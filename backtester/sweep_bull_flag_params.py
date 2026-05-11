"""Univariate parameter sweep for the Bull Flag strategy — Top-5 focus.

Sweeps each of the **5 params with measurable impact** from the prior
12-param baseline (see ``docs/strategy_notes/bull_flag_filter_audit.md``)
while holding the others at the page-default value, on a fixed
universe + window. For each (param, value) pair it runs the full
:class:`MultiBullFlagStrategy` end-to-end and writes the resulting
metrics to a CSV one row at a time.

Why these 5: the original 12-param sweep showed that 7 params produced
identical metrics across all 3 swept values (binding cap elsewhere or
the value doesn't reach a regime change on this dataset). Dropping
them cuts runtime ``36 → 15 iter`` while preserving the only
sensitivity-bearing dimensions:

    target_at_r_multiple, pole_min_pct, pullback_volume_ratio,
    pole_lookback, add_at_r

A subsequent **multivariate refinement** across these 5 is a separate
follow-up (cross-effects are not captured by univariate).

Logging / persistence guarantees
    - CSV is opened in append mode and flushed after every row →
      ``Ctrl-C`` mid-run loses at most the in-flight combo.
    - On startup the script reads any pre-existing rows and **skips
      already-completed (param, value) pairs** so re-running resumes.
    - Per-iteration stderr line includes timestamp + elapsed seconds
      so users can see hangs (vs slow-but-progressing).
    - Final summary written to ``{out}_summary.json`` with best
      param value per metric (total_return / win_rate / pl_ratio).

Usage
    docker compose exec backtester python sweep_bull_flag_params.py \
        --start 2026-04-26 --end 2026-05-10 \
        --universe nasdaq_full --out /tmp/sweep_bull_flag.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time as t_mod
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from data.adapters.composed_fundamentals import build_default_fundamentals
from data.adapters.composed_market_data import build_default_market_data
from data.adapters.regular_session_filter import RegularSessionFilterAdapter
from data.adapters.wikipedia_universe import default_universe_provider
from data.domain.market_calendar import NY
from pattern.adapters.bull_flag import BullFlagDetector
from strategy.adapters.bull_flag_strategy import BullFlagStrategy
from strategy.adapters.multi_bull_flag_strategy import MultiBullFlagStrategy
from strategy.domain.models import MultiStrategyConfig, TossFeeSchedule

# Page defaults (= 21_Bull_Flag_Strategy.py / 22_Multi_Bull_Flag.py).
# Anything not in SWEEP_GRID stays at this value across all sweep runs.
DEFAULTS: dict[str, Any] = {
    # Stock selection
    "min_gap_pct": 0.02,
    "min_rvol": 5.0,
    "min_price": 2.0,
    "max_price": 20.0,
    "max_float_shares": 10_000_000.0,
    # Pole geometry
    "pole_lookback": 7,
    "pole_min_pct": 0.04,
    "pole_min_green_bars": 3,
    # Flag geometry
    "flag_max_bars": 4,
    "flag_max_retrace": 0.7,
    # Time windows
    "latest_entry_hour": 12,
    # Volume profile
    "pullback_volume_ratio": 0.7,
    "breakout_volume_ratio": 0.0,
    # Topping tail
    "max_pole_topping_tail_ratio": 0.5,
    # MTF
    "enable_mtf_check": True,
    "mtf_tolerance_seconds": 600,
    # Continuity guards
    "split_blackout_days": 30,
    "price_floor_lookback_days": 30,
    # Strategy
    "target_min_r_multiple": 2.0,
    "target_at_r_multiple": 2.0,
    "add_at_r": 1.5,
    "risk_per_trade": 0.02,
    "max_session_losses": 1,
}

# Sweep grid — Top 5 impactful params identified by the 4/11~5/11
# nasdaq_full sweep (MTF=False, 8 trades baseline).
#
# Reasoning:
#   - The full 12-param sweep showed that 7 of the 12 params produced
#     identical metrics across all 3 swept values (mtf_tolerance,
#     max_session_losses, max_pole_topping_tail_ratio, risk_per_trade,
#     min_gap_pct, max_float_shares, latest_entry_hour). These either
#     don't bind on this dataset or are blocked by other constraints
#     (e.g. max_position_pct_of_equity cap masks risk_per_trade).
#   - Keeping the same 5 params + same value sets so this re-run is
#     directly comparable to the prior result table. Multivariate
#     refinement (cross-effects) is a separate follow-up.
#
# Value sets: identical to prior sweep so resumed CSVs cross-validate.
# Each param's winning value (from prior baseline) is in **bold** in
# the runtime log. Total iter count = Σ |grid_i| = 15.
SWEEP_GRID: dict[str, list[Any]] = {
    "target_at_r_multiple": [2.0, 2.5, 3.0],      # winner: 3.0 → +11.95%
    "pole_min_pct": [0.03, 0.05, 0.08],           # winner: 0.08 → +9.57%, win 100%
    "pullback_volume_ratio": [0.5, 0.7, 1.0],     # winner: 1.0 → +6.11%, win 75%
    "pole_lookback": [5, 7, 10],                  # winner: 5 → +5.80%, P/L 3.41
    "add_at_r": [1.0, 1.5, 2.0],                  # winner: 1.0 → +5.20%
}

CSV_FIELDS = [
    "ts",
    "sweep_param",
    "sweep_value",
    "tickers_scanned",
    "failed_tickers",
    "signals",
    "trades",
    "win_rate",
    "total_return_pct",
    "max_dd_pct",
    "final_capital",
    "total_commission",
    "avg_win_pnl",
    "avg_loss_pnl",
    "pl_ratio",
    "elapsed_s",
]


def setup_logger() -> logging.Logger:
    log = logging.getLogger("sweep")
    log.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
    log.addHandler(handler)
    log.propagate = False
    return log


def run_one(
    params: dict[str, Any],
    start_d: date,
    end_d: date,
    universe: str,
    max_tickers: int,
    max_workers: int,
) -> dict[str, Any]:
    """Run MultiBullFlagStrategy once with given params, return metrics."""
    md = RegularSessionFilterAdapter(build_default_market_data(), market=NY)
    # Daily도 composed adapter 통해 — yfinance primary + Massive fallback
    # + 디스크 캐시. Raw YFinanceAdapter는 cache 0 + rate-limit 시 fail
    # → ticker scan_universe에서 전부 failed로 빠짐.
    fundamentals = build_default_fundamentals()
    universe_provider = default_universe_provider()
    fee = TossFeeSchedule()

    def det_factory(*, float_shares, splits):
        return BullFlagDetector(
            float_shares=float_shares,
            require_float_filter=True,
            min_gap_pct=params["min_gap_pct"],
            min_rvol=params["min_rvol"],
            min_price=params["min_price"],
            max_price=params["max_price"],
            max_float_shares=params["max_float_shares"],
            pole_lookback=int(params["pole_lookback"]),
            pole_min_pct=params["pole_min_pct"],
            pole_min_green_bars=int(params["pole_min_green_bars"]),
            flag_max_bars=int(params["flag_max_bars"]),
            flag_max_retrace=params["flag_max_retrace"],
            latest_entry_local=time(int(params["latest_entry_hour"]), 0),
            splits=splits,
            split_blackout_days=int(params["split_blackout_days"]),
            price_floor_lookback_days=int(params["price_floor_lookback_days"]),
            enable_mtf_check=bool(params["enable_mtf_check"]),
            mtf_tolerance_seconds=int(params["mtf_tolerance_seconds"]),
            enable_volume_profile=True,
            pullback_volume_ratio=params["pullback_volume_ratio"],
            breakout_volume_ratio=params["breakout_volume_ratio"],
            max_pole_topping_tail_ratio=params["max_pole_topping_tail_ratio"],
        )

    def strat_factory(*, detector):
        return BullFlagStrategy(
            detector=detector,
            target_min_r_multiple=params["target_min_r_multiple"],
            target_at_r_multiple=params["target_at_r_multiple"],
            enable_add_to_winner=True,
            add_at_r=params["add_at_r"],
            add_confirm_on_close=True,
            max_session_losses=int(params["max_session_losses"]),
            fee_schedule=fee,
        )

    multi = MultiBullFlagStrategy(
        market_data=md,
        market_data_5m=md,
        daily_market_data=md,
        fundamentals=fundamentals,
        universe_provider=universe_provider,
        detector_factory=det_factory,
        strategy_factory=strat_factory,
        market=NY,
        max_workers=max_workers,
        require_float_filter=True,
    )
    cfg = MultiStrategyConfig(
        universe=universe,
        start_date=start_d,
        end_date=end_d,
        pattern_name="bull_flag",
        initial_capital=100_000.0,
        risk_per_trade=params["risk_per_trade"],
        max_holding_days=1,
        max_tickers=max_tickers if max_tickers > 0 else None,
        fee_schedule=fee,
    )
    res = multi.run(cfg)

    # Per-trade aggregates for P/L ratio + win-side / loss-side breakdown.
    wins = [tr.pnl for tr in res.trades if tr.pnl > 0]
    losses = [tr.pnl for tr in res.trades if tr.pnl <= 0]
    avg_win = sum(wins) / len(wins) if wins else 0.0
    avg_loss = sum(losses) / len(losses) if losses else 0.0
    pl_ratio = abs(avg_win / avg_loss) if avg_loss < 0 else 0.0

    return {
        "tickers_scanned": res.tickers_scanned,
        "failed_tickers": len(res.failed_tickers),
        "signals": res.total_signals,
        "trades": res.trades_taken,
        "win_rate": round(res.win_rate, 4),
        "total_return_pct": round(res.total_return_pct, 6),
        "max_dd_pct": round(res.max_drawdown_pct, 4),
        "final_capital": round(res.final_capital, 2),
        "total_commission": round(res.total_commission, 2),
        "avg_win_pnl": round(avg_win, 2),
        "avg_loss_pnl": round(avg_loss, 2),
        "pl_ratio": round(pl_ratio, 2),
    }


def load_completed(out_path: Path) -> set[tuple[str, str]]:
    """Read prior CSV rows so we can resume — skip already-done combos."""
    if not out_path.exists():
        return set()
    try:
        df = pd.read_csv(out_path)
    except Exception:
        return set()
    return {(str(r["sweep_param"]), str(r["sweep_value"])) for _, r in df.iterrows()}


def write_summary(out_path: Path, sweep_params: list[str]) -> None:
    """Pick best value per param × metric, dump to {out}_summary.json."""
    if not out_path.exists():
        return
    df = pd.read_csv(out_path)
    summary: dict[str, dict[str, Any]] = {}
    for param in sweep_params:
        sub = df[df["sweep_param"] == param]
        if sub.empty:
            continue
        # Multiple metrics, pick best per. NaN → drop.
        result: dict[str, Any] = {}
        for metric, ascending in (
            ("total_return_pct", False),
            ("win_rate", False),
            ("pl_ratio", False),
            ("max_dd_pct", True),  # smaller is better
        ):
            if metric not in sub.columns:
                continue
            sub_clean = sub.dropna(subset=[metric])
            if sub_clean.empty:
                continue
            ranked = sub_clean.sort_values(metric, ascending=ascending)
            best = ranked.iloc[0]
            result[f"best_by_{metric}"] = {
                "value": best["sweep_value"],
                metric: float(best[metric]),
                "trades": int(best["trades"]),
            }
        summary[param] = result

    summary_path = out_path.with_name(out_path.stem + "_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"[summary] wrote {summary_path}", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2026-04-26")
    parser.add_argument("--end", default="2026-05-10")
    parser.add_argument("--universe", default="nasdaq_full")
    parser.add_argument("--max-tickers", type=int, default=0, help="0 = full universe (NASDAQ ~2200)")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--out", default="/tmp/sweep_bull_flag.csv")
    parser.add_argument(
        "--params",
        nargs="*",
        default=None,
        help="Subset of params to sweep. Default = all from SWEEP_GRID.",
    )
    parser.add_argument(
        "--mtf",
        choices=["true", "false"],
        default="true",
        help="Default value of enable_mtf_check across sweep. "
        "Run twice (true / false) to get both videos-canonical and "
        "lenient regimes — they share data + fundamentals cache.",
    )
    args = parser.parse_args()
    DEFAULTS["enable_mtf_check"] = args.mtf == "true"

    log = setup_logger()
    start_d = date.fromisoformat(args.start)
    end_d = date.fromisoformat(args.end)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sweep_params = args.params or list(SWEEP_GRID.keys())
    for p in sweep_params:
        if p not in SWEEP_GRID:
            log.error("Unknown sweep param: %r (known: %s)", p, sorted(SWEEP_GRID.keys()))
            return 2

    completed = load_completed(out_path)
    if completed:
        log.info("Resume: %d combos already in %s", len(completed), out_path)

    write_header = not out_path.exists()
    f_out = open(out_path, "a", newline="")
    writer = csv.DictWriter(f_out, fieldnames=CSV_FIELDS)
    if write_header:
        writer.writeheader()
        f_out.flush()

    total = sum(len(SWEEP_GRID[p]) for p in sweep_params)
    log.info(
        "Sweep start: universe=%s window=%s..%s max_tickers=%s " "params=%d total_combos=%d",
        args.universe,
        start_d,
        end_d,
        args.max_tickers if args.max_tickers > 0 else "ALL",
        len(sweep_params),
        total,
    )

    done = 0
    try:
        for param in sweep_params:
            for val in SWEEP_GRID[param]:
                done += 1
                key = (param, str(val))
                if key in completed:
                    log.info("[%d/%d] SKIP %s=%s (resume)", done, total, param, val)
                    continue

                params = dict(DEFAULTS)
                params[param] = val
                t0 = t_mod.time()
                log.info("[%d/%d] %s=%s ...", done, total, param, val)
                try:
                    metrics = run_one(
                        params,
                        start_d,
                        end_d,
                        args.universe,
                        args.max_tickers,
                        args.max_workers,
                    )
                    elapsed = t_mod.time() - t0
                except Exception as exc:
                    elapsed = t_mod.time() - t0
                    log.exception(
                        "[%d/%d] FAIL %s=%s after %.1fs: %s", done, total, param, val, elapsed, exc
                    )
                    continue

                row = {
                    "ts": datetime.now().isoformat(timespec="seconds"),
                    "sweep_param": param,
                    "sweep_value": val,
                    "elapsed_s": round(elapsed, 1),
                    **metrics,
                }
                writer.writerow(row)
                f_out.flush()
                log.info(
                    "[%d/%d] %s=%s → trades=%d ret=%+.2f%% win=%.0f%% " "pl=%.2f dd=%.2f%% (%.1fs)",
                    done,
                    total,
                    param,
                    val,
                    metrics["trades"],
                    metrics["total_return_pct"] * 100,
                    metrics["win_rate"] * 100,
                    metrics["pl_ratio"],
                    metrics["max_dd_pct"] * 100,
                    elapsed,
                )
                # Incremental summary refresh every iteration so the
                # JSON snapshot is always current.
                write_summary(out_path, sweep_params)
    finally:
        f_out.close()
        write_summary(out_path, sweep_params)
        log.info("Done. CSV=%s summary=%s", out_path, out_path.with_name(out_path.stem + "_summary.json"))

    return 0


if __name__ == "__main__":
    sys.exit(main())
