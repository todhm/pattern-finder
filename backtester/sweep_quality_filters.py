"""Quality-filter relaxation sweep — find tunings that **add trades**
while preserving the win-rate improvement.

Background
----------
The 7 quality filters introduced after the 2026-05-12 CSV analysis
brought 34 → 7 trades (53% → 71% win, P/L 1.23 → 1.94). Goal now is
to find which **single-axis relaxations** keep the win-rate gain
while bringing back some of the lost trade volume.

Sweep grid
----------
Univariate around the current strict default. Each row tests one
relaxed value with all others held at the strict baseline. Same
``MultiStrategyConfig`` (universe, window, capital, risk) used for
every cell so trade counts are directly comparable to the baseline
CSV row.

Output
------
CSV with one row per (filter, value) — trades, win_rate, total_return,
P/L ratio, max_dd. Final summary picks the Pareto front (more trades
+ no win-rate regression).

Usage
-----
    docker compose exec backtester python sweep_quality_filters.py \\
        --start 2025-01-01 --end 2026-05-11 \\
        --universe nasdaq_full \\
        --out /tmp/sweep_quality.csv
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


# Strict baseline — what produced the 7-trade CSV. Each sweep cell
# overrides ONE field of this dict.
BASELINE: dict[str, Any] = {
    # Stock selection (영상 정통, 변경 X)
    "min_gap_pct": 0.02,
    "min_rvol": 5.0,
    "min_price": 2.0,
    "max_price": 20.0,
    "max_float_shares": 10_000_000.0,
    # Pole / flag (영상 정통)
    "pole_lookback": 7,
    "pole_min_pct": 0.08,
    "pole_min_green_bars": 3,
    "flag_max_bars": 4,
    "flag_max_retrace": 0.7,
    "latest_entry_hour": 12,
    # Volume profile
    "pullback_volume_ratio": 0.7,
    "breakout_volume_ratio": 0.0,
    "max_pole_topping_tail_ratio": 0.5,
    # MTF (off — already verified 0 trades)
    "enable_mtf_check": False,
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
    # Quality filters — 2026-05-13 sweep 결과 적용된 새 default
    "max_rvol": 30.0,
    "max_gap_pct": 0.50,            # 0.30 → 0.50 (sweep winner)
    "max_stop_distance_pct": 0.05,
    "require_9ema_support": True,
    "ema9_tolerance_pct": 0.025,    # 0.015 → 0.025 (sweep winner)
    "also_accept_20ema_support": False,
    "require_daily_trend": False,   # True → False (Ross 영상 X)
    "daily_trend_sma_period": 50,
    "max_nth_pullback": 2,
    "use_premarket_high": True,
}


# Univariate relaxation grid — single-axis tests.
# Each entry: list of overrides to try. ``None`` for boolean keys
# means "toggle off". Range values for numeric keys.
SWEEP_GRID: dict[str, list[Any]] = {
    # Ross-derived relaxations (영상 정통 안 벗어남)
    "max_nth_pullback": [2, 3, 4, 5],           # Ross: 3rd "cautious"
    "ema9_tolerance_pct": [0.010, 0.015, 0.020, 0.025, 0.030],
    "also_accept_20ema_support": [False, True],  # Brett Burgett 추가
    # Custom-added relaxations
    "max_rvol": [20.0, 30.0, 40.0, 50.0, None],  # None = no cap
    "max_gap_pct": [0.20, 0.30, 0.50, 0.70, None],
    "max_stop_distance_pct": [0.04, 0.05, 0.06, 0.07, 0.10],
    "require_daily_trend": [True, False],
    "daily_trend_sma_period": [20, 30, 50, 100],  # only when on
    "use_premarket_high": [True, False],
}

CSV_FIELDS = [
    "ts", "sweep_param", "sweep_value",
    "tickers_scanned", "failed_tickers", "signals",
    "trades", "win_rate", "total_return_pct", "max_dd_pct",
    "avg_win_pnl", "avg_loss_pnl", "pl_ratio",
    "final_capital", "elapsed_s",
]


def setup_logger() -> logging.Logger:
    log = logging.getLogger("sweep")
    log.setLevel(logging.INFO)
    h = logging.StreamHandler(sys.stderr)
    h.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
    ))
    log.addHandler(h)
    log.propagate = False
    return log


def run_one(
    overrides: dict[str, Any],
    start_d: date, end_d: date,
    universe: str, max_tickers: int, max_workers: int,
    chunk_months: int,
) -> dict[str, Any]:
    """One sweep cell — baseline params + single-axis override."""
    p = {**BASELINE, **overrides}

    md_raw = build_default_market_data()
    md = RegularSessionFilterAdapter(md_raw, market=NY)
    fp = build_default_fundamentals()
    fee = TossFeeSchedule()

    def det_factory(*, float_shares, splits, pm_high_by_date=None):
        return BullFlagDetector(
            float_shares=float_shares,
            require_float_filter=True,
            min_gap_pct=p["min_gap_pct"], min_rvol=p["min_rvol"],
            min_price=p["min_price"], max_price=p["max_price"],
            max_float_shares=p["max_float_shares"],
            pole_lookback=int(p["pole_lookback"]),
            pole_min_pct=p["pole_min_pct"],
            pole_min_green_bars=int(p["pole_min_green_bars"]),
            flag_max_bars=int(p["flag_max_bars"]),
            flag_max_retrace=p["flag_max_retrace"],
            latest_entry_local=time(int(p["latest_entry_hour"]), 0),
            splits=splits,
            split_blackout_days=int(p["split_blackout_days"]),
            price_floor_lookback_days=int(p["price_floor_lookback_days"]),
            enable_mtf_check=bool(p["enable_mtf_check"]),
            mtf_tolerance_seconds=int(p["mtf_tolerance_seconds"]),
            enable_volume_profile=True,
            pullback_volume_ratio=p["pullback_volume_ratio"],
            breakout_volume_ratio=p["breakout_volume_ratio"],
            max_pole_topping_tail_ratio=p["max_pole_topping_tail_ratio"],
            max_rvol=p["max_rvol"],
            max_gap_pct=p["max_gap_pct"],
            max_stop_distance_pct=p["max_stop_distance_pct"],
            require_9ema_support=bool(p["require_9ema_support"]),
            ema9_tolerance_pct=p["ema9_tolerance_pct"],
            also_accept_20ema_support=bool(p["also_accept_20ema_support"]),
            require_daily_trend=bool(p["require_daily_trend"]),
            daily_trend_sma_period=int(p["daily_trend_sma_period"]),
            max_nth_pullback=int(p["max_nth_pullback"]),
            premarket_high_by_date=pm_high_by_date if p["use_premarket_high"] else None,
        )

    def strat_factory(*, detector):
        return BullFlagStrategy(
            detector=detector,
            target_min_r_multiple=p["target_min_r_multiple"],
            target_at_r_multiple=p["target_at_r_multiple"],
            enable_add_to_winner=True,
            add_at_r=p["add_at_r"],
            add_confirm_on_close=True,
            max_session_losses=int(p["max_session_losses"]),
            fee_schedule=fee,
        )

    multi = MultiBullFlagStrategy(
        market_data=md, market_data_5m=md, daily_market_data=md,
        fundamentals=fp, universe_provider=default_universe_provider(),
        detector_factory=det_factory, strategy_factory=strat_factory,
        market=NY, max_workers=max_workers, require_float_filter=True,
        chunk_months=chunk_months,
        raw_market_data=md_raw if p["use_premarket_high"] else None,
    )
    cfg = MultiStrategyConfig(
        universe=universe, start_date=start_d, end_date=end_d,
        pattern_name="bull_flag", initial_capital=100_000.0,
        risk_per_trade=p["risk_per_trade"], max_holding_days=1,
        max_tickers=max_tickers if max_tickers > 0 else None,
        fee_schedule=fee,
    )
    res = multi.simulate(cfg)

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
        "avg_win_pnl": round(avg_win, 2),
        "avg_loss_pnl": round(avg_loss, 2),
        "pl_ratio": round(pl_ratio, 2),
        "final_capital": round(res.final_capital, 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end", default="2026-05-11")
    parser.add_argument("--universe", default="nasdaq_full")
    parser.add_argument("--max-tickers", type=int, default=0)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--chunk-months", type=int, default=1)
    parser.add_argument("--out", default="/tmp/sweep_quality.csv")
    parser.add_argument(
        "--params", nargs="*", default=None,
        help="Subset of params to sweep. Default = all from SWEEP_GRID.",
    )
    parser.add_argument(
        "--include-baseline", action="store_true",
        help="Also run the strict baseline (no overrides) as a reference row.",
    )
    args = parser.parse_args()

    log = setup_logger()
    start_d = date.fromisoformat(args.start)
    end_d = date.fromisoformat(args.end)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sweep_params = args.params or list(SWEEP_GRID.keys())
    for p in sweep_params:
        if p not in SWEEP_GRID:
            log.error("Unknown param: %r", p)
            return 2

    # Resume support — read existing CSV
    completed: set[tuple[str, str]] = set()
    if out_path.exists():
        try:
            df = pd.read_csv(out_path)
            for _, row in df.iterrows():
                completed.add((str(row["sweep_param"]), str(row["sweep_value"])))
            log.info("Resume — %d combos already in CSV", len(completed))
        except Exception:
            pass

    write_header = not out_path.exists()
    f_out = open(out_path, "a", newline="")
    writer = csv.DictWriter(f_out, fieldnames=CSV_FIELDS)
    if write_header:
        writer.writeheader()
        f_out.flush()

    total = sum(len(SWEEP_GRID[p]) for p in sweep_params)
    if args.include_baseline:
        total += 1
    log.info(
        "Sweep start: universe=%s window=%s..%s max_tickers=%s combos=%d",
        args.universe, start_d, end_d,
        args.max_tickers if args.max_tickers > 0 else "ALL", total,
    )

    done = 0

    def write_row(label_param: str, label_value: str, metrics: dict[str, Any], elapsed: float) -> None:
        writer.writerow({
            "ts": datetime.now().isoformat(timespec="seconds"),
            "sweep_param": label_param, "sweep_value": label_value,
            "elapsed_s": round(elapsed, 1),
            **metrics,
        })
        f_out.flush()

    try:
        if args.include_baseline:
            done += 1
            log.info("[%d/%d] BASELINE (strict 7-filter)...", done, total)
            t0 = t_mod.time()
            metrics = run_one(
                {}, start_d, end_d, args.universe,
                args.max_tickers, args.max_workers, args.chunk_months,
            )
            elapsed = t_mod.time() - t0
            write_row("__baseline__", "baseline", metrics, elapsed)
            log.info(
                "[%d/%d] BASELINE → trades=%d win=%.0f%% ret=%+.2f%% pl=%.2f (%.1fs)",
                done, total, metrics["trades"], metrics["win_rate"] * 100,
                metrics["total_return_pct"] * 100, metrics["pl_ratio"], elapsed,
            )

        for param in sweep_params:
            for val in SWEEP_GRID[param]:
                done += 1
                key = (param, str(val))
                if key in completed:
                    log.info("[%d/%d] SKIP %s=%s (resume)", done, total, param, val)
                    continue
                log.info("[%d/%d] %s=%s ...", done, total, param, val)
                t0 = t_mod.time()
                try:
                    metrics = run_one(
                        {param: val}, start_d, end_d, args.universe,
                        args.max_tickers, args.max_workers, args.chunk_months,
                    )
                except Exception as exc:
                    log.exception("[%d/%d] FAIL %s=%s: %s", done, total, param, val, exc)
                    continue
                elapsed = t_mod.time() - t0
                write_row(param, str(val), metrics, elapsed)
                log.info(
                    "[%d/%d] %s=%s → trades=%d win=%.0f%% ret=%+.2f%% pl=%.2f dd=%.2f%% (%.1fs)",
                    done, total, param, val,
                    metrics["trades"], metrics["win_rate"] * 100,
                    metrics["total_return_pct"] * 100,
                    metrics["pl_ratio"], metrics["max_dd_pct"] * 100, elapsed,
                )
    finally:
        f_out.close()
        # Summary — pareto-style report
        try:
            df = pd.read_csv(out_path)
            baseline_row = df[df["sweep_param"] == "__baseline__"].head(1)
            base_trades = int(baseline_row["trades"].iloc[0]) if not baseline_row.empty else None
            base_win = float(baseline_row["win_rate"].iloc[0]) if not baseline_row.empty else None
            summary = {
                "baseline": {
                    "trades": base_trades, "win_rate": base_win,
                } if base_trades is not None else None,
                "pareto_front": [],  # more trades + win_rate >= baseline
            }
            if base_trades is not None and base_win is not None:
                non_base = df[df["sweep_param"] != "__baseline__"]
                pareto = non_base[
                    (non_base["trades"] > base_trades)
                    & (non_base["win_rate"] >= base_win * 0.9)  # 10% win-rate slack
                ].sort_values("total_return_pct", ascending=False)
                summary["pareto_front"] = pareto.to_dict(orient="records")
            summary_path = out_path.with_name(out_path.stem + "_summary.json")
            summary_path.write_text(json.dumps(summary, indent=2, default=str))
            log.info("Summary written: %s", summary_path)
        except Exception as exc:
            log.exception("Summary write failed: %s", exc)

    return 0


if __name__ == "__main__":
    sys.exit(main())
