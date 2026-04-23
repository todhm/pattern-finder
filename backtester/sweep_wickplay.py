"""Full-grid parameter sweep for Wick Play strategy.

Sweeps seven knobs (exit, entry, regime, macro) against 500 Nasdaq
common-stock tickers and saves:

    - ``sweep_results/<ts>/summary.csv`` — one row per config with
      aggregate metrics (trades / win_rate / total_return / max_dd /
      avg_R / profit_factor / avg_hold_days).
    - ``sweep_results/<ts>/trades.parquet`` — one row per individual
      trade, keyed by ``config_id``, so later analysis can dig into
      winners vs losers per config.
    - ``sweep_results/<ts>/config.json`` — the sweep definition
      (universe, window, grid dimensions) for reproducibility.

Perf trick: fetch + ``_with_indicators`` + exhaustion detection are
CONFIG-INDEPENDENT along most dimensions, so we pre-compute them
once per ``(ticker, ema_trail)`` pair and reuse. Only the wick
detector + walk step re-runs per config. Target: < 15 s per config.

Run inside the container::

    docker compose exec backtester python sweep_wickplay.py
"""

from __future__ import annotations

import itertools
import json
import multiprocessing as mp
import os
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from data.adapters.cached_market_data import CachedMarketDataAdapter
from data.adapters.macro_calendar import blackout_dates as build_blackout_dates
from data.adapters.wikipedia_universe import default_universe_provider
from data.adapters.yfinance_adapter import YFinanceAdapter
from pattern.adapters.exhaustion_extension_top import (
    ExhaustionExtensionTopDetector,
)
from pattern.adapters.wick_play import WickPlayDetector
from strategy.adapters.multi_wickplay_strategy import MultiWickPlayStrategy
from strategy.adapters.wickplay_strategy import WickPlayStrategy
from strategy.domain.models import MultiStrategyConfig, TossFeeSchedule


# ---- Sweep window ----------------------------------------------------

START_DATE = date(2024, 1, 1)
END_DATE = date(2026, 4, 22)
MAX_TICKERS = 500
INITIAL_CAPITAL = 100_000.0
RISK_PER_TRADE = 0.02
REGIME_SYMBOL = "^GSPC"

# ---- Grid dimensions -------------------------------------------------
# 2 x 2 x 2 x 2 x 5 x 6 = 480 configs. ~5-10s each → ~40-80 min total.

EMA_TRAILS = [10, 20]
MIN_TRAIL_BARS_LIST = [2, 5]
MAX_HOLDING_DAYS_LIST = [40, 60]
MIN_PSYCH_SCORES = [3, 4]

REGIMES: dict[str, dict[str, Any]] = {
    "none": {},
    "sma20": {"regime_min_above_sma": 20},
    "sma50": {"regime_min_above_sma": 50},
    "dd5_2pct": {"regime_max_n_day_drawdown": (5, 0.02)},
    "sma50+dd5_2pct": {
        "regime_min_above_sma": 50,
        "regime_max_n_day_drawdown": (5, 0.02),
    },
}

# Macro dimension — now explicitly separates FOMC / CPI / window options.
# Each value is a (include_fomc, include_cpi, window_days) tuple.
MACRO_SPECS: dict[str, tuple[bool, bool, int]] = {
    "none": (False, False, 0),
    "fomc_w0": (True, False, 0),
    "fomc_w1": (True, False, 1),
    "fomc_cpi_w0": (True, True, 0),
    "cpi_w0": (False, True, 0),
    "cpi_w1": (False, True, 1),
}


# ---- Cache-aware runner ---------------------------------------------


class _CachedMultiWickPlay(MultiWickPlayStrategy):
    """Override ``_scan_universe`` to return pre-built ticker_state
    instead of re-scanning the universe. Set via ``set_state()`` before
    calling ``run()``.
    """

    def set_state(
        self,
        ticker_state: dict[str, dict[str, Any]],
        failed: list[str],
    ) -> None:
        self._cached_state = ticker_state
        self._cached_failed = failed

    def _scan_universe(self, tickers, config):
        return self._cached_state, list(self._cached_failed)


class _StaticUniverse:
    def __init__(self, tickers: list[str]):
        self._tickers = list(tickers)

    def get_tickers(self, universe: str) -> list[str]:
        return list(self._tickers)


# ---- Grid config container -------------------------------------------


@dataclass(frozen=True)
class GridConfig:
    config_id: int
    ema_trail: int
    min_trail_bars: int
    max_holding_days: int
    min_psych_score: int
    regime_key: str
    macro_key: str


def build_grid() -> list[GridConfig]:
    rows = []
    cid = 0
    for (ema, mtb, mhd, psych, regime, macro) in itertools.product(
        EMA_TRAILS,
        MIN_TRAIL_BARS_LIST,
        MAX_HOLDING_DAYS_LIST,
        MIN_PSYCH_SCORES,
        REGIMES.keys(),
        MACRO_SPECS.keys(),
    ):
        rows.append(
            GridConfig(
                config_id=cid,
                ema_trail=ema,
                min_trail_bars=mtb,
                max_holding_days=mhd,
                min_psych_score=psych,
                regime_key=regime,
                macro_key=macro,
            )
        )
        cid += 1
    return rows


# ---- Builders --------------------------------------------------------


def build_detector(
    cfg: GridConfig,
    regime_df: pd.DataFrame,
    blackout_sets: dict[str, set[date] | None],
) -> WickPlayDetector:
    overrides = dict(REGIMES[cfg.regime_key])
    blackout = blackout_sets.get(cfg.macro_key)
    has_regime = bool(overrides)
    return WickPlayDetector(
        min_upper_wick_ratio=0.5,
        max_volume_dryup=1.0,
        breakout_trigger="wick_high",
        stop_mode="wick_low",
        max_wick_range_atr=2.5,
        cooldown_bars=5,
        psych_vol_lookback=20,
        psych_wick_vol_exhaustion_mult=1.0,
        psych_breakout_vol_expansion_mult=1.0,
        psych_prior_red_streak=2,
        psych_dramatic_wick_ratio=0.65,
        min_psych_score=cfg.min_psych_score,
        min_breakout_strength_atr=0.3,
        min_prior_trend_20d=-0.01,
        min_wick_close_location=0.15,
        regime_df=regime_df if has_regime else None,
        blackout_dates=blackout,
        **overrides,
    )


def build_exit_detector(ema_trail: int) -> ExhaustionExtensionTopDetector:
    return ExhaustionExtensionTopDetector(
        extension_atr_mult=2.5,
        min_slow_slope=0.005,
        ema_fast=ema_trail,
    )


def build_strategy(
    market_data: CachedMarketDataAdapter,
    detector: WickPlayDetector,
    exit_detector: ExhaustionExtensionTopDetector,
    cfg: GridConfig,
) -> WickPlayStrategy:
    return WickPlayStrategy(
        market_data=market_data,
        detector=detector,
        exit_detector=exit_detector,
        ema_trail=cfg.ema_trail,
        atr_period=14,
        min_trail_bars=cfg.min_trail_bars,
        enable_same_day_reversal_exit=False,
        enable_gap_down_rejection=True,
        max_entry_gap_down=0.005,
        enable_breakeven_stop=False,
    )


# ---- Pre-fetch + pre-compute -----------------------------------------


def prefetch_all(
    tickers: list[str],
    market_data: CachedMarketDataAdapter,
    min_bars: int,
) -> dict[str, pd.DataFrame]:
    """Fetch raw OHLCV for each ticker (no indicators yet).

    Runs sequentially — the CachedMarketDataAdapter already memoizes
    to disk, so after the first pass every call is a parquet read.
    """
    raw: dict[str, pd.DataFrame] = {}
    fetch_start = START_DATE - timedelta(days=400)
    for i, t in enumerate(tickers):
        if i % 50 == 0:
            print(f"  fetch {i}/{len(tickers)}...", flush=True)
        try:
            df = market_data.fetch_ohlcv(t, fetch_start, END_DATE)
        except Exception:
            continue
        if df is None or df.empty or len(df) < min_bars:
            continue
        if df.index.tz is not None:
            df = df.copy()
            df.index = df.index.tz_localize(None)
        raw[t] = df
    return raw


def build_indicator_cache(
    raw_dfs: dict[str, pd.DataFrame], ema_trails: list[int]
) -> dict[tuple[str, int], pd.DataFrame]:
    """Pre-compute indicator frames per ``(ticker, ema_trail)``.

    Exposing ``WickPlayStrategy._with_indicators`` as the builder keeps
    the math identical to what the real strategy uses at run time.
    """
    cache: dict[tuple[str, int], pd.DataFrame] = {}
    for ema in ema_trails:
        stub = WickPlayStrategy(
            market_data=None,  # type: ignore[arg-type]
            detector=None,     # type: ignore[arg-type]
            ema_trail=ema,
        )
        for ticker, df in raw_dfs.items():
            cache[(ticker, ema)] = stub._with_indicators(df)
    return cache


def build_exit_dates_cache(
    df_ind_cache: dict[tuple[str, int], pd.DataFrame],
    ema_trails: list[int],
) -> dict[tuple[str, int], set[date]]:
    """Pre-compute exhaustion exit dates per ``(ticker, ema_trail)``."""
    cache: dict[tuple[str, int], set[date]] = {}
    for ema in ema_trails:
        exh = build_exit_detector(ema)
        for (ticker, key_ema), df_ind in df_ind_cache.items():
            if key_ema != ema:
                continue
            cache[(ticker, ema)] = {s.date for s in exh.detect(df_ind)}
    return cache


def build_blackout_sets() -> dict[str, set[date] | None]:
    """Materialize every macro blackout option once up-front."""
    out: dict[str, set[date] | None] = {}
    for key, (fomc, cpi, window) in MACRO_SPECS.items():
        if not fomc and not cpi:
            out[key] = None
            continue
        out[key] = build_blackout_dates(
            start=START_DATE - timedelta(days=400),
            end=END_DATE,
            include_fomc=fomc,
            include_cpi=cpi,
            window_days=window,
        )
    return out


# ---- One-config run --------------------------------------------------


def run_one_config(
    cfg: GridConfig,
    raw_dfs: dict[str, pd.DataFrame],
    df_ind_cache: dict[tuple[str, int], pd.DataFrame],
    exit_dates_cache: dict[tuple[str, int], set[date]],
    blackout_sets: dict[str, set[date] | None],
    regime_df: pd.DataFrame,
    market_data: CachedMarketDataAdapter,
    multi_cfg_template: MultiStrategyConfig,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run one config and return (summary_row, trade_rows)."""
    detector = build_detector(cfg, regime_df, blackout_sets)
    exit_detector = build_exit_detector(cfg.ema_trail)
    strategy = build_strategy(market_data, detector, exit_detector, cfg)

    # Build ticker_state using cached indicators + fresh detect
    ticker_state: dict[str, dict[str, Any]] = {}
    for ticker in raw_dfs:
        df_ind = df_ind_cache[(ticker, cfg.ema_trail)]
        signals = detector.detect(df_ind)
        exit_dates = exit_dates_cache.get((ticker, cfg.ema_trail), set())
        ticker_state[ticker] = {
            "df": df_ind,
            "signals": signals,
            "exit_dates": exit_dates,
        }

    runner = _CachedMultiWickPlay(
        market_data=market_data,
        universe_provider=_StaticUniverse(list(raw_dfs.keys())),
        detector=detector,
        strategy=strategy,
        max_workers=1,
    )
    runner.set_state(ticker_state, failed=[])

    multi_cfg = multi_cfg_template.model_copy(
        update={"max_holding_days": cfg.max_holding_days}
    )
    result = runner.run(multi_cfg)

    # --- Build trade rows with R multiple and days_held ---
    trade_rows: list[dict[str, Any]] = []
    wins = losses = 0
    total_win_pct = total_loss_pct = 0.0
    total_r = 0.0
    total_hold = 0
    for t in result.trades:
        risk_per_share = t.entry_price - t.stop_loss
        if risk_per_share > 0:
            r_multiple = (t.pnl / t.shares) / risk_per_share
        else:
            r_multiple = 0.0
        days_held = (t.exit_date - t.entry_date).days
        if t.pnl > 0:
            wins += 1
            total_win_pct += t.pnl_pct
        else:
            losses += 1
            total_loss_pct += t.pnl_pct
        total_r += r_multiple
        total_hold += days_held
        trade_rows.append(
            {
                "config_id": cfg.config_id,
                "ticker": t.ticker,
                "entry_date": str(t.entry_date),
                "exit_date": str(t.exit_date),
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "stop_loss": t.stop_loss,
                "shares": t.shares,
                "exit_reason": t.exit_reason,
                "gross_pnl": t.gross_pnl,
                "commission": t.commission,
                "net_pnl": t.pnl,
                "pnl_pct": t.pnl_pct,
                "r_multiple": round(r_multiple, 4),
                "days_held": days_held,
                "signal_buy_sell_ratio": t.signal_buy_sell_ratio,
            }
        )

    n = len(result.trades)
    gross_win = sum(t.pnl for t in result.trades if t.pnl > 0)
    gross_loss = -sum(t.pnl for t in result.trades if t.pnl < 0)
    profit_factor = (gross_win / gross_loss) if gross_loss > 0 else float("inf") if gross_win > 0 else 0.0

    summary = {
        "config_id": cfg.config_id,
        "ema_trail": cfg.ema_trail,
        "min_trail_bars": cfg.min_trail_bars,
        "max_holding_days": cfg.max_holding_days,
        "min_psych_score": cfg.min_psych_score,
        "regime_key": cfg.regime_key,
        "macro_key": cfg.macro_key,
        "trades": n,
        "wins": wins,
        "losses": losses,
        "win_rate": round(result.win_rate, 4),
        "total_return_pct": round(result.total_return_pct, 4),
        "final_capital": round(result.final_capital, 2),
        "max_drawdown_pct": round(result.max_drawdown_pct, 4),
        "total_commission": round(result.total_commission, 2),
        "avg_win_pct": round(total_win_pct / wins, 4) if wins else 0.0,
        "avg_loss_pct": round(total_loss_pct / losses, 4) if losses else 0.0,
        "avg_r": round(total_r / n, 4) if n else 0.0,
        "profit_factor": round(profit_factor, 4) if profit_factor != float("inf") else None,
        "avg_hold_days": round(total_hold / n, 1) if n else 0.0,
    }
    return summary, trade_rows


# ---- Parallel worker plumbing ---------------------------------------
#
# Pool workers are spawned via fork() on Linux (docker container), so
# they inherit module-level globals copy-on-write. We stash the big
# caches on this module before the Pool is created; workers read them
# without paying the pickle tax.

_G_RAW_DFS: dict[str, pd.DataFrame] = {}
_G_DF_IND_CACHE: dict[tuple[str, int], pd.DataFrame] = {}
_G_EXIT_DATES_CACHE: dict[tuple[str, int], set[date]] = {}
_G_BLACKOUT_SETS: dict[str, set[date] | None] = {}
_G_REGIME_DF: pd.DataFrame | None = None
_G_MULTI_CFG_TEMPLATE: MultiStrategyConfig | None = None


def _worker(cfg: GridConfig) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Pool worker. Reads prefetched state from module globals."""
    # market_data is only used as plumbing by WickPlayStrategy — no
    # I/O fires because the cached df is passed directly via
    # ticker_state. Re-create inside the worker so we avoid passing
    # an unpickleable object through the Pool.
    market_data = CachedMarketDataAdapter(YFinanceAdapter())
    t0 = time.monotonic()
    summary, trades = run_one_config(
        cfg=cfg,
        raw_dfs=_G_RAW_DFS,
        df_ind_cache=_G_DF_IND_CACHE,
        exit_dates_cache=_G_EXIT_DATES_CACHE,
        blackout_sets=_G_BLACKOUT_SETS,
        regime_df=_G_REGIME_DF,  # type: ignore[arg-type]
        market_data=market_data,
        multi_cfg_template=_G_MULTI_CFG_TEMPLATE,  # type: ignore[arg-type]
    )
    summary["runtime_s"] = round(time.monotonic() - t0, 2)
    return summary, trades


# ---- Main ------------------------------------------------------------


def main() -> None:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(__file__).resolve().parent / "sweep_results" / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[sweep] output dir: {out_dir}", flush=True)

    grid = build_grid()
    print(f"[sweep] total configs: {len(grid)}", flush=True)

    # --- Save config metadata ---
    (out_dir / "config.json").write_text(
        json.dumps(
            {
                "start_date": str(START_DATE),
                "end_date": str(END_DATE),
                "max_tickers": MAX_TICKERS,
                "initial_capital": INITIAL_CAPITAL,
                "risk_per_trade": RISK_PER_TRADE,
                "regime_symbol": REGIME_SYMBOL,
                "dimensions": {
                    "ema_trail": EMA_TRAILS,
                    "min_trail_bars": MIN_TRAIL_BARS_LIST,
                    "max_holding_days": MAX_HOLDING_DAYS_LIST,
                    "min_psych_score": MIN_PSYCH_SCORES,
                    "regimes": list(REGIMES.keys()),
                    "macros": list(MACRO_SPECS.keys()),
                },
                "n_configs": len(grid),
            },
            indent=2,
        )
    )

    market_data = CachedMarketDataAdapter(YFinanceAdapter())
    universe_provider = default_universe_provider()
    tickers = sorted(universe_provider.get_tickers("nasdaq_full"))[:MAX_TICKERS]
    print(f"[sweep] tickers sampled: {len(tickers)}", flush=True)

    # --- Pre-fetch regime ---
    print(f"[sweep] fetching {REGIME_SYMBOL}...", flush=True)
    regime_df = market_data.fetch_ohlcv(
        REGIME_SYMBOL, START_DATE - timedelta(days=400), END_DATE
    )
    print(f"[sweep] regime rows: {len(regime_df)}", flush=True)

    # --- Pre-fetch tickers ---
    t0 = time.monotonic()
    print("[sweep] prefetching ticker OHLCV...", flush=True)
    raw_dfs = prefetch_all(tickers, market_data, min_bars=15)
    print(
        f"[sweep] fetched {len(raw_dfs)} usable tickers in "
        f"{time.monotonic() - t0:.1f}s",
        flush=True,
    )

    # --- Pre-compute indicators ---
    t0 = time.monotonic()
    df_ind_cache = build_indicator_cache(raw_dfs, EMA_TRAILS)
    print(
        f"[sweep] indicators: {len(df_ind_cache)} frames cached in "
        f"{time.monotonic() - t0:.1f}s",
        flush=True,
    )

    # --- Pre-compute exhaustion exit dates ---
    t0 = time.monotonic()
    exit_dates_cache = build_exit_dates_cache(df_ind_cache, EMA_TRAILS)
    print(
        f"[sweep] exhaustion exit dates: {len(exit_dates_cache)} sets in "
        f"{time.monotonic() - t0:.1f}s",
        flush=True,
    )

    # --- Pre-build blackout sets ---
    blackout_sets = build_blackout_sets()
    print(
        "[sweep] blackout sets: "
        + ", ".join(
            f"{k}={len(v) if v else 0}" for k, v in blackout_sets.items()
        ),
        flush=True,
    )

    fee_schedule = TossFeeSchedule(
        buy_commission_pct=0.001,
        sell_commission_pct=0.001,
        sec_fee_pct=0.0000229,
    )
    multi_cfg_template = MultiStrategyConfig(
        universe="nasdaq_full",
        start_date=START_DATE,
        end_date=END_DATE,
        pattern_name="wick_play",
        initial_capital=INITIAL_CAPITAL,
        risk_per_trade=RISK_PER_TRADE,
        max_holding_days=40,  # overridden per config
        max_tickers=len(raw_dfs),
        fee_schedule=fee_schedule,
    )

    # --- Stash caches on module globals so fork'd workers inherit them ---
    global _G_RAW_DFS, _G_DF_IND_CACHE, _G_EXIT_DATES_CACHE
    global _G_BLACKOUT_SETS, _G_REGIME_DF, _G_MULTI_CFG_TEMPLATE
    _G_RAW_DFS = raw_dfs
    _G_DF_IND_CACHE = df_ind_cache
    _G_EXIT_DATES_CACHE = exit_dates_cache
    _G_BLACKOUT_SETS = blackout_sets
    _G_REGIME_DF = regime_df
    _G_MULTI_CFG_TEMPLATE = multi_cfg_template

    n_workers = min(int(os.environ.get("SWEEP_WORKERS", 8)), os.cpu_count() or 4)
    print(f"[sweep] launching pool with {n_workers} workers", flush=True)

    # Use fork — default on Linux, inherits module globals without
    # pickling the huge cache.
    ctx = mp.get_context("fork")

    # --- Sweep loop (parallel) ---
    summary_rows: list[dict[str, Any]] = []
    all_trade_rows: list[dict[str, Any]] = []
    completed = 0
    sweep_start = time.monotonic()

    with ctx.Pool(processes=n_workers) as pool:
        for summary, trades in pool.imap_unordered(
            _worker, grid, chunksize=1
        ):
            summary_rows.append(summary)
            all_trade_rows.extend(trades)
            completed += 1

            if completed % 20 == 0 or completed == len(grid):
                pd.DataFrame(summary_rows).to_csv(
                    out_dir / "summary.csv", index=False
                )
                if all_trade_rows:
                    pd.DataFrame(all_trade_rows).to_parquet(
                        out_dir / "trades.parquet", index=False
                    )

            elapsed = time.monotonic() - sweep_start
            rate = completed / elapsed
            remain = (
                (len(grid) - completed) / rate if rate > 0 else 0.0
            )
            print(
                f"[{completed:>3}/{len(grid)}] id={summary['config_id']:>3} "
                f"ema={summary['ema_trail']:>2} mtb={summary['min_trail_bars']} "
                f"hold={summary['max_holding_days']} psy={summary['min_psych_score']} "
                f"reg={summary['regime_key']:<17} mac={summary['macro_key']:<12} "
                f"→ trades={summary['trades']:>3} win={summary['win_rate']:>5.1%} "
                f"ret={summary['total_return_pct']:>+6.2%} avg_r={summary['avg_r']:>+5.2f} "
                f"[{summary['runtime_s']:>5.2f}s elapsed={elapsed/60:.1f}m "
                f"remain={remain/60:.1f}m]",
                flush=True,
            )

    # --- Final reports ---
    summary_df = pd.DataFrame(summary_rows)
    trades_df = pd.DataFrame(all_trade_rows)
    summary_df.to_csv(out_dir / "summary.csv", index=False)
    if not trades_df.empty:
        trades_df.to_parquet(out_dir / "trades.parquet", index=False)

    print()
    print("=" * 110)
    print("TOP 10 BY TOTAL RETURN")
    print("=" * 110)
    top = summary_df.sort_values("total_return_pct", ascending=False).head(10)
    print(top.to_string(index=False))

    print()
    print("=" * 110)
    print("TOP 10 BY PROFIT FACTOR (win_rate ≥ 0.35, trades ≥ 10)")
    print("=" * 110)
    filt = summary_df[
        (summary_df["win_rate"] >= 0.35) & (summary_df["trades"] >= 10)
    ]
    if not filt.empty:
        top_pf = filt.sort_values("profit_factor", ascending=False).head(10)
        print(top_pf.to_string(index=False))

    print()
    print(f"Saved: {out_dir}")
    print(f"  summary.csv ({len(summary_df)} rows)")
    print(f"  trades.parquet ({len(trades_df)} rows)")


if __name__ == "__main__":
    sys.exit(main() or 0)
