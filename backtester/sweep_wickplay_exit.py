"""Exit-focused sweep for Wick Play — Exhaustion + Breakeven grid.

Motivated by the 2017-2025 S&P 500 export analysis:

    - 73 trades, +$75.9K net, PF 1.36 — but highly concentrated.
    - Exhaustion Extension Top exits: 7 trades, 100% win rate, +$127.6K
      → the actual alpha source, but fires only 3.9% of the time.
    - Wick Low Stop: 17 trades, 0% win, -$145.1K.
    - Breakeven stop is OFF by default → wicks that briefly hit +1R then
      collapse turn into full -1R losers.

Hypothesis: Loosening ``extension_atr_mult`` (currently 2.5) should fire
Exhaustion sooner / more often; turning Breakeven ON with a moderate
arm multiple should convert a chunk of the -1R Wick Low Stops into
flat-to-small-green trades.

Entry / regime / macro held at the prior-sweep best (config_id=137):

    ema_trail=10, min_trail_bars=5, max_holding_days=40,
    min_psych_score=3, regime=sma50, macro=cpi_w1.

Grid: 5 × 3 × 4 = 60 configs. ~15-20 min on 8 workers with pre-fetch.

Run inside the container::

    docker compose exec backtester python sweep_wickplay_exit.py
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

START_DATE = date(2017, 1, 1)
END_DATE = date(2025, 12, 31)
UNIVERSE = "sp500"  # Wikipedia S&P 500 constituents, ~503 tickers
MAX_TICKERS = 503
INITIAL_CAPITAL = 100_000.0
RISK_PER_TRADE = 0.02
MAX_HOLDING_DAYS = 40
REGIME_SYMBOL = "^GSPC"

# Entry / regime / macro fixed to prior-sweep optimum.
FIXED_EMA_TRAIL = 10
FIXED_MIN_TRAIL_BARS = 5
FIXED_MIN_PSYCH_SCORE = 3
FIXED_REGIME_SMA = 50
FIXED_MACRO = {"fomc": False, "cpi": True, "window": 1}

# ---- Grid dimensions -------------------------------------------------

EXTENSION_ATR_MULTS = [1.5, 1.8, 2.0, 2.2, 2.5]
MIN_SLOW_SLOPES = [0.0, 0.005, 0.01]

# (label, enabled, arm_r, offset_r)
BREAKEVEN_SPECS: list[tuple[str, bool, float, float]] = [
    ("off",            False, 1.5, 0.5),  # no BE
    ("arm1.0_off0.0",  True,  1.0, 0.0),  # pure breakeven
    ("arm1.5_off0.5",  True,  1.5, 0.5),  # current default when on
    ("arm2.0_off1.0",  True,  2.0, 1.0),  # lock +1R after +2R
]


# ---- Cache-aware runner ---------------------------------------------


class _CachedMultiWickPlay(MultiWickPlayStrategy):
    def set_state(self, ticker_state, failed):
        self._cached_state = ticker_state
        self._cached_failed = failed

    def _scan_universe(self, tickers, config):
        return self._cached_state, list(self._cached_failed)


class _StaticUniverse:
    def __init__(self, tickers):
        self._tickers = list(tickers)

    def get_tickers(self, universe):
        return list(self._tickers)


# ---- Grid config container -------------------------------------------


@dataclass(frozen=True)
class GridConfig:
    config_id: int
    extension_atr_mult: float
    min_slow_slope: float
    breakeven_label: str
    breakeven_enabled: bool
    breakeven_arm_r: float
    breakeven_offset_r: float


def build_grid() -> list[GridConfig]:
    rows = []
    cid = 0
    for ext, slope, be in itertools.product(
        EXTENSION_ATR_MULTS, MIN_SLOW_SLOPES, BREAKEVEN_SPECS
    ):
        be_label, be_en, arm, offset = be
        rows.append(
            GridConfig(
                config_id=cid,
                extension_atr_mult=ext,
                min_slow_slope=slope,
                breakeven_label=be_label,
                breakeven_enabled=be_en,
                breakeven_arm_r=arm,
                breakeven_offset_r=offset,
            )
        )
        cid += 1
    return rows


# ---- Detector / strategy construction --------------------------------


def build_detector(
    regime_df: pd.DataFrame,
    blackout_set: set[date] | None,
) -> WickPlayDetector:
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
        min_psych_score=FIXED_MIN_PSYCH_SCORE,
        min_breakout_strength_atr=0.3,
        min_prior_trend_20d=-0.01,
        min_wick_close_location=0.15,
        regime_df=regime_df,
        regime_min_above_sma=FIXED_REGIME_SMA,
        blackout_dates=blackout_set,
    )


def build_exit_detector(cfg: GridConfig) -> ExhaustionExtensionTopDetector:
    return ExhaustionExtensionTopDetector(
        extension_atr_mult=cfg.extension_atr_mult,
        min_slow_slope=cfg.min_slow_slope,
        ema_fast=FIXED_EMA_TRAIL,
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
        ema_trail=FIXED_EMA_TRAIL,
        atr_period=14,
        min_trail_bars=FIXED_MIN_TRAIL_BARS,
        enable_same_day_reversal_exit=False,
        enable_gap_down_rejection=True,
        max_entry_gap_down=0.005,
        enable_breakeven_stop=cfg.breakeven_enabled,
        breakeven_arm_r_multiple=cfg.breakeven_arm_r,
        breakeven_exit_offset_r=cfg.breakeven_offset_r,
    )


# ---- Pre-fetch + caches ---------------------------------------------


def prefetch_all(
    tickers: list[str],
    market_data: CachedMarketDataAdapter,
    min_bars: int,
) -> dict[str, pd.DataFrame]:
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
    raw_dfs: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Compute indicators ONCE per ticker — ema_trail is fixed at 10
    for this sweep, so a single cache per ticker suffices."""
    stub = WickPlayStrategy(
        market_data=None,  # type: ignore[arg-type]
        detector=None,     # type: ignore[arg-type]
        ema_trail=FIXED_EMA_TRAIL,
    )
    return {t: stub._with_indicators(df) for t, df in raw_dfs.items()}


def build_exit_dates_cache(
    df_ind_cache: dict[str, pd.DataFrame],
) -> dict[tuple[float, float], dict[str, set[date]]]:
    """Pre-compute exhaustion exit_dates for each ``(extension_atr_mult,
    min_slow_slope)`` combination. 5 × 3 = 15 unique exhaustion configs
    × ~500 tickers = one-time cost.
    """
    cache: dict[tuple[float, float], dict[str, set[date]]] = {}
    combos = list(itertools.product(EXTENSION_ATR_MULTS, MIN_SLOW_SLOPES))
    for i, (ext, slope) in enumerate(combos):
        exh = ExhaustionExtensionTopDetector(
            extension_atr_mult=ext,
            min_slow_slope=slope,
            ema_fast=FIXED_EMA_TRAIL,
        )
        by_ticker: dict[str, set[date]] = {}
        for ticker, df_ind in df_ind_cache.items():
            by_ticker[ticker] = {s.date for s in exh.detect(df_ind)}
        cache[(ext, slope)] = by_ticker
        print(
            f"  exhaustion combo {i+1}/{len(combos)} "
            f"(ext={ext}, slope={slope}): "
            f"{sum(len(v) for v in by_ticker.values())} fire events",
            flush=True,
        )
    return cache


# ---- Result row building --------------------------------------------


def run_one_config(
    cfg: GridConfig,
    raw_dfs: dict[str, pd.DataFrame],
    df_ind_cache: dict[str, pd.DataFrame],
    exit_dates_cache: dict[tuple[float, float], dict[str, set[date]]],
    blackout_set: set[date] | None,
    regime_df: pd.DataFrame,
    market_data: CachedMarketDataAdapter,
    multi_cfg_template: MultiStrategyConfig,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    detector = build_detector(regime_df, blackout_set)
    exit_detector = build_exit_detector(cfg)
    strategy = build_strategy(market_data, detector, exit_detector, cfg)

    exit_dates_by_ticker = exit_dates_cache[
        (cfg.extension_atr_mult, cfg.min_slow_slope)
    ]

    ticker_state: dict[str, dict[str, Any]] = {}
    for ticker in raw_dfs:
        df_ind = df_ind_cache[ticker]
        signals = detector.detect(df_ind)
        ticker_state[ticker] = {
            "df": df_ind,
            "signals": signals,
            "exit_dates": exit_dates_by_ticker.get(ticker, set()),
        }

    runner = _CachedMultiWickPlay(
        market_data=market_data,
        universe_provider=_StaticUniverse(list(raw_dfs.keys())),
        detector=detector,
        strategy=strategy,
        max_workers=1,
    )
    runner.set_state(ticker_state, failed=[])

    result = runner.run(multi_cfg_template)

    trade_rows: list[dict[str, Any]] = []
    wins = losses = 0
    total_win_pct = total_loss_pct = 0.0
    total_r = 0.0
    total_hold = 0
    exit_counts = {"exhaustion_exit": 0, "ema_trail": 0, "wick_low_stop": 0,
                   "breakeven_stop": 0, "time_stop": 0, "end_of_data": 0,
                   "same_day_reversal": 0}
    for t in result.trades:
        risk_per_share = t.entry_price - t.stop_loss
        r_multiple = (
            (t.pnl / t.shares) / risk_per_share
            if risk_per_share > 0 else 0.0
        )
        days_held = (t.exit_date - t.entry_date).days
        if t.pnl > 0:
            wins += 1
            total_win_pct += t.pnl_pct
        else:
            losses += 1
            total_loss_pct += t.pnl_pct
        total_r += r_multiple
        total_hold += days_held
        exit_counts[t.exit_reason] = exit_counts.get(t.exit_reason, 0) + 1
        trade_rows.append({
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
        })

    n = len(result.trades)
    gross_win = sum(t.pnl for t in result.trades if t.pnl > 0)
    gross_loss = -sum(t.pnl for t in result.trades if t.pnl < 0)
    profit_factor = (
        gross_win / gross_loss if gross_loss > 0
        else (float("inf") if gross_win > 0 else 0.0)
    )

    summary = {
        "config_id": cfg.config_id,
        "extension_atr_mult": cfg.extension_atr_mult,
        "min_slow_slope": cfg.min_slow_slope,
        "breakeven_label": cfg.breakeven_label,
        "breakeven_arm_r": cfg.breakeven_arm_r if cfg.breakeven_enabled else None,
        "breakeven_offset_r": cfg.breakeven_offset_r if cfg.breakeven_enabled else None,
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
        "profit_factor": (
            round(profit_factor, 4) if profit_factor != float("inf") else None
        ),
        "avg_hold_days": round(total_hold / n, 1) if n else 0.0,
        "exhaustion_exits": exit_counts.get("exhaustion_exit", 0),
        "ema_trail_exits": exit_counts.get("ema_trail", 0),
        "wick_stop_exits": exit_counts.get("wick_low_stop", 0),
        "breakeven_exits": exit_counts.get("breakeven_stop", 0),
    }
    return summary, trade_rows


# ---- Pool worker plumbing -------------------------------------------

_G_RAW_DFS: dict[str, pd.DataFrame] = {}
_G_DF_IND_CACHE: dict[str, pd.DataFrame] = {}
_G_EXIT_DATES_CACHE: dict[tuple[float, float], dict[str, set[date]]] = {}
_G_BLACKOUT_SET: set[date] | None = None
_G_REGIME_DF: pd.DataFrame | None = None
_G_MULTI_CFG: MultiStrategyConfig | None = None


def _worker(cfg: GridConfig) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    market_data = CachedMarketDataAdapter(YFinanceAdapter())
    t0 = time.monotonic()
    summary, trades = run_one_config(
        cfg=cfg,
        raw_dfs=_G_RAW_DFS,
        df_ind_cache=_G_DF_IND_CACHE,
        exit_dates_cache=_G_EXIT_DATES_CACHE,
        blackout_set=_G_BLACKOUT_SET,
        regime_df=_G_REGIME_DF,  # type: ignore[arg-type]
        market_data=market_data,
        multi_cfg_template=_G_MULTI_CFG,  # type: ignore[arg-type]
    )
    summary["runtime_s"] = round(time.monotonic() - t0, 2)
    return summary, trades


# ---- Main ------------------------------------------------------------


def main() -> None:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = (
        Path(__file__).resolve().parent
        / "sweep_results"
        / f"exit_{ts}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[sweep] output dir: {out_dir}", flush=True)

    grid = build_grid()
    print(f"[sweep] total configs: {len(grid)}", flush=True)

    (out_dir / "config.json").write_text(
        json.dumps(
            {
                "start_date": str(START_DATE),
                "end_date": str(END_DATE),
                "universe": UNIVERSE,
                "max_tickers": MAX_TICKERS,
                "initial_capital": INITIAL_CAPITAL,
                "risk_per_trade": RISK_PER_TRADE,
                "regime_symbol": REGIME_SYMBOL,
                "fixed": {
                    "ema_trail": FIXED_EMA_TRAIL,
                    "min_trail_bars": FIXED_MIN_TRAIL_BARS,
                    "min_psych_score": FIXED_MIN_PSYCH_SCORE,
                    "regime_sma": FIXED_REGIME_SMA,
                    "macro": FIXED_MACRO,
                },
                "dimensions": {
                    "extension_atr_mult": EXTENSION_ATR_MULTS,
                    "min_slow_slope": MIN_SLOW_SLOPES,
                    "breakeven": [s[0] for s in BREAKEVEN_SPECS],
                },
                "n_configs": len(grid),
            },
            indent=2,
        )
    )

    market_data = CachedMarketDataAdapter(YFinanceAdapter())
    universe_provider = default_universe_provider()
    tickers = sorted(universe_provider.get_tickers(UNIVERSE))[:MAX_TICKERS]
    print(f"[sweep] tickers: {len(tickers)}", flush=True)

    print(f"[sweep] fetching {REGIME_SYMBOL}...", flush=True)
    regime_df = market_data.fetch_ohlcv(
        REGIME_SYMBOL, START_DATE - timedelta(days=400), END_DATE
    )
    print(f"[sweep] regime rows: {len(regime_df)}", flush=True)

    t0 = time.monotonic()
    print("[sweep] prefetching ticker OHLCV...", flush=True)
    raw_dfs = prefetch_all(tickers, market_data, min_bars=15)
    print(
        f"[sweep] fetched {len(raw_dfs)} usable tickers in "
        f"{time.monotonic() - t0:.1f}s", flush=True,
    )

    t0 = time.monotonic()
    df_ind_cache = build_indicator_cache(raw_dfs)
    print(
        f"[sweep] indicators: {len(df_ind_cache)} frames in "
        f"{time.monotonic() - t0:.1f}s", flush=True,
    )

    t0 = time.monotonic()
    print("[sweep] pre-computing exhaustion exit dates...", flush=True)
    exit_dates_cache = build_exit_dates_cache(df_ind_cache)
    print(
        f"[sweep] exhaustion cache: 15 combos × {len(df_ind_cache)} tickers "
        f"in {time.monotonic() - t0:.1f}s", flush=True,
    )

    blackout_set = build_blackout_dates(
        start=START_DATE - timedelta(days=400),
        end=END_DATE,
        include_fomc=FIXED_MACRO["fomc"],
        include_cpi=FIXED_MACRO["cpi"],
        window_days=FIXED_MACRO["window"],
    )
    print(f"[sweep] blackout dates: {len(blackout_set)}", flush=True)

    fee_schedule = TossFeeSchedule(
        buy_commission_pct=0.001,
        sell_commission_pct=0.001,
        sec_fee_pct=0.0000229,
    )
    multi_cfg = MultiStrategyConfig(
        universe=UNIVERSE,
        start_date=START_DATE,
        end_date=END_DATE,
        pattern_name="wick_play",
        initial_capital=INITIAL_CAPITAL,
        risk_per_trade=RISK_PER_TRADE,
        max_holding_days=MAX_HOLDING_DAYS,
        max_tickers=len(raw_dfs),
        fee_schedule=fee_schedule,
    )

    global _G_RAW_DFS, _G_DF_IND_CACHE, _G_EXIT_DATES_CACHE
    global _G_BLACKOUT_SET, _G_REGIME_DF, _G_MULTI_CFG
    _G_RAW_DFS = raw_dfs
    _G_DF_IND_CACHE = df_ind_cache
    _G_EXIT_DATES_CACHE = exit_dates_cache
    _G_BLACKOUT_SET = blackout_set
    _G_REGIME_DF = regime_df
    _G_MULTI_CFG = multi_cfg

    n_workers = min(int(os.environ.get("SWEEP_WORKERS", 8)), os.cpu_count() or 4)
    print(f"[sweep] launching pool with {n_workers} workers", flush=True)
    ctx = mp.get_context("fork")

    summary_rows: list[dict[str, Any]] = []
    all_trade_rows: list[dict[str, Any]] = []
    completed = 0
    sweep_start = time.monotonic()

    with ctx.Pool(processes=n_workers) as pool:
        for summary, trades in pool.imap_unordered(_worker, grid, chunksize=1):
            summary_rows.append(summary)
            all_trade_rows.extend(trades)
            completed += 1

            if completed % 10 == 0 or completed == len(grid):
                pd.DataFrame(summary_rows).to_csv(
                    out_dir / "summary.csv", index=False
                )
                if all_trade_rows:
                    pd.DataFrame(all_trade_rows).to_parquet(
                        out_dir / "trades.parquet", index=False
                    )

            elapsed = time.monotonic() - sweep_start
            rate = completed / elapsed
            remain = (len(grid) - completed) / rate if rate > 0 else 0.0
            print(
                f"[{completed:>3}/{len(grid)}] id={summary['config_id']:>3} "
                f"ext={summary['extension_atr_mult']} slope={summary['min_slow_slope']} "
                f"be={summary['breakeven_label']:<13} → "
                f"tr={summary['trades']:>3} win={summary['win_rate']:>5.1%} "
                f"ret={summary['total_return_pct']:>+7.2%} "
                f"exh={summary['exhaustion_exits']:>2} wstop={summary['wick_stop_exits']:>2} "
                f"be_ex={summary['breakeven_exits']:>2} "
                f"[{summary['runtime_s']:>5.2f}s elapsed={elapsed/60:.1f}m "
                f"remain={remain/60:.1f}m]", flush=True,
            )

    summary_df = pd.DataFrame(summary_rows)
    trades_df = pd.DataFrame(all_trade_rows)
    summary_df.to_csv(out_dir / "summary.csv", index=False)
    if not trades_df.empty:
        trades_df.to_parquet(out_dir / "trades.parquet", index=False)

    print()
    print("=" * 120)
    print("TOP 10 BY TOTAL RETURN")
    print("=" * 120)
    top = summary_df.sort_values("total_return_pct", ascending=False).head(10)
    print(top[[
        "config_id", "extension_atr_mult", "min_slow_slope",
        "breakeven_label", "trades", "win_rate", "total_return_pct",
        "max_drawdown_pct", "avg_r", "profit_factor",
        "exhaustion_exits", "wick_stop_exits", "breakeven_exits",
    ]].to_string(index=False))

    print()
    print("=" * 120)
    print("EXHAUSTION HIT COUNT BY extension_atr_mult (averaged over slope & breakeven)")
    print("=" * 120)
    by_ext = summary_df.groupby("extension_atr_mult").agg(
        avg_exh=("exhaustion_exits", "mean"),
        avg_wstop=("wick_stop_exits", "mean"),
        avg_ret=("total_return_pct", "mean"),
        avg_win=("win_rate", "mean"),
    ).round(4)
    print(by_ext.to_string())

    print()
    print("=" * 120)
    print("BREAKEVEN EFFECT (averaged over extension_atr_mult & slope)")
    print("=" * 120)
    by_be = summary_df.groupby("breakeven_label").agg(
        avg_ret=("total_return_pct", "mean"),
        avg_win=("win_rate", "mean"),
        avg_wstop=("wick_stop_exits", "mean"),
        avg_be_ex=("breakeven_exits", "mean"),
        avg_dd=("max_drawdown_pct", "mean"),
    ).round(4)
    print(by_be.to_string())

    print()
    print(f"Saved: {out_dir}")
    print(f"  summary.csv ({len(summary_df)} rows)")
    print(f"  trades.parquet ({len(trades_df)} rows)")


if __name__ == "__main__":
    sys.exit(main() or 0)
