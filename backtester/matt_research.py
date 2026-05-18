"""Matt Diamond win-rate research scratchpad.

Iteratively measure win rate / total return / profit factor on a basket
of liquid large-caps under different parameter combinations. Saves
fetched DataFrames per ticker so the slow data-fetch only happens once.
"""

from __future__ import annotations

import json
import os
import pickle
import sys
from collections import defaultdict
from datetime import date, time, timedelta
from typing import Any

import pandas as pd

from data.adapters.composed_market_data import build_default_market_data
from data.adapters.regular_session_filter import RegularSessionFilterAdapter
from data.domain.market_calendar import NY
from pattern.adapters.matt_diamond_bull_flag import (
    MattDiamondBullFlagDetector,
    compute_market_regime_ok,
    compute_premarket_high_by_date,
)
from strategy.adapters.matt_diamond_bull_flag_strategy import (
    MattDiamondBullFlagStrategy,
)
from strategy.domain.models import StrategyConfig, TossFeeSchedule

TICKERS = ["NVDA", "TSLA", "QQQ", "AAPL", "AMD"]
START = date(2025, 1, 1)
END = date.today()
CACHE = "/tmp/matt_cache.pkl"


def fetch_universe() -> dict[str, dict[str, Any]]:
    """One-shot fetch + cache so subsequent runs are fast."""
    if os.path.exists(CACHE):
        with open(CACHE, "rb") as f:
            return pickle.load(f)
    md_raw = build_default_market_data()
    md = RegularSessionFilterAdapter(md_raw, market=NY)
    df_spy = md.fetch_ohlcv("SPY", START - timedelta(days=120), END, interval="1d")
    regime_ok = compute_market_regime_ok(df_spy, sma_period=50)
    store: dict[str, dict[str, Any]] = {"_regime": regime_ok}
    for t in TICKERS:
        print(f"  fetching {t}...", file=sys.stderr)
        df_intra = md.fetch_ohlcv(t, START, END, interval="1m")
        df_daily = md.fetch_ohlcv(t, START - timedelta(days=120), END, interval="1d")
        df_raw = md_raw.fetch_ohlcv(t, START, END, interval="1m")
        if df_intra.index.tz is not None and str(df_intra.index.tz) != NY.tz:
            df_intra.index = df_intra.index.tz_convert(NY.tz)
        if df_raw.index.tz is None:
            df_raw.index = df_raw.index.tz_localize(NY.tz)
        elif str(df_raw.index.tz) != NY.tz:
            df_raw.index = df_raw.index.tz_convert(NY.tz)
        pm = compute_premarket_high_by_date(df_raw)
        store[t] = {"intra": df_intra, "daily": df_daily, "pm": pm}
    with open(CACHE, "wb") as f:
        pickle.dump(store, f)
    return store


def run_one(
    store: dict[str, Any],
    *,
    detector_kwargs: dict[str, Any] | None = None,
    strategy_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run all tickers with given kwargs. Return aggregated metrics."""
    detector_kwargs = detector_kwargs or {}
    strategy_kwargs = strategy_kwargs or {}
    regime_ok = store["_regime"]

    by_ticker: dict[str, dict[str, Any]] = {}
    agg_trades = []
    for t in TICKERS:
        td = store[t]
        det_kw = dict(
            require_premarket_high=True,
            premarket_high_by_date=td["pm"],
            market_regime_ok_by_date={},
        )
        det_kw.update(detector_kwargs)
        det = MattDiamondBullFlagDetector(**det_kw)
        strat = MattDiamondBullFlagStrategy(det, fee_schedule=TossFeeSchedule(), **strategy_kwargs)
        cfg = StrategyConfig(
            ticker=t,
            start_date=START,
            end_date=END,
            pattern_name="matt_diamond_bull_flag",
            initial_capital=100_000.0,
            risk_per_trade=0.01,
        )
        result = strat.run(td["intra"], td["daily"], cfg)
        p = result.performance
        by_ticker[t] = {
            "n": p.total_trades,
            "win": p.win_rate,
            "ret": p.total_return_pct,
            "avg_win": p.avg_win_pct,
            "avg_loss": p.avg_loss_pct,
            "dd": p.max_drawdown_pct,
        }
        agg_trades.extend(p.trades)

    wins = [t for t in agg_trades if t.pnl > 0]
    losses = [t for t in agg_trades if t.pnl <= 0]
    total_pnl_w = sum(t.pnl for t in wins)
    total_pnl_l = abs(sum(t.pnl for t in losses)) or 1e-9
    pf = total_pnl_w / total_pnl_l
    win_rate = len(wins) / max(1, len(agg_trades))
    avg_win = sum(t.pnl_pct for t in wins) / max(1, len(wins))
    avg_loss = sum(t.pnl_pct for t in losses) / max(1, len(losses))
    return {
        "n": len(agg_trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": win_rate,
        "profit_factor": pf,
        "avg_win_pct": avg_win,
        "avg_loss_pct": avg_loss,
        "total_pnl": sum(t.pnl for t in agg_trades),
        "by_ticker": by_ticker,
        "trades": agg_trades,
    }


def print_summary(label: str, r: dict[str, Any]) -> None:
    print(
        f"{label:65s}  n={r['n']:>4d}  W={r['wins']:>3d}/L={r['losses']:>3d}  "
        f"win%={r['win_rate']*100:>5.1f}  PF={r['profit_factor']:>4.2f}  "
        f"avgW={r['avg_win_pct']*100:+5.2f}%  avgL={r['avg_loss_pct']*100:+5.2f}%  "
        f"PnL=${r['total_pnl']:>+9,.0f}"
    )
