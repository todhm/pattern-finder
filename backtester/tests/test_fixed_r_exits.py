"""Fixed-R exit framework: take_profit + hard_initial_stop.

Pin the four invariants the 15m default exit set relies on:

1. ``take_profit_r_multiple=2.0`` exits when bar HIGH reaches
   ``entry + 2 × initial_risk`` — at the target, not at the bar's
   high (so a wide spike past the target doesn't gift extra fill).
2. ``enable_hard_initial_stop=True`` exits when bar LOW touches
   ``stop_loss`` — at the stop level, not at the low.
3. Daily defaults (both options OFF) preserve the legacy structural-
   only behavior — no new exits fire on a 1d-style frame.
4. Same-bar conflict between TP and stop resolves to TP (the loop
   checks TP first; intentional bias since intraday wedge pops are
   bullish setups and we want to bank gains rather than assume the
   worst).
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from data.domain.ports import MarketDataPort
from pattern.adapters.wedge_pop import WedgePopDetector
from pattern.domain.models import PatternSignal
from strategy.adapters.wedgepop_strategy import WedgepopStrategy
from strategy.domain.models import StrategyConfig


class _FakeMarket(MarketDataPort):
    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df

    def fetch_ohlcv(self, symbol, start, end, interval="1d"):
        return self._df


def _flat_df_then_spike(spike_high: float, n: int = 30) -> pd.DataFrame:
    """30-bar daily frame: flat at $100 for the warmup window, then a
    single bar with custom spike_high so we can dial which exit fires."""
    idx = pd.date_range("2024-03-01", periods=n, freq="B")
    closes = [100.0] * n
    highs = [100.5] * n
    lows = [99.5] * n
    opens = [100.0] * n
    # Bar at index 25 is the "spike bar" we'll target with TP.
    highs[25] = spike_high
    return pd.DataFrame(
        {
            "Open": opens,
            "High": highs,
            "Low": lows,
            "Close": closes,
            "Volume": [1_000] * n,
        },
        index=idx,
    )


def _flat_df_then_dip(dip_low: float, n: int = 30) -> pd.DataFrame:
    idx = pd.date_range("2024-03-01", periods=n, freq="B")
    closes = [100.0] * n
    highs = [100.5] * n
    lows = [99.5] * n
    opens = [100.0] * n
    lows[25] = dip_low
    return pd.DataFrame(
        {
            "Open": opens,
            "High": highs,
            "Low": lows,
            "Close": closes,
            "Volume": [1_000] * n,
        },
        index=idx,
    )


def _detector():
    # Concrete detector — needed for WedgepopStrategy to instantiate,
    # but we never actually run detect() in these tests.
    return WedgePopDetector(require_above_long_smas=False)


def _signal_at(df: pd.DataFrame, idx: int, entry_price: float, stop: float) -> PatternSignal:
    return PatternSignal(
        date=df.index[idx].date(),
        timestamp=pd.Timestamp(df.index[idx]).to_pydatetime(),
        pattern_name="wedge_pop",
        entry_price=entry_price,
        stop_loss=stop,
        metadata={"ema_slow_slope": 0.0},
    )


# ---- take_profit ---------------------------------------------------------


def test_take_profit_fires_when_high_reaches_target() -> None:
    """Bar's HIGH crossing entry + 2R must fire ``take_profit``
    at the target — even if the bar's high blew well past it."""
    # entry=100, stop=95, initial_risk=5. 2R target = 110. spike high = 115.
    df = _flat_df_then_spike(spike_high=115.0)
    strat = WedgepopStrategy(
        market_data=_FakeMarket(df),
        detector=_detector(),
        max_ema_slope_decline=None,
        take_profit_r_multiple=2.0,
        enable_hard_initial_stop=False,
    )
    df_ind = strat._with_indicators(df)
    signal = _signal_at(df_ind, idx=20, entry_price=100.0, stop=95.0)
    cfg = StrategyConfig(
        ticker="TEST",
        start_date=df_ind.index[0].date(),
        end_date=df_ind.index[-1].date(),
        pattern_name="wedge_pop",
    )
    trade, exit_idx = strat._execute_trade(
        df_ind, signal, entry_idx=21, capital=100_000.0, config=cfg
    )
    assert trade is not None
    assert trade.exit_reason == "take_profit"
    # Exit at the target price, not the spike high.
    assert trade.exit_price == 110.0
    assert exit_idx == 25


def test_take_profit_off_by_default() -> None:
    """Default ``take_profit_r_multiple=None`` must not fire even on
    a giant spike — preserves legacy 1d behavior bit-for-bit."""
    df = _flat_df_then_spike(spike_high=200.0)
    strat = WedgepopStrategy(
        market_data=_FakeMarket(df),
        detector=_detector(),
        max_ema_slope_decline=None,
    )
    df_ind = strat._with_indicators(df)
    signal = _signal_at(df_ind, idx=20, entry_price=100.0, stop=95.0)
    cfg = StrategyConfig(
        ticker="TEST",
        start_date=df_ind.index[0].date(),
        end_date=df_ind.index[-1].date(),
        pattern_name="wedge_pop",
    )
    trade, _ = strat._execute_trade(
        df_ind, signal, entry_idx=21, capital=100_000.0, config=cfg
    )
    assert trade is not None
    assert trade.exit_reason != "take_profit"


# ---- hard_initial_stop ---------------------------------------------------


def test_hard_stop_fires_when_low_touches_stop() -> None:
    df = _flat_df_then_dip(dip_low=94.0)
    strat = WedgepopStrategy(
        market_data=_FakeMarket(df),
        detector=_detector(),
        max_ema_slope_decline=None,
        enable_hard_initial_stop=True,
    )
    df_ind = strat._with_indicators(df)
    signal = _signal_at(df_ind, idx=20, entry_price=100.0, stop=95.0)
    cfg = StrategyConfig(
        ticker="TEST",
        start_date=df_ind.index[0].date(),
        end_date=df_ind.index[-1].date(),
        pattern_name="wedge_pop",
    )
    trade, exit_idx = strat._execute_trade(
        df_ind, signal, entry_idx=21, capital=100_000.0, config=cfg
    )
    assert trade is not None
    assert trade.exit_reason == "initial_stop"
    # Fill at the stop level (not the bar's low).
    assert trade.exit_price == 95.0
    assert exit_idx == 25


def test_hard_stop_off_by_default() -> None:
    """Daily callers leave the option off — even a deep dip should
    not fire ``initial_stop`` in the default config."""
    df = _flat_df_then_dip(dip_low=80.0)
    strat = WedgepopStrategy(
        market_data=_FakeMarket(df),
        detector=_detector(),
        max_ema_slope_decline=None,
    )
    df_ind = strat._with_indicators(df)
    signal = _signal_at(df_ind, idx=20, entry_price=100.0, stop=95.0)
    cfg = StrategyConfig(
        ticker="TEST",
        start_date=df_ind.index[0].date(),
        end_date=df_ind.index[-1].date(),
        pattern_name="wedge_pop",
    )
    trade, _ = strat._execute_trade(
        df_ind, signal, entry_idx=21, capital=100_000.0, config=cfg
    )
    assert trade is not None
    assert trade.exit_reason != "initial_stop"


# ---- ordering ------------------------------------------------------------


def test_same_bar_tp_and_stop_fires_take_profit() -> None:
    """A bar that pierces BOTH the stop and the target on the same
    candle resolves to take_profit — the loop checks TP first because
    the wedge-pop direction is bullish and we want to bank target gains
    rather than assume worst-case fill."""
    # Single bar: low=94 (below 95 stop), high=115 (above 110 target).
    n = 30
    idx = pd.date_range("2024-03-01", periods=n, freq="B")
    df = pd.DataFrame(
        {
            "Open": [100.0] * n,
            "High": [100.5] * n,
            "Low": [99.5] * n,
            "Close": [100.0] * n,
            "Volume": [1_000] * n,
        },
        index=idx,
    )
    df.iloc[25, df.columns.get_loc("Low")] = 94.0
    df.iloc[25, df.columns.get_loc("High")] = 115.0

    strat = WedgepopStrategy(
        market_data=_FakeMarket(df),
        detector=_detector(),
        max_ema_slope_decline=None,
        take_profit_r_multiple=2.0,
        enable_hard_initial_stop=True,
    )
    df_ind = strat._with_indicators(df)
    signal = _signal_at(df_ind, idx=20, entry_price=100.0, stop=95.0)
    cfg = StrategyConfig(
        ticker="TEST",
        start_date=df_ind.index[0].date(),
        end_date=df_ind.index[-1].date(),
        pattern_name="wedge_pop",
    )
    trade, _ = strat._execute_trade(
        df_ind, signal, entry_idx=21, capital=100_000.0, config=cfg
    )
    assert trade is not None
    assert trade.exit_reason == "take_profit"
    assert trade.exit_price == 110.0


# ---- 15m subclass defaults ----------------------------------------------


def test_15m_subclass_enables_fixed_r_framework_by_default() -> None:
    """The 15m subclass turns the framework on — pin the defaults so
    a refactor can't silently drift back to the daily structural-only
    behavior that wrecked the multiwedge.csv backtest."""
    from strategy.adapters.wedgepop_15m_strategy import Wedgepop15mStrategy

    class _NoopDet:
        name = "n"

        def detect(self, df, **k):
            return []

    strat = Wedgepop15mStrategy(market_data=_FakeMarket(pd.DataFrame()), detector=_NoopDet())
    assert strat.take_profit_r_multiple == 2.0
    assert strat.enable_hard_initial_stop is True
    assert strat.enable_breakeven_stop is True
    assert strat.use_smart_trail is False
