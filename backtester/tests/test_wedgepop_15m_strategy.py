"""15-minute Wedge Pop strategy unit tests.

Scope: the B-plan guarantees — that intraday subclasses can override
the four bar-identity hooks on ``WedgepopStrategy`` to key by exact
``DatetimeIndex`` values instead of session dates, and that the
inherited entry/exit/sizing logic still resolves correctly on a 15m
frame. We also pin the "daily Trade always carries ts fields now"
data-upgrade so downstream UI code can rely on it.
"""

from __future__ import annotations

from datetime import date, datetime

import pandas as pd
import pytest

from data.domain.ports import MarketDataPort
from pattern.domain.models import PatternSignal
from pattern.domain.ports import PatternDetector
from pattern.helpers.sessions import NY_TZ
from strategy.adapters.wedgepop_15m_strategy import Wedgepop15mStrategy
from strategy.adapters.wedgepop_strategy import WedgepopStrategy
from strategy.domain.models import StrategyConfig, Trade


class _FakeMarket(MarketDataPort):
    """Records the interval it was called with so we can assert
    that the inherited ``run()`` path forwards ``_interval`` to the
    port."""

    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df
        self.last_interval: str | None = None

    def fetch_ohlcv(self, symbol, start, end, interval: str = "1d"):
        self.last_interval = interval
        return self._df


class _NoopDetector(PatternDetector):
    name = "noop"

    def detect(self, df, weekly_df=None, monthly_df=None):
        return []


def _intraday_df(n_sessions: int = 3) -> pd.DataFrame:
    """Build an ``n_sessions``-long 15m frame (26 bars per session)
    with a gentle uptrend so every filter stays inert."""
    frames = []
    base = 100.0
    for day_offset in range(n_sessions):
        day = pd.Timestamp("2024-03-18", tz=NY_TZ) + pd.Timedelta(days=day_offset)
        idx = pd.date_range(
            start=day.replace(hour=9, minute=30),
            periods=26,
            freq="15min",
            tz=NY_TZ,
        )
        closes = [base + day_offset * 5 + i * 0.1 for i in range(26)]
        frames.append(
            pd.DataFrame(
                {
                    "Open": [c - 0.05 for c in closes],
                    "High": [c + 0.20 for c in closes],
                    "Low": [c - 0.20 for c in closes],
                    "Close": closes,
                    "Volume": [1_000 + i for i in range(26)],
                },
                index=idx,
            )
        )
    return pd.concat(frames)


# ---- bar-identity hooks -------------------------------------------------


def test_daily_defaults_preserve_legacy_keys() -> None:
    """Default WedgepopStrategy hooks must still produce the exact
    keys the old date-based code used: midnight Timestamp of the
    signal's session and the bar's session date."""
    sig = PatternSignal(
        date=date(2024, 3, 18),
        pattern_name="wedge_pop",
        entry_price=100.0,
        stop_loss=99.0,
    )
    assert WedgepopStrategy._signal_key(sig) == pd.Timestamp("2024-03-18 00:00:00")
    assert WedgepopStrategy._signal_match_ts(sig) == pd.Timestamp(
        "2024-03-18 00:00:00"
    )

    df = _intraday_df(1)
    # Intraday timestamps collapse to midnight under the 1d default —
    # same behavior as the original ``df.index[i].date()``.
    assert WedgepopStrategy._bar_key(df, 5) == pd.Timestamp("2024-03-18 00:00:00")


def test_15m_hooks_require_timestamp_and_preserve_resolution() -> None:
    """The 15m subclass rejects signals without an intraday timestamp
    and preserves bar-level resolution in both signal and bar keys
    so matches land on the exact bar, not the whole session."""
    no_ts = PatternSignal(
        date=date(2024, 3, 18),
        pattern_name="wedge_pop",
        entry_price=100.0,
        stop_loss=99.0,
    )
    with pytest.raises(ValueError, match="timestamp"):
        Wedgepop15mStrategy._signal_key(no_ts)
    with pytest.raises(ValueError, match="timestamp"):
        Wedgepop15mStrategy._signal_match_ts(no_ts)

    ts = datetime(2024, 3, 18, 10, 15, tzinfo=pd.Timestamp("now", tz=NY_TZ).tzinfo)
    with_ts = PatternSignal(
        date=date(2024, 3, 18),
        pattern_name="wedge_pop",
        entry_price=100.0,
        stop_loss=99.0,
        timestamp=ts,
    )
    key = Wedgepop15mStrategy._signal_key(with_ts)
    assert key == pd.Timestamp(ts)
    # Different intraday bars on the same session must produce
    # distinct keys — this is the whole reason for the override.
    df = _intraday_df(1)
    assert Wedgepop15mStrategy._bar_key(df, 3) != Wedgepop15mStrategy._bar_key(df, 4)


def test_15m_exit_key_uses_trade_exit_ts() -> None:
    exit_ts = datetime(2024, 3, 19, 15, 45, tzinfo=pd.Timestamp("now", tz=NY_TZ).tzinfo)
    trade = Trade(
        pattern_name="wedge_pop",
        entry_date=date(2024, 3, 18),
        exit_date=date(2024, 3, 19),
        entry_price=100.0,
        exit_price=105.0,
        stop_loss=99.0,
        shares=10,
        pnl=50.0,
        pnl_pct=0.05,
        exit_reason="end_of_data",
        entry_ts=datetime(2024, 3, 18, 9, 30, tzinfo=exit_ts.tzinfo),
        exit_ts=exit_ts,
    )
    assert Wedgepop15mStrategy._trade_exit_key(trade) == pd.Timestamp(exit_ts)

    # Daily default falls back to exit_date (midnight Timestamp).
    assert WedgepopStrategy._trade_exit_key(trade) == pd.Timestamp("2024-03-19")


# ---- run() forwards interval -------------------------------------------


def test_15m_run_forwards_interval_to_market_data() -> None:
    df = _intraday_df(2)
    market = _FakeMarket(df)
    strat = Wedgepop15mStrategy(market_data=market, detector=_NoopDetector())
    cfg = StrategyConfig(
        ticker="TEST",
        start_date=date(2024, 3, 18),
        end_date=date(2024, 3, 20),
        pattern_name="wedge_pop",
        initial_capital=100_000.0,
        risk_per_trade=0.02,
        max_holding_days=520,
    )
    strat.run(cfg)
    assert market.last_interval == "15m"


def test_daily_run_still_forwards_1d() -> None:
    """Inherited run() on the base class must still ask for 1d — the
    generalization is a pure kwarg addition."""
    # Minimal daily frame — detector is a no-op so no signal paths.
    idx = pd.date_range("2023-01-02", periods=30, freq="B")
    df = pd.DataFrame(
        {
            "Open": range(30),
            "High": [x + 1 for x in range(30)],
            "Low": [x - 1 for x in range(30)],
            "Close": [x + 0.5 for x in range(30)],
            "Volume": [1_000] * 30,
        },
        index=idx,
    )
    market = _FakeMarket(df)
    strat = WedgepopStrategy(market_data=market, detector=_NoopDetector())
    cfg = StrategyConfig(
        ticker="TEST",
        start_date=date(2023, 1, 2),
        end_date=date(2023, 2, 10),
        pattern_name="wedge_pop",
    )
    strat.run(cfg)
    assert market.last_interval == "1d"


# ---- daily Trade.ts population -----------------------------------------


def test_daily_trades_always_carry_ts_fields() -> None:
    """After the generalization, even the 1d path populates
    ``entry_ts`` / ``exit_ts`` on every Trade. Downstream consumers
    (intraday-capable UIs, serializers) can rely on the fields never
    being None after a strategy run."""
    idx = pd.date_range("2023-01-02", periods=30, freq="B")
    df = pd.DataFrame(
        {
            "Open": range(30),
            "High": [x + 1 for x in range(30)],
            "Low": [x - 1 for x in range(30)],
            "Close": [x + 0.5 for x in range(30)],
            "Volume": [1_000] * 30,
        },
        index=idx,
    )
    # Use the force_entry path — cheapest way to synthesize a trade
    # without fighting the detector.
    strat = WedgepopStrategy(market_data=_FakeMarket(df), detector=_NoopDetector())
    df_ind = strat._with_indicators(df)
    signal = PatternSignal(
        date=df_ind.index[5].date(),
        pattern_name="wedge_pop",
        entry_price=float(df_ind["Close"].iloc[5]),
        stop_loss=float(df_ind["Low"].iloc[5]) - 1.0,
    )
    cfg = StrategyConfig(
        ticker="TEST",
        start_date=df_ind.index[0].date(),
        end_date=df_ind.index[-1].date(),
        pattern_name="wedge_pop",
    )
    trade, _ = strat._execute_trade(df_ind, signal, entry_idx=6, capital=100_000.0, config=cfg)
    assert trade is not None
    assert trade.entry_ts is not None
    assert trade.exit_ts is not None
    assert trade.entry_ts.date() == trade.entry_date
    assert trade.exit_ts.date() == trade.exit_date


# ---- subclass defaults ---------------------------------------------------


def test_15m_defaults_rescale_indicator_periods() -> None:
    """Pin the session-count mapping so accidental refactors don't
    silently drift the intraday calibration."""
    strat = Wedgepop15mStrategy(
        market_data=_FakeMarket(_intraday_df(1)),
        detector=_NoopDetector(),
    )
    assert strat.ema_trail == 26  # 1 session
    assert strat.ema_slow == 78  # 3 sessions
    assert strat.atr_period == 26
    assert strat.swing_pivot_left == 4
    assert strat.swing_pivot_right == 4
    assert strat.swing_pivot_lookback == 520
    # Slope filter is off by default — the 1d -1% threshold over
    # 20 bars does not translate to a meaningful intraday gate.
    assert strat.min_ema_slow_slope is None
    assert strat.max_ema_slow_slope is None
