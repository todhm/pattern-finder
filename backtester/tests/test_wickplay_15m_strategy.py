"""15-minute Wick Play strategy unit tests.

Mirrors ``test_wedgepop_15m_strategy.py``: pins the bar-identity
hook overrides (date default → intraday timestamp), the tz-
preserving ``_with_indicators`` switch, and the factory's
session-count defaults so an accidental refactor can't silently
drift the intraday calibration.
"""

from __future__ import annotations

from datetime import date, datetime

import pandas as pd
import pytest

from data.domain.ports import MarketDataPort
from pattern.domain.models import PatternSignal
from pattern.domain.ports import PatternDetector
from pattern.helpers.sessions import NY_TZ
from strategy.adapters.wickplay_15m_strategy import (
    Wickplay15mStrategy,
    build_wickplay_15m_detector,
)
from strategy.adapters.wickplay_strategy import WickPlayStrategy
from strategy.domain.models import Trade


class _FakeMarket(MarketDataPort):
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


def test_daily_hooks_preserve_legacy_keys() -> None:
    """Default WickPlayStrategy hooks must still produce midnight-
    Timestamp keys, matching the old ``set[date]`` semantics."""
    sig = PatternSignal(
        date=date(2024, 3, 18),
        pattern_name="wick_play",
        entry_price=100.0,
        stop_loss=99.0,
    )
    assert WickPlayStrategy._signal_key(sig) == pd.Timestamp("2024-03-18")
    assert WickPlayStrategy._signal_match_ts(sig) == pd.Timestamp("2024-03-18")


def test_15m_hooks_require_timestamp() -> None:
    no_ts = PatternSignal(
        date=date(2024, 3, 18),
        pattern_name="wick_play",
        entry_price=100.0,
        stop_loss=99.0,
    )
    with pytest.raises(ValueError, match="timestamp"):
        Wickplay15mStrategy._signal_key(no_ts)

    ts = datetime(2024, 3, 18, 10, 15, tzinfo=pd.Timestamp("now", tz=NY_TZ).tzinfo)
    with_ts = PatternSignal(
        date=date(2024, 3, 18),
        pattern_name="wick_play",
        entry_price=100.0,
        stop_loss=99.0,
        timestamp=ts,
    )
    assert Wickplay15mStrategy._signal_key(with_ts) == pd.Timestamp(ts)


def test_15m_strategy_keeps_index_tz() -> None:
    """Daily strategy strips tz to match pre-existing naive-datetime
    semantics; the 15m subclass must NOT strip — otherwise
    ``_next_open_index`` fails to match ``signal.timestamp``."""
    assert WickPlayStrategy._strip_index_tz is True
    assert Wickplay15mStrategy._strip_index_tz is False


def test_15m_run_forwards_interval() -> None:
    idx = pd.date_range(
        "2024-03-18 09:30", periods=30, freq="15min", tz=NY_TZ
    )
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
    from strategy.domain.models import StrategyConfig

    strat = Wickplay15mStrategy(market_data=market, detector=_NoopDetector())
    strat.run(
        StrategyConfig(
            ticker="TEST",
            start_date=date(2024, 3, 18),
            end_date=date(2024, 3, 18),
            pattern_name="wick_play",
            max_holding_days=26,
        )
    )
    assert market.last_interval == "15m"


def test_15m_strategy_defaults_rescaled_to_sessions() -> None:
    strat = Wickplay15mStrategy(
        market_data=_FakeMarket(pd.DataFrame()),
        detector=_NoopDetector(),
    )
    assert strat.ema_trail == 26        # 1 session
    assert strat.atr_period == 26       # 1 session
    assert strat.min_trail_bars == 8    # ~2 hours


def test_15m_detector_factory_rescales_lookbacks() -> None:
    """The WickPlayDetector takes day-named lookbacks; the 15m
    factory must convert them to session-count units. Pin the
    conversions so an accidental refactor doesn't silently revert
    to daily scale."""
    d = build_wickplay_15m_detector()
    # 20 daily bars → 20 sessions × 26 = 520 intraday bars.
    assert d.psych_vol_lookback == 520
    assert d.prior_trend_lookback == 520
    assert d.pct_high_lookback == 520
    # 14-day ATR → 1 session of intraday.
    assert d.atr_period == 26
    # 5-day cooldown → 1 session so no back-to-back intraday signals.
    assert d.cooldown_bars == 26


def test_15m_trade_exit_key_uses_exit_ts() -> None:
    exit_ts = datetime(
        2024, 3, 19, 15, 45, tzinfo=pd.Timestamp("now", tz=NY_TZ).tzinfo
    )
    trade = Trade(
        pattern_name="wick_play",
        entry_date=date(2024, 3, 18),
        exit_date=date(2024, 3, 19),
        entry_price=100.0,
        exit_price=105.0,
        stop_loss=99.0,
        shares=10,
        pnl=50.0,
        pnl_pct=0.05,
        exit_reason="ema_trail",
        entry_ts=datetime(2024, 3, 18, 9, 30, tzinfo=exit_ts.tzinfo),
        exit_ts=exit_ts,
    )
    assert Wickplay15mStrategy._trade_exit_key(trade) == pd.Timestamp(exit_ts)
    # Daily fallback: exit_date midnight.
    assert WickPlayStrategy._trade_exit_key(trade) == pd.Timestamp("2024-03-19")
