"""Unit tests for IntervalRoutingMarketData."""

from __future__ import annotations

from datetime import date

import pandas as pd

from data.adapters.interval_routing_market_data import (
    IntervalRoutingMarketData,
)
from data.domain.ports import MarketDataPort


class _RecordingAdapter(MarketDataPort):
    def __init__(self, label: str):
        self.label = label
        self.calls: list[dict] = []

    def fetch_ohlcv(self, symbol, start, end, interval="1d"):
        self.calls.append(
            dict(symbol=symbol, start=start, end=end, interval=interval)
        )
        # Identifying frame so callers can confirm which adapter served.
        return pd.DataFrame({"src": [self.label]})


def _build():
    sub = _RecordingAdapter("massive")
    daily = _RecordingAdapter("yfinance")
    return IntervalRoutingMarketData(sub, daily), sub, daily


def test_daily_interval_routes_to_daily_adapter() -> None:
    router, sub, daily = _build()
    out = router.fetch_ohlcv(
        "AAPL", date(2024, 1, 1), date(2024, 12, 31), interval="1d"
    )
    assert out.iloc[0]["src"] == "yfinance"
    assert sub.calls == []
    assert len(daily.calls) == 1


def test_sub_daily_us_ticker_routes_to_sub_daily_adapter() -> None:
    router, sub, daily = _build()
    out = router.fetch_ohlcv(
        "AAPL", date(2024, 1, 1), date(2024, 1, 7), interval="15m"
    )
    assert out.iloc[0]["src"] == "massive"
    assert len(sub.calls) == 1
    assert daily.calls == []


def test_korean_ticker_sub_daily_routes_to_sub_daily_adapter() -> None:
    """Routing is interval-only; KR tickers go to Massive on
    sub-daily intervals just like US tickers. Whether Massive
    actually returns data is the upstream's concern — this layer
    doesn't pre-filter by ticker."""
    router, sub, daily = _build()
    for interval in ["1m", "5m", "15m", "30m"]:
        router.fetch_ohlcv(
            "005930.KS", date(2024, 1, 1), date(2024, 1, 7), interval=interval
        )
    assert len(sub.calls) == 4
    assert daily.calls == []


def test_korean_ticker_daily_still_routes_to_daily_adapter() -> None:
    router, sub, daily = _build()
    router.fetch_ohlcv(
        "005930.KS", date(2024, 1, 1), date(2024, 12, 31), interval="1d"
    )
    assert sub.calls == []
    assert len(daily.calls) == 1


def test_unknown_long_interval_falls_through_to_daily() -> None:
    """``1wk`` / ``1mo`` aren't useful on the sub-daily provider —
    route them to the daily adapter."""
    router, sub, daily = _build()
    for interval in ["1wk", "1mo"]:
        router.fetch_ohlcv(
            "AAPL", date(2024, 1, 1), date(2024, 12, 31), interval=interval
        )
    assert sub.calls == []
    assert len(daily.calls) == 2
