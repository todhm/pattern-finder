"""Smoke tests for the build_default_market_data factory.

Heavy network behaviour (actual Alpha Vantage / yfinance calls) lives in
the per-adapter test files. This file verifies wiring only:
- factory falls back to a yfinance-only path when no paid key is set
  (so dev / CI without secrets still loads pages)
- with an Alpha Vantage key, **daily** bars route AlphaVantage →
  YFinance (Mongo-cached each) and sub-daily routes AlphaVantage first.
"""

from __future__ import annotations

import pytest

from data.adapters.alphavantage_adapter import AlphaVantageAdapter
from data.adapters.cached_market_data import CachedMarketDataAdapter
from data.adapters.composed_market_data import (
    build_default_market_data,
    is_eodhd_configured,
)
from data.adapters.fallback_market_data import FallbackMarketDataAdapter
from data.adapters.interval_routing_market_data import (
    IntervalRoutingMarketData,
)
from data.adapters.yfinance_adapter import YFinanceAdapter

_KEYS = ("ALPHAVANTAGE_API_KEY", "EODHD_API_KEY", "MASSIVE_API_KEY")


@pytest.fixture
def no_keys(monkeypatch):
    for k in _KEYS:
        monkeypatch.delenv(k, raising=False)
    return monkeypatch


def test_falls_back_to_yfinance_when_no_api_key(no_keys) -> None:
    market = build_default_market_data()
    # No routing layer when fallback path is taken — the page sees a
    # plain cached yfinance adapter.
    assert isinstance(market, CachedMarketDataAdapter)
    assert isinstance(market._upstream, YFinanceAdapter)
    assert is_eodhd_configured() is False


def test_daily_routes_alphavantage_then_yfinance(no_keys) -> None:
    no_keys.setenv("ALPHAVANTAGE_API_KEY", "test-key")
    market = build_default_market_data()
    assert isinstance(market, IntervalRoutingMarketData)

    daily = market._daily
    assert isinstance(daily, FallbackMarketDataAdapter)
    assert isinstance(daily._primary._upstream, AlphaVantageAdapter)
    assert isinstance(daily._fallback._upstream, YFinanceAdapter)

    # sub-daily: AlphaVantage 체인 → yfinance (오늘 장중 갭)
    sub = market._sub_daily
    assert isinstance(sub._primary._upstream, AlphaVantageAdapter)
    assert isinstance(sub._fallback._upstream, YFinanceAdapter)
