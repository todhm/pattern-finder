"""Smoke tests for the build_default_market_data factory.

Heavy network behaviour (actual Massive / yfinance calls) lives in
the per-adapter test files. This file verifies wiring only:
- factory falls back to a yfinance-only path when the API key is
  missing (so dev / CI without secrets still loads pages)
- factory composes the routing layer when the key is present, with
  a Massive cache that lives in its own directory so previously-
  cached yfinance-clamped sub-daily parquets can't be served.
"""

from __future__ import annotations

from data.adapters.cached_market_data import CachedMarketDataAdapter
from data.adapters.composed_market_data import (
    build_default_market_data,
    is_eodhd_configured,
)
from data.adapters.interval_routing_market_data import (
    IntervalRoutingMarketData,
)
from data.adapters.yfinance_adapter import YFinanceAdapter


def test_falls_back_to_yfinance_when_no_api_key(monkeypatch) -> None:
    monkeypatch.delenv("EODHD_API_KEY", raising=False)
    market = build_default_market_data()
    # No routing layer when fallback path is taken — the page sees a
    # plain cached yfinance adapter.
    assert isinstance(market, CachedMarketDataAdapter)
    assert isinstance(market._upstream, YFinanceAdapter)
    assert is_eodhd_configured() is False


def test_composes_routing_when_api_key_present(monkeypatch) -> None:
    monkeypatch.setenv("EODHD_API_KEY", "test-key")
    market = build_default_market_data()
    assert isinstance(market, IntervalRoutingMarketData)

    # Sub-daily branch wraps EODHD in its own cache (separate dir
    # so legacy yfinance parquets don't bleed in).
    sub = market._sub_daily
    assert isinstance(sub, CachedMarketDataAdapter)
    assert sub._cache_dir.name == "eodhd"

    # Daily branch keeps the legacy /tmp/pattern-finder-cache so
    # existing yfinance daily caches stay valid.
    daily = market._daily
    assert isinstance(daily, CachedMarketDataAdapter)
    assert isinstance(daily._upstream, YFinanceAdapter)
    assert is_eodhd_configured() is True
