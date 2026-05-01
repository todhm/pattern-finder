"""Factory composing the production market-data stack.

Pages should call :func:`build_default_market_data` instead of
hand-wiring the adapters. The factory:

1. Wraps each upstream source (yfinance, EODHD) with its own
   parquet-cache directory so cached parquets never cross-pollinate.
   Without per-source caching, a previously-cached yfinance-clamped
   sub-daily window would be served on subsequent calls even after
   we started routing sub-daily fetches to a paid provider — the
   cache decorator wrapping the router would short-circuit before
   the upstream ever saw the request.

2. Routes by interval: daily → yfinance (free, sufficient depth),
   sub-daily → EODHD (10y+ for 15m, 2y+ for 1m, KOSPI included).
   When ``EODHD_API_KEY`` isn't configured (dev / test), the factory
   silently falls back to yfinance for all intervals so the page
   still loads.

Cache layout::

    /tmp/pattern-finder-cache/                 # daily yfinance parquets
    /tmp/pattern-finder-cache/eodhd/           # EODHD sub-daily parquets
"""

from __future__ import annotations

from data.adapters.cached_market_data import CachedMarketDataAdapter
from data.adapters.eodhd_adapter import EODHDAdapter
from data.adapters.interval_routing_market_data import (
    IntervalRoutingMarketData,
)
from data.adapters.yfinance_adapter import YFinanceAdapter
from data.domain.ports import MarketDataPort


def build_default_market_data() -> MarketDataPort:
    """Return the routed + cached market-data stack used by every
    Streamlit page.

    Composition::

        IntervalRoutingMarketData(
            sub_daily = CachedMarketDataAdapter(EODHDAdapter(),
                                                cache_dir=".../eodhd"),
            daily     = CachedMarketDataAdapter(YFinanceAdapter()),
        )

    Falls back to ``CachedMarketDataAdapter(YFinanceAdapter())`` when
    EODHD is unavailable (no API key).
    """
    yf_cached = CachedMarketDataAdapter(YFinanceAdapter())
    try:
        eodhd_cached = CachedMarketDataAdapter(
            EODHDAdapter(),
            cache_dir="/tmp/pattern-finder-cache/eodhd",
        )
    except ValueError:
        return yf_cached
    return IntervalRoutingMarketData(
        sub_daily=eodhd_cached,
        daily=yf_cached,
    )


def is_eodhd_configured() -> bool:
    """Cheap check pages can use to surface a 'EODHD disabled'
    warning when the API key isn't loaded."""
    try:
        EODHDAdapter()
        return True
    except ValueError:
        return False
