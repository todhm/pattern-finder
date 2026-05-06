"""Factory composing the production market-data stack.

Pages should call :func:`build_default_market_data` instead of
hand-wiring the adapters. The factory:

1. Wraps each upstream source (yfinance, EODHD, Massive) with its
   own parquet-cache directory so cached parquets never cross-
   pollinate. Without per-source caching, a previously-cached
   yfinance-clamped sub-daily window would be served on subsequent
   calls even after we started routing sub-daily fetches to a paid
   provider.

2. Routes by interval: daily → yfinance (free, sufficient depth),
   sub-daily → **EODHD with Massive fallback**. When EODHD's daily
   quota is exhausted (HTTP 402) or any other call fails, the
   fallback transparently routes to Massive/Polygon — which has
   10y+ depth on US equities and a separate quota.

3. When neither EODHD nor MASSIVE keys are configured, falls back
   to yfinance for all intervals.

Cache layout::

    /tmp/pattern-finder-cache/                 # daily yfinance parquets
    /tmp/pattern-finder-cache/eodhd/           # EODHD sub-daily parquets
    /tmp/pattern-finder-cache/massive/         # Massive sub-daily parquets
"""

from __future__ import annotations

from data.adapters.cached_market_data import CachedMarketDataAdapter
from data.adapters.eodhd_adapter import EODHDAdapter
from data.adapters.fallback_market_data import FallbackMarketDataAdapter
from data.adapters.interval_routing_market_data import (
    IntervalRoutingMarketData,
)
from data.adapters.massive_adapter import MassiveAdapter
from data.adapters.yfinance_adapter import YFinanceAdapter
from data.domain.ports import MarketDataPort


def build_default_market_data() -> MarketDataPort:
    """Return the routed + cached market-data stack used by every
    Streamlit page.

    Composition::

        IntervalRoutingMarketData(
            sub_daily = FallbackMarketDataAdapter(
                primary  = Cached(EODHD,   .../eodhd),
                fallback = Cached(Massive, .../massive),
            ),
            daily     = Cached(YFinance, .../),
        )

    Each leg degrades gracefully:
      - No EODHD key   → primary is just Massive (no fallback).
      - No MASSIVE key → primary is just EODHD (no fallback).
      - Neither key    → all intervals via yfinance.
    """
    yf_cached = CachedMarketDataAdapter(YFinanceAdapter())

    eodhd_cached = None
    try:
        eodhd_cached = CachedMarketDataAdapter(
            EODHDAdapter(),
            cache_dir="/tmp/pattern-finder-cache/eodhd",
        )
    except ValueError:
        pass

    massive_cached = None
    try:
        massive_cached = CachedMarketDataAdapter(
            MassiveAdapter(),
            cache_dir="/tmp/pattern-finder-cache/massive",
        )
    except ValueError:
        pass

    # Compose sub-daily source. Prefer EODHD primary (KR coverage,
    # cheaper, well-tested) with Massive fallback for quota-exhaust
    # days. If only one is available, use it directly. If neither,
    # all intervals go via yfinance.
    if eodhd_cached is not None and massive_cached is not None:
        sub_daily: MarketDataPort = FallbackMarketDataAdapter(
            primary=eodhd_cached,
            fallback=massive_cached,
            primary_label="EODHD",
            fallback_label="Massive",
        )
    elif eodhd_cached is not None:
        sub_daily = eodhd_cached
    elif massive_cached is not None:
        sub_daily = massive_cached
    else:
        return yf_cached

    return IntervalRoutingMarketData(
        sub_daily=sub_daily,
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
