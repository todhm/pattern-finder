"""Factory for the production fundamentals (float + splits) stack.

Mirror of ``composed_market_data.build_default_market_data`` for the
:class:`FundamentalsPort` interface. Pages should call
:func:`build_default_fundamentals` instead of hand-wiring adapters.

Composition::

    Cached(
        Fallback(
            primary  = EODHDFundamentalsAdapter(),
            fallback = MassiveFundamentalsAdapter(),
        ),
        cache_dir="/tmp/pattern-finder-cache/fundamentals",
        ttl_days=7,
    )

Degradation:
  - No EODHD key → primary is just Massive.
  - No MASSIVE key → primary is just EODHD.
  - Neither → empty stub returning ``TickerFundamentals(symbol, None, None)``.
"""

from __future__ import annotations

from data.adapters.cached_fundamentals import CachedFundamentalsAdapter
from data.adapters.eodhd_fundamentals import EODHDFundamentalsAdapter
from data.adapters.fallback_fundamentals import FallbackFundamentalsAdapter
from data.adapters.massive_fundamentals import MassiveFundamentalsAdapter
from data.domain.ports import FundamentalsPort, TickerFundamentals


class _NullFundamentalsAdapter(FundamentalsPort):
    """Returned when neither EODHD nor Massive keys are configured.

    Always returns empty fundamentals — the strategy's
    ``require_float_filter`` then decides whether to drop the ticker.
    """

    def fetch(self, symbol: str) -> TickerFundamentals:
        return TickerFundamentals(symbol=symbol, float_shares=None, splits=None)


def build_default_fundamentals(
    cache_dir: str = "/tmp/pattern-finder-cache/fundamentals",
    ttl_days: int = 7,
) -> FundamentalsPort:
    eodhd = None
    try:
        eodhd = EODHDFundamentalsAdapter()
    except ValueError:
        pass

    massive = None
    try:
        massive = MassiveFundamentalsAdapter()
    except ValueError:
        pass

    inner: FundamentalsPort
    if eodhd is not None and massive is not None:
        inner = FallbackFundamentalsAdapter(
            primary=eodhd,
            fallback=massive,
            primary_label="EODHD",
            fallback_label="Massive",
        )
    elif eodhd is not None:
        inner = eodhd
    elif massive is not None:
        inner = massive
    else:
        # Neither configured — return null adapter wrapped in cache so
        # the cache layer is uniform regardless.
        inner = _NullFundamentalsAdapter()

    return CachedFundamentalsAdapter(inner, cache_dir=cache_dir, ttl_days=ttl_days)
