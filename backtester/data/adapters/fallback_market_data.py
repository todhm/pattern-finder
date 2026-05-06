"""Primary → fallback chain for market-data sources.

Wraps two :class:`MarketDataPort` implementations and tries the
primary first; on any exception (HTTP 402 quota, 429 rate limit,
422 unsupported, network error, etc.) falls through to the
fallback. The fallback's output is returned unchanged so callers
can't tell which source succeeded.

Composition pattern (used by ``composed_market_data``)::

    sub_daily = FallbackMarketDataAdapter(
        primary  = Cached(EODHDAdapter(),   eodhd_cache_dir),
        fallback = Cached(MassiveAdapter(), massive_cache_dir),
    )

Cache wrappers go *inside* the fallback so each source uses its
own parquet cache. A cache hit on the primary skips the network
entirely; only on cache miss does the HTTP call happen, and only
on HTTP failure does the fallback engage.

The intended trigger is **EODHD daily quota exhaustion** — the
server returns ``HTTP 402 You exceeded your daily API requests
limit`` and the chain transparently routes the rest of the day
through Polygon/Massive (which has 10y+ depth on US equities).
"""

from __future__ import annotations

import logging
from datetime import date

import pandas as pd

from data.domain.ports import MarketDataPort

log = logging.getLogger(__name__)


class FallbackMarketDataAdapter(MarketDataPort):
    """Try ``primary``, fall through to ``fallback`` on exception."""

    def __init__(
        self,
        primary: MarketDataPort,
        fallback: MarketDataPort,
        primary_label: str = "primary",
        fallback_label: str = "fallback",
    ) -> None:
        self._primary = primary
        self._fallback = fallback
        self._primary_label = primary_label
        self._fallback_label = fallback_label
        # Last error from the primary — useful to surface in
        # diagnostic UIs ("EODHD quota exhausted, served via
        # Massive").
        self.last_primary_error: str | None = None

    def fetch_ohlcv(
        self,
        symbol: str,
        start: date,
        end: date,
        interval: str = "1d",
    ) -> pd.DataFrame:
        try:
            return self._primary.fetch_ohlcv(
                symbol, start, end, interval=interval
            )
        except Exception as exc:
            self.last_primary_error = (
                f"{type(exc).__name__}: {exc}"
            )
            log.warning(
                "%s failed for %s %s: %s — falling through to %s",
                self._primary_label, symbol, interval,
                self.last_primary_error, self._fallback_label,
            )
        return self._fallback.fetch_ohlcv(
            symbol, start, end, interval=interval
        )
