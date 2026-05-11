"""Primary → fallback chain for market-data sources.

Wraps two :class:`MarketDataPort` implementations and resolves a
fetch in three stages:

  1. **Cross-source cache peek (no network)** — call ``peek_cache``
     on both adapters in primary→fallback order; the first parquet
     hit wins. This stage exists so a Massive cache hit is preferred
     over an EODHD live call that would otherwise burn quota.
  2. **Primary live fetch** — both caches missed; call the primary
     normally. The wrapped ``CachedMarketDataAdapter`` writes a
     parquet on success and a ``.fail.json`` negative marker on any
     exception (so this primary won't be retried until the marker's
     TTL elapses).
  3. **Fallback live fetch** — primary raised; route to the secondary.

Composition pattern (used by ``composed_market_data``)::

    sub_daily = FallbackMarketDataAdapter(
        primary  = Cached(EODHDAdapter(),   eodhd_cache_dir),
        fallback = Cached(MassiveAdapter(), massive_cache_dir),
    )

The intended trigger for the live-fetch fallback is **EODHD daily
quota exhaustion** (HTTP 402); the cache-peek stage avoids hitting
EODHD at all when Massive already covers the window.
"""

from __future__ import annotations

import logging
from datetime import date

import pandas as pd

from data.domain.ports import MarketDataPort

log = logging.getLogger(__name__)


class FallbackMarketDataAdapter(MarketDataPort):
    """Cache-first across both sources, then primary→fallback live."""

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
        # --- Stage 1: cross-source cache peek (no network call) ---
        for adapter, label in (
            (self._primary, self._primary_label),
            (self._fallback, self._fallback_label),
        ):
            peek = getattr(adapter, "peek_cache", None)
            if peek is None:
                continue
            try:
                df = peek(symbol, start, end, interval=interval)
            except Exception:
                df = None
            if df is not None:
                return df

        # --- Stage 2: primary live (cache will write parquet/fail) ---
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

        # --- Stage 3: fallback live ---
        return self._fallback.fetch_ohlcv(
            symbol, start, end, interval=interval
        )
