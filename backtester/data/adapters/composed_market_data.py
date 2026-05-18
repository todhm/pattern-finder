"""Factory composing the production market-data stack.

Pages should call :func:`build_default_market_data` instead of
hand-wiring the adapters. The factory:

1. Wraps each upstream source (yfinance, EODHD, Massive) with the
   **MongoDB-backed cache** (`MongoDayCacheAdapter`). One Mongo
   collection per source so a cross-source peek can find a hit even
   if the primary source has only a fail marker.

2. Routes by interval:
     - daily       → yfinance primary + Massive fallback
     - sub-daily   → EODHD primary + Massive fallback

   When EODHD or yfinance throws (HTTP 402 quota, 429 rate limit),
   the fallback transparently routes to Massive/Polygon.

3. When neither EODHD nor MASSIVE keys are configured, falls back
   to a single uncached yfinance for all intervals.

Storage layout
--------------
- OHLCV cache is **day-chunked** — one Mongo doc per
  (symbol, interval, date) — in collections ``bars_eodhd`` /
  ``bars_massive`` / ``bars_yfinance`` inside the ``pattern_finder``
  database (see ``MONGO_DB`` env var). Different windows that share
  any business days reuse the same per-day docs.
- Mongo data files are bind-mounted from the host
  (``./mongodata`` per ``docker-compose.yaml``) — the docker image
  stays small while the cache survives container rebuilds.

Today-partial policy
--------------------
A day fetched while it was *today* is stored with
``is_partial=True``. On the next request after the system date
advances, that single day is refetched (past days in the same range
are still served from cache). Past-day entries (``is_partial=False``)
are permanent.

Signal pages
------------
Signal pages (live entry-candidate scanners — e.g.
``4_Multi_Wedgepop_Signals.py``) keep using the parquet-based
``CachedMarketDataAdapter`` directly with ``bypass_today=True``. They
need a clean "always re-fetch the partial day" opt-in and don't
benefit from cross-machine cache sharing.
"""

from __future__ import annotations

from data.adapters.cached_market_data import CachedMarketDataAdapter
from data.adapters.eodhd_adapter import EODHDAdapter
from data.adapters.fallback_market_data import FallbackMarketDataAdapter
from data.adapters.interval_routing_market_data import (
    IntervalRoutingMarketData,
)
from data.adapters.massive_adapter import MassiveAdapter
from data.adapters.mongo_day_cache import MongoDayCacheAdapter
from data.adapters.yfinance_adapter import YFinanceAdapter
from data.domain.ports import MarketDataPort


def build_default_market_data(
    *, bypass_today: bool = False
) -> MarketDataPort:
    """Return the routed + Mongo-cached market-data stack used by every
    backtest Streamlit page / sweep CLI.

    ``bypass_today``:
        Threaded into every cache adapter. **Signal pages** that need
        a fresh fetch when the window includes today should set True
        — but those pages typically wire ``CachedMarketDataAdapter``
        themselves rather than going through this factory. Backtest
        and parameter-sweep pages leave it False (default) so the
        Mongo cache is reused across iterations.

    Composition::

        IntervalRoutingMarketData(
            sub_daily = FallbackMarketDataAdapter(
                primary  = FallbackMarketDataAdapter(
                    primary  = MongoDayCache(EODHD,   "bars_eodhd"),
                    fallback = MongoDayCache(Massive, "bars_massive"),
                ),
                fallback = MongoDayCache(YFinance, "bars_yfinance"),
            ),
            daily     = FallbackMarketDataAdapter(
                primary  = MongoDayCache(YFinance, "bars_yfinance"),
                fallback = MongoDayCache(Massive,  "bars_massive"),
            ),
        )

    Each leg degrades gracefully:
      - No EODHD key   → sub-daily primary is just (Massive → YFinance).
      - No MASSIVE key → sub-daily is (EODHD → YFinance); daily has no fallback.
      - Neither paid key → all intervals via yfinance (no Mongo wrap).

    Why yfinance is the *final* sub-daily fallback (added 2026-05):
        EODHD and Massive both refuse to serve today's not-yet-closed
        intraday on most of their tiers. yfinance returns live 1m bars
        (15-min delay on the free tier) — so wiring it at the tail of
        the sub-daily fallback chain lets ``end_date == today`` pages
        keep working without each page hand-wiring a yfinance bypass.
    """
    yf_cached = MongoDayCacheAdapter(
        YFinanceAdapter(),
        source_name="yfinance",
        bypass_today=bypass_today,
    )

    eodhd_cached = None
    try:
        eodhd_cached = MongoDayCacheAdapter(
            EODHDAdapter(),
            source_name="eodhd",
            bypass_today=bypass_today,
        )
    except ValueError:
        pass

    massive_cached = None
    try:
        massive_cached = MongoDayCacheAdapter(
            MassiveAdapter(),
            source_name="massive",
            bypass_today=bypass_today,
        )
    except ValueError:
        pass

    # Compose sub-daily source. Prefer EODHD primary (KR coverage,
    # cheaper, well-tested) with Massive fallback for quota-exhaust
    # days. If only one is available, use it directly. If neither,
    # all intervals go via uncached yfinance.
    if eodhd_cached is not None and massive_cached is not None:
        sub_daily_paid: MarketDataPort = FallbackMarketDataAdapter(
            primary=eodhd_cached,
            fallback=massive_cached,
            primary_label="EODHD",
            fallback_label="Massive",
        )
    elif eodhd_cached is not None:
        sub_daily_paid = eodhd_cached
    elif massive_cached is not None:
        sub_daily_paid = massive_cached
    else:
        # No paid sub-daily source configured — fall back to yfinance
        # for all intervals. Drop Mongo wrap to avoid a useless
        # Mongo dependency on the local-dev path.
        return CachedMarketDataAdapter(
            YFinanceAdapter(), bypass_today=bypass_today,
        )

    # Tail the sub-daily chain with yfinance for the today-intraday gap
    # (EODHD/Massive don't publish live un-closed bars on most tiers).
    # yfinance only goes back ~30 days on 1m and ~60 days on 5m, but
    # for those windows it works — and historical sub-daily requests
    # are served from the upstream paid sources / Mongo cache *before*
    # we ever fall through to here.
    sub_daily: MarketDataPort = FallbackMarketDataAdapter(
        primary=sub_daily_paid,
        fallback=yf_cached,
        primary_label="EODHD/Massive",
        fallback_label="YFinance",
    )

    # Daily leg: yfinance primary (free, fast on hits) + Massive
    # fallback (paid, separate quota) so a yfinance 429 doesn't kill
    # the page.
    if massive_cached is not None:
        daily: MarketDataPort = FallbackMarketDataAdapter(
            primary=yf_cached,
            fallback=massive_cached,
            primary_label="YFinance",
            fallback_label="Massive",
        )
    else:
        daily = yf_cached

    return IntervalRoutingMarketData(
        sub_daily=sub_daily,
        daily=daily,
    )


def is_eodhd_configured() -> bool:
    """Cheap check pages can use to surface a 'EODHD disabled'
    warning when the API key isn't loaded."""
    try:
        EODHDAdapter()
        return True
    except ValueError:
        return False
