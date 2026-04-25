"""Filter wrapper that keeps only regular US equity session bars.

Composition::

    RegularSessionFilterAdapter(
        CachedMarketDataAdapter(YFinanceAdapter())
    )

The ``YFinanceAdapter`` fetches **everything** (``prepost=True``) and
``CachedMarketDataAdapter`` parquet-caches that raw frame on disk.
This wrapper sits *outside* the cache so the filter is applied at
read time — a future ETH-aware strategy can compose without this
wrapper and read the *same* cached files, no duplicate fetch.

Daily (``interval="1d"``) requests pass through unchanged because
yfinance daily bars are already session-only by construction. The
filter only operates on sub-daily intervals where a bar's
``DatetimeIndex`` time-of-day is meaningful.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from data.domain.ports import MarketDataPort

# Regular US equity session window. ``between_time`` is inclusive on
# both ends — the last 15m bar starts at 15:45 (covering 15:45–16:00),
# so 15:59 is the latest start time we keep.
SESSION_OPEN = "09:30"
SESSION_CLOSE = "15:59"


class RegularSessionFilterAdapter(MarketDataPort):
    """Wrap any ``MarketDataPort`` and drop non-regular-session bars.

    On a sub-daily interval the upstream frame is expected to have a
    tz-aware ``DatetimeIndex`` (``YFinanceAdapter`` normalizes to
    ``America/New_York``). Pre-market (04:00–09:29 ET) and after-hours
    (16:00–20:00 ET) bars get filtered out here so downstream
    indicators (EMA, ATR, average volume) only see the high-liquidity
    regular session — the regime they're calibrated for.
    """

    def __init__(self, upstream: MarketDataPort):
        self._upstream = upstream

    def fetch_ohlcv(
        self,
        symbol: str,
        start: date,
        end: date,
        interval: str = "1d",
    ) -> pd.DataFrame:
        df = self._upstream.fetch_ohlcv(
            symbol, start, end, interval=interval
        )
        if interval == "1d":
            return df
        return df.between_time(SESSION_OPEN, SESSION_CLOSE)
