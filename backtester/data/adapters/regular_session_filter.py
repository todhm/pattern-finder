"""Filter wrapper that keeps only regular-session bars for the
configured market calendar.

Composition::

    RegularSessionFilterAdapter(
        CachedMarketDataAdapter(YFinanceAdapter()),
        market=NY,  # or KR, etc.
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

For non-US calendars: yfinance returns intraday data with a tz-aware
index (default ``America/New_York``). The filter converts to the
calendar's *local* tz before applying ``between_time`` so KR bars
gate against 09:00–15:30 KST instead of 09:30–16:00 ET — without
this, a KR ticker's bars at ~02:30 ET (= 15:30 KST close) would
all be dropped as "non-RTH" and the universe scan returns 0 trades
on every Korean ticker (the user-flagged failure mode).
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from data.domain.market_calendar import NY, MarketCalendar
from data.domain.ports import MarketDataPort


class RegularSessionFilterAdapter(MarketDataPort):
    """Wrap any ``MarketDataPort`` and drop non-regular-session bars
    for a given :class:`MarketCalendar`."""

    def __init__(
        self,
        upstream: MarketDataPort,
        market: MarketCalendar = NY,
    ) -> None:
        self._upstream = upstream
        self._market = market

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
        if interval == "1d" or df is None or df.empty:
            return df
        # Convert to the calendar's local tz so ``between_time``
        # compares apples-to-apples. yfinance hands us tz-aware NY
        # frames by default, which would silently misclassify every
        # KR bar without this conversion.
        if df.index.tz is not None:
            local_df = df.copy()
            local_df.index = df.index.tz_convert(self._market.tz)
        else:
            local_df = df
        # ``between_time`` is inclusive on both ends. The last 15m
        # bar starts ``rth_close - 15min`` (e.g. 15:45 ET / 15:15
        # KST), so we set the close-side bound one minute below
        # ``rth_close`` to keep that final bar.
        end_bound = self._market.rth_close.replace(
            minute=max(0, self._market.rth_close.minute - 1)
        )
        if self._market.rth_close.minute == 0:
            # 16:00 → 15:59. Subtract one minute by walking the hour.
            end_bound = self._market.rth_close.replace(
                hour=self._market.rth_close.hour - 1, minute=59
            )
        filtered = local_df.between_time(
            self._market.rth_open.strftime("%H:%M"),
            end_bound.strftime("%H:%M"),
        )
        # Restore the upstream tz so downstream code keeps seeing
        # the same tz it always did (NY for cached parquet files).
        if df.index.tz is not None:
            filtered = filtered.copy()
            filtered.index = filtered.index.tz_convert(df.index.tz)
        return filtered
