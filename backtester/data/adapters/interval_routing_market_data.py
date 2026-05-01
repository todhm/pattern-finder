"""Route market-data fetches to the right source based on interval.

Composition pattern::

    IntervalRoutingMarketData(
        sub_daily=CachedMarketDataAdapter(EODHDAdapter()),
        daily=CachedMarketDataAdapter(YFinanceAdapter()),
    )

Routing table::

    interval == "1d" / "1wk" / "1mo"  →  daily     (free, sufficient depth)
    everything else                   →  sub_daily (Massive: deep intraday)

KR tickers (``.KS`` / ``.KQ``) are sent to Massive too — if the
upstream's plan covers KOSPI/KOSDAQ, the data flows through; if not,
Massive's "no data" error surfaces and the caller can decide.

Both underlying adapters return the same NY-tz indexed OHLCV
``pd.DataFrame`` shape so the routing is purely about cost/depth.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from data.domain.ports import MarketDataPort

# yfinance "daily and longer" intervals — these stay on the free
# upstream. Massive would also serve them but at API-call cost.
_DAILY_OR_LONGER = {"1d", "5d", "1wk", "1mo", "3mo"}


class IntervalRoutingMarketData(MarketDataPort):
    """Dispatch ``fetch_ohlcv`` based on interval."""

    def __init__(
        self,
        sub_daily: MarketDataPort,
        daily: MarketDataPort,
    ) -> None:
        self._sub_daily = sub_daily
        self._daily = daily

    def fetch_ohlcv(
        self,
        symbol: str,
        start: date,
        end: date,
        interval: str = "1d",
    ) -> pd.DataFrame:
        if self._is_daily_or_longer(interval):
            return self._daily.fetch_ohlcv(
                symbol, start, end, interval=interval
            )
        return self._sub_daily.fetch_ohlcv(
            symbol, start, end, interval=interval
        )

    @staticmethod
    def _is_daily_or_longer(interval: str) -> bool:
        return interval in _DAILY_OR_LONGER
