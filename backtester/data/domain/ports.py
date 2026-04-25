from abc import ABC, abstractmethod
from datetime import date

import pandas as pd


class MarketDataPort(ABC):
    """Port: market data fetching interface."""

    @abstractmethod
    def fetch_ohlcv(
        self,
        symbol: str,
        start: date,
        end: date,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Fetch OHLCV data.

        Returns DataFrame with columns: Open, High, Low, Close, Volume
        indexed by DatetimeIndex.

        ``interval`` follows yfinance conventions (e.g. ``"1d"``, ``"1h"``,
        ``"30m"``, ``"15m"``, ``"5m"``). For sub-daily intervals the
        returned DatetimeIndex is tz-aware in ``America/New_York`` and
        contains only regular US equity session bars (09:30–16:00 ET).
        Daily (``"1d"``) data keeps the legacy tz-naive contract.
        """
        ...


class UniverseProviderPort(ABC):
    """Port: ticker-universe provider.

    Resolves a universe identifier (e.g. ``"sp500"``, ``"nasdaq100"``) to
    the list of tickers belonging to that index. Used by multi-ticker
    strategies that scan an entire universe for opportunities.
    """

    @abstractmethod
    def get_tickers(self, universe: str) -> list[str]:
        """Return the list of tickers for the named universe."""
        ...
