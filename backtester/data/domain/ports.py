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
    ) -> pd.DataFrame:
        """Fetch OHLCV data.

        Returns DataFrame with columns: Open, High, Low, Close, Volume
        indexed by DatetimeIndex.
        """
        ...
