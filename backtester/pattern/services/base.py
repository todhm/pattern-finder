from abc import ABC, abstractmethod

import pandas as pd

from pattern.models.signal import PatternSignal


class PatternDetector(ABC):
    """Base class for all pattern detectors.

    Subclass this and implement `detect()` to add a new pattern.
    """

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def detect(self, df: pd.DataFrame) -> list[PatternSignal]:
        """Detect pattern signals from OHLCV DataFrame.

        Args:
            df: DataFrame with columns Open, High, Low, Close, Volume
                indexed by DatetimeIndex.

        Returns:
            List of detected pattern signals.
        """
        ...

    def _add_ema(self, df: pd.DataFrame, period: int) -> pd.Series:
        return df["Close"].ewm(span=period, adjust=False).mean()

    def _avg_volume(self, df: pd.DataFrame, period: int = 20) -> pd.Series:
        return df["Volume"].rolling(period).mean()
