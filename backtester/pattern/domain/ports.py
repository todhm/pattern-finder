from abc import ABC, abstractmethod

import pandas as pd

from pattern.domain.models import PatternSignal


class PatternDetector(ABC):
    """Port: pattern detection interface.

    Subclass this and implement `detect()` to add a new pattern.
    """

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def detect(
        self,
        df: pd.DataFrame,
        weekly_df: pd.DataFrame | None = None,
        monthly_df: pd.DataFrame | None = None,
    ) -> list[PatternSignal]:
        """Detect pattern signals from OHLCV DataFrame.

        Args:
            df: Daily DataFrame with columns Open, High, Low, Close, Volume
                indexed by DatetimeIndex.
            weekly_df: Optional weekly OHLCV DataFrame (same column format).
            monthly_df: Optional monthly OHLCV DataFrame (same column format).

        Returns:
            List of detected pattern signals.
        """
        ...

    def _add_ema(self, df: pd.DataFrame, period: int) -> pd.Series:
        return df["Close"].ewm(span=period, adjust=False).mean()

    def _avg_volume(self, df: pd.DataFrame, period: int = 20) -> pd.Series:
        return df["Volume"].rolling(period).mean()
