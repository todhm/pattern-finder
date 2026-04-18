from abc import ABC, abstractmethod

import numpy as np
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

    @staticmethod
    def _regression_slope(series: pd.Series, window: int) -> pd.Series:
        """Rolling linear-regression slope, normalised by the window
        mean so the result is comparable to a percent-change metric.

        For each position ``i``, fits ``y = a + b*x`` to
        ``series[i-window+1 : i+1]`` and returns ``b * window /
        mean(y)`` — the total predicted change over the window as a
        fraction of the window's average value.

        This captures the *direction the line is heading right now*,
        unlike the endpoint formula ``(y[-1] - y[0]) / y[0]`` which
        can be positive even while the series is falling (e.g. rose
        then fell back, ending slightly above the start).
        """
        x = np.arange(window, dtype=float)
        x_mean = x.mean()
        x_var = ((x - x_mean) ** 2).sum()

        def _slope(vals):
            if len(vals) < window:
                return np.nan
            y = vals.values
            y_mean = y.mean()
            if y_mean == 0:
                return 0.0
            b = ((x - x_mean) * (y - y_mean)).sum() / x_var
            return b * window / y_mean

        return series.rolling(window).apply(_slope, raw=False)
