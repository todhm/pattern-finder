import pandas as pd

from pattern.models.signal import PatternSignal
from pattern.services.base import PatternDetector


class WedgePopDetector(PatternDetector):
    """Detects Wedge Pop pattern.

    Conditions:
    1. Consolidation: close below both 10 & 20 EMA for `consolidation_days`.
    2. Volatility compression: recent ATR < average ATR (tightening range).
    3. Breakout: close breaks above both 10 & 20 EMA.
    4. Volume confirmation: volume > `volume_multiplier` * 20-day avg.
    """

    name = "wedge_pop"

    def __init__(
        self,
        consolidation_days: int = 10,
        volume_multiplier: float = 1.3,
        atr_compression: float = 0.8,
    ):
        self.consolidation_days = consolidation_days
        self.volume_multiplier = volume_multiplier
        self.atr_compression = atr_compression

    def detect(self, df: pd.DataFrame) -> list[PatternSignal]:
        df = df.copy()
        df["ema10"] = self._add_ema(df, 10)
        df["ema20"] = self._add_ema(df, 20)
        df["vol_avg"] = self._avg_volume(df)
        df["atr"] = self._atr(df, 14)
        df["atr_avg"] = df["atr"].rolling(20).mean()

        min_index = max(30, self.consolidation_days + 1)
        signals: list[PatternSignal] = []

        for i in range(min_index, len(df)):
            if not self._was_consolidated(df, i):
                continue

            if not self._is_breakout(df, i):
                continue

            if not self._volatility_compressed(df, i):
                continue

            if not self._volume_confirmed(df, i):
                continue

            signals.append(self._build_signal(df, i))

        return signals

    def _was_consolidated(self, df: pd.DataFrame, i: int) -> bool:
        for j in range(1, self.consolidation_days + 1):
            idx = i - j
            if (
                df["Close"].iloc[idx] >= df["ema10"].iloc[idx]
                or df["Close"].iloc[idx] >= df["ema20"].iloc[idx]
            ):
                return False
        return True

    def _is_breakout(self, df: pd.DataFrame, i: int) -> bool:
        return (
            df["Close"].iloc[i] > df["ema10"].iloc[i]
            and df["Close"].iloc[i] > df["ema20"].iloc[i]
        )

    def _volatility_compressed(self, df: pd.DataFrame, i: int) -> bool:
        if pd.isna(df["atr_avg"].iloc[i]):
            return False
        return df["atr"].iloc[i] < self.atr_compression * df["atr_avg"].iloc[i]

    def _volume_confirmed(self, df: pd.DataFrame, i: int) -> bool:
        return df["Volume"].iloc[i] > self.volume_multiplier * df["vol_avg"].iloc[i]

    def _build_signal(self, df: pd.DataFrame, i: int) -> PatternSignal:
        consolidation_low = df["Low"].iloc[i - self.consolidation_days : i].min()
        return PatternSignal(
            date=df.index[i].date(),
            pattern_name=self.name,
            entry_price=df["Close"].iloc[i],
            stop_loss=consolidation_low,
            confidence=round(
                df["Volume"].iloc[i] / df["vol_avg"].iloc[i], 2
            ),
            metadata={
                "atr_ratio": round(
                    df["atr"].iloc[i] / df["atr_avg"].iloc[i], 4
                ),
                "volume_ratio": round(
                    df["Volume"].iloc[i] / df["vol_avg"].iloc[i], 2
                ),
                "consolidation_low": round(consolidation_low, 2),
            },
        )

    @staticmethod
    def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        high_low = df["High"] - df["Low"]
        high_prev_close = (df["High"] - df["Close"].shift(1)).abs()
        low_prev_close = (df["Low"] - df["Close"].shift(1)).abs()
        true_range = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(
            axis=1
        )
        return true_range.rolling(period).mean()
