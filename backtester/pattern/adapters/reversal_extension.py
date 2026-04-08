import pandas as pd

from pattern.domain.models import PatternSignal
from pattern.domain.ports import PatternDetector


class ReversalExtensionDetector(PatternDetector):
    """Detects Reversal Extension pattern.

    Conditions:
    1. Downtrend: close below 10 EMA for `lookback` consecutive days.
    2. Extension: low extends > `extension_pct` below 10 EMA.
    3. Volume spike: volume > `volume_multiplier` * 20-day avg volume.
    4. Reversal candle: close > open (bullish).
    """

    name = "reversal_extension"

    def __init__(
        self,
        ema_period: int = 10,
        extension_pct: float = 0.05,
        volume_multiplier: float = 1.5,
        lookback: int = 5,
    ):
        self.ema_period = ema_period
        self.extension_pct = extension_pct
        self.volume_multiplier = volume_multiplier
        self.lookback = lookback

    def detect(
        self,
        df: pd.DataFrame,
        weekly_df: pd.DataFrame | None = None,
        monthly_df: pd.DataFrame | None = None,
    ) -> list[PatternSignal]:
        df = df.copy()
        df["ema"] = self._add_ema(df, self.ema_period)
        df["vol_avg"] = self._avg_volume(df)

        min_index = max(20, self.lookback)
        signals: list[PatternSignal] = []

        for i in range(min_index, len(df)):
            if not self._in_downtrend(df, i):
                continue

            extension = self._extension_ratio(df, i)
            if extension < self.extension_pct:
                continue

            if not self._volume_spike(df, i):
                continue

            if not self._is_reversal_candle(df, i):
                continue

            signals.append(self._build_signal(df, i, extension))

        return signals

    def _in_downtrend(self, df: pd.DataFrame, i: int) -> bool:
        for j in range(1, self.lookback + 1):
            if df["Close"].iloc[i - j] >= df["ema"].iloc[i - j]:
                return False
        return True

    def _extension_ratio(self, df: pd.DataFrame, i: int) -> float:
        ema_val = df["ema"].iloc[i]
        low = df["Low"].iloc[i]
        return (ema_val - low) / ema_val

    def _volume_spike(self, df: pd.DataFrame, i: int) -> bool:
        return df["Volume"].iloc[i] > self.volume_multiplier * df["vol_avg"].iloc[i]

    def _is_reversal_candle(self, df: pd.DataFrame, i: int) -> bool:
        return df["Close"].iloc[i] > df["Open"].iloc[i]

    def _build_signal(
        self, df: pd.DataFrame, i: int, extension: float
    ) -> PatternSignal:
        return PatternSignal(
            date=df.index[i].date(),
            pattern_name=self.name,
            entry_price=df["Close"].iloc[i],
            stop_loss=df["Low"].iloc[i],
            confidence=min(extension / self.extension_pct, 2.0),
            metadata={
                "extension_pct": round(extension, 4),
                "volume_ratio": round(
                    df["Volume"].iloc[i] / df["vol_avg"].iloc[i], 2
                ),
            },
        )
