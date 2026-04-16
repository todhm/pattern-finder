import pandas as pd

from pattern.domain.models import PatternSignal
from pattern.domain.ports import PatternDetector


class WedgeDropDetector(PatternDetector):
    """Wedge Drop — mirror of Wedge Pop, confirms a top.

    Prior ``lookback`` bars consolidated ABOVE the fast EMA after an
    Exhaustion Extension. This bar breaks down through BOTH the 10
    and 20 EMA on a bearish candle, flipping the MAs from support to
    resistance. The Downtrends doc calls this the signal that the
    uptrend has ended.

    Short signal: ``entry_price`` at close, ``stop_loss`` above the
    consolidation high (invalidation level).

    Conditions
        - Prior ``lookback`` window: >= ``consolidation_pct`` of closes
          sat ABOVE the fast EMA.
        - Previous bar closed ABOVE fast EMA (fresh crossdown).
        - This bar closes BELOW both fast AND slow EMA.
        - Bearish bar: ``close < open``.
        - Breakdown strength: ``max(ema_distance, daily_move) / ATR
          >= breakdown_atr_mult``.
    """

    name = "wedge_drop"

    def __init__(
        self,
        lookback: int = 10,
        consolidation_pct: float = 0.6,
        breakdown_atr_mult: float = 0.5,
        ema_fast: int = 10,
        ema_slow: int = 20,
        atr_period: int = 14,
        cooldown_bars: int | None = None,
    ):
        self.lookback = lookback
        self.consolidation_pct = consolidation_pct
        self.breakdown_atr_mult = breakdown_atr_mult
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.atr_period = atr_period
        self.cooldown_bars = cooldown_bars if cooldown_bars is not None else lookback

    def detect(
        self,
        df: pd.DataFrame,
        weekly_df: pd.DataFrame | None = None,
        monthly_df: pd.DataFrame | None = None,
    ) -> list[PatternSignal]:
        df = df.copy()
        df["ema_fast"] = self._add_ema(df, self.ema_fast)
        df["ema_slow"] = self._add_ema(df, self.ema_slow)
        df["vol_avg"] = self._avg_volume(df)
        df["atr"] = self._compute_atr(df, self.atr_period)

        signals: list[PatternSignal] = []
        cooldown_until = -1
        start = max(self.lookback, self.ema_slow)

        for i in range(start, len(df)):
            if i <= cooldown_until:
                continue
            atr = df["atr"].iloc[i]
            if pd.isna(atr) or atr <= 0:
                continue

            # Prior window: mostly ABOVE fast EMA (prior uptrend)
            above = 0
            for j in range(1, self.lookback + 1):
                idx = i - j
                if df["Close"].iloc[idx] > df["ema_fast"].iloc[idx]:
                    above += 1
            if above / self.lookback < self.consolidation_pct:
                continue

            # Fresh crossdown: yesterday above fast EMA
            prev_close = float(df["Close"].iloc[i - 1])
            prev_ema_fast = float(df["ema_fast"].iloc[i - 1])
            if prev_close <= prev_ema_fast:
                continue

            close = float(df["Close"].iloc[i])
            open_ = float(df["Open"].iloc[i])
            ema_fast = float(df["ema_fast"].iloc[i])
            ema_slow = float(df["ema_slow"].iloc[i])

            # Close below BOTH EMAs
            if close >= ema_fast or close >= ema_slow:
                continue
            # Bearish bar
            if close >= open_:
                continue

            support = min(ema_fast, ema_slow)
            ema_distance = (support - close) / float(atr)
            daily_move = (prev_close - close) / float(atr)
            strength = max(ema_distance, daily_move)
            if strength < self.breakdown_atr_mult:
                continue

            signals.append(self._build_signal(df, i, strength))
            cooldown_until = i + self.cooldown_bars

        return signals

    def _build_signal(self, df: pd.DataFrame, i: int, strength: float) -> PatternSignal:
        close = float(df["Close"].iloc[i])
        lookback = min(self.lookback, i)
        consolidation_high = float(df["High"].iloc[i - lookback : i].max())
        vol_ratio = (
            round(float(df["Volume"].iloc[i] / df["vol_avg"].iloc[i]), 2)
            if not pd.isna(df["vol_avg"].iloc[i]) and df["vol_avg"].iloc[i] > 0
            else 1.0
        )
        return PatternSignal(
            date=df.index[i].date(),
            pattern_name=self.name,
            entry_price=close,
            stop_loss=round(consolidation_high, 2),
            confidence=min(1.0 + strength * 2, 3.0),
            metadata={
                "direction": "short",
                "breakdown_strength_atr": round(strength, 4),
                "consolidation_high": round(consolidation_high, 2),
                "volume_ratio": vol_ratio,
            },
        )

    @staticmethod
    def _compute_atr(df: pd.DataFrame, period: int) -> pd.Series:
        prev_close = df["Close"].shift(1)
        true_range = pd.concat(
            [
                df["High"] - df["Low"],
                (df["High"] - prev_close).abs(),
                (df["Low"] - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        return true_range.ewm(span=period, adjust=False).mean()
