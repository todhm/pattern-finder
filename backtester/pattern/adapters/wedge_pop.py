import pandas as pd

from pattern.domain.models import PatternSignal
from pattern.domain.ports import PatternDetector


class WedgePopDetector(PatternDetector):
    """Detects Wedge Pop pattern using EMA-based consolidation + breakout.

    Conditions:
    1. Consolidation: >= `consolidation_pct` of last `lookback` bars
       closed below the fast EMA (allows real-market noise).
    2. Breakout: close > both fast & slow EMA.
    3. Breakout strength: close exceeds the higher EMA by at least
       `breakout_pct`, filtering out weak bounces.

    Volume is NOT a hard gate — real breakouts can have average volume
    when the rolling average is skewed by prior spikes (e.g. earnings).
    Volume ratio is reported in `metadata.volume_ratio` as a confidence
    indicator instead.
    """

    name = "wedge_pop"

    def __init__(
        self,
        lookback: int = 15,
        ema_fast: int = 10,
        ema_slow: int = 20,
        consolidation_pct: float = 0.6,
        breakout_pct: float = 0.015,
    ):
        self.lookback = lookback
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.consolidation_pct = consolidation_pct
        self.breakout_pct = breakout_pct

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

        min_idx = self.lookback
        signals: list[PatternSignal] = []
        cooldown_until = -1

        for i in range(min_idx, len(df)):
            if i <= cooldown_until:
                continue
            if not self._was_consolidated(df, i):
                continue
            if not self._is_breakout(df, i):
                continue
            signals.append(self._build_signal(df, i))
            cooldown_until = i + self.lookback

        return signals

    # ---- condition checks ----

    def _was_consolidated(self, df: pd.DataFrame, i: int) -> bool:
        """Require >= consolidation_pct of lookback days below fast EMA."""
        below = 0
        for j in range(1, self.lookback + 1):
            idx = i - j
            if df["Close"].iloc[idx] < df["ema_fast"].iloc[idx]:
                below += 1
        return below / self.lookback >= self.consolidation_pct

    def _is_breakout(self, df: pd.DataFrame, i: int) -> bool:
        """Close above both EMAs with meaningful strength.

        Passes if EITHER:
        - EMA distance: close is >= breakout_pct above the higher EMA, OR
        - Daily momentum: close rose >= breakout_pct from previous close.
        The second condition catches breakouts where the EMAs are close to
        price (tight consolidation) but the daily candle is clearly bullish.
        """
        close = df["Close"].iloc[i]
        fast = df["ema_fast"].iloc[i]
        slow = df["ema_slow"].iloc[i]

        if close <= fast or close <= slow:
            return False

        resistance = max(fast, slow)
        ema_strength = (close - resistance) / resistance

        prev_close = df["Close"].iloc[i - 1]
        daily_move = (close - prev_close) / prev_close

        return ema_strength >= self.breakout_pct or daily_move >= self.breakout_pct

    # ---- signal building ----

    def _build_signal(self, df: pd.DataFrame, i: int) -> PatternSignal:
        lookback = min(self.lookback, i)
        consolidation_low = df["Low"].iloc[i - lookback : i].min()

        vol_ratio = (
            round(df["Volume"].iloc[i] / df["vol_avg"].iloc[i], 2)
            if not pd.isna(df["vol_avg"].iloc[i]) and df["vol_avg"].iloc[i] > 0
            else 1.0
        )

        fast = df["ema_fast"].iloc[i]
        slow = df["ema_slow"].iloc[i]
        resistance = max(fast, slow)
        breakout_strength = round((df["Close"].iloc[i] - resistance) / resistance, 4)

        return PatternSignal(
            date=df.index[i].date(),
            pattern_name=self.name,
            entry_price=df["Close"].iloc[i],
            stop_loss=consolidation_low,
            confidence=min(1.0 + breakout_strength * 10, 3.0),
            metadata={
                "breakout_strength": breakout_strength,
                "volume_ratio": vol_ratio,
                "consolidation_low": round(consolidation_low, 2),
            },
        )
