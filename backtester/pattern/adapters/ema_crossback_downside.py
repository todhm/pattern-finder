import pandas as pd

from pattern.domain.models import PatternSignal
from pattern.domain.ports import PatternDetector


class EmaCrossbackDownsideDetector(PatternDetector):
    """EMA Crossback (downside) — failed retest of declining EMAs.

    In a confirmed downtrend, the price rallies UP toward the declining
    fast EMA and is rejected: today's ``High`` touches or pierces the
    fast EMA, but the close closes back below it on a bearish candle.
    Per the Downtrends doc, the failure to reclaim the moving averages
    confirms further downside and provides a low-risk short entry.

    Short signal: ``entry_price`` at close, ``stop_loss`` above the
    rally's high (invalidation level).

    Conditions
        - Downtrend: ``ema_fast < ema_slow``.
        - Slow EMA declining: ``ema_slow_slope <= max_slow_slope``.
        - Prior ``prior_below_bars`` mostly closed BELOW fast EMA
          (confirms context — this IS a retest, not a first crossdown).
        - Today's ``High >= ema_fast`` (rally reached resistance).
        - Today's ``close < ema_fast`` (rejected).
        - Bearish bar (``close < open``).
    """

    name = "ema_crossback_downside"

    def __init__(
        self,
        max_slow_slope: float = -0.02,
        prior_below_bars: int = 5,
        prior_below_pct: float = 0.6,
        ema_fast: int = 10,
        ema_slow: int = 20,
        atr_period: int = 14,
        slope_lookback: int = 20,
        cooldown_bars: int = 5,
    ):
        self.max_slow_slope = max_slow_slope
        self.prior_below_bars = prior_below_bars
        self.prior_below_pct = prior_below_pct
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.atr_period = atr_period
        self.slope_lookback = slope_lookback
        self.cooldown_bars = cooldown_bars

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
        n = self.slope_lookback
        df["ema_slow_slope"] = (df["ema_slow"] - df["ema_slow"].shift(n)) / df["ema_slow"].shift(n)

        signals: list[PatternSignal] = []
        cooldown_until = -1
        start = max(self.ema_slow, self.slope_lookback, self.prior_below_bars)

        for i in range(start, len(df)):
            if i <= cooldown_until:
                continue
            atr = df["atr"].iloc[i]
            if pd.isna(atr) or atr <= 0:
                continue

            ema_fast = float(df["ema_fast"].iloc[i])
            ema_slow = float(df["ema_slow"].iloc[i])
            if ema_fast >= ema_slow:
                continue

            slope = df["ema_slow_slope"].iloc[i]
            if pd.isna(slope) or slope > self.max_slow_slope:
                continue

            high = float(df["High"].iloc[i])
            close = float(df["Close"].iloc[i])
            open_ = float(df["Open"].iloc[i])
            if high < ema_fast:
                continue
            if close >= ema_fast:
                continue
            if close >= open_:
                continue

            below = 0
            for j in range(1, self.prior_below_bars + 1):
                if df["Close"].iloc[i - j] < df["ema_fast"].iloc[i - j]:
                    below += 1
            if below / self.prior_below_bars < self.prior_below_pct:
                continue

            signals.append(self._build_signal(df, i))
            cooldown_until = i + self.cooldown_bars

        return signals

    def _build_signal(self, df: pd.DataFrame, i: int) -> PatternSignal:
        close = float(df["Close"].iloc[i])
        high = float(df["High"].iloc[i])
        ema_fast = float(df["ema_fast"].iloc[i])
        ema_slow = float(df["ema_slow"].iloc[i])
        slope = float(df["ema_slow_slope"].iloc[i])
        atr = float(df["atr"].iloc[i])
        rejection_atr = (ema_fast - close) / atr if atr > 0 else 0.0
        vol_ratio = (
            round(float(df["Volume"].iloc[i] / df["vol_avg"].iloc[i]), 2)
            if not pd.isna(df["vol_avg"].iloc[i]) and df["vol_avg"].iloc[i] > 0
            else 1.0
        )
        return PatternSignal(
            date=df.index[i].date(),
            pattern_name=self.name,
            entry_price=close,
            stop_loss=round(high, 2),
            confidence=min(1.0 + rejection_atr * 2, 3.0),
            metadata={
                "direction": "short",
                "rally_high": round(high, 2),
                "ema_fast": round(ema_fast, 2),
                "ema_slow": round(ema_slow, 2),
                "ema_slow_slope": round(slope, 4),
                "rejection_atr": round(rejection_atr, 4),
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
