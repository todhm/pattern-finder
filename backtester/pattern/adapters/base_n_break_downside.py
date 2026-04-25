import pandas as pd

from pattern.domain.models import PatternSignal
from pattern.domain.ports import PatternDetector


class BaseNBreakDownsideDetector(PatternDetector):
    """Base N Break (downside) — continuation breakdown in a downtrend.

    Mirror of the uptrend Base N Break. During a confirmed downtrend,
    the stock consolidates in a tight range BELOW the declining EMAs
    (the EMAs act as resistance above), then breaks below the
    consolidation low on a bearish candle — signalling continuation.

    The ``max_range_atr`` gate enforces the "volatility contraction"
    the Downtrends doc calls out as the setup trigger — wait for the
    fear-driven wide bars to calm down before the breakdown counts.

    Short signal: ``entry_price`` at close, ``stop_loss`` above the
    consolidation high (invalidation level).

    Conditions
        - Downtrend: ``ema_fast < ema_slow``.
        - Slow EMA declining: ``ema_slow_slope <= max_slow_slope``.
        - Prior ``lookback`` window: >= ``consolidation_below_pct`` of
          closes BELOW fast EMA.
        - Tight range: ``(window_high - window_low) / ATR <= max_range_atr``.
        - This bar: ``close < window_low`` (breakdown through pivot).
        - Bearish bar: ``close < open``.
        - Breakdown strength: ``(window_low - close) / ATR >= breakdown_atr_mult``.
    """

    name = "base_n_break_downside"

    def __init__(
        self,
        lookback: int = 10,
        consolidation_below_pct: float = 0.7,
        max_range_atr: float = 2.5,
        breakdown_atr_mult: float = 0.3,
        max_slow_slope: float = -0.02,
        ema_fast: int = 10,
        ema_slow: int = 20,
        atr_period: int = 14,
        slope_lookback: int = 20,
        cooldown_bars: int | None = None,
    ):
        self.lookback = lookback
        self.consolidation_below_pct = consolidation_below_pct
        self.max_range_atr = max_range_atr
        self.breakdown_atr_mult = breakdown_atr_mult
        self.max_slow_slope = max_slow_slope
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.atr_period = atr_period
        self.slope_lookback = slope_lookback
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
        n = self.slope_lookback
        df["ema_slow_slope"] = self._regression_slope(df["ema_slow"], n)

        signals: list[PatternSignal] = []
        cooldown_until = -1
        start = max(self.lookback, self.ema_slow, self.slope_lookback)

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

            win_close = df["Close"].iloc[i - self.lookback : i]
            win_fast = df["ema_fast"].iloc[i - self.lookback : i]
            below = int((win_close < win_fast).sum())
            if below / self.lookback < self.consolidation_below_pct:
                continue

            window_high = float(df["High"].iloc[i - self.lookback : i].max())
            window_low = float(df["Low"].iloc[i - self.lookback : i].min())
            range_atr = (window_high - window_low) / float(atr)
            if range_atr > self.max_range_atr:
                continue

            close = float(df["Close"].iloc[i])
            open_ = float(df["Open"].iloc[i])
            if close >= window_low:
                continue
            if close >= open_:
                continue

            breakdown_atr = (window_low - close) / float(atr)
            if breakdown_atr < self.breakdown_atr_mult:
                continue

            signals.append(
                self._build_signal(df, i, window_high, window_low, range_atr, breakdown_atr)
            )
            cooldown_until = i + self.cooldown_bars

        return signals

    def _build_signal(
        self,
        df: pd.DataFrame,
        i: int,
        window_high: float,
        window_low: float,
        range_atr: float,
        breakdown_atr: float,
    ) -> PatternSignal:
        close = float(df["Close"].iloc[i])
        vol_ratio = (
            round(float(df["Volume"].iloc[i] / df["vol_avg"].iloc[i]), 2)
            if not pd.isna(df["vol_avg"].iloc[i]) and df["vol_avg"].iloc[i] > 0
            else 1.0
        )
        return PatternSignal(
            date=df.index[i].date(),
            timestamp=pd.Timestamp(df.index[i]).to_pydatetime(),
            pattern_name=self.name,
            entry_price=close,
            stop_loss=round(window_high, 2),
            confidence=min(1.0 + breakdown_atr * 2, 3.0),
            metadata={
                "direction": "short",
                "breakdown_strength_atr": round(breakdown_atr, 4),
                "range_atr": round(range_atr, 4),
                "consolidation_high": round(window_high, 2),
                "consolidation_low": round(window_low, 2),
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
