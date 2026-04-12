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
        max_consolidation_pct: float | None = None,
        breakout_pct: float = 0.015,
        max_breakout_pct: float | None = None,
        slope_lookback: int = 20,
        cooldown_bars: int | None = None,
        require_above_long_smas: bool = False,
        sma_mid: int = 50,
        sma_long: int = 200,
    ):
        self.lookback = lookback
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        # ``consolidation_pct`` is the MIN ratio of prior-lookback
        # closes that must sit below the fast EMA for a signal to
        # arm. ``max_consolidation_pct``, when set, is the upper
        # bound — useful for isolating "base-on-base" setups where
        # too deep a consolidation (e.g. 95%+) is actually a
        # capitulation signature, not a coil.
        self.consolidation_pct = consolidation_pct
        self.max_consolidation_pct = max_consolidation_pct
        # Same idea for the breakout strength gate. ``breakout_pct``
        # is the MIN expansion required. ``max_breakout_pct`` caps
        # the upper end so you can filter out runaway gaps that
        # would normally pass the strength check but are already
        # overextended (e.g. you can set the cap to 10% to skip
        # +15% single-bar moves).
        self.breakout_pct = breakout_pct
        self.max_breakout_pct = max_breakout_pct
        self.slope_lookback = slope_lookback
        # Cooldown — number of bars to wait after a fire before the
        # next signal can arm. Historically this was hard-coded to
        # ``lookback`` (so a consolidation window and the cooldown
        # shared the same value), which meant tuning the
        # consolidation window also tightened/loosened the cooldown
        # as a side effect. It's now a separate knob, defaulting to
        # ``lookback`` for backwards compatibility. Set to 0 to
        # allow back-to-back signals.
        self.cooldown_bars = (
            cooldown_bars if cooldown_bars is not None else lookback
        )
        self.require_above_long_smas = require_above_long_smas
        self.sma_mid = sma_mid
        self.sma_long = sma_long

    def detect(
        self,
        df: pd.DataFrame,
        weekly_df: pd.DataFrame | None = None,
        monthly_df: pd.DataFrame | None = None,
    ) -> list[PatternSignal]:
        df = df.copy()
        df["ema_fast"] = self._add_ema(df, self.ema_fast)
        df["ema_slow"] = self._add_ema(df, self.ema_slow)
        df["sma_mid"] = df["Close"].rolling(self.sma_mid).mean()
        df["sma_long"] = df["Close"].rolling(self.sma_long).mean()
        df["vol_avg"] = self._avg_volume(df)
        # Recent EMA slopes (percent change over ``slope_lookback``
        # bars). Exposed purely as a signal-metadata variable so
        # downstream strategies can filter out wedge pops where the
        # medium-term trend is still rolling over. NOT used as a
        # rejection condition inside the detector itself.
        n = self.slope_lookback
        fast_prev = df["ema_fast"].shift(n)
        slow_prev = df["ema_slow"].shift(n)
        df["ema_fast_slope"] = (df["ema_fast"] - fast_prev) / fast_prev
        df["ema_slow_slope"] = (df["ema_slow"] - slow_prev) / slow_prev

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
            cooldown_until = i + self.cooldown_bars

        return signals

    # ---- condition checks ----

    def _was_consolidated(self, df: pd.DataFrame, i: int) -> bool:
        """The ratio of prior-``lookback`` closes below the fast EMA
        must sit inside ``[consolidation_pct, max_consolidation_pct]``.
        The upper bound is optional — leave ``max_consolidation_pct``
        as ``None`` for the classic "at least X% below" semantic.
        """
        below = 0
        for j in range(1, self.lookback + 1):
            idx = i - j
            if df["Close"].iloc[idx] < df["ema_fast"].iloc[idx]:
                below += 1
        ratio = below / self.lookback
        if ratio < self.consolidation_pct:
            return False
        if (
            self.max_consolidation_pct is not None
            and ratio > self.max_consolidation_pct
        ):
            return False
        return True

    def _is_breakout(self, df: pd.DataFrame, i: int) -> bool:
        """Close above both EMAs with meaningful strength.

        Passes if EITHER:
        - EMA distance: close is >= breakout_pct above the higher EMA, OR
        - Daily momentum: close rose >= breakout_pct from previous close.
        The second condition catches breakouts where the EMAs are close to
        price (tight consolidation) but the daily candle is clearly bullish.

        Additional guards:
        - ``close >= prev_open``: a "breakout" whose close is below where
          the prior session opened is by definition not a wedge pop —
          it's an upthrust or failed rally.
        - ``require_above_long_smas``: when True, the close must sit
          above BOTH the 50 SMA and 200 SMA. Structural filter for
          "wedge pop inside an established uptrend only".
        """
        close = df["Close"].iloc[i]
        open_ = df["Open"].iloc[i]
        fast = df["ema_fast"].iloc[i]
        slow = df["ema_slow"].iloc[i]

        if close <= fast or close <= slow:
            return False

        # The breakout candle itself must be bullish (close > open).
        if close <= open_:
            return False

        # The breakout must be a FRESH cross above the fast EMA. If
        # the previous bar already closed above the fast EMA, today
        # is a continuation — not a wedge pop.
        prev_close = df["Close"].iloc[i - 1]
        prev_ema_fast = df["ema_fast"].iloc[i - 1]
        if prev_close >= prev_ema_fast:
            return False

        prev_open = df["Open"].iloc[i - 1]
        if close < prev_open:
            return False

        if self.require_above_long_smas:
            sma_mid = df["sma_mid"].iloc[i]
            sma_long = df["sma_long"].iloc[i]
            # If either SMA is still NaN (not enough history) we treat
            # the filter as failing — better to skip than to let an
            # unconfirmed long-term trend through.
            if pd.isna(sma_mid) or pd.isna(sma_long):
                return False
            if close <= sma_mid or close <= sma_long:
                return False

        resistance = max(fast, slow)
        ema_strength = (close - resistance) / resistance

        prev_close = df["Close"].iloc[i - 1]
        daily_move = (close - prev_close) / prev_close

        # "Strength" is the max of the two candidate metrics —
        # whichever leg would have triggered the OR gate is the
        # one driving the breakout. Gate on
        # ``[breakout_pct, max_breakout_pct]``: min bound is the
        # required expansion; optional max bound caps runaway gaps
        # that are already overextended.
        strength = max(ema_strength, daily_move)
        if strength < self.breakout_pct:
            return False
        if (
            self.max_breakout_pct is not None
            and strength > self.max_breakout_pct
        ):
            return False
        return True

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

        fast_slope_raw = df["ema_fast_slope"].iloc[i]
        slow_slope_raw = df["ema_slow_slope"].iloc[i]
        ema_fast_slope = (
            round(float(fast_slope_raw), 4)
            if not pd.isna(fast_slope_raw)
            else 0.0
        )
        ema_slow_slope = (
            round(float(slow_slope_raw), 4)
            if not pd.isna(slow_slope_raw)
            else 0.0
        )

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
                "ema_fast_slope": ema_fast_slope,
                "ema_slow_slope": ema_slow_slope,
                "slope_lookback": self.slope_lookback,
            },
        )
