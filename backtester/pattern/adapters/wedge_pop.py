import numpy as np
import pandas as pd

from pattern.domain.models import PatternSignal
from pattern.domain.ports import PatternDetector


class WedgePopDetector(PatternDetector):
    """Detects Wedge Pop pattern using EMA-based consolidation + breakout.

    Conditions:
    1. Consolidation: >= `consolidation_pct` of last `lookback` bars
       closed below the fast EMA (allows real-market noise).
    2. Breakout: close > both fast & slow EMA.
    3. Breakout strength: the breakout move exceeds
       ``breakout_atr_mult × ATR``, filtering out weak bounces.
       ATR-based thresholds automatically adapt to each stock's
       volatility, so the same multiplier works consistently
       across $10 micro-caps and $500 mega-caps.

    Volume is NOT a hard gate — real breakouts can have average volume
    when the rolling average is skewed by prior spikes (e.g. earnings).
    Volume ratio is reported in `metadata.volume_ratio` as a confidence
    indicator instead.

    **Late-entry path** (``late_entry_bars > 0``)
        The primary path requires a *fresh* cross — the previous bar
        must still sit below the fast EMA. That gates out continuation
        bars on Day 2/3 of a breakout, which is usually the right
        call, but some of the most decisive wedge-pop candles print
        one or two bars *after* the first close-above-EMA (the Day-1
        cross is often a marginal / indecisive bar, and the real
        buying arrives on Day 2/3). Setting ``late_entry_bars`` to
        ``N > 0`` adds a secondary path: if the fresh-cross check
        fails but *any* bar within the last ``N + 1`` bars closed
        below the fast EMA (i.e. the cross happened at most ``N``
        bars ago), the signal still fires and is tagged
        ``trigger=late_entry`` in metadata. The consolidation
        ratio check still applies, so noise in sideways regimes
        is still rejected.
    """

    name = "wedge_pop"

    def __init__(
        self,
        lookback: int = 10,
        ema_fast: int = 10,
        ema_slow: int = 20,
        consolidation_pct: float = 0.6,
        max_consolidation_pct: float | None = None,
        breakout_atr_mult: float = 0.5,
        max_breakout_atr_mult: float | None = None,
        atr_period: int = 14,
        slope_lookback: int = 20,
        cooldown_bars: int | None = None,
        require_above_long_smas: bool = True,
        sma_mid: int = 50,
        sma_long: int = 200,
        late_entry_bars: int = 0,
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
        # Breakout strength gate in ATR units.
        # ``breakout_atr_mult`` is the MIN expansion required,
        # measured as ``max(ema_distance, daily_move) / ATR``.
        # ``max_breakout_atr_mult`` caps the upper end so you can
        # filter out runaway gaps that are already overextended
        # (e.g. cap at 3.0 to skip +4 ATR single-bar moves).
        # Using ATR instead of fixed % means the gate automatically
        # scales with each stock's volatility — "0.5 ATR" is a
        # comparable move on a $10 penny stock and a $500 blue chip.
        self.breakout_atr_mult = breakout_atr_mult
        self.max_breakout_atr_mult = max_breakout_atr_mult
        self.atr_period = atr_period
        self.slope_lookback = slope_lookback
        # Cooldown — number of bars to wait after a fire before the
        # next signal can arm. Historically this was hard-coded to
        # ``lookback`` (so a consolidation window and the cooldown
        # shared the same value), which meant tuning the
        # consolidation window also tightened/loosened the cooldown
        # as a side effect. It's now a separate knob, defaulting to
        # ``lookback`` for backwards compatibility. Set to 0 to
        # allow back-to-back signals.
        self.cooldown_bars = cooldown_bars if cooldown_bars is not None else lookback
        self.require_above_long_smas = require_above_long_smas
        self.sma_mid = sma_mid
        self.sma_long = sma_long
        # Late-entry window (in bars). 0 = strict "fresh cross only"
        # behaviour. When > 0, a continuation bar fires as long as the
        # original cross-above-EMA happened within the last N bars.
        self.late_entry_bars = late_entry_bars

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
        df["atr"] = self._compute_atr(df, self.atr_period)
        # Rolling linear-regression slope of each EMA over
        # ``slope_lookback`` bars. Captures the *current direction*
        # of the EMA, unlike endpoint percent-change which can be
        # positive while the EMA is actually falling.
        n = self.slope_lookback
        df["ema_fast_slope"] = self._regression_slope(df["ema_fast"], n)
        df["ema_slow_slope"] = self._regression_slope(df["ema_slow"], n)

        # Vectorized candidate mask — equivalent to running the per-bar
        # ``_was_consolidated`` + ``_breakout_trigger`` checks via numpy
        # array ops. Cooldown is still applied sequentially because it
        # has stateful dependency on prior fires. Preserves the exact
        # semantics of the loop-based implementation, including the
        # late-entry path, the NaN-resilient SMA gate, and the ATR
        # strength bounds.
        n = len(df)
        close = df["Close"].to_numpy(dtype=float)
        open_ = df["Open"].to_numpy(dtype=float)
        fast = df["ema_fast"].to_numpy(dtype=float)
        slow = df["ema_slow"].to_numpy(dtype=float)
        atr = df["atr"].to_numpy(dtype=float)
        sma_mid = df["sma_mid"].to_numpy(dtype=float)
        sma_long = df["sma_long"].to_numpy(dtype=float)

        # Consolidation ratio — prior ``lookback`` bars where
        # ``close < ema_fast``. Rolling sum over the boolean, shifted
        # one bar so it's strictly prior (matches the original
        # ``for j in range(1, lookback+1)`` semantics).
        below_mask = close < fast
        prior_below = (
            pd.Series(below_mask.astype(np.float64))
            .rolling(self.lookback)
            .sum()
            .shift(1)
            .to_numpy()
        )
        cons_ratio = prior_below / self.lookback
        cons_ok = np.where(
            np.isnan(cons_ratio), False, cons_ratio >= self.consolidation_pct
        )
        if self.max_consolidation_pct is not None:
            cons_ok = cons_ok & np.where(
                np.isnan(cons_ratio), False, cons_ratio <= self.max_consolidation_pct
            )

        # Prior-bar values for primary/late-entry trigger + gap gate.
        prev_close = np.roll(close, 1)
        prev_open = np.roll(open_, 1)
        prev_ema_fast = np.roll(fast, 1)
        # Index 0 has no valid prior; will be masked out via min_idx.

        primary_mask = prev_close < prev_ema_fast
        if self.late_entry_bars > 0:
            recent_below = (
                pd.Series(below_mask.astype(np.float64))
                .rolling(self.late_entry_bars + 1)
                .sum()
                .shift(1)
                .to_numpy()
                > 0
            )
            late_mask = (~primary_mask) & recent_below
        else:
            late_mask = np.zeros(n, dtype=bool)
        trigger_ok = primary_mask | late_mask

        # ATR strength (max of ema-distance and daily-move in ATR
        # units). Matches the loop's ``max(ema_distance, daily_move)``.
        atr_valid = atr > 0
        resistance = np.maximum(fast, slow)
        with np.errstate(divide="ignore", invalid="ignore"):
            ema_distance = np.where(
                atr_valid, (close - resistance) / atr, -np.inf
            )
            daily_move = np.where(
                atr_valid, (close - prev_close) / atr, -np.inf
            )
        strength = np.maximum(ema_distance, daily_move)
        strength_ok = atr_valid & (strength >= self.breakout_atr_mult)
        if self.max_breakout_atr_mult is not None:
            strength_ok = strength_ok & (strength <= self.max_breakout_atr_mult)

        # Long-term SMA gate — NaNs resolve to False (matches
        # ``pd.isna`` short-circuit in the loop).
        if self.require_above_long_smas:
            sma_valid = ~(np.isnan(sma_mid) | np.isnan(sma_long))
            above_sma = sma_valid & (close > sma_mid) & (close > sma_long)
        else:
            above_sma = np.ones(n, dtype=bool)

        # Composite candidate mask. Early bars (< lookback) are
        # excluded to match ``for i in range(min_idx, len(df))``.
        candidate_mask = (
            cons_ok
            & (close > fast)
            & (close > slow)
            & (close > open_)
            & trigger_ok
            & (close >= prev_open)  # loop returns None when close < prev_open
            & strength_ok
            & above_sma
        )
        candidate_mask[: self.lookback] = False

        # Cooldown is sequential (prior fires gate future candidates).
        signals: list[PatternSignal] = []
        cooldown_until = -1
        for i in np.flatnonzero(candidate_mask):
            if i <= cooldown_until:
                continue
            trigger = "primary" if primary_mask[i] else "late_entry"
            signals.append(self._build_signal(df, int(i), trigger=trigger))
            cooldown_until = int(i) + self.cooldown_bars

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
        if self.max_consolidation_pct is not None and ratio > self.max_consolidation_pct:
            return False
        return True

    def _breakout_trigger(self, df: pd.DataFrame, i: int) -> str | None:
        """Return the trigger label (``"primary"`` / ``"late_entry"``)
        when bar ``i`` qualifies as a wedge-pop breakout, or ``None``
        when it doesn't.

        Passes the breakout-strength gate if EITHER:
        - EMA distance: (close - resistance) >= breakout_atr_mult × ATR, OR
        - Daily momentum: (close - prev_close) >= breakout_atr_mult × ATR.
        The second condition catches breakouts where the EMAs are close to
        price (tight consolidation) but the daily candle is clearly bullish.

        Using ATR as the yardstick means the same multiplier (e.g. 0.5)
        automatically adapts to high-vol and low-vol regimes — no need to
        retune the threshold per stock or per market cycle.

        Additional guards:
        - ``close >= prev_open``: a "breakout" whose close is below where
          the prior session opened is by definition not a wedge pop —
          it's an upthrust or failed rally.
        - ``require_above_long_smas``: when True, the close must sit
          above BOTH the 50 SMA and 200 SMA. Structural filter for
          "wedge pop inside an established uptrend only".

        **Fresh cross vs late entry**:
        The primary path requires the previous bar to still be below
        the fast EMA (fresh breakout). When that fails, the late-entry
        path fires instead if the cross happened within the last
        ``late_entry_bars`` bars (``late_entry_bars=0`` disables).
        """
        close = df["Close"].iloc[i]
        open_ = df["Open"].iloc[i]
        fast = df["ema_fast"].iloc[i]
        slow = df["ema_slow"].iloc[i]

        if close <= fast or close <= slow:
            return None

        # The breakout candle itself must be bullish (close > open).
        if close <= open_:
            return None

        # Decide primary vs late-entry based on where the cross sits.
        prev_close = df["Close"].iloc[i - 1]
        prev_ema_fast = df["ema_fast"].iloc[i - 1]
        if prev_close < prev_ema_fast:
            trigger = "primary"
        else:
            # Continuation bar. Allow only when the cross is still
            # recent — i.e. some bar within the last
            # ``late_entry_bars + 1`` closed below the fast EMA.
            if self.late_entry_bars <= 0:
                return None
            recent_below = False
            window = min(self.late_entry_bars + 1, i)
            for k in range(1, window + 1):
                if df["Close"].iloc[i - k] < df["ema_fast"].iloc[i - k]:
                    recent_below = True
                    break
            if not recent_below:
                return None
            trigger = "late_entry"

        prev_open = df["Open"].iloc[i - 1]
        if close < prev_open:
            return None

        if self.require_above_long_smas:
            sma_mid = df["sma_mid"].iloc[i]
            sma_long = df["sma_long"].iloc[i]
            if pd.isna(sma_mid) or pd.isna(sma_long):
                return None
            if close <= sma_mid or close <= sma_long:
                return None

        atr = float(df["atr"].iloc[i]) if not pd.isna(df["atr"].iloc[i]) else 0.0
        if atr <= 0:
            return None

        resistance = max(fast, slow)
        ema_distance = (close - resistance) / atr
        daily_move = (close - prev_close) / atr

        # "Strength" is the max of the two candidate metrics —
        # whichever leg would have triggered the OR gate is the
        # one driving the breakout. Gate on
        # ``[breakout_atr_mult, max_breakout_atr_mult]``.
        strength = max(ema_distance, daily_move)
        if strength < self.breakout_atr_mult:
            return None
        if self.max_breakout_atr_mult is not None and strength > self.max_breakout_atr_mult:
            return None
        return trigger

    # ---- signal building ----

    def _build_signal(
        self, df: pd.DataFrame, i: int, trigger: str = "primary"
    ) -> PatternSignal:
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
        atr = float(df["atr"].iloc[i]) if not pd.isna(df["atr"].iloc[i]) else 0.0
        # Report both %-based and ATR-based breakout strength in
        # metadata so the user can compare. The ATR-based value is
        # the one used for the gate; the %-based value is kept for
        # human readability (e.g. "broke out +2.5% above EMA").
        breakout_strength_pct = round((df["Close"].iloc[i] - resistance) / resistance, 4)
        breakout_strength_atr = round((df["Close"].iloc[i] - resistance) / atr, 4) if atr > 0 else 0.0

        fast_slope_raw = df["ema_fast_slope"].iloc[i]
        slow_slope_raw = df["ema_slow_slope"].iloc[i]
        ema_fast_slope = round(float(fast_slope_raw), 4) if not pd.isna(fast_slope_raw) else 0.0
        ema_slow_slope = round(float(slow_slope_raw), 4) if not pd.isna(slow_slope_raw) else 0.0

        return PatternSignal(
            date=df.index[i].date(),
            timestamp=pd.Timestamp(df.index[i]).to_pydatetime(),
            pattern_name=self.name,
            entry_price=df["Close"].iloc[i],
            stop_loss=consolidation_low,
            confidence=min(1.0 + breakout_strength_atr * 2, 3.0),
            metadata={
                "trigger": trigger,
                "breakout_strength": breakout_strength_pct,
                "breakout_strength_atr": breakout_strength_atr,
                "volume_ratio": vol_ratio,
                "consolidation_low": round(consolidation_low, 2),
                "ema_fast_slope": ema_fast_slope,
                "ema_slow_slope": ema_slow_slope,
                "slope_lookback": self.slope_lookback,
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
