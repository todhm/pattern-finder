import pandas as pd

from pattern.domain.models import PatternSignal
from pattern.domain.ports import PatternDetector


class ExhaustionExtensionTopDetector(PatternDetector):
    """Exhaustion Extension — euphoric blow-off at a top.

    Fires when, inside a confirmed uptrend, the bar's ``High`` prints
    more than ``extension_atr_mult × ATR`` above the fast EMA. This is
    the "air between price and the 10 EMA" described in the Downtrends
    doc — an early *warning* that the uptrend is overextended. The
    actual trend change only gets confirmed when a Wedge Drop follows.

    The detector combines two **distribution confirmations** as
    additional evidence of selling at the highs:

        - **Bearish Candlestick**: close in the LOWER
          ``max_close_location`` of the bar's range (shooting-star /
          upper-wick shape — sellers absorbed the intraday rally).
          Set to 1.0 to disable.

        - **Sell Dominance**: over ``pressure_lookback`` bars,
          ``Σ(down-bar volume) / Σ(up-bar volume) >= min_sell_dominance``.
          Compares actual selling vs buying volume in the SAME window —
          strictly captures whether sellers outweigh buyers, instead of
          merely whether total volume is up. Down-bar = close < open,
          up-bar = close >= open. Set to 0.0 to disable.

    ``require_all_confirmations=True`` flips the combinator from OR
    (any sign is enough) to AND (both signs required) for tighter
    filtering. Disabled confirmations (off) are ignored in either
    mode; if ALL are disabled, price extension alone fires.

    **Upper-wick Rejection Override** (``enable_rejection_override``)
    adds a secondary detection path: if the bar prints a very strong
    upper-wick rejection (``close_location <= rejection_close_location``,
    default 0.25 — lower quarter of the bar) while still inside an
    uptrend with extension / slope within ``rejection_leniency`` of
    their thresholds (90% by default), the bar fires **bypassing the
    sell-dominance confirmation and the cooldown window**. Rationale:
    a long upper wick IS direct evidence of distribution (sellers
    absorbed the intraday rally at the highs), so requiring an
    independent volume confirmation would double-count the same
    signal. Cooldown is bypassed because a fresh rejection candle
    represents a new topping attempt regardless of prior activity.

    Short signal: ``entry_price`` at close, ``stop_loss`` above the
    blow-off high (invalidation level for the short).

    Conditions (primary path)
        - Uptrend: ``ema_fast > ema_slow``.
        - Slow EMA rising: ``ema_slow_slope >= min_slow_slope``.
        - ``(high - ema_fast) / ATR >= extension_atr_mult``.
        - Distribution confirmation: combined with OR (default) or
          AND (``require_all_confirmations=True``).

    Conditions (rejection-override path, when enabled)
        - Uptrend: ``ema_fast > ema_slow``.
        - ``ema_slow_slope >= min_slow_slope × rejection_leniency``.
        - ``extension_atr >= extension_atr_mult × rejection_leniency``.
        - ``close_location <= rejection_close_location``.
    """

    name = "exhaustion_extension_top"

    def __init__(
        self,
        extension_atr_mult: float = 3.0,
        min_slow_slope: float = 0.005,
        max_close_location: float = 0.5,
        min_sell_dominance: float = 1.5,
        pressure_lookback: int = 10,
        require_all_confirmations: bool = False,
        enable_rejection_override: bool = True,
        rejection_close_location: float = 0.25,
        rejection_leniency: float = 0.9,
        ema_fast: int = 10,
        ema_slow: int = 20,
        atr_period: int = 14,
        slope_lookback: int = 20,
        cooldown_bars: int = 10,
    ):
        self.extension_atr_mult = extension_atr_mult
        self.min_slow_slope = min_slow_slope
        self.max_close_location = max_close_location
        self.min_sell_dominance = min_sell_dominance
        self.pressure_lookback = pressure_lookback
        self.require_all_confirmations = require_all_confirmations
        self.enable_rejection_override = enable_rejection_override
        self.rejection_close_location = rejection_close_location
        self.rejection_leniency = rejection_leniency
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
        df["atr"] = self._compute_atr(df, self.atr_period)
        n = self.slope_lookback
        df["ema_slow_slope"] = self._regression_slope(df["ema_slow"], n)

        signals: list[PatternSignal] = []
        cooldown_until = -1
        start = max(self.ema_slow, self.slope_lookback)

        for i in range(start, len(df)):
            atr = df["atr"].iloc[i]
            if pd.isna(atr) or atr <= 0:
                continue

            ema_fast = float(df["ema_fast"].iloc[i])
            ema_slow = float(df["ema_slow"].iloc[i])
            if ema_fast <= ema_slow:
                continue

            slope = df["ema_slow_slope"].iloc[i]
            high = float(df["High"].iloc[i])
            low = float(df["Low"].iloc[i])
            close = float(df["Close"].iloc[i])
            bar_range = high - low
            close_location = (
                (close - low) / bar_range if bar_range > 0 else 0.5
            )
            extension_atr = (high - ema_fast) / float(atr)

            # --- Rejection override path ---
            # Long upper-wick itself is direct distribution evidence, so
            # we bypass both the sell-dominance confirmation AND the
            # cooldown window here. Still gated on uptrend + ~90% of
            # extension/slope thresholds so it won't fire on trivial
            # pullbacks.
            if self.enable_rejection_override:
                lenient = self.rejection_leniency
                if (
                    close_location <= self.rejection_close_location
                    and not pd.isna(slope)
                    and slope >= self.min_slow_slope * lenient
                    and extension_atr >= self.extension_atr_mult * lenient
                ):
                    sell_dominance = self._sell_dominance(df, i)
                    signals.append(
                        self._build_signal(
                            df, i, extension_atr, close_location, sell_dominance,
                            trigger="rejection_override",
                        )
                    )
                    cooldown_until = i + self.cooldown_bars
                    continue

            # --- Normal path ---
            if i <= cooldown_until:
                continue
            if pd.isna(slope) or slope < self.min_slow_slope:
                continue
            if extension_atr < self.extension_atr_mult:
                continue

            sell_dominance = self._sell_dominance(df, i)

            close_enabled = self.max_close_location < 1.0
            dominance_enabled = self.min_sell_dominance > 0.0

            if close_enabled or dominance_enabled:
                close_ok = (
                    close_location <= self.max_close_location
                    if close_enabled
                    else None
                )
                dominance_ok = (
                    sell_dominance >= self.min_sell_dominance
                    if dominance_enabled
                    else None
                )
                checks = [c for c in (close_ok, dominance_ok) if c is not None]
                if self.require_all_confirmations:
                    if not all(checks):
                        continue
                else:
                    if not any(checks):
                        continue

            signals.append(
                self._build_signal(
                    df, i, extension_atr, close_location, sell_dominance,
                    trigger="primary",
                )
            )
            cooldown_until = i + self.cooldown_bars

        return signals

    def _sell_dominance(self, df: pd.DataFrame, i: int) -> float:
        """Σ(down-bar volume) / Σ(up-bar volume) over ``pressure_lookback``.

        Down-bar = ``Close < Open`` (bearish), up-bar = ``Close >= Open``.
        Higher value means sellers outweigh buyers in the recent window.
        Returns 1.0 when data is insufficient and ``inf`` when there's
        only down-bar volume (no buying at all).
        """
        lb = self.pressure_lookback
        if i < lb:
            return 1.0

        window = df.iloc[i - lb: i]
        is_down = window["Close"] < window["Open"]
        down_vol = float(window["Volume"].where(is_down, 0.0).sum())
        up_vol = float(window["Volume"].where(~is_down, 0.0).sum())

        if up_vol <= 0:
            return float("inf") if down_vol > 0 else 1.0
        return down_vol / up_vol

    def _build_signal(
        self,
        df: pd.DataFrame,
        i: int,
        extension_atr: float,
        close_location: float,
        sell_dominance: float,
        trigger: str = "primary",
    ) -> PatternSignal:
        close = float(df["Close"].iloc[i])
        high = float(df["High"].iloc[i])
        dominance_bonus = (
            min(sell_dominance, 5.0) - 1.0 if sell_dominance != float("inf") else 4.0
        )
        confidence = (
            1.0
            + (extension_atr - self.extension_atr_mult) * 0.5
            + (1.0 - close_location) * 0.5
            + max(dominance_bonus, 0.0) * 0.3
        )
        return PatternSignal(
            date=df.index[i].date(),
            pattern_name=self.name,
            entry_price=close,
            stop_loss=round(high, 2),
            confidence=min(confidence, 3.0),
            metadata={
                "direction": "short",
                "trigger": trigger,
                "extension_atr": round(extension_atr, 4),
                "close_location": round(close_location, 3),
                "sell_dominance": (
                    round(sell_dominance, 2)
                    if sell_dominance != float("inf")
                    else "inf"
                ),
                "ema_fast": round(float(df["ema_fast"].iloc[i]), 2),
                "ema_slow": round(float(df["ema_slow"].iloc[i]), 2),
                "blow_off_high": round(high, 2),
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
