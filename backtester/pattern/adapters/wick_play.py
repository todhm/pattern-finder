import pandas as pd

from pattern.domain.models import PatternSignal
from pattern.domain.ports import PatternDetector


class WickPlayDetector(PatternDetector):
    """Wick Play — Oliver Kell's 3-bar reversal setup.

    The Wick Play compresses a full battle between buyers and sellers
    into three bars:

      - Bar ``i-2`` — *Wick bar*. Sellers push price down from the
        session high, leaving a prominent UPPER wick (close sits in
        the lower portion of the bar's range).
      - Bar ``i-1`` — *Inside bar*. Trades entirely inside the wick
        bar's range (``H_{i-1} <= H_{i-2}`` and ``L_{i-1} >= L_{i-2}``).
        Volume typically **dries up**, signalling that selling
        pressure has subsided — a stalemate.
      - Bar ``i`` — *Breakout*. A bullish candle that closes above
        the wick high (``close > H_{i-2}``). Buyers have overwhelmed
        the sellers that created the wick, which often kicks off a
        multi-week move.

    Entry fires at the close of bar ``i``. Stop is either the inside
    bar low (tight) or the wick low (loose) — controlled by
    ``stop_mode``.

    Key risk controls
        - ``min_upper_wick_ratio`` — minimum *upper wick / total
          range* on the wick bar. Defines "prominent" wick.
        - ``max_volume_dryup`` — inside bar volume ceiling relative
          to the wick bar. Values below 1.0 enforce an actual
          dry-up; 1.0 only requires non-expansion.
        - ``max_wick_range_atr`` — Oliver Kell: "if the wick is
          too tall you carry too much risk". Reject setups whose
          wick-bar range exceeds this many ATRs. Set to ``None``
          to disable.

    Psychology score (NVDA 2021-07-12 post-mortem)
        The structural gates above only check the *shape* of the
        three bars. A setup whose shape passes can still fail when
        the underlying psychology doesn't — the wick is distribution
        rather than exhaustion, the breakout has no real buying,
        the setup forms inside an active sell-off, or the wick is
        just marginal. Four additional checks score whether the
        *psychology* of Kell's "sellers → stalemate → buyers"
        transition is actually supported by the tape:

          1. **Wick-volume exhaustion** — wick bar volume must be
             ``<=`` its ``psych_vol_lookback``-day average times
             ``psych_wick_vol_exhaustion_mult``. A wick printed on
             *elevated* volume is distribution (pros selling into
             every bounce), not exhaustion.
          2. **Breakout-volume expansion** — breakout bar volume
             must be ``>`` wick bar volume times
             ``psych_breakout_vol_expansion_mult``. Buyers
             overwhelming sellers should show up as real volume
             expansion, not a mechanical drift above the trigger.
          3. **No prior red streak** — the ``psych_prior_red_streak``
             bars *immediately before* the wick bar must not all
             be bearish. Wick inside an active multi-day sell-off
             is a failed bounce, not a pause in an uptrend.
          4. **Dramatic wick** — ``upper_wick_ratio`` must be
             ``>=`` ``psych_dramatic_wick_ratio``. 0.5 is the
             structural floor; 0.65+ is the "clear rejection
             candle" regime Kell's best examples sit in.

        ``min_psych_score`` (0–4) sets how many of the four must
        pass. Default is 3 — "at least 3 of 4 psychological
        confirmations must support the setup". Set to 0 to turn
        the whole gate off (structural-only detection).
    """

    name = "wick_play"

    def __init__(
        self,
        min_upper_wick_ratio: float = 0.5,
        max_upper_wick_ratio: float | None = None,
        max_volume_dryup: float = 1.0,
        breakout_trigger: str = "wick_high",
        stop_mode: str = "inside_low",
        max_wick_range_atr: float | None = 2.5,
        atr_period: int = 14,
        cooldown_bars: int = 5,
        # --- Psychology score filters ---
        psych_vol_lookback: int = 20,
        psych_wick_vol_exhaustion_mult: float = 1.0,
        psych_breakout_vol_expansion_mult: float = 1.0,
        psych_prior_red_streak: int = 2,
        psych_dramatic_wick_ratio: float = 0.65,
        min_psych_score: int = 3,
        # --- Hard gates from failure-case post-mortem ---
        min_breakout_strength_atr: float = 0.3,
        min_prior_trend_20d: float | None = -0.01,
        max_prior_trend_20d: float | None = None,
        prior_trend_lookback: int = 20,
        min_wick_close_location: float = 0.15,
        min_breakout_close_location: float | None = None,
        # --- Trend Template gates (opt-in; default OFF) ---
        # Data-driven: Wick Play is a CAPITULATION REVERSAL setup,
        # not a Minervini Trend Template continuation. Empirically,
        # the biggest winners (DDOG +21.7%, APA +7.06%) were BELOW
        # their 200-SMA and >25% off 52-week highs at the wick bar
        # — exactly the "weak-looking" stocks that produce the
        # strongest absorption bounces. Forcing ``require_above_sma200``
        # on cuts net P&L by more than the small-loser reduction
        # buys back. Exposed as opt-in for users who want a
        # Minervini-flavored variant of the setup.
        require_above_sma200: bool = False,
        sma_period: int = 200,
        min_pct_of_52w_high: float | None = None,
        pct_high_lookback: int = 252,
    ):
        if breakout_trigger not in ("wick_high", "inside_high"):
            raise ValueError(
                "breakout_trigger must be 'wick_high' or 'inside_high'"
            )
        if stop_mode not in ("inside_low", "wick_low"):
            raise ValueError("stop_mode must be 'inside_low' or 'wick_low'")
        if not 0 <= min_psych_score <= 4:
            raise ValueError("min_psych_score must be between 0 and 4")
        self.min_upper_wick_ratio = min_upper_wick_ratio
        # Upper-wick CEILING — opt-in (off by default). An initial
        # analysis of 20 trades suggested extreme wicks (>0.55)
        # correlated with losers, but that used an off-by-one
        # wick-bar index. Re-analysis with the correct wick bar
        # (entry_idx − 3, since entry = signal + 1) showed the
        # opposite: winner mean uw_ratio 0.672 vs loser 0.591,
        # with heavy overlap and no clean cutoff. DDOG (+21.7%
        # top winner) had uw_ratio = 0.782 and would be dropped
        # by a 0.55 ceiling. Kept as a configurable knob but not
        # a recommended filter.
        self.max_upper_wick_ratio = max_upper_wick_ratio
        self.max_volume_dryup = max_volume_dryup
        self.breakout_trigger = breakout_trigger
        self.stop_mode = stop_mode
        self.max_wick_range_atr = max_wick_range_atr
        self.atr_period = atr_period
        self.cooldown_bars = cooldown_bars
        self.psych_vol_lookback = psych_vol_lookback
        self.psych_wick_vol_exhaustion_mult = psych_wick_vol_exhaustion_mult
        self.psych_breakout_vol_expansion_mult = (
            psych_breakout_vol_expansion_mult
        )
        self.psych_prior_red_streak = psych_prior_red_streak
        self.psych_dramatic_wick_ratio = psych_dramatic_wick_ratio
        self.min_psych_score = min_psych_score
        # Hard gates — derived from deep-dive on 11 failed trades
        # (PODD/LITE/AIZ/CB/PLTR/AEE/HUM/SBAC/ETR/ADM/KHC).
        #
        # ``min_breakout_strength_atr`` — the breakout close must sit
        # at least this many ATRs above the trigger level. Previously
        # implicit at 0; 6 of 11 failures had breakouts under 0.3 ATR
        # (barely-poking-above breakouts = stop-run noise, not a
        # real "buyers overwhelm sellers" event).
        #
        # ``min_prior_trend_20d`` — the wick bar's close must have a
        # 20-day return no worse than this (default -1%). Originally
        # -3% from the HUM/ADM/KHC/AIZ post-mortem; tightened to
        # -1% after walk-through on 20 live trades (2024-06 →
        # 2026-04) showed ret20d is the single most-discriminating
        # feature: every winner had ret20d ≥ -0.5%, while two
        # losers (HRL -2.1%, CHTR -2.0%) sat clearly below. The
        # -0.01 cutoff drops those two without touching any
        # winner. Kell's Wick Play works inside uptrend
        # pullbacks, not mid-decline bounces.
        #
        # ``max_prior_trend_20d`` — optional parabolic cap. When set
        # (e.g. 0.15), rejects setups that have already rallied more
        # than this in the prior N days (PLTR +21% case). Exhausted
        # uptrends turn Wick Plays into tops, not reversals.
        self.min_breakout_strength_atr = min_breakout_strength_atr
        self.min_prior_trend_20d = min_prior_trend_20d
        self.max_prior_trend_20d = max_prior_trend_20d
        self.prior_trend_lookback = prior_trend_lookback
        # ``min_wick_close_location`` — wick-bar close must sit at
        # least this far above the bar's low (as a fraction of its
        # range). Kell's Wick Play psychology: sellers pushed price
        # down from high BUT buyers absorbed at the lows. A wick
        # bar that closes *at* its low (e.g. CL = 0.03) is a
        # bearish-dominance bar, not absorption — sellers were in
        # control all session, not just early. Caught the 2024
        # PODD / ZTS / BG failures where wick CL was 0.03–0.05.
        #
        # ``min_breakout_close_location`` — optional gate on where
        # the BREAKOUT bar closes inside its own range. Weak
        # breakouts that close mid-range (ZTS 0.61) signal the
        # intraday buying faded before the bell.
        self.min_wick_close_location = min_wick_close_location
        self.min_breakout_close_location = min_breakout_close_location
        # Trend Template gates — Minervini-style leadership filter,
        # off by default because Wick Play is a CAPITULATION
        # REVERSAL setup, not a continuation-in-uptrend. Walk-through
        # on 20 live trades (2024-06 → 2026-04) showed both top
        # winners (DDOG +21.7%, APA +7.06%) sat BELOW their 200-SMA
        # and >25% off 52-week highs at the wick bar — the "weak-
        # looking" setups that produce the strongest absorption
        # bounces. With ``require_above_sma200=True``, saved small
        # losers (APTV/REGN/HRL/CL/MKC ≈ +$4.6K) are dwarfed by
        # lost winners (DDOG + APA ≈ -$14K). Net −$9K.
        #
        # Kept as opt-in for users who want a Minervini-flavored
        # variant — e.g. combine with a different pattern where the
        # Trend Template actually applies.
        self.require_above_sma200 = require_above_sma200
        self.sma_period = sma_period
        self.min_pct_of_52w_high = min_pct_of_52w_high
        self.pct_high_lookback = pct_high_lookback

    def detect(
        self,
        df: pd.DataFrame,
        weekly_df: pd.DataFrame | None = None,
        monthly_df: pd.DataFrame | None = None,
    ) -> list[PatternSignal]:
        df = df.copy()
        df["atr"] = self._compute_atr(df, self.atr_period)
        df["vol_avg_psych"] = (
            df["Volume"].rolling(self.psych_vol_lookback).mean()
        )
        df["sma_long"] = (
            df["Close"].rolling(self.sma_period).mean()
            if self.require_above_sma200
            else pd.Series(index=df.index, dtype=float)
        )
        df["pct_of_52w_high"] = (
            df["Close"] / df["Close"].rolling(self.pct_high_lookback).max()
            if self.min_pct_of_52w_high is not None
            else pd.Series(index=df.index, dtype=float)
        )
        is_red = df["Close"] < df["Open"]

        signals: list[PatternSignal] = []
        cooldown_until = -1

        for i in range(2, len(df)):
            if i <= cooldown_until:
                continue

            # --- Bar i-2: wick bar -------------------------------
            w = i - 2
            w_open = float(df["Open"].iloc[w])
            w_high = float(df["High"].iloc[w])
            w_low = float(df["Low"].iloc[w])
            w_close = float(df["Close"].iloc[w])
            w_volume = float(df["Volume"].iloc[w])
            w_range = w_high - w_low
            if w_range <= 0:
                continue

            body_top = max(w_open, w_close)
            upper_wick = w_high - body_top
            upper_wick_ratio = upper_wick / w_range
            if upper_wick_ratio < self.min_upper_wick_ratio:
                continue
            if (
                self.max_upper_wick_ratio is not None
                and upper_wick_ratio > self.max_upper_wick_ratio
            ):
                continue

            # --- Hard gate: wick bar close_location (finding #7) ---
            # Close sitting at the very bottom of the range = sellers
            # dominated to the bell, not "absorbed at the lows"
            # psychology Kell is after.
            wick_close_location = (w_close - w_low) / w_range
            if wick_close_location < self.min_wick_close_location:
                continue

            atr_w = float(df["atr"].iloc[w])
            if self.max_wick_range_atr is not None:
                if pd.isna(atr_w) or atr_w <= 0:
                    continue
                if w_range / atr_w > self.max_wick_range_atr:
                    continue

            # --- Bar i-1: inside bar -----------------------------
            n_high = float(df["High"].iloc[i - 1])
            n_low = float(df["Low"].iloc[i - 1])
            n_volume = float(df["Volume"].iloc[i - 1])
            if n_high > w_high or n_low < w_low:
                continue
            if w_volume > 0 and n_volume > w_volume * self.max_volume_dryup:
                continue

            # --- Bar i: breakout ---------------------------------
            b_open = float(df["Open"].iloc[i])
            b_close = float(df["Close"].iloc[i])
            if b_close <= b_open:
                continue

            trigger_level = (
                w_high if self.breakout_trigger == "wick_high" else n_high
            )
            if b_close <= trigger_level:
                continue

            # --- Hard gate: breakout strength (finding #1) ---------
            # "close > trigger_level" alone accepts 0.01-ATR pokes.
            # Require the breakout to clear the trigger by a
            # meaningful ATR fraction.
            atr_b_gate = float(df["atr"].iloc[i])
            if self.min_breakout_strength_atr > 0:
                if pd.isna(atr_b_gate) or atr_b_gate <= 0:
                    continue
                strength = (b_close - trigger_level) / atr_b_gate
                if strength < self.min_breakout_strength_atr:
                    continue

            # --- Optional gate: breakout-bar close location -------
            # Breakouts that close mid-range (e.g. ZTS 0.61) mean
            # intraday buying faded before the bell. Off by default;
            # set to 0.70+ to require strong top-of-range closes.
            b_range = float(df["High"].iloc[i]) - float(df["Low"].iloc[i])
            breakout_close_location = (
                (b_close - float(df["Low"].iloc[i])) / b_range
                if b_range > 0
                else 0.5
            )
            if self.min_breakout_close_location is not None:
                if breakout_close_location < self.min_breakout_close_location:
                    continue

            # --- Hard gate: 20-day prior-trend filter (finding #2/#3)
            # Wick Plays work inside uptrend pullbacks, not mid-
            # decline bounces. Also optionally cap the upper end to
            # exclude parabolic extensions.
            lb = self.prior_trend_lookback
            prior_trend_20d: float | None = None
            if w >= lb:
                ref_close = float(df["Close"].iloc[w - lb])
                if ref_close > 0:
                    prior_trend_20d = (w_close - ref_close) / ref_close
            if (
                self.min_prior_trend_20d is not None
                or self.max_prior_trend_20d is not None
            ):
                if prior_trend_20d is None:
                    continue
                if (
                    self.min_prior_trend_20d is not None
                    and prior_trend_20d < self.min_prior_trend_20d
                ):
                    continue
                if (
                    self.max_prior_trend_20d is not None
                    and prior_trend_20d > self.max_prior_trend_20d
                ):
                    continue

            # --- Trend Template: primary uptrend (SMA200) -----------
            # Kell/Minervini Wick Play is a pullback inside an
            # uptrend. Stock below its 200-SMA is in a primary
            # downtrend — bounces there are dead-cat, not setups.
            sma_long_w = float(df["sma_long"].iloc[w])
            if self.require_above_sma200:
                if pd.isna(sma_long_w) or w_close <= sma_long_w:
                    continue

            # --- Trend Template: leadership (near 52w high) ---------
            # Leaders trade within 25% of their 52-week highs
            # during momentum regimes; laggards sit 30–50% off.
            # Wick Plays on laggards fail more often than they
            # run (VTRS/HRL/MKC/CL type).
            pct_of_high_w = float(df["pct_of_52w_high"].iloc[w])
            if self.min_pct_of_52w_high is not None:
                if (
                    pd.isna(pct_of_high_w)
                    or pct_of_high_w < self.min_pct_of_52w_high
                ):
                    continue

            # --- Psychology score (4 checks) ---------------------
            # Each check expresses ONE leg of Kell's
            # "sellers → stalemate → buyers" story. A setup whose
            # shape passes but whose psychology doesn't is the
            # 2021-07-12 NVDA failure mode (wick was distribution,
            # breakout had no buying, formed inside a sell-off,
            # upper wick was marginal).
            vol_avg_w = df["vol_avg_psych"].iloc[w]
            check_vol_exhaustion = (
                not pd.isna(vol_avg_w)
                and vol_avg_w > 0
                and w_volume
                <= vol_avg_w * self.psych_wick_vol_exhaustion_mult
            )
            b_volume = float(df["Volume"].iloc[i])
            check_vol_expansion = (
                w_volume > 0
                and b_volume
                > w_volume * self.psych_breakout_vol_expansion_mult
            )
            if self.psych_prior_red_streak > 0 and w >= self.psych_prior_red_streak:
                streak_start = w - self.psych_prior_red_streak
                reds = is_red.iloc[streak_start:w]
                check_no_red_streak = not bool(reds.all())
            else:
                # Not enough prior history to assert a streak —
                # treat as "no streak detected" (pass).
                check_no_red_streak = True
            check_dramatic_wick = (
                upper_wick_ratio >= self.psych_dramatic_wick_ratio
            )

            psych_checks = {
                "vol_exhaustion": check_vol_exhaustion,
                "vol_expansion": check_vol_expansion,
                "no_prior_red_streak": check_no_red_streak,
                "dramatic_wick": check_dramatic_wick,
            }
            psych_score = sum(psych_checks.values())
            if psych_score < self.min_psych_score:
                continue

            stop_loss = n_low if self.stop_mode == "inside_low" else w_low
            atr_b = float(df["atr"].iloc[i])
            atr_safe = atr_b if not pd.isna(atr_b) and atr_b > 0 else 0.0
            breakout_atr = (
                round((b_close - trigger_level) / atr_safe, 4)
                if atr_safe > 0
                else 0.0
            )
            wick_range_atr = (
                round(w_range / atr_w, 3)
                if not pd.isna(atr_w) and atr_w > 0
                else 0.0
            )
            vol_ratio_inside = (
                round(n_volume / w_volume, 2) if w_volume > 0 else 1.0
            )

            signals.append(
                PatternSignal(
                    date=df.index[i].date(),
                    pattern_name=self.name,
                    entry_price=b_close,
                    stop_loss=round(stop_loss, 2),
                    confidence=min(1.0 + upper_wick_ratio, 3.0),
                    metadata={
                        "direction": "long",
                        "wick_high": round(w_high, 2),
                        "wick_low": round(w_low, 2),
                        "inside_high": round(n_high, 2),
                        "inside_low": round(n_low, 2),
                        "upper_wick_ratio": round(upper_wick_ratio, 3),
                        "wick_range_atr": wick_range_atr,
                        "inside_vol_ratio_vs_wick": vol_ratio_inside,
                        "breakout_strength_atr": breakout_atr,
                        "trigger": self.breakout_trigger,
                        "stop_mode": self.stop_mode,
                        "psych_score": psych_score,
                        "psych_vol_exhaustion": check_vol_exhaustion,
                        "psych_vol_expansion": check_vol_expansion,
                        "psych_no_prior_red_streak": check_no_red_streak,
                        "psych_dramatic_wick": check_dramatic_wick,
                        "prior_trend_20d": (
                            round(prior_trend_20d, 4)
                            if prior_trend_20d is not None
                            else None
                        ),
                        "wick_close_location": round(wick_close_location, 3),
                        "breakout_close_location": round(
                            breakout_close_location, 3
                        ),
                        "above_sma_long": (
                            None
                            if pd.isna(sma_long_w) or sma_long_w <= 0
                            else round(w_close / sma_long_w, 3)
                        ),
                        "pct_of_52w_high": (
                            None
                            if pd.isna(pct_of_high_w)
                            else round(pct_of_high_w, 3)
                        ),
                    },
                )
            )
            cooldown_until = i + self.cooldown_bars

        return signals

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
