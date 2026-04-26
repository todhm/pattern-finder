"""Fair Value Gap (FVG) — long-side intraday reversal pattern.

Three structural conditions must line up, in order, **within a single
NY session** for the detector to emit a signal:

1. **First Change of Character (CHoCH)** — a *true* lower-lows-then-
   reversal break. Specifically:

      - The session has printed at least two confirmed swing lows
        and the most recent is **strictly lower** than the one
        before it (down-structure: L2 < L1).
      - There is a swing high *between* L1 and L2 — call it H1.
      - A bar's close finally exceeds H1 *after* L2 has printed.

   This matches the framework's ``down → up → down → up → new low →
   rally above prior swing`` definition — without the lower-lows
   precondition, the looser "first close above any prior swing high"
   reading produces false positives in trending markets.

2. **Bullish Fair Value Gap** after the CHoCH — three consecutive
   bars ``[i-2, i-1, i]`` that satisfy *both*:

      - The first bar's high is strictly below the third bar's low
        (wicks don't overlap → the gap is real).
      - All three bars are bullish (``close > open``). The framework's
        diagrams show three green candles; without this same-color
        requirement the detector picks up "gap-up after a red candle"
        sequences that are mechanically gaps but not the impulse-leg
        FVGs the strategy targets.

3. **Mid-point hold** — by construction the third bar of a bullish
   FVG has ``low > high[i-2]``, so its close is automatically above
   the gap's midpoint. Entry trigger satisfied at the bar that
   completes the gap.

The detector emits one ``PatternSignal`` per qualifying FVG, with
**at most one signal per CHoCH structure**. As soon as an FVG's
midpoint retest fires, the active CHoCH state is cleared and any
queued (but unfired) FVGs from that same structural break are
discarded — a second signal in the same session requires a fresh
CHoCH (price retraces, prints a new lower-low, then breaks the
intermediate swing high again). ``max_signals_per_session`` then
caps how many *distinct* CHoCHs can produce signals on the same NY
date. Downstream strategy code is still responsible for the "one
position at a time" rule.

Metadata attached to each signal::

    fvg_low                       # gap's lower edge (= high of the 1st bar)
    fvg_high                      # gap's upper edge (= low of the 3rd bar)
    fvg_mid                       # midpoint, used as the BOS-trail target
    fvg_producing_low             # low of the middle (2nd) bar — the stop
    choch_high                    # the swing high (H1) the CHoCH cleared
    choch_bar_offset              # bars between CHoCH and FVG completion
    choch_timestamp               # ISO ts of the CHoCH break bar
    choch_break_level_start_ts    # ISO ts of H1 (= where the broken
                                  # level first printed)
    fvg_start_timestamp           # ISO ts of bar i-2 (FVG zone left edge)
    gap_size_pct                  # gap width as a fraction of close
"""

from __future__ import annotations

from datetime import time as _dt_time

import numpy as np
import pandas as pd

from pattern.domain.models import PatternSignal
from pattern.domain.ports import PatternDetector
from pattern.helpers.pivots import find_swing_highs, find_swing_lows

# US regular session window. Bars whose START falls inside this window
# count as RTH (a 15m bar starting at 15:45 covers 15:45–16:00). The
# detector forms FVGs only within RTH, even though the upstream frame
# may include pre/post bars — those still flow into ChoCH detection
# so an early structural break still anchors the session, but the
# 3-bar gap itself only forms during the high-liquidity window.
_RTH_OPEN = _dt_time(9, 30)
_RTH_CLOSE = _dt_time(16, 0)


class FairValueGapDetector(PatternDetector):
    """First-CHoCH → Bullish FVG detector.

    Designed for sub-daily intraday frames (15m default; works on 1m
    / 5m too). On daily frames the "session" partitioning collapses
    to one bar per day so the detector emits at most a handful of
    signals — usable but not its primary cadence.
    """

    name = "fair_value_gap"

    def __init__(
        self,
        swing_left: int = 2,
        swing_right: int = 2,
        min_gap_pct: float = 0.003,
        max_bars_after_choch: int = 40,
        max_signals_per_session: int = 2,
        min_choch_swing_atr: float | None = 2.0,
        atr_period: int = 14,
        max_retest_bars: int = 30,
    ) -> None:
        # Swing fractal half-widths. Default 2/2 matches Bill Williams'
        # 5-bar fractal — small enough to surface micro-structure on
        # 15m bars, large enough to ignore single-bar noise.
        self.swing_left = swing_left
        self.swing_right = swing_right
        # Minimum gap size as a fraction of the FVG-completion bar's
        # close. ``0.003`` ≈ 30 bps — wide enough that the gap
        # represents a real impulse leg and not bid/ask noise on a
        # quiet bar. Earlier default (``0.001``) was too permissive
        # and surfaced micro-gaps that the user flagged as
        # "looks-like-FVG-but-isn't".
        self.min_gap_pct = min_gap_pct
        # FVG must form within this many bars of the CHoCH or it's
        # too late — the structural break has already faded into a
        # generic uptrend the detector wasn't built to chase.
        self.max_bars_after_choch = max_bars_after_choch
        # Cap signals per session. With the "1 signal per CHoCH"
        # rule, this effectively caps how many distinct CHoCH
        # structures (lower-lows + break) can fire signals on the
        # same NY date. Two strong intraday reversals in one session
        # are rare but possible; default 2 keeps that headroom.
        self.max_signals_per_session = max_signals_per_session
        # Minimum CHoCH swing magnitude in ATR multiples. ``H1 - L2``
        # (the down-leg the rally has to undo) must be ≥ this many
        # ATRs measured at L2's bar. Filters out the user-flagged
        # failure mode where the structural break happens on a
        # micro-bounce: tiny swings look like CHoCH on paper but
        # carry no momentum into the FVG. ``None`` disables the
        # gate entirely.
        self.min_choch_swing_atr = min_choch_swing_atr
        # ATR period for the magnitude filter above. EMA-smoothed,
        # matching the convention the wedgepop / wickplay detectors use.
        self.atr_period = atr_period
        # Maximum bars between FVG completion and the midpoint
        # retest entry trigger. After this many bars the pending FVG
        # is discarded — a setup that hasn't been retested in 30
        # bars is essentially a strong directional move that won't
        # come back, no entry available.
        self.max_retest_bars = max_retest_bars

    # ---- detection ------------------------------------------------------

    def detect(
        self,
        df: pd.DataFrame,
        weekly_df: pd.DataFrame | None = None,
        monthly_df: pd.DataFrame | None = None,
    ) -> list[PatternSignal]:
        n = len(df)
        if n < 5:
            return []

        # Pre-compute swing pivots once. ``find_swing_*`` returns NaN
        # for non-pivot bars; we read them off as we sweep.
        swing_high_arr = find_swing_highs(
            df, left=self.swing_left, right=self.swing_right
        ).to_numpy()
        swing_low_arr = find_swing_lows(
            df, left=self.swing_left, right=self.swing_right
        ).to_numpy()

        opens = df["Open"].to_numpy(dtype=float)
        highs = df["High"].to_numpy(dtype=float)
        lows = df["Low"].to_numpy(dtype=float)
        closes = df["Close"].to_numpy(dtype=float)
        atr_arr = self._compute_atr(df, self.atr_period).to_numpy(dtype=float)

        # NY calendar date per bar — session boundary key. Daily frames
        # naturally map 1 bar → 1 date; intraday frames map ~26 bars
        # (15m) or ~390 bars (1m) → 1 date.
        session_dates = self._session_dates(df)
        # RTH mask: True when the bar's start falls inside 09:30–16:00
        # NY local time. Pre/post bars still flow into ChoCH state
        # (they print swing highs/lows like any other bar) but the
        # 3-bar FVG and the midpoint retest only fire on RTH bars —
        # an FVG forming on thin pre-market liquidity isn't a real
        # tradable level.
        rth_mask = self._rth_mask(df)

        signals: list[PatternSignal] = []

        # Per-session state, reset on date roll-over.
        current_session: object = None
        sess_swing_highs: list[tuple[int, float]] = []
        sess_swing_lows: list[tuple[int, float]] = []
        choch_bar: int | None = None
        choch_high: float | None = None
        choch_break_level_start_idx: int | None = None
        sess_signal_count = 0
        # FVGs detected but not yet retested → pending entry queue.
        # Each entry is a dict carrying the structural levels and
        # the bar where the FVG completed; a subsequent bar's low
        # touching ``fvg_mid`` (without breaking ``fvg_low``) fires
        # the signal at THAT retest bar.
        pending_fvgs: list[dict] = []

        for i in range(n):
            sess = session_dates[i]
            if sess != current_session:
                current_session = sess
                sess_swing_highs = []
                sess_swing_lows = []
                choch_bar = None
                choch_high = None
                choch_break_level_start_idx = None
                sess_signal_count = 0
                pending_fvgs = []

            # Confirm pivots ``swing_right`` bars after the fact.
            confirm_idx = i - self.swing_right
            if confirm_idx >= 0 and session_dates[confirm_idx] == current_session:
                sh = swing_high_arr[confirm_idx]
                sl = swing_low_arr[confirm_idx]
                if not np.isnan(sh):
                    sess_swing_highs.append((confirm_idx, float(sh)))
                if not np.isnan(sl):
                    sess_swing_lows.append((confirm_idx, float(sl)))

            # CHoCH detection (bullish): strict structural sequence.
            #   - At least 2 confirmed swing lows in this session.
            #   - The most recent low is strictly lower than the prior
            #     low (lower-lows = down structure confirmed).
            #   - There must be a swing high BETWEEN the two lows
            #     (the bounce H1 — this is the level the rally has to
            #     break to flip the structure).
            #   - A bar's close finally exceeds H1 after the most
            #     recent low has printed.
            if choch_bar is None and len(sess_swing_lows) >= 2:
                last_low_idx, last_low_val = sess_swing_lows[-1]
                _, prior_low_val = sess_swing_lows[-2]
                if last_low_val < prior_low_val:
                    prior_low_idx = sess_swing_lows[-2][0]
                    between = [
                        (idx, val)
                        for idx, val in sess_swing_highs
                        if prior_low_idx < idx < last_low_idx
                    ]
                    if between and i > last_low_idx:
                        ref_high_idx, ref_high_val = between[-1]
                        # Magnitude gate — the down-leg ``H1 → L2``
                        # has to span at least N ATRs or we treat
                        # the structural break as a weak-bounce
                        # false-positive (see user feedback on the
                        # "ChoCH on tiny green candle" screenshot).
                        atr_at_L2 = atr_arr[last_low_idx]
                        swing_magnitude = ref_high_val - last_low_val
                        passes_magnitude = (
                            self.min_choch_swing_atr is None
                            or (
                                not np.isnan(atr_at_L2)
                                and atr_at_L2 > 0
                                and swing_magnitude
                                >= self.min_choch_swing_atr * atr_at_L2
                            )
                        )
                        if passes_magnitude and closes[i] > ref_high_val:
                            choch_bar = i
                            choch_high = ref_high_val
                            choch_break_level_start_idx = ref_high_idx

            # ---- A. FVG queueing -----------------------------------
            # On a fresh CHoCH-anchored 3-bar bullish gap, push the
            # FVG onto the pending list rather than emitting a
            # signal. Entry is deferred to the midpoint retest below.
            #
            # RTH gate: all three FVG bars must fall inside the
            # regular session. ChoCH itself can come from pre/post
            # liquidity (early structural break) but the gap zone is
            # only valid where there's enough flow for a limit fill
            # to actually execute at the midpoint.
            # All three FVG bars (i-2, i-1, i) must be **strictly
            # after** the CHoCH bar. ``i > choch_bar`` alone allowed
            # the CHoCH bar itself to anchor the FVG (e.g., when
            # ``i = choch_bar + 2`` the first FVG bar i-2 is the
            # CHoCH bar). The user flagged this on SWKS 2026-04-17:
            # the same big green CHoCH-break candle was also being
            # used as the FVG-producing structure, so the gap was
            # measured against the bar that just gapped up to break
            # — circular and structurally meaningless. Requiring
            # ``i - 2 > choch_bar`` ⇔ ``i >= choch_bar + 3`` forces
            # the FVG to be a separate impulse leg printed AFTER the
            # break confirmed.
            if (
                choch_bar is not None
                and choch_high is not None
                and i >= 2
                and i - 2 > choch_bar
                and i - choch_bar <= self.max_bars_after_choch
                and sess_signal_count < self.max_signals_per_session
                and session_dates[i - 2] == current_session
                and rth_mask[i - 2]
                and rth_mask[i - 1]
                and rth_mask[i]
            ):
                all_bullish = (
                    closes[i - 2] > opens[i - 2]
                    and closes[i - 1] > opens[i - 1]
                    and closes[i] > opens[i]
                )
                fvg_low = highs[i - 2]
                fvg_high = lows[i]
                if all_bullish and fvg_high > fvg_low:
                    gap_size = fvg_high - fvg_low
                    close_i = closes[i]
                    if close_i > 0 and gap_size / close_i >= self.min_gap_pct:
                        fvg_mid = (fvg_low + fvg_high) / 2.0
                        fvg_producing_low = lows[i - 1]
                        # Stop must be strictly below FVG-completion close,
                        # otherwise the setup is already invalid.
                        if fvg_producing_low < close_i:
                            pending_fvgs.append(
                                {
                                    "fvg_bar_idx": i,
                                    "fvg_low": float(fvg_low),
                                    "fvg_high": float(fvg_high),
                                    "fvg_mid": float(fvg_mid),
                                    "fvg_producing_low": float(
                                        fvg_producing_low
                                    ),
                                    "gap_size": float(gap_size),
                                    "choch_high": float(choch_high),
                                    "choch_bar": int(choch_bar),
                                    "choch_break_level_start_idx": (
                                        int(choch_break_level_start_idx)
                                        if choch_break_level_start_idx is not None
                                        else None
                                    ),
                                }
                            )

            # ---- B. Midpoint retest → signal -----------------------
            # Walk pending FVGs and check whether the current bar is
            # the retest entry trigger. Three outcomes per pending
            # FVG at this bar:
            #   1. Bar's LOW pierces fvg_low → gap fully filled, the
            #      structural setup is broken; drop it.
            #   2. Bar's LOW touches fvg_mid AND close stays above
            #      fvg_mid → midpoint hold, fire the signal at this
            #      bar (entry executes at next bar's open per
            #      strategy convention).
            #   3. Too many bars elapsed since FVG formed → drop.
            # Rule (3) keeps the pending queue from carrying stale
            # FVGs that the user has clearly missed.
            still_pending: list[dict] = []
            for fvg in pending_fvgs:
                bars_since = i - fvg["fvg_bar_idx"]
                if bars_since <= 0:
                    still_pending.append(fvg)
                    continue
                if bars_since > self.max_retest_bars:
                    continue  # drop — stale
                if lows[i] < fvg["fvg_low"]:
                    continue  # drop — gap filled
                if (
                    sess_signal_count < self.max_signals_per_session
                    and rth_mask[i]
                    and lows[i] <= fvg["fvg_mid"]
                    and closes[i] > fvg["fvg_mid"]
                ):
                    # Retest entry — emit signal at THIS bar.
                    choch_ts_iso = pd.Timestamp(
                        df.index[fvg["choch_bar"]]
                    ).isoformat()
                    fvg_start_ts_iso = pd.Timestamp(
                        df.index[fvg["fvg_bar_idx"] - 2]
                    ).isoformat()
                    fvg_completion_ts_iso = pd.Timestamp(
                        df.index[fvg["fvg_bar_idx"]]
                    ).isoformat()
                    choch_break_start_ts_iso = (
                        pd.Timestamp(
                            df.index[fvg["choch_break_level_start_idx"]]
                        ).isoformat()
                        if fvg["choch_break_level_start_idx"] is not None
                        else None
                    )
                    close_i = closes[i]
                    # Entry is the FVG midpoint — a resting limit buy
                    # placed at ``fvg_mid`` after the gap forms gets
                    # filled when the retest bar's low touches that
                    # level. Using ``closes[i]`` would over-state the
                    # fill (the bar typically closes well above mid)
                    # and break the 1:R math the framework prescribes.
                    signal = PatternSignal(
                        date=df.index[i].date(),
                        timestamp=pd.Timestamp(df.index[i]).to_pydatetime(),
                        pattern_name=self.name,
                        entry_price=float(fvg["fvg_mid"]),
                        stop_loss=float(fvg["fvg_producing_low"]),
                        confidence=min(
                            1.0 + fvg["gap_size"] / close_i * 100.0, 3.0
                        ),
                        metadata={
                            "fvg_low": round(fvg["fvg_low"], 4),
                            "fvg_high": round(fvg["fvg_high"], 4),
                            "fvg_mid": round(fvg["fvg_mid"], 4),
                            "fvg_producing_low": round(
                                fvg["fvg_producing_low"], 4
                            ),
                            "choch_high": round(fvg["choch_high"], 4),
                            "choch_bar_offset": int(
                                fvg["fvg_bar_idx"] - fvg["choch_bar"]
                            ),
                            "choch_timestamp": choch_ts_iso,
                            "choch_break_level_start_ts": (
                                choch_break_start_ts_iso
                            ),
                            "fvg_start_timestamp": fvg_start_ts_iso,
                            "fvg_completion_timestamp": fvg_completion_ts_iso,
                            "retest_bar_offset": int(bars_since),
                            "gap_size_pct": round(
                                float(fvg["gap_size"] / close_i), 6
                            ),
                        },
                    )
                    signals.append(signal)
                    sess_signal_count += 1
                    # The ChoCH that anchored this FVG has now done
                    # its job — clear it so the *next* FVG queue from
                    # the same structural break can't fire as a
                    # second back-to-back signal. Emitting another
                    # FVG from the same ChoCH while the first leg is
                    # still extending re-uses already-played-out
                    # structure (the user flagged this as a META
                    # double-entry on 2026-03-31). Clear the
                    # accumulated lower-lows too — without this, the
                    # same L1/L2 pair would re-fire ChoCH at every
                    # subsequent bar that closes above the same H1.
                    # A genuinely new signal now requires fresh
                    # structure: two new swing lows (lower-lows
                    # against each other), a new intermediate swing
                    # high, and a fresh break.
                    choch_bar = None
                    choch_high = None
                    choch_break_level_start_idx = None
                    pending_fvgs = []
                    still_pending = []
                    sess_swing_lows = []
                    sess_swing_highs = []
                    break
                # No trigger this bar, keep pending for later bars.
                still_pending.append(fvg)
            pending_fvgs = still_pending

        return signals

    # ---- helpers --------------------------------------------------------

    @staticmethod
    def _compute_atr(df: pd.DataFrame, period: int) -> pd.Series:
        """EMA-smoothed ATR — same convention as wedgepop / wickplay
        detectors. Used to gate CHoCH magnitude (a tiny green-candle
        bounce shouldn't qualify as a structural break)."""
        prev_close = df["Close"].shift(1)
        tr = pd.concat(
            [
                df["High"] - df["Low"],
                (df["High"] - prev_close).abs(),
                (df["Low"] - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        return tr.ewm(span=period, adjust=False).mean()

    @staticmethod
    def _session_dates(df: pd.DataFrame) -> np.ndarray:
        """Per-bar NY calendar date. Reused across the detector loop
        as the session-boundary key — bars on the same date belong to
        the same session even when the index is tz-naive (daily) or
        tz-aware (intraday)."""
        idx = df.index
        if hasattr(idx, "tz") and idx.tz is not None:
            return np.array([ts.tz_convert("America/New_York").date() for ts in idx])
        return np.array([ts.date() for ts in idx])

    @staticmethod
    def _rth_mask(df: pd.DataFrame) -> np.ndarray:
        """Per-bar RTH boolean. ``True`` when bar START falls in
        09:30 ≤ t < 16:00 NY local time. Daily frames (where every
        bar is at midnight) collapse to all-True so the detector's
        FVG step still fires — the RTH gate matters only when the
        upstream frame mixes pre/post bars with regular session
        bars (yfinance ``prepost=True``)."""
        idx = df.index
        if hasattr(idx, "tz") and idx.tz is not None:
            local = idx.tz_convert("America/New_York")
        else:
            local = idx
        times = np.array([ts.time() for ts in local])
        # All bars at midnight → daily frame. Skip the time-of-day
        # gate so daily callers (tests, future daily detectors)
        # aren't accidentally filtered out.
        if all(t == _dt_time(0, 0) for t in times):
            return np.ones(len(df), dtype=bool)
        return np.array(
            [(_RTH_OPEN <= t < _RTH_CLOSE) for t in times], dtype=bool
        )
