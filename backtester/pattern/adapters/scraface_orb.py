"""Scraface Trade Pattern detector — opening-range box breakout-retest.

Pattern source: ``ScrafaceTradePattern.docx``. Three-timeframe rule:

    1. **Daily** — draw the 50 SMA. Above = buyer dominance (long
       bias), below = seller dominance (short bias).
    2. **5 minute** — at session open (e.g., 09:30 ET) the FIRST
       5-minute candle's OHLC range becomes "the box". Its high =
       intraday resistance, low = intraday support. No trading
       inside the box.
    3. **1 minute** — wait for price to break the box and *retest*
       it from the breakout side, then enter at the retest. Stop on
       the opposite side of the box, target 1:2 R/R.

This detector implements the **bull-only** variant per the page's
explicit ask: only LONG signals are emitted, and a regime gate
requires today + at least N-of-M of the most recent daily closes to
sit *below* the 50 SMA. That turns the pattern into a
reversal-from-suppression bounce rather than a trend continuation —
the user specifically wants to fish for longs after a sustained
sub-SMA stretch.

The detector is multi-timeframe — it consumes both the 1-minute and
daily frames — so it doesn't subclass :class:`PatternDetector` (which
is single-frame). Callers pass both frames explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, time

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ScrafaceORBSignal:
    """A single bull retest entry signal.

    The pattern uses a *two-box* structure:

      - **First box** (``box_low`` / ``box_high``) — the opening N-minute
        candle's OHLC. ``box_high`` is the resistance the breakout
        clears and the level the retest defends.
      - **Second box** (``second_box_low`` / ``second_box_high``) — the
        post-breakout consolidation envelope: highest high and lowest
        low across the rally + pullback that follows the first
        breakout. Used for visualization only; trade triggers are
        keyed off ``box_high``.

    Entry triggers in three phases:
      1. First close > ``box_high`` (breakout).
      2. Some later bar **closes** below ``box_high`` (the closing
         retest dip — intrabar wick touches alone don't count).
      3. The next bar that closes back above ``box_high`` IS the
         entry. Stop = that bar's own ``Low``; TP = entry + ``R`` ×
         (entry − stop). All long-only.
    """

    session_date: date
    direction: str
    box_low: float
    box_high: float
    box_open_ts: pd.Timestamp
    box_close_ts: pd.Timestamp
    breakout_ts: pd.Timestamp
    breakout_price: float
    second_box_low: float
    second_box_high: float
    retest_ts: pd.Timestamp
    entry_ts: pd.Timestamp
    entry_price: float
    stop_loss: float
    take_profit: float


class ScrafaceORBDetector:
    """Bull-only opening-range breakout-retest detector.

    Parameters
    ----------
    sma_period:
        Daily-chart SMA used for regime gating. ``50`` per the doc.
    days_below_lookback / min_days_below:
        Of the last ``days_below_lookback`` *daily* closes (today
        inclusive), at least ``min_days_below`` must sit ABOVE the
        SMA. Defaults to 10 / 7 — long entries only fire in a
        sustained bull regime. Field names kept for backwards
        compat; semantics now reads as "min days *above* SMA".
    box_minutes:
        First N minutes of the regular session that define the box.
        ``5`` per the doc; configurable so users can experiment with
        15-minute opening ranges, etc.
    session_open_local:
        Wall-clock open in the data's *local* timezone. Defaults to
        09:30 ET (NY cash equity session). The detector never sees
        the calendar object directly — callers pre-localize the
        1-minute frame to NY tz before passing it in.
    breakout_buffer_atr:
        Optional buffer above ``box_high`` that the breakout close
        must clear (in units of the box's intrabar range). ``0`` =
        any close > box_high counts. ``0.1`` ≈ 10% of the box height.
    retest_tolerance_atr:
        Reserved for future intrabar-touch variants; currently unused
        because the retest is detected from the bar **close** (a true
        closing dip below ``box_high``), not the wick.
    max_retest_bars:
        Cap on how many 1-minute bars after the *initial* breakout
        to keep scanning for the closing retest + continuation. After
        this window the setup is considered failed.
    latest_entry_local:
        Hard cutoff time (local market wall clock) past which entries
        are not allowed. The doc-derived rule is "09:30–11:00 ET only"
        — the morning auction-driven volatility window. After this the
        regime breaks down and the setup is skipped.
    target_r_multiple:
        TP = entry + ``target_r_multiple`` × (entry − stop). Doc says
        1:2. Stop is the **entry candle's own low** — risk per share
        is measured off the bar that fires the trade so sizing reacts
        to actual entry-time volatility.
    """

    name = "scraface_orb"

    def __init__(
        self,
        sma_period: int = 50,
        days_below_lookback: int = 10,
        min_days_below: int = 7,
        box_minutes: int = 5,
        session_open_local: time = time(9, 30),
        latest_entry_local: time = time(11, 0),
        breakout_buffer_atr: float = 0.0,
        retest_tolerance_atr: float = 0.1,
        max_retest_bars: int = 60,
        target_r_multiple: float = 2.0,
    ) -> None:
        self.sma_period = sma_period
        self.days_below_lookback = days_below_lookback
        self.min_days_below = min_days_below
        self.box_minutes = box_minutes
        self.session_open_local = session_open_local
        self.latest_entry_local = latest_entry_local
        self.breakout_buffer_atr = breakout_buffer_atr
        self.retest_tolerance_atr = retest_tolerance_atr
        self.max_retest_bars = max_retest_bars
        self.target_r_multiple = target_r_multiple

    # ---- public API ---------------------------------------------------

    def detect(
        self,
        df_1m: pd.DataFrame,
        df_daily: pd.DataFrame,
    ) -> list[ScrafaceORBSignal]:
        """Scan ``df_1m`` for bull retest entries on qualifying days.

        ``df_1m`` must have a tz-aware DatetimeIndex (typically
        ``America/New_York`` from the EODHD adapter); ``df_daily`` is
        tz-naive midnight-indexed. The returned list is ordered by
        ``entry_ts``.
        """
        if df_1m.empty or df_daily.empty:
            return []

        sma = df_daily["Close"].rolling(self.sma_period).mean()
        # Long bias gate: today AND at least N-of-M of the most recent
        # daily closes must sit ABOVE the SMA. Doc rule — "50 SMA 위에는
        # Buyer가 더 많다". Below-SMA days are bearish regime; longs
        # don't fire there regardless of intraday structure.
        above_sma = (df_daily["Close"] > sma).astype(int)
        rolling_above = above_sma.rolling(
            self.days_below_lookback, min_periods=self.days_below_lookback
        ).sum()

        qualifying_dates: set[date] = set()
        for ts, count in rolling_above.items():
            if pd.isna(count):
                continue
            if count >= self.min_days_below and above_sma.loc[ts] == 1:
                qualifying_dates.add(ts.date())

        signals: list[ScrafaceORBSignal] = []
        # Group 1m bars by NY local date once — the index can be
        # tz-aware (intraday from EODHD) or tz-naive; both paths
        # collapse to a date key here.
        local_dates = self._local_dates(df_1m.index)
        for sess_date, sess_slice in self._iter_sessions(df_1m, local_dates):
            if sess_date not in qualifying_dates:
                continue
            sig = self._scan_session(sess_date, sess_slice)
            if sig is not None:
                signals.append(sig)
        return signals

    # ---- internals ---------------------------------------------------

    @staticmethod
    def _local_dates(idx: pd.DatetimeIndex) -> np.ndarray:
        if getattr(idx, "tz", None) is not None:
            return np.array([ts.date() for ts in idx])
        return np.array([ts.date() for ts in idx])

    @staticmethod
    def _iter_sessions(
        df_1m: pd.DataFrame, local_dates: np.ndarray
    ):
        """Yield ``(date, df_session)`` pairs, ordered by date."""
        if len(df_1m) == 0:
            return
        # Identify contiguous date runs without sorting (df_1m is
        # already chronological from the data adapter).
        change = np.r_[True, local_dates[1:] != local_dates[:-1]]
        boundaries = np.flatnonzero(change)
        for i, start in enumerate(boundaries):
            end = boundaries[i + 1] if i + 1 < len(boundaries) else len(df_1m)
            yield local_dates[start], df_1m.iloc[start:end]

    def _scan_session(
        self, sess_date: date, sess: pd.DataFrame
    ) -> ScrafaceORBSignal | None:
        """Locate box → breakout → retest within a single session."""
        # The box: first ``box_minutes`` 1m bars whose start falls
        # inside ``[session_open, session_open + box_minutes)``. We
        # don't trust the slice's first row blindly because pre-market
        # bars might leak in if the upstream filter is off — match by
        # wall-clock time explicitly.
        local_idx = sess.index
        if hasattr(local_idx, "tz") and local_idx.tz is not None:
            # Index already in NY local; tz_convert no-ops if already there.
            local_times = local_idx.tz_convert(local_idx.tz)
        else:
            local_times = local_idx

        open_ts = self.session_open_local
        # End of the box window: open + box_minutes (exclusive).
        # Using minute math keeps the boundary check clean across
        # hour rollovers (e.g., 09:30 + 60 → 10:30).
        open_minutes = open_ts.hour * 60 + open_ts.minute
        end_minutes = open_minutes + self.box_minutes
        bar_minutes = np.array(
            [t.hour * 60 + t.minute for t in local_times.time]
        )
        box_mask = (bar_minutes >= open_minutes) & (bar_minutes < end_minutes)
        box_bars = sess[box_mask]
        if len(box_bars) == 0:
            return None  # session has no bars in the open window

        box_high = float(box_bars["High"].max())
        box_low = float(box_bars["Low"].min())
        box_height = box_high - box_low
        if box_height <= 0:
            return None  # degenerate box (single-tick bar) — skip
        box_open_ts = box_bars.index[0]
        box_close_ts = box_bars.index[-1]

        # Post-box bars only — entries inside the box are explicitly
        # forbidden by the framework.
        post_box = sess[bar_minutes >= end_minutes]
        if len(post_box) == 0:
            return None

        breakout_thresh = box_high + self.breakout_buffer_atr * box_height
        closes = post_box["Close"].to_numpy(dtype=float)
        highs = post_box["High"].to_numpy(dtype=float)
        lows = post_box["Low"].to_numpy(dtype=float)
        timestamps = post_box.index

        # Three-phase scan, all bound by ``max_retest_bars`` from the
        # initial breakout and the ``latest_entry_local`` time gate:
        #
        #   PHASE 1 — wait_breakout
        #     First close > ``breakout_thresh``. Track running peak +
        #     deepest low for the second-box visualization. Anything
        #     dipping to box_low along the way invalidates.
        #
        #   PHASE 2 — wait_retest_close
        #     After the initial breakout, price either keeps holding
        #     above box_high or fades. The retest is a *closing* dip:
        #     a bar that **closes** below box_high (low touches alone
        #     don't count — TSLA 2026-04-14 10:23 dipped to box_high
        #     intrabar but closed back above, so it was a wick, not a
        #     real retest). Track running peak + min low for the
        #     second-box visualization.
        #
        #   PHASE 3 — wait_continuation_close
        #     After the closing dip, the next bar that closes back
        #     above box_high IS the entry. Stop = entry candle's own
        #     low; TP = entry + R × (entry − stop). Time gate
        #     (entry's bar-start minute < cutoff_minutes) applies —
        #     past 11:00 ET the morning auction-driven volatility
        #     window has decayed and the setup is abandoned.
        cutoff_minutes = (
            self.latest_entry_local.hour * 60
            + self.latest_entry_local.minute
        )
        # ``bar_minutes`` was computed across the full session; slice
        # to the same post-box mask so indices align with ``closes`` /
        # ``highs`` / ``lows`` below.
        post_box_minutes = bar_minutes[bar_minutes >= end_minutes]

        phase = "wait_breakout"
        peak = -1.0
        deepest_low = float("inf")
        breakout_idx: int | None = None
        retest_idx: int | None = None
        scan_end = len(post_box)
        for i in range(scan_end):
            if lows[i] <= box_low:
                return None  # downside flip — abandon the long thesis
            if (
                breakout_idx is not None
                and i - breakout_idx > self.max_retest_bars
            ):
                return None  # retest+continuation didn't develop in time
            if post_box_minutes[i] >= cutoff_minutes:
                return None  # past the morning entry window

            if phase == "wait_breakout":
                if closes[i] > breakout_thresh:
                    breakout_idx = i
                    peak = highs[i]
                    deepest_low = lows[i]
                    phase = "wait_retest_close"
                continue

            # Track running peak + deepest low across phases 2-3 so
            # the second-box rectangle reflects the full consolidation.
            if highs[i] > peak:
                peak = highs[i]
            if lows[i] < deepest_low:
                deepest_low = lows[i]

            if phase == "wait_retest_close":
                if closes[i] < box_high:
                    retest_idx = i
                    phase = "wait_continuation_close"
                continue

            # phase == "wait_continuation_close"
            if closes[i] > box_high:
                entry_price = float(closes[i])
                stop = float(lows[i])  # entry candle's own low
                if entry_price - stop <= 0:
                    return None  # degenerate marubozu — skip
                tp = entry_price + self.target_r_multiple * (
                    entry_price - stop
                )
                return ScrafaceORBSignal(
                    session_date=sess_date,
                    direction="long",
                    box_low=box_low,
                    box_high=box_high,
                    box_open_ts=box_open_ts,
                    box_close_ts=box_close_ts,
                    breakout_ts=timestamps[breakout_idx],
                    breakout_price=float(closes[breakout_idx]),
                    second_box_low=float(deepest_low),
                    second_box_high=float(peak),
                    retest_ts=timestamps[retest_idx],
                    entry_ts=timestamps[i],
                    entry_price=entry_price,
                    stop_loss=stop,
                    take_profit=tp,
                )
        return None
