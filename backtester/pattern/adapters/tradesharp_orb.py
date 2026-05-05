"""Trade Sharp ORB pattern detector — opening-range pullback-rejection-breakout.

Adapted from a Trade Sharp methodology video describing a 15-minute
opening-range variant that explicitly **rejects** the naive
"first close above the box" entry as a trap. Trader claims an 88%+
win rate over a recent quarter using the structure below.

Pattern (long-only, NYSE 09:30 ET):

    1. **Daily bias** — previous daily candle must be bullish
       (close > open). The video stresses checking higher-timeframe
       direction; without that bias the breakout is just noise.

    2. **Opening range** — first 15 minutes' high–low (3 × 5m bars).

    3. **Initial breakout (the trap)** — wait for the first 5m bar
       that *closes* above ``box_high``. This is **NOT** the entry.
       The video repeatedly calls this the "slapped in the face"
       moment when retail buys the breakout.

    4. **Liquidity grab / pullback** — price pulls back, ideally
       *back inside* the open range (low ≤ box_high). This grabs
       stops below the breakout and creates the rubber-band setup.
       If lows breach ``box_low`` outright, the structure has
       failed — abandon.

    5. **Rejection candle** — first bullish 5m close (close > open)
       after the pullback low. The video shows a clean bullish
       candle with a lower wick into the OR — buyers absorbing the
       grab.

    6. **Entry trigger** — the *next* bar's high breaks above the
       rejection candle's high. Fills at ``rejection_high``
       (buy-stop semantics), within ``max_entry_after_rejection_bars``
       of the rejection candle.

    7. **Stop** = rejection candle's low. The video: "even nicer if
       you can put it back inside of the open range" — that's
       naturally true since the rejection often dips into the OR.
       Stop is the level the rejection defended.

    8. **Take profit** = entry + R × (entry − stop). Video uses 1:1
       to 1:2 examples; the trader said 1:2 R:R is the practical
       target.

Time gate: entries must fire before ``latest_entry_local`` (default
11:30 ET — first two hours of NYSE). Past that the morning-auction
volatility regime breaks down.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, time

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TradeSharpORBSignal:
    """Single bull entry signal for the Trade Sharp ORB pattern.

    All timestamps are tz-aware in the market's local tz.

    Field semantics:
      - ``box_low / box_high`` — opening 15-minute range.
      - ``initial_breakout_*`` — first 5m close > box_high (the trap).
      - ``pullback_low_*`` — the deepest low reached during the
        pullback consolidation (= the stop level).
      - ``consol_start_ts`` — timestamp the pullback first dipped
        back into the open range. Marks the start of the
        consolidation phase.
      - ``consol_high`` — highest high during the consolidation
        phase, BEFORE the entry bar broke through it. The entry
        order sits at this level (buy-stop fill).
      - ``entry_*`` — the bar whose high crossed ``consol_high``,
        triggering the buy-stop fill at ``consol_high``.
    """

    session_date: date
    direction: str  # always "long" in this build
    box_low: float
    box_high: float
    box_open_ts: pd.Timestamp
    box_close_ts: pd.Timestamp
    initial_breakout_ts: pd.Timestamp
    initial_breakout_price: float
    pullback_low_ts: pd.Timestamp
    pullback_low: float
    consol_start_ts: pd.Timestamp
    consol_high: float
    entry_ts: pd.Timestamp
    entry_price: float
    stop_loss: float
    take_profit: float


class TradeSharpORBDetector:
    """Trade Sharp opening-range pullback-rejection-breakout detector.

    Parameters
    ----------
    box_minutes:
        Opening range duration. Video standard = 15 minutes.
    session_open_local:
        Market local-tz wall-clock open. NY = 09:30 ET, KR = 09:00 KST.
    latest_entry_local:
        Past this local time, no new entries fire. Video implies the
        morning auction regime — defaults to 11:30 (first two hours).
    require_daily_bullish:
        When True, only fire on sessions where the *previous* daily
        candle closed above its open. Video's "previous daily candle
        bullish" bias gate.
    require_pullback_into_box:
        When True, the post-breakout pullback must dip **back into
        the open range** (low ≤ box_high). When False, any retracement
        counts. Default True per the video's "liquidity grab back
        into the OR" emphasis.
    min_pullback_depth_pct:
        Optional extra-strict pullback gate — the pullback low must
        reach at least this fraction of the box height below
        ``box_high`` (0.0 = touch is enough, 1.0 = pullback must
        reach ``box_low``). Default 0.0 = touch-the-line.
    max_consolidation_bars:
        Hard cap on 5m bars between the initial breakout and the
        entry trigger. After this window, the setup is stale.
    min_consolidation_bars:
        Minimum bars that must elapse during the consolidation phase
        before any entry can fire. This is the **fix** for the
        "took the first pullback" bug — the trader's screenshot
        shows price needs time to develop a real base before the
        breakout-of-consolidation entry. Default 3 (= 15 min on 5m).
    target_r_multiple:
        TP = entry + R × (entry − stop). Video: 1:1 to 1:2 examples.
    """

    name = "tradesharp_orb"

    def __init__(
        self,
        box_minutes: int = 15,
        session_open_local: time = time(9, 30),
        latest_entry_local: time = time(11, 30),
        require_daily_bullish: bool = True,
        require_pullback_into_box: bool = True,
        min_pullback_depth_pct: float = 0.0,
        max_consolidation_bars: int = 25,
        min_consolidation_bars: int = 3,
        target_r_multiple: float = 1.5,
    ) -> None:
        if box_minutes <= 0:
            raise ValueError("box_minutes must be > 0")
        if not (0.0 <= min_pullback_depth_pct <= 1.0):
            raise ValueError("min_pullback_depth_pct ∈ [0, 1]")
        if min_consolidation_bars < 1:
            raise ValueError("min_consolidation_bars must be ≥ 1")
        self.box_minutes = box_minutes
        self.session_open_local = session_open_local
        self.latest_entry_local = latest_entry_local
        self.require_daily_bullish = require_daily_bullish
        self.require_pullback_into_box = require_pullback_into_box
        self.min_pullback_depth_pct = min_pullback_depth_pct
        self.max_consolidation_bars = max_consolidation_bars
        self.min_consolidation_bars = min_consolidation_bars
        self.target_r_multiple = target_r_multiple

    # ---- public API ------------------------------------------------

    def detect(
        self,
        df_5m: pd.DataFrame,
        df_daily: pd.DataFrame,
    ) -> list[TradeSharpORBSignal]:
        if df_5m.empty or df_daily.empty:
            return []

        # Daily bias filter — previous session's daily candle must be
        # bullish (close > open). ``shift(1)`` makes this strictly
        # causal (today qualifies based on yesterday).
        if self.require_daily_bullish:
            prev_close = df_daily["Close"].shift(1)
            prev_open = df_daily["Open"].shift(1)
            bullish_yesterday = prev_close > prev_open
            qualifying_dates = {
                ts.date()
                for ts, ok in bullish_yesterday.items()
                if bool(ok)
            }
        else:
            qualifying_dates = {ts.date() for ts in df_daily.index}

        signals: list[TradeSharpORBSignal] = []
        local_dates = self._local_dates(df_5m.index)
        for sess_date, sess_5m in self._iter_sessions(df_5m, local_dates):
            if sess_date not in qualifying_dates:
                continue
            sig = self._scan_session(sess_date, sess_5m)
            if sig is not None:
                signals.append(sig)
        return signals

    # ---- internals -------------------------------------------------

    @staticmethod
    def _local_dates(idx: pd.DatetimeIndex) -> np.ndarray:
        return np.array([ts.date() for ts in idx])

    @staticmethod
    def _iter_sessions(df: pd.DataFrame, local_dates: np.ndarray):
        if len(df) == 0:
            return
        change = np.r_[True, local_dates[1:] != local_dates[:-1]]
        boundaries = np.flatnonzero(change)
        for i, start in enumerate(boundaries):
            end = boundaries[i + 1] if i + 1 < len(boundaries) else len(df)
            yield local_dates[start], df.iloc[start:end]

    def _scan_session(
        self, sess_date: date, sess: pd.DataFrame
    ) -> TradeSharpORBSignal | None:
        idx = sess.index
        local_time = idx if idx.tz is None else idx.tz_convert(idx.tz)
        bar_minutes = np.array(
            [t.hour * 60 + t.minute for t in local_time.time]
        )
        open_minutes = (
            self.session_open_local.hour * 60
            + self.session_open_local.minute
        )
        cutoff_minutes = (
            self.latest_entry_local.hour * 60
            + self.latest_entry_local.minute
        )
        # Box = first ``box_minutes`` of the session. With 5m bars
        # and box_minutes=15, that's 3 bars.
        box_end = open_minutes + self.box_minutes
        box_mask = (bar_minutes >= open_minutes) & (bar_minutes < box_end)
        box_bars = sess[box_mask]
        if len(box_bars) == 0:
            return None
        box_high = float(box_bars["High"].max())
        box_low = float(box_bars["Low"].min())
        box_height = box_high - box_low
        if box_height <= 0:
            return None
        box_open_ts = box_bars.index[0]
        box_close_ts = box_bars.index[-1]

        post_mask = bar_minutes >= box_end
        post = sess[post_mask]
        if len(post) == 0:
            return None
        post_minutes = bar_minutes[post_mask]
        opens = post["Open"].to_numpy(dtype=float)
        highs = post["High"].to_numpy(dtype=float)
        lows = post["Low"].to_numpy(dtype=float)
        closes = post["Close"].to_numpy(dtype=float)
        timestamps = post.index

        # 3-phase state machine. The Trade Sharp screenshot shows
        # entry happens at the breakout of the **post-pullback
        # consolidation high** (not the first rejection candle's
        # high) — so we collapse the previous wait_rejection +
        # wait_entry phases into a single ``wait_consolidation_break``
        # phase that tracks the running consolidation max:
        #
        #   wait_breakout            — first 5m close > box_high
        #                               (the trap)
        #   wait_pullback            — pullback into [box_low, box_high]
        #   wait_consolidation_break — track pullback's deepest low
        #                               (= stop) and the consolidation's
        #                               highest high (= entry trigger).
        #                               Entry only fires after at least
        #                               ``min_consolidation_bars`` have
        #                               elapsed AND a bar's high
        #                               crosses the running ``consol_high``.
        phase = "wait_breakout"
        breakout_idx: int | None = None
        breakout_price: float | None = None
        peak: float = -1.0
        pullback_idx: int | None = None
        pullback_low: float = float("inf")
        consol_high: float = -1.0

        depth_threshold = box_high - self.min_pullback_depth_pct * box_height

        for i in range(len(post)):
            # Hard invalidation across all phases.
            if lows[i] < box_low:
                return None
            # Stale-setup timeout from the breakout bar onwards.
            if (
                breakout_idx is not None
                and phase != "wait_breakout"
                and i - breakout_idx > self.max_consolidation_bars
            ):
                return None
            # Time-of-day cutoff applies once we're scanning for
            # entries — earlier phases can keep watching even past
            # the cutoff (no entry until they advance, which won't
            # happen anyway because the cutoff blocks below).
            if (
                phase == "wait_consolidation_break"
                and post_minutes[i] >= cutoff_minutes
            ):
                return None

            if phase == "wait_breakout":
                if closes[i] > box_high:
                    breakout_idx = i
                    breakout_price = float(closes[i])
                    peak = highs[i]
                    phase = "wait_pullback"
                continue

            if phase == "wait_pullback":
                if highs[i] > peak:
                    peak = highs[i]
                pullback_target = (
                    depth_threshold
                    if self.require_pullback_into_box
                    else peak
                )
                if lows[i] <= pullback_target:
                    pullback_idx = i
                    pullback_low = float(lows[i])
                    consol_high = float(highs[i])
                    phase = "wait_consolidation_break"
                continue

            # phase == "wait_consolidation_break"
            # Update the deepest low — this is the actual stop level.
            if lows[i] < pullback_low:
                pullback_low = float(lows[i])
            bars_since_pullback_start = i - pullback_idx
            # Entry can only fire after the consolidation has
            # developed for ``min_consolidation_bars``. Until then
            # we just track structure (consol_high updated below).
            if bars_since_pullback_start >= self.min_consolidation_bars:
                # Entry trigger: bar's high pierces the running
                # ``consol_high``. ``consol_high`` is intentionally
                # measured **before** including this bar, so a
                # fresh-high bar IS the breakout. Fill at
                # ``consol_high`` — the buy-stop level the order
                # would have been parked at.
                if highs[i] > consol_high:
                    entry_price = float(consol_high)
                    stop = float(pullback_low)
                    if entry_price - stop <= 0:
                        return None
                    tp = entry_price + self.target_r_multiple * (
                        entry_price - stop
                    )
                    return TradeSharpORBSignal(
                        session_date=sess_date,
                        direction="long",
                        box_low=box_low,
                        box_high=box_high,
                        box_open_ts=box_open_ts,
                        box_close_ts=box_close_ts,
                        initial_breakout_ts=timestamps[breakout_idx],
                        initial_breakout_price=breakout_price,
                        pullback_low_ts=timestamps[pullback_idx],
                        pullback_low=pullback_low,
                        consol_start_ts=timestamps[pullback_idx],
                        consol_high=consol_high,
                        entry_ts=timestamps[i],
                        entry_price=entry_price,
                        stop_loss=stop,
                        take_profit=tp,
                    )
            # Update consol_high AFTER the entry check so the
            # current bar's high doesn't trivially become the
            # threshold it just broke.
            if highs[i] > consol_high:
                consol_high = float(highs[i])
        return None
