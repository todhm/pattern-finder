"""First Candle Rule pattern detector.

Adapted from "This Scalping Strategy Works Everyday" (Smart Trading
Blueprint). Three mechanical steps anchored to the first 5-minute
candle, with **two distinct entry paths**:

    1. **Range** — mark high/low of the 5m bar formed at 09:30–09:35
       ET. These define the day's first-act range.

    2. **Range break** — drop to 1m, wait for a candle that
       *closes* above ``box_high`` (or below ``box_low`` for short).
       Wick-only breaks don't count — close must be on the
       breakout side. This is the "candle that broke through and
       closed through it".

    3. **Direction-confirming FVG** — a 3-bar bullish FVG whose
       upper edge sits above the OR. There are two paths to entry:

       **Path A (immediate)** — the range-break candle IS the
       FVG-displacement bar (bar ``i`` in the 3-bar window). As
       soon as that candle closes, the FVG is confirmed and we
       enter on its close.

       **Path B (retest)** — the range-break candle alone made no
       FVG. Wait for price to retest the OR (low dips back into
       the range). After the retest, watch for a new bullish FVG
       to form off the OR. Enter on close of that FVG bar.

       Critical: in **both** paths, the **stop sits at the
       original range-break candle's low** — not the FVG-creator's
       low. The video is explicit about this for Path B.

    4. **Take profit** = entry + 2 × (entry − stop). Fixed 2:1.

Long-only. The original video shows both long and short examples
but the implementation here is bull-only — the rule set is
symmetric so a short variant is a straight inversion of the price
comparators.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, time

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FirstCandleRuleSignal:
    """One bull entry signal.

    Field semantics:
      - ``box_*`` — the first 5-minute candle's range.
      - ``range_break_*`` — the candle that first *closed* above
        ``box_high``. Kept as informational context (chart marker);
        no longer used for the stop level (per user pref).
      - ``path`` — ``"A"`` when the range-break candle was itself
        the FVG-displacement bar; ``"B"`` when the FVG only
        formed after a retest into the OR.
      - ``fvg_*`` — the 3-bar bullish FVG that triggered entry.
        Zone = ``[fvg_low, fvg_high]`` where
        ``fvg_low = bar[i-2].High`` and ``fvg_high = bar[i].Low``.
        ``fvg_pre_low`` = ``bar[i-2].Low`` — the absolute floor of
        the first FVG bar, which is now the **stop level**: if
        price violates this, the FVG's structural foundation is
        gone and the setup is invalidated.
      - ``retest_ts / retest_low`` — the bar whose low dipped back
        into the OR. ``None`` for Path A.
      - ``entry_ts / entry_price`` — close of the FVG-confirming
        bar (= ``fvg_form_ts``).
    """

    session_date: date
    direction: str  # always "long"
    box_low: float
    box_high: float
    box_open_ts: pd.Timestamp
    box_close_ts: pd.Timestamp
    range_break_ts: pd.Timestamp
    range_break_low: float
    fvg_low: float
    fvg_high: float
    fvg_form_ts: pd.Timestamp
    fvg_pre_ts: pd.Timestamp  # bar i-2 — start of the 3-bar FVG window
    fvg_pre_low: float  # bar i-2's low — stop reference
    path: str  # "A" or "B"
    retest_ts: pd.Timestamp | None
    retest_low: float | None
    entry_ts: pd.Timestamp
    entry_price: float
    stop_loss: float
    take_profit: float


class FirstCandleRuleDetector:
    """Long-only First Candle Rule detector.

    Parameters
    ----------
    box_minutes:
        Length of the opening-range candle. Video standard = 5
        (the literal "first 5-minute candle").
    session_open_local:
        Wall-clock open in the data's local tz. NY = 09:30 ET.
    latest_entry_local:
        Hard cutoff past which entries are not allowed. The video
        focuses on the morning auction window — default 11:00 ET.
    require_daily_bullish:
        Optional regime filter — only fire when yesterday's daily
        candle closed bullish (close > open). Off by default
        because the video doesn't specify a daily bias gate.
    require_bullish_displacement:
        When True, the FVG-creating bar must close bullish
        (close > open). The video calls this an "energetic" candle
        — bullish close is the directional commitment.
    min_fvg_gap_pct:
        Min FVG height (= ``low[i] − high[i-2]``) as a fraction
        of price. ``0`` disables. Useful for filtering micro-gaps
        that fill in 1–2 bars.
    max_total_bars:
        Cap on bars from the range-break candle to the entry FVG.
        After this window the setup is considered stale. Default
        45 (covers ~45 minutes on 1m / ~225 minutes on 5m).
    target_r_multiple:
        TP = entry + R × (entry − stop). Video says 2:1.
    stop_tick_buffer:
        Subtracted from the **first FVG bar's low** (``bar[i-2].Low``)
        to set the stop. Default $0.01 for US equities. The stop
        sits one tick below the absolute floor of the 3-bar FVG
        window — if breached, the FVG's foundation is gone.
    require_retest_for_path_b:
        When True (default), Path B entries fire only after price
        has dipped back to ``box_high`` or below following the
        range break. When False, the detector accepts any
        post-break FVG regardless of retest. Default True per the
        video.
    """

    name = "first_candle_rule"

    def __init__(
        self,
        box_minutes: int = 5,
        session_open_local: time = time(9, 30),
        latest_entry_local: time = time(11, 0),
        require_daily_bullish: bool = False,
        require_bullish_displacement: bool = True,
        min_fvg_gap_pct: float = 0.0003,
        max_total_bars: int = 45,
        target_r_multiple: float = 2.0,
        stop_tick_buffer: float = 0.01,
        require_retest_for_path_b: bool = True,
        # ---- Advanced filters (defaults from cross-ticker sweep) -
        # Skip sessions whose open vs prior daily close gapped up
        # more than this fraction. Default 1.5% — picked from a
        # 144-combo grid sweep on NVDA Jan-May 2026 + cross-ticker
        # validation on QQQ/MSFT/AAPL/TSLA/SPY. Filters classic
        # "gap-and-fade" traps where overnight buyers dump into the
        # OR-break long. None = disabled.
        max_gap_up_pct: float | None = 0.015,
        # Require today's daily close > N-day SMA. Disabled by
        # default — empirically didn't help in cross-ticker
        # validation (the NVDA sweep's bear period was too short
        # for SMA filtering to differentiate). 50 / 200 are common.
        require_above_daily_sma: int | None = None,
        # Skip sessions where today's volume is below this multiple
        # of the 20-day average daily volume. Default 0.85 — real
        # institutional moves come on volume; thin tape = noise.
        # The cross-ticker sweep showed this is the single most
        # robust filter (improves NVDA + TSLA without hurting others).
        min_daily_rvol: float | None = 0.85,
        # FVG-bar volume gate. Disabled by default — overfit to NVDA
        # in the sweep (improved NVDA but hurt QQQ/TSLA). Toggle on
        # for very volatile names where intrabar volume actually
        # tracks institutional flow.
        min_entry_bar_rvol: float | None = None,
    ) -> None:
        if box_minutes <= 0:
            raise ValueError("box_minutes must be > 0")
        if min_fvg_gap_pct < 0:
            raise ValueError("min_fvg_gap_pct must be ≥ 0")
        self.box_minutes = box_minutes
        self.session_open_local = session_open_local
        self.latest_entry_local = latest_entry_local
        self.require_daily_bullish = require_daily_bullish
        self.require_bullish_displacement = require_bullish_displacement
        self.min_fvg_gap_pct = min_fvg_gap_pct
        self.max_total_bars = max_total_bars
        self.target_r_multiple = target_r_multiple
        self.stop_tick_buffer = stop_tick_buffer
        self.require_retest_for_path_b = require_retest_for_path_b
        self.max_gap_up_pct = max_gap_up_pct
        self.require_above_daily_sma = require_above_daily_sma
        self.min_daily_rvol = min_daily_rvol
        self.min_entry_bar_rvol = min_entry_bar_rvol

    # ---- public API ------------------------------------------------

    def detect(
        self,
        df_1m: pd.DataFrame,
        df_daily: pd.DataFrame,
    ) -> list[FirstCandleRuleSignal]:
        if df_1m.empty:
            return []

        # Pre-compute daily-level filter masks once. Each is a
        # set[date] of qualifying sessions; ``None`` means "all
        # qualify" (filter disabled). Final qualifying set is the
        # intersection of all enabled filters.
        qualifying_sets: list[set[date] | None] = []
        if self.require_daily_bullish and not df_daily.empty:
            prev_close = df_daily["Close"].shift(1)
            prev_open = df_daily["Open"].shift(1)
            ok = prev_close > prev_open
            qualifying_sets.append(
                {ts.date() for ts, q in ok.items() if bool(q)}
            )
        if self.max_gap_up_pct is not None and not df_daily.empty:
            prev_close = df_daily["Close"].shift(1)
            gap = (df_daily["Open"] - prev_close) / prev_close
            ok = gap <= self.max_gap_up_pct
            qualifying_sets.append(
                {ts.date() for ts, q in ok.items() if bool(q)}
            )
        if self.require_above_daily_sma and not df_daily.empty:
            sma = df_daily["Close"].rolling(self.require_above_daily_sma).mean()
            ok = df_daily["Close"] > sma
            qualifying_sets.append(
                {ts.date() for ts, q in ok.items() if bool(q)}
            )
        if self.min_daily_rvol is not None and not df_daily.empty:
            avg_vol = df_daily["Volume"].rolling(20).mean()
            rvol = df_daily["Volume"] / avg_vol
            ok = rvol >= self.min_daily_rvol
            qualifying_sets.append(
                {ts.date() for ts, q in ok.items() if bool(q)}
            )
        qualifying: set[date] | None
        if qualifying_sets:
            qualifying = qualifying_sets[0]
            for s in qualifying_sets[1:]:
                qualifying = qualifying & s
        else:
            qualifying = None

        signals: list[FirstCandleRuleSignal] = []
        local_dates = self._local_dates(df_1m.index)
        for sess_date, sess in self._iter_sessions(df_1m, local_dates):
            if qualifying is not None and sess_date not in qualifying:
                continue
            sig = self._scan_session(sess_date, sess)
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
    ) -> FirstCandleRuleSignal | None:
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
        box_end = open_minutes + self.box_minutes

        # OR = bars whose start falls inside [open, open + box_minutes).
        # On 1m data with a 5m box this is the first 5 bars; on 5m
        # data it's a single bar (which is fine — the OR collapses to
        # one candle, matching the video's "first 5-minute candle"
        # framing literally).
        box_mask = (bar_minutes >= open_minutes) & (bar_minutes < box_end)
        box_bars = sess[box_mask]
        if len(box_bars) == 0:
            return None
        box_high = float(box_bars["High"].max())
        box_low = float(box_bars["Low"].min())
        if box_high - box_low <= 0:
            return None
        box_open_ts = box_bars.index[0]
        box_close_ts = box_bars.index[-1]

        # Post-box bars only — this is where the FVG / retest /
        # engulfing logic plays out.
        post_mask = bar_minutes >= box_end
        post = sess[post_mask]
        if len(post) < 3:
            return None
        post_minutes = bar_minutes[post_mask]
        opens = post["Open"].to_numpy(dtype=float)
        highs = post["High"].to_numpy(dtype=float)
        lows = post["Low"].to_numpy(dtype=float)
        closes = post["Close"].to_numpy(dtype=float)
        volumes = post["Volume"].to_numpy(dtype=float) if "Volume" in post.columns else None
        timestamps = post.index
        # Reference volume for the entry-bar RVOL filter — average
        # volume of the first 30 bars of the session (or all post-box
        # bars if the session is shorter). Computed once per session.
        early_vol_avg: float | None = None
        if (
            self.min_entry_bar_rvol is not None
            and volumes is not None
            and len(volumes) > 0
        ):
            n = min(30, len(volumes))
            avg = float(np.nanmean(volumes[:n]))
            early_vol_avg = avg if avg > 0 else None

        # State machine:
        #   - range_break_idx is None  → still waiting for the first
        #     candle that closes above box_high.
        #   - After the range break, EVERY subsequent bar (including
        #     the range-break bar itself) is checked for a 3-bar
        #     bullish FVG above the OR. If found:
        #       Path A — fires on the range-break bar (the FVG bar
        #                 IS the range-break bar, ``fvg_form_idx ==
        #                 range_break_idx``).
        #       Path B — fires on a later bar, AFTER price has
        #                 dipped back into the OR (``retest_idx`` set).
        # Stop in both paths = ``range_break_low − tick``. Critically
        # not the FVG-bar's low for Path B (per the video).
        range_break_idx: int | None = None
        range_break_low: float | None = None
        retest_idx: int | None = None
        retest_low: float | None = None

        def fvg_at(i: int) -> tuple[float, float] | None:
            """Return ``(fvg_low, fvg_high)`` if bar ``i`` is the
            displacement bar of a valid bullish 3-bar FVG, else None.

            The FVG zone position vs. ``box_high`` is intentionally
            NOT checked — the video defines a valid setup as "FVG
            created from the range break", meaning the displacement
            bar that breaks ``box_high`` and simultaneously forms a
            3-bar gap qualifies even if the gap zone itself sits
            below ``box_high`` (e.g., QQQ 2026-02-06 09:38: bar i
            closed at 603.78 above box_high 603.24, but its FVG
            window [602.66, 602.95] is fully inside the OR — still
            a valid Path A setup).

            For Path B, the breakout-side context is enforced by
            requiring a retest first; the FVG zone position is
            again secondary.
            """
            if i < 2:
                return None
            gap_low = highs[i - 2]
            gap_high = lows[i]
            if gap_high <= gap_low:
                return None
            if (
                self.require_bullish_displacement
                and closes[i] <= opens[i]
            ):
                return None
            if self.min_fvg_gap_pct > 0:
                if (gap_high - gap_low) / closes[i] < self.min_fvg_gap_pct:
                    return None
            # Entry-bar RVOL gate. Real institutional displacement
            # comes with volume; a thin algorithmic gap that
            # geometrically passes the FVG check but lacks volume
            # is just noise.
            if (
                self.min_entry_bar_rvol is not None
                and volumes is not None
                and early_vol_avg is not None
                and i < len(volumes)
            ):
                bar_rvol = volumes[i] / early_vol_avg
                if bar_rvol < self.min_entry_bar_rvol:
                    return None
            return float(gap_low), float(gap_high)

        def fire_signal(
            path: str, fvg_form_idx: int, fvg_low: float, fvg_high: float
        ) -> FirstCandleRuleSignal | None:
            entry_price = float(closes[fvg_form_idx])
            # Stop is the LOW of the FIRST bar in the 3-bar FVG
            # window (bar i-2). If price falls below this level the
            # FVG's structural support is broken — same level for
            # Path A and Path B. (Replaces the prior range-break-low
            # rule which felt out-of-place for Path B retest setups.)
            fvg_pre_low_val = float(lows[fvg_form_idx - 2])
            stop = fvg_pre_low_val - self.stop_tick_buffer
            if entry_price - stop <= 0:
                return None
            tp = entry_price + self.target_r_multiple * (
                entry_price - stop
            )
            return FirstCandleRuleSignal(
                session_date=sess_date,
                direction="long",
                box_low=box_low,
                box_high=box_high,
                box_open_ts=box_open_ts,
                box_close_ts=box_close_ts,
                range_break_ts=timestamps[range_break_idx],
                range_break_low=float(range_break_low),
                fvg_low=fvg_low,
                fvg_high=fvg_high,
                fvg_form_ts=timestamps[fvg_form_idx],
                fvg_pre_ts=timestamps[fvg_form_idx - 2],
                fvg_pre_low=fvg_pre_low_val,
                path=path,
                retest_ts=(
                    timestamps[retest_idx] if retest_idx is not None else None
                ),
                retest_low=retest_low,
                entry_ts=timestamps[fvg_form_idx],
                entry_price=entry_price,
                stop_loss=stop,
                take_profit=tp,
            )

        for i in range(len(post)):
            # **Short-side hard fail — applies in EVERY phase.** The
            # video's range-break rule is close-based ("the candle
            # broke through and closed through it" — close > box_high
            # for long). Mirror for the bearish direction: a 1m bar
            # that **closes** below ``box_low`` means sellers took
            # control first. A long setup that develops later in the
            # same session is fighting that flip and should be
            # abandoned.
            #
            # TSLA 2026-01-22: 09:36 closed 432.63 < box_low 433.30,
            # 09:48 closed 433.16 < box_low. Under the old wick-only
            # rule (and under the rule that only fired AFTER a long
            # range break) neither invalidated, and the session went
            # on to fake a Path A long at 09:52. Close-based gating
            # in every phase blocks that.
            if closes[i] < box_low:
                return None
            # Stale-setup + time-of-day caps only apply once we've
            # already started tracking a long range break.
            if range_break_idx is not None:
                if i - range_break_idx > self.max_total_bars:
                    return None
                if post_minutes[i] >= cutoff_minutes:
                    return None

            if range_break_idx is None:
                # Phase 1 — wait for the first candle that closes
                # above ``box_high``. "Closed through it", per the
                # video. Wick-only breaks don't qualify.
                if closes[i] > box_high:
                    range_break_idx = i
                    range_break_low = float(lows[i])
                    # Path A check — does this same bar form a
                    # valid FVG?
                    fvg = fvg_at(i)
                    if fvg is not None:
                        sig = fire_signal("A", i, *fvg)
                        if sig is not None:
                            return sig
                continue

            # Phase 2 — post range break. Track retests so Path B
            # can gate on them.
            if lows[i] <= box_high:
                if retest_idx is None:
                    retest_idx = i
                    retest_low = float(lows[i])
                else:
                    if lows[i] < retest_low:
                        retest_low = float(lows[i])

            # Path B FVG check — only fires after a retest if the
            # ``require_retest_for_path_b`` knob is set (default).
            if (
                self.require_retest_for_path_b
                and retest_idx is None
            ):
                continue
            fvg = fvg_at(i)
            if fvg is not None:
                sig = fire_signal("B", i, *fvg)
                if sig is not None:
                    return sig
        return None
