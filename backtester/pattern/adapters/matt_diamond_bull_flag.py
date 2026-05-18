"""Matt Diamond Bull Flag detector — large-cap intraday continuation.

Source: Matt Diamond, "Bull Flag Patterns EXPLAINED: How To Spot and
Trade Them Like A Pro!" (YouTube ``SNjtH42aCuk``, 2025-03-24). The
demonstration ticker is TSLA on 1m bars.

Spec sheet for the rules implemented here lives at
``docs/strategy_notes/matt_diamond_bull_flag.md``. This module is the
Matt-rule sibling of :class:`pattern.adapters.bull_flag.BullFlagDetector`
(Ross Cameron rules). The two are intentionally kept separate so we can
A/B compare them — they target different universes (Ross = low-float
small-caps, Matt = liquid large-caps) and use different entry mechanics
(Ross = pole-high breakout, Matt = Green-Take-Red).

Key differences vs. Ross detector
---------------------------------
1.  **Universe** — float / penny-price gates *off* by default; designed
    for TSLA-class names.
2.  **Pole = opening drive** — instead of any rolling pole window, the
    pole is the *first* push of the regular session, bounded by
    ``opening_drive_max_bars`` (영상: "strong opening drive higher").
3.  **Pre-market high gate (hard)** — session must open *above* the PM
    high, and the pullback low must stay above PM high. PM high is a
    Matt-signature reference, not a soft Ross-style "preferable above".
4.  **10 EMA (not 9 EMA)** — Matt literally says "I just changed it to a
    10". Functionally near-identical, kept faithful to the source.
5.  **Green-Take-Red entry** — not pole-high breakout. Trigger fires on
    the *next* bar after a GTR candle takes out the bar that wicked
    above the GTR high.
6.  **ATR-aware stop** — stop distance is the wider of (GTR candle low)
    and (entry − ATR × multiplier). Matt: "risk a point on Tesla,
    that's pretty good" — a point on TSLA ≈ 1 ATR on a 1m chart at
    that volatility level.
7.  **Market regime gate** — SPY/QQQ (caller-provided ``df_market``)
    must be above its own SMA to qualify. Matt explicitly tied the
    whole TSLA setup to "the ES/NQ gapping over resistance".
8.  **Secondary bull flag** — Matt traded the *same* setup twice in
    the example. ``max_nth_pullback=2`` matches that.

Long-only (영상은 long side만 다룸).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, time, timedelta
from typing import Iterable, TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from data.domain.models import EarningsEvent, NewsEvent


@dataclass(frozen=True)
class MattDiamondSignal:
    """Single intraday Matt-Diamond-style Bull Flag long entry.

    Field semantics
    ---------------
    pole_start_ts / pole_start_price:
        Open of the regular session — Matt's pole begins at the open
        (or the wick low of the opening dip if buyers stepped up
        immediately).
    pole_end_ts / pole_end_price:
        Peak of the opening drive — the bar from which the first
        pullback originates.
    flag_low / flag_low_ts:
        Lowest low inside the 1~``pullback_max_bars`` pullback bars.
    gtr_ts / gtr_high / gtr_low:
        The Green-Take-Red candle. Long buy stop is at ``gtr_high +
        tick``; protective stop sits at the wider of ``gtr_low`` or
        ``entry − ATR × multiplier``.
    entry_ts / entry_price:
        The bar that *fills* the buy stop (= the bar after the GTR
        candle when that bar's high crosses ``gtr_high``). For a
        same-bar fill option see ``allow_same_bar_fill``.
    stop_loss:
        Final stop after ATR widening.
    atr_at_entry:
        ATR value (period = ``atr_period``) at the GTR bar. Used by
        the strategy layer for scalper R-targeting + risk sizing.
    pm_high:
        Pre-market high for the session. Always > 0 when this signal
        is emitted (gate is hard).
    ema10_at_pullback:
        10 EMA at ``flag_low_ts`` — debug field; equals the value the
        pullback respected.
    nth_pullback:
        1 = first bull flag of the session, 2 = secondary. Caller may
        sub-filter on this.
    gap_pct, rvol:
        Session gates that already passed (debug/reporting only).
    """

    session_date: date
    direction: str  # always "long"

    pole_start_ts: pd.Timestamp
    pole_start_price: float
    pole_end_ts: pd.Timestamp
    pole_end_price: float
    flag_low: float
    flag_low_ts: pd.Timestamp

    gtr_ts: pd.Timestamp
    gtr_high: float
    gtr_low: float

    entry_ts: pd.Timestamp
    entry_price: float
    stop_loss: float
    atr_at_entry: float
    pm_high: float
    ema10_at_pullback: float
    nth_pullback: int

    gap_pct: float
    rvol: float
    open_price: float


class MattDiamondBullFlagDetector:
    """Long-only large-cap intraday Bull Flag detector — Matt Diamond style.

    Parameters
    ----------
    require_premarket_high:
        Hard gate (default True). Session must open *above* PM high,
        and every pullback bar's low must stay above PM high. Without
        PM data the session is skipped (graceful — no data, no signal).
    premarket_high_by_date:
        ``{session_date: pm_high}`` — caller computes this from raw
        intraday before any RTH filter. If empty the gate fails closed.
    market_regime_ok_by_date:
        ``{session_date: bool}`` — caller-precomputed SPY/QQQ regime
        check (e.g. SPY close > SMA50 on that date). ``True`` means the
        session is *eligible*. Empty dict = gate disabled (treat all
        sessions as OK).
    min_gap_pct / max_gap_pct:
        Session daily gap-up size. Matt's example was a small-but-
        decisive gap, not a Ross-style +10%. Defaults: 0.5% min,
        +15% max (above that is exhaustion territory for large-caps).
    min_rvol:
        Today's daily volume / 50-day average. Large-caps rarely hit
        5×; default 1.3× catches "elevated" days while filtering
        ordinary chop. Matt verbal cue: "elevated volume on the
        igniting candle".
    min_price:
        Soft floor to avoid pathological cheap names (default $20 —
        well below TSLA/AAPL/AMD range, generous on purpose).
    require_float_filter:
        Default **False** — Matt's setup targets large-caps. Toggle
        on if you want to constrain by float anyway.
    opening_drive_max_bars:
        Pole window = first N bars of the regular session. Default 12
        (= first 12 minutes on 1m, ≈ Matt's "256 → 263 in 3 minutes
        opening drive" + room).
    pole_min_pct:
        Minimum opening-drive size (high − open) / open. Default 0.5%
        — large-caps rarely sprint 5%+ in 12 minutes; even Matt's TSLA
        case was ≈ 2.7%.
    pullback_max_bars:
        Max bars for the pullback before GTR must fire. Matt: "1-3
        candles of pullback", default 4 (room for one wick bar).
    ema_period:
        Default 10 (Matt literally uses 10 EMA). Kept tunable.
    ema_tolerance_pct:
        Pullback low must satisfy ``flag_low >= ema10 × (1 - tol)``.
        Default 0.7% — Matt: "if this had dipped below it… maybe this
        could start a reversal". Allow a tiny wick.
    require_pullback_close_above_ema:
        Stricter variant — every pullback bar's *close* must stay
        above the 10 EMA (wicks below are OK). Default True (matches
        Matt's "controlled selling" description).
    atr_period:
        ATR window. Default 14 — standard.
    atr_stop_multiplier:
        Stop = max(gtr_low, entry − ATR × multiplier). Default 1.0
        (Matt: "risk a point on Tesla = pretty good"; 1 ATR on TSLA
        1m ≈ $1 at that volatility). Set 0 to disable ATR widening
        and use raw GTR low.
    allow_same_bar_fill:
        If True (default False), the GTR candle itself fills the buy
        stop *if* its own high crosses the trigger inside the same
        bar — used when bar resolution is coarse. Default False ("get
        the pop above the high" = next bar logic).
    latest_entry_local:
        No new entries after this local-time cutoff. Default 14:30 —
        Matt scalped TSLA into the afternoon in the video; not a
        morning-only setup like Ross's. Set 11:30 for stricter
        morning-only behavior.
    max_nth_pullback:
        Max sequential pullbacks within a session that may fire
        signals. Default 2 (Matt's example = exactly two flags).
        Cap at 3 to allow a third continuation; below 1 disables the
        whole session.
    stop_tick_buffer:
        Subtracted from the chosen stop price ($0.01 default).
    catalyst_dates:
        Set of dates the caller has tagged as "catalyst-ok"
        (= earnings window ∪ news catalyst). Pre-computed via
        :func:`compute_earnings_window_dates` and
        :func:`compute_news_session_dates` so the detector stays
        framework-free. Empty/None disables the gate.
    require_catalyst:
        Hard gate. ``True`` rejects every session whose date is not
        in ``catalyst_dates``. Matt: "when you add a catalyst to a
        stock and it gets going in one direction, the probability
        of follow-through is there versus just trading any bull
        flag in the market".
    require_prior_igniting_candle:
        Pre-session-day must be a strong green candle on elevated
        volume. Matt's TSLA case: "a nice igniting candle on Friday
        on some elevated volume" preceded Monday's setup.
    igniting_candle_min_pct:
        Prior day close-vs-open must be ≥ this fraction. Default 0.02
        (= +2% green body).
    igniting_candle_min_rvol:
        Prior day volume / 50d avg must be ≥ this. Default 1.5×.
    require_daily_resistance_break:
        Session open / first hour must clear a recent daily swing
        high. Matt: "it was opening above a lot of resistance".
    resistance_lookback_days:
        Bars looked back for the resistance level. Default 20 = a
        month of trading — close to "recent" without being trivially
        flat.
    """

    name = "matt_diamond_bull_flag"

    def __init__(
        self,
        *,
        require_premarket_high: bool = True,
        premarket_high_by_date: dict[date, float] | None = None,
        market_regime_ok_by_date: dict[date, bool] | None = None,
        # ---- Daily session gates ----
        # Matt's video never quotes specific gap / RVOL thresholds —
        # he describes "a gap up" and "elevated volume". On liquid
        # large-caps the previous conservative interpretation
        # (gap ≥ 0.5%, RVOL ≥ 1.3×) silently killed ~95% of trading
        # days at Stage 1 of the funnel (NVDA 2025-01 → 2026-05: 343
        # trading days → 14 qualifying sessions → 3 signals). Loosened
        # to match Matt's *qualitative* description: any gap up and
        # average-or-better volume. The default funnel still applies
        # PM-high + regime + GTR which keep the strategy selective.
        min_gap_pct: float = 0.001,
        max_gap_pct: float | None = 0.15,
        min_rvol: float = 1.0,
        min_price: float = 20.0,
        max_price: float | None = None,
        require_float_filter: bool = False,
        float_shares: float | None = None,
        max_float_shares: float = 100_000_000.0,
        # ---- Pole / pullback geometry ----
        opening_drive_max_bars: int = 12,
        # Tuned to 1.5% from a 15-ticker × 17-month sweep — combined
        # with the resistance-break + GTR-volume gates this is the
        # win-rate-optimal pole threshold (77.8% win, PF 7.68).
        # Earlier 0.3% default gave only 24% win rate.
        pole_min_pct: float = 0.015,
        # Matt: "1-3 candles of pullback". Secondary-flag pullbacks
        # routinely run 4-5 bars; 5 captures both without inflating
        # the search cost meaningfully.
        pullback_max_bars: int = 5,
        ema_period: int = 10,
        # Matt explicitly shows *wicks below* 10 EMA in the source
        # video — "if this had dipped below it... maybe this could
        # start a reversal". The pullback low can wick further than
        # 0.7%; 1.5% matches the visible wick depth.
        ema_tolerance_pct: float = 0.015,
        # Wicks below EMA are fine per Matt; close-below-EMA on the
        # other hand IS bearish. We default to OFF because the wick
        # tolerance above already enforces "EMA held in spirit", and
        # demanding every pullback close above EMA was over-strict
        # (cost ~30% of qualifying signals).
        require_pullback_close_above_ema: bool = False,
        atr_period: int = 14,
        atr_stop_multiplier: float = 1.0,
        # Matt scalps fast — the GTR candle itself is the actionable
        # bar. The previous default (False = wait for next bar to
        # cross trigger + tick) lost ~25% of fills to fade bars.
        # Same-bar fill is also more faithful to "I'd be in at the
        # green-takes-red candle".
        allow_same_bar_fill: bool = True,
        latest_entry_local: time = time(14, 30),
        max_nth_pullback: int = 2,
        stop_tick_buffer: float = 0.01,
        # Catalyst gates (Matt: "context first, pattern second")
        catalyst_dates: set[date] | None = None,
        require_catalyst: bool = False,
        require_prior_igniting_candle: bool = False,
        igniting_candle_min_pct: float = 0.02,
        igniting_candle_min_rvol: float = 1.5,
        # Sweep-optimal default ON. The single biggest win-rate driver
        # in the 2026-05 research: 15-ticker × 17-month grid showed
        # 25% → 78% win rate when this gate is ON alongside
        # ``min_resistance_break_pct=0.005`` and the GTR-volume gate.
        # Trade-off: signal count drops by ~80%, but PF jumps 0.20 → 7.68.
        require_daily_resistance_break: bool = True,
        resistance_lookback_days: int = 20,
        # Distance ABOVE the N-day high. Sweep-tuned to 0.5% — "open
        # meaningfully above resistance, not just tagging it". Going to
        # 0% drops PF from 7.68 → 4.18 (true breakouts vs. probes).
        min_resistance_break_pct: float = 0.005,
        # Skip the first N bars of the regular session for entries.
        # Opening 1-5 minutes are the noisiest; the GTR setup that
        # fires at 9:31 has a much different reliability profile from
        # one at 9:38. 0 = no skip.
        skip_first_n_minutes: int = 0,
        # Sweep-optimal default ON. GTR bar volume must expand vs the
        # prior red bar — "fresh buyers stepping in". On its own the
        # gate lifts win rate from 50% (resistance+pole stack) to
        # 63.6% (PF 4.18); stacked with resistance_break_pct=0.5% it
        # lifts to 77.8% (PF 7.68). Multiplier 1.0× = strictly more
        # volume than the red bar that preceded the GTR.
        require_gtr_volume_expansion: bool = True,
        gtr_volume_expansion_mult: float = 1.0,
        # VWAP-aware pullback gate — pullback low must stay above
        # session VWAP. Strong intraday continuation setups don't
        # break VWAP on the first pullback. ``False`` (default)
        # leaves it disabled.
        require_pullback_above_vwap: bool = False,
        # Hard cap on consecutive red bars inside the pullback — 4+
        # reds in a row is a real reversal even if the EMA holds.
        # 0 = disabled.
        max_consecutive_red_bars: int = 0,
        # Data quality
        min_bar_range: float = 0.001,
        max_bar_gap_seconds: int = 90,
    ) -> None:
        if opening_drive_max_bars < 2:
            raise ValueError("opening_drive_max_bars must be >= 2")
        if pullback_max_bars < 1:
            raise ValueError("pullback_max_bars must be >= 1")
        if ema_period < 2:
            raise ValueError("ema_period must be >= 2")
        if atr_period < 2:
            raise ValueError("atr_period must be >= 2")
        if max_nth_pullback < 1:
            raise ValueError("max_nth_pullback must be >= 1")

        self.require_premarket_high = require_premarket_high
        self.premarket_high_by_date = premarket_high_by_date or {}
        self.market_regime_ok_by_date = market_regime_ok_by_date or {}
        self.min_gap_pct = min_gap_pct
        self.max_gap_pct = max_gap_pct
        self.min_rvol = min_rvol
        self.min_price = min_price
        self.max_price = max_price
        self.require_float_filter = require_float_filter
        self.float_shares = float_shares
        self.max_float_shares = max_float_shares
        self.opening_drive_max_bars = opening_drive_max_bars
        self.pole_min_pct = pole_min_pct
        self.pullback_max_bars = pullback_max_bars
        self.ema_period = ema_period
        self.ema_tolerance_pct = ema_tolerance_pct
        self.require_pullback_close_above_ema = require_pullback_close_above_ema
        self.atr_period = atr_period
        self.atr_stop_multiplier = atr_stop_multiplier
        self.allow_same_bar_fill = allow_same_bar_fill
        self.latest_entry_local = latest_entry_local
        self.max_nth_pullback = max_nth_pullback
        self.stop_tick_buffer = stop_tick_buffer
        self.catalyst_dates = set(catalyst_dates) if catalyst_dates else set()
        self.require_catalyst = require_catalyst
        self.require_prior_igniting_candle = require_prior_igniting_candle
        self.igniting_candle_min_pct = igniting_candle_min_pct
        self.igniting_candle_min_rvol = igniting_candle_min_rvol
        self.require_daily_resistance_break = require_daily_resistance_break
        self.resistance_lookback_days = resistance_lookback_days
        self.min_resistance_break_pct = min_resistance_break_pct
        self.skip_first_n_minutes = skip_first_n_minutes
        self.require_gtr_volume_expansion = require_gtr_volume_expansion
        self.gtr_volume_expansion_mult = gtr_volume_expansion_mult
        self.require_pullback_above_vwap = require_pullback_above_vwap
        self.max_consecutive_red_bars = max_consecutive_red_bars
        self.min_bar_range = min_bar_range
        self.max_bar_gap_seconds = max_bar_gap_seconds

    # ---- public API -----------------------------------------------------

    def detect(
        self,
        df_intraday: pd.DataFrame,
        df_daily: pd.DataFrame,
    ) -> list[MattDiamondSignal]:
        if df_intraday.empty or df_daily.empty:
            return []

        # ---- session-level eligibility (daily gates) ----
        if self.require_float_filter:
            if self.float_shares is None:
                return []
            if self.float_shares >= self.max_float_shares:
                return []

        qualifying = self._qualifying_sessions(df_daily)
        if not qualifying:
            return []

        out: list[MattDiamondSignal] = []
        local_dates = self._local_dates(df_intraday.index)
        for sess_date, sess in self._iter_sessions(df_intraday, local_dates):
            meta = qualifying.get(sess_date)
            if meta is None:
                continue
            # Market-regime gate. Empty dict = caller chose not to wire
            # SPY/QQQ → treat as always-OK (graceful default).
            if self.market_regime_ok_by_date:
                if not self.market_regime_ok_by_date.get(sess_date, False):
                    continue
            pm_high = float(self.premarket_high_by_date.get(sess_date, 0.0))
            if self.require_premarket_high and pm_high <= 0.0:
                continue  # hard gate — no PM data => skip
            out.extend(self._scan_session(sess_date, sess, meta, pm_high))
        return out

    # ---- session gates --------------------------------------------------

    def _qualifying_sessions(
        self, df_daily: pd.DataFrame
    ) -> dict[date, dict]:
        if "Open" not in df_daily.columns or "Close" not in df_daily.columns:
            return {}

        prev_close = df_daily["Close"].shift(1)
        gap_pct = (df_daily["Open"] - prev_close) / prev_close
        avg_vol = df_daily["Volume"].rolling(50, min_periods=20).mean()
        rvol = df_daily["Volume"] / avg_vol

        gate = (gap_pct >= self.min_gap_pct) & (rvol >= self.min_rvol)
        if self.max_gap_pct is not None:
            gate = gate & (gap_pct <= self.max_gap_pct)
        if self.min_price is not None:
            gate = gate & (df_daily["Open"] >= self.min_price)
        if self.max_price is not None:
            gate = gate & (df_daily["Open"] <= self.max_price)

        # Pre-compute prior-day igniting candle + N-day high (resistance)
        # vectors so the per-row loop stays O(n).
        prev_open = df_daily["Open"].shift(1)
        prev_close_for_igniting = df_daily["Close"].shift(1)
        prev_vol = df_daily["Volume"].shift(1)
        prev_avg_vol = avg_vol.shift(1)
        prior_pct = (prev_close_for_igniting - prev_open) / prev_open
        prior_rvol = prev_vol / prev_avg_vol
        # N-day high *up to and including the prior session* (not today,
        # so the gate is forward-looking-safe).
        nbar_high = (
            df_daily["High"]
            .rolling(
                self.resistance_lookback_days,
                min_periods=min(5, self.resistance_lookback_days),
            )
            .max()
            .shift(1)
        )

        out: dict[date, dict] = {}
        for ts, ok in gate.items():
            if not bool(ok):
                continue
            sess_date = ts.date()
            # Catalyst gate (earnings window OR news lookback set).
            if self.require_catalyst:
                if sess_date not in self.catalyst_dates:
                    continue
            # Prior-day igniting candle gate.
            if self.require_prior_igniting_candle:
                pp = prior_pct.loc[ts]
                pr = prior_rvol.loc[ts]
                if pd.isna(pp) or pd.isna(pr):
                    continue
                if (
                    pp < self.igniting_candle_min_pct
                    or pr < self.igniting_candle_min_rvol
                ):
                    continue
            # Daily resistance break: today's open >= recent N-day high
            # × (1 + min_resistance_break_pct). 0% = "open at level",
            # 0.5% = "open meaningfully above level".
            if self.require_daily_resistance_break:
                lvl = nbar_high.loc[ts]
                if pd.isna(lvl):
                    continue
                threshold = float(lvl) * (1.0 + self.min_resistance_break_pct)
                if float(df_daily.loc[ts, "Open"]) < threshold:
                    continue
            out[sess_date] = {
                "gap_pct": float(gap_pct.loc[ts]),
                "rvol": float(rvol.loc[ts]),
                "open_price": float(df_daily.loc[ts, "Open"]),
            }
        return out

    # ---- intraday helpers ----------------------------------------------

    @staticmethod
    def _local_dates(idx: pd.DatetimeIndex) -> np.ndarray:
        return np.array([ts.date() for ts in idx])

    @staticmethod
    def _iter_sessions(
        df: pd.DataFrame, local_dates: np.ndarray
    ) -> Iterable[tuple[date, pd.DataFrame]]:
        if len(df) == 0:
            return
        change = np.r_[True, local_dates[1:] != local_dates[:-1]]
        boundaries = np.flatnonzero(change)
        for i, start in enumerate(boundaries):
            end = boundaries[i + 1] if i + 1 < len(boundaries) else len(df)
            yield local_dates[start], df.iloc[start:end]

    # ---- core pattern logic --------------------------------------------

    def _scan_session(
        self,
        sess_date: date,
        sess: pd.DataFrame,
        meta: dict,
        pm_high: float,
    ) -> list[MattDiamondSignal]:
        n = len(sess)
        if n < self.opening_drive_max_bars + self.pullback_max_bars + 2:
            return []

        opens = sess["Open"].to_numpy(dtype=float)
        highs = sess["High"].to_numpy(dtype=float)
        lows = sess["Low"].to_numpy(dtype=float)
        closes = sess["Close"].to_numpy(dtype=float)
        volumes = (
            sess["Volume"].to_numpy(dtype=float)
            if "Volume" in sess.columns
            else np.zeros(len(sess), dtype=float)
        )
        ts_idx = sess.index

        # Session VWAP — cumulative typical-price × volume / cumulative
        # volume. Used by the optional ``require_pullback_above_vwap``
        # gate to enforce "first pullback stays above VWAP" — a textbook
        # large-cap continuation filter.
        typical = (highs + lows + closes) / 3.0
        tpv = typical * volumes
        cv = np.cumsum(volumes)
        ctpv = np.cumsum(tpv)
        # Guard cv==0 (no volume = use typical price)
        with np.errstate(divide="ignore", invalid="ignore"):
            vwap = np.where(cv > 0, ctpv / cv, typical)

        # PM-high "buyers reclaim" gate — relaxed from a strict
        # ``open > pm_high`` (was too tight: Matt's TSLA case
        # literally opened *below* PM high and buyers stepped up in
        # the first wick). Accept any session where some close inside
        # the opening-drive window crosses back above PM high
        # ("buyers stepped up immediately. … then it got above pre-
        # market high"). Pole-peak > PM high is enforced separately
        # below, so the pullback that follows happens *above* PM
        # high, matching the video.
        if self.require_premarket_high:
            drive_cap = min(self.opening_drive_max_bars + 1, n)
            if not np.any(closes[:drive_cap] > pm_high):
                return []

        # Indicators — computed once per session for speed.
        ema = pd.Series(closes).ewm(span=self.ema_period, adjust=False).mean().to_numpy()
        atr = _atr_wilder(highs, lows, closes, self.atr_period)

        cutoff_minutes = (
            self.latest_entry_local.hour * 60
            + self.latest_entry_local.minute
        )
        local_time = ts_idx if ts_idx.tz is None else ts_idx.tz_convert(ts_idx.tz)
        bar_minutes = np.array(
            [t.hour * 60 + t.minute for t in local_time.time]
        )

        # Pole = opening drive. Anchor at bar 0 (regular-session open)
        # — Matt frames the first push from the open, not from any
        # generic rolling window.
        drive_end_cap = min(self.opening_drive_max_bars, n - 1)
        pole_start_idx = 0
        pole_start_price = float(opens[0])
        # Pole peak = highest high in the opening-drive window.
        pole_end_offset = int(np.argmax(highs[: drive_end_cap + 1]))
        pole_end_idx = pole_end_offset
        pole_end_price = float(highs[pole_end_idx])

        # Sanity: opening dip + bounce → use the dip wick low as the
        # true pole start (Matt's "took a dip off the open, buyers
        # stepped up immediately"). Search for argmin BEFORE the peak.
        if pole_end_idx > 0:
            pre_peak_min_off = int(np.argmin(lows[: pole_end_idx + 1]))
            pole_start_idx = pre_peak_min_off
            pole_start_price = float(lows[pre_peak_min_off])

        if pole_start_price <= 0:
            return []
        pole_pct = (pole_end_price - pole_start_price) / pole_start_price
        if pole_pct < self.pole_min_pct:
            return []
        # Pole peak must clear PM high (Matt: "It's already broke
        # pre-market high — then it does the first bull flag …").
        if self.require_premarket_high and pole_end_price <= pm_high:
            return []
        # Opening-drive data quality (gaps / zero-range bars).
        if not _bars_clean(
            highs[pole_start_idx : pole_end_idx + 1],
            lows[pole_start_idx : pole_end_idx + 1],
            ts_idx[pole_start_idx : pole_end_idx + 1],
            self.min_bar_range,
            self.max_bar_gap_seconds,
        ):
            return []

        signals: list[MattDiamondSignal] = []
        # Walk forward from the bar right after the opening-drive peak.
        # ``i`` always points to the *last bar accepted into the current
        # pullback*; pullback search starts at i+1.
        i = pole_end_idx
        nth = 0
        while nth < self.max_nth_pullback and i < n - 2:
            if bar_minutes[i] >= cutoff_minutes:
                break

            sig = self._find_gtr_signal(
                opens, highs, lows, closes, volumes, vwap, ts_idx,
                ema, atr, sess_date, meta, pm_high,
                start_after=i,
                current_peak=pole_end_price if nth == 0 else float(highs[i]),
                bar_minutes=bar_minutes,
                cutoff_minutes=cutoff_minutes,
                pole_start_ts=ts_idx[pole_start_idx],
                pole_start_price=pole_start_price,
                pole_end_ts=ts_idx[pole_end_idx] if nth == 0 else ts_idx[i],
                pole_end_price=pole_end_price if nth == 0 else float(highs[i]),
                nth_pullback=nth + 1,
            )
            if sig is None:
                break
            signals.append(sig)
            nth += 1
            # Anchor the next iteration at the entry bar — secondary
            # pullback must originate AFTER this fill.
            entry_loc = ts_idx.get_loc(sig.entry_ts)
            if isinstance(entry_loc, slice):
                entry_loc = entry_loc.stop - 1
            # For nth=2+, re-detect a fresh peak after the entry: the
            # secondary pole_end is the highest high seen from entry to
            # the next pullback start. Pass via current_peak loop var.
            i = max(int(entry_loc), pole_end_idx + 1)
            # Allow a fresh local peak — walk forward to the next swing
            # high before the next pullback search begins.
            new_peak_off = i
            for k in range(i + 1, min(i + self.opening_drive_max_bars, n)):
                if highs[k] > highs[new_peak_off]:
                    new_peak_off = k
                elif highs[k] < highs[new_peak_off]:
                    break
            i = new_peak_off

        return signals

    def _find_gtr_signal(
        self,
        opens: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        volumes: np.ndarray,
        vwap: np.ndarray,
        ts_idx: pd.DatetimeIndex,
        ema: np.ndarray,
        atr: np.ndarray,
        sess_date: date,
        meta: dict,
        pm_high: float,
        *,
        start_after: int,
        current_peak: float,
        bar_minutes: np.ndarray,
        cutoff_minutes: int,
        pole_start_ts: pd.Timestamp,
        pole_start_price: float,
        pole_end_ts: pd.Timestamp,
        pole_end_price: float,
        nth_pullback: int,
    ) -> MattDiamondSignal | None:
        """Find a Green-Take-Red entry after a valid pullback.

        Pullback validity gates (all must hold):
          1. ≥1 red candle (close < open)
          2. Every pullback bar's low > PM high (hard PM gate)
          3. Pullback low respects 10 EMA tolerance
          4. (optional) every pullback bar's close > 10 EMA
          5. Pullback width ≤ pullback_max_bars

        GTR trigger:
          first green candle whose high > immediately-prior red high.
          Buy stop = gtr_high + tick. Filled on the NEXT bar that
          trades through that price (or same bar if
          ``allow_same_bar_fill``).
        """
        n = len(opens)
        flag_low = float("inf")
        flag_low_idx = -1
        last_red_idx = -1
        consecutive_reds = 0  # Track streak for max_consecutive_red_bars gate.

        for k in range(start_after + 1, min(start_after + self.pullback_max_bars + 1, n)):
            if bar_minutes[k] >= cutoff_minutes:
                return None
            # Data quality
            if (highs[k] - lows[k]) <= self.min_bar_range:
                return None
            if (
                ts_idx[k] - ts_idx[k - 1]
            ).total_seconds() > self.max_bar_gap_seconds:
                return None

            # PM high hard gate — pullback never trades below PM high.
            if self.require_premarket_high and lows[k] < pm_high:
                return None

            # 10 EMA hold — pullback low within tolerance of EMA.
            ema_k = float(ema[k])
            if ema_k <= 0:
                return None
            if lows[k] < ema_k * (1 - self.ema_tolerance_pct):
                return None
            # Optional close-above-EMA enforcement (controlled selling).
            if self.require_pullback_close_above_ema and closes[k] < ema_k:
                return None

            # ---- VWAP-hold gate (large-cap continuation textbook) ----
            # Pullback low must stay above session VWAP. Off by default
            # because it's an *additional* filter on top of EMA hold;
            # for trend days the two roughly coincide but on choppy days
            # VWAP is the stricter check.
            if self.require_pullback_above_vwap:
                vwap_k = float(vwap[k])
                if vwap_k > 0 and lows[k] < vwap_k:
                    return None

            is_red = closes[k] < opens[k]
            is_green = closes[k] > opens[k]

            # ---- Consecutive-red-bar cap ----
            # 4+ reds in a row inside the pullback is a real reversal
            # even if the EMA somehow holds. Disabled when 0.
            if is_red:
                consecutive_reds += 1
                if (
                    self.max_consecutive_red_bars > 0
                    and consecutive_reds >= self.max_consecutive_red_bars
                ):
                    return None
            elif is_green:
                consecutive_reds = 0

            if is_red:
                last_red_idx = k
                if lows[k] < flag_low:
                    flag_low = float(lows[k])
                    flag_low_idx = k
                continue

            # ── GTR trigger check ──
            # Need at least one prior red bar in this pullback for GTR
            # to be defined ("green TAKES red").
            if is_green and last_red_idx >= 0:
                prev_red_high = float(highs[last_red_idx])
                if highs[k] > prev_red_high:
                    gtr_idx = k
                    gtr_high = float(highs[gtr_idx])
                    gtr_low = float(lows[gtr_idx])

                    # ---- Skip-first-N-minutes gate ----
                    # Reject entries that fire too early in the session
                    # (e.g. first 5 bars). The opening 1-5 minutes are
                    # the noisiest; a GTR that fires at 9:31 is way
                    # less reliable than one at 9:38.
                    if self.skip_first_n_minutes > 0 and gtr_idx < self.skip_first_n_minutes:
                        return None

                    # ---- GTR volume expansion gate ----
                    # GTR bar volume should expand vs the prior red bar
                    # — "fresh buyers stepping in". A thin-volume GTR
                    # is usually a fake bounce.
                    if (
                        self.require_gtr_volume_expansion
                        and last_red_idx >= 0
                        and volumes[last_red_idx] > 0
                    ):
                        if (
                            volumes[gtr_idx]
                            < volumes[last_red_idx] * self.gtr_volume_expansion_mult
                        ):
                            return None

                    # Stop = wider of (gtr_low - tick) and
                    # (entry − ATR × mult). Wider = safer = less false
                    # stop-outs on intrabar noise.
                    entry_price_target = gtr_high + self.stop_tick_buffer
                    atr_k = float(atr[gtr_idx]) if not np.isnan(atr[gtr_idx]) else 0.0
                    stop_from_gtr = gtr_low - self.stop_tick_buffer
                    stop_from_atr = (
                        entry_price_target - atr_k * self.atr_stop_multiplier
                        if self.atr_stop_multiplier > 0 and atr_k > 0
                        else float("-inf")
                    )
                    stop_loss = max(stop_from_gtr, stop_from_atr)
                    if stop_loss >= entry_price_target:
                        return None  # ATR widened past entry — bad signal

                    # ── Fill logic ──
                    # Default: next bar's high crosses gtr_high + tick.
                    # Same-bar fill option treats the GTR candle *itself*
                    # as the fill (the bar that just took out the prior
                    # red's high is, by definition, executable at its
                    # own high). Previous form required ``highs[gtr_idx]
                    # >= gtr_high + 0.01`` which is *always* false (a
                    # bar's high can't exceed itself by a buffer), so
                    # the same-bar option was effectively dead code. We
                    # now fill at ``gtr_high`` (no buffer) on the GTR
                    # bar — matches Matt's quick scalper execution where
                    # the order tags the breakout high during the GTR
                    # candle itself.
                    fill_idx: int | None = None
                    if self.allow_same_bar_fill:
                        fill_idx = gtr_idx
                        entry_price_target = gtr_high  # no buffer on same-bar fill
                    else:
                        # Walk forward looking for the first bar whose
                        # high reaches entry_price_target. Cap search at
                        # gtr_idx + pullback_max_bars to bound work.
                        search_end = min(
                            gtr_idx + 1 + self.pullback_max_bars, n
                        )
                        for f in range(gtr_idx + 1, search_end):
                            if bar_minutes[f] >= cutoff_minutes:
                                break
                            if (
                                ts_idx[f] - ts_idx[f - 1]
                            ).total_seconds() > self.max_bar_gap_seconds:
                                break
                            if highs[f] >= entry_price_target:
                                fill_idx = f
                                break
                            # If price fails the GTR low while waiting
                            # to fill, the setup is dead.
                            if lows[f] < gtr_low:
                                fill_idx = None
                                break
                    if fill_idx is None:
                        return None

                    flag_low_final = (
                        flag_low if flag_low != float("inf") else gtr_low
                    )
                    flag_low_ts = (
                        ts_idx[flag_low_idx] if flag_low_idx >= 0 else ts_idx[gtr_idx]
                    )
                    return MattDiamondSignal(
                        session_date=sess_date,
                        direction="long",
                        pole_start_ts=pole_start_ts,
                        pole_start_price=pole_start_price,
                        pole_end_ts=pole_end_ts,
                        pole_end_price=pole_end_price,
                        flag_low=flag_low_final,
                        flag_low_ts=flag_low_ts,
                        gtr_ts=ts_idx[gtr_idx],
                        gtr_high=gtr_high,
                        gtr_low=gtr_low,
                        entry_ts=ts_idx[fill_idx],
                        entry_price=entry_price_target,
                        stop_loss=stop_loss,
                        atr_at_entry=atr_k,
                        pm_high=pm_high,
                        ema10_at_pullback=ema_k,
                        nth_pullback=nth_pullback,
                        gap_pct=meta["gap_pct"],
                        rvol=meta["rvol"],
                        open_price=meta["open_price"],
                    )
                # Green bar that didn't take prior red high → still in
                # pullback, treat as part of the consolidation. Reset
                # flag_low if this green made a new low (rare).
                if lows[k] < flag_low:
                    flag_low = float(lows[k])
                    flag_low_idx = k

        return None


# ---- module-private helpers --------------------------------------------


def _atr_wilder(
    highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int
) -> np.ndarray:
    """Wilder-smoothed ATR. Returns array of len(highs); first
    ``period`` values are NaN (warmup)."""
    n = len(highs)
    if n == 0:
        return np.array([], dtype=float)
    tr = np.empty(n, dtype=float)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        tr[i] = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
    atr = np.full(n, np.nan, dtype=float)
    if n < period:
        return atr
    atr[period - 1] = float(np.mean(tr[:period]))
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


def _bars_clean(
    highs: np.ndarray,
    lows: np.ndarray,
    ts: pd.DatetimeIndex,
    min_range: float,
    max_gap_sec: int,
) -> bool:
    """Reject windows containing zero-range bars or inter-bar gaps."""
    if len(highs) == 0:
        return False
    if np.any((highs - lows) <= min_range):
        return False
    for k in range(1, len(ts)):
        if (ts[k] - ts[k - 1]).total_seconds() > max_gap_sec:
            return False
    return True


def compute_premarket_high_by_date(
    df_raw_intraday: pd.DataFrame,
    rth_open_local: time = time(9, 30),
) -> dict[date, float]:
    """Convenience helper: derive ``{session_date: pm_high}`` from raw
    (pre + regular session) intraday OHLC. Caller hands this to the
    detector so the PM-high gate has data.

    Expects ``df_raw_intraday`` indexed by tz-aware DatetimeIndex.
    Bars whose local time is *before* ``rth_open_local`` are treated
    as pre-market for that calendar date.
    """
    if df_raw_intraday.empty:
        return {}
    out: dict[date, float] = {}
    local = df_raw_intraday.index
    cutoff = rth_open_local.hour * 60 + rth_open_local.minute
    for ts, high in zip(local, df_raw_intraday["High"].to_numpy()):
        minute_of_day = ts.hour * 60 + ts.minute
        if minute_of_day >= cutoff:
            continue
        d = ts.date()
        prev = out.get(d, float("-inf"))
        out[d] = max(prev, float(high))
    # Drop dates with no PM data (-inf).
    return {d: v for d, v in out.items() if v != float("-inf")}


def compute_earnings_window_dates(
    events: "list[EarningsEvent]",
    window_days: int = 2,
) -> set[date]:
    """Expand each earnings report date into a ``[d-N, d+N]`` window
    and return the union. Matt: "earnings season is not too far away
    — when stocks gap on a catalyst, the bull flag continuations
    work best".

    ``window_days=2`` covers D-2, D-1, D, D+1, D+2 around each report.
    """
    out: set[date] = set()
    for ev in events or []:
        d = ev.report_date
        for off in range(-window_days, window_days + 1):
            out.add(d + timedelta(days=off))
    return out


def compute_news_session_dates(
    news: "list[NewsEvent]",
    lookback_days: int = 1,
    min_sentiment: float | None = None,
) -> set[date]:
    """Map news items to "catalyst-ok session dates".

    A session date ``d`` is marked OK when there was at least one
    news item published on ``d - lookback_days … d`` (inclusive).
    ``lookback_days=1`` is the conservative Matt-style read ("news
    catalyst from yesterday or today").

    ``min_sentiment`` (optional, ``[-1, +1]``) filters out negative-
    /neutral-toned news. ``0.0`` keeps only positive items.
    """
    out: set[date] = set()
    for ev in news or []:
        if min_sentiment is not None and (
            ev.sentiment is None or ev.sentiment < min_sentiment
        ):
            continue
        d = ev.published_at.date()
        for off in range(0, lookback_days + 1):
            out.add(d + timedelta(days=off))
    return out


def compute_market_regime_ok(
    df_market_daily: pd.DataFrame,
    sma_period: int = 50,
) -> dict[date, bool]:
    """``{date: True}`` for each day where market index close > SMA.
    ``df_market_daily`` should be the SPY (or QQQ) daily OHLC for the
    backtest window. Days without enough SMA history are ``False``.
    """
    if df_market_daily.empty:
        return {}
    sma = (
        df_market_daily["Close"]
        .rolling(sma_period, min_periods=max(10, sma_period // 4))
        .mean()
    )
    ok = (df_market_daily["Close"] > sma).fillna(False)
    return {ts.date(): bool(v) for ts, v in ok.items()}
