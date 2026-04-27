"""Fair Value Gap (FVG) trading strategy.

Self-contained intraday strategy designed for the structural setup
emitted by :class:`FairValueGapDetector`. Lifecycle (one position at
a time, intraday-style):

Entry
    Limit-buy fills at ``signal.entry_price`` (= ``fvg_mid``) on the
    retest bar — by detector contract that bar's low has touched the
    midpoint and its close is back above it. Mechanically the resting
    limit at ``fvg_mid`` would have executed when the bar's low first
    reached that level. Using the bar's *close* (well above mid)
    would inflate the entry price and break the framework's R math.

Initial stop
    ``signal.stop_loss`` — the FVG-producing (middle) candle's low.
    A LOW pierce of this level exits at the stop (gap-down → open).

Take profit
    Bar HIGH reaching ``entry + take_profit_r_multiple × initial_risk``
    fires the exit at the target level (not the bar's high). Default
    ``3.0`` matches the framework's "1:3-4 TP" guidance — the upper
    end (``4.0``) is exposed as a knob for users who want to bias
    toward fewer winners with bigger banks.

Same-bar TP + stop ordering
    On a bar where ``high >= target`` AND ``low <= stop`` (no gap),
    OHLC tells us nothing about which level was hit first within the
    bar. We use a distance-from-open heuristic: whichever level is
    closer to the open in price is more likely to have been hit
    first. Ties resolve to stop-first (conservative).

Optional asymmetric protections (off by default unless toggled):

- **Break-even stop** — once any close hits ``entry + 1R``, a
  subsequent low touching entry exits at zero P&L. Kills the
  "almost made it" attrition where +1R winners pull back through
  entry to the initial stop.
- **BOS trail** — when close exceeds the FIRST post-entry swing
  high (the rally peak in the framework's step-4 picture), the stop
  ratchets up to the FVG midpoint. The reference is *frozen* at the
  first confirmed 2/2 fractal high; later (higher) peaks don't
  promote it. ETH bars are excluded — after-hours liquidity prints
  off-chart spikes that would render the gate unreachable. NOT
  ``close > choch_high`` (that was the pre-entry break level, not
  the framework's step-4 BOS).

Exit reason codes (added by this strategy):
    ``take_profit``      — TP target reached
    ``initial_stop``     — bar low touched the FVG-producing low
    ``breakeven_stop``   — armed at +1R, pulled back to entry
    ``bos_trail_stop``   — BOS happened, then bar low pierced FVG mid
    ``session_close``    — last RTH bar of the session, still flat-
                           less, close at that bar's close price
    ``end_of_data``      — window expired with no exit (only fires
                           when ``force_close_at_session_end`` is OFF
                           or there's no RTH bar after entry)

ETH stop gate (``disable_stops_outside_rth``, default ON)
    Bars outside 09:30–16:00 NY local are allowed to fire take-profit
    but **not** stop exits. After-hours liquidity is thin enough that
    spike-and-revert wicks routinely "stop out" the OHLC backtest at
    a depth that wouldn't fill in practice; deferring stops to the
    next RTH open produces fills that match what the user would
    actually see live.

Session-end forced close (``force_close_at_session_end``, default ON)
    The framework is intraday by design — most live FVG traders flat
    the position at the cash close rather than carry overnight gap
    risk. With this on, any trade still open on the last RTH bar of
    its NY session (e.g., the 15:45 15m bar or the 15:59 1m bar) is
    closed at that bar's CLOSE price, regardless of whether TP / stop
    fired. Off → trade can run across multiple sessions until TP /
    stop / max-holding-bars / end-of-data resolves it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data.domain.ports import MarketDataPort
from pattern.domain.models import PatternSignal
from pattern.domain.ports import PatternDetector
from pattern.helpers.pivots import find_swing_highs
from strategy.domain.models import (
    EquityPoint,
    StrategyConfig,
    StrategyPerformance,
    StrategyResult,
    Trade,
)
from strategy.domain.ports import StrategyRunnerPort


class FairValueGapStrategy(StrategyRunnerPort):
    """Single-ticker FVG strategy with fixed-R exit envelope."""

    # Forwarded to ``MarketDataPort.fetch_ohlcv``. 15m is the default
    # because yfinance caps 1m intraday at 7 calendar days, which is
    # too tight to backtest meaningfully — the user can switch to
    # ``"1m"`` (the framework's stated time-frame) for live signal
    # scanning, or to ``"5m"`` / ``"30m"`` for a less noisy variant.
    _interval: str = "15m"

    def __init__(
        self,
        market_data: MarketDataPort,
        detector: PatternDetector,
        take_profit_r_multiple: float = 3.0,
        enable_breakeven_stop: bool = False,
        breakeven_arm_r_multiple: float = 1.0,
        breakeven_exit_offset_r: float = 0.0,
        enable_bos_trail: bool = True,
        disable_stops_outside_rth: bool = True,
        force_close_at_session_end: bool = True,
    ) -> None:
        self._market_data = market_data
        self._detector = detector
        self.take_profit_r_multiple = take_profit_r_multiple
        self.enable_breakeven_stop = enable_breakeven_stop
        self.breakeven_arm_r_multiple = breakeven_arm_r_multiple
        self.breakeven_exit_offset_r = breakeven_exit_offset_r
        self.enable_bos_trail = enable_bos_trail
        # When True, ETH bars (pre/post-market) cannot fire stop
        # exits (initial / BE / BOS trail). Thin after-hours prints
        # regularly stop-out by spike-then-revert and the user ends
        # up with worse fills than letting the position breathe to
        # the next RTH open.
        self.disable_stops_outside_rth = disable_stops_outside_rth
        # Session-bound day-trading semantics. When True (default),
        # any position still open at the last RTH bar of the session
        # closes at that bar's CLOSE price ("session_close" exit
        # reason). Matches how most ICT/SMC FVG traders run the
        # framework — they don't carry the structural thesis past
        # the cash close because pre/post liquidity invalidates the
        # OHLC backtest fills and overnight gap risk is unrelated
        # to the intraday setup.
        self.force_close_at_session_end = force_close_at_session_end

    # ---- public API -----------------------------------------------------

    def run(self, config: StrategyConfig) -> StrategyResult:
        df = self._market_data.fetch_ohlcv(
            config.ticker,
            config.start_date,
            config.end_date,
            interval=self._interval,
        )
        return self.execute(df, config)

    def execute(self, df: pd.DataFrame, config: StrategyConfig) -> StrategyResult:
        if df.empty:
            return StrategyResult(
                config=config,
                performance=self._build_performance(
                    config.initial_capital, config.initial_capital, 0.0, []
                ),
                equity_curve=[
                    EquityPoint(date=config.start_date, equity=config.initial_capital)
                ],
            )

        signals = self._detector.detect(df)
        signals = [
            s
            for s in signals
            if config.start_date <= s.date <= config.end_date
        ]

        capital = config.initial_capital
        peak = capital
        max_dd = 0.0
        trades: list[Trade] = []
        curve: list[EquityPoint] = [
            EquityPoint(date=config.start_date, equity=capital)
        ]

        next_open_idx = 0
        for sig in signals:
            entry_idx = self._signal_bar_index(df, sig)
            if entry_idx is None or entry_idx < next_open_idx:
                continue

            trade, exit_idx = self._execute_trade(
                df, sig, entry_idx, capital, config
            )
            if trade is None:
                continue

            capital += trade.pnl
            peak = max(peak, capital)
            max_dd = max(max_dd, (peak - capital) / peak if peak > 0 else 0.0)
            trades.append(trade)
            curve.append(
                EquityPoint(date=trade.exit_date, equity=round(capital, 2))
            )
            next_open_idx = exit_idx + 1

        performance = self._build_performance(
            config.initial_capital, capital, max_dd, trades
        )
        return StrategyResult(
            config=config, performance=performance, equity_curve=curve
        )

    # ---- entry / sizing -------------------------------------------------

    @staticmethod
    def _signal_bar_index(df: pd.DataFrame, signal: PatternSignal) -> int | None:
        """Locate the FVG retest bar (the signal bar itself).

        With the limit-fill-at-fvg_mid entry convention, the signal
        bar IS the entry bar — the resting limit would have executed
        the moment that bar's low reached ``fvg_mid``. Falls back to
        a date-based lookup for hand-built daily test fixtures.
        """
        ts = (
            pd.Timestamp(signal.timestamp)
            if signal.timestamp is not None
            else pd.Timestamp(signal.date)
        )
        if df.index.tz is not None and ts.tz is None:
            ts = ts.tz_localize(df.index.tz)
        elif df.index.tz is None and ts.tz is not None:
            ts = ts.tz_localize(None)
        # Exact match first (intraday); fall back to first bar on
        # the same date (daily-style fixtures).
        try:
            return df.index.get_loc(ts)
        except KeyError:
            on_date = df.index[df.index.date == ts.date()]
            if len(on_date) == 0:
                return None
            return df.index.get_loc(on_date[0])

    def _execute_trade(
        self,
        df: pd.DataFrame,
        signal: PatternSignal,
        entry_idx: int,
        capital: float,
        config: StrategyConfig,
    ) -> tuple[Trade | None, int]:
        # Limit fill at fvg_mid. Detector-side contract: on the
        # signal bar (``entry_idx``) the bar's low has already touched
        # ``signal.entry_price`` so the resting limit executed.
        entry_price = float(signal.entry_price)
        stop = float(signal.stop_loss)
        risk_per_share = entry_price - stop
        if risk_per_share <= 0:
            return None, entry_idx

        risk_amount = capital * config.risk_per_trade
        shares = max(1, int(risk_amount / risk_per_share))
        # No-leverage cap — same rule WedgepopStrategy enforces, so a
        # tight stop on a high-priced ticker can't size into 3-5×
        # capital and turn a 1R move into a -3R real-world loss.
        max_shares_by_capital = int(capital / entry_price)
        if max_shares_by_capital < 1:
            return None, entry_idx
        shares = min(shares, max_shares_by_capital)

        exit_price, exit_idx, exit_reason = self._find_exit(
            df, entry_idx, entry_price, stop, signal, config.max_holding_days
        )

        pnl = (exit_price - entry_price) * shares
        entry_ts = pd.Timestamp(df.index[entry_idx]).to_pydatetime()
        exit_ts = pd.Timestamp(df.index[exit_idx]).to_pydatetime()
        return (
            Trade(
                pattern_name=signal.pattern_name,
                entry_date=df.index[entry_idx].date(),
                exit_date=df.index[exit_idx].date(),
                entry_price=round(entry_price, 4),
                exit_price=round(exit_price, 4),
                stop_loss=round(stop, 4),
                shares=shares,
                pnl=round(pnl, 2),
                pnl_pct=round((exit_price - entry_price) / entry_price, 4),
                exit_reason=exit_reason,
                entry_ts=entry_ts,
                exit_ts=exit_ts,
            ),
            exit_idx,
        )

    # ---- exit walk ------------------------------------------------------

    def _find_exit(
        self,
        df: pd.DataFrame,
        entry_idx: int,
        entry_price: float,
        stop: float,
        signal: PatternSignal,
        max_holding_days: int,
    ) -> tuple[float, int, str]:
        last_idx = len(df) - 1
        deadline = min(entry_idx + max_holding_days, last_idx)

        opens = df["Open"].to_numpy(dtype=float)
        highs = df["High"].to_numpy(dtype=float)
        lows = df["Low"].to_numpy(dtype=float)
        closes = df["Close"].to_numpy(dtype=float)
        # Pre-compute swing highs once for the whole frame; the BOS
        # trail uses the FIRST post-entry swing high as its trigger
        # level. 1/1 fractal (each side's neighbor strictly lower)
        # matches the visualization helper — relaxed from the
        # detector's 2/2 because on 1m bars the strict version
        # routinely overrides obvious chart peaks two bars later
        # with marginally-higher prints, anchoring the trail at the
        # wrong level.
        #
        # ETH gate: bars whose START is outside 09:30–16:00 NY local
        # are excluded from BOS swing-high tracking. Post-market
        # prints on thin liquidity routinely spike above the real
        # RTH peak (TSLA 2026-04-24 case: 17:50 ETH high at $391 vs
        # the visible RTH peak at $382). Letting them anchor the
        # trail level would make the BOS trail effectively never
        # fire and the chart line render off-screen.
        _SWING_RIGHT = 1
        swing_high_arr = find_swing_highs(
            df, left=1, right=_SWING_RIGHT
        ).to_numpy()
        rth_bar_mask = self._rth_mask(df)
        # Pre-compute the index of the LAST RTH bar in each NY
        # session — that's where ``force_close_at_session_end`` fires
        # if the position is still open. Doing this once outside the
        # loop avoids an O(n²) per-bar scan; ``last_rth_idx`` is a
        # set so the per-bar check is O(1).
        last_rth_idx: set[int] = set()
        if self.force_close_at_session_end:
            session_dates_arr = self._session_dates(df)
            seen: dict[object, int] = {}
            for j in range(len(df)):
                if rth_bar_mask[j]:
                    seen[session_dates_arr[j]] = j
            last_rth_idx = set(seen.values())

        initial_risk = entry_price - stop
        target = entry_price + self.take_profit_r_multiple * initial_risk
        breakeven_armed = False
        bos_trail_active = False
        bos_line: float | None = None  # running max of post-entry swing highs
        breakeven_exit_price = (
            entry_price + self.breakeven_exit_offset_r * initial_risk
        )
        fvg_mid = float(signal.metadata.get("fvg_mid", stop))

        for i in range(entry_idx, deadline + 1):
            open_bar = float(opens[i])
            high_bar = float(highs[i])
            low_bar = float(lows[i])
            close_bar = float(closes[i])

            # ETH stop gate — when on (default), stop exits fire only
            # on RTH bars. After-hours spikes look like stop-outs on
            # OHLC but rarely fill that deep in practice; deferring
            # to the next RTH open is the realistic read.
            is_rth_bar = bool(rth_bar_mask[i])
            stops_armed_for_bar = (
                is_rth_bar or not self.disable_stops_outside_rth
            )

            # (T+S) Take profit / hard initial stop — joint resolution
            #     so a bar that touches both gets the *closer-to-open*
            #     side instead of an ordering accident. Gap opens
            #     (open already past a level) settle first; if the
            #     bar then prints both within its range, distance
            #     from open is the heuristic. Stop is suppressed once
            #     break-even arms or BOS trail activates — those have
            #     their own floors below.
            stop_active = (
                i > entry_idx
                and not breakeven_armed
                and not bos_trail_active
                and stops_armed_for_bar
            )
            if i > entry_idx and initial_risk > 0:
                # Gaps first: an open above target gives an immediate
                # TP fill; an open below stop gives an immediate stop
                # fill. If the open already cleared *both* (huge gap
                # + reversal), tie-break on stop-first as the
                # conservative read.
                gap_tp = open_bar >= target
                gap_stop = stop_active and open_bar <= stop
                if gap_tp and gap_stop:
                    return open_bar, i, "initial_stop"
                if gap_tp:
                    return open_bar, i, "take_profit"
                if gap_stop:
                    return open_bar, i, "initial_stop"

                hit_tp = high_bar >= target
                hit_stop = stop_active and low_bar <= stop
                if hit_tp and hit_stop:
                    # Distance heuristic: smaller open-to-level move
                    # is the one likely traversed first within the
                    # bar. Tie → stop wins.
                    if (open_bar - stop) <= (target - open_bar):
                        return stop, i, "initial_stop"
                    return target, i, "take_profit"
                if hit_tp:
                    return target, i, "take_profit"
                if hit_stop:
                    return stop, i, "initial_stop"

            # (BOS) Trail — framework definition: after entry, the
            #     first new swing high formed by the rally is the
            #     "BOS reference". When a later close pierces that
            #     reference (or any higher peak that printed since),
            #     structural continuation is confirmed and the stop
            #     ratchets up to the FVG midpoint. Implementation:
            #     update ``bos_line`` whenever a confirmed post-entry
            #     swing high prints (running max), then trigger the
            #     trail on the first close above ``bos_line``. NOTE:
            #     earlier versions used ``close > choch_high`` here,
            #     which broke the framework — choch_high is the
            #     pre-entry break level, not the post-entry rally
            #     peak the framework actually means.
            confirm_idx = i - _SWING_RIGHT
            if (
                bos_line is None  # freeze on first peak — see chart helper
                and confirm_idx > entry_idx
                and rth_bar_mask[confirm_idx]
            ):
                sh_val = swing_high_arr[confirm_idx]
                if not np.isnan(sh_val):
                    bos_line = float(sh_val)
            if (
                self.enable_bos_trail
                and not bos_trail_active
                and bos_line is not None
                and close_bar > bos_line
                and i > entry_idx
            ):
                bos_trail_active = True
            if bos_trail_active and i > entry_idx and stops_armed_for_bar:
                if open_bar <= fvg_mid:
                    return open_bar, i, "bos_trail_stop"
                if low_bar <= fvg_mid:
                    return fvg_mid, i, "bos_trail_stop"

            # (B) Break-even — armed when any close clears +arm×R;
            #     a subsequent low touching the entry (+ optional
            #     offset R) exits at that level. Same-bar arm +
            #     pierce can't fire because the arm uses THIS bar's
            #     close while the pierce check uses LAST bar's
            #     ``breakeven_armed`` state.
            if self.enable_breakeven_stop:
                if (
                    breakeven_armed
                    and i > entry_idx
                    and low_bar <= breakeven_exit_price
                    and stops_armed_for_bar
                ):
                    return (
                        min(open_bar, breakeven_exit_price),
                        i,
                        "breakeven_stop",
                    )
                if (
                    not breakeven_armed
                    and initial_risk > 0
                    and close_bar
                    >= entry_price + self.breakeven_arm_r_multiple * initial_risk
                ):
                    breakeven_armed = True

            # (SC) Session close — last RTH bar of the NY session and
            #     the trade is still open. Forces a flat close at the
            #     bar's CLOSE price, matching how most ICT/SMC FVG
            #     traders run the framework (intraday day-trade,
            #     never carry overnight). Runs LAST so TP / stop /
            #     BE / BOS trail still get a chance on the same bar.
            if (
                self.force_close_at_session_end
                and i > entry_idx
                and i in last_rth_idx
            ):
                return close_bar, i, "session_close"

        # No exit fired within the holding window — close at deadline.
        return float(closes[deadline]), deadline, "end_of_data"

    @staticmethod
    def _session_dates(df: pd.DataFrame) -> np.ndarray:
        """Per-bar NY calendar date — the session-boundary key the
        ``force_close_at_session_end`` rule scans against. Tz-aware
        intraday frames convert to NY local first; tz-naive daily
        frames return their existing date directly."""
        idx = df.index
        if hasattr(idx, "tz") and idx.tz is not None:
            return np.array(
                [ts.tz_convert("America/New_York").date() for ts in idx]
            )
        return np.array([ts.date() for ts in idx])

    @staticmethod
    def _rth_mask(df: pd.DataFrame) -> np.ndarray:
        """Per-bar RTH boolean. ``True`` when bar START falls in
        09:30 ≤ t < 16:00 NY local time. Daily frames (every bar at
        midnight) collapse to all-True so the strategy's BOS gate
        is a no-op there."""
        from datetime import time as _t

        idx = df.index
        if hasattr(idx, "tz") and idx.tz is not None:
            local = idx.tz_convert("America/New_York")
        else:
            local = idx
        times = [ts.time() for ts in local]
        if all(t == _t(0, 0) for t in times):
            return np.ones(len(df), dtype=bool)
        rth_open = _t(9, 30)
        rth_close = _t(16, 0)
        return np.array(
            [(rth_open <= t < rth_close) for t in times], dtype=bool
        )

    # ---- performance ----------------------------------------------------

    @staticmethod
    def _build_performance(
        initial_capital: float,
        final_capital: float,
        max_drawdown: float,
        trades: list[Trade],
    ) -> StrategyPerformance:
        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]
        return StrategyPerformance(
            initial_capital=initial_capital,
            final_capital=round(final_capital, 2),
            total_return_pct=round(
                (final_capital - initial_capital) / initial_capital, 4
            )
            if initial_capital > 0
            else 0.0,
            total_trades=len(trades),
            win_rate=round(len(wins) / len(trades), 4) if trades else 0.0,
            avg_win_pct=round(
                sum(t.pnl_pct for t in wins) / len(wins), 4
            )
            if wins
            else 0.0,
            avg_loss_pct=round(
                sum(t.pnl_pct for t in losses) / len(losses), 4
            )
            if losses
            else 0.0,
            max_drawdown_pct=round(max_drawdown, 4),
            trades=trades,
        )
