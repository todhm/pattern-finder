from datetime import date

import numpy as np
import pandas as pd

from data.domain.ports import MarketDataPort
from pattern.domain.models import PatternSignal
from pattern.domain.ports import PatternDetector
from strategy.domain.models import (
    EquityPoint,
    StrategyConfig,
    StrategyPerformance,
    StrategyResult,
    Trade,
)
from strategy.domain.ports import StrategyRunnerPort


class WickPlayStrategy(StrategyRunnerPort):
    """Oliver Kell Wick Play strategy — minimal recipe.

    Entry
        - Injected ``detector`` (a ``WickPlayDetector``) fires on the
          breakout bar (Bar i).
        - Buy at the NEXT bar's OPEN. Avoids paying the breakout
          candle's close and gives a clean fill.

    Initial stop
        - ``signal.stop_loss`` — comes from the detector, which is
          expected to be configured with ``stop_mode="wick_low"``
          (Kell's "for more room, use the low of the wick as your
          stop level").

    Position size
        - Fixed-fractional risk: ``shares = capital * risk_per_trade
          / (entry - stop)``.

    Exits (OR — first to fire wins)
        1. **10 EMA trail** — once ``close < 10 EMA`` on any bar
           (from ``entry_bar + min_trail_bars``), exit at that
           bar's close. The "stairstep" trail Kell / Minervini use:
           while price keeps riding the 10 EMA, hold; when it
           snaps, exit.
        2. **Exhaustion Extension Top** (optional but ON by default)
           — a paired detector (typically
           ``ExhaustionExtensionTopDetector``) fires on the current
           bar → exit at that bar's close. The natural counter-signal
           to Wick Play, from the same Kell masterclass: Wick Play
           is the entry, Exhaustion Extension Top is the exit.
        3. **Breakeven stop** (OPT-IN, off by default) — once any
           close hits ``entry + arm_r × initial_risk``, the stop
           moves up to ``entry + offset_r × initial_risk``. A
           subsequent bar's low touching that level exits there.
           Guarded off by default because Wick Play's wick_low
           stop is wide (often 5–10%), so a naive arm at +1R /
           offset 0 lock exits winners during routine pullbacks
           (APA/MO/ARES 2024–25). Users who want loss-tail
           protection should pair ``arm_r ≥ 1.5`` with a positive
           ``offset_r`` (e.g. 0.5R) so the profit lock asymmetry
           matches the stop asymmetry.
        4. **Time stop** — after ``max_holding_days`` bars, exit at
           the next bar's open. Safety net so a flat / drifting
           trade doesn't occupy capital indefinitely.

    Initial stop is enforced as well: if any bar's low pierces the
    wick low, exit at the stop price (or the open on a gap-down).
    """

    def __init__(
        self,
        market_data: MarketDataPort,
        detector: PatternDetector,
        exit_detector: PatternDetector | None = None,
        ema_trail: int = 10,
        atr_period: int = 14,
        min_trail_bars: int = 2,
        enable_same_day_reversal_exit: bool = False,
        max_same_day_close_location: float = 0.3,
        enable_gap_down_rejection: bool = True,
        max_entry_gap_down: float = 0.005,
        enable_breakeven_stop: bool = False,
        breakeven_arm_r_multiple: float = 1.5,
        breakeven_exit_offset_r: float = 0.5,
    ):
        self.market_data = market_data
        self.detector = detector
        self.exit_detector = exit_detector
        self.ema_trail = ema_trail
        self.atr_period = atr_period
        # Don't let the trail fire on the entry bar itself or the bar
        # right after — gives the breakout room to digest before
        # the EMA catches up.
        self.min_trail_bars = min_trail_bars
        # Same-day reversal exit (opt-in). On the entry bar itself,
        # if the close lands in the bottom ``max_same_day_close_location``
        # fraction of its range, exit at close — the breakout that
        # lured us in has already reversed within the session
        # (PODD/AIZ/SBAC 2024 failure mode). Off by default: this
        # fires alongside the normal wick-low stop path, it doesn't
        # replace it, so the wider stop still gets first claim when
        # price actually pierces it intraday.
        self.enable_same_day_reversal_exit = enable_same_day_reversal_exit
        self.max_same_day_close_location = max_same_day_close_location
        # Gap-down rejection (opt-in). Entry bar's open gapping
        # below the breakout close by more than ``max_entry_gap_down``
        # (fraction of breakout close) means the setup broke down
        # overnight — skip the trade entirely (PODD 2024 -2.9% gap).
        # Off by default.
        self.enable_gap_down_rejection = enable_gap_down_rejection
        self.max_entry_gap_down = max_entry_gap_down
        # Breakeven stop — once close hits entry + arm_r × risk, move
        # the stop up to entry + offset_r × risk. offset=0 means pure
        # breakeven; small positive offsets lock in a tiny profit.
        # Fires BEFORE EMA/exhaustion exits so a post-1R pullback can't
        # be "rescued" into a worse fill by the trail.
        self.enable_breakeven_stop = enable_breakeven_stop
        self.breakeven_arm_r_multiple = breakeven_arm_r_multiple
        self.breakeven_exit_offset_r = breakeven_exit_offset_r

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    # Interval forwarded to ``MarketDataPort.fetch_ohlcv``. Daily is
    # the historical default; intraday subclasses override via class
    # attribute so ``run()`` pulls the right granularity without
    # overriding the method itself.
    _interval: str = "1d"

    def run(self, config: StrategyConfig) -> StrategyResult:
        df = self.market_data.fetch_ohlcv(
            config.ticker,
            config.start_date,
            config.end_date,
            interval=self._interval,
        )
        return self.execute(df, config)

    def execute(
        self, df: pd.DataFrame, config: StrategyConfig
    ) -> StrategyResult:
        df = self._with_indicators(df)
        signals = self.detector.detect(df)

        # Pre-compute exhaustion exit-bar identity keys once;
        # detector docstrings guarantee causal (end-of-bar) detection.
        # Keys go through ``_signal_key`` so intraday subclasses can
        # swap the identity granularity (date → timestamp).
        exit_keys: set[pd.Timestamp] = set()
        if self.exit_detector is not None:
            exit_keys = {self._signal_key(s) for s in self.exit_detector.detect(df)}

        capital = config.initial_capital
        trades: list[Trade] = []
        equity_points: list[EquityPoint] = [
            EquityPoint(date=df.index[0].date(), equity=capital)
        ]
        position_until: int = -1  # one-position-at-a-time

        for sig in signals:
            entry_idx = self._next_open_index(df, sig)
            if entry_idx is None:
                continue
            if entry_idx <= position_until:
                continue  # still in a prior trade

            trade, exit_idx = self._execute_trade(
                df=df,
                signal=sig,
                entry_idx=entry_idx,
                capital=capital,
                config=config,
                exit_keys=exit_keys,
            )
            if trade is None:
                continue

            capital += trade.pnl
            trades.append(trade)
            # Block new entries until the day after this trade exits.
            position_until = int(exit_idx)
            equity_points.append(
                EquityPoint(date=trade.exit_date, equity=capital)
            )

        performance = self._build_performance(
            config.initial_capital, capital, trades
        )
        return StrategyResult(
            config=config,
            performance=performance,
            equity_curve=equity_points,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    # Daily path drops index tz to match pre-existing naive-timestamp
    # semantics in the rest of the daily pipeline (trade dates, chart
    # builders). Intraday subclasses override to keep the NY tz so
    # ``_next_open_index`` can match ``signal.timestamp`` against the
    # index without a tz-mismatch.
    _strip_index_tz: bool = True

    def _with_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if self._strip_index_tz and df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df["ema_trail"] = (
            df["Close"].ewm(span=self.ema_trail, adjust=False).mean()
        )
        df["atr"] = self._atr(df, self.atr_period)
        return df

    @staticmethod
    def _atr(df: pd.DataFrame, period: int) -> pd.Series:
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

    # ---- interval-identity hooks ----
    # Default implementations preserve the historical 1d behavior
    # bit-for-bit (keys = session date midnight). Intraday subclasses
    # override these to key by the exact ``DatetimeIndex`` value so
    # same-session 15m bars stay distinct.

    @staticmethod
    def _signal_match_ts(signal: PatternSignal) -> pd.Timestamp:
        return pd.Timestamp(signal.date)

    @staticmethod
    def _signal_key(signal: PatternSignal) -> pd.Timestamp:
        return pd.Timestamp(signal.date)

    @staticmethod
    def _bar_key(df: pd.DataFrame, i: int) -> pd.Timestamp:
        return pd.Timestamp(df.index[i].date())

    @staticmethod
    def _trade_exit_key(trade: Trade) -> pd.Timestamp:
        return pd.Timestamp(trade.exit_date)

    def _next_open_index(
        self, df: pd.DataFrame, signal: PatternSignal
    ) -> int | None:
        ts = self._signal_match_ts(signal)
        if df.index.tz is not None and ts.tz is None:
            ts = ts.tz_localize(df.index.tz)
        try:
            signal_idx = df.index.get_loc(ts)
        except KeyError:
            return None
        next_idx = signal_idx + 1
        if next_idx >= len(df):
            return None
        return next_idx

    def _execute_trade(
        self,
        df: pd.DataFrame,
        signal: PatternSignal,
        entry_idx: int,
        capital: float,
        config: StrategyConfig,
        exit_keys: set[pd.Timestamp] | None = None,
    ) -> tuple[Trade | None, int]:
        if exit_keys is None:
            exit_keys = set()
        entry_price = float(df["Open"].iloc[entry_idx])
        stop = float(signal.stop_loss)

        # Gap-down rejection (opt-in, ⑨). Entry-bar open sitting
        # below the breakout close by more than ``max_entry_gap_down``
        # means the setup broke overnight — skip.
        if self.enable_gap_down_rejection and entry_idx > 0:
            breakout_close = float(df["Close"].iloc[entry_idx - 1])
            if breakout_close > 0:
                gap = (entry_price - breakout_close) / breakout_close
                if gap < -self.max_entry_gap_down:
                    return None, entry_idx

        risk_per_share = entry_price - stop
        if risk_per_share <= 0:
            return None, entry_idx

        shares = int((capital * config.risk_per_trade) // risk_per_share)
        if shares <= 0:
            return None, entry_idx

        exit_price, exit_idx, exit_reason = self._find_exit(
            df=df,
            entry_idx=entry_idx,
            entry_price=entry_price,
            stop=stop,
            max_holding_days=config.max_holding_days,
            exit_keys=exit_keys,
        )

        pnl = (exit_price - entry_price) * shares
        pnl_pct = (exit_price - entry_price) / entry_price

        entry_ts = pd.Timestamp(df.index[entry_idx]).to_pydatetime()
        exit_ts = pd.Timestamp(df.index[exit_idx]).to_pydatetime()
        trade = Trade(
            pattern_name=signal.pattern_name,
            entry_date=df.index[entry_idx].date(),
            exit_date=df.index[exit_idx].date(),
            entry_price=entry_price,
            exit_price=exit_price,
            stop_loss=stop,
            shares=shares,
            pnl=pnl,
            pnl_pct=pnl_pct,
            exit_reason=exit_reason,
            entry_ts=entry_ts,
            exit_ts=exit_ts,
        )
        return trade, exit_idx

    def _find_exit(
        self,
        df: pd.DataFrame,
        entry_idx: int,
        entry_price: float,
        stop: float,
        max_holding_days: int,
        exit_keys: set[pd.Timestamp],
    ) -> tuple[float, int, str]:
        last_idx = len(df) - 1
        opens = df["Open"].to_numpy(dtype=float)
        highs = df["High"].to_numpy(dtype=float)
        lows = df["Low"].to_numpy(dtype=float)
        closes = df["Close"].to_numpy(dtype=float)
        ema = df["ema_trail"].to_numpy(dtype=float)
        ema = np.where(np.isnan(ema), 0.0, ema)

        deadline = entry_idx + max_holding_days
        initial_risk = entry_price - stop
        breakeven_armed = False
        breakeven_exit_price = (
            entry_price + self.breakeven_exit_offset_r * initial_risk
        )

        for i in range(entry_idx, last_idx + 1):
            # (A) Hard stop on wick low — pierce on intraday low.
            # Fires even on the entry bar itself; a gap-down that
            # blows through the stop exits at the open.
            if lows[i] <= stop:
                fill = min(float(opens[i]), stop)
                return fill, i, "wick_low_stop"

            # (A') Same-day reversal exit (opt-in). Only fires on
            # the entry bar itself: if the bar closes in the bottom
            # ``max_same_day_close_location`` fraction of its range,
            # the breakout has already unraveled intraday, so exit
            # at close rather than wait for the wick-low stop or
            # next-day trail to catch up.
            if (
                i == entry_idx
                and self.enable_same_day_reversal_exit
            ):
                rng = highs[i] - lows[i]
                if rng > 0:
                    close_loc = (closes[i] - lows[i]) / rng
                    if close_loc < self.max_same_day_close_location:
                        return float(closes[i]), i, "same_day_reversal"

            # (B) Breakeven stop — once a prior bar's close reached
            # entry + arm_r × risk, a subsequent low touching the
            # breakeven level exits at that price (gap-downs fill
            # at the open). Same-bar arm + pierce cannot fire:
            # ``breakeven_armed`` uses last bar's state when we
            # check the pierce, and only flips after the check.
            if self.enable_breakeven_stop and initial_risk > 0:
                if (
                    breakeven_armed
                    and i > entry_idx
                    and lows[i] <= breakeven_exit_price
                ):
                    fill = min(float(opens[i]), breakeven_exit_price)
                    return fill, i, "breakeven_stop"
                if (
                    not breakeven_armed
                    and closes[i]
                    >= entry_price
                    + self.breakeven_arm_r_multiple * initial_risk
                ):
                    breakeven_armed = True

            # The follow-through exits below wait at least
            # ``min_trail_bars`` after entry so the breakout has
            # room to breathe.
            if i - entry_idx < self.min_trail_bars:
                continue

            # (C) Exhaustion Extension Top — inject from composition
            # root. Exit at the bar's close (the fire bar itself).
            # Bar-identity uses ``_bar_key`` so intraday subclasses
            # can swap the granularity (date → timestamp) without
            # rewriting this loop.
            if (
                self.exit_detector is not None
                and self._bar_key(df, i) in exit_keys
            ):
                return float(closes[i]), i, "exhaustion_exit"

            # (D) EMA trail — Kell / Minervini stairstep. Close
            # below EMA snaps the trail; exit at the close of that
            # bar.
            if ema[i] > 0 and closes[i] < ema[i]:
                return float(closes[i]), i, "ema_trail"

            # (E) Time stop — exit at next bar's open.
            if i >= deadline:
                if i + 1 <= last_idx:
                    return float(opens[i + 1]), i + 1, "time_stop"
                return float(closes[i]), i, "time_stop"

        # End-of-data fallback — held to the last bar.
        return float(closes[last_idx]), last_idx, "end_of_data"

    def _build_performance(
        self,
        initial_capital: float,
        final_capital: float,
        trades: list[Trade],
    ) -> StrategyPerformance:
        n = len(trades)
        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]
        win_rate = len(wins) / n if n else 0.0
        avg_win = (
            sum(t.pnl_pct for t in wins) / len(wins) if wins else 0.0
        )
        avg_loss = (
            sum(t.pnl_pct for t in losses) / len(losses) if losses else 0.0
        )

        # Max drawdown from cumulative equity.
        equity = [initial_capital]
        for t in trades:
            equity.append(equity[-1] + t.pnl)
        peak = equity[0]
        max_dd = 0.0
        for eq in equity:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak if peak > 0 else 0.0
            if dd > max_dd:
                max_dd = dd

        total_return = (
            (final_capital - initial_capital) / initial_capital
            if initial_capital > 0
            else 0.0
        )
        return StrategyPerformance(
            initial_capital=initial_capital,
            final_capital=final_capital,
            total_return_pct=total_return,
            total_trades=n,
            win_rate=win_rate,
            avg_win_pct=avg_win,
            avg_loss_pct=avg_loss,
            max_drawdown_pct=max_dd,
            trades=trades,
        )
