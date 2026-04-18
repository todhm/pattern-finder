from datetime import date

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


class WedgepopStrategy(StrategyRunnerPort):
    """TraderLion 'Wedge Pop' (Oliver Kell - The Money Pattern) trading strategy.

    Self-contained strategy adapter: depends only on the strategy domain
    plus inbound ports (`MarketDataPort`, `PatternDetector`). Holds no
    reference to the backtest domain — execution, sizing, exits, and
    performance accounting all live inside the strategy boundary.

    All distance-based thresholds (exhaustion, entry extension, climax)
    are expressed in **ATR multiples** rather than fixed percentages.
    ATR automatically scales with each stock's volatility, so "2.5 ATR
    above EMA" represents a comparable visual distance on a quiet $500
    blue-chip and a jumpy $20 growth name — eliminating the "1% is huge
    on stock A but invisible on stock B" problem that plagued the
    old fixed-% gates.

    Lifecycle (one position at a time)

    Entry
        - The injected `PatternDetector` (expected to be a wedge-pop
          detector) signals the first close above both EMAs after a
          consolidation. Buy at the NEXT bar's OPEN — gives a clean fill
          and avoids paying the breakout candle's close.

    Initial stop ("logical pivot stop")
        - Just below the consolidation low (provided as `signal.stop_loss`).
          This is the level the doc recommends for failed-breakout protection.

    Position size
        - Fixed-fractional risk: ``shares = capital * risk_per_trade /
          (entry - stop)``. Caps downside at `risk_per_trade` per trade.

    Exit — two modes controlled by ``use_smart_trail``:

        **Smart Trail mode** (``use_smart_trail=True``):
            1. Hard stop     consolidation low → ``min(open, stop)``
            2. Smart trail   Chandelier exit: ``highest_high - N × ATR``
                             N widens with R-profit: <2R→3, 2-4R→4, >4R→5.
                             Arms after 3 bars. LOW-based (wick-proof).
            3. Time stop     held >= max_holding_days → exit at close.

        **Legacy mode** (``use_smart_trail=False``):
            1. Hard stop     consolidation low → ``min(open, stop)``
            2. Exhaustion    ``ref_ema + ATR × extension_atr_mult``
            3. Climax bar    ``prev_close + climax_atr_mult × ATR``
            4. EMA trail     close < fast EMA → exit at close
            5. Time stop     held >= max_holding_days → exit at close.

    **Pattern-based exit** (opt-in via ``exit_detector``)
        When any `PatternDetector` is injected, any bar inside the
        holding period that fires a signal from that detector triggers
        an immediate exit at that bar's close. Typical use: inject an
        `ExhaustionExtensionTopDetector` so that a long exits once the
        market prints a euphoric blow-off top. All conditions those
        detectors check (extensions, slopes, candle shapes, rolling
        volume ratios) are end-of-bar quantities, so exiting at the
        signal bar's close uses no future data — every input to the
        decision is known by the time the bar finishes. Signals are
        pre-computed once on the full dataframe in ``execute()``;
        pandas' EWM/rolling indicators are causal, so each signal's
        date maps to a bar whose detection used only data up to and
        including that bar. The entry bar itself is skipped
        regardless of whether it fires — a same-bar exit against our
        own entry is nonsensical.

    The strategy's exit logic is tied to the wedge-pop lifecycle (Wedge
    Pop -> EMA Crossback -> Base n' Break -> Exhaustion Extension), so the
    injected detector must be a wedge-pop detector for the rules to make
    sense. The dependency is typed against the port (`PatternDetector`)
    to keep the adapter free of concrete-adapter coupling; composition is
    the responsibility of the caller.
    """

    def __init__(
        self,
        market_data: MarketDataPort,
        detector: PatternDetector,
        ema_trail: int = 10,
        ema_slow: int = 20,
        atr_period: int = 14,
        extension_atr_mult: float = 2.5,
        climax_atr_mult: float = 1.5,
        max_entry_ema_extension_atr: float | None = None,
        max_ema_slope_decline: float | None = 0.01,
        min_ema_slow_slope: float | None = None,
        max_ema_slow_slope: float | None = None,
        require_gap_up: bool = False,
        use_smart_trail: bool = False,
        exit_detector: PatternDetector | None = None,
    ):
        self._market_data = market_data
        self._detector = detector
        self._exit_detector = exit_detector
        self.ema_trail = ema_trail
        self.ema_slow = ema_slow
        self.atr_period = atr_period
        self.extension_atr_mult = extension_atr_mult
        self.climax_atr_mult = climax_atr_mult
        self.max_entry_ema_extension_atr = max_entry_ema_extension_atr
        # Slope entry filter as a RANGE ``[min, max]``. The legacy
        # ``max_ema_slope_decline`` param still works as a
        # backwards-compatible shim: setting it to e.g. 0.01 is
        # equivalent to ``min_ema_slow_slope=-0.01``. The new
        # ``min_ema_slow_slope`` / ``max_ema_slow_slope`` take
        # precedence when both are set — the old shim only fills
        # in the min bound if it's ``None``. This lets users
        # require a POSITIVE slope (e.g. ``min=0.05`` → "ema20
        # must be rising by at least +5% over the slope_lookback
        # window"), not just reject steep declines.
        self.max_ema_slope_decline = max_ema_slope_decline
        if min_ema_slow_slope is None and max_ema_slope_decline is not None:
            min_ema_slow_slope = -max_ema_slope_decline
        self.min_ema_slow_slope = min_ema_slow_slope
        self.max_ema_slow_slope = max_ema_slow_slope
        self.require_gap_up = require_gap_up
        self.use_smart_trail = use_smart_trail

    # ---- public API ----

    def run(self, config: StrategyConfig) -> StrategyResult:
        """Fetch data via the market-data port and execute the strategy."""
        df = self._market_data.fetch_ohlcv(config.ticker, config.start_date, config.end_date)
        return self.execute(df, config)

    def execute(self, df: pd.DataFrame, config: StrategyConfig) -> StrategyResult:
        """Run the strategy on a pre-fetched OHLCV DataFrame.

        Useful when the caller already holds the market data (e.g. a UI
        page rendering both the chart and the strategy from a single
        fetch). The DataFrame must have the same shape as
        `MarketDataPort.fetch_ohlcv` returns.

        The DataFrame may contain bars before ``config.start_date``
        (warmup for indicator convergence — e.g. 200 extra bars so
        the 200 SMA is valid from day 1). Signals that fire during
        the warmup period are silently discarded.
        """
        df = self._with_indicators(df)
        signals = self._detector.detect(df)
        # Discard warmup-period signals outside the user's date range.
        signals = [
            s
            for s in signals
            if config.start_date <= s.date <= config.end_date
        ]

        # Pre-compute pattern-based exit dates once. Safe against
        # look-ahead: the detector's conditions (extensions, slopes,
        # candle shapes, rolling volume) are all causal — EWM/rolling
        # indicators only read past+current values — so each signal's
        # date corresponds to a bar that could have been detected at
        # its own close in real time.
        exit_dates: set[date] = set()
        if self._exit_detector is not None:
            exit_dates = {s.date for s in self._exit_detector.detect(df)}

        performance, equity_curve = self._run_signals(
            df, signals, config, exit_dates
        )

        return StrategyResult(
            config=config,
            performance=performance,
            equity_curve=equity_curve,
        )

    # ---- indicators ----

    def _with_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["ema_trail"] = df["Close"].ewm(span=self.ema_trail, adjust=False).mean()
        df["ema_slow"] = df["Close"].ewm(span=self.ema_slow, adjust=False).mean()
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

    # ---- execution loop ----

    def _run_signals(
        self,
        df: pd.DataFrame,
        signals: list[PatternSignal],
        config: StrategyConfig,
        exit_dates: set[date],
    ) -> tuple[StrategyPerformance, list[EquityPoint]]:
        capital = config.initial_capital
        peak = capital
        max_dd = 0.0
        trades: list[Trade] = []
        curve: list[EquityPoint] = [EquityPoint(date=config.start_date, equity=capital)]

        next_open_idx = 0  # blocks new entries while a position is open
        for signal in signals:
            entry_idx = self._next_open_index(df, signal.date)
            if entry_idx is None or entry_idx < next_open_idx:
                continue

            trade, exit_idx = self._execute_trade(
                df, signal, entry_idx, capital, config, exit_dates
            )
            if trade is None:
                continue

            capital += trade.pnl
            peak = max(peak, capital)
            max_dd = max(max_dd, (peak - capital) / peak if peak > 0 else 0.0)
            trades.append(trade)
            curve.append(EquityPoint(date=trade.exit_date, equity=round(capital, 2)))
            next_open_idx = exit_idx + 1

        performance = self._build_performance(config.initial_capital, capital, max_dd, trades)
        return performance, curve

    @staticmethod
    def _next_open_index(df: pd.DataFrame, signal_date) -> int | None:
        # `signal_date` is a plain `date`. yfinance returns a tz-aware
        # DatetimeIndex (America/New_York), so a naive Timestamp would
        # raise on comparison. Localize the Timestamp to the index tz
        # whenever the index has one.
        ts = pd.Timestamp(signal_date)
        if df.index.tz is not None and ts.tz is None:
            ts = ts.tz_localize(df.index.tz)
        after = df.index[df.index > ts]
        if len(after) == 0:
            return None
        return df.index.get_loc(after[0])

    def _execute_trade(
        self,
        df: pd.DataFrame,
        signal: PatternSignal,
        entry_idx: int,
        capital: float,
        config: StrategyConfig,
        exit_dates: set[date] | None = None,
    ) -> tuple[Trade | None, int]:
        if exit_dates is None:
            exit_dates = set()
        entry_price = float(df["Open"].iloc[entry_idx])

        # Reject entry when the open gaps BELOW both EMAs — if we're
        # buying a long and the open is already under the 10 EMA and
        # 20 EMA (computed at the signal bar, the last known values
        # at market open), the breakout has failed overnight and
        # entering is chasing a dead setup.
        if entry_idx > 0:
            sig_ema_fast = float(df["ema_trail"].iloc[entry_idx - 1])
            sig_ema_slow = float(df["ema_slow"].iloc[entry_idx - 1])
            if entry_price < sig_ema_fast and entry_price < sig_ema_slow:
                return None, entry_idx

        # Optional gap-up confirmation (TraderLion: "Most Wedge Pops that
        # start new trends often include unfilled gaps with strong volume").
        # If enabled, the next bar's open must clear the breakout bar's
        # close — otherwise the breakout failed to follow through and we
        # skip this signal entirely.
        if self.require_gap_up and entry_idx > 0:
            prev_close = float(df["Close"].iloc[entry_idx - 1])
            if entry_price <= prev_close:
                return None, entry_idx

        # "EMA-extension" entry filter (ATR-scaled): reject entries
        # where the entry-bar open is more than
        # ``max_entry_ema_extension_atr × ATR`` above
        # ``max(ema_fast, ema_slow)`` at the signal bar. Using ATR
        # as the yardstick means the same multiplier (e.g. 1.5)
        # works across high- and low-volatility regimes — no need
        # to retune a fixed % per stock. Setting to ``None``
        # disables the filter entirely.
        if (
            self.max_entry_ema_extension_atr is not None
            and entry_idx > 0
        ):
            sig_fast = float(df["ema_trail"].iloc[entry_idx - 1])
            sig_slow = float(df["ema_slow"].iloc[entry_idx - 1])
            ref_ema = max(sig_fast, sig_slow)
            atr_at_signal = float(df["atr"].iloc[entry_idx - 1])
            if ref_ema > 0 and atr_at_signal > 0:
                extension_atr = (entry_price - ref_ema) / atr_at_signal
                if extension_atr > self.max_entry_ema_extension_atr:
                    return None, entry_idx

        # Slope range filter: reject entries where the signal's
        # medium-term EMA slope (metadata variable set by the
        # detector) sits outside
        # ``[min_ema_slow_slope, max_ema_slow_slope]``. The MIN
        # bound catches dead-cat-bounce setups inside rolling-over
        # trends (e.g. AMD 2025-01-06 with ema20_slope20 = -10.5%)
        # OR can be used to require a confirmed uptrend (e.g.
        # ``min_ema_slow_slope = 0.05`` → "only take wedge pops
        # where the 20 EMA has risen at least 5% over the last N
        # bars"). The MAX bound filters out pops that are already
        # inside a parabolic move (e.g. cap at 0.30 to skip names
        # that are up 30%+ on the slope window).
        slow_slope = signal.metadata.get("ema_slow_slope")
        if slow_slope is not None:
            if (
                self.min_ema_slow_slope is not None
                and slow_slope < self.min_ema_slow_slope
            ):
                return None, entry_idx
            if (
                self.max_ema_slow_slope is not None
                and slow_slope > self.max_ema_slow_slope
            ):
                return None, entry_idx

        stop = float(signal.stop_loss)
        risk_per_share = entry_price - stop
        if risk_per_share <= 0:
            return None, entry_idx

        risk_amount = capital * config.risk_per_trade
        shares = max(1, int(risk_amount / risk_per_share))

        exit_price, exit_idx = self._find_exit(
            df,
            entry_idx,
            entry_price,
            stop,
            config.max_holding_days,
            exit_dates,
        )

        pnl = (exit_price - entry_price) * shares
        trade = Trade(
            pattern_name=signal.pattern_name,
            entry_date=df.index[entry_idx].date(),
            exit_date=df.index[exit_idx].date(),
            entry_price=round(entry_price, 2),
            exit_price=round(exit_price, 2),
            stop_loss=round(stop, 2),
            shares=shares,
            pnl=round(pnl, 2),
            pnl_pct=round((exit_price - entry_price) / entry_price, 4),
        )
        return trade, exit_idx

    def _find_exit(
        self,
        df: pd.DataFrame,
        entry_idx: int,
        entry_price: float,
        stop: float,
        max_holding_days: int,
        exit_dates: set[date] | None = None,
    ) -> tuple[float, int]:
        if exit_dates is None:
            exit_dates = set()
        last_idx = min(entry_idx + max_holding_days, len(df) - 1)

        # Smart trail state — Chandelier exit with profit-tier widening.
        highest_high = 0.0
        initial_risk = entry_price - stop  # 1R

        # The loop starts at ``entry_idx`` (NOT ``entry_idx + 1``) so
        # the entry bar itself is checked for same-day exits. A bar
        # that spikes intraday past the exhaustion line, or an
        # entry-day reversal that pierces the consolidation-low stop,
        # both fire on the very first iteration. Breakeven arming at
        # end-of-bar then correctly flows into the next bar's hard
        # stop check (AMD 2019-08-22 case is still handled — arming
        # happens at the end of the 08-22 iteration, before 08-23's
        # hard-stop check sees ``stop = entry_price``).
        for i in range(entry_idx, last_idx + 1):
            open_bar = float(df["Open"].iloc[i])
            high_bar = float(df["High"].iloc[i])
            low_bar = float(df["Low"].iloc[i])
            low = low_bar
            close = float(df["Close"].iloc[i])
            ema_fast = float(df["ema_trail"].iloc[i])
            ema_slow = float(df["ema_slow"].iloc[i])
            atr = float(df["atr"].iloc[i]) if not pd.isna(df["atr"].iloc[i]) else 0.0

            if high_bar > highest_high:
                highest_high = high_bar

            # 1. Hard stop — consolidation low.
            #    LOW-based limit-order model: intraday pierce fills
            #    at the stop line, gap-down fills at the open.
            if low <= stop:
                return min(open_bar, stop), i

            # 1b. Pattern-based exit — fires when the injected
            #     ``exit_detector`` signals on this bar. Typical use:
            #     ``ExhaustionExtensionTopDetector`` so that a long
            #     exits on a euphoric blow-off top. All conditions
            #     those detectors check are end-of-bar quantities,
            #     and the detection was pre-run in ``execute()`` with
            #     causal indicators, so firing on this bar's close
            #     uses no future data. Skip the entry bar itself —
            #     a same-bar exit against our own entry is nonsense.
            if i > entry_idx and df.index[i].date() in exit_dates:
                return close, i

            if self.use_smart_trail:
                # ── Smart Trail mode ──
                # Exit conditions: hard stop (above) + smart trail + time stop only.
                # Exhaustion / climax / EMA trail are all disabled.
                bars_held = i - entry_idx
                if bars_held >= 3 and atr > 0 and initial_risk > 0:
                    unrealized_r = (close - entry_price) / initial_risk
                    if unrealized_r >= 4.0:
                        eff_mult = 5.0
                    elif unrealized_r >= 2.0:
                        eff_mult = 4.0
                    else:
                        eff_mult = 3.0
                    trail_level = highest_high - eff_mult * atr
                    if trail_level > stop and low <= trail_level:
                        return min(open_bar, trail_level), i
            else:
                # ── Legacy mode ──
                # Exhaustion + climax + EMA trail all active.

                # 2. Exhaustion Extension — HIGH-based, ATR-scaled.
                ref_ema = max(ema_fast, ema_slow)
                if ref_ema > 0 and atr > 0:
                    trigger = ref_ema + atr * self.extension_atr_mult
                    if trigger > entry_price and high_bar >= trigger:
                        return max(open_bar, trigger), i

                # 3. Climax bar — HIGH-based limit-order model.
                if atr > 0 and i > 0:
                    prev_close = float(df["Close"].iloc[i - 1])
                    climax_line = prev_close + self.climax_atr_mult * atr
                    bar_range = high_bar - low_bar
                    if (
                        climax_line > entry_price
                        and bar_range > self.climax_atr_mult * atr
                        and high_bar >= climax_line
                    ):
                        return max(open_bar, climax_line), i

                # 4. EMA trail — close < fast EMA → exit at close.
                if close < ema_fast:
                    return close, i

        # 5. Time stop
        return float(df["Close"].iloc[last_idx]), last_idx

    # ---- performance builder ----

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
            total_return_pct=round((final_capital - initial_capital) / initial_capital, 4),
            total_trades=len(trades),
            win_rate=round(len(wins) / len(trades), 4) if trades else 0.0,
            avg_win_pct=round(sum(t.pnl_pct for t in wins) / len(wins), 4) if wins else 0.0,
            avg_loss_pct=round(sum(t.pnl_pct for t in losses) / len(losses), 4) if losses else 0.0,
            max_drawdown_pct=round(max_drawdown, 4),
            trades=trades,
        )
