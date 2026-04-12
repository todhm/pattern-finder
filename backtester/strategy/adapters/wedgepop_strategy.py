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

    Entry filter (optional)
        ``max_entry_chase_ratio`` rejects "chasing" entries where the
        next-bar open has already pushed above the signal bar's high
        by more than ``ratio × signal_bar_range``. Structural reason:
        if the entry fill is already that far above the breakout
        bar's top, the breakout has spent its move and we're buying
        at an exhausted level. Real example — AMD 2020-06-09: signal
        bar closes $56.39 with high $56.46, next bar opens $57.20,
        20% of the signal bar's range above its high. The filter
        rejects that signal instead of chasing a -14% loss.

    Exit (whichever fires first after entry)
        1. Hard stop:    LOW-based limit-order model, symmetric to
                         the exhaustion exit below. The bar's Low
                         touching the stop line fills at
                         ``min(open, stop)`` — a normal intraday
                         pierce fills at the stop; a gap-down that
                         opens below the stop fills at the open
                         (realistic slippage). The stop starts at
                         the consolidation low (``signal.stop_loss``).
                         When ``arm_breakeven_after_profit=True``
                         and the trade has closed above
                         ``entry_price`` at least once, the stop
                         **ratchets up to ``entry_price``
                         (breakeven)** and stays there for the rest
                         of the trade. This is the stateful 손절
                         mechanism: a trade that briefly went green
                         and then collapsed exits at breakeven
                         instead of giving back the entire
                         unrealized gain. Real example — CDNS
                         2026-02-11: close > entry on 02-18, then
                         collapsed; the breakeven stop catches it
                         on 02-19 instead of riding to a -7% loss.
        2. Exhaustion:   HIGH-based limit-order model. Each bar two
                         trigger lines are computed from the reference
                         EMA = ``max(ema_fast, ema_slow)``:
                             pct_line = ref × (1 + extension_pct)
                             atr_line = ref + ATR(14) × extension_atr_mult
                         The trade has an implicit sell limit at the
                         LOWER of the two lines — that's the one the
                         price reaches first as the rally stretches.
                         As soon as the bar's ``High`` touches the
                         line, the order fills **at the line**, not at
                         the high. A bar that gaps through the line
                         fills at ``max(open, line)`` (the gap favours
                         the seller). Both triggers must sit above
                         ``entry_price`` to count — otherwise the line
                         would be a loss-taking level, not a
                         profit-take.
        3. Climax bar:   HIGH-based limit-order model anchored at
                         yesterday's close. Each bar compute
                             climax_line = prev_close + climax_atr_mult × ATR
                         As soon as the bar's ``High`` touches that
                         line — AND the bar's own range exceeds the
                         same ``climax_atr_mult × ATR`` threshold
                         (i.e. it's a genuinely wide bar, not a
                         one-tick wick) — the trade fills at the
                         line (or at ``open`` on a gap-up). The rule
                         is intentionally ``High``-based so it also
                         fires on bars that spike intraday and then
                         close back down (ALL 2025-05-08). AMD
                         2019-03-19 still fires because its +11%
                         day prints a high well above prev_close.
        4. EMA trail:    close < 10 EMA -> exit at close. The doc
                         warns that the breakout candle often
                         retests the EMAs, so we only arm the trail
                         while the trade is in profit
                         (``trail_after_profit=True``). The check is
                         intentionally per-bar (not sticky): the
                         trail only fires when *this* bar's close is
                         above entry — preserving the original 익절
                         behaviour. The 손절 side is handled by the
                         breakeven stop above, which avoids killing
                         trades that briefly dipped below entry and
                         later recovered into a bigger profit.
        5. Time stop:    held >= max_holding_days  -> exit at close.

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
        extension_pct: float = 0.15,
        extension_atr_mult: float = 2.5,
        climax_atr_mult: float = 1.5,
        max_entry_chase_ratio: float = 0.15,
        max_entry_ema_extension_pct: float | None = None,
        max_ema_slope_decline: float | None = 0.01,
        min_ema_slow_slope: float | None = None,
        max_ema_slow_slope: float | None = None,
        trail_after_profit: bool = True,
        arm_breakeven_after_profit: bool = True,
        require_gap_up: bool = False,
    ):
        self._market_data = market_data
        self._detector = detector
        self.ema_trail = ema_trail
        self.ema_slow = ema_slow
        self.atr_period = atr_period
        self.extension_pct = extension_pct
        self.extension_atr_mult = extension_atr_mult
        self.climax_atr_mult = climax_atr_mult
        self.max_entry_chase_ratio = max_entry_chase_ratio
        self.max_entry_ema_extension_pct = max_entry_ema_extension_pct
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
        self.trail_after_profit = trail_after_profit
        self.arm_breakeven_after_profit = arm_breakeven_after_profit
        self.require_gap_up = require_gap_up

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

        performance, equity_curve = self._run_signals(df, signals, config)

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

            trade, exit_idx = self._execute_trade(df, signal, entry_idx, capital, config)
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
    ) -> tuple[Trade | None, int]:
        entry_price = float(df["Open"].iloc[entry_idx])

        # Optional gap-up confirmation (TraderLion: "Most Wedge Pops that
        # start new trends often include unfilled gaps with strong volume").
        # If enabled, the next bar's open must clear the breakout bar's
        # close — otherwise the breakout failed to follow through and we
        # skip this signal entirely.
        if self.require_gap_up and entry_idx > 0:
            prev_close = float(df["Close"].iloc[entry_idx - 1])
            if entry_price <= prev_close:
                return None, entry_idx

        # "Chase-entry" filter: measure how far the entry-bar open has
        # pushed above the SIGNAL bar's high relative to that bar's
        # range. If the next-bar open is already more than
        # ``max_entry_chase_ratio`` of the signal range above its high,
        # the breakout has "spent" its move and we're buying at an
        # exhausted level. Set ``max_entry_chase_ratio`` very high to
        # disable. (AMD 2020-06-09 case.)
        if self.max_entry_chase_ratio < float("inf") and entry_idx > 0:
            sig_high = float(df["High"].iloc[entry_idx - 1])
            sig_low = float(df["Low"].iloc[entry_idx - 1])
            sig_range = sig_high - sig_low
            if sig_range > 0:
                chase_ratio = (entry_price - sig_high) / sig_range
                if chase_ratio > self.max_entry_chase_ratio:
                    return None, entry_idx

        # "EMA-extension" entry filter: reject entries where the
        # entry-bar open is more than ``max_entry_ema_extension_pct``
        # above ``max(ema_fast, ema_slow)`` at the signal bar. The
        # motivation is the same as the chase filter but anchored at
        # the EMA stack instead of the signal bar's high — catches
        # entries where the breakout has already stretched well past
        # the trend line before we buy. Setting this to ``None``
        # disables the filter entirely.
        if (
            self.max_entry_ema_extension_pct is not None
            and entry_idx > 0
        ):
            sig_fast = float(df["ema_trail"].iloc[entry_idx - 1])
            sig_slow = float(df["ema_slow"].iloc[entry_idx - 1])
            ref_ema = max(sig_fast, sig_slow)
            if ref_ema > 0:
                extension = (entry_price - ref_ema) / ref_ema
                if extension > self.max_entry_ema_extension_pct:
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

        exit_price, exit_idx = self._find_exit(df, entry_idx, entry_price, stop, config.max_holding_days)

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
    ) -> tuple[float, int]:
        last_idx = min(entry_idx + max_holding_days, len(df) - 1)
        # `armed_stop` is the stateful 손절 mechanism. It starts at the
        # consolidation-low stop and ratchets up to `entry_price`
        # (breakeven) the first time the trade closes above entry.
        # Only goes up, never down.
        armed_stop = stop

        # The loop starts at ``entry_idx`` (NOT ``entry_idx + 1``) so
        # the entry bar itself is checked for same-day exits. A bar
        # that spikes intraday past the exhaustion line, or an
        # entry-day reversal that pierces the consolidation-low stop,
        # both fire on the very first iteration. Breakeven arming at
        # end-of-bar then correctly flows into the next bar's hard
        # stop check (AMD 2019-08-22 case is still handled — arming
        # happens at the end of the 08-22 iteration, before 08-23's
        # hard-stop check sees ``armed_stop = entry_price``).
        for i in range(entry_idx, last_idx + 1):
            open_bar = float(df["Open"].iloc[i])
            high_bar = float(df["High"].iloc[i])
            low_bar = float(df["Low"].iloc[i])
            low = low_bar
            close = float(df["Close"].iloc[i])
            ema_fast = float(df["ema_trail"].iloc[i])
            ema_slow = float(df["ema_slow"].iloc[i])
            atr = float(df["atr"].iloc[i]) if not pd.isna(df["atr"].iloc[i]) else 0.0

            # 1. Hard stop — LOW-based limit-order model. Symmetric
            #    to the exhaustion exit:
            #      - exhaustion: bar's High touches the profit line
            #        → fill at ``max(open, line)``. Gap-up favours
            #        the seller and fills at the higher open.
            #      - hard stop: bar's Low touches the stop line
            #        → fill at ``min(open, stop)``. Gap-down hurts
            #        the seller and fills at the lower open.
            #    When the bar opened ABOVE the stop and then the
            #    intraday low pierces it, the stop order is hit
            #    during the bar and fills at the stop line. When
            #    the bar opened BELOW the stop (gap-down through
            #    the level), the sell executes at market open
            #    (realistic slippage). Covers both the original
            #    consolidation-low stop and the breakeven ratchet.
            if low <= armed_stop:
                return min(open_bar, armed_stop), i

            # 2. Exhaustion Extension — HIGH-based limit-order model.
            #    Two trigger lines are computed each bar from the
            #    higher of the fast/slow EMAs:
            #        pct_line =   ref_ema × (1 + extension_pct)
            #        atr_line =   ref_ema + atr × extension_atr_mult
            #    The trade has an implicit sell limit at the LOWER of
            #    the two — that's the line the price reaches first as
            #    the rally stretches. As soon as the bar's HIGH touches
            #    that line, the order fills at the LINE — NOT at the
            #    bar's high. The gap-up case is handled by
            #    ``max(open, trigger)``: when a bar gaps clean through
            #    the line, the limit order fills at the open (the gap
            #    favours the seller).
            #
            #    Both triggers must sit above ``entry_price`` to count;
            #    right after entry the ref_ema is typically below entry,
            #    so the "trigger line" would sit at a loss and firing
            #    it makes no sense as a sell-into-strength exit. This
            #    guard lets the EMAs catch up before the rule arms.
            ref_ema = max(ema_fast, ema_slow)
            if ref_ema > 0:
                pct_trigger = ref_ema * (1 + self.extension_pct)
                atr_trigger = (
                    ref_ema + atr * self.extension_atr_mult
                    if atr > 0
                    else float("inf")
                )
                candidates = [
                    t
                    for t in (pct_trigger, atr_trigger)
                    if t > entry_price
                ]
                if candidates:
                    trigger = min(candidates)
                    if high_bar >= trigger:
                        return max(open_bar, trigger), i

            # 3. Climax bar — HIGH-based limit-order model.
            #    Triggers on a "parabolic pop" bar: the bar's HIGH
            #    prints more than ``climax_atr_mult × atr`` above
            #    yesterday's close, AND the bar's range itself
            #    exceeds the same multiple (wide bar, not a tiny
            #    wick). Fills at the climax line
            #    ``prev_close + climax_atr_mult × atr`` (gap-up
            #    fills at open). The line is recomputed each bar
            #    so it works as an adaptive "N-ATR pop from
            #    yesterday's close" profit target that tracks the
            #    trade without caring whether ``close`` is near the
            #    bar high — AMD 2019-03-19 still fires, and bars
            #    like ALL 2025-05-08 that spike intraday but close
            #    back down also fire.
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

            # Ratchet the stop up to entry on the first profit-close.
            # Sticky: never relaxes back down. Only the 손절 leg is
            # stateful — the EMA trail below stays per-bar so the
            # 익절 path keeps its original (recoverable) semantics.
            if self.arm_breakeven_after_profit and close > entry_price and armed_stop < entry_price:
                armed_stop = entry_price

            # 4. EMA trail — Wedge Drop / failed EMA Crossback.
            #    Per-bar check (intentionally NOT sticky), so the
            #    profit-side trail keeps its original behaviour.
            if close < ema_fast:
                if not self.trail_after_profit or close > entry_price:
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
