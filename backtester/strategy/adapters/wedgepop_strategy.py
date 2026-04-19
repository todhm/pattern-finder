from datetime import date

import pandas as pd

from data.domain.ports import MarketDataPort
from pattern.domain.models import PatternSignal
from pattern.domain.ports import PatternDetector
from pattern.helpers.pivots import (
    find_swing_highs,
    find_swing_lows,
    fit_lower_trendline,
    recent_swing_high,
)
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

    Exit — only four opt-in conditions, each independent:

        1. ``exit_detector`` fires   — Exhaustion Extension Top exit
                                       (inject ``ExhaustionExtensionTopDetector``).
        2. ``enable_trendline_exit`` — bar's LOW pierces the higher-low
                                       trendline fitted through recent swing
                                       lows. LOW-based limit-order fill: the
                                       sell fills at the line (or at the open
                                       on a gap-down).
        3. ``use_smart_trail``       — Chandelier exit:
                                       ``highest_high - N × ATR``, N widens with
                                       R-profit (<2R→3, 2-4R→4, >4R→5). Arms
                                       after 3 bars. LOW-based (wick-proof).
        4. ``enable_resistance_break_exit`` — false-breakout rule. Entry-time
                                       swing resistance is stored; once close
                                       confirms above it, a subsequent low
                                       piercing back below fires an exit at
                                       the resistance line (gap-down → open).

        If none is active (or none fires), the position is held to the
        last bar of the dataframe. There is no hard stop, EMA trail,
        exhaustion-line, climax-bar, or time stop.

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
        enable_swing_resistance_filter: bool = False,
        swing_pivot_left: int = 2,
        swing_pivot_right: int = 2,
        swing_pivot_lookback: int = 60,
        swing_resistance_tolerance_atr: float = 0.5,
        enable_trendline_exit: bool = False,
        trendline_max_pivots: int = 3,
        trendline_min_pivots: int = 2,
        enable_resistance_break_exit: bool = False,
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
        self.enable_swing_resistance_filter = enable_swing_resistance_filter
        self.swing_pivot_left = swing_pivot_left
        self.swing_pivot_right = swing_pivot_right
        self.swing_pivot_lookback = swing_pivot_lookback
        self.swing_resistance_tolerance_atr = swing_resistance_tolerance_atr
        self.enable_trendline_exit = enable_trendline_exit
        self.trendline_max_pivots = trendline_max_pivots
        self.trendline_min_pivots = trendline_min_pivots
        self.enable_resistance_break_exit = enable_resistance_break_exit

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
        if (
            self.enable_swing_resistance_filter
            or self.enable_trendline_exit
            or self.enable_resistance_break_exit
        ):
            df["swing_high"] = find_swing_highs(
                df, left=self.swing_pivot_left, right=self.swing_pivot_right
            )
            df["swing_low"] = find_swing_lows(
                df, left=self.swing_pivot_left, right=self.swing_pivot_right
            )
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

        # Swing-high resistance filter: reject if the entry-bar open
        # sits just beneath a recent confirmed swing high (the entry
        # would be buying directly into supply). Passing overhead
        # resistance entirely is fine — we only reject entries that
        # are close enough that the pivot blocks further upside.
        if (
            self.enable_swing_resistance_filter
            and entry_idx > 0
            and "swing_high" in df.columns
        ):
            signal_idx = entry_idx - 1
            res = recent_swing_high(
                df["swing_high"],
                upto_idx=signal_idx,
                lookback=self.swing_pivot_lookback,
                right=self.swing_pivot_right,
            )
            if res is not None:
                _, pivot_price = res
                atr_at_signal = float(df["atr"].iloc[signal_idx])
                if atr_at_signal > 0:
                    gap = pivot_price - entry_price
                    if (
                        gap > 0
                        and gap < self.swing_resistance_tolerance_atr * atr_at_signal
                    ):
                        return None, entry_idx

        stop = float(signal.stop_loss)
        risk_per_share = entry_price - stop
        if risk_per_share <= 0:
            return None, entry_idx

        risk_amount = capital * config.risk_per_trade
        shares = max(1, int(risk_amount / risk_per_share))

        # Entry-time swing resistance — fed to the resistance-break
        # exit rule. Computed here so the exit loop doesn't recompute
        # per bar. ``None`` means no confirmable pivot was available.
        entry_resistance: float | None = None
        if self.enable_resistance_break_exit and entry_idx > 0 and "swing_high" in df.columns:
            res = recent_swing_high(
                df["swing_high"],
                upto_idx=entry_idx - 1,
                lookback=self.swing_pivot_lookback,
                right=self.swing_pivot_right,
            )
            if res is not None:
                _, entry_resistance = res

        exit_price, exit_idx, exit_reason = self._find_exit(
            df,
            entry_idx,
            entry_price,
            stop,
            config.max_holding_days,
            exit_dates,
            entry_resistance,
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
            exit_reason=exit_reason,
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
        entry_resistance: float | None = None,
    ) -> tuple[float, int, str]:
        if exit_dates is None:
            exit_dates = set()
        last_idx = len(df) - 1

        # Smart trail state — Chandelier exit with profit-tier widening.
        # ``initial_risk`` is the 1R unit for R-based trail widening.
        # ``stop`` is no longer checked as a hard stop (removed per
        # spec), but it still defines the risk basis computed by
        # ``_execute_trade`` for position sizing.
        highest_high = 0.0
        initial_risk = entry_price - stop

        # Resistance-break exit state. If the entry is already above
        # the stored resistance (fresh-breakout entry), the breakout
        # is considered confirmed immediately — a subsequent low
        # piercing back through the line fires the exit.
        resistance_confirmed = (
            entry_resistance is not None and entry_price >= entry_resistance
        )

        # Only three exit paths are active:
        #   (1) ``exit_detector`` firing (Exhaustion Extension Top)
        #   (2) higher-low trendline break
        #   (3) Smart Trail (Chandelier)
        # The entry bar itself is skipped for (1) and (2) — a same-bar
        # exit against our own entry is nonsense. (3) also effectively
        # waits ≥ 3 bars via its arming guard. If no condition fires,
        # the position holds until the final bar of the dataframe.
        for i in range(entry_idx, last_idx + 1):
            open_bar = float(df["Open"].iloc[i])
            high_bar = float(df["High"].iloc[i])
            low_bar = float(df["Low"].iloc[i])
            close = float(df["Close"].iloc[i])
            atr = float(df["atr"].iloc[i]) if not pd.isna(df["atr"].iloc[i]) else 0.0

            if high_bar > highest_high:
                highest_high = high_bar

            # (1) Exhaustion Extension Top exit — fires when the
            #     injected ``exit_detector`` signals on this bar.
            #     Detection was pre-run causally, so firing on this
            #     bar's close uses no future data.
            if i > entry_idx and df.index[i].date() in exit_dates:
                return close, i, "exhaustion_exit"

            # (2) Higher-low trendline break — LOW-based limit-order
            #     model. The instant the bar's low pierces the trendline
            #     fit through recent confirmable swing lows, the sell
            #     limit at the line fills. Gap-downs fill at the open
            #     (realistic slippage) — symmetric to the smart-trail
            #     fill model. Causality: only pivots with
            #     ``idx ≤ i - right`` contribute, so the trendline at
            #     bar i uses no future data.
            if (
                self.enable_trendline_exit
                and i > entry_idx
                and "swing_low" in df.columns
            ):
                tl = fit_lower_trendline(
                    df["swing_low"],
                    upto_idx=i,
                    lookback=self.swing_pivot_lookback,
                    right=self.swing_pivot_right,
                    max_points=self.trendline_max_pivots,
                    min_points=self.trendline_min_pivots,
                )
                if tl is not None:
                    slope, intercept, _ = tl
                    if slope > 0:
                        trendline_y = slope * i + intercept
                        if low_bar <= trendline_y:
                            return (
                                min(open_bar, trendline_y),
                                i,
                                "trendline_break",
                            )

            # (4) Resistance-break (false-breakout) exit. Once the
            #     close has confirmed above the entry-time swing
            #     resistance, a later low piercing back through the
            #     line fires a LOW-based limit-order exit. Skip the
            #     entry bar itself — confirmation and break on the
            #     same bar is a no-op round trip.
            if (
                self.enable_resistance_break_exit
                and entry_resistance is not None
                and i > entry_idx
            ):
                if not resistance_confirmed and close >= entry_resistance:
                    resistance_confirmed = True
                elif resistance_confirmed and low_bar <= entry_resistance:
                    return (
                        min(open_bar, entry_resistance),
                        i,
                        "resistance_break",
                    )

            # (3) Smart Trail (Chandelier with profit-tier widening).
            if self.use_smart_trail:
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
                    if low_bar <= trail_level:
                        return min(open_bar, trail_level), i, "smart_trail"

        # Fallback — no exit fired. Hold to the last bar of data.
        return float(df["Close"].iloc[last_idx]), last_idx, "end_of_data"

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
