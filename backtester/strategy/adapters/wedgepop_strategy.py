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

    Exit (whichever fires first after entry)
        1. Hard stop:    intraday Low <= stop_loss   -> exit at stop_loss
        2. Exhaustion:   close has stretched far above the 10 EMA -> sell
                         into strength at the close. Triggered when
                         ``(close - ema) / ema >= extension_pct`` OR
                         ``close - ema >= ATR(14) * extension_atr_mult``.
                         Captures TraderLion's "scale out into euphoric
                         price action" rule.
        3. EMA trail:    close < 10 EMA -> exit at close. The doc warns
                         that the breakout candle often retests the EMAs,
                         so we only arm the trail once the trade is in
                         profit (`trail_after_profit=True`); before that
                         the consolidation-low stop is the only protection.
        4. Time stop:    held >= max_holding_days  -> exit at close.

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
        atr_period: int = 14,
        extension_pct: float = 0.15,
        extension_atr_mult: float = 2.5,
        trail_after_profit: bool = True,
        require_gap_up: bool = False,
    ):
        self._market_data = market_data
        self._detector = detector
        self.ema_trail = ema_trail
        self.atr_period = atr_period
        self.extension_pct = extension_pct
        self.extension_atr_mult = extension_atr_mult
        self.trail_after_profit = trail_after_profit
        self.require_gap_up = require_gap_up

    # ---- public API ----

    def run(self, config: StrategyConfig) -> StrategyResult:
        """Fetch data via the market-data port and execute the strategy."""
        df = self._market_data.fetch_ohlcv(
            config.ticker, config.start_date, config.end_date
        )
        return self.execute(df, config)

    def execute(
        self, df: pd.DataFrame, config: StrategyConfig
    ) -> StrategyResult:
        """Run the strategy on a pre-fetched OHLCV DataFrame.

        Useful when the caller already holds the market data (e.g. a UI
        page rendering both the chart and the strategy from a single
        fetch). The DataFrame must have the same shape as
        `MarketDataPort.fetch_ohlcv` returns.
        """
        df = self._with_indicators(df)
        signals = self._detector.detect(df)

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
        curve: list[EquityPoint] = [
            EquityPoint(date=config.start_date, equity=capital)
        ]

        next_open_idx = 0  # blocks new entries while a position is open
        for signal in signals:
            entry_idx = self._next_open_index(df, signal.date)
            if entry_idx is None or entry_idx < next_open_idx:
                continue

            trade, exit_idx = self._execute_trade(
                df, signal, entry_idx, capital, config
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

        stop = float(signal.stop_loss)
        risk_per_share = entry_price - stop
        if risk_per_share <= 0:
            return None, entry_idx

        risk_amount = capital * config.risk_per_trade
        shares = max(1, int(risk_amount / risk_per_share))

        exit_price, exit_idx = self._find_exit(
            df, entry_idx, entry_price, stop, config.max_holding_days
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
    ) -> tuple[float, int]:
        last_idx = min(entry_idx + max_holding_days, len(df) - 1)

        for i in range(entry_idx + 1, last_idx + 1):
            low = float(df["Low"].iloc[i])
            close = float(df["Close"].iloc[i])
            ema = float(df["ema_trail"].iloc[i])
            atr = float(df["atr"].iloc[i]) if not pd.isna(df["atr"].iloc[i]) else 0.0

            # 1. Hard stop — failed breakout
            if low <= stop:
                return stop, i

            # 2. Exhaustion Extension — sell into strength
            if ema > 0:
                distance = close - ema
                pct_extended = distance / ema >= self.extension_pct
                atr_extended = (
                    atr > 0 and distance >= atr * self.extension_atr_mult
                )
                if pct_extended or atr_extended:
                    return close, i

            # 3. EMA trail — Wedge Drop / failed EMA Crossback
            if close < ema:
                if not self.trail_after_profit or close > entry_price:
                    return close, i

        # 4. Time stop
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
            total_return_pct=round(
                (final_capital - initial_capital) / initial_capital, 4
            ),
            total_trades=len(trades),
            win_rate=round(len(wins) / len(trades), 4) if trades else 0.0,
            avg_win_pct=round(sum(t.pnl_pct for t in wins) / len(wins), 4)
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
