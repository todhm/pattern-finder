import pandas as pd

from backtest.models.result import BacktestResult
from backtest.models.trade import Trade
from pattern.models.signal import PatternSignal


class BacktestEngine:
    """Simulates trades based on pattern signals.

    Exit strategy:
    - Stop loss: exit if low <= stop_loss.
    - Trailing stop: exit if close < 10 EMA after entry.
    - Max holding: force exit after `max_holding_days`.
    """

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        risk_per_trade: float = 0.02,
        max_holding_days: int = 60,
    ):
        self.initial_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.max_holding_days = max_holding_days

    def run(
        self, df: pd.DataFrame, signals: list[PatternSignal]
    ) -> BacktestResult:
        df = df.copy()
        df["ema10"] = df["Close"].ewm(span=10, adjust=False).mean()

        capital = self.initial_capital
        peak_capital = capital
        max_drawdown = 0.0
        trades: list[Trade] = []

        for signal in signals:
            trade = self._execute_trade(df, signal, capital)
            if trade is None:
                continue

            capital += trade.pnl
            peak_capital = max(peak_capital, capital)
            drawdown = (peak_capital - capital) / peak_capital
            max_drawdown = max(max_drawdown, drawdown)
            trades.append(trade)

        return self._build_result(capital, max_drawdown, trades)

    def _execute_trade(
        self, df: pd.DataFrame, signal: PatternSignal, capital: float
    ) -> Trade | None:
        signal_date = pd.Timestamp(signal.date)
        dates_after = df.index[df.index > signal_date]
        if len(dates_after) < 2:
            return None

        entry_idx = df.index.get_loc(dates_after[0])
        entry_price = df["Open"].iloc[entry_idx]

        risk_per_share = entry_price - signal.stop_loss
        if risk_per_share <= 0:
            return None

        risk_amount = capital * self.risk_per_trade
        shares = max(1, int(risk_amount / risk_per_share))

        exit_price, exit_idx = self._find_exit(
            df, entry_idx, signal.stop_loss
        )

        pnl = (exit_price - entry_price) * shares
        pnl_pct = (exit_price - entry_price) / entry_price

        return Trade(
            pattern_name=signal.pattern_name,
            entry_date=df.index[entry_idx].date(),
            exit_date=df.index[exit_idx].date(),
            entry_price=round(entry_price, 2),
            exit_price=round(exit_price, 2),
            stop_loss=round(signal.stop_loss, 2),
            shares=shares,
            pnl=round(pnl, 2),
            pnl_pct=round(pnl_pct, 4),
        )

    def _find_exit(
        self, df: pd.DataFrame, entry_idx: int, stop_loss: float
    ) -> tuple[float, int]:
        for i in range(entry_idx + 1, min(entry_idx + self.max_holding_days, len(df))):
            if df["Low"].iloc[i] <= stop_loss:
                return stop_loss, i

            if df["Close"].iloc[i] < df["ema10"].iloc[i]:
                return df["Close"].iloc[i], i

        last_idx = min(entry_idx + self.max_holding_days, len(df)) - 1
        return df["Close"].iloc[last_idx], last_idx

    def _build_result(
        self,
        final_capital: float,
        max_drawdown: float,
        trades: list[Trade],
    ) -> BacktestResult:
        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]

        return BacktestResult(
            initial_capital=self.initial_capital,
            final_capital=round(final_capital, 2),
            total_return_pct=round(
                (final_capital - self.initial_capital) / self.initial_capital, 4
            ),
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
