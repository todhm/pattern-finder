from pydantic import BaseModel

from backtest.models.trade import Trade


class BacktestResult(BaseModel):
    initial_capital: float
    final_capital: float
    total_return_pct: float
    total_trades: int
    win_rate: float
    avg_win_pct: float
    avg_loss_pct: float
    max_drawdown_pct: float
    trades: list[Trade]
