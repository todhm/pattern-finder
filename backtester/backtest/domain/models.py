from datetime import date

from pydantic import BaseModel


class Trade(BaseModel):
    pattern_name: str
    entry_date: date
    exit_date: date
    entry_price: float
    exit_price: float
    stop_loss: float
    shares: int
    pnl: float
    pnl_pct: float


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
