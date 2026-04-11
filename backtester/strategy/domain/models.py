from datetime import date
from typing import Any

from pydantic import BaseModel, Field


class StrategyConfig(BaseModel):
    ticker: str
    start_date: date
    end_date: date
    pattern_name: str
    initial_capital: float = 100_000.0
    risk_per_trade: float = 0.02
    max_holding_days: int = 60
    pattern_params: dict[str, Any] = Field(default_factory=dict)


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


class StrategyPerformance(BaseModel):
    initial_capital: float
    final_capital: float
    total_return_pct: float
    total_trades: int
    win_rate: float
    avg_win_pct: float
    avg_loss_pct: float
    max_drawdown_pct: float
    trades: list[Trade]


class EquityPoint(BaseModel):
    date: date
    equity: float


class StrategyResult(BaseModel):
    config: StrategyConfig
    performance: StrategyPerformance
    equity_curve: list[EquityPoint]
