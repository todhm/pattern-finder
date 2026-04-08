from datetime import date
from typing import Any

from pydantic import BaseModel, Field

from backtest.domain.models import BacktestResult


class StrategyConfig(BaseModel):
    ticker: str
    start_date: date
    end_date: date
    pattern_name: str
    initial_capital: float = 100_000.0
    risk_per_trade: float = 0.02
    max_holding_days: int = 60
    pattern_params: dict[str, Any] = Field(default_factory=dict)


class EquityPoint(BaseModel):
    date: date
    equity: float


class StrategyResult(BaseModel):
    config: StrategyConfig
    backtest_result: BacktestResult
    equity_curve: list[EquityPoint]
