"""Pydantic request/response models for backtest API."""
from datetime import date, datetime
from typing import Optional, List
from pydantic import BaseModel, Field


class BacktestRequest(BaseModel):
    country: str = Field(..., description="KR or US")
    start_date: date
    end_date: date
    initial_cash: float = 10_000_000
    top_n: int = 10
    rebal_freq_days: int = 5
    grades_filter: List[str] = Field(
        default_factory=lambda: ["STRONG_BUY", "BUY", "강력매수", "매수"]
    )
    commission_rate: float = 0.0025
    slippage_rate: float = 0.001


class GradeGenerationRequest(BaseModel):
    country: str = Field(..., description="KR or US")
    start_date: date
    end_date: date
    skip_existing: bool = True
    use_prefilter: bool = False
    prefilter_top_n: int = 500


class BacktestSummary(BaseModel):
    run_id: str
    country: str
    start_date: date
    end_date: date
    status: str
    metrics: Optional[dict] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
