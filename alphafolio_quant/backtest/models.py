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
    # NOTE: us_stock_grade stores Korean labels with a space — "강력 매수" / "매수".
    # The old default used "강력매수" (no space), which silently matched nothing,
    # so STRONG_BUY names were dropped from every backtest (English labels are
    # never produced by the grader either, kept only for forward-compat).
    grades_filter: List[str] = Field(
        default_factory=lambda: ["STRONG_BUY", "BUY", "강력 매수", "매수"]
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
    # Whether to apply the event_engine total_modifier (earnings/options/GEX/
    # insider/news) to the score. Pass-A runs BEFORE options are collected, so
    # it sets this False (faster, no stale-option noise — it only ranks symbols
    # for option selection). Pass-B sets it True to fold in the freshly
    # collected option signals (options_modifier + gex_modifier).
    with_event_modifier: bool = True
    # reco-only 팩터 개선(value 게이팅 등). 기본 False → backtest baseline 불변.
    # reco DAG 의 grade 태스크만 True 로 전달(OOS 재검증 전까지 reco 한정).
    improved_factors: bool = False


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
