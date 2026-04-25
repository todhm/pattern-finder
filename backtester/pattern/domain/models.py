from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, Field


class PatternSignal(BaseModel):
    date: date
    pattern_name: str
    entry_price: float
    stop_loss: float
    confidence: float = 1.0
    metadata: dict[str, Any] = Field(default_factory=dict)
    # Intraday bar timestamp. ``None`` for daily detectors (the
    # session ``date`` is a sufficient bar identifier at 1-bar-per-day
    # resolution); intraday detectors (15m, 1h, …) populate this with
    # the signal bar's exact DatetimeIndex value so the strategy
    # layer can unambiguously map the signal back to its bar.
    timestamp: datetime | None = None
