from datetime import date
from typing import Any

from pydantic import BaseModel, Field


class PatternSignal(BaseModel):
    date: date
    pattern_name: str
    entry_price: float
    stop_loss: float
    confidence: float = 1.0
    metadata: dict[str, Any] = Field(default_factory=dict)
