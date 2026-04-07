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
