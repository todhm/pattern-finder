from datetime import date, datetime

from pydantic import BaseModel


class OHLCV(BaseModel):
    date: date
    open: float
    high: float
    low: float
    close: float
    volume: int


class EarningsEvent(BaseModel):
    """Single earnings announcement for a symbol.

    ``report_date`` is the calendar date the report is released. EODHD
    additionally exposes ``before_after_market`` (BMO/AMC); we keep it
    optional because not every source provides it. Strategies use only
    the date for window gates (e.g. "trade ± N days around earnings").
    """

    symbol: str
    report_date: date
    before_after_market: str | None = None  # "BMO" | "AMC" | None
    eps_actual: float | None = None
    eps_estimate: float | None = None


class NewsEvent(BaseModel):
    """A single news-catalyst item attached to a symbol.

    ``date`` is the calendar date of publication (tz-stripped to NY).
    ``sentiment`` is a normalized score in ``[-1.0, +1.0]`` when the
    source provides one; ``None`` if not. Strategies use the date only
    by default; sentiment is exposed for callers that want to filter
    on it (e.g. accept only positive-sentiment days).
    """

    symbol: str
    published_at: datetime
    title: str
    sentiment: float | None = None
    source: str = "eodhd"

    @property
    def date(self) -> date:
        return self.published_at.date()
