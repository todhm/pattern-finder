from datetime import date, datetime, timedelta

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


class QuarterlyFinancials(BaseModel):
    """한 분기의 실적 스냅샷 — 미너비니류 성장주 스크리닝용.

    ``fiscal_date``는 분기 말일, ``report_date``는 실제 발표일.
    과거 시점 재현(point-in-time)에서는 반드시
    ``effective_report_date <= 기준일`` 로 필터해야 look-ahead가 없다.
    발표일이 없으면 보수적으로 ``fiscal_date + 90일``을 가정한다.

    금액 필드는 소스가 값을 안 주면 ``None`` — 소비자(스크리너)가
    항목별로 '데이터 없음' 처리한다.
    """

    symbol: str
    fiscal_date: date
    report_date: date | None = None
    eps_actual: float | None = None
    eps_estimate: float | None = None
    revenue: float | None = None
    net_income: float | None = None

    @property
    def effective_report_date(self) -> date:
        return self.report_date or (self.fiscal_date + timedelta(days=90))

    @property
    def net_margin(self) -> float | None:
        if self.revenue and self.net_income is not None:
            return self.net_income / self.revenue
        return None


class AnnualFinancials(BaseModel):
    """회계연도 하나의 실적 — 미너비니 2-5(코드 33)·2-6·2-10 연간 판정용.

    ``fiscal_date``는 회계연도 말일. ``report_date``는 연간 실적 발표일 —
    소스가 연간 발표일을 주지 않으면 어댑터가 같은 회계연도 말의 Q4
    분기 발표일을 채워 넣고, 그것도 없으면 ``fiscal_date + 90일``로
    간주한다 (point-in-time 보수 가정).
    """

    symbol: str
    fiscal_date: date
    report_date: date | None = None
    eps: float | None = None
    revenue: float | None = None
    net_income: float | None = None

    @property
    def effective_report_date(self) -> date:
        return self.report_date or (self.fiscal_date + timedelta(days=90))

    @property
    def net_margin(self) -> float | None:
        if self.revenue and self.net_income is not None:
            return self.net_income / self.revenue
        return None


class QuarterlyBalance(BaseModel):
    """분기 재무상태표 발췌 — 미너비니 2-9 (재고·매출채권 적색경보)용.

    ``report_date``는 같은 분기 실적 발표일(어댑터가 EARNINGS에서 채움),
    없으면 ``fiscal_date + 45일``로 간주 (10-Q 제출 기한 근사).
    재고가 없는 서비스업은 ``inventory=None`` 또는 0.
    """

    symbol: str
    fiscal_date: date
    report_date: date | None = None
    inventory: float | None = None
    receivables: float | None = None

    @property
    def effective_report_date(self) -> date:
        return self.report_date or (self.fiscal_date + timedelta(days=45))


class GrowthSnapshot(BaseModel):
    """티커 하나의 성장 펀더멘털 묶음 (분기·연간 리스트 + 소속 산업군).

    ``quarters``·``annuals``는 fiscal_date 오름차순. 데이터 소스 장애 시
    빈 리스트 — 스크리너가 '펀더멘털 미확인'으로 처리한다 (탈락과 구분).
    """

    symbol: str
    sector: str | None = None
    industry: str | None = None
    quarters: list[QuarterlyFinancials] = []
    annuals: list[AnnualFinancials] = []


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
