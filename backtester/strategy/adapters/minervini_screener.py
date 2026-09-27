"""미너비니 SEPA 자동 스크리너 — 순수 평가 로직.

페이지(``44_Minervini_Screener``)가 유니버스 루프/데이터 fetch를 맡고,
여기는 DataFrame·분기 리스트를 받아 판정만 한다 (전부 순수 함수 —
네트워크/스트림릿 의존 없음, pytest로 직접 검증).

체크 기준 근거: ``pages/_shared/minervini_manual.md``
    STAGE 1  추세 템플릿 1-1~1-7 (1-8 RS는 유니버스 백분위라 페이지에서)
    STAGE 2  분기 EPS·매출·마진 (point-in-time: 발표일 ≤ 기준일)
    STAGE 4  피봇(최근 압축 구간 고점) + 조밀도·거래량 드라이업 힌트
    STAGE 5  손절선 = 피봇 × (1 − stop%), 익절선 = 피봇 × (1 + target%)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from data.domain.models import AnnualFinancials, QuarterlyBalance, QuarterlyFinancials

TRADING_DAYS = {"1m": 21, "3m": 63, "6m": 126, "9m": 189, "12m": 252}

# 추세 템플릿에 필요한 최소 봉 수: 52주 레인지·12개월 모멘텀(252) +
# SMA200을 21거래일 전과 비교(221). 252 + 1(당일)로 잡는다.
MIN_BARS = 253


# ---------------------------------------------------------------------------
# STAGE 1 — 추세 템플릿 (1-1 ~ 1-7)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TrendTemplateResult:
    close: float
    sma50: float
    sma150: float
    sma200: float
    sma200_prev21: float
    low_52w: float
    high_52w: float
    checks: dict[str, bool]
    passed: bool

    @property
    def pct_above_low(self) -> float:
        return self.close / self.low_52w - 1.0

    @property
    def pct_from_high(self) -> float:
        return self.close / self.high_52w - 1.0


def evaluate_trend_template(df: pd.DataFrame) -> TrendTemplateResult | None:
    """1-1 ~ 1-7 판정. ``df``는 기준일까지로 잘린 일봉 (미래 봉 금지).

    봉이 :data:`MIN_BARS` 미만이면 None (판정 불가 — 신규 상장 등).
    """
    if len(df) < MIN_BARS:
        return None
    c = df["Close"]
    close = float(c.iloc[-1])
    sma50 = float(c.rolling(50).mean().iloc[-1])
    sma150 = float(c.rolling(150).mean().iloc[-1])
    sma200_series = c.rolling(200).mean()
    sma200 = float(sma200_series.iloc[-1])
    sma200_prev21 = float(sma200_series.iloc[-22])
    low_52w = float(df["Low"].tail(252).min())
    high_52w = float(df["High"].tail(252).max())
    checks = {
        "1-1 주가>150·200일선": close > sma150 and close > sma200,
        "1-2 150>200일선": sma150 > sma200,
        "1-3 200일선 상승": sma200 > sma200_prev21,
        "1-4 50>150·200일선": sma50 > sma150 and sma50 > sma200,
        "1-5 주가>50일선": close > sma50,
        "1-6 신저가+30%↑": close >= low_52w * 1.30,
        "1-7 신고가-25%내": close >= high_52w * 0.75,
    }
    return TrendTemplateResult(
        close=close, sma50=sma50, sma150=sma150, sma200=sma200,
        sma200_prev21=sma200_prev21, low_52w=low_52w, high_52w=high_52w,
        checks=checks, passed=all(checks.values()),
    )


def momentum_components(df: pd.DataFrame) -> dict[str, float] | None:
    """3/6/9/12개월 수익률 — RS 근거 표시용."""
    c = df["Close"]
    if len(c) < TRADING_DAYS["12m"] + 1:
        return None
    last = float(c.iloc[-1])
    return {
        k: last / float(c.iloc[-1 - n]) - 1.0
        for k, n in TRADING_DAYS.items()
        if k != "1m"
    }


def momentum_score(df: pd.DataFrame) -> float | None:
    """1-8 RS용 가중 모멘텀 = 2×3M + 6M + 9M + 12M 수익률.

    유니버스 전체에서 이 값의 백분위(rank pct)를 RS로 쓴다.
    """
    r = momentum_components(df)
    if r is None:
        return None
    return 2.0 * r["3m"] + r["6m"] + r["9m"] + r["12m"]


def rs_percentiles(scores: dict[str, float]) -> dict[str, float]:
    """모멘텀 점수 dict → 0~100 백분위 dict (유니버스 상대 순위)."""
    if not scores:
        return {}
    s = pd.Series(scores)
    return (s.rank(pct=True) * 100.0).to_dict()


# ---------------------------------------------------------------------------
# STAGE 2 — 펀더멘털 (2-1 ~ 2-4, point-in-time)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GrowthResult:
    eps_yoy: float | None          # 최신 발표 분기 EPS YoY
    eps_yoy_history: list[float]   # 오래된 → 최신 (최대 3개)
    eps_accelerating: bool | None
    revenue_yoy: float | None
    revenue_accelerating: bool | None
    margin_now: float | None
    margin_year_ago: float | None
    checks: dict[str, bool] = field(default_factory=dict)
    n_passed: int = 0
    n_required: int = 3
    passed: bool = False
    data_available: bool = True    # False = 분기 데이터 부족(미확인)


def _yoy(values: list[float | None], i: int) -> float | None:
    """values[i] vs values[i-4] YoY. 전년동기 값이 0/None이면 None."""
    if i < 4:
        return None
    cur, prev = values[i], values[i - 4]
    if cur is None or prev is None or prev == 0:
        return None
    return (cur - prev) / abs(prev)


def evaluate_growth(
    quarters: list[QuarterlyFinancials],
    as_of: date,
    eps_min_yoy: float = 0.25,
    rev_min_yoy: float = 0.20,
    n_required: int = 3,
) -> GrowthResult:
    """2-1~2-4 판정. **발표일 ≤ 기준일** 분기만 사용 (look-ahead 방지).

    checks 4종 중 ``n_required``개 이상이면 통과:
        2-1 최신 분기 EPS YoY ≥ eps_min_yoy
        2-2 EPS YoY 가속 (최근 2~3개 발표 분기에서 증가)
        2-3 매출 YoY ≥ rev_min_yoy 또는 매출 YoY 가속
        2-4 순이익률이 전년 동기 대비 개선
    """
    usable = sorted(
        (q for q in quarters if q.effective_report_date <= as_of),
        key=lambda q: q.fiscal_date,
    )
    eps = [q.eps_actual for q in usable]
    rev = [q.revenue for q in usable]
    margins = [q.net_margin for q in usable]

    eps_yoys = [y for i in range(len(eps)) if (y := _yoy(eps, i)) is not None]
    rev_yoys = [y for i in range(len(rev)) if (y := _yoy(rev, i)) is not None]

    if not eps_yoys:
        return GrowthResult(
            eps_yoy=None, eps_yoy_history=[], eps_accelerating=None,
            revenue_yoy=None, revenue_accelerating=None,
            margin_now=None, margin_year_ago=None,
            n_required=n_required, data_available=False,
        )

    eps_yoy = eps_yoys[-1]
    eps_hist = eps_yoys[-3:]
    eps_accel = (
        all(b > a for a, b in zip(eps_hist, eps_hist[1:]))
        if len(eps_hist) >= 2 else None
    )
    rev_yoy = rev_yoys[-1] if rev_yoys else None
    rev_accel = (
        rev_yoys[-1] > rev_yoys[-2] if len(rev_yoys) >= 2 else None
    )
    margin_now = margins[-1] if margins else None
    margin_year_ago = margins[-5] if len(margins) >= 5 else None

    checks = {
        "2-1 EPS YoY": eps_yoy >= eps_min_yoy,
        "2-2 EPS 가속": bool(eps_accel),
        "2-3 매출": (
            (rev_yoy is not None and rev_yoy >= rev_min_yoy)
            or bool(rev_accel)
        ),
        "2-4 마진 개선": (
            margin_now is not None
            and margin_year_ago is not None
            and margin_now > margin_year_ago
        ),
    }
    n_passed = sum(checks.values())
    return GrowthResult(
        eps_yoy=eps_yoy, eps_yoy_history=eps_hist, eps_accelerating=eps_accel,
        revenue_yoy=rev_yoy, revenue_accelerating=rev_accel,
        margin_now=margin_now, margin_year_ago=margin_year_ago,
        checks=checks, n_passed=n_passed, n_required=n_required,
        passed=n_passed >= n_required,
    )


# ---------------------------------------------------------------------------
# STAGE 2 — 2-7 어닝 서프라이즈 · 2-9 재고/매출채권 (수동 항목의 데이터 보조)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SurpriseResult:
    """2-7 — 발표일 ≤ 기준일 최근 n개 분기의 actual vs estimate."""

    rows: list[dict]               # {분기말, 발표일, 실제, 추정, 서프라이즈, 비트}
    latest_beat: bool | None       # 최신 발표 분기 actual > estimate
    streak: int                    # 최신부터 연속 비트 횟수
    data_available: bool


def evaluate_surprise(
    quarters: list[QuarterlyFinancials], as_of: date, n: int = 4,
) -> SurpriseResult:
    usable = sorted(
        (q for q in quarters
         if q.effective_report_date <= as_of
         and q.eps_actual is not None and q.eps_estimate is not None),
        key=lambda q: q.fiscal_date,
    )[-n:]
    rows = []
    for q in usable:
        est = q.eps_estimate
        surprise = ((q.eps_actual - est) / abs(est)) if est else None
        rows.append({
            "분기말": q.fiscal_date.isoformat(),
            "발표일": q.effective_report_date.isoformat(),
            "실제": q.eps_actual, "추정": est, "서프라이즈": surprise,
            "비트": q.eps_actual > est,
        })
    if not rows:
        return SurpriseResult(rows=[], latest_beat=None, streak=0, data_available=False)
    streak = 0
    for r in reversed(rows):
        if r["비트"]:
            streak += 1
        else:
            break
    return SurpriseResult(rows=rows, latest_beat=rows[-1]["비트"],
                          streak=streak, data_available=True)


@dataclass(frozen=True)
class BalanceResult:
    """2-9 — 최신 발표 분기의 재고·매출채권 YoY vs 매출 YoY.

    - ``inventory_flag``: True = 재고 증가율이 매출 증가율을 ``inv_margin``
      이상 초과 (책: +79% vs +11% → 탈락). None = 재고 없음/측정 불가.
    - ``receivables_flag``: True = 채권 증가율 > 매출 증가율 (경고).
    """

    fiscal_date: date | None
    revenue_yoy: float | None
    inventory_yoy: float | None
    receivables_yoy: float | None
    inventory_flag: bool | None
    receivables_flag: bool | None
    inventory_absent: bool          # 서비스업 등 재고 항목 없음
    data_available: bool

    @property
    def passed(self) -> bool | None:
        if not self.data_available:
            return None
        return not self.inventory_flag


def evaluate_balance(
    balances: list[QuarterlyBalance],
    quarters: list[QuarterlyFinancials],
    as_of: date,
    inv_margin: float = 0.25,
) -> BalanceResult:
    """발표일 ≤ 기준일 분기 중 최신 것과 4분기 전을 비교."""
    usable = sorted((b for b in balances if b.effective_report_date <= as_of),
                    key=lambda b: b.fiscal_date)
    rev_by = {q.fiscal_date: q.revenue for q in quarters
              if q.effective_report_date <= as_of}
    if len(usable) < 5:
        return BalanceResult(None, None, None, None, None, None, False, False)
    cur, prev = usable[-1], usable[-5]

    def _g(a, b):
        if a is None or b is None or b == 0:
            return None
        return (a - b) / abs(b)

    rev_yoy = _g(rev_by.get(cur.fiscal_date), rev_by.get(prev.fiscal_date))
    # 서비스업은 None/0, 소스 아티팩트로 '1' 같은 값이 오기도 → $1,000 미만은 없음 취급
    inv_absent = (cur.inventory or 0) < 1e3 and (prev.inventory or 0) < 1e3
    inv_yoy = None if inv_absent else _g(cur.inventory, prev.inventory)
    rec_yoy = _g(cur.receivables, prev.receivables)
    inv_flag = (None if inv_yoy is None or rev_yoy is None
                else inv_yoy - rev_yoy > inv_margin)
    rec_flag = (None if rec_yoy is None or rev_yoy is None
                else rec_yoy > rev_yoy)
    return BalanceResult(
        fiscal_date=cur.fiscal_date, revenue_yoy=rev_yoy,
        inventory_yoy=inv_yoy, receivables_yoy=rec_yoy,
        inventory_flag=inv_flag, receivables_flag=rec_flag,
        inventory_absent=inv_absent,
        data_available=rev_yoy is not None or rec_yoy is not None or inv_yoy is not None,
    )


# ---------------------------------------------------------------------------
# STAGE 2 — 연간 판정 (2-5 코드 33 · 2-6 EPS 신고 · 2-10 감속 경고)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AnnualResult:
    """연간 실적 기반 판정 묶음 (point-in-time: 발표일 ≤ 기준일 회계연도만).

    - 2-5 코드 33: EPS 가속 · 매출 가속 · 순이익률 상승, 3개 지표 각각
      최근 3년 판정 → ``checks`` / ``n_passed`` (0~3) / ``passed`` (3/3).
    - 2-6 EPS 신고: 최신 연간 EPS가 이전 3~7년 최고치를 넘었는가.
    - 2-10 감속 경고: 연간 EPS 증가율 3개가 뚜렷이 줄어드는가 (탈락 사유).
    측정에 필요한 연도 수가 부족한 항목은 ``None`` (미확인).
    """

    years: list[date]                 # 사용한 회계연도 말일 (오름차순, 최대 4)
    eps: list[float | None]
    revenue: list[float | None]
    margins: list[float | None]
    eps_yoy: list[float]              # 최대 3개 (오래된 → 최신)
    revenue_yoy: list[float]
    checks: dict[str, bool | None] = field(default_factory=dict)
    n_passed: int = 0
    passed: bool = False
    eps_breakout: bool | None = None  # 2-6
    eps_prev_high: float | None = None
    decel_warning: bool | None = None  # 2-10
    data_available: bool = False


def _accelerating(yoys: list[float]) -> bool | None:
    """'3년 연속 가속' 실무 판정 — 3개 YoY 전부 양수이고 마지막이 첫 해보다 높다.

    책의 몬스터 사례(EPS +75→+214→+195%)도 엄격한 단조 증가는 아니므로
    중간 한 해의 소폭 둔화는 허용한다. 3개 미만이면 None.
    """
    if len(yoys) < 3:
        return None
    y = yoys[-3:]
    return all(v > 0 for v in y) and y[-1] > y[0]


def evaluate_annual(
    annuals: list[AnnualFinancials],
    as_of: date,
    breakout_lookback: int = 7,
    decel_ratio: float = 0.5,
) -> AnnualResult:
    """연간 실적으로 2-5 / 2-6 / 2-10 을 판정한다. **발표일 ≤ 기준일** 연도만.

    - 코드 33 (2-5): EPS YoY 3개 가속, 매출 YoY 3개 가속, 순이익률 3년
      연속 상승 (m1>m0>m-1>m-2 — 4개 연도 필요).
    - 2-6: 최신 EPS > 0 이고 이전 ``breakout_lookback``년 최고치 초과
      (비교 대상이 3년 미만이면 None).
    - 2-10: EPS YoY 3개가 두 번 연속 줄고 마지막이 첫 해의
      ``decel_ratio`` 미만 (델: 80→65→28%). 3개 미만이면 None.

    ⚠️ 소스(Alpha Vantage)의 연간 EPS는 GAAP/non-GAAP가 섞일 수 있어
    일회성 손익이 낀 해는 판정이 튄다 — 페이지가 4년 수치를 같이 보여준다.
    """
    usable = sorted(
        (a for a in annuals if a.effective_report_date <= as_of),
        key=lambda a: a.fiscal_date,
    )
    if not usable:
        return AnnualResult(years=[], eps=[], revenue=[], margins=[],
                            eps_yoy=[], revenue_yoy=[])

    last4 = usable[-4:]
    years = [a.fiscal_date for a in last4]
    eps = [a.eps for a in last4]
    rev = [a.revenue for a in last4]
    margins = [a.net_margin for a in last4]

    def _yoys(vals: list[float | None]) -> list[float]:
        out: list[float] = []
        for i in range(1, len(vals)):
            cur, prev = vals[i], vals[i - 1]
            if cur is None or prev is None or prev == 0:
                continue
            out.append((cur - prev) / abs(prev))
        return out

    # YoY 는 연속된 연도끼리만 의미가 있으므로 결측이 끼면 그 자리는 건너뛴다
    eps_yoy = _yoys(eps)
    rev_yoy = _yoys(rev)

    margin_up: bool | None
    if len(margins) == 4 and all(m is not None for m in margins):
        margin_up = all(b > a for a, b in zip(margins, margins[1:]))
    else:
        margin_up = None

    checks: dict[str, bool | None] = {
        "2-5a EPS 3년 가속": _accelerating(eps_yoy),
        "2-5b 매출 3년 가속": _accelerating(rev_yoy),
        "2-5c 순이익률 3년 상승": margin_up,
    }
    n_passed = sum(1 for v in checks.values() if v)
    passed = all(v is True for v in checks.values())

    # 2-6 EPS 신고 — 이전 3~7년 최고치와 비교
    latest_eps = usable[-1].eps
    prev = [a.eps for a in usable[-1 - breakout_lookback:-1] if a.eps is not None]
    if latest_eps is None or len(prev) < 3:
        eps_breakout, eps_prev_high = None, None
    else:
        eps_prev_high = max(prev)
        eps_breakout = latest_eps > 0 and latest_eps > eps_prev_high

    # 2-10 감속 경고
    if len(eps_yoy) >= 3:
        y = eps_yoy[-3:]
        decel = y[0] > y[1] > y[2] and y[2] < y[0] * decel_ratio
    else:
        decel = None

    return AnnualResult(
        years=years, eps=eps, revenue=rev, margins=margins,
        eps_yoy=eps_yoy, revenue_yoy=rev_yoy,
        checks=checks, n_passed=n_passed, passed=passed,
        eps_breakout=eps_breakout, eps_prev_high=eps_prev_high,
        decel_warning=decel,
        data_available=any(v is not None for v in checks.values())
        or eps_breakout is not None or decel is not None,
    )


def annual_evidence(
    annuals: list[AnnualFinancials],
    as_of: date,
    n: int = 7,
) -> list[dict]:
    """상세 근거 표시용 — 기준일까지 발표된 최근 ``n``개 회계연도 테이블."""
    usable = sorted(
        (a for a in annuals if a.effective_report_date <= as_of),
        key=lambda a: a.fiscal_date,
    )
    rows = []
    for i in range(max(0, len(usable) - n), len(usable)):
        a, prev = usable[i], (usable[i - 1] if i > 0 else None)

        def _g(cur, pv):
            if cur is None or pv is None or pv == 0:
                return None
            return (cur - pv) / abs(pv)

        rows.append({
            "회계연도": a.fiscal_date.isoformat(),
            "발표일": a.effective_report_date.isoformat(),
            "EPS": a.eps,
            "EPS YoY": _g(a.eps, prev.eps if prev else None),
            "매출": a.revenue,
            "매출 YoY": _g(a.revenue, prev.revenue if prev else None),
            "순이익률": a.net_margin,
        })
    return rows


# ---------------------------------------------------------------------------
# STAGE 0 — 시장 환경 (지수 하나에 대한 0-1·0-2 판정)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MarketVerdict:
    """지수 하나(SPY·QQQ 등)의 0-1 추세 + 0-2 분산일 판정.

    미너비니는 S&P 500만 보지 않는다 — 나스닥 종합을 가장 자주 보고,
    IBD식 분산일은 S&P 500·나스닥 중 **어느 한쪽**이라도 쌓이면 경고다.
    페이지는 이 결과를 지수별로 만들어 합친다 (``combine_market``).
    """

    symbol: str
    close: float
    sma200: float
    sma200_prev21: float
    trend_ok: bool                 # 0-1: 종가 > 200일선 & 200일선 상승
    dist_days: int | None          # 0-2: 최근 25거래일 분산일 (None = 봉 부족)
    dist_ok: bool | None           # 분산일 < 5

    @property
    def ok(self) -> bool:
        return self.trend_ok and (self.dist_ok is not False)

    @property
    def note(self) -> str:
        arrow = "상승" if self.sma200 > self.sma200_prev21 else "하락"
        dd = "?" if self.dist_days is None else str(self.dist_days)
        return (f"{self.symbol} {self.close:,.0f} vs 200일선 {self.sma200:,.0f} "
                f"({arrow} 중) · 분산일 {dd}회")


def evaluate_market(
    df: pd.DataFrame, symbol: str = "SPY", max_dist_days: int = 5,
) -> MarketVerdict | None:
    """기준일까지의 지수 일봉으로 0-1·0-2 판정. 봉이 222개 미만이면 None."""
    if len(df) < 222:
        return None
    c = df["Close"]
    sma = c.rolling(200).mean()
    close, s200, s200_prev = float(c.iloc[-1]), float(sma.iloc[-1]), float(sma.iloc[-22])
    dd = distribution_days(df)
    return MarketVerdict(
        symbol=symbol, close=close, sma200=s200, sma200_prev21=s200_prev,
        trend_ok=close > s200 and s200 > s200_prev,
        dist_days=dd, dist_ok=None if dd is None else dd < max_dist_days,
    )


def combine_market(verdicts: list[MarketVerdict]) -> tuple[bool | None, list[str]]:
    """여러 지수 판정 합산 → (all_ok, 실패 항목 목록).

    - 0-1 은 **모든** 지수가 200일선 위·상승이어야 통과 (하나라도 아래면 경고).
    - 0-2 는 **어느 한 지수**라도 분산일 5회↑면 경고 (IBD 규칙).
    비어 있으면 (None, []) — 판정 불가.
    """
    if not verdicts:
        return None, []
    failed: list[str] = []
    bad_trend = [v.symbol for v in verdicts if not v.trend_ok]
    if bad_trend:
        failed.append(f"0-1 지수 추세 ({'·'.join(bad_trend)} 200일선 아래/하락)")
    bad_dist = [v.symbol for v in verdicts if v.dist_ok is False]
    if bad_dist:
        failed.append(f"0-2 분산일 ({'·'.join(bad_dist)} 5회↑)")
    return not failed, failed


def distribution_days(
    df: pd.DataFrame,
    window: int = 25,
    min_drop: float = 0.002,
) -> int | None:
    """0-2 랠리의 질 — IBD식 **분산일** 카운트.

    분산일 = 지수가 전일 대비 ``min_drop``(-0.2%) 이상 하락했는데
    거래량은 전일보다 **늘어난** 날 (기관 매도 흔적). 최근
    ``window``(25)거래일 안에서 센다. **4~5회 이상이면 조정 경고**
    (매뉴얼 STAGE 0-2). 봉이 부족하면 None.
    """
    if len(df) < window + 1:
        return None
    tail = df.tail(window + 1)
    ret = tail["Close"].pct_change()
    vol_up = tail["Volume"].diff() > 0
    mask = (ret <= -min_drop) & vol_up
    return int(mask.tail(window).sum())


def split_factor(splits: "pd.Series | None", as_of: date) -> float:
    """기준일 **이후** 분할들의 누적 배수 — 조정가 → 당시 실제 가격 환산용.

    yfinance식 조정가는 미래 분할비율로 나눠져 있으므로, 기준일 이후
    분할비율을 다시 곱하면 당시 거래 가격에 근접한다::

        당시 가격 ≈ 조정 종가 × split_factor(splits, 기준일)

    (배당 조정분은 남아 있어 완전히 정확하진 않다 — 최소 주가 필터
    용도로는 충분. 배당 잔차는 가격을 낮추는 쪽이라 보수적.)

    ``splits``: yfinance ``Ticker.splits`` 관례 — 분할 발효일 index,
    값 = 비율(4:1 forward → 4.0). None/빈 시리즈면 1.0.
    """
    if splits is None or len(splits) == 0:
        return 1.0
    cutoff = pd.Timestamp(as_of)
    idx = splits.index
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
        splits = pd.Series(splits.values, index=idx)
    after = splits[splits.index > cutoff]
    if len(after) == 0:
        return 1.0
    return float(after.prod())


def quarterly_evidence(
    quarters: list[QuarterlyFinancials],
    as_of: date,
    n: int = 6,
) -> list[dict]:
    """상세 근거 표시용 — 기준일까지 발표된 최근 ``n``개 분기 테이블.

    YoY는 usable 리스트 안에서 4분기 전 대비 (evaluate_growth와 동일).
    """
    usable = sorted(
        (q for q in quarters if q.effective_report_date <= as_of),
        key=lambda q: q.fiscal_date,
    )
    eps = [q.eps_actual for q in usable]
    rev = [q.revenue for q in usable]
    rows = []
    for i in range(max(0, len(usable) - n), len(usable)):
        q = usable[i]
        rows.append({
            "분기말": q.fiscal_date.isoformat(),
            "발표일": q.effective_report_date.isoformat(),
            "EPS": q.eps_actual,
            "EPS YoY": _yoy(eps, i),
            "매출": q.revenue,
            "매출 YoY": _yoy(rev, i),
            "순이익률": q.net_margin,
        })
    return rows


# ---------------------------------------------------------------------------
# STAGE 4·5 — VCP 구조 기반 피봇 + 손절선/익절선
# ---------------------------------------------------------------------------

def _swing_points(
    highs: "pd.Series", lows: "pd.Series", reversal: float
) -> list[tuple[int, float, str]]:
    """퍼센트 zigzag — 교대되는 스윙 고점/저점 [(index, price, 'H'|'L')].

    방향이 위일 때 고가 극값을 추적하다 저가가 극값 대비 ``reversal``
    이상 빠지면 스윙 고점 확정(그 반대도 동일). 마지막 미확정 극값은
    포함하지 않는다 — 진행 중인 파동은 호출자가 별도 처리.
    """
    h = highs.to_numpy(dtype=float)
    l = lows.to_numpy(dtype=float)
    pivots: list[tuple[int, float, str]] = []
    direction: str | None = None
    hi_i, hi = 0, h[0]
    lo_i, lo = 0, l[0]
    for i in range(1, len(h)):
        if direction is None:
            if h[i] > hi:
                hi, hi_i = h[i], i
            if l[i] < lo:
                lo, lo_i = l[i], i
            if l[i] <= hi * (1 - reversal):
                pivots.append((hi_i, hi, "H"))
                direction = "down"
                lo, lo_i = l[i], i
            elif h[i] >= lo * (1 + reversal):
                pivots.append((lo_i, lo, "L"))
                direction = "up"
                hi, hi_i = h[i], i
        elif direction == "up":
            if h[i] > hi:
                hi, hi_i = h[i], i
            if l[i] <= hi * (1 - reversal):
                pivots.append((hi_i, hi, "H"))
                direction = "down"
                lo, lo_i = l[i], i
        else:  # down
            if l[i] < lo:
                lo, lo_i = l[i], i
            if h[i] >= lo * (1 + reversal):
                pivots.append((lo_i, lo, "L"))
                direction = "up"
                hi, hi_i = h[i], i
    return pivots


@dataclass(frozen=True)
class PivotLevels:
    pivot: float           # 매수 트리거 가격 (아래 method 참고)
    close: float
    dist_to_pivot: float   # close/pivot − 1 (0이면 피봇에 도달)
    tightness_10d: float   # 최근 10일 (고−저)/고 — <10%면 조밀 (4-6)
    volume_dryup: float    # 10일 평균 거래량 / 50일 평균 (<0.7이면 드라이업, 4-7)
    stop: float            # 손절선 = pivot × (1 − stop_pct)
    target: float          # 익절선 = pivot × (1 + target_pct)
    # --- VCP 구조 분석 (정밀 피봇의 근거) ---
    method: str = "recent-high"        # contraction-high | tight-shelf(cheat) | recent-high
    status: str = ""                   # 피봇 근접 / 베이스 진행 중 / 압축 없음(추격 금지) 등
    contractions: tuple[float, ...] = ()  # 왼쪽부터 각 축소 깊이 (0.18 = −18%)
    contraction_ok: bool | None = None    # 4-4·4-5: 2~6회 & 체감 (None=측정 부족)
    base_depth: float | None = None       # 4-3: 베이스 고점→저점 낙폭
    base_weeks: float | None = None       # 4-2: 베이스 고점 이후 경과 주 수


def pivot_levels(
    df: pd.DataFrame,
    lookback: int = 25,
    stop_pct: float = 0.08,
    target_pct: float = 0.22,
    base_lookback: int = 120,
    reversal: float = 0.04,
    shelf_bars: int = 10,
    shelf_max_range: float = 0.06,
) -> PivotLevels | None:
    """VCP 구조 기반 정밀 피봇 + 손절/익절선.

    책의 정의(피봇 = **마지막 축소 구간의 천장**)를 그대로 계산한다:

    1. 최근 ``base_lookback``(120일 ≈ 24주) 창에서 ``reversal``(4%)
       zigzag로 스윙 고점/저점을 찾고, 베이스 고점 이후의
       고점→저점 파동들 = **축소(T) 시퀀스**를 실측한다 (4-4·4-5).
    2. 피봇 = 마지막 스윙 고점 (마지막 축소의 천장, method
       ``contraction-high``).
    3. 단, 그 아래로 최근 ``shelf_bars``일이 ``shelf_max_range``(6%)
       이내로 뭉친 더 조밀한 선반이 있으면 그 선반의 고가를 피봇으로
       쓴다 — 책의 속임수(cheat) 피봇 (method ``tight-shelf(cheat)``).
    4. 축소가 하나도 없으면(4% 이상 눌림 없는 직선 랠리) 책 기준
       "추격 금지" — 최근 ``lookback``일 고가로 폴백하되 status에
       명시 (method ``recent-high``).

    부산물로 4-2(베이스 주 수)·4-3(베이스 깊이)·4-4/4-5(축소
    횟수·체감)가 함께 판정된다. 최종 확정은 여전히 차트 눈 판독.
    """
    if len(df) < 50:
        return None
    close = float(df["Close"].iloc[-1])
    window = df.tail(base_lookback)
    highs, lows = window["High"], window["Low"]

    # --- 베이스 구조 (4-2, 4-3) ---
    # 베이스 천장은 최근 5일을 제외하고 잡는다 — 오늘 막 돌파해 신고가를
    # 찍은 경우 '오늘'이 베이스 고점이 되어 좌측 구조가 통째로 무시되는
    # 것을 방지 (돌파 진행은 dist>0으로 자연스럽게 드러난다).
    struct = highs.iloc[:-5] if len(highs) > 10 else highs
    base_high = float(struct.max())
    base_high_pos = int(struct.to_numpy().argmax())
    after_high_lows = lows.iloc[base_high_pos:]
    base_low = float(after_high_lows.min())
    base_depth = 1.0 - base_low / base_high if base_high > 0 else None
    base_weeks = (len(window) - 1 - base_high_pos) / 5.0

    # --- 축소 시퀀스 (4-4, 4-5) ---
    swings = _swing_points(highs, lows, reversal)
    contractions: list[float] = []
    last_H: float | None = None
    for j, (idx, price, kind) in enumerate(swings):
        if kind != "H" or idx < base_high_pos:
            continue  # 베이스 고점 이전 파동은 상승 추세의 일부
        last_H = price
        lows_after = lows.iloc[idx + 1:]
        # 다음 스윙 저점(확정) 또는 진행 중 최저가까지의 깊이
        if j + 1 < len(swings) and swings[j + 1][2] == "L":
            trough = swings[j + 1][1]
        elif len(lows_after):
            trough = float(lows_after.min())
        else:
            continue
        contractions.append(1.0 - trough / price)
    # 베이스 고점 자체가 첫 축소의 시작점인데 zigzag가 고점 확정 전이면 보강
    if last_H is None and base_high_pos < len(window) - 1:
        trough = float(after_high_lows.min())
        depth = 1.0 - trough / base_high
        if depth >= reversal:
            last_H = base_high
            contractions.append(depth)

    n = len(contractions)
    if n >= 2:
        # '각 축소는 직전의 약 절반' — 실무 판정: 15% 슬랙의 감소 수열
        decreasing = all(b <= a * 0.85 for a, b in zip(contractions, contractions[1:]))
        contraction_ok = (2 <= n <= 6) and decreasing
    else:
        contraction_ok = None  # 측정 부족 — 판정 불가

    # --- 피봇 결정 ---
    if last_H is not None:
        pivot, method = last_H, "contraction-high"
        shelf = df.tail(shelf_bars)
        shelf_hi = float(shelf["High"].max())
        shelf_lo = float(shelf["Low"].min())
        if (
            shelf_hi > 0
            and (shelf_hi - shelf_lo) / shelf_hi <= shelf_max_range
            and shelf_hi < last_H * 0.97
        ):
            pivot, method = shelf_hi, "tight-shelf(cheat)"
        dist = close / pivot - 1.0
        if dist >= 0.05:
            status = "피봇 위 5%+ 확장 — 추격 주의, 재보합 대기(4-10)"
        elif dist >= -0.05:
            status = "피봇 근접 — 돌파 감시"
        else:
            status = f"베이스 진행 중 — 피봇까지 {-dist:.0%}"
    else:
        pivot, method = float(df["High"].tail(lookback).max()), "recent-high"
        dist = close / pivot - 1.0
        status = "압축 없음(4%+ 눌림 無) — 직선 랠리, 추격 금지·보합 대기"

    hi10 = float(df["High"].tail(10).max())
    lo10 = float(df["Low"].tail(10).min())
    vol10 = float(df["Volume"].tail(10).mean())
    vol50 = float(df["Volume"].tail(50).mean())
    return PivotLevels(
        pivot=pivot,
        close=close,
        dist_to_pivot=dist,
        tightness_10d=(hi10 - lo10) / hi10 if hi10 > 0 else float("nan"),
        volume_dryup=vol10 / vol50 if vol50 > 0 else float("nan"),
        stop=pivot * (1.0 - stop_pct),
        target=pivot * (1.0 + target_pct),
        method=method,
        status=status,
        contractions=tuple(contractions),
        contraction_ok=contraction_ok,
        base_depth=base_depth,
        base_weeks=base_weeks,
    )


# ---------------------------------------------------------------------------
# 종합 점수 — 최종 후보 정렬용 (RS 단일 기준 대신 SEPA 요소 가중 합산)
# ---------------------------------------------------------------------------

# 각 요소의 가중치 (합 100). 측정 불가(None)한 요소는 빼고 남은 가중치로
# 재정규화하므로 펀더멘털 미사용/산업군 미분류여도 0~100 스케일이 유지된다.
COMPOSITE_WEIGHTS: dict[str, float] = {
    "RS": 30.0,          # 1-8 유니버스 백분위
    "성장": 20.0,        # 2-1~2-4 통과 개수
    "코드33": 5.0,       # 2-5 연간 3지표 가속 (보너스)
    "VCP구조": 15.0,     # 4-4·4-5 축소 횟수·체감
    "피봇위치": 15.0,    # 매수 가능 구간인가 (근접=최고, 확장/멀리=감점)
    "조밀도": 5.0,       # 4-6 최근 10일 레인지
    "거래량": 5.0,       # 4-7 드라이업
    "산업군": 5.0,       # 3-5 생존자 집중도
}


@dataclass(frozen=True)
class CompositeScore:
    total: float                  # 0~100
    parts: dict[str, float | None]  # 요소별 0~100 (None = 측정 불가, 제외)


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def composite_score(
    rs: float | None,
    growth: GrowthResult | None,
    pl: PivotLevels,
    industry_survivors: int | None = None,
    weights: dict[str, float] | None = None,
    annual: "AnnualResult | None" = None,
) -> CompositeScore:
    """최종 후보 정렬용 종합 점수 (0~100, 높을수록 '지금 사기 좋은' 셋업).

    RS만으로 정렬하면 이미 5% 이상 확장됐거나 VCP가 없는 종목이 위로
    오므로, 책의 SEPA 요소를 가중 합산한다:

    - RS: 백분위 그대로.
    - 성장: n_passed/4. 데이터 없음(``data_available=False``)·미사용은 제외.
    - 코드33: 연간 3지표 중 통과 수/3. 연간 데이터 없으면 제외.
    - VCP구조: ``contraction_ok`` True 100 / None(측정 부족) 50 / False 20.
      축소가 전혀 없는 직선 랠리(method ``recent-high``)는 0.
    - 피봇위치: −5%~0% 100(돌파 감시 구간), 0~+5% 70(돌파 직후·아직
      매수 가능), +5% 이상 20(추격 금지), −5% 아래는 −20%까지 선형 감소.
    - 조밀도: 10일 레인지 ≤10% 100 → 25%에서 0.
    - 거래량: 10/50일 비율 ≤0.7 100 → 1.5에서 0.
    - 산업군: 생존자 1종목 30 / 2종목 60 / 3종목↑ 100. None이면 제외.
    """
    w = weights or COMPOSITE_WEIGHTS
    parts: dict[str, float | None] = {}

    parts["RS"] = float(rs) if rs is not None else None

    if growth is None or not growth.data_available:
        parts["성장"] = None
    else:
        parts["성장"] = 100.0 * growth.n_passed / 4.0

    n_code33 = sum(1 for v in (annual.checks.values() if annual else ()) if v is not None)
    if annual is None or n_code33 == 0:
        parts["코드33"] = None
    else:
        parts["코드33"] = 100.0 * annual.n_passed / 3.0

    if pl.method == "recent-high":
        parts["VCP구조"] = 0.0
    elif pl.contraction_ok is None:
        parts["VCP구조"] = 50.0
    else:
        parts["VCP구조"] = 100.0 if pl.contraction_ok else 20.0

    d = pl.dist_to_pivot
    if d >= 0.05:
        pivot_pos = 20.0
    elif d >= 0.0:
        pivot_pos = 70.0
    elif d >= -0.05:
        pivot_pos = 100.0
    else:
        pivot_pos = 100.0 * _clamp01(1.0 - (-d - 0.05) / 0.15)
    parts["피봇위치"] = pivot_pos

    t = pl.tightness_10d
    parts["조밀도"] = (
        100.0 * _clamp01((0.25 - t) / 0.15) if t == t else None  # NaN 가드
    )
    v = pl.volume_dryup
    parts["거래량"] = (
        100.0 * _clamp01((1.5 - v) / 0.8) if v == v else None
    )

    if industry_survivors is None:
        parts["산업군"] = None
    elif industry_survivors >= 3:
        parts["산업군"] = 100.0
    elif industry_survivors == 2:
        parts["산업군"] = 60.0
    else:
        parts["산업군"] = 30.0

    used = {k: s for k, s in parts.items() if s is not None and w.get(k, 0) > 0}
    total_w = sum(w[k] for k in used)
    if total_w <= 0:
        return CompositeScore(total=0.0, parts=parts)
    total = sum(s * w[k] for k, s in used.items()) / total_w
    return CompositeScore(total=round(total, 1), parts=parts)


# ---------------------------------------------------------------------------
# 트레이드 리플레이 — 규칙 변형별로 진입~청산을 재생 (45번 페이지)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TradeReplay:
    """피봇 돌파 매수 1건의 재생 결과.

    ``exit_reason``: 손절 | 익절 | 본전청산 | {ma}일선 이탈 | 보유 중 | 미돌파.
    ``ret``는 청산 시 실현 수익률, 보유 중이면 마지막 종가 기준 평가.
    """

    entered: bool
    entry_date: date | None = None
    entry_price: float | None = None
    breakout_volume_ratio: float | None = None  # 돌파일 거래량 / 직전 50일 평균 (4-12)
    exit_date: date | None = None
    exit_price: float | None = None
    exit_reason: str = "미돌파"
    ret: float | None = None
    days_held: int | None = None
    mfe: float | None = None   # 진입 후 최대 상승 (고가 기준)
    mae: float | None = None   # 진입 후 최대 하락 (저가 기준)


def replay_trade(
    df_all: pd.DataFrame,
    as_of: date,
    pivot: float,
    stop_pct: float | None = 0.08,
    target_pct: float | None = 0.22,
    entry_window: int = 20,
    ma_exit: int | None = None,
    breakeven_trigger: float | None = None,
) -> TradeReplay:
    """기준일 이후의 피봇 돌파 매수를 규칙 조합으로 재생한다.

    - 진입: 기준일 이후 ``entry_window``거래일 안에 고가가 피봇에 닿으면
      그날 체결 (시가가 피봇 위면 시가 = 갭 반영).
    - 하루 안의 판정 순서(보수적): ① 손절선(저가) → ② 익절선(고가) →
      ③ ``ma_exit``일선 종가 이탈. 손절·익절 동시 도달은 손절.
    - ``breakeven_trigger``: 종가가 진입가 +trigger 이상을 찍은 다음날부터
      손절선을 본전(진입가)으로 올린다 (책의 '무료 롤').
    - ``stop_pct/target_pct/ma_exit`` 전부 ``None``이면 끝까지 보유
      (존버 벤치마크).

    ``df_all``: 기준일 이전 구간을 **포함**한 일봉 — 이동평균과 돌파일
    거래량 비율(50일 평균 대비)을 look-ahead 없이 계산하기 위함.
    """
    idx = df_all.index
    forward_mask = idx > pd.Timestamp(as_of)
    forward = df_all[forward_mask]
    if forward.empty:
        return TradeReplay(entered=False)

    sma = (
        df_all["Close"].rolling(ma_exit).mean() if ma_exit else None
    )
    vol50 = df_all["Volume"].rolling(50).mean()

    entry_pos_in_all: int | None = None
    n_forward_checked = 0
    for pos in range(len(df_all)):
        if not forward_mask[pos]:
            continue
        n_forward_checked += 1
        if n_forward_checked > entry_window:
            break
        if float(df_all["High"].iloc[pos]) >= pivot:
            entry_pos_in_all = pos
            break
    if entry_pos_in_all is None:
        return TradeReplay(entered=False)

    entry_open = float(df_all["Open"].iloc[entry_pos_in_all])
    entry_price = max(pivot, entry_open)
    entry_date = idx[entry_pos_in_all].date()
    prev_vol50 = float(vol50.iloc[entry_pos_in_all - 1]) if entry_pos_in_all >= 50 else None
    breakout_vol_ratio = (
        float(df_all["Volume"].iloc[entry_pos_in_all]) / prev_vol50
        if prev_vol50 and prev_vol50 > 0 else None
    )

    stop_level = entry_price * (1.0 - stop_pct) if stop_pct is not None else None
    target_level = entry_price * (1.0 + target_pct) if target_pct is not None else None
    breakeven_armed = False

    exit_pos: int | None = None
    exit_price: float | None = None
    exit_reason = "보유 중"
    mfe, mae = 0.0, 0.0
    for pos in range(entry_pos_in_all, len(df_all)):
        row = df_all.iloc[pos]
        hi, lo, cl = float(row["High"]), float(row["Low"]), float(row["Close"])
        mfe = max(mfe, hi / entry_price - 1.0)
        mae = min(mae, lo / entry_price - 1.0)
        effective_stop = stop_level
        if breakeven_armed and (effective_stop is None or entry_price > effective_stop):
            effective_stop = entry_price
        if effective_stop is not None and lo <= effective_stop:
            exit_pos, exit_price = pos, effective_stop
            exit_reason = "본전청산" if breakeven_armed and effective_stop == entry_price else "손절"
            break
        if target_level is not None and hi >= target_level:
            exit_pos, exit_price = pos, target_level
            exit_reason = "익절"
            break
        if sma is not None and pos > entry_pos_in_all:
            ma_val = float(sma.iloc[pos])
            if ma_val == ma_val and cl < ma_val:  # NaN 가드
                exit_pos, exit_price = pos, cl
                exit_reason = f"{ma_exit}일선 이탈"
                break
        if (
            breakeven_trigger is not None
            and not breakeven_armed
            and cl >= entry_price * (1.0 + breakeven_trigger)
        ):
            breakeven_armed = True  # 다음날부터 본전 스톱

    if exit_pos is not None:
        return TradeReplay(
            entered=True, entry_date=entry_date, entry_price=entry_price,
            breakout_volume_ratio=breakout_vol_ratio,
            exit_date=idx[exit_pos].date(), exit_price=exit_price,
            exit_reason=exit_reason, ret=exit_price / entry_price - 1.0,
            days_held=exit_pos - entry_pos_in_all, mfe=mfe, mae=mae,
        )
    last_close = float(df_all["Close"].iloc[-1])
    return TradeReplay(
        entered=True, entry_date=entry_date, entry_price=entry_price,
        breakout_volume_ratio=breakout_vol_ratio,
        exit_reason="보유 중", ret=last_close / entry_price - 1.0,
        days_held=len(df_all) - 1 - entry_pos_in_all, mfe=mfe, mae=mae,
    )


# ---------------------------------------------------------------------------
# 검증용 — 기준일 이후 데이터로 돌파/손절/익절 시뮬레이션
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ForwardOutcome:
    entered: bool
    entry_date: date | None = None
    entry_price: float | None = None
    first_hit: str = "none"        # "stop" | "target" | "none"
    hit_date: date | None = None
    ret_1m: float | None = None    # 진입 후 21/63/126/252거래일 종가 수익률
    ret_3m: float | None = None
    ret_6m: float | None = None
    ret_12m: float | None = None
    max_runup: float | None = None     # 진입 후 12개월 내 최대 상승
    max_drawdown: float | None = None  # 진입 후 12개월 내 최대 하락


def simulate_forward(
    forward_df: pd.DataFrame,
    pivot: float,
    stop_pct: float = 0.08,
    target_pct: float = 0.22,
    entry_window: int = 20,
) -> ForwardOutcome:
    """기준일 이후 일봉으로 '피봇 돌파 매수'를 재현한다.

    - 진입: ``entry_window``거래일 안에 고가가 피봇에 닿으면 그날 체결.
      갭 상승으로 시가가 피봇 위면 시가 체결 (현실 반영).
    - 이후 손절선/익절선 중 **어느 쪽이 먼저** 닿는지 저가/고가로 판정.
      같은 날 둘 다 닿으면 보수적으로 손절로 센다.
    - 수익률(1/3/6/12M)·최대 상승/하락은 진입일 기준.
    """
    if forward_df.empty:
        return ForwardOutcome(entered=False)
    highs = forward_df["High"]
    entry_idx: int | None = None
    for i in range(min(entry_window, len(forward_df))):
        if float(highs.iloc[i]) >= pivot:
            entry_idx = i
            break
    if entry_idx is None:
        return ForwardOutcome(entered=False)
    entry_open = float(forward_df["Open"].iloc[entry_idx])
    entry_price = max(pivot, entry_open)
    entry_date = forward_df.index[entry_idx].date()
    stop_level = entry_price * (1.0 - stop_pct)
    target_level = entry_price * (1.0 + target_pct)

    held = forward_df.iloc[entry_idx:]
    first_hit, hit_date = "none", None
    for ts, row in held.iterrows():
        hit_stop = float(row["Low"]) <= stop_level
        hit_target = float(row["High"]) >= target_level
        if hit_stop:               # 동시 히트는 보수적으로 손절
            first_hit, hit_date = "stop", ts.date()
            break
        if hit_target:
            first_hit, hit_date = "target", ts.date()
            break

    closes = held["Close"]

    def _ret(n: int) -> float | None:
        if len(closes) <= n:
            return None
        return float(closes.iloc[n]) / entry_price - 1.0

    year = held.head(TRADING_DAYS["12m"] + 1)
    return ForwardOutcome(
        entered=True, entry_date=entry_date, entry_price=entry_price,
        first_hit=first_hit, hit_date=hit_date,
        ret_1m=_ret(TRADING_DAYS["1m"]), ret_3m=_ret(TRADING_DAYS["3m"]),
        ret_6m=_ret(TRADING_DAYS["6m"]), ret_12m=_ret(TRADING_DAYS["12m"]),
        max_runup=float(year["High"].max()) / entry_price - 1.0,
        max_drawdown=float(year["Low"].min()) / entry_price - 1.0,
    )
