"""와인스타인 «최적의 타이밍» 종목 스크린 — 기준일 하나로 유니버스 전체를 돌린다 (Streamlit 비의존).

문서 ``docs/strategy_notes/와인스타인_최적의타이밍_체크리스트_2026_10.md`` §E 의 깔때기를
그대로 코드로 옮긴 것이다. TQQQ 검증(``와인스타인_TQQQ_검증_2026_10.md`` §8-3)의 결론
— "책이 수익을 더하는 쪽은 지수가 아니라 **주도 업종의 1→2단계 전환 종목**" — 을 실제
종목 유니버스에서 검증하기 위한 모듈.

    시장 STAGE 0 (SPY·QQQ·ACWI 30주 MA + 유니버스 시장 폭)
    → 업종 STAGE 1 (종목 섹터 → SPDR 섹터 ETF 의 단계·MA·RS)
    → 종목: 주봉 30주 MA · 단계 분류 · 트레이딩 범위(베이스) · **이번 주 저항 돌파**
      (또는 돌파 대기 = 저항 −5% 이내 워치)
    → STAGE 2~5 체크리스트 (:func:`weinstein_stage.evaluate_breakout`) → §7 등급
    → 기준일 이후 데이터로 7·8장 진입·청산 재현 (:func:`simulate_forward`) → 등급별 성과.

단계 분류·베이스·체크리스트 평가는 :mod:`strategy.adapters.weinstein_stage` 를 그대로 쓴다.
데이터 소스는 전부 주입 — composition root(페이지·CLI)가 고른다.

Look-ahead 규칙: 스캔은 기준일까지의 일봉만 본다(``clip_daily``). 사후 검증은 돌파 주
금요일 **다음 거래일 시가**에 진입한다 (문서 §E.7).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Callable

import numpy as np
import pandas as pd

from strategy.adapters.minervini_scan import fetch_with_retry
from strategy.adapters.weinstein_stage import (
    Base,
    Check,
    StageParams,
    add_ma,
    breadth_from_universe,
    classify_stages,
    evaluate_breakout,
    find_base,
    grade,
    mansfield_rs,
    to_weekly,
)

INDEX_SYMBOLS = ("SPY", "QQQ", "ACWI")

# Alpha Vantage OVERVIEW 의 Sector(야후식 11개) → SPDR 섹터 ETF. 책의 '업종'은 이보다 좁지만
# (알루미늄·항공 등) 유니버스 전 종목의 업종 지수를 합성하려면 종목마다 OVERVIEW 호출이
# 필요해 비싸다 → 1차 근사로 섹터 ETF 를 쓴다 (문서 STAGE 1 각주).
SECTOR_ETF: dict[str, str] = {
    "TECHNOLOGY": "XLK",
    "HEALTHCARE": "XLV",
    "FINANCIAL SERVICES": "XLF",
    "CONSUMER CYCLICAL": "XLY",
    "INDUSTRIALS": "XLI",
    "COMMUNICATION SERVICES": "XLC",
    "CONSUMER DEFENSIVE": "XLP",
    "BASIC MATERIALS": "XLB",
    "ENERGY": "XLE",
    "REAL ESTATE": "XLRE",
    "UTILITIES": "XLU",
}

GRADE_ORDER = ["A+", "A", "B", "B-", "C-"]  # 탈락(...) 은 맨 뒤


@dataclass(frozen=True)
class ScanParams:
    mode: str = "breakout"            # breakout = 최근 N주 안에 저항 돌파 | watch = 저항 −watch_pct 이내 대기
    recent_weeks: int = 1             # breakout: 이번 주만(1) 또는 최근 N주
    watch_pct: float = 0.05
    min_price: float = 10.0           # 4-6 유동성 (책: "거래량 적은 종목 회피")
    min_dollar_vol_m: float = 5.0
    lookback_years: int = 11          # 10년 저항(3-5) + 30주 MA
    forward_days: int = 400           # 사후 검증용 조회 범위
    include_partial_week: bool = True # 기준일이 금요일 마감 전이면 진행 중인 주봉을 포함할지
    stage: StageParams = field(default_factory=StageParams)


@dataclass
class Weekly:
    """한 심볼의 주봉 묶음 — OHLCV + MA/slope/above/ext, 단계, 맨스필드 RS."""

    w: pd.DataFrame
    stages: pd.Series
    rs: pd.Series


@dataclass
class Setup:
    week_i: int        # 돌파 주 (watch 는 현재 주) 의 iloc 위치
    base: Base
    kind: str          # "breakout" | "watch"


@dataclass
class Candidate:
    ticker: str
    kind: str
    week_end: pd.Timestamp
    base: Base
    checks: list[Check]
    grade: str                 # 시장·업종 포함 최종 (문서 §7)
    stock_grade: str           # 0-x·1-x 제외 — 종목 자체 등급 (대조군 분석용)
    close: float
    resistance: float
    support: float
    vol_ratio: float
    rs: float
    stage_now: int
    entry_type: str            # "1→2" | "2단계 지속"
    sector: str | None
    sector_etf: str | None
    weekly: Weekly
    entry_ok: bool = False
    entry_note: str = ""

    @property
    def aplus_flag(self) -> bool:
        return any(c.code == "3-5" and c.passed for c in self.checks)


@dataclass
class ScanResult:
    as_of: date
    week_end: pd.Timestamp | None
    partial_week: bool
    market_checks: list[Check]
    market_pass: bool | None
    breadth: pd.DataFrame | None
    funnel: dict[str, int]
    candidates: list[Candidate]
    index_weekly: dict[str, Weekly]
    sector_weekly: dict[str, Weekly]
    frames: dict[str, pd.DataFrame]        # 후보 티커의 일봉 (기준일 이후 포함 — 사후 검증용)
    errors: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# 데이터 준비
# --------------------------------------------------------------------------- #
def clip_daily(df: pd.DataFrame, as_of: date, completed_weeks_only: bool = False) -> pd.DataFrame:
    """tz 제거·정규화 후 기준일까지만. ``completed_weeks_only`` 면 마지막 금요일 마감분까지만."""
    out = df.copy()
    idx = pd.to_datetime(out.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    out.index = idx.normalize()
    out = out[out.index <= pd.Timestamp(as_of)]
    if completed_weeks_only and len(out) and out.index[-1].weekday() != 4:
        last_fri = out.index[-1] - pd.Timedelta(days=(out.index[-1].weekday() - 4) % 7)
        out = out[out.index <= last_fri]
    return out


def prepare_weekly(daily: pd.DataFrame, p: StageParams, index_close: pd.Series | None) -> Weekly | None:
    """일봉 → 주봉 + MA + 단계 + RS. 30주 MA 가 5주 미만이면 None."""
    if len(daily) < 60:
        return None
    w = add_ma(to_weekly(daily), p)
    if w["MA"].notna().sum() < 5:
        return None
    stages = classify_stages(w, p)
    if index_close is not None:
        rs = mansfield_rs(w["Close"], index_close, p.rs_window)
    else:
        rs = pd.Series(np.nan, index=w.index)
    return Weekly(w=w, stages=stages, rs=rs)


def find_setup(wk: Weekly, p: ScanParams) -> Setup | None:
    """돌파(최근 ``recent_weeks`` 주 중 가장 최근) 또는 워치(저항 바로 아래) 셋업."""
    w = wk.w
    n = len(w)
    if n < 40:
        return None
    if p.mode == "watch":
        i = n - 1
        base = find_base(w, i + 1, p.stage)          # 현재 주를 베이스에 포함
        if base is None:
            return None
        close = float(w["Close"].iloc[i])
        if base.high * (1 - p.watch_pct) <= close <= base.high and int(wk.stages.iloc[i]) in (1, 2):
            return Setup(i, base, "watch")
        return None
    for i in range(n - 1, max(n - 1 - p.recent_weeks, 0), -1):
        base = find_base(w, i, p.stage)
        if base is None:
            continue
        if float(w["Close"].iloc[i]) > base.high >= float(w["Close"].iloc[i - 1]):
            return Setup(i, base, "breakout")
    return None


# --------------------------------------------------------------------------- #
# STAGE 0 / 1
# --------------------------------------------------------------------------- #
def _last(wk: Weekly, d: pd.Timestamp):
    return wk.w.loc[:d].iloc[-1]


def market_checks(idx: dict[str, Weekly], breadth: pd.DataFrame | None, d: pd.Timestamp) -> dict:
    """문서 STAGE 0 — 0-1 필수, 0-2~0-7 정보/경고. (0-8 P/D 는 외부 데이터라 제외)"""
    checks: list[Check] = []
    spy, qqq = idx.get("SPY"), idx.get("QQQ")
    if spy is None or qqq is None:
        return {"checks": checks, "pass": None}
    s, q = _last(spy, d), _last(qqq, d)
    ok01 = bool(s.above and s.slope >= 0 and q.above and q.slope >= 0)
    checks.append(Check(
        "0-1", "지수(SPY·QQQ) > 30주 MA & MA 상승",
        f"SPY {s.Close:.0f}/{s.MA:.0f} slope {s.slope * 100:+.2f}% · QQQ {q.Close:.0f}/{q.MA:.0f} slope {q.slope * 100:+.2f}%",
        ok01, True, "" if ok01 else "하락세 시장 — 매수 금지 규칙 1"))
    st = int(spy.stages.loc[:d].iloc[-1])
    checks.append(Check("0-2", "지수(SPY) 단계 ∈ {1, 2}", f"{st}단계", st in (1, 2), False))
    if breadth is not None and len(breadth.loc[:d]):
        b = breadth.loc[:d]
        br = b.iloc[-1]
        br8 = b.iloc[-9] if len(b) > 9 else br
        if not np.isnan(br.pct_stage12):
            checks.append(Check("0-3", "유니버스 1·2단계 종목 비율 ≥ 50%",
                                f"{br.pct_stage12 * 100:.0f}% (8주 전 {br8.pct_stage12 * 100:.0f}%, 2단계만 {br.pct_stage2 * 100:.0f}%)",
                                bool(br.pct_stage12 >= 0.5), False, "지수 하락 중 비율 상승 = 강세 다이버전스(1982)"))
        spy26 = spy.w.loc[:d]["Close"].iloc[-26:]
        at_high = bool(s.Close >= spy26.max() * 0.99)
        ad26 = b["ad_line"].iloc[-26:]
        ad_at_high = bool(br.ad_line >= ad26.max() - 0.01 * abs(ad26.max()) - 50)
        checks.append(Check("0-4", "A/D선 약세 다이버전스 없음",
                            f"지수 26주 고점권 {at_high} / A/D 26주 고점권 {ad_at_high}",
                            not (at_high and not ad_at_high), False, "지수 신고점인데 A/D 미동반이면 경고(1972→73)"))
        if not np.isnan(br.mi):
            mi = b["mi"].dropna()
            cross = ""
            if len(mi) > 13:
                sgn = np.sign(mi.iloc[-13:])
                if sgn.iloc[-1] > 0 and sgn.min() < 0:
                    cross = "최근 13주 내 −→+ 교차(장기 강세)"
                elif sgn.iloc[-1] < 0 and sgn.max() > 0:
                    cross = "최근 13주 내 +→− 교차(하락 징후)"
            checks.append(Check("0-5", "탄력지수 MI(순상승 200일 평균) > 0", f"{br.mi:+.1f} {cross}", bool(br.mi > 0), False))
        nh8 = b["nh_nl"].iloc[-8:]
        slope = float(np.polyfit(range(len(nh8)), nh8.values, 1)[0]) if len(nh8) >= 4 else float("nan")
        checks.append(Check("0-6", "주간 신고−신저 > 0 & 방향", f"{br.nh_nl:+.0f} (8주 기울기 {slope:+.1f}/주)",
                            bool(br.nh_nl > 0), False, "지수 상승 중 지표 하락 = 표면 아래 부패(1987 G)"))
    acwi = idx.get("ACWI")
    if acwi is not None and len(acwi.w.loc[:d]):
        a = _last(acwi, d)
        if not np.isnan(a.MA):
            checks.append(Check("0-7", "세계 증시(ACWI) > 30주 MA & 상승", f"{a.Close:.1f}/{a.MA:.1f} slope {a.slope * 100:+.2f}%",
                                bool(a.above and a.slope >= 0), False))
    m = d.month
    checks.append(Check("0-9", "계절·선거주기",
                        f"{m}월{' (12월 최고)' if m == 12 else ' (9월 최악)' if m == 9 else ''}, 선거 다음 해 {(d.year % 4) == 1}", None))
    return {"checks": checks, "pass": ok01}


def sector_checks(sec: Weekly | None, etf: str | None, sector: str | None, d: pd.Timestamp) -> dict:
    """문서 STAGE 1 — 섹터 ETF 의 단계·MA·RS. 섹터를 모르면 정보 없음(판정 보류, 차단 아님)."""
    label = f"{sector or '?'}→{etf or '?'}"
    if sec is None or etf is None or not len(sec.w.loc[:d]):
        checks = [Check("1-1", "업종 단계 ∈ {1, 2}", f"{label}: 섹터 정보 없음", None, True, "펀더멘털 소스 없음/미분류 — 수동 확인"),
                  Check("1-2", "업종 지수 > 30주 MA & 상승", "n/a", None, True),
                  Check("1-3", "업종 RS(vs SPY 맨스필드) > 0", "n/a", None, True)]
        return {"checks": checks, "pass": None}
    r = _last(sec, d)
    st = int(sec.stages.loc[:d].iloc[-1])
    rs = sec.rs.loc[:d].dropna()
    rs_now = float(rs.iloc[-1]) if len(rs) else float("nan")
    rs_prev = float(rs.iloc[-14]) if len(rs) > 14 else float("nan")
    checks = [
        Check("1-1", "업종 단계 ∈ {1, 2}", f"{label}: {st}단계", st in (1, 2), True,
              "업종 2단계 초기 돌파 + 종목 돌파 동시 = 최상"),
        Check("1-2", "업종 지수 > 30주 MA & 상승", f"{etf} {r.Close:.1f}/{r.MA:.1f} slope {r.slope * 100:+.2f}%",
              bool(r.above and r.slope >= 0), True, "MA 아래에서의 돌파는 가짜(엔터테인먼트 1980s)"),
        Check("1-3", "업종 RS(vs SPY 맨스필드) > 0", f"{rs_prev:.2f} → {rs_now:.2f}" if not np.isnan(rs_now) else "n/a",
              bool(rs_now > 0) if not np.isnan(rs_now) else None, True,
              "상승 중" if (not np.isnan(rs_prev) and rs_now >= rs_prev) else "하락 중(경고)"),
    ]
    return {"checks": checks, "pass": all(c.passed for c in checks if c.required and c.passed is not None)}


# --------------------------------------------------------------------------- #
# 스캔
# --------------------------------------------------------------------------- #
def _grade_key(g: str) -> int:
    return GRADE_ORDER.index(g) if g in GRADE_ORDER else len(GRADE_ORDER)


def run_scan(
    as_of: date,
    tickers: list[str],
    market_data,
    params: ScanParams = ScanParams(),
    growth_adapters: list | None = None,
    max_workers: int = 4,
    log: Callable[[str], None] = lambda _m: None,
) -> ScanResult:
    p, sp = params, params.stage
    completed_only = not p.include_partial_week
    start = as_of - timedelta(days=365 * p.lookback_years)
    end = min(as_of + timedelta(days=p.forward_days), date.today())
    errors: list[str] = []

    def _fetch(sym: str) -> pd.DataFrame | None:
        try:
            return fetch_with_retry(market_data, sym, start, end)
        except Exception as e:  # noqa: BLE001 — 한 종목 실패는 건너뛴다
            errors.append(f"{sym}: {type(e).__name__}")
            return None

    # --- 지수 ---
    index_weekly: dict[str, Weekly] = {}
    last_bar: pd.Timestamp | None = None
    for sym in INDEX_SYMBOLS:
        df = _fetch(sym)
        if df is None:
            continue
        d = clip_daily(df, as_of, completed_only)
        if sym == "SPY" and len(d):
            last_bar = d.index[-1]
        wk = prepare_weekly(d, sp, None)
        if wk is not None:
            index_weekly[sym] = wk
    spy = index_weekly.get("SPY")
    spy_close = spy.w["Close"] if spy is not None else None
    for sym in ("QQQ", "ACWI"):
        if sym in index_weekly and spy_close is not None:
            index_weekly[sym].rs = mansfield_rs(index_weekly[sym].w["Close"], spy_close, sp.rs_window)
    if spy is None:
        return ScanResult(as_of, None, False, [], None, None, {"유니버스": len(tickers)}, [], index_weekly, {}, {},
                          errors + ["SPY 조회 실패 — 시장 판정 불가"])
    week_end = spy.w.index[-1]
    partial = bool(last_bar is not None and week_end > last_bar)   # 주봉 라벨(금요일) > 마지막 일봉 = 진행 중

    # --- 유니버스 일봉 (병렬) ---
    raw: dict[str, pd.DataFrame] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for i, (t, df) in enumerate(zip(tickers, ex.map(_fetch, tickers))):
            if df is not None:
                raw[t] = df
            if (i + 1) % 100 == 0:
                log(f"  일봉 {i + 1}/{len(tickers)}")

    # --- 종목별 주봉·단계·셋업 ---
    clipped: dict[str, pd.DataFrame] = {}
    stages_of: dict[str, pd.Series] = {}
    weeklies: dict[str, Weekly] = {}
    setups: dict[str, Setup] = {}
    n_liquid = n_weekly = 0
    for t, df in raw.items():
        d = clip_daily(df, as_of, completed_only)
        if len(d) < 250:
            continue
        close = float(d["Close"].iloc[-1])
        dollar_vol = float((d["Close"] * d["Volume"]).tail(20).mean())
        if close < p.min_price or dollar_vol < p.min_dollar_vol_m * 1e6:
            continue
        n_liquid += 1
        wk = prepare_weekly(d, sp, spy_close)
        if wk is None:
            continue
        n_weekly += 1
        clipped[t] = d
        stages_of[t] = wk.stages
        weeklies[t] = wk
        st = find_setup(wk, p)
        if st is not None:
            setups[t] = st
    log(f"  유동성 {n_liquid} → 주봉 {n_weekly} → 셋업 {len(setups)}")

    breadth = breadth_from_universe(clipped, sp, stages=stages_of) if clipped else None
    market = market_checks(index_weekly, breadth, week_end)

    # --- 업종: 셋업 종목만 섹터 조회 (캐시) ---
    sector_of: dict[str, str | None] = {}
    if growth_adapters:
        for t in setups:
            sec = None
            for a in growth_adapters:
                try:
                    snap = a.fetch(t)
                except Exception:  # noqa: BLE001
                    snap = None
                if snap is not None and getattr(snap, "sector", None):
                    sec = str(snap.sector).upper()
                    break
            sector_of[t] = sec
    sector_weekly: dict[str, Weekly] = {}
    for etf in {SECTOR_ETF.get(s) for s in sector_of.values() if s} - {None}:
        df = _fetch(etf)
        if df is None:
            continue
        wk = prepare_weekly(clip_daily(df, as_of, completed_only), sp, spy_close)
        if wk is not None:
            sector_weekly[etf] = wk

    # --- 체크리스트 · 등급 ---
    cands: list[Candidate] = []
    for t, st in setups.items():
        wk = weeklies[t]
        sec = sector_of.get(t)
        etf = SECTOR_ETF.get(sec) if sec else None
        sc = sector_checks(sector_weekly.get(etf), etf, sec, week_end)
        checks = evaluate_breakout(wk.w, wk.stages, wk.rs, st.week_i, st.base, sp, market=market, sector=sc)
        skip = () if st.kind == "breakout" else ("3-2", "4-1")   # 워치: 돌파·거래량은 아직 모름
        g = grade([c for c in checks if c.code not in skip])
        sg = grade([c for c in checks if c.code not in skip and c.code[0] not in "01"])
        cmap = {c.code: c for c in checks}
        row = wk.w.iloc[st.week_i]
        try:
            vol_ratio = float(cmap["4-1"].value.rstrip("x"))
        except (KeyError, ValueError):
            vol_ratio = float("nan")
        was_stage1 = bool((wk.stages.iloc[max(0, st.week_i - 8): st.week_i] == 1).any())
        cands.append(Candidate(
            ticker=t, kind=st.kind, week_end=wk.w.index[st.week_i], base=st.base, checks=checks,
            grade=g, stock_grade=sg, close=float(row.Close), resistance=st.base.high, support=st.base.low,
            vol_ratio=vol_ratio, rs=float(wk.rs.iloc[st.week_i]) if not np.isnan(wk.rs.iloc[st.week_i]) else float("nan"),
            stage_now=int(wk.stages.iloc[st.week_i]), entry_type="1→2" if was_stage1 else "2단계 지속",
            sector=sec, sector_etf=etf, weekly=wk,
        ))
    cands.sort(key=lambda c: (_grade_key(c.grade), -(c.vol_ratio if not np.isnan(c.vol_ratio) else 0.0)))
    # 책: "A 이상만 산다" — 시장·업종 포함 최종 등급 기준
    for c in cands:
        c.entry_ok = c.kind == "breakout" and c.grade in ("A+", "A")
        c.entry_note = ("✅ 매수 자격 (buy-stop 체결 대상)" if c.entry_ok else
                        "👀 워치 — 돌파 대기" if c.kind == "watch" else
                        f"⛔ {c.grade}" if c.grade.startswith("탈락") else f"👀 {c.grade} — 아주 좋은 것만 산다")

    funnel = {
        "유니버스": len(tickers), "조회 실패": len(tickers) - len(raw),
        "유동성 통과": n_liquid, "주봉 판정 가능": n_weekly,
        ("돌파 셋업" if p.mode == "breakout" else "워치 셋업"): len(setups),
        "A 이상": sum(c.grade in ("A+", "A") for c in cands),
    }
    return ScanResult(as_of=as_of, week_end=week_end, partial_week=partial, market_checks=market["checks"],
                      market_pass=market["pass"], breadth=breadth, funnel=funnel, candidates=cands,
                      index_weekly=index_weekly, sector_weekly=sector_weekly,
                      frames={c.ticker: raw[c.ticker] for c in cands}, errors=errors)


# --------------------------------------------------------------------------- #
# 사후 검증 — 7·8장 진입·청산 재현
# --------------------------------------------------------------------------- #
@dataclass
class Outcome:
    entered: bool
    entry_date: date | None = None
    entry_px: float | None = None
    exit_date: date | None = None
    exit_px: float | None = None
    reason: str = ""
    ret: float | None = None
    weeks: int = 0
    mfe: float | None = None        # 보유 중 최대 상승 (주봉 고가 기준)
    mae: float | None = None        # 보유 중 최대 하락 (주봉 저가 기준)
    ret_13w: float | None = None
    ret_26w: float | None = None
    ret_52w: float | None = None
    pullback_fill: bool | None = None   # 7-3 풀백 2차 매수 조건이 6주 안에 발생했는가
    vol_2w_ratio: float | None = None   # 4-2 돌파 후 2주 평균 거래량 / 돌파 주


def simulate_forward(
    daily_full: pd.DataFrame,
    cand: Candidate,
    p: StageParams,
    slip: float = 0.005,
    max_weeks: int = 52,
    stop_pct: float = 0.05,
) -> Outcome:
    """돌파 주 다음 거래일 시가 100% 진입 → 주봉 규칙으로 청산.

    7-5 돌파 후 6주 내 종가 < 돌파가 · 7-7 스톱(돌파가 −5%) · 8-3 종가 < MA & 기울기 ≤ 0
    · 8-4 4단계 또는 지지선 이탈. ``max_weeks`` 넘으면 그 주 종가로 평가(open).
    책의 50/50 분할(7-2·7-3)은 수익률 계산을 단순화하려고 100% 진입으로 두고,
    풀백 조건 발생 여부만 ``pullback_fill`` 로 기록한다.
    """
    d = daily_full.copy()
    idx = pd.to_datetime(d.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    d.index = idx.normalize()
    w = add_ma(to_weekly(d), p)
    stages = classify_stages(w, p)
    after = d[d.index > cand.week_end]
    if after.empty:
        return Outcome(entered=False)
    entry_date = after.index[0]
    entry_px = float(after["Open"].iloc[0]) * (1 + slip)
    wi = int(w.index.get_indexer([cand.week_end])[0])
    if wi < 0:
        return Outcome(entered=False)
    res, sup = cand.resistance, cand.support
    bo_vol = float(w["Volume"].iloc[wi])
    stop = res * (1 - stop_pct)
    mfe = mae = 0.0
    pullback = False
    reason, exit_week = "", None
    for k in range(1, max_weeks + 1):
        j = wi + k
        if j >= len(w):
            break
        row = w.iloc[j]
        mfe = max(mfe, float(row.High) / entry_px - 1)
        mae = min(mae, float(row.Low) / entry_px - 1)
        if k <= 6 and bo_vol > 0 and (row.Low <= res * 1.02 and row.Close > res and row.Close > row.MA
                                      and row.Volume <= 0.6 * bo_vol):
            pullback = True
        st = int(stages.iloc[j])
        if k <= 6 and row.Close < res:
            reason = "7-5 돌파 후 6주 내 종가 < 돌파가 (가짜 돌파)"
        elif row.Close < stop:
            reason = f"7-7 스톱 (돌파가 −{stop_pct:.0%})"
        elif (not row.above) and row.slope <= 0:
            reason = "8-3 종가 < 30주 MA & MA 평평/하락"
        elif st == 4 or row.Close < sup:
            reason = "8-4 4단계 / 지지선 이탈"
        if reason:
            exit_week = j
            break

    def _ret_at(n: int):
        j = wi + n
        return float(w["Close"].iloc[j]) / entry_px - 1 if j < len(w) else None

    vol2 = w["Volume"].iloc[wi + 1: wi + 3]
    vol_2w = float(vol2.mean() / bo_vol) if len(vol2) == 2 and bo_vol > 0 else None
    if exit_week is None:
        last_j = min(wi + max_weeks, len(w) - 1)
        exit_px = float(w["Close"].iloc[last_j])
        return Outcome(True, entry_date.date(), round(entry_px, 4), w.index[last_j].date(), round(exit_px, 4),
                       "open" if last_j < wi + max_weeks else "time", exit_px / entry_px - 1, last_j - wi,
                       mfe, mae, _ret_at(13), _ret_at(26), _ret_at(52), pullback, vol_2w)
    nxt = d[d.index > w.index[exit_week]]
    exit_px = (float(nxt["Open"].iloc[0]) if len(nxt) else float(w["Close"].iloc[exit_week])) * (1 - slip)
    exit_date = nxt.index[0].date() if len(nxt) else w.index[exit_week].date()
    return Outcome(True, entry_date.date(), round(entry_px, 4), exit_date, round(exit_px, 4), reason,
                   exit_px / entry_px - 1, exit_week - wi, mfe, mae, _ret_at(13), _ret_at(26), _ret_at(52),
                   pullback, vol_2w)


def regrade_volume_optional(checks: list[Check]) -> str:
    """4-1 거래량을 정보 항목으로 두고 다시 매긴 등급 — 대형주용 보정.

    S&P 대형주는 돌파 주 거래량이 4주 평균 2배를 넘는 일이 드물어(검증: 55건 중 3건)
    책 그대로면 A 등급이 거의 안 나온다. 저항·RS·베이스만으로 등급을 보려는 용도.
    """
    from dataclasses import replace as _replace
    return grade([_replace(c, required=False) if c.code == "4-1" else c for c in checks])


def grade_bucket(g: str) -> str:
    """'탈락(2-10,4-1)' 같은 상세 등급을 거친 묶음으로 — 성과표 행이 쪼개지지 않게."""
    return "탈락" if g.startswith("탈락") else g


def grade_stats(rows: list[dict], key: str = "grade") -> pd.DataFrame:
    """등급(또는 다른 키)별 사후 성과 — 문서 §E.3 '등급 체계가 변별력 있는가'."""
    df = pd.DataFrame(rows)
    if df.empty or "ret" not in df:
        return pd.DataFrame()
    if key == "grade":
        df = df.assign(grade=df["grade"].map(grade_bucket))
    ent = df[df["entered"] == True]  # noqa: E712
    out = []
    for g, grp in ent.groupby(key):
        r = grp["ret"].dropna()
        out.append({
            key: g, "진입": len(grp), "승률": float((r > 0).mean()) if len(r) else np.nan,
            "평균": float(r.mean()) if len(r) else np.nan, "중앙값": float(r.median()) if len(r) else np.nan,
            "평균 보유주": float(grp["weeks"].mean()), "7-5 실패": int(grp["reason"].str.startswith("7-5").sum()),
            "평균 MFE": float(grp["mfe"].mean()), "평균 MAE": float(grp["mae"].mean()),
        })
    res = pd.DataFrame(out)
    res["_k"] = res[key].map(lambda g: _grade_key(str(g)) if key == "grade" else 0)
    return res.sort_values(["_k", key]).drop(columns="_k").reset_index(drop=True)


def check_stats(rows: list[dict]) -> pd.DataFrame:
    """항목별 예측력 — 각 체크 코드를 통과한 거래 vs 미통과 거래의 사후 수익률.

    ``rows`` 의 각 항목에 ``checks``(``{code: passed}`` 또는 Check 리스트) 가 있어야 한다.
    등급표가 '어떤 조합이 좋았나'를 보여준다면 이 표는 '어느 규칙 하나가 성과를 가르나'를 본다.
    """
    recs = []
    for r in rows:
        if not r.get("entered") or r.get("ret") is None:
            continue
        chk = r.get("checks") or {}
        if not isinstance(chk, dict):
            chk = {c.code: c.passed for c in chk}
        for code, passed in chk.items():
            if passed is None:
                continue
            recs.append({"code": code, "passed": bool(passed), "ret": r["ret"]})
    df = pd.DataFrame(recs)
    if df.empty:
        return df
    out = []
    for code, g in df.groupby("code"):
        p, f = g[g.passed]["ret"], g[~g.passed]["ret"]
        out.append({"항목": code, "통과 n": len(p), "통과 평균": p.mean() if len(p) else np.nan,
                    "통과 승률": (p > 0).mean() if len(p) else np.nan,
                    "미통과 n": len(f), "미통과 평균": f.mean() if len(f) else np.nan,
                    "미통과 승률": (f > 0).mean() if len(f) else np.nan})
    res = pd.DataFrame(out)
    res["차이"] = res["통과 평균"] - res["미통과 평균"]
    return res.sort_values("항목").reset_index(drop=True)
