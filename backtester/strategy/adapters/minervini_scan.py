"""미너비니 SEPA 스크린 1회 실행 — 44번 페이지 파이프라인의 Streamlit 비의존 버전.

44 스크리너 페이지와 **같은 깔때기·같은 기본값**을 코드로 고정해 배치
검증(``minervini_batch_cli.py``)에서 날짜 여러 개를 돌릴 수 있게 한다:

    STAGE 0 시장(SPY·QQQ) → 가격/유동성 프리필터 → 추세 템플릿 1-1~1-7
    → RS 백분위 1-8 → 분기·연간 펀더멘털 → 산업군 → 피봇/손익선
    → 종합점수 정렬 → 시장 상태별 '진입' 허용

데이터 소스(시세·펀더멘털·분할)는 전부 주입받는다 — composition root
(페이지/CLI)가 구현체를 고른다.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Callable

import pandas as pd

from strategy.adapters.minervini_manual_auto import industry_rank
from strategy.adapters.minervini_screener import (
    MIN_BARS,
    MarketVerdict,
    PivotLevels,
    combine_market,
    composite_score,
    evaluate_annual,
    evaluate_growth,
    evaluate_market,
    evaluate_trend_template,
    market_warnings,
    momentum_score,
    pivot_levels,
    rs_percentiles,
    split_factor,
)

INDEX_SYMBOLS = ("SPY", "QQQ")


@dataclass(frozen=True)
class ScreenParams:
    """44 페이지 사이드바 기본값과 동일."""

    min_price: float = 10.0
    min_dollar_vol_m: float = 5.0
    rs_cutoff: float = 70.0
    use_fundamentals: bool = True
    eps_min: float = 0.25
    rev_min: float = 0.20
    growth_required: int = 3
    pivot_lookback: int = 25
    near_pivot_only: bool = True
    stop_pct: float = 0.08
    target_pct: float = 0.22
    stage0_gate: bool = True           # 0-1·0-1b 실패 시 스캔 생략
    caution_max_entries: int = 3       # 0-2 경고 시 품질 통과 중 RS 상위 N개
    entry_rs_min: float = 90.0         # 진입 품질 필터 RS 하한
    short_slope_days: int = 20         # 0-1b 50일선 기울기 비교 기간 (OOS 검증으로 10→20)
    breadth_rule: str = "level"        # 0-3: "level"(신고가>신저가) | "ratio_trend"
    breadth_universe: str = "all"      # 0-3 집계 대상: "all"(조회 전 종목) | "liquid"(가격·거래대금 프리필터 통과)


@dataclass
class Candidate:
    ticker: str
    score: float
    rs: float | None
    industry: str
    industry_survivors: int
    pl: PivotLevels
    tt: object
    growth: object | None
    annual: object | None
    snapshot: object | None
    industry_top: bool | None = None   # 3-4 스크린 생존자 내 산업군 RS 1~2위
    entry_ok: bool = True
    entry_note: str = "✅ 진입 가능"


@dataclass
class ScreenResult:
    as_of: date
    market: list[MarketVerdict]
    market_failed: list[str]
    market_warns: list[str]
    nh: int | None = None
    nl: int | None = None
    nh_prev: int | None = None         # 20거래일 전 같은 유니버스의 신고가/신저가 근접 수
    nl_prev: int | None = None
    nh_liq: int | None = None          # 프리필터(최소 주가·거래대금) 통과 종목만의 신고가/신저가 근접 수
    nl_liq: int | None = None
    nh_liq_prev: int | None = None
    nl_liq_prev: int | None = None
    breadth_ok: bool | None = None     # 0-3 (``breadth_rule`` · ``breadth_universe``)
    skipped: bool = False
    funnel: dict[str, int] = field(default_factory=dict)
    candidates: list[Candidate] = field(default_factory=list)
    frames: dict[str, pd.DataFrame] = field(default_factory=dict)
    index_frames: dict[str, pd.DataFrame] = field(default_factory=dict)


def _clip(df: pd.DataFrame, as_of: date) -> pd.DataFrame:
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index = df.index.normalize()
    return df[df.index <= pd.Timestamp(as_of)]


def _is_rate_limited(e: Exception) -> bool:
    return "RateLimit" in type(e).__name__ or "Too Many" in str(e) or "429" in str(e)


def fetch_with_retry(market_data, sym: str, start: date, end: date,
                     retries: int = 4, wait_s: float = 30.0) -> pd.DataFrame:
    """요청 제한(429)이면 ``wait_s``×시도횟수 만큼 쉬고 재시도. 그 외 예외는 그대로."""
    for i in range(retries + 1):
        try:
            return market_data.fetch_ohlcv(sym, start, end)
        except Exception as e:  # noqa: BLE001
            if not _is_rate_limited(e) or i == retries:
                raise
            time.sleep(wait_s * (i + 1))
    raise RuntimeError("unreachable")


@dataclass(frozen=True)
class EntryInput:
    """진입 품질 필터 입력 — 후보 1개."""

    rs: float | None
    growth_full: bool               # 분기 성장 2-1~2-4 전부 (4/4)
    verified: bool                  # 분기 펀더멘털 확인됨 (미확인/미사용이면 False)
    industry_top: bool | None       # 3-4 산업군 1~2위 (None = 미분류)
    flat_box: bool = False          # 가격 고정 — M&A 확정·우선주·ETN 처럼 상승 여지가 없는 종목 (아래 is_flat_box). 책 규칙 아님


FLAT_BOX_RANGE = 0.025     # 최근 10일 고저 폭
FLAT_BOX_DEPTH = 0.03      # 베이스 깊이


def is_flat_box(tight10d: float | None, base_depth: float | None) -> bool:
    """'조밀한 VCP' 로 오인되는 가격 고정 종목 판별 — 미너비니 책 규칙이 아니라 유니버스 정제 규칙.

    VCP 의 "조용함" 은 매도 물량 소진이지만, 인수가에 묶인 종목·우선주·ETN 의 "조용함" 은 구조적으로 오를 수
    없어서 생긴다. 겉모양이 같아 스크리너가 속으므로 가격 움직임으로 걸러낸다.

    2026-10-10 스크리너 61종목 중 SMTI(합병 투표 대기)·DSGR(피인수)·STRC/STRF(MSTR 우선주)·USOI(ETN) 가
    10일 폭 1.2~1.6%, 베이스 깊이 0.2~2% 로 상위에 올라왔다. 진짜 VCP 의 마지막 축소는 3~5% 이고
    베이스 깊이는 10% 이상이므로, **10일 폭 < 2.5% 이고 베이스 깊이 < 3%** 면 주가가 묶인 종목으로 본다.
    """
    if tight10d is None or base_depth is None:
        return False
    return tight10d < FLAT_BOX_RANGE and base_depth < FLAT_BOX_DEPTH


def breadth_ok(nh: int, nl: int, nh_prev: int | None = None, nl_prev: int | None = None,
               rule: str = "level", max_drop: float = 0.15) -> bool:
    """0-3 시장 폭.

    - ``level``: 신고가 근접 > 신저가 근접 (기본, 미너비니 원문).
    - ``ratio_trend``: 신고가 비중 NH/(NH+NL) ≥ 50% **그리고** 20거래일 전보다
      ``max_drop``(15%p) 넘게 줄지 않음 — 비중이 꺾이는 걸 미리 잡으려는 변형.

    2026-10 검증으로 **전체 종목 기준 level 을 진입 차단**으로 유지한다
    (docs/strategy_notes/미너비니_0-3_시장폭_검증_2026-10.md):
    게이트 통과일 1,378일 중 0-3 실패 72일은 이후 3개월 −10% 급락 확률 46% (통과일 15%),
    0-3 이 갈린 날 품질 필터 진입 12건 전부 손절. 유동성 종목 기준은 2022-01-17 을 놓쳤고,
    ratio_trend 는 level 과 판정이 한 번도 다르지 않았다.
    """
    if rule == "level":
        return nh > nl
    r = nh / (nh + nl) if nh + nl else 0.0
    if r < 0.5:
        return False
    if nh_prev is None or nl_prev is None or nh_prev + nl_prev == 0:
        return True
    return r - nh_prev / (nh_prev + nl_prev) >= -max_drop


def entry_permissions(
    items: list[EntryInput], breadth: bool, warns: list[str], max_entries: int,
    rs_min: float = 90.0,
) -> list[tuple[bool, str]]:
    """후보별 진입 허용 — 44 페이지 '진입' 열과 배치가 같은 규칙을 쓴다.

    0. 가격 고정(M&A 확정·우선주·ETN — 상승 여지 없음) → 제외 (시장 상태와 무관하게 먼저 표시, 책 규칙이 아닌 유니버스 정제)
    1. 0-3 시장 폭 실패 → 전부 관찰
    2. 펀더멘털 미확인 → 진입 불가
    3. 품질 필터: RS ≥ ``rs_min`` · 성장 4/4 · 3-4 산업군 1~2위 — 미달은 관찰
    4. 0-2 경고 → 품질 통과 중 RS 상위 ``max_entries``개만

    2026-10 샘플 7개 날짜: 종합점수 상위 3 은 거래당 −1.3%, 품질 필터는 +3.2%.
    """
    out: list[tuple[bool, str]] = []
    passed: list[int] = []
    for i, it in enumerate(items):
        if it.flat_box:
            out.append((False, "⛔ 제외 (가격 고정 — 상승 여지 없음: M&A 확정·우선주·ETN)"))
            continue
        if not breadth:
            out.append((False, "👀 관찰 (0-3 시장 폭 실패)"))
            continue
        if not it.verified:
            out.append((False, "❌ 펀더멘털 미확인 — 진입 불가"))
            continue
        miss = [label for ok, label in (
            ((it.rs or 0.0) >= rs_min, f"RS<{rs_min:g}"),
            (it.growth_full, "성장<4/4"),
            (it.industry_top is True, "산업군 3위↓"),
        ) if not ok]
        if miss:
            out.append((False, f"👀 관찰 (품질 미달: {' · '.join(miss)})"))
            continue
        out.append((True, "✅ 진입 가능"))
        passed.append(i)
    if warns and len(passed) > max_entries:
        keep = set(sorted(passed, key=lambda i: items[i].rs or 0.0, reverse=True)[:max_entries])
        for i in passed:
            if i not in keep:
                out[i] = (False, f"👀 관찰 (경고: RS 상위 {max_entries}개만)")
    return out


def run_screen(
    as_of: date,
    tickers: list[str],
    market_data,
    growth_adapters: list | None = None,
    splits_provider=None,
    params: ScreenParams = ScreenParams(),
    max_workers: int = 2,
    log: Callable[[str], None] = lambda _m: None,
) -> ScreenResult:
    p = params
    fetch_start = as_of - timedelta(days=460)

    # --- STAGE 0 ---
    verdicts: list[MarketVerdict] = []
    index_frames: dict[str, pd.DataFrame] = {}
    for sym in INDEX_SYMBOLS:
        try:
            idf = _clip(fetch_with_retry(market_data, sym, as_of - timedelta(days=700), as_of), as_of)
        except Exception as e:  # noqa: BLE001
            log(f"{sym} 조회 실패: {e}")
            continue
        index_frames[sym] = idf
        v = evaluate_market(idf, symbol=sym, short_slope_days=p.short_slope_days)
        if v is not None:
            verdicts.append(v)
    ok, failed = combine_market(verdicts)
    if len(verdicts) < len(INDEX_SYMBOLS):
        # 지수 판정 불가 상태로 스캔하면 게이트가 조용히 열린다 → 중단 사유로 취급
        failed = failed + [f"STAGE 0 판정 불가 (지수 {len(verdicts)}/{len(INDEX_SYMBOLS)}개만 조회)"]
    warns = market_warnings(verdicts)
    res = ScreenResult(as_of=as_of, market=verdicts, market_failed=failed,
                       market_warns=warns, index_frames=index_frames)
    if p.stage0_gate and failed:
        res.skipped = True
        res.funnel = {"유니버스": len(tickers), "스캔": 0, "최종 후보": 0}
        return res

    # --- STEP 1~3: 가격 (병렬 조회) → 프리필터 → 템플릿 → RS ---
    def _fetch(t: str):
        try:
            return t, _clip(fetch_with_retry(market_data, t, fetch_start, as_of), as_of)
        except Exception:  # noqa: BLE001
            return t, None

    raw: dict[str, pd.DataFrame | None] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for i, (t, df) in enumerate(ex.map(_fetch, tickers)):
            raw[t] = df
            if (i + 1) % 250 == 0:
                log(f"  가격 {i + 1}/{len(tickers)}")

    def _effective_price(t: str, adj: float) -> float:
        if splits_provider is None:
            return adj
        try:
            return adj * split_factor(splits_provider.fetch(t).splits, as_of)
        except Exception:  # noqa: BLE001
            return adj

    fetch_failed = nh = nl = nh_prev = nl_prev = 0
    nh_liq = nl_liq = nh_liq_prev = nl_liq_prev = 0
    scores: dict[str, float] = {}
    prefilter_pass: list[str] = []
    template: dict[str, object] = {}
    frames: dict[str, pd.DataFrame] = {}
    for t in tickers:
        df = raw.get(t)
        if df is None:
            fetch_failed += 1
            continue
        if len(df) < MIN_BARS:
            continue
        close = float(df["Close"].iloc[-1])
        dollar_vol = float((df["Close"] * df["Volume"]).tail(20).mean())
        s = momentum_score(df)
        if s is not None:
            scores[t] = s
        hi52, lo52 = float(df["High"].tail(252).max()), float(df["Low"].tail(252).min())
        is_nh = hi52 > 0 and close >= 0.95 * hi52
        is_nl = not is_nh and close <= 1.05 * lo52
        nh += is_nh
        nl += is_nl
        prev_nh = prev_nl = False
        if len(df) >= 272:  # 20거래일 전 시점의 같은 판정 (0-3 비율 추세용)
            c_prev = float(df["Close"].iloc[-21])
            past = df.iloc[-272:-20]
            prev_nh = c_prev >= 0.95 * float(past["High"].max())
            prev_nl = not prev_nh and c_prev <= 1.05 * float(past["Low"].min())
            nh_prev += prev_nh
            nl_prev += prev_nl
        if dollar_vol < p.min_dollar_vol_m * 1e6:
            continue
        if close < p.min_price and _effective_price(t, close) < p.min_price:
            continue
        nh_liq += is_nh
        nl_liq += is_nl
        nh_liq_prev += prev_nh
        nl_liq_prev += prev_nl
        prefilter_pass.append(t)
        frames[t] = df
        tt = evaluate_trend_template(df)
        if tt is not None:
            template[t] = tt
    rs = rs_percentiles(scores)
    template_pass = [t for t in prefilter_pass if t in template and template[t].passed]
    rs_pass = [t for t in template_pass if rs.get(t, 0.0) >= p.rs_cutoff]
    log(f"  프리필터 {len(prefilter_pass)} → 템플릿 {len(template_pass)} → RS {len(rs_pass)}")

    # --- STEP 4: 펀더멘털 (순차 — Alpha Vantage 레이트리밋) ---
    growth: dict[str, object] = {}
    annual: dict[str, object] = {}
    snaps: dict[str, object] = {}
    if p.use_fundamentals and growth_adapters:
        for t in rs_pass:
            snap = None
            for a in growth_adapters:
                try:
                    snap = a.fetch(t)
                except Exception:  # noqa: BLE001
                    snap = None
                if snap is not None and snap.quarters:
                    break
            snaps[t] = snap
            growth[t] = evaluate_growth(
                snap.quarters if snap else [], as_of,
                eps_min_yoy=p.eps_min, rev_min_yoy=p.rev_min, n_required=p.growth_required,
            )
            annual[t] = evaluate_annual(snap.annuals if snap else [], as_of)

    def _growth_ok(t: str) -> bool:
        g = growth.get(t)
        return g is None or g.passed or not g.data_available

    growth_pass = [t for t in rs_pass if _growth_ok(t)]

    # --- STEP 5~6: 산업군 + 피봇 + 종합점수 ---
    industry_of = {t: (getattr(snaps.get(t), "industry", None) or "미분류") for t in growth_pass}
    ind_counts = pd.Series(list(industry_of.values()), dtype=object).value_counts()
    cands: list[Candidate] = []
    for t in growth_pass:
        pl = pivot_levels(frames[t], lookback=p.pivot_lookback,
                          stop_pct=p.stop_pct, target_pct=p.target_pct)
        if pl is None or (p.near_pivot_only and pl.dist_to_pivot < -0.05):
            continue
        ind = industry_of[t]
        n_ind = int(ind_counts.get(ind, 1))
        cs = composite_score(rs.get(t), growth.get(t), pl,
                             industry_survivors=None if ind == "미분류" else n_ind,
                             annual=annual.get(t))
        cands.append(Candidate(
            ticker=t, score=cs.total, rs=rs.get(t), industry=ind,
            industry_survivors=n_ind, pl=pl, tt=template[t],
            growth=growth.get(t), annual=annual.get(t), snapshot=snaps.get(t),
        ))
    cands.sort(key=lambda c: (c.score, c.rs or 0.0), reverse=True)
    ranks = industry_rank([(c.ticker, c.industry, c.rs) for c in cands])
    for c in cands:
        c.industry_top = ranks.get(c.ticker, (None, ""))[0]
    if p.breadth_universe == "liquid":
        b_ok = breadth_ok(nh_liq, nl_liq, nh_liq_prev, nl_liq_prev, rule=p.breadth_rule)
    else:
        b_ok = breadth_ok(nh, nl, nh_prev, nl_prev, rule=p.breadth_rule)
    inputs = [EntryInput(
        rs=c.rs,
        growth_full=bool(c.growth is not None and c.growth.data_available and c.growth.n_passed == 4),
        verified=bool(c.growth is not None and c.growth.data_available),
        industry_top=c.industry_top,
        flat_box=is_flat_box(c.pl.tightness_10d, c.pl.base_depth),
    ) for c in cands]
    for c, (ok_, note) in zip(cands, entry_permissions(inputs, b_ok, warns,
                                                      p.caution_max_entries, p.entry_rs_min)):
        c.entry_ok, c.entry_note = ok_, note

    res.nh, res.nl, res.nh_prev, res.nl_prev, res.breadth_ok = nh, nl, nh_prev, nl_prev, b_ok
    res.nh_liq, res.nl_liq, res.nh_liq_prev, res.nl_liq_prev = nh_liq, nl_liq, nh_liq_prev, nl_liq_prev
    res.candidates = cands
    res.frames = {c.ticker: frames[c.ticker] for c in cands}
    res.funnel = {
        "유니버스": len(tickers), "조회 실패": fetch_failed,
        "가격/유동성 통과": len(prefilter_pass), "추세 템플릿 통과": len(template_pass),
        f"RS≥{p.rs_cutoff:g} 통과": len(rs_pass), "성장 통과": len(growth_pass),
        "최종 후보": len(cands),
    }
    return res
