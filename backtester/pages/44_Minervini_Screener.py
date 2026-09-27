"""미너비니 SEPA 자동 스크리너 — 기준일 하나로 매수 후보 리스트 뽑기.

43번 페이지(수동 체크리스트)의 STEP 0~5 깔때기를 코드로 돌린다:
유니버스 → 가격/유동성 프리필터 → 추세 템플릿(1-1~1-7) → RS 백분위
→ 분기 EPS·매출 성장(발표일 기준 point-in-time) → 산업군 집계 →
피봇 + 손절선/익절선. 마지막 VCP 눈 판독(STAGE 4)은 자동화하지 않고
힌트 컬럼(조밀도·거래량 드라이업)까지만 제공 — 최종 확정은 43번
페이지 체크리스트로.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from data.adapters.composed_market_data import build_default_market_data
from data.adapters.wikipedia_universe import default_universe_provider
from pages._shared.growth_sources import build_growth_adapters
from pages._shared.manual_evidence import load_manual_evidence
from pages._shared.market_regime import (
    combine_rows,
    fetch_market_verdicts,
    stage0_lines,
    verdicts_to_state,
)
from pages._shared.minervini_manual_checks import render_manual_checklist
from strategy.adapters.minervini_screener import (
    MIN_BARS,
    COMPOSITE_WEIGHTS,
    annual_evidence,
    composite_score,
    evaluate_annual,
    evaluate_growth,
    evaluate_trend_template,
    momentum_components,
    momentum_score,
    pivot_levels,
    quarterly_evidence,
    rs_percentiles,
    simulate_forward,
    split_factor,
)

st.set_page_config(page_title="미너비니 스크리너", layout="wide")
st.title("미너비니 SEPA 스크리너 — 기준일 매수 후보 자동 추출")
st.caption(
    "기준일(과거의 '오늘')을 정하면 유니버스 전체에 추세 템플릿 → RS → "
    "분기 EPS·매출(발표일 ≤ 기준일만 사용) → 산업군 → 피봇/손절선/익절선 "
    "깔때기를 돌려 매수 후보를 뽑는다. ⚠️ 유니버스는 **현재** 지수 구성 "
    "종목(위키피디아)이라 과거 시점엔 생존 편향이 있다 — 오래된 기준일일수록 "
    "결과가 실제보다 좋게 나올 수 있음을 감안할 것. 최종 진입은 43번 "
    "체크리스트(VCP 눈 판독)로 확정."
)

with st.sidebar:
    st.header("기준일 / 유니버스")
    as_of = st.date_input(
        "기준일", value=date(2020, 1, 2),
        min_value=date(2005, 1, 1), max_value=date.today(),
        help="이날까지의 데이터만 사용한다 (look-ahead 없음).",
    )
    universe_choice = st.selectbox(
        "유니버스",
        ["nasdaq_full@기준일", "sp500+nasdaq100", "sp500", "nasdaq100", "nasdaq_full",
         "us_all@기준일"],
        help="**@기준일** = 그날 실제 상장돼 있던 종목 (Alpha Vantage LISTING_STATUS, "
        "Mongo 영구 캐시) — 이후 상장된 종목은 빠지고 폐지·인수된 종목은 들어가 "
        "생존 편향이 없다. 폐지 종목 가격은 EODHD 폴백으로 채운다. "
        "nasdaq_full(@없음)은 **오늘** 리스트라 과거 기준일엔 편향이 있다. "
        "첫 실행은 수천 종목 fetch로 느리고 이후엔 Mongo 캐시.",
    )
    max_tickers = st.number_input(
        "최대 종목 수 (0=전체)", value=0, min_value=0, step=50,
        help="스모크 테스트용. 처음엔 100 정도로 돌려보길.",
    )
    stage0_gate = st.toggle(
        "시장 조건(0-1·0-2) 미충족 시 스캔 생략", value=True,
        help="SPY와 QQQ 둘 다 보고 판정한다: 0-1은 두 지수 모두 200일선 위·상승이어야 "
        "통과, 0-2는 어느 한 지수라도 분산일 5회↑면 경고(IBD 규칙). 하나라도 "
        "실패하면 유니버스를 돌리지 않고 즉시 '후보 없음'으로 끝낸다 — 매뉴얼상 "
        "신규 매수 중단 구간이라 수천 종목을 조회할 이유가 없다. 약세장에서 "
        "관찰용 워치리스트를 뽑고 싶으면 끈다.",
    )

    st.header("프리필터 / RS")
    min_price = st.number_input(
        "최소 주가 ($)", value=10.0, min_value=0.0, step=1.0,
        help="**당시 실제 가격** 기준으로 검사한다: 분할 조정가가 이 값 "
        "밑이면 기준일 이후 분할 배수를 곱해 환산 후 재검사 (예: NVDA "
        "2020-01-02 조정가 $5.96 → ×40 = $238 → 통과). 환산에 쓰는 분할 "
        "이력은 조정가 탈락 종목만 조회 + 디스크 캐시라 느려지지 않음.",
    )
    min_dollar_vol_m = st.number_input(
        "최소 일평균 거래대금 ($M)", value=5.0, min_value=0.0, step=1.0
    )
    rs_cutoff = st.number_input(
        "RS 백분위 컷 (1-8)", value=70, min_value=0, max_value=99, step=5,
        help="가중 모멘텀(2×3M+6M+9M+12M)의 유니버스 내 백분위.",
    )

    st.header("펀더멘털 (STAGE 2)")
    use_fundamentals = st.toggle(
        "분기 EPS·매출 검증 (Alpha Vantage)", value=True,
        help="ALPHAVANTAGE_API_KEY 사용 (EODHD fundamentals는 현 플랜에서 "
        "403). 티커당 3콜 — 가격 스크린 통과 종목에만 호출하고 디스크 "
        "캐시(7일)한다.",
    )
    eps_min = st.number_input("EPS YoY 최소 (%)", value=25.0, step=5.0)
    rev_min = st.number_input("매출 YoY 최소 (%)", value=20.0, step=5.0)
    growth_required = st.number_input(
        "성장 체크 통과 개수 (4개 중)", value=3, min_value=1, max_value=4,
        help="2-1 EPS / 2-2 가속 / 2-3 매출 / 2-4 마진 중 몇 개 이상.",
    )

    st.header("피봇 / 손익선 (STAGE 4·5)")
    pivot_lookback = st.number_input(
        "피봇 폴백 기간 (일)", value=25, min_value=10, max_value=60,
        help="피봇은 기본적으로 VCP 구조(4% zigzag 스윙)로 정밀 계산된다 — "
        "마지막 축소의 천장, 또는 그 아래 조밀 선반(cheat). 이 값은 축소가 "
        "하나도 없는 직선 랠리에서만 쓰는 폴백(최근 N일 고가).",
    )
    near_pivot_only = st.toggle("피봇 −5% 이내만", value=True,
                                help="꺼면 베이스 형성 중(워치리스트)도 표시.")
    stop_pct = st.number_input("손절폭 (%)", value=8.0, min_value=1.0, max_value=10.0, step=0.5)
    target_pct = st.number_input("익절 목표 (%)", value=22.0, min_value=5.0, max_value=50.0, step=1.0)

    st.header("검증 (기준일 이후 — 평가 뒤에만!)")
    show_forward = st.toggle(
        "돌파/손절/익절 시뮬레이션 표시", value=False,
        help="기준일 이후 데이터로 피봇 돌파→손절 vs 익절 선착·1/3/6/12M "
        "수익률을 계산. 눈 판독 훈련 중이면 끄고 시작할 것.",
    )
    run_btn = st.button("Run Screen", type="primary", use_container_width=True)


_VERDICT = {True: "✅ 통과", False: "❌ 실패", None: "ℹ️ 참고"}


def _fmt_yoys(ys) -> str:
    return " → ".join(f"{y:+.0%}" for y in ys) if ys else "없음"


def _add_annual_checks(add, a) -> None:
    """2-5(코드 33)·2-6·2-10 — 연간 실적 자동 판정 행 (발표일 ≤ 기준일 연도만)."""
    if a is None or not a.data_available:
        add("2 펀더멘털", "2-5 코드 33 / 2-6 / 2-10 (연간)", None,
            "연간 데이터 부족 (발표일 ≤ 기준일 회계연도 4개 미만) — 미확인")
        return
    yrs = f"{a.years[0].year}~{a.years[-1].year}" if a.years else ""
    mg = " → ".join(f"{m:.1%}" if m is not None else "?" for m in a.margins)
    add("2 펀더멘털", "2-5a 코드 33: 연간 EPS 3년 가속", a.checks["2-5a EPS 3년 가속"],
        f"FY{yrs} EPS YoY {_fmt_yoys(a.eps_yoy)} (3개 전부 +, 마지막 > 첫 해)")
    add("2 펀더멘털", "2-5b 코드 33: 연간 매출 3년 가속", a.checks["2-5b 매출 3년 가속"],
        f"매출 YoY {_fmt_yoys(a.revenue_yoy)}")
    add("2 펀더멘털", "2-5c 코드 33: 순이익률 3년 연속 상승", a.checks["2-5c 순이익률 3년 상승"],
        f"순이익률 {mg}")
    add("2 펀더멘털", "2-5 코드 33 종합 (보너스)", a.passed,
        f"{a.n_passed}/3 — 3/3이면 초고수익 최상급 후보 (몬스터: EPS +75→+214→+195%, "
        "매출 +20→+63→+93%, NPM 3.3→11.3→18%)")
    eps_last = a.eps[-1] if a.eps else None
    add("2 펀더멘털", "2-6 연간 EPS 과거 고점 돌파 (보너스)", a.eps_breakout,
        (f"최신 연간 EPS ${eps_last:,.2f} vs 이전 최고 ${a.eps_prev_high:,.2f}"
         if a.eps_prev_high is not None and eps_last is not None
         else "비교할 이전 연도 3개 미만 — 미확인"))
    add("2 펀더멘털", "2-10 연간 EPS 성장률 감속 경고",
        (None if a.decel_warning is None else not a.decel_warning),
        (f"EPS YoY {_fmt_yoys(a.eps_yoy[-3:])} — "
         + ("뚜렷한 감속(두 번 연속 하락, 마지막 < 첫 해 ½) → 천장 경고, 신규 진입 금지"
            if a.decel_warning else "급감속 아님")
         if a.decel_warning is not None else "YoY 3개 미만 — 미확인")
        + " · ⚠️ 소스 EPS는 GAAP/non-GAAP 혼재 가능 — 튀는 해는 아래 연간 표로 확인")


def _build_detail_checks(
    tt, rs_val, rs_cut, comps, g, use_fund, pl,
    stop_pct_v, target_pct_v, industry, ind_count,
    eps_min_v: float = 25.0, rev_min_v: float = 20.0,
    cs=None, annual=None,
) -> pd.DataFrame:
    """최종 후보 1종목의 '항목 · 판정 · 근거(실제 수치)' 테이블.

    이미 계산된 값만 재사용 — 추가 데이터 조회 없음. 자동 판정이
    없는 항목(VCP 구조 등)은 ℹ️ 참고로 표시하고 43번 페이지로 넘긴다.
    """
    rows: list[dict] = []

    def add(stage: str, item: str, ok: bool | None, evidence: str) -> None:
        rows.append({"단계": stage, "항목": item,
                     "판정": _VERDICT[ok], "근거": evidence})

    if cs is not None:
        parts_txt = " · ".join(
            f"{k} {v:.0f}(×{COMPOSITE_WEIGHTS[k]:g})" if v is not None
            else f"{k} 제외"
            for k, v in cs.parts.items()
        )
        add("0 정렬", "종합점수", None,
            f"{cs.total:.1f}/100 = 가중 평균 — {parts_txt}. "
            "측정 불가 요소는 빼고 남은 가중치로 재정규화.")

    c = tt.checks
    add("1 추세", "1-1 주가 > 150·200일선", c["1-1 주가>150·200일선"],
        f"종가 ${tt.close:,.2f} vs SMA150 ${tt.sma150:,.2f} · "
        f"SMA200 ${tt.sma200:,.2f}")
    add("1 추세", "1-2 150일선 > 200일선", c["1-2 150>200일선"],
        f"SMA150 ${tt.sma150:,.2f} vs SMA200 ${tt.sma200:,.2f}")
    sma200_chg = tt.sma200 / tt.sma200_prev21 - 1.0
    add("1 추세", "1-3 200일선 상승 중", c["1-3 200일선 상승"],
        f"SMA200 ${tt.sma200:,.2f} vs 21거래일 전 ${tt.sma200_prev21:,.2f} "
        f"({sma200_chg:+.1%})")
    add("1 추세", "1-4 50일선 > 150·200일선", c["1-4 50>150·200일선"],
        f"SMA50 ${tt.sma50:,.2f} vs SMA150 ${tt.sma150:,.2f} · "
        f"SMA200 ${tt.sma200:,.2f}")
    add("1 추세", "1-5 주가 > 50일선", c["1-5 주가>50일선"],
        f"종가 ${tt.close:,.2f} vs SMA50 ${tt.sma50:,.2f}")
    add("1 추세", "1-6 52주 신저가 +30% 이상", c["1-6 신저가+30%↑"],
        f"52주 최저 ${tt.low_52w:,.2f} 대비 {tt.pct_above_low:+.0%} "
        f"(기준 ≥ +30%)")
    add("1 추세", "1-7 52주 신고가 −25% 이내", c["1-7 신고가-25%내"],
        f"52주 최고 ${tt.high_52w:,.2f} 대비 {tt.pct_from_high:+.1%} "
        f"(기준 ≥ −25%)")
    comp_txt = (
        " — " + " · ".join(f"{k.upper()} {v:+.0%}" for k, v in comps.items())
        if comps else ""
    )
    add("1 추세", "1-8 RS 백분위", rs_val >= rs_cut,
        f"RS {rs_val:.0f} (컷 {rs_cut}) = 가중 모멘텀 2×3M+6M+9M+12M의 "
        f"유니버스 백분위{comp_txt}")

    if not use_fund:
        add("2 펀더멘털", "2-1~2-4", None, "펀더멘털 검증 미사용 (사이드바 토글)")
    elif g is None or not g.data_available:
        add("2 펀더멘털", "2-1~2-4", None,
            "분기 데이터 미확인 (소스에 YoY 계산 가능한 분기 부족) — 통과로 취급됨")
    else:
        hist = " → ".join(f"{y:+.0%}" for y in g.eps_yoy_history)
        add("2 펀더멘털", "2-1 분기 EPS YoY", g.checks["2-1 EPS YoY"],
            f"최근 발표 분기 YoY {g.eps_yoy:+.0%} (설정 기준 ≥ +{eps_min_v:g}% "
            f"— 책은 고정값 대신 20~25%+와 '가속'을 요구, 슈퍼스톡은 "
            f"30~40%+·세 자릿수 흔함)")
        add("2 펀더멘털", "2-2 EPS 가속", g.checks["2-2 EPS 가속"],
            f"최근 발표 분기 YoY 흐름: {hist}")
        rev_txt = (f"매출 YoY {g.revenue_yoy:+.0%}"
                   if g.revenue_yoy is not None else "매출 데이터 없음")
        add("2 펀더멘털", "2-3 매출 성장/가속", g.checks["2-3 매출"],
            f"{rev_txt} (설정 기준 ≥ +{rev_min_v:g}% 또는 가속)"
            + (" · 직전 분기 대비 가속" if g.revenue_accelerating else ""))
        if g.margin_now is not None and g.margin_year_ago is not None:
            mg = (f"순이익률 {g.margin_now:.1%} vs 전년 동기 "
                  f"{g.margin_year_ago:.1%}")
        else:
            mg = "마진 데이터 부족"
        add("2 펀더멘털", "2-4 순이익률 개선", g.checks["2-4 마진 개선"], mg)
        add("2 펀더멘털", "종합", g.passed,
            f"{g.n_passed}/4 통과 (기준 ≥ {g.n_required})")
    if use_fund:
        _add_annual_checks(add, annual)

    add("3 주도주", "3-5 산업군 집계", None,
        f"{industry} — 이번 스크린 생존자 {ind_count}종목 "
        f"(생존자가 몰린 산업군일수록 주도 그룹 가능성)")
    add("3 주도주", "3-7 유동성·가격", True,
        "프리필터 통과 (최소 주가·거래대금 조건)")

    if pl.base_weeks is not None:
        if pl.dist_to_pivot >= 0:
            base_txt = (
                f"돌파 **전** 베이스 길이 {pl.base_weeks:.0f}주 — 현재가는 "
                f"이미 천장 위(신고가/돌파 진행)라 다지기는 끝난 상태. "
                f"천장 = 최근 5일 제외한 구조적 고점"
            )
        else:
            base_txt = f"베이스 천장 이후 {pl.base_weeks:.0f}주째 다지는 중"
        add("4 진입", "4-2 베이스 기간", 3.0 <= pl.base_weeks <= 60.0,
            f"{base_txt} (기준 3~60주, 측정창 24주 한도 — 더 긴 베이스는 "
            f"차트로 확인)")
    if pl.base_depth is not None:
        depth_ok = 0.10 <= pl.base_depth <= 0.35
        note = " · **60%+ 조정은 무조건 탈락**" if pl.base_depth > 0.60 else ""
        add("4 진입", "4-3 베이스 깊이", depth_ok,
            f"천장→저점 낙폭 {pl.base_depth:.0%} (기준 10~35%){note}")
    contr_txt = (" → ".join(f"−{d:.0%}" for d in pl.contractions)
                 if pl.contractions else "감지된 축소 없음")
    add("4 진입", "4-4 축소 횟수 2~6회",
        (2 <= len(pl.contractions) <= 6) if pl.contractions else False,
        f"4% zigzag로 실측한 축소: {contr_txt} ({len(pl.contractions)}회)")
    add("4 진입", "4-5 축소의 체감(직전의 절반)", pl.contraction_ok,
        f"{contr_txt} — 각 축소가 직전 대비 감소하는지 (15% 슬랙)"
        if pl.contractions else "축소 2회 미만 — 판정 불가")
    add("4 진입", "4-6 최종 압축 조밀도", pl.tightness_10d < 0.10,
        f"최근 10일 고저폭 {pl.tightness_10d:.1%} (기준 <10%, 3~5%면 최상)")
    add("4 진입", "4-7 거래량 드라이업", pl.volume_dryup < 0.70,
        f"10일 평균 거래량 = 50일 평균의 {pl.volume_dryup:.0%} (기준 <70%)")
    add("4 진입", "4-11 피봇", None,
        f"${pl.pivot:,.2f} [{pl.method}] · 현재가와 거리 {pl.dist_to_pivot:+.1%} "
        f"· {pl.status} — contraction-high = 마지막 축소의 천장, "
        f"tight-shelf(cheat) = 그 아래 6% 이내로 뭉친 선반의 고가")

    add("5 리스크", "5-1 손절선", None,
        f"${pl.stop:,.2f} = 피봇 − {stop_pct_v:g}%")
    add("5 리스크", "익절선 · 손익비", target_pct_v / stop_pct_v >= 2.0,
        f"${pl.target:,.2f} = 피봇 + {target_pct_v:g}% → 손익비 "
        f"{target_pct_v / stop_pct_v:.1f}:1 (기준 ≥ 2:1)")
    return pd.DataFrame(rows)


def _detail_chart(df: pd.DataFrame, pivot: float, stop: float, target: float):
    """기준일까지 최근 12개월: 캔들 + SMA50/150/200 + 피봇/손절/익절선."""
    tail = df.tail(260)
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.78, 0.22], vertical_spacing=0.02,
    )
    fig.add_trace(go.Candlestick(
        x=tail.index, open=tail["Open"], high=tail["High"],
        low=tail["Low"], close=tail["Close"], name="가격",
        increasing_line_color="#26A69A", decreasing_line_color="#EF5350",
    ), row=1, col=1)
    closes = df["Close"]
    for n, color in ((50, "#FFA726"), (150, "#42A5F5"), (200, "#AB47BC")):
        sma = closes.rolling(n).mean().reindex(tail.index)
        fig.add_trace(go.Scatter(
            x=tail.index, y=sma, name=f"SMA{n}",
            line=dict(width=1.2, color=color),
        ), row=1, col=1)
    for level, name, color in (
        (pivot, f"피봇 {pivot:,.2f}", "#FFEE58"),
        (stop, f"손절 {stop:,.2f}", "#EF5350"),
        (target, f"익절 {target:,.2f}", "#26A69A"),
    ):
        fig.add_hline(y=level, line_dash="dot", line_color=color,
                      annotation_text=name, annotation_position="right",
                      row=1, col=1)
    fig.add_trace(go.Bar(
        x=tail.index, y=tail["Volume"], name="거래량",
        marker_color="#78909C", opacity=0.7,
    ), row=2, col=1)
    fig.update_layout(
        template="plotly_dark", height=560, showlegend=True,
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=40, r=90, t=30, b=20),
    )
    return fig


@st.cache_data(ttl=86400, show_spinner=False)
def _universe_tickers(choice: str, as_of_date: date | None = None) -> list[str]:
    provider = default_universe_provider()
    if choice.endswith("@기준일"):
        if as_of_date is None:
            raise ValueError("기준일 유니버스에는 as_of 가 필요하다")
        return provider.get_tickers(f"{choice[:-len('@기준일')]}@{as_of_date.isoformat()}")
    if choice == "sp500+nasdaq100":
        merged = provider.get_tickers("sp500") + provider.get_tickers("nasdaq100")
        return list(dict.fromkeys(merged))
    return provider.get_tickers(choice)


if run_btn:
    # Mongo 일 단위 캐시 스택 (bars_yfinance 컬렉션 + Massive 폴백).
    # 파퀘(/tmp)와 달리 ① 컨테이너 재빌드에도 데이터 유지(호스트
    # ./mongodata 바인드), ② 캐시가 '하루' 단위라 기준일을 바꿔도
    # 겹치는 날짜는 전부 재사용된다 (파퀘는 (symbol,start,end) 키라
    # 기준일이 바뀌면 전량 재fetch였음).
    market_data = build_default_market_data()
    fetch_start = as_of - timedelta(days=460)

    # 분할 이력 fetcher — 최소 주가 필터의 '당시 실제 가격' 환산용.
    # 조정가로 탈락하는 종목에만 호출되고 디스크 캐시(7일)라 저렴하다.
    splits_provider = None
    try:
        from data.adapters.cached_fundamentals import CachedFundamentalsAdapter
        from data.adapters.eodhd_fundamentals import EODHDFundamentalsAdapter
        from data.adapters.fallback_fundamentals import FallbackFundamentalsAdapter
        from data.adapters.massive_fundamentals import MassiveFundamentalsAdapter
        try:
            _splits_src = FallbackFundamentalsAdapter(
                EODHDFundamentalsAdapter(), MassiveFundamentalsAdapter(),
                primary_label="eodhd", fallback_label="massive",
            )
        except ValueError:
            _splits_src = MassiveFundamentalsAdapter()
        splits_provider = CachedFundamentalsAdapter(_splits_src)
    except Exception:
        splits_provider = None  # 키 없음 → 조정가 그대로 필터 (경고 표시)

    def _effective_price(ticker: str, adj_close: float) -> float:
        """조정 종가 → 기준일 당시 실제 가격 근사 (분할 배수 복원)."""
        if splits_provider is None:
            return adj_close
        try:
            f = splits_provider.fetch(ticker)
            return adj_close * split_factor(f.splits, as_of)
        except Exception:
            return adj_close

    with st.spinner("유니버스 로드..."):
        try:
            tickers = _universe_tickers(universe_choice, as_of)
        except Exception as e:
            st.error(f"유니버스 로드 실패: {e}")
            st.stop()
    if max_tickers:
        tickers = tickers[: int(max_tickers)]

    # --- STAGE 0: 시장 확인 — SPY + QQQ 이중 판정 ---
    # 미너비니는 S&P 500만 보지 않는다(나스닥 종합을 가장 자주 봄). 0-1은
    # 두 지수 모두 통과해야, 0-2는 한 지수라도 분산일 5회↑면 경고 (IBD).
    market_verdicts, market_errors = fetch_market_verdicts(market_data, as_of)
    market_rows = verdicts_to_state(market_verdicts)
    market_ok, stage0_failed = combine_rows(market_rows)
    market_note = " · ".join(market_errors)

    # 0-1·0-2 실패 = 신규 매수 중단 구간 → 유니버스 스캔 생략 (토글)
    if stage0_gate and stage0_failed:
        st.session_state["minervini_screen"] = {
            "as_of": as_of, "universe": universe_choice,
            "df": pd.DataFrame(), "details": {},
            "funnel": {"유니버스": len(tickers), "스캔": 0, "최종 후보": 0},
            "split_rescued": 0, "splits_available": splits_provider is not None,
            "stage0": {"indexes": market_rows, "nh": None, "nl": None,
                       "scanned": 0, "skipped": stage0_failed},
            "market_ok": market_ok, "market_note": market_note,
            "growth_skipped": "", "show_forward": show_forward,
        }
        st.rerun()

    # --- STEP 1~3: 가격 히스토리 → 프리필터 → 템플릿 → 모멘텀 ---
    prog = st.progress(0.0, text="가격 데이터 수집 중...")
    frames: dict[str, pd.DataFrame] = {}
    prefilter_pass: list[str] = []
    template_results: dict[str, object] = {}
    scores: dict[str, float] = {}
    fetch_failed = 0
    split_rescued = 0
    nh_count = 0  # 0-3: 52주 신고가 −5% 이내 종목 수 (프리필터와 무관)
    nl_count = 0  # 0-3: 52주 신저가 +5% 이내 종목 수
    for i, t in enumerate(tickers):
        prog.progress((i + 1) / len(tickers), text=f"가격 수집 {i + 1}/{len(tickers)} — {t}")
        try:
            df = market_data.fetch_ohlcv(t, fetch_start, as_of)
        except Exception:
            fetch_failed += 1
            continue
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df.index = df.index.normalize()
        df = df[df.index <= pd.Timestamp(as_of)]
        if len(df) < MIN_BARS:
            continue
        close = float(df["Close"].iloc[-1])
        dollar_vol = float((df["Close"] * df["Volume"]).tail(20).mean())
        score = momentum_score(df)
        if score is not None:
            scores[t] = score  # RS 백분위 분모는 프리필터와 무관하게 전체
        # 0-3 신고가/신저가 카운트 — 프리필터 이전, 조회된 전 종목 대상
        hi52 = float(df["High"].tail(252).max())
        lo52 = float(df["Low"].tail(252).min())
        if hi52 > 0 and close >= 0.95 * hi52:
            nh_count += 1
        elif close <= 1.05 * lo52:
            nl_count += 1
        # 거래대금은 분할 불변(가격÷N × 거래량×N)이라 조정가 그대로 OK.
        if dollar_vol < min_dollar_vol_m * 1e6:
            continue
        # 최소 주가는 '당시 실제 가격' 기준 — 조정가로 탈락하는 경우에만
        # 분할 배수를 복원해 재검사 (미래 분할로 위너가 잘못 걸러지는
        # look-ahead 왜곡 방지. 예: NVDA 2020-01 조정가 $5.96, 실제 $238).
        if close < min_price:
            if _effective_price(t, close) < min_price:
                continue
            split_rescued += 1
        prefilter_pass.append(t)
        frames[t] = df
        tt = evaluate_trend_template(df)
        if tt is not None:
            template_results[t] = tt
    prog.empty()

    rs = rs_percentiles(scores)
    template_pass = [t for t in prefilter_pass
                     if t in template_results and template_results[t].passed]
    rs_pass = [t for t in template_pass if rs.get(t, 0.0) >= rs_cutoff]

    # --- STEP 4: 펀더멘털 (통과 종목에만 — 쿼터 절약) ---
    growth_results: dict[str, object] = {}
    annual_results: dict[str, object] = {}
    snapshots: dict[str, object] = {}
    growth_skipped_reason = ""
    if use_fundamentals and rs_pass:
        # 소스 체인: Alpha Vantage(기본 — EPS 발표일 포함 풀 히스토리)
        # → EODHD(플랜에 fundamentals 있으면). 앞 소스가 빈 결과일 때만
        # 다음 소스를 시도한다.
        growth_adapters = build_growth_adapters()
        if not growth_adapters:
            growth_skipped_reason = (
                "펀더멘털 소스 없음 (ALPHAVANTAGE_API_KEY / EODHD_API_KEY "
                "미설정) — 생략"
            )
        else:
            gprog = st.progress(0.0, text="분기 실적 수집 중...")
            for i, t in enumerate(rs_pass):
                gprog.progress((i + 1) / len(rs_pass),
                               text=f"분기 실적 {i + 1}/{len(rs_pass)} — {t}")
                snap = None
                for adapter in growth_adapters:
                    snap = adapter.fetch(t)
                    if snap.quarters:
                        break
                snapshots[t] = snap
                growth_results[t] = evaluate_growth(
                    snap.quarters if snap else [], as_of,
                    eps_min_yoy=eps_min / 100.0, rev_min_yoy=rev_min / 100.0,
                    n_required=int(growth_required),
                )
                annual_results[t] = evaluate_annual(
                    snap.annuals if snap else [], as_of,
                )
            gprog.empty()
    elif use_fundamentals:
        growth_skipped_reason = "RS 통과 종목 없음"

    def _growth_ok(t: str) -> bool:
        if not use_fundamentals or t not in growth_results:
            return True  # 펀더멘털 미사용/미확인은 통과로 두고 컬럼에 표시
        g = growth_results[t]
        return g.passed or not g.data_available

    growth_pass = [t for t in rs_pass if _growth_ok(t)]

    # --- STEP 5~6: 산업군 집계 + 피봇/손익선 ---
    industry_of = {
        t: (getattr(snapshots.get(t), "industry", None) or "미분류")
        for t in growth_pass
    }
    industry_counts = pd.Series(list(industry_of.values())).value_counts()

    rows = []
    pls: dict[str, object] = {}
    composites: dict[str, object] = {}
    for t in growth_pass:
        pl = pivot_levels(frames[t], lookback=int(pivot_lookback),
                          stop_pct=stop_pct / 100.0, target_pct=target_pct / 100.0)
        if pl is None:
            continue
        pls[t] = pl
        near_pivot = pl.dist_to_pivot >= -0.05
        if near_pivot_only and not near_pivot:
            continue
        tt = template_results[t]
        g = growth_results.get(t)
        ind = industry_of[t]
        n_ind = int(industry_counts.get(ind, 1))
        a = annual_results.get(t)
        cs = composite_score(
            rs.get(t), g, pl,
            industry_survivors=None if ind == "미분류" else n_ind,
            annual=a,
        )
        composites[t] = cs
        rows.append({
            "티커": t,
            "종합점수": cs.total,
            "산업군": ind,
            "산업군 생존자": n_ind,
            "종가": pl.close,
            "RS": rs.get(t),
            "신고가대비": tt.pct_from_high,
            "신저가대비": tt.pct_above_low,
            "EPS YoY": getattr(g, "eps_yoy", None),
            "EPS가속": getattr(g, "eps_accelerating", None),
            "매출 YoY": getattr(g, "revenue_yoy", None),
            "성장통과": (f"{g.n_passed}/4" if g and g.data_available
                       else ("미확인" if use_fundamentals else "미사용")),
            "코드33": (f"{a.n_passed}/3" if a and a.data_available
                      else ("미확인" if use_fundamentals else "미사용")),
            "연간감속": ("❗" if a and a.decel_warning
                       else ("—" if a and a.decel_warning is not None else "?")),
            "피봇": pl.pivot,
            "피봇거리": pl.dist_to_pivot,
            "손절선": pl.stop,
            "익절선": pl.target,
            "축소(T)": (" → ".join(f"−{d:.0%}" for d in pl.contractions)
                       or "없음"),
            "VCP": ("✅" if pl.contraction_ok
                    else ("—" if pl.contraction_ok is None else "❌")),
            "피봇상태": pl.status,
            "조밀도10d": pl.tightness_10d,
            "거래량드라이업": pl.volume_dryup,
        })
    result_df = pd.DataFrame(rows)
    if len(result_df):
        # 종합점수(SEPA 요소 가중 합산) 내림차순 — 동점이면 RS로 결정
        result_df = (
            result_df.sort_values(["종합점수", "RS"], ascending=[False, False])
            .reset_index(drop=True)
        )

    # --- 최종 후보별 상세 근거 (이미 계산된 값 재사용 — 추가 조회 없음) ---
    details: dict[str, dict] = {}
    for t in (result_df["티커"] if len(result_df) else []):
        g = growth_results.get(t)
        snap = snapshots.get(t)
        pl = pls[t]
        details[t] = {
            "checks": _build_detail_checks(
                template_results[t], rs.get(t, 0.0), rs_cutoff,
                momentum_components(frames[t]), g, use_fundamentals, pl,
                stop_pct, target_pct, industry_of.get(t, "미분류"),
                int(industry_counts.get(industry_of.get(t, "미분류"), 1)),
                eps_min_v=eps_min, rev_min_v=rev_min,
                cs=composites.get(t), annual=annual_results.get(t),
            ),
            "quarters": (quarterly_evidence(snap.quarters, as_of)
                         if snap is not None else []),
            "annuals": (annual_evidence(snap.annuals, as_of)
                        if snap is not None else []),
            "chart": frames[t],
            "pivot": pl.pivot, "stop": pl.stop, "target": pl.target,
        }

    # --- 검증: 기준일 이후 시뮬레이션 (옵션) ---
    if show_forward and len(result_df):
        fprog = st.progress(0.0, text="기준일 이후 데이터 수집 중...")
        fwd_end = min(as_of + timedelta(days=430), date.today())
        outcomes = []
        for i, row in result_df.iterrows():
            fprog.progress((i + 1) / len(result_df),
                           text=f"검증 {i + 1}/{len(result_df)} — {row['티커']}")
            try:
                fdf = market_data.fetch_ohlcv(row["티커"], as_of + timedelta(days=1), fwd_end)
                if fdf.index.tz is not None:
                    fdf.index = fdf.index.tz_localize(None)
                out = simulate_forward(
                    fdf, pivot=row["피봇"],
                    stop_pct=stop_pct / 100.0, target_pct=target_pct / 100.0,
                )
            except Exception:
                out = None
            outcomes.append(out)
        fprog.empty()
        result_df["돌파"] = [
            ("미돌파" if o and not o.entered else
             {"stop": "손절 먼저", "target": "익절 먼저", "none": "보유 중"}.get(
                 o.first_hit) if o else "?")
            for o in outcomes
        ]
        for col, attr in (("+1M", "ret_1m"), ("+3M", "ret_3m"),
                          ("+6M", "ret_6m"), ("+12M", "ret_12m"),
                          ("최대상승", "max_runup"), ("최대하락", "max_drawdown")):
            result_df[col] = [getattr(o, attr, None) if o else None for o in outcomes]

    st.session_state["minervini_screen"] = {
        "as_of": as_of, "universe": universe_choice, "df": result_df,
        "details": details,
        "funnel": {
            "유니버스": len(tickers), "조회 실패": fetch_failed,
            "가격/유동성 통과": len(prefilter_pass),
            "추세 템플릿 통과": len(template_pass),
            f"RS≥{rs_cutoff} 통과": len(rs_pass),
            "성장 통과": len(growth_pass), "최종 후보": len(result_df),
        },
        "split_rescued": split_rescued,
        "splits_available": splits_provider is not None,
        "stage0": {"indexes": market_rows, "nh": nh_count, "nl": nl_count,
                   "scanned": len(tickers) - fetch_failed},
        "market_ok": market_ok, "market_note": market_note,
        "growth_skipped": growth_skipped_reason,
        "show_forward": show_forward,
    }

_state = st.session_state.get("minervini_screen")
if _state is None:
    st.info("좌측에서 기준일·유니버스를 정하고 **Run Screen**. 처음엔 "
            "최대 종목 수 100으로 스모크 테스트 권장 (첫 실행은 종목당 "
            "가격 fetch가 있어 느리고, 이후엔 디스크 캐시로 빨라진다).")
    st.stop()

result_df = _state["df"]
st.subheader(f"{_state['as_of']} 기준 — {_state['universe']}")

s0 = _state.get("stage0") or {}
if _state["market_ok"] is not None:
    checks_s0, trend_all, dd_ok = stage0_lines(s0.get("indexes") or [])
    if _state.get("market_note"):
        checks_s0.append(f"⚠️ {_state['market_note']}")
    nh, nl = s0.get("nh"), s0.get("nl")
    skipped = s0.get("skipped") or []
    if nh is None or nl is None:
        nhnl_ok = True
        checks_s0.append("⏭ **0-3 신고가 > 신저가** — 스캔 생략으로 미측정")
    else:
        nhnl_ok = nh > nl
        checks_s0.append(
            f"{'✅' if nhnl_ok else '❌'} **0-3 신고가 > 신저가** — 스캔 "
            f"{s0.get('scanned', 0):,}종목 중 52주 신고가 근접(−5% 이내) "
            f"{nh:,} vs 신저가 근접(+5% 이내) {nl:,}"
        )
    stage0_all_ok = bool(_state["market_ok"]) and nhnl_ok
    box = st.success if stage0_all_ok else st.error
    if skipped:
        tail = (f" — **{' · '.join(skipped)} 실패 → 신규 매수 중단 구간. "
                f"유니버스 {_state['funnel'].get('유니버스', 0):,}종목 스캔을 생략했다.**")
    elif stage0_all_ok:
        tail = ""
    else:
        tail = " — **신규 매수 보류가 원칙** (아래 후보는 관찰/검증용)"
    box("STAGE 0 시장 환경" + tail + "\n\n" + "\n\n".join(checks_s0))
    st.caption(
        "0-1·0-2는 SPY(S&P 500)와 QQQ(나스닥 100) 이중 판정 — 0-1은 둘 다 통과해야, "
        "0-2는 한쪽이라도 5회↑면 경고(IBD 규칙). 미너비니는 나스닥 종합을 가장 자주 본다. "
        "0-3의 분모는 선택한 유니버스라 IBD의 전체 시장 NH/NL과는 다름 — "
        "유니버스가 클수록(nasdaq_full) 정확. 0-4(강세장 초입 여부)는 "
        "저점 정의가 주관적이라 자동화하지 않음 → 43번 페이지에서 수동 판단."
    )
elif _state["market_note"]:
    st.warning(f"STAGE 0 시장: {_state['market_note']}")
if _state["growth_skipped"]:
    st.warning(f"STAGE 2: {_state['growth_skipped']}")

funnel = _state["funnel"]
cols = st.columns(len(funnel))
for col, (name, n) in zip(cols, funnel.items()):
    col.metric(name, f"{n:,}")
if _state.get("split_rescued"):
    st.caption(
        f"↳ 최소 주가 필터: 분할 환산으로 {_state['split_rescued']}종목 구제 "
        "(조정가 < 기준이었지만 당시 실제 가격은 기준 이상)"
    )
if not _state.get("splits_available", True):
    st.warning(
        "분할 이력 소스 없음(EODHD/MASSIVE 키) — 최소 주가 필터가 분할 "
        "**조정가** 기준으로 동작해 과거 기준일에서 위너를 잘못 거를 수 "
        "있다. 최소 주가를 0으로 낮추고 거래대금 필터에 의존할 것."
    )

if not len(result_df):
    if (_state.get("stage0") or {}).get("skipped"):
        st.warning("후보 없음 — 시장 조건 미충족으로 종목 스캔 자체를 생략했다. "
                   "약세장 관찰용 워치리스트가 필요하면 사이드바 "
                   "'시장 조건 미충족 시 스캔 생략'을 끄고 다시 실행.")
    else:
        st.warning("최종 후보 없음 — 시장이 약세이거나 필터가 빡빡한 경우다. "
                   "'피봇 −5% 이내만'을 끄면 베이스 형성 중 워치리스트가 보인다.")
    st.stop()

pct_cols = {c: st.column_config.NumberColumn(format="percent")
            for c in ("신고가대비", "신저가대비", "EPS YoY", "매출 YoY", "피봇거리",
                      "조밀도10d", "+1M", "+3M", "+6M", "+12M", "최대상승", "최대하락")
            if c in result_df.columns}
price_cols = {c: st.column_config.NumberColumn(format="dollar")
              for c in ("종가", "피봇", "손절선", "익절선") if c in result_df.columns}
st.dataframe(
    result_df, use_container_width=True, hide_index=True,
    column_config={
        **pct_cols, **price_cols,
        "종합점수": st.column_config.NumberColumn(
            format="%.1f",
            help="RS 30 · 성장 20 · 코드33 5 · VCP구조 15 · 피봇위치 15 · 조밀도 5 · "
                 "거래량 5 · 산업군 5 가중 평균 (0~100). 측정 불가 요소는 "
                 "제외 후 재정규화. 상세 근거 표 첫 줄에 요소별 점수 표시."),
        "코드33": st.column_config.TextColumn(
            help="2-5 연간 3지표(EPS 가속·매출 가속·순이익률 상승) 통과 수. 3/3 = 몬스터 후보."),
        "연간감속": st.column_config.TextColumn(
            help="2-10 연간 EPS 증가율 급감속(델 80→65→28%) 경고. ❗면 신규 진입 금지."),
        "RS": st.column_config.NumberColumn(format="%.0f"),
        "거래량드라이업": st.column_config.NumberColumn(
            format="%.2f", help="10일 평균 거래량 ÷ 50일 평균. <0.7이면 드라이업(4-7)."),
    },
)
st.caption(
    "매수 규칙: **피봇 돌파 시** 진입(피봇거리 0% 근접 종목 주시), 돌파일 거래량이 "
    "50일 평균의 +40~50%인지 확인(4-12). 손절선·익절선은 피봇 기준. "
    "조밀도10d<10% + 거래량드라이업<0.7이면 VCP 마지막 압축(4-6·4-7) 가능성 — "
    "최종 확정은 43번 페이지에서 차트 눈 판독으로."
)

# --- 종목별 상세 근거 -------------------------------------------------------
st.divider()
st.subheader("종목별 상세 근거 — 어떤 항목을 왜 통과했나")
details = _state.get("details") or {}
if details:
    sel = st.selectbox("종목 선택", list(result_df["티커"]))
    d = details[sel]
    st.plotly_chart(
        _detail_chart(d["chart"], d["pivot"], d["stop"], d["target"]),
        use_container_width=True,
    )
    st.dataframe(
        d["checks"], use_container_width=True, hide_index=True,
        column_config={
            "단계": st.column_config.TextColumn(width="small"),
            "판정": st.column_config.TextColumn(width="small"),
            "근거": st.column_config.TextColumn(width="large"),
        },
    )
    if d["quarters"]:
        st.markdown("**분기 실적 근거 (기준일까지 발표된 분기만 — YoY는 4분기 전 대비)**")
        st.dataframe(
            pd.DataFrame(d["quarters"]), use_container_width=True, hide_index=True,
            column_config={
                "EPS YoY": st.column_config.NumberColumn(format="percent"),
                "매출 YoY": st.column_config.NumberColumn(format="percent"),
                "순이익률": st.column_config.NumberColumn(format="percent"),
                "매출": st.column_config.NumberColumn(format="compact"),
            },
        )
    if d.get("annuals"):
        st.markdown("**연간 실적 근거 (기준일까지 발표된 회계연도만 — 2-5 코드 33 · 2-6 · 2-10)**")
        st.dataframe(
            pd.DataFrame(d["annuals"]), use_container_width=True, hide_index=True,
            column_config={
                "EPS YoY": st.column_config.NumberColumn(format="percent"),
                "매출 YoY": st.column_config.NumberColumn(format="percent"),
                "순이익률": st.column_config.NumberColumn(format="percent"),
                "매출": st.column_config.NumberColumn(format="compact"),
            },
        )
        st.caption("연간 EPS는 소스(Alpha Vantage)가 GAAP/non-GAAP를 섞어 줄 수 있다 — "
                   "한 해만 튀면 일회성 손익(세금 환입 등)일 가능성이 크니 보도자료로 재확인.")
    st.divider()
    render_manual_checklist(
        f"{_state['as_of']}_{sel}", ticker=sel, as_of=_state["as_of"],
        evidence=load_manual_evidence(sel, _state["as_of"]),
    )
else:
    st.caption("상세 근거 없음 — 스크린을 다시 실행하면 생성된다.")

if _state["show_forward"] and "돌파" in result_df.columns:
    st.subheader("검증 — 기준일 이후 실제 결과")
    entered = result_df[result_df["돌파"] != "미돌파"]
    n_target = int((result_df["돌파"] == "익절 먼저").sum())
    n_stop = int((result_df["돌파"] == "손절 먼저").sum())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("돌파(진입) 종목", f"{len(entered)}/{len(result_df)}")
    c2.metric("익절 먼저", n_target)
    c3.metric("손절 먼저", n_stop)
    if len(entered) and entered["+6M"].notna().any():
        c4.metric("진입 종목 평균 +6M", f"{entered['+6M'].mean():+.1%}")
    st.caption("동일 날짜 손절·익절 동시 도달은 보수적으로 손절로 집계. "
               "진입가 = max(피봇, 돌파일 시가) — 갭 상승 반영.")

st.download_button(
    "결과 CSV 다운로드",
    result_df.to_csv(index=False).encode("utf-8-sig"),
    file_name=f"minervini_screen_{_state['as_of']}.csv",
    mime="text/csv",
)
