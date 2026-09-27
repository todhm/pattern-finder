"""미너비니 초고수익 성장주 — 과거 시점 수동 검증 체크리스트.

의도적으로 **데이터 자동 연동 없음**: 과거의 어느 하루로 돌아가
그날 알 수 있었던 데이터를 사용자가 직접 하나하나 찾아보며
체크하는 훈련/검증 도구다. 각 항목에 "무슨 데이터를 어디서 어떻게
확인하는지"가 적혀 있다. 상세 매뉴얼:
``docs/strategy_notes/미너비니_초고수익_체크리스트_2026_09.md``
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

from pages._shared.manual_evidence import evidence_markdown, load_manual_evidence
from pages._shared.source_links import earnings_surprise_links, guidance_links

st.set_page_config(page_title="미너비니 체크리스트", layout="wide")
st.title("미너비니 초고수익 성장주 체크리스트 — 과거 시점 수동 검증")
st.caption(
    "«초고수익 성장주 투자»(SEPA) 기준을 단계별 체크리스트로 정리. "
    "티커와 **과거 기준일**을 정하고, 기준일 이후 데이터는 가리고 "
    "각 항목을 실제 데이터로 하나하나 확인한 뒤 체크한다. "
    "평가를 끝낸 후에야 기준일 오른쪽 차트를 열어 성과를 기록한다."
)


# ---------------------------------------------------------------------------
# 체크리스트 정의
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Item:
    key: str
    label: str          # "1-1 주가 > 150·200일선"
    criterion: str      # 통과 기준 (숫자 포함)
    verify: str         # 데이터 확인법
    group: str = "core"  # core | guard(하나라도 실패 시 즉시 탈락) | bonus


@dataclass(frozen=True)
class Stage:
    key: str
    title: str
    intro: str
    items: tuple[Item, ...]
    min_core: int | None = None  # None = core 전부 통과 필요


STAGES: tuple[Stage, ...] = (
    Stage(
        key="s0",
        title="STAGE 0 — 시장 환경",
        intro="종목을 보기 전에 시장부터. 0-1·0-2 실패 시 신규 매수 중단이 원칙.",
        items=(
            Item("s0_trend", "0-1 지수 추세",
                 "SPY(또는 ^IXIC) 종가 > 200일선, 200일선이 하락 중 아님",
                 "기준일까지의 지수 일봉 + 200일 SMA. TradingView에서 "
                 "기준일로 이동해 주봉+200일선 확인 (1분).", "guard"),
            Item("s0_quality", "0-2 랠리의 질",
                 "최근 4주: 상승일 거래량 증가·하락일 거래량 감소. "
                 "분산일(하락+거래량증가) 4~5회/4주면 경고",
                 "지수 일봉·거래량 4주치 눈 판독.", "guard"),
            Item("s0_nhnl", "0-3 신고가 > 신저가",
                 "52주 신고가 근접 종목 수 > 신저가 근접 종목 수",
                 "유니버스(S&P500+NDX100) 스캔 또는 당시 시황 기사로 근사.",
                 "core"),
            Item("s0_window", "0-4 강세장 초입 (보너스)",
                 "새 강세장 첫 4~8주라면 최상의 기회의 창",
                 "지수 저점 이후 경과 주수 계산. 이때는 지수 저점일에 "
                 "신고가 찍던 종목이 최우선 후보.", "bonus"),
        ),
        min_core=0,
    ),
    Stage(
        key="s1",
        title="STAGE 1 — 추세 템플릿 8항목 (전부 통과 필수)",
        intro="기준일까지 최소 260거래일 일봉(분할조정가). 하나라도 실패 = 탈락, 예외 없음.",
        items=(
            Item("s1_1", "1-1 주가 > 150·200일선",
                 "종가 P > SMA150 그리고 P > SMA200",
                 "fetch_ohlcv로 일봉 → c.rolling(150/200).mean() 끝값과 비교."),
            Item("s1_2", "1-2 150일선 > 200일선",
                 "SMA150 > SMA200", "위와 동일 데이터."),
            Item("s1_3", "1-3 200일선 상승 중",
                 "SMA200(오늘) > SMA200(21거래일 전). 4~5개월 연속 상승이면 우량",
                 "SMA200 시리즈에서 21거래일 전 값과 비교."),
            Item("s1_4", "1-4 50일선 > 150·200일선",
                 "SMA50 > SMA150 그리고 SMA50 > SMA200", "위와 동일."),
            Item("s1_5", "1-5 주가 > 50일선",
                 "P > SMA50", "위와 동일."),
            Item("s1_6", "1-6 52주 신저가 대비 +30% 이상",
                 "P / 최근 252거래일 최저가 − 1 ≥ +30% (최고 종목은 100~300%)",
                 "df['Low'].tail(252).min() 대비 종가."),
            Item("s1_7", "1-7 52주 신고가 대비 −25% 이내",
                 "P / 최근 252거래일 최고가 − 1 ≥ −25% (−15% 이내면 우량)",
                 "df['High'].tail(252).max() 대비 종가."),
            Item("s1_8", "1-8 상대강도 상위 (RS ≥ 70)",
                 "가중 모멘텀 2×3M + 6M + 9M + 12M 수익률의 유니버스 백분위 ≥ 70. "
                 "간이 판정: 3·6·12개월 수익률 모두 SPY보다 높고 RS라인(종가/SPY) 1개월 상승",
                 "종목·SPY 각각 3/6/9/12개월 수익률 계산 후 비교."),
        ),
    ),
    Stage(
        key="s2",
        title="STAGE 2 — 펀더멘털 (이익·매출·마진)",
        intro="발표일(reportDate)이 기준일 이전인 분기만 사용 — 분기 말일 기준 금지 (look-ahead). "
              "2-1~2-4 중 3개 이상 + 적색경보 0개.",
        items=(
            Item("s2_1", "2-1 분기 EPS YoY +20~25% 이상",
                 "최근 발표 분기 EPS YoY ≥ +20~25% (초고수익 종목은 +100%도 흔함)",
                 "EODHD /fundamentals Earnings::History 또는 macrotrends "
                 "quarterly EPS — 기준일 이전 발표분까지만."),
            Item("s2_2", "2-2 EPS 증가율 가속",
                 "최근 2~3분기 YoY 증가율이 분기마다 커짐 (엘란: +13→+80→+100→+120%)",
                 "분기 EPS YoY를 3분기 나열해 비교."),
            Item("s2_3", "2-3 매출 동반 성장",
                 "최근 분기 매출 YoY ≥ +20% 또는 2~3분기 가속 "
                 "(F5: EPS 22→30→47→65% / 매출 2→15→34→46%)",
                 "Income_Statement::quarterly totalRevenue YoY."),
            Item("s2_4", "2-4 순이익률 개선",
                 "netIncome/totalRevenue가 전년 동기 대비 상승 추세",
                 "최근 2~3분기 순이익률 vs 전년 동기."),
            Item("s2_5", "2-5 코드 33 (보너스)",
                 "연간 EPS·매출·순이익률 3개가 3년 연속 가속 "
                 "(몬스터: EPS +75→+214→+195%)",
                 "44 스크리너 상세 근거 표 / 45 리플레이 '연간 펀더멘털 자동 판정'의 "
                 "2-5a·b·c 값을 옮겨 적는다 (Alpha Vantage 연간, 발표일 ≤ 기준일).", "bonus"),
            Item("s2_6", "2-6 연간 EPS 박스권 돌파 (보너스)",
                 "수년 박스권·과거 고점을 새 연간(후행 12개월) EPS가 돌파",
                 "44/45 자동 판정 2-6 (최신 연간 EPS vs 이전 3~7년 최고치).", "bonus"),
            Item("s2_7", "2-7 어닝 서프라이즈(+)",
                 "최근 발표 분기 epsActual > epsEstimate — '추정치 30일 상향' "
                 "항목의 과거 재현용 대체물",
                 "Earnings::History의 actual vs estimate.", "core"),
            Item("s2_8", "2-8 [경보 없음] 가이던스 하향 없음",
                 "최근 실적 발표에서 하향 가이던스 없음 (상향이면 가점)",
                 "당시 실적 발표 보도자료·뉴스 기간검색.", "guard"),
            Item("s2_9", "2-9 [경보 없음] 재고·매출채권 정상",
                 "완제품 재고 YoY ≫ 매출 YoY 아님 (예: +79% vs +11%면 탈락), "
                 "매출채권 YoY > 매출 YoY도 경고",
                 "분기 재무상태표 inventory·receivables vs 매출.", "guard"),
            Item("s2_10", "2-10 [경보 없음] 성장률 감속 없음",
                 "EPS 증가율 뚜렷한 감속(델: 80→65→28%) 없음 — 감속은 천장 신호",
                 "44 '연간감속' 컬럼 / 45 자동 판정 2-10 (연간 EPS YoY 두 번 연속 하락 + "
                 "마지막 < 첫 해 ½). 분기 가속은 2-2 자동.", "guard"),
        ),
        min_core=3,
    ),
    Stage(
        key="s3",
        title="STAGE 3 — 주도주·산업군",
        intro="주도주는 시장보다 먼저 바닥을 떠난다. 핵심(3-1~3-5) 중 3개 이상 + 3-3 필수.",
        items=(
            Item("s3_1", "3-1 시장 대비 선행 바닥",
                 "종목이 지수보다 먼저 바닥을 만들고 먼저 2단계 진입 "
                 "(AMZN 2001-10 vs 나스닥 2002-10)",
                 "직전 조정 구간에서 종목 저점일 vs 지수 저점일 비교."),
            Item("s3_2", "3-2 지수 약세일의 신고가",
                 "지수가 조정 신저점 찍은 날 종목은 52주 신고가(근접) "
                 "(PCYC: 이후 33개월 +1,500%)",
                 "지수 저점일 날짜에 종목 주가 위치 확인."),
            Item("s3_3", "3-3 조정 깊이가 시장의 2~3배 미만",
                 "직전 시장 조정에서 종목 낙폭이 지수 낙폭의 2~3배 넘으면 탈락 "
                 "(MNKD −60% vs 나스닥 −6% → 실패)",
                 "같은 구간 고점→저점 낙폭 두 개 계산해 비율.", "guard"),
            Item("s3_4", "3-4 산업군 내 상위 1~2위",
                 "같은 산업군 수익률 비교에서 상위 1~3위 (2등 기업까지만)",
                 "동종 5~10개 티커의 6개월 수익률 나열 (수동 판단, 근거 메모)."),
            Item("s3_5", "3-5 산업군 자체가 주도 그룹",
                 "신고가 종목이 몰린 3~4개(최대 8~10개) 산업군에 포함",
                 "당시 신고가 목록/시황에서 강한 업종 확인 (수동)."),
            Item("s3_6", "3-6 기업의 신선함 (보너스)",
                 "상장 10년 이내·신제품·신시장·신경영 등 '새로움' (오닐 N)",
                 "상장일·당시 뉴스.", "bonus"),
            Item("s3_7", "3-7 유동성·가격",
                 "주가 ≥ $10~12, 일평균 거래대금이 포지션의 수십 배",
                 "종가 × 평균 거래량.", "guard"),
        ),
        min_core=3,
    ),
    Stage(
        key="s4",
        title="STAGE 4 — VCP·진입 포인트 (타이밍)",
        intro="기준일까지 12개월 일봉 차트(가격+거래량)를 자로 재듯 측정. "
              "실패 시 '종목은 좋으나 지금은 살 때 아님' → 워치리스트.",
        items=(
            Item("s4_1", "4-1 선행 상승 존재",
                 "베이스 이전 뚜렷한 상승 (최소 +30%, 파워 플레이는 8주 내 +100%)",
                 "베이스 시작 전 저점→고점 상승폭 측정."),
            Item("s4_2", "4-2 베이스 기간 3~60주",
                 "통상 5~26주. 3주 미만은 미성숙",
                 "베이스 시작 고점부터 기준일까지 주 수."),
            Item("s4_3", "4-3 베이스 깊이 10~35%",
                 "고점→저점 낙폭 10~35%. 60% 이상 조정은 무조건 탈락",
                 "베이스 내 최고가·최저가로 낙폭 계산.", "guard"),
            Item("s4_4", "4-4 축소(T) 2~6회",
                 "베이스 안 조정 파동 수 2~6회 (대다수 2~4회)",
                 "왼쪽부터 고점→저점 파동을 센다."),
            Item("s4_5", "4-5 축소의 체감",
                 "각 축소가 직전의 약 절반 (−25% → −15% → −8%)",
                 "각 파동 낙폭 %를 왼쪽부터 나열."),
            Item("s4_6", "4-6 최종 축소 조밀함 <10%",
                 "마지막 축소(손잡이/속임수) 고저 낙폭 10% 미만 (3~5%면 최상, "
                 "속임수 구간 기준 5~10%)",
                 "마지막 구간 고가·저가 측정."),
            Item("s4_7", "4-7 거래량 급감",
                 "마지막 축소 평균 거래량이 50일 평균 대비 뚜렷한 감소(−30%↑이면 명확)",
                 "구간 평균 거래량 ÷ 50일 평균 거래량."),
            Item("s4_8", "4-8 털어내기 (보너스)",
                 "베이스/손잡이에서 전저점 하회 후 빠른 복귀 — 매물 소진 증거 (DECK, VIVO)",
                 "차트에서 전저점 이탈 후 회복 여부.", "bonus"),
            Item("s4_9", "4-9 손잡이 위치 상단 1/3",
                 "손잡이는 컵 상단 1/3 (중간 이하면 '속임수' 피봇으로 별도 취급)",
                 "베이스 높이 대비 손잡이 수직 위치."),
            Item("s4_10", "4-10 우측 급등 추격 아님",
                 "베이스 오른쪽 단기 급등 직후가 아님 — 급등했다면 재보합 대기 (MGA)",
                 "최근 1~2주 급등 여부 확인.", "guard"),
            Item("s4_11", "4-11 피봇 확정",
                 "마지막 축소 구간의 고점 = 매수 피봇. 숫자로 기록",
                 "피봇 가격을 아래 기록란에 입력.", "guard"),
            Item("s4_12", "4-12 돌파 + 거래량 +40~50%",
                 "피봇 돌파 당일 거래량이 50일 평균의 +40~50% 이상",
                 "돌파일 거래량 ÷ 50일 평균 거래량.", "guard"),
            Item("s4_13", "4-13 돌파 후 정상 반응 (사후)",
                 "눌림이 1~7일 내 회복 후 신고가 = 테니스공 액션. "
                 "돌파가 −8% 이탈은 실패",
                 "돌파 이후 확인용 — 평가 시점에는 체크 불가.", "bonus"),
        ),
    ),
    Stage(
        key="s5",
        title="STAGE 5 — 리스크·포지션 (진입 전 숫자 확정)",
        intro="네 개 숫자를 기록해야 매수 자격. 전부 필수.",
        items=(
            Item("s5_1", "5-1 손절가 확정 (≤10%)",
                 "손절폭 최대 10%, 기본 7~8% 이하. 기술적 손절이 10%보다 멀면 진입 포기",
                 "피봇 × (1−손절%). 아래 기록란에 입력.", "guard"),
            Item("s5_2", "5-2 계좌 리스크 1.25~2.5%",
                 "포지션 비중 × 손절폭 ≤ 계좌의 1.25~2.5% "
                 "(최근 성적 나쁘면 0.75~1.25%)",
                 "예: 2% ÷ 8% = 포지션 25%.", "guard"),
            Item("s5_3", "5-3 포지션 상한 20~25%",
                 "한 종목 20~25% 상한, 총 4~8종목 집중",
                 "계좌 리스크 ÷ 손절폭.", "guard"),
            Item("s5_4", "5-4 손익비 ≥ 2:1",
                 "기대 이익(평균 +20%) ÷ 손절폭 ≥ 2",
                 "익절 목표 +20~25% 기준으로 계산.", "guard"),
        ),
    ),
)

SELL_RULES = """
**매도/관리 규칙 (성과 기록 시 청산 판정에 사용)**
- 이익 **+20~25%** 도달 시 강세에 매도(전량 또는 절반). 단 돌파 후 **1~3주 만에 +20%**면 8주 보유 규칙.
- **손절 도달 시 무조건 청산** — 예외 없음.
- 절반 익절 후 나머지는 본전 스톱 → **50일선 종가 이탈(대량 거래)** 시 청산.
- 천장 경고: 최대 상승일·소진 갭·2~3주 내 +25~50% 클라이맥스, 연간 EPS 증가율 감속.
"""


# ---------------------------------------------------------------------------
# 사이드바 — 평가 대상
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("평가 대상")
    ticker = st.text_input("티커", value="", placeholder="예: MNST").strip().upper()
    as_of = st.date_input(
        "기준일 (과거의 '오늘')", value=date(2020, 1, 2),
        min_value=date(1995, 1, 1), max_value=date.today(),
        help="이날 이후의 차트·실적은 평가가 끝날 때까지 절대 보지 않는다.",
    )
    st.header("진입 숫자 기록 (STAGE 4·5)")
    pivot_price = st.number_input("피봇 가격 (4-11)", value=0.0, min_value=0.0, format="%.2f")
    stop_pct = st.number_input("손절폭 % (5-1)", value=8.0, min_value=1.0, max_value=10.0, step=0.5)
    position_pct = st.number_input("포지션 비중 % (5-3)", value=20.0, min_value=0.0, max_value=25.0, step=5.0)
    memo = st.text_area("메모 (판단 근거·탈락 사유)", value="", height=80)
    if st.button("체크 초기화", use_container_width=True):
        for stage in STAGES:
            for item in stage.items:
                st.session_state[f"chk_{item.key}"] = False
        st.rerun()

if pivot_price > 0:
    st.sidebar.caption(
        f"→ 손절가 **{pivot_price * (1 - stop_pct / 100):,.2f}** · "
        f"계좌 리스크 **{position_pct * stop_pct / 100:.2f}%** "
        f"(1.25~2.5% 안이어야 5-2 통과)"
    )


# ---------------------------------------------------------------------------
# 체크리스트 본문
# ---------------------------------------------------------------------------

GROUP_BADGE = {"guard": "🚫 필수(실패=즉시탈락)", "core": "핵심", "bonus": "➕ 보너스"}

_EVIDENCE_KEY = {"s2_7": "m2_7", "s2_8": "m2_8", "s2_9": "m2_9"}
_evidence = load_manual_evidence(ticker, as_of) if ticker else {}

stage_results: dict[str, dict] = {}
for stage in STAGES:
    with st.expander(stage.title, expanded=(stage.key in ("s0", "s1"))):
        st.caption(stage.intro)
        checked: dict[str, bool] = {}
        for item in stage.items:
            cols = st.columns([0.55, 0.45])
            with cols[0]:
                checked[item.key] = st.checkbox(
                    f"**{item.label}** — {item.criterion}",
                    key=f"chk_{item.key}",
                )
            with cols[1]:
                _links = ""
                if ticker and item.key == "s2_7":
                    _links = earnings_surprise_links(ticker)
                elif ticker and item.key == "s2_8":
                    _links = guidance_links(ticker, as_of)
                _ev = evidence_markdown(_evidence.get(_EVIDENCE_KEY.get(item.key, "")))
                st.caption(f"{GROUP_BADGE[item.group]} · 확인법: {item.verify}"
                           + (f"  \n{_ev}" if _ev else "")
                           + (f"  \n🔗 {_links}" if _links else ""))
        core_items = [i for i in stage.items if i.group == "core"]
        guard_items = [i for i in stage.items if i.group == "guard"]
        bonus_items = [i for i in stage.items if i.group == "bonus"]
        core_pass = sum(checked[i.key] for i in core_items)
        guard_pass = sum(checked[i.key] for i in guard_items)
        bonus_pass = sum(checked[i.key] for i in bonus_items)
        need_core = len(core_items) if stage.min_core is None else stage.min_core
        ok = guard_pass == len(guard_items) and core_pass >= need_core
        stage_results[stage.key] = {
            "ok": ok,
            "text": f"핵심 {core_pass}/{len(core_items)}"
                    + (f" (필요 {need_core})" if stage.min_core is not None else "")
                    + f" · 필수 {guard_pass}/{len(guard_items)}"
                    + (f" · 보너스 +{bonus_pass}" if bonus_items else ""),
            "counts": f"{core_pass + guard_pass + bonus_pass}/{len(stage.items)}",
        }
        (st.success if ok else st.warning)(
            f"{'통과' if ok else '미통과'} — {stage_results[stage.key]['text']}"
        )

st.markdown(SELL_RULES)


# ---------------------------------------------------------------------------
# 종합 판정
# ---------------------------------------------------------------------------

st.divider()
st.subheader("종합 판정")

all_ok = all(r["ok"] for r in stage_results.values())
only_timing_fail = (
    not stage_results["s4"]["ok"]
    and all(r["ok"] for k, r in stage_results.items() if k != "s4")
)
if all_ok:
    verdict = "매수"
    st.success("✅ **매수 자격** — 전 단계 통과. 피봇 돌파 시 계획대로 진입.")
elif only_timing_fail:
    verdict = "워치"
    st.info("👀 **워치리스트** — 종목은 합격이나 STAGE 4(진입 타이밍) 미완성. "
            "피봇 형성/돌파를 기다린다.")
else:
    verdict = "탈락"
    failed = [s.title.split("—")[0].strip() for s in STAGES
              if not stage_results[s.key]["ok"]]
    st.error(f"❌ **탈락** — 미통과: {', '.join(failed)}")

cols = st.columns(len(STAGES))
for col, stage in zip(cols, STAGES):
    r = stage_results[stage.key]
    col.metric(stage.title.split("—")[0].strip(),
               ("✅ " if r["ok"] else "❌ ") + r["counts"])


# ---------------------------------------------------------------------------
# 성과 확인 안내 + 평가 기록 누적
# ---------------------------------------------------------------------------

st.divider()
st.subheader("성과 확인 (평가를 저장한 뒤에야 기준일 오른쪽을 연다)")
st.markdown(
    """
1. **진입가** = 피봇 돌파가 (돌파일 종가 근사, 슬리피지 +0.5%).
2. 이후 일봉에서 **손절(−8%)과 익절(+20~25%) 중 어느 쪽이 먼저인지** 저가/고가로 확인.
3. 기록: 1/3/6/12개월 수익률, 최대 상승폭(MFE)·최대 하락폭(MAE), 손절 히트 날짜,
   매도 규칙(50일선 이탈 등) 적용 시 청산일.
4. 이 레포로: `fetch_ohlcv(ticker, 기준일, 기준일+1년)` 한 줄. 수동으로는 TradingView에서
   기준일 오른쪽을 다시 연다.
5. **판정 기준**: 표본 30건 이상(슈퍼스톡 A그룹 + 무작위 B그룹) 모은 뒤,
   매수 판정 그룹 vs 탈락 그룹의 6개월 평균 수익률·승률 비교.
   목표: 승률 ≥ 40~50%, 손익비 ≥ 2:1. 상세 절차는 전략 노트 §3.
"""
)

fwd_cols = st.columns(6)
fwd_1m = fwd_cols[0].number_input("1개월 %", value=0.0, step=1.0, format="%.1f")
fwd_3m = fwd_cols[1].number_input("3개월 %", value=0.0, step=1.0, format="%.1f")
fwd_6m = fwd_cols[2].number_input("6개월 %", value=0.0, step=1.0, format="%.1f")
fwd_12m = fwd_cols[3].number_input("12개월 %", value=0.0, step=1.0, format="%.1f")
stop_hit = fwd_cols[4].selectbox("손절 먼저?", ["미확인", "손절 먼저", "익절 먼저"])
group = fwd_cols[5].selectbox("표본 그룹", ["A(슈퍼스톡)", "B(무작위)"])

if "minervini_log" not in st.session_state:
    st.session_state["minervini_log"] = []

if st.button("이 평가를 기록에 추가", type="primary", disabled=not ticker):
    st.session_state["minervini_log"].append({
        "티커": ticker,
        "기준일": as_of.isoformat(),
        "그룹": group,
        "판정": verdict,
        **{s.title.split("—")[0].strip(): stage_results[s.key]["counts"]
           for s in STAGES},
        "피봇": pivot_price or None,
        "손절%": stop_pct,
        "포지션%": position_pct,
        "1M%": fwd_1m, "3M%": fwd_3m, "6M%": fwd_6m, "12M%": fwd_12m,
        "손절/익절": stop_hit,
        "메모": memo,
    })
    st.toast(f"{ticker} @ {as_of} 기록 추가")

log = st.session_state["minervini_log"]
if log:
    st.subheader(f"평가 기록 ({len(log)}건)")
    log_df = pd.DataFrame(log)
    st.dataframe(log_df, use_container_width=True, hide_index=True)
    taken = log_df[log_df["판정"] == "매수"]
    dropped = log_df[log_df["판정"] == "탈락"]
    if len(taken) and len(dropped):
        st.caption(
            f"매수 판정 {len(taken)}건 평균 6M {taken['6M%'].mean():+.1f}% vs "
            f"탈락 {len(dropped)}건 평균 6M {dropped['6M%'].mean():+.1f}% — "
            "매수 그룹이 뚜렷이 높아야 체크리스트가 작동하는 것."
        )
    st.download_button(
        "기록 CSV 다운로드",
        log_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="minervini_checklist_log.csv",
        mime="text/csv",
    )
else:
    st.info("아직 기록 없음. 체크 후 '이 평가를 기록에 추가'를 누르면 여기에 쌓인다. "
            "⚠️ 세션 메모리 저장 — 브라우저 새로고침 전에 CSV로 내려받을 것.")


# ---------------------------------------------------------------------------
# 전체 매뉴얼 (전략 노트 원문)
# ---------------------------------------------------------------------------

_doc = Path(__file__).resolve().parent / "_shared" / "minervini_manual.md"
with st.expander("📖 전체 매뉴얼 원문 (데이터 소스·look-ahead 규칙·검증 절차 상세)"):
    if _doc.exists():
        st.markdown(_doc.read_text(encoding="utf-8"))
    else:
        st.caption(f"문서를 찾을 수 없음: {_doc}")
