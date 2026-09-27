"""미너비니 수동 체크리스트 — 자동화 불가/부적합 항목의 공용 컴포넌트.

44(스크리너)·45(리플레이) 페이지 하단에 붙어, 자동 판정과 합쳐
'완전한 체크리스트'를 이룬다. 어떤 항목이 왜 수동인지와 확인 방법을
항목마다 명시한다. 체크 상태는 (페이지 무관하게) 종목+기준일 단위로
세션에 유지된다 — 44에서 체크한 것이 45에서도 보인다.

전체 기준 원문: ``pages/_shared/minervini_manual.md`` (43번 페이지 렌더).
"""

from __future__ import annotations

import streamlit as st

from pages._shared.manual_evidence import evidence_markdown
from pages._shared.source_links import earnings_surprise_links, guidance_links

# (key, 단계, 라벨, 확인 방법) — 순서 = 표시 순서
# 2-5 코드 33 · 2-6 연간 EPS 신고 · 2-10 연간 감속은 Alpha Vantage 연간
# 데이터로 자동 판정된다 (44 상세 근거 표 / 45 '연간 펀더멘털 자동 판정')
# — 여기서 뺐다.
_ITEMS: list[tuple[str, str, str, str]] = [
    ("m0_4", "STAGE 0", "0-4 새 강세장 초입(첫 4~8주)인가",
     "지수 저점 이후 경과 주수를 센다. 초입이면 최상의 기회의 창 — 이때 "
     "지수 저점일에 신고가를 찍던 종목이 최우선. '저점'의 정의가 주관적이라 수동."),
    ("m2_7", "STAGE 2", "2-7 최근 분기 어닝 서프라이즈(+) / 추정치 상향",
     "상세 패널의 분기 실적 테이블 + 당시 발표 기사에서 actual vs estimate 확인. "
     "추정치 30일 전 대비 상향은 과거 재현 불가 → 서프라이즈로 대체."),
    ("m2_8", "STAGE 2", "2-8 가이던스 하향 없음 (상향이면 가점)",
     "최근 실적 발표 보도자료·컨콜 요약을 기준일 이전 날짜로 기간검색. "
     "하향 가이던스면 즉시 탈락."),
    ("m2_9", "STAGE 2", "2-9 완제품 재고·매출채권 증가율 ≤ 매출 증가율",
     "10-Q 재무상태표: 완제품 재고 +79% vs 매출 +11% 같으면 적색경보 탈락. "
     "재고 세부 항목은 API에 없어 수동."),
    ("m3_1", "STAGE 3", "3-1 지수보다 먼저 바닥을 만들었나",
     "직전 시장 조정에서 종목 저점일 vs 지수 저점일 비교 (AMZN 2001-10 vs 나스닥 2002-10)."),
    ("m3_2", "STAGE 3", "3-2 지수 조정 저점일에 신고가(근접)였나",
     "지수 최저일 날짜에 종목 주가 위치 확인 — 최상위 주도주 신호 (PCYC +1,500%)."),
    ("m3_3", "STAGE 3", "3-3 조정 깊이가 지수의 2~3배 미만",
     "같은 조정 구간의 종목 낙폭 ÷ 지수 낙폭. MNKD −60% vs 나스닥 −6% → 탈락 사례."),
    ("m3_4", "STAGE 3", "3-4 산업군 내 1~2위 종목인가",
     "동종 5~10개 티커의 6개월 수익률·차트 비교. 상세 패널의 '산업군 생존자' 수와 "
     "RS를 출발점으로 쓰되 최종 순위 판단은 육안."),
    ("m3_6", "STAGE 3", "3-6 기업의 신선함 (보너스)",
     "상장 10년 이내, 신제품/신시장/신경영 등 '새로움' (오닐의 N). 정성 판단."),
    ("m4_1", "STAGE 4", "4-1 베이스 이전 선행 상승 +30% 이상",
     "상세 차트에서 베이스 시작 전 저점→고점 상승폭 측정 (파워 플레이는 8주 내 +100%)."),
    ("m4_8", "STAGE 4", "4-8 털어내기 존재 (가점)",
     "베이스/손잡이에서 전저점 하회 후 빠른 복귀 — 매물 소진 증거 (DECK·VIVO). 차트 육안."),
    ("m4_9", "STAGE 4", "4-9 손잡이가 베이스 상단 1/3에 위치",
     "자동 피봇이 tight-shelf(cheat)면 중단부 속임수 피봇일 수 있음 — 컵 높이 대비 "
     "선반의 수직 위치를 차트에서 확인."),
    ("m4_10", "STAGE 4", "4-10 베이스 우측 급등 추격이 아님",
     "'피봇상태'가 확장/압축 없음이면 특히 주의 — 재보합(핸들) 형성을 기다린다 (MGA)."),
    ("m4_vcp", "STAGE 4", "4-종합 VCP 모양 최종 눈판독",
     "자동 실측(축소 시퀀스·조밀도·드라이업)과 차트 인상이 일치하는가. "
     "왼쪽 느슨→오른쪽 조밀, 등락폭·거래량이 함께 줄어야 진짜 VCP."),
    ("m5_2", "STAGE 5", "5-2 계좌 리스크 1.25~2.5% 이내",
     "포지션 비중 × 손절폭. 최근 성적이 나쁘면 0.75~1.25%로 축소. 계좌 상태 의존이라 수동."),
    ("m5_3", "STAGE 5", "5-3 포지션 20~25% 상한 · 총 4~8종목 집중",
     "이 종목을 사면 포트폴리오 구성이 규칙 안에 남는지 확인."),
]

# 돌파 이후에만 판정 가능 — 45(리플레이) 페이지 전용 섹션
_POST_ITEMS: list[tuple[str, str, str, str]] = [
    ("p4_12", "돌파 후", "4-12 돌파일 거래량 ≥ 50일 평균 ×1.4",
     "리플레이의 '돌파일 거래량' 자동 표시값 확인 — 미달이면 가짜 돌파 의심, "
     "실전이라면 진입 보류 대상이었다."),
    ("p4_13", "돌파 후", "4-13 테니스공 액션 — 눌림이 1~7일 내 회복",
     "일자별 테이블에서 돌파 후 첫 눌림의 회복 기간 확인 (NFLX 5~7일, BEBE 1·4·6일). "
     "돌파가 −8% 이탈은 실패."),
    ("p_sell", "돌파 후", "매도 규칙 준수 — 계획대로 청산했는가",
     "+20~25% 강세 매도 / 돌파 1~3주 내 +20%면 8주 보유 / 50일선 대량 거래 이탈 청산 / "
     "손절 무조건. 규칙 변형 비교 테이블과 대조."),
]


def _item_links(key: str, ticker: str | None, as_of) -> str:
    """항목별 외부 소스 링크 (티커가 있을 때만). 2-7·2-8."""
    if not ticker:
        return ""
    if key == "m2_7":
        return earnings_surprise_links(ticker)
    if key == "m2_8":
        return guidance_links(ticker, as_of)
    return ""


def render_manual_checklist(
    context_key: str,
    include_post_breakout: bool = False,
    ticker: str | None = None,
    as_of=None,
    evidence: dict[str, dict] | None = None,
) -> None:
    """종목+기준일(context_key) 단위 수동 체크리스트를 렌더.

    체크 상태 키가 페이지와 무관하게 context_key 기준이라 44↔45를
    오가도 유지된다 (세션 한정 — 영구 기록은 43번 페이지 CSV로).
    ``ticker``/``as_of``를 주면 2-7·2-8에 확인용 외부 링크(발표일별
    서프라이즈, 기준일 이전 8-K·뉴스)를 붙인다. ``evidence``
    (:func:`pages._shared.manual_evidence.load_manual_evidence`)를 주면
    2-7·2-8·2-9 밑에 Alpha Vantage/EODHD 근거와 자동 판정을 띄운다.
    """
    items = _ITEMS + (_POST_ITEMS if include_post_breakout else [])
    total = len(items)
    checked = sum(
        bool(st.session_state.get(f"mmc_{context_key}_{k}"))
        for k, *_ in items
    )
    st.subheader(f"수동 체크리스트 — 데이터 자동화 밖 항목 ({checked}/{total})")
    st.caption(
        "자동 판정(위)과 이 수동 항목이 **전부 모여야 완전한 체크리스트**다. "
        "각 항목의 확인 방법을 따라 직접 보고 체크할 것 — 체크 상태는 종목+기준일 "
        "단위로 세션에 유지되며 44↔45 페이지 간 공유된다. "
        "⚠️ 새로고침하면 사라지므로 최종 기록은 43번 페이지에서 CSV로."
    )
    st.progress(checked / total if total else 0.0)
    current_stage = None
    for key, stage, label, guide in items:
        if stage != current_stage:
            st.markdown(f"**{stage}**")
            current_stage = stage
        cols = st.columns([0.52, 0.48])
        with cols[0]:
            st.checkbox(label, key=f"mmc_{context_key}_{key}")
        with cols[1]:
            links = _item_links(key, ticker, as_of)
            ev_line = evidence_markdown((evidence or {}).get(key))
            st.caption(guide
                       + (f"  \n{ev_line}" if ev_line else "")
                       + (f"  \n🔗 {links}" if links else ""))
