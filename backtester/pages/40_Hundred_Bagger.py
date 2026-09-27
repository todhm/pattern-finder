import json
import os

import pandas as pd
import streamlit as st

st.set_page_config(page_title="100배 주식 리서치", layout="wide")
st.title("100배 주식 리서치 — 검증된 사례와 스크리너")
st.caption(
    "완전한 백테스트가 불가능한 영역이라 접근을 바꾼다: **문서로 검증된 "
    "100배 사례**(크리스 메이어의 365개 연구 등)에서 반복 패턴을 추출하고, "
    "그 패턴을 alphafolio 펀더멘털 데이터로 **지금 시장에서 스크리닝**한다. "
    "이 페이지는 매수 리스트가 아니라 '깊이 조사할 후보'를 좁히는 도구다."
)

# ---------------------------------------------------------------------------
st.header("1. 검증된 사례 — 무엇이 실제로 100배가 됐나")
st.markdown(
    """
[크리스 메이어의 연구](https://www.amazon.com/100-Baggers-Stocks-100-1/dp/1621291650)는
1962~2014년 미국 시장에서 **100배를 달성한 365개 종목 전부**를 분석했다. 대표 사례
(수치는 공개 문헌 기준 근사):

| 회사 | 대략 기간 | 배수 | 걸린 시간 | 핵심 엔진 |
|---|---|---|---|---|
| Monster Beverage | 1995~2015 | ~700배 | 20년 | 초소형 시총 + 이익 폭발 + PER 리레이팅 |
| Berkshire Hathaway | 1965~ | 수만 배 | 50년+ | 오너 경영 + 이익 전액 재투자 |
| Amazon | 1997~2009 | 100배 | 12년 | 매출 초고성장, 이익은 나중 |
| Netflix | 2002~2013 | 100배 | 11년 | 사업모델 전환(DVD→스트리밍) |
| Home Depot | 1981~1992 | 100배 | 11년 | 검증된 매장 모델의 전국 복제 |
| Walmart | 1970~1985 | 100배 | 15년 | 위와 동일 — '복제 가능한 단위경제' |
| Altria (필립모리스) | 1962~ | 100배+ | 수십 년 | 저성장이지만 압도적 ROE + 재투자 |
| Domino's Pizza | 2008~2020 | ~130배 | 12년 | 턴어라운드 + 자사주 매입 |

**메이어 연구의 통계적 사실**:
- 100배까지 걸린 시간의 중앙값은 **약 26년** — 연 20% 복리를 26년 유지한 것
  ([요약](https://fifthperson.com/100-baggers-by-christopher-mayer/))
- 시작 시점 중앙값: **시총 ~$5억, 매출 ~$1.7억** — 대부분 소형주에서 출발
  ([스크리닝 정리](https://stablebread.com/chris-mayer-100-baggers-screen/))
- 쌍발 엔진(twin engines): **이익 성장 × 멀티플 확장**이 곱해질 때 100배가 나온다
- 베센바인더 연구(1926~2016): 미국 주식 순부 창출의 전부를 **상위 ~4% 종목**이
  만들었다 — 나머지 대부분은 국채만도 못했다

**가장 불편한 진실**: 100배의 본체는 종목 선정이 아니라 **26년을 안 파는 것**이다.
메이어가 '커피캔 포트폴리오'(사서 잊어라)를 결론으로 둔 이유. 우리가 한 달간 확인한
"잦은 익절 = 세금·복리 파괴"와 정확히 같은 교훈이다.
"""
)

# ---------------------------------------------------------------------------
st.header("2. 반복 패턴 → 체크리스트")
st.markdown(
    """
사례들에서 공통으로 확인되고, **데이터로 점검 가능한** 조건:

| # | 패턴 | 스크리닝 지표 |
|---|---|---|
| 1 | 작게 시작 (100배의 산술적 여지) | 시총 < $10억 |
| 2 | 이미 검증된 수익 모델 | ROE ≥ 15%, 흑자 마진 |
| 3 | 고성장 지속 | 매출/EPS 성장 ≥ 15%/년 |
| 4 | 이익을 배당 대신 재투자 | 배당수익률 ≈ 0 |
| 5 | 오너가 직접 운전 | 내부자 지분 ≥ 5~10% |
| 6 | 출발 밸류에이션이 미치지 않았을 것 | PEG ≤ 2 안팎 |
| 7 | 복제 가능한 단위경제 (질적) | 스크리너 불가 — 손으로 조사 |

한계도 명시한다: 아래 스크리너는 **TTM 스냅샷**이라 '지속성'(수년간 ROE 유지)을
검증하지 못하고, 7번(사업의 질)은 데이터 밖이다. 그리고 메이어의 365개는
살아남은 종목만 본 것(생존편향) — 같은 조건으로 시작해 사라진 회사가 훨씬 많다.
그래서 이 목록은 '복권 후보'이며, 포트폴리오에선 소액·분산·장기보유가 전제다.
"""
)

# ---------------------------------------------------------------------------
st.header("3. 스크리너 — 지금 시장에서 패턴 찾기")

DATA_PATH = "sweep_results/hundred_bagger_screen.json"
if not os.path.exists(DATA_PATH):
    st.error(
        "스크리너 데이터가 없어. 갱신 명령:\n\n"
        "`docker compose exec alphafolio_data python ... > "
        "backtester/sweep_results/hundred_bagger_screen.json` "
        "(README/대화 기록 참조)"
    )
    st.stop()
with open(DATA_PATH) as f:
    data = json.load(f)
df = pd.DataFrame(data["rows"])
snapshot = df["snapshot"].max() if len(df) else "?"
st.caption(
    f"데이터: alphafolio `us_stock_basic` 최신 스냅샷 ({snapshot}) · "
    f"{len(df):,}개 보통주. 갱신하려면 alphafolio 파이프라인 최신화 후 "
    "추출 스크립트를 재실행."
)

c1, c2, c3, c4 = st.columns(4)
mc_max = c1.number_input("시총 상한 ($백만)", value=1000, min_value=50, step=100)
roe_min = c2.number_input("ROE 하한 (%)", value=15.0, min_value=0.0, step=1.0)
rev_min = c3.number_input("매출성장 하한 (%/년)", value=15.0, min_value=0.0, step=5.0)
eps_min = c4.number_input("EPS성장 하한 (%/년)", value=0.0, min_value=-100.0, step=5.0)
c5, c6, c7, c8 = st.columns(4)
div_max = c5.number_input("배당수익률 상한 (%)", value=1.0, min_value=0.0, step=0.5)
ins_min = c6.number_input("내부자 지분 하한 (%)", value=5.0, min_value=0.0, step=1.0)
peg_max = c7.number_input("PEG 상한 (0=무시)", value=0.0, min_value=0.0, step=0.5)
sectors = c8.multiselect(
    "섹터 (비우면 전체)", sorted(df["sector"].dropna().unique().tolist())
)

f = df.copy()
f = f[f["market_cap"] < mc_max * 1e6]
f = f[f["roe"].fillna(-1) >= roe_min / 100.0]
f = f[f["rev_growth_yoy"].fillna(-1) >= rev_min / 100.0]
f = f[f["eps_growth_yoy"].fillna(-999) >= eps_min / 100.0]
f = f[f["dividend_yield"].fillna(0.0) <= div_max / 100.0]
f = f[f["insiders_pct"].fillna(0.0) >= ins_min / 100.0]
if peg_max > 0:
    f = f[(f["peg"].notna()) & (f["peg"] > 0) & (f["peg"] <= peg_max)]
if sectors:
    f = f[f["sector"].isin(sectors)]
f = f.sort_values("rev_growth_yoy", ascending=False)

st.subheader(f"통과 종목: {len(f)}개")
show = f[[
    "symbol", "name", "sector", "industry", "market_cap", "roe",
    "rev_growth_yoy", "eps_growth_yoy", "profit_margin",
    "insiders_pct", "dividend_yield", "pe", "peg", "ps",
]].rename(columns={
    "symbol": "티커", "name": "회사", "sector": "섹터", "industry": "산업",
    "market_cap": "시총($)", "roe": "ROE",
    "rev_growth_yoy": "매출성장", "eps_growth_yoy": "EPS성장",
    "profit_margin": "순이익률", "insiders_pct": "내부자",
    "dividend_yield": "배당", "pe": "PER", "peg": "PEG", "ps": "PSR",
})
st.dataframe(
    show, use_container_width=True, hide_index=True,
    column_config={
        "시총($)": st.column_config.NumberColumn(format="compact"),
        "ROE": st.column_config.NumberColumn(format="percent"),
        "매출성장": st.column_config.NumberColumn(format="percent"),
        "EPS성장": st.column_config.NumberColumn(format="percent"),
        "순이익률": st.column_config.NumberColumn(format="percent"),
        "내부자": st.column_config.NumberColumn(format="percent"),
        "배당": st.column_config.NumberColumn(format="percent"),
        "PER": st.column_config.NumberColumn(format="%.1f"),
        "PEG": st.column_config.NumberColumn(format="%.2f"),
        "PSR": st.column_config.NumberColumn(format="%.2f"),
    },
)
if len(f):
    st.download_button(
        f"후보 CSV 다운로드 ({len(f)}개)",
        show.to_csv(index=False).encode("utf-8-sig"),
        file_name="hundred_bagger_candidates.csv",
        mime="text/csv",
    )

st.markdown(
    """
---
**이 목록을 쓰는 법** (체크리스트 7번 — 데이터 밖의 숙제):

1. 후보별로 "이 회사의 단위경제가 복제 가능한가?"를 손으로 조사
   (10-K, 오너 서한, 마진 추이 5년)
2. 통과한 소수만 **소액·분산**으로 편입 — 베센바인더의 4%에 걸리려면
   여러 장의 복권이 필요하다
3. 그리고 가장 어려운 것: **안 판다.** 중앙값 26년. 잦은 익절이 복리와
   세금에서 어떻게 지는지는 이 백테스터의 한 달 결론 그 자체다.
"""
)
