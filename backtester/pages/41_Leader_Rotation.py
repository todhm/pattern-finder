from datetime import date

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data.adapters.cached_market_data import CachedMarketDataAdapter
from data.adapters.yfinance_adapter import YFinanceAdapter
from strategy.adapters.leader_rotation_strategy import LeaderRotationStrategy
from strategy.adapters.risk_metrics import compute_risk_metrics
from strategy.domain.models import LeaderRotationConfig, TossFeeSchedule

st.set_page_config(page_title="리더 로테이션", layout="wide")
st.title("리더 로테이션 — '폭발 성장주 갈아타기' 방법론 검증")

CLOSE_PATH = "sweep_results/leader_rotation_close.parquet"
DVOL_PATH = "sweep_results/leader_rotation_dollarvol.parquet"

st.markdown(
    """
### 1. 실제로 갈아타기로 초고수익을 낸 사람들

'26년 보유 100배'가 아니라 **폭발 구간의 주도주만 갈아타며** 몇 년 만에
수백~수만 %를 만든, 기록이 남아 있는 사례들
([정리 글](https://www.financialwisdomtv.com/post/how-legendary-traders-enter-breakouts-minervini-kullamagi-darvas-o-neil),
[Momentum Masters](https://www.amazon.com/Momentum-Masters-Roundtable-Interview-Minervini/dp/0996307923),
[서평](https://www.newtraderu.com/2015/08/22/book-review-momentum-masters/)):

| 트레이더 | 기록 | 기간 | 방법 |
|---|---|---|---|
| 니콜라스 다바스 | $3.6만 → $225만 (~60배) | 18개월 (1957~59) | 박스 돌파 + 신고가 주도주만 보유 |
| 댄 잰저 | $1.1만 → $1,800만 (**감사 기록**) | 18개월 (1998~00) | 차트 패턴 돌파 + 마진 풀가동 |
| 마크 미너비니 | 5년 연평균 220% (누적 33,500%), USIC 우승 155% | 1994~99 | SEPA: 추세 템플릿 + 변동성 수축 돌파 |
| 크리스티안 쿨라매기 | $5천 → $1억 (주장, 방송으로 부분 검증) | ~10년 | 3/6개월 모멘텀 상위주 돌파 스윙 |
| 윌리엄 오닐 | CANSLIM으로 26세에 NYSE 회원권 | 1960s~ | 실적+신고가 주도주, 약세장 현금 |

**전원이 공유하는 뼈대는 3가지다**: ① 지금 가장 강한 종목만 산다(모멘텀/신고가)
② 리더십이 식으면 미련 없이 다음 리더로 갈아탄다 ③ 약세장에서는 시장에서 나온다.

### 2. 검증 방법 — 일봉으로 근사한 규칙

장중 체결·재량 판단은 재현 불가능하므로, 뼈대만 기계적 규칙으로 옮겼다:
**유동성 있는 미국 주식 전체(3,900여 종목)** 중 [200일선 위 + 52주 고점 근처]
종목을 [N개월 수익률]로 줄 세워 **상위 K개를 균등 보유**, [M거래일]마다
재랭킹해 갈아타고, **시장(QQQ)이 200일선 아래면 전량 현금**.
수수료·슬리피지·양도세 22% 반영.
"""
)
st.warning(
    "정직한 한계 — 이 백테스트는 **낙관 쪽으로 기운다**: ① 유니버스가 "
    "현재 상장 종목 기준이라 상폐된 종목이 빠져 있다(생존편향) ② 데이터가 "
    "2017-11부터라 2000·2008급 붕괴가 없다 ③ 종가 체결 가정. 반대로 "
    "실제 트레이더들의 기록은 장중 타이밍 + 집중 베팅 + 레버리지의 "
    "산물이라 이 근사보다 위·아래로 모두 훨씬 과격하다. 여기서 확인할 "
    "것은 '방법론의 방향성이 유효한가'다."
)

st.markdown(
    """
### 3. 사전 그리드 판정 (2018-11 ~ 2026-06, 세후, 1억 시작)

같은 뼈대라도 **디테일이 생사를 갈랐다**:

| 변형 | 세후 | CAGR | MDD | 한 줄 판정 |
|---|---|---|---|---|
| 나이브: 6개월 수익률 상위 8종목, 거래대금 >$20M | 0.5억 | -8% | -96% | 💀 과열 소형주 꼭지 매수 — 참사 |
| + 50일선 즉시 손절 / 변동성 조정 랭킹 | 0.7~1.4억 | -4~+5% | -64~-90% | 완화될 뿐 구조 못 바꿈 |
| **12-1 모멘텀 + 거래대금 >$200M (대형주)** | **8.5억** | **+34%** | **-56%** | ✅ 유일하게 작동 |
| 위 설정 + 5종목 집중 | ~11억 | +40% | -72% | 수익↑ 낙폭↑ |
| 위 설정 + 12종목 분산 | ~8억 | +33% | -47% | 단독 Calmar 최고 0.70 |
| 위 설정, 슬리피지 40bp 스트레스 | ~8억 | +33% | -59% | 대형주라 비용에 강함 |
| (비교) TQQQ 존버 | 9.5억 | +39% | -82% | 순수 금액 근소 우위, 낙폭 지옥 |
| (비교) QQQ 존버 | 3.7억 | +22% | -35% | — |
| **TQQQ 30% + 로테이션 70% 혼합** | **11.1억** | **+41%** | **-51%** | ✅ **Calmar 0.80 — 전체 1위** |

**두 가지가 결정적이었다**: ① **최근 1개월을 랭킹에서 제외** (12-1 모멘텀 —
블로우오프 꼭지 매수 방지) ② **거래대금 $200M 이상 대형주만** (소형 잡주
모멘텀은 2021~23년 -56%/-30%/-61%로 붕괴). 전설들의 '재량'이 하던 일을
이 두 규칙이 대신한다. 반대로 빠른 손절선은 대형주에선 휩쏘 비용만 냈다.

**vs TQQQ 존버**: 순수 금액은 TQQQ가 근소하게 앞서지만(9.5 vs 8.5억)
낙폭이 -82% vs -56%. 그리고 둘은 수익 원천이 달라서(레버리지 베타 vs
종목선정 알파) **섞으면 둘 다 이긴다** — 30:70 혼합이 세후 11.1억,
MDD -51%, Calmar 0.80으로 여태 테스트한 전체 전략 중 위험효율 1위.
(혼합 수치는 일일 리밸런스 근사 + 최종 청산 과세만 반영한 상한 추정)
"""
)

import os

if not (os.path.exists(CLOSE_PATH) and os.path.exists(DVOL_PATH)):
    st.error(
        "유니버스 데이터가 아직 없어. 생성 명령: "
        "`docker compose exec backtester python fetch_leader_universe.py`"
    )
    st.stop()


@st.cache_data(show_spinner="유니버스 로딩 중 (3,900종목)...")
def load_universe():
    close = pd.read_parquet(CLOSE_PATH)
    dvol = pd.read_parquet(DVOL_PATH)
    if close.index.tz is not None:
        close.index = close.index.tz_localize(None)
        dvol.index = dvol.index.tz_localize(None)
    return close, dvol


@st.cache_data(show_spinner=False)
def load_benchmark(ticker: str, start: date, end: date) -> pd.Series:
    # EODHD 우선 (yfinance는 벌크 다운로드 후 레이트리밋 잦음).
    try:
        from data.adapters.eodhd_adapter import EODHDAdapter

        s = EODHDAdapter().fetch_ohlcv(ticker, start, end, "1d")["Close"]
    except Exception:
        md = CachedMarketDataAdapter(YFinanceAdapter())
        s = md.fetch_ohlcv(ticker, start, end)["Close"]
    s = s.dropna()
    if s.index.tz is not None:
        s.index = s.index.tz_localize(None)
    s.index = s.index.normalize()
    return s


with st.sidebar:
    st.header("규칙 파라미터")
    lookback = st.selectbox(
        "모멘텀 기간 (거래일)", [21, 63, 126, 252], index=3,
        help="학계 정석은 12개월. 소형주+짧은 룩백 조합은 참사였다.",
    )
    skip = st.selectbox(
        "최근 제외 기간 (거래일)", [0, 21], index=1,
        help="'12-1 모멘텀'의 -1. 최근 1개월을 빼면 블로우오프 꼭지를 "
        "사는 걸 피한다 — 그리드에서 결정적 개선.",
    )
    top_n = st.number_input("보유 종목 수", 1, 30, 8)
    rebal = st.selectbox("재랭킹 주기 (거래일)", [5, 10, 21, 42], index=2)
    near_high = st.slider("52주 고점 대비 허용 낙폭 (%)", 5, 50, 25) / 100.0
    trend_ma = st.selectbox("종목 추세 필터 (이평선)", [0, 100, 200], index=2,
                            format_func=lambda x: "없음" if x == 0 else f"{x}일선 위")
    min_dv = st.number_input(
        "최소 20일 평균 거래대금 ($백만)", 1, 1000, 200,
        help="200 미만으로 내리면 소형 잡주가 들어와 결과가 붕괴한다 "
        "(그리드 검증: $20M → 세후 0.5억, $200M → 9억).",
    ) * 1e6
    rank_mode = st.selectbox(
        "랭킹 방식", ["raw", "vol_adj"], index=0,
        format_func=lambda x: "수익률 그대로" if x == "raw" else "변동성 조정",
    )
    stop_ma = st.selectbox(
        "보유 중 손절선 (이평)", [0, 20, 50], index=0,
        format_func=lambda x: "없음 (월간 재랭킹만)" if x == 0 else f"{x}일선 이탈 시 즉시",
        help="대형주에선 손절선이 오히려 휩쏘로 성과를 깎았다.",
    )
    market_filter = st.checkbox("시장 필터 (QQQ < 200일선 → 전량 현금)", value=True)
    slippage = st.number_input("슬리피지 (bp, 편도)", 0, 100, 20)
    start = st.date_input("시작일", date(2018, 11, 1),
                          min_value=date(2018, 6, 1))
    end = st.date_input("종료일", date(2026, 6, 12))
    run_btn = st.button("백테스트 실행", type="primary", use_container_width=True)

if run_btn:
    close, dvol = load_universe()
    market = load_benchmark("QQQ", date(2017, 1, 1), end)
    cfg = LeaderRotationConfig(
        start_date=start, end_date=end,
        lookback_days=int(lookback), skip_days=int(skip), top_n=int(top_n),
        rebalance_days=int(rebal), near_high_pct=near_high,
        trend_ma_days=int(trend_ma), min_dollar_volume=float(min_dv),
        rank_mode=rank_mode, stop_ma_days=int(stop_ma),
        market_filter=market_filter, slippage_bp=float(slippage),
        fee_schedule=TossFeeSchedule(),
    )
    with st.spinner("로테이션 시뮬레이션 중..."):
        result = LeaderRotationStrategy().execute(close, dvol, cfg, market=market)
    st.session_state["leader_rotation_result"] = result

result = st.session_state.get("leader_rotation_result")
if result is None:
    st.info("좌측에서 파라미터를 정하고 **백테스트 실행**을 눌러줘.")
    st.stop()

cfg = result.config
curve = pd.Series(
    {pd.Timestamp(p.date): p.equity for p in result.equity_curve}
)
metrics = compute_risk_metrics(curve)
liq = result.liquidation

st.header("결과")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("세후 최종", f"{liq.final_value_after_tax / 1e8:.2f}억")
c2.metric("CAGR (세전)", f"{metrics['cagr']:.1%}")
c3.metric("MDD", f"{metrics['mdd']:.1%}")
c4.metric("Calmar", f"{metrics['calmar']:.2f}")
c5.metric("수수료+세금", f"{(liq.total_fees + liq.total_tax) / 1e8:.2f}억")

qqq = load_benchmark("QQQ", cfg.start_date, cfg.end_date)
tqqq = load_benchmark("TQQQ", cfg.start_date, cfg.end_date)
fig = go.Figure()
fig.add_trace(go.Scatter(x=curve.index, y=curve.values, name="리더 로테이션",
                         line=dict(width=2.5)))
for name, s in [("QQQ 존버", qqq), ("TQQQ 존버", tqqq)]:
    s = s[s.index >= curve.index[0]]
    fig.add_trace(go.Scatter(
        x=s.index, y=s / s.iloc[0] * cfg.initial_capital, name=name,
        line=dict(width=1.2, dash="dot"),
    ))
fig.update_layout(yaxis_type="log", yaxis_title="평가액 (로그)",
                  height=450, legend=dict(orientation="h"))
st.plotly_chart(fig, use_container_width=True)

st.subheader("연도별 수익률")
qqq_y = qqq.groupby(qqq.index.year).apply(lambda g: g.iloc[-1] / g.iloc[0] - 1)
yr = pd.DataFrame({
    "전략": pd.Series(result.yearly_returns),
    "QQQ": qqq_y,
}).dropna(how="all")
st.dataframe(
    yr.T.style.format("{:+.1%}", na_rep="-"), use_container_width=True
)

risk_off = sum(1 for r in result.rebalances if r.regime == "risk_off")
st.caption(
    f"리밸런스 {len(result.rebalances)}회 중 위험회피(전량 현금) "
    f"{risk_off}회 · 평균 보유 {result.avg_holdings:.1f}종목 · "
    f"누적 회전율 {result.total_turnover:.0f}배"
)

st.subheader("갈아타기 내역 (최근 30회)")
rows = [
    {
        "날짜": r.date, "레짐": "🟢 진입" if r.regime == "risk_on" else "🔴 현금",
        "보유 종목": ", ".join(r.picks) if r.picks else "(현금)",
        "회전율": r.turnover,
    }
    for r in result.rebalances[-30:]
][::-1]
st.dataframe(
    rows, use_container_width=True, hide_index=True,
    column_config={"회전율": st.column_config.NumberColumn(format="percent")},
)

last = result.rebalances[-1]
if last.regime == "risk_on" and last.picks:
    st.success(
        "**마지막 랭킹 기준 리더** (지금 이 규칙을 따른다면): "
        + ", ".join(
            f"{s} ({last.momentum.get(s, 0):+.0%})" for s in last.picks
        )
    )
else:
    st.warning("마지막 랭킹 기준: 시장 필터 발동 — 전량 현금 구간.")
