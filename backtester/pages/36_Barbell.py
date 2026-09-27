from datetime import date, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data.adapters.cached_market_data import CachedMarketDataAdapter
from data.adapters.yfinance_adapter import YFinanceAdapter
from strategy.adapters import portfolio_lab as pl
from strategy.adapters.band_rebalance_strategy import BandRebalanceStrategy
from strategy.adapters.barbell import combine_barbell_detailed
from strategy.adapters.risk_metrics import compute_risk_metrics
from strategy.adapters.volatility_breakout_strategy import (
    VolatilityBreakoutStrategy,
)
from strategy.domain.models import (
    BandRebalanceConfig,
    TossFeeSchedule,
    VolBreakoutConfig,
)

st.set_page_config(page_title="바벨 포트폴리오", layout="wide")
st.title("바벨 포트폴리오 — 안전 주머니 + 위험 주머니")

st.markdown(
    """
### 바벨이 뭔가요?

아령(barbell)처럼 **양 끝에만** 돈을 둡니다. 어중간한 중간은 없습니다.

|  | 🛡️ **안전 주머니** (예: 95%) | 🎲 **위험 주머니** (예: 5%) |
|---|---|---|
| 무엇에 | TQQQ 50% + 금 + 거시레짐 (검증된 안정 전략) | BTC 변동성 돌파 × 레버리지 (초고위험) |
| 역할 | 계좌를 지키며 꾸준히 불림 | 대박 복권 — 터지면 계좌 전체를 끌어올림 |
| 최악의 경우 | 낙폭 -30%대 | **전부 날려도 계좌의 5%만 손실** |

**1년에 한 번 "비중 원위치"가 핵심입니다**: 위험 주머니가 불어나 있으면 그만큼 **익절해서 안전으로 옮기고**,
쪼그라들어 있으면 안전에서 조금 떼어 **다시 5%로 채웁니다**. 이게 "오르면 팔고, 망하면 소액만 재도전"을
자동으로 해줍니다.
"""
)
with st.expander("표·그래프에 나오는 이름 설명 (클릭)"):
    st.markdown(
        """
| 이름 | 뜻 |
|---|---|
| **내 바벨** | 좌측 설정대로 안전+위험을 나누고 정한 주기로 비중 원위치 |
| **안전 주머니만 100%** | 비교용 — 1억 전부를 안전 전략에만 넣었다면? |
| **위험 주머니만 100% (참고)** | 비교용 극단 — 1억 전부를 위험 전략에. **실제로 하면 안 되는 선**이고, 위험 주머니의 "원료"가 어떤 물건인지 보여주는 용도 |
| **방치 바벨** | 처음 한 번만 나누고 다시는 비중을 안 건드림 — 위험 주머니가 커지는 대로 놔둬서 결국 계좌가 위험 전략에 지배됨 (아래 '위험 주머니 비중' 차트에서 확인) |

> ⚠️ 위험 주머니(BTC 돌파)의 과거 수익엔 2015~2020 코인 황금기가 포함돼 있어 절대값 재현은 기대하지 말 것.
> 유효한 결론은 **구조**: 안전 코어 + 소량 위험 + 정기 원위치가 '낙폭 대비 수익'을 끌어올린다는 것.
"""
    )

with st.expander("🎲 위험 주머니 전략(BTC 변동성 돌파)이 뭔가요? — 쉬운 설명"):
    st.markdown(
        """
매일 딱 세 가지만 합니다:

1. **어제 BTC가 얼마나 출렁였는지 잽니다** — 어제 최고가 − 최저가 = "레인지"
2. **오늘 가격이 `오늘 시작가 + 레인지 × 0.7`(매수선)을 뚫고 오르면 그 즉시 매수** —
   "오늘은 유난히 강한 날"이라는 증거가 나올 때만 삽니다
3. **그날 일봉이 마감되면 무조건 팝니다** — 내일은 처음부터 다시. 밤새 들고 가지 않습니다

**숫자 예시** (k=0.7): 어제 최고 105, 최저 100 → 레인지 5. 오늘 시작가 102 → 매수선 = 102 + 5×0.7 = **105.5**.
가격이 105.5를 찍으면 매수 → 마감이 109면 **+3.3%** 먹고 종료. 매수선을 못 찍는 날은 **아무것도 안 하고 현금** —
그래서 하락장은 대부분 구경만 하게 되고, 이게 이 전략의 숨은 방어막입니다.

**레버리지 2배란**: 500만 원으로 1,000만 원어치 포지션(500만은 거래소에서 빌림). 수익·손실이 2배가 되고,
빌린 돈에는 **펀딩비**(무기한 선물의 이자 같은 비용, 연 ~10%)가 붙습니다 — 좌측 설정값이 백테스트에서 실제로 차감됩니다.
"""
    )

GOLD_BASKETS = {
    "금 100%": {"GOLD": 1.0},
    "금 80% + 달러 20%": {"GOLD": 0.8, "UUP": 0.2},
    "금 60% + 달러 40%": {"GOLD": 0.6, "UUP": 0.4},
}
BASKET_FETCH = {"GOLD": "GC=F", "UUP": "UUP"}

with st.sidebar:
    st.header("기간 / 자본")
    start_date = st.date_input(
        "Start Date", value=date(2014, 9, 17),
        min_value=date(2014, 9, 17), max_value=date.today(),
        help="BTC 데이터 시작(2014-09-17)이 하한.",
    )
    end_date = st.date_input(
        "End Date", value=date.today(),
        min_value=date(2014, 9, 17), max_value=date.today(),
    )
    initial_capital = st.number_input(
        "Initial Capital", value=100_000_000, min_value=1_000, step=10_000_000
    )

    st.header("🛡️ 안전 주머니")
    basket_name = st.selectbox("방어 바스켓", list(GOLD_BASKETS), index=0)
    st.caption("TQQQ 50% + 방어 바스켓 50%, 밴드 리밸런싱 + 거시 레짐.")

    st.header("🎲 위험 주머니")
    risk_k = st.number_input(
        "BTC 돌파 계수 k", value=0.7, min_value=0.1, max_value=2.0, step=0.1,
        help="매수선 = 오늘 시작가 + 어제 출렁임(고가−저가) × k. "
        "클수록 확실한 급등에만 진입.",
    )
    risk_lev = st.number_input(
        "BTC 레버리지", value=2.0, min_value=0.5, max_value=4.0, step=0.5,
        help="500만으로 1,000만어치 포지션(2배). 수익·손실 2배 + 빌린 "
        "부분에 아래 펀딩비가 실제로 차감된다.",
    )
    financing_pct = st.number_input(
        "펀딩비 (연 %, 빌린 부분에만)", value=10.0, min_value=0.0,
        max_value=50.0, step=1.0,
        help="무기한 선물 펀딩비 평균(8시간당 ~0.01% ≈ 연 11%)의 근사. "
        "트레이드가 있는 날마다 1일치 차감. 레버리지 1배면 비용 0.",
    )

    st.header("⚖️ 바벨 구조")
    w_safe = st.number_input(
        "안전 주머니 비중 (%)", value=95.0, min_value=50.0, max_value=99.0, step=1.0,
        help="나머지가 위험 주머니. 95면 1억 기준 위험에 500만 원.",
    )
    rebalance_label = st.selectbox(
        "비중 원위치 주기", ["연 1회", "분기 1회", "방치 (원위치 없음)"], index=0,
    )

    st.header("수수료 / 세금")
    commission_pct = st.number_input(
        "미국주식 수수료 (%, 편도)", value=0.10, min_value=0.0, max_value=1.0,
        step=0.01, format="%.2f",
    )
    crypto_commission_pct = st.number_input(
        "크립토 수수료 (%, 편도)", value=0.05, min_value=0.0, max_value=1.0,
        step=0.01, format="%.2f",
    )
    tax_pct = st.number_input(
        "양도소득세율 (%)", value=22.0, min_value=0.0, max_value=50.0, step=1.0
    )

    run_btn = st.button("Run Backtest", type="primary", use_container_width=True)

if run_btn:
    market_data = CachedMarketDataAdapter(YFinanceAdapter())

    def _clean(x):
        if x.index.tz is not None:
            x.index = x.index.tz_localize(None)
        x.index = x.index.normalize()
        return x

    with st.spinner("가격 데이터 fetch..."):
        try:
            tqqq = _clean(market_data.fetch_ohlcv("TQQQ", start_date, end_date)["Close"])
            btc = _clean(market_data.fetch_ohlcv("BTC-USD", start_date, end_date))
            legs = {}
            for key in GOLD_BASKETS[basket_name]:
                legs[key] = _clean(
                    market_data.fetch_ohlcv(BASKET_FETCH[key], start_date, end_date)["Close"]
                )
        except Exception as e:
            st.error(f"데이터 fetch 실패: {e}")
            st.stop()

    risk_flags = None
    with st.spinner("거시 레짐 계산..."):
        try:
            from data.adapters.fred_adapter import FredCsvAdapter
            from strategy.adapters import macro_regime as mr

            fred = FredCsvAdapter()
            warm = start_date - timedelta(days=500)
            components = {
                "curve": mr.score_yield_curve(
                    fred.fetch_series("T10Y3M", date(1982, 1, 4), end_date)
                ),
                "credit": mr.score_credit_spread(
                    fred.fetch_series("BAA10Y", date(1986, 1, 2), end_date)
                ),
                "vix": mr.score_vix(
                    _clean(market_data.fetch_ohlcv("^VIX", warm, end_date)["Close"])
                ),
                "sahm": mr.score_sahm_rule(
                    fred.fetch_series("UNRATE", date(1990, 1, 1), end_date)
                ),
                "consumer": mr.score_ratio_trend(
                    _clean(market_data.fetch_ohlcv("XLY", warm, end_date)["Close"]),
                    _clean(market_data.fetch_ohlcv("XLP", warm, end_date)["Close"]),
                ),
            }
            composite = mr.composite_score(
                components, {k: 1.0 for k in components}, tqqq.index
            )
            trend = mr.score_price_trend(
                _clean(market_data.fetch_ohlcv("QQQ", warm, end_date)["Close"])
            ).reindex(tqqq.index, method="ffill")
            risk_flags = mr.hybrid_risk_on(trend, composite, 0.6, 10)
        except Exception as e:
            st.warning(f"거시 레짐 계산 실패 — 레짐 없이 진행: {e}")

    with st.spinner("두 주머니 백테스트..."):
        try:
            basket = pl.build_basket_series(legs, GOLD_BASKETS[basket_name])
            safe_res = BandRebalanceStrategy().execute(
                tqqq, basket,
                BandRebalanceConfig(
                    aggressive_ticker="TQQQ", defensive_ticker=basket_name,
                    start_date=start_date, end_date=end_date,
                    initial_capital=float(initial_capital),
                    fee_schedule=TossFeeSchedule(
                        buy_commission_pct=commission_pct / 100.0,
                        sell_commission_pct=commission_pct / 100.0,
                    ),
                    capital_gains_tax_pct=tax_pct / 100.0,
                ),
                risk_on_series=risk_flags,
            )
            risky_res = VolatilityBreakoutStrategy().execute(
                btc,
                VolBreakoutConfig(
                    ticker="BTC-USD", start_date=start_date, end_date=end_date,
                    initial_capital=float(initial_capital),
                    k=float(risk_k), leverage=float(risk_lev), slippage_bp=5.0,
                    financing_annual_rate=financing_pct / 100.0,
                    fee_schedule=TossFeeSchedule(
                        buy_commission_pct=crypto_commission_pct / 100.0,
                        sell_commission_pct=crypto_commission_pct / 100.0,
                        sec_fee_pct=0.0,
                    ),
                    capital_gains_tax_pct=tax_pct / 100.0,
                ),
            )
        except Exception as e:
            st.error(f"Backtest failed: {e}")
            st.stop()

    safe_v = pd.Series(
        [p.total for p in safe_res.curve],
        index=pd.to_datetime([p.date for p in safe_res.curve]),
    )
    risky_v = pd.Series(
        [p.equity for p in risky_res.equity_curve],
        index=pd.to_datetime([p.date for p in risky_res.equity_curve]),
    )
    st.session_state["barbell_state"] = {
        "safe_v": safe_v, "risky_v": risky_v,
        "w_safe": w_safe / 100.0,
        "rebalance": {"연 1회": "yearly", "분기 1회": "quarterly",
                      "방치 (원위치 없음)": "never"}[rebalance_label],
        "rebalance_label": rebalance_label,
        "capital": float(initial_capital),
    }

_state = st.session_state.get("barbell_state")
if _state is None:
    st.info("좌측에서 두 주머니와 비중을 정하고 **Run Backtest**를 눌러줘.")
    st.stop()
safe_v = _state["safe_v"]
risky_v = _state["risky_v"]
w = _state["w_safe"]
capital = _state["capital"]
w_risk = 1.0 - w

my_df, my_events = combine_barbell_detailed(
    safe_v, risky_v, w, _state["rebalance"], capital
)
never_df, _ = combine_barbell_detailed(safe_v, risky_v, w, "never", capital)
safe_only = (safe_v / float(safe_v.iloc[0]) * capital).dropna()
risky_only = (risky_v / float(risky_v.iloc[0]) * capital).dropna()

my_name = f"내 바벨 (안전 {w:.0%} + 위험 {w_risk:.0%}, {_state['rebalance_label']})"

# --- 한눈에 결과 ---
st.subheader("한눈에 결과")
final_my = float(my_df["total"].iloc[-1])
final_safe = float(safe_only.iloc[-1])
final_risky = float(risky_only.iloc[-1])
mdd_my = compute_risk_metrics(my_df["total"])["mdd"]
mdd_safe = compute_risk_metrics(safe_only)["mdd"]
c1, c2, c3 = st.columns(3)
c1.metric(
    "🛡️ 안전 주머니만 100%",
    f"{final_safe:,.0f}",
    f"MDD {mdd_safe:.0%}",
    delta_color="off",
)
c2.metric(
    f"⚖️ {my_name}",
    f"{final_my:,.0f}",
    f"안전만보다 {final_my / final_safe - 1.0:+.0%} · MDD {mdd_my:.0%}",
)
c3.metric(
    "🎲 위험 주머니만 100% (참고)",
    f"{final_risky:,.0f}",
    "실제로 하면 안 되는 극단 비교선",
    delta_color="off",
)
st.success(
    f"쉽게 말하면: {capital:,.0f}원을 전부 안전 전략에 넣었으면 "
    f"**{final_safe:,.0f}원**. 여기서 {capital * w_risk:,.0f}원"
    f"({w_risk:.0%})만 위험 주머니에 떼어 두고 "
    f"{_state['rebalance_label']}로 비중을 원위치했더니 "
    f"**{final_my:,.0f}원** — 안전만보다 "
    f"{final_my - final_safe:+,.0f}원 차이. 최대 낙폭은 "
    f"{mdd_safe:.0%} → {mdd_my:.0%}."
)

# --- 위험 주머니 비중 차트 (왜 원위치가 필요한가) ---
st.subheader("위험 주머니 비중 추이 — '비중 원위치'가 하는 일")
wt = go.Figure()
wt.add_trace(go.Scatter(
    x=my_df.index, y=my_df["risky_weight"], mode="lines",
    name=f"내 바벨 ({_state['rebalance_label']})",
    line=dict(color="#2196F3", width=1.8),
))
wt.add_trace(go.Scatter(
    x=never_df.index, y=never_df["risky_weight"], mode="lines",
    name="방치했다면 (원위치 없음)",
    line=dict(color="#E53935", width=1.4, dash="dash"),
))
wt.add_hline(
    y=w_risk, line_dash="dot", line_color="#9E9E9E",
    annotation_text=f"목표 {w_risk:.0%}",
)
wt.update_layout(
    template="plotly_dark", height=340,
    yaxis_tickformat=".0%", yaxis_title="계좌에서 위험 주머니가 차지하는 비중",
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    margin=dict(l=50, r=50, t=40, b=30),
)
st.plotly_chart(wt, use_container_width=True)
st.caption(
    "빨간 점선(방치)이 올라가는 건 위험 주머니가 불어나 **계좌 전체가 "
    "위험 전략으로 변해가는 것** — 바벨의 보호막이 사라진다. 파란 선은 "
    "원위치 덕에 비중이 계속 목표로 되돌아온다."
)

# --- 리밸런싱(원위치) 내역 ---
if my_events:
    st.subheader(f"비중 원위치 내역 ({len(my_events)}회)")
    ev_rows = [
        {
            "날짜": e["date"],
            "원위치 전 위험 비중": e["risky_weight_before"],
            "이동 금액": round(abs(e["moved"])),
            "방향": "🎲→🛡️ 익절 회수" if e["moved"] > 0 else "🛡️→🎲 재충전",
        }
        for e in my_events
    ]
    st.dataframe(
        ev_rows, use_container_width=True, hide_index=True,
        column_config={
            "원위치 전 위험 비중": st.column_config.NumberColumn(format="percent"),
            "이동 금액": st.column_config.NumberColumn(format="localized"),
        },
    )
    st.caption(
        "**익절 회수** = 위험 주머니가 목표보다 커져서 초과분을 안전으로 "
        "옮김 (수익 확정). **재충전** = 위험 주머니가 쪼그라들어 안전에서 "
        "소액을 떼어 다시 목표 비중으로 채움 (복권 재구매)."
    )

# --- 상세 비교 테이블 ---
st.subheader("상세 비교")
variants = {
    my_name: my_df["total"],
    "안전 주머니만 100%": safe_only,
    f"방치 바벨 (원위치 없음)": never_df["total"],
    "위험 주머니만 100% (참고)": risky_only,
}
DESCRIPTIONS = {
    my_name: "좌측 설정 그대로",
    "안전 주머니만 100%": "전액 안전 전략 — 바벨의 비교 기준",
    "방치 바벨 (원위치 없음)": "나누기만 하고 안 건드림 — 위험이 계좌를 지배",
    "위험 주머니만 100% (참고)": "전액 위험 전략 — 극단 비교선 (비추천)",
}
rows = []
for name, v in variants.items():
    v = v.dropna()
    if len(v) < 2:
        continue
    m = compute_risk_metrics(v)
    rows.append(
        {
            "포트폴리오": name,
            "설명": DESCRIPTIONS.get(name, ""),
            "최종": round(float(v.iloc[-1])),
            "CAGR": m["cagr"],
            "MDD": m["mdd"],
            "낙폭 1%당 수익 (Calmar)": round(m["calmar"], 2),
            "최악 연도": m["worst_year"],
        }
    )
st.dataframe(
    rows, use_container_width=True, hide_index=True,
    column_config={
        "최종": st.column_config.NumberColumn(format="localized"),
        "CAGR": st.column_config.NumberColumn(format="percent"),
        "MDD": st.column_config.NumberColumn(format="percent"),
        "최악 연도": st.column_config.NumberColumn(format="percent"),
    },
)

# --- 평가액 곡선 ---
log_scale = st.toggle("로그 스케일", value=True)
eq = go.Figure()
COLORS = {"내": "#2196F3", "안전": "#43A047", "방치": "#FFD600", "위험": "#E53935"}
for name, v in variants.items():
    color = ("#2196F3" if name.startswith("내") else
             "#43A047" if name.startswith("안전") else
             "#FFD600" if name.startswith("방치") else "#E53935")
    eq.add_trace(go.Scatter(
        x=v.index, y=v.values, mode="lines", name=name,
        line=dict(color=color,
                  width=2.2 if name.startswith("내") else 1.2,
                  dash="dot" if "참고" in name else None),
    ))
eq.update_layout(
    template="plotly_dark", height=470,
    yaxis_type="log" if log_scale else "linear",
    xaxis_title="Date", yaxis_title="Portfolio Value",
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    margin=dict(l=50, r=50, t=40, b=30),
)
st.plotly_chart(eq, use_container_width=True)

# --- 연도별 수익률 ---
st.subheader("연도별 수익률")
yr_rows = []
for name in (my_name, "안전 주머니만 100%", "위험 주머니만 100% (참고)"):
    v = variants[name].dropna()
    yearly = v.groupby(v.index.year).agg(["first", "last"])
    for y, row in yearly.iterrows():
        yr_rows.append({"연도": int(y), "포트폴리오": name,
                        "수익률": float(row["last"] / row["first"] - 1.0)})
yr_df = pd.DataFrame(yr_rows).pivot(index="연도", columns="포트폴리오", values="수익률")
st.dataframe(
    yr_df.reset_index(), use_container_width=True, hide_index=True,
    column_config={c: st.column_config.NumberColumn(format="percent")
                   for c in yr_df.columns},
)
