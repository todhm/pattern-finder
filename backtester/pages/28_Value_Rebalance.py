from datetime import date

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data.adapters.cached_market_data import CachedMarketDataAdapter
from data.adapters.yfinance_adapter import YFinanceAdapter
from strategy.adapters.value_rebalance_strategy import ValueRebalanceStrategy
from strategy.domain.models import TossFeeSchedule, ValueRebalanceConfig

st.set_page_config(page_title="라오어 밸류 리밸런싱", layout="wide")
st.title("라오어 밸류 리밸런싱 (VR) — TQQQ 거치식")
st.caption(
    "책 『라오어의 미국주식 밸류 리밸런싱』의 거치식 VR: 주식 75% : "
    "Pool 25%로 시작, **2주(10거래일)마다 V += Pool/G** (G=10)로 "
    "밸류패스를 올리고, 평가금이 밴드 상단(V+15%)을 넘으면 초과분 "
    "매도, 하단(V−15%) 아래면 Pool 한도 내 매수. **기본공식과 "
    "실력공식(√(E/V) 보정 근사)을 같이 돌려 비교**한다. 토스증권 "
    "수수료 + 양도세 22%(연 250만 공제) 반영. 실력공식 원본은 서적 "
    "전용이라 공개된 설명 기반 근사임."
)

TQQQ_INCEPTION = date(2010, 2, 11)

# --- Sidebar ---
with st.sidebar:
    st.header("기간 / 자본")
    start_date = st.date_input(
        "Start Date",
        value=TQQQ_INCEPTION,
        min_value=TQQQ_INCEPTION,
        max_value=date.today(),
    )
    end_date = st.date_input(
        "End Date",
        value=date.today(),
        min_value=TQQQ_INCEPTION,
        max_value=date.today(),
    )
    initial_capital = st.number_input(
        "Initial Capital", value=100_000_000, min_value=1_000, step=10_000_000
    )

    st.header("VR 파라미터")
    stock_ratio = st.number_input(
        "초기 주식 비율 (%)",
        value=75.0, min_value=10.0, max_value=100.0, step=5.0,
        help="나머지는 Pool(현금). 거치식 기준.",
    )
    gradient = st.number_input(
        "G (기울기 분모)",
        value=10.0, min_value=1.0, max_value=50.0, step=1.0,
        help="사이클마다 V += Pool/G. 클수록 밸류패스가 완만해져 "
        "보수적. 거치식·적립식 10, 인출식 20 권장 (책 기준).",
    )
    cycle_days = st.number_input(
        "사이클 (거래일)",
        value=10, min_value=1, max_value=60, step=1,
        help="V값 갱신 주기. 책 기준 2주 = 10거래일.",
    )
    band_pct = st.number_input(
        "밴드 폭 (±%)",
        value=15.0, min_value=1.0, max_value=50.0, step=1.0,
    )
    check_daily = st.checkbox(
        "매일 밴드 검사",
        value=True,
        help="켜면 매일 밴드 이탈을 검사 (책의 매수표·매도표 LOC "
        "방식 근사). 끄면 2주 시점에만 검사.",
    )
    to_edge = st.selectbox(
        "체결 방식",
        ["밴드 가장자리까지 (책 방식 — 분할 체결 등가)", "V까지 한 번에 (공격적)"],
        index=0,
        help="책의 매수표·매도표는 가격대별 LOC 분할 주문이다. 매일 "
        "종가에 밴드 가장자리까지만 복원하면 종가가 한 단계 더 "
        "벗어질 때마다 그만큼만 추가 체결되므로 분할 매매와 등가. "
        "'V까지 한 번에'는 이탈 즉시 부족분 전액을 채우는 공격적 "
        "변형.",
    ).startswith("밴드")
    pool_rate = st.number_input(
        "Pool 연이율 (%) — P2P 채권 운용",
        value=0.0, min_value=0.0, max_value=30.0, step=1.0,
        help="Pool(현금)을 P2P 채권 등으로 굴릴 때의 연이율. 0이면 "
        "무수익 현금(순정 VR). 일할 누적이며 즉시 인출 가능 가정 — "
        "1년 만기 락업의 유동성 제약은 반영하지 않는 낙관적 상한.",
    )

    st.header("수수료 / 세금")
    commission_pct = st.number_input(
        "거래수수료 (%, 매수·매도 각각)",
        value=0.10, min_value=0.0, max_value=1.0, step=0.01, format="%.2f",
    )
    tax_pct = st.number_input(
        "양도소득세율 (%)",
        value=22.0, min_value=0.0, max_value=50.0, step=1.0,
    )
    tax_deduction = st.number_input(
        "연간 기본공제", value=2_500_000, min_value=0, step=500_000
    )

    run_btn = st.button("Run Backtest", type="primary", use_container_width=True)

if not run_btn:
    st.info("좌측에서 VR 파라미터를 설정하고 **Run Backtest**를 눌러줘.")
    st.stop()

# --- Fetch ---
market_data = CachedMarketDataAdapter(YFinanceAdapter())
with st.spinner("Fetching TQQQ..."):
    try:
        df = market_data.fetch_ohlcv("TQQQ", start_date, end_date)
    except Exception as e:
        st.error(f"Failed to fetch data: {e}")
        st.stop()

close = df["Close"]
if close.index.tz is not None:
    close.index = close.index.tz_localize(None)
close.index = close.index.normalize()


def _make_config(advanced: bool) -> ValueRebalanceConfig:
    return ValueRebalanceConfig(
        start_date=start_date,
        end_date=end_date,
        initial_capital=float(initial_capital),
        stock_ratio=stock_ratio / 100.0,
        gradient=float(gradient),
        cycle_days=int(cycle_days),
        band_pct=band_pct / 100.0,
        check_daily=check_daily,
        advanced_formula=advanced,
        rebalance_to_edge=to_edge,
        pool_annual_rate=pool_rate / 100.0,
        fee_schedule=TossFeeSchedule(
            buy_commission_pct=commission_pct / 100.0,
            sell_commission_pct=commission_pct / 100.0,
        ),
        capital_gains_tax_pct=tax_pct / 100.0,
        tax_deduction=float(tax_deduction),
    )


with st.spinner("Running backtest (기본공식 + 실력공식)..."):
    try:
        strategy = ValueRebalanceStrategy()
        result = strategy.execute(close, _make_config(advanced=False))
        adv_result = strategy.execute(close, _make_config(advanced=True))
    except Exception as e:
        st.error(f"Backtest failed: {e}")
        st.stop()

s = result.summary
liq = result.liquidation
st.subheader(s.name)
st.caption(
    f"{result.curve[0].date} → {result.curve[-1].date} · 실제 TQQQ "
    "배당 조정 종가 · 체결은 당일 종가"
)

# --- Headline metrics (기본공식) ---
m1, m2, m3, m4 = st.columns(4)
m1.metric("최종 평가액 (세전)", f"{liq.final_value_pre_tax:,.0f}")
m2.metric(
    "세후 청산 가치",
    f"{liq.final_value_after_tax:,.0f}",
    help="마지막 날 전량 매도 가정: 매도 수수료 + 양도세 차감.",
)
m3.metric("CAGR (세전)", f"{s.cagr_pct:+.2%}")
m4.metric("최대 낙폭 (MDD)", f"{s.max_drawdown_pct:.1%}")

m5, m6, m7, m8 = st.columns(4)
m5.metric("매도 (상단 초과)", f"{result.sell_count}회")
m6.metric("매수 (하단 이탈)", f"{result.buy_count}회")
m7.metric("총 수수료", f"{liq.total_fees:,.0f}")
m8.metric(
    "총 양도소득세",
    f"{liq.total_tax:,.0f}",
    help=f"연 단위 정산 + 최종 청산분 {liq.final_tax:,.0f} 포함.",
)

# --- Comparison table ---
st.subheader("기본공식 vs 실력공식 vs 벤치마크 (세후 청산 기준)")
rows = []
for r in (result, adv_result):
    lq = r.liquidation
    rows.append(
        {
            "포트폴리오": r.summary.name,
            "최종 평가액 (세전)": f"{lq.final_value_pre_tax:,.0f}",
            "세후 청산가": f"{lq.final_value_after_tax:,.0f}",
            "세후 수익률": (
                f"{lq.final_value_after_tax / float(initial_capital) - 1.0:+.1%}"
            ),
            "CAGR (세전)": f"{r.summary.cagr_pct:+.2%}",
            "MDD": f"{r.summary.max_drawdown_pct:.1%}",
            "매수/매도": f"{r.buy_count} / {r.sell_count}",
        }
    )
for b in result.benchmarks:
    at = result.benchmark_after_tax.get(b.name, b.final_value)
    rows.append(
        {
            "포트폴리오": b.name,
            "최종 평가액 (세전)": f"{b.final_value:,.0f}",
            "세후 청산가": f"{at:,.0f}",
            "세후 수익률": f"{at / float(initial_capital) - 1.0:+.1%}",
            "CAGR (세전)": f"{b.cagr_pct:+.2%}",
            "MDD": f"{b.max_drawdown_pct:.1%}",
            "매수/매도": "—",
        }
    )
st.dataframe(rows, use_container_width=True, hide_index=True)

# --- Equity curves ---
log_scale = st.toggle("로그 스케일", value=True)
curve_dates = [pt.date for pt in result.curve]

eq_fig = go.Figure()
eq_fig.add_trace(
    go.Scatter(
        x=curve_dates,
        y=[pt.total for pt in result.curve],
        mode="lines",
        name="VR 기본공식",
        line=dict(color="#2196F3", width=2.5),
    )
)
eq_fig.add_trace(
    go.Scatter(
        x=[pt.date for pt in adv_result.curve],
        y=[pt.total for pt in adv_result.curve],
        mode="lines",
        name="VR 실력공식 (√보정)",
        line=dict(color="#FFD600", width=2.0, dash="dash"),
    )
)
for color, (name, points) in zip(
    ["#E53935", "#9E9E9E"], result.benchmark_curves.items()
):
    eq_fig.add_trace(
        go.Scatter(
            x=[p.date for p in points],
            y=[p.equity for p in points],
            mode="lines",
            name=name,
            line=dict(color=color, width=1.3, dash="dot"),
        )
    )
eq_fig.update_layout(
    title="평가액 추이 — VR vs 벤치마크",
    template="plotly_dark",
    height=500,
    yaxis_type="log" if log_scale else "linear",
    xaxis_title="Date",
    yaxis_title="Portfolio Value",
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    margin=dict(l=50, r=50, t=80, b=30),
)
st.plotly_chart(eq_fig, use_container_width=True)

# --- 밸류패스 vs 평가금 (핵심 시각화) ---
st.subheader("밸류패스(V) vs 주식 평가금(E) — 기본공식")
vp_fig = go.Figure()
vp = pd.Series([pt.value_path for pt in result.curve], index=curve_dates)
ev = pd.Series([pt.stock_value for pt in result.curve], index=curve_dates)
vp_fig.add_trace(
    go.Scatter(
        x=vp.index, y=(vp * (1 + band_pct / 100.0)).values, mode="lines",
        name=f"밴드 상단 (+{band_pct:.0f}%)",
        line=dict(color="#43A047", width=1, dash="dot"),
    )
)
vp_fig.add_trace(
    go.Scatter(
        x=vp.index, y=(vp * (1 - band_pct / 100.0)).values, mode="lines",
        name=f"밴드 하단 (−{band_pct:.0f}%)",
        line=dict(color="#E53935", width=1, dash="dot"),
        fill="tonexty", fillcolor="rgba(158,158,158,0.08)",
    )
)
vp_fig.add_trace(
    go.Scatter(
        x=vp.index, y=vp.values, mode="lines",
        name="밸류패스 V",
        line=dict(color="#FFC107", width=1.8, dash="dash"),
    )
)
vp_fig.add_trace(
    go.Scatter(
        x=ev.index, y=ev.values, mode="lines",
        name="주식 평가금 E",
        line=dict(color="#2196F3", width=1.8),
    )
)
sell_events = [e for e in result.events if e.kind == "sell"]
buy_events = [e for e in result.events if e.kind == "buy"]
if sell_events:
    vp_fig.add_trace(
        go.Scatter(
            x=[e.date for e in sell_events],
            y=[e.stock_value_after for e in sell_events],
            mode="markers", name="매도 (초과분)",
            marker=dict(symbol="triangle-down", size=9, color="#43A047"),
        )
    )
if buy_events:
    vp_fig.add_trace(
        go.Scatter(
            x=[e.date for e in buy_events],
            y=[e.stock_value_after for e in buy_events],
            mode="markers", name="매수 (부족분)",
            marker=dict(symbol="triangle-up", size=9, color="#E53935"),
        )
    )
vp_fig.update_layout(
    template="plotly_dark",
    height=450,
    yaxis_type="log" if log_scale else "linear",
    xaxis_title="Date",
    yaxis_title="Value",
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    margin=dict(l=50, r=50, t=60, b=30),
)
st.plotly_chart(vp_fig, use_container_width=True)

# --- Pool 비중 추이 ---
st.subheader("Pool(현금) 비중 추이")
pool_fig = go.Figure()
for name_, res, color in (
    ("기본공식", result, "#2196F3"),
    ("실력공식", adv_result, "#FFD600"),
):
    pool_w = pd.Series(
        [pt.pool / pt.total if pt.total > 0 else 0.0 for pt in res.curve],
        index=[pt.date for pt in res.curve],
    )
    pool_fig.add_trace(
        go.Scatter(
            x=pool_w.index, y=pool_w.values, mode="lines",
            name=f"Pool 비중 — {name_}",
            line=dict(color=color, width=1.5),
        )
    )
pool_fig.add_hline(
    y=1.0 - stock_ratio / 100.0,
    line_dash="dash", line_color="#9E9E9E",
    annotation_text=f"초기 {100 - stock_ratio:.0f}%",
)
pool_fig.update_layout(
    template="plotly_dark", height=300,
    xaxis_title="Date", yaxis_title="Pool Weight",
    yaxis_tickformat=".0%",
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    margin=dict(l=50, r=50, t=40, b=30),
)
st.plotly_chart(pool_fig, use_container_width=True)

# --- Drawdown ---
st.subheader("낙폭 (Drawdown)")
dd_fig = go.Figure()
for name_, res, color, width in (
    ("VR 기본공식", result, "#2196F3", 2.0),
    ("VR 실력공식", adv_result, "#FFD600", 1.5),
):
    values = pd.Series(
        [pt.total for pt in res.curve], index=[pt.date for pt in res.curve]
    )
    dd = values / values.cummax() - 1.0
    dd_fig.add_trace(
        go.Scatter(
            x=dd.index, y=dd.values, mode="lines", name=name_,
            line=dict(color=color, width=width),
        )
    )
for color, (name, points) in zip(
    ["#E53935", "#9E9E9E"], result.benchmark_curves.items()
):
    bench_values = pd.Series(
        [p.equity for p in points], index=[p.date for p in points]
    )
    bench_dd = bench_values / bench_values.cummax() - 1.0
    dd_fig.add_trace(
        go.Scatter(
            x=bench_dd.index, y=bench_dd.values, mode="lines", name=name,
            line=dict(color=color, width=1, dash="dot"),
        )
    )
dd_fig.update_layout(
    template="plotly_dark", height=350,
    xaxis_title="Date", yaxis_title="Drawdown",
    yaxis_tickformat=".0%",
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    margin=dict(l=50, r=50, t=60, b=30),
)
st.plotly_chart(dd_fig, use_container_width=True)

# --- Events table ---
st.subheader(f"매매 이벤트 — 기본공식 ({len(result.events)}건)")
KIND_LABELS = {"sell": "매도 (상단 초과)", "buy": "매수 (하단 이탈)", "tax": "양도세 납부"}
event_rows = [
    {
        "날짜": e.date,
        "구분": KIND_LABELS.get(e.kind, e.kind),
        "거래액 (+매수/−매도)": f"{e.traded:,.0f}",
        "밸류패스 V": f"{e.value_path:,.0f}",
        "주식 평가금": f"{e.stock_value_after:,.0f}",
        "Pool": f"{e.pool_after:,.0f}",
    }
    for e in result.events
]
st.dataframe(event_rows, use_container_width=True, hide_index=True)
