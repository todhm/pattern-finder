from datetime import date

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data.adapters.cached_market_data import CachedMarketDataAdapter
from data.adapters.yfinance_adapter import YFinanceAdapter
from pages._shared.formatting import fmt_price
from strategy.adapters.risk_metrics import compute_risk_metrics
from strategy.adapters.volatility_breakout_strategy import (
    VolatilityBreakoutStrategy,
)
from strategy.domain.models import TossFeeSchedule, VolBreakoutConfig

st.set_page_config(page_title="미국주식 돌파 상세", layout="wide")
st.title("미국주식 변동성 돌파 — 종목별 트레이드 상세")
st.caption(
    "37번(알트 상세)과 같은 규칙을 **미국 주식/ETF**에 적용: 매일 시가 "
    "+ k×전일 레인지 돌파 시 매수(트리거가 체결가), 당일 종가 청산, "
    "노셔널 = 평가액 × 레버리지(데이트레이드 마진, 당일 청산이라 이자 "
    "0). 토스 수수료(매수·매도 각 0.1% + SEC fee)·슬리피지·양도세 반영."
)
st.caption(
    "⚠️ **사전 검증 요지 (2026-08)**: 3배 ETF(TQQQ)에서는 비용 전 "
    "엣지가 연 +10% 수준이라 왕복 ~0.25% 비용이 전부 삼켜 **전 조합 "
    "손실**이었다. 미국 대형주는 크립토보다 변동성이 낮아 돌파 "
    "전략에 구조적으로 불리 — 이 페이지는 그걸 종목별로 직접 확인하는 "
    "용도다. 가격은 분할·배당 조정가."
)

TICKERS = [
    # 레버리지 ETF
    "TQQQ", "SOXL", "UPRO", "TNA", "FAS", "QLD",
    # 지수
    "QQQ", "SPY", "IWM",
    # 고변동 대형주
    "NVDA", "TSLA", "PLTR", "COIN", "MSTR", "AMD",
    "AAPL", "MSFT", "AMZN", "META",
]

with st.sidebar:
    st.header("종목 / 기간")
    ticker_choice = st.selectbox(
        "종목", [*TICKERS, "직접 입력"], index=0,
        help="레버리지 ETF·지수·고변동 대형주 프리셋. 임의 티커는 직접 입력.",
    )
    if ticker_choice == "직접 입력":
        ticker = st.text_input("티커", value="NVDA").strip().upper()
    else:
        ticker = ticker_choice
    start_date = st.date_input(
        "Start Date", value=date(2020, 1, 2),
        min_value=date(2000, 1, 3), max_value=date.today(),
        help="올해 성적만 보려면 2026-01-01로. 상장일 이전이면 데이터 "
        "시작부터.",
    )
    end_date = st.date_input(
        "End Date", value=date.today(),
        min_value=date(2000, 1, 3), max_value=date.today(),
    )
    initial_capital = st.number_input(
        "Initial Capital", value=100_000_000, min_value=1_000, step=10_000_000
    )

    st.header("돌파 규칙")
    k = st.number_input("돌파 계수 k", value=0.7, min_value=0.1, max_value=2.0, step=0.1)
    leverage = st.number_input(
        "레버리지", value=1.0, min_value=0.5, max_value=4.0, step=0.5,
        help="데이트레이드 마진 가정 (당일 청산 — 이자 0).",
    )
    trend_days = st.number_input(
        "추세 필터 (N일선, 0=끔)", value=0, min_value=0, max_value=300, step=10,
        help="전일 종가가 N일선 위일 때만 진입.",
    )

    st.header("비용 / 세금")
    commission_pct = st.number_input(
        "거래수수료 (%, 매수·매도 각각)",
        value=0.10, min_value=0.0, max_value=1.0, step=0.01, format="%.2f",
    )
    slippage_bp = st.number_input(
        "슬리피지 (bp, 편도)", value=5.0, min_value=0.0, max_value=100.0, step=1.0
    )
    tax_pct = st.number_input(
        "양도소득세율 (%)", value=22.0, min_value=0.0, max_value=50.0, step=1.0
    )

    run_btn = st.button("Run Backtest", type="primary", use_container_width=True)

if run_btn:
    market_data = CachedMarketDataAdapter(YFinanceAdapter())
    with st.spinner(f"Fetching {ticker}..."):
        try:
            daily = market_data.fetch_ohlcv(ticker, start_date, end_date)
        except Exception as e:
            st.error(f"데이터 fetch 실패: {e}")
            st.stop()
    if daily.index.tz is not None:
        daily.index = daily.index.tz_localize(None)
    daily.index = daily.index.normalize()
    if len(daily) < 40:
        st.error(f"{ticker}: 데이터 {len(daily)}봉 — 최소 40일 필요.")
        st.stop()

    cfg = VolBreakoutConfig(
        ticker=ticker,
        start_date=daily.index[0].date(),
        end_date=daily.index[-1].date(),
        initial_capital=float(initial_capital),
        k=float(k), leverage=float(leverage),
        trend_filter_days=int(trend_days),
        slippage_bp=float(slippage_bp),
        fee_schedule=TossFeeSchedule(
            buy_commission_pct=commission_pct / 100.0,
            sell_commission_pct=commission_pct / 100.0,
        ),
        capital_gains_tax_pct=tax_pct / 100.0,
    )
    with st.spinner("Running backtest..."):
        try:
            result = VolatilityBreakoutStrategy().execute(daily, cfg)
        except Exception as e:
            st.error(f"Backtest failed: {e}")
            st.stop()
    st.session_state["stock_detail_state"] = {"result": result, "daily": daily}

_state = st.session_state.get("stock_detail_state")
if _state is None:
    st.info("종목과 규칙을 정하고 **Run Backtest**를 눌러줘.")
    st.stop()
result = _state["result"]
daily = _state["daily"]
cfg = result.config
ticker = cfg.ticker

s = result.summary
liq = result.liquidation
st.subheader(s.name)
st.caption(
    f"{result.equity_curve[0].date} → {result.equity_curve[-1].date} · "
    f"거래일 {len(result.equity_curve):,}일 중 트레이드 {len(result.trades):,}회"
)

m1, m2, m3, m4 = st.columns(4)
m1.metric("최종 평가액 (세전)", f"{liq.final_value_pre_tax:,.0f}")
m2.metric("세후 청산 가치", f"{liq.final_value_after_tax:,.0f}")
m3.metric("CAGR (세전)", f"{s.cagr_pct:+.2%}")
m4.metric("최대 낙폭 (MDD)", f"{s.max_drawdown_pct:.1%}")

m5, m6, m7, m8 = st.columns(4)
m5.metric("승률", f"{result.win_rate:.1%}")
m6.metric(
    "평균 손익 (노셔널)",
    f"+{result.avg_win_pct:.2%} / −{result.avg_loss_pct:.2%}",
)
m7.metric(
    "추정 켈리 f*", f"{result.kelly_fraction:.2f}",
    help="이보다 큰 레버리지는 장기적으로 파산 방향. 0이면 엣지 없음.",
)
m8.metric("총 비용 / 세금", f"{liq.total_fees:,.0f} / {liq.total_tax:,.0f}")
if result.ruined:
    st.error("💀 이 설정은 파산했습니다 (평가액이 초기의 10% 이하).")

# --- 평가액 곡선 ---
log_scale = st.toggle("로그 스케일", value=True)
eq = go.Figure()
eq.add_trace(go.Scatter(
    x=[p.date for p in result.equity_curve],
    y=[p.equity for p in result.equity_curve],
    mode="lines", name="전략", line=dict(color="#2196F3", width=2),
))
bh = daily["Close"] / float(daily["Close"].iloc[0]) * float(cfg.initial_capital)
eq.add_trace(go.Scatter(
    x=bh.index, y=bh.values, mode="lines", name=f"{ticker} 존버",
    line=dict(color="#E53935", width=1.2, dash="dot"),
))
eq.update_layout(
    template="plotly_dark", height=430,
    yaxis_type="log" if log_scale else "linear",
    xaxis_title="Date", yaxis_title="Equity",
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    margin=dict(l=50, r=50, t=40, b=30),
)
st.plotly_chart(eq, use_container_width=True)

# --- 가격 + 트레이드 마커 ---
st.subheader("가격 · 트레이드 지점 (초록=익절, 빨강=손실)")
px_fig = go.Figure()
px_fig.add_trace(go.Scatter(
    x=daily.index, y=daily["Close"], mode="lines",
    name=f"{ticker} 종가", line=dict(color="#B0BEC5", width=1.0),
))
wins = [t for t in result.trades if t.pnl > 0]
losses = [t for t in result.trades if t.pnl <= 0]
if wins:
    px_fig.add_trace(go.Scatter(
        x=[t.date for t in wins], y=[t.entry for t in wins],
        mode="markers", name=f"익절 진입 ({len(wins)})",
        marker=dict(symbol="triangle-up", size=6, color="#43A047"),
        hovertemplate="%{x}<br>진입 %{y:,.4f}<extra></extra>",
    ))
if losses:
    px_fig.add_trace(go.Scatter(
        x=[t.date for t in losses], y=[t.entry for t in losses],
        mode="markers", name=f"손실 진입 ({len(losses)})",
        marker=dict(symbol="triangle-down", size=6, color="#E53935"),
        hovertemplate="%{x}<br>진입 %{y:,.4f}<extra></extra>",
    ))
px_fig.update_layout(
    template="plotly_dark", height=430,
    yaxis_type="log" if log_scale else "linear",
    xaxis_title="Date", yaxis_title="Price",
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    margin=dict(l=50, r=50, t=40, b=30),
)
st.plotly_chart(px_fig, use_container_width=True)

# --- 연×월 수익률 매트릭스 ---
st.subheader("월별 수익률")
values = pd.Series(
    [p.equity for p in result.equity_curve],
    index=pd.to_datetime([p.date for p in result.equity_curve]),
)
monthly = values.resample("ME").last().pct_change()
first_month = values.resample("ME").last()
if len(first_month) > 0 and float(values.iloc[0]) > 0:
    monthly.iloc[0] = float(first_month.iloc[0] / values.iloc[0] - 1.0)
mm = monthly.to_frame("ret")
mm["연도"] = mm.index.year
mm["월"] = mm.index.month
pivot = mm.pivot(index="연도", columns="월", values="ret")
st.dataframe(
    pivot.reset_index(), use_container_width=True, hide_index=True,
    column_config={c: st.column_config.NumberColumn(format="percent")
                   for c in pivot.columns},
)

# --- 트레이드 전체 기록 ---
st.subheader(f"트레이드 전체 기록 ({len(result.trades):,}건)")
year_options = ["전체"] + sorted(
    {t.date.year for t in result.trades}, reverse=True
)
sel_year = st.selectbox("연도 필터", year_options, index=0)
shown = [
    t for t in result.trades
    if sel_year == "전체" or t.date.year == sel_year
]
trade_rows = [
    {
        "날짜": t.date,
        "진입 (트리거)": fmt_price(t.entry),
        "청산 (종가)": fmt_price(t.exit),
        "수익률 (노셔널)": t.ret_pct,
        "손익": round(t.pnl),
        "체결 후 평가액": round(t.equity_after),
    }
    for t in shown
]
st.dataframe(
    trade_rows, use_container_width=True, hide_index=True,
    column_config={
        "수익률 (노셔널)": st.column_config.NumberColumn(format="percent"),
        "손익": st.column_config.NumberColumn(format="localized"),
        "체결 후 평가액": st.column_config.NumberColumn(format="localized"),
    },
)
all_df = pd.DataFrame([t.model_dump() for t in result.trades])
if len(all_df):
    st.download_button(
        f"전체 트레이드 CSV ({len(result.trades):,}건)",
        all_df.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"stock_breakout_{ticker}_k{cfg.k:g}_x{cfg.leverage:g}.csv",
        mime="text/csv",
    )

m = compute_risk_metrics(values)
st.caption(
    f"리스크: 등급 **{m['grade']}** · 최장 수면기간 "
    f"{m['longest_underwater_days'] / 365.25:.1f}년 · Sharpe {m['sharpe']:.2f} · "
    f"Calmar {m['calmar']:.2f} · 최악 연도 {m['worst_year']:+.1%} · "
    f"일간 VaR95 {m['var95']:.1%}"
)
