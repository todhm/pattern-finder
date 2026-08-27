from datetime import date

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data.adapters.cached_market_data import CachedMarketDataAdapter
from data.adapters.yfinance_adapter import YFinanceAdapter
from strategy.adapters.dip_buy_strategy import DipBuyStrategy
from strategy.adapters.risk_metrics import compute_risk_metrics
from pages._shared.formatting import fmt_price
from strategy.domain.models import DipBuyConfig, TossFeeSchedule

st.set_page_config(page_title="급락 매수 상세", layout="wide")
st.title("전일 급락 매수 — 단일 조합 상세 분석")
st.caption(
    "31번 그리드에서 고른 조합 하나를 **매매 한 건 한 건까지** 뜯어보는 "
    "페이지. 전날 −X% → 다음 날 시가 1/N 매수(최대 N회 스택) → 평단 "
    "+Y% 지정가 익절, 손절 없음. 토스 수수료 + 양도세 22% 반영."
)
st.caption("가격은 **분할·배당 조정가** 기준 — 실제 당시 호가와 다르다 (예: SOXL 2010년 표시 $0.66 = 실제 $40, 누적 60배 분할 반영). 규칙이 전부 %기반이라 수익률 결과는 동일하며, 호가단위·정수주 제약은 무시한다(1억 규모에서 오차 <0.01%).")

TICKERS = ["TQQQ", "SOXL", "UPRO", "KORU", "EDC", "YINN", "INDL", "QQQ", "SPY"]

with st.sidebar:
    st.header("종목 / 기간")
    ticker_choice = st.selectbox("종목", [*TICKERS, "직접 입력"], index=0)
    if ticker_choice == "직접 입력":
        ticker = st.text_input("티커", value="KORU").strip().upper()
    else:
        ticker = ticker_choice
    start_date = st.date_input(
        "Start Date", value=date(2010, 2, 11),
        min_value=date(2000, 1, 1), max_value=date.today(),
    )
    end_date = st.date_input(
        "End Date", value=date.today(),
        min_value=date(2000, 1, 1), max_value=date.today(),
    )
    initial_capital = st.number_input(
        "Initial Capital", value=100_000_000, min_value=1_000, step=10_000_000
    )

    st.header("전략 파라미터")
    drop_pct = st.number_input(
        "급락 기준 −X% (전날 종가)", value=5.0, min_value=1.0, max_value=20.0,
        step=0.5,
        help="그리드 최적은 −3%였지만 기본값은 원래 아이디어인 −5%.",
    )
    target_pct = st.number_input(
        "익절 목표 +Y%", value=1.2, min_value=0.3, max_value=20.0, step=0.1
    )
    split = st.number_input(
        "분할 수 (1/N씩, 최대 N회 스택)",
        value=3, min_value=1, max_value=10, step=1,
    )

    st.header("수수료 / 세금")
    commission_pct = st.number_input(
        "거래수수료 (%, 매수·매도 각각)",
        value=0.10, min_value=0.0, max_value=1.0, step=0.01, format="%.2f",
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

    config = DipBuyConfig(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        initial_capital=float(initial_capital),
        drop_pct=drop_pct / 100.0,
        target_pct=target_pct / 100.0,
        split=int(split),
        fee_schedule=TossFeeSchedule(
            buy_commission_pct=commission_pct / 100.0,
            sell_commission_pct=commission_pct / 100.0,
        ),
        capital_gains_tax_pct=tax_pct / 100.0,
    )

    with st.spinner("Running backtest..."):
        try:
            result = DipBuyStrategy().execute(daily, config)
        except Exception as e:
            st.error(f"Backtest failed: {e}")
            st.stop()
    # 세션에 보관 — 사이클 선택 등 위젯 조작으로 rerun 되어도
    # 백테스트를 다시 돌리지 않고 결과 화면을 유지한다.
    st.session_state["dip_detail_state"] = {"result": result, "daily": daily}

_state = st.session_state.get("dip_detail_state")
if _state is None:
    st.info("파라미터를 정하고 **Run Backtest**를 눌러줘.")
    st.stop()
result = _state["result"]
daily = _state["daily"]
# 표시용 값은 실행 시점 config에서 복원 — 사이드바를 바꿔도 화면은
# 마지막 실행 기준 (다시 반영하려면 Run Backtest).
config = result.config
ticker = config.ticker
target_pct = config.target_pct * 100.0
start_date, end_date = config.start_date, config.end_date
initial_capital = config.initial_capital

s = result.summary
liq = result.liquidation
closed = result.closed_cycles
open_cycles = [c for c in result.cycles if c.outcome == "open"]

st.subheader(s.name)
m1, m2, m3, m4 = st.columns(4)
m1.metric("최종 평가액 (세전)", f"{liq.final_value_pre_tax:,.0f}")
m2.metric("세후 청산 가치", f"{liq.final_value_after_tax:,.0f}")
m3.metric("CAGR (세전)", f"{s.cagr_pct:+.2%}")
m4.metric("최대 낙폭 (MDD)", f"{s.max_drawdown_pct:.1%}")

m5, m6, m7, m8 = st.columns(4)
m5.metric("완결 사이클", f"{len(closed)}회")
avg_hold = sum(c.holding_days for c in closed) / len(closed) if closed else 0
max_hold = max((c.holding_days for c in result.cycles), default=0)
m6.metric("평균 / 최장 보유", f"{avg_hold:.0f}일 / {max_hold:,}일")
m7.metric(
    "미청산 평가손익",
    f"{open_cycles[0].pnl:,.0f}" if open_cycles else "없음",
)
m8.metric("총 수수료 / 세금", f"{liq.total_fees:,.0f} / {liq.total_tax:,.0f}")

# --- 평가액 곡선 ---
log_scale = st.toggle("로그 스케일", value=True)
eq_fig = go.Figure()
eq_fig.add_trace(
    go.Scatter(
        x=[p.date for p in result.equity_curve],
        y=[p.equity for p in result.equity_curve],
        mode="lines", name=s.name, line=dict(color="#2196F3", width=2),
    )
)
bh = daily["Close"] / float(daily["Close"].iloc[0]) * float(initial_capital)
eq_fig.add_trace(
    go.Scatter(
        x=bh.index, y=bh.values, mode="lines", name=f"{ticker} 100%",
        line=dict(color="#E53935", width=1.2, dash="dot"),
    )
)
eq_fig.update_layout(
    template="plotly_dark", height=430,
    yaxis_type="log" if log_scale else "linear",
    xaxis_title="Date", yaxis_title="Portfolio Value",
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    margin=dict(l=50, r=50, t=40, b=30),
)
st.plotly_chart(eq_fig, use_container_width=True)

# --- 가격 + 매매 마커 ---
st.subheader("가격 · 매수/매도 지점")
px_fig = go.Figure()
px_fig.add_trace(
    go.Scatter(
        x=daily.index, y=daily["Close"], mode="lines",
        name=f"{ticker} 종가", line=dict(color="#B0BEC5", width=1.1),
    )
)
buys = [e for e in result.events if e.kind == "buy"]
sells = [e for e in result.events if e.kind == "sell"]
if buys:
    px_fig.add_trace(
        go.Scatter(
            x=[e.date for e in buys], y=[e.price for e in buys],
            mode="markers", name="매수 (시가)",
            marker=dict(symbol="triangle-up", size=7, color="#E53935"),
            hovertemplate="%{x}<br>매수 %{y:,.2f}<extra></extra>",
        )
    )
if sells:
    px_fig.add_trace(
        go.Scatter(
            x=[e.date for e in sells], y=[e.price for e in sells],
            mode="markers", name="익절 매도",
            marker=dict(symbol="triangle-down", size=8, color="#43A047"),
            hovertemplate="%{x}<br>매도 %{y:,.2f}<extra></extra>",
        )
    )
px_fig.update_layout(
    template="plotly_dark", height=430,
    yaxis_type="log" if log_scale else "linear",
    xaxis_title="Date", yaxis_title="Price",
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    margin=dict(l=50, r=50, t=40, b=30),
)
st.plotly_chart(px_fig, use_container_width=True)

# --- 사이클 요약 ---
st.subheader(f"사이클 요약 ({len(result.cycles)}회)")
cycle_rows = [
    {
        "사이클": i + 1,
        "시작": c.start,
        "종료": c.end if c.end else "진행 중",
        "보유일": c.holding_days,
        "매수 횟수": c.n_buys,
        "평단": fmt_price(c.avg_price),
        "청산가": fmt_price(c.exit_price) if c.exit_price else "—",
        "투입": round(c.invested),
        "손익": round(c.pnl),
        "결과": "익절" if c.outcome == "profit" else "진행 중",
    }
    for i, c in enumerate(result.cycles)
]
st.dataframe(
    cycle_rows,
    use_container_width=True,
    hide_index=True,
    column_config={
        "투입": st.column_config.NumberColumn(format="localized"),
        "손익": st.column_config.NumberColumn(format="localized"),
    },
)

# --- 사이클 상세: 매매 하나하나 ---
st.subheader("사이클 상세 — 매매 전체 기록")
events_by_cycle: dict[int, list] = {}
for e in result.events:
    events_by_cycle.setdefault(e.cycle_no, []).append(e)

cycle_options = {
    f"#{i + 1}  {c.start} ~ {c.end if c.end else '진행 중'} · "
    f"{'익절' if c.outcome == 'profit' else '진행 중'} ({c.pnl:+,.0f})": i + 1
    for i, c in enumerate(result.cycles)
}
if cycle_options:
    selected_label = st.selectbox("사이클 선택", list(cycle_options), index=0)
    selected_no = cycle_options[selected_label]
    KIND_LABELS = {"buy": "매수 (시가)", "sell": "익절 매도", "tax": "양도세 납부"}
    detail_rows = [
        {
            "날짜": e.date,
            "구분": KIND_LABELS.get(e.kind, e.kind),
            "물타기 회차": e.stack_no if e.stack_no else "",
            "트리거 (전날 등락)": f"{e.trigger_ret:+.2%}" if e.kind == "buy" else "",
            "체결가": fmt_price(e.price),
            "수량": f"{e.qty:,.2f}" if e.qty else "",
            "금액 (+매수/−매도)": f"{e.notional:,.0f}",
            "체결 후 보유": f"{e.shares_after:,.2f}",
            "체결 후 평단": fmt_price(e.avg_price_after),
            "익절 목표가": fmt_price(e.target_price),
            "현금 잔고": f"{e.cash_after:,.0f}",
        }
        for e in events_by_cycle.get(selected_no, [])
    ]
    st.dataframe(detail_rows, use_container_width=True, hide_index=True)
    st.caption(
        "**트리거** = 매수를 유발한 전날 일간 등락률 · **익절 목표가** = "
        f"체결 후 평단 × (1+{target_pct:.1f}%) — 물타기로 평단이 내려가면 "
        "목표가도 같이 내려온다."
    )
else:
    st.info("이 구간엔 매매가 한 번도 없었어.")

all_events_df = pd.DataFrame([e.model_dump() for e in result.events])
if len(all_events_df):
    st.download_button(
        f"전체 매매 로그 CSV 다운로드 ({len(result.events):,}건)",
        all_events_df.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"dip_buy_{ticker}_{start_date}_{end_date}.csv",
        mime="text/csv",
    )

# --- 리스크 요약 ---
values = pd.Series(
    [p.equity for p in result.equity_curve],
    index=pd.to_datetime([p.date for p in result.equity_curve]),
)
m = compute_risk_metrics(values)
st.caption(
    f"리스크: 등급 **{m['grade']}** · 최장 수면기간 "
    f"{m['longest_underwater_days'] / 365.25:.1f}년 · Sharpe {m['sharpe']:.2f} · "
    f"Calmar {m['calmar']:.2f} · 최악 연도 {m['worst_year']:+.1%} · "
    f"동일 구간 존버 세후 {result.benchmark_after_tax:,.0f}"
)
