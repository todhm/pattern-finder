from datetime import date, timedelta

import streamlit as st

from data.adapters.yfinance_adapter import YFinanceAdapter
from pattern.adapters.wedge_pop import WedgePopDetector
from strategy.adapters.wedgepop_strategy import WedgepopStrategy
from strategy.domain.models import StrategyConfig
from visualization.adapters.plotly_charts import PlotlyChartBuilder

st.set_page_config(page_title="Strategy", layout="wide")
st.title("Wedge Pop Strategy")
st.caption(
    "Wedge Pop 신호 다음날 open 매수 → 손절(consolidation low) / "
    "Exhaustion 익절 / 10 EMA trail / time stop"
)

# --- Sidebar inputs ---
with st.sidebar:
    st.header("Market")
    ticker = st.text_input("Ticker", value="AAPL")
    start_date = st.date_input(
        "Start Date", value=date.today() - timedelta(days=365)
    )
    end_date = st.date_input("End Date", value=date.today())

    st.header("Risk")
    initial_capital = st.number_input(
        "Initial Capital ($)",
        value=100_000,
        min_value=1_000,
        step=10_000,
    )
    risk_pct = st.number_input(
        "Risk per Trade (%)",
        value=2.0,
        min_value=0.1,
        max_value=100.0,
        step=0.5,
        help="자본의 몇 %를 거래당 위험에 노출할지. 일반적으로 1~2%, "
        "공격적으로 운용하면 5~10%, 실험적으론 그 이상까지 가능.",
    )
    max_holding_days = st.number_input(
        "Max Holding Days",
        value=60,
        min_value=1,
        max_value=2_000,
        step=5,
        help="Time stop 발동까지 최대 보유 일수.",
    )

    st.header("Entry Filter")
    require_gap_up = st.checkbox(
        "Require gap-up confirmation",
        value=False,
        help="다음날 시초가가 전날 종가보다 높을 때만 진입 (TraderLion의 "
        "'Wedge Pops with unfilled gaps show strong momentum' 룰).",
    )

    st.header("Exit Tuning")
    extension_pct = st.number_input(
        "Exhaustion % above 10 EMA",
        value=15.0,
        min_value=1.0,
        max_value=500.0,
        step=1.0,
        help="종가가 10 EMA보다 이 % 이상 위로 벌어지면 익절. "
        "보통 10~20%. 변동성 큰 종목/장기 보유엔 더 크게.",
    )
    extension_atr_mult = st.number_input(
        "Exhaustion ATR multiplier",
        value=2.5,
        min_value=0.1,
        max_value=50.0,
        step=0.1,
        help="대안 익절 트리거: close − 10 EMA ≥ ATR(14) × 이 값. "
        "보통 2~4. 큰 값일수록 더 많이 풀어주고 익절 늦춤.",
    )

    run_btn = st.button("Run Strategy", type="primary", use_container_width=True)


# --- Main ---
if not run_btn:
    st.info("좌측에서 ticker / 기간 / 위험 파라미터를 설정하고 **Run Strategy**를 눌러줘.")
    st.stop()

market_data = YFinanceAdapter()

with st.spinner(f"Fetching {ticker} {start_date} → {end_date}..."):
    try:
        df = market_data.fetch_ohlcv(ticker, start_date, end_date)
    except Exception as e:
        st.error(f"Failed to fetch data: {e}")
        st.stop()

if df is None or df.empty:
    st.warning("No data returned for that range.")
    st.stop()

config = StrategyConfig(
    ticker=ticker,
    start_date=start_date,
    end_date=end_date,
    pattern_name="wedge_pop",
    initial_capital=float(initial_capital),
    risk_per_trade=risk_pct / 100.0,
    max_holding_days=max_holding_days,
)

strategy = WedgepopStrategy(
    market_data=market_data,
    detector=WedgePopDetector(),
    extension_pct=extension_pct / 100.0,
    extension_atr_mult=extension_atr_mult,
    require_gap_up=require_gap_up,
)

with st.spinner("Running strategy..."):
    try:
        result = strategy.execute(df, config)
    except Exception as e:
        st.error(f"Strategy failed: {e}")
        st.stop()

perf = result.performance

# --- Headline metrics ---
st.subheader(f"{ticker} — Wedge Pop Strategy")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Trades", perf.total_trades)
m2.metric(
    "Win Rate",
    f"{perf.win_rate:.0%}" if perf.total_trades else "—",
)
m3.metric("Total Return", f"{perf.total_return_pct:.2%}")
m4.metric("Final Capital", f"${perf.final_capital:,.0f}")

m5, m6, m7, m8 = st.columns(4)
m5.metric("Initial Capital", f"${perf.initial_capital:,.0f}")
m6.metric("Avg Win", f"{perf.avg_win_pct:.2%}" if perf.trades else "—")
m7.metric("Avg Loss", f"{perf.avg_loss_pct:.2%}" if perf.trades else "—")
m8.metric("Max Drawdown", f"{perf.max_drawdown_pct:.2%}")

# --- Candlestick with trade markers ---
chart_builder = PlotlyChartBuilder()
trade_fig = chart_builder.build_candlestick_with_trades(
    df,
    perf.trades,
    title=f"{ticker} — Buy / Sell / Stop",
)
st.plotly_chart(trade_fig, use_container_width=True)

if not perf.trades:
    st.info("이 기간엔 Wedge Pop 신호가 발생하지 않았어.")
    st.stop()

# --- Equity curve ---
if len(result.equity_curve) > 1:
    eq_fig = chart_builder.build_equity_curve(
        result.equity_curve, title="Equity Curve"
    )
    st.plotly_chart(eq_fig, use_container_width=True)

# --- Trade table with dollar values ---
st.subheader("Trades")
trade_rows = []
for t in perf.trades:
    is_stop = abs(t.exit_price - t.stop_loss) < 1e-6
    outcome = "STOP" if is_stop else ("WIN" if t.pnl > 0 else "LOSS")
    trade_rows.append(
        {
            "Entry Date": t.entry_date,
            "Exit Date": t.exit_date,
            "Outcome": outcome,
            "Entry Price": f"${t.entry_price:,.2f}",
            "Exit Price": f"${t.exit_price:,.2f}",
            "Stop Loss": f"${t.stop_loss:,.2f}",
            "Shares": t.shares,
            "Cost": f"${t.entry_price * t.shares:,.0f}",
            "Proceeds": f"${t.exit_price * t.shares:,.0f}",
            "P&L ($)": f"${t.pnl:,.2f}",
            "P&L (%)": f"{t.pnl_pct:.2%}",
        }
    )
st.dataframe(trade_rows, use_container_width=True)
