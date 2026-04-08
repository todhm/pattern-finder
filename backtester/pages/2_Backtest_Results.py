from datetime import date, timedelta

import streamlit as st

from backtest.adapters.engine import BacktestEngine
from data.adapters.yfinance_adapter import YFinanceAdapter
from pattern.adapters.reversal_extension import ReversalExtensionDetector
from pattern.adapters.wedge_pop import WedgePopDetector
from strategy.adapters.runner import StrategyRunner
from strategy.domain.models import StrategyConfig
from visualization.adapters.plotly_charts import PlotlyChartBuilder

DETECTORS = {
    "reversal_extension": ReversalExtensionDetector(),
    "wedge_pop": WedgePopDetector(),
}

st.set_page_config(page_title="Backtest Results", layout="wide")
st.title("Backtest Results")

# --- Sidebar inputs ---
with st.sidebar:
    st.header("Strategy Config")
    ticker = st.text_input("Ticker", value="AAPL")
    start_date = st.date_input("Start Date", value=date.today() - timedelta(days=365))
    end_date = st.date_input("End Date", value=date.today())
    pattern_name = st.selectbox("Pattern", options=list(DETECTORS.keys()))

    st.header("Backtest Params")
    initial_capital = st.number_input(
        "Initial Capital ($)", value=100_000, min_value=1_000, step=10_000
    )
    risk_per_trade = st.slider(
        "Risk per Trade (%)", min_value=0.5, max_value=10.0, value=2.0, step=0.5
    )
    max_holding_days = st.slider(
        "Max Holding Days", min_value=5, max_value=120, value=60, step=5
    )

    run_btn = st.button("Run Backtest", type="primary", use_container_width=True)

# --- Main area ---
if run_btn:
    config = StrategyConfig(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        pattern_name=pattern_name,
        initial_capital=float(initial_capital),
        risk_per_trade=risk_per_trade / 100.0,
        max_holding_days=max_holding_days,
    )

    runner = StrategyRunner(
        market_data=YFinanceAdapter(),
        detectors=DETECTORS,
        engine=BacktestEngine(
            initial_capital=config.initial_capital,
            risk_per_trade=config.risk_per_trade,
            max_holding_days=config.max_holding_days,
        ),
    )

    with st.spinner("Running backtest..."):
        try:
            result = runner.run(config)
        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()

    br = result.backtest_result

    # --- Metrics ---
    st.subheader(f"{ticker} — {pattern_name} Backtest")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Return", f"{br.total_return_pct:.2%}")
    c2.metric("Win Rate", f"{br.win_rate:.2%}")
    c3.metric("Max Drawdown", f"{br.max_drawdown_pct:.2%}")
    c4.metric("Total Trades", br.total_trades)

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Final Capital", f"${br.final_capital:,.0f}")
    c6.metric("Avg Win", f"{br.avg_win_pct:.2%}")
    c7.metric("Avg Loss", f"{br.avg_loss_pct:.2%}")
    c8.metric("Initial Capital", f"${br.initial_capital:,.0f}")

    # --- Equity Curve ---
    if len(result.equity_curve) > 1:
        chart_builder = PlotlyChartBuilder()
        eq_fig = chart_builder.build_equity_curve(
            result.equity_curve, title="Equity Curve"
        )
        st.plotly_chart(eq_fig, use_container_width=True)

    # --- Trade List ---
    if br.trades:
        st.subheader("Trades")
        trade_data = [
            {
                "Pattern": t.pattern_name,
                "Entry Date": t.entry_date,
                "Exit Date": t.exit_date,
                "Entry Price": f"{t.entry_price:.2f}",
                "Exit Price": f"{t.exit_price:.2f}",
                "Stop Loss": f"{t.stop_loss:.2f}",
                "Shares": t.shares,
                "PnL": f"${t.pnl:,.2f}",
                "PnL %": f"{t.pnl_pct:.2%}",
            }
            for t in br.trades
        ]
        st.dataframe(trade_data, use_container_width=True)
    else:
        st.info("No trades were executed.")
