from datetime import date, timedelta

import streamlit as st

from data.adapters.yfinance_adapter import YFinanceAdapter
from pattern.adapters.reversal_extension import ReversalExtensionDetector
from pattern.adapters.wedge_pop import WedgePopDetector
from visualization.adapters.plotly_charts import PlotlyChartBuilder

st.set_page_config(page_title="Pattern Detection", layout="wide")
st.title("Pattern Detection")

# --- Sidebar inputs ---
with st.sidebar:
    st.header("Settings")
    ticker = st.text_input("Ticker", value="AAPL")
    start_date = st.date_input("Start Date", value=date.today() - timedelta(days=365))
    end_date = st.date_input("End Date", value=date.today())
    pattern_name = st.selectbox("Pattern", options=["wedge_pop", "reversal_extension"])

    st.header("Wedge Pop Detector")
    st.caption("Used only when `wedge_pop` is selected above.")
    detect_lookback = st.number_input(
        "Consolidation lookback (days)",
        value=15,
        min_value=3,
        max_value=60,
        step=1,
    )
    cooldown_bars_ui = st.number_input(
        "Cooldown bars (after a signal)",
        value=15,
        min_value=0,
        max_value=60,
        step=1,
        help="signal 이후 건너뛰는 바 수. 0 = 연속 허용.",
    )
    ema_fast = st.number_input(
        "Fast EMA period", value=10, min_value=2, max_value=100, step=1
    )
    ema_slow = st.number_input(
        "Slow EMA period", value=20, min_value=2, max_value=200, step=1
    )
    consolidation_pct = st.number_input(
        "Min consolidation %",
        value=60.0,
        min_value=0.0,
        max_value=100.0,
        step=5.0,
        help="직전 lookback 일 중 close < fast EMA 요구 **최소** 비율.",
    )
    enable_max_cp = st.checkbox(
        "Cap max consolidation %",
        value=False,
    )
    max_consolidation_pct_ui = st.number_input(
        "Max consolidation %",
        value=95.0,
        min_value=0.0,
        max_value=100.0,
        step=5.0,
        disabled=not enable_max_cp,
    )
    breakout_pct = st.number_input(
        "Min breakout strength %",
        value=1.5,
        min_value=0.0,
        max_value=20.0,
        step=0.1,
        format="%.2f",
        help="max(ema_strength, daily_move) 의 **하한**.",
    )
    enable_max_bp = st.checkbox(
        "Cap max breakout strength",
        value=False,
    )
    max_breakout_pct_ui = st.number_input(
        "Max breakout strength %",
        value=10.0,
        min_value=0.0,
        max_value=50.0,
        step=0.5,
        format="%.2f",
        disabled=not enable_max_bp,
    )
    slope_lookback = st.number_input(
        "Slope lookback (days)",
        value=20,
        min_value=5,
        max_value=120,
        step=1,
        help="ema_fast_slope / ema_slow_slope 측정 기간. signal metadata에 기록됨.",
    )
    require_above_long_smas = st.checkbox(
        "Require close above 50 & 200 SMA",
        value=False,
        help="signal 캔들의 close가 50 SMA 및 200 SMA 위일 때만 wedge "
        "pop으로 인정.",
    )

    run_btn = st.button("Detect", type="primary", use_container_width=True)

# --- Main area ---
if run_btn:
    # Fetch extra history so 50/200 SMA are converged from day 1.
    fetch_start = start_date - timedelta(days=400)
    with st.spinner("Fetching data & detecting patterns..."):
        try:
            adapter = YFinanceAdapter()
            df = adapter.fetch_ohlcv(ticker, fetch_start, end_date)
            if pattern_name == "wedge_pop":
                detector = WedgePopDetector(
                    lookback=int(detect_lookback),
                    ema_fast=int(ema_fast),
                    ema_slow=int(ema_slow),
                    consolidation_pct=consolidation_pct / 100.0,
                    max_consolidation_pct=(
                        max_consolidation_pct_ui / 100.0
                        if enable_max_cp
                        else None
                    ),
                    breakout_pct=breakout_pct / 100.0,
                    max_breakout_pct=(
                        max_breakout_pct_ui / 100.0
                        if enable_max_bp
                        else None
                    ),
                    slope_lookback=int(slope_lookback),
                    cooldown_bars=int(cooldown_bars_ui),
                    require_above_long_smas=require_above_long_smas,
                )
            else:
                detector = ReversalExtensionDetector()
            signals = detector.detect(df)
        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()

    st.subheader(f"{ticker} — {pattern_name} ({len(signals)} signals)")

    chart_builder = PlotlyChartBuilder()
    fig = chart_builder.build_candlestick_with_signals(
        df, signals, title=f"{ticker} — {pattern_name}"
    )
    fig.update_xaxes(range=[str(start_date), str(end_date)])
    st.plotly_chart(fig, use_container_width=True)

    if signals:
        st.subheader("Detected Signals")
        signal_data = [
            {
                "Date": s.date,
                "Entry Price": f"{s.entry_price:.2f}",
                "Stop Loss": f"{s.stop_loss:.2f}",
                "Confidence": f"{s.confidence:.2f}",
                **{k: f"{v:.4f}" if isinstance(v, float) else v for k, v in s.metadata.items()},
            }
            for s in signals
        ]
        st.dataframe(signal_data, use_container_width=True)
    else:
        st.info("No patterns detected in this date range.")
