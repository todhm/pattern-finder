from datetime import date, timedelta

import streamlit as st

from data.adapters.yfinance_adapter import YFinanceAdapter
from pattern.adapters.reversal_extension import ReversalExtensionDetector
from pattern.adapters.wedge_pop import WedgePopDetector
from visualization.adapters.plotly_charts import PlotlyChartBuilder

DETECTORS = {
    "reversal_extension": ReversalExtensionDetector(),
    "wedge_pop": WedgePopDetector(),
}

st.set_page_config(page_title="Pattern Detection", layout="wide")
st.title("Pattern Detection")

# --- Sidebar inputs ---
with st.sidebar:
    st.header("Settings")
    ticker = st.text_input("Ticker", value="AAPL")
    start_date = st.date_input("Start Date", value=date.today() - timedelta(days=365))
    end_date = st.date_input("End Date", value=date.today())
    pattern_name = st.selectbox("Pattern", options=list(DETECTORS.keys()))
    run_btn = st.button("Detect", type="primary", use_container_width=True)

# --- Main area ---
if run_btn:
    with st.spinner("Fetching data & detecting patterns..."):
        try:
            adapter = YFinanceAdapter()
            df = adapter.fetch_ohlcv(ticker, start_date, end_date)
            detector = DETECTORS[pattern_name]
            signals = detector.detect(df)
        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()

    st.subheader(f"{ticker} — {pattern_name} ({len(signals)} signals)")

    chart_builder = PlotlyChartBuilder()
    fig = chart_builder.build_candlestick_with_signals(
        df, signals, title=f"{ticker} — {pattern_name}"
    )
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
