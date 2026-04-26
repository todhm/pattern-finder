"""Single-ticker Wick Play backtest on 15-minute bars.

Intraday counterpart to :file:`5_Wick_Play_Strategy.py`. Uses
:class:`Wickplay15mStrategy` (which inherits every stop and exit
rule from the daily strategy) plus the
:func:`build_wickplay_15m_detector` factory which rescales the
detector's daily-named lookbacks to session-count units.

yfinance caps 15m history at 60 calendar days — the period picker
reflects that.
"""

from datetime import date, timedelta

import streamlit as st

from data.adapters.cached_market_data import CachedMarketDataAdapter
from data.adapters.regular_session_filter import RegularSessionFilterAdapter
from data.adapters.yfinance_adapter import YFinanceAdapter
from pages._shared.wedgepop_results import (
    apply_fees_to_trades,
    render_single_ticker_headline_metrics,
    render_single_ticker_trade_table,
    render_toss_fee_inputs,
)
from pattern.adapters.exhaustion_extension_top import ExhaustionExtensionTopDetector
from strategy.adapters.wickplay_15m_strategy import (
    Wickplay15mStrategy,
    build_wickplay_15m_detector,
)
from strategy.domain.models import StrategyConfig
from visualization.adapters.plotly_charts import PlotlyChartBuilder

st.set_page_config(page_title="Wick Play — 15m", layout="wide")
st.title("Wick Play Strategy — 15-Minute Bars")
st.caption(
    "단일 ticker에 대한 15분봉 Wick Play 백테스트 (Oliver Kell 카피튜레이션 "
    "리버설). ⚠️ yfinance 15분봉 캡 = 최근 60일."
)

with st.sidebar:
    st.header("Market")
    ticker = st.text_input("Ticker", value="AAPL")
    start_date = st.date_input(
        "Start Date",
        value=date.today() - timedelta(days=7),
        min_value=date.today() - timedelta(days=60),
        max_value=date.today(),
    )
    end_date = st.date_input(
        "End Date",
        value=date.today(),
        min_value=date(2000, 1, 1),
        max_value=date.today(),
    )

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
    )
    max_holding_bars = st.number_input(
        "Max Holding Bars",
        value=520,
        min_value=1,
        max_value=5_000,
        step=26,
        help="26 = 1 session, 130 = 1 week, 520 = 20 sessions (≈1 month).",
    )

    st.header("Detector (psychology)")
    min_psych_score = st.slider(
        "Min psychology score (0–4)",
        min_value=0,
        max_value=4,
        value=3,
        help="4개 심리 필터 중 최소 몇 개를 통과해야 하는지. 0 = filter off.",
    )
    min_breakout_strength_atr = st.number_input(
        "Min breakout strength (× ATR)",
        value=0.3,
        min_value=0.0,
        max_value=5.0,
        step=0.05,
    )

    st.header("Strategy")
    ema_trail = st.number_input(
        "EMA trail period (bars)", value=26, min_value=5, max_value=200, step=1
    )
    atr_period = st.number_input(
        "ATR period (bars)", value=26, min_value=5, max_value=200, step=1
    )
    min_trail_bars = st.number_input(
        "Min bars before trail can fire", value=8, min_value=0, max_value=100, step=1
    )
    enable_gap_down_rejection = st.checkbox(
        "Reject gap-down opens",
        value=True,
    )
    enable_exh_exit = st.checkbox(
        "Exhaustion Extension Top exit", value=True
    )
    enable_breakeven = st.checkbox("Break-even stop (arm at +1.5R)", value=False)

    fee_schedule = render_toss_fee_inputs(key_prefix="wp15m_")

    run_btn = st.button("Run 15m Backtest", type="primary", use_container_width=True)

if not run_btn:
    st.info("좌측에서 설정 후 **Run 15m Backtest**를 눌러.")
    st.stop()

market_data = RegularSessionFilterAdapter(
    CachedMarketDataAdapter(YFinanceAdapter())
)

detector = build_wickplay_15m_detector(
    min_psych_score=int(min_psych_score),
    min_breakout_strength_atr=float(min_breakout_strength_atr),
    atr_period=int(atr_period),
)
exit_detector = (
    ExhaustionExtensionTopDetector(atr_period=int(atr_period))
    if enable_exh_exit
    else None
)
strategy = Wickplay15mStrategy(
    market_data=market_data,
    detector=detector,
    exit_detector=exit_detector,
    ema_trail=int(ema_trail),
    atr_period=int(atr_period),
    min_trail_bars=int(min_trail_bars),
    enable_gap_down_rejection=enable_gap_down_rejection,
    enable_breakeven_stop=enable_breakeven,
)
config = StrategyConfig(
    ticker=ticker.upper(),
    start_date=start_date,
    end_date=end_date,
    pattern_name="wick_play",
    initial_capital=float(initial_capital),
    risk_per_trade=risk_pct / 100.0,
    max_holding_days=int(max_holding_bars),
)

with st.spinner(f"Running 15m Wick Play on {ticker.upper()}..."):
    try:
        result = strategy.run(config)
    except Exception as exc:
        st.error(f"Backtest failed: {exc}")
        st.stop()

perf = result.performance
chart_builder = PlotlyChartBuilder()

# Trade-derived widgets only render when something fired. Falling
# through to the chart on zero trades lets the user see the raw 15m
# price action and tune parameters visually.
if perf.total_trades == 0:
    st.subheader(f"{ticker.upper()} — 15m")
    st.metric("Trades", 0)
    st.info(
        "이 기간엔 15m Wick Play signal이 잡히지 않았어. 파라미터/기간을 "
        "조정해봐. 차트는 그대로 아래에 표시돼."
    )
else:
    fee_rows = apply_fees_to_trades(perf.trades, fee_schedule)
    render_single_ticker_headline_metrics(
        perf, fee_rows, subtitle=f"{ticker.upper()} — 15m"
    )

    st.subheader("Equity Curve")
    st.plotly_chart(
        chart_builder.build_equity_curve(
            result.equity_curve,
            title=f"{ticker.upper()} 15m Wick Play equity (gross)",
        ),
        use_container_width=True,
    )
    st.caption(
        "Equity curve는 전략의 gross PnL 기반이야. Toss 수수료는 metric과 "
        "trade table에서 차감돼서 표시돼."
    )

    render_single_ticker_trade_table(fee_rows)

# Full-period chart with every trade overlaid — always rendered,
# even when zero trades fired, so price action is visible for tuning.
# Fetch range is exactly the user's window (no warmup pad): yfinance
# caps 15m history at 60 calendar days, so any extra pad would push
# wider windows past the cap and silently kill the chart. Indicator
# overlays may not converge in the first few bars of the window —
# acceptable cost for keeping the chart available across the full
# 60-day cap range.
st.subheader("Chart")
try:
    full_df = market_data.fetch_ohlcv(
        ticker.upper(), start_date, end_date, interval="15m"
    )
except Exception as exc:
    st.warning(f"Failed to fetch chart data: {exc}")
else:
    full_fig = chart_builder.build_candlestick_with_trades(
        full_df,
        perf.trades,
        title=f"{ticker.upper()} — {start_date} → {end_date} (15m Wick Play)",
    )
    full_fig.update_xaxes(range=[str(start_date), str(end_date)])
    st.plotly_chart(full_fig, use_container_width=True)
