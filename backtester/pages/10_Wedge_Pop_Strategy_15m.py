"""Single-ticker Wedge Pop backtest on 15-minute bars.

Intraday counterpart to :file:`9_Wedge_Pop_Strategy.py`. Designed as
a focused smoke-test / parameter-tuning surface for the 15m
strategy: fewer knobs than the daily page, 15m-appropriate defaults
(session = 26 × 15m bars), and the yfinance 60-day cap surfaced in
the sidebar so a 180-day window doesn't silently fail.

The inherited :class:`Wedgepop15mStrategy` handles every entry
filter, position sizer, and exit rule unchanged from the daily
strategy — the only difference is the bar-identity hooks resolve to
the exact ``DatetimeIndex`` value and the class forwards
``interval="15m"`` to :class:`MarketDataPort.fetch_ohlcv`.
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
from pattern.adapters.wedge_pop import WedgePopDetector
from strategy.adapters.wedgepop_15m_strategy import Wedgepop15mStrategy
from strategy.domain.models import StrategyConfig
from visualization.adapters.plotly_charts import PlotlyChartBuilder

st.set_page_config(page_title="Wedge Pop — 15m", layout="wide")
st.title("Wedge Pop Strategy — 15-Minute Bars")
st.caption(
    "단일 ticker에 대한 15분봉 Wedge Pop 백테스트. ⚠️ yfinance 15분봉 캡 = "
    "최근 60일. 더 긴 기간은 별도 MarketDataPort 어댑터 필요."
)

with st.sidebar:
    st.header("Market")
    ticker = st.text_input("Ticker", value="AAPL")
    st.caption("yfinance 15m 캡 = 최근 60 calendar days.")
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
        value=5.0,
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

    st.header("Pattern Detection — 15m")
    detect_lookback = st.number_input(
        "Consolidation lookback (bars)",
        value=26,
        min_value=5,
        max_value=520,
        step=1,
    )
    ema_fast = st.number_input(
        "Fast EMA (bars)",
        value=26,
        min_value=5,
        max_value=200,
        step=1,
    )
    ema_slow = st.number_input(
        "Slow EMA (bars)",
        value=78,
        min_value=ema_fast + 1,
        max_value=520,
        step=1,
    )
    atr_period = st.number_input(
        "ATR period (bars)",
        value=26,
        min_value=5,
        max_value=200,
        step=1,
    )
    slope_lookback = st.number_input(
        "EMA slope lookback (bars)",
        value=78,
        min_value=5,
        max_value=520,
        step=1,
    )
    breakout_atr_mult = st.number_input(
        "Min breakout strength (× ATR)",
        value=0.005,
        min_value=0.0,
        max_value=10.0,
        step=0.001,
        format="%.3f",
    )
    consolidation_pct = st.number_input(
        "Min consolidation %",
        value=30.0,
        min_value=0.0,
        max_value=100.0,
        step=5.0,
    )
    cooldown_bars = st.number_input(
        "Cooldown between signals (bars)",
        value=26,
        min_value=0,
        max_value=520,
        step=1,
    )

    st.header("Strategy")
    use_smart_trail = st.checkbox("Smart trail (Chandelier)", value=True)
    enable_breakeven = st.checkbox("Break-even stop after +1R", value=False)
    enable_exh_exit = st.checkbox("Exhaustion Extension Top exit", value=False)

    fee_schedule = render_toss_fee_inputs(key_prefix="wpg15m_")

    run_btn = st.button("Run 15m Backtest", type="primary", use_container_width=True)

if not run_btn:
    st.info("좌측에서 ticker/기간을 설정하고 **Run 15m Backtest**를 눌러.")
    st.stop()

# yfinance fetches pre+post → cache stores raw → filter wraps the
# read so strategy/chart only see RTH bars. Decoupling lets a future
# ETH strategy share the same parquet cache.
market_data = RegularSessionFilterAdapter(
    CachedMarketDataAdapter(YFinanceAdapter())
)

detector = WedgePopDetector(
    lookback=int(detect_lookback),
    ema_fast=int(ema_fast),
    ema_slow=int(ema_slow),
    consolidation_pct=consolidation_pct / 100.0,
    breakout_atr_mult=breakout_atr_mult,
    slope_lookback=int(slope_lookback),
    cooldown_bars=int(cooldown_bars),
    atr_period=int(atr_period),
    require_above_long_smas=False,
)
exit_detector = (
    ExhaustionExtensionTopDetector(
        ema_fast=int(ema_fast),
        ema_slow=int(ema_slow),
        slope_lookback=int(slope_lookback),
        atr_period=int(atr_period),
    )
    if enable_exh_exit
    else None
)
strategy = Wedgepop15mStrategy(
    market_data=market_data,
    detector=detector,
    ema_trail=int(ema_fast),
    ema_slow=int(ema_slow),
    atr_period=int(atr_period),
    use_smart_trail=use_smart_trail,
    enable_breakeven_stop=enable_breakeven,
    exit_detector=exit_detector,
)
config = StrategyConfig(
    ticker=ticker.upper(),
    start_date=start_date,
    end_date=end_date,
    pattern_name="wedge_pop",
    initial_capital=float(initial_capital),
    risk_per_trade=risk_pct / 100.0,
    # Bar count for 15m — treated as bar count by the engine.
    max_holding_days=int(max_holding_bars),
)

with st.spinner(f"Running 15m backtest on {ticker.upper()}..."):
    try:
        result = strategy.run(config)
    except Exception as exc:
        st.error(f"Backtest failed: {exc}")
        st.stop()

perf = result.performance
chart_builder = PlotlyChartBuilder()

# Trade-derived widgets (fee-augmented metrics, equity curve, trade
# table) only make sense when at least one trade fired. When the
# detector found nothing, fall through to the chart so the user can
# still inspect price action and tune parameters from the visual.
if perf.total_trades == 0:
    st.subheader(f"{ticker.upper()} — 15m")
    st.metric("Trades", 0)
    st.info(
        "이 기간엔 15m Wedge Pop signal이 잡히지 않았어. 파라미터/기간을 "
        "조정해봐. 차트는 그대로 아래에 표시돼."
    )
else:
    # Augment gross trades with Toss commission so the table / metrics
    # show net-of-fees numbers. Original Trade objects stay untouched
    # so the candlestick chart below can draw them unchanged.
    fee_rows = apply_fees_to_trades(perf.trades, fee_schedule)
    render_single_ticker_headline_metrics(
        perf, fee_rows, subtitle=f"{ticker.upper()} — 15m"
    )

    st.subheader("Equity Curve")
    eq_fig = chart_builder.build_equity_curve(
        result.equity_curve,
        title=f"{ticker.upper()} 15m Wedge Pop equity (gross)",
    )
    st.plotly_chart(eq_fig, use_container_width=True)
    st.caption(
        "Equity curve는 전략의 gross PnL 기반이야. Toss 수수료는 위 metric과 "
        "trade table에서 차감돼서 표시돼."
    )

    render_single_ticker_trade_table(fee_rows)

# Full-period 15m candlestick — always rendered, with whatever trades
# fired (possibly zero) overlaid. Lets the user see the raw price
# action of the window even when the detector didn't pick anything,
# so parameter tuning has a visual reference.
st.subheader("Chart")
fetch_start = start_date - timedelta(days=10)
try:
    full_df = market_data.fetch_ohlcv(
        ticker.upper(), fetch_start, end_date, interval="15m"
    )
except Exception as exc:
    st.warning(f"Failed to fetch chart data: {exc}")
else:
    full_fig = chart_builder.build_candlestick_with_trades(
        full_df,
        perf.trades,
        title=f"{ticker.upper()} — {start_date} → {end_date} (15m)",
    )
    full_fig.update_xaxes(range=[str(start_date), str(end_date)])
    st.plotly_chart(full_fig, use_container_width=True)
