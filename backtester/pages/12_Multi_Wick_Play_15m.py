"""Multi-ticker Wick Play backtest on 15-minute bars.

Intraday counterpart to :file:`6_Multi_Wick_Play.py`. Uses
:class:`MultiWickplay15mStrategy` (which injects
:class:`Wickplay15mStrategy` by default) and shares result rendering
with ``8_Multi_Wedgepop_15m.py`` via the
``pages._shared.wedgepop_results`` helpers — the rendering is
strategy-agnostic, driven by :class:`MultiStrategyResult`.
"""

from datetime import date, timedelta

import streamlit as st

from data.adapters.cached_market_data import CachedMarketDataAdapter
from data.adapters.regular_session_filter import RegularSessionFilterAdapter
from data.adapters.wikipedia_universe import WikipediaUniverseAdapter
from data.adapters.yfinance_adapter import YFinanceAdapter
from pages._shared.wedgepop_results import (
    render_equity_curve,
    render_failed_tickers,
    render_headline_metrics,
    render_per_trade_charts,
    render_ticker_contribution,
    render_top_trades,
    render_trade_table,
)
from pattern.adapters.exhaustion_extension_top import ExhaustionExtensionTopDetector
from strategy.adapters.wickplay_15m_strategy import (
    MultiWickplay15mStrategy,
    Wickplay15mStrategy,
    build_wickplay_15m_detector,
)
from strategy.domain.models import MultiStrategyConfig, TossFeeSchedule
from visualization.adapters.plotly_charts import PlotlyChartBuilder

st.set_page_config(page_title="Multi Wick Play — 15m", layout="wide")
st.title("Multi-Ticker Wick Play — 15-Minute Bars")
st.caption(
    "S&P 500 / Nasdaq-100 universe에서 15분봉 Wick Play 신호를 스캔. "
    "한 포지션이 청산될 때까지 다음 종목은 잡지 않음. ⚠️ yfinance 15분봉 "
    "최근 60일 캡."
)

with st.sidebar:
    st.header("Universe")
    universe = st.selectbox(
        "Universe",
        options=["sp500", "nasdaq100"],
        index=0,
        format_func=lambda x: "S&P 500" if x == "sp500" else "Nasdaq-100",
    )
    max_tickers = st.number_input(
        "Max tickers (0 = all)",
        value=30,
        min_value=0,
        max_value=600,
        step=10,
    )
    max_workers = st.number_input(
        "Parallel workers",
        value=4,
        min_value=1,
        max_value=32,
        step=1,
    )

    st.header("Period")
    st.caption("yfinance 15m 캡 = 최근 60 calendar days.")
    start_date = st.date_input(
        "Start Date",
        value=date.today() - timedelta(days=30),
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
    )

    st.header("Detector (psychology)")
    min_psych_score = st.slider(
        "Min psychology score (0–4)",
        min_value=0,
        max_value=4,
        value=3,
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
        "Min bars before trail can fire",
        value=8,
        min_value=0,
        max_value=100,
        step=1,
    )
    enable_gap_down_rejection = st.checkbox("Reject gap-down opens", value=True)
    enable_exh_exit = st.checkbox("Exhaustion Extension Top exit", value=True)
    enable_breakeven = st.checkbox("Break-even stop (arm at +1.5R)", value=False)

    st.header("Fees (Toss)")
    buy_fee_pct = st.number_input(
        "Buy (%)",
        value=0.10,
        min_value=0.0,
        max_value=5.0,
        step=0.01,
        format="%.4f",
    )
    sell_fee_pct = st.number_input(
        "Sell (%)",
        value=0.10,
        min_value=0.0,
        max_value=5.0,
        step=0.01,
        format="%.4f",
    )
    sec_fee_pct = st.number_input(
        "SEC (%)",
        value=0.00229,
        min_value=0.0,
        max_value=1.0,
        step=0.0001,
        format="%.5f",
    )

    run_btn = st.button(
        "Run 15m Universe Scan", type="primary", use_container_width=True
    )

if not run_btn:
    st.info("좌측에서 설정 후 **Run 15m Universe Scan**을 눌러.")
    st.stop()

market_data = RegularSessionFilterAdapter(
    CachedMarketDataAdapter(YFinanceAdapter())
)
universe_provider = WikipediaUniverseAdapter()

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
per_ticker_strategy = Wickplay15mStrategy(
    market_data=market_data,
    detector=detector,
    exit_detector=exit_detector,
    ema_trail=int(ema_trail),
    atr_period=int(atr_period),
    min_trail_bars=int(min_trail_bars),
    enable_gap_down_rejection=enable_gap_down_rejection,
    enable_breakeven_stop=enable_breakeven,
)
runner = MultiWickplay15mStrategy(
    market_data=market_data,
    universe_provider=universe_provider,
    detector=detector,
    strategy=per_ticker_strategy,
    max_workers=int(max_workers),
)

fee_schedule = TossFeeSchedule(
    buy_commission_pct=buy_fee_pct / 100.0,
    sell_commission_pct=sell_fee_pct / 100.0,
    sec_fee_pct=sec_fee_pct / 100.0,
)
config = MultiStrategyConfig(
    universe=universe,
    start_date=start_date,
    end_date=end_date,
    pattern_name="wick_play",
    initial_capital=float(initial_capital),
    risk_per_trade=risk_pct / 100.0,
    max_holding_days=int(max_holding_bars),
    max_tickers=int(max_tickers) if max_tickers > 0 else None,
    fee_schedule=fee_schedule,
)

with st.spinner(
    f"Scanning {universe} on 15m bars ({config.max_tickers or 'all'} tickers)..."
):
    try:
        result = runner.run(config)
    except Exception as exc:
        st.error(f"Universe scan failed: {exc}")
        st.stop()

render_headline_metrics(result, universe_label=f"{universe.upper()} · 15m · Wick Play")

if result.tickers_scanned == 0:
    st.warning("Universe returned 0 tickers — check selection.")
    st.stop()
if not result.trades:
    st.info(
        "이 기간엔 15m Wick Play signal이 잡히지 않았거나 모두 필터에 걸렸어. "
        "기간/필터를 조정해봐."
    )
    render_failed_tickers(result)
    st.stop()

chart_builder = PlotlyChartBuilder()
render_equity_curve(
    chart_builder,
    result,
    title=f"Single-position Wick Play across {universe.upper()} (15m)",
)
render_top_trades(chart_builder, result)
render_ticker_contribution(chart_builder, result)
render_trade_table(result)
render_per_trade_charts(
    chart_builder,
    market_data,
    result,
    interval="15m",
    context_before_days=3,
    context_after_days=2,
    warmup_days=10,
)
render_failed_tickers(result)
