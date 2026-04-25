"""Multi-ticker Wedge Pop backtest on 15-minute bars.

Parallels :file:`3_Multi_Wedgepop.py` but runs every detector,
strategy, and exit rule on 15m bars instead of daily. The pipeline
is:

1. ``MarketDataPort.fetch_ohlcv(..., interval="15m")`` pulls NY-tz
   intraday bars (yfinance caps this at 60 calendar days — the
   sidebar defaults the window to 30 days to stay inside the cap).
2. ``Wedgepop15mStrategy`` inherits every entry filter / position
   sizer / exit rule from the daily :class:`WedgepopStrategy` and
   overrides only the bar-identity hooks so signals and exits key
   off the exact 15m ``DatetimeIndex`` value.
3. ``MultiWedgepop15mStrategy`` then walks signals chronologically
   with the same single-position portfolio lock, just keyed by
   timestamp instead of session date.

Result rendering is delegated to ``pages._shared.wedgepop_results``
so the daily page and this one show the same metrics / trade table
/ per-trade chart layout. The only 15m-specific wiring is:

- Interval forwarded to the per-trade chart fetch
  (``interval="15m"``).
- Warmup window dropped from 400 days to 20 (yfinance cap).
- Tighter default period (30 days).

Adding a 5m or 1h page later is a 10-line copy of this file with
different defaults — the strategy plumbing already supports any
``interval`` the market-data adapter will serve.
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
from pattern.adapters.wedge_pop import WedgePopDetector
from strategy.adapters.wedgepop_15m_strategy import (
    MultiWedgepop15mStrategy,
    Wedgepop15mStrategy,
)
from strategy.domain.models import MultiStrategyConfig, TossFeeSchedule
from visualization.adapters.plotly_charts import PlotlyChartBuilder

st.set_page_config(page_title="Multi Wedge Pop — 15m", layout="wide")
st.title("Multi-Ticker Wedge Pop — 15-Minute Bars")
st.caption(
    "S&P 500 / Nasdaq-100 universe를 15분봉으로 스캔. 한 포지션이 청산될 때까지 "
    "다음 종목은 잡지 않아. ⚠️ yfinance 15분봉은 최근 60일만 제공 — "
    "기간을 더 길게 잡으려면 다른 MarketDataPort 어댑터가 필요해."
)

# --- Sidebar inputs ---
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
        help="15m은 종목당 데이터 양이 일봉의 ~26배라 scan이 느려. "
        "작게 시작해서 동작 확인 후 늘려.",
    )
    max_workers = st.number_input(
        "Parallel workers",
        value=4,
        min_value=1,
        max_value=32,
        step=1,
        help="15m은 각 ticker가 fetch당 데이터 양이 커서 동시호출 많으면 "
        "yfinance rate-limit 더 빨리 맞음. 4 정도로 시작.",
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
        value=5.0,
        min_value=0.1,
        max_value=100.0,
        step=0.5,
    )
    max_holding_bars = st.number_input(
        "Max Holding Bars",
        value=520,  # 20 sessions ≈ 1 month of 15m bars
        min_value=1,
        max_value=5_000,
        step=26,
        help="15m bar 기준 최대 보유 bar 수. 26 = 1 session (intraday only), "
        "130 = 1주, 520 = 20 sessions (≈ 1개월 swing).",
    )

    st.header("Pattern Detection — 15m tuning")
    st.caption(
        "Session = 26 × 15m bars. Detector periods use session-count units."
    )
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
        help="동일 ticker에서 signal 간 최소 간격. 15m에서 26 = 1 session.",
    )

    st.header("Strategy toggles")
    use_smart_trail = st.checkbox(
        "Smart trail exit (Chandelier)",
        value=True,
        help="ATR-scaled trailing stop. 15m에서도 동일하게 작동.",
    )
    enable_breakeven_stop = st.checkbox(
        "Break-even stop after +1R",
        value=False,
    )
    enable_gap_down_rejection = st.checkbox(
        "Reject gap-down opens",
        value=False,
        help="15m 세션 내에서는 'gap'이 보통 오버나잇 갭 ≈ 세션 첫 bar만 해당.",
    )
    enable_exh_exit = st.checkbox(
        "Exhaustion Extension Top exit",
        value=False,
        help="ATR-based 피로 청산 패턴. 15m에서 쓰면 extension 임계치가 "
        "intraday 변동성에 맞게 더 커질 수 있음.",
    )

    st.header("Fees (Toss Securities)")
    buy_fee_pct = st.number_input(
        "Buy commission (%)",
        value=0.10,
        min_value=0.0,
        max_value=5.0,
        step=0.01,
        format="%.4f",
    )
    sell_fee_pct = st.number_input(
        "Sell commission (%)",
        value=0.10,
        min_value=0.0,
        max_value=5.0,
        step=0.01,
        format="%.4f",
    )
    sec_fee_pct = st.number_input(
        "SEC fee (%) — sell only",
        value=0.00229,
        min_value=0.0,
        max_value=1.0,
        step=0.0001,
        format="%.5f",
    )

    run_btn = st.button(
        "Run 15m Universe Scan", type="primary", use_container_width=True
    )

# --- Main ---
if not run_btn:
    st.info(
        "좌측에서 기간 / universe를 설정하고 **Run 15m Universe Scan** 을 눌러. "
        "처음엔 max_tickers를 작게(20~30) 잡고 smoke test하는 걸 권장."
    )
    st.stop()

# Composition: yfinance pulls pre+post (prepost=True) into the
# parquet cache, then RegularSessionFilterAdapter drops non-RTH bars
# on read. Cache stays raw for future ETH-aware strategies.
market_data = RegularSessionFilterAdapter(
    CachedMarketDataAdapter(YFinanceAdapter())
)
universe_provider = WikipediaUniverseAdapter()

detector = WedgePopDetector(
    lookback=int(detect_lookback),
    ema_fast=int(ema_fast),
    ema_slow=int(ema_slow),
    consolidation_pct=consolidation_pct / 100.0,
    breakout_atr_mult=breakout_atr_mult,
    slope_lookback=int(slope_lookback),
    cooldown_bars=int(cooldown_bars),
    atr_period=int(atr_period),
    # Intraday sessions don't need the 50/200 SMA long-term regime
    # gate — that's a multi-month filter that loses meaning on a
    # 60-day 15m window.
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

per_ticker_strategy = Wedgepop15mStrategy(
    market_data=market_data,
    detector=detector,
    ema_trail=int(ema_fast),
    ema_slow=int(ema_slow),
    atr_period=int(atr_period),
    use_smart_trail=use_smart_trail,
    enable_breakeven_stop=enable_breakeven_stop,
    enable_gap_down_rejection=enable_gap_down_rejection,
    exit_detector=exit_detector,
)
runner = MultiWedgepop15mStrategy(
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
    pattern_name="wedge_pop",
    initial_capital=float(initial_capital),
    risk_per_trade=risk_pct / 100.0,
    # ``max_holding_days`` on the config is the loop guard; at 15m
    # it's in units of bars, not days (the strategy treats it as bar
    # count all the way down).
    max_holding_days=int(max_holding_bars),
    max_tickers=int(max_tickers) if max_tickers > 0 else None,
    fee_schedule=fee_schedule,
)

with st.spinner(
    f"Scanning {universe} on 15m bars ({config.max_tickers or 'all'} tickers)..."
):
    try:
        result = runner.run(config)
    except Exception as e:
        st.error(f"Universe scan failed: {e}")
        st.stop()

# --- Results --------------------------------------------------------

render_headline_metrics(result, universe_label=f"{universe.upper()} · 15m")

if result.tickers_scanned == 0:
    st.warning("Universe returned 0 tickers — check your selection.")
    st.stop()
if not result.trades:
    st.info(
        "이 기간엔 universe 전체에서 15m Wedge Pop signal이 잡히지 않았거나, "
        "모두 진입 조건에서 걸렸어. 기간/필터를 조정해봐."
    )
    render_failed_tickers(result)
    st.stop()

chart_builder = PlotlyChartBuilder()
render_equity_curve(
    chart_builder,
    result,
    title=f"Single-position portfolio across {universe.upper()} (15m)",
)
render_top_trades(chart_builder, result)
render_ticker_contribution(chart_builder, result)
render_trade_table(result)

# Per-trade candlestick uses 15m bars for context rendering. Warmup
# buys a few extra sessions so EMAs drawn on the chart converge.
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
