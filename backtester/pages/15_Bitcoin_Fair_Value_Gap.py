"""Bitcoin / Crypto Fair Value Gap (FVG) backtest page.

Same FVG framework as ``13_Fair_Value_Gap.py`` — CHoCH → bullish 3-bar
gap → midpoint retest entry → 1:3 R take-profit — but pinned to the
``CRYPTO`` market calendar (24/7, UTC-anchored). All session-bound
behavior in the shared detector / strategy / chart code is short-
circuited via ``MarketCalendar.is_24_7``:

  - No RTH gate (every bar counts as "regular trading").
  - No force-close at session end (positions ride to TP / stop / max
    holding bars).
  - No weekend rangebreaks on the chart.
  - No "min bars left to session close" filter (irrelevant for 24/7).

Tickers follow EODHD's ``.CC`` convention: ``BTC-USD.CC``,
``ETH-USD.CC``, etc. The composed market-data routing layer already
sends sub-daily fetches to EODHD and matches up to the per-source
parquet cache.
"""

from datetime import date, timedelta

import streamlit as st

from data.adapters.composed_market_data import build_default_market_data
from data.domain.market_calendar import CRYPTO
from pages._shared.wedgepop_results import (
    apply_fees_to_trades,
    render_single_ticker_headline_metrics,
    render_single_ticker_trade_table,
    render_toss_fee_inputs,
)
from pattern.adapters.fair_value_gap import FairValueGapDetector
from strategy.adapters.fair_value_gap_strategy import FairValueGapStrategy
from strategy.domain.models import StrategyConfig
from visualization.adapters.plotly_charts import PlotlyChartBuilder

# Pre-vetted EODHD crypto symbols (probed 2026-04-26: each returns
# 192 bars over a 2-day 15m window). MATIC was rebranded to POL on
# Polygon's side and EODHD now returns 0 rows for MATIC-USD.CC, so
# it's intentionally excluded.
CRYPTO_TICKER_PRESETS = [
    "BTC-USD.CC",
    "ETH-USD.CC",
    "SOL-USD.CC",
    "XRP-USD.CC",
    "DOGE-USD.CC",
    "ADA-USD.CC",
    "BNB-USD.CC",
    "AVAX-USD.CC",
    "LINK-USD.CC",
]

# Same wall-clock targeting as the equity FVG page — the gap structure
# matters in bar-count, not market hours, so the equity defaults port
# directly. See pages/13_Fair_Value_Gap.py for the derivation.
INTERVAL_RETEST_DEFAULTS = {"1m": 40, "5m": 15, "15m": 5, "30m": 3}
INTERVAL_CHOCH_FVG_DEFAULTS = {"1m": 60, "5m": 20, "15m": 8, "30m": 4}

st.set_page_config(page_title="Bitcoin Fair Value Gap", layout="wide")
st.title("Bitcoin / Crypto FVG Strategy")
st.caption(
    "24/7 시장 — CHoCH → bullish 3-bar FVG → midpoint retest entry → 1:3 R take-profit. "
    "정규장 개념이 없으므로 force-close, RTH stop gate, weekend rangebreak 모두 자동 비활성화."
)

with st.sidebar:
    st.header("Market")
    if "crypto_ticker" not in st.session_state:
        st.session_state.crypto_ticker = "BTC-USD.CC"

    preset = st.selectbox(
        "Preset",
        options=CRYPTO_TICKER_PRESETS,
        index=CRYPTO_TICKER_PRESETS.index(st.session_state.crypto_ticker)
        if st.session_state.crypto_ticker in CRYPTO_TICKER_PRESETS
        else 0,
        help="EODHD가 검증된 Top crypto ticker. 그 외엔 아래 입력란에 직접.",
    )
    if preset != st.session_state.crypto_ticker:
        st.session_state.crypto_ticker = preset
    ticker = st.text_input(
        "Ticker (EODHD `.CC` suffix)",
        key="crypto_ticker",
        help="EODHD crypto symbol — `BTC-USD.CC`, `ETH-USD.CC` 등.",
    )

    interval = st.radio(
        "Bar interval",
        options=["1m", "5m", "15m", "30m"],
        index=2,
        horizontal=True,
    )
    st.caption("EODHD intraday 크립토 데이터 — 24/7, UTC anchored.")

    start_date = st.date_input(
        "Start Date",
        value=date.today() - timedelta(days=30),
        min_value=date(2018, 1, 1),
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
        value=96,  # 24h on 15m
        min_value=1,
        max_value=2_000,
        step=10,
        help="해당 bar 단위 기준. 15m=96이면 24시간, 1m=96이면 ~96분.",
    )

    st.header("Detector")
    swing_left = st.number_input(
        "Swing pivot left (bars)",
        value=2,
        min_value=1,
        max_value=10,
        step=1,
    )
    swing_right = st.number_input(
        "Swing pivot right (bars)",
        value=2,
        min_value=1,
        max_value=10,
        step=1,
    )
    min_choch_swing_atr = st.number_input(
        "Min ChoCH swing magnitude (× ATR)",
        value=2.0,
        min_value=0.0,
        max_value=10.0,
        step=0.5,
        format="%.1f",
        help="H1-L2 down-leg이 N×ATR 이상이어야 ChoCH 인정.",
    )
    min_gap_pct = st.number_input(
        "Min FVG size (%)",
        value=0.30,
        min_value=0.0,
        max_value=5.0,
        step=0.05,
        format="%.2f",
        help="close 대비 FVG 폭의 최소값. 크립토 변동성 → 0.30% 권장.",
    )
    max_bars_after_choch = st.number_input(
        "Max bars CHoCH → FVG",
        value=INTERVAL_CHOCH_FVG_DEFAULTS[interval],
        min_value=3,
        max_value=200,
        step=1,
    )
    max_retest_bars = st.number_input(
        "Max bars FVG → retest entry",
        value=INTERVAL_RETEST_DEFAULTS[interval],
        min_value=1,
        max_value=200,
        step=1,
    )
    max_signals_per_session = st.number_input(
        "Max signals per UTC day",
        value=4,
        min_value=1,
        max_value=20,
        step=1,
        help="크립토는 24/7이라 'session' = UTC 일자. 한 UTC day당 진입 시그널 상한.",
    )

    st.header("Exits")
    take_profit_r = st.number_input(
        "Take profit (× R)",
        value=3.0,
        min_value=0.5,
        max_value=10.0,
        step=0.5,
        format="%.1f",
    )
    enable_breakeven = st.checkbox(
        "Break-even stop after +1R",
        value=False,
    )
    enable_bos_trail = st.checkbox(
        "BOS trail to FVG midpoint",
        value=True,
        help="close가 진입 후 첫 swing high를 돌파(BOS)하면 stop을 FVG midpoint로.",
    )

    fee_schedule = render_toss_fee_inputs(key_prefix="crypto_fvg_")

    run_btn = st.button(
        "Run Crypto FVG Backtest", type="primary", use_container_width=True
    )

if not run_btn:
    st.info(
        "좌측에서 ticker / interval / 기간을 설정하고 "
        "**Run Crypto FVG Backtest**를 눌러."
    )
    st.stop()

# 24/7 markets — no RegularSessionFilter wrap. Every bar is RTH.
market_data = build_default_market_data()

detector = FairValueGapDetector(
    swing_left=int(swing_left),
    swing_right=int(swing_right),
    min_gap_pct=min_gap_pct / 100.0,
    max_bars_after_choch=int(max_bars_after_choch),
    max_retest_bars=int(max_retest_bars),
    max_signals_per_session=int(max_signals_per_session),
    min_choch_swing_atr=float(min_choch_swing_atr),
    # 24/7: no session-end gate. Pending FVGs still clear at UTC
    # midnight by design (stale-gap protection); structure carries
    # so the CHoCH-FVG-retest sequence can span days.
    min_bars_to_session_close=0,
    market=CRYPTO,
    allow_cross_session_carryover=True,
)


class _IntervalScopedFVGStrategy(FairValueGapStrategy):
    pass


_IntervalScopedFVGStrategy._interval = interval

strategy = _IntervalScopedFVGStrategy(
    market_data=market_data,
    detector=detector,
    take_profit_r_multiple=float(take_profit_r),
    enable_breakeven_stop=enable_breakeven,
    enable_bos_trail=enable_bos_trail,
    # 24/7: no after-hours / no session close. Both flags off so the
    # is_24_7 short-circuits in the strategy don't fight a True flag.
    disable_stops_outside_rth=False,
    force_close_at_session_end=False,
    market=CRYPTO,
)

config = StrategyConfig(
    ticker=ticker.upper(),
    start_date=start_date,
    end_date=end_date,
    pattern_name="fair_value_gap",
    initial_capital=float(initial_capital),
    risk_per_trade=risk_pct / 100.0,
    max_holding_days=int(max_holding_bars),
)

with st.spinner(f"Running FVG backtest on {ticker.upper()} ({interval})..."):
    try:
        result = strategy.run(config)
    except Exception as exc:
        st.error(f"Backtest failed: {exc}")
        st.stop()

perf = result.performance
chart_builder = PlotlyChartBuilder()

if perf.total_trades == 0:
    st.subheader(f"{ticker.upper()} — {interval}")
    st.metric("Trades", 0)
    st.info(
        "이 기간엔 FVG signal이 안 잡혔거나 진입 조건을 다 통과한 게 없음. "
        "기간/필터를 조정해봐 — 차트는 그대로 아래에 표시돼."
    )
else:
    fee_rows = apply_fees_to_trades(perf.trades, fee_schedule)
    render_single_ticker_headline_metrics(
        perf, fee_rows, subtitle=f"{ticker.upper()} — {interval}"
    )

    st.subheader("Equity Curve")
    st.plotly_chart(
        chart_builder.build_equity_curve(
            result.equity_curve,
            title=f"{ticker.upper()} {interval} FVG equity (gross)",
        ),
        use_container_width=True,
    )
    st.caption("Equity curve는 gross PnL 기반. 위 metric / trade table은 수수료 차감 후 net.")

    render_single_ticker_trade_table(fee_rows)

st.subheader("Chart")
try:
    full_df = market_data.fetch_ohlcv(
        ticker.upper(), start_date, end_date, interval=interval
    )
except Exception as exc:
    st.warning(f"Failed to fetch chart data: {exc}")
else:
    chart_signals = detector.detect(full_df)
    full_fig = chart_builder.build_candlestick_with_trades(
        full_df,
        perf.trades,
        title=f"{ticker.upper()} — {start_date} → {end_date} ({interval} FVG)",
        fvg_signals=chart_signals,
        take_profit_r=float(take_profit_r),
        market=CRYPTO,
    )
    full_fig.update_xaxes(range=[str(start_date), str(end_date)])
    st.plotly_chart(full_fig, use_container_width=True)
    st.caption(
        "🟩 FVG zone (light green box)  ·  ━━ midpoint hold/break level "
        "(dashed)  ·  ChoCH (purple dashdot)  ·  BOS (orange dotted)  ·  "
        "┄┄ red dash = stop, green dot = take-profit (R-target). "
        "24/7 시장이라 weekend rangebreak 없이 연속 시간축."
    )
