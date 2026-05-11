"""Multi-ticker Bull Flag (Ross Cameron) scan — universe runner.

Sister page to ``21_Bull_Flag_Strategy.py``. Walks an NY-equity
universe day-by-day and, when multiple tickers fire on the same
session, picks the one with the strongest Ross-style profile.

**Tiebreaker — same-day signals**:
    1. **RVOL 내림차순** — 오늘의 runner (Ross: "leading percentage gainer
       with thick volume"). Higher daily RVOL = more institutional
       participation + thicker market = better fills + bigger move ahead.
    2. **Float 오름차순** — supply 제약이 강한 쪽 (같은 매수 압력에 더
       큰 % 이동). Ross의 #1 selection criterion.
    3. **Entry_ts 오름차순** — 영상 sweet spot 09:30~10:30 첫 풀백.

One trade per day — once a position is taken, other same-day
signals are dropped. Position is exited intraday (TP / stop /
session_close) so the next day starts flat.
"""

from datetime import date, time, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data.adapters.composed_fundamentals import build_default_fundamentals
from data.adapters.composed_market_data import build_default_market_data
from data.adapters.regular_session_filter import RegularSessionFilterAdapter
from data.adapters.wikipedia_universe import default_universe_provider
from data.domain.market_calendar import NY
from pages._shared.wedgepop_results import (
    render_failed_tickers,
    render_headline_metrics,
)
from pattern.adapters.bull_flag import BullFlagDetector
from strategy.adapters.bull_flag_strategy import BullFlagStrategy
from strategy.adapters.multi_bull_flag_strategy import MultiBullFlagStrategy
from strategy.domain.models import (
    MultiStrategyConfig,
    TossFeeSchedule,
)

INTRADAY_HISTORY_FLOOR = date(2022, 1, 1)

st.set_page_config(page_title="Multi Bull Flag", layout="wide")
st.title("Multi Bull Flag Scan — Ross Cameron")
st.caption(
    "NASDAQ universe-wide Bull Flag scan. 동일 세션에 여러 종목이 시그널을 만들면 "
    "**RVOL 내림차순 → Float 오름차순 → Entry 시각 빠른 순** 으로 한 종목 선택. "
    "한 번에 하루 한 종목만 트레이드. Toss 증권 수수료 적용."
)

# ---- Sidebar -------------------------------------------------------
with st.sidebar:
    st.header("Universe")
    universe = st.selectbox(
        "Universe",
        options=["nasdaq_full", "nasdaq100", "sp500"],
        index=0,
        help="nasdaq_full = NASDAQ 전체 common stock (~2200). 영상 정통 = nasdaq_full.",
    )
    max_tickers = st.number_input(
        "Max tickers (0 = all)",
        value=200,
        min_value=0,
        max_value=2500,
        step=50,
        help="0 = universe 전체 (~2,200). 1m+5m+daily+float 모두 병렬 fetch.",
    )
    max_workers = st.number_input(
        "Parallel workers",
        value=8,
        min_value=1,
        max_value=32,
        step=1,
        help="ticker별 1m + 5m + daily + float 병렬 fetch worker 수.",
    )

    st.header("Date range")
    start_date = st.date_input(
        "Start Date",
        value=date.today() - timedelta(days=14),
        min_value=INTRADAY_HISTORY_FLOOR,
        max_value=date.today(),
    )
    end_date = st.date_input(
        "End Date",
        value=date.today(),
        min_value=INTRADAY_HISTORY_FLOOR,
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
    max_position_pct = st.number_input(
        "Max position (% of equity)",
        value=30.0,
        min_value=1.0,
        max_value=100.0,
        step=5.0,
    )

    st.header("Stock Selection (4 criteria — 영상 정통)")
    min_gap_pct = st.number_input(
        "Min pre-market gap (%)", value=2.0, min_value=0.0, max_value=50.0,
        step=0.5, format="%.1f",
    )
    min_rvol = st.number_input(
        "Min Relative Volume (×)", value=5.0, min_value=1.0, max_value=20.0,
        step=0.5, format="%.1f",
    )
    min_price = st.number_input(
        "Min price ($)", value=2.0, min_value=0.5, max_value=100.0, step=0.5,
    )
    max_price = st.number_input(
        "Max price ($)", value=20.0, min_value=1.0, max_value=500.0, step=1.0,
    )
    max_float_shares_mil = st.number_input(
        "Max float (millions)", value=10.0, min_value=0.5, max_value=500.0,
        step=1.0,
    )
    require_float = st.checkbox(
        "Require float data", value=True,
        help="True (영상 정통): float 모르면 그 종목 skip. False: pass-through.",
    )

    st.header("Pattern (Bull Flag geometry)")
    pole_lookback = st.number_input("Pole lookback (bars)", value=7, min_value=2, max_value=30, step=1)
    pole_min_pct = st.number_input("Pole min rise (%)", value=8.0, min_value=0.5, max_value=50.0, step=0.5, format="%.1f")
    pole_min_green_bars = st.number_input("Pole min green bars", value=3, min_value=1, max_value=20, step=1)
    flag_max_bars = st.number_input("Flag max bars", value=4, min_value=1, max_value=10, step=1)
    flag_max_retrace = st.number_input("Flag max retrace (% of pole)", value=70.0, min_value=10.0, max_value=99.0, step=5.0, format="%.0f")
    latest_entry_hour = st.number_input("Latest entry hour (ET)", value=12, min_value=10, max_value=15, step=1)
    min_bar_range = st.number_input("Min bar range ($)", value=0.001, min_value=0.0, max_value=1.0, step=0.001, format="%.3f")
    max_bar_gap_seconds = st.number_input("Max bar gap (seconds)", value=90, min_value=30, max_value=600, step=15)

    st.header("Exit / Sizing")
    target_min_r = st.number_input("Min R/R required", value=2.0, min_value=1.0, max_value=10.0, step=0.5, format="%.1f")
    use_fixed_target = st.checkbox("Use fixed R-multiple target", value=True)
    fixed_target_r = st.number_input("Fixed target R-multiple", value=2.0, min_value=1.0, max_value=10.0, step=0.5, format="%.1f", disabled=not use_fixed_target)
    enable_add = st.checkbox("Add to winner at +R", value=True)
    add_at_r = st.number_input("Add trigger (R)", value=1.5, min_value=0.3, max_value=3.0, step=0.1, format="%.1f", disabled=not enable_add)
    add_confirm_on_close = st.checkbox("Add: close-confirmed", value=True, disabled=not enable_add)
    max_session_losses = st.number_input("Max session losses", value=1, min_value=0, max_value=10, step=1)
    be_stop_buffer_pct = st.number_input("BE stop buffer (%)", value=0.3, min_value=0.0, max_value=2.0, step=0.05, format="%.2f")

    st.header("MTF (DP4 영상 정통)")
    enable_mtf = st.checkbox("Enable MTF check (1m + 5m)", value=False)
    mtf_tolerance_seconds = st.number_input(
        "MTF tolerance (s)", value=600, min_value=60, max_value=1800, step=60,
        disabled=not enable_mtf,
    )

    st.header("Reverse-split Guard")
    split_blackout_days = st.number_input("Split blackout days (±N)", value=30, min_value=0, max_value=180, step=5)
    price_floor_lookback_days = st.number_input("Price-floor lookback (days)", value=30, min_value=0, max_value=120, step=5)

    st.header("Toss 수수료")
    buy_pct = st.number_input("Buy (%)", value=0.10, min_value=0.0, max_value=2.0, step=0.01, format="%.3f")
    sell_pct = st.number_input("Sell (%)", value=0.10, min_value=0.0, max_value=2.0, step=0.01, format="%.3f")
    sec_pct = st.number_input("SEC (%)", value=0.0023, min_value=0.0, max_value=0.1, step=0.0001, format="%.4f")

    run_btn = st.button("Run Multi Bull Flag Scan", type="primary", use_container_width=True)

if not run_btn:
    st.info(
        "좌측에서 universe / 기간 / Max tickers를 설정하고 **Run** 버튼을 눌러. "
        "NASDAQ 전체(~2,200) 스캔도 가능 — 시간이 오래 걸리면 Max tickers를 줄여 시작."
    )
    st.stop()

# ---- Build adapters + factories ------------------------------------
md = RegularSessionFilterAdapter(build_default_market_data(), market=NY)
md_5m = md  # 같은 어댑터로 5m 호출 (interval만 다름)
# Daily도 composed adapter 통해 fetch (yfinance primary + Massive fallback)
md_daily = md
fundamentals = build_default_fundamentals()
universe_provider = default_universe_provider()

fee_schedule = TossFeeSchedule(
    buy_commission_pct=float(buy_pct) / 100.0,
    sell_commission_pct=float(sell_pct) / 100.0,
    sec_fee_pct=float(sec_pct) / 100.0,
)


def detector_factory(*, float_shares, splits):
    return BullFlagDetector(
        float_shares=float_shares,
        require_float_filter=bool(require_float),
        min_gap_pct=float(min_gap_pct) / 100.0,
        min_rvol=float(min_rvol),
        min_price=float(min_price),
        max_price=float(max_price),
        max_float_shares=float(max_float_shares_mil) * 1e6,
        pole_lookback=int(pole_lookback),
        pole_min_pct=float(pole_min_pct) / 100.0,
        pole_min_green_bars=int(pole_min_green_bars),
        flag_max_bars=int(flag_max_bars),
        flag_max_retrace=float(flag_max_retrace) / 100.0,
        latest_entry_local=time(int(latest_entry_hour), 0),
        splits=splits,
        split_blackout_days=int(split_blackout_days),
        price_floor_lookback_days=int(price_floor_lookback_days),
        enable_mtf_check=bool(enable_mtf),
        mtf_tolerance_seconds=int(mtf_tolerance_seconds),
        min_bar_range=float(min_bar_range),
        max_bar_gap_seconds=int(max_bar_gap_seconds),
    )


def strategy_factory(*, detector):
    return BullFlagStrategy(
        detector=detector,
        max_position_pct_of_equity=float(max_position_pct) / 100.0,
        target_min_r_multiple=float(target_min_r),
        target_at_r_multiple=float(fixed_target_r) if use_fixed_target else None,
        enable_add_to_winner=bool(enable_add),
        add_at_r=float(add_at_r),
        add_confirm_on_close=bool(add_confirm_on_close),
        max_session_losses=int(max_session_losses),
        be_stop_buffer_pct=float(be_stop_buffer_pct) / 100.0,
        fee_schedule=fee_schedule,
    )


multi = MultiBullFlagStrategy(
    market_data=md,
    market_data_5m=md_5m,
    daily_market_data=md_daily,
    fundamentals=fundamentals,
    universe_provider=universe_provider,
    detector_factory=detector_factory,
    strategy_factory=strategy_factory,
    market=NY,
    max_workers=int(max_workers),
    require_float_filter=bool(require_float),
)

config = MultiStrategyConfig(
    universe=universe,
    start_date=start_date,
    end_date=end_date,
    pattern_name="bull_flag",
    initial_capital=float(initial_capital),
    risk_per_trade=float(risk_pct) / 100.0,
    max_holding_days=1,
    max_tickers=int(max_tickers) if max_tickers > 0 else None,
    fee_schedule=fee_schedule,
)

with st.spinner("Scanning universe... (1m+5m+daily fetch + float lookup throttled)"):
    try:
        result = multi.run(config)
    except Exception as exc:
        st.error(f"Scan failed: {exc}")
        st.stop()

# ---- Render results -------------------------------------------------
render_headline_metrics(result, universe_label=universe)

# Equity curve
if len(result.equity_curve) > 1:
    st.subheader("Portfolio Equity Curve")
    eq_df = pd.DataFrame(
        [(e.date, e.equity) for e in result.equity_curve],
        columns=["date", "equity"],
    )
    fig = go.Figure(
        go.Scatter(x=eq_df["date"], y=eq_df["equity"], mode="lines+markers",
                   line=dict(color="#1976D2", width=2))
    )
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="Equity ($)")
    st.plotly_chart(fig, use_container_width=True)

# Trade table
st.subheader("Trades — Details (one per day)")
EXIT_LABELS = {
    "take_profit": "✅ Take Profit",
    "stop_loss": "❌ Stop Loss",
    "breakeven_stop": "🟰 BE Stop (after add)",
    "session_close": "🕒 Session Close",
    "end_of_data": "📭 End of Data",
}
if not result.trades:
    st.info("이 기간/유니버스에서 4 criteria + Bull Flag + MTF 통과 후 trade가 발생하지 않음.")
else:
    rows = []
    for t in result.trades:
        rows.append({
            "Date": str(t.entry_date),
            "Ticker": t.ticker,
            "Entry": str(t.entry_ts)[:16] if t.entry_ts else str(t.entry_date),
            "Exit": str(t.exit_ts)[:16] if t.exit_ts else str(t.exit_date),
            "Reason": EXIT_LABELS.get(t.exit_reason, t.exit_reason),
            "Entry $": f"${t.entry_price:.2f}",
            "Exit $": f"${t.exit_price:.2f}",
            "Stop $": f"${t.stop_loss:.2f}",
            "Shares": t.shares,
            "RVOL": f"{t.signal_volume:.1f}x",
            "Float (M)": f"{t.signal_buy_volume/1e6:.2f}M" if t.signal_buy_volume > 0 else "—",
            "Gross $": f"${t.gross_pnl:+,.0f}",
            "Comm $": f"${t.commission:,.2f}",
            "Net $": f"${t.pnl:+,.0f}",
            "Net %": f"{t.pnl_pct:+.2%}",
        })
    st.dataframe(rows, use_container_width=True)

# 종목별 P&L 기여도
if result.trades:
    st.subheader("P&L by Ticker")
    by_ticker = {}
    for t in result.trades:
        by_ticker[t.ticker] = by_ticker.get(t.ticker, 0.0) + t.pnl
    by_ticker_sorted = sorted(by_ticker.items(), key=lambda x: x[1], reverse=True)
    fig = go.Figure(
        go.Bar(
            x=[x[0] for x in by_ticker_sorted],
            y=[x[1] for x in by_ticker_sorted],
            marker_color=["#26A69A" if v > 0 else "#EF5350" for _, v in by_ticker_sorted],
            text=[f"${v:+,.0f}" for _, v in by_ticker_sorted],
            textposition="outside",
        )
    )
    fig.update_layout(
        height=320, margin=dict(l=10, r=10, t=10, b=10),
        yaxis_title="Net P&L ($)", xaxis_title="Ticker",
    )
    st.plotly_chart(fig, use_container_width=True)

render_failed_tickers(result)
