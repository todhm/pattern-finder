from datetime import date, timedelta

import streamlit as st

from data.adapters.cached_market_data import CachedMarketDataAdapter
from data.adapters.wikipedia_universe import WikipediaUniverseAdapter
from data.adapters.yfinance_adapter import YFinanceAdapter
from pattern.adapters.exhaustion_extension_top import (
    ExhaustionExtensionTopDetector,
)
from pattern.adapters.wick_play import WickPlayDetector
from strategy.adapters.multi_wickplay_strategy import MultiWickPlayStrategy
from strategy.adapters.wickplay_strategy import WickPlayStrategy
from strategy.domain.models import MultiStrategyConfig, TossFeeSchedule
from visualization.adapters.plotly_charts import PlotlyChartBuilder

st.set_page_config(page_title="Multi Wick Play Scan", layout="wide")
st.title("Multi-Ticker Wick Play Strategy")
st.caption(
    "S&P 500 / Nasdaq-100 전체에서 매일 발생한 Wick Play signal 중 "
    "buy/sell 거래량 비율이 가장 높은 종목 하나를 매수. "
    "한 포지션이 청산될 때까지 다른 종목은 사지 않음. "
    "진입=다음날 open, stop=wick low, exit=10 EMA trail / Exhaustion Extension Top / time stop."
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
        value=50,
        min_value=0,
        max_value=600,
        step=10,
        help="스캔할 최대 종목 수. 0이면 전체. 처음엔 작게 시작.",
    )
    max_workers = st.number_input(
        "Parallel workers",
        value=8,
        min_value=1,
        max_value=32,
        step=1,
    )

    st.header("Period")
    start_date = st.date_input(
        "Start Date",
        value=date.today() - timedelta(days=365 * 2),
        min_value=date(2000, 1, 1),
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
    max_holding_days = st.number_input(
        "Max Holding Days",
        value=40,
        min_value=1,
        max_value=500,
        step=5,
        help="Wick Play는 multi-week move를 노리는 setup이라 40일 기본.",
    )

    st.header("Wick Play Detector")
    min_upper_wick_ratio = st.number_input(
        "Min upper wick / range",
        value=0.5,
        min_value=0.0,
        max_value=1.0,
        step=0.05,
        format="%.2f",
    )
    max_volume_dryup = st.number_input(
        "Max inside-bar vol vs wick bar",
        value=1.0,
        min_value=0.1,
        max_value=2.0,
        step=0.05,
        format="%.2f",
    )
    breakout_trigger = st.selectbox(
        "Breakout trigger",
        options=["wick_high", "inside_high"],
        index=0,
    )
    enable_max_wick_range = st.checkbox(
        "Cap wick-bar range (× ATR)", value=True
    )
    max_wick_range_atr = st.number_input(
        "Max wick-bar range (× ATR)",
        value=2.5,
        min_value=0.5,
        max_value=10.0,
        step=0.1,
        format="%.2f",
        disabled=not enable_max_wick_range,
    )
    cooldown_bars = st.number_input(
        "Cooldown bars",
        value=5,
        min_value=0,
        max_value=60,
        step=1,
    )

    st.subheader("Psychology score (4 checks)")
    min_psych_score = st.slider("Min psych score", 0, 4, 3, 1)
    psych_vol_lookback = st.number_input(
        "Vol avg lookback (days)",
        value=20, min_value=5, max_value=100, step=5,
    )
    psych_wick_vol_exhaustion_mult = st.number_input(
        "Check 1: wick vol ≤ avg × mult",
        value=1.0, min_value=0.1, max_value=3.0, step=0.05, format="%.2f",
    )
    psych_breakout_vol_expansion_mult = st.number_input(
        "Check 2: breakout vol > wick vol × mult",
        value=1.0, min_value=0.1, max_value=3.0, step=0.05, format="%.2f",
    )
    psych_prior_red_streak = st.number_input(
        "Check 3: prior red streak (fail if all red)",
        value=2, min_value=0, max_value=10, step=1,
    )
    psych_dramatic_wick_ratio = st.number_input(
        "Check 4: dramatic wick ratio",
        value=0.65, min_value=0.3, max_value=1.0, step=0.05, format="%.2f",
    )

    st.subheader("Hard gates (failure post-mortem)")
    st.caption(
        "실패 11건 분석으로 추가 — 약한 돌파 + 하락추세 wick 차단."
    )
    min_breakout_strength_atr_wk = st.number_input(
        "① Min breakout strength (× ATR)",
        value=0.3, min_value=0.0, max_value=5.0, step=0.05, format="%.2f",
        help="0.0 = 예전 동작. 0.3 = 기본, 약한 돌파 차단.",
    )
    enable_min_prior_trend = st.checkbox(
        "② Enforce min 20-day prior trend",
        value=True,
    )
    min_prior_trend_20d_wk = st.number_input(
        "② Min 20-day trend (fraction)",
        value=-0.03, min_value=-0.5, max_value=0.5, step=0.005, format="%.3f",
        disabled=not enable_min_prior_trend,
        help="-0.03 = 20일 -3% 이상 빠진 종목 거부.",
    )
    enable_max_prior_trend = st.checkbox(
        "③ Enforce max 20-day prior trend (parabolic cap)",
        value=False,
    )
    max_prior_trend_20d_wk = st.number_input(
        "③ Max 20-day trend (fraction)",
        value=0.15, min_value=0.0, max_value=1.0, step=0.01, format="%.2f",
        disabled=not enable_max_prior_trend,
        help="0.15 = 20일 +15% 초과 parabolic 종목 거부.",
    )
    min_wick_close_location_wk = st.number_input(
        "⑥ Min wick close location (0=low, 1=high)",
        value=0.15, min_value=0.0, max_value=1.0, step=0.05, format="%.2f",
        help="Wick bar close가 range 하단 N% 이상이어야 통과. "
        "0.15 = 하단 15% 미만 마감은 'outright bearish' 봉으로 보고 거부.",
    )
    enable_min_breakout_cl_wk = st.checkbox(
        "⑧ Enforce min breakout close location",
        value=False,
    )
    min_breakout_close_location_wk = st.number_input(
        "⑧ Min breakout close location",
        value=0.70, min_value=0.0, max_value=1.0, step=0.05, format="%.2f",
        disabled=not enable_min_breakout_cl_wk,
    )

    st.header("Exits")
    ema_trail = st.number_input(
        "EMA trail period",
        value=10, min_value=3, max_value=50, step=1,
    )
    min_trail_bars = st.number_input(
        "Min bars before trail/exhaustion fires",
        value=2, min_value=0, max_value=10, step=1,
    )
    enable_exh_exit = st.checkbox(
        "Enable Exhaustion Extension Top exit", value=True
    )
    exh_extension_atr = st.number_input(
        "Exh extension above EMA (× ATR)",
        value=1.9, min_value=0.5, max_value=10.0, step=0.1,
        disabled=not enable_exh_exit,
    )
    exh_min_slope = st.number_input(
        "Exh min slow-EMA slope",
        value=0.005, min_value=-1.0, max_value=1.0, step=0.001, format="%.4f",
        disabled=not enable_exh_exit,
    )

    st.subheader("Same-day reversal exit (opt-in)")
    enable_same_day_reversal = st.checkbox(
        "⑤ Enable same-day reversal exit", value=False,
        help="진입 당일 close가 range 하단에 마감되면 바로 청산. PODD 케이스 대응. 기본 off.",
    )
    max_same_day_close_location = st.number_input(
        "Max same-day close location (0=low, 1=high)",
        value=0.3, min_value=0.0, max_value=1.0, step=0.05, format="%.2f",
        disabled=not enable_same_day_reversal,
    )

    st.subheader("Gap-down rejection (opt-in)")
    enable_gap_down_rejection = st.checkbox(
        "⑨ Reject entry if gap-down vs breakout close", value=False,
        help="진입 open이 breakout close 대비 N% 이상 갭다운이면 진입 거부. 기본 off.",
    )
    max_entry_gap_down = st.number_input(
        "Max entry gap-down (fraction)",
        value=0.005, min_value=0.0, max_value=0.1, step=0.001, format="%.3f",
        disabled=not enable_gap_down_rejection,
    )

    st.header("Fees (Toss Securities)")
    buy_fee_pct = st.number_input(
        "Buy commission (%)",
        value=0.10, min_value=0.0, max_value=5.0, step=0.01, format="%.4f",
    )
    sell_fee_pct = st.number_input(
        "Sell commission (%)",
        value=0.10, min_value=0.0, max_value=5.0, step=0.01, format="%.4f",
    )
    sec_fee_pct = st.number_input(
        "SEC fee (%) — sell only",
        value=0.00229, min_value=0.0, max_value=1.0, step=0.0001, format="%.5f",
    )

    run_btn = st.button(
        "Run Universe Scan", type="primary", use_container_width=True
    )


# --- Main ---
if not run_btn:
    st.info(
        "좌측에서 universe / 기간을 설정하고 **Run Universe Scan** 을 눌러줘. "
        "처음 실행 땐 max_tickers를 작게 잡고 스모크 테스트."
    )
    st.stop()

market_data = CachedMarketDataAdapter(YFinanceAdapter())
universe_provider = WikipediaUniverseAdapter()

detector = WickPlayDetector(
    min_upper_wick_ratio=float(min_upper_wick_ratio),
    max_volume_dryup=float(max_volume_dryup),
    breakout_trigger=breakout_trigger,
    stop_mode="wick_low",
    max_wick_range_atr=(
        float(max_wick_range_atr) if enable_max_wick_range else None
    ),
    cooldown_bars=int(cooldown_bars),
    psych_vol_lookback=int(psych_vol_lookback),
    psych_wick_vol_exhaustion_mult=float(psych_wick_vol_exhaustion_mult),
    psych_breakout_vol_expansion_mult=float(psych_breakout_vol_expansion_mult),
    psych_prior_red_streak=int(psych_prior_red_streak),
    psych_dramatic_wick_ratio=float(psych_dramatic_wick_ratio),
    min_psych_score=int(min_psych_score),
    min_breakout_strength_atr=float(min_breakout_strength_atr_wk),
    min_prior_trend_20d=(float(min_prior_trend_20d_wk) if enable_min_prior_trend else None),
    max_prior_trend_20d=(float(max_prior_trend_20d_wk) if enable_max_prior_trend else None),
    min_wick_close_location=float(min_wick_close_location_wk),
    min_breakout_close_location=(
        float(min_breakout_close_location_wk) if enable_min_breakout_cl_wk else None
    ),
)

exit_detector = (
    ExhaustionExtensionTopDetector(
        extension_atr_mult=float(exh_extension_atr),
        min_slow_slope=float(exh_min_slope),
        ema_fast=int(ema_trail),
    )
    if enable_exh_exit
    else None
)

per_ticker_strategy = WickPlayStrategy(
    market_data=market_data,
    detector=detector,
    exit_detector=exit_detector,
    ema_trail=int(ema_trail),
    min_trail_bars=int(min_trail_bars),
    enable_same_day_reversal_exit=enable_same_day_reversal,
    max_same_day_close_location=float(max_same_day_close_location),
    enable_gap_down_rejection=enable_gap_down_rejection,
    max_entry_gap_down=float(max_entry_gap_down),
)

runner = MultiWickPlayStrategy(
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
    max_holding_days=int(max_holding_days),
    max_tickers=int(max_tickers) if max_tickers > 0 else None,
    fee_schedule=fee_schedule,
)

with st.spinner(
    f"Scanning {universe} ({config.max_tickers or 'all'} tickers)..."
):
    try:
        result = runner.run(config)
    except Exception as e:
        st.error(f"Universe scan failed: {e}")
        st.stop()

# --- Headline metrics ---
st.subheader(f"Universe — {universe.upper()}")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Tickers Scanned", result.tickers_scanned)
m2.metric("Total Signals", result.total_signals)
m3.metric("Trades Taken", result.trades_taken)
m4.metric("Failed Tickers", len(result.failed_tickers))

m5, m6, m7, m8 = st.columns(4)
m5.metric("Total Return (net)", f"{result.total_return_pct:.2%}")
m6.metric("Win Rate", f"{result.win_rate:.0%}" if result.trades_taken else "—")
m7.metric("Final Capital", f"${result.final_capital:,.0f}")
m8.metric("Max Drawdown", f"{result.max_drawdown_pct:.2%}")

m9, m10, m11, m12 = st.columns(4)
gross_pnl = sum(t.gross_pnl for t in result.trades)
m9.metric("Total Commission", f"${result.total_commission:,.2f}")
m10.metric("Gross P&L", f"${gross_pnl:,.2f}")
m11.metric("Net P&L", f"${gross_pnl - result.total_commission:,.2f}")
m12.metric(
    "Fees as % of Gross",
    f"{(result.total_commission / gross_pnl):.2%}" if gross_pnl else "—",
)

if result.tickers_scanned == 0:
    st.warning("Universe returned 0 tickers — check your selection.")
    st.stop()

if not result.trades:
    st.info(
        "이 기간엔 universe 전체에서 Wick Play signal이 잡히지 않았거나, "
        "진입 조건에서 모두 걸렸어. 기간을 늘리거나 psych score를 낮춰봐."
    )
    st.stop()

chart_builder = PlotlyChartBuilder()

st.subheader("Portfolio Equity Curve")
eq_fig = chart_builder.build_equity_curve(
    result.equity_curve,
    title=f"Single-position portfolio across {universe.upper()}",
)
st.plotly_chart(eq_fig, use_container_width=True)

st.subheader("Best Trades")
trades_fig = chart_builder.build_top_trades_bar(
    result.trades, title="Top trades by % return"
)
st.plotly_chart(trades_fig, use_container_width=True)

st.subheader("P&L by Ticker")
contrib_fig = chart_builder.build_ticker_contribution_bar(
    result.trades, title="Total P&L contribution per ticker"
)
st.plotly_chart(contrib_fig, use_container_width=True)

st.subheader("Trades — Details")
EXIT_LABELS = {
    "wick_low_stop": "Wick Low Stop (initial)",
    "same_day_reversal": "Same-Day Reversal (intraday fail)",
    "ema_trail": "10 EMA Trail (Kell stairstep)",
    "exhaustion_exit": "Exhaustion Extension Top",
    "time_stop": "Time Stop",
    "end_of_data": "End of Data (no exit fired)",
}

rows = [
    {
        "Ticker": t.ticker,
        "Entry Date": t.entry_date,
        "Exit Date": t.exit_date,
        "Exit Reason": EXIT_LABELS.get(t.exit_reason, t.exit_reason),
        "Entry": f"${t.entry_price:,.2f}",
        "Exit": f"${t.exit_price:,.2f}",
        "Stop": f"${t.stop_loss:,.2f}",
        "Shares": t.shares,
        "Sig Vol": f"{t.signal_volume:,.0f}",
        "Buy/Sell": f"{t.signal_buy_sell_ratio:.2f}",
        "Gross P&L": f"${t.gross_pnl:,.2f}",
        "Commission": f"${t.commission:,.2f}",
        "Net P&L ($)": f"${t.pnl:,.2f}",
        "Net P&L (%)": f"{t.pnl_pct:.2%}",
    }
    for t in result.trades
]
st.dataframe(rows, use_container_width=True)

st.subheader("Trade Charts")
CONTEXT_BEFORE_DAYS = 60
CONTEXT_AFTER_DAYS = 30

for i, t in enumerate(result.trades):
    pnl_sign = "+" if t.pnl >= 0 else ""
    label = (
        f"{t.ticker} — {t.entry_date} → {t.exit_date}  "
        f"({pnl_sign}{t.pnl_pct:.2%})"
    )
    with st.expander(label, expanded=(i < 3)):
        chart_start = t.entry_date - timedelta(days=CONTEXT_BEFORE_DAYS)
        chart_end = t.exit_date + timedelta(days=CONTEXT_AFTER_DAYS)
        fetch_start = chart_start - timedelta(days=150)
        try:
            ticker_df = market_data.fetch_ohlcv(
                t.ticker, fetch_start, chart_end
            )
        except Exception:
            st.warning(f"Failed to fetch data for {t.ticker}")
            continue
        if ticker_df is None or ticker_df.empty:
            st.warning(f"No data for {t.ticker}")
            continue

        trade_fig = chart_builder.build_candlestick_with_trades(
            ticker_df,
            [t],
            title=f"{t.ticker} — {t.entry_date} → {t.exit_date}",
        )
        trade_fig.update_xaxes(range=[str(chart_start), str(chart_end)])
        st.plotly_chart(trade_fig, use_container_width=True)

if result.failed_tickers:
    with st.expander(f"Failed tickers ({len(result.failed_tickers)})"):
        st.write(", ".join(result.failed_tickers))
