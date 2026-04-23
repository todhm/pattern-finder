from datetime import date, timedelta

import streamlit as st

from data.adapters.yfinance_adapter import YFinanceAdapter
from pattern.adapters.exhaustion_extension_top import (
    ExhaustionExtensionTopDetector,
)
from pattern.adapters.wick_play import WickPlayDetector
from strategy.adapters.wickplay_strategy import WickPlayStrategy
from strategy.domain.models import StrategyConfig
from visualization.adapters.plotly_charts import PlotlyChartBuilder

st.set_page_config(page_title="Wick Play Strategy", layout="wide")
st.title("Wick Play Strategy")
st.caption(
    "Oliver Kell의 Wick Play 진입 → wick low 손절 / 10 EMA trail / "
    "Exhaustion Extension Top / time stop"
)

# --- Sidebar ---
with st.sidebar:
    st.header("Market")
    ticker = st.text_input("Ticker", value="AAPL")
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
        help="거래당 자본 대비 위험 %.",
    )
    max_holding_days = st.number_input(
        "Max Holding Days (time stop)",
        value=40,
        min_value=5,
        max_value=500,
        step=5,
        help="multi-week 이동을 노리는 setup이라 40일 기본. "
        "시간 초과 시 다음 봉 open에서 청산.",
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
    st.caption("4개 중 N개 이상 통과해야 신호 발동.")
    min_psych_score = st.slider(
        "Min psych score", 0, 4, 3, 1
    )
    psych_vol_lookback = st.number_input(
        "Vol avg lookback (days)", value=20, min_value=5, max_value=100, step=5
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
        "Check 3: prior red streak bars (fail if all red)",
        value=2, min_value=0, max_value=10, step=1,
    )
    psych_dramatic_wick_ratio = st.number_input(
        "Check 4: dramatic wick ratio",
        value=0.65, min_value=0.3, max_value=1.0, step=0.05, format="%.2f",
    )

    st.subheader("Hard gates (2024–2026 failure post-mortem)")
    st.caption(
        "11개 실패 트레이드(PODD/LITE/AIZ/CB/PLTR/AEE/HUM/SBAC/ETR/ADM/KHC) "
        "분석으로 추가된 필터."
    )
    min_breakout_strength_atr_wk = st.number_input(
        "① Min breakout strength (× ATR)",
        value=0.3, min_value=0.0, max_value=5.0, step=0.05, format="%.2f",
        help="돌파 close가 trigger level을 ATR의 N배 이상 넘어야 함. "
        "0.0 = 예전 동작 (아무리 작은 돌파도 OK). "
        "0.3 = 기본 — 기계적 stop-run 돌파 차단. "
        "실패 11건 중 6건이 이 값 미만으로 돌파.",
    )
    enable_min_prior_trend = st.checkbox(
        "② Enforce min 20-day prior trend",
        value=True,
        help="Wick Play는 상승추세 pullback에서만 작동. "
        "하락 추세 한복판의 wick은 대부분 실패 (HUM -17%, ADM -7.5% 케이스).",
    )
    min_prior_trend_20d_wk = st.number_input(
        "② Min 20-day trend (fraction)",
        value=-0.03, min_value=-0.5, max_value=0.5, step=0.005, format="%.3f",
        disabled=not enable_min_prior_trend,
        help="-0.03 = 직전 20일 -3% 이상 빠진 종목 거부. "
        "-2% (-0.02)로 하면 더 엄격, -5% (-0.05)로 하면 느슨.",
    )
    enable_max_prior_trend = st.checkbox(
        "③ Enforce max 20-day prior trend (parabolic cap)",
        value=False,
        help="이미 너무 오른 종목 제외 (PLTR +21% 케이스). 기본 off.",
    )
    max_prior_trend_20d_wk = st.number_input(
        "③ Max 20-day trend (fraction)",
        value=0.15, min_value=0.0, max_value=1.0, step=0.01, format="%.2f",
        disabled=not enable_max_prior_trend,
        help="0.15 = 직전 20일 +15% 초과 종목 거부 (parabolic).",
    )
    min_wick_close_location_wk = st.number_input(
        "⑥ Min wick close location (0=low, 1=high)",
        value=0.15, min_value=0.0, max_value=1.0, step=0.05, format="%.2f",
        help="Wick bar close가 range 하단 N% 이상이어야 통과. "
        "0.15 = 하단 15% 미만 마감은 'outright bearish' 봉으로 보고 거부. "
        "ZTS/BG/PODD 2024 실패(모두 CL < 0.05) 차단.",
    )
    enable_min_breakout_cl_wk = st.checkbox(
        "⑧ Enforce min breakout close location",
        value=False,
        help="돌파 봉 close가 range 상단 N% 이상 마감 요구. 약한 돌파 마감(ZTS 0.61) 차단.",
    )
    min_breakout_close_location_wk = st.number_input(
        "⑧ Min breakout close location",
        value=0.70, min_value=0.0, max_value=1.0, step=0.05, format="%.2f",
        disabled=not enable_min_breakout_cl_wk,
    )

    st.header("Exits")
    st.caption(
        "**Stop** = wick_low (Kell 여유 stop). "
        "**Trail** = 10 EMA — close가 EMA 아래로 떨어지면 그 봉 close에 청산."
    )
    ema_trail = st.number_input(
        "EMA trail period",
        value=10, min_value=3, max_value=50, step=1,
        help="Kell / Minervini의 stairstep trail. 10 EMA 기본.",
    )
    min_trail_bars = st.number_input(
        "Min bars before trail/exhaustion fires",
        value=2, min_value=0, max_value=10, step=1,
        help="진입 직후 N봉은 trail/exhaustion exit 비활성 — 돌파 봉이 소화되도록 여유.",
    )
    enable_exh_exit = st.checkbox(
        "Enable Exhaustion Extension Top exit",
        value=True,
        help="Kell 본인의 exit signal (윗꼬리 blowoff top). Wick Play의 자연스러운 쌍.",
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
    st.caption(
        "⑤ 진입 당일 close가 당일 range 하단에 마감되면 바로 청산. "
        "PODD 2024-05-08 (close_loc=0.05) 같은 인트라데이 reversal 케이스 대응. "
        "기본 off."
    )
    enable_same_day_reversal = st.checkbox(
        "Enable same-day reversal exit",
        value=False,
    )
    max_same_day_close_location = st.number_input(
        "Max same-day close location (0=low, 1=high)",
        value=0.3, min_value=0.0, max_value=1.0, step=0.05, format="%.2f",
        disabled=not enable_same_day_reversal,
        help="진입 봉 close가 (close-low)/(high-low) < 이 값이면 당일 close 청산. "
        "0.3 = 하단 30% 이내 마감 시 청산.",
    )

    st.subheader("Gap-down rejection (opt-in)")
    enable_gap_down_rejection = st.checkbox(
        "⑨ Reject entry if gap-down vs breakout close",
        value=False,
        help="진입 open이 breakout close 대비 N% 이상 갭다운이면 아예 진입 거부. "
        "PODD 2024 -2.9% 갭다운 케이스 대응. 기본 off.",
    )
    max_entry_gap_down = st.number_input(
        "Max entry gap-down (fraction)",
        value=0.005, min_value=0.0, max_value=0.1, step=0.001, format="%.3f",
        disabled=not enable_gap_down_rejection,
        help="0.005 = -0.5% 이상 갭다운이면 거부. 0.01 = -1% 이상.",
    )

    run_btn = st.button("Run Strategy", type="primary", use_container_width=True)


# --- Main ---
if not run_btn:
    st.info("좌측에서 ticker / 기간 / 파라미터를 설정하고 **Run Strategy**를 눌러줘.")
    st.stop()

market_data = YFinanceAdapter()
fetch_start = start_date - timedelta(days=150)  # EMA + psych vol warmup

with st.spinner(f"Fetching {ticker} {fetch_start} → {end_date}..."):
    try:
        df = market_data.fetch_ohlcv(ticker, fetch_start, end_date)
    except Exception as e:
        st.error(f"Failed to fetch data: {e}")
        st.stop()

if df is None or df.empty:
    st.warning("No data returned.")
    st.stop()

detector = WickPlayDetector(
    min_upper_wick_ratio=float(min_upper_wick_ratio),
    max_volume_dryup=float(max_volume_dryup),
    breakout_trigger=breakout_trigger,
    stop_mode="wick_low",  # per user choice — Kell's "more room" stop
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

strategy = WickPlayStrategy(
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

config = StrategyConfig(
    ticker=ticker,
    start_date=start_date,
    end_date=end_date,
    pattern_name="wick_play",
    initial_capital=float(initial_capital),
    risk_per_trade=risk_pct / 100.0,
    max_holding_days=int(max_holding_days),
)

with st.spinner("Running strategy..."):
    try:
        result = strategy.execute(df, config)
    except Exception as e:
        st.error(f"Strategy failed: {e}")
        st.stop()

perf = result.performance

# --- Metrics ---
st.subheader(f"{ticker} — Wick Play Strategy")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Trades", perf.total_trades)
m2.metric(
    "Win Rate", f"{perf.win_rate:.0%}" if perf.total_trades else "—"
)
m3.metric("Total Return", f"{perf.total_return_pct:.2%}")
m4.metric("Final Capital", f"${perf.final_capital:,.0f}")

m5, m6, m7, m8 = st.columns(4)
m5.metric("Initial Capital", f"${perf.initial_capital:,.0f}")
m6.metric("Avg Win", f"{perf.avg_win_pct:.2%}" if perf.trades else "—")
m7.metric("Avg Loss", f"{perf.avg_loss_pct:.2%}" if perf.trades else "—")
m8.metric("Max Drawdown", f"{perf.max_drawdown_pct:.2%}")

# --- Chart with trades ---
chart_builder = PlotlyChartBuilder()
trade_fig = chart_builder.build_candlestick_with_trades(
    df, perf.trades, title=f"{ticker} — Buy / Sell / Stop"
)
trade_fig.update_xaxes(range=[str(start_date), str(end_date)])
st.plotly_chart(trade_fig, use_container_width=True)

# --- Equity curve ---
if len(result.equity_curve) > 1:
    eq_fig = chart_builder.build_equity_curve(
        result.equity_curve, title="Equity Curve"
    )
    st.plotly_chart(eq_fig, use_container_width=True)

# --- Trades table ---
if not perf.trades:
    st.info("이 기간엔 Wick Play 신호가 발생하지 않았어.")
    st.stop()

EXIT_LABELS = {
    "wick_low_stop": "Wick Low Stop (initial)",
    "same_day_reversal": "Same-Day Reversal (intraday fail)",
    "ema_trail": "10 EMA Trail (Kell stairstep)",
    "exhaustion_exit": "Exhaustion Extension Top",
    "time_stop": "Time Stop",
    "end_of_data": "End of Data (no exit fired)",
}

st.subheader("Trades")
rows = []
for t in perf.trades:
    rows.append(
        {
            "Entry Date": t.entry_date,
            "Exit Date": t.exit_date,
            "Outcome": "WIN" if t.pnl > 0 else "LOSS",
            "Exit Reason": EXIT_LABELS.get(t.exit_reason, t.exit_reason),
            "Entry Price": f"${t.entry_price:,.2f}",
            "Exit Price": f"${t.exit_price:,.2f}",
            "Stop Loss": f"${t.stop_loss:,.2f}",
            "Shares": t.shares,
            "P&L ($)": f"${t.pnl:,.2f}",
            "P&L (%)": f"{t.pnl_pct:.2%}",
        }
    )
st.dataframe(rows, use_container_width=True)
