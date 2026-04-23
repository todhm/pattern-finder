from datetime import date, timedelta

import streamlit as st

from data.adapters.macro_calendar import blackout_dates as build_blackout_dates
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
    "Oliver Kell의 Wick Play 진입 → wick low 손절 / 10 EMA trail / " "Exhaustion Extension Top / time stop"
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
        value=5.0,
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
        help="multi-week 이동을 노리는 setup이라 40일 기본. " "시간 초과 시 다음 봉 open에서 청산.",
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
    enable_max_upper_wick = st.checkbox(
        "Cap upper wick / range (opt-in, 기본 OFF)",
        value=False,
        help="실측 재분석: winner와 loser uw_ratio 겹침 큼. 유의하지 않음. 실험용.",
    )
    max_upper_wick_ratio = st.number_input(
        "Max upper wick / range",
        value=0.55,
        min_value=0.3,
        max_value=1.0,
        step=0.05,
        format="%.2f",
        disabled=not enable_max_upper_wick,
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
    enable_max_wick_range = st.checkbox("Cap wick-bar range (× ATR)", value=True)
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
    min_psych_score = st.slider("Min psych score", 0, 4, 3, 1)
    psych_vol_lookback = st.number_input(
        "Vol avg lookback (days)", value=20, min_value=5, max_value=100, step=5
    )
    psych_wick_vol_exhaustion_mult = st.number_input(
        "Check 1: wick vol ≤ avg × mult",
        value=1.0,
        min_value=0.1,
        max_value=3.0,
        step=0.05,
        format="%.2f",
    )
    psych_breakout_vol_expansion_mult = st.number_input(
        "Check 2: breakout vol > wick vol × mult",
        value=1.0,
        min_value=0.1,
        max_value=3.0,
        step=0.05,
        format="%.2f",
    )
    psych_prior_red_streak = st.number_input(
        "Check 3: prior red streak bars (fail if all red)",
        value=2,
        min_value=0,
        max_value=10,
        step=1,
    )
    psych_dramatic_wick_ratio = st.number_input(
        "Check 4: dramatic wick ratio",
        value=0.65,
        min_value=0.3,
        max_value=1.0,
        step=0.05,
        format="%.2f",
    )

    st.subheader("Hard gates (2024–2026 failure post-mortem)")
    st.caption(
        "11개 실패 트레이드(PODD/LITE/AIZ/CB/PLTR/AEE/HUM/SBAC/ETR/ADM/KHC) " "분석으로 추가된 필터."
    )
    min_breakout_strength_atr_wk = st.number_input(
        "① Min breakout strength (× ATR)",
        value=0.3,
        min_value=0.0,
        max_value=5.0,
        step=0.05,
        format="%.2f",
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
        value=-0.01,
        min_value=-0.5,
        max_value=0.5,
        step=0.005,
        format="%.3f",
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
        value=0.15,
        min_value=0.0,
        max_value=1.0,
        step=0.01,
        format="%.2f",
        disabled=not enable_max_prior_trend,
        help="0.15 = 직전 20일 +15% 초과 종목 거부 (parabolic).",
    )
    min_wick_close_location_wk = st.number_input(
        "⑥ Min wick close location (0=low, 1=high)",
        value=0.15,
        min_value=0.0,
        max_value=1.0,
        step=0.05,
        format="%.2f",
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
        value=0.70,
        min_value=0.0,
        max_value=1.0,
        step=0.05,
        format="%.2f",
        disabled=not enable_min_breakout_cl_wk,
    )
    st.markdown("**Trend Template (opt-in, 기본 OFF)**")
    st.caption(
        "⚠️ Wick Play는 capitulation reversal이라 Trend Template가 winner를 자름. "
        "Minervini-flavored 변형이 필요할 때만 사용."
    )
    require_above_sma200_wk = st.checkbox(
        "⑩ Require close > 200 SMA (primary uptrend)",
        value=False,
    )
    enable_pct_high_wk = st.checkbox(
        "⑪ Require near 52-week high (leadership)",
        value=False,
    )
    min_pct_of_52w_high_wk = st.number_input(
        "⑪ Min close / 52w high",
        value=0.75,
        min_value=0.3,
        max_value=1.0,
        step=0.05,
        format="%.2f",
        disabled=not enable_pct_high_wk,
        help="0.75 = 52주 최고가의 75% 이상 (고점 대비 25% 이내).",
    )

    st.subheader("Macro regime gates")
    st.caption(
        "2026-04 480-config 스윕 결과 반영 (500 Nasdaq 티커, 2024-01 → 2026-04). "
        "상위 config: ema=10 / mtb=5 / regime=sma50 / macro=cpi±1 → +4.44% net, "
        "baseline(off)은 -0.52%. 기본 ON으로 전환."
    )
    enable_regime_filter = st.checkbox(
        "⑫ Enable index regime filter (SPY)",
        value=True,
        help="SPY OHLCV를 받아 신호 봉에서 regime 조건 검사. "
        "엄격 가운데 하나라도 실패하면 해당 신호 거부.",
    )
    regime_symbol = st.text_input(
        "Regime symbol",
        value="^GSPC",
        disabled=not enable_regime_filter,
        help="SPY / QQQ / ^GSPC / ^NDX 등.",
    )
    regime_sma = st.number_input(
        "Require SPY close > SMA period (0 = off)",
        value=50,
        min_value=0,
        max_value=400,
        step=10,
        disabled=not enable_regime_filter,
        help="스윕 결과 sma20/sma50/dd5_2pct 동률. sma50이 가장 단순한 단일 필터.",
    )
    regime_ema = st.number_input(
        "Require SPY close > EMA period (0 = off)",
        value=0,
        min_value=0,
        max_value=200,
        step=5,
        disabled=not enable_regime_filter,
    )
    regime_dd_n = st.number_input(
        "Max N-day drawdown period (0 = off)",
        value=0,
        min_value=0,
        max_value=60,
        step=1,
        disabled=not enable_regime_filter,
        help="SPY N일 수익률이 -max_dd보다 나쁘면 거부. " "sma50과 대체로 중복되므로 0(off)이 기본.",
    )
    regime_dd_max = st.number_input(
        "Max N-day drawdown (magnitude, fraction)",
        value=0.02,
        min_value=0.0,
        max_value=0.5,
        step=0.005,
        format="%.3f",
        disabled=not enable_regime_filter,
        help="0.02 = -2%.",
    )

    enable_macro_blackout = st.checkbox(
        "⑬ Enable macro-event blackout",
        value=True,
        help="FOMC / CPI 발표일 근처 신호·entry를 거부. "
        "스윕 결과 CPI ±1일이 가장 유효 (FOMC 단독보다 우수).",
    )
    include_fomc_blackout = st.checkbox(
        "Include FOMC announcement days",
        value=False,
        disabled=not enable_macro_blackout,
        help="스윕 기간엔 CPI 블랙아웃보다 약간 약했지만 SHOP 2024-12-18 같은 "
        "단일 이벤트 방어에는 여전히 유효.",
    )
    include_cpi_blackout = st.checkbox(
        "Include CPI release days",
        value=True,
        disabled=not enable_macro_blackout,
        help="스윕 상위 모든 config의 macro 선택. ON 권장.",
    )
    blackout_window_days = st.number_input(
        "Blackout window (± days)",
        value=1,
        min_value=0,
        max_value=3,
        step=1,
        disabled=not enable_macro_blackout,
        help="0 = 이벤트 당일만. 1 = ±1일 포함 — 스윕 최적값.",
    )

    st.header("Exits")
    st.caption(
        "**Stop** = wick_low (Kell 여유 stop). "
        "**Trail** = 10 EMA — close가 EMA 아래로 떨어지면 그 봉 close에 청산."
    )
    ema_trail = st.number_input(
        "EMA trail period",
        value=10,
        min_value=3,
        max_value=50,
        step=1,
        help="Kell/Minervini stairstep. 10=기본 (Wick Play 원래 세팅).",
    )
    min_trail_bars = st.number_input(
        "Min bars before trail/exhaustion fires",
        value=5,
        min_value=0,
        max_value=20,
        step=1,
        help="진입 직후 N봉은 trail/exhaustion exit 비활성 — 돌파 봉 소화 여유. "
        "2026-04 스윕: 5가 2보다 top-config 기준 우수 (+4.44% vs +1.58%).",
    )
    enable_exh_exit = st.checkbox(
        "Enable Exhaustion Extension Top exit",
        value=True,
        help="Kell 본인의 exit signal (윗꼬리 blowoff top). Wick Play의 자연스러운 쌍.",
    )
    exh_extension_atr = st.number_input(
        "Exh extension above EMA (× ATR)",
        value=2.2,
        min_value=0.5,
        max_value=10.0,
        step=0.1,
        disabled=not enable_exh_exit,
        help="Exit-sweep 2026-04 (S&P 500 2017-2025): 2.2가 sweet spot. "
        "2.5는 exhaustion 기회 놓침 (9년 5건), 1.5는 winner 일찍 잘라냄. "
        "2.2에서 exhaustion 13건, PF 1.67 (ext=2.5 대비 +13%p 수익).",
    )
    exh_min_slope = st.number_input(
        "Exh min slow-EMA slope",
        value=0.005,
        min_value=-1.0,
        max_value=1.0,
        step=0.001,
        format="%.4f",
        disabled=not enable_exh_exit,
    )

    st.subheader("Breakeven stop")
    st.caption(
        "Exit-sweep 2026-04 결과: arm=2.0 / offset=1.0 조합이 최적 "
        "(+2R 도달 winner가 reverse 시 +1R 잠금). "
        "arm=1.0 (pure BE)은 오히려 winner를 잘라 win_rate 45%→40% 하락."
    )
    enable_breakeven = st.checkbox(
        "Enable breakeven / partial-profit stop",
        value=True,
    )
    breakeven_arm_r = st.number_input(
        "Arm at +R multiple",
        value=2.0,
        min_value=0.3,
        max_value=5.0,
        step=0.1,
        format="%.2f",
        disabled=not enable_breakeven,
        help="+2R 도달해야 stop을 +1R로 이동. 1.0으로 내리면 너무 빨리 트리거 → winner 잘림.",
    )
    breakeven_offset_r = st.number_input(
        "Exit offset (× R)",
        value=1.0,
        min_value=-0.5,
        max_value=2.0,
        step=0.05,
        format="%.2f",
        disabled=not enable_breakeven,
        help="0 = 순수 breakeven. 1.0 = arm=2.0과 쌍 — 최소 +1R lock-in.",
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
        value=0.3,
        min_value=0.0,
        max_value=1.0,
        step=0.05,
        format="%.2f",
        disabled=not enable_same_day_reversal,
        help="진입 봉 close가 (close-low)/(high-low) < 이 값이면 당일 close 청산. "
        "0.3 = 하단 30% 이내 마감 시 청산.",
    )

    st.subheader("Gap-down rejection")
    enable_gap_down_rejection = st.checkbox(
        "⑨ Reject entry if gap-down vs breakout close",
        value=True,
        help="진입 open이 breakout close 대비 N% 이상 갭다운이면 아예 진입 거부. "
        "PODD 2024 -2.9% 갭다운 케이스 대응. 기본 ON.",
    )
    max_entry_gap_down = st.number_input(
        "Max entry gap-down (fraction)",
        value=0.005,
        min_value=0.0,
        max_value=0.1,
        step=0.001,
        format="%.3f",
        disabled=not enable_gap_down_rejection,
        help="0.005 = -0.5% 이상 갭다운이면 거부. 0.01 = -1% 이상.",
    )

    run_btn = st.button("Run Strategy", type="primary", use_container_width=True)


# --- Main ---
if not run_btn:
    st.info("좌측에서 ticker / 기간 / 파라미터를 설정하고 **Run Strategy**를 눌러줘.")
    st.stop()

market_data = YFinanceAdapter()
fetch_start = start_date - timedelta(days=400)  # EMA + psych vol + SMA200 warmup

with st.spinner(f"Fetching {ticker} {fetch_start} → {end_date}..."):
    try:
        df = market_data.fetch_ohlcv(ticker, fetch_start, end_date)
    except Exception as e:
        st.error(f"Failed to fetch data: {e}")
        st.stop()

if df is None or df.empty:
    st.warning("No data returned.")
    st.stop()

# Fetch regime benchmark if enabled. Uses the same cached path so
# subsequent runs across pages reuse the parquet on disk.
regime_df = None
if enable_regime_filter:
    with st.spinner(f"Fetching regime symbol {regime_symbol}..."):
        try:
            regime_df = market_data.fetch_ohlcv(regime_symbol, fetch_start, end_date)
        except Exception as e:
            st.error(f"Failed to fetch regime symbol: {e}")
            st.stop()
    if regime_df is None or regime_df.empty:
        st.warning(f"No data for regime symbol {regime_symbol} — disabling filter.")
        regime_df = None

blackout_set = None
if enable_macro_blackout and (include_fomc_blackout or include_cpi_blackout):
    blackout_set = build_blackout_dates(
        start=fetch_start,
        end=end_date,
        include_fomc=include_fomc_blackout,
        include_cpi=include_cpi_blackout,
        window_days=int(blackout_window_days),
    )

detector = WickPlayDetector(
    min_upper_wick_ratio=float(min_upper_wick_ratio),
    max_upper_wick_ratio=(float(max_upper_wick_ratio) if enable_max_upper_wick else None),
    max_volume_dryup=float(max_volume_dryup),
    breakout_trigger=breakout_trigger,
    stop_mode="wick_low",  # per user choice — Kell's "more room" stop
    max_wick_range_atr=(float(max_wick_range_atr) if enable_max_wick_range else None),
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
    require_above_sma200=bool(require_above_sma200_wk),
    min_pct_of_52w_high=(float(min_pct_of_52w_high_wk) if enable_pct_high_wk else None),
    regime_df=regime_df,
    regime_min_above_sma=(int(regime_sma) if enable_regime_filter and regime_sma > 0 else None),
    regime_min_above_ema=(int(regime_ema) if enable_regime_filter and regime_ema > 0 else None),
    regime_max_n_day_drawdown=(
        (int(regime_dd_n), float(regime_dd_max)) if enable_regime_filter and regime_dd_n > 0 else None
    ),
    blackout_dates=blackout_set,
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
    enable_breakeven_stop=enable_breakeven,
    breakeven_arm_r_multiple=float(breakeven_arm_r),
    breakeven_exit_offset_r=float(breakeven_offset_r),
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
m2.metric("Win Rate", f"{perf.win_rate:.0%}" if perf.total_trades else "—")
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
    eq_fig = chart_builder.build_equity_curve(result.equity_curve, title="Equity Curve")
    st.plotly_chart(eq_fig, use_container_width=True)

# --- Trades table ---
if not perf.trades:
    st.info("이 기간엔 Wick Play 신호가 발생하지 않았어.")
    st.stop()

EXIT_LABELS = {
    "wick_low_stop": "Wick Low Stop (initial)",
    "same_day_reversal": "Same-Day Reversal (intraday fail)",
    "breakeven_stop": "Breakeven Stop (post-1R pullback)",
    "ema_trail": "EMA Trail (Kell stairstep)",
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
