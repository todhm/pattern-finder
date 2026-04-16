from datetime import date, timedelta

import streamlit as st

from data.adapters.yfinance_adapter import YFinanceAdapter
from pattern.adapters.base_n_break_downside import BaseNBreakDownsideDetector
from pattern.adapters.ema_crossback_downside import EmaCrossbackDownsideDetector
from pattern.adapters.exhaustion_extension_top import ExhaustionExtensionTopDetector
from pattern.adapters.reversal_extension import ReversalExtensionDetector
from pattern.adapters.wedge_drop import WedgeDropDetector
from pattern.adapters.wedge_pop import WedgePopDetector
from visualization.adapters.plotly_charts import PlotlyChartBuilder

st.set_page_config(page_title="Pattern Detection", layout="wide")
st.title("Pattern Detection")

PATTERN_OPTIONS = [
    "wedge_pop",
    "reversal_extension",
    "exhaustion_extension_top",
    "wedge_drop",
    "ema_crossback_downside",
    "base_n_break_downside",
]

# --- Sidebar inputs ---
with st.sidebar:
    st.header("Settings")
    ticker = st.text_input("Ticker", value="AAPL")
    start_date = st.date_input("Start Date", value=date.today() - timedelta(days=365))
    end_date = st.date_input("End Date", value=date.today())
    pattern_name = st.selectbox("Pattern", options=PATTERN_OPTIONS)

    # Shared EMA/ATR controls used by most detectors
    st.header("Common (EMA / ATR)")
    ema_fast = st.number_input("Fast EMA period", value=10, min_value=2, max_value=100, step=1)
    ema_slow = st.number_input("Slow EMA period", value=20, min_value=2, max_value=200, step=1)
    atr_period = st.number_input("ATR period", value=14, min_value=2, max_value=100, step=1)
    slope_lookback = st.number_input(
        "Slope lookback (days)",
        value=20,
        min_value=5,
        max_value=120,
        step=1,
        help="ema_fast_slope / ema_slow_slope 측정 기간.",
    )

    # ------------------------------------------------------------
    # Per-pattern parameter groups — each group is only rendered
    # when the corresponding pattern is selected above.
    # ------------------------------------------------------------

    if pattern_name == "wedge_pop":
        st.header("Wedge Pop")
        detect_lookback = st.number_input(
            "Consolidation lookback (days)",
            value=10,
            min_value=3,
            max_value=60,
            step=1,
        )
        cooldown_bars_ui = st.number_input(
            "Cooldown bars",
            value=15,
            min_value=0,
            max_value=60,
            step=1,
            help="signal 이후 건너뛰는 바 수. 0 = 연속 허용.",
        )
        consolidation_pct = st.number_input(
            "Min consolidation %",
            value=60.0,
            min_value=0.0,
            max_value=100.0,
            step=5.0,
            help="직전 lookback 중 close < fast EMA 요구 최소 비율.",
        )
        enable_max_cp = st.checkbox("Cap max consolidation %", value=False)
        max_consolidation_pct_ui = st.number_input(
            "Max consolidation %",
            value=95.0,
            min_value=0.0,
            max_value=100.0,
            step=5.0,
            disabled=not enable_max_cp,
        )
        breakout_atr_mult = st.number_input(
            "Min breakout strength (× ATR)",
            value=0.01,
            min_value=0.0,
            max_value=10.0,
            step=0.001,
            format="%.3f",
        )
        enable_max_bp = st.checkbox("Cap max breakout strength", value=False)
        max_breakout_atr_mult_ui = st.number_input(
            "Max breakout strength (× ATR)",
            value=3.0,
            min_value=0.0,
            max_value=20.0,
            step=0.5,
            format="%.2f",
            disabled=not enable_max_bp,
        )
        require_above_long_smas = st.checkbox(
            "Require close above 50 & 200 SMA",
            value=True,
        )

    elif pattern_name == "exhaustion_extension_top":
        st.header("Exhaustion Extension (Top)")
        st.caption("상승 추세 꼭지에서 10 EMA 위로 과도하게 벌어지는 블로우오프.")
        extension_atr_mult_top = st.number_input(
            "Min extension above 10 EMA (× ATR)",
            value=1.9,
            min_value=0.5,
            max_value=20.0,
            step=0.01,
            help="high가 fast EMA 위로 ATR의 몇 배 이상이어야 하는지.",
        )
        min_slow_slope_top = st.number_input(
            "Min slow EMA slope",
            value=0.05,
            min_value=-1.0,
            max_value=1.0,
            step=0.01,
            format="%.3f",
            help="상승 추세 확인용. 20일간 slow EMA가 +N% 이상 올라야 함.",
        )
        st.caption(
            "**Distribution 확증 신호** — 기본적으로 **OR**로 결합돼 "
            "둘 중 하나라도 통과하면 시그널 발동. 매도가 매수를 압도함을 "
            "반드시 확인하려면 'Require ALL'을 체크하고 sell dominance를 "
            "켜두세요."
        )
        min_sell_dominance_top = st.number_input(
            "Min sell dominance (down_vol / up_vol) — 0=off",
            value=1.5,
            min_value=0.0,
            max_value=10.0,
            step=0.1,
            help="직전 N봉 구간 안에서 "
            "음봉(close<open) 거래량 합 / 양봉(close>=open) 거래량 합 "
            "비율이 이 값 이상이어야 통과. 같은 윈도우 안에서 매도 vs "
            "매수를 직접 비교하므로 '매도가 매수를 압도한다'를 엄격하게 "
            "확인. 예: 1.5 = 매도 거래량이 매수 거래량의 1.5배 이상. "
            "0으로 두면 off.",
        )
        pressure_lookback_top = st.number_input(
            "Pressure lookback (bars)",
            value=10,
            min_value=3,
            max_value=60,
            step=1,
            help="sell dominance를 측정할 직전 봉 개수.",
            disabled=min_sell_dominance_top <= 0.0,
        )
        max_close_location_top = st.number_input(
            "Max close location (0=low, 1=high) — 1.0=off",
            value=0.5,
            min_value=0.0,
            max_value=1.0,
            step=0.05,
            format="%.2f",
            help="close가 (close-low)/(high-low) 이 값 이하. "
            "shooting star / bearish candle 확증. 1.0으로 두면 off.",
        )
        require_all_confirmations_top = st.checkbox(
            "Require ALL confirmations (AND instead of OR)",
            value=False,
            help="체크 시 활성화된 모든 확증 조건이 동시에 통과해야 함. "
            "기본(off)은 OR — 하나만 통과해도 시그널 발동.",
        )
        enable_rejection_override_top = st.checkbox(
            "Upper-wick rejection override",
            value=True,
            help="close_location ≤ 0.25 (강한 샹팅스타 거절 캔들)이면 "
            "sell_dominance 확증과 cooldown을 건너뛰고 신호 발동. "
            "긴 윗꼬리 자체가 고점 매도 증거라 별도 거래량 확증이 중복됨. "
            "이 경로에서는 extension/slope가 기본 임계값의 90% 이상이면 통과.",
        )
        cooldown_bars_top = st.number_input(
            "Cooldown bars",
            value=10,
            min_value=0,
            max_value=60,
            step=1,
        )

    elif pattern_name == "wedge_drop":
        st.header("Wedge Drop")
        st.caption("Wedge Pop의 거울 — 밀집 후 양 EMA 아래로 이탈하며 꼭지 확정.")
        drop_lookback = st.number_input(
            "Consolidation lookback (days)",
            value=10,
            min_value=3,
            max_value=60,
            step=1,
        )
        drop_consolidation_pct = st.number_input(
            "Min consolidation %",
            value=60.0,
            min_value=0.0,
            max_value=100.0,
            step=5.0,
            help="직전 lookback 중 close > fast EMA 요구 최소 비율.",
        )
        breakdown_atr_mult_drop = st.number_input(
            "Min breakdown strength (× ATR)",
            value=0.5,
            min_value=0.0,
            max_value=10.0,
            step=0.1,
        )
        cooldown_bars_drop = st.number_input(
            "Cooldown bars",
            value=10,
            min_value=0,
            max_value=60,
            step=1,
        )

    elif pattern_name == "ema_crossback_downside":
        st.header("EMA Crossback (Downside)")
        st.caption("하락 추세 내 EMA로의 반등이 실패하는 지점.")
        max_slow_slope_cb = st.number_input(
            "Max slow EMA slope",
            value=-0.02,
            min_value=-1.0,
            max_value=1.0,
            step=0.01,
            format="%.3f",
            help="하락 추세 확인용. slow EMA 기울기가 이 값 이하여야 함. " "-0.02 = 20일간 -2% 이상 하락.",
        )
        prior_below_bars_cb = st.number_input(
            "Prior below bars",
            value=5,
            min_value=2,
            max_value=30,
            step=1,
            help="직전 N 봉 동안 fast EMA 아래에 있었는지 체크.",
        )
        prior_below_pct_cb = st.number_input(
            "Prior below pct",
            value=0.6,
            min_value=0.0,
            max_value=1.0,
            step=0.05,
            format="%.2f",
            help="직전 N 봉 중 몇 %가 fast EMA 아래여야 하는지.",
        )
        cooldown_bars_cb = st.number_input(
            "Cooldown bars",
            value=5,
            min_value=0,
            max_value=60,
            step=1,
        )

    elif pattern_name == "base_n_break_downside":
        st.header("Base N Break (Downside)")
        st.caption("하락 추세 내 타이트 밀집 → 하방 이탈 지속 신호.")
        bnb_lookback = st.number_input(
            "Consolidation lookback (days)",
            value=10,
            min_value=3,
            max_value=60,
            step=1,
        )
        bnb_below_pct = st.number_input(
            "Min consolidation below %",
            value=0.7,
            min_value=0.0,
            max_value=1.0,
            step=0.05,
            format="%.2f",
            help="밀집 구간 중 close < fast EMA 비율 최소값.",
        )
        bnb_max_range_atr = st.number_input(
            "Max range (× ATR)",
            value=2.5,
            min_value=0.5,
            max_value=10.0,
            step=0.1,
            help="밀집 구간 total range가 ATR의 이 배수 이하여야 함 (변동성 수축).",
        )
        bnb_breakdown_atr_mult = st.number_input(
            "Min breakdown strength (× ATR)",
            value=0.3,
            min_value=0.0,
            max_value=10.0,
            step=0.1,
        )
        bnb_max_slow_slope = st.number_input(
            "Max slow EMA slope",
            value=-0.02,
            min_value=-1.0,
            max_value=1.0,
            step=0.01,
            format="%.3f",
        )
        cooldown_bars_bnb = st.number_input(
            "Cooldown bars",
            value=10,
            min_value=0,
            max_value=60,
            step=1,
        )

    run_btn = st.button("Detect", type="primary", use_container_width=True)

# --- Main area ---
if run_btn:
    # Fetch extra history so 50/200 SMA are converged from day 1.
    fetch_start = start_date - timedelta(days=400)
    with st.spinner("Fetching data & detecting patterns..."):
        try:
            adapter = YFinanceAdapter()
            df = adapter.fetch_ohlcv(ticker, fetch_start, end_date)

            if pattern_name == "wedge_pop":
                detector = WedgePopDetector(
                    lookback=int(detect_lookback),
                    ema_fast=int(ema_fast),
                    ema_slow=int(ema_slow),
                    consolidation_pct=consolidation_pct / 100.0,
                    max_consolidation_pct=(max_consolidation_pct_ui / 100.0 if enable_max_cp else None),
                    breakout_atr_mult=breakout_atr_mult,
                    max_breakout_atr_mult=(max_breakout_atr_mult_ui if enable_max_bp else None),
                    atr_period=int(atr_period),
                    slope_lookback=int(slope_lookback),
                    cooldown_bars=int(cooldown_bars_ui),
                    require_above_long_smas=require_above_long_smas,
                )
            elif pattern_name == "reversal_extension":
                detector = ReversalExtensionDetector()
            elif pattern_name == "exhaustion_extension_top":
                detector = ExhaustionExtensionTopDetector(
                    extension_atr_mult=extension_atr_mult_top,
                    min_slow_slope=min_slow_slope_top,
                    max_close_location=max_close_location_top,
                    min_sell_dominance=min_sell_dominance_top,
                    pressure_lookback=int(pressure_lookback_top),
                    require_all_confirmations=require_all_confirmations_top,
                    enable_rejection_override=enable_rejection_override_top,
                    ema_fast=int(ema_fast),
                    ema_slow=int(ema_slow),
                    atr_period=int(atr_period),
                    slope_lookback=int(slope_lookback),
                    cooldown_bars=int(cooldown_bars_top),
                )
            elif pattern_name == "wedge_drop":
                detector = WedgeDropDetector(
                    lookback=int(drop_lookback),
                    consolidation_pct=drop_consolidation_pct / 100.0,
                    breakdown_atr_mult=breakdown_atr_mult_drop,
                    ema_fast=int(ema_fast),
                    ema_slow=int(ema_slow),
                    atr_period=int(atr_period),
                    cooldown_bars=int(cooldown_bars_drop),
                )
            elif pattern_name == "ema_crossback_downside":
                detector = EmaCrossbackDownsideDetector(
                    max_slow_slope=max_slow_slope_cb,
                    prior_below_bars=int(prior_below_bars_cb),
                    prior_below_pct=prior_below_pct_cb,
                    ema_fast=int(ema_fast),
                    ema_slow=int(ema_slow),
                    atr_period=int(atr_period),
                    slope_lookback=int(slope_lookback),
                    cooldown_bars=int(cooldown_bars_cb),
                )
            elif pattern_name == "base_n_break_downside":
                detector = BaseNBreakDownsideDetector(
                    lookback=int(bnb_lookback),
                    consolidation_below_pct=bnb_below_pct,
                    max_range_atr=bnb_max_range_atr,
                    breakdown_atr_mult=bnb_breakdown_atr_mult,
                    max_slow_slope=bnb_max_slow_slope,
                    ema_fast=int(ema_fast),
                    ema_slow=int(ema_slow),
                    atr_period=int(atr_period),
                    slope_lookback=int(slope_lookback),
                    cooldown_bars=int(cooldown_bars_bnb),
                )
            else:
                st.error(f"Unknown pattern: {pattern_name}")
                st.stop()

            signals = detector.detect(df)
        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()

    st.subheader(f"{ticker} — {pattern_name} ({len(signals)} signals)")

    chart_builder = PlotlyChartBuilder()
    fig = chart_builder.build_candlestick_with_signals(df, signals, title=f"{ticker} — {pattern_name}")
    fig.update_xaxes(range=[str(start_date), str(end_date)])
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
