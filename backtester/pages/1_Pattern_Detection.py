from datetime import date, timedelta

import plotly.graph_objects as go
import streamlit as st

from data.adapters.yfinance_adapter import YFinanceAdapter
from pattern.adapters.base_n_break_downside import BaseNBreakDownsideDetector
from pattern.adapters.ema_crossback_downside import EmaCrossbackDownsideDetector
from pattern.adapters.exhaustion_extension_top import ExhaustionExtensionTopDetector
from pattern.adapters.reversal_extension import ReversalExtensionDetector
from pattern.adapters.wedge_drop import WedgeDropDetector
from pattern.adapters.wedge_pop import WedgePopDetector
from pattern.adapters.wick_play import WickPlayDetector
from visualization.adapters.plotly_charts import PlotlyChartBuilder

st.set_page_config(page_title="Pattern Detection", layout="wide")
st.title("Pattern Detection")

PATTERN_OPTIONS = [
    "wedge_pop",
    "wick_play",
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
        value=10,
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
            value=30.0,
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
            value=0.005,
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
        late_entry_bars_wp = st.number_input(
            "Late-entry bars (0 = strict fresh-cross)",
            value=0,
            min_value=0,
            max_value=10,
            step=1,
            help="기본 0: 전일이 반드시 EMA 아래여야 함 (엄격한 첫 돌파). "
            "N>0: 돌파가 N봉 이내에 일어났으면 continuation 봉도 "
            "wedge pop으로 인정. 예: 2 = 첫 돌파일 포함 최대 3봉까지 "
            "'여진' 봉 catch. metadata trigger 필드로 primary/late_entry "
            "구분됨.",
        )
        st.caption(
            "**EMA slow slope 범위 필터** — detector가 metadata로 내보내는 "
            "`ema_slow_slope`를 post-filter함. strategy 페이지의 slope 범위 "
            "필터와 동일한 로직."
        )
        enable_min_slope_wp = st.checkbox(
            "Enforce min EMA slow slope",
            value=False,
            help="ema_slow_slope의 **하한**. 양수로 설정하면 '반드시 +N% "
            "이상 올라가는 추세'만 통과 (예: 0.005 → slope_lookback 동안 "
            "+0.5% 이상). 음수로 설정하면 dead-cat bounce를 걸러냄.",
        )
        min_ema_slow_slope_wp = st.number_input(
            "Min EMA slow slope",
            value=-0.01,
            min_value=-1.0,
            max_value=1.0,
            step=0.001,
            format="%.4f",
            disabled=not enable_min_slope_wp,
        )
        enable_max_slope_wp = st.checkbox(
            "Enforce max EMA slow slope",
            value=False,
            help="ema_slow_slope의 **상한**. 이미 너무 가파르게 오른 "
            "(parabolic) 종목 제외. 예: 0.30 → slope_lookback 동안 +30% "
            "이상 오른 종목 제외.",
        )
        max_ema_slow_slope_wp = st.number_input(
            "Max EMA slow slope",
            value=0.30,
            min_value=-1.0,
            max_value=5.0,
            step=0.01,
            format="%.3f",
            disabled=not enable_max_slope_wp,
        )

    elif pattern_name == "wick_play":
        st.header("Wick Play")
        st.caption(
            "Oliver Kell의 3봉 리버설 셋업 — 윗꼬리 봉 → 거래량 말라붙은 "
            "inside bar → wick high 돌파. "
            "sellers → stalemate → buyers 전환을 micro 타임프레임에 압축."
        )
        min_upper_wick_ratio_wp_wk = st.number_input(
            "Min upper wick / range",
            value=0.5,
            min_value=0.0,
            max_value=1.0,
            step=0.05,
            format="%.2f",
            help="Wick 봉의 `(high - max(open,close)) / (high - low)` "
            "최소값. 0.5 = 윗꼬리가 당일 range의 50% 이상 차지해야 함.",
        )
        max_volume_dryup_wk = st.number_input(
            "Max inside-bar volume vs wick bar",
            value=1.0,
            min_value=0.1,
            max_value=2.0,
            step=0.05,
            format="%.2f",
            help="Inside bar 거래량 / wick bar 거래량 상한. "
            "1.0 = 넘지만 않으면 OK, 0.7 = 엄격한 dry-up (30%+ 감소 요구).",
        )
        breakout_trigger_wk = st.selectbox(
            "Breakout trigger",
            options=["wick_high", "inside_high"],
            index=0,
            help="wick_high = 보수적 (Bar i-2 고점 돌파 요구). "
            "inside_high = 공격적 (Bar i-1 고점만 돌파하면 OK, 더 타이트한 stop).",
        )
        stop_mode_wk = st.selectbox(
            "Stop placement",
            options=["inside_low", "wick_low"],
            index=0,
            help="inside_low = Bar i-1 저점 (타이트). " "wick_low = Bar i-2 저점 (여유).",
        )
        enable_max_wick_range_wk = st.checkbox(
            "Cap wick-bar range (× ATR)",
            value=True,
            help="Oliver Kell: 'wick이 너무 크면 리스크가 커진다'. "
            "Wick 봉의 total range가 ATR의 N배를 넘으면 거부.",
        )
        max_wick_range_atr_wk = st.number_input(
            "Max wick-bar range (× ATR)",
            value=2.5,
            min_value=0.5,
            max_value=10.0,
            step=0.1,
            format="%.2f",
            disabled=not enable_max_wick_range_wk,
        )
        cooldown_bars_wk = st.number_input(
            "Cooldown bars",
            value=5,
            min_value=0,
            max_value=60,
            step=1,
            help="신호 발동 후 건너뛸 바 개수.",
        )

        st.subheader("Psychology score (4 checks)")
        st.caption(
            "Wick Play의 심리 3단계(sellers 탈진 → stalemate → buyers 장악)가 "
            "실제 tape에서 뒷받침되는지 확인하는 4개 필터. "
            "NVDA 2021-07-12처럼 **형태는 맞는데 심리가 비어있는** 셋업을 거름. "
            "4개 중 `min_psych_score` 이상 통과해야 신호 발동 (기본 3 of 4)."
        )
        min_psych_score_wk = st.slider(
            "Min psych score (0 = off)",
            min_value=0,
            max_value=4,
            value=3,
            step=1,
            help="0 = 심리 필터 off (구조만 체크). "
            "3 = 4개 중 3개 이상 통과 요구 (기본, 권장). "
            "4 = 모두 통과 (엄격, 샘플 매우 적어짐).",
        )
        psych_vol_lookback_wk = st.number_input(
            "Vol avg lookback (days)",
            value=20,
            min_value=5,
            max_value=100,
            step=5,
            help="Check 1(wick-vol exhaustion)에 쓰는 이동평균 거래량 기간.",
        )
        psych_wick_vol_exhaustion_mult_wk = st.number_input(
            "Check 1: wick vol ≤ avg × mult",
            value=1.0,
            min_value=0.1,
            max_value=3.0,
            step=0.05,
            format="%.2f",
            help="Wick bar 거래량이 ≤ (N-day 평균 × mult) 이어야 통과. "
            "1.0 = 평균 이하 (탈진 시나리오). 거래량이 평균을 넘으면 "
            "매도자가 힘을 다 쓴 게 아니라 **분산** 중 → fail.",
        )
        psych_breakout_vol_expansion_mult_wk = st.number_input(
            "Check 2: breakout vol > wick vol × mult",
            value=1.0,
            min_value=0.1,
            max_value=3.0,
            step=0.05,
            format="%.2f",
            help="Breakout bar 거래량이 > (wick bar 거래량 × mult) 이어야 통과. "
            "1.0 = wick bar 대비 *더 많이* 붙어야 (매수자 우세 거래량 확증). "
            "1.5로 올리면 더 강한 확증 요구.",
        )
        psych_prior_red_streak_wk = st.number_input(
            "Check 3: prior red streak bars (fail if all red)",
            value=2,
            min_value=0,
            max_value=10,
            step=1,
            help="Wick bar 직전 N봉이 **전부** 음봉(close<open)이면 fail. "
            "2 = 직전 2봉 연속 음봉이면 거절 (기본). 0 = 체크 off. "
            "하락 추세 중간에 찍힌 wick을 거름.",
        )
        psych_dramatic_wick_ratio_wk = st.number_input(
            "Check 4: dramatic wick ratio",
            value=0.65,
            min_value=0.3,
            max_value=1.0,
            step=0.05,
            format="%.2f",
            help="upper_wick_ratio가 이 값 이상이면 통과. "
            "0.5는 구조 하한이고, 0.65+가 Kell의 베스트 예시 수준 "
            "('극적인 거부 캔들'). 0.53 같은 마진널 케이스를 거름.",
        )

        st.subheader("Hard gates (failure post-mortem)")
        st.caption(
            "실패 11건(PODD/LITE/AIZ/CB/PLTR/AEE/HUM/SBAC/ETR/ADM/KHC) "
            "분석으로 추가된 필터."
        )
        min_breakout_strength_atr_pd = st.number_input(
            "① Min breakout strength (× ATR)",
            value=0.3,
            min_value=0.0,
            max_value=5.0,
            step=0.05,
            format="%.2f",
            help="돌파 close가 trigger level을 ATR의 N배 이상 넘어야 함. "
            "0.0 = 예전 동작. 0.3 = 기본 (약한 돌파 차단).",
        )
        enable_min_prior_trend_pd = st.checkbox(
            "② Enforce min 20-day prior trend",
            value=True,
            help="하락 추세 중간의 wick 거부.",
        )
        min_prior_trend_20d_pd = st.number_input(
            "② Min 20-day trend (fraction)",
            value=-0.03,
            min_value=-0.5,
            max_value=0.5,
            step=0.005,
            format="%.3f",
            disabled=not enable_min_prior_trend_pd,
        )
        enable_max_prior_trend_pd = st.checkbox(
            "③ Enforce max 20-day prior trend (parabolic cap)",
            value=False,
            help="이미 너무 오른 종목 제외 (PLTR +21% 케이스).",
        )
        max_prior_trend_20d_pd = st.number_input(
            "③ Max 20-day trend (fraction)",
            value=0.15,
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            format="%.2f",
            disabled=not enable_max_prior_trend_pd,
        )
        min_wick_close_location_pd = st.number_input(
            "⑥ Min wick close location (0=low, 1=high)",
            value=0.15,
            min_value=0.0,
            max_value=1.0,
            step=0.05,
            format="%.2f",
            help="Wick bar close가 range 하단 N% 이상이어야 통과. "
            "0.15 = 하단 15% 미만 마감은 'outright bearish' 봉으로 보고 거부. "
            "ZTS/BG/PODD 2024 실패 패턴(모두 wick CL < 0.05)을 차단.",
        )
        enable_min_breakout_cl_pd = st.checkbox(
            "⑧ Enforce min breakout close location",
            value=False,
            help="돌파 봉 close가 range 상단 N% 이상에 마감해야. "
            "약한 돌파 마감(ZTS 0.61) 차단.",
        )
        min_breakout_close_location_pd = st.number_input(
            "⑧ Min breakout close location",
            value=0.70,
            min_value=0.0,
            max_value=1.0,
            step=0.05,
            format="%.2f",
            disabled=not enable_min_breakout_cl_pd,
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
            value=0.005,
            min_value=-1.0,
            max_value=1.0,
            step=0.001,
            format="%.4f",
            help="상승 추세 확인용. slope_lookback 바 동안 slow EMA가 "
            "+N% 이상 움직여야 함. 0.005 = 10봉 +0.5% (연 ~13% 속도). "
            "0.05 = 10봉 +5% (연 ~340%, 파라볼릭).",
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
            value=30.0,
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
                    late_entry_bars=int(late_entry_bars_wp),
                )
            elif pattern_name == "wick_play":
                detector = WickPlayDetector(
                    min_upper_wick_ratio=float(min_upper_wick_ratio_wp_wk),
                    max_volume_dryup=float(max_volume_dryup_wk),
                    breakout_trigger=breakout_trigger_wk,
                    stop_mode=stop_mode_wk,
                    max_wick_range_atr=(float(max_wick_range_atr_wk) if enable_max_wick_range_wk else None),
                    atr_period=int(atr_period),
                    cooldown_bars=int(cooldown_bars_wk),
                    psych_vol_lookback=int(psych_vol_lookback_wk),
                    psych_wick_vol_exhaustion_mult=float(psych_wick_vol_exhaustion_mult_wk),
                    psych_breakout_vol_expansion_mult=float(psych_breakout_vol_expansion_mult_wk),
                    psych_prior_red_streak=int(psych_prior_red_streak_wk),
                    psych_dramatic_wick_ratio=float(psych_dramatic_wick_ratio_wk),
                    min_psych_score=int(min_psych_score_wk),
                    min_breakout_strength_atr=float(min_breakout_strength_atr_pd),
                    min_prior_trend_20d=(float(min_prior_trend_20d_pd) if enable_min_prior_trend_pd else None),
                    max_prior_trend_20d=(float(max_prior_trend_20d_pd) if enable_max_prior_trend_pd else None),
                    min_wick_close_location=float(min_wick_close_location_pd),
                    min_breakout_close_location=(
                        float(min_breakout_close_location_pd)
                        if enable_min_breakout_cl_pd else None
                    ),
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

            # Post-filter: apply wedge_pop slope range. Detector reports
            # ema_slow_slope in metadata; filtering here mirrors the
            # strategy page's entry filter so the detection page and
            # the strategy page see the same signal set.
            if pattern_name == "wedge_pop":

                def _passes_slope(s):
                    slope = s.metadata.get("ema_slow_slope")
                    if slope is None:
                        return True
                    if enable_min_slope_wp and slope < min_ema_slow_slope_wp:
                        return False
                    if enable_max_slope_wp and slope > max_ema_slow_slope_wp:
                        return False
                    return True

                signals = [s for s in signals if _passes_slope(s)]
        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()

    st.subheader(f"{ticker} — {pattern_name} ({len(signals)} signals)")

    chart_builder = PlotlyChartBuilder()
    fig = chart_builder.build_candlestick_with_signals(df, signals, title=f"{ticker} — {pattern_name}")
    fig.update_xaxes(range=[str(start_date), str(end_date)])

    # Wick Play 3-bar overlay — annotate Bar i-2 (W = wick),
    # Bar i-1 (I = inside), Bar i (B = breakout) so the user can
    # see the full setup structure, not just the entry.
    if pattern_name == "wick_play" and signals:
        df_tz = df.copy()
        if df_tz.index.tz is not None:
            df_tz.index = df_tz.index.tz_localize(None)
        date_to_pos = {idx.date(): p for p, idx in enumerate(df_tz.index)}
        first_group = True
        for s in signals:
            pos_i = date_to_pos.get(s.date)
            if pos_i is None or pos_i < 2:
                continue
            ts_w = df_tz.index[pos_i - 2]
            ts_n = df_tz.index[pos_i - 1]
            ts_b = df_tz.index[pos_i]
            high_w = float(df_tz["High"].iloc[pos_i - 2])
            high_n = float(df_tz["High"].iloc[pos_i - 1])
            high_b = float(df_tz["High"].iloc[pos_i])
            wick_high = float(s.metadata.get("wick_high", high_w))

            # Letter markers above each bar's high
            fig.add_trace(
                go.Scatter(
                    x=[ts_w, ts_n, ts_b],
                    y=[high_w * 1.01, high_n * 1.01, high_b * 1.01],
                    mode="markers+text",
                    marker=dict(
                        symbol=["circle", "square", "diamond"],
                        size=[16, 16, 16],
                        color=["#FFB300", "#9E9E9E", "#2196F3"],
                        line=dict(width=1, color="white"),
                    ),
                    text=["W", "I", "B"],
                    textposition="middle center",
                    textfont=dict(color="white", size=10, family="Arial Black"),
                    name="Wick / Inside / Breakout",
                    legendgroup="wick_play_bars",
                    showlegend=first_group,
                    hovertext=[
                        f"Bar i-2 — Wick bar<br>Date: {ts_w.date()}<br>"
                        f"High: {high_w:.2f}<br>"
                        f"Upper wick ratio: {s.metadata.get('upper_wick_ratio')}",
                        f"Bar i-1 — Inside bar<br>Date: {ts_n.date()}<br>"
                        f"High: {high_n:.2f}  Low: {s.metadata.get('inside_low')}<br>"
                        f"Vol vs wick: {s.metadata.get('inside_vol_ratio_vs_wick')}",
                        f"Bar i — Breakout<br>Date: {ts_b.date()}<br>"
                        f"Close: {s.entry_price:.2f}<br>"
                        f"Breakout strength: {s.metadata.get('breakout_strength_atr')} ATR",
                    ],
                    hoverinfo="text",
                ),
                row=1,
                col=1,
            )

            # Trigger line — wick_high from Bar i-2 through Bar i
            fig.add_trace(
                go.Scatter(
                    x=[ts_w, ts_b],
                    y=[wick_high, wick_high],
                    mode="lines",
                    line=dict(color="#FFB300", width=1.2, dash="dash"),
                    name="Wick High (trigger)",
                    legendgroup="wick_play_trigger",
                    showlegend=first_group,
                    hoverinfo="skip",
                    opacity=0.75,
                ),
                row=1,
                col=1,
            )

            first_group = False

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
