from datetime import date, timedelta

import streamlit as st

from data.adapters.cached_market_data import CachedMarketDataAdapter
from data.adapters.wikipedia_universe import WikipediaUniverseAdapter
from data.adapters.yfinance_adapter import YFinanceAdapter
from pattern.adapters.exhaustion_extension_top import ExhaustionExtensionTopDetector
from pattern.adapters.wedge_pop import WedgePopDetector
from strategy.adapters.multi_wedgepop_strategy import MultiWedgepopStrategy
from strategy.adapters.wedgepop_strategy import WedgepopStrategy
from strategy.domain.models import MultiStrategyConfig, TossFeeSchedule
from visualization.adapters.plotly_charts import PlotlyChartBuilder

st.set_page_config(page_title="Multi Wedge Pop Scan", layout="wide")
st.title("Multi-Ticker Wedge Pop Strategy")
st.caption(
    "S&P 500 / Nasdaq-100 전체에서 매일 발생한 Wedge Pop signal 중 거래량이 "
    "가장 큰 종목 하나를 매수. 한 포지션이 청산될 때까지 다른 종목은 사지 않음."
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
        value=50,
        min_value=0,
        max_value=600,
        step=10,
        help="스캔할 최대 종목 수. 0이면 전체. 처음엔 작게(20~50) 시작해서 "
        "동작을 확인한 뒤 늘려. yfinance 호출이 종목당 1회 발생.",
    )
    max_workers = st.number_input(
        "Parallel workers",
        value=8,
        min_value=1,
        max_value=32,
        step=1,
        help="yfinance 호출을 동시에 몇 개 보낼지. 너무 크면 rate-limit.",
    )

    st.header("Period")
    start_date = st.date_input(
        "Start Date",
        value=date.today() - timedelta(days=365),
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
        help="포트폴리오 시작 자본. 한 번에 한 종목만 보유하므로 매수 시 "
        "전체 자본 기준으로 risk_per_trade × capital 로 사이즈 계산.",
    )
    risk_pct = st.number_input(
        "Risk per Trade (%)",
        value=5.0,
        min_value=0.1,
        max_value=100.0,
        step=0.5,
    )
    max_holding_days = st.number_input("Max Holding Days", value=60, min_value=1, max_value=2_000, step=5)

    st.header("Pattern Detection")
    consolidation_pct = st.number_input(
        "Min consolidation %",
        value=30.0,
        min_value=0.0,
        max_value=100.0,
        step=5.0,
        help="직전 N일 중 close가 fast EMA 아래에 있어야 하는 " "**최소** 비율.",
    )
    enable_max_cp = st.checkbox(
        "Cap max consolidation %",
        value=False,
        help="상한 추가 — 너무 깊은 consolidation (95%+)을 제외하려면.",
    )
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
        help="breakout move가 ATR의 몇 배 이상이어야 wedge pop으로 인정. "
        "ATR 기반이라 종목 변동성에 자동 적응. 0이면 하한 off.",
    )
    enable_max_bp = st.checkbox(
        "Cap max breakout strength",
        value=False,
        help="상한 추가 — 이미 너무 큰 단일봉 gap(+4 ATR+)을 overextended로 제외.",
    )
    max_breakout_atr_mult_ui = st.number_input(
        "Max breakout strength (× ATR)",
        value=3.0,
        min_value=0.0,
        max_value=20.0,
        step=0.5,
        format="%.2f",
        disabled=not enable_max_bp,
        help="breakout이 ATR의 이 배수를 넘으면 overextended로 제외.",
    )
    require_above_long_smas = st.checkbox(
        "Require close above 50 & 200 SMA",
        value=True,
        help="signal 캔들의 close가 50 SMA와 200 SMA 둘 다 위에 있을 "
        "때만 wedge pop으로 인정. 장기 추세 안에 있는 wedge pop만 "
        "필터링하고 싶을 때 켜세요.",
    )
    late_entry_bars_wp = st.number_input(
        "Late-entry bars (0 = strict fresh-cross)",
        value=0,
        min_value=0,
        max_value=10,
        step=1,
        help="기본 0: 전일이 반드시 EMA 아래여야 함 (엄격한 첫 돌파). "
        "N>0: 돌파가 N봉 이내에 일어났으면 continuation 봉도 wedge pop "
        "으로 인정. 예: 2 = 첫 돌파일 포함 최대 3봉까지 '여진' 봉 catch. "
        "metadata trigger 필드로 primary / late_entry 구분됨.",
    )
    detect_lookback = st.number_input(
        "Consolidation lookback (days)",
        value=10,
        min_value=3,
        max_value=60,
        step=1,
        help="Consolidation 체크 / consolidation-low stop 계산 기간.",
    )
    cooldown_bars_ui = st.number_input(
        "Cooldown bars (after a signal)",
        value=0,
        min_value=0,
        max_value=60,
        step=1,
        help="signal fire 후 건너뛰는 바 수. 0 = 연속 signal 허용. "
        "예전엔 lookback을 그대로 썼는데 이제 독립 파라미터.",
    )
    ema_fast = st.number_input(
        "Fast EMA period",
        value=10,
        min_value=2,
        max_value=100,
        step=1,
        help="Detector와 strategy가 공통으로 쓰는 fast EMA.",
    )
    ema_slow = st.number_input(
        "Slow EMA period",
        value=20,
        min_value=2,
        max_value=200,
        step=1,
        help="Detector breakout 및 strategy exhaustion reference = " "max(fast, slow) 계산에 쓰임.",
    )
    slope_lookback = st.number_input(
        "Slope lookback (days)",
        value=10,
        min_value=5,
        max_value=120,
        step=1,
        help="ema_fast_slope / ema_slow_slope 측정 기간. signal metadata "
        "에 기록되어 downstream slope 필터의 입력이 됨.",
    )

    st.header("Entry Filter")
    require_gap_up = st.checkbox(
        "Require gap-up confirmation",
        value=False,
        help="다음날 시초가가 breakout 종가보다 높을 때만 진입.",
    )
    enable_entry_ema_filter = st.checkbox(
        "Enable EMA-extension entry filter",
        value=True,
        help="Entry open이 signal bar의 max(10EMA, 20EMA) 위로 N × ATR "
        "이상 초과하면 진입 거부. ATR 기반이라 변동성에 자동 적응.",
    )
    max_entry_ema_extension_atr = st.number_input(
        "Max entry extension above EMA (× ATR)",
        value=1.5,
        min_value=0.0,
        max_value=20.0,
        step=0.1,
        format="%.2f",
        disabled=not enable_entry_ema_filter,
        help="(entry_open - max(ema10, ema20)) / ATR > 이 값이면 거부.",
    )
    st.caption("**EMA slow slope 범위 필터**")
    enable_min_slope = st.checkbox(
        "Enforce min EMA slow slope",
        value=True,
        help="ema_slow_slope의 **하한**. 양수로 설정하면 '반드시 +N% "
        "이상 상승 추세'만 통과 (예: 0.05). 음수면 dead-cat bounce "
        "거부 (예: -0.01).",
    )
    min_ema_slow_slope_ui = st.number_input(
        "Min EMA slow slope",
        value=0.005,
        min_value=-1.0,
        max_value=1.0,
        step=0.01,
        format="%.3f",
        disabled=not enable_min_slope,
    )
    enable_max_slope = st.checkbox(
        "Enforce max EMA slow slope",
        value=False,
        help="ema_slow_slope의 **상한**. parabolic하게 오른 종목을 " "제외하려면 켜세요 (예: 0.30).",
    )
    max_ema_slow_slope_ui = st.number_input(
        "Max EMA slow slope",
        value=0.30,
        min_value=-1.0,
        max_value=5.0,
        step=0.05,
        format="%.3f",
        disabled=not enable_max_slope,
    )

    st.header("Swing Pivot (S/R + Trendline)")
    swing_pivot_left_ui = st.number_input(
        "Pivot window — left bars",
        value=2,
        min_value=1,
        max_value=10,
        step=1,
    )
    swing_pivot_right_ui = st.number_input(
        "Pivot window — right bars",
        value=2,
        min_value=1,
        max_value=10,
        step=1,
        help="pivot은 이만큼 뒤에 확정 (lag).",
    )
    swing_pivot_lookback_ui = st.number_input(
        "Pivot lookback (bars)",
        value=60,
        min_value=10,
        max_value=500,
        step=5,
    )
    enable_swing_resistance = st.checkbox(
        "Enable swing-high resistance filter (Entry)",
        value=True,
        help="Entry open이 직전 swing high 바로 아래 " "tolerance × ATR 이내면 거부.",
    )
    swing_resistance_tol_atr = st.number_input(
        "Resistance tolerance (× ATR)",
        value=0.5,
        min_value=0.0,
        max_value=5.0,
        step=0.1,
        format="%.2f",
        disabled=not enable_swing_resistance,
    )
    enable_trendline_exit = st.checkbox(
        "Enable higher-low trendline exit",
        value=True,
        help="최근 swing low 추세선 아래로 low 침투 시 청산 " "(limit 모델).",
    )
    trendline_max_pivots_ui = st.number_input(
        "Trendline — max pivots",
        value=3,
        min_value=2,
        max_value=10,
        step=1,
        disabled=not enable_trendline_exit,
    )
    enable_resistance_break_exit = st.checkbox(
        "Enable resistance-break exit (false-breakout)",
        value=False,
        help="window 내 모든 swing high를 독립 stop으로. close가 "
        "level + confirm_buffer×ATR 넘으면 confirmed. 이후 low가 "
        "level - pierce_buffer×ATR 밑으로 가면 체결.",
    )
    resistance_break_pierce_buffer_atr = st.number_input(
        "Pierce buffer (× ATR)",
        value=0.5,
        min_value=0.0,
        max_value=3.0,
        step=0.05,
        format="%.2f",
        disabled=not enable_resistance_break_exit,
    )
    resistance_break_confirm_buffer_atr = st.number_input(
        "Confirm buffer (× ATR)",
        value=0.1,
        min_value=0.0,
        max_value=3.0,
        step=0.05,
        format="%.2f",
        disabled=not enable_resistance_break_exit,
    )

    st.header("Extras (no tuning — toggle only)")
    enable_market_regime_filter = st.checkbox(
        "Market regime filter — SPY > 200 SMA (Entry)",
        value=True,
        help="진입 시점 SPY 종가가 SPY 200 SMA 위일 때만 허용. "
        "약세장 wedge pop 실패율이 높은 문제를 차단.",
    )
    enable_breakeven_stop = st.checkbox(
        "Break-even stop after ≥ 1R profit (Exit)",
        value=True,
        help="close가 entry+1R 이상 한번이라도 도달한 뒤 " "low가 entry 아래로 가면 entry 가격에 청산.",
    )
    breakeven_arm_r_multiple_ui = st.number_input(
        "Break-even arm R multiple",
        value=1.0,
        min_value=0.5,
        max_value=5.0,
        step=0.1,
        format="%.2f",
        disabled=not enable_breakeven_stop,
        help="close가 entry + N×(entry-stop) 이상 도달해야 arm. "
        "1.0 = 기존(진입 후 +1R 즉시 arm). 1.5~2.0으로 올리면 더 증명된 "
        "뒤 arm하지만, +1R~+1.5R 사이에서 반등했다 빠지는 트레이드들이 "
        "원 stop까지 가서 손실 커질 수 있음 — 실측상 악화 확인됨.",
    )
    structural_exit_grace_bars_ui = st.number_input(
        "Structural-exit grace bars (Exit)",
        value=0,
        min_value=0,
        max_value=20,
        step=1,
        help="진입 후 N 바까지 trendline break + resistance break "
        "exit 억제. 0 = 기존. N>0으로 올리면 초기 chop 통과하지만, "
        "dud 신호에서 빠르게 털지 못해 손실 커지는 부작용 관찰됨.",
    )
    breakeven_exit_offset_r_ui = st.number_input(
        "Break-even exit offset (× R)",
        value=0.0,
        min_value=0.0,
        max_value=2.0,
        step=0.05,
        format="%.2f",
        disabled=not enable_breakeven_stop,
        help="Breakeven armed 이후 low가 entry+offset×R 닿으면 "
        "exit. 0 = entry 가격(기존). 작은 universe에서는 0.3 정도가 "
        "도움되지만, max_tickers=0 전체 유니버스에선 rotation을 줄여 "
        "오히려 역효과였음. 기본값은 0으로 유지.",
    )
    enable_structural_close_confirm = st.checkbox(
        "Structural-exit CLOSE confirmation",
        value=False,
        help="trendline/resistance break을 LOW 기반(기존) 대신 "
        "CLOSE 기반으로. 작은 universe(~150 tickers)에선 1일 wick "
        "탈출 차단해 큰 개선이나, 전체 universe에선 포지션이 "
        "오래 남아 다음 winner 신호 놓치므로 기본 OFF.",
    )
    enable_gap_down_rejection = st.checkbox(
        "Gap-down open rejection (Entry)",
        value=True,
        help="entry open이 signal bar low 아래로 갭다운 시 거부.",
    )
    enable_signal_close_strength = st.checkbox(
        "Signal close-strength filter (Entry)",
        value=False,
        help="signal 봉의 close가 당일 range의 상단 N% 이상일 때만 진입. "
        "close_location = (close - low) / (high - low). 윗꼬리 큰 pump-and-fade "
        "캔들을 차단. 켜면 샘플 수가 줄어드는 경향.",
    )
    min_signal_close_location_ui = st.number_input(
        "Min signal close location (0=low, 1=high)",
        value=0.5,
        min_value=0.0,
        max_value=1.0,
        step=0.05,
        format="%.2f",
        disabled=not enable_signal_close_strength,
        help="0.5 = 하단 절반 close 거부. 0.7 = 상단 30%만 통과 (더 보수적).",
    )
    enable_swing_breakout = st.checkbox(
        "Swing-breakout filter (Entry)",
        value=False,
        help="signal 봉의 high가 직전 swing high를 buffer×ATR 이상 "
        "돌파해야 진입. 구조적 돌파 확증. buffer=0에서도 상당히 제한적.",
    )
    swing_breakout_buffer_atr_ui = st.number_input(
        "Swing-breakout buffer (× ATR)",
        value=0.0,
        min_value=0.0,
        max_value=3.0,
        step=0.05,
        format="%.2f",
        disabled=not enable_swing_breakout,
        help="0 = 단순 돌파만 요구. 0.3 = 0.3×ATR 이상 확실히 돌파해야.",
    )
    enable_euphoria_cap = st.checkbox(
        "Signal bar euphoria cap (Entry)",
        value=True,
        help="signal 봉의 당일 상승폭(close-open)이 ATR의 N배를 초과하면 "
        "거부. 다종목 스캔에서 buy/sell ratio 랭킹이 euphoric 급등 후보를 "
        "선호하는 adverse selection을 방지. 이미 한계 돌파한 pump 캔들은 "
        "다음날 mean revert 확률이 높음.",
    )
    max_signal_bar_gain_atr_ui = st.number_input(
        "Max signal bar gain (× ATR)",
        value=2.5,
        min_value=0.5,
        max_value=10.0,
        step=0.1,
        format="%.2f",
        disabled=not enable_euphoria_cap,
        help="일봉 (close-open)/ATR 상한. 2.5 = 일반적 breakout 허용, " "극단 pump 제거. 낮출수록 보수적.",
    )

    st.header("Exit Tuning")
    use_smart_trail = st.checkbox(
        "Smart Trail (Chandelier + Profit-tier)",
        value=False,
        help="기존 10 EMA trail 대신 Chandelier Exit 사용. "
        "최고가에서 ATR 기반 trailing, 수익 커질수록 trail 넓힘. "
        "**<2R → 3×ATR, 2~4R → 4×ATR, >4R → 5×ATR**. "
        "진입 후 3봉 동안은 trail 비활성.",
    )
    st.caption(
        "**Exhaustion Extension Top exit** — 상승추세 꼭지에서 "
        "10 EMA 위로 과도하게 벌어지는 블로우오프 캔들이 찍히면 "
        "그 봉 종가에서 long 청산. 감지가 end-of-bar라 미래 데이터 "
        "참조 없음."
    )
    enable_exh_exit = st.checkbox(
        "Enable Exhaustion Extension Top exit",
        value=True,
        help="보유 중인 long에 대해, ExhaustionExtensionTopDetector가 해당 "
        "봉에서 fire하면 그 봉 종가에서 청산. 진입 바 자체는 스킵.",
    )
    exh_exit_extension_atr = st.number_input(
        "Exh exit — Min extension above EMA (× ATR)",
        value=1.9,
        min_value=0.5,
        max_value=20.0,
        step=0.1,
        disabled=not enable_exh_exit,
    )
    exh_exit_min_slope = st.number_input(
        "Exh exit — Min slow EMA slope",
        value=0.005,
        min_value=-1.0,
        max_value=1.0,
        step=0.001,
        format="%.4f",
        disabled=not enable_exh_exit,
        help="slope_lookback 바 동안 slow EMA가 이 비율 이상 상승. " "0.005 ≈ 연 13% 속도.",
    )
    exh_exit_max_close_loc = st.number_input(
        "Exh exit — Max close location (0=low, 1=high)",
        value=0.5,
        min_value=0.0,
        max_value=1.0,
        step=0.05,
        format="%.2f",
        disabled=not enable_exh_exit,
        help="윗꼬리 거절 캔들 확증. 1.0 = off.",
    )
    exh_exit_min_sell_dom = st.number_input(
        "Exh exit — Min sell dominance",
        value=1.5,
        min_value=0.0,
        max_value=10.0,
        step=0.1,
        disabled=not enable_exh_exit,
        help="최근 봉에서 음봉 vol / 양봉 vol ≥ 이 값이면 통과. 0 = off.",
    )
    exh_exit_rejection_override = st.checkbox(
        "Exh exit — Upper-wick rejection override",
        value=True,
        disabled=not enable_exh_exit,
        help="close_location ≤ 0.25 강한 샹팅스타는 sell_dominance / " "cooldown 건너뛰고 fire.",
    )

    st.header("Fees (Toss Securities)")
    st.caption("토스증권 미국주식 기본 수수료. 매수/매도 각 0.1% + SEC fee 0.00229% (매도시).")
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
        help="미국 SEC Section 31 규정 수수료. 매도 거래대금에 부과.",
    )

    run_btn = st.button("Run Universe Scan", type="primary", use_container_width=True)


# --- Main ---
if not run_btn:
    st.info(
        "좌측에서 universe / 기간을 설정하고 **Run Universe Scan** 을 눌러줘. "
        "처음 실행 땐 max_tickers를 작게 잡고 스모크 테스트해보는 게 안전해."
    )
    st.stop()

market_data = CachedMarketDataAdapter(YFinanceAdapter())
universe_provider = WikipediaUniverseAdapter()

# Fetch SPY once up-front when the market-regime filter is on so
# every per-ticker WedgepopStrategy reuses the same regime lookup
# instead of re-fetching SPY for each scanned ticker.
market_regime_df = None
if enable_market_regime_filter:
    from datetime import timedelta as _td

    _fetch_start = start_date - _td(days=400)
    try:
        market_regime_df = market_data.fetch_ohlcv("SPY", _fetch_start, end_date)
    except Exception as e:
        st.error(f"Failed to fetch SPY: {e}")
        st.stop()
detector = WedgePopDetector(
    lookback=int(detect_lookback),
    ema_fast=int(ema_fast),
    ema_slow=int(ema_slow),
    consolidation_pct=consolidation_pct / 100.0,
    max_consolidation_pct=(max_consolidation_pct_ui / 100.0 if enable_max_cp else None),
    breakout_atr_mult=breakout_atr_mult,
    max_breakout_atr_mult=(max_breakout_atr_mult_ui if enable_max_bp else None),
    slope_lookback=int(slope_lookback),
    cooldown_bars=int(cooldown_bars_ui),
    require_above_long_smas=require_above_long_smas,
    late_entry_bars=int(late_entry_bars_wp),
)
exit_detector = (
    ExhaustionExtensionTopDetector(
        extension_atr_mult=exh_exit_extension_atr,
        min_slow_slope=exh_exit_min_slope,
        max_close_location=exh_exit_max_close_loc,
        min_sell_dominance=exh_exit_min_sell_dom,
        enable_rejection_override=exh_exit_rejection_override,
        ema_fast=int(ema_fast),
        ema_slow=int(ema_slow),
        slope_lookback=int(slope_lookback),
    )
    if enable_exh_exit
    else None
)

per_ticker_strategy = WedgepopStrategy(
    market_data=market_data,
    detector=detector,
    ema_trail=int(ema_fast),
    ema_slow=int(ema_slow),
    max_entry_ema_extension_atr=(max_entry_ema_extension_atr if enable_entry_ema_filter else None),
    max_ema_slope_decline=None,
    min_ema_slow_slope=(min_ema_slow_slope_ui if enable_min_slope else None),
    max_ema_slow_slope=(max_ema_slow_slope_ui if enable_max_slope else None),
    require_gap_up=require_gap_up,
    use_smart_trail=use_smart_trail,
    exit_detector=exit_detector,
    enable_swing_resistance_filter=enable_swing_resistance,
    swing_pivot_left=int(swing_pivot_left_ui),
    swing_pivot_right=int(swing_pivot_right_ui),
    swing_pivot_lookback=int(swing_pivot_lookback_ui),
    swing_resistance_tolerance_atr=float(swing_resistance_tol_atr),
    enable_trendline_exit=enable_trendline_exit,
    trendline_max_pivots=int(trendline_max_pivots_ui),
    enable_resistance_break_exit=enable_resistance_break_exit,
    resistance_break_pierce_buffer_atr=float(resistance_break_pierce_buffer_atr),
    resistance_break_confirm_buffer_atr=float(resistance_break_confirm_buffer_atr),
    enable_market_regime_filter=enable_market_regime_filter,
    market_regime_df=market_regime_df,
    enable_breakeven_stop=enable_breakeven_stop,
    enable_gap_down_rejection=enable_gap_down_rejection,
    enable_signal_close_strength_filter=enable_signal_close_strength,
    min_signal_close_location=float(min_signal_close_location_ui),
    enable_swing_breakout_filter=enable_swing_breakout,
    swing_breakout_buffer_atr=float(swing_breakout_buffer_atr_ui),
    max_signal_bar_gain_atr=(float(max_signal_bar_gain_atr_ui) if enable_euphoria_cap else None),
    breakeven_arm_r_multiple=float(breakeven_arm_r_multiple_ui),
    structural_exit_grace_bars=int(structural_exit_grace_bars_ui),
    breakeven_exit_offset_r=float(breakeven_exit_offset_r_ui),
    structural_exit_close_confirm=enable_structural_close_confirm,
)
runner = MultiWedgepopStrategy(
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
    max_holding_days=int(max_holding_days),
    max_tickers=int(max_tickers) if max_tickers > 0 else None,
    fee_schedule=fee_schedule,
)

with st.spinner(f"Scanning {universe} ({config.max_tickers or 'all'} tickers)..."):
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
        "이 기간엔 universe 전체에서 Wedge Pop signal이 잡히지 않았거나, "
        "잡혔어도 모두 진입 조건에서 걸렸어. 기간을 늘려보거나 universe를 "
        "넓혀봐."
    )
    st.stop()

chart_builder = PlotlyChartBuilder()

# --- Equity curve ---
st.subheader("Portfolio Equity Curve")
eq_fig = chart_builder.build_equity_curve(
    result.equity_curve,
    title=f"Single-position portfolio across {universe.upper()}",
)
st.plotly_chart(eq_fig, use_container_width=True)

# --- Best individual trades ---
st.subheader("Best Trades")
trades_fig = chart_builder.build_top_trades_bar(
    result.trades,
    title="Top trades by % return",
)
st.plotly_chart(trades_fig, use_container_width=True)

# --- Per-ticker contribution ---
st.subheader("P&L by Ticker")
contrib_fig = chart_builder.build_ticker_contribution_bar(
    result.trades,
    title="Total P&L contribution per ticker",
)
st.plotly_chart(contrib_fig, use_container_width=True)

# --- Trade table ---
st.subheader("Trades — Details")
EXIT_REASON_LABELS = {
    "exhaustion_exit": "Exhaustion Extension Top",
    "trendline_break": "Higher-Low Trendline Break",
    "smart_trail": "Smart Trail (Chandelier)",
    "resistance_break": "Resistance Break (false breakout)",
    "breakeven_stop": "Break-even Stop (≥1R unrealized)",
    "end_of_data": "End of Data (no exit fired)",
}

rows = [
    {
        "Ticker": t.ticker,
        "Entry Date": t.entry_date,
        "Exit Date": t.exit_date,
        "Exit Reason": EXIT_REASON_LABELS.get(t.exit_reason, t.exit_reason),
        "Entry": f"${t.entry_price:,.2f}",
        "Exit": f"${t.exit_price:,.2f}",
        "Stop": f"${t.stop_loss:,.2f}",
        "Shares": t.shares,
        "Sig Vol": f"{t.signal_volume:,.0f}",
        "Buy Vol": f"{t.signal_buy_volume:,.0f}",
        "Sell Vol": f"{t.signal_sell_volume:,.0f}",
        "Buy/Sell": f"{t.signal_buy_sell_ratio:.2f}",
        "Gross P&L": f"${t.gross_pnl:,.2f}",
        "Commission": f"${t.commission:,.2f}",
        "Net P&L ($)": f"${t.pnl:,.2f}",
        "Net P&L (%)": f"{t.pnl_pct:.2%}",
    }
    for t in result.trades
]
st.dataframe(rows, use_container_width=True)

# --- Per-trade candlestick charts ---
st.subheader("Trade Charts")
CONTEXT_BEFORE_DAYS = 60  # calendar days before entry to show
CONTEXT_AFTER_DAYS = 30  # calendar days after exit to show

for i, t in enumerate(result.trades):
    pnl_sign = "+" if t.pnl >= 0 else ""
    label = f"{t.ticker} — {t.entry_date} → {t.exit_date}  " f"({pnl_sign}{t.pnl_pct:.2%})"
    with st.expander(label, expanded=(i < 3)):
        chart_start = t.entry_date - timedelta(days=CONTEXT_BEFORE_DAYS)
        chart_end = t.exit_date + timedelta(days=CONTEXT_AFTER_DAYS)
        # Extra warmup so 50/200 SMA lines are converged
        fetch_start = chart_start - timedelta(days=400)
        try:
            ticker_df = market_data.fetch_ohlcv(t.ticker, fetch_start, chart_end)
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
