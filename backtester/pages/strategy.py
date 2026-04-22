from datetime import date, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data.adapters.yfinance_adapter import YFinanceAdapter
from pattern.adapters.exhaustion_extension_top import ExhaustionExtensionTopDetector
from pattern.adapters.wedge_pop import WedgePopDetector
from pattern.helpers.pivots import (
    fit_lower_trendline,
    last_swing_high,
    recent_swing_high,
)
from strategy.adapters.wedgepop_strategy import WedgepopStrategy
from strategy.domain.models import StrategyConfig
from visualization.adapters.plotly_charts import PlotlyChartBuilder

st.set_page_config(page_title="Strategy", layout="wide")
st.title("Wedge Pop Strategy")
st.caption(
    "Wedge Pop 신호 다음날 open 매수 → 손절(consolidation low) / "
    "Exhaustion 익절 / 10 EMA trail / time stop"
)

# --- Sidebar inputs ---
with st.sidebar:
    st.header("Market")
    ticker = st.text_input("Ticker", value="AAPL")
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
    )
    risk_pct = st.number_input(
        "Risk per Trade (%)",
        value=5.0,
        min_value=0.1,
        max_value=100.0,
        step=0.5,
        help="자본의 몇 %를 거래당 위험에 노출할지. 일반적으로 1~2%, "
        "공격적으로 운용하면 5~10%, 실험적으론 그 이상까지 가능.",
    )
    max_holding_days = st.number_input(
        "Max Holding Days",
        value=60,
        min_value=1,
        max_value=2_000,
        step=5,
        help="Time stop 발동까지 최대 보유 일수.",
    )

    st.header("Pattern Detection")
    consolidation_pct = st.number_input(
        "Min consolidation %",
        value=30.0,
        min_value=0.0,
        max_value=100.0,
        step=5.0,
        help="직전 N일 중 close가 fast EMA 아래에 있어야 하는 " "**최소** 비율. 0%면 하한 off.",
    )
    enable_max_cp = st.checkbox(
        "Cap max consolidation %",
        value=False,
        help="상한 추가 — 너무 깊은 consolidation (95%+)은 coil이 "
        "아니라 capitulation이라 제외하고 싶을 때.",
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
        help="상한 추가 — 이미 너무 큰 단일봉 gap(+4 ATR+)은 overextended " "라 제외하고 싶을 때.",
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
        "때만 wedge pop으로 인정. 장기 추세 confirm 용.",
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
        help="Consolidation 체크 및 stop-loss (consolidation low) 계산에 "
        "쓰이는 직전 봉 개수. 기본 15일.",
    )
    cooldown_bars_ui = st.number_input(
        "Cooldown bars (after a signal)",
        value=0,
        min_value=0,
        max_value=60,
        step=1,
        help="한 signal이 fire된 후 다음 signal까지 건너뛰는 바 개수. "
        "0으로 두면 쿨다운 없이 연속 signal 허용 (예전엔 "
        "Consolidation lookback 값을 그대로 쓰던 부분을 독립 파라미터로 "
        "분리). Case: AAPL 2025-08-06은 07-16 signal의 기본 쿨다운에 "
        "걸려서 안 잡히는데, 5 정도로 낮추면 잡힘.",
    )
    ema_fast = st.number_input(
        "Fast EMA period",
        value=10,
        min_value=2,
        max_value=100,
        step=1,
        help="Detector와 strategy가 공통으로 쓰는 fast EMA (기본 10).",
    )
    ema_slow = st.number_input(
        "Slow EMA period",
        value=20,
        min_value=2,
        max_value=200,
        step=1,
        help="Detector의 breakout 체크와 strategy의 exhaustion reference "
        "라인 max(fast, slow) 계산에 쓰이는 slow EMA (기본 20).",
    )
    slope_lookback = st.number_input(
        "Slope lookback (days)",
        value=10,
        min_value=5,
        max_value=120,
        step=1,
        help="ema_fast_slope / ema_slow_slope를 계산하는 기간. "
        "signal metadata에 담겨 downstream slope 필터의 입력이 됨.",
    )

    st.header("Entry Filter")
    require_gap_up = st.checkbox(
        "Require gap-up confirmation",
        value=False,
        help="다음날 시초가가 전날 종가보다 높을 때만 진입 (TraderLion의 "
        "'Wedge Pops with unfilled gaps show strong momentum' 룰).",
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
    st.caption("**EMA slow slope 범위 필터** — 양수/음수 모두 가능.")
    enable_min_slope = st.checkbox(
        "Enforce min EMA slow slope",
        value=True,
        help="ema_slow_slope의 **하한**. 양수로 설정하면 '반드시 +N% "
        "이상 올라가는 추세'만 통과 (예: 0.05 → 20일 동안 +5% 이상). "
        "음수로 설정하면 dead-cat bounce를 걸러냄 (예: -0.01).",
    )
    min_ema_slow_slope_ui = st.number_input(
        "Min EMA slow slope",
        value=0.005,
        min_value=-1.0,
        max_value=1.0,
        step=0.001,
        format="%.3f",
        disabled=not enable_min_slope,
    )
    enable_max_slope = st.checkbox(
        "Enforce max EMA slow slope",
        value=False,
        help="ema_slow_slope의 **상한**. 이미 너무 가파르게 오른 "
        "종목 (parabolic)을 제외하려면 켜세요 (예: 0.30 → +30% 이상 "
        "오른 종목 제외).",
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
    st.caption(
        "Bill Williams 5-bar fractal로 swing high/low를 찾아 "
        "**수평 저항선 필터**(Entry)와 **추세선 이탈 exit**를 "
        "적용. pivot은 `right` 봉 뒤에 확정되므로 검출에 그만큼 "
        "lag 있음 (미래 데이터 참조 아님)."
    )
    swing_pivot_left_ui = st.number_input(
        "Pivot window — left bars",
        value=2,
        min_value=1,
        max_value=10,
        step=1,
        help="swing high/low 검출용 좌측 비교 봉 수. 2면 " "`High[i] > High[i-1], High[i-2]` 둘 다 필요.",
    )
    swing_pivot_right_ui = st.number_input(
        "Pivot window — right bars",
        value=2,
        min_value=1,
        max_value=10,
        step=1,
        help="우측 비교 봉 수 — 이만큼 지나야 pivot 확정.",
    )
    swing_pivot_lookback_ui = st.number_input(
        "Pivot lookback (bars)",
        value=60,
        min_value=10,
        max_value=500,
        step=5,
        help="직전 N봉 안에서 pivot 탐색 (저항선 / 추세선 둘 다 공통).",
    )
    enable_swing_resistance = st.checkbox(
        "Enable swing-high resistance filter (Entry)",
        value=True,
        help="Entry open이 직전 swing high 바로 아래 "
        "`tolerance × ATR` 이내면 거부 (저항선 충돌). "
        "저항선을 돌파한 상태면 통과 — 이미 깨고 올라간 경우는 OK.",
    )
    swing_resistance_tol_atr = st.number_input(
        "Resistance tolerance (× ATR)",
        value=0.5,
        min_value=0.0,
        max_value=5.0,
        step=0.1,
        format="%.2f",
        disabled=not enable_swing_resistance,
        help="swing_high - entry_price < 이 값 × ATR 이면 거부.",
    )
    enable_trendline_exit = st.checkbox(
        "Enable higher-low trendline exit",
        value=True,
        help="최근 swing low들을 이어 만든 상승 추세선 아래로 "
        "low가 침투하면 추세선 가격에 체결 (limit 모델). "
        "slope ≤ 0이면 비활성 (higher-low 구조 없음).",
    )
    enable_resistance_break_exit = st.checkbox(
        "Enable resistance-break exit (false-breakout)",
        value=False,
        help="진입 시점 window 내 모든 swing high를 독립 stop level로 "
        "관리. 각 level별로: close가 (level + confirm_buffer × ATR) "
        "위로 올라가면 confirmed. 이후 low가 (level - pierce_buffer × ATR) "
        "아래로 내려가면 해당 가격에 체결. entry가 이미 level 위라면 "
        "즉시 confirmed. Entry 필터와 독립.",
    )
    resistance_break_pierce_buffer_atr = st.number_input(
        "Pierce buffer (× ATR below level)",
        value=0.5,
        min_value=0.0,
        max_value=3.0,
        step=0.05,
        format="%.2f",
        disabled=not enable_resistance_break_exit,
        help="level - 이값×ATR 밑으로 low 가 내려가야 fire. 0 이면 "
        "level 정확히 tag해도 trigger (노이즈 많음). 0.3~0.5면 의미있는 "
        "이탈만 잡아줌.",
    )
    resistance_break_confirm_buffer_atr = st.number_input(
        "Confirm buffer (× ATR above level)",
        value=0.1,
        min_value=0.0,
        max_value=3.0,
        step=0.05,
        format="%.2f",
        disabled=not enable_resistance_break_exit,
        help="close가 level + 이값×ATR 위로 가야 confirmed. 약한 " "돌파 거르는 용.",
    )
    trendline_max_pivots_ui = st.number_input(
        "Trendline — max pivots",
        value=3,
        min_value=2,
        max_value=10,
        step=1,
        disabled=not enable_trendline_exit,
        help="최근 N개 swing low까지 써서 선형회귀. 2면 정확히 " "최근 두 점을 잇는 선.",
    )
    # Master swing-overlay toggle retired — each pivot / line is now
    # gated on the specific strategy flag that actually uses it, so
    # the chart only renders what's currently active. Keeping the
    # name bound to True preserves downstream references without
    # adding dead code.
    show_swing_overlay = True

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
        help="기존 10 EMA trail 대신 고수들이 쓰는 Chandelier Exit 사용. "
        "진입 후 최고가에서 ATR 기반으로 trailing하며, 수익이 커질수록 "
        "(R배수 기준) trail을 넓혀서 큰 수익은 더 달리게 함. "
        "**<2R → 3×ATR, 2~4R → 4×ATR, >4R → 5×ATR**. "
        "진입 후 최소 3봉은 trail 비활성화해서 초기 흔들림에 안 걸림.",
    )
    st.caption(
        "**Exhaustion Extension Top exit** — 상승추세 꼭지에서 "
        "10 EMA 위로 과도하게 벌어지는 블로우오프 캔들이 찍히면 "
        "그 봉의 종가에서 long 청산. 감지가 end-of-bar라 미래 "
        "데이터 참조 없음."
    )
    enable_exh_exit = st.checkbox(
        "Enable Exhaustion Extension Top exit",
        value=True,
        help="보유 중인 long 포지션에 대해, ExhaustionExtensionTopDetector "
        "가 해당 봉에서 fire하면 그 봉 종가에서 청산. 진입 바 자체는 "
        "스킵 (같은 바 exit 방지).",
    )
    exh_exit_extension_atr = st.number_input(
        "Exh exit — Min extension above EMA (× ATR)",
        value=1.9,
        min_value=0.5,
        max_value=20.0,
        step=0.1,
        disabled=not enable_exh_exit,
        help="bar의 high가 fast EMA 위로 ATR의 몇 배 이상이어야 하는지.",
    )
    exh_exit_min_slope = st.number_input(
        "Exh exit — Min slow EMA slope",
        value=0.005,
        min_value=-1.0,
        max_value=1.0,
        step=0.001,
        format="%.4f",
        disabled=not enable_exh_exit,
        help="slope_lookback 바 동안 slow EMA가 이 비율 이상 상승해야 함. " "0.005 = 연 ~13% 속도.",
    )
    exh_exit_max_close_loc = st.number_input(
        "Exh exit — Max close location (0=low, 1=high)",
        value=0.5,
        min_value=0.0,
        max_value=1.0,
        step=0.05,
        format="%.2f",
        disabled=not enable_exh_exit,
        help="close가 (close-low)/(high-low) 이 값 이하일 때 통과. " "윗꼬리 거절 캔들 확증. 1.0 = off.",
    )
    exh_exit_min_sell_dom = st.number_input(
        "Exh exit — Min sell dominance",
        value=1.5,
        min_value=0.0,
        max_value=10.0,
        step=0.1,
        disabled=not enable_exh_exit,
        help="최근 pressure_lookback 봉에서 음봉 거래량 합 / 양봉 거래량 " "합 ≥ 이 값이면 통과. 0 = off.",
    )
    exh_exit_rejection_override = st.checkbox(
        "Exh exit — Upper-wick rejection override",
        value=True,
        disabled=not enable_exh_exit,
        help="close_location ≤ 0.25 인 강한 샹팅스타는 sell_dominance / " "cooldown을 건너뛰고 바로 fire.",
    )

    run_btn = st.button("Run Strategy", type="primary", use_container_width=True)


# --- Main ---
if not run_btn:
    st.info("좌측에서 ticker / 기간 / 위험 파라미터를 설정하고 **Run Strategy**를 눌러줘.")
    st.stop()

market_data = YFinanceAdapter()

# Fetch extra history so 50/200 SMA are converged from day 1 of the
# user's date range. ~400 calendar days ≈ 200+ trading days.
fetch_start = start_date - timedelta(days=400)
with st.spinner(f"Fetching {ticker} {fetch_start} → {end_date} (SMA warmup)..."):
    try:
        df = market_data.fetch_ohlcv(ticker, fetch_start, end_date)
    except Exception as e:
        st.error(f"Failed to fetch data: {e}")
        st.stop()

if df is None or df.empty:
    st.warning("No data returned for that range.")
    st.stop()

market_regime_df = None
if enable_market_regime_filter:
    # Fetch SPY once; strategy needs 200 trading days of history
    # before start_date for the 200 SMA to converge on day 1.
    with st.spinner("Fetching SPY for market-regime filter..."):
        try:
            market_regime_df = market_data.fetch_ohlcv("SPY", fetch_start, end_date)
        except Exception as e:
            st.error(f"Failed to fetch SPY: {e}")
            st.stop()

config = StrategyConfig(
    ticker=ticker,
    start_date=start_date,
    end_date=end_date,
    pattern_name="wedge_pop",
    initial_capital=float(initial_capital),
    risk_per_trade=risk_pct / 100.0,
    max_holding_days=max_holding_days,
)

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

strategy = WedgepopStrategy(
    market_data=market_data,
    detector=detector,
    ema_trail=int(ema_fast),
    ema_slow=int(ema_slow),
    max_entry_ema_extension_atr=(max_entry_ema_extension_atr if enable_entry_ema_filter else None),
    max_ema_slope_decline=None,  # superseded by min/max_ema_slow_slope
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

with st.spinner("Running strategy..."):
    try:
        result = strategy.execute(df, config)
    except Exception as e:
        st.error(f"Strategy failed: {e}")
        st.stop()

perf = result.performance

# --- Headline metrics ---
st.subheader(f"{ticker} — Wedge Pop Strategy")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Trades", perf.total_trades)
m2.metric(
    "Win Rate",
    f"{perf.win_rate:.0%}" if perf.total_trades else "—",
)
m3.metric("Total Return", f"{perf.total_return_pct:.2%}")
m4.metric("Final Capital", f"${perf.final_capital:,.0f}")

m5, m6, m7, m8 = st.columns(4)
m5.metric("Initial Capital", f"${perf.initial_capital:,.0f}")
m6.metric("Avg Win", f"{perf.avg_win_pct:.2%}" if perf.trades else "—")
m7.metric("Avg Loss", f"{perf.avg_loss_pct:.2%}" if perf.trades else "—")
m8.metric("Max Drawdown", f"{perf.max_drawdown_pct:.2%}")

# --- Candlestick with trade markers + trigger lines ---
chart_builder = PlotlyChartBuilder()
trade_fig = chart_builder.build_candlestick_with_trades(
    df,
    perf.trades,
    title=f"{ticker} — Buy / Sell / Stop",
)

# Overlay trigger lines during each trade's holding period so the user
# can see exactly where the exhaustion / climax exits sit — makes
# ATR-based multipliers visually intuitive.
df_ind = strategy._with_indicators(df)
if show_swing_overlay and "swing_high" not in df_ind.columns:
    from pattern.helpers.pivots import find_swing_highs, find_swing_lows

    df_ind = df_ind.copy()
    df_ind["swing_high"] = find_swing_highs(
        df_ind, left=int(swing_pivot_left_ui), right=int(swing_pivot_right_ui)
    )
    df_ind["swing_low"] = find_swing_lows(
        df_ind, left=int(swing_pivot_left_ui), right=int(swing_pivot_right_ui)
    )
if df_ind.index.tz is not None:
    df_ind = df_ind.copy()
    df_ind.index = df_ind.index.tz_localize(None)
date_map = {idx.date(): idx for idx in df_ind.index}

# Pre-compute exhaustion-detector fire dates once (causal per the
# detector's docstring) so each trade can highlight them in O(1).
exit_fire_dates: set = set()
if enable_exh_exit:
    try:
        exit_fire_dates = {s.date for s in exit_detector.detect(df_ind)}
    except Exception:
        exit_fire_dates = set()

# ------------------------------------------------------------
# Global overlays — drawn once across the entire chart,
# independent of any trade window. These represent "what the
# strategy sees at every bar" (pivots, rolling filter-resistance,
# all potential resistance stops, exhaustion fires) so the user
# can evaluate the signal landscape even outside actual trades.
# ------------------------------------------------------------
if "swing_high" in df_ind.columns:
    # Swing high markers — shown only when an active filter/exit
    # consumes the swing_high column (otherwise visual noise).
    swing_high_users = (
        enable_swing_resistance
        or enable_resistance_break_exit
        or enable_swing_breakout
    )
    if swing_high_users:
        sh_all = df_ind["swing_high"].dropna()
        if len(sh_all) > 0:
            trade_fig.add_trace(
                go.Scatter(
                    x=sh_all.index,
                    y=sh_all.values,
                    mode="markers",
                    marker=dict(symbol="triangle-down", color="#E53935", size=9),
                    name="Swing High",
                    hovertemplate="Swing High: $%{y:.2f}<extra></extra>",
                ),
                row=1,
                col=1,
            )
    else:
        sh_all = df_ind["swing_high"].dropna()  # needed below if flags change

    # Swing low markers — only the trendline exit uses swing_low.
    if enable_trendline_exit:
        sl_all = df_ind["swing_low"].dropna()
        if len(sl_all) > 0:
            trade_fig.add_trace(
                go.Scatter(
                    x=sl_all.index,
                    y=sl_all.values,
                    mode="markers",
                    marker=dict(symbol="triangle-up", color="#43A047", size=9),
                    name="Swing Low",
                    hovertemplate="Swing Low: $%{y:.2f}<extra></extra>",
                ),
                row=1,
                col=1,
            )

    # Resistance Stop lines — every confirmed swing high extends
    # rightward for ``lookback + right`` bars (the window in which
    # it remains a valid stop level), independent of trades.
    if enable_resistance_break_exit:
        lookback_total = int(swing_pivot_lookback_ui) + int(swing_pivot_right_ui)
        n = len(df_ind)
        first = True
        for ts, level in sh_all.items():
            pivot_loc = df_ind.index.get_loc(ts)
            end_loc = min(n - 1, pivot_loc + lookback_total)
            trade_fig.add_trace(
                go.Scatter(
                    x=[ts, df_ind.index[end_loc]],
                    y=[level, level],
                    mode="lines",
                    line=dict(color="#FB8C00", width=1.2, dash="dashdot"),
                    name="Resistance Stop",
                    showlegend=first,
                    legendgroup="res_stops",
                    opacity=0.55,
                    hovertemplate="Resistance Stop: $%{y:.2f}<extra></extra>",
                ),
                row=1,
                col=1,
            )
            first = False

    # Swing Resistance (filter) — rolling MAX of confirmable swing
    # highs. Plotted as a step-line across the whole chart so the
    # user can see where the filter would have blocked entries at
    # any moment in time.
    if enable_swing_resistance:
        filter_xs = []
        filter_ys = []
        for i in range(len(df_ind)):
            res = recent_swing_high(
                df_ind["swing_high"],
                upto_idx=i,
                lookback=int(swing_pivot_lookback_ui),
                right=int(swing_pivot_right_ui),
            )
            if res is not None:
                _, pivot_price = res
                filter_xs.append(df_ind.index[i])
                filter_ys.append(pivot_price)
        if filter_xs:
            trade_fig.add_trace(
                go.Scatter(
                    x=filter_xs,
                    y=filter_ys,
                    mode="lines",
                    line=dict(color="#E53935", width=1.4, dash="dot"),
                    name="Swing Resistance (filter)",
                    hovertemplate="Swing Resistance: $%{y:.2f}<extra></extra>",
                ),
                row=1,
                col=1,
            )

# HL trendline — global rolling fit. At every bar, compute the
# trendline that the trendline-exit rule would use, then plot as
# a continuous line across the whole chart. A gap in the line
# means the fit is unavailable at that bar (< 2 pivots) or slope
# ≤ 0 (no higher-low structure to defend).
if enable_trendline_exit and "swing_low" in df_ind.columns:
    tl_x = []
    tl_y = []
    for i in range(len(df_ind)):
        tl = fit_lower_trendline(
            df_ind["swing_low"],
            upto_idx=i,
            lookback=int(swing_pivot_lookback_ui),
            right=int(swing_pivot_right_ui),
            max_points=int(trendline_max_pivots_ui),
        )
        if tl is None:
            continue
        slope, intercept, _ = tl
        if slope <= 0:
            continue
        tl_x.append(df_ind.index[i])
        tl_y.append(slope * i + intercept)
    if tl_x:
        trade_fig.add_trace(
            go.Scatter(
                x=tl_x,
                y=tl_y,
                mode="lines+markers",
                line=dict(color="#FF1744", width=4.5, dash="solid"),
                marker=dict(size=4, color="#FF1744"),
                name="HL Trendline",
                hovertemplate="HL Trendline: $%{y:.2f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

# Exhaustion Extension Top — always plot the continuous threshold
# so the user can see "what price needs to cross" at every bar,
# regardless of whether a trade is open or the detector fires.
# Line = ``ema_fast + extension_atr_mult × ATR`` = the primary-path
# high-extension threshold. Rejection-override path uses
# ``× rejection_leniency`` (0.9 default), drawn as a secondary
# fainter line below. Fire markers (⭐) remain on top to mark bars
# where ALL confirming conditions (slope / close-loc / sell-dom) lined
# up, not just the extension check.
if enable_exh_exit and "atr" in df_ind.columns:
    ema_fast_col = (
        "ema_trail" if "ema_trail" in df_ind.columns else "ema_fast"
    )
    if ema_fast_col in df_ind.columns:
        ema_f = df_ind[ema_fast_col]
        atr_col = df_ind["atr"]
        primary_thr = ema_f + float(exh_exit_extension_atr) * atr_col
        trade_fig.add_trace(
            go.Scatter(
                x=primary_thr.index,
                y=primary_thr.values,
                mode="lines",
                line=dict(color="#FFC107", width=2.0, dash="dash"),
                name=f"Exh primary ({exh_exit_extension_atr}×ATR)",
                hovertemplate="Exh primary: $%{y:.2f}<extra></extra>",
            ),
            row=1,
            col=1,
        )
        if exh_exit_rejection_override:
            # 0.9 = ExhaustionExtensionTopDetector default leniency.
            rejection_thr = ema_f + (
                float(exh_exit_extension_atr) * 0.9 * atr_col
            )
            trade_fig.add_trace(
                go.Scatter(
                    x=rejection_thr.index,
                    y=rejection_thr.values,
                    mode="lines",
                    line=dict(color="#FFE082", width=1.2, dash="dot"),
                    name=f"Exh rejection ({exh_exit_extension_atr * 0.9:.2f}×ATR)",
                    hovertemplate="Exh rejection: $%{y:.2f}<extra></extra>",
                ),
                row=1,
                col=1,
            )

# Star markers at bars where the detector fully fires (all confirms
# satisfied, not just the extension threshold). Rendered on top of
# the threshold line.
if enable_exh_exit and exit_fire_dates:
    exh_x = []
    exh_y = []
    for d in sorted(exit_fire_dates):
        ts = date_map.get(d)
        if ts is None:
            continue
        exh_x.append(ts)
        exh_y.append(float(df_ind.loc[ts, "High"]) * 1.005)
    if exh_x:
        trade_fig.add_trace(
            go.Scatter(
                x=exh_x,
                y=exh_y,
                mode="markers",
                marker=dict(
                    symbol="star",
                    size=18,
                    color="#FFEB3B",
                    line=dict(width=2, color="#B71C1C"),
                ),
                name="Exhaustion Fire",
                hovertemplate="Exhaustion Fire @ $%{y:.2f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

for t in perf.trades:
    if t.entry_date not in date_map or t.exit_date not in date_map:
        continue
    entry_ts = date_map[t.entry_date]
    exit_ts = date_map[t.exit_date]
    entry_loc = df_ind.index.get_loc(entry_ts)
    exit_loc = df_ind.index.get_loc(exit_ts)
    trade_slice = df_ind.iloc[entry_loc : exit_loc + 1]

    atr_s = trade_slice["atr"]

    # Smart Trail (Chandelier) line — only when enabled
    if use_smart_trail and len(trade_slice) > 0:
        entry_p = t.entry_price
        init_risk = entry_p - t.stop_loss
        if init_risk > 0:
            highs = trade_slice["High"]
            highest = highs.expanding().max()
            closes = trade_slice["Close"]
            unrealized_r = (closes - entry_p) / init_risk
            eff_mult = unrealized_r.apply(lambda r: 5.0 if r >= 4.0 else (4.0 if r >= 2.0 else 3.0))
            chandelier = highest - eff_mult * atr_s
            # Only show from bar 3 onward (min_trail_bars)
            chandelier = chandelier.iloc[3:] if len(chandelier) > 3 else chandelier.iloc[0:0]
            if len(chandelier) > 0:
                trade_fig.add_trace(
                    go.Scatter(
                        x=chandelier.index,
                        y=chandelier,
                        mode="lines",
                        line=dict(color="#2196F3", width=1.5, dash="dashdot"),
                        name="Smart Trail",
                        showlegend=(t == perf.trades[0]),
                        hovertemplate="Smart Trail: $%{y:.2f}<extra></extra>",
                    ),
                    row=1,
                    col=1,
                )

# Zoom the chart to the user's date range while keeping the warmup
# bars accessible via scroll/pan (needed for converged 50/200 SMA).
trade_fig.update_xaxes(range=[str(start_date), str(end_date)])
st.plotly_chart(trade_fig, use_container_width=True)

# --- Weekly & Yearly context charts ---
st.subheader("주봉 / 연봉 컨텍스트")

# Weekly: resample the already-fetched daily df (warmup + user range)
# to weekly OHLCV. Same trade markers + volume + MAs as the daily
# chart, zoomed to the user's date range.
agg = {
    "Open": "first",
    "High": "max",
    "Low": "min",
    "Close": "last",
    "Volume": "sum",
}
df_weekly = df.resample("W").agg(agg).dropna()
w_fig = chart_builder.build_candlestick_with_trades(
    df_weekly,
    perf.trades,
    title=f"{ticker} — Weekly",
)
w_fig.update_xaxes(range=[str(start_date), str(end_date)])
st.plotly_chart(w_fig, use_container_width=True)

# Yearly: needs a longer history (~10y) to have enough bars to be
# meaningful. Plain candlestick — no trades/volume (one year per bar
# makes intra-year trade markers meaningless).
context_fetch_start = end_date - timedelta(days=365 * 10)
with st.spinner(f"Fetching {ticker} long history for yearly chart..."):
    try:
        df_long = market_data.fetch_ohlcv(ticker, context_fetch_start, end_date)
    except Exception as e:
        st.warning(f"Long-history fetch failed: {e}")
        df_long = df

if df_long is not None and not df_long.empty:
    df_yearly = df_long.resample("YE").agg(agg).dropna()
    y_fig = chart_builder.build_simple_candlestick(df_yearly, title=f"{ticker} — Yearly")
    st.plotly_chart(y_fig, use_container_width=True)

if not perf.trades:
    st.info("이 기간엔 Wedge Pop 신호가 발생하지 않았어.")
    st.stop()

# --- Equity curve ---
if len(result.equity_curve) > 1:
    eq_fig = chart_builder.build_equity_curve(result.equity_curve, title="Equity Curve")
    st.plotly_chart(eq_fig, use_container_width=True)

# --- Trade table with dollar values ---
st.subheader("Trades")
EXIT_REASON_LABELS = {
    "exhaustion_exit": "Exhaustion Extension Top",
    "trendline_break": "Higher-Low Trendline Break",
    "smart_trail": "Smart Trail (Chandelier)",
    "resistance_break": "Resistance Break (false breakout)",
    "breakeven_stop": "Break-even Stop (≥1R unrealized)",
    "end_of_data": "End of Data (no exit fired)",
}

trade_rows = []
for t in perf.trades:
    outcome = "WIN" if t.pnl > 0 else "LOSS"
    trade_rows.append(
        {
            "Entry Date": t.entry_date,
            "Exit Date": t.exit_date,
            "Outcome": outcome,
            "Exit Reason": EXIT_REASON_LABELS.get(t.exit_reason, t.exit_reason),
            "Entry Price": f"${t.entry_price:,.2f}",
            "Exit Price": f"${t.exit_price:,.2f}",
            "Stop Loss": f"${t.stop_loss:,.2f}",
            "Shares": t.shares,
            "Cost": f"${t.entry_price * t.shares:,.0f}",
            "Proceeds": f"${t.exit_price * t.shares:,.0f}",
            "P&L ($)": f"${t.pnl:,.2f}",
            "P&L (%)": f"{t.pnl_pct:.2%}",
        }
    )
st.dataframe(trade_rows, use_container_width=True)
