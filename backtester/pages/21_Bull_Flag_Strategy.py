"""Bull Flag (Ross Cameron) single-ticker intraday backtest page.

영상 ``m5zu_X-_51I``의 Bull Flag 셋업을 그대로 구현. 4가지 필수
필터(News 제외):

  1. Pre-market gap +2% 이상
  2. 5x Relative Volume (50일 평균 대비)
  3. 주가 $2 ~ $20
  4. Float < 10M shares  (yfinance ``floatShares`` 사용)

세션 게이트 통과 + Bull Flag geometry (pole + 50% 풀백 + 첫 신고가
캔들) 매칭 시 진입. Stop = 풀백 저점, Target = HoD 재돌파.
선택적으로 +1R 도달 시 더블링 + BE stop (영상의 add-to-winner).
"""

from datetime import date, time, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from data.adapters.composed_fundamentals import build_default_fundamentals
from data.adapters.composed_market_data import build_default_market_data
from data.adapters.regular_session_filter import RegularSessionFilterAdapter
from data.domain.market_calendar import NY
from pattern.adapters.bull_flag import BullFlagDetector
from strategy.adapters.bull_flag_strategy import BullFlagStrategy
from strategy.domain.models import StrategyConfig, TossFeeSchedule

INTRADAY_HISTORY_FLOOR = date(2022, 1, 1)

st.set_page_config(page_title="Bull Flag Strategy", layout="wide")
st.title("Bull Flag (Ross Cameron) — Intraday")
st.caption(
    "Pre-market gap ≥ +2% / RVOL ≥ 5x / Price ∈ [$2, $20] / Float < 10M 의 "
    "**4가지 supply-demand 필터**를 통과한 세션에서만 인트라데이 Bull Flag "
    "(폴 5~10% 상승 → 50% 안쪽 풀백 → 첫 신고가 캔들) 진입. "
    "Stop = 풀백 저점, Target = HoD. 옵션: +1R에서 더블링 + BE stop. "
    "**영상 시간 프레임 = 1분봉 (default)** — 10초/5분/15분도 사용 가능 (영상 46:17~47:02)."
)
# 영상에서 직접 거론된 ticker 예시 (자막 grep 기준):
#   - OSR  : 영상 촬영 당일 +120% (\$2.50→\$5.50, 20분), \$12,227 trade
#   - ATNF : +564% gap-up, 5800만주, \$98,754 day
#   - MLGO : Float ~800K, 3억주 거래, +430%
#   - IMTE : 비슷한 저-Float 케이스
#   - Ford : counter-example (라지캡 횡보 — 절대 트레이드 X)
EXAMPLE_TICKERS_FROM_VIDEO = ["OSR", "ATNF", "MLGO", "IMTE"]

# ---- Sidebar -------------------------------------------------------
with st.sidebar:
    st.header("Market")
    if "bf_ticker" not in st.session_state:
        # 영상에서 직접 거론된 첫 번째 사례. 다른 후보는
        # EXAMPLE_TICKERS_FROM_VIDEO 참고. 데이터 가용 시점에 따라
        # 결과가 다를 수 있음 — 영상 시점 패턴은 이미 사라졌을 수 있다.
        st.session_state.bf_ticker = "OSR"
    ticker = st.text_input(
        "Ticker", key="bf_ticker",
        help=(
            "영상 직접 인용 사례: " + ", ".join(EXAMPLE_TICKERS_FROM_VIDEO)
            + ". 영상 시간 프레임 = 1분봉 (default)."
        ),
    )
    st.caption(
        "💡 **영상에서 거론된 종목**: "
        + " · ".join(f"`{t}`" for t in EXAMPLE_TICKERS_FROM_VIDEO)
        + "  ·  ❌ Counter-example: `F` (Ford — 라지캡 횡보)"
    )
    market = NY  # 영상은 US 시장 전용

    start_date = st.date_input(
        "Start Date",
        value=date.today() - timedelta(days=30),
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
        help="자본의 몇 %를 거래당 risk로 노출할지. 영상의 quarter-cushion 룰을 " "근사하려면 1~2%로 시작.",
    )
    max_position_pct = st.number_input(
        "Max position (% of equity)",
        value=30.0,
        min_value=1.0,
        max_value=100.0,
        step=5.0,
        help="단일 트레이드 notional 상한.",
    )

    st.header("Stock Selection (4 criteria — 영상 핵심)")
    st.caption("영상의 5 criteria 중 News를 제외한 4개. **모두 만족하지 않으면 그날은 진입 자체를 막음**.")
    min_gap_pct = st.number_input(
        "1) Min pre-market gap (%)",
        value=2.0,
        min_value=0.0,
        max_value=50.0,
        step=0.5,
        format="%.1f",
        help="영상: '최소 +2%, 이상적 +10%'. 백테스트 후보 부족 시 2%, " "엄격하게는 10%.",
    )
    min_rvol = st.number_input(
        "2) Min Relative Volume (×)",
        value=5.0,
        min_value=1.0,
        max_value=20.0,
        step=0.5,
        format="%.1f",
        help="오늘 거래량 / 50일 평균 거래량.",
    )
    min_price = st.number_input(
        "3a) Min price ($)",
        value=2.0,
        min_value=0.5,
        max_value=100.0,
        step=0.5,
    )
    max_price = st.number_input(
        "3b) Max price ($)",
        value=20.0,
        min_value=1.0,
        max_value=500.0,
        step=1.0,
    )
    max_float_shares_mil = st.number_input(
        "4) Max float (millions of shares)",
        value=10.0,
        min_value=0.5,
        max_value=500.0,
        step=1.0,
        help="영상 기본 10M. 더 빡빡하게는 5M. yfinance ``floatShares`` 사용.",
    )
    require_float = st.checkbox(
        "Require float data (off = float 모르면 통과)",
        value=True,
        help="True (기본): float 데이터 못 받으면 그 종목은 trade 안 함 — "
        "영상의 가장 핵심 필터라 보수적으로. False: 데이터 결측 시 필터 스킵.",
    )

    st.header("Pattern (Bull Flag geometry)")
    pole_lookback = st.number_input(
        "Pole lookback (bars)",
        value=7,
        min_value=2,
        max_value=30,
        step=1,
        help="breakout 후보 직전 N봉 안에서 폴 검출. " "영상: '5~7개의 그린 캔들'.",
    )
    pole_min_pct = st.number_input(
        "Pole min rise (%)",
        value=8.0,
        min_value=0.5,
        max_value=50.0,
        step=0.5,
        format="%.1f",
        help="폴 시작점 저점 → 끝점 고점 누적 상승률. 영상은 '5% 8% 10%' "
        "예시 — 4%는 데이터 sparse 보정용 lower bound.",
    )
    pole_min_green_bars = st.number_input(
        "Pole min green bars",
        value=3,
        min_value=1,
        max_value=20,
        step=1,
        help="폴 구간 내 bullish 봉(close>open) 최소 수. 영상의 '5-7 candles' "
        "는 패턴 전체(pole+flag+breakout) 기준 — 폴 자체는 3 green이 영상 도식과 일치.",
    )
    min_bar_range = st.number_input(
        "Min bar range ($) — zero-range filter",
        value=0.001,
        min_value=0.0,
        max_value=1.0,
        step=0.001,
        format="%.3f",
        help="봉의 high-low가 이 값 이하면 \"무거래 doji\"로 보고 폴/풀백 후보에서 거름. "
        "데이터 품질 fix — Ross 화면엔 doji 없음.",
    )
    max_bar_gap_seconds = st.number_input(
        "Max bar gap (seconds)",
        value=90,
        min_value=30,
        max_value=600,
        step=15,
        help="폴+풀백+돌파 윈도우 내 인접 봉 시간차 상한. 1m차트에서 90s = 한 봉 갭까지 허용. "
        "데이터 품질 fix — Ross 화면엔 분봉 갭 없음.",
    )
    flag_max_bars = st.number_input(
        "Flag max bars (pullback length)",
        value=4,
        min_value=1,
        max_value=10,
        step=1,
        help="풀백을 구성하는 최대 봉 수. 영상: '1~3개', 백테스트 default 4 "
        "(첫 풀백이 4봉까지 지속되는 케이스 자주 발생).",
    )
    flag_max_retrace = st.number_input(
        "Flag max retrace (% of pole)",
        value=70.0,
        min_value=10.0,
        max_value=99.0,
        step=5.0,
        format="%.0f",
        help="풀백 저점이 폴 상승분의 N% 이내로 유지되어야 bullish. "
        "**영상 룰: 50%** — 하지만 실제 1m 시장에서는 너무 빡빡해 거의 "
        "매칭 0건. 백테스트 default 70% (SPRC 4/21에서 50%로는 0 signal, "
        "70%로 2 signal 발견됨). 영상 정통성을 원하면 50으로 낮춰서 시도.",
    )
    latest_entry_hour = st.number_input(
        "Latest entry hour (ET)",
        value=12,
        min_value=10,
        max_value=15,
        step=1,
        help="이 시각 이후엔 신규 진입 안 함. 영상 sweet spot = 개장 후 1~2시간 "
        "(09:30 + 2h = 11:30). 백테스트 default 12시 — 두 번째 풀백이 11:30~12:00 "
        "사이에 자주 형성되는 케이스 catch.",
    )

    st.header("Quality Filters (CSV 분석 + Ross 영상 추가 룰)")
    max_rvol_input = st.number_input(
        "Max RVOL (over-extended cap)",
        value=30.0,
        min_value=5.0,
        max_value=100.0,
        step=5.0,
        format="%.1f",
        help="CSV 분석: ≥ 30x는 setup played out → 36% win. "
        "0으로 두면 비활성.",
    )
    max_gap_pct_input = st.number_input(
        "Max pre-market gap (%)",
        value=50.0,
        min_value=0.0,
        max_value=500.0,
        step=5.0,
        format="%.1f",
        help="CSV 분석: > 30% gap은 mean-reversion risk. 0이면 비활성.",
    )
    max_stop_dist_pct_input = st.number_input(
        "Max stop distance (%)",
        value=5.0,
        min_value=0.5,
        max_value=20.0,
        step=0.5,
        format="%.1f",
        help="entry → stop 거리. > 5% 이면 entry가 지지선에서 너무 멀어진 "
        "셋업으로 reject. CSV 분석에서 25% win rate.",
    )
    require_9ema = st.checkbox(
        "Require 9 EMA support (Ross 영상)",
        value=True,
        help='Ross: "I use 9 EMA on every timeframe". 풀백 저점이 9 EMA '
        "근방(±tolerance%)에 있어야 통과. False면 게이트 비활성.",
    )
    ema9_tol_pct = st.number_input(
        "9 EMA tolerance (%)",
        value=2.5,
        min_value=0.1,
        max_value=10.0,
        step=0.1,
        format="%.1f",
        disabled=not require_9ema,
    )
    also_accept_20ema = st.checkbox(
        "20 EMA fallback (opt-in)",
        value=False,
        help="9 EMA 못 닿더라도 20 EMA 근방이면 통과. Brett Burgett / "
        "Nathan Michaud 룰. Default OFF — sweep으로 효과 검증 권장.",
        disabled=not require_9ema,
    )
    require_daily_trend = st.checkbox(
        "Require daily uptrend (Ross 영상 X — sweep default OFF)",
        value=False,
        help='Ross: "stock should be in daily uptrend". 진입일 close > '
        "SMA{period}일 때만 통과.",
    )
    daily_sma_period = st.number_input(
        "Daily trend SMA period",
        value=50,
        min_value=10,
        max_value=200,
        step=10,
        disabled=not require_daily_trend,
    )
    max_nth_pullback = st.number_input(
        "Max N-th pullback",
        value=2,
        min_value=1,
        max_value=5,
        step=1,
        help='Ross: "1st/2nd pullback work well, 3rd start to be cautious". '
        "같은 세션 내 N번째 풀백까지만 통과.",
    )
    use_premarket_high = st.checkbox(
        "Require entry > pre-market high (Ross 영상)",
        value=True,
        help='Ross: "breaking PM high = confirmation". 진입가가 PM high보다 '
        "위여야 통과. False면 비활성. PM 데이터 fetch는 자동.",
    )

    st.header("Exit / Sizing")
    target_min_r = st.number_input(
        "Min R/R required to enter",
        value=2.0,
        min_value=1.0,
        max_value=10.0,
        step=0.5,
        format="%.1f",
        help="(HoD - entry) / risk 가 이 값 미만이면 trade 자체를 거름. " "영상의 2:1 게이트.",
    )
    use_fixed_target = st.checkbox(
        "Use fixed R-multiple target instead of HoD",
        value=True,
        help="기본(on): 익절가 = ``entry + R × risk`` 고정 (백테스트 default). "
        "off로 끄면 HoD를 target으로 — 영상의 'first target = retest of HoD' "
        "충실하지만, 영상은 HoD 도달 후 더 holding 하는데 우리는 HoD touch 시 "
        "전량 청산이라 winner를 작게 가져가게 됨. 백테스트에선 fixed R이 "
        "유리해서 default on.",
    )
    fixed_target_r = st.number_input(
        "Fixed target R-multiple",
        value=2.0,
        min_value=1.0,
        max_value=10.0,
        step=0.5,
        format="%.1f",
        disabled=not use_fixed_target,
        help="영상의 \"P/L ratio 2:1\" 룰 그대로. 3R로 올리면 winner는 더 크지만 "
        "TP 도달 빈도 ↓. SPRC 4/21에서 2R = 1 winner +$728 / 3R = TP 미도달.",
    )
    enable_add = st.checkbox(
        "Add to winner at +R (영상 doubling)",
        value=True,
        help="진입 후 +R 도달 시 동일 수량 추가 + 손절을 평단가(BE)로 끌어올림.",
    )
    add_at_r = st.number_input(
        "Add trigger (R)",
        value=1.5,
        min_value=0.3,
        max_value=3.0,
        step=0.1,
        format="%.1f",
        disabled=not enable_add,
        help="기본 1.5R — 1.0R은 진입 직후 wick 한 번에 트리거되어 BE stop 즉시 발화하는 케이스 다수 발생.",
    )
    add_confirm_on_close = st.checkbox(
        "Add: close-confirmed (wick filter)",
        value=True,
        disabled=not enable_add,
        help="True(기본): 봉 close가 +R 트리거 위에서 닫혀야 add 발화. "
        "False: high(wick) 기준. wick 한 번에 fake-add 트리거되는 것 차단.",
    )
    max_session_losses = st.number_input(
        "Max session losses (stop trading after N stop-outs)",
        value=1,
        min_value=0,
        max_value=10,
        step=1,
        help="한 세션에서 stop/BE-stop 누적 N회 시 그날 추가 진입 차단. "
        "영상의 'loss 후 quit' 룰. 0=비활성.",
    )
    be_stop_buffer_pct = st.number_input(
        "BE stop buffer (% below avg entry)",
        value=0.3,
        min_value=0.0,
        max_value=2.0,
        step=0.05,
        format="%.2f",
        help="add 후 BE stop을 평단가에서 N% 아래로 둠 — 평단가 정확히 "
        "한 번 tag에 즉시 청산되는 UGRO 케이스 방지. 기본 0.3% = $10 종목 3¢.",
    )

    st.header("Multi-Timeframe Alignment (DP4 영상)")
    st.caption(
        "영상 33:30~35:00: \"both 1m and 5m giving the same signal\". "
        "1m 신호 시점에 5m 차트에도 동일한 bull flag 신호가 있는지 검증. "
        "Ross의 VVPR 사례 — 1m 풀백 형성 + 5m 막 돌파 = alignment ✅."
    )
    enable_mtf = st.checkbox(
        "Enable MTF check (1m + 5m bull flag alignment)",
        value=False,
        help="False(기본 — 4/11~5/11 nasdaq_full sweep에서 영향 0 확인): "
        "같은 BullFlagDetector를 5m에도 돌려 두 분봉 모두에서 신호가 잡힌 "
        "시점만 통과. True로 켜면 영상 DP4 33:30~35:00 VVPR 정통 룰 적용 "
        "(Ross가 두 화면 동시에 보고 판단) — 보수적이라 trade 수 ↓.",
    )
    mtf_tolerance_seconds = st.number_input(
        "MTF time tolerance (seconds)",
        value=600,
        min_value=60, max_value=1800, step=60,
        disabled=not enable_mtf,
        help="1m 신호 ±N초 안에 5m 신호가 있어야 alignment 인정. "
        "**영상 정통 = 동시 발생** (Ross는 두 화면 동시에 보고 판단). "
        "5m bar = 300s 폭이라 1 bar 슬랙 = 600s (= ±10분) 이 합리적 default. "
        "더 빡빡하게 = 300s (정확히 1 bar 안). 더 느슨하게 = 900s (= ±15분).",
    )

    st.header("Reverse-split / Continuity Guard")
    split_blackout_days = st.number_input(
        "Split blackout days (±N)",
        value=30,
        min_value=0,
        max_value=180,
        step=5,
        help="split(forward/reverse) 발생일 전후 N거래일은 거름. 0=비활성. "
        "yfinance Ticker.splits 사용. UGRO처럼 reverse-split 직전후 "
        "데이터 불연속 차단.",
    )
    price_floor_lookback_days = st.number_input(
        "Price-floor lookback (days)",
        value=30,
        min_value=0,
        max_value=120,
        step=5,
        help="직전 N거래일 daily Low가 한 번이라도 ``min_price`` 미만이면 "
        "거름. \$0.41→\$7.35 점프 같은 split 흔적 자동 차단.",
    )

    run_btn = st.button(
        "Run Bull Flag Backtest",
        type="primary",
        use_container_width=True,
    )

if not run_btn:
    st.info(
        "좌측에서 ticker / 기간을 설정하고 **Run Bull Flag Backtest**를 눌러. "
        "영상은 소형 저-Float 모멘텀주(예: 바이오, IPO, 마이크로캡)에서 통하는 "
        "전략이므로 NVDA/QQQ 같은 라지캡에선 4가지 필터를 통과하는 세션이 거의 "
        "없을 수 있음 (의도된 동작)."
    )
    st.stop()

# ---- Data fetch ----------------------------------------------------
md_raw = build_default_market_data()  # PM bars 살아있음 — PM high 계산용
md = RegularSessionFilterAdapter(md_raw, market=market)

with st.spinner(f"Fetching 1m / 5m / daily for {ticker.upper()}..."):
    try:
        df_intraday = md.fetch_ohlcv(ticker.upper(), start_date, end_date, interval="1m")
        # 5m: MTF alignment 검증 + 시각화에 사용
        df_5m = md.fetch_ohlcv(ticker.upper(), start_date, end_date, interval="5m")
        # 50일 RVOL 계산을 위해 충분한 daily 히스토리 확보.
        # composed adapter 통해 yfinance primary + Massive fallback.
        df_daily = md.fetch_ohlcv(
            ticker.upper(), start_date - timedelta(days=120), end_date, interval="1d",
        )
        # PM bars (4:00 ~ 9:30 ET) — RegularSessionFilter 우회한 raw 데이터.
        # 페이지 옵션이 켜져 있을 때만 fetch.
        df_intraday_raw = None
        if bool(use_premarket_high):
            df_intraday_raw = md_raw.fetch_ohlcv(
                ticker.upper(), start_date, end_date, interval="1m",
            )
    except Exception as exc:
        st.error(f"Data fetch failed: {exc}")
        st.stop()

# 세션별 PM high 계산 (사용자가 PM high 게이트 켰을 때만).
premarket_high_by_date: dict = {}
if bool(use_premarket_high) and df_intraday_raw is not None and not df_intraday_raw.empty:
    _raw = df_intraday_raw
    if _raw.index.tz is None:
        _raw = _raw.copy()
        _raw.index = _raw.index.tz_localize(market.tz)
    elif str(_raw.index.tz) != market.tz:
        _raw = _raw.copy()
        _raw.index = _raw.index.tz_convert(market.tz)
    # 4:00 ~ 9:29 ET 범위만 → 세션 일자별 max High
    _pm = _raw.between_time("04:00", "09:29")
    if not _pm.empty:
        for d, group in _pm.groupby(_pm.index.date):
            premarket_high_by_date[d] = float(group["High"].max())

if df_intraday is None or df_intraday.empty:
    st.warning("No 1m data returned for that range.")
    st.stop()
if df_intraday.index.tz is not None and str(df_intraday.index.tz) != market.tz:
    df_intraday = df_intraday.copy()
    df_intraday.index = df_intraday.index.tz_convert(market.tz)

# 5m 타임존 정합 (1m과 동일 처리)
if df_5m is not None and not df_5m.empty:
    if df_5m.index.tz is not None and str(df_5m.index.tz) != market.tz:
        df_5m = df_5m.copy()
        df_5m.index = df_5m.index.tz_convert(market.tz)
else:
    df_5m = None  # 데이터 없으면 MTF check 자동 비활성화

# ---- Float + splits lookup -----------------------------------------
# EODHD primary + Massive fallback + 7-day disk cache. yfinance
# rate-limit에 의존하지 않음.
float_shares: float | None = None
splits_series = None
with st.spinner(f"Fetching float / splits for {ticker.upper()}..."):
    try:
        fund = build_default_fundamentals().fetch(ticker.upper())
        float_shares = fund.float_shares
        splits_series = fund.splits
    except Exception as exc:
        st.warning(f"Lookup failed: {exc} — float/split filters may behave conservatively.")

# ---- 4 criteria summary --------------------------------------------
st.subheader(f"{ticker.upper()} — Stock Selection 진단")
c1, c2, c3, c4 = st.columns(4)
c1.metric(
    "Float",
    f"{float_shares/1e6:.1f}M" if float_shares else "—",
    delta=(
        "✅ < cap"
        if float_shares and float_shares < max_float_shares_mil * 1e6
        else "❌ over cap" if float_shares else "no data"
    ),
)
# 일별 게이트 통과 세션 수 미리 계산
_prev = df_daily["Close"].shift(1)
_gap = (df_daily["Open"] - _prev) / _prev
_avg_vol = df_daily["Volume"].rolling(50, min_periods=20).mean()
_rvol = df_daily["Volume"] / _avg_vol
_in_window = (df_daily.index.date >= start_date) & (df_daily.index.date <= end_date)
_qualifying = (
    (_gap >= min_gap_pct / 100.0)
    & (_rvol >= min_rvol)
    & (df_daily["Open"] >= min_price)
    & (df_daily["Open"] <= max_price)
    & _in_window
)
n_qual = int(_qualifying.sum())
n_total = int(_in_window.sum())
c2.metric(
    "Qualifying sessions",
    f"{n_qual} / {n_total}",
    help="요청한 기간 안에서 4 criteria(gap/RVOL/price/float)를 모두 만족한 세션 수.",
)
if n_qual > 0 and not _qualifying[_qualifying].empty:
    avg_gap = float(_gap[_qualifying].mean())
    avg_rvol = float(_rvol[_qualifying].mean())
    c3.metric("Avg gap (qual sessions)", f"{avg_gap*100:+.1f}%")
    c4.metric("Avg RVOL (qual sessions)", f"{avg_rvol:.1f}x")
else:
    c3.metric("Avg gap (qual sessions)", "—")
    c4.metric("Avg RVOL (qual sessions)", "—")

# ---- Build detector + strategy + run -------------------------------
detector = BullFlagDetector(
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
    splits=splits_series,
    split_blackout_days=int(split_blackout_days),
    price_floor_lookback_days=int(price_floor_lookback_days),
    enable_mtf_check=bool(enable_mtf and df_5m is not None),
    mtf_tolerance_seconds=int(mtf_tolerance_seconds),
    min_bar_range=float(min_bar_range),
    max_bar_gap_seconds=int(max_bar_gap_seconds),
    # ---- New quality filters ----
    max_rvol=float(max_rvol_input) if max_rvol_input > 0 else None,
    max_gap_pct=float(max_gap_pct_input) / 100.0 if max_gap_pct_input > 0 else None,
    max_stop_distance_pct=float(max_stop_dist_pct_input) / 100.0,
    require_9ema_support=bool(require_9ema),
    ema9_tolerance_pct=float(ema9_tol_pct) / 100.0,
    also_accept_20ema_support=bool(also_accept_20ema),
    require_daily_trend=bool(require_daily_trend),
    daily_trend_sma_period=int(daily_sma_period),
    max_nth_pullback=int(max_nth_pullback),
    premarket_high_by_date=premarket_high_by_date if use_premarket_high else None,
)
strategy = BullFlagStrategy(
    detector=detector,
    max_position_pct_of_equity=float(max_position_pct) / 100.0,
    target_min_r_multiple=float(target_min_r),
    target_at_r_multiple=float(fixed_target_r) if use_fixed_target else None,
    enable_add_to_winner=bool(enable_add),
    add_at_r=float(add_at_r),
    add_confirm_on_close=bool(add_confirm_on_close),
    max_session_losses=int(max_session_losses),
    be_stop_buffer_pct=float(be_stop_buffer_pct) / 100.0,
    fee_schedule=TossFeeSchedule(),  # Toss 증권 수수료 (buy/sell 0.1% + SEC 0.0023%)
)

config = StrategyConfig(
    ticker=ticker.upper(),
    start_date=start_date,
    end_date=end_date,
    pattern_name="bull_flag",
    initial_capital=float(initial_capital),
    risk_per_trade=risk_pct / 100.0,
    max_holding_days=1,
)

with st.spinner("Running backtest..."):
    try:
        result = strategy.run(df_intraday, df_daily, config, df_5m=df_5m)
    except Exception as exc:
        st.error(f"Backtest failed: {exc}")
        st.stop()

perf = result.performance
all_signals = detector.detect(df_intraday, df_daily, df_5m=df_5m)
# MTF check 진단 (filter 적용 전 raw signals)
chart_signals_pre_mtf = detector.detect(df_intraday, df_daily, df_5m=None) if enable_mtf else all_signals

# 차트에는 strategy가 실제로 진입한 신호만 표시.
# (max_session_losses 차단, 다른 포지션 open 중, R/R gate 등으로 skip된
#  신호는 chart에서 숨김. 사용자가 헷갈리지 않도록.)
def _ts_to_ny_naive(ts):
    """tz-aware/naive 모두 NY 시간 naive datetime으로 정규화."""
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        return t
    return t.tz_convert("America/New_York").tz_localize(None)

_taken_ts = {_ts_to_ny_naive(t.entry_ts) for t in perf.trades}
chart_signals = [s for s in all_signals if _ts_to_ny_naive(s.entry_ts) in _taken_ts]

# 5m chart에 표시할 독립 5m bull flag 신호들
# (1m 신호와 매칭된 5m 신호를 강조 표시하기 위해 별도 계산)
chart_signals_5m: list = []
if df_5m is not None and not df_5m.empty:
    det_5m_only = BullFlagDetector(
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
        splits=splits_series,
        split_blackout_days=int(split_blackout_days),
        price_floor_lookback_days=int(price_floor_lookback_days),
        # 5m bar gap은 ~300s, slack 100s 더해 400s
        max_bar_gap_seconds=400,
        min_bar_range=float(min_bar_range),
    )
    chart_signals_5m = det_5m_only.detect(df_5m, df_daily, df_5m=None)

# 1m signal 별로 매칭된 5m signal 찾기 (timestamp ±tolerance 안)
def _find_matching_5m_signal(sig_1m, sigs_5m_list, tol_sec):
    sig_ts = pd.Timestamp(sig_1m.entry_ts)
    best = None
    best_gap = None
    for s5 in sigs_5m_list:
        if s5.session_date != sig_ts.date():
            continue
        s5_ts = pd.Timestamp(s5.entry_ts)
        gap = abs((sig_ts - s5_ts).total_seconds())
        if gap <= tol_sec and (best_gap is None or gap < best_gap):
            best = s5; best_gap = gap
    return best

# ---- Headline metrics ----------------------------------------------
st.subheader(f"{ticker.upper()} — Bull Flag Backtest")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Trades", perf.total_trades)
m2.metric(
    "Win Rate",
    f"{perf.win_rate:.0%}" if perf.total_trades else "—",
    help="영상 본인 기준 71.4% — 4 criteria가 빡빡할수록 winrate ↑.",
)
m3.metric("Total Return", f"{perf.total_return_pct:+.2%}")
m4.metric("Final Capital", f"${perf.final_capital:,.0f}")
m5, m6, m7, m8 = st.columns(4)
m5.metric("Avg Win", f"{perf.avg_win_pct:+.2%}" if perf.trades else "—")
m6.metric("Avg Loss", f"{perf.avg_loss_pct:+.2%}" if perf.trades else "—")
if perf.trades:
    avg_win_dol = sum(t.pnl for t in perf.trades if t.pnl > 0) / max(
        1, sum(1 for t in perf.trades if t.pnl > 0)
    )
    avg_loss_dol = sum(t.pnl for t in perf.trades if t.pnl <= 0) / max(
        1, sum(1 for t in perf.trades if t.pnl <= 0)
    )
    pl_ratio = abs(avg_win_dol / avg_loss_dol) if avg_loss_dol else 0.0
    m7.metric(
        "P/L Ratio",
        f"{pl_ratio:.2f}:1",
        help="영상 본인 기준 ~2.4:1.",
    )
else:
    m7.metric("P/L Ratio", "—")
m8.metric("Max DD", f"{perf.max_drawdown_pct:.2%}")

# ---- Intraday chart with pattern overlays --------------------------
st.subheader("1m — Bull Flag overlays + Volume")

# 2-row subplot: 위 = 캔들 + 패턴 오버레이, 아래 = 거래량 막대 + 20봉 평균.
# x축 공유로 zoom/pan 동기화. 거래량 row를 크게 (35%) — 사용자 요청.
intra_fig = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.03,
    row_heights=[0.65, 0.35],
    subplot_titles=("", "Volume"),
)

# Row 1 — 캔들스틱
intra_fig.add_trace(
    go.Candlestick(
        x=df_intraday.index,
        open=df_intraday["Open"],
        high=df_intraday["High"],
        low=df_intraday["Low"],
        close=df_intraday["Close"],
        name="1m",
        showlegend=False,
    ),
    row=1, col=1,
)

# Row 2 — 거래량 막대 (그린/레드 candle별 색상 분리) + 20봉 평균선
_is_green = df_intraday["Close"] >= df_intraday["Open"]
_vol_colors = ["#26A69A" if g else "#EF5350" for g in _is_green]
intra_fig.add_trace(
    go.Bar(
        x=df_intraday.index,
        y=df_intraday["Volume"],
        marker=dict(color=_vol_colors),
        name="Volume",
        showlegend=False,
        hovertemplate="%{x|%Y-%m-%d %H:%M}<br>Volume: %{y:,.0f}<extra></extra>",
    ),
    row=2, col=1,
)
# 20봉 rolling average — Bull Flag breakout이 평균 위에서 나오는지
# 시각적으로 확인 가능 (영상의 "higher volume on the move up,
# lighter volume on the selling" 룰).
_vol_ma = df_intraday["Volume"].rolling(20, min_periods=5).mean()
intra_fig.add_trace(
    go.Scatter(
        x=_vol_ma.index,
        y=_vol_ma.values,
        mode="lines",
        line=dict(color="#FB8C00", width=1.5, dash="dash"),
        name="Volume MA20",
        showlegend=False,
        hovertemplate="%{x|%Y-%m-%d %H:%M}<br>MA20: %{y:,.0f}<extra></extra>",
    ),
    row=2, col=1,
)

# 9 EMA 오버레이 (DP4 영상 P0) — 풀백 지지선 시각화.
# Ross: "I use this on all my time frames 5 minute 1 minute daily."
_ema9_1m = df_intraday["Close"].ewm(span=9, adjust=False).mean()
intra_fig.add_trace(
    go.Scatter(
        x=_ema9_1m.index, y=_ema9_1m.values,
        mode="lines",
        line=dict(color="#9C27B0", width=1.5),
        name="9 EMA (1m)",
        hovertemplate="9 EMA: $%{y:.3f}<extra></extra>",
    ),
    row=1, col=1,
)

# 패턴 + entry/stop/target 마커 — 각 trade의 lifetime 안에서만 라인 그림.
# add_hline은 차트 전체 x축에 라인을 늘리는 문제가 있어서 add_trace + bounded
# x로 해당 trade 영역에만 그림.
_target_r = float(fixed_target_r) if use_fixed_target else None

# trade 별로 signal 매핑 (entry_ts 정규화로 매칭)
_sig_by_entry = {_ts_to_ny_naive(s.entry_ts): s for s in chart_signals}
_trade_by_entry = {_ts_to_ny_naive(t.entry_ts): t for t in perf.trades}

for sig in chart_signals:
    sig_key = _ts_to_ny_naive(sig.entry_ts)
    trade = _trade_by_entry.get(sig_key)
    if trade is None:
        continue  # 안전장치 — chart_signals는 이미 taken만 필터됨

    risk = sig.entry_price - sig.stop_loss
    target_price = (
        sig.entry_price + _target_r * risk if _target_r else sig.hod_at_entry
    )
    target_label = f"{_target_r:.1f}R" if _target_r else "HoD"

    # 라인 x 범위.
    # - Entry/Target 라인: **pole_end_ts → exit_ts** — entry 가격을 정의한
    #   폴 고점 봉부터 그려서 "어떤 캔들이 entry trigger 기준인지" 시각적
    #   으로 즉시 식별 가능. (이전엔 entry_ts부터라 어느 봉이 trigger인지
    #   알기 어려웠음.)
    # - Stop 라인: flag_low_ts → exit_ts — 풀백 저점이 손절 기준.
    line_x_start = sig.pole_end_ts
    # exit_ts는 naive datetime이라 tz 일치시키기
    exit_ts = pd.Timestamp(trade.exit_ts) if trade.exit_ts else pd.Timestamp(sig.entry_ts)
    if exit_ts.tzinfo is None and sig.entry_ts.tzinfo is not None:
        exit_ts = exit_ts.tz_localize(sig.entry_ts.tz)
    line_x_end = exit_ts
    stop_x_start = sig.flag_low_ts

    # 폴 (점선) — pole_start → pole_end
    intra_fig.add_trace(
        go.Scatter(
            x=[sig.pole_start_ts, sig.pole_end_ts],
            y=[sig.pole_start_price, sig.pole_end_price],
            mode="lines+markers",
            line=dict(color="#1976D2", width=2, dash="dot"),
            marker=dict(size=8, color="#1976D2"),
            name="Pole", showlegend=False,
            hovertemplate="Pole<br>%{x}<br>$%{y:.2f}<extra></extra>",
        ),
        row=1, col=1,
    )
    # ── ENTRY 라인 (pole_end_ts → exit_ts) ──
    intra_fig.add_trace(
        go.Scatter(
            x=[line_x_start, line_x_end],
            y=[sig.entry_price, sig.entry_price],
            mode="lines",
            line=dict(color="#2E7D32", width=2.5),
            name=f"Entry ${sig.entry_price:.3f}", showlegend=False,
            hovertemplate=f"Entry ${sig.entry_price:.3f}<extra></extra>",
        ),
        row=1, col=1,
    )
    # ── MAX LOSS 라인 (flag_low_ts → exit_ts) ──
    # Stop 가격은 풀백 저점부터 의미가 생기므로 flag_low_ts부터 그림.
    intra_fig.add_trace(
        go.Scatter(
            x=[stop_x_start, line_x_end],
            y=[sig.stop_loss, sig.stop_loss],
            mode="lines",
            line=dict(color="#C62828", width=2.5, dash="dash"),
            name=f"Max Loss ${sig.stop_loss:.3f}", showlegend=False,
            hovertemplate=f"Max Loss ${sig.stop_loss:.3f}<extra></extra>",
        ),
        row=1, col=1,
    )
    # ── PROFIT TARGET 라인 (pole_end_ts → exit_ts) ──
    intra_fig.add_trace(
        go.Scatter(
            x=[line_x_start, line_x_end],
            y=[target_price, target_price],
            mode="lines",
            line=dict(color="#1565C0", width=2.5, dash="dash"),
            name=f"Profit ({target_label}) ${target_price:.3f}", showlegend=False,
            hovertemplate=f"Profit ({target_label}) ${target_price:.3f}<extra></extra>",
        ),
        row=1, col=1,
    )
    # 라인 가격 라벨을 line 끝에 텍스트로 표시
    for y_val, color, label in (
        (sig.entry_price, "#2E7D32", f"Entry ${sig.entry_price:.3f}"),
        (sig.stop_loss, "#C62828", f"Stop ${sig.stop_loss:.3f}"),
        (target_price, "#1565C0", f"Target ${target_price:.3f}"),
    ):
        intra_fig.add_annotation(
            x=line_x_end, y=y_val,
            text=label, showarrow=False, xanchor="left", yanchor="middle",
            xshift=6, font=dict(color=color, size=11),
            row=1, col=1,
        )

    # 풀백 저점 마커 (실제 flag_low가 있는 봉의 low 위치)
    intra_fig.add_trace(
        go.Scatter(
            x=[sig.flag_low_ts], y=[sig.flag_low],
            mode="markers",
            marker=dict(symbol="triangle-down", color="#FB8C00", size=14),
            name="Flag low", showlegend=False,
            hovertemplate=f"Flag low @ %{{x}}<br>$%{{y:.3f}}<extra></extra>",
        ),
        row=1, col=1,
    )
    # 진입 마커 (entry_ts 위치)
    intra_fig.add_trace(
        go.Scatter(
            x=[sig.entry_ts], y=[sig.entry_price],
            mode="markers",
            marker=dict(
                symbol="triangle-up", color="#2E7D32", size=18,
                line=dict(width=2, color="white"),
            ),
            name="Entry", showlegend=False,
            hovertemplate=(
                f"Entry @ %{{x}}<br>$%{{y:.3f}}<br>"
                f"stop ${sig.stop_loss:.3f}  target ${target_price:.3f}"
                "<extra></extra>"
            ),
        ),
        row=1, col=1,
    )

# 트레이드 결과 마커 — exit 시점에 그려서 진입 봉과 헷갈리지 않게.
# (이전엔 entry_ts에 마커가 그려져 "같은 캔들에서 stop된 것처럼" 보이는 시각적 버그)
if perf.trades:
    win_x, win_y, win_text = [], [], []
    lose_x, lose_y, lose_text = [], [], []
    for t in perf.trades:
        exit_ts = t.exit_ts or pd.Timestamp(t.exit_date)
        text = (
            f"Entry: {t.entry_ts}<br>"
            f"Exit: {t.exit_ts} ({t.exit_reason})<br>"
            f"entry ${t.entry_price:.3f} → exit ${t.exit_price:.3f}<br>"
            f"stop ${t.stop_loss:.3f}<br>"
            f"PnL ${t.pnl:+,.0f} ({t.pnl_pct:+.2%})"
        )
        # 마커는 exit_price 위치에 정확히 — 라인(target/stop)과 *동일한 가격*에
        # 찍어야 시각적으로 "이 라인에 닿아서 청산"이 명확. 이전엔 ±0.5% 오프셋
        # 으로 라인에서 떨어져 그려져서 "익절 점이 익절 라인이랑 다르다"는
        # 사용자 컴플레인 발생.
        if t.pnl > 0:
            win_x.append(exit_ts)
            win_y.append(t.exit_price)
            win_text.append(text)
        else:
            lose_x.append(exit_ts)
            lose_y.append(t.exit_price)
            lose_text.append(text)
    if win_x:
        intra_fig.add_trace(
            go.Scatter(
                x=win_x,
                y=win_y,
                mode="markers",
                marker=dict(symbol="circle", color="#2E7D32", size=10),
                name="Win",
                text=win_text,
                hovertemplate="%{text}<extra></extra>",
            ),
            row=1, col=1,
        )
    if lose_x:
        intra_fig.add_trace(
            go.Scatter(
                x=lose_x,
                y=lose_y,
                mode="markers",
                marker=dict(symbol="circle", color="#C62828", size=10),
                name="Loss",
                text=lose_text,
                hovertemplate="%{text}<extra></extra>",
            ),
            row=1, col=1,
        )

# 사용자 요청: 차트 더 크게 + 거래량 부분도 잘 보이게 (이미 row_heights 0.65/0.35).
# x축 공유로 zoom 시 거래량 부분도 함께 zoom됨 (shared_xaxes=True).
intra_fig.update_layout(
    height=1000,  # 720 → 1000 (사용자 요청 "좀 크게")
    margin=dict(l=10, r=80, t=10, b=10),  # right margin 늘림 (annotation 표시용)
    xaxis_rangeslider_visible=False,
    showlegend=False,
)
intra_fig.update_yaxes(title_text="Price ($)", row=1, col=1)
intra_fig.update_yaxes(title_text="Volume", row=2, col=1)
st.plotly_chart(intra_fig, use_container_width=True)

# ---- 5m Multi-Timeframe Alignment chart -----------------------------
if df_5m is not None and not df_5m.empty:
    n_total_signals = len(chart_signals_pre_mtf) if enable_mtf else len(all_signals)
    n_mtf_passed = len(all_signals)
    n_taken = len(chart_signals)
    if enable_mtf:
        st.subheader(
            f"5m — Multi-Timeframe Alignment "
            f"(detected: {n_total_signals} → MTF passed: {n_mtf_passed} → "
            f"strategy taken: {n_taken})"
        )
        st.caption(
            f"MTF 게이트 (DP4 영상 33:30 정통 룰): 같은 BullFlagDetector를 5m에도 돌려 "
            f"1m 신호 ±**{int(mtf_tolerance_seconds)}**초 안에 5m 신호가 있으면 alignment ✅."
        )
    else:
        st.subheader("5m — Context Chart")
        st.caption("MTF check 비활성. 5m 차트는 컨텍스트 시각화 용도.")

    mtf_fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.65, 0.35],
        subplot_titles=("", "Volume (5m)"),
    )
    mtf_fig.add_trace(
        go.Candlestick(
            x=df_5m.index,
            open=df_5m["Open"], high=df_5m["High"],
            low=df_5m["Low"], close=df_5m["Close"],
            name="5m", showlegend=False,
        ),
        row=1, col=1,
    )
    # 5m 9 EMA — 1m과 동일한 색
    _ema9_5m = df_5m["Close"].ewm(span=9, adjust=False).mean()
    mtf_fig.add_trace(
        go.Scatter(
            x=_ema9_5m.index, y=_ema9_5m.values,
            mode="lines",
            line=dict(color="#9C27B0", width=1.8),
            name="9 EMA (5m)",
            hovertemplate="9 EMA: $%{y:.3f}<extra></extra>",
        ),
        row=1, col=1,
    )

    # 5m 거래량 막대
    _is_green_5m = df_5m["Close"] >= df_5m["Open"]
    _vol_colors_5m = ["#26A69A" if g else "#EF5350" for g in _is_green_5m]
    mtf_fig.add_trace(
        go.Bar(
            x=df_5m.index, y=df_5m["Volume"],
            marker=dict(color=_vol_colors_5m),
            name="Volume", showlegend=False,
            hovertemplate="%{x|%Y-%m-%d %H:%M}<br>Volume: %{y:,.0f}<extra></extra>",
        ),
        row=2, col=1,
    )

    # ── 5m bull flag 독립 신호들 — 폴/플래그/엔트리 마커 + textbook 라인 ──
    matched_5m_ts = set()  # 1m과 매칭된 5m 신호 timestamps
    for sig_1m in chart_signals:
        s5_match = _find_matching_5m_signal(
            sig_1m, chart_signals_5m, int(mtf_tolerance_seconds)
        )
        if s5_match is not None:
            matched_5m_ts.add(s5_match.entry_ts)

    for s5 in chart_signals_5m:
        is_matched = s5.entry_ts in matched_5m_ts
        # 1m과 매칭 안 된 5m 신호는 차트에 표시하지 않음 — "정확히 똑같이
        # 보이는 5m flag"가 1m 차트에도 있는 노이즈를 제거. MTF 게이트가
        # 통과시킨 alignment된 5m 신호만 시각화.
        if not is_matched:
            continue
        entry_color = "#2E7D32"
        marker_size = 22
        line_w = 3
        # 폴 라인
        mtf_fig.add_trace(
            go.Scatter(
                x=[s5.pole_start_ts, s5.pole_end_ts],
                y=[s5.pole_start_price, s5.pole_end_price],
                mode="lines+markers",
                line=dict(color="#1976D2", width=line_w, dash="dot"),
                marker=dict(size=8),
                name="5m Pole", showlegend=False,
                hovertemplate="5m Pole<br>%{x}<br>$%{y:.2f}<extra></extra>",
            ),
            row=1, col=1,
        )
        # Flag low 마커 (5m 차트는 MTF 검증 시각화 전용 —
        # 실제 entry/stop/target 라인은 1m 차트에만)
        mtf_fig.add_trace(
            go.Scatter(
                x=[s5.flag_low_ts], y=[s5.flag_low],
                mode="markers",
                marker=dict(symbol="triangle-down", color="#FB8C00", size=14),
                name="5m Flag low", showlegend=False,
                hovertemplate=f"5m Flag low<br>$%{{y:.3f}}<extra></extra>",
            ),
            row=1, col=1,
        )
        # Entry 마커
        mtf_fig.add_trace(
            go.Scatter(
                x=[s5.entry_ts], y=[s5.entry_price],
                mode="markers",
                marker=dict(
                    symbol="triangle-up", color=entry_color, size=marker_size,
                    line=dict(width=2, color="white"),
                ),
                name="5m Entry (MATCHED)", showlegend=False,
                hovertemplate=(
                    f"✅ MATCHED 5m Entry @ %{{x}}<br>$%{{y:.3f}}"
                    "<extra></extra>"
                ),
            ),
            row=1, col=1,
        )

    # ── 1m → 5m 매칭 연결선 (점선) — 어떤 1m 신호가 어떤 5m 신호와 짝지어졌는지 ──
    for sig_1m in chart_signals:
        s5_match = _find_matching_5m_signal(
            sig_1m, chart_signals_5m, int(mtf_tolerance_seconds)
        )
        if s5_match is None:
            continue
        # 1m signal price (top), 5m signal price (bottom on 5m chart)
        mtf_fig.add_trace(
            go.Scatter(
                x=[sig_1m.entry_ts, s5_match.entry_ts],
                y=[s5_match.entry_price * 1.02, s5_match.entry_price],
                mode="lines",
                line=dict(color="#7B1FA2", width=1.5, dash="dot"),
                name="1m↔5m link", showlegend=False,
                hovertemplate=(
                    f"MTF link<br>1m {sig_1m.entry_ts.strftime('%H:%M')} → "
                    f"5m {s5_match.entry_ts.strftime('%H:%M')}"
                    "<extra></extra>"
                ),
            ),
            row=1, col=1,
        )

    # MTF 거절 / strategy skip 신호는 차트에서 숨김 (사용자 요청).
    # 진단 카운트는 위 subheader에 detected/passed/taken 으로 노출.

    mtf_fig.update_layout(
        height=800,  # 520 → 800 (사용자 요청 "좀 크게")
        margin=dict(l=10, r=80, t=10, b=10),
        xaxis_rangeslider_visible=False,
        showlegend=False,
    )
    mtf_fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    mtf_fig.update_yaxes(title_text="Volume", row=2, col=1)
    st.plotly_chart(mtf_fig, use_container_width=True)
else:
    st.info("5m 데이터 없음 — Multi-Timeframe Alignment 차트 생략.")


# ---- Equity curve --------------------------------------------------
if len(result.equity_curve) > 1:
    st.subheader("Equity Curve")
    eq_df = pd.DataFrame(
        [(e.date, e.equity) for e in result.equity_curve],
        columns=["date", "equity"],
    )
    eq_fig = go.Figure(
        go.Scatter(
            x=eq_df["date"],
            y=eq_df["equity"],
            mode="lines",
            line=dict(color="#1976D2", width=2),
        )
    )
    eq_fig.update_layout(
        height=280,
        margin=dict(l=10, r=10, t=10, b=10),
        yaxis_title="Equity ($)",
    )
    st.plotly_chart(eq_fig, use_container_width=True)

# ---- Trade table ---------------------------------------------------
st.subheader("Trades")
if not perf.trades:
    st.info(
        "이 기간에 4 criteria + Bull Flag 매칭이 발생하지 않았어. "
        "영상 전략은 소형 저-Float 모멘텀주에 특화되어 있으니 "
        "ticker를 SPRC, MLGO, ATNF 같은 소형주로 바꿔서 시도해봐. "
        "또는 필터를 완화 (gap 2%, RVOL 3x 등)."
    )
else:
    EXIT_LABELS = {
        "take_profit": "✅ Take Profit (HoD/R-target)",
        "stop_loss": "❌ Stop Loss (flag low)",
        "breakeven_stop": "🟰 Break-even Stop (after add)",
        "session_close": "🕒 Session Close",
        "end_of_data": "📭 End of Data",
    }
    rows = []
    for t in perf.trades:
        rows.append(
            {
                "Entry": str(t.entry_ts or t.entry_date)[:16],
                "Exit": str(t.exit_ts or t.exit_date)[:16],
                "Outcome": "WIN" if t.pnl > 0 else "LOSS",
                "Exit Reason": EXIT_LABELS.get(t.exit_reason, t.exit_reason),
                "Entry $": f"${t.entry_price:.2f}",
                "Exit $": f"${t.exit_price:.2f}",
                "Stop $": f"${t.stop_loss:.2f}",
                "Shares": t.shares,
                "PnL $": f"${t.pnl:+,.2f}",
                "PnL %": f"{t.pnl_pct:+.2%}",
            }
        )
    st.dataframe(rows, use_container_width=True)
