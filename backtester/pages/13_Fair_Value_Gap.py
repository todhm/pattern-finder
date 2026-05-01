"""Single-ticker Fair Value Gap (FVG) backtest page.

Implements the framework from ``FairGapValue.docx``:

  Step 1 — first CHoCH of the NY session
  Step 2 — bullish 3-bar Fair Value Gap after the CHoCH
  Step 3 — entry trigger when close holds above the FVG midpoint
  Step 4 — 1:3 / 1:4 R-target take profit, FVG-producing candle
           low as the initial stop, optional BOS-trail to FVG
           midpoint after Break of Structure

Default interval is 15m (the framework names "1M time frame" but
yfinance caps 1m intraday at 7 calendar days, so 15m is what you
can realistically backtest end-to-end). 1m / 5m / 30m are exposed
as opt-in radio buttons for live signal scanning.
"""

from datetime import date, timedelta

import streamlit as st

from data.adapters.composed_market_data import build_default_market_data
from data.adapters.regular_session_filter import RegularSessionFilterAdapter
from data.domain.market_calendar import KR, NY, market_for_ticker
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

# Sub-daily fetches go through Massive (Polygon-compatible), which
# carries 10+ years of intraday history. The yfinance caps that used
# to clamp Start Date here no longer apply — the sidebar lets the
# user pick any start back to 2003 (Polygon's earliest stocks
# coverage). The constant stays around for the caption text below.
INTRADAY_HISTORY_FLOOR = date(2003, 1, 1)

# Default ``max_retest_bars`` per interval. Scaled so the wall-clock
# window stays in the 30–90 minute range across timeframes:
#
#   1m × 40 = 40 min   (noisy bars need a longer window)
#   5m × 15 = 75 min
#   15m × 5 = 75 min
#   30m × 3 = 90 min
#
# Past the window the structural FVG has aged out — price either ran
# away or consolidated long enough that the limit-at-midpoint thesis
# stops applying.
INTERVAL_RETEST_DEFAULTS = {
    "1m": 40,
    "5m": 15,
    "15m": 5,
    "30m": 3,
}

# Default ``max_bars_after_choch`` per interval — the cap on the
# CHoCH → FVG rally phase. A bit more headroom than the retest
# defaults: the impulse rally typically takes longer to develop than
# the subsequent pullback to the FVG midpoint.
#
#   1m × 60  = 60 min
#   5m × 20  = 100 min
#   15m × 8  = 2 hours
#   30m × 4  = 2 hours
INTERVAL_CHOCH_FVG_DEFAULTS = {
    "1m": 60,
    "5m": 20,
    "15m": 8,
    "30m": 4,
}

# KOSPI-tuned defaults — matching pages/14_Multi_Fair_Value_Gap.py.
# KR data has slower velocity in % terms; relaxed timing windows +
# 0.10% min-gap (vs NY 0.30%) match what KOSPI 15m actually prints
# while keeping the strict 3-bar bullish FVG structure intact.
KR_INTERVAL_RETEST_DEFAULTS = {
    "1m": 60,
    "5m": 30,
    "15m": 15,
    "30m": 8,
}
KR_INTERVAL_CHOCH_FVG_DEFAULTS = {
    "1m": 90,
    "5m": 30,
    "15m": 20,
    "30m": 10,
}
KR_MIN_GAP_PCT = 0.10

# Per-interval default for "minimum bars left in session before
# accepting a retest entry". Roughly 1 hour of wall-clock so the
# trade has room for TP / stop / BOS to play out before the
# session-close forced-flat fires. Calendar-aware filtering
# happens inside the detector — it counts session bars in the
# market's local tz, so KR's 09:00–15:00 (15 hours-ish gap to next
# session) and NY's 09:30–16:00 (extended hours optional) both work.
INTERVAL_MIN_BARS_TO_CLOSE = {
    "1m": 60,    # 1h
    "5m": 12,    # 1h
    "15m": 4,    # 1h
    "30m": 2,    # 1h
}

st.set_page_config(page_title="Fair Value Gap", layout="wide")
st.title("Fair Value Gap Strategy")
st.caption(
    "세션 시작 후 첫 CHoCH → bullish FVG → midpoint 위 holding entry → 1:3 R-target. "
    "초기 stop은 FVG-producing candle 바깥, 옵션으로 BOS 트레일 (close가 CHoCH high 돌파 시 stop을 FVG midpoint로)."
)

@st.cache_data(ttl=3600, show_spinner=False)
def _search_kr_eodhd(query: str) -> list[tuple[str, str, str]]:
    """EODHD `/api/search/{q}` filtered to Korean common stocks.

    Returns ``[(code, name, yfinance_suffix), ...]`` — empty when
    EODHD_API_KEY isn't loaded, the call fails, or no KR matches
    surface. Cached for 1 h so successive keystrokes on the same
    query don't hammer the API.
    """
    import os
    import httpx

    key = os.environ.get("EODHD_API_KEY")
    if not key or len(query.strip()) < 2:
        return []
    try:
        resp = httpx.get(
            f"https://eodhd.com/api/search/{query.strip()}",
            params={"api_token": key, "fmt": "json"},
            timeout=10.0,
        )
    except Exception:
        return []
    if resp.status_code != 200:
        return []
    out: list[tuple[str, str, str]] = []
    for row in resp.json():
        if row.get("Type") != "Common Stock":
            continue
        if row.get("Country") != "Korea":
            continue
        code = row.get("Code") or ""
        exchange = row.get("Exchange") or ""
        if not code:
            continue
        # EODHD's exchange code → yfinance suffix used by the rest
        # of the system. KO (KOSPI) maps back to .KS, KQ stays.
        if exchange == "KO":
            suffix = ".KS"
        elif exchange == "KQ":
            suffix = ".KQ"
        else:
            continue
        out.append((code, row.get("Name", ""), suffix))
    return out


with st.sidebar:
    st.header("Market")
    if "ticker" not in st.session_state:
        st.session_state.ticker = "AAPL"
    ticker = st.text_input("Ticker", key="ticker")
    with st.expander("🔍 Search Korean ticker"):
        query = st.text_input(
            "회사명 (English)",
            placeholder="samsung, kakao, hyundai, naver, sk hynix...",
            key="kr_search_query",
            help="EODHD `/search` 엔드포인트 — 한글은 미지원이라 영문으로. "
            "결과 클릭 시 위 Ticker 박스에 자동 입력.",
        )
        if query and query.strip():
            hits = _search_kr_eodhd(query)
            if not hits:
                st.caption("No KR Common-Stock matches.")
            for code, name, suffix in hits[:12]:
                full = f"{code}{suffix}"
                if st.button(
                    f"**{full}** — {name}",
                    key=f"kr_pick_{full}",
                    use_container_width=True,
                ):
                    st.session_state.ticker = full
                    st.rerun()
    # Market calendar — NY by default, KR auto-selected when ticker
    # carries a Korean exchange suffix (.KS / .KQ). User can still
    # override via the dropdown for edge cases.
    detected_market = market_for_ticker(ticker)
    market_choice = st.selectbox(
        "Market calendar",
        options=["NY", "KR"],
        index=0 if detected_market.name == "NY" else 1,
        format_func=lambda x: {
            "NY": "🇺🇸 US (NYSE / Nasdaq)  — 09:30–16:00 ET",
            "KR": "🇰🇷 Korea (KOSPI/KOSDAQ) — 09:00–15:30 KST",
        }[x],
        help="ticker suffix(.KS/.KQ)로 자동 감지하지만 필요하면 override.",
    )
    market = NY if market_choice == "NY" else KR
    interval = st.radio(
        "Bar interval",
        options=["1m", "5m", "15m", "30m"],
        index=2,  # default 15m
        horizontal=True,
        help="원문의 'time frame'은 1M (1분봉)이지만 yfinance 1m cap이 "
        "7일이라 백테스트엔 15m이 현실적. 1m은 실시간 시그널용.",
    )
    st.caption(
        f"{interval}봉은 Massive(Polygon)에서 가져오므로 기간 제한 없음."
    )

    start_date = st.date_input(
        "Start Date",
        value=date.today() - timedelta(days=30),
        min_value=INTRADAY_HISTORY_FLOOR,
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
        value=78,  # 3 sessions on 15m; ~78 minutes on 1m
        min_value=1,
        max_value=2_000,
        step=10,
        help="해당 bar 단위 기준. 15m=78이면 3세션, 1m=78이면 ~80분.",
    )

    st.header("Detector")
    # KR data trades on KRW-tick increments → bar-high ties common,
    # strict 2/2 fractal misses pivots. 1/1 default for KR.
    _swing_default = 1 if market_choice == "KR" else 2
    swing_left = st.number_input(
        "Swing pivot left (bars)",
        value=_swing_default,
        min_value=1,
        max_value=10,
        step=1,
    )
    swing_right = st.number_input(
        "Swing pivot right (bars)",
        value=_swing_default,
        min_value=1,
        max_value=10,
        step=1,
    )
    # KOSPI swings smaller in ATR-multiples than US.
    _atr_default = 1.0 if market_choice == "KR" else 2.0
    min_choch_swing_atr = st.number_input(
        "Min ChoCH swing magnitude (× ATR)",
        value=_atr_default,
        min_value=0.0,
        max_value=10.0,
        step=0.5,
        format="%.1f",
        help="H1-L2 down-leg이 N×ATR 이상이어야 ChoCH 인정. "
        "KR=1.0, NY=2.0 권장.",
    )
    _min_gap_default = KR_MIN_GAP_PCT if market_choice == "KR" else 0.30
    min_gap_pct = st.number_input(
        "Min FVG size (%)",
        value=_min_gap_default,
        min_value=0.0,
        max_value=5.0,
        step=0.05,
        format="%.2f",
        help="close 대비 FVG 폭의 최소값. NY=0.30%, KR=0.10% (KOSPI는 % 변동이 작음).",
    )
    _choch_fvg_table = (
        KR_INTERVAL_CHOCH_FVG_DEFAULTS
        if market_choice == "KR"
        else INTERVAL_CHOCH_FVG_DEFAULTS
    )
    max_bars_after_choch = st.number_input(
        "Max bars CHoCH → FVG",
        value=_choch_fvg_table[interval],
        min_value=3,
        max_value=200,
        step=1,
        help="CHoCH 이후 N bar 안에 FVG가 형성돼야 valid signal. "
        "NY 15m=8, KR 15m=20.",
    )
    _retest_table = (
        KR_INTERVAL_RETEST_DEFAULTS
        if market_choice == "KR"
        else INTERVAL_RETEST_DEFAULTS
    )
    max_retest_bars = st.number_input(
        "Max bars FVG → retest entry",
        value=_retest_table[interval],
        min_value=1,
        max_value=200,
        step=1,
        help="FVG 형성 후 N bar 안에 midpoint를 retest해야 진입 시그널 발화. "
        "NY 15m=5, KR 15m=15.",
    )
    max_signals_per_session = st.number_input(
        "Max signals per session",
        value=2,
        min_value=1,
        max_value=10,
        step=1,
    )
    min_bars_to_session_close = st.number_input(
        "Min bars left to session close (entry filter)",
        value=INTERVAL_MIN_BARS_TO_CLOSE.get(interval, 4),
        min_value=0,
        max_value=200,
        step=1,
        help="Retest entry가 정규장 마감까지 N bar 미만 남았을 때 발동되면 "
        "force_close_at_session_end가 거의 즉시 청산해서 의미 없는 trade가 됨. "
        "1m=60, 5m=12, 15m=4, 30m=2 (≈1시간). 0이면 필터 끔. "
        "캘린더(NY 16:00 ET / KR 15:00 KST)는 자동.",
    )

    st.header("Exits")
    take_profit_r = st.number_input(
        "Take profit (× R)",
        value=3.0,
        min_value=0.5,
        max_value=10.0,
        step=0.5,
        format="%.1f",
        help="원문 step 4의 1:3 ~ 1:4 권고. 3.0 표준, 4.0은 적은 winner / 큰 banking.",
    )
    enable_breakeven = st.checkbox(
        "Break-even stop after +1R",
        value=False,
        help="+1R close 후 stop을 entry로 — winner의 -R 전환 방어.",
    )
    enable_bos_trail = st.checkbox(
        "BOS trail to FVG midpoint",
        value=True,
        help="close가 진입 후 첫 swing high를 돌파(BOS)하면 stop을 "
        "FVG midpoint로 끌어올림. 원문 step 4 명시 규칙 — 기본 ON. "
        "FVG mid = entry이므로 사실상 break-even으로 동작.",
    )
    disable_stops_outside_rth = st.checkbox(
        "Disable stops outside RTH",
        value=True,
        help="정규장 외 봉에서 initial / BE / BOS trail stop 모두 차단. "
        "After-hours 저유동성 spike에 false stop-out되는 걸 방지.",
    )
    force_close_session_end = st.checkbox(
        "Force close at session end (day-trade mode)",
        value=True,
        help="정규장 마지막 RTH 봉(15m=15:45, 1m=15:59)까지 TP에 못 닿으면 "
        "그 봉 close에서 강제 청산. 대부분의 ICT/SMC FVG 트레이더가 쓰는 "
        "intraday day-trade 방식 — 오버나잇 갭 리스크 회피, ETH 저유동성 "
        "구간에서 시각적/실체결 모순 방지. OFF면 max holding bars까지 보유.",
    )

    st.header("Session")
    include_pre_post = st.checkbox(
        "Include pre/post market data",
        value=True,
        help="원문은 9:30 직후의 첫 CHoCH를 찾으라고 하지만 실전에선 "
        "장 시작 전(04:00–09:30)에 이미 CHoCH가 형성돼 있을 수 있어. "
        "ON이면 prepost 모두 fetch해서 detector가 그 구조를 본다. "
        "OFF면 정규장 (09:30–16:00 ET)만.",
    )

    fee_schedule = render_toss_fee_inputs(key_prefix="fvg_")

    run_btn = st.button("Run FVG Backtest", type="primary", use_container_width=True)

if not run_btn:
    st.info(
        "좌측에서 ticker / interval / 기간을 설정하고 **Run FVG Backtest**를 눌러."
    )
    st.stop()

# Composition: yfinance fetches everything → cache → conditional
# RTH filter. When ``include_pre_post`` is on, we skip the filter so
# the detector can see CHoCH structures forming in the 04:00–09:30
# pre-market window. The cache layer is shared with other pages, so
# turning the toggle on/off doesn't double-fetch.
_base_market = build_default_market_data()
market_data = (
    _base_market
    if include_pre_post
    else RegularSessionFilterAdapter(_base_market, market=market)
)

detector = FairValueGapDetector(
    swing_left=int(swing_left),
    swing_right=int(swing_right),
    min_gap_pct=min_gap_pct / 100.0,
    max_bars_after_choch=int(max_bars_after_choch),
    max_retest_bars=int(max_retest_bars),
    max_signals_per_session=int(max_signals_per_session),
    min_choch_swing_atr=float(min_choch_swing_atr),
    min_bars_to_session_close=int(min_bars_to_session_close),
    market=market,
    # KR has no pre-market feed in EODHD; let yesterday's PM
    # structure carry into today's session so 6h RTH isn't the
    # only window for ChoCH+FVG+retest to form. NY already gets
    # 04:00–09:30 ET pre-market via include_pre_post=True so it
    # doesn't need the carryover.
    allow_cross_session_carryover=(market_choice == "KR"),
)


# Override the strategy's default 15m interval with whatever the user
# picked on the radio (1m/5m/30m). The class attribute is what
# ``run()`` forwards to ``MarketDataPort.fetch_ohlcv``.
class _IntervalScopedFVGStrategy(FairValueGapStrategy):
    pass


_IntervalScopedFVGStrategy._interval = interval

strategy = _IntervalScopedFVGStrategy(
    market_data=market_data,
    detector=detector,
    take_profit_r_multiple=float(take_profit_r),
    enable_breakeven_stop=enable_breakeven,
    enable_bos_trail=enable_bos_trail,
    disable_stops_outside_rth=disable_stops_outside_rth,
    force_close_at_session_end=force_close_session_end,
    market=market,
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

# ---- metrics + table (only when trades fired) ----
if perf.total_trades == 0:
    st.subheader(f"{ticker.upper()} — {interval}")
    st.metric("Trades", 0)
    st.info(
        "이 기간엔 FVG signal이 안 잡혔거나 진입 조건을 다 통과한 게 없음. "
        "기간/필터를 조정해봐 — 차트는 그대로 아래에 표시돼."
    )
else:
    fee_rows = apply_fees_to_trades(perf.trades, fee_schedule)
    render_single_ticker_headline_metrics(perf, fee_rows, subtitle=f"{ticker.upper()} — {interval}")

    st.subheader("Equity Curve")
    st.plotly_chart(
        chart_builder.build_equity_curve(
            result.equity_curve,
            title=f"{ticker.upper()} {interval} FVG equity (gross)",
        ),
        use_container_width=True,
    )
    st.caption("Equity curve는 gross PnL 기반. 위 metric / trade table은 Toss 수수료 차감 후 net.")

    render_single_ticker_trade_table(fee_rows)

# ---- always render full-period chart with FVG overlays + trade markers ----
# Re-run the detector against the chart's df (separate from the
# strategy's run, but reuses the same cached data) so we can paint
# *every* qualifying FVG / CHoCH on the chart, including ones that
# the strategy skipped because the portfolio was already busy.
st.subheader("Chart")
try:
    full_df = market_data.fetch_ohlcv(ticker.upper(), start_date, end_date, interval=interval)
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
        market=market,
    )
    full_fig.update_xaxes(range=[str(start_date), str(end_date)])
    st.plotly_chart(full_fig, use_container_width=True)
    st.caption(
        "🟩 FVG zone (light green box)  ·  ━━ midpoint hold/break level "
        "(dashed)  ·  ChoCH (purple dashdot, H1→break bar)  ·  "
        "BOS (orange dotted, ChoCH→continuation bar where close>choch_high — "
        "이 시점에 enable_bos_trail이 stop을 FVG mid로 끌어올려)  ·  "
        "┄┄ red dash = stop, green dot = take-profit (R-target)."
    )
