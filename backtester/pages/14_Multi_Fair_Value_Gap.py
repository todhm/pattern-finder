"""Multi-ticker Fair Value Gap backtest.

Universe-wide counterpart to :file:`13_Fair_Value_Gap.py`. Scans every
ticker in the chosen universe (S&P 500 / Nasdaq-100) at the chosen
intraday interval, picks the daily highest-conviction signal by
buy/sell-volume A/D proxy, and walks one position at a time.

Default detector params match :file:`13_Fair_Value_Gap.py` (RTH-only
FVG formation, 0.30% min gap, 2.0× ATR ChoCH magnitude). Default
strategy params: TP at 3R, BE off, BOS trail on (framework step-4
rule).
"""

from datetime import date, timedelta

import streamlit as st

from data.adapters.cached_market_data import CachedMarketDataAdapter
from data.adapters.regular_session_filter import RegularSessionFilterAdapter
from data.adapters.wikipedia_universe import default_universe_provider
from data.domain.market_calendar import KR, NY
from data.adapters.yfinance_adapter import YFinanceAdapter
from pages._shared.wedgepop_results import (
    render_equity_curve,
    render_failed_tickers,
    render_headline_metrics,
    render_per_trade_charts,
    render_ticker_contribution,
    render_top_trades,
    render_trade_table,
    render_toss_fee_inputs,
)
from pattern.adapters.fair_value_gap import FairValueGapDetector
from strategy.adapters.fair_value_gap_strategy import FairValueGapStrategy
from strategy.adapters.multi_fair_value_gap_strategy import (
    MultiFairValueGapStrategy,
)
from strategy.domain.models import MultiStrategyConfig
from visualization.adapters.plotly_charts import PlotlyChartBuilder

# yfinance intraday history caps. Sidebar uses these to clamp Start
# Date so the user can't pick a window the upstream silently rejects.
INTRADAY_CAPS = {
    "1m": 7,
    "5m": 60,
    "15m": 60,
    "30m": 60,
}

# Default ``max_retest_bars`` per interval. Scaled to a roughly 30–90
# minute wall-clock window across timeframes (1m=40min, 5m=75min,
# 15m=75min, 30m=90min). Past that the FVG has aged out structurally.
INTERVAL_RETEST_DEFAULTS = {
    "1m": 40,
    "5m": 15,
    "15m": 5,
    "30m": 3,
}

# Default ``max_bars_after_choch`` per interval — the CHoCH → FVG
# rally window. A bit more headroom than retest defaults since the
# impulse rally typically takes longer to develop than the pullback.
# 1m=60min, 5m=100min, 15m=2h, 30m=2h.
INTERVAL_CHOCH_FVG_DEFAULTS = {
    "1m": 60,
    "5m": 20,
    "15m": 8,
    "30m": 4,
}

st.set_page_config(page_title="Multi Fair Value Gap", layout="wide")
st.title("Multi-Ticker Fair Value Gap Strategy")
st.caption(
    "S&P 500 / Nasdaq-100 universe에서 intraday FVG 신호를 스캔. 한 봉에 "
    "여러 종목이 신호 발화하면 buy/sell-volume A/D ratio가 가장 높은 종목 "
    "하나를 매수. 한 포지션이 청산될 때까지 다음 종목 잡지 않음. "
    "⚠️ yfinance 1m 캡 7일 / 15m 캡 60일."
)

with st.sidebar:
    st.header("Market")
    market_choice = st.selectbox(
        "Market",
        options=["NY", "KR"],
        index=0,
        format_func=lambda x: {
            "NY": "🇺🇸 US (NYSE / Nasdaq)",
            "KR": "🇰🇷 Korea (KOSPI)",
        }[x],
        help="Detector / strategy / chart 모두 이 calendar에 따라 RTH "
        "윈도우와 session 경계를 잡아. KR은 09:00–15:30 KST.",
    )
    market = NY if market_choice == "NY" else KR

    st.header("Universe")
    if market_choice == "NY":
        universe_options = ["sp500", "nasdaq100", "nasdaq_full"]
        universe_labels = {
            "sp500": "S&P 500 (~500)",
            "nasdaq100": "Nasdaq-100 (~100)",
            "nasdaq_full": "Nasdaq All Common Stocks (~2,200)",
        }
    else:
        universe_options = ["kospi200"]
        universe_labels = {"kospi200": "KOSPI 200 (~200)"}
    universe = st.selectbox(
        "Universe",
        options=universe_options,
        index=0,
        format_func=lambda x: universe_labels[x],
    )
    max_tickers = st.number_input(
        "Max tickers (0 = all)",
        value=30,
        min_value=0,
        max_value=2_500,
        step=10,
        help="처음엔 작게(20~50) 시작해서 동작 확인 후 늘려. "
        "nasdaq_full은 ~2,200 종목 — 1m 캡 7일 / 15m 캡 60일 안에서 "
        "병렬 호출이 많아지므로 max_workers 신중히.",
    )
    max_workers = st.number_input(
        "Parallel workers",
        value=4,
        min_value=1,
        max_value=32,
        step=1,
        help="yfinance 호출을 동시에 몇 개 보낼지. 너무 크면 rate-limit.",
    )

    st.header("Interval / Period")
    interval = st.radio(
        "Bar interval",
        options=["1m", "5m", "15m", "30m"],
        index=2,  # default 15m
        horizontal=True,
        help="1m은 7일 캡, 나머지는 60일.",
    )
    cap_days = INTRADAY_CAPS[interval]
    st.caption(f"yfinance {interval} cap = 최근 {cap_days} calendar days.")

    start_date = st.date_input(
        "Start Date",
        value=date.today() - timedelta(days=min(7, cap_days)),
        min_value=date.today() - timedelta(days=cap_days),
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
        max_value=5_000,
        step=10,
        help="해당 bar 단위 기준. 15m=78이면 3세션, 1m=78이면 ~80분.",
    )

    st.header("Detector")
    # Per-market default swing fractal width. KR data trades on
    # KRW-tick increments (Samsung at 219,000 KRW has a 500-KRW tick
    # = 0.23%) so bar-high ties are common; the strict 2/2 fractal
    # misses most pivots and the universe scan hits 0 signals. 1/1
    # gives KOSPI realistic detection without being noisy on US.
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
    min_gap_pct = st.number_input(
        "Min FVG size (%)",
        value=0.30,
        min_value=0.0,
        max_value=5.0,
        step=0.05,
        format="%.2f",
        help="close 대비 FVG 폭 최소값. 0.30 (= 0.3%) = detector default.",
    )
    max_bars_after_choch = st.number_input(
        "Max bars CHoCH → FVG",
        value=INTERVAL_CHOCH_FVG_DEFAULTS[interval],
        min_value=3,
        max_value=200,
        step=1,
        help="CHoCH 이후 N bar 안에 FVG가 형성돼야 valid. "
        "1m=60, 5m=20, 15m=8, 30m=4 (1-2시간).",
    )
    max_retest_bars = st.number_input(
        "Max bars FVG → retest entry",
        value=INTERVAL_RETEST_DEFAULTS[interval],
        min_value=1,
        max_value=200,
        step=1,
        help="FVG 형성 후 N bar 안에 midpoint retest 못하면 만료. "
        "1m=40, 5m=15, 15m=5, 30m=3 (대략 30~90분 윈도우).",
    )
    max_signals_per_session = st.number_input(
        "Max signals per session (per ticker)",
        value=2,
        min_value=1,
        max_value=10,
        step=1,
    )
    # KR equity intraday swings tend to be smaller in ATR-multiples
    # than US — 2.0× rejects nearly every ChoCH on KOSPI 15m.
    _atr_default = 1.0 if market_choice == "KR" else 2.0
    min_choch_swing_atr = st.number_input(
        "Min ChoCH swing magnitude (× ATR)",
        value=_atr_default,
        min_value=0.0,
        max_value=10.0,
        step=0.5,
        format="%.1f",
        help="H1-L2 down-leg이 N×ATR 이상이어야 ChoCH로 인정. KR은 1.0× "
        "(KOSPI intraday swing이 미장 대비 작음), NY는 2.0× 권장.",
    )

    st.header("Exits")
    take_profit_r = st.number_input(
        "Take profit (× R)",
        value=3.0,
        min_value=0.5,
        max_value=10.0,
        step=0.5,
        format="%.1f",
        help="원문 step 4의 1:3~1:4 권고.",
    )
    enable_breakeven = st.checkbox(
        "Break-even stop after +1R",
        value=False,
        help="+1R close 시 stop을 entry로. BOS trail이 default ON이라 보조 안전장치.",
    )
    enable_bos_trail = st.checkbox(
        "BOS trail to FVG midpoint",
        value=True,
        help="진입 후 첫 swing high를 close가 돌파하면 stop을 FVG mid로 끌어올림. "
        "원문 step 4 명시 — default ON.",
    )
    disable_stops_outside_rth = st.checkbox(
        "Disable stops outside RTH",
        value=True,
        help="정규장 외 봉에서 모든 stop 차단. ETH 저유동성 spike로 "
        "인한 false stop-out 방지.",
    )
    force_close_session_end = st.checkbox(
        "Force close at session end (day-trade mode)",
        value=True,
        help="정규장 마지막 RTH 봉까지 TP에 못 닿으면 그 봉 close에서 "
        "강제 청산. 대부분의 ICT/SMC FVG 트레이더가 쓰는 intraday "
        "day-trade 방식 — overnight 갭 리스크 회피.",
    )

    st.header("Session")
    if market_choice == "NY":
        include_pre_post = st.checkbox(
            "Include pre/post market data",
            value=True,
            help="ChoCH가 04:00–09:30 프리마켓에서 형성되는 경우도 잡으려면 ON. "
            "FVG 자체는 detector가 RTH에서만 형성하도록 강제 (별도 필터).",
        )
    else:
        # Korean cash market has no formal pre/post session that
        # yfinance reports — leave the toggle off so the upstream
        # frame is RTH-only by construction.
        include_pre_post = False
        st.caption(
            "🇰🇷 KR: 정규장 외 세션 미지원 (yfinance KR 피드는 RTH-only)."
        )

    fee_schedule = render_toss_fee_inputs(key_prefix="multifvg_")

    run_btn = st.button(
        "Run FVG Universe Scan", type="primary", use_container_width=True
    )

if not run_btn:
    st.info(
        f"좌측에서 universe / interval / 기간을 설정하고 **Run FVG Universe "
        f"Scan**을 눌러. yfinance {interval} 캡이 {cap_days}일이라 그 이상의 "
        "기간을 보려면 다른 데이터 어댑터가 필요해."
    )
    st.stop()

# Composition: yfinance fetch → cache → conditional RTH filter. The
# RTH filter is calendar-aware so KR tickers (whose bars yfinance
# normalizes to NY tz) gate against 09:00–15:30 KST not 09:30–16:00
# ET. Without that, every KR ticker drops every bar and the universe
# scan reports 0 trades / 100% failures.
_base_market = CachedMarketDataAdapter(YFinanceAdapter())
market_data = (
    _base_market
    if include_pre_post
    else RegularSessionFilterAdapter(_base_market, market=market)
)
universe_provider = default_universe_provider()

detector = FairValueGapDetector(
    swing_left=int(swing_left),
    swing_right=int(swing_right),
    min_gap_pct=min_gap_pct / 100.0,
    max_bars_after_choch=int(max_bars_after_choch),
    max_retest_bars=int(max_retest_bars),
    max_signals_per_session=int(max_signals_per_session),
    min_choch_swing_atr=float(min_choch_swing_atr),
    market=market,
)


# Per-ticker strategy with the user's chosen interval. We use a thin
# subclass instead of mutating the class attribute on
# ``FairValueGapStrategy`` itself, which would leak across runs of the
# Streamlit page.
class _IntervalScopedFVGStrategy(FairValueGapStrategy):
    pass


_IntervalScopedFVGStrategy._interval = interval

per_ticker_strategy = _IntervalScopedFVGStrategy(
    market_data=market_data,
    detector=detector,
    take_profit_r_multiple=float(take_profit_r),
    enable_breakeven_stop=enable_breakeven,
    enable_bos_trail=enable_bos_trail,
    disable_stops_outside_rth=disable_stops_outside_rth,
    force_close_at_session_end=force_close_session_end,
    market=market,
)

runner = MultiFairValueGapStrategy(
    market_data=market_data,
    universe_provider=universe_provider,
    detector=detector,
    strategy=per_ticker_strategy,
    max_workers=int(max_workers),
)

config = MultiStrategyConfig(
    universe=universe,
    start_date=start_date,
    end_date=end_date,
    pattern_name="fair_value_gap",
    initial_capital=float(initial_capital),
    risk_per_trade=risk_pct / 100.0,
    max_holding_days=int(max_holding_bars),
    max_tickers=int(max_tickers) if max_tickers > 0 else None,
    fee_schedule=fee_schedule,
)

with st.spinner(
    f"Scanning {universe} on {interval} bars "
    f"({config.max_tickers or 'all'} tickers)..."
):
    try:
        result = runner.run(config)
    except Exception as exc:
        st.error(f"Universe scan failed: {exc}")
        st.stop()

render_headline_metrics(
    result, universe_label=f"{universe.upper()} · {interval} · FVG"
)

if result.tickers_scanned == 0:
    st.warning("Universe returned 0 tickers — check selection.")
    st.stop()
if not result.trades:
    st.info(
        f"이 기간엔 {interval} FVG signal이 잡히지 않았거나 모두 필터에 걸렸어. "
        "기간/필터를 조정해봐."
    )
    render_failed_tickers(result)
    st.stop()

chart_builder = PlotlyChartBuilder()
render_equity_curve(
    chart_builder,
    result,
    title=(
        f"Single-position FVG across {universe.upper()} ({interval})"
    ),
)
render_top_trades(chart_builder, result)
render_ticker_contribution(chart_builder, result)
render_trade_table(result)
render_per_trade_charts(
    chart_builder,
    market_data,
    result,
    interval=interval,
    context_before_days=2,
    context_after_days=2,
    warmup_days=5,
    market=market,
)
render_failed_tickers(result)
