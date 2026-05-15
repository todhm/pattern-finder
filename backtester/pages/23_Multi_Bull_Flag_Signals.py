"""Live trading workflow for the Multi Bull Flag (Ross Cameron) strategy.

Companion to ``22_Multi_Bull_Flag.py``. 1분봉 Bull Flag는 신호가 한
세션 안에서 빠르게 사라지므로 "전체 유니버스를 매분 풀스캔"하는
워크플로우는 비현실적. 대신 두 단계로 분리:

    1. **🌅 Pre-market Screen** (장 시작 전 / 직후 1회)
       Daily 4-criteria 만 빠르게 통과시켜 오늘 볼 종목을 5~30개로 좁힘.
       Universe(~2,200) 전체에서도 daily-only fetch라 수 초 안에 끝남.

    2. **🟢 Live Monitor** (장중, 30~60초 주기 새로고침)
       1번에서 좁힌 종목 리스트에만 인트라데이 detector를 돌려
       풀백·돌파·손절 상태를 실시간 추적. 신호가 막 떨어진 종목은
       **BUY NOW 카드**로 부각.

영상 정통 워크플로우와 동일 — Ross도 장 시작 직후 "leading percentage
gainers" 리스트(이게 4-criteria 통과 종목)를 좁혀놓고 거기서만 1m
풀백을 본다.
"""

from datetime import date, datetime, time, timedelta

import pandas as pd
import streamlit as st

from data.adapters.composed_fundamentals import build_default_fundamentals
from data.adapters.composed_market_data import build_default_market_data
from data.adapters.eodhd_realtime import EODHDRealtimeAdapter
from data.adapters.regular_session_filter import RegularSessionFilterAdapter
from data.adapters.wikipedia_universe import default_universe_provider
from data.domain.market_calendar import NY
from data.domain.ports import RealtimeQuote
from pattern.adapters.bull_flag import BullFlagDetector
from signals.adapters.bull_flag_scanner import BullFlagBuySignalScanner
from signals.adapters.in_memory_repo import InMemorySignalRepo
from signals.adapters.postgres_signal_repo import PostgresSignalRepo
from signals.domain.models import BuySignal, SignalStatus
from strategy.adapters.bull_flag_strategy import BullFlagStrategy
from strategy.domain.models import TossFeeSchedule

INTRADAY_HISTORY_FLOOR = date(2022, 1, 1)
PATTERN_TAG = "bull_flag"
INTERVAL_TAG = "1m"
FRESH_SIGNAL_MINUTES = 10  # 신호 발생 후 N분 이내 = "BUY NOW" 등급

st.set_page_config(page_title="Multi Bull Flag — Live", layout="wide")
st.title("Multi Bull Flag — Live Workflow")
st.caption(
    "**오전 워크플로우**: ① Pre-market Screen으로 오늘 볼 종목 5~30개로 좁히기 "
    "→ ② Live Monitor로 그 종목들만 30~60초 주기 새로고침 → "
    "③ 신호가 떨어지면 BUY NOW 카드의 entry/stop/target/shares 그대로 매수. "
    "Postgres `buy_signals` 테이블 공유 (interval='1m', pattern_name='bull_flag')."
)
try:
    st.page_link(
        "pages/22_Multi_Bull_Flag.py",
        label="📈 Multi Bull Flag — 같은 전략의 백테스트 튜닝",
        icon="⬅️",
    )
except Exception:
    pass


# --- Real-time quote provider (EODHD /api/real-time/) ---
# Scoped to Bull Flag pages only. Falls back gracefully if EODHD_API_KEY
# isn't configured — pages keep working with their existing data sources.
if "bf_rt_adapter" not in st.session_state:
    try:
        st.session_state.bf_rt_adapter = EODHDRealtimeAdapter()
        st.session_state.bf_rt_kind = "eodhd"
    except Exception:
        st.session_state.bf_rt_adapter = None
        st.session_state.bf_rt_kind = "disabled"
rt_adapter: EODHDRealtimeAdapter | None = st.session_state.bf_rt_adapter


def _fetch_live_quotes(tickers: list[str]) -> dict[str, RealtimeQuote]:
    """Wrapper that tolerates a disabled / failing real-time adapter
    so the rest of the page keeps rendering."""
    if rt_adapter is None or not tickers:
        return {}
    try:
        return rt_adapter.fetch_quotes(tickers)
    except Exception:
        return {}


# --- Repository ---
if "signal_repo" not in st.session_state:
    try:
        st.session_state.signal_repo = PostgresSignalRepo()
        st.session_state.signal_repo_kind = "postgres"
    except Exception as e:
        st.warning(f"Postgres 연결 실패 → in-memory 저장소로 fallback: {e}")
        st.session_state.signal_repo = InMemorySignalRepo()
        st.session_state.signal_repo_kind = "in-memory"
repo = st.session_state.signal_repo


def compute_sizing(
    entry_price: float,
    stop_loss: float,
    capital: float,
    risk_pct: float,
    max_position_pct: float,
) -> dict:
    risk_per_share = entry_price - stop_loss
    if risk_per_share <= 0 or entry_price <= 0 or capital <= 0:
        return {"shares": 0}
    risk_amount = capital * (risk_pct / 100.0)
    shares_by_risk = int(risk_amount / risk_per_share)
    max_notional = capital * (max_position_pct / 100.0)
    shares_by_notional = int(max_notional / entry_price)
    shares_by_capital = int(capital / entry_price)
    shares = max(0, min(shares_by_risk, shares_by_notional, shares_by_capital))
    if shares == 0:
        return {"shares": 0}
    position_value = shares * entry_price
    binding = "risk"
    if shares == shares_by_notional and shares_by_notional <= shares_by_risk:
        binding = "max_position_pct"
    if shares == shares_by_capital and shares_by_capital < shares_by_risk:
        binding = "capital"
    return {
        "shares": shares,
        "position_value": position_value,
        "pct_of_capital": position_value / capital * 100.0,
        "intended_risk_dollar": shares * risk_per_share,
        "intended_risk_pct": shares * risk_per_share / capital * 100.0,
        "binding_constraint": binding,
        "shares_by_risk_uncapped": shares_by_risk,
    }


# --- Sidebar ---
with st.sidebar:
    st.header("Universe")
    universe = st.selectbox(
        "Universe",
        options=["nasdaq_full", "nasdaq100", "sp500"],
        index=0,
        help="nasdaq_full = NASDAQ 전체 (~2,200). 영상 정통.",
    )
    max_tickers = st.number_input(
        "Max tickers (0 = all)",
        value=0, min_value=0, max_value=2500, step=50,
        help="Pre-market screen 의 universe 상한. 0이면 전체. "
        "Daily-only fetch라 nasdaq_full 전체도 수 초.",
    )
    max_workers = st.number_input(
        "Parallel workers", value=12, min_value=1, max_value=32, step=1,
    )

    st.header("Risk / Position Sizing")
    account_capital_ui = st.number_input(
        "Account Capital ($)", value=100_000, min_value=100, step=1_000,
    )
    risk_pct_ui = st.number_input(
        "Risk per Trade (%)", value=2.0, min_value=0.1, max_value=100.0,
        step=0.5,
    )
    max_position_pct_ui = st.number_input(
        "Max position (% of equity)", value=30.0, min_value=1.0,
        max_value=100.0, step=5.0,
    )

    st.header("Stock Selection (4 criteria)")
    min_gap_pct = st.number_input(
        "Min pre-market gap (%)", value=2.0, min_value=0.0, max_value=50.0,
        step=0.5, format="%.1f",
    )
    min_rvol = st.number_input(
        "Min Relative Volume (×)", value=5.0, min_value=1.0, max_value=20.0,
        step=0.5, format="%.1f",
    )
    min_price = st.number_input(
        "Min price ($)", value=2.0, min_value=0.5, max_value=100.0, step=0.5,
    )
    max_price = st.number_input(
        "Max price ($)", value=20.0, min_value=1.0, max_value=500.0, step=1.0,
    )
    max_float_shares_mil = st.number_input(
        "Max float (millions)", value=10.0, min_value=0.5, max_value=500.0,
        step=1.0,
    )
    require_float = st.checkbox("Require float data", value=True)

    st.header("Pattern (Bull Flag geometry)")
    pole_lookback = st.number_input("Pole lookback (bars)", value=7, min_value=2, max_value=30, step=1)
    pole_min_pct = st.number_input("Pole min rise (%)", value=8.0, min_value=0.5, max_value=50.0, step=0.5, format="%.1f")
    pole_min_green_bars = st.number_input("Pole min green bars", value=3, min_value=1, max_value=20, step=1)
    flag_max_bars = st.number_input("Flag max bars", value=4, min_value=1, max_value=10, step=1)
    flag_max_retrace = st.number_input("Flag max retrace (% of pole)", value=70.0, min_value=10.0, max_value=99.0, step=5.0, format="%.0f")
    latest_entry_hour = st.number_input("Latest entry hour (ET)", value=12, min_value=10, max_value=15, step=1)
    min_bar_range = st.number_input("Min bar range ($)", value=0.001, min_value=0.0, max_value=1.0, step=0.001, format="%.3f")
    max_bar_gap_seconds = st.number_input("Max bar gap (seconds)", value=90, min_value=30, max_value=600, step=15)

    st.header("Quality Filters")
    max_rvol_input = st.number_input("Max RVOL", value=30.0, min_value=5.0, max_value=100.0, step=5.0, format="%.1f")
    max_gap_pct_input = st.number_input("Max gap (%)", value=50.0, min_value=0.0, max_value=500.0, step=5.0, format="%.1f")
    max_stop_dist_pct_input = st.number_input("Max stop distance (%)", value=5.0, min_value=0.5, max_value=20.0, step=0.5, format="%.1f")
    require_9ema = st.checkbox("Require 9 EMA support", value=True)
    ema9_tol_pct = st.number_input("9 EMA tolerance (%)", value=2.5, min_value=0.1, max_value=10.0, step=0.1, format="%.1f", disabled=not require_9ema)
    also_accept_20ema = st.checkbox("20 EMA fallback", value=False, disabled=not require_9ema)
    require_daily_trend = st.checkbox("Require daily uptrend", value=False)
    daily_sma_period = st.number_input("Daily SMA period", value=50, min_value=10, max_value=200, step=10, disabled=not require_daily_trend)
    max_nth_pullback = st.number_input("Max N-th pullback", value=2, min_value=1, max_value=5, step=1)
    use_premarket_high = st.checkbox("Require entry > PM high", value=True)

    st.header("Exit / Sizing")
    target_min_r = st.number_input("Min R/R required", value=2.0, min_value=1.0, max_value=10.0, step=0.5, format="%.1f")
    use_fixed_target = st.checkbox("Use fixed R-multiple target", value=True)
    fixed_target_r = st.number_input("Fixed target R-multiple", value=2.0, min_value=1.0, max_value=10.0, step=0.5, format="%.1f", disabled=not use_fixed_target)
    enable_add = st.checkbox("Add to winner at +R", value=True)
    add_at_r = st.number_input("Add trigger (R)", value=1.5, min_value=0.3, max_value=3.0, step=0.1, format="%.1f", disabled=not enable_add)
    add_confirm_on_close = st.checkbox("Add: close-confirmed", value=True, disabled=not enable_add)
    max_session_losses = st.number_input("Max session losses", value=1, min_value=0, max_value=10, step=1)
    be_stop_buffer_pct = st.number_input("BE stop buffer (%)", value=0.3, min_value=0.0, max_value=2.0, step=0.05, format="%.2f")

    st.header("MTF")
    enable_mtf = st.checkbox("Enable MTF check (1m + 5m)", value=False)
    mtf_tolerance_seconds = st.number_input(
        "MTF tolerance (s)", value=600, min_value=60, max_value=1800, step=60,
        disabled=not enable_mtf,
    )

    st.header("Reverse-split Guard")
    split_blackout_days = st.number_input("Split blackout days (±N)", value=30, min_value=0, max_value=180, step=5)
    price_floor_lookback_days = st.number_input("Price-floor lookback (days)", value=30, min_value=0, max_value=120, step=5)

    st.header("Toss 수수료")
    buy_pct = st.number_input("Buy (%)", value=0.10, min_value=0.0, max_value=2.0, step=0.01, format="%.3f")
    sell_pct = st.number_input("Sell (%)", value=0.10, min_value=0.0, max_value=2.0, step=0.01, format="%.3f")
    sec_pct = st.number_input("SEC (%)", value=0.0023, min_value=0.0, max_value=0.1, step=0.0001, format="%.4f")


# --- Composition root ---
def build_scanner() -> BullFlagBuySignalScanner:
    md_raw = build_default_market_data()
    md = RegularSessionFilterAdapter(md_raw, market=NY)
    fundamentals = build_default_fundamentals()
    universe_provider = default_universe_provider()
    fee_schedule = TossFeeSchedule(
        buy_commission_pct=float(buy_pct) / 100.0,
        sell_commission_pct=float(sell_pct) / 100.0,
        sec_fee_pct=float(sec_pct) / 100.0,
    )

    def detector_factory(*, float_shares, splits, pm_high_by_date=None):
        return BullFlagDetector(
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
            splits=splits,
            split_blackout_days=int(split_blackout_days),
            price_floor_lookback_days=int(price_floor_lookback_days),
            enable_mtf_check=bool(enable_mtf),
            mtf_tolerance_seconds=int(mtf_tolerance_seconds),
            min_bar_range=float(min_bar_range),
            max_bar_gap_seconds=int(max_bar_gap_seconds),
            max_rvol=float(max_rvol_input) if max_rvol_input > 0 else None,
            max_gap_pct=float(max_gap_pct_input) / 100.0 if max_gap_pct_input > 0 else None,
            max_stop_distance_pct=float(max_stop_dist_pct_input) / 100.0,
            require_9ema_support=bool(require_9ema),
            ema9_tolerance_pct=float(ema9_tol_pct) / 100.0,
            also_accept_20ema_support=bool(also_accept_20ema),
            require_daily_trend=bool(require_daily_trend),
            daily_trend_sma_period=int(daily_sma_period),
            max_nth_pullback=int(max_nth_pullback),
            premarket_high_by_date=pm_high_by_date if use_premarket_high else None,
        )

    def strategy_factory(*, detector):
        return BullFlagStrategy(
            detector=detector,
            max_position_pct_of_equity=float(max_position_pct_ui) / 100.0,
            target_min_r_multiple=float(target_min_r),
            target_at_r_multiple=float(fixed_target_r) if use_fixed_target else None,
            enable_add_to_winner=bool(enable_add),
            add_at_r=float(add_at_r),
            add_confirm_on_close=bool(add_confirm_on_close),
            max_session_losses=int(max_session_losses),
            be_stop_buffer_pct=float(be_stop_buffer_pct) / 100.0,
            fee_schedule=fee_schedule,
        )

    return BullFlagBuySignalScanner(
        market_data=md,
        market_data_5m=md,
        daily_market_data=md,
        fundamentals=fundamentals,
        universe_provider=universe_provider,
        detector_factory=detector_factory,
        strategy_factory=strategy_factory,
        market=NY,
        max_workers=int(max_workers),
        require_float_filter=bool(require_float),
        raw_market_data=md_raw if use_premarket_high else None,
    )


# ====================================================================
# Section 1 — 🌅 Pre-market Screen (today's watchlist builder)
# ====================================================================
st.header("🌅 Pre-market Screen — 오늘 볼 종목 좁히기")
st.caption(
    "Daily 4-criteria(gap / RVOL / price / float)만 적용해서 오늘 시장에서 "
    "Bull Flag 가능한 종목 후보를 추림. **Daily-only fetch** 라 universe 전체 "
    "(~2,200) 스캔도 수 초. 한 번 빌드하면 Live Monitor 가 이 리스트만 본다."
)

if "bf_today_watchlist" not in st.session_state:
    st.session_state.bf_today_watchlist = []  # list[dict]
if "bf_today_screen_date" not in st.session_state:
    st.session_state.bf_today_screen_date = None
if "bf_today_screen_at" not in st.session_state:
    st.session_state.bf_today_screen_at = None

screen_cols = st.columns([2, 2, 2])
screen_date = screen_cols[0].date_input(
    "Screen date",
    value=date.today(),
    min_value=INTRADAY_HISTORY_FLOOR,
    max_value=date.today(),
    help="장 시작 직후 = today. 과거 날짜로 두면 그 날의 4-criteria 통과 종목 재현.",
)
screen_btn = screen_cols[1].button(
    "🔍 Build today's watchlist",
    type="primary",
    use_container_width=True,
    help="Universe 전체에 daily 4-criteria 게이트만 적용. 통과한 종목만 다음 단계로.",
)
clear_btn = screen_cols[2].button(
    "🗑 Clear watchlist", use_container_width=True
)
if clear_btn:
    st.session_state.bf_today_watchlist = []
    st.session_state.bf_today_screen_date = None
    st.session_state.bf_today_screen_at = None

if screen_btn:
    scanner = build_scanner()
    with st.spinner(
        f"Daily 4-criteria 스캔 ({universe}, {max_tickers or 'all'} tickers)..."
    ):
        try:
            rows = scanner.screen_premarket(
                universe=universe,
                target_date=screen_date,
                max_tickers=int(max_tickers) if max_tickers > 0 else None,
            )
        except Exception as e:
            st.error(f"Screen 실패: {e}")
            rows = []
    st.session_state.bf_today_watchlist = rows
    st.session_state.bf_today_screen_date = screen_date
    st.session_state.bf_today_screen_at = datetime.utcnow().isoformat(timespec="seconds")
    st.success(f"{len(rows)}개 종목이 4-criteria 통과.")

wl_rows: list[dict] = st.session_state.get("bf_today_watchlist", [])
wl_date = st.session_state.get("bf_today_screen_date")
wl_at = st.session_state.get("bf_today_screen_at")

if wl_rows:
    age_min = None
    if wl_at:
        try:
            age_min = (datetime.utcnow() - datetime.fromisoformat(wl_at)).total_seconds() / 60
        except Exception:
            pass
    cap = f"📋 **{len(wl_rows)} ticker(s)** · screened for {wl_date}"
    if age_min is not None:
        cap += f" · {age_min:.1f}m ago"
    st.markdown(cap)

    # Fetch live quotes for the screened tickers — one EODHD bulk
    # HTTP for the whole watchlist. Lets the user see "where is this
    # stock right now vs today's open" before committing to monitor.
    live_quotes = _fetch_live_quotes([r["ticker"] for r in wl_rows])
    df_wl = pd.DataFrame([
        {
            "Ticker": r["ticker"],
            "Gap %": round(r["gap_pct"] * 100, 1),
            "RVOL ×": round(r["rvol"], 1),
            "Open $": round(r["open_price"], 2),
            "Live $": (
                round(live_quotes[r["ticker"]].last, 2)
                if r["ticker"] in live_quotes else None
            ),
            "Day Δ%": (
                round(live_quotes[r["ticker"]].change_pct, 2)
                if r["ticker"] in live_quotes else None
            ),
            "Day H/L": (
                f"${live_quotes[r['ticker']].high:.2f} / ${live_quotes[r['ticker']].low:.2f}"
                if r["ticker"] in live_quotes else None
            ),
            "Float (M)": (
                round(r["float_shares"] / 1e6, 2)
                if r.get("float_shares") else None
            ),
        }
        for r in wl_rows
    ])
    if live_quotes:
        st.caption(
            f"📡 EODHD real-time: {len(live_quotes)}/{len(wl_rows)} tickers, "
            f"snapshot ~ {max(q.timestamp for q in live_quotes.values()).strftime('%H:%M:%S UTC')}"
        )
    elif rt_adapter is None:
        st.caption(
            "📡 EODHD real-time 비활성 (EODHD_API_KEY 미설정). "
            "Live $ / Day Δ% 열은 비어있음."
        )
    st.dataframe(df_wl, use_container_width=True, height=300, hide_index=True)
    # Optional manual prune
    keep = st.multiselect(
        "Tickers to monitor (deselect to drop from live monitor)",
        options=[r["ticker"] for r in wl_rows],
        default=[r["ticker"] for r in wl_rows],
        key="bf_wl_keep",
    )
    if keep != [r["ticker"] for r in wl_rows]:
        st.session_state.bf_today_watchlist = [
            r for r in wl_rows if r["ticker"] in keep
        ]
        wl_rows = st.session_state.bf_today_watchlist
else:
    st.info(
        "위 **🔍 Build today's watchlist** 를 눌러 오늘 볼 종목을 좁혀줘. "
        "결과가 비어있으면 4-criteria 가 너무 빡빡한 거니까 좌측 사이드바에서 "
        "min_gap_pct / min_rvol 을 완화해서 다시 시도."
    )


# ====================================================================
# Section 2 — 🟢 Live Monitor
# ====================================================================
st.divider()
st.header("🟢 Live Monitor — 1분봉 Bull Flag 신호")
st.caption(
    "Pre-market screen 으로 좁힌 종목들만 인트라데이 detector 돌림. "
    "신호가 막 떨어진 종목은 **BUY NOW** 카드로 부각. 30~60초 주기로 새로고침해서 "
    "준라이브 모니터링."
)

# Auto-refresh controls
auto_cols = st.columns([1, 1, 1, 2])
auto_on = auto_cols[0].checkbox(
    "Auto-refresh", value=False, key="bf_auto_refresh",
    help="HTML meta-refresh 로 페이지 전체를 N초마다 재로드. "
    "세션 상태(watchlist)는 그대로 유지.",
)
refresh_sec = auto_cols[1].selectbox(
    "Every", options=[30, 45, 60, 90, 120], index=2,
    format_func=lambda x: f"{x}s",
    disabled=not auto_on,
    key="bf_refresh_sec",
)
manual_refresh = auto_cols[2].button(
    "🔄 Refresh Now", use_container_width=True
)
last_monitor_at = st.session_state.get("bf_last_monitor_at")
if last_monitor_at:
    auto_cols[3].caption(
        f"📡 Last monitor refresh: {last_monitor_at} UTC "
        f"({(datetime.utcnow() - datetime.fromisoformat(last_monitor_at)).total_seconds():.0f}s ago)"
    )

if auto_on:
    # HTML meta-refresh: simple, no extra deps. Streamlit session state
    # survives the full-page reload, so the watchlist + sidebar don't reset.
    st.markdown(
        f'<meta http-equiv="refresh" content="{int(refresh_sec)}">',
        unsafe_allow_html=True,
    )

# Decide whether to run a monitor pass: auto-on (always re-fetch on
# every reload), manual button click, or no cached signals yet.
should_monitor = bool(wl_rows) and (
    auto_on or manual_refresh or "bf_monitor_signals" not in st.session_state
)
if should_monitor and wl_rows:
    scanner = build_scanner()
    tickers = [r["ticker"] for r in wl_rows]
    with st.spinner(f"Monitoring {len(tickers)} ticker(s)..."):
        try:
            sigs = scanner.monitor_tickers(
                tickers=tickers,
                target_date=wl_date or date.today(),
            )
        except Exception as e:
            st.error(f"Monitor 실패: {e}")
            sigs = []
    st.session_state.bf_monitor_signals = sigs
    st.session_state.bf_last_monitor_at = datetime.utcnow().isoformat(timespec="seconds")

monitor_signals: list[BuySignal] = st.session_state.get("bf_monitor_signals", [])


def _signal_age_minutes(sig: BuySignal) -> float | None:
    """Minutes since the breakout candle printed (entry_ts) — not since
    we refreshed the signal."""
    ts = sig.metadata.get("entry_ts")
    if not ts:
        return None
    try:
        entry_dt = datetime.fromisoformat(ts)
    except Exception:
        return None
    # Normalize to UTC for diffing — entry_ts is tz-aware NY time.
    if entry_dt.tzinfo is None:
        return None
    try:
        return (datetime.now(entry_dt.tzinfo) - entry_dt).total_seconds() / 60
    except Exception:
        return None


def _classify(sig: BuySignal) -> str:
    """Coarse state for visual hierarchy: buy_now > fired > stopped > target."""
    meta = sig.metadata
    if meta.get("stop_tripped"):
        return "stopped"
    current_hod = meta.get("current_hod")
    target = meta.get("target_strategy")
    if current_hod and target and current_hod >= target:
        return "target_hit"
    age_min = _signal_age_minutes(sig)
    if age_min is not None and age_min <= FRESH_SIGNAL_MINUTES:
        return "buy_now"
    return "fired"


if not wl_rows:
    st.info("위 Section 1 에서 today's watchlist 를 먼저 빌드해줘.")
elif not monitor_signals:
    st.warning(
        f"📭 {len(wl_rows)}개 종목 모두 아직 Bull Flag 신호 미감지. "
        f"풀백·돌파가 진행 중일 수 있으니 30~60초 후 다시 새로고침. "
        f"(detector_factory + 4-criteria 통과 종목이라도 인트라데이 "
        "geometry 가 안 잡히면 신호 안 떨어짐.)"
    )
else:
    # Sort: BUY NOW first, then fired, then stopped/target_hit.
    state_order = {"buy_now": 0, "fired": 1, "target_hit": 2, "stopped": 3}
    monitor_signals.sort(
        key=lambda s: (
            state_order[_classify(s)],
            -(s.metadata.get("rvol", 0)),
        )
    )

    buy_now = [s for s in monitor_signals if _classify(s) == "buy_now"]
    fired = [s for s in monitor_signals if _classify(s) == "fired"]
    target_hit = [s for s in monitor_signals if _classify(s) == "target_hit"]
    stopped = [s for s in monitor_signals if _classify(s) == "stopped"]

    # Real-time quotes for the signal tickers — one EODHD bulk call.
    # ``latest_close`` from the intraday fetch is ~15 min stale if it
    # routed through yfinance; EODHD real-time is current-tick. Used
    # to recompute unrealized R / target-hit / stop-tripped one more
    # time per cycle so BUY NOW cards reflect the freshest price.
    mon_quotes = _fetch_live_quotes([s.ticker for s in monitor_signals])
    if mon_quotes:
        snap_ts = max(q.timestamp for q in mon_quotes.values())
        st.caption(
            f"📡 EODHD real-time overlay: {len(mon_quotes)}/{len(monitor_signals)} "
            f"snapshot ~ {snap_ts.strftime('%H:%M:%S UTC')}"
        )

    # ---- BUY NOW cards (loud) ----
    if buy_now:
        st.markdown("### 🟢 BUY NOW — 신호 발생 직후 (< 10분)")
        for sig in buy_now:
            meta = sig.metadata
            sizing = compute_sizing(
                entry_price=sig.entry_price,
                stop_loss=sig.stop_loss,
                capital=float(account_capital_ui),
                risk_pct=float(risk_pct_ui),
                max_position_pct=float(max_position_pct_ui),
            )
            target_strategy = meta.get("target_strategy") or sig.entry_price
            target_label = meta.get("target_strategy_label", "?")
            r_to_target = (
                (target_strategy - sig.entry_price)
                / (sig.entry_price - sig.stop_loss)
            )
            age_min = _signal_age_minutes(sig) or 0
            entry_time_str = meta.get("entry_ts", "")[:16].replace("T", " ")

            # Prefer EODHD real-time quote (current tick) over the 1m
            # fetch's latest_close (15-min lagged via yfinance). Fall
            # back to latest_close when EODHD is disabled / no quote.
            rt = mon_quotes.get(sig.ticker)
            risk = sig.entry_price - sig.stop_loss
            if rt is not None and risk > 0:
                live_price = rt.last
                live_unrealized_r = (live_price - sig.entry_price) / risk
                live_source = "EODHD live"
            else:
                live_price = meta.get("latest_close")
                live_unrealized_r = meta.get("unrealized_r")
                live_source = "1m fetch"

            with st.container(border=True):
                head_cols = st.columns([2, 3])
                head_cols[0].markdown(
                    f"### 🟢 **{sig.ticker}**\n"
                    f"신호 시각 `{entry_time_str}` · **{age_min:.0f}분 전**"
                )
                if live_price and live_unrealized_r is not None:
                    head_cols[1].metric(
                        f"Latest ${live_price:.2f} ({live_source})",
                        f"{live_unrealized_r:+.2f}R",
                        delta=f"{(live_price - sig.entry_price)/sig.entry_price*100:+.2f}% vs entry",
                    )

                # Order ticket — copy-able numbers
                t1, t2, t3 = st.columns(3)
                t1.metric(
                    "🟢 BUY @ Entry",
                    f"${sig.entry_price:.2f}",
                    help="Pole 끝 high = breakout price. 영상 정통.",
                )
                t2.metric(
                    "🔴 STOP @ Flag Low",
                    f"${sig.stop_loss:.2f}",
                    delta=f"-${sig.entry_price - sig.stop_loss:.2f}/share",
                )
                t3.metric(
                    f"🎯 TARGET ({target_label})",
                    f"${target_strategy:.2f}",
                    delta=f"+{r_to_target:.2f}R",
                )

                if sizing["shares"] > 0:
                    s1, s2, s3, s4 = st.columns(4)
                    s1.metric("Shares", f"**{sizing['shares']:,}**")
                    s2.metric(
                        "Position $", f"${sizing['position_value']:,.0f}"
                    )
                    s3.metric(
                        "% of capital", f"{sizing['pct_of_capital']:.1f}%"
                    )
                    s4.metric(
                        "Intended risk",
                        f"${sizing['intended_risk_dollar']:,.0f}",
                        delta=f"{sizing['intended_risk_pct']:.2f}%",
                    )
                    if sizing["binding_constraint"] == "max_position_pct":
                        st.caption(
                            f"⚠ Position cap binding "
                            f"({max_position_pct_ui:.0f}% notional)."
                        )

                # Context — terse
                ctx_cols = st.columns(4)
                ctx_cols[0].caption(
                    f"Gap **{meta.get('gap_pct', 0)*100:+.1f}%** · "
                    f"RVOL **{meta.get('rvol', 0):.1f}×**"
                )
                float_shares = meta.get("float_shares")
                ctx_cols[1].caption(
                    f"Float **{float_shares/1e6:.1f}M**"
                    if float_shares else "Float —"
                )
                pm_high = meta.get("premarket_high")
                ctx_cols[2].caption(
                    f"PM high ${pm_high:.2f}"
                    if pm_high else "PM high —"
                )
                ctx_cols[3].caption(
                    f"9 EMA now ${meta.get('current_ema9', 0):.2f}"
                )

                # Save / dismiss
                act = st.columns([1, 1, 4])
                if act[0].button(
                    "💾 Save to watchlist",
                    key=f"bf_save_{sig.id}",
                    type="primary",
                ):
                    repo.save(sig)
                    st.success(f"저장됨: {sig.ticker} {sig.signal_date}")
                if act[1].button("🚫 Dismiss", key=f"bf_dismiss_{sig.id}"):
                    st.session_state.bf_monitor_signals = [
                        s for s in monitor_signals if s.id != sig.id
                    ]
                    st.rerun()

    # ---- Fired (older than 10 min) — compact rows ----
    if fired:
        st.markdown("### ✅ Fired earlier (10분+ 경과 — 진입 윈도우 지났을 수 있음)")
        df_fired = []
        for s in fired:
            meta = s.metadata
            df_fired.append({
                "Ticker": s.ticker,
                "Entry time": meta.get("entry_ts", "")[:16].replace("T", " "),
                "Entry $": round(s.entry_price, 2),
                "Stop $": round(s.stop_loss, 2),
                "Target $": round(meta.get("target_strategy") or 0, 2),
                "Latest $": round(meta.get("latest_close") or 0, 2),
                "Unrealized R": round(meta.get("unrealized_r") or 0, 2),
                "Current HoD": round(meta.get("current_hod") or 0, 2),
                "Gap %": round(meta.get("gap_pct", 0) * 100, 1),
                "RVOL ×": round(meta.get("rvol", 0), 1),
            })
        st.dataframe(
            df_fired, use_container_width=True, height=200, hide_index=True
        )

    # ---- Stopped / target hit — informational ----
    if stopped or target_hit:
        cols = st.columns(2)
        if stopped:
            with cols[0]:
                st.markdown("### ❌ Stopped out")
                for s in stopped:
                    st.caption(
                        f"**{s.ticker}** {s.metadata.get('entry_ts','')[11:16]} · "
                        f"entry ${s.entry_price:.2f} / stop ${s.stop_loss:.2f} · "
                        f"low ${s.metadata.get('session_low_since_entry', 0):.2f}"
                    )
        if target_hit:
            with cols[1]:
                st.markdown("### 🎯 Target hit")
                for s in target_hit:
                    target = s.metadata.get("target_strategy")
                    st.caption(
                        f"**{s.ticker}** entry ${s.entry_price:.2f} · "
                        f"target ${target:.2f} · "
                        f"HoD ${s.metadata.get('current_hod', 0):.2f}"
                    )


# ====================================================================
# Section 3 — Manual Add
# ====================================================================
st.divider()
st.subheader("➕ Manual Add (특정 종목·날짜)")
st.caption(
    "Universe 밖 종목을 직접 watchlist 에 저장. detector 가 그 날짜에 신호를 "
    "찍었으면 detector 메타데이터로, 아니면 **수동 등록(pattern=`manual`)** 으로 "
    "fallback — entry = 세션 open / stop = max(session_low, open × 0.95)."
)
ma1, ma2, ma3 = st.columns([2, 2, 1])
manual_ticker = ma1.text_input(
    "Ticker", value="", placeholder="OSR / MLGO / ATNF",
    key="bf_manual_ticker",
).strip().upper()
manual_date = ma2.date_input(
    "Signal Date",
    value=date.today(),
    min_value=INTRADAY_HISTORY_FLOOR,
    max_value=date.today(),
    key="bf_manual_date",
)
manual_notes = ma3.text_input("Notes (optional)", value="", key="bf_manual_notes")
manual_add_btn = st.button(
    "Add to Watchlist", disabled=not manual_ticker, key="bf_manual_add_btn",
)
if manual_add_btn and manual_ticker:
    builder = build_scanner()
    try:
        sig = builder.build_signal_at(manual_ticker, manual_date)
        sig.notes = manual_notes
        repo.save(sig)
        if sig.metadata.get("manually_added_no_signal"):
            st.warning(
                f"{manual_ticker} · {manual_date} 수동 등록 — detector signal "
                f"없어서 fallback. entry ${sig.entry_price:.2f} / "
                f"stop ${sig.stop_loss:.2f}"
            )
        else:
            st.success(
                f"{manual_ticker} · {manual_date} 저장 (detector 감지 · "
                f"entry ${sig.entry_price:.2f} / stop ${sig.stop_loss:.2f})"
            )
    except ValueError as e:
        st.error(f"추가 실패: {e}")
    except Exception as e:
        st.error(f"예상치 못한 오류: {e}")


# ====================================================================
# Section 4 — Saved Watchlist (Postgres)
# ====================================================================
st.divider()
st.subheader("📌 Saved Watchlist (Bull Flag · 1m)")
st.caption(
    "Postgres `buy_signals` 테이블 공유 (다른 전략과 함께). "
    "여기선 `pattern_name=bull_flag` + `interval=1m` 만 표시."
)
wl_cols = st.columns([3, 1])
status_filter = wl_cols[0].selectbox(
    "Status filter",
    options=["all", *[s.value for s in SignalStatus]],
    index=0,
    key="bf_status_filter",
)
refresh_saved_btn = wl_cols[1].button(
    "🔄 Refresh saved signals",
    use_container_width=True,
    help="저장된 각 signal의 current HoD / 9 EMA / latest close 재계산.",
    key="bf_local_refresh_btn",
)
status_value = None if status_filter == "all" else SignalStatus(status_filter)
all_saved = repo.list(status=status_value, interval=INTERVAL_TAG)
saved = [s for s in all_saved if s.pattern_name in {PATTERN_TAG, "manual"}]


def _refresh_saved(signals: list[BuySignal]) -> None:
    if not signals:
        return
    refresher = build_scanner()
    progress = st.progress(0.0, text="새로고침 중...")
    for i, sig in enumerate(signals):
        try:
            updated = refresher.refresh_targets(sig)
            repo.save(updated)
        except Exception as e:
            st.warning(f"{sig.ticker} 새로고침 실패: {e}")
        progress.progress(
            (i + 1) / len(signals),
            text=f"{sig.ticker} ({i + 1}/{len(signals)})",
        )
    progress.empty()


if saved and refresh_saved_btn:
    _refresh_saved(saved)
    st.success(f"{len(saved)}개 저장 signal 새로고침 완료.")
    all_saved = repo.list(status=status_value, interval=INTERVAL_TAG)
    saved = [s for s in all_saved if s.pattern_name in {PATTERN_TAG, "manual"}]

if not saved:
    st.info("저장된 Bull Flag signal 없음.")
else:
    for sig in saved:
        meta = sig.metadata
        unrealized_r = meta.get("unrealized_r")
        stop_tripped = meta.get("stop_tripped", False)
        status_prefix = "❌" if stop_tripped else (
            "✅" if (unrealized_r is not None and unrealized_r >= 2.0) else ""
        )
        with st.expander(
            f"[{sig.status.value.upper()}] {status_prefix} {sig.ticker} — "
            f"{sig.signal_date} · entry ${sig.entry_price:.2f}",
            expanded=False,
        ):
            st.write(f"**Pattern**: {sig.pattern_name} · interval: {sig.interval}")
            st.write(
                f"**Entry / Stop**: ${sig.entry_price:.2f} / ${sig.stop_loss:.2f}"
            )
            target_strategy = meta.get("target_strategy")
            target_label = meta.get("target_strategy_label", "?")
            if target_strategy:
                st.write(
                    f"**Target ({target_label})**: ${target_strategy:.2f} "
                    f"· 1R/2R/3R: "
                    f"${meta.get('target_1r', 0):.2f} / "
                    f"${meta.get('target_2r', 0):.2f} / "
                    f"${meta.get('target_3r', 0):.2f}"
                )
            latest_close = meta.get("latest_close")
            if latest_close:
                delta_pct = (
                    (latest_close - sig.entry_price) / sig.entry_price * 100
                )
                st.write(
                    f"**Latest close** ({meta.get('latest_date', '?')}): "
                    f"${latest_close:.2f} ({delta_pct:+.2f}% vs entry)"
                )
            if unrealized_r is not None:
                st.write(
                    f"**Unrealized R**: {unrealized_r:+.2f}R "
                    f"{'⚠ stop tripped' if stop_tripped else ''}"
                )
            refreshed_at = meta.get("refreshed_at")
            if refreshed_at:
                st.caption(f"마지막 새로고침: {refreshed_at} UTC")

            w_sizing = compute_sizing(
                entry_price=sig.entry_price,
                stop_loss=sig.stop_loss,
                capital=float(account_capital_ui),
                risk_pct=float(risk_pct_ui),
                max_position_pct=float(max_position_pct_ui),
            )
            if w_sizing["shares"] > 0:
                st.write(
                    f"**포지션 사이즈**: {w_sizing['shares']:,}주 · "
                    f"${w_sizing['position_value']:,.0f} · "
                    f"{w_sizing['pct_of_capital']:.2f}% of capital"
                )
            st.write(f"**Saved at**: {sig.created_at.isoformat(timespec='seconds')} UTC")
            if sig.notes:
                st.write(f"**Notes**: {sig.notes}")
            new_notes = st.text_area(
                "Edit notes", value=sig.notes,
                key=f"bf_notes_edit_{sig.id}", height=80,
            )
            new_status = st.selectbox(
                "Status",
                options=[s.value for s in SignalStatus],
                index=[s.value for s in SignalStatus].index(sig.status.value),
                key=f"bf_status_sel_{sig.id}",
            )
            c1, c2, _ = st.columns(3)
            if c1.button("Update", key=f"bf_update_{sig.id}"):
                repo.update_notes(sig.id, new_notes)
                repo.update_status(sig.id, SignalStatus(new_status))
                st.rerun()
            if c2.button("Delete", key=f"bf_delete_{sig.id}"):
                repo.delete(sig.id)
                st.rerun()
