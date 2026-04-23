"""Live signals view for the Multi Wick Play strategy.

Companion to ``6_Multi_Wick_Play.py`` — same detector / exit /
sizing knobs, but applied to the **most recent bars** to surface
actionable Wick Play buy setups right now. Signals persist in the
shared ``buy_signals`` Postgres table (same repo as Multi Wedgepop
signals), tagged by ``pattern_name`` so both strategies coexist on
one watchlist.

Unlike the wedgepop scanner, Wick Play's detector already applies
psych-score / regime / macro / prior-trend filters internally, so no
post-detection gate re-evaluation is needed here — every signal the
scanner returns has already survived the full detector gate stack.
"""

from datetime import date, datetime, timedelta

import streamlit as st

from data.adapters.cached_market_data import CachedMarketDataAdapter
from data.adapters.macro_calendar import blackout_dates as build_blackout_dates
from data.adapters.wikipedia_universe import default_universe_provider
from data.adapters.yfinance_adapter import YFinanceAdapter
from pattern.adapters.exhaustion_extension_top import (
    ExhaustionExtensionTopDetector,
)
from pattern.adapters.wick_play import WickPlayDetector
from signals.adapters.in_memory_repo import InMemorySignalRepo
from signals.adapters.postgres_signal_repo import PostgresSignalRepo
from signals.adapters.wick_play_scanner import WickPlayBuySignalScanner
from signals.domain.models import BuySignal, SignalStatus
from strategy.adapters.wickplay_strategy import WickPlayStrategy

st.set_page_config(page_title="Multi Wick Play Signals", layout="wide")
st.title("Multi Wick Play Signals")
st.caption(
    "**6_Multi_Wick_Play** 백테스트 페이지와 동일한 detector / exit / 사이징 "
    "설정으로 현재 시점의 실매수 후보를 스캔. Watchlist는 Postgres "
    "`buy_signals` 테이블 공유 (pattern_name으로 wick_play / wedge_pop 구분). "
    "Wick Play detector는 psych / regime / macro / prior-trend 필터를 "
    "내부에서 모두 적용 — scanner는 추가 gate 없이 통과 signal만 표시."
)
try:
    st.page_link(
        "pages/6_Multi_Wick_Play.py",
        label="📉 Multi Wick Play — 같은 전략의 백테스트 튜닝",
        icon="⬅️",
    )
except Exception:
    pass


# ------------------------------------------------------------------
# Prominent "refresh all targets" CTA — identical UX to the wedgepop
# signals page. Session flag consumed further down once the scanner
# has been built.
# ------------------------------------------------------------------
st.divider()
top_refresh_cols = st.columns([3, 1])
top_refresh_cols[0].markdown(
    "### 🔄 최신 데이터로 새로고침\n"
    "scan 결과 + watchlist의 **Exhaustion TP · Breakeven arm/exit · "
    "EMA trail · latest close**를 오늘 바 기준으로 재계산. "
    "signal 수만큼 yfinance 호출 발생. 자동 새로고침은 비활성화."
)
if top_refresh_cols[1].button(
    "🔄 모두 새로고침",
    type="primary",
    use_container_width=True,
    key="wp_refresh_all_top_btn",
):
    st.session_state._wp_refresh_all_requested = True
    st.session_state._wp_top_refresh_just_clicked = True
st.divider()

# Loud banner when "모두 새로고침" was clicked but there is literally
# nothing to refresh — last-scan empty AND watchlist empty for this
# pattern. Without it, the button silently does nothing, which the
# user experienced as "눌렀는데 아무것도 안 나와".
if st.session_state.pop("_wp_top_refresh_just_clicked", False):
    _has_scan = bool(st.session_state.get("wp_last_scan"))
    _has_saved = any(
        s.pattern_name in {"wick_play", "manual"}
        for s in (
            st.session_state.signal_repo.list()
            if "signal_repo" in st.session_state
            else []
        )
    )
    if not _has_scan and not _has_saved:
        st.warning(
            "🔄 새로고침할 대상이 없어. **좌측 사이드바의 "
            "`Scan for Signals`** 를 먼저 눌러 신호를 찾거나, "
            "**Manual Add** 로 watchlist에 종목을 추가해줘. "
            "이 버튼은 *이미 떠 있는* scan 결과 / watchlist의 "
            "TP·Breakeven·EMA trail 레벨을 오늘 바 기준으로 재계산하는 용도."
        )


# --- Repository (shared with Multi Wedgepop Signals) ---
if "signal_repo" not in st.session_state:
    try:
        st.session_state.signal_repo = PostgresSignalRepo()
        st.session_state.signal_repo_kind = "postgres"
    except Exception as e:
        st.warning(
            f"Postgres 연결 실패 → in-memory 저장소로 fallback: {e}"
        )
        st.session_state.signal_repo = InMemorySignalRepo()
        st.session_state.signal_repo_kind = "in-memory"
repo = st.session_state.signal_repo


def compute_sizing(
    entry_price: float,
    stop_loss: float,
    capital: float,
    risk_pct: float,
) -> dict:
    """Position-sizing math — mirrors the backtest rule
    (``shares = capital × risk / (entry − stop)``) plus the no-leverage
    cap (``shares × entry ≤ capital``) so the shown number matches
    what the strategy would actually buy.
    """
    risk_per_share = entry_price - stop_loss
    if risk_per_share <= 0 or entry_price <= 0 or capital <= 0:
        return {"shares": 0}
    risk_amount = capital * (risk_pct / 100.0)
    shares_by_risk = int(risk_amount / risk_per_share)
    shares_by_capital = int(capital / entry_price)
    shares = max(0, min(shares_by_risk, shares_by_capital))
    if shares == 0:
        return {"shares": 0}
    position_value = shares * entry_price
    return {
        "shares": shares,
        "position_value": position_value,
        "pct_of_capital": position_value / capital * 100.0,
        "intended_risk_dollar": shares * risk_per_share,
        "intended_risk_pct": shares * risk_per_share / capital * 100.0,
        "binding_constraint": (
            "capital" if shares_by_capital <= shares_by_risk else "risk"
        ),
        "shares_by_risk_uncapped": shares_by_risk,
    }


# --- Sidebar inputs — mirrors 6_Multi_Wick_Play.py ---
with st.sidebar:
    st.header("Universe")
    universe = st.selectbox(
        "Universe",
        options=["sp500", "nasdaq100", "nasdaq_full"],
        index=0,
        format_func=lambda x: {
            "sp500": "S&P 500 (~500)",
            "nasdaq100": "Nasdaq-100 (~100)",
            "nasdaq_full": "Nasdaq All Common Stocks (~2,200)",
        }[x],
    )
    max_tickers = st.number_input(
        "Max tickers (0 = all)",
        value=50, min_value=0, max_value=3000, step=10,
        help="스캔할 최대 종목 수. 0이면 전체.",
    )
    max_workers = st.number_input(
        "Parallel workers",
        value=8, min_value=1, max_value=32, step=1,
    )

    st.header("Period")
    lookback_days = st.number_input(
        "Signal window (days)",
        value=5, min_value=1, max_value=60, step=1,
        help="오늘로부터 N일 이내에 발생한 signal만 표시.",
    )

    st.header("Risk / Position Sizing")
    account_capital_ui = st.number_input(
        "Account Capital ($)",
        value=100_000, min_value=100, step=1_000,
    )
    risk_pct_ui = st.number_input(
        "Risk per Trade (%)",
        value=2.0, min_value=0.1, max_value=100.0, step=0.5,
    )

    st.header("Wick Play Detector")
    min_upper_wick_ratio = st.number_input(
        "Min upper wick / range",
        value=0.5, min_value=0.0, max_value=1.0, step=0.05, format="%.2f",
    )
    max_volume_dryup = st.number_input(
        "Max inside-bar vol vs wick bar",
        value=1.0, min_value=0.1, max_value=2.0, step=0.05, format="%.2f",
    )
    breakout_trigger = st.selectbox(
        "Breakout trigger",
        options=["wick_high", "inside_high"],
        index=0,
    )
    enable_max_wick_range = st.checkbox("Cap wick-bar range (× ATR)", value=True)
    max_wick_range_atr = st.number_input(
        "Max wick-bar range (× ATR)",
        value=2.5, min_value=0.5, max_value=10.0, step=0.1, format="%.2f",
        disabled=not enable_max_wick_range,
    )
    cooldown_bars = st.number_input(
        "Cooldown bars",
        value=5, min_value=0, max_value=60, step=1,
    )

    st.subheader("Psychology score (4 checks)")
    min_psych_score = st.slider("Min psych score", 0, 4, 3, 1)
    psych_vol_lookback = st.number_input(
        "Vol avg lookback (days)",
        value=20, min_value=5, max_value=100, step=5,
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
        "Check 3: prior red streak (fail if all red)",
        value=2, min_value=0, max_value=10, step=1,
    )
    psych_dramatic_wick_ratio = st.number_input(
        "Check 4: dramatic wick ratio",
        value=0.65, min_value=0.3, max_value=1.0, step=0.05, format="%.2f",
    )

    st.subheader("Hard gates")
    min_breakout_strength_atr = st.number_input(
        "Min breakout strength (× ATR)",
        value=0.3, min_value=0.0, max_value=5.0, step=0.05, format="%.2f",
    )
    enable_min_prior_trend = st.checkbox(
        "Enforce min 20-day prior trend", value=True,
    )
    min_prior_trend_20d = st.number_input(
        "Min 20-day trend (fraction)",
        value=-0.01, min_value=-0.5, max_value=0.5, step=0.005, format="%.3f",
        disabled=not enable_min_prior_trend,
    )
    min_wick_close_location = st.number_input(
        "Min wick close location (0=low, 1=high)",
        value=0.15, min_value=0.0, max_value=1.0, step=0.05, format="%.2f",
    )

    st.subheader("Macro regime gates")
    enable_regime_filter = st.checkbox(
        "Enable index regime filter (SPY)", value=True,
    )
    regime_symbol = st.text_input(
        "Regime symbol", value="^GSPC",
        disabled=not enable_regime_filter,
    )
    regime_sma = st.number_input(
        "Require SPY close > SMA period (0 = off)",
        value=50, min_value=0, max_value=400, step=10,
        disabled=not enable_regime_filter,
    )
    enable_macro_blackout = st.checkbox(
        "Enable macro-event blackout", value=True,
    )
    include_fomc_blackout = st.checkbox(
        "Include FOMC announcement days", value=False,
        disabled=not enable_macro_blackout,
    )
    include_cpi_blackout = st.checkbox(
        "Include CPI release days", value=True,
        disabled=not enable_macro_blackout,
    )
    blackout_window_days = st.number_input(
        "Blackout window (± days)",
        value=1, min_value=0, max_value=3, step=1,
        disabled=not enable_macro_blackout,
    )

    st.header("Exits (for TP / BE level display)")
    ema_trail = st.number_input(
        "EMA trail period",
        value=10, min_value=3, max_value=50, step=1,
    )
    enable_exh_exit = st.checkbox(
        "Enable Exhaustion Extension Top exit (TP level)", value=True,
    )
    exh_extension_atr = st.number_input(
        "Exh extension above EMA (× ATR)",
        value=2.2, min_value=0.5, max_value=10.0, step=0.1,
        disabled=not enable_exh_exit,
    )
    exh_min_slope = st.number_input(
        "Exh min slow-EMA slope",
        value=0.005, min_value=-1.0, max_value=1.0, step=0.001, format="%.4f",
        disabled=not enable_exh_exit,
    )
    enable_breakeven = st.checkbox(
        "Enable breakeven stop (BE level)", value=True,
    )
    breakeven_arm_r = st.number_input(
        "Arm at +R multiple",
        value=2.0, min_value=0.3, max_value=5.0, step=0.1, format="%.2f",
        disabled=not enable_breakeven,
    )
    breakeven_offset_r = st.number_input(
        "Exit offset (× R)",
        value=1.0, min_value=-0.5, max_value=2.0, step=0.05, format="%.2f",
        disabled=not enable_breakeven,
    )

    scan_btn = st.button(
        "Scan for Signals", type="primary", use_container_width=True
    )


# --- Composition root: build scanner ---
def build_scanner() -> WickPlayBuySignalScanner:
    market_data = CachedMarketDataAdapter(YFinanceAdapter())
    universe_provider = default_universe_provider()

    regime_df = None
    if enable_regime_filter:
        try:
            fetch_start = date.today() - timedelta(days=400)
            regime_df = market_data.fetch_ohlcv(
                regime_symbol, fetch_start, date.today()
            )
        except Exception as e:
            st.error(f"{regime_symbol} fetch 실패: {e}")
            st.stop()

    blackout_set = None
    if enable_macro_blackout and (include_fomc_blackout or include_cpi_blackout):
        blackout_set = build_blackout_dates(
            start=date.today() - timedelta(days=400),
            end=date.today(),
            include_fomc=include_fomc_blackout,
            include_cpi=include_cpi_blackout,
            window_days=int(blackout_window_days),
        )

    detector = WickPlayDetector(
        min_upper_wick_ratio=float(min_upper_wick_ratio),
        max_volume_dryup=float(max_volume_dryup),
        breakout_trigger=breakout_trigger,
        stop_mode="wick_low",
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
        min_breakout_strength_atr=float(min_breakout_strength_atr),
        min_prior_trend_20d=(
            float(min_prior_trend_20d) if enable_min_prior_trend else None
        ),
        min_wick_close_location=float(min_wick_close_location),
        regime_df=regime_df if enable_regime_filter else None,
        regime_min_above_sma=(
            int(regime_sma) if enable_regime_filter and regime_sma > 0 else None
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
        enable_breakeven_stop=bool(enable_breakeven),
        breakeven_arm_r_multiple=float(breakeven_arm_r),
        breakeven_exit_offset_r=float(breakeven_offset_r),
    )
    return WickPlayBuySignalScanner(
        market_data=market_data,
        universe_provider=universe_provider,
        detector=detector,
        strategy=strategy,
        max_workers=int(max_workers),
    )


# --- Scan action ---
def _run_scan() -> None:
    scanner = build_scanner()
    with st.spinner(
        f"Scanning {universe} ({max_tickers or 'all'} tickers, last {lookback_days}d)..."
    ):
        try:
            found = scanner.scan(
                universe=universe,
                lookback_days=int(lookback_days),
                max_tickers=int(max_tickers) if max_tickers > 0 else None,
            )
        except Exception as e:
            st.error(f"Scan 실패: {e}")
            st.stop()
    st.session_state.wp_last_scan = found
    st.session_state.wp_last_scan_at = datetime.utcnow().isoformat(timespec="seconds")
    # Carry diagnostic counters forward so the empty-result banner
    # below can distinguish "no data" from "no matching pattern".
    st.session_state.wp_last_scan_stats = scanner.last_scan_stats
    st.success(f"{len(found)} active Wick Play signal(s) found.")


if scan_btn:
    _run_scan()


# --- Scan results ---
st.subheader("Scan Results")
st.caption(
    "**같은 날짜 내에서 buy/sell 비율 내림차순**으로 정렬 — Multi Wick Play "
    "walker가 그날 고를 후보 순서와 동일. 최근 날짜가 위. Detector의 "
    "psych/regime/macro 필터를 이미 통과한 signal만 여기 도달."
)


def _wp_scan_age_hours() -> float | None:
    s = st.session_state.get("wp_last_scan_at")
    if not s:
        return None
    try:
        return (datetime.utcnow() - datetime.fromisoformat(s)).total_seconds() / 3600
    except Exception:
        return None


last_scan: list[BuySignal] = st.session_state.get("wp_last_scan", [])
age = _wp_scan_age_hours()

if last_scan and age is not None and age >= 24:
    st.warning(
        f"이전 스캔이 {age:.1f}시간 전입니다. 최신 신호를 보려면 **Scan for Signals**를 다시 눌러주세요."
    )

if last_scan:
    latest_bar_dates = [
        s.metadata.get("latest_date") for s in last_scan
        if s.metadata.get("latest_date")
    ]
    latest_bar = max(latest_bar_dates) if latest_bar_dates else None
    anchor = (
        date.fromisoformat(latest_bar)
        if isinstance(latest_bar, str)
        else (latest_bar if isinstance(latest_bar, date) else date.today())
    )
    window_days = int(lookback_days)
    expected: list[date] = []
    d = anchor
    while len(expected) < window_days and (anchor - d).days <= window_days * 2:
        if d.weekday() < 5:
            expected.append(d)
        d -= timedelta(days=1)
    for s in last_scan:
        if s.signal_date not in expected:
            expected.append(s.signal_date)
    expected.sort(reverse=True)

    counts = {d: 0 for d in expected}
    for s in last_scan:
        if s.signal_date in counts:
            counts[s.signal_date] += 1
    dist = ", ".join(f"{d.isoformat()}({counts[d]})" for d in expected)

    age_label = f"{age:.1f}h ago" if age is not None else "unknown"
    latest_label = latest_bar if latest_bar else "?"
    st.caption(
        f"📊 마지막 스캔: **{age_label}** · 데이터 최신 바: "
        f"**{latest_label}** · signal_date 분포: {dist}"
    )

# Manual refresh: consumes the top-level flag (also shared with
# watchlist refresh below) to rerun refresh_targets on every scan
# result.
if last_scan and st.session_state.get("_wp_refresh_all_requested"):
    refresher = build_scanner()
    refreshed: dict[str, BuySignal] = {}
    with st.spinner(f"Scan 결과 {len(last_scan)}개 최신화 중..."):
        for sig in last_scan:
            try:
                refreshed[sig.id] = refresher.refresh_targets(sig)
            except Exception as e:
                st.warning(f"{sig.ticker} scan 새로고침 실패: {e}")
                refreshed[sig.id] = sig
    last_scan = [refreshed.get(s.id, s) for s in last_scan]
    st.session_state.wp_last_scan = last_scan

if not last_scan:
    stats = st.session_state.get("wp_last_scan_stats")
    if stats is None:
        # Never scanned this session.
        st.info(
            "좌측에서 **Scan for Signals** 를 눌러 최근 Wick Play 매수 신호를 불러와."
        )
    else:
        # Scanned but got 0 signals — explain WHY using the
        # collected stats. Three distinct modes:
        #   (a) no data was fetchable         → "no data"
        #   (b) data ok, detector never fired → "no pattern"
        #   (c) detector fired only before cutoff → "old signals"
        req = stats.get("tickers_requested", 0)
        ok = stats.get("tickers_with_data", 0)
        failed = stats.get("tickers_fetch_failed", 0)
        short = stats.get("tickers_too_few_bars", 0)
        total_hits = stats.get("total_detector_hits_history", 0)
        in_window = stats.get("in_window_hits", 0)
        lookback = stats.get("lookback_days", lookback_days)

        if ok == 0:
            st.error(
                f"📡 **데이터를 받지 못했습니다** — 요청 {req}종목 중 "
                f"fetch 실패 {failed}, 바 수 부족 {short}. "
                "yfinance 네트워크 / ticker 심볼 / 날짜 범위를 확인해줘."
            )
        elif total_hits == 0:
            st.warning(
                f"🔍 **패턴 자체가 없음** — {ok}/{req}종목을 스캔했지만 "
                f"설정값으로 Wick Play가 단 한 번도 감지되지 않음. "
                "좌측 detector 필터(psych score, min prior trend, "
                "regime SMA, macro blackout 등)가 너무 엄격할 수 있어. "
                "min_psych_score나 regime 필터를 완화해봐."
            )
        elif in_window == 0:
            st.warning(
                f"📅 **최근엔 패턴이 없음** — {ok}/{req}종목에서 과거 총 "
                f"{total_hits}개의 Wick Play가 감지됐지만 모두 "
                f"최근 {lookback}일 밖. Signal window를 늘리거나 "
                "더 오래된 setup도 보려면 lookback_days를 키워."
            )
        else:
            # Shouldn't happen (in_window > 0 but len(last_scan) == 0)
            # but guard anyway.
            st.info(
                f"Scan 완료 · 요청 {req} / 데이터 {ok} / 과거 히트 "
                f"{total_hits} / 윈도우 내 {in_window} — 그런데 "
                "최종 결과가 비어있어. 내부 버그일 수 있으니 재시도."
            )
        # Always show the raw counter block for transparency.
        with st.expander("🔬 Scan diagnostics", expanded=False):
            st.json(stats)
else:
    for sig in last_scan:
        meta = sig.metadata
        entry_confirmed = meta.get("entry_confirmed", False)
        entry_label = "entry" if entry_confirmed else "entry (prov.)"
        header = (
            f"{sig.ticker} — signal {sig.signal_date} · "
            f"{entry_label} ${sig.entry_price:.2f} / stop ${sig.stop_loss:.2f}"
        )
        if not entry_confirmed:
            header += " · ⏳ next-bar open pending"
        with st.expander(header, expanded=False):
            src_cap = (
                f"**Entry source**: actual next-bar open on "
                f"{meta.get('entry_date', '?')} (${sig.entry_price:.2f})"
                if entry_confirmed
                else "**Entry source**: provisional = signal close "
                f"(${meta.get('signal_close', sig.entry_price):.2f}) — "
                "entry bar not printed yet"
            )
            st.caption(src_cap)

            # Detector rationale
            c1, c2, c3 = st.columns(3)
            c1.metric("Psych score", f"{meta.get('psych_score', 0)}/4")
            c2.metric("Upper wick ratio", f"{meta.get('upper_wick_ratio', 0):.3f}")
            c3.metric("Breakout strength (×ATR)", f"{meta.get('breakout_strength_atr', 0):.2f}")

            d1, d2, d3 = st.columns(3)
            d1.metric("Wick range (×ATR)", f"{meta.get('wick_range_atr', 0):.2f}")
            d2.metric("20d prior trend", (
                f"{meta.get('prior_trend_20d', 0):+.2%}"
                if meta.get("prior_trend_20d") is not None
                else "—"
            ))
            d3.metric("Wick close loc (0=low)", f"{meta.get('wick_close_location', 0):.2f}")

            # Latest close vs entry
            latest_close = meta.get("latest_close")
            if latest_close:
                delta_pct = (latest_close - sig.entry_price) / sig.entry_price * 100
                st.metric(
                    f"Latest close ({meta.get('latest_date', '?')})",
                    f"${latest_close:.2f}",
                    delta=f"{delta_pct:+.2f}% vs entry",
                )

            # Stop level (wick low, from signal)
            st.markdown("**손절 — Wick Low (WickPlayStrategy 초기 stop)**")
            st.metric(
                "Wick Low stop",
                f"${sig.stop_loss:.2f}",
                help="진입 직후 low ≤ wick_low 면 즉시 청산. Wick Play canonical stop.",
            )

            # Take-profit levels
            st.markdown("**익절 라인 — WickPlayStrategy exit 레벨**")
            exh_primary = meta.get("target_exhaustion_primary")
            r_to_exh = meta.get("r_to_exhaustion_primary")
            ema_trail_now = meta.get("ema_trail_current")
            t2r = meta.get("target_2r")
            t3r = meta.get("target_3r")

            e1, e2, e3 = st.columns(3)
            e1.metric(
                "Exhaustion Top — primary",
                f"${exh_primary:.2f}" if exh_primary else "—",
                delta=f"{r_to_exh:.2f}R" if r_to_exh is not None else None,
                help="ExhaustionExtensionTopDetector: high ≥ ema_fast + "
                "extension_atr_mult × ATR 이면 발동 후보. 발동 시 그 바 close에서 exit.",
            )
            e2.metric(
                "EMA trail — current",
                f"${ema_trail_now:.2f}" if ema_trail_now else "—",
                help=f"{int(ema_trail)} EMA 현재값. 종가가 이 선 아래로 떨어지면 Kell stairstep trail 발동.",
            )
            e3.metric(
                "+2R target (reference)",
                f"${t2r:.2f}" if t2r else "—",
                delta=f"{t3r:+.2f} = +3R" if t3r else None,
                help="순수 R-multiple 기준선. Exit trigger 아님 — 수익률 잣대로만.",
            )

            # Breakeven levels (if enabled in strategy)
            be_arm = meta.get("breakeven_arm_price")
            be_exit = meta.get("breakeven_exit_price")
            if be_arm and be_exit:
                st.markdown("**Breakeven stop — armed / lock level**")
                b1, b2 = st.columns(2)
                b1.metric(
                    f"Arm at +{meta.get('breakeven_arm_r', '?')}R",
                    f"${be_arm:.2f}",
                    help="종가가 이 레벨에 닿으면 breakeven stop 활성화.",
                )
                b2.metric(
                    f"Lock at +{meta.get('breakeven_offset_r', '?')}R",
                    f"${be_exit:.2f}",
                    help="arm 이후 price가 이 레벨을 밟으면 lock-in profit으로 청산.",
                )

            # Risk per share + sizing
            risk_ps = meta.get("risk_per_share")
            if risk_ps:
                st.caption(
                    f"Risk per share ≈ ${risk_ps:.2f} (entry − wick_low). "
                    f"1R × shares = 실제 손실."
                )

            sizing = compute_sizing(
                entry_price=sig.entry_price,
                stop_loss=sig.stop_loss,
                capital=float(account_capital_ui),
                risk_pct=float(risk_pct_ui),
            )
            st.markdown("**포지션 사이즈**")
            if sizing["shares"] == 0:
                st.warning("계산 불가 — entry ≤ stop 이거나 자본/가격 문제.")
            else:
                p1, p2, p3, p4 = st.columns(4)
                p1.metric("Shares", f"{sizing['shares']:,}")
                p2.metric("Position $", f"${sizing['position_value']:,.0f}")
                p3.metric("% of capital", f"{sizing['pct_of_capital']:.2f}%")
                p4.metric(
                    "Intended risk",
                    f"${sizing['intended_risk_dollar']:,.0f}",
                    delta=f"{sizing['intended_risk_pct']:.2f}%",
                )
                binding = sizing["binding_constraint"]
                if binding == "capital":
                    uncapped = sizing["shares_by_risk_uncapped"]
                    st.caption(
                        f"⚠ No-leverage cap binding — risk 공식은 {uncapped:,}주를 "
                        f"요구하지만 자본 한도로 {sizing['shares']:,}주까지만."
                    )

            st.caption(
                f"Trigger: {meta.get('trigger', '?')} · "
                f"Buy/Sell: {meta.get('buy_sell_ratio', 0):.2f} · "
                f"Stop mode: {meta.get('stop_mode', '?')}"
            )
            notes_input = st.text_area(
                "Notes (근거 메모)", key=f"wp_note_input_{sig.id}", height=80
            )
            save_btn = st.button(
                "Save to Watchlist",
                key=f"wp_save_{sig.id}",
                type="primary",
            )
            if save_btn:
                sig.notes = notes_input
                repo.save(sig)
                st.success(f"저장됨: {sig.ticker} {sig.signal_date}")


# --- Manual Add ---
st.divider()
st.subheader("Manual Add")
st.caption(
    "특정 종목을 특정 날짜로 watchlist에 직접 저장. Wick Play detector가 그 "
    "날짜에 signal을 찍었으면 detector 메타데이터와 함께, 아니면 "
    "**수동 등록(pattern=`manual`)** 으로 fallback."
)
ma1, ma2, ma3 = st.columns([2, 2, 1])
manual_ticker = ma1.text_input(
    "Ticker", value="", placeholder="AAPL", key="wp_manual_ticker"
).strip().upper()
manual_date = ma2.date_input(
    "Signal Date",
    value=date.today(),
    min_value=date(2000, 1, 1),
    max_value=date.today(),
    key="wp_manual_date",
)
manual_notes = ma3.text_input(
    "Notes (optional)", value="", key="wp_manual_notes"
)
manual_add_btn = st.button(
    "Add to Watchlist",
    disabled=not manual_ticker,
    key="wp_manual_add_btn",
)
if manual_add_btn and manual_ticker:
    builder = build_scanner()
    try:
        sig = builder.build_signal_at(manual_ticker, manual_date)
        sig.notes = manual_notes
        repo.save(sig)
        if sig.metadata.get("manually_added_no_signal"):
            st.warning(
                f"{manual_ticker} · {manual_date} 수동 등록 완료 — "
                f"해당 날짜에 detector signal 없어서 fallback. "
                f"entry ${sig.entry_price:.2f} / stop ${sig.stop_loss:.2f}"
            )
        else:
            st.success(
                f"{manual_ticker} · {manual_date} 저장 완료 "
                f"(detector signal 감지 · "
                f"entry ${sig.entry_price:.2f} / stop ${sig.stop_loss:.2f})"
            )
    except ValueError as e:
        st.error(f"추가 실패: {e}")
    except Exception as e:
        st.error(f"예상치 못한 오류: {e}")


# --- Watchlist ---
st.divider()
st.subheader("Watchlist")
st.caption(
    "전체 watchlist (wedge_pop / wick_play 등 모든 전략 공유). "
    "아래 필터는 pattern_name = `wick_play` 또는 `manual` 로 제한."
)
wl_cols = st.columns([3, 1])
status_filter = wl_cols[0].selectbox(
    "Status filter",
    options=["all", *[s.value for s in SignalStatus]],
    index=0,
    key="wp_status_filter",
)
refresh_btn = wl_cols[1].button(
    "🔄 최신 데이터로 새로고침",
    use_container_width=True,
    help="저장된 각 signal의 TP / BE / EMA trail 을 오늘 바 기준으로 재계산.",
    key="wp_local_refresh_btn",
)
status_value = None if status_filter == "all" else SignalStatus(status_filter)
all_saved = repo.list(status=status_value)
# Only show wick-play signals here. Manual-added signals with
# ``wick_play`` detector fallback also carry pattern_name="manual", so
# include them when the metadata flags them as a Wick Play manual add.
saved = [
    s for s in all_saved
    if s.pattern_name in {"wick_play", "manual"}
]


def _refresh_signals(signals: list[BuySignal]) -> None:
    if not signals:
        return
    refresher = build_scanner()
    progress = st.progress(0.0, text="최신 데이터로 새로고침 중...")
    for i, sig in enumerate(signals):
        try:
            updated = refresher.refresh_targets(sig)
            repo.save(updated)
        except Exception as e:
            st.warning(f"{sig.ticker} 새로고침 실패: {e}")
        progress.progress(
            (i + 1) / len(signals), text=f"{sig.ticker} 완료 ({i + 1}/{len(signals)})"
        )
    progress.empty()


top_flag = st.session_state.pop("_wp_refresh_all_requested", False)
if saved and (top_flag or refresh_btn):
    _refresh_signals(saved)
    st.success(f"{len(saved)}개 Wick Play watchlist signal 새로고침 완료.")
    all_saved = repo.list(status=status_value)
    saved = [s for s in all_saved if s.pattern_name in {"wick_play", "manual"}]

if saved:
    oldest_hours = None
    for s in saved:
        r = s.metadata.get("refreshed_at")
        if not r:
            oldest_hours = float("inf")
            break
        try:
            age = (datetime.utcnow() - datetime.fromisoformat(r)).total_seconds() / 3600
        except Exception:
            continue
        oldest_hours = age if oldest_hours is None else max(oldest_hours, age)
    if oldest_hours is not None:
        if oldest_hours == float("inf"):
            st.warning(
                "⚠️ 일부 watchlist signal이 아직 새로고침되지 않았어. 상단 **모두 새로고침** 또는 이 섹션 버튼 클릭."
            )
        elif oldest_hours >= 1.0:
            st.warning(
                f"⚠️ watchlist가 {oldest_hours:.1f}시간 전 데이터 기준. 최신값 보려면 새로고침."
            )

if not saved:
    st.info("저장된 Wick Play signal 없음.")
else:
    for sig in saved:
        with st.expander(
            f"[{sig.status.value.upper()}] {sig.ticker} — {sig.signal_date} "
            f"· entry ${sig.entry_price:.2f}",
            expanded=False,
        ):
            st.write(f"**Pattern**: {sig.pattern_name}")
            st.write(f"**Entry / Stop**: ${sig.entry_price:.2f} / ${sig.stop_loss:.2f}")
            meta = sig.metadata
            refreshed_at = meta.get("refreshed_at")
            latest_close = meta.get("latest_close")
            latest_date = meta.get("latest_date")
            if latest_close:
                delta_pct = (latest_close - sig.entry_price) / sig.entry_price * 100
                st.write(
                    f"**Latest close** ({latest_date or '?'}): "
                    f"${latest_close:.2f} ({delta_pct:+.2f}% vs entry)"
                )

            tp_bits = []
            if meta.get("target_exhaustion_primary") is not None:
                t = meta["target_exhaustion_primary"]
                r = meta.get("r_to_exhaustion_primary")
                tp_bits.append(
                    f"Exh ${t:.2f}" + (f" ({r:.2f}R)" if r is not None else "")
                )
            if meta.get("ema_trail_current") is not None:
                tp_bits.append(f"EMA trail ${meta['ema_trail_current']:.2f}")
            if tp_bits:
                st.write(f"**익절 라인**: {' · '.join(tp_bits)}")

            if meta.get("breakeven_arm_price") is not None:
                st.write(
                    f"**Breakeven**: arm +{meta.get('breakeven_arm_r', '?')}R "
                    f"@ ${meta['breakeven_arm_price']:.2f} → lock "
                    f"+{meta.get('breakeven_offset_r', '?')}R "
                    f"@ ${meta['breakeven_exit_price']:.2f}"
                )

            if refreshed_at:
                st.caption(f"마지막 새로고침: {refreshed_at} UTC")
            else:
                st.caption(
                    "⚠ 저장 시점 스냅샷 — 상단 '최신 데이터로 새로고침' 버튼으로 업데이트."
                )

            w_sizing = compute_sizing(
                entry_price=sig.entry_price,
                stop_loss=sig.stop_loss,
                capital=float(account_capital_ui),
                risk_pct=float(risk_pct_ui),
            )
            if w_sizing["shares"] > 0:
                st.write(
                    f"**포지션 사이즈** (현재 자본 ${float(account_capital_ui):,.0f} / "
                    f"risk {risk_pct_ui:.1f}%): "
                    f"{w_sizing['shares']:,}주 · "
                    f"${w_sizing['position_value']:,.0f} · "
                    f"{w_sizing['pct_of_capital']:.2f}% of capital"
                )
            st.write(f"**Saved at**: {sig.created_at.isoformat(timespec='seconds')} UTC")
            if sig.notes:
                st.write(f"**Notes**: {sig.notes}")
            new_notes = st.text_area(
                "Edit notes",
                value=sig.notes,
                key=f"wp_notes_edit_{sig.id}",
                height=80,
            )
            new_status = st.selectbox(
                "Status",
                options=[s.value for s in SignalStatus],
                index=[s.value for s in SignalStatus].index(sig.status.value),
                key=f"wp_status_sel_{sig.id}",
            )
            c1, c2, _ = st.columns(3)
            if c1.button("Update", key=f"wp_update_{sig.id}"):
                repo.update_notes(sig.id, new_notes)
                repo.update_status(sig.id, SignalStatus(new_status))
                st.rerun()
            if c2.button("Delete", key=f"wp_delete_{sig.id}"):
                repo.delete(sig.id)
                st.rerun()
