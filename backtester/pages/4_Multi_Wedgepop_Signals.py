"""Live signals view for the Multi Wedgepop strategy.

Companion to ``3_Multi_Wedgepop.py`` — same detector / filter /
sizing knobs, but applied to the **most recent bars** to surface
actionable buy setups right now. Signals persist in Postgres
(``buy_signals`` table) with notes, status transitions, deletion.

Future strategies (base-n-break, downside reversal, etc.) each get
their own numbered signals page that reuses the same
``SignalScannerPort`` + ``SignalRepositoryPort`` seams + shared UI
patterns. The watchlist table is strategy-agnostic so cross-strategy
holdings live in one place.
"""

from datetime import date, datetime, timedelta

import streamlit as st

from data.adapters.cached_market_data import CachedMarketDataAdapter
from data.adapters.wikipedia_universe import WikipediaUniverseAdapter
from data.adapters.yfinance_adapter import YFinanceAdapter
from pattern.adapters.exhaustion_extension_top import ExhaustionExtensionTopDetector
from pattern.adapters.wedge_pop import WedgePopDetector
from signals.adapters.in_memory_repo import InMemorySignalRepo
from signals.adapters.postgres_signal_repo import PostgresSignalRepo
from signals.adapters.universe_scanner import UniverseBuySignalScanner
from signals.domain.models import BuySignal, SignalStatus
from strategy.adapters.wedgepop_strategy import WedgepopStrategy

st.set_page_config(page_title="Multi Wedgepop Signals", layout="wide")
st.title("Multi Wedgepop Signals")
st.caption(
    "**3_Multi_Wedgepop** 백테스트 페이지와 동일한 detector / 필터 / 사이징 "
    "설정으로, 지금 시점의 실매수 후보를 스캔. 관심 종목은 watchlist로 저장 "
    "(Postgres `buy_signals` 테이블, DB 불가 시 in-memory fallback). "
    "향후 다른 전략 추가 시 각 전략별 Signals 페이지를 별도로 생성 — "
    "watchlist 저장소는 전략 간 공유."
)
try:
    st.page_link(
        "pages/3_Multi_Wedgepop.py",
        label="📉 Multi Wedgepop — 같은 전략의 백테스트 튜닝",
        icon="⬅️",
    )
except Exception:
    pass


# --- Repository: Postgres by default, fall back to in-memory if the
#     DB is unreachable (e.g. running the page outside docker-compose).
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


def _needs_refresh(sig: BuySignal, ttl_minutes: int = 30) -> bool:
    """Auto-refresh guard: any signal missing a ``refreshed_at`` stamp
    or whose last refresh is older than ``ttl_minutes`` is treated
    as stale. New-day rollovers always fall through since the stamp
    is older than 24h × 60 min."""
    r = sig.metadata.get("refreshed_at")
    if not r:
        return True
    try:
        rd = datetime.fromisoformat(r)
    except Exception:
        return True
    return (datetime.utcnow() - rd).total_seconds() >= ttl_minutes * 60


def compute_sizing(
    entry_price: float,
    stop_loss: float,
    capital: float,
    risk_pct: float,
) -> dict:
    """Return the position sizing plan for a buy signal.

    Mirrors the backtest rule (``shares = capital × risk / (entry −
    stop)``) plus the no-leverage cap (``shares × entry ≤ capital``)
    so the page displays the same number of shares the strategy
    would actually buy. Also reports which constraint binds:
    ``"risk"`` when the risk-budget caps shares, ``"capital"`` when
    a tight stop inflated share count beyond account equity and the
    no-leverage cap kicked in.
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


# --- Sidebar inputs — 3_Multi_Wedgepop 섹션 순서 그대로 ---
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
        help="스캔할 최대 종목 수. 0이면 전체.",
    )
    max_workers = st.number_input(
        "Parallel workers",
        value=8,
        min_value=1,
        max_value=32,
        step=1,
        help="yfinance 호출을 동시에 몇 개 보낼지.",
    )

    st.header("Period")
    lookback_days = st.number_input(
        "Signal window (days)",
        value=5,
        min_value=1,
        max_value=60,
        step=1,
        help="오늘로부터 N일 이내에 발생한 signal만 표시. "
        "(백테스트의 Start/End Date 대신 '현재 기준 N일 윈도우'로 대체.)",
    )

    st.header("Risk / Position Sizing")
    account_capital_ui = st.number_input(
        "Account Capital ($)",
        value=100_000,
        min_value=100,
        step=1_000,
        help="현재 보유 자금. 아래 Risk per Trade와 no-leverage cap이 이 금액 기준으로 계산.",
    )
    risk_pct_ui = st.number_input(
        "Risk per Trade (%)",
        value=5.0,
        min_value=0.1,
        max_value=100.0,
        step=0.5,
        help="한 트레이드에서 1R 손절 시 감수할 자본 비율. "
        "shares = 자본 × risk% / (entry − stop).",
    )

    st.header("Pattern Detection")
    consolidation_pct = st.number_input(
        "Min consolidation %",
        value=30.0,
        min_value=0.0,
        max_value=100.0,
        step=5.0,
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
    require_above_long_smas = st.checkbox("Require close above 50 & 200 SMA", value=True)
    late_entry_bars_wp = st.number_input(
        "Late-entry bars (0 = strict fresh-cross)",
        value=0,
        min_value=0,
        max_value=10,
        step=1,
    )
    detect_lookback = st.number_input(
        "Consolidation lookback (days)",
        value=10,
        min_value=3,
        max_value=60,
        step=1,
    )
    cooldown_bars_ui = st.number_input(
        "Cooldown bars (after a signal)",
        value=0,
        min_value=0,
        max_value=60,
        step=1,
    )
    ema_fast = st.number_input("Fast EMA period", value=10, min_value=2, max_value=100, step=1)
    ema_slow = st.number_input("Slow EMA period", value=20, min_value=2, max_value=200, step=1)
    slope_lookback = st.number_input(
        "Slope lookback (days)",
        value=10,
        min_value=5,
        max_value=120,
        step=1,
    )

    st.header("Entry Filter")
    st.caption("**EMA slow slope 범위 필터**")
    enable_min_slope = st.checkbox("Enforce min EMA slow slope", value=True)
    min_ema_slow_slope_ui = st.number_input(
        "Min EMA slow slope",
        value=0.005,
        min_value=-1.0,
        max_value=1.0,
        step=0.01,
        format="%.3f",
        disabled=not enable_min_slope,
    )
    enable_max_slope = st.checkbox("Enforce max EMA slow slope", value=False)
    max_ema_slow_slope_ui = st.number_input(
        "Max EMA slow slope",
        value=0.30,
        min_value=-1.0,
        max_value=5.0,
        step=0.05,
        format="%.3f",
        disabled=not enable_max_slope,
    )
    st.caption(
        "require_gap_up / EMA-extension / swing-resistance 같은 "
        "**entry bar 필터**는 실제 진입 시점(다음 바 open)에만 판정 가능 — "
        "여기선 생략."
    )

    st.header("Swing Pivot (S/R + Trendline)")
    swing_pivot_left_ui = st.number_input(
        "Pivot window — left bars", value=2, min_value=1, max_value=10, step=1
    )
    swing_pivot_right_ui = st.number_input(
        "Pivot window — right bars", value=2, min_value=1, max_value=10, step=1
    )
    swing_pivot_lookback_ui = st.number_input(
        "Pivot lookback (bars)", value=60, min_value=10, max_value=500, step=5
    )
    enable_trendline_exit_ui = st.checkbox(
        "Enable higher-low trendline exit",
        value=True,
        help="카드에서 HL Trendline 손절 라인을 계산/표시할지.",
    )
    enable_resistance_break_exit_ui = st.checkbox(
        "Enable resistance-break exit",
        value=False,
        help="카드에서 Resistance-break supports / overhead hurdles / "
        "Next resistance 익절 라인을 계산/표시할지. 10년 전체 "
        "스캔에선 이 exit가 drag여서 기본 OFF.",
    )

    st.header("Extras (signal-bar gates)")
    enable_market_regime_filter = st.checkbox(
        "Market regime filter — SPY > 200 SMA",
        value=True,
        help="signal 날짜의 SPY close가 200 SMA 위일 때만 통과.",
    )
    enable_signal_close_strength = st.checkbox(
        "Signal close-strength filter",
        value=False,
        help="signal 봉 close가 당일 range 상단 N% 이상.",
    )
    min_signal_close_location_ui = st.number_input(
        "Min signal close location (0=low, 1=high)",
        value=0.5,
        min_value=0.0,
        max_value=1.0,
        step=0.05,
        format="%.2f",
        disabled=not enable_signal_close_strength,
    )
    enable_swing_breakout = st.checkbox(
        "Swing-breakout filter",
        value=False,
        help="signal 봉 high가 직전 swing high를 buffer×ATR 이상 돌파.",
    )
    swing_breakout_buffer_atr_ui = st.number_input(
        "Swing-breakout buffer (× ATR)",
        value=0.0,
        min_value=0.0,
        max_value=3.0,
        step=0.05,
        format="%.2f",
        disabled=not enable_swing_breakout,
    )
    enable_euphoria_cap = st.checkbox(
        "Signal bar euphoria cap",
        value=True,
        help="(close-open)/ATR가 임계값 초과 시 거부.",
    )
    max_signal_bar_gain_atr_ui = st.number_input(
        "Max signal bar gain (× ATR)",
        value=2.5,
        min_value=0.5,
        max_value=10.0,
        step=0.1,
        format="%.2f",
        disabled=not enable_euphoria_cap,
    )

    st.header("Exit — Exhaustion Top (TP 레벨 계산용)")
    enable_exh_exit = st.checkbox(
        "Enable Exhaustion Extension Top exit",
        value=True,
        help="Exit detector 파라미터 — TP 레벨 계산에만 씀. "
        "scan gate에는 영향 없음.",
    )
    exh_exit_extension_atr = st.number_input(
        "Min extension above EMA (× ATR)",
        value=1.9,
        min_value=0.5,
        max_value=20.0,
        step=0.1,
        disabled=not enable_exh_exit,
    )
    exh_exit_min_slope = st.number_input(
        "Min slow EMA slope",
        value=0.005,
        min_value=-1.0,
        max_value=1.0,
        step=0.001,
        format="%.4f",
        disabled=not enable_exh_exit,
    )
    exh_exit_max_close_loc = st.number_input(
        "Max close location",
        value=0.5,
        min_value=0.0,
        max_value=1.0,
        step=0.05,
        format="%.2f",
        disabled=not enable_exh_exit,
    )
    exh_exit_min_sell_dom = st.number_input(
        "Min sell dominance",
        value=1.5,
        min_value=0.0,
        max_value=10.0,
        step=0.1,
        disabled=not enable_exh_exit,
    )
    exh_exit_rejection_override = st.checkbox(
        "Upper-wick rejection override",
        value=True,
        disabled=not enable_exh_exit,
    )
    st.caption(
        "break-even / gap-down / structural-exit 관련 knob은 **exit 전용** — "
        "scan 결과에 영향 없어 생략."
    )

    scan_btn = st.button("Scan for Signals", type="primary", use_container_width=True)


# --- Composition root: build the scanner from adapters + filter knobs ---
def build_scanner() -> UniverseBuySignalScanner:
    market_data = CachedMarketDataAdapter(YFinanceAdapter())
    universe_provider = WikipediaUniverseAdapter()

    market_regime_df = None
    if enable_market_regime_filter:
        fetch_start = date.today() - timedelta(days=400)
        try:
            market_regime_df = market_data.fetch_ohlcv("SPY", fetch_start, date.today())
        except Exception as e:
            st.error(f"SPY fetch 실패: {e}")
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

    # Respect the user-facing exit toggles so the scanner only
    # computes/surfaces the levels the user intends to act on. If
    # *either* exit flag is on, swing columns still get populated
    # in ``_with_indicators`` (they're needed by whichever flag is
    # on). ``_compute_targets`` then gates each field individually
    # on the same flag the strategy would check at exit time.
    strategy = WedgepopStrategy(
        market_data=market_data,
        detector=detector,
        ema_trail=int(ema_fast),
        ema_slow=int(ema_slow),
        max_entry_ema_extension_atr=None,
        max_ema_slope_decline=None,
        min_ema_slow_slope=(min_ema_slow_slope_ui if enable_min_slope else None),
        max_ema_slow_slope=(max_ema_slow_slope_ui if enable_max_slope else None),
        exit_detector=exit_detector,
        enable_swing_resistance_filter=False,
        swing_pivot_left=int(swing_pivot_left_ui),
        swing_pivot_right=int(swing_pivot_right_ui),
        swing_pivot_lookback=int(swing_pivot_lookback_ui),
        enable_trendline_exit=enable_trendline_exit_ui,
        enable_resistance_break_exit=enable_resistance_break_exit_ui,
        enable_market_regime_filter=enable_market_regime_filter,
        market_regime_df=market_regime_df,
        enable_signal_close_strength_filter=enable_signal_close_strength,
        min_signal_close_location=float(min_signal_close_location_ui),
        enable_swing_breakout_filter=enable_swing_breakout,
        swing_breakout_buffer_atr=float(swing_breakout_buffer_atr_ui),
        max_signal_bar_gain_atr=(
            float(max_signal_bar_gain_atr_ui) if enable_euphoria_cap else None
        ),
    )
    return UniverseBuySignalScanner(
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
    st.session_state.last_scan = found
    st.session_state.last_scan_at = datetime.utcnow().isoformat(timespec="seconds")
    st.success(f"{len(found)} active signal(s) found.")


if scan_btn:
    _run_scan()


# --- Show last scan results ---
st.subheader("Scan Results")
st.caption(
    "**같은 날짜 내에서 buy/sell 비율 내림차순**으로 정렬 — Multi Wedge Pop이 "
    "그날 몇 등부터 사는지의 순서. 위에서부터 top-pick. 최근 날짜가 위, "
    "동일 날짜 내 후보가 여럿이면 ratio 높은 순으로. `volume_ratio ≥ 1.0` 하드 필터 포함."
)

def _scan_age_hours() -> float | None:
    s = st.session_state.get("last_scan_at")
    if not s:
        return None
    try:
        return (datetime.utcnow() - datetime.fromisoformat(s)).total_seconds() / 3600
    except Exception:
        return None


# Auto-rescan if last scan is older than 24h — signal_date is a
# historical fact tied to the bar it fired on, so stale scan results
# simply miss signals that fired after the last scan.
age = _scan_age_hours()
if (not st.session_state.get("last_scan")) or (age is not None and age >= 24):
    if st.session_state.get("last_scan"):
        st.info(
            f"이전 스캔이 {age:.1f}시간 전이라 자동 재스캔합니다 "
            "(signal_date는 해당 바에 고정되므로 재스캔 없이는 새 신호 못 잡음)."
        )
    _run_scan()

last_scan: list[BuySignal] = st.session_state.get("last_scan", [])
age = _scan_age_hours()  # re-read after potential auto-rescan

# Diagnostic banner: when was the scan run, what's the latest bar
# seen, signal-date distribution. Lets the user verify "scanner DID
# look at today" even if no new signals fired on today's bar.
if last_scan:
    latest_bar_dates = [
        s.metadata.get("latest_date") for s in last_scan
        if s.metadata.get("latest_date")
    ]
    latest_bar = max(latest_bar_dates) if latest_bar_dates else None

    # Cover every weekday in the lookback window — zero-count days
    # show explicitly (e.g. `2026-04-20(0)`) so the user can tell
    # "scanner looked at 4/20, nothing fired" vs "scanner didn't
    # reach 4/20". Anchor on the latest bar seen (not today) so
    # non-trading days at the tail don't fabricate phantom zeros.
    # Falls back to today when no latest_bar is available.
    from datetime import timedelta as _td
    anchor = (
        date.fromisoformat(latest_bar)
        if isinstance(latest_bar, str)
        else (latest_bar if isinstance(latest_bar, date) else date.today())
    )
    window_days = int(lookback_days)
    expected = []
    d = anchor
    while len(expected) < window_days and (anchor - d).days <= window_days * 2:
        if d.weekday() < 5:  # Mon-Fri (holidays may still show 0)
            expected.append(d)
        d -= _td(days=1)
    # Merge in any signal_dates not in the weekday list (unlikely but
    # covers edge cases).
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

# Auto-refresh scan results' metadata against latest bar. The scan
# itself (signal ordering, which tickers passed) reflects the
# universe+filter state at scan-click time, but the per-signal
# latest_close / HL-trendline / exhaustion / resistance levels
# should always reflect today. Uses the same ``_needs_refresh``
# staleness guard (30 min TTL) as the watchlist below.
if last_scan:
    stale_scans = [s for s in last_scan if _needs_refresh(s)]
    if stale_scans:
        refresher = build_scanner()
        refreshed_map: dict[str, BuySignal] = {}
        with st.spinner(f"Scan 결과 {len(stale_scans)}개 최신화 중..."):
            for sig in stale_scans:
                try:
                    refreshed_map[sig.id] = refresher.refresh_targets(sig)
                except Exception:
                    refreshed_map[sig.id] = sig
        last_scan = [refreshed_map.get(s.id, s) for s in last_scan]
        st.session_state.last_scan = last_scan
if not last_scan:
    st.info("좌측에서 **Scan for Signals** 를 눌러 최근 매수 신호를 불러와.")
else:
    for sig in last_scan:
        meta = sig.metadata
        gates = meta.get("filter_gates", {})
        gate_str = " ".join(
            f"{'✅' if v else '❌'} {k}" for k, v in gates.items()
        ) or "(no gates evaluated)"
        entry_confirmed = meta.get("entry_confirmed", False)
        entry_label = "entry" if entry_confirmed else "entry (prov.)"
        header = (
            f"{sig.ticker} — signal {sig.signal_date} · "
            f"{entry_label} ${sig.entry_price:.2f} / stop ${sig.stop_loss:.2f}"
        )
        if not entry_confirmed:
            header += " · ⏳ next-bar open pending"
        with st.expander(header, expanded=False):
            # Entry provenance: actual next-bar open (once printed)
            # or provisional = signal bar's close. 1R and all TPs are
            # recomputed off whichever applies, so gaps move the
            # reference levels with the price.
            src_cap = (
                f"**Entry source**: actual next-bar open on "
                f"{meta.get('entry_date', '?')} (${sig.entry_price:.2f})"
                if entry_confirmed
                else "**Entry source**: provisional = signal close "
                f"(${meta.get('signal_close', sig.entry_price):.2f}) — "
                "entry bar not printed yet"
            )
            st.caption(src_cap)

            c1, c2, c3 = st.columns(3)
            c1.metric("Breakout ATR", f"{meta.get('breakout_strength_atr', 0):.2f}")
            c2.metric("EMA slow slope", f"{meta.get('ema_slow_slope', 0):.4f}")
            c3.metric("Volume ratio", f"{meta.get('volume_ratio', 0):.2f}")

            # Latest close vs entry — how far price has already moved
            # since the computed entry. Useful to decide whether the
            # setup has run away too far to participate.
            latest_close = meta.get("latest_close")
            if latest_close:
                delta_pct = (latest_close - sig.entry_price) / sig.entry_price * 100
                st.metric(
                    f"Latest close ({meta.get('latest_date', '?')})",
                    f"${latest_close:.2f}",
                    delta=f"{delta_pct:+.2f}% vs entry",
                )

            # 손절 · 익절 라인 — *only* the levels that
            # WedgepopStrategy._find_exit actually uses to trigger
            # exits. Pure reward-framing references (consolidation
            # low as sizing basis, 2R/3R multiples) are intentionally
            # omitted per user request.
            st.markdown("**손절 라인 — actual WedgepopStrategy exit levels**")
            stop_tl = meta.get("stop_trendline_at_entry")
            tl_slope = meta.get("stop_trendline_slope")
            st.metric(
                "HL Trendline",
                f"${stop_tl:.2f}" if stop_tl else "—",
                delta=(
                    f"slope +${tl_slope:.4f}/bar" if tl_slope is not None else None
                ),
                help="trendline_break: 이후 바에서 low ≤ slope×i+intercept면 청산.",
            )

            supports = meta.get("stop_resistance_supports") or []
            if supports:
                st.markdown(
                    "Resistance-break supports (entry 아래 · low가 "
                    "`pierce_trigger` 밑으로 가면 level에서 청산):"
                )
                st.table(
                    [
                        {
                            "level": f"${s['level']:.2f}",
                            "pierce trigger (− pierce_buffer × ATR)":
                                f"${s['pierce_trigger']:.2f}",
                        }
                        for s in supports
                    ]
                )

            st.markdown("**익절 라인 — actual WedgepopStrategy exit levels**")
            exh_primary = meta.get("target_exhaustion_primary")
            r_to_exh_p = meta.get("r_to_exhaustion_primary")
            exh_reject = meta.get("target_exhaustion_rejection")
            r_to_exh_r = meta.get("r_to_exhaustion_rejection")
            next_res = meta.get("target_next_resistance")
            r_to_res = meta.get("r_to_next_resistance")

            e1, e2, e3 = st.columns(3)
            e1.metric(
                "Exhaustion Top — primary",
                f"${exh_primary:.2f}" if exh_primary else "—",
                delta=f"{r_to_exh_p:.2f}R" if r_to_exh_p is not None else None,
                help="ExhaustionExtensionTopDetector: high ≥ ema_fast + "
                "extension_atr_mult × ATR 이면 발동 후보 (slope/close-loc/"
                "sell-dom 추가 확인 필요, 미리 계산 불가). 발동 시 그 바 "
                "종가에서 exit.",
            )
            e2.metric(
                "Exhaustion Top — rejection",
                f"${exh_reject:.2f}" if exh_reject else "—",
                delta=f"{r_to_exh_r:.2f}R" if r_to_exh_r is not None else None,
                help="rejection_leniency × extension_atr_mult 로 완화된 "
                "threshold. 윗꼬리 strong인 캔들에서 조기 발동.",
            )
            e3.metric(
                "Next resistance (swing high)",
                f"${next_res:.2f}" if next_res else "—",
                delta=f"{r_to_res:.2f}R" if r_to_res is not None else None,
                help="resistance_break: high가 level+confirm_buffer×ATR 돌파 → "
                "confirmed. 실패 시 exit at level.",
            )

            hurdles = meta.get("target_resistance_hurdles") or []
            if hurdles:
                st.markdown(
                    "All overhead swing-high hurdles (돌파 후 실패 시 "
                    "해당 level에서 청산):"
                )
                st.table(
                    [
                        {
                            "level": f"${h['level']:.2f}",
                            "confirm trigger (+ confirm_buffer × ATR)":
                                f"${h['confirm_trigger']:.2f}",
                            "R-multiple": (
                                f"{h['r_multiple']:.2f}R"
                                if h.get("r_multiple") is not None else "—"
                            ),
                        }
                        for h in hurdles
                    ]
                )
            risk_ps = meta.get("risk_per_share")
            if risk_ps:
                st.caption(
                    f"Risk per share (sizing only) ≈ ${risk_ps:.2f} "
                    f"(entry − consolidation low). 전략은 이 값으로 "
                    f"position size를 잡지만 **exit trigger로는 쓰지 않음**."
                )

            # Position sizing for THIS signal at the sidebar-configured
            # capital + risk%. Matches the backtest rule + no-leverage
            # cap so what you see here is what you'd actually buy.
            sizing = compute_sizing(
                entry_price=sig.entry_price,
                stop_loss=sig.stop_loss,
                capital=float(account_capital_ui),
                risk_pct=float(risk_pct_ui),
            )
            st.markdown("**포지션 사이즈**")
            if sizing["shares"] == 0:
                st.warning(
                    "계산 불가 — entry ≤ stop 이거나 자본/가격 문제."
                )
            else:
                p1, p2, p3, p4 = st.columns(4)
                p1.metric("Shares", f"{sizing['shares']:,}")
                p2.metric(
                    "Position $",
                    f"${sizing['position_value']:,.0f}",
                )
                p3.metric(
                    "% of capital",
                    f"{sizing['pct_of_capital']:.2f}%",
                )
                p4.metric(
                    "Intended risk",
                    f"${sizing['intended_risk_dollar']:,.0f}",
                    delta=f"{sizing['intended_risk_pct']:.2f}%",
                )
                binding = sizing["binding_constraint"]
                if binding == "capital":
                    uncapped = sizing["shares_by_risk_uncapped"]
                    st.caption(
                        f"⚠ No-leverage cap binding — risk 공식은 "
                        f"{uncapped:,}주를 요구하지만 자본 한도로 "
                        f"{sizing['shares']:,}주까지만. 실제 1R 손실은 "
                        f"intended risk %보다 작을 수 있음."
                    )
                else:
                    st.caption(
                        f"Risk-capped at {risk_pct_ui:.1f}% of capital · "
                        f"Capital: ${float(account_capital_ui):,.0f}"
                    )

            st.caption(f"Filter gates: {gate_str}")
            st.caption(
                f"Trigger: {meta.get('trigger', '?')} · "
                f"Consolidation low: ${meta.get('consolidation_low', 0):.2f} · "
                f"Buy/Sell: {meta.get('buy_sell_ratio', 0):.2f}"
            )
            notes_input = st.text_area(
                "Notes (근거 메모)", key=f"note_input_{sig.id}", height=80
            )
            save_btn = st.button(
                "Save to Watchlist",
                key=f"save_{sig.id}",
                type="primary",
            )
            if save_btn:
                sig.notes = notes_input
                repo.save(sig)
                st.success(f"저장됨: {sig.ticker} {sig.signal_date}")


# --- Manual add (ticker + date) ---
st.divider()
st.subheader("Manual Add")
st.caption(
    "특정 종목을 특정 날짜로 watchlist에 직접 저장. detector가 그 "
    "날짜에 signal을 찍었으면 검출 메타데이터와 함께, 아니면 "
    "**수동 등록(pattern=`manual`)** 으로 fallback — 이 경우 "
    "entry=다음 바 open(없으면 그 바 close), stop=그 바 low 사용. "
    "stop/익절 라인은 최신 바 기준 계산됨."
)
ma1, ma2, ma3 = st.columns([2, 2, 1])
manual_ticker = ma1.text_input(
    "Ticker", value="", placeholder="AAPL", key="manual_add_ticker"
).strip().upper()
manual_date = ma2.date_input(
    "Signal Date",
    value=date.today(),
    min_value=date(2000, 1, 1),
    max_value=date.today(),
    key="manual_add_date",
)
manual_notes = ma3.text_input(
    "Notes (optional)", value="", key="manual_add_notes"
)
manual_add_btn = st.button(
    "Add to Watchlist",
    disabled=not manual_ticker,
    key="manual_add_btn",
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

# --- Watchlist (saved signals) ---
st.divider()
st.subheader("Watchlist")
wl_cols = st.columns([3, 1])
status_filter = wl_cols[0].selectbox(
    "Status filter",
    options=["all", *[s.value for s in SignalStatus]],
    index=0,
)
refresh_btn = wl_cols[1].button(
    "🔄 최신 데이터로 새로고침",
    use_container_width=True,
    help="저장된 각 signal의 stop/TP/latest-close를 "
    "오늘 바 기준으로 재계산. yfinance 호출이 signal 수만큼 발생.",
)
status_value = None if status_filter == "all" else SignalStatus(status_filter)
saved = repo.list(status=status_value)


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


# Auto-refresh on page view for signals that are stale or
# never-refreshed. Guarantees the watchlist always shows today's
# trendline / exhaustion / latest-close without the user clicking.
stale = [s for s in saved if _needs_refresh(s)]
if stale:
    _refresh_signals(stale)
    saved = repo.list(status=status_value)

if refresh_btn and saved:
    _refresh_signals(saved)
    st.success(f"{len(saved)}개 signal 강제 새로고침 완료.")
    saved = repo.list(status=status_value)

if not saved:
    st.info("저장된 signal 없음.")
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
            stop_bits = []
            if meta.get("stop_trendline_at_entry"):
                tl_val = meta["stop_trendline_at_entry"]
                slope = meta.get("stop_trendline_slope")
                label = f"HL Trendline ${tl_val:.2f}"
                if slope is not None:
                    label += f" (slope +${slope:.3f}/bar)"
                stop_bits.append(label)
            if meta.get("stop_resistance_supports"):
                top_sup = meta["stop_resistance_supports"][0]
                stop_bits.append(
                    f"Res support ${top_sup['level']:.2f} "
                    f"(pierce ${top_sup['pierce_trigger']:.2f})"
                )
            if stop_bits:
                st.write(f"**손절 라인**: {' · '.join(stop_bits)}")
            tp_bits = []
            if meta.get("target_exhaustion_primary"):
                tp_bits.append(
                    f"Exh ${meta['target_exhaustion_primary']:.2f}"
                    + (
                        f" ({meta['r_to_exhaustion_primary']:.2f}R)"
                        if meta.get("r_to_exhaustion_primary") is not None
                        else ""
                    )
                )
            if meta.get("target_next_resistance"):
                label = f"Res ${meta['target_next_resistance']:.2f}"
                if meta.get("r_to_next_resistance") is not None:
                    label += f" ({meta['r_to_next_resistance']:.2f}R)"
                tp_bits.append(label)
            if tp_bits:
                st.write(f"**익절 라인**: {' · '.join(tp_bits)}")
            if refreshed_at:
                st.caption(f"마지막 새로고침: {refreshed_at} UTC")
            else:
                st.caption(
                    "⚠ 저장 시점 스냅샷 — 위 '최신 데이터로 새로고침' 버튼으로 업데이트."
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
                key=f"notes_edit_{sig.id}",
                height=80,
            )
            new_status = st.selectbox(
                "Status",
                options=[s.value for s in SignalStatus],
                index=[s.value for s in SignalStatus].index(sig.status.value),
                key=f"status_sel_{sig.id}",
            )
            c1, c2, c3 = st.columns(3)
            if c1.button("Update", key=f"update_{sig.id}"):
                repo.update_notes(sig.id, new_notes)
                repo.update_status(sig.id, SignalStatus(new_status))
                st.rerun()
            if c2.button("Delete", key=f"delete_{sig.id}"):
                repo.delete(sig.id)
                st.rerun()
