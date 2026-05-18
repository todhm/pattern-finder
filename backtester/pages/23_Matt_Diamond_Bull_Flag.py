"""Matt Diamond Bull Flag — large-cap intraday scalper page.

Mirrors the source video (YouTube ``SNjtH42aCuk``, TSLA 2025-03-24)
mechanics rather than Ross Cameron's low-float setup. See
``docs/strategy_notes/matt_diamond_bull_flag.md`` for the rule
derivation.

Layout choices on this page
---------------------------
* Default ticker = **TSLA** (the source-video case).
* PM-high gate **on** by default (Matt-signature).
* Market-regime gate **on** by default using SPY > SMA50 — Matt
  explicitly tied the TSLA setup to "ES/NQ gapping over resistance".
* Scalper exits: fixed R-multiple TP + time stop. Add-to-winner is
  intentionally *not* exposed — Matt didn't use it.
* All intraday + daily fetches go through ``build_default_market_data()``
  so the EODHD → Massive → YFinance fallback chain (composed_market_data
  2026-05 update) handles quota exhaustion and today's not-yet-closed
  intraday bars without the page hand-wiring a yfinance bypass.
* Page does **not** auto-run on slider changes — click the **▶ Run
  backtest** button. Lets users dial parameters without paying the
  data + sim cost on every keystroke.
"""

from __future__ import annotations

from datetime import date, time, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from data.adapters.composed_market_data import build_default_market_data
from data.adapters.regular_session_filter import RegularSessionFilterAdapter
from data.domain.market_calendar import NY
from pattern.adapters.matt_diamond_bull_flag import (
    MattDiamondBullFlagDetector,
    compute_earnings_window_dates,
    compute_market_regime_ok,
    compute_news_session_dates,
    compute_premarket_high_by_date,
)
from strategy.adapters.matt_diamond_bull_flag_strategy import (
    MattDiamondBullFlagStrategy,
)
from strategy.domain.models import StrategyConfig, TossFeeSchedule

st.set_page_config(page_title="Matt Diamond Bull Flag", layout="wide")
st.title("Matt Diamond Bull Flag — Large-Cap Intraday Scalper")
st.caption(
    "TSLA-class names. Opens **above PM high** → opening drive → 1~3 bar "
    "pullback to **10 EMA** → **Green-Take-Red** entry → fixed-R scalp exit. "
    "Optional **SPY/QQQ regime** gate (영상의 'ES/NQ context first'). "
    "데이터: composed_market_data (EODHD → Massive → YFinance fallback)."
)

# ---- Sidebar -------------------------------------------------------
with st.sidebar:
    st.header("Market")
    ticker = st.text_input("Ticker", value="TSLA").strip().upper()
    today = date.today()
    # Default 90-day window — large enough to get a workable trade
    # sample on a catalyst-driven sparse setup, small enough that the
    # first fetch isn't a multi-second wait. Users can dial it to a
    # full year (EODHD intraday goes back ~2y).
    start_date = st.date_input("Start", value=today - timedelta(days=90))
    end_date = st.date_input("End", value=today)
    intraday_interval = st.selectbox(
        "Intraday bar size",
        options=["1m", "5m"],
        index=0,
        help="1m matches the source video. 5m for longer-window backtests "
        "(yfinance 1m history caps at ~30 days; 5m goes ~60 days).",
    )

    st.header("Capital / Risk")
    initial_capital = st.number_input(
        "Initial capital ($)", value=100_000.0, step=1_000.0, format="%.0f"
    )
    risk_per_trade = st.slider(
        "Risk per trade", 0.001, 0.05, 0.01, 0.001,
        help="Dollar risk per trade as a fraction of equity.",
    )
    max_position_pct = st.slider(
        "Max notional / equity", 0.05, 1.0, 0.30, 0.05,
        help="Cap shares so notional ≤ this fraction of equity.",
    )

    st.header("Session gates (Matt)")
    require_pm_high = st.checkbox(
        "Require open > PM high (hard)", value=True,
        help="Matt rule. Off only for ablation tests.",
    )
    require_regime = st.checkbox(
        "Require SPY > SMA50 (regime)", value=False,
        help="**Default OFF** — Matt 영상의 source 케이스 (TSLA 2025-03-24) "
        "는 SPY가 SMA50 *아래* 였던 날인데, SPY > SMA50 게이트로 해석하면 "
        "정통 케이스가 reject됨. Matt 원문은 'ES/NQ gapping over resistance' "
        "= 그 날 selling-into-strength 가 아닌지 정도의 컨텍스트 체크, "
        "장기 추세 필터 아님. Opt-in 으로만 켜고 보수적 필터링용으로 사용.",
    )
    regime_sma_period = st.slider(
        "Regime SMA period", 10, 200, 50, 10,
        help="Used only when SPY-regime gate is on.",
    )
    min_gap_pct = st.slider(
        "Min gap %", 0.0, 0.05, 0.001, 0.001,
        format="%.3f",
        help="Daily gap-up size (open vs prev close). Matt's video says "
        "'if there's a gap up' — no specific %. 0.1% catches 'any positive "
        "gap'. Raise to 0.5%+ for selectivity.",
    )
    max_gap_pct = st.slider(
        "Max gap %", 0.02, 0.50, 0.15, 0.01,
        help="Cap above which the gap is exhaustion territory for large-caps.",
    )
    min_rvol = st.slider(
        "Min RVOL (vs 50d avg)", 0.5, 5.0, 1.0, 0.1,
        help="Liquid large-caps (NVDA / AAPL / QQQ) average ~1.0× by "
        "definition; 1.3× catches only ~5% of days for those names. "
        "1.0× = 'not below average' which is what Matt's 'elevated' "
        "qualitatively means for large-caps. Raise to 1.3-1.5× for "
        "selectivity on catalyst days.",
    )
    min_price = st.number_input("Min open price ($)", value=20.0, step=1.0)

    st.header("Catalyst gates (Matt — 'context first')")
    require_catalyst = st.checkbox(
        "Require earnings ∪ news catalyst window", value=False,
        help="영상의 'when you add a catalyst' 게이트. EODHD earnings + news 어댑터 fetch.",
    )
    earnings_window_days = st.slider(
        "Earnings ± days", 0, 5, 2,
        help="Trade only within D±N of an earnings report.",
    )
    news_lookback_days = st.slider(
        "News lookback (days)", 0, 5, 1,
        help="Session date counts if any news in the last N days.",
    )
    news_min_sentiment = st.slider(
        "Min news sentiment (-1 to 1)", -1.0, 1.0, 0.0, 0.1,
        help="Only positive-toned news counts. Set to -1 to keep everything.",
    )
    require_igniting = st.checkbox(
        "Require prior-day igniting candle", value=False,
        help="Matt: 'a nice igniting candle on Friday on some elevated volume'.",
    )
    igniting_pct = st.slider(
        "Igniting candle min % body", 0.005, 0.10, 0.02, 0.005, format="%.3f",
    )
    igniting_rvol = st.slider("Igniting candle min RVOL", 1.0, 5.0, 1.5, 0.1)
    require_resistance_break = st.checkbox(
        "Require daily resistance break (open ≥ N-day high)", value=True,
        help="**Sweep-tuned ON** (2026-05): single biggest win-rate driver. "
        "Matt: 'it was opening above a lot of resistance'. 15-ticker sweep "
        "showed 25% → 78% win rate when this is on alongside the "
        "GTR-volume + 0.5% break-distance gates.",
    )
    resistance_lookback = st.slider("Resistance lookback (days)", 5, 60, 20)
    min_resistance_break_pct = st.slider(
        "Min break distance above N-day high (%)", 0.0, 0.05, 0.005, 0.001,
        format="%.3f",
        help="Open must be >= N-day high × (1 + this). 0.5% sweep-optimal: "
        "PF 7.68 vs 4.18 at 0% (true breakouts vs. probes).",
    )

    st.header("Pole / Pullback")
    opening_drive_max_bars = st.slider(
        "Opening drive max bars", 3, 30, 12,
        help="Pole = first push in the first N bars of the regular session.",
    )
    pole_min_pct = st.slider(
        "Pole min %", 0.001, 0.05, 0.015, 0.001, format="%.3f",
        help="**Sweep-tuned 1.5%** (2026-05). With resistance-break gate on, "
        "this is the win-rate-optimal threshold. Lower values let in weak "
        "drives that get stopped out.",
    )
    pullback_max_bars = st.slider(
        "Pullback max bars", 1, 8, 5,
        help="Matt: '1-3 candles' for the first flag; secondary flags can "
        "run longer. 5 covers both without bloating search.",
    )
    ema_period = st.number_input(
        "EMA period", min_value=5, max_value=50, value=10,
        help="Matt literally uses 10 EMA (not 9).",
    )
    ema_tol_pct = st.slider(
        "10 EMA tolerance %", 0.0, 0.03, 0.015, 0.001, format="%.3f",
        help="Pullback low wick depth below 10 EMA. Matt's video shows "
        "visible wicks below — 1.5% matches what he tolerates on screen.",
    )
    require_close_above_ema = st.checkbox(
        "Pullback close > 10 EMA (strict)", value=False,
        help="Off = wicks OR closes below EMA OK as long as wick depth "
        "is within tolerance. On = every pullback bar must close above "
        "EMA (stricter; Matt's 'controlled selling' is implied but he "
        "explicitly shows wicks below in the video).",
    )

    st.header("Entry / Stop")
    atr_period = st.number_input("ATR period", min_value=5, max_value=30, value=14)
    atr_mult = st.slider(
        "ATR stop multiplier", 0.0, 3.0, 1.0, 0.1,
        help="Stop = max(gtr_low, entry − ATR × mult). 0 disables ATR widening.",
    )
    allow_same_bar_fill = st.checkbox(
        "Allow same-bar fill on GTR", value=True,
        help="ON (default) = fill at the GTR candle's own high — Matt "
        "scalps fast and the GTR candle itself is the actionable bar. "
        "OFF = wait for the next bar to print > gtr_high + tick, which "
        "loses ~25% of fills to fade bars (and the prior implementation "
        "of this option was effectively dead code).",
    )
    require_gtr_volume_expansion = st.checkbox(
        "Require GTR vol >= prior red vol × mult", value=True,
        help="**Sweep-tuned ON** (2026-05). Volume confirmation on the "
        "GTR candle — 'fresh buyers stepping in'. Single-gate sweep: "
        "lifts win rate 50% → 64%, stacked with resistance_break it goes "
        "to 78% (PF 7.68).",
    )
    gtr_volume_expansion_mult = st.slider(
        "GTR vol expansion mult", 0.5, 3.0, 1.0, 0.1, format="%.1f",
        disabled=not require_gtr_volume_expansion,
        help="GTR bar volume / prior red bar volume must exceed this. "
        "1.0× = strictly more volume than the red bar before. 1.5× = "
        "noticeably more (trade count drops sharply).",
    )
    skip_first_n_minutes = st.slider(
        "Skip first N bars of session", 0, 15, 0, 1,
        help="Reject entries firing before bar N. Opening 1-5 minutes "
        "are noisiest. 0 = no skip (sweep showed no effect for current "
        "stack because pole_end_idx is already past minute 5+).",
    )
    require_pullback_above_vwap = st.checkbox(
        "Require pullback above session VWAP", value=False,
        help="Pullback low must stay above cumulative session VWAP. "
        "Sweep: improves win rate to 83% but cuts trade count to 6 — "
        "decent quality filter but small sample. Opt-in.",
    )
    max_consecutive_red_bars = st.slider(
        "Max consecutive red bars in pullback (0 = off)", 0, 5, 0, 1,
        help="Cap consecutive red bars inside the pullback. 4+ reds in "
        "a row is a real reversal even if 10 EMA holds. Sweep: minor "
        "effect on top of resistance + GTR-vol stack.",
    )
    max_nth_pullback = st.slider(
        "Max pullbacks per session", 1, 3, 2,
        help="2 = 'secondary bull flag continuation' (영상 정통).",
    )
    cutoff_hour = st.slider(
        "Latest entry (hour, local)", 10, 15, 14,
    )
    cutoff_min = st.slider("Latest entry (min)", 0, 59, 30, 5)

    st.header("Scalper exit")
    target_r = st.slider(
        "Target (R multiple)", 0.5, 5.0, 2.5, 0.1,
        help="**Sweep-tuned 2.5R** (2026-05). With resistance-break + "
        "GTR-vol gates, 2.5R is win-rate-optimal — let winners run far "
        "enough to compensate for the few losers. 1.5R is too greedy on "
        "TPs that don't trigger; 4.0R misses the few real big wins.",
    )
    enable_time_stop = st.checkbox("Enable time stop", value=True)
    time_stop_bars = st.slider("Time stop bars", 3, 60, 10)
    time_stop_min_r = st.slider("Time stop min R", -0.5, 1.5, 0.5, 0.1)
    enable_be = st.checkbox("Breakeven after +1R", value=True)

    st.markdown("---")
    # Explicit run trigger — page is dial-then-run, not auto-run, so
    # the data fetch + sim doesn't fire on every slider change.
    run_clicked = st.button("▶ Run backtest", type="primary", use_container_width=True)
    clear_clicked = st.button("✕ Clear results", use_container_width=True)

market = NY

# ---- Result cache (per-click) --------------------------------------
# Streamlit reruns on every widget change. Persist the last-run result
# in session_state so the chart + tables stay on screen between
# parameter tweaks; user reruns by hitting the button.
if "md_bf_result" not in st.session_state:
    st.session_state["md_bf_result"] = None

if clear_clicked:
    st.session_state["md_bf_result"] = None

if not run_clicked and st.session_state["md_bf_result"] is None:
    st.info(
        "Tune parameters in the sidebar, then click **▶ Run backtest** to "
        "fetch data and execute the simulation. Subsequent slider changes "
        "do **not** re-trigger the run (avoids burning fetch quota)."
    )
    st.stop()

if run_clicked:
    # ---- Data fetch ------------------------------------------------
    # Composed adapter handles the EODHD → Massive → YFinance chain.
    md_raw = build_default_market_data(bypass_today=(end_date >= today))
    md = RegularSessionFilterAdapter(md_raw, market=market)

    with st.spinner(
        f"Fetching {intraday_interval} + 1d for {ticker}"
        + (" + SPY" if require_regime else "")
        + (" + raw 1m PM bars" if require_pm_high else "")
        + " ..."
    ):
        try:
            df_intraday = md.fetch_ohlcv(
                ticker, start_date, end_date, interval=intraday_interval
            )
            df_daily = md.fetch_ohlcv(
                ticker, start_date - timedelta(days=120), end_date, interval="1d",
            )
            # PM high needs RAW bars (before RTH session filter).
            df_intraday_raw = (
                md_raw.fetch_ohlcv(
                    ticker, start_date, end_date, interval="1m"
                )
                if require_pm_high
                else pd.DataFrame()
            )
            df_spy_daily = (
                md.fetch_ohlcv(
                    "SPY",
                    start_date - timedelta(days=120),
                    end_date,
                    interval="1d",
                )
                if require_regime
                else pd.DataFrame()
            )
        except Exception as exc:
            st.error(f"Data fetch failed: {exc}")
            st.stop()

    if df_intraday is None or df_intraday.empty:
        st.warning("No intraday data returned for that range.")
        st.stop()

    # tz alignment
    if df_intraday.index.tz is not None and str(df_intraday.index.tz) != market.tz:
        df_intraday = df_intraday.copy()
        df_intraday.index = df_intraday.index.tz_convert(market.tz)

    # Derive PM high from raw (un-session-filtered) 1m bars.
    pm_by_date: dict = {}
    if require_pm_high and not df_intraday_raw.empty:
        _raw = df_intraday_raw
        if _raw.index.tz is None:
            _raw = _raw.copy()
            _raw.index = _raw.index.tz_localize(market.tz)
        elif str(_raw.index.tz) != market.tz:
            _raw = _raw.copy()
            _raw.index = _raw.index.tz_convert(market.tz)
        pm_by_date = compute_premarket_high_by_date(_raw)

    regime_ok: dict = {}
    if require_regime and not df_spy_daily.empty:
        regime_ok = compute_market_regime_ok(
            df_spy_daily, sma_period=regime_sma_period
        )

    # Catalyst dates — earnings ∪ news. Fetched only when the gate is
    # on, so users not opted-in don't pay the API round-trip.
    catalyst_dates: set = set()
    earnings_count = 0
    news_count = 0
    if require_catalyst:
        from data.adapters.eodhd_earnings import EODHDEarningsAdapter
        from data.adapters.eodhd_news import EODHDNewsAdapter
        try:
            earnings = EODHDEarningsAdapter().fetch_earnings(
                ticker, start_date, end_date
            )
            news = EODHDNewsAdapter().fetch_news(
                ticker, start_date, end_date, limit_per_day=3
            )
            earnings_count = len(earnings)
            news_count = len(news)
            min_sent = (
                None if news_min_sentiment <= -0.999 else float(news_min_sentiment)
            )
            catalyst_dates = (
                compute_earnings_window_dates(
                    earnings, window_days=int(earnings_window_days)
                )
                | compute_news_session_dates(
                    news,
                    lookback_days=int(news_lookback_days),
                    min_sentiment=min_sent,
                )
            )
        except Exception as exc:
            st.warning(
                f"Catalyst fetch failed (EODHD): {type(exc).__name__}. "
                "Disable the catalyst gate or set EODHD_API_KEY."
            )

    detector = MattDiamondBullFlagDetector(
        require_premarket_high=bool(require_pm_high),
        premarket_high_by_date=pm_by_date,
        market_regime_ok_by_date=regime_ok,
        catalyst_dates=catalyst_dates,
        require_catalyst=bool(require_catalyst),
        require_prior_igniting_candle=bool(require_igniting),
        igniting_candle_min_pct=float(igniting_pct),
        igniting_candle_min_rvol=float(igniting_rvol),
        require_daily_resistance_break=bool(require_resistance_break),
        resistance_lookback_days=int(resistance_lookback),
        min_resistance_break_pct=float(min_resistance_break_pct),
        require_gtr_volume_expansion=bool(require_gtr_volume_expansion),
        gtr_volume_expansion_mult=float(gtr_volume_expansion_mult),
        skip_first_n_minutes=int(skip_first_n_minutes),
        require_pullback_above_vwap=bool(require_pullback_above_vwap),
        max_consecutive_red_bars=int(max_consecutive_red_bars),
        min_gap_pct=float(min_gap_pct),
        max_gap_pct=float(max_gap_pct),
        min_rvol=float(min_rvol),
        min_price=float(min_price),
        opening_drive_max_bars=int(opening_drive_max_bars),
        pole_min_pct=float(pole_min_pct),
        pullback_max_bars=int(pullback_max_bars),
        ema_period=int(ema_period),
        ema_tolerance_pct=float(ema_tol_pct),
        require_pullback_close_above_ema=bool(require_close_above_ema),
        atr_period=int(atr_period),
        atr_stop_multiplier=float(atr_mult),
        allow_same_bar_fill=bool(allow_same_bar_fill),
        latest_entry_local=time(int(cutoff_hour), int(cutoff_min)),
        max_nth_pullback=int(max_nth_pullback),
    )

    strategy = MattDiamondBullFlagStrategy(
        detector,
        max_position_pct_of_equity=float(max_position_pct),
        target_at_r_multiple=float(target_r),
        enable_time_stop=bool(enable_time_stop),
        time_stop_bars=int(time_stop_bars),
        time_stop_min_r=float(time_stop_min_r),
        enable_breakeven_after_r=1.0 if enable_be else None,
        fee_schedule=TossFeeSchedule(),
    )

    config = StrategyConfig(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        pattern_name="matt_diamond_bull_flag",
        initial_capital=float(initial_capital),
        risk_per_trade=float(risk_per_trade),
    )

    with st.spinner("Running backtest..."):
        result = strategy.run(df_intraday, df_daily, config)
        all_signals = detector.detect(df_intraday, df_daily)

    st.session_state["md_bf_result"] = {
        "result": result,
        "all_signals": all_signals,
        "df_intraday": df_intraday,
        "pm_by_date": pm_by_date,
        "regime_ok": regime_ok,
        "catalyst_dates": catalyst_dates,
        "earnings_count": earnings_count,
        "news_count": news_count,
        "ema_period": int(ema_period),
        "ticker": ticker,
        "interval": intraday_interval,
    }

# ---- Render last-run results --------------------------------------
state = st.session_state["md_bf_result"]
if state is None:
    st.stop()

result = state["result"]
all_signals = state["all_signals"]
df_intraday = state["df_intraday"]
pm_by_date = state["pm_by_date"]
regime_ok = state["regime_ok"]
ema_period_used = state["ema_period"]
ticker_used = state["ticker"]
interval_used = state["interval"]
perf = result.performance

st.markdown(
    f"**Last run**: {ticker_used} {interval_used} "
    f"({result.config.start_date} → {result.config.end_date})"
)

cols = st.columns(6)
cols[0].metric("Total trades", perf.total_trades)
cols[1].metric("Win rate", f"{perf.win_rate*100:.1f}%")
cols[2].metric("Return", f"{perf.total_return_pct*100:+.2f}%")
cols[3].metric(
    "Final equity",
    f"${perf.final_capital:,.0f}",
    delta=f"${perf.final_capital - perf.initial_capital:+,.0f}",
)
cols[4].metric("Avg win", f"{perf.avg_win_pct*100:+.2f}%")
cols[5].metric("Max DD", f"{perf.max_drawdown_pct*100:.2f}%")

catalyst_dates_used = state.get("catalyst_dates") or set()
st.markdown(
    f"**Detector raw signals**: {len(all_signals)} · "
    f"**Sessions with PM data**: {len(pm_by_date)} · "
    f"**Regime-OK sessions**: "
    f"{sum(regime_ok.values()) if regime_ok else 'N/A'} · "
    f"**Catalyst sessions**: {len(catalyst_dates_used) or 'N/A'} "
    f"(earnings={state.get('earnings_count', 0)}, "
    f"news={state.get('news_count', 0)})"
)
if not all_signals:
    st.info(
        "No raw signals. Try loosening one of: min_gap_pct, min_rvol, "
        "pole_min_pct, ema tolerance, require_pm_high, require_regime."
    )

if perf.trades:
    trades_df = pd.DataFrame([t.model_dump() for t in perf.trades])
    st.markdown("### Trades")
    st.dataframe(trades_df, use_container_width=True)

if result.equity_curve:
    eq_df = pd.DataFrame([p.model_dump() for p in result.equity_curve])
    fig_eq = go.Figure()
    fig_eq.add_trace(
        go.Scatter(
            x=eq_df["date"], y=eq_df["equity"], mode="lines+markers",
            name="Equity",
        )
    )
    fig_eq.update_layout(
        title="Equity curve",
        xaxis_title="Date",
        yaxis_title="Equity ($)",
        height=300,
    )
    st.plotly_chart(fig_eq, use_container_width=True)

if all_signals:
    st.markdown("### Intraday chart with signals")
    sig_dates = sorted({s.session_date for s in all_signals})
    pick = st.selectbox(
        "Session to view",
        sig_dates,
        format_func=lambda d: d.isoformat(),
    )
    day_df = df_intraday[df_intraday.index.date == pick]
    day_sigs = [s for s in all_signals if s.session_date == pick]
    if not day_df.empty:
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03,
            row_heights=[0.75, 0.25],
        )
        fig.add_trace(
            go.Candlestick(
                x=day_df.index,
                open=day_df["Open"], high=day_df["High"],
                low=day_df["Low"], close=day_df["Close"],
                name="OHLC",
            ),
            row=1, col=1,
        )
        ema_series = (
            day_df["Close"].ewm(span=ema_period_used, adjust=False).mean()
        )
        fig.add_trace(
            go.Scatter(
                x=day_df.index, y=ema_series, mode="lines",
                name=f"{ema_period_used} EMA", line=dict(width=1.2),
            ),
            row=1, col=1,
        )
        pm = pm_by_date.get(pick)
        if pm:
            fig.add_hline(
                y=pm, line=dict(dash="dash"),
                annotation_text=f"PM high {pm:.2f}",
                row=1, col=1,
            )
        for s in day_sigs:
            fig.add_trace(
                go.Scatter(
                    x=[s.entry_ts], y=[s.entry_price],
                    mode="markers+text",
                    marker=dict(symbol="triangle-up", size=14, color="green"),
                    text=[f"E{s.nth_pullback}"],
                    textposition="top center",
                    name=f"Entry #{s.nth_pullback}",
                ),
                row=1, col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=[s.entry_ts], y=[s.stop_loss],
                    mode="markers",
                    marker=dict(symbol="x", size=10, color="red"),
                    name=f"Stop #{s.nth_pullback}",
                ),
                row=1, col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=[s.gtr_ts], y=[s.gtr_high],
                    mode="markers",
                    marker=dict(symbol="diamond", size=10, color="orange"),
                    name=f"GTR #{s.nth_pullback}",
                ),
                row=1, col=1,
            )
        fig.add_trace(
            go.Bar(
                x=day_df.index, y=day_df["Volume"], name="Volume",
                marker_color="#666",
            ),
            row=2, col=1,
        )
        fig.update_layout(
            height=600,
            xaxis_rangeslider_visible=False,
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Signal details")
        sigs_df = pd.DataFrame(
            [
                dict(
                    nth=s.nth_pullback,
                    gtr_ts=s.gtr_ts,
                    gtr_high=s.gtr_high,
                    gtr_low=s.gtr_low,
                    entry_ts=s.entry_ts,
                    entry=s.entry_price,
                    stop=s.stop_loss,
                    atr=s.atr_at_entry,
                    pm_high=s.pm_high,
                    ema10=s.ema10_at_pullback,
                    gap_pct=s.gap_pct,
                    rvol=s.rvol,
                )
                for s in day_sigs
            ]
        )
        st.dataframe(sigs_df, use_container_width=True)
