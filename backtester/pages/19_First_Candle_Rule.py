"""First Candle Rule single-ticker page.

Implements the "This Scalping Strategy Works Everyday" video by
Smart Trading Blueprint. Two-path entry logic:

  - **Path A (immediate):** the candle that closes above the OR
    is itself the displacement bar of a 3-bar bullish FVG. Enter
    on its close.
  - **Path B (retest):** range break with no FVG → wait for price
    to dip back into the OR → enter on the next post-retest FVG.

**Exit policy (per user pref):** TP touch OR Nth-strike stop.
**Stop = first FVG bar's low − 1 tick** (= ``fvg_pre_low``,
shared across Path A & B). The stop fires only on the **Nth
distinct excursion** of price below it (default N=3), letting
the first two wicks slide as liquidity grabs. Consecutive bars
below the stop count as one strike; the streak resets when a
bar's low climbs back at or above the stop. If TP isn't touched
and the stop hasn't accumulated enough strikes by session end,
the trade force-closes at the last bar's close.
"""

from datetime import date, time, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data.adapters.composed_market_data import build_default_market_data
from data.adapters.regular_session_filter import RegularSessionFilterAdapter
from data.adapters.yfinance_adapter import YFinanceAdapter
from data.domain.market_calendar import KR, NY, market_for_ticker
from pattern.adapters.first_candle_rule import FirstCandleRuleDetector
from strategy.adapters.first_candle_rule_strategy import (
    FirstCandleRuleStrategy,
)
from strategy.domain.models import StrategyConfig

INTRADAY_HISTORY_FLOOR = date(2022, 1, 1)

st.set_page_config(page_title="First Candle Rule", layout="wide")
st.title("First Candle Rule (Casper SMC)")
st.caption(
    "첫 5분봉 OR → 1분봉으로 전환 → 박스 밖으로 break하는 bullish FVG → "
    "Path A: range-break 봉이 곧 FVG displacement면 즉시 entry. "
    "Path B: range-break 후 FVG 없으면 retest 후 새 FVG에 entry. "
    "Exit = TP touch (고정 2:1) / Stop = FVG 첫 봉(bar i-2)의 low − tick, "
    "N번째 침투에서 발화 (기본 3회) / 둘 다 안 닿으면 session close."
)

with st.sidebar:
    st.header("Market")
    if "fcr_ticker" not in st.session_state:
        st.session_state.fcr_ticker = "QQQ"
    ticker = st.text_input("Ticker", key="fcr_ticker")
    detected_market = market_for_ticker(ticker)
    market_choice = st.selectbox(
        "Market calendar",
        options=["NY", "KR"],
        index=0 if detected_market.name == "NY" else 1,
        format_func=lambda x: {
            "NY": "🇺🇸 US (NYSE / Nasdaq)  — 09:30–16:00 ET",
            "KR": "🇰🇷 Korea (KOSPI/KOSDAQ) — 09:00–15:00 KST",
        }[x],
    )
    market = NY if market_choice == "NY" else KR

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
    )
    max_position_pct = st.number_input(
        "Max position (% of equity)",
        value=30.0,
        min_value=1.0,
        max_value=100.0,
        step=5.0,
        help="단일 트레이드 notional 상한.",
    )
    max_below_stop_strikes = st.number_input(
        "Stop strikes (N-touch tolerant stop)",
        value=3,
        min_value=0,
        max_value=10,
        step=1,
        help="가격이 stop 라인 아래로 N번째 내려갔을 때 stop 발화. "
        "1=즉시 stop, 2=두 번째 wick부터 stop, 3=세 번째 (기본). "
        "연속 봉이 stop 아래에 있는 동안은 같은 strike (1회)로 카운트. "
        "0이면 stop 비활성 (TP 또는 session close만).",
    )
    enable_be_time = st.checkbox(
        "Move stop to break-even after N minutes (Crabel)",
        value=True,
        help="진입 후 N분 경과 시 stop을 entry로 끌어올림. "
        "session_close에서 -1R로 끝나는 trade를 break-even으로 변환. "
        "Crabel: '이상적 트레이드는 즉시 수익이 보임; 늦게 갈수록 vulnerable'.",
    )
    breakeven_after_min = st.number_input(
        "BE after minutes",
        value=60, min_value=0, max_value=390, step=15,
        disabled=not enable_be_time,
    )
    enable_be_r = st.checkbox(
        "Move stop to break-even after R reached",
        value=False,
        help="미실현 +N×R 도달 시 stop을 entry로. 1R touch 후 -R로 회귀하는 "
        "trade를 break-even으로 변환. winner 일부가 BE에서 청산되는 trade-off.",
    )
    breakeven_after_r = st.number_input(
        "BE after R-multiple",
        value=1.0, min_value=0.0, max_value=5.0, step=0.1, format="%.1f",
        disabled=not enable_be_r,
    )

    st.header("Pattern")
    box_minutes = st.number_input(
        "OR length (minutes)",
        value=5,
        min_value=1,
        max_value=30,
        step=1,
        help="영상은 첫 5분봉 = 5. 1m frame이면 5봉, 5m frame이면 1봉.",
    )
    require_bullish_displacement = st.checkbox(
        "Require bullish displacement bar (FVG-creator)",
        value=True,
        help="FVG를 만든 bar(i)가 양봉이어야 함. 거짓 spike-and-fade FVG 거름.",
    )
    min_fvg_gap_pct = st.number_input(
        "Min FVG gap (% of price)",
        value=0.03,
        min_value=0.0,
        max_value=2.0,
        step=0.01,
        format="%.2f",
        help="FVG height ≥ 이 값 × price 일 때만 valid. 0.03% = 약 $0.18 on $600 ETF. "
        "0.5센트짜리 micro-FVG는 거르되 0.05% 같은 borderline displacement는 통과. "
        "0이면 OFF.",
    )
    max_total_bars = st.number_input(
        "Max bars from range break to entry FVG",
        value=45,
        min_value=3,
        max_value=200,
        step=5,
        help="Range break 봉 이후 N봉 안에 entry FVG가 안 나오면 setup invalidate. "
        "Path A는 0봉, Path B는 retest+FVG 형성 시간 필요. 기본 45봉(≈45분 on 1m).",
    )
    require_retest_for_path_b = st.checkbox(
        "Require retest into OR for Path B",
        value=True,
        help="Range break 후 즉시 FVG가 안 생기면, OR 안으로 retest dip 후에야 "
        "Path B FVG entry 허용. OFF면 retest 없이도 post-break FVG 진입.",
    )
    target_r_multiple = st.number_input(
        "Take profit (R-multiple)",
        value=2.0,
        min_value=0.5,
        max_value=10.0,
        step=0.5,
        format="%.1f",
        help="영상 = 고정 2:1.",
    )
    stop_tick_buffer = st.number_input(
        "Stop tick buffer ($)",
        value=0.01,
        min_value=0.0,
        max_value=1.0,
        step=0.01,
        format="%.2f",
        help="Retest 봉 low에서 N달러 더 아래에 stop. 영상의 'one tick beyond'.",
    )
    latest_entry_hour = st.number_input(
        "Latest entry hour (local, exclusive)",
        value=11,
        min_value=10,
        max_value=16,
        step=1,
    )
    require_daily_bullish = st.checkbox(
        "Require previous daily candle bullish (regime gate)",
        value=False,
        help="옵션 — 영상은 daily bias 명시 안 함. ON이면 전날 양봉 마감일에만 진입.",
    )

    st.header("Advanced filters (sweep-tuned)")
    st.caption(
        "144-combo grid sweep on NVDA Jan-May 2026 + 6-ticker validation. "
        "기본값 = 검증된 robust combo (NVDA +0.94pp, TSLA +0.43pp, 평균 +0.17pp)."
    )
    enable_gap_filter = st.checkbox(
        "Skip gap-up days",
        value=True,
        help="Today's open vs prev_close 갭이 X% 초과면 그 세션 스킵. "
        "Gap-and-fade 트랩 회피.",
    )
    max_gap_up_pct = st.number_input(
        "Max gap-up (%)",
        value=1.5,
        min_value=0.0,
        max_value=10.0,
        step=0.5,
        format="%.2f",
        disabled=not enable_gap_filter,
    )
    enable_drvol_filter = st.checkbox(
        "Skip low-volume days",
        value=True,
        help="Today's volume / 20-day avg volume < X면 스킵. "
        "Cross-ticker 검증에서 가장 robust한 단일 필터.",
    )
    min_daily_rvol = st.number_input(
        "Min daily RVOL",
        value=0.85,
        min_value=0.0,
        max_value=3.0,
        step=0.05,
        format="%.2f",
        disabled=not enable_drvol_filter,
    )
    enable_sma_filter = st.checkbox(
        "Require above daily SMA(N)",
        value=False,
        help="Today's close > N-day SMA. Bear regime 필터. "
        "Sweep에서 효과 미미했지만 longer 백테스트 윈도우에서 의미 있을 수 있음.",
    )
    sma_period = st.selectbox(
        "SMA period", options=[50, 200], index=0,
        disabled=not enable_sma_filter,
    )
    enable_brvol_filter = st.checkbox(
        "Require FVG-bar RVOL",
        value=False,
        help="FVG 봉 volume / early-session avg ≥ X. NVDA에 overfit돼서 "
        "기본 OFF. 변동 큰 종목에 ON 시도 가능.",
    )
    min_entry_bar_rvol = st.number_input(
        "Min FVG-bar RVOL",
        value=1.0,
        min_value=0.0,
        max_value=5.0,
        step=0.1,
        format="%.2f",
        disabled=not enable_brvol_filter,
    )

    run_btn = st.button(
        "Run First Candle Rule Backtest",
        type="primary",
        use_container_width=True,
    )

if not run_btn:
    st.info(
        "좌측에서 ticker / 기간을 설정하고 **Run First Candle Rule Backtest**를 눌러. "
        "영상 권장 = NYSE 오픈 + 액티브 종목 (QQQ, SPY, NVDA, TSLA 등)."
    )
    st.stop()

# Composition root
_base_md = build_default_market_data()
md = RegularSessionFilterAdapter(_base_md, market=market)
yf = YFinanceAdapter()

# EODHD doesn't carry 1m for KR — fall back to 5m for KR tickers.
strat_interval = "5m" if market.name == "KR" else "1m"

with st.spinner(f"Fetching {strat_interval} / daily for {ticker}..."):
    try:
        df_intraday = md.fetch_ohlcv(
            ticker.upper(), start_date, end_date, interval=strat_interval
        )
        # Pad ~400 calendar days so the 200 SMA on the daily chart
        # converges from the very first session of the display
        # window. 200 trading days ≈ 280 cal days; 400 keeps a
        # comfortable buffer including weekends / holidays.
        df_daily = yf.fetch_ohlcv(
            ticker.upper(), start_date - timedelta(days=400), end_date
        )
    except Exception as exc:
        st.error(f"Data fetch failed: {exc}")
        st.stop()

if df_intraday is None or df_intraday.empty:
    st.warning(f"No {strat_interval} data returned for that range.")
    st.stop()

if df_intraday.index.tz is not None and str(df_intraday.index.tz) != market.tz:
    df_intraday = df_intraday.copy()
    df_intraday.index = df_intraday.index.tz_convert(market.tz)

detector = FirstCandleRuleDetector(
    box_minutes=int(box_minutes),
    session_open_local=market.rth_open,
    latest_entry_local=time(int(latest_entry_hour), 0),
    require_daily_bullish=bool(require_daily_bullish),
    require_bullish_displacement=bool(require_bullish_displacement),
    min_fvg_gap_pct=float(min_fvg_gap_pct) / 100.0,
    max_total_bars=int(max_total_bars),
    require_retest_for_path_b=bool(require_retest_for_path_b),
    target_r_multiple=float(target_r_multiple),
    stop_tick_buffer=float(stop_tick_buffer),
    max_gap_up_pct=(
        float(max_gap_up_pct) / 100.0 if enable_gap_filter else None
    ),
    min_daily_rvol=(
        float(min_daily_rvol) if enable_drvol_filter else None
    ),
    require_above_daily_sma=(
        int(sma_period) if enable_sma_filter else None
    ),
    min_entry_bar_rvol=(
        float(min_entry_bar_rvol) if enable_brvol_filter else None
    ),
)
strategy = FirstCandleRuleStrategy(
    detector,
    max_position_pct_of_equity=float(max_position_pct) / 100.0,
    max_below_stop_strikes=int(max_below_stop_strikes),
    breakeven_after_minutes=(
        int(breakeven_after_min) if enable_be_time else 0
    ),
    breakeven_after_r_multiple=(
        float(breakeven_after_r) if enable_be_r else 0.0
    ),
)

config = StrategyConfig(
    ticker=ticker.upper(),
    start_date=start_date,
    end_date=end_date,
    pattern_name="first_candle_rule",
    initial_capital=float(initial_capital),
    risk_per_trade=risk_pct / 100.0,
    max_holding_days=1,
)

with st.spinner("Running backtest..."):
    try:
        result = strategy.run(df_intraday, df_daily, config)
    except Exception as exc:
        st.error(f"Backtest failed: {exc}")
        st.stop()

perf = result.performance
chart_signals = detector.detect(df_intraday, df_daily)

# ---- Headline metrics ---------------------------------------------
st.subheader(f"{ticker.upper()} — First Candle Rule")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Trades", perf.total_trades)
m2.metric(
    "Win Rate", f"{perf.win_rate:.0%}" if perf.total_trades else "—"
)
m3.metric("Total Return", f"{perf.total_return_pct:+.2%}")
m4.metric("Final Capital", f"${perf.final_capital:,.0f}")
m5, m6, m7, m8 = st.columns(4)
m5.metric("Avg Win", f"{perf.avg_win_pct:+.2%}" if perf.trades else "—")
m6.metric("Avg Loss", f"{perf.avg_loss_pct:+.2%}" if perf.trades else "—")
m7.metric("Max DD", f"{perf.max_drawdown_pct:.2%}")
m8.metric("Signals", len(chart_signals))

# ---- Daily chart --------------------------------------------------
st.subheader("Daily — context")
_cutoff = pd.Timestamp(start_date)
if df_daily.index.tz is not None and _cutoff.tz is None:
    _cutoff = _cutoff.tz_localize(df_daily.index.tz)
# Compute MAs on the FULL series (warmup + display window) so SMA200
# converges from the first display bar, then slice for rendering.
_ema10 = df_daily["Close"].ewm(span=10, adjust=False).mean()
_ema20 = df_daily["Close"].ewm(span=20, adjust=False).mean()
_sma50 = df_daily["Close"].rolling(50).mean()
_sma200 = df_daily["Close"].rolling(200).mean()
d_disp = df_daily[df_daily.index >= _cutoff]
daily_fig = go.Figure(go.Candlestick(
    x=d_disp.index, open=d_disp["Open"], high=d_disp["High"],
    low=d_disp["Low"], close=d_disp["Close"],
    name="Daily", showlegend=False,
))
# Overlay the four canonical MAs — short EMAs to show recent
# momentum, long SMAs for regime / structural levels.
for ma_series, name, color, width in (
    (_ema10[df_daily.index >= _cutoff], "10 EMA", "#FFB300", 1.4),
    (_ema20[df_daily.index >= _cutoff], "20 EMA", "#FB8C00", 1.4),
    (_sma50[df_daily.index >= _cutoff], "50 SMA", "#1976D2", 1.6),
    (_sma200[df_daily.index >= _cutoff], "200 SMA", "#6A1B9A", 1.8),
):
    daily_fig.add_trace(go.Scatter(
        x=ma_series.index, y=ma_series.values,
        mode="lines", line=dict(color=color, width=width),
        name=name,
        hovertemplate=name + " $%{y:.2f}<extra></extra>",
    ))
if perf.trades:
    daily_dates = {ts.date(): ts for ts in d_disp.index}
    win_x, win_y, win_text = [], [], []
    lose_x, lose_y, lose_text = [], [], []
    for t in perf.trades:
        ts = daily_dates.get(t.entry_date)
        if ts is None:
            continue
        marker_y = float(d_disp.loc[ts, "Low"]) * 0.995
        text = (
            f"{t.entry_date} {t.exit_reason}<br>"
            f"entry ${t.entry_price:.2f}  stop ${t.stop_loss:.2f}<br>"
            f"PnL ${t.pnl:+,.0f} ({t.pnl_pct:+.2%})"
        )
        if t.pnl > 0:
            win_x.append(ts)
            win_y.append(marker_y)
            win_text.append(text)
        else:
            lose_x.append(ts)
            lose_y.append(marker_y)
            lose_text.append(text)
    if win_x:
        daily_fig.add_trace(go.Scatter(
            x=win_x, y=win_y, mode="markers",
            marker=dict(symbol="triangle-up", color="#2E7D32", size=14),
            name="Win", text=win_text,
            hovertemplate="%{text}<extra></extra>",
        ))
    if lose_x:
        daily_fig.add_trace(go.Scatter(
            x=lose_x, y=lose_y, mode="markers",
            marker=dict(symbol="triangle-up", color="#C62828", size=14),
            name="Loss", text=lose_text,
            hovertemplate="%{text}<extra></extra>",
        ))
daily_fig.update_layout(
    height=320, xaxis_rangeslider_visible=False,
    margin=dict(l=10, r=10, t=10, b=10),
)
st.plotly_chart(daily_fig, use_container_width=True)

# ---- Intraday chart with full pattern overlays --------------------
st.subheader(f"{strat_interval} — pattern overlays")

last_intraday_per_date = {}
for ts in df_intraday.index:
    last_intraday_per_date[ts.date()] = ts

intra_fig = go.Figure(go.Candlestick(
    x=df_intraday.index, open=df_intraday["Open"], high=df_intraday["High"],
    low=df_intraday["Low"], close=df_intraday["Close"],
    name=strat_interval, showlegend=False,
))
first_legend = {
    "rb": True, "retest": True, "entry": True,
    "tp": True, "stop": True, "exit": True,
}
for sig in chart_signals:
    end_ts = last_intraday_per_date.get(sig.session_date, sig.box_close_ts)
    # OR rectangle (orange).
    intra_fig.add_shape(
        type="rect",
        x0=sig.box_open_ts, x1=end_ts,
        y0=sig.box_low, y1=sig.box_high,
        line=dict(color="#FB8C00", width=1.4),
        fillcolor="rgba(251,140,0,0.15)",
        layer="below",
    )
    # FVG zone — drawn as a prominent purple box spanning the
    # entire 3-bar window (bar i-2 → bar i) extended through the
    # entry bar so the structural level is visually clear, even
    # for micro-gaps where the y-axis band is thin.
    fvg_height = sig.fvg_high - sig.fvg_low
    intra_fig.add_shape(
        type="rect",
        x0=sig.fvg_pre_ts, x1=sig.entry_ts,
        y0=sig.fvg_low, y1=sig.fvg_high,
        line=dict(color="#7B1FA2", width=2),
        fillcolor="rgba(186,104,200,0.30)",
        layer="below",
    )
    # FVG label — shows the gap height in dollars so micro-gaps
    # are immediately obvious in the trade table view.
    intra_fig.add_annotation(
        x=sig.fvg_pre_ts, y=sig.fvg_high,
        text=(
            f"FVG [{sig.fvg_low:.2f}, {sig.fvg_high:.2f}] "
            f"(Δ ${fvg_height:.3f})"
        ),
        showarrow=False, xanchor="left", yanchor="bottom",
        font=dict(color="#4A148C", size=10),
        bgcolor="rgba(255,255,255,0.75)",
    )
    # Range-break candle marker (cyan ▲ at the candle's high) —
    # the candle that defines the stop level. Same in Path A and B.
    rb_high = float(df_intraday.loc[sig.range_break_ts, "High"])
    intra_fig.add_trace(go.Scatter(
        x=[sig.range_break_ts], y=[rb_high],
        mode="markers",
        marker=dict(symbol="triangle-up", color="#00ACC1", size=11),
        name="Range break candle",
        legendgroup="rb",
        showlegend=first_legend.get("rb", True),
        hovertemplate=(
            f"Range break (Path {sig.path}) @ ${rb_high:.2f}"
            f"<br>stop ref low ${sig.range_break_low:.2f}<extra></extra>"
        ),
    ))
    first_legend["rb"] = False
    # Path B retest marker — only present when retest happened.
    if sig.retest_ts is not None and sig.retest_low is not None:
        intra_fig.add_trace(go.Scatter(
            x=[sig.retest_ts], y=[sig.retest_low],
            mode="markers",
            marker=dict(symbol="triangle-down", color="#1976D2", size=11),
            name="Retest into OR (Path B)",
            legendgroup="retest",
            showlegend=first_legend["retest"],
            hovertemplate=f"Retest @ ${sig.retest_low:.2f}<extra></extra>",
        ))
        first_legend["retest"] = False
    # Entry marker (green ⬆ at FVG-confirming bar's close).
    intra_fig.add_trace(go.Scatter(
        x=[sig.entry_ts], y=[sig.entry_price],
        mode="markers",
        marker=dict(
            symbol="arrow-up", color="#2E7D32", size=14,
            line=dict(color="#1B5E20", width=1.5),
        ),
        name=f"Entry (Path {sig.path})",
        legendgroup="entry",
        showlegend=first_legend["entry"],
        hovertemplate=(
            f"Path {sig.path} entry @ ${sig.entry_price:.2f}<extra></extra>"
        ),
    ))
    first_legend["entry"] = False
    # Stop / TP horizontal levels.
    intra_fig.add_trace(go.Scatter(
        x=[sig.entry_ts, end_ts],
        y=[sig.take_profit, sig.take_profit],
        mode="lines",
        line=dict(color="#2E7D32", width=1.6, dash="dash"),
        name=f"Take profit ({target_r_multiple:.1f}R)",
        legendgroup="tp",
        showlegend=first_legend["tp"],
        hovertemplate=f"TP @ ${sig.take_profit:.2f}<extra></extra>",
    ))
    # Stop level — N-strike tolerant stop. The line fires on the
    # Nth excursion (default 3rd) of price below it. Drawn as a
    # dashed red line so it's clearly visible but with a label
    # showing the strike count.
    intra_fig.add_trace(go.Scatter(
        x=[sig.entry_ts, end_ts],
        y=[sig.stop_loss, sig.stop_loss],
        mode="lines",
        line=dict(color="#C62828", width=1.6, dash="dash"),
        name=f"Stop (after {max_below_stop_strikes}× below)",
        legendgroup="stop",
        showlegend=first_legend["stop"],
        hovertemplate=f"Stop @ ${sig.stop_loss:.2f}<extra></extra>",
    ))
    first_legend["tp"] = False
    first_legend["stop"] = False
    intra_fig.add_annotation(
        x=end_ts, y=sig.take_profit,
        text=f"TP {sig.take_profit:.2f}",
        showarrow=False, xanchor="right", yanchor="bottom",
        font=dict(color="#1B5E20", size=10),
        bgcolor="rgba(255,255,255,0.65)",
    )
    intra_fig.add_annotation(
        x=end_ts, y=sig.stop_loss,
        text=f"Stop {sig.stop_loss:.2f}",
        showarrow=False, xanchor="right", yanchor="top",
        font=dict(color="#7F0000", size=10),
        bgcolor="rgba(255,255,255,0.65)",
    )

# Trade exit markers
for t in perf.trades:
    if not t.exit_ts:
        continue
    exit_ts = pd.Timestamp(t.exit_ts)
    if exit_ts.tz is None and df_intraday.index.tz is not None:
        exit_ts = exit_ts.tz_localize(df_intraday.index.tz)
    color = "#2E7D32" if t.pnl > 0 else "#C62828"
    if t.exit_reason == "session_close":
        symbol, color = "diamond-open", color
    elif t.exit_reason == "stop_loss":
        symbol = "x"
    elif t.exit_reason == "breakeven_stop":
        symbol = "square-open"
    else:  # take_profit
        symbol = "circle"
    intra_fig.add_trace(go.Scatter(
        x=[exit_ts], y=[t.exit_price],
        mode="markers",
        marker=dict(symbol=symbol, color=color, size=12),
        name=f"Exit ({t.exit_reason})",
        legendgroup="exit",
        showlegend=first_legend["exit"],
        hovertemplate=(
            f"{t.exit_reason} @ ${t.exit_price:.2f} (PnL ${t.pnl:+,.2f})"
            "<extra></extra>"
        ),
    ))
    first_legend["exit"] = False

intra_fig.update_layout(
    height=560,
    xaxis_rangeslider_visible=False,
    xaxis_rangebreaks=[
        dict(
            bounds=[
                market.rth_close.hour + market.rth_close.minute / 60,
                market.rth_open.hour + market.rth_open.minute / 60,
            ],
            pattern="hour",
        ),
        dict(bounds=["sat", "mon"]),
    ],
    margin=dict(l=10, r=10, t=10, b=10),
)
st.plotly_chart(intra_fig, use_container_width=True)
st.caption(
    "🟧 첫 5분봉 OR  ·  🟪 FVG zone (3-bar bullish above OR)  ·  "
    "▲ 청록 = range-break candle (stop reference)  ·  "
    "▽ 파랑 = retest into OR (Path B만)  ·  "
    "⬆ 초록 = entry (Path A=즉시, Path B=retest 후 FVG)  ·  "
    f"초록 점선 = TP, 빨간 점선 = stop (FVG 첫봉 low, N={max_below_stop_strikes}회 침투 시 발화)  ·  "
    "⭕ TP 체결, ❌ stop, ◇ session close (TP 미터치)"
)

# ---- Trades table -------------------------------------------------
if perf.trades:
    st.subheader("Trades")
    EXIT_LABELS = {
        "take_profit": "Take Profit (R-target)",
        "session_close": "Session Close (TP not touched)",
        "stop_loss": f"Stop Loss ({max_below_stop_strikes}× below FVG-pre low)",
        "breakeven_stop": "Break-even stop (BE-armed, ~0R)",
    }
    rows = []
    for t in perf.trades:
        rows.append({
            "Entry": t.entry_ts,
            "Exit": t.exit_ts,
            "Reason": EXIT_LABELS.get(t.exit_reason, t.exit_reason),
            "Entry Price": f"${t.entry_price:,.2f}",
            "Exit Price": f"${t.exit_price:,.2f}",
            "Stop": f"${t.stop_loss:,.2f}",
            "Shares": t.shares,
            "P&L ($)": f"${t.pnl:,.2f}",
            "P&L (%)": f"{t.pnl_pct:.2%}",
        })
    st.dataframe(rows, use_container_width=True)
else:
    st.info(
        "이 기간엔 패턴이 트리거되지 않았어. "
        "min_fvg_gap_pct / max_total_bars / require_retest_for_path_b를 조정해봐."
    )
