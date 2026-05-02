"""Scraface Trade Pattern — single-ticker bull-only ORB retest.

Implements ``ScrafaceTradePattern.docx``: 50-SMA daily regime →
opening 5-minute box on the intraday → 1-minute breakout-and-retest
entry with 1:2 R/R. Bull-only (long) per page spec.

Regime gate: TODAY plus at least N-of-M of the most recent daily
closes must sit *below* the 50 SMA. The default 7-of-10 (incl.
today) targets sustained-suppression bounces — the bullish ORB
retest after a stretch of weak closes.

Three timeframes are rendered side-by-side so the multi-TF logic
is visually inspectable:
  - **Daily** with 50 SMA, qualifying days highlighted.
  - **5 minute** intraday context, opening 5m candle (the box)
    highlighted.
  - **1 minute** with the box rectangle drawn precisely from
    box_open_ts → session end at y=[box_low, box_high], plus
    breakout / retest / trade markers.
"""

from datetime import date, time, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from data.adapters.composed_market_data import build_default_market_data
from data.adapters.regular_session_filter import RegularSessionFilterAdapter
from data.adapters.yfinance_adapter import YFinanceAdapter
from data.domain.market_calendar import NY
from pattern.adapters.scraface_orb import ScrafaceORBDetector
from strategy.adapters.scraface_strategy import ScrafaceORBStrategy
from strategy.domain.models import StrategyConfig

# EODHD 1m data depth ≈ 2 years; we still let users go further back
# (the call will simply 422 on too-old start), but cap the date input
# floor to a sensible year so the picker isn't infinite.
INTRADAY_HISTORY_FLOOR = date(2022, 1, 1)

st.set_page_config(page_title="Scraface Pattern", layout="wide")
st.title("Scraface Trade Pattern (Bull-only ORB Retest)")
st.caption(
    "Daily 50 SMA 아래 눌린 종목의 첫 5분봉 box를 1분봉에서 위로 돌파 → "
    "박스 상단 retest → long 진입 (1:2 R/R). 박스 안에서는 트레이드 안 함."
)

with st.sidebar:
    st.header("Market")
    ticker = st.text_input("Ticker", value="TSLA")
    start_date = st.date_input(
        "Start Date",
        value=date.today() - timedelta(days=14),
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

    st.header("Daily regime filter")
    sma_period = st.number_input(
        "SMA period (daily)",
        value=50,
        min_value=5,
        max_value=400,
        step=5,
    )
    days_below_lookback = st.number_input(
        "Lookback days",
        value=10,
        min_value=2,
        max_value=60,
        step=1,
        help="이 기간 동안 close가 SMA 아래에 있던 일수를 셈.",
    )
    min_days_below = st.number_input(
        "Min days below SMA (incl. today)",
        value=7,
        min_value=1,
        max_value=60,
        step=1,
        help="해당 당일 + 직전 lookback일 중 N일 이상 close가 SMA 아래여야 "
        "long 시그널 fire. 기본 7/10.",
    )

    st.header("Box / breakout")
    box_minutes = st.number_input(
        "Box minutes (opening range)",
        value=5,
        min_value=1,
        max_value=60,
        step=1,
        help="첫 N분 동안의 1분봉 OHLC 범위 = box.",
    )
    breakout_buffer = st.number_input(
        "Breakout buffer (× box height)",
        value=0.0,
        min_value=0.0,
        max_value=1.0,
        step=0.05,
        format="%.2f",
        help="close가 box_high + 이 값 × box height 위로 가야 breakout 인정. "
        "0이면 아주 살짝만 넘어도 OK.",
    )
    retest_tolerance = st.number_input(
        "Retest tolerance (× box height)",
        value=0.10,
        min_value=0.0,
        max_value=1.0,
        step=0.05,
        format="%.2f",
        help="bar의 low가 box_high - 이 값 × box height까지 내려와야 'retest' "
        "로 인정. 0 = 정확히 box_high tag만 인정.",
    )
    max_retest_bars = st.number_input(
        "Max bars from breakout to retest",
        value=60,
        min_value=5,
        max_value=300,
        step=5,
        help="breakout 봉 이후 N분 안에 retest 안 되면 setup invalidate.",
    )
    target_r_multiple = st.number_input(
        "Target (R-multiple)",
        value=2.0,
        min_value=0.5,
        max_value=10.0,
        step=0.5,
        format="%.1f",
        help="원문 1:2 R/R 권고. Stop은 entry bar의 low, TP = entry + R × (entry − stop).",
    )
    latest_entry_hour = st.number_input(
        "Latest entry hour (local, exclusive)",
        value=11,
        min_value=10,
        max_value=16,
        step=1,
        help="이 시각 이전(09:30 ≤ t < 이 시각)에만 진입 허용. "
        "기본 11 → 09:30~11:00 morning window. 이후 entry는 reject.",
    )

    run_btn = st.button(
        "Run Scraface Backtest", type="primary", use_container_width=True
    )

if not run_btn:
    st.info("좌측에서 ticker / 기간을 설정하고 **Run Scraface Backtest**를 눌러.")
    st.stop()

# Session-bound day-trading. Wrap the composed sub-daily/daily router
# in an RTH filter so 1m bars only contain 09:30–16:00 ET prints
# (otherwise pre/post-market bars leak in and stop/TP/session-end
# logic prices off thin extended-hours data).
_base_md = build_default_market_data()
md = RegularSessionFilterAdapter(_base_md, market=NY)
yf = YFinanceAdapter()

# Pad daily history so the SMA + the lookback window are converged
# from the very first session in the user's date range.
daily_pad_days = int(sma_period) + int(days_below_lookback) + 30
daily_start = start_date - timedelta(days=daily_pad_days * 2)

with st.spinner(f"Fetching 1m / 5m / daily for {ticker}..."):
    try:
        df_1m = md.fetch_ohlcv(
            ticker.upper(), start_date, end_date, interval="1m"
        )
        df_5m = md.fetch_ohlcv(
            ticker.upper(), start_date, end_date, interval="5m"
        )
        df_daily = yf.fetch_ohlcv(ticker.upper(), daily_start, end_date)
    except Exception as exc:
        st.error(f"Data fetch failed: {exc}")
        st.stop()

if df_1m is None or df_1m.empty:
    st.warning("No 1-minute data returned for that range.")
    st.stop()

detector = ScrafaceORBDetector(
    sma_period=int(sma_period),
    days_below_lookback=int(days_below_lookback),
    min_days_below=int(min_days_below),
    box_minutes=int(box_minutes),
    session_open_local=time(9, 30),
    latest_entry_local=time(int(latest_entry_hour), 0),
    breakout_buffer_atr=float(breakout_buffer),
    retest_tolerance_atr=float(retest_tolerance),
    max_retest_bars=int(max_retest_bars),
    target_r_multiple=float(target_r_multiple),
)
strategy = ScrafaceORBStrategy(detector)

config = StrategyConfig(
    ticker=ticker.upper(),
    start_date=start_date,
    end_date=end_date,
    pattern_name="scraface_orb",
    initial_capital=float(initial_capital),
    risk_per_trade=risk_pct / 100.0,
    max_holding_days=1,
)

with st.spinner("Running backtest..."):
    try:
        result = strategy.run(df_1m, df_daily, config)
    except Exception as exc:
        st.error(f"Backtest failed: {exc}")
        st.stop()

perf = result.performance
# Re-detect against the chart frames so we can paint signals even on
# days where the strategy was already in a position (none expected
# here since we're 1-pos-at-a-time and trades close intraday, but
# keeps the chart authoritative).
chart_signals = detector.detect(df_1m, df_daily)

# ---- Headline metrics -----------------------------------------------
st.subheader(f"{ticker.upper()} — Scraface ORB")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Trades", perf.total_trades)
m2.metric(
    "Win Rate", f"{perf.win_rate:.0%}" if perf.total_trades else "—"
)
m3.metric("Total Return", f"{perf.total_return_pct:.2%}")
m4.metric("Final Capital", f"${perf.final_capital:,.0f}")

m5, m6, m7, m8 = st.columns(4)
m5.metric("Initial Capital", f"${perf.initial_capital:,.0f}")
m6.metric("Avg Win", f"{perf.avg_win_pct:.2%}" if perf.trades else "—")
m7.metric("Avg Loss", f"{perf.avg_loss_pct:.2%}" if perf.trades else "—")
m8.metric("Max Drawdown", f"{perf.max_drawdown_pct:.2%}")

# ---- Daily chart with SMA + qualifying-day shading ------------------
st.subheader("Daily — 50 SMA & qualifying days")

sma = df_daily["Close"].rolling(int(sma_period)).mean()
below = (df_daily["Close"] < sma).astype(int)
rolling_below = below.rolling(
    int(days_below_lookback), min_periods=int(days_below_lookback)
).sum()
qualifying = rolling_below >= int(min_days_below)
qualifying = qualifying & (below == 1)
# Slice to the user's display window (we kept the warmup pad just for
# convergence — don't overwhelm the chart with it). yfinance hands
# back a tz-aware NY index, so the cutoff has to match — naive
# Timestamp comparison errors out under newer pandas.
_cutoff = pd.Timestamp(start_date)
if df_daily.index.tz is not None and _cutoff.tz is None:
    _cutoff = _cutoff.tz_localize(df_daily.index.tz)
display_mask = df_daily.index >= _cutoff
df_d_disp = df_daily[display_mask]
sma_disp = sma[display_mask]
qual_disp = qualifying[display_mask]

daily_fig = go.Figure()
daily_fig.add_trace(
    go.Candlestick(
        x=df_d_disp.index,
        open=df_d_disp["Open"],
        high=df_d_disp["High"],
        low=df_d_disp["Low"],
        close=df_d_disp["Close"],
        name="Daily",
        showlegend=False,
    )
)
daily_fig.add_trace(
    go.Scatter(
        x=sma_disp.index,
        y=sma_disp.values,
        mode="lines",
        line=dict(color="#1976D2", width=1.6),
        name=f"{int(sma_period)} SMA",
    )
)
# Highlight qualifying days as soft green vertical bands so the user
# can see at a glance which sessions the detector will scan.
for ts, q in qual_disp.items():
    if q:
        daily_fig.add_vrect(
            x0=ts - pd.Timedelta(hours=12),
            x1=ts + pd.Timedelta(hours=12),
            fillcolor="rgba(67,160,71,0.18)",
            line_width=0,
            layer="below",
        )
daily_fig.update_layout(
    height=380,
    xaxis_rangeslider_visible=False,
    margin=dict(l=10, r=10, t=30, b=10),
)
st.plotly_chart(daily_fig, use_container_width=True)
st.caption("🟢 shaded = 해당 당일 + 직전 N일 중 ≥M일이 50 SMA 아래인 qualifying day.")

# ---- 5-minute chart -------------------------------------------------
st.subheader("5 minute")
if df_5m is not None and not df_5m.empty:
    five_fig = go.Figure()
    five_fig.add_trace(
        go.Candlestick(
            x=df_5m.index,
            open=df_5m["Open"],
            high=df_5m["High"],
            low=df_5m["Low"],
            close=df_5m["Close"],
            name="5m",
            showlegend=False,
        )
    )
    # Highlight the opening 5-minute candle (the box) for each
    # qualifying session — that's the bar whose OHLC define the
    # trade's structural levels.
    qualifying_dates = {ts.date() for ts, q in qualifying.items() if q}
    sigs_by_date = {s.session_date: s for s in chart_signals}
    for ts in df_5m.index:
        if ts.time() != time(9, 30):
            continue
        if ts.date() not in qualifying_dates:
            continue
        sig = sigs_by_date.get(ts.date())
        # Whether or not a signal fired, every qualifying day's
        # opening 5m bar is the box. Color = orange when no signal
        # fired (box never produced a tradeable retest), green when
        # a signal did.
        color = "rgba(67,160,71,0.45)" if sig else "rgba(251,140,0,0.35)"
        five_fig.add_vrect(
            x0=ts,
            x1=ts + pd.Timedelta(minutes=5),
            fillcolor=color,
            line_width=0,
            layer="below",
        )
    five_fig.update_layout(
        height=320,
        xaxis_rangeslider_visible=False,
        xaxis_rangebreaks=[
            dict(bounds=[16, 9.5], pattern="hour"),
            dict(bounds=["sat", "mon"]),
        ],
        margin=dict(l=10, r=10, t=30, b=10),
    )
    st.plotly_chart(five_fig, use_container_width=True)
    st.caption("🟩 첫 5분봉 (qualifying day, signal fired)  ·  🟧 첫 5분봉 (qualifying, no retest)")
else:
    st.caption("5분봉 데이터 없음.")

# ---- 1 minute chart with precise box overlays ----------------------
st.subheader("1 minute (with opening box highlighted)")

trades_by_entry = {pd.Timestamp(t.entry_ts): t for t in perf.trades if t.entry_ts}

one_fig = make_subplots(
    rows=1, cols=1, shared_xaxes=True, vertical_spacing=0.02
)
one_fig.add_trace(
    go.Candlestick(
        x=df_1m.index,
        open=df_1m["Open"],
        high=df_1m["High"],
        low=df_1m["Low"],
        close=df_1m["Close"],
        name="1m",
        showlegend=False,
    )
)

# Build a date → last-1m-bar lookup so each session's box rectangle
# extends from box_open_ts to the actual final RTH bar of that day
# (filter already trimmed pre/post — last bar = ~15:59 ET).
last_bar_per_date: dict[date, pd.Timestamp] = {}
for ts in df_1m.index:
    last_bar_per_date[ts.date()] = ts

for sig in chart_signals:
    end_ts = last_bar_per_date.get(sig.session_date, sig.box_close_ts)
    # First box (opening 5m) — orange. Drawn as a shape so it's
    # locked to data coordinates (zoom preserves the band).
    one_fig.add_shape(
        type="rect",
        x0=sig.box_open_ts,
        x1=end_ts,
        y0=sig.box_low,
        y1=sig.box_high,
        line=dict(color="#FB8C00", width=1.4),
        fillcolor="rgba(251,140,0,0.18)",
        layer="below",
    )
    # Second box — purple. Spans from the initial breakout bar to the
    # entry bar, with high = peak (first-breakout bar's high) and low
    # = deepest pullback during the retest. The user's "두번째 박스".
    one_fig.add_shape(
        type="rect",
        x0=sig.breakout_ts,
        x1=sig.entry_ts,
        y0=sig.second_box_low,
        y1=sig.second_box_high,
        line=dict(color="#7E57C2", width=1.4, dash="dash"),
        fillcolor="rgba(126,87,194,0.18)",
        layer="below",
    )
    # Box edge labels for clarity (top resistance, bottom support).
    one_fig.add_annotation(
        x=sig.box_open_ts,
        y=sig.box_high,
        text=f"box high {sig.box_high:.2f}",
        showarrow=False,
        font=dict(color="#E65100", size=10),
        xanchor="left",
        yanchor="bottom",
    )
    one_fig.add_annotation(
        x=sig.box_open_ts,
        y=sig.box_low,
        text=f"box low {sig.box_low:.2f}",
        showarrow=False,
        font=dict(color="#E65100", size=10),
        xanchor="left",
        yanchor="top",
    )
    # Breakout marker (purple triangle-up at breakout bar's close).
    one_fig.add_trace(
        go.Scatter(
            x=[sig.breakout_ts],
            y=[sig.breakout_price],
            mode="markers",
            marker=dict(symbol="triangle-up", color="#7E57C2", size=12),
            name="Breakout",
            legendgroup="breakout",
            showlegend=(sig is chart_signals[0]),
            hovertemplate="Breakout @ $%{y:.2f}<extra></extra>",
        )
    )
    # Entry marker (signal — green up-arrow). Trade entry uses a
    # distinct marker below so the user sees both even when they
    # coincide.
    one_fig.add_trace(
        go.Scatter(
            x=[sig.entry_ts],
            y=[sig.entry_price],
            mode="markers",
            marker=dict(
                symbol="arrow-up", color="#2E7D32", size=14,
                line=dict(color="#1B5E20", width=1.5),
            ),
            name="Retest entry",
            legendgroup="entry",
            showlegend=(sig is chart_signals[0]),
            hovertemplate="Entry @ $%{y:.2f}<extra></extra>",
        )
    )
    # TP / Stop dotted lines from entry → end of session.
    one_fig.add_shape(
        type="line",
        x0=sig.entry_ts, x1=end_ts,
        y0=sig.take_profit, y1=sig.take_profit,
        line=dict(color="#2E7D32", width=1.2, dash="dot"),
    )
    one_fig.add_shape(
        type="line",
        x0=sig.entry_ts, x1=end_ts,
        y0=sig.stop_loss, y1=sig.stop_loss,
        line=dict(color="#C62828", width=1.2, dash="dot"),
    )

# Trade exit markers — green dot for win (TP), red X for stop, gray
# diamond for session-close.
for t in perf.trades:
    if not t.exit_ts:
        continue
    color = "#2E7D32" if t.pnl > 0 else "#C62828"
    if t.exit_reason == "session_close":
        symbol, color = "diamond-open", "#757575"
    elif t.exit_reason == "stop_loss":
        symbol = "x"
    else:
        symbol = "circle"
    one_fig.add_trace(
        go.Scatter(
            x=[pd.Timestamp(t.exit_ts)],
            y=[t.exit_price],
            mode="markers",
            marker=dict(symbol=symbol, color=color, size=12),
            name=f"Exit ({t.exit_reason})",
            showlegend=False,
            hovertemplate=(
                f"{t.exit_reason} @ $%{{y:.2f}} (PnL ${t.pnl:+.2f})<extra></extra>"
            ),
        )
    )

one_fig.update_layout(
    height=560,
    xaxis_rangeslider_visible=False,
    xaxis_rangebreaks=[
        dict(bounds=[16, 9.5], pattern="hour"),
        dict(bounds=["sat", "mon"]),
    ],
    margin=dict(l=10, r=10, t=30, b=10),
)
st.plotly_chart(one_fig, use_container_width=True)
st.caption(
    "🟧 첫 박스 (개장 N분, y=[box_low, box_high])  ·  "
    "🟪 두번째 박스 (breakout→retest 구간, y=[retest low, peak])  ·  "
    "▲ purple = 첫 박스 breakout  ·  ⬆️ green = 두번째 박스 breakout = entry  ·  "
    "초록 점선 = TP (entry+R×R), 빨간 점선 = stop (entry bar low)  ·  "
    "⭕ TP 체결, ❌ stop, ◇ session close 강제청산"
)

# ---- Trade table ---------------------------------------------------
if perf.trades:
    st.subheader("Trades")
    EXIT_LABELS = {
        "stop_loss": "Stop Loss (box low)",
        "take_profit": "Take Profit (R-target)",
        "session_close": "Session Close (forced flat)",
    }
    rows = []
    for t in perf.trades:
        rows.append(
            {
                "Entry": t.entry_ts,
                "Exit": t.exit_ts,
                "Reason": EXIT_LABELS.get(t.exit_reason, t.exit_reason),
                "Entry Price": f"${t.entry_price:,.2f}",
                "Exit Price": f"${t.exit_price:,.2f}",
                "Stop": f"${t.stop_loss:,.2f}",
                "Shares": t.shares,
                "P&L ($)": f"${t.pnl:,.2f}",
                "P&L (%)": f"{t.pnl_pct:.2%}",
            }
        )
    st.dataframe(rows, use_container_width=True)
else:
    st.info("이 기간엔 Scraface signal이 트리거되지 않았어. SMA / lookback / box 설정을 조정해봐.")
