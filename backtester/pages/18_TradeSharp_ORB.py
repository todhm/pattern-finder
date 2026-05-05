"""Trade Sharp ORB single-ticker page.

Implements the variant from the YouTube video "The ORB Strategy
Wasn't Working… Until I Did This!" by Trade Sharp. The key
differences vs the Scraface page (16):

  - **15-minute opening range** (not 5-min). Box = first 3 × 5m bars.
  - **5-minute execution timeframe** (not 1-min) — the video's
    explicit recommendation for the cleaner setup.
  - **First breakout = trap, not entry**. Wait for the liquidity
    grab (pullback into the OR) → bullish rejection candle →
    next-bar high break of the rejection's high. THAT is entry.
  - **Stop = rejection candle's low**, not box low or entry low.
  - **Daily bias** = previous daily candle bullish (close > open).
    Simpler regime check than Scraface's 7-of-10 SMA gate.

Charts: daily (bias context) + 5m (full play-by-play with all
state-machine markers) + 1m (zoomed-in entry detail when
available).
"""

from datetime import date, time, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data.adapters.composed_market_data import build_default_market_data
from data.adapters.regular_session_filter import RegularSessionFilterAdapter
from data.adapters.yfinance_adapter import YFinanceAdapter
from data.domain.market_calendar import KR, NY, market_for_ticker
from pattern.adapters.tradesharp_orb import TradeSharpORBDetector
from strategy.adapters.tradesharp_strategy import TradeSharpORBStrategy
from strategy.domain.models import StrategyConfig

INTRADAY_HISTORY_FLOOR = date(2022, 1, 1)

st.set_page_config(page_title="Trade Sharp ORB", layout="wide")
st.title("Trade Sharp ORB Strategy")
st.caption(
    "15분 OR → 첫 박스 돌파(트랩) → 박스 안으로 pullback (liquidity grab) → "
    "rejection candle (bullish close) → 다음 봉이 rejection high를 깰 때 진입. "
    "Stop = rejection low, TP = entry + R × (entry − stop). 영상 출처의 5분봉 변형."
)

with st.sidebar:
    st.header("Market")
    if "tradesharp_ticker" not in st.session_state:
        st.session_state.tradesharp_ticker = "QQQ"
    ticker = st.text_input("Ticker", key="tradesharp_ticker")
    detected_market = market_for_ticker(ticker)
    market_choice = st.selectbox(
        "Market calendar",
        options=["NY", "KR"],
        index=0 if detected_market.name == "NY" else 1,
        format_func=lambda x: {
            "NY": "🇺🇸 US (NYSE / Nasdaq)  — 09:30–16:00 ET",
            "KR": "🇰🇷 Korea (KOSPI/KOSDAQ) — 09:00–15:00 KST",
        }[x],
        help="영상은 NYSE 오픈 + 고변동 종목(NASDAQ/US30/SPX/gold)에 한정. "
        "다른 시장도 시도 가능하지만 비디오의 88% win rate는 NY 한정 주장.",
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
        help="단일 트레이드 notional 상한. 좁은 stop으로 commission 폭증 방지.",
    )

    st.header("Opening Range")
    box_minutes = st.number_input(
        "OR length (minutes)",
        value=15,
        min_value=5,
        max_value=60,
        step=5,
        help="영상 표준 = 15분 (5m × 3봉). 30분 / 60분도 시도 가능.",
    )
    latest_entry_hour = st.number_input(
        "Latest entry hour (local, exclusive)",
        value=11,
        min_value=10,
        max_value=16,
        step=1,
    )
    latest_entry_minute = st.number_input(
        "Latest entry minute",
        value=30,
        min_value=0,
        max_value=59,
        step=15,
        help="영상은 morning auction window. 기본 11:30 = NYSE 첫 2시간.",
    )

    st.header("Setup conditions")
    require_daily_bullish = st.checkbox(
        "Require previous daily candle bullish (close > open)",
        value=True,
        help="영상의 'higher timeframe bias' 룰. 전날 daily가 양봉이어야 long entry. "
        "OFF면 모든 세션에서 trigger.",
    )
    require_pullback_into_box = st.checkbox(
        "Require pullback back INTO the OR (low ≤ box_high)",
        value=True,
        help="영상의 핵심 — 'liquidity grab back into the open range'. "
        "OFF면 어떤 retracement도 pullback으로 인정 (false positives↑).",
    )
    min_pullback_depth = st.number_input(
        "Min pullback depth (% of box height into OR)",
        value=0.0,
        min_value=0.0,
        max_value=100.0,
        step=10.0,
        format="%.0f",
        help="0 = box_high 터치만 해도 OK. 50 = box 절반까지 내려와야. "
        "100 = box_low 터치 (가장 엄격). 영상은 명시 안 함, 기본 0.",
    )
    min_consolidation_bars = st.number_input(
        "Min consolidation bars (before entry can fire)",
        value=3,
        min_value=1,
        max_value=20,
        step=1,
        help="Pullback이 박스 안으로 들어온 후, 최소 N봉이 지나야 entry trigger 활성화. "
        "기본 3봉(=15분). 너무 작으면 첫 bounce에 성급히 진입함 (영상의 함정).",
    )
    max_consolidation_bars = st.number_input(
        "Max consolidation bars (timeout)",
        value=25,
        min_value=5,
        max_value=200,
        step=5,
        help="Initial breakout 후 N봉 안에 entry 안 트리거되면 setup invalidate. "
        "기본 25봉(≈125분).",
    )
    target_r_multiple = st.number_input(
        "Take profit (R-multiple)",
        value=1.5,
        min_value=0.5,
        max_value=10.0,
        step=0.5,
        format="%.1f",
        help="영상 예시 1:1 ~ 1:2. 기본 1.5.",
    )

    run_btn = st.button(
        "Run Trade Sharp ORB Backtest",
        type="primary",
        use_container_width=True,
    )

if not run_btn:
    st.info(
        "좌측에서 ticker / 기간을 설정하고 **Run Trade Sharp ORB Backtest**를 눌러. "
        "영상에서 추천하는 종목: NASDAQ-100 ETF(QQQ), SPY, GLD, US30 futures."
    )
    st.stop()

# Composition root
_base_md = build_default_market_data()
md = RegularSessionFilterAdapter(_base_md, market=market)
yf = YFinanceAdapter()

with st.spinner(f"Fetching 5m / daily for {ticker}..."):
    try:
        df_5m = md.fetch_ohlcv(
            ticker.upper(), start_date, end_date, interval="5m"
        )
        # Pad daily for the bias check
        df_daily = yf.fetch_ohlcv(
            ticker.upper(), start_date - timedelta(days=30), end_date
        )
    except Exception as exc:
        st.error(f"Data fetch failed: {exc}")
        st.stop()

if df_5m is None or df_5m.empty:
    st.warning("No 5-minute data returned for that range.")
    st.stop()

# tz convert intraday to market-local so detector reads right times
if df_5m.index.tz is not None and str(df_5m.index.tz) != market.tz:
    df_5m = df_5m.copy()
    df_5m.index = df_5m.index.tz_convert(market.tz)

detector = TradeSharpORBDetector(
    box_minutes=int(box_minutes),
    session_open_local=market.rth_open,
    latest_entry_local=time(int(latest_entry_hour), int(latest_entry_minute)),
    require_daily_bullish=bool(require_daily_bullish),
    require_pullback_into_box=bool(require_pullback_into_box),
    min_pullback_depth_pct=float(min_pullback_depth) / 100.0,
    max_consolidation_bars=int(max_consolidation_bars),
    min_consolidation_bars=int(min_consolidation_bars),
    target_r_multiple=float(target_r_multiple),
)
strategy = TradeSharpORBStrategy(
    detector,
    max_position_pct_of_equity=float(max_position_pct) / 100.0,
)

config = StrategyConfig(
    ticker=ticker.upper(),
    start_date=start_date,
    end_date=end_date,
    pattern_name="tradesharp_orb",
    initial_capital=float(initial_capital),
    risk_per_trade=risk_pct / 100.0,
    max_holding_days=1,
)

with st.spinner("Running backtest..."):
    try:
        result = strategy.run(df_5m, df_daily, config)
    except Exception as exc:
        st.error(f"Backtest failed: {exc}")
        st.stop()

perf = result.performance
chart_signals = detector.detect(df_5m, df_daily)

# ---- Headline metrics ---------------------------------------------
st.subheader(f"{ticker.upper()} — Trade Sharp ORB")
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
m8.metric(
    "Signals scanned",
    len(chart_signals),
    help="Detector가 발화한 시그널 총합 — 트레이드와 같음 (1포지션/세션)",
)

# ---- Daily chart with bias indicator ------------------------------
st.subheader("Daily — bias context")
_cutoff = pd.Timestamp(start_date)
if df_daily.index.tz is not None and _cutoff.tz is None:
    _cutoff = _cutoff.tz_localize(df_daily.index.tz)
d_disp = df_daily[df_daily.index >= _cutoff]
daily_fig = go.Figure(
    go.Candlestick(
        x=d_disp.index,
        open=d_disp["Open"],
        high=d_disp["High"],
        low=d_disp["Low"],
        close=d_disp["Close"],
        name="Daily",
        showlegend=False,
    )
)
# Shade days where YESTERDAY was bullish (so today qualifies for the
# bias gate). Helps visually verify the regime filter.
prev_bullish = (df_daily["Close"].shift(1) > df_daily["Open"].shift(1))
for ts, q in prev_bullish.items():
    if not bool(q) or ts < _cutoff:
        continue
    daily_fig.add_vrect(
        x0=ts - pd.Timedelta(hours=12),
        x1=ts + pd.Timedelta(hours=12),
        fillcolor="rgba(67,160,71,0.15)",
        line_width=0,
        layer="below",
    )
# Trade markers
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
    height=360,
    xaxis_rangeslider_visible=False,
    margin=dict(l=10, r=10, t=10, b=10),
)
st.plotly_chart(daily_fig, use_container_width=True)
st.caption(
    "🟢 음영 = 전날 daily가 bullish였던 세션 (bias 통과)  ·  "
    "🔺 초록 = win 트레이드, 🔻 빨강 = loss 트레이드"
)

# ---- 5-minute chart with full state-machine markers --------------
st.subheader("5 minute — full pattern play-by-play")

last_5m = {}
for ts in df_5m.index:
    last_5m[ts.date()] = ts
trades_by_date = {t.entry_date: t for t in perf.trades if t.entry_ts}

five_fig = go.Figure(
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
first_legend = {"box": True, "init_break": True, "pullback": True,
                "rejection": True, "entry": True, "exit": True,
                "stop_line": True, "tp_line": True}
for sig in chart_signals:
    end_ts = last_5m.get(sig.session_date, sig.box_close_ts)
    # First box (orange rectangle, full session width).
    five_fig.add_shape(
        type="rect",
        x0=sig.box_open_ts, x1=end_ts,
        y0=sig.box_low, y1=sig.box_high,
        line=dict(color="#FB8C00", width=1.4),
        fillcolor="rgba(251,140,0,0.15)",
        layer="below",
    )
    # Initial breakout marker (purple ▲ — the trap).
    five_fig.add_trace(go.Scatter(
        x=[sig.initial_breakout_ts], y=[sig.initial_breakout_price],
        mode="markers",
        marker=dict(symbol="triangle-up", color="#7E57C2", size=11),
        name="Initial breakout (trap)",
        legendgroup="init_break",
        showlegend=first_legend["init_break"],
        hovertemplate=(
            f"Initial breakout (NOT entry) @ ${sig.initial_breakout_price:.2f}"
            "<extra></extra>"
        ),
    ))
    first_legend["init_break"] = False
    # Pullback low marker (down-arrow, blue).
    five_fig.add_trace(go.Scatter(
        x=[sig.pullback_low_ts], y=[sig.pullback_low],
        mode="markers",
        marker=dict(symbol="triangle-down", color="#1976D2", size=11),
        name="Pullback low (liquidity grab)",
        legendgroup="pullback",
        showlegend=first_legend["pullback"],
        hovertemplate=(
            f"Pullback low @ ${sig.pullback_low:.2f}<extra></extra>"
        ),
    ))
    first_legend["pullback"] = False
    # Consolidation envelope — purple rectangle from the pullback
    # start to the entry bar at y=[pullback_low, consol_high]. This
    # is the "base" the trader waits to form before the buy-stop.
    five_fig.add_shape(
        type="rect",
        x0=sig.consol_start_ts,
        x1=sig.entry_ts,
        y0=sig.pullback_low,
        y1=sig.consol_high,
        line=dict(color="#7B1FA2", width=2, dash="dash"),
        fillcolor="rgba(186,104,200,0.12)",
        layer="below",
    )
    # Entry marker (green ⬆ at consol_high — buy-stop fill level).
    five_fig.add_trace(go.Scatter(
        x=[sig.entry_ts], y=[sig.entry_price],
        mode="markers",
        marker=dict(
            symbol="arrow-up", color="#2E7D32", size=14,
            line=dict(color="#1B5E20", width=1.5),
        ),
        name="Entry",
        legendgroup="entry",
        showlegend=first_legend["entry"],
        hovertemplate=f"Entry @ ${sig.entry_price:.2f}<extra></extra>",
    ))
    first_legend["entry"] = False
    # Stop / TP horizontal levels — Scatter so they render above
    # candles.
    five_fig.add_trace(go.Scatter(
        x=[sig.entry_ts, end_ts],
        y=[sig.take_profit, sig.take_profit],
        mode="lines",
        line=dict(color="#2E7D32", width=1.6, dash="dash"),
        name="Take profit",
        legendgroup="tp_line",
        showlegend=first_legend["tp_line"],
        hovertemplate=f"TP @ ${sig.take_profit:.2f}<extra></extra>",
    ))
    five_fig.add_trace(go.Scatter(
        x=[sig.entry_ts, end_ts],
        y=[sig.stop_loss, sig.stop_loss],
        mode="lines",
        line=dict(color="#C62828", width=1.6, dash="dash"),
        name="Stop",
        legendgroup="stop_line",
        showlegend=first_legend["stop_line"],
        hovertemplate=f"Stop @ ${sig.stop_loss:.2f}<extra></extra>",
    ))
    first_legend["tp_line"] = False
    first_legend["stop_line"] = False
    five_fig.add_annotation(
        x=end_ts, y=sig.take_profit,
        text=f"TP {sig.take_profit:.2f}",
        showarrow=False, xanchor="right", yanchor="bottom",
        font=dict(color="#1B5E20", size=10),
        bgcolor="rgba(255,255,255,0.65)",
    )
    five_fig.add_annotation(
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
    if exit_ts.tz is None and df_5m.index.tz is not None:
        exit_ts = exit_ts.tz_localize(df_5m.index.tz)
    color = "#2E7D32" if t.pnl > 0 else "#C62828"
    if t.exit_reason == "session_close":
        symbol, color = "diamond-open", "#757575"
    elif t.exit_reason == "stop_loss":
        symbol = "x"
    else:
        symbol = "circle"
    five_fig.add_trace(go.Scatter(
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

five_fig.update_layout(
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
st.plotly_chart(five_fig, use_container_width=True)
st.caption(
    "🟧 박스 (15분 OR)  ·  🟪 작은 보라 박스 = rejection candle 영역  ·  "
    "▲ 보라 = 첫 박스 돌파 (트랩)  ·  ▽ 파랑 = pullback low (liquidity grab)  ·  "
    "⬆ 초록 = entry (rejection high 돌파)  ·  "
    "초록 점선 = TP, 빨간 점선 = stop  ·  "
    "⭕ 초록 = TP 체결, ❌ 빨강 = stop, ◇ 회색 = session close"
)

# ---- Trades table -------------------------------------------------
if perf.trades:
    st.subheader("Trades")
    EXIT_LABELS = {
        "stop_loss": "Stop Loss (rejection low)",
        "take_profit": "Take Profit (R-target)",
        "session_close": "Session Close (forced flat)",
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
        "Daily bias / pullback depth / max bars 설정을 조정해봐."
    )
