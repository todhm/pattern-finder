from datetime import date, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data.adapters.yfinance_adapter import YFinanceAdapter
from pattern.adapters.wedge_pop import WedgePopDetector
from strategy.adapters.wedgepop_strategy import WedgepopStrategy
from strategy.domain.models import StrategyConfig
from visualization.adapters.plotly_charts import PlotlyChartBuilder

st.set_page_config(page_title="Strategy", layout="wide")
st.title("Wedge Pop Strategy")
st.caption(
    "Wedge Pop 신호 다음날 open 매수 → 손절(consolidation low) / "
    "Exhaustion 익절 / 10 EMA trail / time stop"
)

# --- Sidebar inputs ---
with st.sidebar:
    st.header("Market")
    ticker = st.text_input("Ticker", value="AAPL")
    start_date = st.date_input("Start Date", value=date.today() - timedelta(days=365))
    end_date = st.date_input("End Date", value=date.today())

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
        help="자본의 몇 %를 거래당 위험에 노출할지. 일반적으로 1~2%, "
        "공격적으로 운용하면 5~10%, 실험적으론 그 이상까지 가능.",
    )
    max_holding_days = st.number_input(
        "Max Holding Days",
        value=60,
        min_value=1,
        max_value=2_000,
        step=5,
        help="Time stop 발동까지 최대 보유 일수.",
    )

    st.header("Pattern Detection")
    consolidation_pct = st.number_input(
        "Min consolidation %",
        value=60.0,
        min_value=0.0,
        max_value=100.0,
        step=5.0,
        help="직전 N일 중 close가 fast EMA 아래에 있어야 하는 " "**최소** 비율. 0%면 하한 off.",
    )
    enable_max_cp = st.checkbox(
        "Cap max consolidation %",
        value=False,
        help="상한 추가 — 너무 깊은 consolidation (95%+)은 coil이 "
        "아니라 capitulation이라 제외하고 싶을 때.",
    )
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
        value=0.01,
        min_value=0.0,
        max_value=10.0,
        step=0.001,
        format="%.3f",
        help="breakout move가 ATR의 몇 배 이상이어야 wedge pop으로 인정. "
        "ATR 기반이라 종목 변동성에 자동 적응. 0이면 하한 off.",
    )
    enable_max_bp = st.checkbox(
        "Cap max breakout strength",
        value=False,
        help="상한 추가 — 이미 너무 큰 단일봉 gap(+4 ATR+)은 overextended " "라 제외하고 싶을 때.",
    )
    max_breakout_atr_mult_ui = st.number_input(
        "Max breakout strength (× ATR)",
        value=3.0,
        min_value=0.0,
        max_value=20.0,
        step=0.5,
        format="%.2f",
        disabled=not enable_max_bp,
        help="breakout이 ATR의 이 배수를 넘으면 overextended로 제외.",
    )
    require_above_long_smas = st.checkbox(
        "Require close above 50 & 200 SMA",
        value=True,
        help="signal 캔들의 close가 50 SMA와 200 SMA 둘 다 위에 있을 "
        "때만 wedge pop으로 인정. 장기 추세 confirm 용.",
    )
    detect_lookback = st.number_input(
        "Consolidation lookback (days)",
        value=10,
        min_value=3,
        max_value=60,
        step=1,
        help="Consolidation 체크 및 stop-loss (consolidation low) 계산에 "
        "쓰이는 직전 봉 개수. 기본 15일.",
    )
    cooldown_bars_ui = st.number_input(
        "Cooldown bars (after a signal)",
        value=15,
        min_value=0,
        max_value=60,
        step=1,
        help="한 signal이 fire된 후 다음 signal까지 건너뛰는 바 개수. "
        "0으로 두면 쿨다운 없이 연속 signal 허용 (예전엔 "
        "Consolidation lookback 값을 그대로 쓰던 부분을 독립 파라미터로 "
        "분리). Case: AAPL 2025-08-06은 07-16 signal의 기본 쿨다운에 "
        "걸려서 안 잡히는데, 5 정도로 낮추면 잡힘.",
    )
    ema_fast = st.number_input(
        "Fast EMA period",
        value=10,
        min_value=2,
        max_value=100,
        step=1,
        help="Detector와 strategy가 공통으로 쓰는 fast EMA (기본 10).",
    )
    ema_slow = st.number_input(
        "Slow EMA period",
        value=20,
        min_value=2,
        max_value=200,
        step=1,
        help="Detector의 breakout 체크와 strategy의 exhaustion reference "
        "라인 max(fast, slow) 계산에 쓰이는 slow EMA (기본 20).",
    )
    slope_lookback = st.number_input(
        "Slope lookback (days)",
        value=20,
        min_value=5,
        max_value=120,
        step=1,
        help="ema_fast_slope / ema_slow_slope를 계산하는 기간. "
        "signal metadata에 담겨 downstream slope 필터의 입력이 됨.",
    )

    st.header("Entry Filter")
    require_gap_up = st.checkbox(
        "Require gap-up confirmation",
        value=False,
        help="다음날 시초가가 전날 종가보다 높을 때만 진입 (TraderLion의 "
        "'Wedge Pops with unfilled gaps show strong momentum' 룰).",
    )
    enable_chase_filter = st.checkbox(
        "Enable entry chase filter",
        value=True,
        help="Entry open이 signal bar high 위로 너무 멀리 gap-up하면 "
        "'chasing'으로 간주해 진입 거부. 끄면 gap-up 제한 없음.",
    )
    max_entry_chase_ratio = st.number_input(
        "Max entry chase (× signal range)",
        value=0.15,
        min_value=0.0,
        max_value=5.0,
        step=0.05,
        format="%.2f",
        disabled=not enable_chase_filter,
        help="Entry open이 signal bar high 위로 signal bar 범위의 몇 배 "
        "까지 초과해도 허용할지. 넘으면 'chasing' 거부. "
        "주의: 0 = 모든 gap-up 차단 (가장 빡빡), 5 = 사실상 off.",
    )
    enable_entry_ema_filter = st.checkbox(
        "Enable EMA-extension entry filter",
        value=False,
        help="Entry open이 signal bar의 max(10EMA, 20EMA) 위로 N × ATR "
        "이상 초과하면 진입 거부. ATR 기반이라 변동성에 자동 적응.",
    )
    max_entry_ema_extension_atr = st.number_input(
        "Max entry extension above EMA (× ATR)",
        value=1.5,
        min_value=0.0,
        max_value=20.0,
        step=0.1,
        format="%.2f",
        disabled=not enable_entry_ema_filter,
        help="(entry_open - max(ema10, ema20)) / ATR > 이 값이면 거부.",
    )
    st.caption("**EMA slow slope 범위 필터** — 양수/음수 모두 가능.")
    enable_min_slope = st.checkbox(
        "Enforce min EMA slow slope",
        value=True,
        help="ema_slow_slope의 **하한**. 양수로 설정하면 '반드시 +N% "
        "이상 올라가는 추세'만 통과 (예: 0.05 → 20일 동안 +5% 이상). "
        "음수로 설정하면 dead-cat bounce를 걸러냄 (예: -0.01).",
    )
    min_ema_slow_slope_ui = st.number_input(
        "Min EMA slow slope",
        value=-0.01,
        min_value=-1.0,
        max_value=1.0,
        step=0.01,
        format="%.3f",
        disabled=not enable_min_slope,
    )
    enable_max_slope = st.checkbox(
        "Enforce max EMA slow slope",
        value=False,
        help="ema_slow_slope의 **상한**. 이미 너무 가파르게 오른 "
        "종목 (parabolic)을 제외하려면 켜세요 (예: 0.30 → +30% 이상 "
        "오른 종목 제외).",
    )
    max_ema_slow_slope_ui = st.number_input(
        "Max EMA slow slope",
        value=0.30,
        min_value=-1.0,
        max_value=5.0,
        step=0.05,
        format="%.3f",
        disabled=not enable_max_slope,
    )

    st.header("Exit Tuning")
    extension_atr_mult = st.number_input(
        "Exhaustion (× ATR above EMA)",
        value=1.5,
        min_value=0.1,
        max_value=50.0,
        step=0.1,
        help="Exit 룰: bar의 **High**가 max(fast, slow) EMA + ATR × 이 "
        "값 선을 터치하면 그 라인에서 체결 (limit order 모델). ATR "
        "기반이라 변동성에 자동 적응 — 같은 2.5가 모든 종목에서 비슷한 "
        "시각적 거리를 의미함.",
    )
    climax_atr_mult = st.number_input(
        "Climax bar ATR multiplier",
        value=0.8,
        min_value=0.0,
        max_value=10.0,
        step=0.001,
        help="단일봉 climax/blow-off 감지 임계. bar range 및 단일봉 "
        "move가 ATR × 이 값 이상이면서 close가 상단 20%에 위치 → "
        "익절 (예: AMD 2019-03-19 climax).",
    )
    atr_period = st.number_input(
        "ATR period",
        value=14,
        min_value=2,
        max_value=100,
        step=1,
        help="ATR(N) 윈도우. Exhaustion / climax 모두 이 ATR 사용 (기본 14).",
    )
    trail_after_profit = st.checkbox(
        "Arm EMA trail only after profit",
        value=True,
        help="꺼두면 첫 bar부터 EMA trail이 발동 — breakout 캔들의 EMA "
        "retest에 손절될 수 있어 권장 안 함.",
    )
    arm_breakeven = st.checkbox(
        "Ratchet stop to breakeven after profit",
        value=True,
        help="수익권 진입 시 consolidation-low stop을 entry price까지 "
        "올림. 이후 하락하면 breakeven 청산 (CDNS 2026-02 case).",
    )

    run_btn = st.button("Run Strategy", type="primary", use_container_width=True)


# --- Main ---
if not run_btn:
    st.info("좌측에서 ticker / 기간 / 위험 파라미터를 설정하고 **Run Strategy**를 눌러줘.")
    st.stop()

market_data = YFinanceAdapter()

# Fetch extra history so 50/200 SMA are converged from day 1 of the
# user's date range. ~400 calendar days ≈ 200+ trading days.
fetch_start = start_date - timedelta(days=400)
with st.spinner(f"Fetching {ticker} {fetch_start} → {end_date} (SMA warmup)..."):
    try:
        df = market_data.fetch_ohlcv(ticker, fetch_start, end_date)
    except Exception as e:
        st.error(f"Failed to fetch data: {e}")
        st.stop()

if df is None or df.empty:
    st.warning("No data returned for that range.")
    st.stop()

config = StrategyConfig(
    ticker=ticker,
    start_date=start_date,
    end_date=end_date,
    pattern_name="wedge_pop",
    initial_capital=float(initial_capital),
    risk_per_trade=risk_pct / 100.0,
    max_holding_days=max_holding_days,
)

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
)
strategy = WedgepopStrategy(
    market_data=market_data,
    detector=detector,
    ema_trail=int(ema_fast),
    ema_slow=int(ema_slow),
    atr_period=int(atr_period),
    extension_atr_mult=extension_atr_mult,
    climax_atr_mult=climax_atr_mult,
    max_entry_chase_ratio=(max_entry_chase_ratio if enable_chase_filter else float("inf")),
    max_entry_ema_extension_atr=(max_entry_ema_extension_atr if enable_entry_ema_filter else None),
    max_ema_slope_decline=None,  # superseded by min/max_ema_slow_slope
    min_ema_slow_slope=(min_ema_slow_slope_ui if enable_min_slope else None),
    max_ema_slow_slope=(max_ema_slow_slope_ui if enable_max_slope else None),
    trail_after_profit=trail_after_profit,
    arm_breakeven_after_profit=arm_breakeven,
    require_gap_up=require_gap_up,
)

with st.spinner("Running strategy..."):
    try:
        result = strategy.execute(df, config)
    except Exception as e:
        st.error(f"Strategy failed: {e}")
        st.stop()

perf = result.performance

# --- Headline metrics ---
st.subheader(f"{ticker} — Wedge Pop Strategy")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Trades", perf.total_trades)
m2.metric(
    "Win Rate",
    f"{perf.win_rate:.0%}" if perf.total_trades else "—",
)
m3.metric("Total Return", f"{perf.total_return_pct:.2%}")
m4.metric("Final Capital", f"${perf.final_capital:,.0f}")

m5, m6, m7, m8 = st.columns(4)
m5.metric("Initial Capital", f"${perf.initial_capital:,.0f}")
m6.metric("Avg Win", f"{perf.avg_win_pct:.2%}" if perf.trades else "—")
m7.metric("Avg Loss", f"{perf.avg_loss_pct:.2%}" if perf.trades else "—")
m8.metric("Max Drawdown", f"{perf.max_drawdown_pct:.2%}")

# --- Candlestick with trade markers + trigger lines ---
chart_builder = PlotlyChartBuilder()
trade_fig = chart_builder.build_candlestick_with_trades(
    df,
    perf.trades,
    title=f"{ticker} — Buy / Sell / Stop",
)

# Overlay trigger lines during each trade's holding period so the user
# can see exactly where the exhaustion / climax exits sit — makes
# ATR-based multipliers visually intuitive.
df_ind = strategy._with_indicators(df)
if df_ind.index.tz is not None:
    df_ind = df_ind.copy()
    df_ind.index = df_ind.index.tz_localize(None)
date_map = {idx.date(): idx for idx in df_ind.index}

for t in perf.trades:
    if t.entry_date not in date_map or t.exit_date not in date_map:
        continue
    entry_ts = date_map[t.entry_date]
    exit_ts = date_map[t.exit_date]
    entry_loc = df_ind.index.get_loc(entry_ts)
    exit_loc = df_ind.index.get_loc(exit_ts)
    trade_slice = df_ind.iloc[entry_loc : exit_loc + 1]

    ema_f = trade_slice["ema_trail"]
    ema_s = trade_slice["ema_slow"]
    atr_s = trade_slice["atr"]
    ref_ema = pd.concat([ema_f, ema_s], axis=1).max(axis=1)

    # Exhaustion line: ref_ema + ATR × extension_atr_mult
    exhaust_line = ref_ema + atr_s * extension_atr_mult
    trade_fig.add_trace(
        go.Scatter(
            x=trade_slice.index,
            y=exhaust_line,
            mode="lines",
            line=dict(color="#4CAF50", width=1.2, dash="dash"),
            name="Exhaustion",
            showlegend=(t == perf.trades[0]),
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )

    # Climax line: prev_close + climax_atr_mult × ATR
    prev_close = trade_slice["Close"].shift(1)
    climax_line = prev_close + climax_atr_mult * atr_s
    climax_line = climax_line.iloc[1:]  # skip first bar (NaN shift)
    if len(climax_line) > 0:
        trade_fig.add_trace(
            go.Scatter(
                x=climax_line.index,
                y=climax_line,
                mode="lines",
                line=dict(color="#FF9800", width=1.2, dash="dot"),
                name="Climax",
                showlegend=(t == perf.trades[0]),
                hoverinfo="skip",
            ),
            row=1,
            col=1,
        )

# Zoom the chart to the user's date range while keeping the warmup
# bars accessible via scroll/pan (needed for converged 50/200 SMA).
trade_fig.update_xaxes(range=[str(start_date), str(end_date)])
st.plotly_chart(trade_fig, use_container_width=True)

if not perf.trades:
    st.info("이 기간엔 Wedge Pop 신호가 발생하지 않았어.")
    st.stop()

# --- Equity curve ---
if len(result.equity_curve) > 1:
    eq_fig = chart_builder.build_equity_curve(result.equity_curve, title="Equity Curve")
    st.plotly_chart(eq_fig, use_container_width=True)

# --- Trade table with dollar values ---
st.subheader("Trades")
trade_rows = []
for t in perf.trades:
    is_stop = abs(t.exit_price - t.stop_loss) < 1e-6
    outcome = "STOP" if is_stop else ("WIN" if t.pnl > 0 else "LOSS")
    trade_rows.append(
        {
            "Entry Date": t.entry_date,
            "Exit Date": t.exit_date,
            "Outcome": outcome,
            "Entry Price": f"${t.entry_price:,.2f}",
            "Exit Price": f"${t.exit_price:,.2f}",
            "Stop Loss": f"${t.stop_loss:,.2f}",
            "Shares": t.shares,
            "Cost": f"${t.entry_price * t.shares:,.0f}",
            "Proceeds": f"${t.exit_price * t.shares:,.0f}",
            "P&L ($)": f"${t.pnl:,.2f}",
            "P&L (%)": f"{t.pnl_pct:.2%}",
        }
    )
st.dataframe(trade_rows, use_container_width=True)
