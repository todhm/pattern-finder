from datetime import date, timedelta

import streamlit as st

from data.adapters.wikipedia_universe import WikipediaUniverseAdapter
from data.adapters.yfinance_adapter import YFinanceAdapter
from pattern.adapters.wedge_pop import WedgePopDetector
from strategy.adapters.multi_wedgepop_strategy import MultiWedgepopStrategy
from strategy.adapters.wedgepop_strategy import WedgepopStrategy
from strategy.domain.models import MultiStrategyConfig, TossFeeSchedule
from visualization.adapters.plotly_charts import PlotlyChartBuilder

st.set_page_config(page_title="Multi Wedge Pop Scan", layout="wide")
st.title("Multi-Ticker Wedge Pop Strategy")
st.caption(
    "S&P 500 / Nasdaq-100 전체에서 매일 발생한 Wedge Pop signal 중 거래량이 "
    "가장 큰 종목 하나를 매수. 한 포지션이 청산될 때까지 다른 종목은 사지 않음."
)

# --- Sidebar inputs ---
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
        help="스캔할 최대 종목 수. 0이면 전체. 처음엔 작게(20~50) 시작해서 "
        "동작을 확인한 뒤 늘려. yfinance 호출이 종목당 1회 발생.",
    )
    max_workers = st.number_input(
        "Parallel workers",
        value=8,
        min_value=1,
        max_value=32,
        step=1,
        help="yfinance 호출을 동시에 몇 개 보낼지. 너무 크면 rate-limit.",
    )

    st.header("Period")
    start_date = st.date_input("Start Date", value=date.today() - timedelta(days=365))
    end_date = st.date_input("End Date", value=date.today())

    st.header("Risk")
    initial_capital = st.number_input(
        "Initial Capital ($)",
        value=100_000,
        min_value=1_000,
        step=10_000,
        help="포트폴리오 시작 자본. 한 번에 한 종목만 보유하므로 매수 시 "
        "전체 자본 기준으로 risk_per_trade × capital 로 사이즈 계산.",
    )
    risk_pct = st.number_input(
        "Risk per Trade (%)",
        value=2.0,
        min_value=0.1,
        max_value=100.0,
        step=0.5,
    )
    max_holding_days = st.number_input("Max Holding Days", value=60, min_value=1, max_value=2_000, step=5)

    st.header("Pattern Detection")
    consolidation_pct = st.number_input(
        "Min consolidation %",
        value=60.0,
        min_value=0.0,
        max_value=100.0,
        step=5.0,
        help="직전 N일 중 close가 fast EMA 아래에 있어야 하는 " "**최소** 비율.",
    )
    enable_max_cp = st.checkbox(
        "Cap max consolidation %",
        value=False,
        help="상한 추가 — 너무 깊은 consolidation (95%+)을 제외하려면.",
    )
    max_consolidation_pct_ui = st.number_input(
        "Max consolidation %",
        value=95.0,
        min_value=0.0,
        max_value=100.0,
        step=5.0,
        disabled=not enable_max_cp,
    )
    breakout_pct = st.number_input(
        "Min breakout strength %",
        value=1.5,
        min_value=0.0,
        max_value=20.0,
        step=0.1,
        format="%.2f",
        help="max(ema_strength, daily_move) 의 **하한**.",
    )
    enable_max_bp = st.checkbox(
        "Cap max breakout strength",
        value=False,
        help="상한 추가 — 이미 너무 큰 단일봉 gap을 overextended로 " "보고 제외하고 싶을 때.",
    )
    max_breakout_pct_ui = st.number_input(
        "Max breakout strength %",
        value=10.0,
        min_value=0.0,
        max_value=50.0,
        step=0.5,
        format="%.2f",
        disabled=not enable_max_bp,
    )
    require_above_long_smas = st.checkbox(
        "Require close above 50 & 200 SMA",
        value=True,
        help="signal 캔들의 close가 50 SMA와 200 SMA 둘 다 위에 있을 "
        "때만 wedge pop으로 인정. 장기 추세 안에 있는 wedge pop만 "
        "필터링하고 싶을 때 켜세요.",
    )
    detect_lookback = st.number_input(
        "Consolidation lookback (days)",
        value=10,
        min_value=3,
        max_value=60,
        step=1,
        help="Consolidation 체크 / consolidation-low stop 계산 기간.",
    )
    cooldown_bars_ui = st.number_input(
        "Cooldown bars (after a signal)",
        value=15,
        min_value=0,
        max_value=60,
        step=1,
        help="signal fire 후 건너뛰는 바 수. 0 = 연속 signal 허용. "
        "예전엔 lookback을 그대로 썼는데 이제 독립 파라미터.",
    )
    ema_fast = st.number_input(
        "Fast EMA period",
        value=10,
        min_value=2,
        max_value=100,
        step=1,
        help="Detector와 strategy가 공통으로 쓰는 fast EMA.",
    )
    ema_slow = st.number_input(
        "Slow EMA period",
        value=20,
        min_value=2,
        max_value=200,
        step=1,
        help="Detector breakout 및 strategy exhaustion reference = " "max(fast, slow) 계산에 쓰임.",
    )
    slope_lookback = st.number_input(
        "Slope lookback (days)",
        value=20,
        min_value=5,
        max_value=120,
        step=1,
        help="ema_fast_slope / ema_slow_slope 측정 기간. signal metadata "
        "에 기록되어 downstream slope 필터의 입력이 됨.",
    )

    st.header("Entry Filter")
    require_gap_up = st.checkbox(
        "Require gap-up confirmation",
        value=False,
        help="다음날 시초가가 breakout 종가보다 높을 때만 진입.",
    )
    enable_chase_filter = st.checkbox(
        "Enable entry chase filter",
        value=True,
        help="Entry open이 signal bar high 위로 너무 멀리 gap-up하면 "
        "'chasing' 거부. 끄면 gap-up 제한 없음.",
    )
    max_entry_chase_ratio = st.number_input(
        "Max entry chase (× signal range)",
        value=0.15,
        min_value=0.0,
        max_value=5.0,
        step=0.05,
        format="%.2f",
        disabled=not enable_chase_filter,
        help="Entry open이 signal bar high 위로 signal range의 몇 배 "
        "초과하면 거부. 주의: 0 = 모든 gap-up 차단, 5 = off.",
    )
    enable_entry_ema_filter = st.checkbox(
        "Enable EMA-extension entry filter",
        value=False,
        help="Entry open이 signal bar의 max(10EMA, 20EMA) 위로 N% "
        "이상 초과하면 진입 거부. chase 필터보다 EMA stack에 anchored.",
    )
    max_entry_ema_extension_pct = st.number_input(
        "Max entry extension above EMA %",
        value=3.0,
        min_value=0.0,
        max_value=50.0,
        step=0.1,
        format="%.2f",
        disabled=not enable_entry_ema_filter,
        help="(entry_open - max(ema10, ema20)) / ema > 이 값 이면 거부.",
    )
    st.caption("**EMA slow slope 범위 필터**")
    enable_min_slope = st.checkbox(
        "Enforce min EMA slow slope",
        value=True,
        help="ema_slow_slope의 **하한**. 양수로 설정하면 '반드시 +N% "
        "이상 상승 추세'만 통과 (예: 0.05). 음수면 dead-cat bounce "
        "거부 (예: -0.01).",
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
        help="ema_slow_slope의 **상한**. parabolic하게 오른 종목을 " "제외하려면 켜세요 (예: 0.30).",
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
    extension_pct = st.number_input(
        "Exhaustion % above EMA",
        value=15.0,
        min_value=0.0,
        max_value=500.0,
        step=0.5,
        help="Exit 룰: bar의 **High**가 max(fast,slow) EMA × (1+이값) "
        "선을 터치하면 해당 라인에서 체결 (limit order 모델, gap-up이면 "
        "open). 0%면 exit 룰이 사실상 꺼짐.",
    )
    extension_atr_mult = st.number_input(
        "Exhaustion ATR multiplier",
        value=2.5,
        min_value=0.1,
        max_value=50.0,
        step=0.1,
        help="대안 익절: close − EMA ≥ ATR × 이 값.",
    )
    climax_atr_mult = st.number_input(
        "Climax bar ATR multiplier",
        value=1.5,
        min_value=0.1,
        max_value=10.0,
        step=0.1,
        help="단일봉 blow-off 감지. range 및 단일봉 move가 ATR × 이 값 "
        "이상이면서 close가 상단 20%이면 익절 (AMD 2019-03-19).",
    )
    atr_period = st.number_input(
        "ATR period",
        value=14,
        min_value=2,
        max_value=100,
        step=1,
        help="ATR(N) 윈도우.",
    )
    trail_after_profit = st.checkbox(
        "Arm EMA trail only after profit",
        value=True,
        help="꺼두면 첫 bar부터 EMA trail 발동 — 권장 안 함.",
    )
    arm_breakeven = st.checkbox(
        "Ratchet stop to breakeven after profit",
        value=True,
        help="수익권 진입 시 stop을 entry price까지 올림 (CDNS 2026-02).",
    )

    st.header("Fees (Toss Securities)")
    st.caption("토스증권 미국주식 기본 수수료. 매수/매도 각 0.1% + SEC fee 0.00229% (매도시).")
    buy_fee_pct = st.number_input(
        "Buy commission (%)",
        value=0.10,
        min_value=0.0,
        max_value=5.0,
        step=0.01,
        format="%.4f",
    )
    sell_fee_pct = st.number_input(
        "Sell commission (%)",
        value=0.10,
        min_value=0.0,
        max_value=5.0,
        step=0.01,
        format="%.4f",
    )
    sec_fee_pct = st.number_input(
        "SEC fee (%) — sell only",
        value=0.00229,
        min_value=0.0,
        max_value=1.0,
        step=0.0001,
        format="%.5f",
        help="미국 SEC Section 31 규정 수수료. 매도 거래대금에 부과.",
    )

    run_btn = st.button("Run Universe Scan", type="primary", use_container_width=True)


# --- Main ---
if not run_btn:
    st.info(
        "좌측에서 universe / 기간을 설정하고 **Run Universe Scan** 을 눌러줘. "
        "처음 실행 땐 max_tickers를 작게 잡고 스모크 테스트해보는 게 안전해."
    )
    st.stop()

market_data = YFinanceAdapter()
universe_provider = WikipediaUniverseAdapter()
detector = WedgePopDetector(
    lookback=int(detect_lookback),
    ema_fast=int(ema_fast),
    ema_slow=int(ema_slow),
    consolidation_pct=consolidation_pct / 100.0,
    max_consolidation_pct=(max_consolidation_pct_ui / 100.0 if enable_max_cp else None),
    breakout_pct=breakout_pct / 100.0,
    max_breakout_pct=(max_breakout_pct_ui / 100.0 if enable_max_bp else None),
    slope_lookback=int(slope_lookback),
    cooldown_bars=int(cooldown_bars_ui),
    require_above_long_smas=require_above_long_smas,
)
per_ticker_strategy = WedgepopStrategy(
    market_data=market_data,
    detector=detector,
    ema_trail=int(ema_fast),
    ema_slow=int(ema_slow),
    atr_period=int(atr_period),
    extension_pct=extension_pct / 100.0,
    extension_atr_mult=extension_atr_mult,
    climax_atr_mult=climax_atr_mult,
    max_entry_chase_ratio=(max_entry_chase_ratio if enable_chase_filter else float("inf")),
    max_entry_ema_extension_pct=(max_entry_ema_extension_pct / 100.0 if enable_entry_ema_filter else None),
    max_ema_slope_decline=None,
    min_ema_slow_slope=(min_ema_slow_slope_ui if enable_min_slope else None),
    max_ema_slow_slope=(max_ema_slow_slope_ui if enable_max_slope else None),
    trail_after_profit=trail_after_profit,
    arm_breakeven_after_profit=arm_breakeven,
    require_gap_up=require_gap_up,
)
runner = MultiWedgepopStrategy(
    market_data=market_data,
    universe_provider=universe_provider,
    detector=detector,
    strategy=per_ticker_strategy,
    max_workers=int(max_workers),
)

fee_schedule = TossFeeSchedule(
    buy_commission_pct=buy_fee_pct / 100.0,
    sell_commission_pct=sell_fee_pct / 100.0,
    sec_fee_pct=sec_fee_pct / 100.0,
)

config = MultiStrategyConfig(
    universe=universe,
    start_date=start_date,
    end_date=end_date,
    pattern_name="wedge_pop",
    initial_capital=float(initial_capital),
    risk_per_trade=risk_pct / 100.0,
    max_holding_days=int(max_holding_days),
    max_tickers=int(max_tickers) if max_tickers > 0 else None,
    fee_schedule=fee_schedule,
)

with st.spinner(f"Scanning {universe} ({config.max_tickers or 'all'} tickers)..."):
    try:
        result = runner.run(config)
    except Exception as e:
        st.error(f"Universe scan failed: {e}")
        st.stop()

# --- Headline metrics ---
st.subheader(f"Universe — {universe.upper()}")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Tickers Scanned", result.tickers_scanned)
m2.metric("Total Signals", result.total_signals)
m3.metric("Trades Taken", result.trades_taken)
m4.metric("Failed Tickers", len(result.failed_tickers))

m5, m6, m7, m8 = st.columns(4)
m5.metric("Total Return (net)", f"{result.total_return_pct:.2%}")
m6.metric("Win Rate", f"{result.win_rate:.0%}" if result.trades_taken else "—")
m7.metric("Final Capital", f"${result.final_capital:,.0f}")
m8.metric("Max Drawdown", f"{result.max_drawdown_pct:.2%}")

m9, m10, m11, m12 = st.columns(4)
gross_pnl = sum(t.gross_pnl for t in result.trades)
m9.metric("Total Commission", f"${result.total_commission:,.2f}")
m10.metric("Gross P&L", f"${gross_pnl:,.2f}")
m11.metric("Net P&L", f"${gross_pnl - result.total_commission:,.2f}")
m12.metric(
    "Fees as % of Gross",
    f"{(result.total_commission / gross_pnl):.2%}" if gross_pnl else "—",
)

if result.tickers_scanned == 0:
    st.warning("Universe returned 0 tickers — check your selection.")
    st.stop()

if not result.trades:
    st.info(
        "이 기간엔 universe 전체에서 Wedge Pop signal이 잡히지 않았거나, "
        "잡혔어도 모두 진입 조건에서 걸렸어. 기간을 늘려보거나 universe를 "
        "넓혀봐."
    )
    st.stop()

chart_builder = PlotlyChartBuilder()

# --- Equity curve ---
st.subheader("Portfolio Equity Curve")
eq_fig = chart_builder.build_equity_curve(
    result.equity_curve,
    title=f"Single-position portfolio across {universe.upper()}",
)
st.plotly_chart(eq_fig, use_container_width=True)

# --- Best individual trades ---
st.subheader("Best Trades")
trades_fig = chart_builder.build_top_trades_bar(
    result.trades,
    title="Top trades by % return",
)
st.plotly_chart(trades_fig, use_container_width=True)

# --- Per-ticker contribution ---
st.subheader("P&L by Ticker")
contrib_fig = chart_builder.build_ticker_contribution_bar(
    result.trades,
    title="Total P&L contribution per ticker",
)
st.plotly_chart(contrib_fig, use_container_width=True)

# --- Trade table ---
st.subheader("Trades — Details")
rows = [
    {
        "Ticker": t.ticker,
        "Entry Date": t.entry_date,
        "Exit Date": t.exit_date,
        "Entry": f"${t.entry_price:,.2f}",
        "Exit": f"${t.exit_price:,.2f}",
        "Stop": f"${t.stop_loss:,.2f}",
        "Shares": t.shares,
        "Sig Vol": f"{t.signal_volume:,.0f}",
        "Buy Vol": f"{t.signal_buy_volume:,.0f}",
        "Sell Vol": f"{t.signal_sell_volume:,.0f}",
        "Buy/Sell": f"{t.signal_buy_sell_ratio:.2f}",
        "Gross P&L": f"${t.gross_pnl:,.2f}",
        "Commission": f"${t.commission:,.2f}",
        "Net P&L ($)": f"${t.pnl:,.2f}",
        "Net P&L (%)": f"{t.pnl_pct:.2%}",
    }
    for t in result.trades
]
st.dataframe(rows, use_container_width=True)

if result.failed_tickers:
    with st.expander(f"Failed tickers ({len(result.failed_tickers)})"):
        st.write(", ".join(result.failed_tickers))
