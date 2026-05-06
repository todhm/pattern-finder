"""Multi-ticker First Candle Rule scan — universe runner.

Sister page to ``19_First_Candle_Rule.py``. Walks an NY-equity
universe day-by-day and, when multiple tickers fire on the same
session, picks the one whose **psychology-fit score** is highest:

    score = fvg_height_pct × entry_bar_rvol × daily_rvol

Each component proxies one of the three institutional-conviction
signals the FCR pattern leans on:
  - ``fvg_height_pct``   → strength of the displacement (3-bar gap).
  - ``entry_bar_rvol``   → institutional volume on the FVG bar.
  - ``daily_rvol``       → day-level institutional participation.

Default risk = 5%, position = 100% of equity. Same detector
filters as the single-ticker page (sweep-tuned: skip gap-up days
> 1.5%, skip days with daily volume < 0.85× 20-day avg).
"""

from datetime import date, time, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data.adapters.composed_market_data import build_default_market_data
from data.adapters.regular_session_filter import RegularSessionFilterAdapter
from data.adapters.wikipedia_universe import default_universe_provider
from data.adapters.yfinance_adapter import YFinanceAdapter
from data.domain.market_calendar import NY
from pages._shared.wedgepop_results import (
    render_failed_tickers,
    render_headline_metrics,
    render_toss_fee_inputs,
    render_trade_table,
)
from pattern.adapters.first_candle_rule import FirstCandleRuleDetector
from strategy.adapters.multi_first_candle_strategy import (
    MultiFirstCandleRuleStrategy,
)
from strategy.domain.models import MultiStrategyConfig

INTRADAY_HISTORY_FLOOR = date(2022, 1, 1)

st.set_page_config(page_title="Multi First Candle Rule", layout="wide")
st.title("Multi First Candle Rule Scan")
st.caption(
    "NY universe-wide FCR scan. 동일 세션에 여러 종목이 시그널을 만들면 "
    "**FVG height % × entry-bar RVOL × daily RVOL** 점수가 가장 높은 후보를 선택. "
    "한 번에 하루 한 종목만 트레이드. 결과 아래에 트레이드별 1m 차트."
)

with st.sidebar:
    st.header("Universe")
    universe = st.selectbox(
        "Universe",
        options=["nasdaq100", "sp500", "nasdaq_full"],
        index=0,
        help="EODHD가 KR 1m 데이터를 안 줘서 NY 유니버스만 노출.",
    )
    max_tickers = st.number_input(
        "Max tickers (0 = all)",
        value=30,
        min_value=0,
        max_value=2500,
        step=10,
        help="0이면 universe 전체. 처음엔 30~50으로 시작.",
    )
    max_workers = st.number_input(
        "Parallel workers",
        value=4,
        min_value=1,
        max_value=16,
        step=1,
    )

    st.header("Date range")
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
        value=5.0,
        min_value=0.1,
        max_value=100.0,
        step=0.5,
        help="기본 5% — high-conviction picks 한 종목씩만 운영하므로 size가 큼.",
    )
    max_position_pct = st.number_input(
        "Max position (% of equity)",
        value=100.0,
        min_value=1.0,
        max_value=100.0,
        step=5.0,
        help="기본 100% — full-port single-position. 작은 stop이라도 "
        "max_position cap이 binding되면 risk_per_trade가 더 작게 implied.",
    )
    max_below_stop_strikes = st.number_input(
        "Stop strikes (N-touch tolerant stop)",
        value=3,
        min_value=0,
        max_value=10,
        step=1,
        help="Stop 라인 아래로 N번째 침투 시 발화. 0이면 stop 비활성.",
    )
    enable_be_time = st.checkbox(
        "Move stop to break-even after N minutes (Crabel)",
        value=True,
        help="진입 후 N분 경과시 stop을 entry로 끌어올림. "
        "5개월 NASDAQ100 데이터에서 session_close 34건이 모두 -1R 가깝게 "
        "끝나는 패턴 → 일부를 break-even (~0R)로 변환. Crabel 룰: "
        "'이상적 트레이드는 즉시 수익이 보임; 늦게 갈수록 vulnerable'.",
    )
    breakeven_after_min = st.number_input(
        "BE after minutes",
        value=60, min_value=0, max_value=390, step=15,
        disabled=not enable_be_time,
    )
    enable_be_r = st.checkbox(
        "Move stop to break-even after R reached",
        value=False,
        help="미실현 +N×R 도달시 stop을 entry로. 1R touch 후 -R로 회귀하는 "
        "트레이드를 break-even으로 변환. 단 TP 못 닿고 BE에서 청산되는 "
        "winner도 생길 수 있어 trade-off.",
    )
    breakeven_after_r = st.number_input(
        "BE after R-multiple",
        value=1.0, min_value=0.0, max_value=5.0, step=0.1, format="%.1f",
        disabled=not enable_be_r,
    )

    st.header("Universe-quality pre-filter")
    st.caption(
        "**SP500처럼 broad universe**일 때 critical. ATR%와 $-volume이 낮은 "
        "종목은 FCR 패턴이 작동 안 함 (variation 부족 + institutional flow 부재). "
        "Nasdaq100 같은 작은 universe에선 OFF로 둬도 OK."
    )
    enable_atr_filter = st.checkbox(
        "Skip low-volatility tickers (min daily ATR %)",
        value=True,
        help="Daily H-L 범위 / Close가 이 % 미만인 종목 제외. "
        "2R 도달이 어려운 저변동 종목 (utilities, REITs) 자동 거름.",
    )
    min_daily_atr_pct = st.number_input(
        "Min daily ATR (%)",
        value=2.0,
        min_value=0.0, max_value=10.0, step=0.25, format="%.2f",
        disabled=not enable_atr_filter,
        help="2.0% ≈ FCR-favorable mega-cap 평균. 1.5% = 더 관대 (small/mid).",
    )
    enable_dollar_vol_filter = st.checkbox(
        "Skip illiquid tickers (min avg $-volume)",
        value=True,
        help="20-day avg dollar volume 이 X 미만인 종목 제외. "
        "Institutional 참여가 충분한 종목만 추림.",
    )
    min_avg_dollar_volume_m = st.number_input(
        "Min avg $-volume (millions)",
        value=500.0,
        min_value=0.0, max_value=10000.0, step=50.0, format="%.0f",
        disabled=not enable_dollar_vol_filter,
        help="500M = ~mid-cap 컷오프. SP500 mega-cap은 보통 1B~10B+.",
    )

    st.header("Pattern (sweep-tuned defaults)")
    box_minutes = st.number_input(
        "OR length (minutes)", value=5, min_value=1, max_value=30, step=1,
    )
    require_bullish_displacement = st.checkbox(
        "Require bullish displacement bar (FVG-creator)", value=True,
    )
    min_fvg_gap_pct = st.number_input(
        "Min FVG gap (% of price)",
        value=0.03, min_value=0.0, max_value=2.0, step=0.01, format="%.2f",
    )
    max_total_bars = st.number_input(
        "Max bars from range break to entry FVG",
        value=45, min_value=3, max_value=200, step=5,
    )
    require_retest_for_path_b = st.checkbox(
        "Require retest into OR for Path B", value=True,
    )
    target_r_multiple = st.number_input(
        "Take profit (R-multiple)", value=2.0, min_value=0.5,
        max_value=10.0, step=0.5, format="%.1f",
    )
    stop_tick_buffer = st.number_input(
        "Stop tick buffer ($)", value=0.01, min_value=0.0,
        max_value=1.0, step=0.01, format="%.2f",
    )
    latest_entry_hour = st.number_input(
        "Latest entry hour (ET, exclusive)",
        value=11, min_value=10, max_value=16, step=1,
    )

    st.header("Advanced filters (sweep-tuned)")
    enable_gap_filter = st.checkbox(
        "Skip gap-up days", value=True,
        help="Today's open vs prev_close 갭이 X% 초과면 그 세션 스킵.",
    )
    max_gap_up_pct = st.number_input(
        "Max gap-up (%)", value=1.5, min_value=0.0, max_value=10.0,
        step=0.5, format="%.2f", disabled=not enable_gap_filter,
    )
    enable_drvol_filter = st.checkbox(
        "Skip low-volume days", value=True,
        help="Today's volume / 20-day avg < X면 스킵.",
    )
    min_daily_rvol = st.number_input(
        "Min daily RVOL", value=0.85, min_value=0.0, max_value=3.0,
        step=0.05, format="%.2f", disabled=not enable_drvol_filter,
    )
    enable_sma_filter = st.checkbox(
        "Require above daily SMA(N)", value=False,
    )
    sma_period = st.selectbox(
        "SMA period", options=[50, 200], index=0,
        disabled=not enable_sma_filter,
    )
    enable_brvol_filter = st.checkbox(
        "Require FVG-bar RVOL", value=False,
    )
    min_entry_bar_rvol = st.number_input(
        "Min FVG-bar RVOL", value=1.0, min_value=0.0, max_value=5.0,
        step=0.1, format="%.2f", disabled=not enable_brvol_filter,
    )

    fee_schedule = render_toss_fee_inputs(key_prefix="multi_fcr_")

    run_btn = st.button(
        "Run Multi First Candle Rule Scan",
        type="primary",
        use_container_width=True,
    )

if not run_btn:
    st.info(
        "좌측에서 universe / 기간 / 파라미터를 설정하고 "
        "**Run Multi First Candle Rule Scan**을 눌러. "
        "기본값 = sweep-tuned (NVDA Jan-May 백테스트 기준 NVDA +0.94pp, TSLA +0.43pp)."
    )
    st.stop()

# Composition root (NY-only — EODHD has no KR 1m).
_base_md = build_default_market_data()
md = RegularSessionFilterAdapter(_base_md, market=NY)
yf = YFinanceAdapter()

detector = FirstCandleRuleDetector(
    box_minutes=int(box_minutes),
    session_open_local=NY.rth_open,
    latest_entry_local=time(int(latest_entry_hour), 0),
    require_daily_bullish=False,
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

runner = MultiFirstCandleRuleStrategy(
    market_data=md,
    daily_market_data=yf,
    universe_provider=default_universe_provider(),
    detector=detector,
    market=NY,
    max_workers=int(max_workers),
    interval="1m",
    max_position_pct_of_equity=float(max_position_pct) / 100.0,
    max_below_stop_strikes=int(max_below_stop_strikes),
    breakeven_after_minutes=(
        int(breakeven_after_min) if enable_be_time else 0
    ),
    breakeven_after_r_multiple=(
        float(breakeven_after_r) if enable_be_r else 0.0
    ),
    min_daily_atr_pct=(
        float(min_daily_atr_pct) / 100.0 if enable_atr_filter else None
    ),
    min_avg_dollar_volume=(
        float(min_avg_dollar_volume_m) * 1_000_000
        if enable_dollar_vol_filter else None
    ),
)

config = MultiStrategyConfig(
    universe=universe,
    start_date=start_date,
    end_date=end_date,
    pattern_name="first_candle_rule",
    initial_capital=float(initial_capital),
    risk_per_trade=risk_pct / 100.0,
    max_holding_days=1,
    max_tickers=(int(max_tickers) if max_tickers > 0 else None),
    fee_schedule=fee_schedule,
)

with st.spinner(
    f"Scanning {universe} ({max_tickers or 'all'} tickers)..."
):
    try:
        result = runner.run(config)
    except Exception as exc:
        st.error(f"Scan failed: {exc}")
        st.stop()

# ---- Headline metrics ---------------------------------------------
render_headline_metrics(result, universe_label=universe)

# Surface the underlying fetch error when most tickers fail —
# without this, the failed-tickers expander looks like a strategy
# problem when it's actually data fetch (e.g., EODHD quota).
if (
    result.tickers_scanned > 0
    and len(result.failed_tickers) >= result.tickers_scanned * 0.5
):
    last_err = getattr(runner, "_last_fetch_error", None)
    if last_err:
        st.error(
            f"**대부분의 ticker fetch 실패** ({len(result.failed_tickers)}/"
            f"{result.tickers_scanned}). 가장 최근 에러:\n\n```\n{last_err}\n```\n\n"
            "EODHD daily quota 초과면 24h 후 reset됨. "
            "Plan upgrade 또는 작은 universe로 재시도."
        )

render_failed_tickers(result)

if not result.trades:
    st.info("이 기간엔 어떤 종목도 FCR signal이 트리거되지 않았어.")
    st.stop()

st.subheader("Equity Curve")
eq_df = pd.DataFrame({
    "date": [p.date for p in result.equity_curve],
    "equity": [p.equity for p in result.equity_curve],
})
eq_fig = go.Figure(go.Scatter(
    x=eq_df["date"], y=eq_df["equity"],
    mode="lines", line=dict(width=2),
))
eq_fig.update_layout(
    height=320,
    margin=dict(l=10, r=10, t=10, b=10),
    yaxis_title="Equity ($)",
)
st.plotly_chart(eq_fig, use_container_width=True)

st.subheader("Trades")
render_trade_table(result, market_tz=NY.tz)
st.caption(
    "**Buy/Sell ratio** 컬럼은 이 페이지에서 의미가 다름 — FCR 픽 메트릭으로 "
    "재활용해서 fvg_pct × entry_bar_rvol × daily_rvol 점수가 표시됨. "
    "동일 세션에 여러 후보가 있을 때 이 점수가 높은 종목이 선택됨. "
    "Buy_volume = fvg_pct, Sell_volume = daily_rvol, Sig_volume = entry_bar_rvol."
)

# ---- Per-trade 1m charts ------------------------------------------
st.subheader("Per-trade 1m charts")
st.caption(
    "각 트레이드별 1분봉 + OR(주황) + FVG zone(보라) + entry/stop/TP 마커. "
    "Path A/B와 retest 위치가 마커로 표시됨."
)

_daily_cache: dict[str, pd.DataFrame] = {}

def _daily_for(ticker: str) -> pd.DataFrame:
    if ticker not in _daily_cache:
        _daily_cache[ticker] = yf.fetch_ohlcv(
            ticker, start_date - timedelta(days=400), end_date
        )
    return _daily_cache[ticker]


for i, t in enumerate(result.trades):
    label = (
        f"{t.ticker} — {t.entry_date} {t.exit_reason}  "
        f"PnL ${t.pnl:+,.0f} ({t.pnl_pct:+.2%})  score={t.signal_buy_sell_ratio:.5f}"
    )
    with st.expander(label, expanded=(i < 3)):
        try:
            df_1m = md.fetch_ohlcv(
                t.ticker,
                t.entry_date - timedelta(days=1),
                t.entry_date + timedelta(days=1),
                interval="1m",
            )
        except Exception as exc:
            st.warning(f"Couldn't fetch {t.ticker} 1m for chart: {exc}")
            continue
        if df_1m is None or df_1m.empty:
            st.warning(f"No 1m bars for {t.ticker} {t.entry_date}.")
            continue
        if df_1m.index.tz is not None and str(df_1m.index.tz) != NY.tz:
            df_1m = df_1m.copy()
            df_1m.index = df_1m.index.tz_convert(NY.tz)
        sess_df = df_1m[df_1m.index.date == t.entry_date]
        if sess_df.empty:
            st.warning(f"No bars on {t.entry_date} for {t.ticker}.")
            continue

        df_d = _daily_for(t.ticker)
        all_signals = detector.detect(df_1m, df_d)
        sig = next(
            (s for s in all_signals if s.session_date == t.entry_date),
            None,
        )

        fig = go.Figure(go.Candlestick(
            x=sess_df.index,
            open=sess_df["Open"], high=sess_df["High"],
            low=sess_df["Low"], close=sess_df["Close"],
            name="1m", showlegend=False,
        ))
        end_ts = sess_df.index[-1]
        if sig is not None:
            # OR rectangle
            fig.add_shape(
                type="rect",
                x0=sig.box_open_ts, x1=end_ts,
                y0=sig.box_low, y1=sig.box_high,
                line=dict(color="#FB8C00", width=1.4),
                fillcolor="rgba(251,140,0,0.15)",
                layer="below",
            )
            # FVG zone (full 3-bar window)
            fvg_height = sig.fvg_high - sig.fvg_low
            fig.add_shape(
                type="rect",
                x0=sig.fvg_pre_ts, x1=sig.entry_ts,
                y0=sig.fvg_low, y1=sig.fvg_high,
                line=dict(color="#7B1FA2", width=2),
                fillcolor="rgba(186,104,200,0.30)",
                layer="below",
            )
            fig.add_annotation(
                x=sig.fvg_pre_ts, y=sig.fvg_high,
                text=(
                    f"FVG [{sig.fvg_low:.2f}, {sig.fvg_high:.2f}] "
                    f"(Δ ${fvg_height:.3f})  Path-{sig.path}"
                ),
                showarrow=False, xanchor="left", yanchor="bottom",
                font=dict(color="#4A148C", size=10),
                bgcolor="rgba(255,255,255,0.75)",
            )
            # Retest marker (Path B only)
            if sig.retest_ts is not None and sig.retest_low is not None:
                fig.add_trace(go.Scatter(
                    x=[sig.retest_ts], y=[sig.retest_low],
                    mode="markers",
                    marker=dict(symbol="triangle-down", color="#1976D2", size=11),
                    name="Retest", showlegend=False,
                    hovertemplate=f"Retest @ ${sig.retest_low:.2f}<extra></extra>",
                ))

        # Entry / stop / TP
        entry_ts = pd.Timestamp(t.entry_ts)
        if entry_ts.tz is None and sess_df.index.tz is not None:
            entry_ts = entry_ts.tz_localize(sess_df.index.tz)
        tp_price = (
            sig.take_profit
            if sig is not None
            else t.entry_price + float(target_r_multiple) * (
                t.entry_price - t.stop_loss
            )
        )
        fig.add_shape(
            type="line",
            x0=entry_ts, x1=end_ts,
            y0=t.stop_loss, y1=t.stop_loss,
            line=dict(color="#C62828", width=1.4, dash="dash"),
        )
        fig.add_shape(
            type="line",
            x0=entry_ts, x1=end_ts,
            y0=tp_price, y1=tp_price,
            line=dict(color="#2E7D32", width=1.4, dash="dash"),
        )
        fig.add_trace(go.Scatter(
            x=[entry_ts], y=[t.entry_price],
            mode="markers",
            marker=dict(
                symbol="arrow-up", color="#2E7D32", size=14,
                line=dict(color="#1B5E20", width=1.5),
            ),
            name="Entry", showlegend=False,
            hovertemplate=f"Entry @ ${t.entry_price:.2f}<extra></extra>",
        ))
        # Exit
        exit_ts = pd.Timestamp(t.exit_ts)
        if exit_ts.tz is None and sess_df.index.tz is not None:
            exit_ts = exit_ts.tz_localize(sess_df.index.tz)
        exit_color = "#2E7D32" if t.pnl > 0 else "#C62828"
        if t.exit_reason == "session_close":
            symbol = "diamond-open"
        elif t.exit_reason == "stop_loss":
            symbol = "x"
        elif t.exit_reason == "breakeven_stop":
            symbol = "square-open"
        else:
            symbol = "circle"
        fig.add_trace(go.Scatter(
            x=[exit_ts], y=[t.exit_price],
            mode="markers",
            marker=dict(symbol=symbol, color=exit_color, size=12),
            name=f"Exit ({t.exit_reason})", showlegend=False,
            hovertemplate=(
                f"{t.exit_reason} @ ${t.exit_price:.2f} "
                f"(PnL ${t.pnl:+,.2f})<extra></extra>"
            ),
        ))
        fig.update_layout(
            height=400,
            xaxis_rangeslider_visible=False,
            xaxis_rangebreaks=[
                dict(bounds=[16, 9.5], pattern="hour"),
                dict(bounds=["sat", "mon"]),
            ],
            margin=dict(l=10, r=10, t=10, b=10),
            title=f"{t.ticker} — {t.entry_date} (1m)",
        )
        st.plotly_chart(fig, use_container_width=True)
