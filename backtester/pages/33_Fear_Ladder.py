from datetime import date, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data.adapters.cached_market_data import CachedMarketDataAdapter
from data.adapters.yfinance_adapter import YFinanceAdapter
from pages._shared.formatting import fmt_price
from strategy.adapters.band_rebalance_strategy import BandRebalanceStrategy
from strategy.adapters.fear_ladder_strategy import FearLadderStrategy
from strategy.adapters.risk_metrics import compute_risk_metrics
from strategy.domain.models import (
    BandRebalanceConfig,
    FearLadderConfig,
    TossFeeSchedule,
)

st.set_page_config(page_title="공포 사다리", layout="wide")
st.title("공포 사다리 — '위기에 사는' 역발상 전략")
st.caption(
    "거시 레짐 필터(위기 회피)의 **정반대** 접근. 버핏('남들이 두려워할 "
    "때 탐욕을')·하워드 막스(사이클 극단 포지셔닝)의 방법론: 평시엔 "
    "주식 X% + 현금 예비대(P2P 이자 운용 가능)를 유지하다가, **낙폭이 "
    "사다리 단계(-20/-35/-50%)에 닿을 때마다 현금을 분할 투입**한다 — "
    "깊을수록 더 산다. 신고점 회복 시 평시 비중으로 익절해 실탄을 "
    "재적립. 기준선을 200일선으로 바꾸면 '200MA 아래 = 기회'를 그대로 "
    "구현한다. VIX 공포 확인 옵션, 토스 수수료 + 양도세 22% 반영."
)
st.caption(
    "가격은 **분할·배당 조정가** 기준 — 실제 당시 호가와 다르다. "
    "규칙이 전부 %기반이라 수익률 결과는 동일. $10 미만은 소수 4자리 표시."
)

TICKERS = ["TQQQ", "SOXL", "UPRO", "QLD", "QQQ", "SPY"]

with st.sidebar:
    st.header("종목 / 기간")
    ticker_choice = st.selectbox("종목", [*TICKERS, "직접 입력"], index=0)
    if ticker_choice == "직접 입력":
        ticker = st.text_input("티커", value="TQQQ").strip().upper()
    else:
        ticker = ticker_choice
    start_date = st.date_input(
        "Start Date", value=date(2010, 2, 11),
        min_value=date(2000, 1, 1), max_value=date.today(),
    )
    end_date = st.date_input(
        "End Date", value=date.today(),
        min_value=date(2000, 1, 1), max_value=date.today(),
    )
    initial_capital = st.number_input(
        "Initial Capital", value=100_000_000, min_value=1_000, step=10_000_000
    )

    st.header("사다리 설계")
    base_weight = st.number_input(
        "평시 주식 비중 (%)", value=50.0, min_value=0.0, max_value=90.0, step=5.0,
        help="나머지는 현금 예비대 — 폭락 때 쓸 실탄.",
    )
    cash_rate = st.number_input(
        "예비대 연이율 (%) — P2P 운용", value=9.0, min_value=0.0,
        max_value=30.0, step=1.0,
        help="0이면 무수익 현금. 즉시 인출 가정(낙관적 상한).",
    )
    trigger_mode = st.selectbox(
        "기준선",
        ["고점 드로다운 (ATH)", "200일선 이탈 (MA)"],
        index=0,
        help="ATH: 역대 고점 대비 낙폭. MA: 200일선 대비 이탈 깊이 — "
        "'200MA 아래는 기회' 버전. MA 모드의 회복은 200일선 위 N일 연속.",
    )
    c1, c2, c3 = st.columns(3)
    lv1 = c1.number_input("1단 (−%)", value=20.0, min_value=1.0, max_value=90.0, step=1.0)
    lv2 = c2.number_input("2단 (−%)", value=35.0, min_value=1.0, max_value=95.0, step=1.0)
    lv3 = c3.number_input("3단 (−%)", value=50.0, min_value=1.0, max_value=99.0, step=1.0)
    f1 = c1.number_input("1단 투입(현금%)", value=33.0, min_value=1.0, max_value=100.0, step=1.0)
    f2 = c2.number_input("2단 투입(현금%)", value=50.0, min_value=1.0, max_value=100.0, step=1.0)
    f3 = c3.number_input("3단 투입(현금%)", value=100.0, min_value=1.0, max_value=100.0, step=1.0)
    recovery_days = st.number_input(
        "MA 회복 확인 일수", value=10, min_value=1, max_value=60, step=1,
        help="MA 모드 전용 — 200일선 위를 N일 연속 유지해야 익절 리밸런싱.",
    )
    vix_confirm = st.checkbox(
        "VIX 공포 확인", value=False,
        help="켜면 VIX 21일 평균이 임계 이상일 때만 사다리 발동 — "
        "'진짜 공포장'에서만 산다.",
    )
    vix_threshold = st.number_input(
        "VIX 임계", value=30.0, min_value=15.0, max_value=60.0, step=1.0,
        disabled=not vix_confirm,
    )

    st.header("비교")
    compare_regime = st.checkbox(
        "200일선 레짐 필터(공포 회피)와 비교", value=True,
        help="같은 종목 + VOO 밴드 리밸런싱 + 200일선 레짐(위기 회피)을 "
        "동일 수수료·세금으로 함께 돌려, '공포에 사기 vs 공포 피하기'를 "
        "직접 비교한다.",
    )

    st.header("수수료 / 세금")
    commission_pct = st.number_input(
        "거래수수료 (%, 매수·매도 각각)",
        value=0.10, min_value=0.0, max_value=1.0, step=0.01, format="%.2f",
    )
    tax_pct = st.number_input(
        "양도소득세율 (%)", value=22.0, min_value=0.0, max_value=50.0, step=1.0
    )

    run_btn = st.button("Run Backtest", type="primary", use_container_width=True)

if run_btn:
    market_data = CachedMarketDataAdapter(YFinanceAdapter())
    with st.spinner(f"Fetching {ticker}..."):
        try:
            close = market_data.fetch_ohlcv(ticker, start_date, end_date)["Close"]
        except Exception as e:
            st.error(f"데이터 fetch 실패: {e}")
            st.stop()
    if close.index.tz is not None:
        close.index = close.index.tz_localize(None)
    close.index = close.index.normalize()

    vix = None
    if vix_confirm:
        with st.spinner("Fetching ^VIX..."):
            try:
                vix = market_data.fetch_ohlcv(
                    "^VIX", start_date - timedelta(days=60), end_date
                )["Close"]
                if vix.index.tz is not None:
                    vix.index = vix.index.tz_localize(None)
                vix.index = vix.index.normalize()
            except Exception as e:
                st.warning(f"VIX fetch 실패 — 공포 확인 없이 진행: {e}")

    config = FearLadderConfig(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        initial_capital=float(initial_capital),
        base_stock_weight=base_weight / 100.0,
        cash_annual_rate=cash_rate / 100.0,
        trigger_mode="ma" if trigger_mode.startswith("200") else "ath",
        levels=sorted([-lv1 / 100.0, -lv2 / 100.0, -lv3 / 100.0], reverse=True),
        deploy_fractions=[f1 / 100.0, f2 / 100.0, f3 / 100.0],
        recovery_confirm_days=int(recovery_days),
        vix_confirm=vix_confirm and vix is not None,
        vix_threshold=float(vix_threshold),
        fee_schedule=TossFeeSchedule(
            buy_commission_pct=commission_pct / 100.0,
            sell_commission_pct=commission_pct / 100.0,
        ),
        capital_gains_tax_pct=tax_pct / 100.0,
    )
    with st.spinner("Running backtest..."):
        try:
            result = FearLadderStrategy().execute(close, config, vix=vix)
        except Exception as e:
            st.error(f"Backtest failed: {e}")
            st.stop()

    regime_result = None
    if compare_regime:
        with st.spinner("레짐 필터(공포 회피) 비교 실행..."):
            try:
                voo_start = max(start_date, date(2010, 9, 9))
                voo = market_data.fetch_ohlcv("VOO", voo_start, end_date)["Close"]
                qqq = market_data.fetch_ohlcv(
                    "QQQ", voo_start - timedelta(days=400), end_date
                )["Close"]
                for s_ in (voo, qqq):
                    if s_.index.tz is not None:
                        s_.index = s_.index.tz_localize(None)
                voo.index = voo.index.normalize()
                qqq.index = qqq.index.normalize()
                bcfg = BandRebalanceConfig(
                    aggressive_ticker=ticker,
                    start_date=voo_start,
                    end_date=end_date,
                    initial_capital=float(initial_capital),
                    fee_schedule=config.fee_schedule,
                    capital_gains_tax_pct=config.capital_gains_tax_pct,
                    tax_deduction=config.tax_deduction,
                )
                regime_result = BandRebalanceStrategy().execute(
                    close[close.index >= pd.Timestamp(voo_start)],
                    voo, bcfg, regime_close=qqq,
                )
            except Exception as e:
                st.warning(f"레짐 비교 실패 (본 전략 결과는 유효): {e}")

    st.session_state["fear_state"] = {
        "result": result, "close": close, "regime": regime_result,
    }

_state = st.session_state.get("fear_state")
if _state is None:
    st.info("사다리를 설계하고 **Run Backtest**를 눌러줘.")
    st.stop()
result = _state["result"]
close = _state["close"]
regime_result = _state["regime"]
config = result.config
ticker = config.ticker
initial_capital = config.initial_capital

s = result.summary
liq = result.liquidation
recovered = [c for c in result.cycles if c.outcome == "recovered"]

st.subheader(s.name)
m1, m2, m3, m4 = st.columns(4)
m1.metric("최종 평가액 (세전)", f"{liq.final_value_pre_tax:,.0f}")
m2.metric("세후 청산 가치", f"{liq.final_value_after_tax:,.0f}")
m3.metric("CAGR (세전)", f"{s.cagr_pct:+.2%}")
m4.metric("최대 낙폭 (MDD)", f"{s.max_drawdown_pct:.1%}")

m5, m6, m7, m8 = st.columns(4)
m5.metric(
    "공포 사이클",
    f"{len(recovered)}회 회수 / 총 {len(result.cycles)}회",
    help="사다리 발동 후 회복까지 완주한 사이클 / 전체.",
)
fear_buys = [e for e in result.events if e.kind == "fear_buy"]
m6.metric("사다리 발동", f"{len(fear_buys)}회")
m7.metric("받은 이자 (P2P)", f"{liq.total_interest:,.0f}")
m8.metric("총 수수료 / 세금", f"{liq.total_fees:,.0f} / {liq.total_tax:,.0f}")

# --- 비교 테이블 ---
st.subheader("전략 vs 벤치마크 (세후 청산 기준)")
rows = [
    {
        "포트폴리오": s.name,
        "세후 청산가": f"{liq.final_value_after_tax:,.0f}",
        "세후 수익률": f"{liq.final_value_after_tax / initial_capital - 1.0:+.1%}",
        "CAGR (세전)": f"{s.cagr_pct:+.2%}",
        "MDD": f"{s.max_drawdown_pct:.1%}",
    }
]
if regime_result is not None:
    rs = regime_result.summary
    rlq = regime_result.liquidation
    rows.append(
        {
            "포트폴리오": f"[공포 회피] {rs.name}",
            "세후 청산가": f"{rlq.final_value_after_tax:,.0f}",
            "세후 수익률": f"{rlq.final_value_after_tax / initial_capital - 1.0:+.1%}",
            "CAGR (세전)": f"{rs.cagr_pct:+.2%}",
            "MDD": f"{rs.max_drawdown_pct:.1%}",
        }
    )
for b in result.benchmarks:
    at = result.benchmark_after_tax.get(b.name, b.final_value)
    rows.append(
        {
            "포트폴리오": b.name,
            "세후 청산가": f"{at:,.0f}",
            "세후 수익률": f"{at / initial_capital - 1.0:+.1%}",
            "CAGR (세전)": f"{b.cagr_pct:+.2%}",
            "MDD": f"{b.max_drawdown_pct:.1%}",
        }
    )
st.dataframe(rows, use_container_width=True, hide_index=True)

# --- Equity ---
log_scale = st.toggle("로그 스케일", value=True)
eq_fig = go.Figure()
eq_fig.add_trace(
    go.Scatter(
        x=[p.date for p in result.equity_curve],
        y=[p.equity for p in result.equity_curve],
        mode="lines", name="공포 사다리", line=dict(color="#2196F3", width=2.2),
    )
)
if regime_result is not None:
    eq_fig.add_trace(
        go.Scatter(
            x=[pt.date for pt in regime_result.curve],
            y=[pt.total for pt in regime_result.curve],
            mode="lines", name="[공포 회피] 밴드+200일선 레짐",
            line=dict(color="#AB47BC", width=1.6, dash="dash"),
        )
    )
for color, (name, points) in zip(
    ["#E53935", "#9E9E9E"], result.benchmark_curves.items()
):
    eq_fig.add_trace(
        go.Scatter(
            x=[p.date for p in points], y=[p.equity for p in points],
            mode="lines", name=name,
            line=dict(color=color, width=1.2, dash="dot"),
        )
    )
eq_fig.update_layout(
    template="plotly_dark", height=470,
    yaxis_type="log" if log_scale else "linear",
    xaxis_title="Date", yaxis_title="Portfolio Value",
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    margin=dict(l=50, r=50, t=40, b=30),
)
st.plotly_chart(eq_fig, use_container_width=True)

# --- 가격 + 사다리 발동 지점 ---
st.subheader("가격 · 사다리 발동 · 회복 익절")
px_fig = go.Figure()
px_fig.add_trace(
    go.Scatter(
        x=close.index, y=close.values, mode="lines",
        name=f"{ticker} 종가", line=dict(color="#B0BEC5", width=1.1),
    )
)
if config.trigger_mode == "ma":
    ma = close.rolling(config.ma_days).mean()
    px_fig.add_trace(
        go.Scatter(
            x=ma.index, y=ma.values, mode="lines",
            name=f"{config.ma_days}일선",
            line=dict(color="#FFC107", width=1.2, dash="dash"),
        )
    )
else:
    ath = close.cummax()
    px_fig.add_trace(
        go.Scatter(
            x=ath.index, y=ath.values, mode="lines", name="역대 고점 (ATH)",
            line=dict(color="#FFC107", width=1.2, dash="dash"),
        )
    )
LEVEL_COLORS = {1: "#FFB74D", 2: "#FF7043", 3: "#E53935"}
for lvl in (1, 2, 3):
    pts = [e for e in fear_buys if e.level == lvl]
    if pts:
        px_fig.add_trace(
            go.Scatter(
                x=[e.date for e in pts], y=[e.price for e in pts],
                mode="markers", name=f"{lvl}단 매수",
                marker=dict(symbol="triangle-up", size=9 + lvl * 2,
                            color=LEVEL_COLORS[lvl]),
            )
        )
recov_events = [e for e in result.events if e.kind == "recovery"]
if recov_events:
    px_fig.add_trace(
        go.Scatter(
            x=[e.date for e in recov_events],
            y=[e.price for e in recov_events],
            mode="markers", name="회복 익절 (평시 비중 복귀)",
            marker=dict(symbol="triangle-down", size=11, color="#43A047"),
        )
    )
px_fig.update_layout(
    template="plotly_dark", height=450,
    yaxis_type="log" if log_scale else "linear",
    xaxis_title="Date", yaxis_title="Price",
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    margin=dict(l=50, r=50, t=40, b=30),
)
st.plotly_chart(px_fig, use_container_width=True)

# --- 사이클 테이블 ---
st.subheader(f"공포 사이클 ({len(result.cycles)}회)")
cycle_rows = [
    {
        "시작": c.start,
        "종료": c.end if c.end else "진행 중",
        "최대 낙폭": c.min_drawdown,
        "발동 단수": c.levels_hit,
        "투입": round(c.invested),
        "회복 익절": round(c.harvested),
        "결과": "회수 완료" if c.outcome == "recovered" else "진행 중",
    }
    for c in result.cycles
]
st.dataframe(
    cycle_rows, use_container_width=True, hide_index=True,
    column_config={
        "최대 낙폭": st.column_config.NumberColumn(format="percent"),
        "투입": st.column_config.NumberColumn(format="localized"),
        "회복 익절": st.column_config.NumberColumn(format="localized"),
    },
)

# --- 이벤트 로그 ---
st.subheader(f"매매 로그 ({len(result.events)}건)")
KIND_LABELS = {
    "base_buy": "평시 매수", "fear_buy": "사다리 매수",
    "recovery": "회복 익절", "tax": "양도세 납부",
}
event_rows = [
    {
        "날짜": e.date,
        "구분": KIND_LABELS.get(e.kind, e.kind)
        + (f" {e.level}단" if e.kind == "fear_buy" else ""),
        "낙폭": f"{e.drawdown:.1%}" if e.kind == "fear_buy" else "",
        "체결가": fmt_price(e.price),
        "금액 (+매수/−매도)": round(e.notional),
        "주식 평가": round(e.stock_value_after),
        "현금": round(e.cash_after),
        "주식 비중": f"{e.stock_weight_after:.0%}",
    }
    for e in result.events
]
st.dataframe(
    event_rows, use_container_width=True, hide_index=True,
    column_config={
        "금액 (+매수/−매도)": st.column_config.NumberColumn(format="localized"),
        "주식 평가": st.column_config.NumberColumn(format="localized"),
        "현금": st.column_config.NumberColumn(format="localized"),
    },
)

values = pd.Series(
    [p.equity for p in result.equity_curve],
    index=pd.to_datetime([p.date for p in result.equity_curve]),
)
m = compute_risk_metrics(values)
st.caption(
    f"리스크: 등급 **{m['grade']}** · 최장 수면기간 "
    f"{m['longest_underwater_days'] / 365.25:.1f}년 · Sharpe {m['sharpe']:.2f} · "
    f"Calmar {m['calmar']:.2f} · 최악 연도 {m['worst_year']:+.1%}"
)
