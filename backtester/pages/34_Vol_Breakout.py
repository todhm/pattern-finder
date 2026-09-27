from datetime import date

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data.adapters.cached_market_data import CachedMarketDataAdapter
from data.adapters.yfinance_adapter import YFinanceAdapter
from pages._shared.formatting import fmt_price
from strategy.adapters.volatility_breakout_strategy import (
    VolatilityBreakoutStrategy,
)
from strategy.domain.models import TossFeeSchedule, VolBreakoutConfig

st.set_page_config(page_title="변동성 돌파 × 켈리", layout="wide")
st.title("변동성 돌파 × 레버리지 그리드 — '연 10배' 검증대")
st.caption(
    "래리 윌리엄스가 1987년 월드컵 트레이딩 챔피언십에서 **1년 "
    "+11,376%(113배, 제3자 검증)**를 낸 방법론 계열: 매일 **시가 + "
    "k×전일 레인지** 돌파 시 매수, 당일 청산(오버나이트 없음). 수익의 "
    "본체는 켈리식 공격 사이징 — 그래서 k(돌파 계수) × **레버리지 "
    "배수** 그리드로 '연 10배가 나오는가 vs 파산하는가'를 정량화한다. "
    "수수료·슬리피지(양방향)·양도세 반영. 레버리지는 데이트레이드 "
    "마진 가정(당일 청산이라 이자 0)."
)
st.caption(
    "가격은 **분할·배당 조정가** 기준. 파산 = 평가액이 초기의 10% "
    "이하로 떨어지면 시뮬레이션 중단."
)

TICKERS = ["TQQQ", "SOXL", "UPRO", "QQQ", "SPY", "BTC-USD"]


def _parse_grid(raw: list) -> list:
    out = set()
    for v in raw:
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if f > 0:
            out.add(f)
    return sorted(out)


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

    st.header("그리드 (직접 입력 가능)")
    k_raw = st.multiselect(
        "돌파 계수 k",
        [0.3, 0.4, 0.5, 0.6, 0.7, 1.0],
        default=[0.3, 0.5, 0.7],
        accept_new_options=True,
    )
    lev_raw = st.multiselect(
        "레버리지 배수",
        [0.5, 1.0, 2.0, 3.0, 4.0],
        default=[1.0, 2.0, 3.0, 4.0],
        accept_new_options=True,
        help="평가액 × 배수만큼 매 트레이드 투입. 윌리엄스의 113배는 "
        "선물 레버리지 산물 — 배수를 올리면 수익과 파산 확률이 함께 "
        "치솟는 걸 확인하는 것이 이 그리드의 목적.",
    )

    st.header("규칙 옵션")
    exit_mode = st.selectbox("청산", ["당일 종가", "익일 시가"], index=0)
    trend_days = st.number_input(
        "추세 필터 (N일선, 0=끔)", value=0, min_value=0, max_value=300, step=10,
        help="켜면 전일 종가가 N일선 위일 때만 진입 — 하락장 돌파 "
        "실패를 걸러낸다.",
    )
    slippage_bp = st.number_input(
        "슬리피지 (bp, 편도)", value=5.0, min_value=0.0, max_value=100.0,
        step=1.0,
        help="돌파 매수는 시장가 추격이라 슬리피지가 필연. 5bp = 0.05%.",
    )

    st.header("수수료 / 세금")
    commission_pct = st.number_input(
        "거래수수료 (%, 매수·매도 각각)",
        value=0.10, min_value=0.0, max_value=1.0, step=0.01, format="%.2f",
    )
    tax_pct = st.number_input(
        "양도소득세율 (%)", value=22.0, min_value=0.0, max_value=50.0, step=1.0
    )

    run_btn = st.button("Run Grid", type="primary", use_container_width=True)

if run_btn:
    ks = _parse_grid(k_raw)
    levs = _parse_grid(lev_raw)
    if not ks or not levs:
        st.error("k와 레버리지를 최소 1개씩 (숫자로) 넣어줘.")
        st.stop()
    market_data = CachedMarketDataAdapter(YFinanceAdapter())
    with st.spinner(f"Fetching {ticker}..."):
        try:
            daily = market_data.fetch_ohlcv(ticker, start_date, end_date)
        except Exception as e:
            st.error(f"데이터 fetch 실패: {e}")
            st.stop()
    if daily.index.tz is not None:
        daily.index = daily.index.tz_localize(None)
    daily.index = daily.index.normalize()

    fee = TossFeeSchedule(
        buy_commission_pct=commission_pct / 100.0,
        sell_commission_pct=commission_pct / 100.0,
    )
    results = {}
    combos = [(k, lv) for k in ks for lv in levs]
    progress = st.progress(0.0, text="그리드 탐색 중...")
    for i, (k, lv) in enumerate(combos):
        cfg = VolBreakoutConfig(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            initial_capital=float(initial_capital),
            k=k,
            leverage=lv,
            exit_mode="close" if exit_mode == "당일 종가" else "next_open",
            trend_filter_days=int(trend_days),
            slippage_bp=float(slippage_bp),
            fee_schedule=fee,
            capital_gains_tax_pct=tax_pct / 100.0,
        )
        results[(k, lv)] = VolatilityBreakoutStrategy().execute(daily, cfg)
        progress.progress((i + 1) / len(combos), text=f"그리드 탐색 중 ({i + 1}/{len(combos)})...")
    progress.empty()
    st.session_state["volbo_state"] = {"results": results, "daily": daily}

_state = st.session_state.get("volbo_state")
if _state is None:
    st.info("그리드를 정하고 **Run Grid**를 눌러줘.")
    st.stop()
results = _state["results"]
daily = _state["daily"]
_cfg = next(iter(results.values())).config
ticker = _cfg.ticker
initial_capital = _cfg.initial_capital
ks = sorted({k for k, _ in results})
levs = sorted({lv for _, lv in results})

# --- 요약: 연 10배 검증 ---
best_key = max(
    results, key=lambda kk: results[kk].liquidation.final_value_after_tax
)
best = results[best_key]
ten_x_years = {
    kk: sum(1 for v in r.yearly_returns.values() if v >= 9.0)
    for kk, r in results.items()
}
best_year = max(best.yearly_returns.values()) if best.yearly_returns else 0.0

st.subheader(
    f"최고 조합: k={best_key[0]:g} × {best_key[1]:g}배 — "
    f"연 10배 달성 연도 {ten_x_years[best_key]}회"
)
m1, m2, m3, m4 = st.columns(4)
m1.metric("세후 청산가", f"{best.liquidation.final_value_after_tax:,.0f}")
m2.metric("CAGR (세전)", f"{best.summary.cagr_pct:+.2%}")
m3.metric("MDD", f"{best.summary.max_drawdown_pct:.1%}")
m4.metric("최고 연도 수익률", f"{best_year:+.0%}")
m5, m6, m7, m8 = st.columns(4)
m5.metric("트레이드", f"{len(best.trades):,}회")
m6.metric("승률", f"{best.win_rate:.1%}")
m7.metric(
    "평균 손익 (노셔널)",
    f"+{best.avg_win_pct:.2%} / −{best.avg_loss_pct:.2%}",
)
m8.metric(
    "추정 켈리 f*",
    f"{best.kelly_fraction:.2f}",
    help="트레이드 통계(승률·손익비)로 계산한 켈리 최적 노셔널 비율. "
    "이보다 큰 레버리지는 장기적으로 파산 방향.",
)
ruined_count = sum(1 for r in results.values() if r.ruined)
st.caption(
    f"그리드 {len(results)}개 중 **파산 {ruined_count}개** · 동일 구간 "
    f"{ticker} 존버 세후 {best.benchmark_after_tax:,.0f}."
)

# --- 히트맵 ---
st.subheader("그리드 히트맵 (행 = k, 열 = 레버리지)")
metric_choice = st.radio(
    "지표", ["세후 청산가", "CAGR (세전)", "MDD", "최고 연도 수익률", "연 10배 연도 수"],
    horizontal=True,
)


def _cell(kk):
    r = results[kk]
    if metric_choice == "세후 청산가":
        return r.liquidation.final_value_after_tax
    if metric_choice == "CAGR (세전)":
        return r.summary.cagr_pct
    if metric_choice == "MDD":
        return r.summary.max_drawdown_pct
    if metric_choice == "최고 연도 수익률":
        return max(r.yearly_returns.values()) if r.yearly_returns else 0.0
    return ten_x_years[kk]


fmt = {
    "세후 청산가": ",.0f", "CAGR (세전)": ".1%", "MDD": ".0%",
    "최고 연도 수익률": ".0%", "연 10배 연도 수": ".0f",
}[metric_choice]
z = [[_cell((k, lv)) for lv in levs] for k in ks]
text = [
    [
        ("💀 " if results[(k, lv)].ruined else "")
        for lv in levs
    ]
    for k in ks
]
heat = go.Figure(
    go.Heatmap(
        z=z,
        x=[f"{lv:g}배" for lv in levs],
        y=[f"k={k:g}" for k in ks],
        colorscale="Viridis" if metric_choice != "MDD" else "Viridis_r",
        texttemplate="%{customdata}%{z:" + fmt + "}",
        customdata=text,
        colorbar=dict(title=metric_choice),
    )
)
heat.update_layout(
    template="plotly_dark", height=380,
    xaxis_title="레버리지", yaxis_title="돌파 계수 k",
    margin=dict(l=60, r=40, t=30, b=50),
)
st.plotly_chart(heat, use_container_width=True)
st.caption("💀 = 파산(평가액이 초기의 10% 이하) 조합.")

# --- 전체 테이블 ---
st.subheader("전체 조합 (세후 청산가 순)")
rows = []
for (k, lv), r in sorted(
    results.items(),
    key=lambda kv: kv[1].liquidation.final_value_after_tax,
    reverse=True,
):
    rows.append(
        {
            "k": k,
            "레버리지": lv,
            "세후 청산가": round(r.liquidation.final_value_after_tax),
            "CAGR": r.summary.cagr_pct,
            "MDD": r.summary.max_drawdown_pct,
            "트레이드": len(r.trades),
            "승률": r.win_rate,
            "켈리 f*": round(r.kelly_fraction, 2),
            "최고 연도": max(r.yearly_returns.values()) if r.yearly_returns else 0.0,
            "연10배 연도": ten_x_years[(k, lv)],
            "파산": "💀" if r.ruined else "",
            "총 비용(수수료+슬리피지)": round(r.liquidation.total_fees),
        }
    )
st.dataframe(
    rows, use_container_width=True, hide_index=True,
    column_config={
        "세후 청산가": st.column_config.NumberColumn(format="localized"),
        "CAGR": st.column_config.NumberColumn(format="percent"),
        "MDD": st.column_config.NumberColumn(format="percent"),
        "승률": st.column_config.NumberColumn(format="percent"),
        "최고 연도": st.column_config.NumberColumn(format="percent"),
        "총 비용(수수료+슬리피지)": st.column_config.NumberColumn(format="localized"),
    },
)

# --- 최고 조합 상세 ---
st.subheader("최고 조합 상세")
log_scale = st.toggle("로그 스케일", value=True)
eq_fig = go.Figure()
eq_fig.add_trace(
    go.Scatter(
        x=[p.date for p in best.equity_curve],
        y=[p.equity for p in best.equity_curve],
        mode="lines", name=best.summary.name,
        line=dict(color="#2196F3", width=2),
    )
)
bh = daily["Close"] / float(daily["Close"].iloc[0]) * float(initial_capital)
eq_fig.add_trace(
    go.Scatter(
        x=bh.index, y=bh.values, mode="lines", name=f"{ticker} 존버",
        line=dict(color="#E53935", width=1.2, dash="dot"),
    )
)
eq_fig.update_layout(
    template="plotly_dark", height=440,
    yaxis_type="log" if log_scale else "linear",
    xaxis_title="Date", yaxis_title="Equity",
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    margin=dict(l=50, r=50, t=40, b=30),
)
st.plotly_chart(eq_fig, use_container_width=True)

yr_rows = [
    {"연도": y, "수익률": v, "연 10배": "O" if v >= 9.0 else ""}
    for y, v in sorted(best.yearly_returns.items())
]
c1, c2 = st.columns([1, 2])
with c1:
    st.dataframe(
        yr_rows, use_container_width=True, hide_index=True,
        column_config={"수익률": st.column_config.NumberColumn(format="percent")},
    )
with c2:
    trade_rows = [
        {
            "날짜": t.date,
            "진입": fmt_price(t.entry),
            "청산": fmt_price(t.exit),
            "수익률(노셔널)": t.ret_pct,
            "손익": round(t.pnl),
            "평가액": round(t.equity_after),
        }
        for t in best.trades[-300:]
    ]
    st.dataframe(
        trade_rows, use_container_width=True, hide_index=True,
        column_config={
            "수익률(노셔널)": st.column_config.NumberColumn(format="percent"),
            "손익": st.column_config.NumberColumn(format="localized"),
            "평가액": st.column_config.NumberColumn(format="localized"),
        },
    )
    st.caption("최근 300개 트레이드만 표시 (전체는 CSV).")
    all_df = pd.DataFrame([t.model_dump() for t in best.trades])
    if len(all_df):
        st.download_button(
            f"전체 트레이드 CSV ({len(best.trades):,}건)",
            all_df.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"vol_breakout_{ticker}_k{best_key[0]:g}_x{best_key[1]:g}.csv",
            mime="text/csv",
        )
