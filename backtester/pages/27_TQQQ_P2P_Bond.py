from datetime import date

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data.adapters.cached_market_data import CachedMarketDataAdapter
from data.adapters.yfinance_adapter import YFinanceAdapter
from strategy.adapters.band_rebalance_strategy import BandRebalanceStrategy
from strategy.adapters.tqqq_p2p_strategy import TqqqP2pStrategy
from strategy.domain.models import (
    BandRebalanceConfig,
    TossFeeSchedule,
    TqqqP2pConfig,
)

st.set_page_config(page_title="TQQQ + P2P Bond", layout="wide")
st.title("TQQQ + P2P 채권 50:50")
st.caption(
    "P2P 투자법인 시나리오: 자본의 50%는 TQQQ, 50%는 **연 9% 채권 "
    "사다리**(매월 이자 지급, 1년 뒤 원금 상환)에 투자. 매월 첫 "
    "거래일에 이자·만기 원금을 **50:50에 모자란 쪽부터** 재투자한다. "
    "TQQQ 매매는 토스증권 수수료(0.1% + SEC fee), 실현 차익엔 "
    "양도소득세 22%(연 기본공제 차감)를 반영. 채권 이자 소득세는 "
    "법인 운영 가정이라 모델링하지 않는다."
)

TQQQ_INCEPTION = date(2010, 2, 11)
VOO_INCEPTION = date(2010, 9, 9)

# 26번(Optimal Portfolio) 리더보드 상위 방어 바스켓 — TQQQ 50% +
# 방어 50% 밴드 리밸런싱 + 거시 하이브리드 레짐으로 재현한다.
OPTIMAL_BASKETS: dict[str, dict[str, float]] = {
    "금 60% + 달러 40%": {"GOLD": 0.6, "UUP": 0.4},
    "금 80% + 달러 20%": {"GOLD": 0.8, "UUP": 0.2},
    "금 100%": {"GOLD": 1.0},
    "금 60% + 달러 20% + 비트코인 20%": {"GOLD": 0.6, "UUP": 0.2, "BTC": 0.2},
}
BASKET_FETCH = {"GOLD": "GC=F", "UUP": "UUP", "BTC": "BTC-USD"}
BASKET_INCEPTION = {
    "GOLD": date(2000, 8, 30),
    "UUP": date(2007, 3, 1),
    "BTC": date(2014, 9, 17),
}

# --- Sidebar inputs ---
with st.sidebar:
    st.header("기간 / 자본")
    start_date = st.date_input(
        "Start Date",
        value=TQQQ_INCEPTION,
        min_value=TQQQ_INCEPTION,
        max_value=date.today(),
        help="TQQQ 상장일(2010-02-11)부터 실데이터.",
    )
    end_date = st.date_input(
        "End Date",
        value=date.today(),
        min_value=TQQQ_INCEPTION,
        max_value=date.today(),
    )
    initial_capital = st.number_input(
        "Initial Capital",
        value=100_000_000,
        min_value=1_000,
        step=10_000_000,
        help="기본 1억 원. 양도세 기본공제(250만 원)와 스케일을 맞출 것.",
    )

    st.header("P2P 채권")
    bond_rate = st.number_input(
        "채권 연이율 (%)",
        value=9.0,
        min_value=0.0,
        max_value=30.0,
        step=0.5,
        help="매월 연이율/12 만큼 이자 지급.",
    )
    maturity_months = st.number_input(
        "만기 (개월)",
        value=12,
        min_value=1,
        max_value=36,
        step=1,
        help="기본 1년 만기. 만기 도래 시 원금이 현금으로 돌아와 "
        "재배분된다.",
    )

    st.header("리밸런싱")
    tqqq_weight = st.number_input(
        "TQQQ 목표 비중 (%)",
        value=50.0,
        min_value=10.0,
        max_value=90.0,
        step=5.0,
    )
    allow_sell = st.checkbox(
        "TQQQ 매도 리밸런싱 허용",
        value=False,
        help="끄면(기본) 채권 이자·만기 원금 같은 **현금흐름만으로** "
        "50:50을 맞춘다 — 상승장에선 TQQQ 비중이 목표를 초과한 채 "
        "유지된다. 켜면 매월 초과분을 팔아 50:50을 강제한다 (매도 "
        "수수료 + 양도세 발생).",
    )

    st.header("수수료 / 세금")
    commission_pct = st.number_input(
        "토스증권 거래수수료 (%, 매수·매도 각각)",
        value=0.10,
        min_value=0.0,
        max_value=1.0,
        step=0.01,
        format="%.2f",
    )
    tax_pct = st.number_input(
        "양도소득세율 (%)",
        value=22.0,
        min_value=0.0,
        max_value=50.0,
        step=1.0,
        help="미국 주식 양도차익에 대한 세율 (지방소득세 포함 22%).",
    )
    tax_deduction = st.number_input(
        "연간 기본공제",
        value=2_500_000,
        min_value=0,
        step=500_000,
        help="연간 실현차익에서 차감 후 과세. 0으로 두면 전액 과세.",
    )

    st.header("비교")
    compare_band = st.checkbox(
        "VOO+TQQQ 밴드 리밸런싱과 비교",
        value=True,
        help="24번 페이지의 기존 전략(15% 밴드 + 200일선 레짐 필터, "
        "QQQ 기준)을 **같은 수수료·양도세를 적용해** 함께 돌려 "
        "비교한다. VOO 상장일(2010-09-09) 이후 구간에서만 계산된다.",
    )
    optimal_selected = st.multiselect(
        "Optimal Portfolio 상위 바스켓 (TQQQ + 방어 바스켓)",
        list(OPTIMAL_BASKETS),
        default=list(OPTIMAL_BASKETS),
        help="26번 페이지 리더보드 상위 조합을 같은 밴드 리밸런싱 + "
        "거시 하이브리드 레짐 필터로 재현하되 **수수료·양도세를 "
        "반영**한다. 실제 TQQQ 데이터 구간(2010-02~, BTC 포함 조합은 "
        "2014-09~)에서 계산되므로 26번의 2007~ 백캐스트 수치와는 "
        "절대값이 다르다.",
    )

    run_btn = st.button("Run Backtest", type="primary", use_container_width=True)

if not run_btn:
    st.info("좌측에서 기간 / 채권 조건 / 세금을 설정하고 **Run Backtest**를 눌러줘.")
    st.stop()

# --- Fetch ---
market_data = CachedMarketDataAdapter(YFinanceAdapter())
with st.spinner("Fetching TQQQ..."):
    try:
        df = market_data.fetch_ohlcv("TQQQ", start_date, end_date)
    except Exception as e:
        st.error(f"Failed to fetch data: {e}")
        st.stop()

close = df["Close"]
if close.index.tz is not None:
    close.index = close.index.tz_localize(None)
close.index = close.index.normalize()

config = TqqqP2pConfig(
    start_date=start_date,
    end_date=end_date,
    initial_capital=float(initial_capital),
    tqqq_weight=tqqq_weight / 100.0,
    bond_annual_rate=bond_rate / 100.0,
    bond_maturity_months=int(maturity_months),
    allow_sell_rebalance=allow_sell,
    fee_schedule=TossFeeSchedule(
        buy_commission_pct=commission_pct / 100.0,
        sell_commission_pct=commission_pct / 100.0,
    ),
    capital_gains_tax_pct=tax_pct / 100.0,
    tax_deduction=float(tax_deduction),
)

with st.spinner("Running backtest..."):
    try:
        result = TqqqP2pStrategy().execute(close, config)
    except Exception as e:
        st.error(f"Backtest failed: {e}")
        st.stop()

# --- 기존 VOO+TQQQ 밴드 리밸런싱 전략과 비교 (24번 페이지 기본 설정) ---
band_result = None
if compare_band:
    from datetime import timedelta

    band_start = max(start_date, VOO_INCEPTION)
    with st.spinner("Fetching VOO / QQQ (밴드 전략 비교)..."):
        try:
            voo = market_data.fetch_ohlcv("VOO", band_start, end_date)["Close"]
            qqq = market_data.fetch_ohlcv(
                "QQQ", band_start - timedelta(days=400), end_date
            )["Close"]
            for s_ in (voo, qqq):
                if s_.index.tz is not None:
                    s_.index = s_.index.tz_localize(None)
            voo.index = voo.index.normalize()
            qqq.index = qqq.index.normalize()
            band_cfg = BandRebalanceConfig(
                start_date=band_start,
                end_date=end_date,
                initial_capital=float(initial_capital),
                fee_schedule=config.fee_schedule,
                capital_gains_tax_pct=config.capital_gains_tax_pct,
                tax_deduction=config.tax_deduction,
            )
            band_result = BandRebalanceStrategy().execute(
                close[close.index >= pd.Timestamp(band_start)],
                voo,
                band_cfg,
                regime_close=qqq,
            )
        except Exception as e:
            st.warning(f"밴드 전략 비교 실패 (본 전략 결과는 유효): {e}")

# --- Optimal Portfolio 상위 바스켓 (26번 페이지) — 세후 재현 ---
optimal_results: dict[str, object] = {}
if optimal_selected:
    from datetime import timedelta as _td

    from data.adapters.fred_adapter import FredCsvAdapter
    from strategy.adapters import macro_regime as mr
    from strategy.adapters import portfolio_lab as pl

    def _clean(s_: pd.Series) -> pd.Series:
        if s_.index.tz is not None:
            s_.index = s_.index.tz_localize(None)
        s_.index = s_.index.normalize()
        return s_

    # 26번과 동일한 거시 하이브리드 레짐 (실패 시 필터 없이 진행).
    risk_flags = None
    with st.spinner("거시 레짐 계산 (Optimal 바스켓 비교)..."):
        try:
            fred = FredCsvAdapter()
            warm = start_date - _td(days=500)
            components = {
                "curve": mr.score_yield_curve(
                    fred.fetch_series("T10Y3M", date(1982, 1, 4), end_date)
                ),
                "credit": mr.score_credit_spread(
                    fred.fetch_series("BAA10Y", date(1986, 1, 2), end_date)
                ),
                "vix": mr.score_vix(
                    _clean(market_data.fetch_ohlcv("^VIX", warm, end_date)["Close"])
                ),
                "sahm": mr.score_sahm_rule(
                    fred.fetch_series("UNRATE", date(1990, 1, 1), end_date)
                ),
                "consumer": mr.score_ratio_trend(
                    _clean(market_data.fetch_ohlcv("XLY", warm, end_date)["Close"]),
                    _clean(market_data.fetch_ohlcv("XLP", warm, end_date)["Close"]),
                ),
            }
            composite = mr.composite_score(
                components, {k: 1.0 for k in components}, close.index
            )
            trend = mr.score_price_trend(
                _clean(market_data.fetch_ohlcv("QQQ", warm, end_date)["Close"])
            ).reindex(close.index, method="ffill")
            risk_flags = mr.hybrid_risk_on(trend, composite, 0.6, 10)
        except Exception as e:
            st.warning(f"거시 레짐 계산 실패 — 레짐 필터 없이 진행: {e}")

    with st.spinner("Optimal 바스켓 백테스트..."):
        leg_cache: dict[str, pd.Series] = {}
        for basket_name in optimal_selected:
            weights = OPTIMAL_BASKETS[basket_name]
            eff_start = max(
                start_date,
                TQQQ_INCEPTION,
                *[BASKET_INCEPTION[k] for k in weights],
            )
            try:
                for key in weights:
                    if key not in leg_cache:
                        leg_cache[key] = _clean(
                            market_data.fetch_ohlcv(
                                BASKET_FETCH[key], eff_start, end_date
                            )["Close"]
                        )
                basket = pl.build_basket_series(
                    {k: leg_cache[k] for k in weights}, weights
                )
                opt_cfg = BandRebalanceConfig(
                    aggressive_ticker="TQQQ",
                    defensive_ticker=basket_name,
                    start_date=eff_start,
                    end_date=end_date,
                    initial_capital=float(initial_capital),
                    aggressive_weight=tqqq_weight / 100.0,
                    fee_schedule=config.fee_schedule,
                    capital_gains_tax_pct=config.capital_gains_tax_pct,
                    tax_deduction=config.tax_deduction,
                )
                optimal_results[basket_name] = BandRebalanceStrategy().execute(
                    close[close.index >= pd.Timestamp(eff_start)],
                    basket,
                    opt_cfg,
                    risk_on_series=risk_flags,
                )
            except Exception as e:
                st.warning(f"바스켓 '{basket_name}' 백테스트 실패: {e}")

s = result.summary
liq = result.liquidation
st.subheader(s.name)
st.caption(
    f"{result.curve[0].date} → {result.curve[-1].date} · 실제 TQQQ "
    "배당 조정 종가 · 채권은 액면 평가 (부도율 0% 가정)"
)

# --- Headline metrics ---
m1, m2, m3, m4 = st.columns(4)
m1.metric("최종 평가액 (세전)", f"{liq.final_value_pre_tax:,.0f}")
m2.metric(
    "세후 청산 가치",
    f"{liq.final_value_after_tax:,.0f}",
    help="마지막 날 TQQQ 전량 매도 가정: 매도 수수료 + 양도세 차감.",
)
m3.metric("CAGR (세전)", f"{s.cagr_pct:+.2%}")
m4.metric("최대 낙폭 (MDD)", f"{s.max_drawdown_pct:.1%}")

after_tax_return = liq.final_value_after_tax / config.initial_capital - 1.0
m5, m6, m7, m8 = st.columns(4)
m5.metric("세후 총 수익률", f"{after_tax_return:+.1%}")
m6.metric("받은 채권 이자 합계", f"{liq.total_interest:,.0f}")
m7.metric("총 수수료", f"{liq.total_fees:,.0f}")
m8.metric(
    "총 양도소득세",
    f"{liq.total_tax:,.0f}",
    help=f"연 단위 정산 + 최종 청산분 {liq.final_tax:,.0f} 포함.",
)

# --- Comparison table ---
st.subheader("전략 vs 벤치마크 (세후 청산 기준)")
rows = [
    {
        "포트폴리오": s.name,
        "최종 평가액 (세전)": f"{liq.final_value_pre_tax:,.0f}",
        "세후 청산가": f"{liq.final_value_after_tax:,.0f}",
        "세후 수익률": f"{after_tax_return:+.1%}",
        "CAGR (세전)": f"{s.cagr_pct:+.2%}",
        "MDD": f"{s.max_drawdown_pct:.1%}",
    }
]
if band_result is not None:
    bs = band_result.summary
    blq = band_result.liquidation
    rows.append(
        {
            "포트폴리오": f"[기존] {bs.name}",
            "최종 평가액 (세전)": f"{blq.final_value_pre_tax:,.0f}",
            "세후 청산가": f"{blq.final_value_after_tax:,.0f}",
            "세후 수익률": (
                f"{blq.final_value_after_tax / config.initial_capital - 1.0:+.1%}"
            ),
            "CAGR (세전)": f"{bs.cagr_pct:+.2%}",
            "MDD": f"{bs.max_drawdown_pct:.1%}",
        }
    )
for basket_name, opt_res in optimal_results.items():
    os_ = opt_res.summary
    olq = opt_res.liquidation
    rows.append(
        {
            "포트폴리오": (
                f"[최적] TQQQ {tqqq_weight:.0f}% + {basket_name} "
                f"(시작 {opt_res.curve[0].date})"
            ),
            "최종 평가액 (세전)": f"{olq.final_value_pre_tax:,.0f}",
            "세후 청산가": f"{olq.final_value_after_tax:,.0f}",
            "세후 수익률": (
                f"{olq.final_value_after_tax / config.initial_capital - 1.0:+.1%}"
            ),
            "CAGR (세전)": f"{os_.cagr_pct:+.2%}",
            "MDD": f"{os_.max_drawdown_pct:.1%}",
        }
    )
for b in result.benchmarks:
    at = result.benchmark_after_tax.get(b.name, b.final_value)
    rows.append(
        {
            "포트폴리오": b.name,
            "최종 평가액 (세전)": f"{b.final_value:,.0f}",
            "세후 청산가": f"{at:,.0f}",
            "세후 수익률": f"{at / config.initial_capital - 1.0:+.1%}",
            "CAGR (세전)": f"{b.cagr_pct:+.2%}",
            "MDD": f"{b.max_drawdown_pct:.1%}",
        }
    )
st.dataframe(rows, use_container_width=True, hide_index=True)
st.caption(
    "비교 전략들은 각 자산의 데이터 시작일부터 계산된다 (VOO "
    "2010-09-09, BTC 포함 바스켓 2014-09-17) — 시작일이 다른 행은 "
    "최종 평가액 대신 **CAGR·MDD**로 비교할 것. [최적] 행은 26번 "
    "페이지 상위 바스켓을 동일 수수료·양도세로 재현한 것."
)

# --- Equity curves ---
log_scale = st.toggle("로그 스케일", value=True)
curve_dates = [pt.date for pt in result.curve]

eq_fig = go.Figure()
eq_fig.add_trace(
    go.Scatter(
        x=curve_dates,
        y=[pt.total for pt in result.curve],
        mode="lines",
        name=s.name,
        line=dict(color="#2196F3", width=2.5),
    )
)
if band_result is not None:
    eq_fig.add_trace(
        go.Scatter(
            x=[pt.date for pt in band_result.curve],
            y=[pt.total for pt in band_result.curve],
            mode="lines",
            name="[기존] VOO+TQQQ 밴드 리밸런싱",
            line=dict(color="#AB47BC", width=1.8, dash="dash"),
        )
    )
OPTIMAL_COLORS = ["#FFD600", "#26C6DA", "#FF7043", "#66BB6A"]
for color, (basket_name, opt_res) in zip(
    OPTIMAL_COLORS, optimal_results.items()
):
    eq_fig.add_trace(
        go.Scatter(
            x=[pt.date for pt in opt_res.curve],
            y=[pt.total for pt in opt_res.curve],
            mode="lines",
            name=f"[최적] TQQQ+{basket_name}",
            line=dict(color=color, width=1.5, dash="dash"),
        )
    )
BENCH_COLORS = ["#E53935", "#43A047", "#9E9E9E"]
for color, (name, points) in zip(BENCH_COLORS, result.benchmark_curves.items()):
    eq_fig.add_trace(
        go.Scatter(
            x=[p.date for p in points],
            y=[p.equity for p in points],
            mode="lines",
            name=name,
            line=dict(color=color, width=1.3, dash="dot"),
        )
    )
eq_fig.update_layout(
    title="평가액 추이 — 전략 vs 벤치마크",
    template="plotly_dark",
    height=500,
    yaxis_type="log" if log_scale else "linear",
    xaxis_title="Date",
    yaxis_title="Portfolio Value",
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    margin=dict(l=50, r=50, t=80, b=30),
)
st.plotly_chart(eq_fig, use_container_width=True)

# --- Allocation over time ---
st.subheader("자산 배분 추이")
alloc_fig = go.Figure()
alloc_fig.add_trace(
    go.Scatter(
        x=curve_dates,
        y=[pt.tqqq_value for pt in result.curve],
        mode="lines",
        name="TQQQ",
        stackgroup="alloc",
        line=dict(width=0.5, color="#E53935"),
    )
)
alloc_fig.add_trace(
    go.Scatter(
        x=curve_dates,
        y=[pt.bond_value for pt in result.curve],
        mode="lines",
        name="P2P 채권 (액면)",
        stackgroup="alloc",
        line=dict(width=0.5, color="#1E88E5"),
    )
)
alloc_fig.update_layout(
    template="plotly_dark",
    height=350,
    xaxis_title="Date",
    yaxis_title="Value",
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    margin=dict(l=50, r=50, t=60, b=30),
)
st.plotly_chart(alloc_fig, use_container_width=True)

# --- TQQQ weight drift ---
st.subheader("TQQQ 비중 추이")
weight_fig = go.Figure()
totals = pd.Series([pt.total for pt in result.curve], index=curve_dates)
tqqq_w = pd.Series(
    [pt.tqqq_value / pt.total if pt.total > 0 else 0.0 for pt in result.curve],
    index=curve_dates,
)
weight_fig.add_trace(
    go.Scatter(
        x=tqqq_w.index,
        y=tqqq_w.values,
        mode="lines",
        name="TQQQ 비중",
        line=dict(color="#FFC107", width=1.5),
    )
)
weight_fig.add_hline(
    y=config.tqqq_weight,
    line_dash="dash",
    line_color="#9E9E9E",
    annotation_text=f"목표 {config.tqqq_weight:.0%}",
)
weight_fig.update_layout(
    template="plotly_dark",
    height=300,
    xaxis_title="Date",
    yaxis_title="Weight",
    yaxis_tickformat=".0%",
    margin=dict(l=50, r=50, t=40, b=30),
)
st.plotly_chart(weight_fig, use_container_width=True)

# --- Drawdown ---
st.subheader("낙폭 (Drawdown)")
dd_fig = go.Figure()
dd = totals / totals.cummax() - 1.0
dd_fig.add_trace(
    go.Scatter(
        x=dd.index,
        y=dd.values,
        mode="lines",
        name=s.name,
        line=dict(color="#2196F3", width=2),
        fill="tozeroy",
        fillcolor="rgba(33,150,243,0.15)",
    )
)
if band_result is not None:
    band_values = pd.Series(
        [pt.total for pt in band_result.curve],
        index=[pt.date for pt in band_result.curve],
    )
    band_dd = band_values / band_values.cummax() - 1.0
    dd_fig.add_trace(
        go.Scatter(
            x=band_dd.index,
            y=band_dd.values,
            mode="lines",
            name="[기존] VOO+TQQQ 밴드 리밸런싱",
            line=dict(color="#AB47BC", width=1.5, dash="dash"),
        )
    )
for color, (name, points) in zip(BENCH_COLORS, result.benchmark_curves.items()):
    bench_values = pd.Series(
        [p.equity for p in points], index=[p.date for p in points]
    )
    bench_dd = bench_values / bench_values.cummax() - 1.0
    dd_fig.add_trace(
        go.Scatter(
            x=bench_dd.index,
            y=bench_dd.values,
            mode="lines",
            name=name,
            line=dict(color=color, width=1, dash="dot"),
        )
    )
dd_fig.update_layout(
    template="plotly_dark",
    height=350,
    xaxis_title="Date",
    yaxis_title="Drawdown",
    yaxis_tickformat=".0%",
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    margin=dict(l=50, r=50, t=60, b=30),
)
st.plotly_chart(dd_fig, use_container_width=True)

# --- Monthly events table ---
st.subheader(f"월별 현금흐름 ({len(result.events)}건)")
event_rows = [
    {
        "날짜": e.date,
        "이자 수령": f"{e.interest:,.0f}",
        "만기 원금": f"{e.matured_principal:,.0f}",
        "양도세 납부": f"{e.tax_paid:,.0f}",
        "TQQQ 매매 (+매수/−매도)": f"{e.tqqq_traded:,.0f}",
        "신규 채권 투자": f"{e.bond_invested:,.0f}",
        "TQQQ 평가액": f"{e.tqqq_value_after:,.0f}",
        "채권 잔액": f"{e.bond_value_after:,.0f}",
        "TQQQ 비중": f"{e.tqqq_weight_after:.1%}",
    }
    for e in result.events
]
st.dataframe(event_rows, use_container_width=True, hide_index=True)
