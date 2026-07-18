from datetime import date, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data.adapters.cached_market_data import CachedMarketDataAdapter
from data.adapters.yfinance_adapter import YFinanceAdapter
from strategy.adapters.band_rebalance_strategy import (
    CALIBRATED_DRAG,
    BandRebalanceStrategy,
    build_synthetic_leveraged,
    splice_series,
)
from strategy.domain.models import BandRebalanceConfig

st.set_page_config(page_title="VOO+TQQQ Rebalance", layout="wide")
st.title("VOO + TQQQ(QLD) 밴드 리밸런싱")
st.caption(
    "초기 50:50 매수 후 공격 자산이 기준가 대비 **−15%** 하락하면 "
    "방어 자산 15%를 팔아 줍줍, **+15%** 상승하면 50:50으로 수익 실현 "
    "리밸런싱. 트리거·체결 모두 당일 종가 기준. "
    "**레짐 필터**를 켜면 QQQ가 200일선 아래로 이탈할 때 방어 태세로 "
    "전환해 닷컴 버블(-97%)류 파국 구간에서 자산을 보존한다."
)

# 실제 ETF 상장일 — 이보다 이른 구간은 합성 모드로만 백테스트 가능.
ETF_INCEPTION = {
    "TQQQ": date(2010, 2, 11),
    "QLD": date(2006, 6, 21),
    "VOO": date(2010, 9, 9),
    "SPY": date(1993, 1, 29),
    "QQQ": date(1999, 3, 10),
}
LEVERAGE = {"TQQQ": 3.0, "QLD": 2.0}

# --- Sidebar inputs ---
with st.sidebar:
    st.header("자산 구성")
    aggressive_ticker = st.selectbox(
        "공격 자산 (창)",
        ["TQQQ", "QLD"],
        index=0,
        help="나스닥 100 레버리지. TQQQ = 3배(수익 극대화형), "
        "QLD = 2배(멘탈 안정형).",
    )
    defensive_ticker = st.selectbox(
        "방어 자산 (방패)",
        ["VOO", "SPY"],
        index=0,
        help="S&P 500. 하락장에서 팔아 공격 자산 물타기에 쓰는 "
        "현금 창고 역할. SPY는 1993년부터 데이터가 있어 장기 "
        "백테스트에 유리.",
    )
    use_synthetic = st.checkbox(
        "상장 이전 구간 백캐스트 (QQQ 기반, 실데이터 스플라이스)",
        value=False,
        help="상장일 이후는 **실제 ETF 가격**을 그대로 쓰고, 상장 "
        "이전(1999~)만 QQQ ×N − 실제 T-bill 자금조달비용 − 실측 "
        "드래그로 백캐스트해 이어붙인다. 모델은 실제 TQQQ/QLD 겹침 "
        "구간으로 보정(16.4년 누적 오차 +4.5%), 실존 2배 펀드 "
        "UOPIX(1997~)의 닷컴 구간으로 교차 검증됨.",
    )
    synthetic_expense = st.number_input(
        "백캐스트 드래그 — 보수+스왑 스프레드 (연 %)",
        value=CALIBRATED_DRAG.get(LEVERAGE[aggressive_ticker], 0.025) * 100,
        min_value=0.0,
        max_value=8.0,
        step=0.05,
        format="%.2f",
        disabled=not use_synthetic,
        help="자금조달비용(2×T-bill, 실데이터 자동 차감) 외에 남는 "
        "드래그. 기본값은 실제 ETF 겹침 구간 실측치 "
        "(TQQQ 2.51% / QLD 1.65%).",
    )

    st.header("기간 / 자본")
    default_start = date(2010, 2, 11) if not use_synthetic else date(1999, 3, 10)
    start_date = st.date_input(
        "Start Date",
        value=default_start,
        min_value=date(1993, 1, 29),
        max_value=date.today(),
    )
    end_date = st.date_input(
        "End Date",
        value=date.today(),
        min_value=date(1993, 1, 29),
        max_value=date.today(),
    )
    initial_capital = st.number_input(
        "Initial Capital",
        value=100_000_000,
        min_value=1_000,
        step=10_000_000,
        help="보고서 기준 1억 원. 소수점 수량을 허용하므로 단위는 "
        "결과 비율에 영향 없음.",
    )

    st.header("리밸런싱 규칙")
    band_pct = st.number_input(
        "밴드 폭 (%)",
        value=15.0,
        min_value=1.0,
        max_value=50.0,
        step=1.0,
        help="기준가 대비 ±이 폭에 도달하면 트리거. 보고서 권장 15%.",
    )
    aggressive_weight = st.number_input(
        "공격 자산 목표 비중 (%)",
        value=50.0,
        min_value=10.0,
        max_value=90.0,
        step=5.0,
        help="초기 매수 및 수익 실현 리밸런싱의 목표 비중. 보고서 "
        "권장 50:50.",
    )
    dip_sell_pct = st.number_input(
        "하락 시 방어 자산 매도 비율 (%)",
        value=15.0,
        min_value=1.0,
        max_value=100.0,
        step=1.0,
        help="줍줍 모드에서 방어 자산 평가액의 몇 %를 팔아 공격 "
        "자산을 살지. 보고서 예시: VOO 평가액의 15%.",
    )

    st.header("하락장 방어 (레짐 필터)")
    enable_regime = st.checkbox(
        "200일선 레짐 필터",
        value=True,
        help="레짐 지수가 SMA 아래로 이탈하면 방어 태세로 전환. "
        "원 전략의 치명 구간(1999-2002: 하락 내내 방어 자산을 팔아 "
        "떨어지는 TQQQ를 사다 원금의 90%를 소실)을 차단한다. "
        "1999~ 백테스트 기준 MDD -97% → -78%, 최종 41억 → 117억.",
    )
    risk_off_label = st.selectbox(
        "Risk-off 행동",
        ["방어자산 대피", "현금 대피", "줍줍만 중단"],
        index=0,
        disabled=not enable_regime,
        help="**방어자산 대피**: 공격 자산 전량을 방어 자산으로 이동 "
        "(성과-방어 균형, 권장). **현금 대피**: 전 자산 현금화 — 위기 "
        "손실은 가장 작지만 강세장 복귀가 늦어 장기 성과 희생 큼. "
        "**줍줍만 중단**: 보유는 유지하고 하락 매수만 멈춤 — 성과 "
        "손실 최소, 방어력도 최소.",
    )
    regime_ticker = st.selectbox(
        "레짐 지수",
        ["QQQ", "SPY"],
        index=0,
        disabled=not enable_regime,
        help="추세 판정에 쓸 지수. 공격 자산이 나스닥 레버리지이므로 "
        "QQQ 권장.",
    )
    regime_sma_days = st.number_input(
        "SMA 일수",
        value=200,
        min_value=50,
        max_value=300,
        step=10,
        disabled=not enable_regime,
    )
    regime_buffer = st.number_input(
        "이탈/복귀 버퍼 (%)",
        value=1.0,
        min_value=0.0,
        max_value=10.0,
        step=0.5,
        disabled=not enable_regime,
        help="SMA ± 이 % 를 넘어야 상태 전환 (휩쏘 완화 히스테리시스).",
    )
    regime_confirm_days = st.number_input(
        "재진입 확인 일수",
        value=15,
        min_value=0,
        max_value=60,
        step=5,
        disabled=not enable_regime,
        help="레짐 지수가 N일 **연속** SMA+버퍼 위를 유지해야 재진입. "
        "2000-02년 베어랠리 휩쏘(짧은 반등에 재진입 → 다음 하락 "
        "직격)를 걸러낸다. 이탈은 즉시 — 방어는 빠르게, 복귀는 "
        "신중하게.",
    )

    run_btn = st.button("Run Backtest", type="primary", use_container_width=True)

if not run_btn:
    st.info("좌측에서 자산 / 기간 / 밴드 폭을 설정하고 **Run Backtest**를 눌러줘.")
    st.stop()

# --- Data availability guard ---
if not use_synthetic:
    floor = max(
        ETF_INCEPTION[aggressive_ticker], ETF_INCEPTION[defensive_ticker]
    )
    if start_date < floor:
        st.warning(
            f"{aggressive_ticker}/{defensive_ticker} 실데이터는 "
            f"{floor}부터 존재해 시작일을 {floor}로 당겨서 실행합니다. "
            "그 이전 구간을 보려면 **합성 레버리지 시계열**을 켜줘."
        )
        start_date = floor
elif start_date < ETF_INCEPTION["QQQ"]:
    st.warning(
        f"합성 모드의 기초지수 QQQ는 {ETF_INCEPTION['QQQ']}부터 "
        "존재해 시작일을 당겨서 실행합니다."
    )
    start_date = ETF_INCEPTION["QQQ"]

# --- Fetch ---
market_data = CachedMarketDataAdapter(YFinanceAdapter())
defensive_fetch = defensive_ticker
# 합성 모드에서 방어 자산이 VOO(2010~)면 SPY로 대체해 과거 구간 확보.
if use_synthetic and start_date < ETF_INCEPTION[defensive_ticker]:
    defensive_fetch = "SPY"
    st.info(
        f"합성 모드: 방어 자산 {defensive_ticker} 상장 이전 구간이라 "
        "SPY 데이터로 대체합니다."
    )
aggressive_fetch = "QQQ" if use_synthetic else aggressive_ticker

with st.spinner(f"Fetching {aggressive_fetch} / {defensive_fetch}..."):
    try:
        agg_df = market_data.fetch_ohlcv(aggressive_fetch, start_date, end_date)
        def_df = market_data.fetch_ohlcv(defensive_fetch, start_date, end_date)
    except Exception as e:
        st.error(f"Failed to fetch data: {e}")
        st.stop()

def _normalize_daily(series: pd.Series) -> pd.Series:
    if series.index.tz is not None:
        series.index = series.index.tz_localize(None)
    series.index = series.index.normalize()
    return series


agg_close = _normalize_daily(agg_df["Close"])
def_close = _normalize_daily(def_df["Close"])

regime_close = None
if enable_regime:
    # SMA 수렴을 위해 시작일 이전 워밍업 포함 fetch (200 거래일
    # ≈ 290 달력일 — 여유 있게 SMA×2 달력일).
    regime_start = start_date - timedelta(days=max(400, int(regime_sma_days) * 2))
    with st.spinner(f"Fetching {regime_ticker} (regime SMA warmup)..."):
        try:
            regime_close = _normalize_daily(
                market_data.fetch_ohlcv(regime_ticker, regime_start, end_date)[
                    "Close"
                ]
            )
        except Exception as e:
            st.error(f"Failed to fetch regime index: {e}")
            st.stop()

if use_synthetic:
    # 백캐스트: 실제 T-bill 금리로 자금조달비용 차감 후, 상장일부터는
    # 실제 ETF 가격으로 스플라이스 — 실데이터가 있는 구간은 전부 실데이터.
    with st.spinner("Fetching ^IRX (자금조달금리) + 실제 ETF..."):
        try:
            irx = _normalize_daily(
                market_data.fetch_ohlcv("^IRX", start_date, end_date)["Close"]
            )
            synthetic = build_synthetic_leveraged(
                agg_close,
                leverage=LEVERAGE[aggressive_ticker],
                annual_expense=synthetic_expense / 100.0,
                financing_rate=irx / 100.0,
            )
            real_agg = None
            inception = ETF_INCEPTION[aggressive_ticker]
            if end_date > inception:
                real_agg = _normalize_daily(
                    market_data.fetch_ohlcv(
                        aggressive_ticker, max(start_date, inception), end_date
                    )["Close"]
                )
            agg_close = splice_series(synthetic, real_agg)
        except Exception as e:
            st.error(f"백캐스트 데이터 fetch 실패: {e}")
            st.stop()

RISK_OFF_MODES = {
    "방어자산 대피": "derisk_defensive",
    "현금 대피": "derisk_cash",
    "줍줍만 중단": "pause_dip",
}
config = BandRebalanceConfig(
    aggressive_ticker=aggressive_ticker,
    defensive_ticker=defensive_ticker,
    start_date=start_date,
    end_date=end_date,
    initial_capital=float(initial_capital),
    band_pct=band_pct / 100.0,
    aggressive_weight=aggressive_weight / 100.0,
    dip_sell_defensive_pct=dip_sell_pct / 100.0,
    regime_sma_days=int(regime_sma_days),
    regime_buffer_pct=regime_buffer / 100.0,
    risk_off_mode=RISK_OFF_MODES[risk_off_label],
    regime_confirm_days=int(regime_confirm_days),
)

with st.spinner("Running backtest..."):
    try:
        strategy = BandRebalanceStrategy()
        result = strategy.execute(
            agg_close, def_close, config, regime_close=regime_close
        )
        # 필터를 켰을 때는 필터 없는 원 전략도 함께 돌려 비교선 제공.
        base_result = (
            strategy.execute(agg_close, def_close, config)
            if enable_regime
            else None
        )
    except Exception as e:
        st.error(f"Backtest failed: {e}")
        st.stop()

s = result.summary
mode_label = (
    "실데이터 + 상장 이전 백캐스트 스플라이스" if use_synthetic else "실제 ETF 데이터"
)
st.subheader(s.name)
st.caption(
    f"{result.curve[0].date} → {result.curve[-1].date} · {mode_label} · "
    f"배당 조정 종가(auto-adjust) 기준"
)

# --- Headline metrics ---
m1, m2, m3, m4 = st.columns(4)
m1.metric("최종 평가액", f"{s.final_value:,.0f}")
m2.metric("총 수익률", f"{s.total_return_pct:+.1%}")
m3.metric("CAGR", f"{s.cagr_pct:+.2%}")
m4.metric("최대 낙폭 (MDD)", f"{s.max_drawdown_pct:.1%}")

m5, m6, m7, m8 = st.columns(4)
m5.metric("줍줍 (하락 매수)", f"{result.dip_buy_count}회")
m6.metric("수익 실현 리밸런싱", f"{result.profit_take_count}회")
risk_off_count = sum(1 for e in result.events if e.kind == "risk_off")
if enable_regime:
    m7.metric("Risk-off 전환", f"{risk_off_count}회")
else:
    best_bench = max(result.benchmarks, key=lambda b: b.final_value)
    m7.metric(
        "최고 벤치마크 대비",
        f"{s.final_value / best_bench.final_value - 1.0:+.1%}",
        help=f"vs {best_bench.name} ({best_bench.final_value:,.0f})",
    )
m8.metric("초기 자본", f"{result.config.initial_capital:,.0f}")

# --- Comparison table ---
st.subheader("전략 vs 벤치마크")
compared = [s]
if base_result is not None:
    compared.append(base_result.summary)
compared.extend(result.benchmarks)
rows = [
    {
        "포트폴리오": p.name,
        "최종 평가액": f"{p.final_value:,.0f}",
        "총 수익률": f"{p.total_return_pct:+.1%}",
        "CAGR": f"{p.cagr_pct:+.2%}",
        "MDD": f"{p.max_drawdown_pct:.1%}",
    }
    for p in compared
]
st.dataframe(rows, use_container_width=True, hide_index=True)

# --- Equity curves ---
log_scale = st.toggle("로그 스케일", value=True)
curve_dates = [pt.date for pt in result.curve]

# risk-off 구간 (연속 False 구간을 [시작, 끝] 리스트로 압축) — 평가액
# / 가격 차트에 음영으로 표시.
risk_off_spans: list[tuple] = []
span_start = None
for pt in result.curve:
    if not pt.risk_on and span_start is None:
        span_start = pt.date
    elif pt.risk_on and span_start is not None:
        risk_off_spans.append((span_start, pt.date))
        span_start = None
if span_start is not None:
    risk_off_spans.append((span_start, result.curve[-1].date))


def _shade_risk_off(fig: go.Figure) -> None:
    for x0, x1 in risk_off_spans:
        fig.add_vrect(
            x0=x0, x1=x1,
            fillcolor="rgba(229,57,53,0.10)",
            line_width=0,
            layer="below",
        )


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
if base_result is not None:
    eq_fig.add_trace(
        go.Scatter(
            x=[pt.date for pt in base_result.curve],
            y=[pt.total for pt in base_result.curve],
            mode="lines",
            name="기존 전략 (필터 없음)",
            line=dict(color="#AB47BC", width=1.5, dash="dash"),
        )
    )
_shade_risk_off(eq_fig)
BENCH_COLORS = ["#9E9E9E", "#E53935", "#43A047"]
for color, (name, points) in zip(
    BENCH_COLORS, result.benchmark_curves.items()
):
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

# --- Aggressive price + trigger events ---
st.subheader(f"{aggressive_ticker} 가격 & 리밸런싱 트리거")
px_fig = go.Figure()
px_fig.add_trace(
    go.Scatter(
        x=agg_close.index,
        y=agg_close.values,
        mode="lines",
        name=f"{aggressive_ticker} 종가",
        line=dict(color="#B0BEC5", width=1.2),
    )
)
ref_series = pd.Series(
    [pt.reference_price for pt in result.curve], index=curve_dates
)
px_fig.add_trace(
    go.Scatter(
        x=ref_series.index,
        y=ref_series.values,
        mode="lines",
        name="기준가",
        line=dict(color="#FFC107", width=1.5, dash="dash"),
    )
)
band = result.config.band_pct
px_fig.add_trace(
    go.Scatter(
        x=ref_series.index,
        y=(ref_series * (1 + band)).values,
        mode="lines",
        name=f"+{band:.0%} (수익 실현)",
        line=dict(color="#43A047", width=1, dash="dot"),
    )
)
px_fig.add_trace(
    go.Scatter(
        x=ref_series.index,
        y=(ref_series * (1 - band)).values,
        mode="lines",
        name=f"−{band:.0%} (줍줍)",
        line=dict(color="#E53935", width=1, dash="dot"),
    )
)
_shade_risk_off(px_fig)
dip_events = [e for e in result.events if e.kind == "dip_buy"]
take_events = [e for e in result.events if e.kind == "profit_take"]
risk_off_events = [e for e in result.events if e.kind == "risk_off"]
risk_on_events = [e for e in result.events if e.kind == "risk_on"]
if dip_events:
    px_fig.add_trace(
        go.Scatter(
            x=[e.date for e in dip_events],
            y=[e.aggressive_price for e in dip_events],
            mode="markers",
            name="줍줍 매수",
            marker=dict(symbol="triangle-up", size=11, color="#E53935"),
            hovertemplate="줍줍 %{x}<br>가격 %{y:,.2f}<extra></extra>",
        )
    )
if take_events:
    px_fig.add_trace(
        go.Scatter(
            x=[e.date for e in take_events],
            y=[e.aggressive_price for e in take_events],
            mode="markers",
            name="수익 실현",
            marker=dict(symbol="triangle-down", size=11, color="#43A047"),
            hovertemplate="수익 실현 %{x}<br>가격 %{y:,.2f}<extra></extra>",
        )
    )
if risk_off_events:
    px_fig.add_trace(
        go.Scatter(
            x=[e.date for e in risk_off_events],
            y=[e.aggressive_price for e in risk_off_events],
            mode="markers",
            name="Risk-off (방어 전환)",
            marker=dict(symbol="x", size=10, color="#FF7043"),
            hovertemplate="Risk-off %{x}<br>가격 %{y:,.2f}<extra></extra>",
        )
    )
if risk_on_events:
    px_fig.add_trace(
        go.Scatter(
            x=[e.date for e in risk_on_events],
            y=[e.aggressive_price for e in risk_on_events],
            mode="markers",
            name="Risk-on (재진입)",
            marker=dict(symbol="circle", size=9, color="#26C6DA"),
            hovertemplate="Risk-on %{x}<br>가격 %{y:,.2f}<extra></extra>",
        )
    )
px_fig.update_layout(
    template="plotly_dark",
    height=450,
    yaxis_type="log" if log_scale else "linear",
    xaxis_title="Date",
    yaxis_title="Price",
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    margin=dict(l=50, r=50, t=60, b=30),
)
st.plotly_chart(px_fig, use_container_width=True)

# --- Allocation over time ---
st.subheader("자산 배분 추이")
alloc_fig = go.Figure()
alloc_fig.add_trace(
    go.Scatter(
        x=curve_dates,
        y=[pt.aggressive_value for pt in result.curve],
        mode="lines",
        name=f"{aggressive_ticker} (공격)",
        stackgroup="alloc",
        line=dict(width=0.5, color="#E53935"),
    )
)
alloc_fig.add_trace(
    go.Scatter(
        x=curve_dates,
        y=[pt.defensive_value for pt in result.curve],
        mode="lines",
        name=f"{defensive_ticker} (방어)",
        stackgroup="alloc",
        line=dict(width=0.5, color="#1E88E5"),
    )
)
if any(pt.cash > 0 for pt in result.curve):
    alloc_fig.add_trace(
        go.Scatter(
            x=curve_dates,
            y=[pt.cash for pt in result.curve],
            mode="lines",
            name="현금 (risk-off 대피)",
            stackgroup="alloc",
            line=dict(width=0.5, color="#9E9E9E"),
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

# --- Drawdown ---
st.subheader("낙폭 (Drawdown)")
dd_fig = go.Figure()
strategy_values = pd.Series([pt.total for pt in result.curve], index=curve_dates)
dd = strategy_values / strategy_values.cummax() - 1.0
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
if base_result is not None:
    base_values = pd.Series(
        [pt.total for pt in base_result.curve],
        index=[pt.date for pt in base_result.curve],
    )
    base_dd = base_values / base_values.cummax() - 1.0
    dd_fig.add_trace(
        go.Scatter(
            x=base_dd.index,
            y=base_dd.values,
            mode="lines",
            name="기존 전략 (필터 없음)",
            line=dict(color="#AB47BC", width=1.5, dash="dash"),
        )
    )
for color, (name, points) in zip(
    BENCH_COLORS, result.benchmark_curves.items()
):
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

# --- Events table ---
st.subheader(f"리밸런싱 이벤트 ({len(result.events)}건)")
if not result.events:
    st.info("이 기간엔 밴드 트리거가 한 번도 발동하지 않았어.")
else:
    KIND_LABELS = {
        "dip_buy": "줍줍 매수",
        "profit_take": "수익 실현",
        "risk_off": "Risk-off (방어 전환)",
        "risk_on": "Risk-on (재진입)",
    }
    event_rows = [
        {
            "날짜": e.date,
            "구분": KIND_LABELS.get(e.kind, e.kind),
            "기준가(이전)": f"{e.reference_price_before:,.2f}",
            f"{aggressive_ticker} 가격": f"{e.aggressive_price:,.2f}",
            "변동률": f"{e.aggressive_price / e.reference_price_before - 1.0:+.1%}",
            "이동 금액(공격 방향 +)": f"{e.traded_amount:,.0f}",
            "공격 평가액": f"{e.aggressive_value_after:,.0f}",
            "방어 평가액": f"{e.defensive_value_after:,.0f}",
            "총 평가액": f"{e.total_value_after:,.0f}",
            "공격 비중": f"{e.aggressive_weight_after:.1%}",
        }
        for e in result.events
    ]
    st.dataframe(event_rows, use_container_width=True, hide_index=True)
