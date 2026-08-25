from datetime import date, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data.adapters.cached_market_data import CachedMarketDataAdapter
from data.adapters.yfinance_adapter import YFinanceAdapter
from strategy.adapters import portfolio_lab as pl
from strategy.adapters.band_rebalance_strategy import BandRebalanceStrategy
from strategy.adapters.risk_metrics import compute_risk_metrics
from strategy.adapters.tqqq_p2p_strategy import TqqqP2pStrategy
from strategy.adapters.value_rebalance_strategy import ValueRebalanceStrategy
from strategy.domain.models import (
    BandRebalanceConfig,
    TossFeeSchedule,
    TqqqP2pConfig,
    ValueRebalanceConfig,
)

st.set_page_config(page_title="Risk Dashboard", layout="wide")
st.title("리스크 대시보드 — 상위 전략 심화 비교")
st.caption(
    "여태 백테스트한 전략 중 성과 상위 조합을 **같은 기간·같은 "
    "수수료(0.1%)·양도세(22%)**로 다시 돌리고, MDD 하나로는 안 보이는 "
    "리스크 팩터 — 최장 수면기간(고점 회복까지 버틴 시간), 최대 낙폭 "
    "회복일수, 변동성, Sharpe/Calmar, 최악 연도, 위기 구간 성적 — 를 "
    "한눈에 비교한다. 등급: 매우 안정(-10%↑) / 안정적(-20%↑) / "
    "시장 수준(-35%↑) / 공격적(-50%↑) / 투기적(-50% 미만)."
)

COMMON_START = date(2010, 9, 9)  # VOO 상장 — 전 전략 공통 시작
STRATEGIES = [
    "VR 실력공식 + Pool P2P 9%",
    "VR 기본공식 + Pool P2P 9%",
    "VR 2주검사만 + Pool P2P 9%",
    "TQQQ+P2P 50:50 매도 리밸런싱",
    "VOO+TQQQ 밴드15% + 200일선 레짐",
    "TQQQ50 + 금 100% (거시레짐)",
    "TQQQ50 + 금80/달러20 (거시레짐)",
    "TQQQ50 + 금60/달러40 (거시레짐)",
    "TQQQ 100% (벤치마크)",
    "TQQQ 75:25 방치 (벤치마크)",
]

with st.sidebar:
    st.header("설정")
    end_date = st.date_input(
        "End Date", value=date.today(),
        min_value=COMMON_START, max_value=date.today(),
    )
    initial_capital = st.number_input(
        "Initial Capital", value=100_000_000, min_value=1_000, step=10_000_000
    )
    selected = st.multiselect("전략 선택", STRATEGIES, default=STRATEGIES)
    st.caption(
        f"공통 시작일 {COMMON_START} (VOO 상장) — 모든 전략이 같은 "
        "구간을 쓰도록 고정."
    )
    run_btn = st.button("Run Dashboard", type="primary", use_container_width=True)

if not run_btn:
    st.info("전략을 고르고 **Run Dashboard**를 눌러줘.")
    st.stop()
if not selected:
    st.error("전략을 최소 1개 선택해줘.")
    st.stop()

FEE = TossFeeSchedule()
TAX, DEDUCT = 0.22, 2_500_000.0
CAP = float(initial_capital)
market_data = CachedMarketDataAdapter(YFinanceAdapter())


def _clean(s: pd.Series) -> pd.Series:
    if s.index.tz is not None:
        s.index = s.index.tz_localize(None)
    s.index = s.index.normalize()
    return s


def _fetch(sym: str, start: date) -> pd.Series:
    return _clean(market_data.fetch_ohlcv(sym, start, end_date)["Close"])


with st.spinner("가격 데이터 fetch..."):
    tqqq = _fetch("TQQQ", COMMON_START)

need_macro = any("거시레짐" in s for s in selected)
risk_flags = None
if need_macro:
    with st.spinner("거시 하이브리드 레짐 계산..."):
        try:
            from data.adapters.fred_adapter import FredCsvAdapter
            from strategy.adapters import macro_regime as mr

            fred = FredCsvAdapter()
            warm = COMMON_START - timedelta(days=500)
            components = {
                "curve": mr.score_yield_curve(
                    fred.fetch_series("T10Y3M", date(1982, 1, 4), end_date)
                ),
                "credit": mr.score_credit_spread(
                    fred.fetch_series("BAA10Y", date(1986, 1, 2), end_date)
                ),
                "vix": mr.score_vix(_fetch("^VIX", warm)),
                "sahm": mr.score_sahm_rule(
                    fred.fetch_series("UNRATE", date(1990, 1, 1), end_date)
                ),
                "consumer": mr.score_ratio_trend(
                    _fetch("XLY", warm), _fetch("XLP", warm)
                ),
            }
            composite = mr.composite_score(
                components, {k: 1.0 for k in components}, tqqq.index
            )
            trend = mr.score_price_trend(_fetch("QQQ", warm)).reindex(
                tqqq.index, method="ffill"
            )
            risk_flags = mr.hybrid_risk_on(trend, composite, 0.6, 10)
        except Exception as e:
            st.warning(f"거시 레짐 계산 실패 — 해당 전략은 레짐 없이 실행: {e}")


def _series_of(curve) -> pd.Series:
    return pd.Series(
        [pt.total for pt in curve],
        index=pd.to_datetime([pt.date for pt in curve]),
    )


def _bh_after_tax(stock_value: float, cost: float, rest: float) -> float:
    sell_rate = FEE.sell_commission_pct + FEE.sec_fee_pct
    f = stock_value * sell_rate
    gain = max(0.0, stock_value - f - cost - DEDUCT)
    return stock_value - f - gain * TAX + rest


def _run_vr(**kw):
    cfg = ValueRebalanceConfig(
        start_date=COMMON_START, end_date=end_date, initial_capital=CAP,
        fee_schedule=FEE, capital_gains_tax_pct=TAX, tax_deduction=DEDUCT,
        **kw,
    )
    r = ValueRebalanceStrategy().execute(tqqq, cfg)
    return _series_of(r.curve), r.liquidation.final_value_after_tax


def _run_band(defensive: pd.Series, regime_close=None, flags=None):
    cfg = BandRebalanceConfig(
        start_date=COMMON_START, end_date=end_date, initial_capital=CAP,
        fee_schedule=FEE, capital_gains_tax_pct=TAX, tax_deduction=DEDUCT,
    )
    r = BandRebalanceStrategy().execute(
        tqqq, defensive, cfg, regime_close=regime_close, risk_on_series=flags
    )
    return _series_of(r.curve), r.liquidation.final_value_after_tax


GOLD_BASKETS = {
    "TQQQ50 + 금 100% (거시레짐)": {"GOLD": 1.0},
    "TQQQ50 + 금80/달러20 (거시레짐)": {"GOLD": 0.8, "UUP": 0.2},
    "TQQQ50 + 금60/달러40 (거시레짐)": {"GOLD": 0.6, "UUP": 0.4},
}
BASKET_FETCH = {"GOLD": "GC=F", "UUP": "UUP"}

runs: dict[str, tuple[pd.Series, float]] = {}
progress = st.progress(0.0, text="전략 실행 중...")
leg_cache: dict[str, pd.Series] = {}
for i, name in enumerate(selected):
    try:
        if name == "VR 실력공식 + Pool P2P 9%":
            runs[name] = _run_vr(advanced_formula=True, pool_annual_rate=0.09)
        elif name == "VR 기본공식 + Pool P2P 9%":
            runs[name] = _run_vr(pool_annual_rate=0.09)
        elif name == "VR 2주검사만 + Pool P2P 9%":
            runs[name] = _run_vr(pool_annual_rate=0.09, check_daily=False)
        elif name == "TQQQ+P2P 50:50 매도 리밸런싱":
            cfg = TqqqP2pConfig(
                start_date=COMMON_START, end_date=end_date,
                initial_capital=CAP, allow_sell_rebalance=True,
                fee_schedule=FEE, capital_gains_tax_pct=TAX,
                tax_deduction=DEDUCT,
            )
            r = TqqqP2pStrategy().execute(tqqq, cfg)
            runs[name] = (
                _series_of(r.curve), r.liquidation.final_value_after_tax
            )
        elif name == "VOO+TQQQ 밴드15% + 200일선 레짐":
            voo = _fetch("VOO", COMMON_START)
            qqq = _fetch("QQQ", COMMON_START - timedelta(days=400))
            runs[name] = _run_band(voo, regime_close=qqq)
        elif name in GOLD_BASKETS:
            weights = GOLD_BASKETS[name]
            for k in weights:
                if k not in leg_cache:
                    leg_cache[k] = _fetch(BASKET_FETCH[k], COMMON_START)
            basket = pl.build_basket_series(
                {k: leg_cache[k] for k in weights}, weights
            )
            runs[name] = _run_band(basket, flags=risk_flags)
        elif name == "TQQQ 100% (벤치마크)":
            values = tqqq / float(tqqq.iloc[0]) * (
                CAP / (1.0 + FEE.buy_commission_pct)
            )
            runs[name] = (
                values, _bh_after_tax(float(values.iloc[-1]), CAP, 0.0)
            )
        elif name == "TQQQ 75:25 방치 (벤치마크)":
            bh = tqqq / float(tqqq.iloc[0]) * (
                CAP / (1.0 + FEE.buy_commission_pct)
            )
            values = bh * 0.75 + CAP * 0.25
            runs[name] = (
                values,
                _bh_after_tax(float(bh.iloc[-1]) * 0.75, CAP * 0.75, CAP * 0.25),
            )
    except Exception as e:
        st.warning(f"'{name}' 실행 실패: {e}")
    progress.progress((i + 1) / len(selected), text=f"전략 실행 중 ({i + 1}/{len(selected)})...")
progress.empty()

if not runs:
    st.error("실행에 성공한 전략이 없어.")
    st.stop()

# --- 리스크 지표 계산 ---
metrics = {name: compute_risk_metrics(values) for name, (values, _) in runs.items()}

# --- 종합 테이블 ---
st.subheader("리스크 팩터 종합")
GRADE_ORDER = {"매우 안정": 0, "안정적": 1, "시장 수준": 2, "공격적": 3, "투기적": 4}
rows = []
for name, (values, after_tax) in runs.items():
    m = metrics[name]
    rows.append(
        {
            "전략": name,
            "등급": m["grade"],
            "세후 청산가": f"{after_tax:,.0f}",
            "CAGR": f"{m['cagr']:+.1%}",
            "MDD": f"{m['mdd']:.1%}",
            "복구 필요 수익률": f"+{m['recovery_needed']:.0%}",
            "최장 수면기간": f"{m['longest_underwater_days'] / 365.25:.1f}년",
            "최대낙폭 회복": (
                f"{m['max_dd_recovery_days']}일"
                if m["max_dd_recovery_days"] is not None
                else "미회복"
            ),
            "연 변동성": f"{m['ann_vol']:.0%}",
            "Sharpe": f"{m['sharpe']:.2f}",
            "Calmar": f"{m['calmar']:.2f}",
            "최악 연도": f"{m['worst_year']:+.1%}",
            "일간 VaR95": f"{m['var95']:.1%}",
            "_grade_order": GRADE_ORDER.get(m["grade"], 9),
        }
    )
rows.sort(key=lambda r: r["_grade_order"])
for r in rows:
    r.pop("_grade_order")
st.dataframe(rows, use_container_width=True, hide_index=True)
st.caption(
    "**최장 수면기간** = 신고점 없이 버틴 최장 시간 (진행 중 포함) · "
    "**최대낙폭 회복** = 최대 낙폭의 직전 고점 → 원금 회복까지 달력일 · "
    "**Calmar** = CAGR ÷ |MDD| (낙폭 1%당 연수익) · 리스크 지표는 세전 "
    "곡선 기준, 세후 청산가만 세금 반영."
)

# --- 리스크-수익 지도 ---
st.subheader("리스크-수익 지도 (MDD vs CAGR)")
GRADE_COLORS = {
    "매우 안정": "#26C6DA", "안정적": "#43A047", "시장 수준": "#FFC107",
    "공격적": "#FF7043", "투기적": "#E53935",
}
sc_fig = go.Figure()
for grade, color in GRADE_COLORS.items():
    pts = [(n, metrics[n]) for n in runs if metrics[n]["grade"] == grade]
    if not pts:
        continue
    sc_fig.add_trace(
        go.Scatter(
            x=[abs(m["mdd"]) for _, m in pts],
            y=[m["cagr"] for _, m in pts],
            mode="markers+text",
            name=grade,
            text=[n.split(" (")[0] for n, _ in pts],
            textposition="top center",
            textfont=dict(size=10),
            marker=dict(size=14, color=color, line=dict(width=1, color="#212121")),
            hovertemplate="%{text}<br>MDD %{x:.1%} · CAGR %{y:.1%}<extra></extra>",
        )
    )
sc_fig.update_layout(
    template="plotly_dark", height=480,
    xaxis=dict(title="MDD (절대값 — 왼쪽일수록 안정)", tickformat=".0%"),
    yaxis=dict(title="CAGR (세전)", tickformat=".0%"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    margin=dict(l=50, r=50, t=40, b=40),
)
st.plotly_chart(sc_fig, use_container_width=True)

# --- 수면 아래 기간 (underwater) ---
st.subheader("낙폭 곡선 (Underwater) — 얼마나 깊게, 얼마나 오래")
uw_fig = go.Figure()
palette = ["#2196F3", "#FFD600", "#26C6DA", "#AB47BC", "#66BB6A",
           "#FF7043", "#EC407A", "#9E9E9E", "#E53935", "#8D6E63"]
for color, (name, (values, _)) in zip(palette, runs.items()):
    dd = values / values.cummax() - 1.0
    uw_fig.add_trace(
        go.Scatter(
            x=dd.index, y=dd.values, mode="lines", name=name,
            line=dict(color=color, width=1.3),
        )
    )
uw_fig.update_layout(
    template="plotly_dark", height=420,
    xaxis_title="Date", yaxis_title="Drawdown", yaxis_tickformat=".0%",
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    margin=dict(l=50, r=50, t=60, b=30),
)
st.plotly_chart(uw_fig, use_container_width=True)

# --- 위기 구간 성적표 ---
st.subheader("위기 구간 성적표")
CRISIS = {
    "2011 미 신용등급 강등": ("2011-07-07", "2011-10-03"),
    "2015-16 위안화 쇼크": ("2015-07-20", "2016-02-11"),
    "2018 4분기 급락": ("2018-10-01", "2018-12-24"),
    "2020 코로나": ("2020-02-19", "2020-03-23"),
    "2022 인플레 약세장": ("2021-11-19", "2022-12-28"),
}
crisis_rows = []
for name, (values, _) in runs.items():
    row = {"전략": name}
    for label, (a, b) in CRISIS.items():
        r_ = pl.window_return(values, a, b)
        row[label] = f"{r_:+.1%}" if r_ is not None else "—"
    crisis_rows.append(row)
st.dataframe(crisis_rows, use_container_width=True, hide_index=True)

# --- 평가액 곡선 ---
st.subheader("평가액 추이 (로그)")
eq_fig = go.Figure()
for color, (name, (values, _)) in zip(palette, runs.items()):
    eq_fig.add_trace(
        go.Scatter(
            x=values.index, y=values.values, mode="lines", name=name,
            line=dict(color=color, width=1.5),
        )
    )
eq_fig.update_layout(
    template="plotly_dark", height=500, yaxis_type="log",
    xaxis_title="Date", yaxis_title="Portfolio Value",
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    margin=dict(l=50, r=50, t=60, b=30),
)
st.plotly_chart(eq_fig, use_container_width=True)
