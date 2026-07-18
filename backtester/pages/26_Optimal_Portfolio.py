from datetime import date, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data.adapters.cached_market_data import CachedMarketDataAdapter
from data.adapters.fred_adapter import FredCsvAdapter
from data.adapters.yfinance_adapter import YFinanceAdapter
from strategy.adapters import macro_regime as mr
from strategy.adapters import portfolio_lab as pl
from strategy.adapters.band_rebalance_strategy import (
    CALIBRATED_DRAG,
    BandRebalanceStrategy,
    build_synthetic_leveraged,
    splice_series,
)
from strategy.domain.models import BandRebalanceConfig

st.set_page_config(page_title="Optimal Portfolio", layout="wide")
st.title("최적 포트폴리오 탐색기 — 위기 자산 바스켓")
st.caption(
    "공격 자산(TQQQ/QLD)은 고정하고 **방어 사이드를 위기 자산 바스켓** "
    "(주식·금·국채·달러·현금)으로 일반화해, 가중치 그리드 전수 탐색으로 "
    "최적 조합을 찾는다. 각 조합은 밴드 리밸런싱 + 거시 하이브리드 레짐 "
    "필터로 백테스트하고, 위기·전쟁 구간 성적을 **위기 방어 점수** "
    "(100=무손실, 0=주식 방어와 동일 손실)로 채점한다."
)

# 방어 바스켓 후보. 금은 GLD(2004~)보다 긴 GC=F(2000~) 선물 사용.
DEFENSIVE_ASSETS = {
    "SPY": {"label": "주식 (S&P 500)", "fetch": "SPY", "inception": date(1993, 1, 29)},
    "GOLD": {"label": "금", "fetch": "GC=F", "inception": date(2000, 8, 30)},
    "TLT": {"label": "장기국채 20Y+", "fetch": "TLT", "inception": date(2002, 7, 30)},
    "IEF": {"label": "중기국채 7-10Y", "fetch": "IEF", "inception": date(2002, 7, 30)},
    "UUP": {"label": "달러 인덱스", "fetch": "UUP", "inception": date(2007, 3, 1)},
    "BTC": {"label": "비트코인 (BTC-USD)", "fetch": "BTC-USD", "inception": date(2014, 9, 17)},
    pl.CASH: {"label": "현금 (무수익)", "fetch": None, "inception": date(1990, 1, 1)},
}
LEVERAGE = {"TQQQ": 3.0, "QLD": 2.0}
ETF_INCEPTION = {"TQQQ": date(2010, 2, 11), "QLD": date(2006, 6, 21)}

# --- Sidebar ---
with st.sidebar:
    st.header("공격 자산")
    aggressive_ticker = st.selectbox("공격 자산", ["TQQQ", "QLD"], index=0)
    use_synthetic = st.checkbox(
        "상장 이전 구간 백캐스트 (실데이터 스플라이스)",
        value=True,
        help="상장일 이후는 실제 ETF 가격, 이전만 QQQ ×N − 실제 "
        "T-bill 자금조달비용 − 실측 드래그로 백캐스트해 연결. 닷컴 "
        "후반·금융위기 포함 백테스트용. 시작일은 선택한 방어 자산들의 "
        "데이터 교집합으로 자동 조정.",
    )
    synthetic_expense = st.number_input(
        "백캐스트 드래그 — 보수+스왑 스프레드 (연 %)",
        value=CALIBRATED_DRAG.get(LEVERAGE[aggressive_ticker], 0.025) * 100,
        min_value=0.0,
        max_value=8.0,
        step=0.05,
        format="%.2f",
        disabled=not use_synthetic,
        help="실제 ETF 겹침 구간 실측 기본값 (TQQQ 2.51% / QLD 1.65%). "
        "자금조달비용은 ^IRX 실데이터로 자동 차감.",
    )

    st.header("방어 바스켓 후보")
    default_on = {"SPY", "GOLD", "TLT"}
    selected_assets = [
        key
        for key, info in DEFENSIVE_ASSETS.items()
        if st.checkbox(
            info["label"],
            value=key in default_on,
            help=f"데이터 시작: {info['inception']}",
        )
    ]
    grid_step = st.selectbox(
        "가중치 그리드 간격",
        [0.25, 0.2, 0.1],
        index=1,
        format_func=lambda v: f"{v:.0%}",
        help="간격이 좁을수록 조합 수가 늘어 오래 걸림 "
        "(4자산 기준 25%→35개 / 20%→56개 / 10%→286개).",
    )

    st.header("기간 / 자본")
    start_date = st.date_input(
        "Start Date", value=date(2002, 7, 30),
        min_value=date(1999, 3, 10), max_value=date.today(),
    )
    end_date = st.date_input(
        "End Date", value=date.today(),
        min_value=date(1999, 3, 10), max_value=date.today(),
    )
    initial_capital = st.number_input(
        "Initial Capital", value=100_000_000, min_value=1_000, step=10_000_000
    )

    st.header("리밸런싱 / 레짐")
    band_pct = st.number_input("밴드 폭 (%)", value=15.0, min_value=1.0, max_value=50.0, step=1.0)
    aggressive_weight = st.number_input(
        "공격 자산 목표 비중 (%)", value=50.0, min_value=10.0, max_value=90.0, step=5.0
    )
    dip_sell_pct = st.number_input(
        "하락 시 방어 매도 비율 (%)", value=15.0, min_value=1.0, max_value=100.0, step=1.0
    )
    enable_regime = st.checkbox(
        "거시 하이브리드 레짐 필터",
        value=True,
        help="추세 붕괴 × 거시 악화 동시 확인 시 방어 전환 "
        "(25 페이지와 동일한 5개 지표 합성).",
    )
    veto_threshold = st.number_input(
        "거시 거부권 임계값", value=0.6, min_value=0.1, max_value=0.9, step=0.05,
        disabled=not enable_regime,
    )
    confirm_days = st.number_input(
        "재진입 확인 일수", value=10, min_value=0, max_value=60, step=5,
        disabled=not enable_regime,
    )

    st.header("점수 가중치")
    crisis_lambda = st.slider(
        "위기 방어 중요도 (λ)",
        min_value=0.0, max_value=1.0, value=0.5, step=0.05,
        help="종합 점수 = (1−λ)×성장 점수 + λ×위기 방어 점수. "
        "0 = 수익률만, 1 = 위기 방어만.",
    )
    top_n = st.number_input("리더보드 표시 수", value=15, min_value=5, max_value=50, step=5)

    run_btn = st.button("Find Optimal Portfolio", type="primary", use_container_width=True)

if not run_btn:
    st.info("좌측에서 방어 바스켓 후보와 설정을 고르고 **Find Optimal Portfolio**를 눌러줘.")
    st.stop()

if not selected_assets:
    st.error("방어 바스켓 후보를 최소 1개 선택해줘.")
    st.stop()
if selected_assets == [pl.CASH]:
    st.error("현금만으로는 방어 바스켓을 만들 수 없어 — 가격 자산을 하나 이상 선택해줘.")
    st.stop()

# --- 공통 시작일: 선택 자산들의 데이터 교집합 (공정 비교) ---
inceptions = [DEFENSIVE_ASSETS[k]["inception"] for k in selected_assets]
if not use_synthetic:
    inceptions.append(ETF_INCEPTION[aggressive_ticker])
else:
    inceptions.append(date(1999, 3, 10))  # QQQ
effective_start = max([start_date, *inceptions])
if effective_start > start_date:
    st.warning(
        f"선택 자산들의 데이터 교집합에 맞춰 시작일을 {effective_start}로 "
        "조정했어 (모든 조합이 같은 기간을 쓰도록 — 공정 비교)."
    )

market_data = CachedMarketDataAdapter(YFinanceAdapter())
fred = FredCsvAdapter()
warm_start = effective_start - timedelta(days=500)


def _normalize_daily(series: pd.Series) -> pd.Series:
    if series.index.tz is not None:
        series.index = series.index.tz_localize(None)
    series.index = series.index.normalize()
    return series


def _fetch_close(sym: str, start: date) -> pd.Series:
    return _normalize_daily(market_data.fetch_ohlcv(sym, start, end_date)["Close"])


with st.spinner("Fetching 가격 데이터..."):
    try:
        if use_synthetic:
            qqq_series = _fetch_close("QQQ", effective_start)
            irx = _fetch_close("^IRX", effective_start)
            synthetic = build_synthetic_leveraged(
                qqq_series,
                leverage=LEVERAGE[aggressive_ticker],
                annual_expense=synthetic_expense / 100.0,
                financing_rate=irx / 100.0,
            )
            inception = ETF_INCEPTION[aggressive_ticker]
            real_agg = (
                _fetch_close(aggressive_ticker, max(effective_start, inception))
                if end_date > inception
                else None
            )
            agg_close = splice_series(synthetic, real_agg)
        else:
            agg_close = _fetch_close(aggressive_ticker, effective_start)
        qqq_full = _fetch_close("QQQ", warm_start)
        asset_prices: dict[str, pd.Series | None] = {}
        for key in selected_assets:
            fetch_sym = DEFENSIVE_ASSETS[key]["fetch"]
            asset_prices[key] = (
                None if fetch_sym is None else _fetch_close(fetch_sym, effective_start)
            )
    except Exception as e:
        st.error(f"가격 데이터 fetch 실패: {e}")
        st.stop()

# 모든 가격 자산 + 공격 자산의 교집합 달력으로 정렬 — 조합별
# 기간 차이가 성적에 섞이지 않게 한다.
calendar_df = pd.DataFrame(
    {"agg": agg_close, **{k: v for k, v in asset_prices.items() if v is not None}}
).dropna()
agg_close = calendar_df["agg"]
for k in asset_prices:
    if asset_prices[k] is not None:
        asset_prices[k] = calendar_df[k]

# --- 거시 레짐 flags (조합 공통 — 한 번만 계산) ---
risk_flags = None
if enable_regime:
    with st.spinner("Fetching 경제지표 + 레짐 계산..."):
        components: dict[str, pd.Series] = {}
        macro_fetchers = {
            "curve": lambda: mr.score_yield_curve(
                fred.fetch_series("T10Y3M", date(1982, 1, 4), end_date)
            ),
            "credit": lambda: mr.score_credit_spread(
                fred.fetch_series("BAA10Y", date(1986, 1, 2), end_date)
            ),
            "vix": lambda: mr.score_vix(_fetch_close("^VIX", warm_start)),
            "sahm": lambda: mr.score_sahm_rule(
                fred.fetch_series("UNRATE", date(1990, 1, 1), end_date)
            ),
            "consumer": lambda: mr.score_ratio_trend(
                _fetch_close("XLY", warm_start), _fetch_close("XLP", warm_start)
            ),
        }
        for key, fetch in macro_fetchers.items():
            try:
                components[key] = fetch()
            except Exception as e:
                st.warning(f"지표 '{key}' 로드 실패 — 제외: {e}")
        if components:
            composite = mr.composite_score(
                components, {k: 1.0 for k in components}, agg_close.index
            )
            trend_score = mr.score_price_trend(qqq_full).reindex(
                agg_close.index, method="ffill"
            )
            risk_flags = mr.hybrid_risk_on(
                trend_score, composite, float(veto_threshold), int(confirm_days)
            )
        else:
            st.warning("지표 전부 실패 — 레짐 필터 없이 진행.")

config = BandRebalanceConfig(
    aggressive_ticker=aggressive_ticker,
    defensive_ticker="BASKET",
    start_date=effective_start,
    end_date=end_date,
    initial_capital=float(initial_capital),
    band_pct=band_pct / 100.0,
    aggressive_weight=aggressive_weight / 100.0,
    dip_sell_defensive_pct=dip_sell_pct / 100.0,
)
strategy = BandRebalanceStrategy()
ASSET_LABELS = {k: v["label"] for k, v in DEFENSIVE_ASSETS.items()}


def _run(defensive_series: pd.Series):
    return strategy.execute(
        agg_close, defensive_series, config, risk_on_series=risk_flags
    )


def _values(curve) -> pd.Series:
    return pd.Series(
        [pt.total for pt in curve],
        index=pd.to_datetime([pt.date for pt in curve]),
    )


# --- 기준 포트폴리오: 방어 100% 주식 (점수 앵커) ---
with st.spinner("기준 포트폴리오 (방어=주식 100%) 실행..."):
    baseline_stock = (
        asset_prices["SPY"]
        if "SPY" in selected_assets
        else _fetch_close("SPY", effective_start).reindex(agg_close.index).dropna()
    )
    baseline_result = _run(pl.build_basket_series({"s": baseline_stock}, {"s": 1.0}))
    baseline_v = _values(baseline_result.curve)
    baseline_cagr = baseline_result.summary.cagr_pct
    baseline_crisis = {
        name: pl.window_return(baseline_v, a, b)
        for name, (a, b) in pl.CRISIS_WINDOWS.items()
    }
    active_windows = [n for n, r in baseline_crisis.items() if r is not None]

# --- 그리드 탐색 ---
grid = pl.enumerate_weight_grid(selected_assets, step=float(grid_step))
grid = [g for g in grid if set(g) != {pl.CASH}]  # 100% 현금 제외
progress = st.progress(0.0, text=f"그리드 탐색 중 (0/{len(grid)})...")
rows = []
runs = {}
for i, weights in enumerate(grid):
    basket = pl.build_basket_series(asset_prices, weights)
    result = _run(basket)
    v = _values(result.curve)
    s = result.summary
    crisis_rets = {
        n: pl.window_return(v, *pl.CRISIS_WINDOWS[n]) for n in active_windows
    }
    valid = [r for r in crisis_rets.values() if r is not None]
    avg_crisis = sum(valid) / len(valid) if valid else 0.0
    base_valid = [baseline_crisis[n] for n in active_windows if baseline_crisis[n] is not None]
    base_avg = sum(base_valid) / len(base_valid) if base_valid else 0.0
    # 위기 방어 = 위기 '구간' 성적 절반 + 전체 MDD 절반 — 지정 위기만
    # 피하고 다른 데서 깊게 깨지는 조합이 1위를 가져가지 못하게.
    d_score = 0.5 * pl.defense_score(avg_crisis, base_avg) + 0.5 * pl.mdd_score(
        s.max_drawdown_pct, baseline_result.summary.max_drawdown_pct
    )
    g_score = pl.growth_score(s.cagr_pct, baseline_cagr)
    total_score = (1 - crisis_lambda) * g_score + crisis_lambda * d_score
    label = pl.label_weights(weights, ASSET_LABELS)
    rows.append(
        {
            "weights": weights,
            "방어 바스켓": label,
            "종합 점수": total_score,
            "성장 점수": g_score,
            "위기 방어 점수": d_score,
            "최종 평가액": s.final_value,
            "CAGR": s.cagr_pct,
            "MDD": s.max_drawdown_pct,
            "평균 위기 수익률": avg_crisis,
            "_crisis": crisis_rets,
        }
    )
    runs[label] = result
    progress.progress((i + 1) / len(grid), text=f"그리드 탐색 중 ({i + 1}/{len(grid)})...")
progress.empty()

rows.sort(key=lambda r: r["종합 점수"], reverse=True)
best = rows[0]
best_result = runs[best["방어 바스켓"]]
best_v = _values(best_result.curve)

# --- 결과: 헤드라인 ---
st.subheader(f"🏆 최적 포트폴리오: {aggressive_ticker} {aggressive_weight:.0f}% + 방어 [{best['방어 바스켓']}]")
st.caption(
    f"{effective_start} → {end_date} · 조합 {len(grid)}개 탐색 · "
    f"λ={crisis_lambda:.2f} · 기준 = 방어 100% 주식 (같은 설정) · "
    f"레짐 필터 {'ON' if risk_flags is not None else 'OFF'}"
)
m1, m2, m3, m4 = st.columns(4)
m1.metric("종합 점수", f"{best['종합 점수']:.0f}점")
m2.metric(
    "위기 방어 점수",
    f"{best['위기 방어 점수']:.0f}점",
    help="위기 구간 성적 50% + 전체 MDD 50%. 100=무손실, 0=주식 방어와 동일",
)
m3.metric("성장 점수", f"{best['성장 점수']:.0f}점", help="100=주식 방어와 동일 CAGR")
m4.metric("최종 평가액", f"{best['최종 평가액']:,.0f}")
m5, m6, m7, m8 = st.columns(4)
m5.metric("CAGR", f"{best['CAGR']:+.2%}")
m6.metric("MDD", f"{best['MDD']:.1%}")
m7.metric("평균 위기 수익률", f"{best['평균 위기 수익률']:+.1%}")
m8.metric(
    "vs 주식 방어",
    f"{best['최종 평가액'] / baseline_result.summary.final_value - 1.0:+.1%}",
    help=f"기준 최종 {baseline_result.summary.final_value:,.0f}",
)

# --- 리더보드 ---
st.subheader(f"리더보드 (상위 {min(int(top_n), len(rows))} / 전체 {len(rows)}개 조합)")
st.dataframe(
    [
        {
            "순위": i + 1,
            "방어 바스켓": r["방어 바스켓"],
            "종합": f"{r['종합 점수']:.0f}",
            "성장": f"{r['성장 점수']:.0f}",
            "위기 방어": f"{r['위기 방어 점수']:.0f}",
            "최종 평가액": f"{r['최종 평가액']:,.0f}",
            "CAGR": f"{r['CAGR']:+.2%}",
            "MDD": f"{r['MDD']:.1%}",
            "평균 위기": f"{r['평균 위기 수익률']:+.1%}",
        }
        for i, r in enumerate(rows[: int(top_n)])
    ],
    use_container_width=True,
    hide_index=True,
)

# --- 효율 프론티어 산점도 ---
st.subheader("전 조합 지형도 — MDD vs CAGR")
scatter_fig = go.Figure()
scatter_fig.add_trace(
    go.Scatter(
        x=[abs(r["MDD"]) for r in rows],
        y=[r["CAGR"] for r in rows],
        mode="markers",
        marker=dict(
            size=8,
            color=[r["종합 점수"] for r in rows],
            colorscale="Viridis",
            colorbar=dict(title="종합 점수"),
        ),
        text=[
            f"{r['방어 바스켓']}<br>종합 {r['종합 점수']:.0f} · "
            f"성장 {r['성장 점수']:.0f} · 위기 {r['위기 방어 점수']:.0f}"
            for r in rows
        ],
        hovertemplate="%{text}<br>MDD %{x:.1%} · CAGR %{y:.2%}<extra></extra>",
        name="조합",
    )
)
top3 = rows[:3]
scatter_fig.add_trace(
    go.Scatter(
        x=[abs(r["MDD"]) for r in top3],
        y=[r["CAGR"] for r in top3],
        mode="markers+text",
        marker=dict(
            symbol="star",
            size=[18, 13, 13],
            color=["#FFD600", "#B0BEC5", "#8D6E63"],
            line=dict(width=1, color="#212121"),
        ),
        text=["1위", "2위", "3위"],
        textposition="top center",
        customdata=[r["방어 바스켓"] for r in top3],
        hovertemplate="%{text}: %{customdata}<br>MDD %{x:.1%} · CAGR %{y:.2%}<extra></extra>",
        name="상위 3 조합",
    )
)
scatter_fig.add_trace(
    go.Scatter(
        x=[abs(baseline_result.summary.max_drawdown_pct)],
        y=[baseline_cagr],
        mode="markers+text",
        marker=dict(symbol="diamond", size=13, color="#E53935"),
        text=["주식 방어 기준"], textposition="bottom center",
        name="기준",
    )
)
scatter_fig.update_layout(
    template="plotly_dark", height=450,
    xaxis=dict(title="MDD (절대값)", tickformat=".0%"),
    yaxis=dict(title="CAGR", tickformat=".0%"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    margin=dict(l=50, r=50, t=40, b=40),
)
st.plotly_chart(scatter_fig, use_container_width=True)

# --- 위기 구간 상세 점수표 ---
st.subheader("위기 구간별 성적표 (최적 vs 기준)")
crisis_rows = []
for name in active_windows:
    r_best = best["_crisis"].get(name)
    r_base = baseline_crisis[name]
    if r_best is None or r_base is None:
        continue
    crisis_rows.append(
        {
            "이벤트": name,
            "최적 포트폴리오": f"{r_best:+.1%}",
            "주식 방어 기준": f"{r_base:+.1%}",
            "구간 점수": f"{pl.defense_score(r_best, r_base):.0f}점",
        }
    )
st.dataframe(crisis_rows, use_container_width=True, hide_index=True)

# --- Equity curve ---
log_scale = st.toggle("로그 스케일", value=True)
eq_fig = go.Figure()
eq_fig.add_trace(
    go.Scatter(
        x=best_v.index, y=best_v.values, mode="lines",
        name=f"최적: {best['방어 바스켓']}",
        line=dict(color="#FFD600", width=2.5),
    )
)
eq_fig.add_trace(
    go.Scatter(
        x=baseline_v.index, y=baseline_v.values, mode="lines",
        name="기준: 방어 100% 주식",
        line=dict(color="#E53935", width=1.4, dash="dash"),
    )
)
for color, (name, points) in zip(
    ["#9E9E9E", "#AB47BC"],
    list(best_result.benchmark_curves.items())[:2],
):
    eq_fig.add_trace(
        go.Scatter(
            x=[p.date for p in points], y=[p.equity for p in points],
            mode="lines", name=name,
            line=dict(color=color, width=1, dash="dot"),
        )
    )
risk_off_spans = []
span_start = None
for pt in best_result.curve:
    if not pt.risk_on and span_start is None:
        span_start = pt.date
    elif pt.risk_on and span_start is not None:
        risk_off_spans.append((span_start, pt.date))
        span_start = None
if span_start is not None:
    risk_off_spans.append((span_start, best_result.curve[-1].date))
for x0, x1 in risk_off_spans:
    eq_fig.add_vrect(
        x0=x0, x1=x1, fillcolor="rgba(229,57,53,0.10)",
        line_width=0, layer="below",
    )
eq_fig.update_layout(
    title="평가액 추이 (붉은 음영 = risk-off)",
    template="plotly_dark", height=500,
    yaxis_type="log" if log_scale else "linear",
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    margin=dict(l=50, r=50, t=80, b=30),
)
st.plotly_chart(eq_fig, use_container_width=True)

# --- Drawdown 비교 ---
dd_fig = go.Figure()
for name, v, color, width in [
    (f"최적: {best['방어 바스켓']}", best_v, "#FFD600", 2.0),
    ("기준: 방어 100% 주식", baseline_v, "#E53935", 1.2),
]:
    dd = v / v.cummax() - 1.0
    dd_fig.add_trace(
        go.Scatter(x=dd.index, y=dd.values, mode="lines", name=name,
                   line=dict(color=color, width=width))
    )
dd_fig.update_layout(
    title="낙폭 비교", template="plotly_dark", height=330,
    yaxis_tickformat=".0%",
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    margin=dict(l=50, r=50, t=60, b=30),
)
st.plotly_chart(dd_fig, use_container_width=True)

# --- 최적 바스켓 구성 ---
st.subheader("최적 방어 바스켓 구성")
pie_fig = go.Figure(
    go.Pie(
        labels=[ASSET_LABELS.get(k, k) for k, w in best["weights"].items() if w > 0],
        values=[w for w in best["weights"].values() if w > 0],
        hole=0.45,
    )
)
pie_fig.update_layout(template="plotly_dark", height=330, margin=dict(l=50, r=50, t=30, b=30))
c1, c2 = st.columns([1, 1])
c1.plotly_chart(pie_fig, use_container_width=True)
with c2:
    st.markdown(
        f"""
**실전 구성 (자본 {initial_capital:,.0f} 기준)**

- **{aggressive_ticker} {aggressive_weight:.0f}%** — 공격 사이드
- 방어 사이드 {100 - aggressive_weight:.0f}%를 아래로 분할:
"""
        + "\n".join(
            f"  - {ASSET_LABELS.get(k, k)}: 전체의 "
            f"{(100 - aggressive_weight) * w / 100:.1f}%"
            for k, w in sorted(best["weights"].items(), key=lambda kv: -kv[1])
            if w > 0
        )
        + f"""

방어 바스켓 내부는 고정비중(일별 근사 — 실전은 월 1회면 충분),
공격↔방어 간에는 **±{band_pct:.0f}% 밴드 리밸런싱** + 레짐 필터 적용.
"""
    )
