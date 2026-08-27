from datetime import date, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data.adapters.cached_market_data import CachedMarketDataAdapter
from data.adapters.fred_adapter import FredCsvAdapter
from data.adapters.yfinance_adapter import YFinanceAdapter
from strategy.adapters import macro_regime as mr
from strategy.adapters.band_rebalance_strategy import (
    CALIBRATED_DRAG,
    BandRebalanceStrategy,
    build_synthetic_leveraged,
    splice_series,
)
from pages._shared.formatting import fmt_price
from strategy.domain.models import BandRebalanceConfig, TossFeeSchedule

st.set_page_config(page_title="Macro Regime Rebalance", layout="wide")
st.title("거시지표 선행 레짐 × 밴드 리밸런싱")
st.caption(
    "금리커브·신용 스프레드·VIX·고용(Sahm)·소비 리스크선호를 0/1로 "
    "**정량화**해 합성 스코어를 만들고, **가격 추세 붕괴 + 거시 악화가 "
    "동시에** 확인될 때만 방어 전환한다. 거시가 건강한 조정·지정학 "
    "이벤트(전쟁 등)에는 반응하지 않아 휩쏘 없이 꾸준히 오르는 것이 "
    "목표. 모든 지표는 발표 지연을 반영해 look-ahead 없음."
)
st.caption(
    "가격은 **분할·배당 조정가** 기준이라 실제 당시 호가와 다르다 — "
    "TQQQ 2010년 표시 $0.4대 = 실제 $25 수준 (누적 60배 분할 반영). "
    "0.30005처럼 보이는 체결가는 에러가 아니라 조정가이며, 규칙이 "
    "전부 %·비중 기반이라 수익률 결과는 실제 가격과 동일하다. "
    "$10 미만 가격은 소수 4자리로 표시한다."
)

ETF_INCEPTION = {
    "TQQQ": date(2010, 2, 11),
    "QLD": date(2006, 6, 21),
    "VOO": date(2010, 9, 9),
    "SPY": date(1993, 1, 29),
    "QQQ": date(1999, 3, 10),
}
LEVERAGE = {"TQQQ": 3.0, "QLD": 2.0}

# 지정학/거시 이벤트 구간 — 전략의 '이벤트 통과 능력' 검증용.
EVENT_WINDOWS = {
    "닷컴 붕괴 (2000-03~2002-10)": ("2000-03-10", "2002-10-09"),
    "9·11 테러 (2001-09)": ("2001-09-10", "2001-10-11"),
    "이라크전 개전 (2003-03)": ("2003-03-18", "2003-05-01"),
    "금융위기 (2007-10~2009-03)": ("2007-10-09", "2009-03-09"),
    "코로나 쇼크 (2020-02~03)": ("2020-02-19", "2020-03-23"),
    "2022 인플레 약세장 (2021-11~2022-12)": ("2021-11-19", "2022-12-28"),
    "우크라이나 침공 (2022-02~06)": ("2022-02-23", "2022-06-30"),
    "이스라엘-하마스 (2023-10~11)": ("2023-10-06", "2023-11-30"),
}

INDICATOR_INFO = {
    "curve": "금리커브 (10Y−3M, FRED T10Y3M) — 역전 시 위험",
    "credit": "신용 스프레드 (무디스 Baa−10Y, FRED BAA10Y) — 200일 평균 위로 확대 시 위험",
    "vix": "VIX 21일 평균 — 25 이상이면 위험",
    "sahm": "고용 Sahm Rule (FRED UNRATE) — 실업률 3개월 평균이 12개월 저점 +0.5%p 이상이면 위험",
    "consumer": "소비 리스크선호 (XLY/XLP) — 200일 평균 아래면 위험",
}

# --- Sidebar ---
with st.sidebar:
    st.header("자산 구성")
    aggressive_ticker = st.selectbox("공격 자산", ["TQQQ", "QLD"], index=0)
    defensive_ticker = st.selectbox("방어 자산", ["VOO", "SPY"], index=0)
    use_synthetic = st.checkbox(
        "상장 이전 구간 백캐스트 (실데이터 스플라이스)",
        value=True,
        help="상장일 이후는 실제 ETF 가격, 이전(1999~)만 QQQ ×N − "
        "실제 T-bill 자금조달비용 − 실측 드래그로 백캐스트해 연결. "
        "닷컴 버블·9·11·이라크전 검증에 필요.",
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

    st.header("기간 / 자본")
    default_start = date(1999, 3, 10) if use_synthetic else date(2010, 9, 9)
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
        "Initial Capital", value=100_000_000, min_value=1_000, step=10_000_000
    )

    st.header("리밸런싱 규칙")
    band_pct = st.number_input(
        "밴드 폭 (%)", value=15.0, min_value=1.0, max_value=50.0, step=1.0
    )
    aggressive_weight = st.number_input(
        "공격 자산 목표 비중 (%)",
        value=50.0,
        min_value=10.0,
        max_value=90.0,
        step=5.0,
    )
    dip_sell_pct = st.number_input(
        "하락 시 방어 자산 매도 비율 (%)",
        value=15.0,
        min_value=1.0,
        max_value=100.0,
        step=1.0,
    )

    st.header("거시 레짐")
    regime_mode = st.selectbox(
        "레짐 판정 방식",
        ["하이브리드 (추세 붕괴 × 거시 확인)", "순수 거시 합성"],
        index=0,
        help="**하이브리드(권장)**: QQQ 200일선 붕괴 AND 거시 합성 "
        "스코어 악화가 동시에 확인될 때만 방어 전환. 거시가 건강한 "
        "기술적 조정에는 버텨서 가격 필터의 휩쏘 비용을 없앤다. "
        "**순수 거시**: 합성 스코어의 히스테리시스만으로 판정.",
    )
    use_indicators = {}
    for key, desc in INDICATOR_INFO.items():
        use_indicators[key] = st.checkbox(desc.split(" — ")[0], value=True, help=desc)
    veto_threshold = st.number_input(
        "거시 거부권 임계값 (하이브리드)",
        value=0.6,
        min_value=0.1,
        max_value=0.9,
        step=0.05,
        help="추세 붕괴 시 합성 스코어가 이 값 미만이어야 방어 전환. "
        "높일수록 필터가 민감해지고(전환 잦음), 낮출수록 확실한 거시 "
        "악화에만 반응.",
    )
    on_threshold = st.number_input(
        "복귀 임계값 (순수 거시)", value=0.65, min_value=0.1, max_value=1.0, step=0.05
    )
    off_threshold = st.number_input(
        "이탈 임계값 (순수 거시)", value=0.45, min_value=0.0, max_value=0.9, step=0.05
    )
    confirm_days = st.number_input(
        "재진입 확인 일수",
        value=10,
        min_value=0,
        max_value=60,
        step=5,
        help="복귀 조건이 N일 연속 유지돼야 재진입 (베어랠리 휩쏘 필터).",
    )
    risk_off_label = st.selectbox(
        "Risk-off 행동", ["방어자산 대피", "현금 대피", "줍줍만 중단"], index=0
    )

    st.header("수수료 / 세금")
    apply_costs = st.checkbox(
        "토스증권 수수료 · 양도소득세 반영",
        value=True,
        help="모든 매매에 거래수수료(매수·매도 각 0.1% + SEC fee), "
        "실현 차익에 연 단위 양도세(기본공제 차감)를 부과. 거시/200SMA/"
        "기존 전략 모두 동일하게 적용된다.",
    )
    commission_pct = st.number_input(
        "거래수수료 (%, 매수·매도 각각)",
        value=0.10, min_value=0.0, max_value=1.0, step=0.01,
        format="%.2f", disabled=not apply_costs,
    )
    tax_pct = st.number_input(
        "양도소득세율 (%)",
        value=22.0, min_value=0.0, max_value=50.0, step=1.0,
        disabled=not apply_costs,
    )
    tax_deduction = st.number_input(
        "연간 기본공제",
        value=2_500_000, min_value=0, step=500_000,
        disabled=not apply_costs,
    )

    run_btn = st.button("Run Backtest", type="primary", use_container_width=True)

if not run_btn:
    st.info("좌측에서 설정 후 **Run Backtest**를 눌러줘.")
    st.stop()

# --- 데이터 가용성 가드 (page 24와 동일 규칙) ---
if not use_synthetic:
    floor = max(ETF_INCEPTION[aggressive_ticker], ETF_INCEPTION[defensive_ticker])
    if start_date < floor:
        st.warning(f"실데이터는 {floor}부터 — 시작일을 당겨서 실행합니다.")
        start_date = floor
elif start_date < ETF_INCEPTION["QQQ"]:
    st.warning(f"합성 모드 기초지수 QQQ는 {ETF_INCEPTION['QQQ']}부터 — 시작일을 당깁니다.")
    start_date = ETF_INCEPTION["QQQ"]

market_data = CachedMarketDataAdapter(YFinanceAdapter())
fred = FredCsvAdapter()
warm_start = start_date - timedelta(days=500)  # 200SMA 워밍업


def _normalize_daily(series: pd.Series) -> pd.Series:
    if series.index.tz is not None:
        series.index = series.index.tz_localize(None)
    series.index = series.index.normalize()
    return series


def _fetch_close(sym: str, start: date) -> pd.Series:
    return _normalize_daily(market_data.fetch_ohlcv(sym, start, end_date)["Close"])


defensive_fetch = defensive_ticker
if use_synthetic and start_date < ETF_INCEPTION[defensive_ticker]:
    defensive_fetch = "SPY"
    st.info(f"합성 모드: {defensive_ticker} 상장 이전 구간은 SPY로 대체.")
aggressive_fetch = "QQQ" if use_synthetic else aggressive_ticker

with st.spinner("Fetching 가격 데이터..."):
    try:
        agg_close = _fetch_close(aggressive_fetch, start_date)
        def_close = _fetch_close(defensive_fetch, start_date)
        qqq_full = _fetch_close("QQQ", warm_start)  # 추세 레짐 지수
    except Exception as e:
        st.error(f"가격 데이터 fetch 실패: {e}")
        st.stop()

if use_synthetic:
    with st.spinner("Fetching ^IRX (자금조달금리) + 실제 ETF..."):
        try:
            irx = _fetch_close("^IRX", start_date)
            synthetic = build_synthetic_leveraged(
                agg_close,
                leverage=LEVERAGE[aggressive_ticker],
                annual_expense=synthetic_expense / 100.0,
                financing_rate=irx / 100.0,
            )
            inception = ETF_INCEPTION[aggressive_ticker]
            real_agg = (
                _fetch_close(aggressive_ticker, max(start_date, inception))
                if end_date > inception
                else None
            )
            agg_close = splice_series(synthetic, real_agg)
        except Exception as e:
            st.error(f"백캐스트 데이터 fetch 실패: {e}")
            st.stop()

# --- 지표 수집 & 정량화 (실패한 지표는 경고 후 제외) ---
components: dict[str, pd.Series] = {}
with st.spinner("Fetching 경제지표 (FRED + yfinance)..."):
    fetchers = {
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
    for key, fetch in fetchers.items():
        if not use_indicators[key]:
            continue
        try:
            components[key] = fetch()
        except Exception as e:
            st.warning(f"지표 '{key}' 로드 실패 — 제외하고 진행: {e}")

if not components:
    st.error("사용 가능한 지표가 없습니다.")
    st.stop()

# --- 합성 스코어 → risk-on 시계열 ---
composite = mr.composite_score(
    components, {k: 1.0 for k in components}, agg_close.index
)
trend_score = mr.score_price_trend(qqq_full).reindex(
    agg_close.index, method="ffill"
)
if regime_mode.startswith("하이브리드"):
    risk_flags = mr.hybrid_risk_on(
        trend_score, composite, float(veto_threshold), int(confirm_days)
    )
else:
    risk_flags = mr.hysteresis_risk_on(
        composite, float(on_threshold), float(off_threshold), int(confirm_days)
    )

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
    risk_off_mode=RISK_OFF_MODES[risk_off_label],
    fee_schedule=TossFeeSchedule(
        buy_commission_pct=commission_pct / 100.0,
        sell_commission_pct=commission_pct / 100.0,
    )
    if apply_costs
    else None,
    capital_gains_tax_pct=tax_pct / 100.0 if apply_costs else 0.0,
    tax_deduction=float(tax_deduction),
)

with st.spinner("Running backtests (거시 / 200SMA / 기존)..."):
    try:
        strategy = BandRebalanceStrategy()
        result = strategy.execute(
            agg_close, def_close, config, risk_on_series=risk_flags
        )
        base_result = strategy.execute(agg_close, def_close, config)
        sma_config = config.model_copy(
            update={"regime_buffer_pct": 0.01, "regime_confirm_days": 15}
        )
        sma_result = strategy.execute(
            agg_close, def_close, sma_config, regime_close=qqq_full
        )
    except Exception as e:
        st.error(f"Backtest failed: {e}")
        st.stop()

s = result.summary
st.subheader(s.name)
st.caption(
    f"{result.curve[0].date} → {result.curve[-1].date} · "
    f"{'실데이터+백캐스트 스플라이스' if use_synthetic else '실제 ETF'} · "
    f"지표 {len(components)}종: {', '.join(components)} · "
    f"{regime_mode}"
)

# --- Headline ---
m1, m2, m3, m4 = st.columns(4)
m1.metric("최종 평가액", f"{s.final_value:,.0f}")
m2.metric("총 수익률", f"{s.total_return_pct:+.1%}")
m3.metric("CAGR", f"{s.cagr_pct:+.2%}")
m4.metric("최대 낙폭 (MDD)", f"{s.max_drawdown_pct:.1%}")

m5, m6, m7, m8 = st.columns(4)
risk_off_count = sum(1 for e in result.events if e.kind == "risk_off")
m5.metric("Risk-off 전환", f"{risk_off_count}회")
m6.metric("줍줍 / 수익실현", f"{result.dip_buy_count} / {result.profit_take_count}회")
m7.metric(
    "vs 기존 전략",
    f"{s.final_value / base_result.summary.final_value - 1.0:+.1%}",
    help=f"기존(필터 없음) 최종 {base_result.summary.final_value:,.0f}",
)
m8.metric(
    "vs 200SMA 필터",
    f"{s.final_value / sma_result.summary.final_value - 1.0:+.1%}",
    help=f"가격 200SMA 필터 최종 {sma_result.summary.final_value:,.0f}",
)

if apply_costs and result.liquidation is not None:
    liq = result.liquidation
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "세후 청산 가치",
        f"{liq.final_value_after_tax:,.0f}",
        help="마지막 날 전량 매도 가정: 매도 수수료 + 양도세 차감.",
    )
    c2.metric(
        "세후 총 수익률",
        f"{liq.final_value_after_tax / config.initial_capital - 1.0:+.1%}",
    )
    c3.metric("총 수수료", f"{liq.total_fees:,.0f}")
    c4.metric(
        "총 양도소득세",
        f"{liq.total_tax:,.0f}",
        help=f"연 단위 정산 + 최종 청산분 {liq.final_tax:,.0f} 포함.",
    )

# --- 비교표 ---
st.subheader("전략 비교")
_after_tax = {
    s.name: result.liquidation,
    sma_result.summary.name: sma_result.liquidation,
    base_result.summary.name: base_result.liquidation,
}
compared = [s, sma_result.summary, base_result.summary, *result.benchmarks]
st.dataframe(
    [
        {
            "포트폴리오": p.name,
            "최종 평가액": f"{p.final_value:,.0f}",
            "세후 청산가": (
                f"{_after_tax[p.name].final_value_after_tax:,.0f}"
                if apply_costs and _after_tax.get(p.name) is not None
                else "—"
            ),
            "총 수익률": f"{p.total_return_pct:+.1%}",
            "CAGR": f"{p.cagr_pct:+.2%}",
            "MDD": f"{p.max_drawdown_pct:.1%}",
        }
        for p in compared
    ],
    use_container_width=True,
    hide_index=True,
)

# --- 이벤트(전쟁/위기) 구간 통과 성적 ---
st.subheader("이벤트 구간 통과 성적")
st.caption(
    "전쟁·위기 이벤트 구간의 수익률. 목표: 이벤트와 무관하게 손실을 "
    "얕게 유지하면서 통과."
)


def _value_series(curve) -> pd.Series:
    return pd.Series(
        [pt.total for pt in curve],
        index=pd.to_datetime([pt.date for pt in curve]),
    )


macro_v = _value_series(result.curve)
sma_v = _value_series(sma_result.curve)
base_v = _value_series(base_result.curve)
event_rows = []
for name, (a, b) in EVENT_WINDOWS.items():
    win = macro_v[a:b]
    if len(win) < 2:
        continue

    def _ret(v: pd.Series) -> str:
        w = v[a:b]
        return f"{w.iloc[-1] / w.iloc[0] - 1.0:+.1%}" if len(w) > 1 else "—"

    event_rows.append(
        {
            "이벤트": name,
            "거시 레짐 전략": _ret(macro_v),
            "200SMA 필터": _ret(sma_v),
            "기존 (필터 없음)": _ret(base_v),
        }
    )
if event_rows:
    st.dataframe(event_rows, use_container_width=True, hide_index=True)
else:
    st.info("백테스트 기간에 포함된 이벤트 구간이 없어.")

# --- Equity curves ---
log_scale = st.toggle("로그 스케일", value=True)
curve_dates = [pt.date for pt in result.curve]

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
        x=curve_dates, y=[pt.total for pt in result.curve],
        mode="lines", name=s.name, line=dict(color="#2196F3", width=2.5),
    )
)
eq_fig.add_trace(
    go.Scatter(
        x=[pt.date for pt in sma_result.curve],
        y=[pt.total for pt in sma_result.curve],
        mode="lines", name="200SMA 가격 필터",
        line=dict(color="#FFA726", width=1.4, dash="dash"),
    )
)
eq_fig.add_trace(
    go.Scatter(
        x=[pt.date for pt in base_result.curve],
        y=[pt.total for pt in base_result.curve],
        mode="lines", name="기존 전략 (필터 없음)",
        line=dict(color="#AB47BC", width=1.4, dash="dash"),
    )
)
BENCH_COLORS = ["#9E9E9E", "#E53935", "#43A047"]
for color, (name, points) in zip(BENCH_COLORS, result.benchmark_curves.items()):
    eq_fig.add_trace(
        go.Scatter(
            x=[p.date for p in points], y=[p.equity for p in points],
            mode="lines", name=name, line=dict(color=color, width=1, dash="dot"),
        )
    )
_shade_risk_off(eq_fig)
eq_fig.update_layout(
    title="평가액 추이 (붉은 음영 = risk-off 구간)",
    template="plotly_dark", height=520,
    yaxis_type="log" if log_scale else "linear",
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    margin=dict(l=50, r=50, t=80, b=30),
)
st.plotly_chart(eq_fig, use_container_width=True)

# --- 합성 스코어 + 컴포넌트 히트맵 ---
st.subheader("거시 합성 스코어")
score_fig = go.Figure()
score_fig.add_trace(
    go.Scatter(
        x=composite.index, y=composite.values,
        mode="lines", name="합성 스코어",
        line=dict(color="#26C6DA", width=1.5),
    )
)
score_fig.add_trace(
    go.Scatter(
        x=trend_score.index, y=trend_score.values * 0.05 + 1.06,
        mode="lines", name="가격 추세 (위=건강)",
        line=dict(color="#FFEE58", width=1),
    )
)
if regime_mode.startswith("하이브리드"):
    score_fig.add_hline(
        y=float(veto_threshold), line_dash="dash", line_color="#E53935",
        annotation_text=f"거부권 임계 {veto_threshold}",
    )
else:
    score_fig.add_hline(y=float(on_threshold), line_dash="dash", line_color="#43A047")
    score_fig.add_hline(y=float(off_threshold), line_dash="dash", line_color="#E53935")
_shade_risk_off(score_fig)
score_fig.update_layout(
    template="plotly_dark", height=320,
    yaxis=dict(range=[-0.05, 1.15], title="score"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    margin=dict(l=50, r=50, t=40, b=30),
)
st.plotly_chart(score_fig, use_container_width=True)

aligned = pd.DataFrame(
    {k: v.reindex(agg_close.index, method="ffill") for k, v in components.items()}
)
heat_fig = go.Figure(
    go.Heatmap(
        z=aligned.T.values,
        x=aligned.index,
        y=[INDICATOR_INFO[k].split(" — ")[0] for k in aligned.columns],
        colorscale=[[0, "#B71C1C"], [1, "#1B5E20"]],
        showscale=False,
        zmin=0, zmax=1,
    )
)
heat_fig.update_layout(
    title="지표별 건강도 (녹색=건강 / 적색=위험)",
    template="plotly_dark", height=260,
    margin=dict(l=50, r=50, t=50, b=30),
)
st.plotly_chart(heat_fig, use_container_width=True)

# --- Drawdown ---
st.subheader("낙폭 (Drawdown)")
dd_fig = go.Figure()
for name, v, color, width, dash in [
    (s.name, macro_v, "#2196F3", 2.0, None),
    ("200SMA 가격 필터", sma_v, "#FFA726", 1.2, "dash"),
    ("기존 전략 (필터 없음)", base_v, "#AB47BC", 1.2, "dash"),
]:
    dd = v / v.cummax() - 1.0
    dd_fig.add_trace(
        go.Scatter(
            x=dd.index, y=dd.values, mode="lines", name=name,
            line=dict(color=color, width=width, dash=dash),
        )
    )
dd_fig.update_layout(
    template="plotly_dark", height=350, yaxis_tickformat=".0%",
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    margin=dict(l=50, r=50, t=60, b=30),
)
st.plotly_chart(dd_fig, use_container_width=True)

# --- 배분 추이 ---
st.subheader("자산 배분 추이")
alloc_fig = go.Figure()
alloc_fig.add_trace(
    go.Scatter(
        x=curve_dates, y=[pt.aggressive_value for pt in result.curve],
        mode="lines", name=f"{aggressive_ticker} (공격)",
        stackgroup="alloc", line=dict(width=0.5, color="#E53935"),
    )
)
alloc_fig.add_trace(
    go.Scatter(
        x=curve_dates, y=[pt.defensive_value for pt in result.curve],
        mode="lines", name=f"{defensive_ticker} (방어)",
        stackgroup="alloc", line=dict(width=0.5, color="#1E88E5"),
    )
)
if any(pt.cash > 0 for pt in result.curve):
    alloc_fig.add_trace(
        go.Scatter(
            x=curve_dates, y=[pt.cash for pt in result.curve],
            mode="lines", name="현금", stackgroup="alloc",
            line=dict(width=0.5, color="#9E9E9E"),
        )
    )
alloc_fig.update_layout(
    template="plotly_dark", height=320,
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    margin=dict(l=50, r=50, t=60, b=30),
)
st.plotly_chart(alloc_fig, use_container_width=True)

# --- 이벤트 테이블 ---
st.subheader(f"리밸런싱/레짐 이벤트 ({len(result.events)}건)")
KIND_LABELS = {
    "dip_buy": "줍줍 매수",
    "profit_take": "수익 실현",
    "risk_off": "Risk-off (방어 전환)",
    "risk_on": "Risk-on (재진입)",
}
if result.events:
    st.dataframe(
        [
            {
                "날짜": e.date,
                "구분": KIND_LABELS.get(e.kind, e.kind),
                f"{aggressive_ticker} 가격": fmt_price(e.aggressive_price),
                "이동 금액(공격 방향 +)": f"{e.traded_amount:,.0f}",
                "총 평가액": f"{e.total_value_after:,.0f}",
                "공격 비중": f"{e.aggressive_weight_after:.1%}",
            }
            for e in result.events
        ],
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info("이 기간엔 이벤트가 없어.")
