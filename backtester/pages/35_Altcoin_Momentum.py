from datetime import date, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data.adapters.cached_market_data import CachedMarketDataAdapter
from data.adapters.yfinance_adapter import YFinanceAdapter
from strategy.adapters.risk_metrics import compute_risk_metrics
from strategy.adapters.volatility_breakout_strategy import (
    VolatilityBreakoutStrategy,
)
from strategy.domain.models import TossFeeSchedule, VolBreakoutConfig

st.set_page_config(page_title="알트코인 신규상장 모멘텀", layout="wide")
st.title("알트코인 신규상장 모멘텀 — 미성숙 시장 엣지 검증")
st.caption(
    "가설: 변동성 돌파(시가+k×전일레인지 돌파 매수, 당일 청산)의 엣지는 "
    "**시장이 미성숙할수록 크다**. 각 코인의 히스토리를 **상장(데이터 "
    "시작) 후 첫 N년 vs 이후**로 잘라 같은 규칙을 돌려 비교한다. "
    "검증 결과(2026-08): 8개 코인 중앙값 기준 초기 2년 CAGR 162% vs "
    "이후 67% — 단 약세장에 상장한 코인(AVAX)은 미성숙 프리미엄이 "
    "아니라 미성숙 리스크를 먹는다. 수수료·슬리피지(양방향)·양도세 반영."
)
st.caption(
    "⚠️ **유동성 경고**: 슬리피지 기본 5bp는 BTC/ETH급 유동성 기준이다. "
    "안유명한 코인의 실제 스프레드(25~50bp+)를 넣으면 코호트 중앙값이 "
    "+33% → **-14%**로 뒤집힌다 (2026-08 스트레스 테스트). 소형 코인 "
    "결과는 슬리피지를 25bp 이상으로 올려 보수적으로 읽을 것. 목록은 "
    "생존 코인만 포함(생존편향)."
)

from pages._shared.coins import DEFAULT_COINS, RECENT_COINS  # noqa: E402

with st.sidebar:
    st.header("코인 / 기간")
    coins_raw = st.multiselect(
        "코인 (yfinance 심볼, 직접 추가 가능)",
        [*DEFAULT_COINS, *RECENT_COINS],
        default=DEFAULT_COINS,
        accept_new_options=True,
        help="목록에는 2023~2025 상장 코인(심볼 검증 완료)이 포함돼 "
        "있다. 2026년 상장 등 새 코인은 yfinance 심볼을 직접 타이핑 — "
        "이름 충돌 시 숫자 접미사 심볼이 정본(예: HYPE32196-USD). "
        "데이터 시작일 = 근사 상장 시점으로 간주.",
    )
    include_recent = st.checkbox(
        "신규 상장(2023~2025) 전부 추가",
        value=False,
        help="검증된 신규 상장 코인 22개를 선택 목록에 일괄 추가. "
        "상장 연도별 미성숙 엣지를 보려면 이걸 켜고 초기 구간을 "
        "1년 정도로 줄여볼 것.",
    )
    end_date = st.date_input(
        "End Date", value=date.today(),
        min_value=date(2015, 1, 1), max_value=date.today(),
    )
    early_years = st.number_input(
        "초기 구간 (년)", value=2.0, min_value=0.5, max_value=5.0, step=0.5,
        help="데이터 시작 후 이 기간을 '미성숙기'로 정의. 2024~2026 "
        "상장 코인은 아직 히스토리가 짧아 '이후' 구간이 비어 있을 수 "
        "있다 (초기 성적만 표시).",
    )
    min_days = st.number_input(
        "최소 데이터 (일)", value=60, min_value=30, max_value=500, step=10,
        help="이보다 짧은 코인은 제외. 2026년 상장 코인을 보려면 낮게 유지.",
    )
    initial_capital = st.number_input(
        "Initial Capital", value=100_000_000, min_value=1_000, step=10_000_000
    )

    st.header("돌파 규칙")
    k = st.number_input("돌파 계수 k", value=0.7, min_value=0.1, max_value=2.0, step=0.1)
    leverage = st.number_input(
        "레버리지", value=1.0, min_value=0.5, max_value=4.0, step=0.5,
        help="빌린 부분에 펀딩비 연 10%가 자동 차감된다 (1배면 0).",
    )

    st.header("비용 / 세금")
    commission_pct = st.number_input(
        "수수료 (%, 편도)", value=0.05, min_value=0.0, max_value=1.0,
        step=0.01, format="%.2f",
    )
    slippage_bp = st.number_input(
        "슬리피지 (bp, 편도)", value=5.0, min_value=0.0, max_value=100.0, step=1.0
    )
    tax_pct = st.number_input(
        "양도소득세율 (%)", value=22.0, min_value=0.0, max_value=50.0, step=1.0,
        help="국내 가상자산 과세 시행 가정. 0으로 두면 비과세.",
    )

    run_btn = st.button("Run Backtest", type="primary", use_container_width=True)

if run_btn:
    coins = {str(c).strip().upper() for c in coins_raw if str(c).strip()}
    if include_recent:
        coins |= set(RECENT_COINS)
    coins = sorted(coins)
    if not coins:
        st.error("코인을 최소 1개 선택해줘.")
        st.stop()
    market_data = CachedMarketDataAdapter(YFinanceAdapter())
    fee = TossFeeSchedule(
        buy_commission_pct=commission_pct / 100.0,
        sell_commission_pct=commission_pct / 100.0,
        sec_fee_pct=0.0,
    )

    rows = []
    curves: dict[str, object] = {}
    progress = st.progress(0.0, text="코인별 백테스트 중...")
    for i, coin in enumerate(coins):
        try:
            df = market_data.fetch_ohlcv(coin, date(2014, 1, 1), end_date)
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            df.index = df.index.normalize()
        except Exception as e:
            st.warning(f"{coin} fetch 실패 — 제외: {e}")
            continue
        if len(df) < int(min_days):
            st.warning(f"{coin}: 데이터 {len(df)}봉 — 최소 {int(min_days)}일 미달로 제외.")
            continue
        t0 = df.index[0].date()
        cut = t0 + timedelta(days=int(early_years * 365.25))

        def _run(seg, label):
            # 신규 상장 코인의 짧은 구간도 평가 가능하게 하한을 낮게.
            if len(seg) < 40:
                return None
            return VolatilityBreakoutStrategy().execute(
                seg,
                VolBreakoutConfig(
                    ticker=coin,
                    start_date=seg.index[0].date(),
                    end_date=seg.index[-1].date(),
                    initial_capital=float(initial_capital),
                    k=float(k), leverage=float(leverage),
                    slippage_bp=float(slippage_bp),
                    financing_annual_rate=0.10,
                    fee_schedule=fee,
                    capital_gains_tax_pct=tax_pct / 100.0,
                ),
            )

        early = _run(df[df.index < pd.Timestamp(cut)], "early")
        late = _run(df[df.index >= pd.Timestamp(cut)], "late")
        full = _run(df, "full")
        if full is None:
            continue
        rows.append(
            {
                "코인": coin,
                "데이터 시작": t0,
                "초기 CAGR": early.summary.cagr_pct if early else None,
                "초기 MDD": early.summary.max_drawdown_pct if early else None,
                "이후 CAGR": late.summary.cagr_pct if late else None,
                "이후 MDD": late.summary.max_drawdown_pct if late else None,
                "전체 세후 청산": round(full.liquidation.final_value_after_tax),
                "전체 CAGR": full.summary.cagr_pct,
                "트레이드": len(full.trades),
                "파산": "💀" if full.ruined else "",
            }
        )
        curves[coin] = full
        progress.progress((i + 1) / len(coins), text=f"코인별 백테스트 중 ({i + 1}/{len(coins)})...")
    progress.empty()
    if not rows:
        st.error("실행 가능한 코인이 없어.")
        st.stop()
    st.session_state["alt_state"] = {"rows": rows, "curves": curves,
                                     "early_years": float(early_years)}

_state = st.session_state.get("alt_state")
if _state is None:
    st.info("코인과 규칙을 정하고 **Run Backtest**를 눌러줘.")
    st.stop()
rows = _state["rows"]
curves = _state["curves"]
early_years_used = _state["early_years"]

# --- 요약 ---
early_vals = [r["초기 CAGR"] for r in rows if r["초기 CAGR"] is not None]
late_vals = [r["이후 CAGR"] for r in rows if r["이후 CAGR"] is not None]
m1, m2, m3 = st.columns(3)
m1.metric(
    f"초기 {early_years_used:g}년 CAGR 중앙값",
    f"{pd.Series(early_vals).median():+.1%}" if early_vals else "—",
)
m2.metric(
    "이후 CAGR 중앙값",
    f"{pd.Series(late_vals).median():+.1%}" if late_vals else "—",
)
ratio = (
    pd.Series(early_vals).median() / max(pd.Series(late_vals).median(), 1e-9)
    if early_vals and late_vals and pd.Series(late_vals).median() > 0
    else None
)
m3.metric("미성숙 프리미엄 (배)", f"{ratio:.1f}배" if ratio else "—")

# --- 코인별 테이블 ---
st.subheader("코인별 초기 vs 이후")
st.dataframe(
    rows, use_container_width=True, hide_index=True,
    column_config={
        "초기 CAGR": st.column_config.NumberColumn(format="percent"),
        "초기 MDD": st.column_config.NumberColumn(format="percent"),
        "이후 CAGR": st.column_config.NumberColumn(format="percent"),
        "이후 MDD": st.column_config.NumberColumn(format="percent"),
        "전체 CAGR": st.column_config.NumberColumn(format="percent"),
        "전체 세후 청산": st.column_config.NumberColumn(format="localized"),
    },
)
st.caption(
    "**해석 주의**: 초기 구간이 우월한 건 '미성숙 프리미엄'이지만, "
    "약세장에 상장한 코인은 반대로 깨진다(AVAX 사례). 신생 코인에 "
    "적용할 땐 상승 추세 확인(예: 20일선 위)과 병행할 것. 레버리지 "
    "펀딩비는 미반영."
)

# --- 초기 vs 이후 CAGR 막대 ---
bar = go.Figure()
coins_axis = [r["코인"] for r in rows]
bar.add_trace(go.Bar(
    x=coins_axis, y=[r["초기 CAGR"] for r in rows],
    name=f"초기 {early_years_used:g}년", marker_color="#2196F3",
))
bar.add_trace(go.Bar(
    x=coins_axis, y=[r["이후 CAGR"] for r in rows],
    name="이후", marker_color="#9E9E9E",
))
bar.update_layout(
    template="plotly_dark", height=380, barmode="group",
    yaxis_tickformat=".0%", yaxis_title="CAGR",
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    margin=dict(l=50, r=50, t=40, b=40),
)
st.plotly_chart(bar, use_container_width=True)

# --- 선택 코인 상세 ---
st.subheader("코인 상세")
sel = st.selectbox("코인 선택", list(curves))
r = curves[sel]
log_scale = st.toggle("로그 스케일", value=True)
eq = go.Figure()
eq.add_trace(go.Scatter(
    x=[p.date for p in r.equity_curve], y=[p.equity for p in r.equity_curve],
    mode="lines", name=r.summary.name, line=dict(color="#2196F3", width=2),
))
eq.update_layout(
    template="plotly_dark", height=400,
    yaxis_type="log" if log_scale else "linear",
    xaxis_title="Date", yaxis_title="Equity",
    margin=dict(l=50, r=50, t=40, b=30),
)
st.plotly_chart(eq, use_container_width=True)
yr_rows = [
    {"연도": y, "수익률": v} for y, v in sorted(r.yearly_returns.items())
]
c1, c2 = st.columns([1, 2])
with c1:
    st.dataframe(
        yr_rows, use_container_width=True, hide_index=True,
        column_config={"수익률": st.column_config.NumberColumn(format="percent")},
    )
with c2:
    values = pd.Series(
        [p.equity for p in r.equity_curve],
        index=pd.to_datetime([p.date for p in r.equity_curve]),
    )
    m = compute_risk_metrics(values)
    st.markdown(
        f"**{sel}** — 승률 {r.win_rate:.1%} · 트레이드 {len(r.trades):,}회 · "
        f"추정 켈리 f* {r.kelly_fraction:.2f}\n\n"
        f"등급 **{m['grade']}** · 최장 수면기간 "
        f"{m['longest_underwater_days'] / 365.25:.1f}년 · "
        f"Sharpe {m['sharpe']:.2f} · Calmar {m['calmar']:.2f} · "
        f"최악 연도 {m['worst_year']:+.1%}"
    )
