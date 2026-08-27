from datetime import date

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data.adapters.cached_market_data import CachedMarketDataAdapter
from data.adapters.composed_market_data import build_default_market_data
from data.adapters.regular_session_filter import RegularSessionFilterAdapter
from data.adapters.yfinance_adapter import YFinanceAdapter
from strategy.adapters.infinite_buying_strategy import InfiniteBuyingStrategy
from strategy.adapters.risk_metrics import compute_risk_metrics
from pages._shared.formatting import fmt_price
from strategy.domain.models import InfiniteBuyingConfig, TossFeeSchedule

st.set_page_config(page_title="무한매수법", layout="wide")
st.title("라오어 무한매수법 — 40분할 LOC 백테스트")
st.caption(
    "사이클 원금을 40분할해 매일 **큰수 LOC T/2 + 평단 LOC T/2** 매수, "
    "평단 +10% 지정가 익절(v2.2는 25% 쿼터매도 + 75% 지정가). LOC는 "
    "종가 체결이라 일봉으로 정확하고, **지정가 익절은 15m 인트라데이 "
    "봉으로 체결 시점·갭을 판정**한다(데이터 없는 구간은 일봉 시가/고가 "
    "근사). 토스 수수료 + 양도세 22% 반영."
)
st.caption("가격은 **분할·배당 조정가** 기준 — 실제 당시 호가와 다르다 (예: SOXL 2010년 표시 $0.66 = 실제 $40, 누적 60배 분할 반영). 규칙이 전부 %기반이라 수익률 결과는 동일하며, 호가단위·정수주 제약은 무시한다(1억 규모에서 오차 <0.01%).")

INCEPTION = {
    "TQQQ": date(2010, 2, 11),
    "SOXL": date(2010, 3, 11),
    "UPRO": date(2009, 6, 25),
    # 3배 신흥국 (Direxion Bull 3X)
    "KORU": date(2013, 4, 10),   # 한국
    "EDC": date(2008, 12, 17),   # 신흥국 전체 (MSCI EM)
    "YINN": date(2009, 12, 3),   # 중국 FTSE China 50
    "INDL": date(2010, 3, 11),   # 인도
    "MEXX": date(2017, 5, 3),    # 멕시코
}

with st.sidebar:
    st.header("종목 / 기간")
    ticker_choice = st.selectbox(
        "종목",
        [*INCEPTION, "직접 입력"],
        index=0,
        help="KORU(한국)·EDC(신흥국)·YINN(중국)·INDL(인도)·MEXX(멕시코) "
        "= 3배 신흥국 레버리지. '직접 입력'으로 임의 티커도 가능.",
    )
    if ticker_choice == "직접 입력":
        ticker = st.text_input("티커", value="KORU").strip().upper()
    else:
        ticker = ticker_choice
    default_start = INCEPTION.get(ticker, date(2010, 2, 11))
    start_date = st.date_input(
        "Start Date", value=default_start,
        min_value=date(2008, 12, 17), max_value=date.today(),
        help="종목 상장일 이전으로 잡아도 데이터 있는 구간부터 자동 시작.",
    )
    end_date = st.date_input(
        "End Date", value=date.today(),
        min_value=date(2008, 12, 17), max_value=date.today(),
    )
    initial_capital = st.number_input(
        "Initial Capital", value=100_000_000, min_value=1_000, step=10_000_000
    )

    st.header("무한매수 파라미터")
    version = st.selectbox(
        "버전", ["v2.2", "v2.1"], index=0,
        help="v2.1: 전량 +10% 지정가. v2.2: 25%는 +5% LOC 쿼터매도, "
        "75%는 +10% 지정가.",
    )
    divisions = st.number_input("분할 수", value=40, min_value=10, max_value=100, step=5)
    target_pct = st.number_input(
        "익절 목표 (%)", value=10.0, min_value=2.0, max_value=30.0, step=1.0
    )
    depletion = st.selectbox(
        "40분할 소진 시", ["홀드 (매도 대기)", "전량 손절 후 재시작"], index=0,
        help="원금을 다 쓴 뒤 행동. 홀드는 반등까지 대기(원조 방식), "
        "손절은 종가 전량 매도 후 다음 날 새 사이클.",
    )
    use_intraday = st.checkbox(
        "15m 인트라데이 체결 판정", value=True,
        help="유료 데이터 소스에서 15m 봉을 가져와 지정가 익절의 체결 "
        "시점을 봉 단위로 판정. 데이터가 없는 과거 구간은 자동으로 "
        "일봉 근사로 폴백.",
    )

    st.header("수수료 / 세금")
    commission_pct = st.number_input(
        "거래수수료 (%, 매수·매도 각각)",
        value=0.10, min_value=0.0, max_value=1.0, step=0.01, format="%.2f",
    )
    tax_pct = st.number_input(
        "양도소득세율 (%)", value=22.0, min_value=0.0, max_value=50.0, step=1.0
    )
    tax_deduction = st.number_input(
        "연간 기본공제", value=2_500_000, min_value=0, step=500_000
    )

    run_btn = st.button("Run Backtest", type="primary", use_container_width=True)

if run_btn:
    daily_source = CachedMarketDataAdapter(YFinanceAdapter())
    with st.spinner(f"Fetching {ticker} (일봉)..."):
        try:
            daily = daily_source.fetch_ohlcv(ticker, start_date, end_date)
        except Exception as e:
            st.error(f"일봉 fetch 실패: {e}")
            st.stop()
    if daily.index.tz is not None:
        daily.index = daily.index.tz_localize(None)
    daily.index = daily.index.normalize()

    intraday = None
    if use_intraday:
        with st.spinner(f"Fetching {ticker} 15m (인트라데이 — 가능한 구간만)..."):
            try:
                intraday_source = RegularSessionFilterAdapter(
                    build_default_market_data()
                )
                intraday = intraday_source.fetch_ohlcv(
                    ticker, start_date, end_date, interval="15m"
                )
                if intraday is not None and len(intraday) == 0:
                    intraday = None
            except Exception as e:
                st.warning(f"15m 데이터 fetch 실패 — 일봉 근사로 진행: {e}")
    intraday_note = None
    if use_intraday and intraday is not None:
        intraday_note = (
            f"15m 데이터 구간: {intraday.index[0].date()} ~ "
            f"{intraday.index[-1].date()} ({len(intraday):,}봉) — 그 이전은 "
            "일봉 시가/고가 근사."
        )

    config = InfiniteBuyingConfig(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        initial_capital=float(initial_capital),
        divisions=int(divisions),
        version=version,
        target_profit_pct=target_pct / 100.0,
        depletion_mode="stop_loss" if depletion.startswith("전량") else "hold",
        fee_schedule=TossFeeSchedule(
            buy_commission_pct=commission_pct / 100.0,
            sell_commission_pct=commission_pct / 100.0,
        ),
        capital_gains_tax_pct=tax_pct / 100.0,
        tax_deduction=float(tax_deduction),
    )

    with st.spinner("Running backtest..."):
        try:
            result = InfiniteBuyingStrategy().execute(
                daily, config, intraday=intraday
            )
        except Exception as e:
            st.error(f"Backtest failed: {e}")
            st.stop()
    # 세션에 보관 — 사이클 선택 등 위젯 조작으로 rerun 되어도 백테스트를
    # 다시 돌리지 않고 결과 화면을 유지한다.
    st.session_state["ib_state"] = {
        "result": result, "daily": daily, "intraday_note": intraday_note,
    }

_state = st.session_state.get("ib_state")
if _state is None:
    st.info("좌측에서 설정 후 **Run Backtest**를 눌러줘.")
    st.stop()
result = _state["result"]
daily = _state["daily"]
if _state["intraday_note"]:
    st.caption(_state["intraday_note"])
# 표시용 값들은 실행 시점의 config에서 복원 (사이드바를 바꿔도 화면은
# 마지막 실행 기준 — 다시 반영하려면 Run Backtest).
config = result.config
ticker = config.ticker
target_pct = config.target_profit_pct * 100.0
start_date, end_date = config.start_date, config.end_date

s = result.summary
liq = result.liquidation
closed = [c for c in result.cycles if c.outcome != "open"]
wins = [c for c in closed if c.pnl > 0]
depleted_cycles = [c for c in result.cycles if c.depleted]

st.subheader(s.name)
st.caption(
    f"{result.curve[0].date} → {result.curve[-1].date} · 인트라데이 체결 "
    f"판정 적용 {result.intraday_days:,}거래일 · 체결: LOC=종가, "
    "지정가=장중 터치(갭 오픈 시 시가)"
)

m1, m2, m3, m4 = st.columns(4)
m1.metric("최종 평가액 (세전)", f"{liq.final_value_pre_tax:,.0f}")
m2.metric("세후 청산 가치", f"{liq.final_value_after_tax:,.0f}")
m3.metric("CAGR (세전)", f"{s.cagr_pct:+.2%}")
m4.metric("최대 낙폭 (MDD)", f"{s.max_drawdown_pct:.1%}")

m5, m6, m7, m8 = st.columns(4)
m5.metric(
    "사이클",
    f"{len(closed)}회 완료",
    help=f"진행 중 {len(result.cycles) - len(closed)}회 포함 총 {len(result.cycles)}회.",
)
m6.metric(
    "사이클 승률",
    f"{len(wins) / len(closed):.0%}" if closed else "—",
    help="익절로 끝난 사이클 비율.",
)
m7.metric(
    "40분할 소진", f"{len(depleted_cycles)}회",
    help="원금을 전부 투입하고도 익절가에 못 간 사이클 수.",
)
m8.metric("총 수수료 / 세금", f"{liq.total_fees:,.0f} / {liq.total_tax:,.0f}")

# --- 비교 테이블 ---
st.subheader("전략 vs 벤치마크 (세후 청산 기준)")
after_tax_ret = liq.final_value_after_tax / config.initial_capital - 1.0
rows = [
    {
        "포트폴리오": s.name,
        "최종 평가액 (세전)": f"{liq.final_value_pre_tax:,.0f}",
        "세후 청산가": f"{liq.final_value_after_tax:,.0f}",
        "세후 수익률": f"{after_tax_ret:+.1%}",
        "CAGR (세전)": f"{s.cagr_pct:+.2%}",
        "MDD": f"{s.max_drawdown_pct:.1%}",
    }
]
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

# --- 평가액 곡선 ---
log_scale = st.toggle("로그 스케일", value=True)
curve_dates = [pt.date for pt in result.curve]
eq_fig = go.Figure()
eq_fig.add_trace(
    go.Scatter(
        x=curve_dates, y=[pt.total for pt in result.curve],
        mode="lines", name=s.name, line=dict(color="#2196F3", width=2.2),
    )
)
for name, points in result.benchmark_curves.items():
    eq_fig.add_trace(
        go.Scatter(
            x=[p.date for p in points], y=[p.equity for p in points],
            mode="lines", name=name,
            line=dict(color="#E53935", width=1.3, dash="dot"),
        )
    )
eq_fig.update_layout(
    template="plotly_dark", height=480,
    yaxis_type="log" if log_scale else "linear",
    xaxis_title="Date", yaxis_title="Portfolio Value",
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    margin=dict(l=50, r=50, t=60, b=30),
)
st.plotly_chart(eq_fig, use_container_width=True)

# --- 가격 vs 평단 + 매매 마커 ---
st.subheader("가격 · 평단 · 익절 체결")
px_fig = go.Figure()
px_fig.add_trace(
    go.Scatter(
        x=daily.index, y=daily["Close"], mode="lines",
        name=f"{ticker} 종가", line=dict(color="#B0BEC5", width=1.1),
    )
)
avg_series = pd.Series(
    [pt.avg_price if pt.avg_price > 0 else None for pt in result.curve],
    index=curve_dates,
)
px_fig.add_trace(
    go.Scatter(
        x=avg_series.index, y=avg_series.values, mode="lines",
        name="평단가", line=dict(color="#FFC107", width=1.4, dash="dash"),
        connectgaps=False,
    )
)
sells = [e for e in result.events if e.kind in ("limit_sell", "quarter_sell")]
stops = [e for e in result.events if e.kind == "stop_loss"]
if sells:
    px_fig.add_trace(
        go.Scatter(
            x=[e.ts for e in sells], y=[e.price for e in sells],
            mode="markers", name="익절 매도",
            marker=dict(symbol="triangle-down", size=8, color="#43A047"),
        )
    )
if stops:
    px_fig.add_trace(
        go.Scatter(
            x=[e.ts for e in stops], y=[e.price for e in stops],
            mode="markers", name="소진 손절",
            marker=dict(symbol="x", size=9, color="#E53935"),
        )
    )
px_fig.update_layout(
    template="plotly_dark", height=420,
    yaxis_type="log" if log_scale else "linear",
    xaxis_title="Date", yaxis_title="Price",
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    margin=dict(l=50, r=50, t=40, b=30),
)
st.plotly_chart(px_fig, use_container_width=True)

# --- 트랜치 투입률 ---
st.subheader("사이클 원금 투입률 (40분할 소진도)")
sp_fig = go.Figure()
sp_fig.add_trace(
    go.Scatter(
        x=curve_dates, y=[pt.tranches_spent_pct for pt in result.curve],
        mode="lines", name="투입률", fill="tozeroy",
        line=dict(color="#FF7043", width=1.2),
    )
)
sp_fig.add_hline(y=1.0, line_dash="dash", line_color="#E53935",
                 annotation_text="소진 (100%)")
sp_fig.update_layout(
    template="plotly_dark", height=280,
    xaxis_title="Date", yaxis_title="투입률", yaxis_tickformat=".0%",
    margin=dict(l=50, r=50, t=40, b=30),
)
st.plotly_chart(sp_fig, use_container_width=True)

# --- 사이클 분포 + 테이블 ---
st.subheader(f"사이클 기록 ({len(result.cycles)}회)")
c1, c2 = st.columns([1, 1])
with c1:
    hist_fig = go.Figure(
        go.Histogram(
            x=[c.trading_days for c in closed],
            nbinsx=30, marker_color="#2196F3",
        )
    )
    hist_fig.update_layout(
        template="plotly_dark", height=300,
        title="사이클 길이 분포 (거래일)",
        xaxis_title="거래일", yaxis_title="사이클 수",
        margin=dict(l=40, r=20, t=50, b=40),
    )
    st.plotly_chart(hist_fig, use_container_width=True)
with c2:
    pnl_fig = go.Figure(
        go.Histogram(
            x=[c.pnl for c in closed], nbinsx=30, marker_color="#43A047",
        )
    )
    pnl_fig.update_layout(
        template="plotly_dark", height=300,
        title="사이클 손익 분포",
        xaxis_title="손익", yaxis_title="사이클 수",
        margin=dict(l=40, r=20, t=50, b=40),
    )
    st.plotly_chart(pnl_fig, use_container_width=True)

events_by_cycle: dict[int, list] = {}
for e in result.events:
    events_by_cycle.setdefault(e.cycle_no, []).append(e)

cycle_rows = [
    {
        "사이클": c.cycle_no,
        "시작": c.start,
        "종료": c.end if c.end else "진행 중",
        "거래일": c.trading_days,
        "매수 횟수": sum(
            1 for e in events_by_cycle.get(c.cycle_no, []) if "buy" in e.kind
        ),
        "최대 투입": round(c.invested_max),
        "손익": round(c.pnl),
        "계좌 수익률": c.pnl_pct,
        "결과": {"profit": "익절", "stop_loss": "손절", "open": "진행 중"}[c.outcome],
        "소진": "O" if c.depleted else "",
    }
    for c in result.cycles
]
st.dataframe(
    cycle_rows,
    use_container_width=True,
    hide_index=True,
    column_config={
        "최대 투입": st.column_config.NumberColumn(format="localized"),
        "손익": st.column_config.NumberColumn(format="localized"),
        "계좌 수익률": st.column_config.NumberColumn(format="percent"),
    },
)

# --- 사이클 상세: 매수 하나하나 전부 ---
st.subheader("사이클 상세 — 매매 전체 기록")
OUTCOME_LABEL = {"profit": "익절", "stop_loss": "손절", "open": "진행 중"}
cycle_options = {
    f"#{c.cycle_no}  {c.start} ~ {c.end if c.end else '진행 중'} · "
    f"{OUTCOME_LABEL[c.outcome]} ({c.pnl:+,.0f})": c.cycle_no
    for c in result.cycles
}
selected_label = st.selectbox("사이클 선택", list(cycle_options), index=0)
selected_no = cycle_options[selected_label]
KIND_LABELS = {
    "start_buy": "시작 매수 (1T)",
    "big_buy": "큰수 LOC 매수 (T/2)",
    "avg_buy": "평단 LOC 매수 (T/2)",
    "quarter_sell": "쿼터 매도 (LOC)",
    "limit_sell": "지정가 익절",
    "stop_loss": "소진 손절",
    "tax": "양도세 납부",
}
detail_rows = [
    {
        "시각": e.ts,
        "구분": KIND_LABELS.get(e.kind, e.kind),
        "매수 회차": e.tranche_no if e.tranche_no else "",
        "체결가": fmt_price(e.price),
        "수량": f"{e.qty:,.2f}" if e.qty else "",
        "금액 (+매수/−매도)": f"{e.notional:,.0f}",
        "체결 후 보유": f"{e.shares_after:,.2f}",
        "체결 후 평단": fmt_price(e.avg_price_after),
        "익절 목표가": fmt_price(e.target_price),
        "원금 투입률": f"{e.spent_pct:.1%}",
        "현금 잔고": f"{e.cash_after:,.0f}",
        "체결 판정": "15m 장중" if e.intraday else "일봉/종가",
    }
    for e in events_by_cycle.get(selected_no, [])
]
st.dataframe(detail_rows, use_container_width=True, hide_index=True)
st.caption(
    "**매수 회차** = 사이클 내 몇 번째 자금 투입인지 (시작 1T 포함, "
    "하루에 큰수+평단이 다 체결되면 2회차씩 증가) · **원금 투입률** = "
    "사이클 원금 대비 누적 매수금 · 익절 목표가 = 체결 후 평단 × "
    f"(1+{target_pct:.0f}%)."
)

all_events_df = pd.DataFrame([e.model_dump() for e in result.events])
st.download_button(
    f"전체 매매 로그 CSV 다운로드 ({len(result.events):,}건)",
    all_events_df.to_csv(index=False).encode("utf-8-sig"),
    file_name=f"infinite_buying_{ticker}_{start_date}_{end_date}.csv",
    mime="text/csv",
)

# --- 리스크 요약 ---
values = pd.Series([pt.total for pt in result.curve], index=pd.to_datetime(curve_dates))
m = compute_risk_metrics(values)
st.caption(
    f"리스크: 등급 **{m['grade']}** · 최장 수면기간 "
    f"{m['longest_underwater_days'] / 365.25:.1f}년 · Sharpe {m['sharpe']:.2f} · "
    f"Calmar {m['calmar']:.2f} · 최악 연도 {m['worst_year']:+.1%}"
)
