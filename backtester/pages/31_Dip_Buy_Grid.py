from datetime import date

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data.adapters.cached_market_data import CachedMarketDataAdapter
from data.adapters.yfinance_adapter import YFinanceAdapter
from pages._shared.formatting import fmt_price
from strategy.adapters.dip_buy_strategy import DipBuyStrategy
from strategy.adapters.multi_dip_buy_strategy import MultiDipBuyStrategy
from strategy.domain.models import (
    DipBuyConfig,
    MultiDipBuyConfig,
    TossFeeSchedule,
)

st.set_page_config(page_title="급락 매수 그리드", layout="wide")
st.title("전일 급락 매수 그리드 — 무한매수법 변형")
st.caption(
    "**전날 −X% 이상 떨어졌으면 다음 날 시가에 계좌 총액의 1/N 매수** "
    "(사이클당 최대 N회 물타기), 보유 중엔 **평단 +Y% 지정가 익절** "
    "상시 대기 (시가 갭이면 시가 체결). 손절 없음. X(급락 기준) × "
    "Y(익절 목표) × **N(분할 수)** 3차원 그리드로 전수 탐색한다. "
    "토스 수수료 + 양도세 22% 반영, 결과는 세후 청산 기준."
)
st.caption(
    "가격은 **분할·배당 조정가** 기준 — 실제 당시 호가와 다르다 "
    "(예: SOXL 2010년 표시 $0.66 = 실제 $40, 누적 60배 분할 반영). "
    "규칙이 전부 %기반이라 수익률 결과는 동일하며, 호가단위·정수주 "
    "제약은 무시한다(1억 규모에서 오차 <0.01%)."
)

TICKERS = ["TQQQ", "SOXL", "UPRO", "KORU", "EDC", "YINN", "INDL", "QQQ", "SPY"]


def _parse_grid(raw: list, as_int: bool = False) -> list:
    """multiselect 값 정규화 — 직접 입력(문자열)과 기본 옵션(숫자) 혼재.

    파싱 불가/비양수 입력은 조용히 버린다.
    """
    out = set()
    for v in raw:
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if f <= 0:
            continue
        out.add(int(round(f)) if as_int else f)
    return sorted(out)


with st.sidebar:
    st.header("종목 / 기간")
    tickers_raw = st.multiselect(
        "종목 (2개 이상 = 한 계좌 공유현금 동시 운용)",
        TICKERS,
        default=["TQQQ"],
        accept_new_options=True,
        help="여러 종목이면 매일 전 종목의 급락 신호를 검사해 낙폭 "
        "깊은 순으로 현금을 배정한다 (종목별 평단·익절 독립). "
        "주의: 3배 ETF끼리는 폭락이 동행해 MDD가 크게 나빠지고, "
        "우하향 종목(신흥국 3배)이 현금을 감금할 수 있다 — 이전 "
        "분석 참조. 목록에 없는 티커는 타이핑으로 추가.",
    )
    tickers = sorted({str(t).strip().upper() for t in tickers_raw if str(t).strip()})
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

    st.header("그리드 범위")
    st.caption("목록에 없는 값은 **직접 타이핑해서 추가**할 수 있다 (숫자만).")
    drop_raw = st.multiselect(
        "급락 기준 −X% (전날 종가 기준)",
        [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0],
        default=[3.0, 4.0, 5.0, 6.0, 7.0],
        accept_new_options=True,
    )
    target_raw = st.multiselect(
        "익절 목표 +Y%",
        [0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 5.0],
        default=[0.8, 1.0, 1.2, 1.5, 2.0, 3.0],
        accept_new_options=True,
    )
    split_raw = st.multiselect(
        "분할 수 N (1/N씩, 최대 N회 스택)",
        [1, 2, 3, 4, 5, 6, 8, 10],
        default=[2, 3, 4],
        accept_new_options=True,
        help="N=1이면 신호 첫날 전액 매수(물타기 없음), N이 클수록 "
        "잘게 나눠 오래 버틴다.",
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
    drop_grid = _parse_grid(drop_raw)
    target_grid = _parse_grid(target_raw)
    split_grid = _parse_grid(split_raw, as_int=True)
    if not tickers:
        st.error("종목을 최소 1개 선택해줘.")
        st.stop()
    if not drop_grid or not target_grid or not split_grid:
        st.error("급락 기준·익절 목표·분할 수를 최소 1개씩 (숫자로) 넣어줘.")
        st.stop()
    market_data = CachedMarketDataAdapter(YFinanceAdapter())
    dailies: dict[str, object] = {}
    for tk in tickers:
        with st.spinner(f"Fetching {tk}..."):
            try:
                df_ = market_data.fetch_ohlcv(tk, start_date, end_date)
            except Exception as e:
                st.warning(f"{tk} fetch 실패 — 제외: {e}")
                continue
        if df_.index.tz is not None:
            df_.index = df_.index.tz_localize(None)
        df_.index = df_.index.normalize()
        if len(df_):
            dailies[tk] = df_
    if not dailies:
        st.error("데이터를 가져온 종목이 없어.")
        st.stop()

    fee = TossFeeSchedule(
        buy_commission_pct=commission_pct / 100.0,
        sell_commission_pct=commission_pct / 100.0,
    )
    combos = [
        (d, t, sp) for d in drop_grid for t in target_grid for sp in split_grid
    ]
    results: dict[tuple[float, float, int], object] = {}
    progress = st.progress(0.0, text="그리드 탐색 중...")
    single = len(dailies) == 1
    for i, (d, t, sp) in enumerate(combos):
        if single:
            only = next(iter(dailies))
            cfg = DipBuyConfig(
                ticker=only,
                start_date=start_date,
                end_date=end_date,
                initial_capital=float(initial_capital),
                drop_pct=d / 100.0,
                target_pct=t / 100.0,
                split=sp,
                fee_schedule=fee,
                capital_gains_tax_pct=tax_pct / 100.0,
            )
            results[(d, t, sp)] = DipBuyStrategy().execute(dailies[only], cfg)
        else:
            mcfg = MultiDipBuyConfig(
                tickers=list(dailies),
                start_date=start_date,
                end_date=end_date,
                initial_capital=float(initial_capital),
                drop_pct=d / 100.0,
                target_pct=t / 100.0,
                split=sp,
                fee_schedule=fee,
                capital_gains_tax_pct=tax_pct / 100.0,
            )
            results[(d, t, sp)] = MultiDipBuyStrategy().execute(dailies, mcfg)
        progress.progress(
            (i + 1) / len(combos),
            text=f"그리드 탐색 중 ({i + 1}/{len(combos)})...",
        )
    progress.empty()
    # 세션에 보관 — 히트맵 탭/지표 라디오 조작으로 rerun 되어도
    # 그리드를 다시 돌리지 않는다.
    st.session_state["dip_grid_state"] = {
        "results": results, "dailies": dailies,
    }

_state = st.session_state.get("dip_grid_state")
if _state is None:
    st.info("그리드 범위를 정하고 **Run Grid**를 눌러줘.")
    st.stop()
results = _state["results"]
dailies = _state["dailies"]

# 실행 시점 값 복원 — 사이드바를 바꿔도 화면은 마지막 실행 기준.
_sample_cfg = next(iter(results.values())).config
ticker = (
    "+".join(_sample_cfg.tickers)
    if hasattr(_sample_cfg, "tickers")
    else _sample_cfg.ticker
)
initial_capital = _sample_cfg.initial_capital
drops = sorted({k[0] for k in results})
targets = sorted({k[1] for k in results})
splits = sorted({k[2] for k in results})

# --- 그리드 요약 ---
best_key = max(
    results, key=lambda k: results[k].liquidation.final_value_after_tax
)
best = results[best_key]

st.subheader(
    f"최적 조합: 전일 −{best_key[0]:g}% → 1/{best_key[2]} 분할 매수 → "
    f"+{best_key[1]:g}% 익절"
)
m1, m2, m3, m4 = st.columns(4)
m1.metric("세후 청산가", f"{best.liquidation.final_value_after_tax:,.0f}")
m2.metric("CAGR (세전)", f"{best.summary.cagr_pct:+.2%}")
m3.metric("MDD", f"{best.summary.max_drawdown_pct:.1%}")
m4.metric(
    "사이클 (승률)",
    f"{len(best.closed_cycles)}회 ({best.win_rate:.0%})",
    help="손절이 없어 완결 사이클은 전부 익절 — 승률 100%가 아니면 "
    "미청산(open) 사이클이 손실 중이라는 뜻.",
)
bh_after = best.benchmark_after_tax
st.caption(
    f"동일 구간 {ticker} 100% 존버 세후: {bh_after:,.0f} — 전략 대비 "
    f"{best.liquidation.final_value_after_tax / bh_after - 1.0:+.1%}."
)

# --- 히트맵: 분할 수별 탭 ---
st.subheader("그리드 히트맵 (탭 = 분할 수)")
metric_choice = st.radio(
    "지표",
    ["세후 청산가", "CAGR (세전)", "MDD", "사이클 수", "평균 보유일"],
    horizontal=True,
)


def _cell(r, metric):
    if metric == "세후 청산가":
        return r.liquidation.final_value_after_tax
    if metric == "CAGR (세전)":
        return r.summary.cagr_pct
    if metric == "MDD":
        return r.summary.max_drawdown_pct
    if metric == "사이클 수":
        return len(r.closed_cycles)
    closed = r.closed_cycles
    return (
        sum(c.holding_days for c in closed) / len(closed) if closed else 0
    )


fmt = {
    "세후 청산가": ",.0f", "CAGR (세전)": ".1%", "MDD": ".1%",
    "사이클 수": ".0f", "평균 보유일": ".0f",
}[metric_choice]
tabs = st.tabs([f"1/{sp} 분할" for sp in splits])
for tab, sp in zip(tabs, splits):
    with tab:
        z = [
            [_cell(results[(d, t, sp)], metric_choice) for t in targets]
            for d in drops
        ]
        heat = go.Figure(
            go.Heatmap(
                z=z,
                x=[f"+{t:g}%" for t in targets],
                y=[f"−{d:g}%" for d in drops],
                colorscale="Viridis" if metric_choice != "MDD" else "Viridis_r",
                texttemplate="%{z:" + fmt + "}",
                colorbar=dict(title=metric_choice),
            )
        )
        heat.update_layout(
            template="plotly_dark", height=400,
            xaxis_title="익절 목표 (+Y%)", yaxis_title="급락 기준 (−X%)",
            margin=dict(l=60, r=40, t=30, b=50),
        )
        st.plotly_chart(heat, use_container_width=True)

# --- 전체 그리드 테이블 ---
st.subheader("전체 조합 결과 (세후 청산가 순)")
# 정렬이 제대로 되도록 값은 **숫자 그대로** 넣고, 표시 포맷은
# column_config로 처리한다 (문자열이면 "99" > "100" 사전순 정렬 버그).
rows = []
for (d, t, sp), r in sorted(
    results.items(),
    key=lambda kv: kv[1].liquidation.final_value_after_tax,
    reverse=True,
):
    closed = r.closed_cycles
    open_cycles = [c for c in r.cycles if c.outcome == "open"]
    rows.append(
        {
            "급락 기준(%)": -d,
            "분할 N": sp,
            "익절 목표(%)": t,
            "세후 청산가": round(r.liquidation.final_value_after_tax),
            "CAGR": r.summary.cagr_pct,
            "MDD": r.summary.max_drawdown_pct,
            "완결 사이클": len(closed),
            "평균 보유일": (
                round(sum(c.holding_days for c in closed) / len(closed))
                if closed
                else None
            ),
            "최장 보유일": (
                max(c.holding_days for c in r.cycles) if r.cycles else None
            ),
            "미청산": round(open_cycles[0].pnl) if open_cycles else None,
            "총 세금": round(r.liquidation.total_tax),
        }
    )
st.dataframe(
    rows,
    use_container_width=True,
    hide_index=True,
    column_config={
        "급락 기준(%)": st.column_config.NumberColumn(format="%g%%"),
        "익절 목표(%)": st.column_config.NumberColumn(format="+%g%%"),
        "세후 청산가": st.column_config.NumberColumn(format="localized"),
        "CAGR": st.column_config.NumberColumn(format="percent"),
        "MDD": st.column_config.NumberColumn(format="percent"),
        "평균 보유일": st.column_config.NumberColumn(format="%d일"),
        "최장 보유일": st.column_config.NumberColumn(format="localized"),
        "미청산": st.column_config.NumberColumn(format="localized"),
        "총 세금": st.column_config.NumberColumn(format="localized"),
    },
)
st.caption(
    "**평균/최장 보유일** = 첫 매수부터 전량 익절까지 달력일 — 짧은 "
    "익절 목표라도 폭락장에 물리면 수백 일 보유가 나올 수 있다 (손절 "
    "없음). **미청산** = 기간 말 목표 미도달로 들고 있는 물량의 평가손익. "
    "개별 매매 내역은 **32번 상세 페이지**에서 같은 조합으로 확인."
)

# --- 최적 조합 상세 ---
st.subheader("최적 조합 상세")
log_scale = st.toggle("로그 스케일", value=True)
eq_fig = go.Figure()
eq_fig.add_trace(
    go.Scatter(
        x=[p.date for p in best.equity_curve],
        y=[p.equity for p in best.equity_curve],
        mode="lines",
        name=best.summary.name,
        line=dict(color="#2196F3", width=2),
    )
)
# 벤치마크 곡선: 단일 = 100% 존버, 멀티 = 1/k 동일가중 존버
# (늦게 상장한 종목의 슬리브는 상장 전까지 현금 대기).
_k = len(dailies)
_sleeves = []
for _df in dailies.values():
    _c = _df["Close"]
    _sleeves.append((_c / float(_c.iloc[0])) * (float(initial_capital) / _k))
_bh = pd.concat(_sleeves, axis=1).ffill()
_bh = _bh.fillna(float(initial_capital) / _k).sum(axis=1)
eq_fig.add_trace(
    go.Scatter(
        x=_bh.index, y=_bh.values, mode="lines",
        name=f"{ticker} 존버 ({'동일가중' if _k > 1 else '100%'})",
        line=dict(color="#E53935", width=1.2, dash="dot"),
    )
)
eq_fig.update_layout(
    template="plotly_dark", height=450,
    yaxis_type="log" if log_scale else "linear",
    xaxis_title="Date", yaxis_title="Portfolio Value",
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    margin=dict(l=50, r=50, t=40, b=30),
)
st.plotly_chart(eq_fig, use_container_width=True)

cycle_rows = [
    {
        "종목": c.ticker or ticker,
        "시작": c.start,
        "종료": c.end if c.end else "진행 중",
        "보유일": c.holding_days,
        "매수 횟수": c.n_buys,
        "평단": fmt_price(c.avg_price),
        "청산가": fmt_price(c.exit_price) if c.exit_price else "—",
        "투입": round(c.invested),
        "손익": round(c.pnl),
        "결과": "익절" if c.outcome == "profit" else "진행 중",
    }
    for c in best.cycles
]
st.dataframe(
    cycle_rows,
    use_container_width=True,
    hide_index=True,
    column_config={
        "투입": st.column_config.NumberColumn(format="localized"),
        "손익": st.column_config.NumberColumn(format="localized"),
    },
)
