"""초단기 전략이 레버리지 ETF 장기보유를 이기는가 — ⑥ 5분 ORB (논문 재현) · ⑦ 오버나이트.

근거·결과: docs/strategy_notes/초단기_ORB_오버나이트_검증_2026-10.md
엔진: strategy/adapters/intraday_lab.py · 배치: intraday_lab_cli.py
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from data.adapters.composed_market_data import build_default_market_data
from strategy.adapters.intraday_lab import (
    KNOWN_SPLITS,
    OrbParams,
    alpha_beta,
    curve_summary,
    overnight_curves,
    run_orb,
    split_factor_series,
    yearly_table,
)
from strategy.adapters.risk_metrics import compute_risk_metrics

st.set_page_config(page_title="초단기 vs 장기보유", layout="wide")
st.title("초단기 전략 vs 장기보유 — 5분 ORB · 오버나이트")
st.caption(
    "**⑥ ORB**: Zarattini & Aziz (2023) *Can Day Trading Really Be Profitable?* 의 규칙을 그대로 재현한다 — 09:30 첫 5분봉 방향으로 "
    "두 번째 봉 시가 진입, 스톱 = 첫 봉 저/고가(변형: 14일 ATR×5%), 목표 10R 또는 장 마감 청산, 계좌 1% 리스크, 레버리지 4배, "
    "수수료 $0.0005/주. 논문: 2016-01~2023-02 QQQ +675% (보유 +169%), TQQQ +1,485%, ATR 5% 스톱 +9,350%. "
    "**⑦ 오버나이트**: 지수 수익은 밤(종가→시가)에 몰린다는 현상을 QQQ·TQQQ 로 확인. "
    "둘 다 **논문식 비용**과 **토스 0.1% + 양도세 22%** 두 가지로 본다 — 국내 계좌에서 살아남는지가 이 페이지의 질문."
)

with st.sidebar:
    st.header("기간 / 종목")
    start = st.date_input("시작", value=date(2016, 1, 1), min_value=date(2005, 1, 1), max_value=date.today())
    end = st.date_input("끝", value=date.today(), min_value=start, max_value=date.today())
    signal = st.selectbox("ORB 신호 봉", ["QQQ", "TQQQ", "SPY"], index=0)
    trade = st.selectbox("ORB 거래 종목", ["QQQ", "TQQQ", "SOXL", "SPY", "UPRO"], index=0,
                         help="논문의 TQQQ 버전은 신호·거래 모두 TQQQ. 신호 QQQ + 거래 TQQQ 는 변형.")
    st.header("⑥ ORB 규칙")
    stop_mode = st.radio("스톱", ["range (첫 봉 저/고가 — 논문 기본)", "atr (14일 ATR × 비율 — 논문 최적)"], index=0)
    atr_frac = st.slider("ATR 비율", 0.01, 0.50, 0.05, 0.01, disabled=not stop_mode.startswith("atr"))
    target_mode = st.radio("목표", ["10R 또는 EoD (논문 기본)", "EoD 만 (논문 최적)", "직접 입력"], index=0)
    target_r = {"10R 또는 EoD (논문 기본)": 10.0, "EoD 만 (논문 최적)": None}.get(target_mode)
    if target_mode == "직접 입력":
        target_r = float(st.number_input("목표 R", 1.0, 30.0, 10.0, 1.0))
    risk_pct = st.slider("거래당 리스크 (%)", 0.25, 3.0, 1.0, 0.25) / 100
    max_lev = st.slider("최대 레버리지", 1.0, 4.0, 4.0, 0.5)
    sides = st.multiselect("방향", ["롱", "숏"], default=["롱", "숏"])
    st.header("비용")
    cost_mode = st.radio("수수료·세금", ["논문 ($0.0005/주, 세금 0)", "IBKR $0.0035/주 + 슬리피지 1센트", "토스 0.1% 양방향 + 양도세 22%", "수수료 0 (상한선)"], index=0)
    capital = st.number_input("초기 자본 ($)", 5_000, 10_000_000, 25_000, 5_000)

cost_kw = {"논문 ($0.0005/주, 세금 0)": dict(commission_per_share=0.0005),
           "IBKR $0.0035/주 + 슬리피지 1센트": dict(commission_per_share=0.0035, slippage_per_share=0.01),
           "토스 0.1% 양방향 + 양도세 22%": dict(commission_per_share=0.0, commission_pct=0.001, tax_rate=0.22, tax_deduction=1_800.0),
           "수수료 0 (상한선)": dict(commission_per_share=0.0)}[cost_mode]
params = OrbParams(stop_mode="atr" if stop_mode.startswith("atr") else "range", atr_frac=atr_frac, target_r=target_r,
                   risk_pct=risk_pct, max_leverage=max_lev, allow_long="롱" in sides, allow_short="숏" in sides,
                   initial_capital=float(capital), **cost_kw)


@st.cache_data(ttl=6 * 3600, show_spinner="5분봉 로딩 (첫 실행은 월 단위 API 호출이라 느림)…")
def _bars5m(sym: str, start: date, end: date) -> pd.DataFrame:
    return build_default_market_data().fetch_ohlcv(sym, start - timedelta(days=30), end, "5m")


@st.cache_data(ttl=6 * 3600, show_spinner="일봉 로딩…")
def _daily(sym: str, start: date, end: date) -> pd.DataFrame:
    d = build_default_market_data().fetch_ohlcv(sym, start, end, "1d")
    if d.index.tz is not None:
        d.index = d.index.tz_localize(None)
    return d


tab_orb, tab_night, tab_read = st.tabs(["⑥ 5분 ORB", "⑦ 오버나이트", "읽는 법 / 한계"])

with tab_orb:
    try:
        sb = _bars5m(signal, start, end)
        tb = sb if trade == signal else _bars5m(trade, start, end)
        qqq_d = _daily("QQQ", start, end)
        tq_d = _daily(trade if trade != "QQQ" else "TQQQ", start, end)
    except Exception as e:  # noqa: BLE001
        st.error(f"데이터 로딩 실패: {e}")
        st.stop()
    cut = pd.Timestamp(start, tz="America/New_York")
    tb_cut = tb[tb.index >= cut]
    factor = split_factor_series(tb_cut.index.tz_convert("America/New_York").tz_localize(None).normalize().unique(),
                                 KNOWN_SPLITS.get(trade, []))
    with st.spinner("ORB 시뮬레이션…"):
        res = run_orb(sb[sb.index >= cut], (tb_cut if trade != signal else None), params, unadjusted_factor=factor)
    if not res.trades:
        st.warning("거래가 없습니다. 기간·종목을 확인하세요.")
        st.stop()
    m = res.metrics
    bh_q = qqq_d["Close"] / qqq_d["Close"].iloc[0] * capital
    bh_t = tq_d["Close"] / tq_d["Close"].iloc[0] * capital
    ab = alpha_beta(res.equity, qqq_d["Close"])
    bh_t_label = trade if trade != "QQQ" else "TQQQ"

    c = st.columns(7)
    c[0].metric("ORB 최종", f"{res.equity.iloc[-1] / capital:.2f}x", f"CAGR {m['cagr']:+.1%}")
    c[1].metric("QQQ 보유", f"{bh_q.iloc[-1] / capital:.2f}x", f"MDD {compute_risk_metrics(bh_q)['mdd']:+.0%}")
    c[2].metric(f"{bh_t_label} 보유", f"{bh_t.iloc[-1] / capital:.2f}x", f"MDD {compute_risk_metrics(bh_t)['mdd']:+.0%}")
    c[3].metric("ORB MDD / 샤프", f"{m['mdd']:+.1%}", f"샤프 {m['sharpe']:.2f}")
    c[4].metric("거래 / 승률", f"{m['trades']}", f"승률 {m['win_rate']:.0%} · 평균 {m['avg_r']:+.2f}R")
    c[5].metric("α(연) / β vs QQQ", f"{ab['alpha_ann']:+.0%}", f"β {ab['beta']:.2f}")
    c[6].metric("레버리지 캡 / 세금", f"{m['capped_share']:.0%}", f"세금 ${res.tax_paid:,.0f}")
    st.caption(f"논문 기준값 (2016-01~2023-02, QQQ, 논문 비용): +675% (7.75x) · 1,795회 · 승률 24% · 0.13R · 샤프 1.12 · α 33%. "
               f"TQQQ: 15.85x · 0.18R · 샤프 1.18 · α 47%. 이 실행: 롱 {m['long_share']:.0%} · 스톱 {m['stop_share']:.0%} · 목표 {m['target_share']:.0%}")

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.03,
                        subplot_titles=("계좌 (로그)", "낙폭"))
    for name, s in ((f"ORB {signal}→{trade}", res.equity), ("QQQ 보유", bh_q), (f"{bh_t_label} 보유", bh_t)):
        fig.add_trace(go.Scatter(x=s.index, y=s, name=name), row=1, col=1)
        fig.add_trace(go.Scatter(x=s.index, y=s / s.cummax() - 1, name=name, showlegend=False, opacity=0.7), row=2, col=1)
    fig.update_yaxes(type="log", row=1, col=1); fig.update_yaxes(tickformat=".0%", row=2, col=1)
    fig.update_layout(height=560, legend=dict(orientation="h", y=1.08), margin=dict(l=40, r=20, t=60, b=30))
    st.plotly_chart(fig, use_container_width=True)

    cA, cB = st.columns(2)
    with cA:
        st.markdown("**연도별 수익률**")
        yr = yearly_table({"ORB": res.equity, "QQQ": bh_q, bh_t_label: bh_t})
        st.dataframe(yr.style.format("{:+.1%}"), use_container_width=True, height=min(500, 40 + 35 * len(yr)))
    with cB:
        st.markdown("**거래 R 분포** (논문 그림 3·6)")
        tr = res.frame
        hist = go.Figure(go.Histogram(x=tr["r_multiple"], nbinsx=60))
        hist.update_layout(height=300, margin=dict(l=30, r=10, t=10, b=30), xaxis_title="R 배수", yaxis_title="거래 수")
        st.plotly_chart(hist, use_container_width=True)
        st.caption(f"청산 사유: {tr['exit_reason'].value_counts().to_dict()} · 평균 이익 {m['avg_win_r']:+.2f}R · 평균 손실 {m['avg_loss_r']:+.2f}R")
    st.markdown("**거래 일지** (최근 300건)")
    st.dataframe(tr.tail(300).iloc[::-1].style.format({"entry": "{:.2f}", "stop": "{:.2f}", "exit": "{:.2f}", "r_value": "{:.3f}",
                                                       "pnl": "{:+,.0f}", "r_multiple": "{:+.2f}"}),
                 use_container_width=True, hide_index=True, height=320)

with tab_night:
    st.markdown("종가에 사서 다음날 시가에 파는 **밤만** 전략과 시가에 사서 종가에 파는 **낮만** 전략을 장기보유와 비교한다. "
                "매일 왕복이라 수수료·세금에 가장 민감하다.")
    for sym in dict.fromkeys(["QQQ", trade if trade != "QQQ" else "TQQQ"]):
        try:
            d = _daily(sym, start, end)
        except Exception as e:  # noqa: BLE001
            st.error(f"{sym}: {e}"); continue
        free = overnight_curves(d, initial_capital=float(capital))
        toss = overnight_curves(d, commission_pct=0.001, tax_rate=0.22, initial_capital=float(capital), tax_deduction=1_800.0)
        st.subheader(sym)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**수수료 0 (상한선)**")
            st.dataframe(curve_summary(free).style.format({"최종 배수": "{:.2f}x", "CAGR": "{:+.1%}", "MDD": "{:+.0%}", "샤프": "{:.2f}", "최악 연도": "{:+.0%}"}),
                         use_container_width=True, hide_index=True)
        with c2:
            st.markdown("**토스 0.1% 양방향 + 양도세 22%**")
            st.dataframe(curve_summary(toss).style.format({"최종 배수": "{:.2f}x", "CAGR": "{:+.1%}", "MDD": "{:+.0%}", "샤프": "{:.2f}", "최악 연도": "{:+.0%}"}),
                         use_container_width=True, hide_index=True)
        f = go.Figure()
        for k in free.columns:
            f.add_trace(go.Scatter(x=free.index, y=free[k], name=f"{k} (무비용)"))
        for k in ("밤만 (종가→시가)", "낮만 (시가→종가)"):
            f.add_trace(go.Scatter(x=toss.index, y=toss[k], name=f"{k} (토스+세금)", line=dict(dash="dot")))
        f.update_yaxes(type="log"); f.update_layout(height=400, legend=dict(orientation="h", y=1.1), margin=dict(l=40, r=20, t=40, b=30))
        st.plotly_chart(f, use_container_width=True)

with tab_read:
    st.markdown("""
- **ORB 는 논문 규칙 그대로**다. 다른 점: 논문은 TQQQ 버전을 TQQQ 자체 5분봉으로 돌렸다(여기선 신호·거래를 같은 종목으로 두면 동일). 한 봉에서 스톱과 목표가 같이 닿으면 스톱 우선(보수적). 스톱을 갭으로 지나면 시가 체결.
- **슬리피지 0** 은 논문 가정이다. ATR 5% 스톱은 TQQQ 기준 $0.08 폭이라 실제로는 스프레드보다 좁다 — 논문도 "대형 계좌에서는 비현실적"이라고 적었다.
- **국내 계좌 비용**: 토스 0.1% 양방향 = 왕복 0.2%. ORB 평균 이익이 0.13R 이고 R 이 가격의 0.3% 안팎이면 왕복 0.2% 가 이익의 절반을 가져간다. 양도세는 연말에 그 해 이익·손실을 합산한 순이익에서 공제(250만 원 ≈ $1,800)를 뺀 뒤 22%. 손실 이월 없음.
- **오버나이트**도 같은 방식(연 단위 손익 합산, 공제 후 22%, 이월 없음). 장기보유는 미실현이라 세금 0.
- 2016~2023 은 논문의 in-sample 이다. **2023-02 이후**가 out-of-sample — 기간을 2023-02-18 부터로 잡아 비교하라.
""")
