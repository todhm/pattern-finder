"""«The Lifecycle Trade» (IPO 생애주기) — 체크리스트 자동 판정 + 과거 IPO 코호트로 책의 통계·매도 규칙 검증.

체크리스트 원문: docs/strategy_notes/라이프사이클_트레이드_체크리스트_2026_10.md (책 75장 스캔 요약)
로직: strategy/adapters/lifecycle_trade.py · IPO 캘린더: data/adapters/finnhub_ipo_calendar.py (→ AV LISTING_STATUS 폴백)
배치: lifecycle_batch_cli.py · 검증 결과: docs/strategy_notes/라이프사이클_트레이드_검증_2026-10.md
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from data.adapters.composed_ipo_calendar import build_default_ipo_calendar
from data.adapters.composed_market_data import build_default_market_data
from data.domain.models import IpoEvent
from pages._shared.growth_sources import build_growth_adapters
from strategy.adapters.lifecycle_portfolio import (
    PortfolioConfig,
    align,
    benchmark_metrics,
    benchmark_series,
    hold_trades,
    run_portfolio,
    trades_from_results,
)
from strategy.adapters.lifecycle_trade import (
    BUY_KINDS,
    RULES,
    LifecycleParams,
    TickerResult,
    analyze_ticker,
    cohort_tables,
    fetch_many,
    ipo_cohort,
    kind_summary,
    revenue_yoy_asof,
    rule_pattern_table,
    rule_summary,
    trades_frame,
)

st.set_page_config(page_title="라이프사이클 트레이드 (IPO)", layout="wide")
st.title("«The Lifecycle Trade» — IPO 생애주기 스크린 & 검증")
st.caption(
    "책의 주장: IPO 는 **IPO-AP(초기 상승) → I-DDP(기관 실사, 길고 지루한 횡보) → I-AP(기관 상승)** 세 단계를 거치고, "
    "6패턴(Late Bloomer·Pump and Dump·Stair Stepper·Rocket Ship·One-Hit Wonder·Disappointment) 중 하나로 움직인다. "
    "매수는 4종(IPO base·initial base·breakaway gap·first mature base) 돌파, 매도는 단계별 규칙(Ascender·Midterm·40-Week·Everest)+10% 손절+MCP. "
    "이 페이지는 ① 기준일 스크린(7장 IPO Alert / Rare Jewels / Liquidity Matters + 단계·베이스) ② 과거 IPO 코호트로 6장 통계 재현 "
    "③ 4매수지점 × 5매도규칙 $100K 시뮬(5장 Table 5.1/5.2 재현) 을 한 번에 돌린다. ⚠️ 패턴 라벨은 사후(hindsight) 라벨이다."
)

_MANUAL = [   # 문서 §B — 코드가 못 보는 것
    ("b1", "B-1 단계 판독: 차트에서 IPO-AP 고점·회색선(첫 주 종가)·난기류 지대(이전 고점)를 직접 짚었는가"),
    ("b2", "B-2 베이스가 '짧고 얕은(구글 18일 13%)' 질 좋은 모양인가, 아니면 넓고 느슨한가 (넓고 느슨하면 회피)"),
    ("b3", "B-3 돌파 일봉이 거래량 동반인가, 매수 지점에서 3~10% 이상 멀어지지 않았는가 (늦은 매수 금지)"),
    ("b4", "B-4 변혁적 기업인가 — 매출 성장·시장 지배·촉매 (Rare Jewels: 최근 분기 매출 +50%)"),
    ("b5", "B-5 락업 해제(~90일)·2차 공모(secondary) 일정이 눈앞에 없는가"),
    ("b6", "B-6 시장이 상승장인가, 내 포트폴리오가 잘 되고 있는가 (아니면 IPO 거래 금지)"),
    ("b7", "B-7 포지션 사이즈: 성숙주의 절반 이하(5%), 1% 손실 한도, 한 번에 IPO 1종목, BOSS(추가 매수 금지)"),
    ("b8", "B-8 단계에 맞는 매도 규칙을 '미리' 정했는가 (IPO-AP: Everest/Ascender · I-AP: Midterm/40-Week + MCP)"),
    ("b9", "B-9 MCP 스톱 가격을 숫자로 적었는가: 매수가 + (피크 − 매수가) × MCP%"),
]


@st.cache_resource(show_spinner=False)
def _calendar():
    return build_default_ipo_calendar()


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def _ipo_events(start: date, end: date) -> list[dict]:
    cal = _calendar()
    if cal is None:
        return []
    return [e.model_dump(mode="json") for e in cal.fetch_ipos(start, end)]


with st.sidebar:
    st.header("기준일 / 코호트")
    as_of = st.date_input("기준일", value=date.today(), min_value=date(2011, 1, 1), max_value=date.today(),
                          help="스크린은 이날까지의 데이터만 본다. 통계·시뮬은 전체 히스토리(사후)를 쓴다.")
    source = st.radio("종목 소스", ["IPO 캘린더 코호트", "티커 직접 입력"], horizontal=True)
    if source == "IPO 캘린더 코호트":
        years_back = st.slider("첫 거래일 범위 (기준일 기준 몇 년 전까지)", 1, 12, 4)
        min_age = st.slider("최소 상장 경과 (주)", 4, 104, 26, help="너무 어린 IPO 는 통계·패턴 판정이 불가")
        exch = st.multiselect("거래소", ["NASDAQ", "NYSE", "AMEX"], default=["NASDAQ", "NYSE"])
        excl_spac = st.toggle("SPAC 제외 (이름에 Acquisition)", value=True)
        min_offer = st.number_input("최소 공모가 ($)", value=5.0, min_value=0.0, step=1.0)
        tickers_text = ""
    else:
        tickers_text = st.text_area("티커:첫거래일[:공모가] (줄바꿈)", value="PANW:2012-07-20:42\nFB:2012-05-18:38\nSHAK:2015-01-30:21\nTWLO:2016-06-23:15",
                                    height=140)
    max_tickers = st.number_input("최대 종목 수 (0=전체)", value=150, min_value=0, step=50,
                                  help="첫 실행은 종목당 전체 일봉 fetch. 코호트 전체(수백~천 종목)는 lifecycle_batch_cli.py 로.")
    st.header("규칙")
    rules = st.multiselect("매도 규칙", list(RULES), default=list(RULES))
    kinds = st.multiselect("매수 지점", list(BUY_KINDS), default=list(BUY_KINDS))
    max_weeks = st.slider("최대 보유 (주)", 26, 260, 156)
    with st.expander("책의 임계값 (LifecycleParams)"):
        ap_min = st.number_input("IPO-AP 성립 최소 상승 (회색선 대비)", value=0.20, step=0.05)
        ap_dd = st.number_input("IPO-AP 종료 = 고점 대비 하락", value=0.40, step=0.05)
        base_depth = st.number_input("베이스 최대 폭", value=0.35, step=0.05)
        gap = st.number_input("breakaway gap 최소 (%)", value=0.25, step=0.05)
        stop = st.number_input("손절 (매수가 대비)", value=0.10, step=0.01)
        mcp = st.number_input("MCP 보전 비율", value=0.45, step=0.05)
        everest_ext = st.number_input("Everest 포물선: 21일 EMA 이격 ≥", value=0.12, step=0.02)
    use_revenue = st.toggle("7장 매출 +50% 판정 (성장 펀더멘털 호출 — 느림)", value=False)
    run_btn = st.button("Run", type="primary", use_container_width=True)


def _chart(r: TickerResult, p: LifecycleParams):
    fr, lc = r.frames, r.lifecycle
    w = fr.w
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.02)
    fig.add_trace(go.Candlestick(x=w.index, open=w.Open, high=w.High, low=w.Low, close=w.Close, name="주봉",
                                 increasing_line_color="#26A69A", decreasing_line_color="#EF5350"), row=1, col=1)
    fig.add_trace(go.Scatter(x=w.index, y=w.SMA10, name="10주 SMA", line=dict(width=1.2, color="#FFA726")), row=1, col=1)
    fig.add_trace(go.Scatter(x=w.index, y=w.SMA40, name="40주 SMA", line=dict(width=1.6, color="#AB47BC")), row=1, col=1)
    fig.add_hline(y=lc.gray, line_dash="dot", line_color="#BDBDBD", annotation_text=f"회색선 {lc.gray:,.2f}", row=1, col=1)
    fig.add_hline(y=lc.turbulence, line_dash="dash", line_color="#FFEE58", annotation_text="난기류 지대", row=1, col=1)
    if lc.offer:
        fig.add_hline(y=lc.offer, line_dash="dot", line_color="#EF5350", annotation_text=f"공모가 {lc.offer:,.2f}", row=1, col=1)
    colors = {"IPO-BASE": "#90CAF9", "IPO-AP": "#66BB6A", "IPO-AF": "#EF5350", "I-DDP": "#F48FB1", "I-AP": "#4FC3F7"}
    ph = lc.phases
    start = ph.index[0]
    for i in range(1, len(ph) + 1):
        if i == len(ph) or ph.iloc[i] != ph.iloc[i - 1]:
            fig.add_vrect(x0=start, x1=ph.index[i - 1], fillcolor=colors.get(ph.iloc[i - 1], "#999"), opacity=0.10,
                          line_width=0, annotation_text=ph.iloc[i - 1], annotation_position="top left", row=1, col=1)
            if i < len(ph):
                start = ph.index[i]
    for b in lc.bases:
        fig.add_shape(type="rect", x0=w.index[b.start_i], x1=w.index[b.end_i], y0=b.low, y1=b.high,
                      line=dict(color="#42A5F5", width=1), row=1, col=1)
    for bp in r.buy_points:
        fig.add_trace(go.Scatter(x=[bp.signal_date], y=[bp.level], mode="markers+text", text=[bp.kind], textposition="bottom center",
                                 marker=dict(symbol="triangle-up", size=12, color="#00E676"), name=f"매수 {bp.kind}"), row=1, col=1)
    for t in r.trades:
        for d_, px, frac, why in t.sells:
            fig.add_trace(go.Scatter(x=[pd.Timestamp(d_)], y=[px], mode="markers", marker=dict(symbol="x", size=8),
                                     name=f"{t.rule} {frac:.0%}", hovertext=why, showlegend=False), row=1, col=1)
    fig.add_trace(go.Bar(x=w.index, y=w.Volume, name="거래량", marker_color="#78909C"), row=2, col=1)
    fig.update_layout(template="plotly_dark", height=620, xaxis_rangeslider_visible=False, showlegend=True,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02), margin=dict(l=40, r=110, t=30, b=20))
    return fig


if run_btn:
    p = LifecycleParams(ap_min_gain=float(ap_min), ap_fail_drawdown=float(ap_dd), base_max_depth=float(base_depth),
                        gap_pct=float(gap), stop_pct=float(stop), mcp=float(mcp), everest_ext=float(everest_ext))
    if source == "IPO 캘린더 코호트":
        start, end = as_of - timedelta(days=365 * years_back), as_of - timedelta(weeks=min_age)
        with st.spinner("IPO 캘린더 조회..."):
            raw = [IpoEvent(**e) for e in _ipo_events(start, end)]
        if not raw:
            st.error("IPO 캘린더 결과 없음 — FINNHUB_API_KEY / ALPHAVANTAGE_API_KEY 또는 Mongo 를 확인하거나 티커 직접 입력.")
            st.stop()
        events = ipo_cohort(raw, tuple(exch), excl_spac, float(min_offer) if min_offer > 0 else None)
        st.caption(f"IPO 캘린더 {len(raw)}건 → 필터 후 {len(events)}건 (소스: {_calendar().last_source})")
    else:
        events = []
        for line in tickers_text.splitlines():
            parts = [x.strip() for x in line.split(":") if x.strip()]
            if len(parts) >= 2:
                events.append(IpoEvent(symbol=parts[0].upper(), ipo_date=date.fromisoformat(parts[1]),
                                       offer_price=float(parts[2]) if len(parts) > 2 else None, source="manual"))
    if max_tickers:
        events = events[: int(max_tickers)]
    if not events:
        st.warning("종목 없음")
        st.stop()
    md = build_default_market_data()
    by_sym = {e.symbol: e for e in events}
    prog = st.progress(0.0, text=f"일봉 수집 ({len(events)}종목)")
    logs: list[str] = []

    def _log(m: str) -> None:
        logs.append(m)
        prog.progress(min(0.9, len(logs) * 50 / max(len(events), 1)), text=m)

    frames, errors = fetch_many(md, list(by_sym), min(e.ipo_date for e in events) - timedelta(days=7), as_of, 4, _log)
    growth = build_growth_adapters() if use_revenue else None
    results: list[TickerResult] = []
    for i, (sym, df) in enumerate(frames.items()):
        prog.progress(0.9 + 0.1 * (i + 1) / len(frames), text=f"분석 {i + 1}/{len(frames)} — {sym}")
        rev = None
        if growth:
            for a in growth:
                try:
                    snap = a.fetch(sym)
                except Exception:  # noqa: BLE001
                    continue
                if snap.quarters:
                    rev = revenue_yoy_asof(snap.quarters, as_of)
                    break
        try:
            r = analyze_ticker(sym, df, by_sym[sym], p, as_of, tuple(rules), tuple(kinds), int(max_weeks) * 5, rev)
        except Exception as e:  # noqa: BLE001
            errors.append(f"{sym}: {type(e).__name__}")
            continue
        if r is not None:
            results.append(r)
    prog.empty()
    st.session_state["lifecycle"] = {"as_of": as_of, "results": results, "errors": errors, "p": p, "n_events": len(events)}

_state = st.session_state.get("lifecycle")
if _state is None:
    st.info("좌측에서 기준일·코호트를 정하고 **Run**. 처음엔 '티커 직접 입력'(책 사례) 또는 코호트 150종목으로 시작. "
            "코호트 전체는 `python lifecycle_batch_cli.py` 로 돌리고 결과는 docs/strategy_notes/lifecycle_runs/ 에 적재.")
    st.stop()

results: list[TickerResult] = _state["results"]
p: LifecycleParams = _state["p"]
st.subheader(f"{_state['as_of']} 기준 — 종목 {_state['n_events']} → 분석 {len(results)} (조회 실패 {len(_state['errors'])})")
if _state["errors"]:
    st.caption("실패: " + " · ".join(_state["errors"][:10]) + (" …" if len(_state["errors"]) > 10 else ""))

tab_screen, tab_study, tab_sim, tab_port, tab_detail = st.tabs(["① 기준일 스크린 (7장)", "② 6장 통계 재현", "③ 매수지점 × 매도규칙 시뮬 (5장)",
                                                                "⑤ 계좌 시뮬 — 그대로 돌리면 얼마?", "④ 종목 상세 + 수동 체크"])

with tab_screen:
    rows = []
    for r in results:
        s = r.screen
        if s is None:
            continue
        rows.append({"티커": s.symbol, "IPO": s.ipo_date, "경과(주)": s.age_weeks, "종가": s.close, "거래대금 $M": s.dollar_vol_m,
                     "52주저 대비": s.above_52w_low, "90일 2배": s.doubled_in_90d, "매출 YoY": s.revenue_yoy,
                     "IPO Alert": s.ipo_alert, "Rare Jewels": s.rare_jewel, "Liquidity Matters": s.liquidity_matters,
                     "단계": s.phase, "패턴(지금까지)": s.pattern, "베이스": s.base_status, "이번주 돌파": s.breakout_this_week,
                     "베이스고점까지": s.dist_to_base_high, "vs 회색선": s.vs_gray, "vs 10주": s.vs_sma10, "vs 40주": s.vs_sma40})
    sdf = pd.DataFrame(rows)
    if sdf.empty:
        st.warning("기준일까지 데이터가 있는 종목 없음")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("IPO Alert", int(sdf["IPO Alert"].sum()))
        c2.metric("Rare Jewels", int(sdf["Rare Jewels"].fillna(False).astype(bool).sum()))
        c3.metric("Liquidity Matters", int(sdf["Liquidity Matters"].fillna(False).astype(bool).sum()))
        c4.metric("이번 주 베이스 돌파", int(sdf["이번주 돌파"].sum()))
        only = st.radio("보기", ["전부", "IPO Alert", "이번 주 돌파", "베이스 형성 중"], horizontal=True)
        view = sdf
        if only == "IPO Alert":
            view = sdf[sdf["IPO Alert"]]
        elif only == "이번 주 돌파":
            view = sdf[sdf["이번주 돌파"]]
        elif only == "베이스 형성 중":
            view = sdf[sdf["베이스"].str.startswith("베이스 형성")]
        view = view.sort_values(["이번주 돌파", "IPO Alert", "거래대금 $M"], ascending=[False, False, False])
        pct = {c: st.column_config.NumberColumn(format="percent") for c in ("52주저 대비", "매출 YoY", "베이스고점까지", "vs 회색선", "vs 10주", "vs 40주")}
        st.dataframe(view, use_container_width=True, hide_index=True,
                     column_config={**pct, "종가": st.column_config.NumberColumn(format="dollar"),
                                    "거래대금 $M": st.column_config.NumberColumn(format="%.1f")})
        st.caption("IPO Alert = 상장 3년 이내 · 52주 저가 +25% 이상 · 20일 평균 거래대금 ≥ $20M. Rare Jewels = 90일 내 +100% + $20M + 최근 분기 매출 +50%. "
                   "Liquidity Matters = 3년 이내 + $20M + 매출 +50%. 매출 판정을 끄면 Rare Jewels/Liquidity Matters 는 '미확인(None)'. "
                   "단계·패턴은 기준일까지의 데이터로 본 것(사후 라벨 아님).")

with tab_study:
    stats = [r.stats for r in results if r.stats is not None]
    st.markdown(f"**첫 거래일 데이터가 있는 {len(stats)}건** (책: 1982~2017 IPO 1,679건). 각 표의 '책' 열이 원문 수치.")
    tabs = cohort_tables(stats)
    if not tabs:
        st.warning("통계 대상 없음 — 첫 거래일(±10일)부터 일봉이 있어야 한다.")
    else:
        pct_fmt = lambda cols: {c: st.column_config.NumberColumn(format="percent") for c in cols}  # noqa: E731
        st.markdown("**Table 6.1 — 첫날 종가 대비 +100% 달성 비율** (대상 n = 그 기간만큼 데이터가 있는 종목)")
        st.dataframe(tabs["table_6_1"], hide_index=True, column_config=pct_fmt(("비율", "책(1982~2017)")))
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Fig 6.1 — 첫날 저가 이탈**")
            st.dataframe(tabs["fig_6_1"], hide_index=True, column_config=pct_fmt(("비율", "책")))
        with c2:
            st.markdown("**Fig 6.2 — 첫날 저가 −10% 손절 발동** (첫날 최저가에 샀다고 가정)")
            st.dataframe(tabs["fig_6_2"], hide_index=True, column_config=pct_fmt(("비율", "책")))
        st.markdown("**Table 6.2 — 빠르게 2배 된 종목이 이후 52주 안에 −50% 조정** (책: 72%가 이익 전부 반납)")
        st.dataframe(tabs["table_6_2"], hide_index=True, column_config=pct_fmt(("코호트 비율", "책 비율", "이후 52주 내 −50% 조정", "책 조정 비율")))
        st.markdown("**80쪽 요약 — 수익 분포 · 유동성 텔(Liquidity Tell)**")
        st.dataframe(tabs["summary"], hide_index=True, column_config=pct_fmt(("값", "책")))
        st.caption("유동성 텔: 거래대금 ≥ $20M 종목이 +200% 갈 확률이 전체의 2배(책 58% vs 28%). 90일 내 2배 종목은 +500% 확률 2배. "
                   "⚠️ 상장폐지 종목의 일봉이 소스에 없으면(실패 목록) 코호트가 생존 편향을 가진다 — 실패 수를 같이 본다.")

with tab_sim:
    tf = trades_frame(results)
    if tf.empty:
        st.warning("거래 없음 — 매수 지점이 잡히지 않았다.")
    else:
        st.markdown("**매도 규칙별 ($100K, 책 Table 5.2 비교)** — 수익률 = 최종 손익/100K, DD = 보유 중 계좌 피크 대비 최대 하락($K), 피크이익 보전 = 최종이익/피크이익")
        rs = rule_summary(tf)
        st.dataframe(rs, hide_index=True, use_container_width=True,
                     column_config={c: st.column_config.NumberColumn(format="percent") for c in ("승률", "중앙값 수익", "평균 수익", "중앙값 피크이익 보전", "책 중앙값 수익", "책 피크이익 보전")})
        st.markdown("**규칙 × 패턴 — 중앙값 수익 (책 Table 5.1)**")
        st.dataframe(rule_pattern_table(tf).style.format("{:+.0%}", na_rep="—"), use_container_width=True)
        st.markdown("**규칙 × 패턴 — 중앙값 DD $K**")
        st.dataframe(rule_pattern_table(tf, "drawdown_k").style.format("{:.0f}", na_rep="—"), use_container_width=True)
        st.markdown("**매수 지점 × 규칙**")
        st.dataframe(kind_summary(tf), hide_index=True, use_container_width=True,
                     column_config={c: st.column_config.NumberColumn(format="percent") for c in ("승률", "중앙값 수익")})
        pat = pd.Series([r.lifecycle.pattern for r in results]).value_counts().rename_axis("패턴").reset_index(name="n")
        st.markdown("**패턴 분포 (사후 라벨)**")
        st.dataframe(pat, hide_index=True)
        st.markdown("**거래 전체**")
        st.dataframe(tf.sort_values(["symbol", "kind", "rule"]), hide_index=True, use_container_width=True,
                     column_config={"pct_gain": st.column_config.NumberColumn(format="percent"), "peak_retained": st.column_config.NumberColumn(format="percent")})
        st.download_button("거래 CSV", tf.to_csv(index=False).encode("utf-8-sig"), file_name=f"lifecycle_trades_{_state['as_of']}.csv", mime="text/csv")
        st.caption("책의 수치는 '손으로 고른' 매수 지점·iconic 종목이라 그대로 비교하면 안 된다. 여기선 규칙으로 잡은 모든 매수 지점에 모든 규칙을 적용했다 — "
                   "같은 종목·같은 진입에서 규칙끼리의 상대 비교(수익 vs DD vs 보유일)가 핵심.")

_PORT_SCENARIOS = {
    "A 필터 없음": PortfolioConfig(liquidity_usd=None, market_filter=False, one_recent_ipo=False),
    "B 유동성 $20M": PortfolioConfig(liquidity_usd=20e6, market_filter=False, one_recent_ipo=False),
    "C 책 그대로 (유동성 + SPY>200일선 + IPO 1년 이내 1종목)": PortfolioConfig(),
    "D 책 그대로 · 첫 성숙 베이스만": PortfolioConfig(kinds=("first mature base",)),
    "E 책 그대로 · 풀 사이즈 20%×5": PortfolioConfig(position_frac=0.20, max_positions=5),
}


@st.cache_data(show_spinner=False)
def _bench(symbol: str, start: date, end: date) -> pd.Series:
    return benchmark_series(build_default_market_data(), symbol, start, end)


with tab_port:
    st.markdown("**$100K 계좌에 책 8장 자금 관리로 거래를 넣으면** — 포지션 = 계좌의 10%(손실 1% ÷ 손절 10%), 동시 보유 ≤ 10, 레버리지 없음, "
                "같은 날 신호가 겹치면 거래대금 큰 순. ③의 거래(진입가·분할 매도)를 그대로 쓴다. 대조군: SPY 그냥 보유, 같은 진입을 손절 없이 252일 보유.")
    c1, c2 = st.columns([2, 1])
    sc_name = c1.selectbox("시나리오", list(_PORT_SCENARIOS), index=2)
    bench_sym = c2.text_input("벤치마크", "SPY")
    if not any(r.trades for r in results):
        st.warning("거래 없음")
    else:
        p_start = min(r.ipo.ipo_date for r in results if r.ipo) - timedelta(days=7)
        p_end = _state["as_of"]
        try:
            spy = _bench(bench_sym, p_start, p_end)
        except Exception as e:  # noqa: BLE001
            st.error(f"벤치마크 조회 실패: {e}")
            spy = None
        if spy is not None:
            cal = spy.index[(spy.index >= pd.Timestamp(p_start)) & (spy.index <= pd.Timestamp(p_end))]
            closes, dvol = align({r.symbol: r.frames.d for r in results}, cal)
            sc = _PORT_SCENARIOS[sc_name]
            runs = {}
            rows = []
            for rule in RULES:
                tr = trades_from_results(results, rule)
                if not tr:
                    continue
                runs[rule] = run_portfolio(tr, closes, dvol, spy, sc, rule)
                rows.append({"규칙": rule, **runs[rule].metrics()})
            if "40-Week" in runs:
                held = hold_trades(trades_from_results(results, "40-Week"), closes, 252)
                runs["보유 252일"] = run_portfolio(held, closes, dvol, spy, sc, "보유 252일")
                rows.append({"규칙": "보유 252일 (대조)", **runs["보유 252일"].metrics()})
            bm = benchmark_metrics(spy, cal)
            rows.append({"규칙": f"{bench_sym} 그냥 보유 (대조)", "최종 자산": bm["최종 자산"], "총수익": bm["총수익"], "CAGR": bm["CAGR"], "MDD": bm["MDD"]})
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True,
                         column_config={**{c: st.column_config.NumberColumn(format="percent") for c in ("총수익", "CAGR", "MDD", "승률", "평균 거래수익", "평균 투자비중")},
                                        "최종 자산": st.column_config.NumberColumn(format="$%d")})
            fig = go.Figure()
            b = spy.reindex(cal).ffill()
            fig.add_trace(go.Scatter(x=cal, y=b / b.iloc[0] * 100_000, name=f"{bench_sym} 보유", line=dict(color="gray")))
            for rule, res in runs.items():
                fig.add_trace(go.Scatter(x=res.equity.index, y=res.equity.values, name=rule))
            fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10), yaxis_title="계좌 가치 $", title=f"{sc_name} — 계좌 곡선")
            st.plotly_chart(fig, use_container_width=True)
            pick = st.selectbox("연도별·거래 내역 규칙", list(runs), index=min(2, len(runs) - 1))
            res = runs[pick]
            st.dataframe(res.yearly().reset_index(), hide_index=True, use_container_width=True,
                         column_config={c: st.column_config.NumberColumn(format="percent") for c in ("연초→연말 수익", "진입 승률")})
            st.caption("건너뛴 진입: " + (", ".join(f"{k} {v}" for k, v in sorted(res.skipped.items(), key=lambda kv: -kv[1])) or "없음"))
            st.dataframe(res.trades.sort_values("pnl", ascending=False), hide_index=True, use_container_width=True,
                         column_config={"pct": st.column_config.NumberColumn(format="percent"), "pnl": st.column_config.NumberColumn(format="$%d"),
                                        "size": st.column_config.NumberColumn(format="$%d")})

with tab_detail:
    by = {r.symbol: r for r in results}
    if not by:
        st.warning("분석된 종목 없음 — 기준일까지 일봉이 있는 종목이 없거나 전부 조회 실패. 좌측 설정(기준일·코호트·티커)을 바꾸고 다시 Run.")
        st.stop()
    sel = st.selectbox("종목", sorted(by))
    r = by[sel]
    lc = r.lifecycle
    st.plotly_chart(_chart(r, p), use_container_width=True)
    st.markdown(f"**패턴(사후)**: {lc.pattern} — {lc.pattern_note} · **지금 단계**: {lc.phase_now} · 회색선 {lc.gray:,.2f} · "
                f"IPO-AP {lc.ap_gain:+.0%} (고점 {lc.ap_peak:,.2f}) · 난기류 지대 {lc.turbulence:,.2f}"
                + (f" · 돌파 {r.frames.w.index[lc.turbulence_cleared_i].date()}" if lc.turbulence_cleared_i is not None else " · 미돌파")
                + (f" · I-AP 시작 {r.frames.w.index[lc.iap_start_i].date()}" if lc.iap_start_i is not None else ""))
    if r.buy_points:
        st.dataframe(pd.DataFrame([{"매수 지점": b.kind, "신호일": b.signal_date.date(), "기준가": b.level, "메모": b.note} for b in r.buy_points]),
                     hide_index=True, use_container_width=True)
    if r.trades:
        st.dataframe(pd.DataFrame([{"매수": t.kind, "규칙": t.rule, "진입": t.entry_date, "진입가": t.entry_px, "청산": t.exit_date, "수익": t.pct_gain,
                                    "DD $K": t.drawdown_k, "피크이익 보전": t.peak_retained, "보유일": t.days, "사유": t.reason, "미청산": t.open_position}
                                   for t in r.trades]), hide_index=True, use_container_width=True,
                     column_config={"수익": st.column_config.NumberColumn(format="percent"), "피크이익 보전": st.column_config.NumberColumn(format="percent")})
    if r.screen is not None:
        s = r.screen
        st.info(f"기준일 {_state['as_of']} 스크린: 단계 {s.phase} · 패턴(지금까지) {s.pattern} · {s.base_status} · IPO Alert {s.ipo_alert} · "
                f"거래대금 ${s.dollar_vol_m:.1f}M · 52주저 대비 {s.above_52w_low:+.0%} · vs 회색선 {s.vs_gray:+.0%}")
    st.markdown(f"**수동 체크 (문서 §B) — {sel} @ {_state['as_of']}**")
    ctx = f"{sel}_{_state['as_of']}"
    done = sum(int(st.checkbox(label, key=f"lcm_{ctx}_{k}")) for k, label in _MANUAL)
    st.progress(done / len(_MANUAL), text=f"{done}/{len(_MANUAL)} 확인")
