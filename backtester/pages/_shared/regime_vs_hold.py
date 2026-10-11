"""하락장 회피 규칙 vs 장기보유 — 49(TQQQ 급락 회피)·50(SOXL 월말 SMA) 페이지 공용 렌더러.

두 페이지는 규칙 계열(kind)과 기본 ETF·신호만 다르고, 오늘의 판단 → 구간별 결과 → 회피 일지 → 시작 시점별 롤링 → 격자 구성은 같다.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from data.adapters.composed_market_data import build_default_market_data
from strategy.adapters.deep_correction_lab import (
    DeepParams,
    MonthlySmaParams,
    avoided_segments,
    decision,
    decision_monthly,
    param_grid,
    param_grid_monthly,
    rolling_study,
    rolling_summary,
    run_pair,
    signal,
    signal_monthly_sma,
)
from strategy.adapters.leveraged_etf_data import ETF_SPECS, UNDERLYING_START, load_lab_prices
from strategy.adapters.leveraged_etf_lab import CostParams, crisis_table, yearly_returns


@st.cache_data(ttl=6 * 3600, show_spinner="가격 로딩…")
def _prices(etf: str, sig_sym: str, start: date, synthetic: bool):
    pr = load_lab_prices(build_default_market_data(), etf, sig_sym, start, date.today(), synthetic=synthetic)
    return pr.lev_close, pr.signal_close, pr.cash_rate, pr.source_note


def _shade_off(fig: go.Figure, s: pd.Series, idx: pd.DatetimeIndex) -> None:
    off_mask = s.reindex(idx).ffill() < 0.5
    if off_mask.any():
        blocks = (off_mask != off_mask.shift()).cumsum()
        for _, grp in off_mask[off_mask].groupby(blocks[off_mask]):
            fig.add_vrect(x0=grp.index[0], x1=grp.index[-1], fillcolor="red", opacity=0.08, line_width=0)


def _decision_deep(sigc: pd.Series, s: pd.Series, p: DeepParams, sig_sym: str, what_if: float | None) -> None:
    dec = decision(sigc, p, hypothetical_close=what_if)
    color = {"보유": st.success, "재진입 매수": st.success, "매도": st.error, "현금 유지": st.warning}[dec["action"]]
    basis = f"가정 종가 {dec['close']:.2f}" if dec["hypothetical"] else f"{dec['as_of']} 종가 {dec['close']:.2f}"
    color(f"**{dec['action']}** — {basis} 기준 · 전날 상태 {'보유' if dec['prev_holding'] else '현금'} → 오늘 {'보유' if dec['holding_after'] else '현금'} "
          f"(현재 상태 {dec['state_days']}일째, 마지막 변경 {dec['last_change']})")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric(f"{sig_sym} vs SMA{p.n}", f"{dec['dist_sma']:+.1%}", f"SMA {dec['sma_n']:.2f}")
    k2.metric("52주 고점 대비", f"{dec['drawdown']:+.1%}", f"고점 {dec['hi252']:.2f}")
    k3.metric(f"이탈선 (고점 −{p.dd:.0%})", f"{dec['exit_level']:.2f}", f"현재가 대비 {dec['dist_exit']:+.1%}")
    k4.metric(f"재진입 SMA{p.reenter_n}", f"{dec['sma_reenter']:.2f}", f"현재가 대비 {dec['dist_reenter']:+.1%}")
    cond = pd.DataFrame([
        {"조건": f"① 종가 < SMA{p.n}", "값": f"{dec['close']:.2f} vs {dec['sma_n']:.2f}", "충족": "✅" if dec["cond_below_sma"] else "❌"},
        {"조건": f"② 종가 < 52주 고점 × (1 − {p.dd:.0%})", "값": f"{dec['close']:.2f} vs {dec['exit_level']:.2f}", "충족": "✅" if dec["cond_deep_dd"] else "❌"},
        {"조건": f"③ 종가 > SMA{p.reenter_n}", "값": f"{dec['close']:.2f} vs {dec['sma_reenter']:.2f}", "충족": "✅" if dec["cond_above_reenter"] else "❌"},
    ])
    cL, cR = st.columns([1, 1])
    with cL:
        st.dataframe(cond, use_container_width=True, hide_index=True)
    with cR:
        if dec["holding_after"]:
            st.markdown(f"**매도 조건**: ①과 ② 둘 다 — 즉 {sig_sym} 종가가 **{dec['sell_trigger']:.2f}** 아래로 마감하면 다음 거래일에 판다 "
                        f"(현재가 대비 {dec['sell_trigger'] / dec['close'] - 1:+.1%}). 그 전까지는 아무것도 하지 않는다.")
        else:
            st.markdown(f"**재진입 조건**: ③ 이고 이탈 조건(①∧②)이 풀려야 한다 — 즉 {sig_sym} 종가가 **{dec['buy_trigger']:.2f}** 위로 마감하면 다음 거래일에 산다 "
                        f"(현재가 대비 {dec['buy_trigger'] / dec['close'] - 1:+.1%}).")
        st.caption("판정은 종가 기준이고 백테스트는 같은 종가에 체결한다. 실제로는 다음 거래일 체결이므로 사이드바 '다음날 종가' 로 보수적 수치를 확인.")
    recent = sigc.tail(300)
    fig0 = go.Figure()
    fig0.add_trace(go.Scatter(x=recent.index, y=recent, name=f"{sig_sym} 종가", line=dict(color="black")))
    fig0.add_trace(go.Scatter(x=recent.index, y=sigc.rolling(p.n).mean().tail(300), name=f"SMA{p.n}", line=dict(color="#1f77b4")))
    fig0.add_trace(go.Scatter(x=recent.index, y=sigc.rolling(p.reenter_n).mean().tail(300), name=f"SMA{p.reenter_n}", line=dict(color="#2ca02c", dash="dot")))
    fig0.add_trace(go.Scatter(x=recent.index, y=(sigc.rolling(252, min_periods=20).max() * (1 - p.dd)).tail(300),
                              name=f"이탈선 (고점 −{p.dd:.0%})", line=dict(color="#d62728", dash="dash")))
    _shade_off(fig0, s, recent.index)
    fig0.update_layout(height=380, legend=dict(orientation="h", y=1.08), margin=dict(l=40, r=20, t=40, b=30), title="최근 300거래일 — 빨간 음영 = 현금 구간")
    st.plotly_chart(fig0, use_container_width=True)


def _decision_monthly(sigc: pd.Series, s: pd.Series, mp: MonthlySmaParams, sig_sym: str, what_if: float | None) -> None:
    dm = decision_monthly(sigc, mp, hypothetical_close=what_if)
    banner = st.success if dm["holding"] else st.warning
    basis = f"가정 종가 {dm['close']:.2f}" if dm["hypothetical"] else f"{dm['as_of']} 종가 {dm['close']:.2f}"
    banner(f"**{dm['action']}** — 지난 월말({dm['last_decision']}) 판정 유지 · 다음 판정 **{dm['next_decision']}** ({dm['days_to_decision']}일 뒤) · "
           f"현재 비중 {dm['exposure_now']:.0%} · 마지막 변경 {dm['last_change']}")
    k1, k2, k3 = st.columns(3)
    k1.metric(f"{sig_sym} vs SMA{mp.n}", f"{dm['dist_sma']:+.1%}", f"SMA {dm['sma']:.2f}")
    k2.metric("오늘이 월말이라면", dm["if_month_end_now"], "변경됨" if dm["would_change"] else "변경 없음", delta_color="inverse" if dm["would_change"] else "off")
    k3.metric("월말 판정 기준가", f"{dm['sma']:.2f}", basis)
    st.markdown(f"**규칙**: {dm['next_decision']} 종가가 SMA{mp.n}({dm['sma']:.2f} 근방) **위**면 다음 달 100% 보유, **아래**면 {mp.off_fraction:.0%} 만 남기고 매도. "
                "월중에는 어떤 가격이 와도 행동하지 않는다.")
    st.caption("SOXL 은 하루 5% 가 보통이라 체결 하루 차이가 결과의 1/3 을 바꾼다 — 사이드바 '다음날 종가' 로 보수적 수치를 꼭 확인.")
    recent = sigc.tail(300)
    fig0 = go.Figure()
    fig0.add_trace(go.Scatter(x=recent.index, y=recent, name=f"{sig_sym} 종가", line=dict(color="black")))
    fig0.add_trace(go.Scatter(x=recent.index, y=sigc.rolling(mp.n).mean().tail(300), name=f"SMA{mp.n}", line=dict(color="#1f77b4")))
    _shade_off(fig0, s, recent.index)
    fig0.update_layout(height=360, legend=dict(orientation="h", y=1.08), margin=dict(l=40, r=20, t=40, b=30), title="최근 300거래일 — 빨간 음영 = 축소 구간")
    st.plotly_chart(fig0, use_container_width=True)


def render(kind: str, *, title: str, caption: str, etf_choices: list[str], signal_choices: list[str], rule_name: str, notes: str) -> None:
    """kind = "deep" (⑩ 급락 회피) | "monthly" (월말 SMA + 잔여 비중)."""
    st.set_page_config(page_title=title, layout="wide")
    st.title(title)
    st.caption(caption)

    with st.sidebar:
        st.header("대상")
        etf = st.selectbox("거래 ETF", etf_choices, index=0)
        spec = ETF_SPECS[etf]
        default_sig = spec.underlying if spec.underlying in signal_choices else signal_choices[0]
        sig_sym = st.selectbox("신호 지수", signal_choices, index=signal_choices.index(default_sig))
        earliest = max(UNDERLYING_START.get(spec.underlying, spec.inception), date(1999, 12, 1))
        use_synthetic = st.toggle("상장 이전 합성 백캐스트", value=True)
        st.header("규칙")
        if kind == "monthly":
            m_n = st.number_input("SMA 일수", 50, 250, 100, step=25)
            m_off = st.slider("이탈 중 잔여 비중", 0.0, 0.5, 0.33, 0.33, help="0.33 ≈ 1배 지수 노출 유지. SOXL 탐색(285개 규칙)에서 0 보다 일관되게 나았다.")
            p = None; mp = MonthlySmaParams(n=int(m_n), off_fraction=m_off)
        else:
            n = st.number_input("추세 SMA 일수", 100, 300, 200, step=50)
            dd = st.slider("이탈 낙폭 (52주 고점 대비 %)", 5, 30, 15, 1, help="15~25% 가 한 묶음. 10~12% 는 휩소로 장기보유의 1/4 이하.") / 100
            reenter = st.number_input("재진입 SMA 일수", 5, 100, 20, step=5)
            off = st.slider("이탈 중 잔여 비중", 0.0, 0.5, 0.0, 0.33)
            p = DeepParams(n=int(n), dd=dd, reenter_n=int(reenter), off_fraction=off); mp = None
        st.header("비용")
        with_tax = st.toggle("양도세 22%", value=True)
        lag = st.radio("체결", ["신호 종가", "다음날 종가"], horizontal=True)
        st.header("시작 시점별 분석")
        horizons = st.multiselect("보유 기간(년)", [1, 2, 3, 5, 7, 10], default=[1, 3, 5, 10])
        step = st.select_slider("시작월 간격(개월)", [1, 3, 6, 12], value=3)

    costs = CostParams(tax_rate=0.22 if with_tax else 0.0, exec_lag=0 if lag == "신호 종가" else 1, rebalance_band=0.02)
    try:
        lev, sigc, cash, note = _prices(etf, sig_sym, earliest if use_synthetic else spec.inception, use_synthetic)
    except Exception as e:  # noqa: BLE001
        st.error(f"데이터 로딩 실패: {e}")
        st.stop()
    st.info(f"{note} · 신호 {sig_sym} · 평가 {lev.index[0].date()} ~ {lev.index[-1].date()} · 규칙 **{rule_name}**")
    s = signal_monthly_sma(sigc, mp) if kind == "monthly" else signal(sigc, p)

    # ── 오늘의 판단
    st.subheader("오늘의 판단 — 보유할까, 팔까")
    what_if = st.number_input(f"{sig_sym} 오늘 종가 가정 (0 = 마지막 종가 그대로)", min_value=0.0, value=0.0, step=1.0,
                              help="장중에 쓰려면 지금 가격을 넣어 본다. 규칙은 종가로 판정하므로 장 마감 전 값은 참고용.")
    wi = float(what_if) if what_if > 0 else None
    if kind == "monthly":
        _decision_monthly(sigc, s, mp, sig_sym, wi)
    else:
        _decision_deep(sigc, s, p, sig_sym, wi)

    # ── 구간별 결과
    st.subheader("구간별 결과")
    rows, pairs = [], {}
    for label, start in (("전체(합성 포함)", lev.index[0].date()), ("실데이터(상장 이후)", spec.inception)):
        if pd.Timestamp(start) < lev.index[0]:
            continue
        hold, strat = run_pair(lev, s, start, None, costs, cash)
        pairs[label] = (hold, strat)
        for name, r in (("장기보유", hold), (rule_name, strat)):
            m = r.metrics
            rows.append({"구간": f"{label} {r.equity.index[0].date()}~", "전략": name, "최종 배수": r.equity.iloc[-1] / r.equity.iloc[0], "CAGR": m["cagr"],
                         "MDD": m["mdd"], "최악 연도": m["worst_year"], "샤프": m["sharpe"], "매매": r.trades, "노출": r.time_in_market,
                         "청산 후 배수": r.final_after_liquidation / r.equity.iloc[0]})
    st.dataframe(pd.DataFrame(rows).style.format({"최종 배수": "{:.1f}x", "CAGR": "{:+.1%}", "MDD": "{:+.0%}", "최악 연도": "{:+.0%}", "샤프": "{:.2f}",
                                                  "노출": "{:.0%}", "청산 후 배수": "{:.1f}x"}), use_container_width=True, hide_index=True)
    label0 = "전체(합성 포함)" if "전체(합성 포함)" in pairs else list(pairs)[0]
    hold0, strat0 = pairs[label0]
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.55, 0.25, 0.2], vertical_spacing=0.03,
                        subplot_titles=("평가액 (로그)", "낙폭", "규칙 비중"))
    for name, r, color in (("장기보유", hold0, "gray"), (rule_name, strat0, "#1f77b4")):
        fig.add_trace(go.Scatter(x=r.equity.index, y=r.equity / r.equity.iloc[0], name=name, line=dict(color=color)), row=1, col=1)
        fig.add_trace(go.Scatter(x=r.equity.index, y=r.equity / r.equity.cummax() - 1, name=name, showlegend=False, line=dict(color=color), opacity=0.7), row=2, col=1)
    fig.add_trace(go.Scatter(x=strat0.exposure.index, y=strat0.exposure, name="비중", fill="tozeroy", line=dict(width=1), showlegend=False), row=3, col=1)
    fig.update_yaxes(type="log", row=1, col=1); fig.update_yaxes(tickformat=".0%", row=2, col=1)
    fig.update_layout(height=640, legend=dict(orientation="h", y=1.06), margin=dict(l=40, r=20, t=50, b=30))
    st.plotly_chart(fig, use_container_width=True)

    cA, cB = st.columns(2)
    with cA:
        st.markdown("**회피 구간 일지** — 팔고 나서 다시 살 때까지 ETF 가 어떻게 됐나 (음수 = 회피 성공, 양수 = 휩소 비용)")
        seg = avoided_segments(strat0, lev, sigc)
        if len(seg):
            st.dataframe(seg.style.format({"회피 구간 ETF": "{:+.0%}", "회피 구간 지수": "{:+.0%}"}), use_container_width=True, hide_index=True, height=320)
            big = seg[seg["회피 구간 ETF"] < -0.3]
            st.caption(f"이탈 {len(seg)}회 · ETF −30% 이상 회피 {len(big)}회 · 휩소(+) {int((seg['회피 구간 ETF'] > 0).sum())}회")
    with cB:
        st.markdown("**연도별 / 위기 구간**")
        yr = yearly_returns({"규칙": strat0, "보유": hold0})
        st.dataframe(yr.style.format("{:+.0%}"), use_container_width=True, height=320)
        ct = crisis_table({"규칙": strat0, "보유": hold0})
        st.dataframe(ct.style.format({"규칙": "{:+.0%}", "보유": "{:+.0%}"}, na_rep="—"), use_container_width=True, hide_index=True)

    # ── 시작 시점별 (롤링)
    st.subheader("시작 시점별 세후 성과 — 분포로 보기")
    st.markdown("각 시작월에서 N년 보유했을 때 **규칙 배수 ÷ 장기보유 배수**. 1보다 크면 규칙 승. 같은 돈·같은 비용.")
    summ_rows, figs = [], {}
    for y in sorted(horizons):
        with st.spinner(f"{y}년 창 계산…"):
            df = rolling_study(lev, s, y, costs, cash, step_months=int(step))
        if df.empty:
            continue
        sm = rolling_summary(df)
        summ_rows.append({"보유 기간": f"{y}년", "창 수": sm["windows"], "규칙 승률": sm["win_rate"], "비율 중앙값": sm["ratio_median"], "비율 하위 10%": sm["ratio_p10"],
                          "비율 최악": sm["ratio_min"], "비율 최고": sm["ratio_max"], "규칙 CAGR 중앙": sm["strat_cagr_median"], "보유 CAGR 중앙": sm["hold_cagr_median"],
                          "규칙 최악 CAGR": sm["strat_cagr_min"], "보유 최악 CAGR": sm["hold_cagr_min"], "규칙 원금 손실 창": sm["strat_loss_rate"], "보유 원금 손실 창": sm["hold_loss_rate"],
                          "규칙 MDD 중앙": sm["strat_mdd_median"], "보유 MDD 중앙": sm["hold_mdd_median"]})
        figs[y] = df
    if summ_rows:
        st.dataframe(pd.DataFrame(summ_rows).style.format({"규칙 승률": "{:.0%}", "비율 중앙값": "{:.2f}", "비율 하위 10%": "{:.2f}", "비율 최악": "{:.2f}", "비율 최고": "{:.2f}",
                                                           "규칙 CAGR 중앙": "{:+.1%}", "보유 CAGR 중앙": "{:+.1%}", "규칙 최악 CAGR": "{:+.1%}", "보유 최악 CAGR": "{:+.1%}",
                                                           "규칙 원금 손실 창": "{:.0%}", "보유 원금 손실 창": "{:.0%}", "규칙 MDD 중앙": "{:+.0%}", "보유 MDD 중앙": "{:+.0%}"}),
                     use_container_width=True, hide_index=True)
        tabs = st.tabs([f"{y}년 보유" for y in figs])
        for tab, (y, df) in zip(tabs, figs.items()):
            with tab:
                f2 = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.5, 0.5], vertical_spacing=0.06,
                                   subplot_titles=(f"시작월별 {y}년 CAGR (세후)", "규칙 ÷ 장기보유 배수 비율 (로그, 1 = 동률)"))
                f2.add_trace(go.Scatter(x=df["start"], y=df["hold_cagr"], name="장기보유", line=dict(color="gray")), row=1, col=1)
                f2.add_trace(go.Scatter(x=df["start"], y=df["strat_cagr"], name=rule_name, line=dict(color="#1f77b4")), row=1, col=1)
                f2.add_trace(go.Bar(x=df["start"], y=df["ratio"], name="비율", marker_color=["#2ca02c" if r > 1 else "#d62728" for r in df["ratio"]], showlegend=False), row=2, col=1)
                f2.add_hline(y=1.0, line_dash="dot", row=2, col=1)
                f2.update_yaxes(tickformat=".0%", row=1, col=1); f2.update_yaxes(type="log", row=2, col=1)
                f2.update_layout(height=520, legend=dict(orientation="h", y=1.08), margin=dict(l=40, r=20, t=50, b=30))
                st.plotly_chart(f2, use_container_width=True)
                st.dataframe(df.style.format({"hold_mult": "{:.2f}x", "strat_mult": "{:.2f}x", "ratio": "{:.2f}", "hold_cagr": "{:+.1%}", "strat_cagr": "{:+.1%}",
                                              "hold_mdd": "{:+.0%}", "strat_mdd": "{:+.0%}"}), use_container_width=True, hide_index=True, height=260)

    # ── 파라미터 격자
    with st.expander("파라미터 격자 (실데이터 구간, 현재 비용) — 이웃에서도 같은 방향인지"):
        if kind == "monthly":
            g = param_grid_monthly(lev, sigc, spec.inception, costs, cash)
            piv = g.pivot(index="n", columns="off", values="vs_hold")
            st.markdown("**장기보유 대비 배수 비율** (행 = SMA 일수, 열 = 잔여 비중)")
            st.dataframe(piv.style.format("{:.2f}"), use_container_width=True)
            st.dataframe(g.style.format({"off": "{:.0%}", "mult": "{:.1f}x", "cagr": "{:+.1%}", "mdd": "{:+.0%}", "vs_hold": "{:.2f}"}), use_container_width=True, hide_index=True)
        else:
            g = param_grid(lev, sigc, spec.inception, costs, cash, n=p.n)
            piv = g.pivot(index="dd", columns="reenter", values="vs_hold")
            st.markdown("**장기보유 대비 배수 비율** (행 = 이탈 낙폭, 열 = 재진입 SMA)")
            st.dataframe(piv.style.format("{:.2f}"), use_container_width=True)
            st.dataframe(g.style.format({"dd": "{:.0%}", "mult": "{:.1f}x", "cagr": "{:+.1%}", "mdd": "{:+.0%}", "vs_hold": "{:.2f}"}), use_container_width=True, hide_index=True)

    with st.expander("읽는 법 / 한계"):
        st.markdown(notes)
