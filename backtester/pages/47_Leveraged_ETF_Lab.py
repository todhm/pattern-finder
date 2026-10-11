"""레버리지 ETF(TQQQ·SOXL) 사고팔기 전략 랩 — "장기보유보다 나은가"를 전략 하나하나 같은 자로 검증.

전략·근거: docs/strategy_notes/레버리지ETF_전략_검증_2026-10.md
엔진: strategy/adapters/leveraged_etf_lab.py (신호·원장·9Sig) · leveraged_etf_data.py (실데이터 + 합성 백캐스트)
배치: leveraged_etf_cli.py (여러 ETF·기간·파라미터 → JSON/MD)
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from data.adapters.composed_market_data import build_default_market_data
from strategy.adapters.leveraged_etf_data import ETF_SPECS, UNDERLYING_START, load_lab_prices
from strategy.adapters.short_term_lab import SHORT_TERM_ORDER, ShortTermParams, run_short_term, trade_stats
from strategy.adapters.leveraged_etf_lab import (
    STRATEGY_ORDER,
    CostParams,
    LabParams,
    crisis_table,
    current_signal,
    param_grid,
    run_lab,
    summary_table,
    trade_log,
    yearly_returns,
)

st.set_page_config(page_title="레버리지 ETF 전략 랩", layout="wide")
st.title("레버리지 ETF 사고팔기 전략 랩 — 장기보유를 이기는가")
st.caption(
    "TQQQ·SOXL 을 **같은 가격·같은 비용(토스 수수료 + 양도세 22%)** 으로 ① 장기보유 ② 200일선 추세 필터(Gayed & Bilello) "
    "③ 변동성 타기팅 ④ ②×③ ⑤ 9Sig(Jason Kelly) 를 하나씩 돌려 CAGR·MDD·위기 구간 손실·매매 횟수를 비교한다. "
    "2010년 이후만 보면 장기보유가 거의 항상 이긴다(역사상 최강 나스닥 강세장) — **닷컴 붕괴(2000~02)를 포함한 합성 백캐스트**로 "
    "봐야 '사고팔기'의 가치가 드러나는지가 이 페이지의 질문. 신호는 종가로 판정하고 같은 종가에 체결(24·28 페이지와 동일)."
)

def _shade(v) -> str:
    """수익률 셀 배경 — matplotlib 없이 부호·크기로 색칠."""
    if v is None or pd.isna(v):
        return ""
    a = min(abs(float(v)), 0.8) / 0.8
    return f"background-color: rgba({'220,53,69' if v < 0 else '40,167,69'},{0.12 + 0.5 * a:.2f})"


_EXPLAIN = {
    STRATEGY_ORDER[0]: "기준선. 첫날 전액 매수 후 보유. 세금은 마지막 날 전량 매도 가정으로 '청산 후 배수'에만 반영.",
    STRATEGY_ORDER[1]: "신호 지수(QQQ 등)가 N일 SMA 위면 100% 보유, 아래면 현금(T-bill 이자). 진입/이탈 밴드를 주면 SMA 근처 휩소가 줄고, "
                       "주봉 판정은 매매 횟수를 더 줄인다. 1928~2015 S&P 2x 백테스트에서 샤프 0.51 vs 0.30 (논문).",
    STRATEGY_ORDER[2]: "비중 = min(상한, 목표 변동성 ÷ 최근 n일 실현 변동성). 레버리지 ETF 의 손실은 하락 자체보다 고변동성 구간의 "
                       "변동성 끌림(∝σ²)에서 오므로, 변동성이 치솟을 때 비중을 줄인다. 매일 리밸런싱 → '리밸런싱 밴드'로 잔거래 억제.",
    STRATEGY_ORDER[3]: "② 가 켜져 있을 때만 ③ 비중으로 보유. 추세 꺾임(현금)과 변동성 급등(축소) 두 방어선.",
    STRATEGY_ORDER[4]: "주식 60 : 현금 40 시작, 분기마다 주식 목표치 V 를 ×1.09 올리고 초과분 매도·부족분 매수(현금 한도). "
                       "라오어 VR(28 페이지)과 같은 밸류패스 계열 — 하락장에서 계속 사므로 MDD 방어는 약하다. 원전의 채권 슬리브는 현금으로 대체.",
    STRATEGY_ORDER[5]: "**큰 조정일 때만 판다.** 신호 지수가 N일 SMA 아래 **그리고** 52주 고점 대비 낙폭 > d% 일 때 이탈, 종가가 20일 SMA 를 "
                       "되찾으면 즉시 재진입. 2026-10 격자 탐색(82개 규칙)에서 세전으로 유일하게 TQQQ 2010~ 장기보유를 넘긴 계열 "
                       "(d=15~18%: 345~489x vs 354x, MDD −61~−68%). 1999~ 는 228~303x vs 1.0x. 세후(22%)로는 장기보유의 55~85%. "
                       "d=10~12% 나 20% 는 절반 이하 — 파라미터 민감. 거래는 27년간 30회 안팎(대부분 2000·2002·2008·2022 의 긴 이탈).",
    SHORT_TERM_ORDER[0]: "Connors & Alvarez (2008). 신호 지수가 200일선 위일 때 2일 RSI < 10 이면 종가 매수, 종가 > 5일 SMA 면 청산. "
                         "책: 1993~2008 SPY 승률 80%+, 평균 보유 3~4일. 2010 이후 효과가 줄었다는 재현 보고가 많아 OOS 가 시험대.",
    SHORT_TERM_ORDER[1]: "Connors 'Double 7s'. 200일선 위에서 종가가 7일 최저 종가면 매수, 7일 최고 종가면 청산. 규칙이 가장 단순한 평균회귀.",
    SHORT_TERM_ORDER[2]: "Raschke 'Holy Grail' (Street Smarts). ADX(14) > 30 상승 추세(5일 전 주가 > 20EMA)에서 저가가 20일 EMA 를 터치, "
                         "3일 안에 종가가 터치 봉 고가를 넘으면 매수. 청산: 되돌림 이전 20일 최고가 도달 / 되돌림 저점 이탈 / 10일.",
    SHORT_TERM_ORDER[3]: "Raschke 'Turtle Soup'. 직전 20일 저점(4일 이상 전)을 오늘 저가가 깨고 종가가 다시 위로 복귀하면 매수(가짜 돌파 반전). "
                         "청산: 진입일 저가 이탈 / 5일. 책은 다음날 매수 스톱(Plus One) 변형도 제시.",
}

with st.sidebar:
    st.header("대상")
    etf = st.selectbox("거래 ETF", list(ETF_SPECS), index=0,
                       help="\n".join(f"{k}: {v.note}" for k, v in ETF_SPECS.items()))
    spec = ETF_SPECS[etf]
    signal = st.selectbox("신호 지수 (SMA 판정)", ["QQQ", "SPY", "SOXX", etf],
                          index=["QQQ", "SPY", "SOXX", etf].index(spec.underlying),
                          help="레버리지 ETF 자체보다 기초지수로 판정하는 게 통례(노이즈 ↓). ETF 자체를 고르면 상장 이후만 가능.")
    use_synthetic = st.toggle("상장 이전 합성 백캐스트", value=True,
                              help=f"{spec.underlying}×{spec.leverage:g} − ^IRX 조달비용 − 실측 드래그로 백캐스트, "
                                   f"{spec.inception} 부터 실데이터 스플라이스 (24 페이지와 같은 모델).")
    earliest = UNDERLYING_START.get(spec.underlying, spec.inception) if use_synthetic else spec.inception
    default_start = max(earliest, date(1999, 6, 1)) if use_synthetic else spec.inception
    start = st.date_input("평가 시작", value=default_start, min_value=earliest, max_value=date.today())
    end = st.date_input("평가 끝", value=date.today(), min_value=start, max_value=date.today())

    st.header("비용 / 체결")
    with_tax = st.toggle("양도세 22% (연 250만 공제)", value=True)
    exec_lag = st.radio("체결", ["신호 종가 (lag 0)", "다음날 종가 (lag 1)"], horizontal=True)
    rebalance_band = st.slider("리밸런싱 밴드 (자본 대비 %)", 0, 20, 5,
                               help="목표 비중과 현재 비중 차이가 이 이하면 거래하지 않음 — ③④ 의 매일 잔거래·세금 억제. ② 0/1 신호엔 영향 없음.") / 100
    costs = CostParams(tax_rate=0.22 if with_tax else 0.0, exec_lag=0 if exec_lag.endswith("0)") else 1,
                       rebalance_band=rebalance_band)

    st.header("② 추세 필터")
    sma_n = st.number_input("SMA 일수", 20, 400, 200, step=10)
    sma_enter = st.slider("진입 밴드 (+% 위)", 0.0, 10.0, 0.0, 0.5) / 100
    sma_exit = st.slider("이탈 밴드 (−% 아래)", 0.0, 10.0, 0.0, 0.5) / 100
    sma_weekly = st.toggle("주봉(금요일)만 판정", value=False)
    sma_monthly = st.toggle("월말에만 판정 (Faber 식)", value=False)
    st.header("⑩ 급락 회피")
    dd_n = st.number_input("추세 SMA 일수", 100, 300, 200, step=50)
    dd_depth = st.slider("이탈 낙폭 (52주 고점 대비 %)", 5, 30, 15, 1,
                         help="SMA 아래이면서 이 낙폭을 넘을 때만 판다. 15~18% 가 한 묶음으로 좋고 10~12%·20% 는 휩소.") / 100
    dd_reenter = st.number_input("재진입 SMA 일수", 5, 100, 20, step=5, help="빠를수록 V 자 반등을 덜 놓치지만 휩소가 는다.")
    dd_off = st.slider("이탈 중 잔여 비중", 0.0, 0.5, 0.0, 0.33, help="0.33 ≈ 1배 지수 노출 유지")
    st.header("③ 변동성 타기팅")
    vol_target = st.slider("목표 변동성 (연 %)", 10, 100, 45, 5) / 100
    vol_lookback = st.number_input("실현 변동성 창(일)", 5, 120, 20, step=5)
    vol_max = st.slider("비중 상한", 0.5, 1.5, 1.0, 0.1, help="1.0 초과 = 신용 가정(여기선 현금 음수 허용 안 함 → 1.0 캡).")
    st.header("⑤ 9Sig")
    sig_w = st.slider("초기 주식 비중", 0.3, 0.9, 0.6, 0.05)
    sig_g = st.slider("분기 목표 성장률 (%)", 3, 15, 9) / 100
    sig_freq = st.radio("리밸런싱 주기", ["Q", "M"], horizontal=True)
    st.header("⑥~⑨ 단기 고수 전략")
    with_short = st.toggle("코너스 RSI(2)·Double 7s / 라시케 Holy Grail·Turtle Soup 포함", value=True)
    rsi_buy = st.slider("RSI(2) 매수 임계", 2, 30, 10, 1)
    adx_min = st.slider("Holy Grail ADX 최소", 20, 40, 30, 5)
    ts_plus_one = st.toggle("Turtle Soup Plus One (다음날 진입)", value=False)

params = LabParams(sma_n=int(sma_n), sma_enter=sma_enter, sma_exit=sma_exit, sma_weekly=sma_weekly, sma_monthly=sma_monthly,
                   dd_n=int(dd_n), dd_depth=dd_depth, dd_reenter=int(dd_reenter), dd_off_fraction=dd_off,
                   vol_target=vol_target, vol_lookback=int(vol_lookback), vol_max=min(vol_max, 1.0),
                   sig_stock_weight=sig_w, sig_growth=sig_g, sig_freq=sig_freq)


@st.cache_data(ttl=3600, show_spinner="가격 데이터 로딩…")
def _prices(etf: str, signal: str, start: date, end: date, synthetic: bool):
    md = build_default_market_data()
    p = load_lab_prices(md, etf, signal, start, end, synthetic=synthetic)
    return p.lev_close, p.signal_close, p.cash_rate, p.synthetic_until, p.source_note, p.signal_df


try:
    lev_close, signal_close, cash_rate, synthetic_until, source_note, signal_df = _prices(etf, signal, start, end, use_synthetic)
except Exception as e:  # noqa: BLE001
    st.error(f"데이터 로딩 실패: {e}")
    st.stop()

if lev_close[lev_close.index >= pd.Timestamp(start)].empty:
    st.warning("평가 구간에 데이터가 없습니다. 시작일을 조정하세요.")
    st.stop()

st.info(f"데이터: {source_note} · 신호 {signal} · 현금 이자 {'^IRX 실제' if cash_rate is not None else '0%'} · "
        f"평가 {max(start, lev_close.index[0].date())} ~ {lev_close.index[-1].date()}")

with st.spinner("전략 시뮬레이션…"):
    results = run_lab(lev_close, signal_close, params, costs, cash_rate, start)
    short_results: dict = {}
    if with_short and signal_df is not None:
        short_results = run_short_term(lev_close, signal_df, ShortTermParams(rsi_buy=float(rsi_buy), adx_min=float(adx_min),
                                                                              ts_plus_one=ts_plus_one), costs, cash_rate, start)
        results.update(short_results)

# ───────────────────────────── 지금 신호
now = current_signal(lev_close, signal_close, params)
c1, c2, c3, c4 = st.columns(4)
c1.metric(f"{signal} vs SMA{params.sma_n}", f"{now['dist_to_sma'] * 100:+.1f}%",
          "② 보유" if now["filter_on"] else "② 현금")
c2.metric(f"{etf} 실현 변동성({params.vol_lookback}일)", f"{now['realized_vol'] * 100:.0f}%",
          f"③ 비중 {now['vol_weight'] * 100:.0f}%")
c3.metric("④ 결합 비중", f"{now['combined_weight'] * 100:.0f}%")
sig_res = results[STRATEGY_ORDER[4]]
c4.metric("⑤ 9Sig 현재 주식 비중", f"{sig_res.exposure.iloc[-1] * 100:.0f}%",
          f"목표 V 대비 {(sig_res.exposure.iloc[-1] / sig_res.target.iloc[-1] - 1) * 100:+.0f}%" if sig_res.target.iloc[-1] > 0 else "")
st.caption(f"기준 {now['as_of']} — 지금 각 전략이 말하는 것. ⑤ 는 다음 분기 첫 거래일에 V 대비 초과분 매도 / 부족분 매수.")

# ───────────────────────────── 비교 요약
st.subheader("비교 요약 — 같은 돈, 같은 비용")
summ = summary_table(results)
st.dataframe(
    summ.style.format({"최종 배수": "{:.1f}x", "CAGR": "{:+.1%}", "MDD": "{:+.1%}", "최악 연도": "{:+.1%}", "연 변동성": "{:.0%}",
                       "샤프": "{:.2f}", "칼마": "{:.2f}", "시장 노출": "{:.0%}", "수수료+세금": "{:.1f}x 원금",
                       "청산 후 배수": "{:.1f}x"}),
    use_container_width=True, hide_index=True,
)
bh = results[STRATEGY_ORDER[0]]
best = summ.sort_values("CAGR", ascending=False).iloc[0]
if best["전략"] == STRATEGY_ORDER[0]:
    st.warning(f"이 구간에서는 **장기보유가 CAGR 1위**({best['CAGR']:+.1%}). 사고팔기의 효과는 MDD 열(장기보유 {bh.metrics['mdd']:+.0%})에서만 나타난다. "
               "닷컴 붕괴를 포함하도록 시작일을 1999~2000년으로 당겨 보라.")
else:
    st.success(f"**{best['전략']}** 가 CAGR {best['CAGR']:+.1%} 로 장기보유({bh.metrics['cagr']:+.1%})를 앞선다. "
               f"MDD {best['MDD']:+.0%} vs {bh.metrics['mdd']:+.0%}.")

fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.03,
                    subplot_titles=("평가액 (로그, 원금=1)", "낙폭"))
for k, r in results.items():
    e = r.equity / r.equity.iloc[0]
    fig.add_trace(go.Scatter(x=e.index, y=e, name=k, mode="lines"), row=1, col=1)
    dd = r.equity / r.equity.cummax() - 1
    fig.add_trace(go.Scatter(x=dd.index, y=dd, name=k, mode="lines", showlegend=False, opacity=0.7), row=2, col=1)
if synthetic_until is not None and pd.Timestamp(synthetic_until) > pd.Timestamp(start):
    fig.add_vrect(x0=pd.Timestamp(start), x1=pd.Timestamp(synthetic_until), fillcolor="gray", opacity=0.08, line_width=0,
                  annotation_text="합성 구간", annotation_position="top left", row=1, col=1)
fig.update_yaxes(type="log", row=1, col=1)
fig.update_yaxes(tickformat=".0%", row=2, col=1)
fig.update_layout(height=620, legend=dict(orientation="h", y=1.08), margin=dict(l=40, r=20, t=60, b=30))
st.plotly_chart(fig, use_container_width=True)

cA, cB = st.columns([1, 1])
with cA:
    st.markdown("**위기 구간 (고점→저점) 손실** — 장기보유 대비 얼마나 덜 잃었나")
    ct = crisis_table(results)
    st.dataframe(ct.style.format({k: "{:+.1%}" for k in STRATEGY_ORDER}, na_rep="데이터 밖"),
                 use_container_width=True, hide_index=True)
with cB:
    st.markdown("**연도별 수익률**")
    yr = yearly_returns(results)
    st.dataframe(yr.style.format("{:+.1%}").map(_shade), use_container_width=True, height=min(600, 40 + 35 * len(yr)))

if short_results:
    st.markdown("**⑥~⑨ 단기 전략 거래 통계** (가격 기준, 비용 전 — 책이 말한 '승률 높고 짧게' 가 이 데이터에서도 성립하는지)")
    ts_rows = []
    for k, r in short_results.items():
        s = trade_stats(r, lev_close)
        ts_rows.append({"전략": k, "거래": s["trades"], "연간": s["per_year"], "승률": s.get("win_rate"), "평균 보유일": s.get("avg_days"),
                        "평균 수익": s.get("avg_ret"), "중앙값": s.get("median_ret"), "최악": s.get("worst"), "최고": s.get("best"),
                        "시장 노출": r.time_in_market})
    st.dataframe(pd.DataFrame(ts_rows).style.format({"연간": "{:.1f}", "승률": "{:.0%}", "평균 보유일": "{:.1f}", "평균 수익": "{:+.2%}",
                                                      "중앙값": "{:+.2%}", "최악": "{:+.1%}", "최고": "{:+.1%}", "시장 노출": "{:.0%}"}, na_rep="—"),
                 use_container_width=True, hide_index=True)

# ───────────────────────────── 전략별 탭
st.subheader("전략 하나하나")
_tab_names = list(results)
tabs = st.tabs(_tab_names + ["민감도(과최적화 점검)"])
for tab, name in zip(tabs[:-1], _tab_names):
    with tab:
        r = results[name]
        st.markdown(_EXPLAIN.get(name, ""))
        m = r.metrics
        k1, k2, k3, k4, k5, k6 = st.columns(6)
        k1.metric("최종 배수", f"{r.equity.iloc[-1] / r.equity.iloc[0]:.1f}x", f"청산 후 {r.final_after_liquidation / r.equity.iloc[0]:.1f}x")
        k2.metric("CAGR", f"{m['cagr']:+.1%}", f"장기보유 {bh.metrics['cagr']:+.1%}")
        k3.metric("MDD", f"{m['mdd']:+.1%}", f"장기보유 {bh.metrics['mdd']:+.1%}")
        k4.metric("매매 횟수", r.trades, f"시장 노출 {r.time_in_market:.0%}")
        k5.metric("수수료+세금", f"{(r.fees + r.tax) / r.equity.iloc[0]:.2f}x 원금")
        k6.metric("최장 수면", f"{m['longest_underwater_days']}일", f"복구 {m['max_dd_recovery_days'] or '미완'}일")
        f2 = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.65, 0.35], vertical_spacing=0.03,
                           subplot_titles=(f"{name} vs 장기보유 (로그)", "보유 비중"))
        f2.add_trace(go.Scatter(x=bh.equity.index, y=bh.equity / bh.equity.iloc[0], name="장기보유", line=dict(color="gray")), row=1, col=1)
        f2.add_trace(go.Scatter(x=r.equity.index, y=r.equity / r.equity.iloc[0], name=name), row=1, col=1)
        f2.add_trace(go.Scatter(x=r.exposure.index, y=r.exposure, name="실제 비중", fill="tozeroy", line=dict(width=1)), row=2, col=1)
        if name == STRATEGY_ORDER[4]:
            f2.add_trace(go.Scatter(x=r.target.index, y=r.target, name="목표 V / 평가액", line=dict(dash="dot")), row=2, col=1)
        f2.update_yaxes(type="log", row=1, col=1)
        f2.update_yaxes(tickformat=".0%", row=2, col=1)
        f2.update_layout(height=520, legend=dict(orientation="h", y=1.08), margin=dict(l=40, r=20, t=60, b=30))
        st.plotly_chart(f2, use_container_width=True)
        if name != STRATEGY_ORDER[0]:
            tl = trade_log(r, min_change=0.05 if name in (STRATEGY_ORDER[2], STRATEGY_ORDER[3]) else 0.01)
            st.markdown(f"**매매 일지** (비중 변화 ≥ {'5' if name in (STRATEGY_ORDER[2], STRATEGY_ORDER[3]) else '1'}%p, {len(tl)}건)")
            st.dataframe(tl.style.format({"비중 변화": "{:+.0%}", "비중": "{:.0%}", "평가액": "{:,.0f}"}),
                         use_container_width=True, hide_index=True, height=300)

with tabs[-1]:
    st.markdown("한 파라미터에서만 좋으면 과최적화다. 이웃 파라미터에서도 같은 방향이어야 믿을 수 있다. (각 행 = 전체 구간 재실행)")
    if st.button("SMA 필터 · 변동성 타기팅 민감도 실행", type="primary"):
        with st.spinner("격자 실행…"):
            sma_grid = [LabParams(sma_n=n, sma_enter=e, sma_exit=x, sma_weekly=w)
                        for n in (100, 150, 200, 250) for (e, x) in ((0.0, 0.0), (0.03, 0.03), (0.05, 0.03)) for w in (False, True)]
            vol_grid = [LabParams(vol_target=t, vol_lookback=lb) for t in (0.30, 0.45, 0.60) for lb in (10, 20, 40)]
            g1 = param_grid(lev_close, signal_close, sma_grid, costs, cash_rate, start, STRATEGY_ORDER[1])
            g2 = param_grid(lev_close, signal_close, vol_grid, costs, cash_rate, start, STRATEGY_ORDER[2])
        st.markdown(f"**② SMA 필터** — 장기보유 CAGR {bh.metrics['cagr']:+.1%} / MDD {bh.metrics['mdd']:+.0%}")
        st.dataframe(g1[["sma_n", "enter", "exit", "weekly", "CAGR", "MDD", "샤프", "매매", "최종 배수"]]
                     .style.format({"enter": "{:.0%}", "exit": "{:.0%}", "CAGR": "{:+.1%}", "MDD": "{:+.0%}", "샤프": "{:.2f}", "최종 배수": "{:.1f}x"})
                     .map(_shade, subset=["CAGR"]), use_container_width=True, hide_index=True)
        st.markdown("**③ 변동성 타기팅**")
        st.dataframe(g2[["vol_target", "lookback", "CAGR", "MDD", "샤프", "매매", "최종 배수"]]
                     .style.format({"vol_target": "{:.0%}", "CAGR": "{:+.1%}", "MDD": "{:+.0%}", "샤프": "{:.2f}", "최종 배수": "{:.1f}x"})
                     .map(_shade, subset=["CAGR"]), use_container_width=True, hide_index=True)

with st.expander("읽는 법 / 한계"):
    st.markdown("""
- **최종 배수**는 연말마다 양도세를 내고 난 평가액, **청산 후 배수**는 마지막 날 전량 매도 + 세금까지 낸 값. 장기보유는 세금이 한 번에 몰려 둘의 차이가 크다.
- **위기 구간**은 지수 고점→저점 창에서 각 전략의 손실. 사고팔기의 효과는 여기와 MDD 에서 나오고, 강세장 CAGR 에서는 보통 장기보유에 진다.
- 2010년 이후 실데이터만 보면 역사상 최강 나스닥 강세장이라 장기보유가 이긴다. 합성 백캐스트(1999~)는 닷컴 붕괴를 포함하지만 **합성 모델의 근사**(드래그·조달비용)에 의존한다.
- 종가 판정 = 종가 체결(lag 0)은 낙관적이다. 보수적 수치는 lag 1 로 확인.
- 변동성 타기팅은 매일 리밸런싱이라 세금이 크다. 리밸런싱 밴드를 키우면 매매가 줄고 수익률은 약간 변한다.
- 9Sig 의 원전은 채권(AGG) 슬리브다. 여기서는 현금(^IRX)이라 2010년대 채권 강세의 이점은 빠지고 2022년의 채권 손실도 빠진다.
""")
