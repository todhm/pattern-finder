"""TQQQ 폭락 스위치 18 — 2026-10-19 확정 전략의 운용 화면. 손잡이 없음.

규칙: strategy.adapters.deep_correction_lab.CRASH_SWITCH_RULE (200일선 & 고점 −18% 이탈 / 20일선 재진입 / 다음날 체결 / TQQQ 100%).
49 페이지(TQQQ_Deep_Correction)는 기존 ⑩ 급락 회피 연구용 그대로이고, 이 페이지는 거기서 확정한 규칙 하나를 고정해 매일 보는 용도다.
선정 근거: docs/strategy_notes/study_cards/README.md (F2·F5·F7), 변동성타기팅_레짐_검증_2026-10.md, 참사_조기경보_검증_2026-10.md.
"""

from datetime import date

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data.adapters.composed_market_data import build_default_market_data
from strategy.adapters.deep_correction_lab import (
    CRASH_SWITCH_NAME, CRASH_SWITCH_PARAMS, CRASH_SWITCH_RULE, avoided_segments, decision, run_pair, signal,
)
from strategy.adapters.leveraged_etf_data import load_lab_prices
from strategy.adapters.leveraged_etf_lab import CostParams

P = CRASH_SWITCH_PARAMS
COSTS = CostParams(tax_rate=0.22, exec_lag=1, rebalance_band=0.02)  # 토스 수수료 + 양도세 22%, 다음날 종가 체결

st.set_page_config(page_title=CRASH_SWITCH_NAME, layout="wide")
st.title(f"{CRASH_SWITCH_NAME} — 오늘 보유 여부")
st.caption(f"**규칙 고정**: {CRASH_SWITCH_RULE} 파라미터를 바꿔 보려면 49 페이지(연구용).")


@st.cache_data(ttl=6 * 3600, show_spinner="가격 로딩…")
def _prices():
    pr = load_lab_prices(build_default_market_data(), "TQQQ", "QQQ", date(1999, 12, 1), date.today(), synthetic=True)
    return pr.lev_close, pr.signal_close, pr.cash_rate, pr.source_note


try:
    lev, sigc, cash, note = _prices()
except Exception as e:  # noqa: BLE001
    st.error(f"데이터 로딩 실패: {e}")
    st.stop()
s = signal(sigc, P)

# ── 오늘 보유 여부
st.subheader(f"오늘 보유 여부 — {sigc.index[-1].date()} 종가 기준")
what_if = st.number_input("QQQ 오늘 종가 가정 (0 = 마지막 종가 그대로)", min_value=0.0, value=0.0, step=1.0,
                          help="장중에 보려면 지금 가격을 넣는다. 규칙은 종가로 판정하므로 장 마감 전 값은 참고용.")
dec = decision(sigc, P, hypothetical_close=float(what_if) if what_if > 0 else None)
color = {"보유": st.success, "재진입 매수": st.success, "매도": st.error, "현금 유지": st.warning}[dec["action"]]
basis = f"가정 종가 {dec['close']:.2f}" if dec["hypothetical"] else f"{dec['as_of']} 종가 {dec['close']:.2f}"
color(f"**{dec['action']}** — {basis} 기준 · 전날 상태 {'보유' if dec['prev_holding'] else '현금'} → 오늘 {'보유' if dec['holding_after'] else '현금'} "
      f"(현재 상태 {dec['state_days']}일째, 마지막 변경 {dec['last_change']})")
k1, k2, k3, k4 = st.columns(4)
k1.metric("QQQ vs 200일선", f"{dec['dist_sma']:+.1%}", f"SMA200 {dec['sma_n']:.2f}")
k2.metric("52주 고점 대비", f"{dec['drawdown']:+.1%}", f"고점 {dec['hi252']:.2f}")
k3.metric("이탈선 (고점 −18%)", f"{dec['exit_level']:.2f}", f"현재가 대비 {dec['dist_exit']:+.1%}")
k4.metric("재진입선 (20일선)", f"{dec['sma_reenter']:.2f}", f"현재가 대비 {dec['dist_reenter']:+.1%}")
cond = pd.DataFrame([
    {"조건": "① 종가 < 200일선", "값": f"{dec['close']:.2f} vs {dec['sma_n']:.2f}", "충족": "✅" if dec["cond_below_sma"] else "❌"},
    {"조건": "② 종가 < 52주 고점 × 0.82", "값": f"{dec['close']:.2f} vs {dec['exit_level']:.2f}", "충족": "✅" if dec["cond_deep_dd"] else "❌"},
    {"조건": "③ 종가 > 20일선", "값": f"{dec['close']:.2f} vs {dec['sma_reenter']:.2f}", "충족": "✅" if dec["cond_above_reenter"] else "❌"},
])
cL, cR = st.columns(2)
with cL:
    st.dataframe(cond, use_container_width=True, hide_index=True)
with cR:
    if dec["holding_after"]:
        st.markdown(f"**매도 조건**: ①과 ② 둘 다 — QQQ 종가가 **{dec['sell_trigger']:.2f}** 아래로 마감하면 다음 거래일에 전량 판다 "
                    f"(현재가 대비 {dec['sell_trigger'] / dec['close'] - 1:+.1%}). 그 전까지는 아무것도 하지 않는다.")
    else:
        st.markdown(f"**재매수 조건**: ③ — QQQ 종가가 **{dec['buy_trigger']:.2f}** 위로 마감하면 다음 거래일에 전량 산다 "
                    f"(현재가 대비 {dec['buy_trigger'] / dec['close'] - 1:+.1%}).")
    st.caption("판정은 종가, 체결은 다음 거래일. 장 마감 후 `cache_fill_cli.py` 로 캐시를 채우면 다음 날 아침 최신 종가로 판정된다.")

recent = sigc.tail(300)
fig0 = go.Figure()
fig0.add_trace(go.Scatter(x=recent.index, y=recent, name="QQQ 종가", line=dict(color="black")))
fig0.add_trace(go.Scatter(x=recent.index, y=sigc.rolling(P.n).mean().tail(300), name="200일선", line=dict(color="#1f77b4")))
fig0.add_trace(go.Scatter(x=recent.index, y=sigc.rolling(P.reenter_n).mean().tail(300), name="20일선", line=dict(color="#2ca02c", dash="dot")))
fig0.add_trace(go.Scatter(x=recent.index, y=(sigc.rolling(252, min_periods=20).max() * (1 - P.dd)).tail(300), name="이탈선 (고점 −18%)", line=dict(color="#d62728", dash="dash")))


def _shade_off(fig, s_, idx):
    off_mask = s_.reindex(idx).ffill() < 0.5
    if off_mask.any():
        blocks = (off_mask != off_mask.shift()).cumsum()
        for _, grp in off_mask[off_mask].groupby(blocks[off_mask]):
            fig.add_vrect(x0=grp.index[0], x1=grp.index[-1], fillcolor="red", opacity=0.08, line_width=0)


_shade_off(fig0, s, recent.index)
fig0.update_layout(height=380, legend=dict(orientation="h", y=1.08), margin=dict(l=40, r=20, t=40, b=30), title="최근 300거래일 — 빨간 음영 = 현금 구간")
st.plotly_chart(fig0, use_container_width=True)


# ── 성적표 · 자산 곡선 · 이력
@st.cache_data(ttl=6 * 3600, show_spinner="세후 성과 계산…")
def _perf(_lev: pd.Series, _sigc: pd.Series, _cash):
    """_sigc = QQQ 종가. run_pair 에는 0/1 스위치 신호를 넘긴다 (가격을 넘기면 항상 보유로 계산되는 버그가 있었음)."""
    sw = signal(_sigc, P)
    rows = []
    for label, start in (("2010-02~ (실제 TQQQ, 폭락 없는 표본)", date(2010, 2, 11)), ("1999-12~ (닷컴·2008 포함, 2010 전 합성)", date(1999, 12, 1))):
        hold, strat = run_pair(_lev, sw, start, None, COSTS, _cash)
        yrs = len(hold.equity) / 252
        hm = hold.final_after_liquidation / hold.equity.iloc[0]; sm = strat.final_after_liquidation / strat.equity.iloc[0]
        rows.append({"표본": label, "보유 배수": hm, "스위치 배수": sm, "보유 대비": sm / hm, "보유 CAGR": hm ** (1 / yrs) - 1, "스위치 CAGR": sm ** (1 / yrs) - 1,
                     "보유 MDD": hold.metrics["mdd"], "스위치 MDD": strat.metrics["mdd"], "매매": strat.trades})
    hold99, strat99 = run_pair(_lev, sw, date(1999, 12, 1), None, COSTS, _cash)
    return pd.DataFrame(rows), hold99.equity, strat99.equity, avoided_segments(strat99, _lev, _sigc)


perf, eq_hold, eq_strat, segs = _perf(lev, sigc, cash)
st.subheader("이 규칙의 세후 성적 (토스 수수료 + 양도세 22%, 다음날 종가 체결)")
st.dataframe(perf.style.format({"보유 배수": "{:.0f}x", "스위치 배수": "{:.0f}x", "보유 대비": "{:.2f}", "보유 CAGR": "{:+.1%}", "스위치 CAGR": "{:+.1%}",
                                "보유 MDD": "{:+.0%}", "스위치 MDD": "{:+.0%}"}), use_container_width=True, hide_index=True)
st.caption("폭락이 없는 2010년대 표본에서는 보유에 약 8% 진다. 폭락이 든 1999~ 표본에서는 보유가 27년째 원금 근처인 동안 100배 이상. "
           "연구 문서(^NDX 신호, 1986~)에서는 10년을 어느 달에 시작했든 보유를 이긴 비율 73~79%. 위 표는 이 페이지의 데이터(QQQ 신호, 1999~)로 다시 계산한 값이라 문서와 조금 다르다.")

fig = go.Figure()
fig.add_trace(go.Scatter(x=eq_hold.index, y=eq_hold / eq_hold.iloc[0], name="TQQQ 보유", line=dict(color="gray")))
fig.add_trace(go.Scatter(x=eq_strat.index, y=eq_strat / eq_strat.iloc[0], name=CRASH_SWITCH_NAME, line=dict(color="#1f77b4", width=2)))
_shade_off(fig, s, eq_strat.index)
fig.update_layout(height=420, yaxis_type="log", yaxis_title="자산 배수 (로그)", title="1999-12 = 1 · 빨간 음영 = 현금 구간", legend=dict(orientation="h", y=1.08),
                  margin=dict(l=40, r=20, t=50, b=30))
st.plotly_chart(fig, use_container_width=True)

st.subheader("매도 → 재매수 이력 (1999~)")
if not segs.empty:
    st.dataframe(segs.sort_values("이탈", ascending=False).style.format({"회피 구간 ETF": "{:+.0%}", "회피 구간 지수": "{:+.0%}"}),
                 use_container_width=True, hide_index=True, height=330)
    st.caption("'회피 구간 ETF' 가 음수면 나가 있는 동안 TQQQ 가 떨어진 것(지킨 돈), 양수면 휩소(놓친 상승). 휩소가 더 많은 게 정상이고, 큰 음수 몇 번이 전체 성적을 만든다.")
