"""TQQQ 스위치·사다리·긴 약세장 패키지 (H43) — 2026-11-11 G1~G7 합격 전략의 운용 화면. 손잡이 없음.

규칙: strategy.adapters.tqqq_package.PACKAGE_RULE (두 단계 이탈 −18% ½·−24% 전량 / 20일선 재진입 / 현금일 때 TQQQ −50/65/80% ⅓ 사다리 /
긴 약세장(QQQ −35%·120일)은 월말 스위치 18 / 다음 거래일 체결).
종전 이 페이지는 2026-10-19 확정 "폭락 스위치 18"(deep_correction_lab.CRASH_SWITCH_RULE) 이었고, 그 규칙은 패키지의 뼈대로 남아 있다(49 페이지 연구용).
쉬운 설명: docs/strategy_notes/TQQQ_탐색_최고결과_쉬운설명_2026-11.md · 기술 원본: TQQQ_탐색_검증_2026-11.md H43.
"""

from datetime import date, datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data.adapters.composed_market_data import build_default_market_data
from strategy.adapters.deep_correction_lab import rolling_study, rolling_summary
from strategy.adapters.leveraged_etf_data import load_lab_prices
from strategy.adapters.tqqq_package import PACKAGE_COSTS, PACKAGE_NAME, PACKAGE_PARAMS, PACKAGE_RULE, change_table, decision, package_frame, run_pair

P = PACKAGE_PARAMS
COSTS = PACKAGE_COSTS  # 토스 수수료 + 슬리피지 0.05% + 양도세 22%(연 통산·250만 공제), 다음날 종가 체결 — 탐색 관문과 동일

st.set_page_config(page_title=PACKAGE_NAME, layout="wide")
_FORCE_KEY = "crash_switch_force_refresh"
tcol, bcol = st.columns([5, 1])
tcol.title(f"{PACKAGE_NAME} — 오늘 무엇을 하나")
if bcol.button("🔄 강제 리프레시", use_container_width=True,
               help="페이지 캐시(6시간)와 성적표 캐시를 비우고, 오늘 날짜가 포함된 구간은 Mongo 캐시를 건너뛰어 소스에서 다시 받는다."):
    st.session_state[_FORCE_KEY] = True
    st.rerun()
st.caption(f"**규칙 고정**: {PACKAGE_RULE}")


@st.cache_data(ttl=6 * 3600, show_spinner="가격 로딩…")
def _prices(_bypass_today: bool):
    """_bypass_today=True 면 end >= 오늘인 요청은 Mongo 일 캐시를 건너뛰고 상위 소스에서 바로 받는다(강제 리프레시 전용).
    앞 밑줄 = 캐시 키에서 제외. 강제 리프레시는 먼저 _prices.clear() 를 하고 True 로 한 번 채우며, 이후 False 호출도 같은 키를 쓴다."""
    pr = load_lab_prices(build_default_market_data(bypass_today=_bypass_today), "TQQQ", "QQQ", date(1999, 12, 1), date.today(), synthetic=True)
    return pr.lev_close, pr.signal_close, pr.cash_rate, pr.source_note, datetime.now()


forced = bool(st.session_state.pop(_FORCE_KEY, False))
if forced:
    _prices.clear()
try:
    lev, sigc, cash, note, loaded_at = _prices(forced)
except Exception as e:  # noqa: BLE001
    st.error(f"데이터 로딩 실패: {e}")
    st.stop()
if forced:
    st.toast("소스에서 다시 받았습니다.", icon="🔄")
st.caption(f"데이터: {note} · 마지막 종가 {sigc.index[-1].date()} · 로딩 {loaded_at:%Y-%m-%d %H:%M} "
           f"({'강제 리프레시' if forced else '캐시 6시간'})")
frame = package_frame(sigc, lev, P)

# ── 오늘의 행동
st.subheader(f"오늘의 행동 — {sigc.index[-1].date()} 종가 기준")
w1, w2 = st.columns(2)
what_if = w1.number_input("QQQ 오늘 종가 가정 (0 = 마지막 종가 그대로)", min_value=0.0, value=0.0, step=1.0,
                          help="장중에 보려면 지금 가격을 넣는다. 규칙은 종가로 판정하므로 장 마감 전 값은 참고용.")
what_if_t = w2.number_input("TQQQ 오늘 종가 가정 (0 = QQQ 변화 × 3 으로 근사)", min_value=0.0, value=0.0, step=1.0,
                            help="사다리 판정은 TQQQ 자기 가격을 본다. 비우면 전날 TQQQ × (1 + 3 × QQQ 변화) 로 근사.")
dec = decision(sigc, lev, P, hypothetical_close=float(what_if) if what_if > 0 else None, hypothetical_tqqq=float(what_if_t) if what_if_t > 0 else None)
act = dec["action"]
color = st.error if act.startswith("매도") else st.success if act.startswith("매수") or act == "보유" else st.warning
basis = f"가정 종가 QQQ {dec['close']:.2f} · TQQQ {dec['tqqq']:.2f}" if dec["hypothetical"] else f"{dec['as_of']} 종가 QQQ {dec['close']:.2f} · TQQQ {dec['tqqq']:.2f}"
color(f"**{act}** — {basis} 기준 · 목표 비중 TQQQ **{dec['target']:.0%}** (전날 {dec['prev_target']:.0%}) · "
      f"{'이유: ' + dec['reason'] + ' · ' if dec['reason'] else ''}현재 상태 {dec['state_days']}일째, 마지막 변경 {dec['last_change']}")
regime_ko = {"A": "평소(상승)", "B": "조정(200일선 아래 또는 고점 −10% 이상)", "C": "긴 약세장 — 월말에만 판정", "D": "약세장 회복 구간"}
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("QQQ vs 200일선", f"{dec['close'] / dec['sma200'] - 1:+.1%}", f"SMA200 {dec['sma200']:.2f}")
k2.metric("QQQ 52주 고점 대비", f"{dec['qqq_dd']:+.1%}", f"고점 {dec['hi252']:.2f}")
k3.metric("QQQ vs 20일선(재진입선)", f"{dec['close'] / dec['sma20'] - 1:+.1%}", f"SMA20 {dec['sma20']:.2f}")
k4.metric("TQQQ 52주 고점 대비", f"{dec['tqqq_dd']:+.1%}", f"고점 {dec['tqqq_hi']:.2f}")
k5.metric("레짐 · 사다리", f"{dec['regime']}", f"칸 소진 {dec['rung_used']}/3 · 이탈 단계 {dec['step_i']}/2")
st.caption(f"레짐 {dec['regime']} = {regime_ko[dec['regime']]}" + (f" · 이 달 마지막 거래일 {dec['month_end']}" if dec["regime"] == "C" else ""))

cond = pd.DataFrame([
    {"조건": "① QQQ 종가 < 200일선", "값": f"{dec['close']:.2f} vs {dec['sma200']:.2f}", "충족": "✅" if dec["cond_below_sma"] else "❌"},
    {"조건": "② QQQ < 52주 고점 −18% (①과 함께 → 절반 매도)", "값": f"{dec['close']:.2f} vs {dec['half_level']:.2f}", "충족": "✅" if dec["cond_dd1"] else "❌"},
    {"조건": "③ QQQ < 52주 고점 −24% (①과 함께 → 전량 매도)", "값": f"{dec['close']:.2f} vs {dec['full_level']:.2f}", "충족": "✅" if dec["cond_dd2"] else "❌"},
    {"조건": "④ QQQ > 20일선 (현금·부분 상태면 → 전량 재매수)", "값": f"{dec['close']:.2f} vs {dec['sma20']:.2f}", "충족": "✅" if dec["cond_above_sma20"] else "❌"},
    {"조건": "⑤ QQQ < 52주 고점 −35% (약세 상태면 → 긴 약세장 모드)", "값": f"{dec['close']:.2f} vs {dec['depth_level']:.2f}", "충족": "✅" if dec["cond_depth"] else "❌"},
] + [{"조건": f"사다리 {i + 1}칸 TQQQ ≤ 52주 고점 −{r:.0%} (현금·부분 상태면 → ⅓ 매수)", "값": f"{dec['tqqq']:.2f} vs {lv:.2f}",
      "충족": "소진" if i < dec["rung_used"] else ("✅" if dec["tqqq"] <= lv else "❌")} for i, (r, lv) in enumerate(zip(P.rungs, dec["ladder_levels"]))])
cL, cR = st.columns(2)
with cL:
    st.dataframe(cond, use_container_width=True, hide_index=True)
with cR:
    st.markdown("**다음에 무엇이 일어나면 움직이나**")
    for t_ in dec["triggers"]:
        st.markdown(f"- {t_}")
    st.caption("판정은 종가, 체결은 다음 거래일. 장 마감 후 `cache_fill_cli.py` 로 캐시를 채우면 다음 날 아침 최신 종가로 판정된다.")

recent = frame.tail(300)
fig0 = go.Figure()
fig0.add_trace(go.Scatter(x=recent.index, y=recent["QQQ"], name="QQQ 종가", line=dict(color="black")))
fig0.add_trace(go.Scatter(x=recent.index, y=recent["sma200"], name="200일선", line=dict(color="#1f77b4")))
fig0.add_trace(go.Scatter(x=recent.index, y=recent["sma20"], name="20일선", line=dict(color="#2ca02c", dash="dot")))
fig0.add_trace(go.Scatter(x=recent.index, y=recent["hi252"] * (1 - P.dd1), name="절반 매도선 (고점 −18%)", line=dict(color="#ff7f0e", dash="dash")))
fig0.add_trace(go.Scatter(x=recent.index, y=recent["hi252"] * (1 - P.dd2), name="전량 매도선 (고점 −24%)", line=dict(color="#d62728", dash="dash")))
fig0.add_trace(go.Scatter(x=recent.index, y=recent["hi252"] * (1 - P.c_depth), name="긴 약세장선 (고점 −35%)", line=dict(color="#9467bd", dash="dot")))
fig0.add_trace(go.Scatter(x=recent.index, y=recent["target"], name="TQQQ 목표 비중 (우축)", yaxis="y2", line=dict(color="#d62728", width=1, shape="hv"), fill="tozeroy", fillcolor="rgba(214,39,40,0.12)"))


def _shade(fig, f_, col="target", lo=1.0, color="red", opacity=0.08):
    m = f_[col] < lo
    if m.any():
        blocks = (m != m.shift()).cumsum()
        for _, grp in m[m].groupby(blocks[m]):
            fig.add_vrect(x0=grp.index[0], x1=grp.index[-1], fillcolor=color, opacity=opacity, line_width=0)


def _shade_regime_c(fig, f_):
    m = f_["regime"] == "C"
    if m.any():
        blocks = (m != m.shift()).cumsum()
        for _, grp in m[m].groupby(blocks[m]):
            fig.add_vrect(x0=grp.index[0], x1=grp.index[-1], fillcolor="#9467bd", opacity=0.12, line_width=0)


_shade_regime_c(fig0, recent)
fig0.update_layout(height=420, legend=dict(orientation="h", y=1.1), margin=dict(l=40, r=20, t=40, b=30), title="최근 300거래일 — 빨간 면 = TQQQ 목표 비중, 보라 음영 = 긴 약세장 모드",
                   yaxis2=dict(overlaying="y", side="right", range=[0, 1.05], showgrid=False, tickformat=".0%"))
st.plotly_chart(fig0, use_container_width=True)

t_recent = frame.tail(300)
fig_t = go.Figure()
fig_t.add_trace(go.Scatter(x=t_recent.index, y=t_recent["tqqq_dd"], name="TQQQ 52주 고점 대비 낙폭", line=dict(color="#2ca02c")))
for r_ in P.rungs:
    fig_t.add_hline(y=-r_, line=dict(color="#2ca02c", dash="dash", width=1), annotation_text=f"사다리 −{r_:.0%}", annotation_position="right")
fig_t.update_layout(height=260, margin=dict(l=40, r=20, t=30, b=30), title="TQQQ 낙폭과 사다리 선 — 현금·부분 상태에서 선에 닿으면 ⅓ 씩 (칸마다 에피소드당 한 번)", yaxis_tickformat=".0%")
st.plotly_chart(fig_t, use_container_width=True)


# ── 성적표 · 자산 곡선 · 이력
@st.cache_data(ttl=6 * 3600, show_spinner="세후 성과 계산…")
def _perf(_lev: pd.Series, _target: pd.Series, _cash, data_key: str):
    """밑줄 인자는 캐시 키에서 빠지므로 data_key(마지막 날짜·길이)로 가격이 바뀌면 다시 계산되게 한다."""
    rows = []
    for label, start in (("2010-02~ (실제 TQQQ)", date(2010, 2, 11)), ("1999-12~ (닷컴·2008 포함, 2010 전 합성)", date(1999, 12, 1))):
        hold, strat = run_pair(_lev, _target, start, None, COSTS, _cash)
        yrs = len(hold.equity) / 252
        hm = hold.final_after_liquidation / hold.equity.iloc[0]; sm = strat.final_after_liquidation / strat.equity.iloc[0]
        rows.append({"표본": label, "보유 배수": hm, "패키지 배수": sm, "보유 대비": sm / hm, "보유 CAGR": hm ** (1 / yrs) - 1, "패키지 CAGR": sm ** (1 / yrs) - 1,
                     "보유 MDD": hold.metrics["mdd"], "패키지 MDD": strat.metrics["mdd"], "매매": strat.trades})
    hold99, strat99 = run_pair(_lev, _target, date(1999, 12, 1), None, COSTS, _cash)
    roll = rolling_study(_lev, _target, 10, COSTS, _cash, first_start=date(2010, 2, 11))
    return pd.DataFrame(rows), hold99.equity, strat99.equity, roll, rolling_summary(roll)


perf, eq_hold, eq_strat, roll, rsum = _perf(lev, frame["target"], cash, f"{sigc.index[-1].date()}_{len(sigc)}_{len(lev)}")
st.subheader("이 규칙의 세후 성적 (토스 수수료 + 슬리피지 + 양도세 22%, 다음날 종가 체결)")
st.dataframe(perf.style.format({"보유 배수": "{:.0f}x", "패키지 배수": "{:.0f}x", "보유 대비": "{:.2f}", "보유 CAGR": "{:+.1%}", "패키지 CAGR": "{:+.1%}",
                                "보유 MDD": "{:+.0%}", "패키지 MDD": "{:+.0%}"}), use_container_width=True, hide_index=True)
if rsum:
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("10년 창 수 (2010-02~, 분기마다)", rsum["windows"])
    r2.metric("TQQQ 보유를 이긴 창", f"{rsum['win_rate']:.0%}")
    r3.metric("보유 대비 배수 중앙", f"{rsum.get('ratio_median', float('nan')):.2f}")
    r4.metric("최악 창 비율", f"{roll['ratio'].min():.2f}")
st.caption("탐색 검증(봉인 해제 판, 2026-10-24 까지): 10년 창 28개 중 27개 승·중앙 1.86·최악 0.93, MDD −69%(TQQQ −82%), 닷컴·2008 포함 합성 10년 창 114개 손실 0. "
           "위 표는 이 페이지의 데이터(QQQ 신호, 1999~ 합성 TQQQ)로 오늘까지 다시 계산한 값이라 문서와 조금 다르다. 규칙이 움직인 에피소드는 2010년 이후 7개뿐 — 큰 사건 몇 개에 기대는 규칙이다.")

fig = go.Figure()
fig.add_trace(go.Scatter(x=eq_hold.index, y=eq_hold / eq_hold.iloc[0], name="TQQQ 보유", line=dict(color="gray")))
fig.add_trace(go.Scatter(x=eq_strat.index, y=eq_strat / eq_strat.iloc[0], name=PACKAGE_NAME, line=dict(color="#1f77b4", width=2)))
_shade(fig, frame.reindex(eq_strat.index).ffill())
_shade_regime_c(fig, frame.reindex(eq_strat.index).ffill())
fig.update_layout(height=420, yaxis_type="log", yaxis_title="자산 배수 (로그)", title="1999-12 = 1 · 빨간 음영 = TQQQ 비중 100% 아래 구간 · 보라 = 긴 약세장 모드", legend=dict(orientation="h", y=1.08),
                  margin=dict(l=40, r=20, t=50, b=30))
st.plotly_chart(fig, use_container_width=True)

with st.expander("시작 시점별 10년 창 (2010-02~ 분기마다)"):
    if not roll.empty:
        st.dataframe(roll.style.format({"hold_mult": "{:.0f}x", "strat_mult": "{:.0f}x", "ratio": "{:.2f}", "hold_cagr": "{:+.1%}", "strat_cagr": "{:+.1%}", "hold_mdd": "{:+.0%}", "strat_mdd": "{:+.0%}"}),
                     use_container_width=True, hide_index=True, height=300)

st.subheader("비중 변경 이력 (1999~, 최신 위)")
hist = change_table(frame)
if not hist.empty:
    st.dataframe(hist.style.format({"이전 비중": "{:.0%}", "이후 비중": "{:.0%}", "QQQ 고점 대비": "{:+.1%}", "QQQ vs 200일선": "{:+.1%}", "TQQQ 고점 대비": "{:+.1%}", "다음 변경까지 TQQQ": "{:+.1%}"}),
                 use_container_width=True, hide_index=True, height=420)
    st.caption("'다음 변경까지 TQQQ' 는 그 비중으로 지낸 구간의 TQQQ 움직임 — 매도 뒤 음수면 지킨 돈, 양수면 휩소(놓친 상승). 사다리 매수 뒤 양수면 바닥 근처에서 산 것.")
