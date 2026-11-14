"""근거중심매매(온휘) 체크리스트 — 과거 시점 블라인드 판정 도구.

종목·기준일을 넣으면 기준일까지의 데이터만으로 (1) 자동 항목(이동평균선·RSI·캔들·FVG·매물대)을 기계가
채우고 (2) 반자동 항목(채널·피보나치)은 후보를 차트에 그려 사람이 확정하며 (3) 수동 항목은 체크박스로 받아
5대 매매 근거 체크리스트 점수와 근거가중치 합으로 베팅/단기/보류/매매 안 함을 판정한다. 평가를 기록한 뒤에야
"봉인 해제"로 기준일 이후 성과를 본다. 설계서: docs/strategy_notes/근거중심매매_검증_체크리스트_2026-10.md
(사본: pages/_shared/evidence_manual.md). 엔진: strategy/adapters/evidence_checklist.py
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from visualization.adapters.plotly_bridge import st_candles
from plotly.subplots import make_subplots

from data.adapters.composed_market_data import build_default_market_data
from strategy.adapters.evidence_checklist import (
    Check, EvidenceParams, TimeframeReport, evaluate, forward_result, fvg_text, verdict, weighted_score,
)

st.set_page_config(page_title="근거중심매매 체크리스트", layout="wide")
st.title("근거중심매매 체크리스트 — 과거 시점 블라인드 판정")
st.caption("온휘 『근거중심매매』의 5대 매매 근거(채널·주봉 이동평균선·캔들·다이버전스·피보나치) 중 **4개 이상 예**면 베팅. "
           "기준일 이후 데이터는 평가가 끝날 때까지 보이지 않는다. 채널·피보나치는 스윙 피벗으로 그린 **근사치**라 사람이 확정한다.")

# ---------------------------------------------------------------------------
# 입력
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("종목·기준일")
    _pre_t = st.session_state.pop("ev_prefill_ticker", None)
    _pre_d = st.session_state.pop("ev_prefill_date", None)
    if _pre_t:
        st.session_state["ev_ticker_input"] = _pre_t
    if _pre_d:
        st.session_state["ev_asof_input"] = _pre_d
    ticker = st.text_input("티커 (yfinance)", value="AAPL", key="ev_ticker_input",
                           help="예: AAPL · NVDA · QQQ · ^NDX · GC=F 금 · CL=F 원유 · BTC-USD · 국내는 005930.KS 처럼 접미사").strip()
    as_of = st.date_input("기준일 (이날 종가까지만 본다)", value=date.today() - timedelta(days=1), max_value=date.today(), key="ev_asof_input")
    use_4h = st.checkbox("4시간봉도 계산 (최근 2년만, 느림)", value=False)
    st.header("작도 파라미터 ([가정])")
    pivot_lr = st.slider("스윙 피벗 좌우 봉 수 (일봉)", 3, 30, 10)
    ch_tol = st.slider("채널선 접촉 허용 ±%", 1.0, 5.0, 3.0, 0.5)
    lvl_tol = st.slider("피보나치 레벨 허용 ±%", 0.5, 3.0, 1.5, 0.5)
    if st.button("체크 초기화", use_container_width=True):
        for k in list(st.session_state.keys()):
            if k.startswith(("ev_chk_", "ev_fix_")):
                del st.session_state[k]
        st.rerun()

P = EvidenceParams(pivot_lr=pivot_lr, channel_tol=ch_tol / 100, level_tol=lvl_tol / 100)


@st.cache_data(ttl=6 * 3600, show_spinner="일봉 로딩…")
def _daily(sym: str, start: date) -> pd.DataFrame:
    return build_default_market_data().fetch_ohlcv(sym, start, date.today())


@st.cache_data(ttl=6 * 3600, show_spinner="시간봉 로딩…")
def _hourly(sym: str) -> pd.DataFrame:
    return build_default_market_data().fetch_ohlcv(sym, date.today() - timedelta(days=720), date.today(), interval="1h")


if not ticker:
    st.stop()
try:
    daily_all = _daily(ticker, as_of - timedelta(days=365 * 10))
    hourly = _hourly(ticker) if use_4h and (date.today() - as_of).days < 700 else None
    reports = evaluate(daily_all, as_of, hourly, P)
except Exception as e:  # noqa: BLE001
    st.error(f"데이터 로딩/계산 실패: {e}")
    st.stop()

weekly, daily = reports["주봉"], reports["일봉"]
h4 = reports.get("4시간봉")


# ---------------------------------------------------------------------------
# 차트
# ---------------------------------------------------------------------------
def _chart(rep: TimeframeReport, bars: int) -> go.Figure:
    df = rep.df.tail(bars)
    off = len(rep.df) - len(df)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
                                 name="가격", increasing_line_color="#d62728", decreasing_line_color="#1f77b4"), 1, 1)
    for n, col in ((20, "red"), (50, "orange"), (100, "skyblue"), (200, "purple")):
        fig.add_trace(go.Scatter(x=df.index, y=df[f"EMA{n}"], name=f"{n}MA", line=dict(color=col, width=1)), 1, 1)
    if f"SMA{P.sma_long}" in df:
        fig.add_trace(go.Scatter(x=df.index, y=df[f"SMA{P.sma_long}"], name="365MA", line=dict(color="white", width=1.2)), 1, 1)
    for ch, col in zip(rep.channel.channels, ("#ffdd57", "#7fffd4")):
        ser = ch.series(len(rep.df)).iloc[off:]
        for name, dash in (("upper", "solid"), ("mid", "dot"), ("lower", "solid")):
            fig.add_trace(go.Scatter(x=df.index, y=ser[name].to_numpy(), name=f"{ch.kind} {name}",
                                     line=dict(color=col, width=1, dash=dash), showlegend=(name == "mid")), 1, 1)
    if rep.fib:
        for r, v in rep.fib.levels.items():
            lab = r if isinstance(r, float) else r.replace("ext", "확장 ")
            fig.add_hline(y=v, line=dict(color="#aaaaaa", width=0.6, dash="dash"), row=1, col=1,
                          annotation_text=f"{lab}", annotation_position="right", annotation_font_size=9)
    for b in rep.boxes:
        if b.end >= off:
            x0 = rep.df.index[max(b.start, off)]
            fig.add_shape(type="rect", x0=x0, x1=df.index[-1], y0=b.bottom, y1=b.top, row=1, col=1,
                          fillcolor="green", opacity=0.08, line_width=0)
    for f in rep.fvgs:
        if f.fill < 1.0 and f.pos >= off:
            fig.add_shape(type="rect", x0=rep.df.index[f.pos], x1=df.index[-1], y0=f.bottom, y1=f.top, row=1, col=1,
                          fillcolor="orange" if f.bullish else "gray", opacity=0.15, line_width=0)
    sw = rep.swings[rep.swings.pos >= off]
    for kind, sym, col in (("H", "triangle-down", "#ff7f0e"), ("L", "triangle-up", "#2ca02c")):
        s = sw[sw.kind == kind]
        if not s.empty:
            fig.add_trace(go.Scatter(x=rep.df.index[s.pos.to_numpy()], y=s.price, mode="markers", name=f"피벗 {kind}",
                                     marker=dict(symbol=sym, color=col, size=9)), 1, 1)
    fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI", line=dict(color="#9467bd")), 2, 1)
    fig.add_hline(y=70, line=dict(color="gray", dash="dot", width=0.7), row=2, col=1)
    fig.add_hline(y=30, line=dict(color="gray", dash="dot", width=0.7), row=2, col=1)
    fig.update_layout(height=620, template="plotly_dark", xaxis_rangeslider_visible=False, yaxis_type="log",
                      margin=dict(l=40, r=70, t=30, b=20), legend=dict(orientation="h", y=1.04, font=dict(size=10)),
                      title=f"{ticker} {rep.label} — {rep.df.index[-1].date()} 까지 (로그)")
    return fig


def _auto_panel(rep: TimeframeReport):
    rows = [("이동평균선", rep.ma.text), ("채널(근사)", rep.channel.text), ("피보나치(근사)", rep.fib.text if rep.fib else "스윙 부족"),
            ("매물대", rep.supply.why), ("FVG", fvg_text(rep.fvgs)),
            ("캔들", rep.candle.text), ("RSI·다이버전스", rep.div.text)]
    st.dataframe(pd.DataFrame(rows, columns=["도구", "자동 판독"]), use_container_width=True, hide_index=True)
    five = pd.DataFrame([{"항목": k, "자동": "✅ 예" if c.ok else ("❌ 아니오" if c.ok is False else "— 판정 불가"), "근거": c.why}
                         for k, c in rep.five.items()])
    st.dataframe(five, use_container_width=True, hide_index=True)
    st.caption(f"자동 '예' {rep.yes_count}/5 — {'주봉 4 이상 = 베팅' if rep.label == '주봉' else '4시간봉·일봉 4 이상 = 단기만'}")


tabs = st.tabs([f"주봉 (예 {weekly.yes_count}/5)", f"일봉 (예 {daily.yes_count}/5)"] + ([f"4시간봉 (예 {h4.yes_count}/5)"] if h4 else []))
with tabs[0]:
    st_candles(_chart(weekly, 200), use_container_width=True)
    _auto_panel(weekly)
with tabs[1]:
    st_candles(_chart(daily, 260), use_container_width=True)
    _auto_panel(daily)
if h4:
    with tabs[2]:
        st_candles(_chart(h4, 300), use_container_width=True)
        _auto_panel(h4)


# ---------------------------------------------------------------------------
# 사람이 확정하는 5대 체크리스트 (주봉 기준) + 보조 근거
# ---------------------------------------------------------------------------
st.subheader("5대 매매 근거 확정 — 주봉 기준 (자동 판독을 보고 사람이 결정)")
st.caption("①·⑤는 근사 작도라 반드시 눈으로 확인. 매도 계획은 ②를 뺀 4요소.")
FIX = ("자동 따름", "예", "아니오")
final: dict[str, bool] = {}
cols = st.columns(5)
for i, (name, c) in enumerate(weekly.five.items()):
    with cols[i]:
        choice = st.radio(name, FIX, key=f"ev_fix_{i}", horizontal=False,
                          help=c.why)
        final[name] = bool(c.ok) if choice == "자동 따름" else (choice == "예")
        st.caption(("자동: ✅" if c.ok else ("자동: ❌" if c.ok is False else "자동: —")) + f" · {c.why[:70]}")

bc = st.columns(4)
sup_choice = bc[0].radio("매물대 겹침 (0.75)", FIX, key="ev_fix_sup", help=weekly.supply.why + " / 일봉: " + daily.supply.why)
sup_ok = (bool(weekly.supply.ok or daily.supply.ok)) if sup_choice == "자동 따름" else (sup_choice == "예")
pattern_pts = bc[1].selectbox("차트패턴 (같은 방향만)", [0.0, 0.5, 0.75], format_func=lambda v: {0.0: "없음", 0.5: "일반 패턴 0.5", 0.75: "어센딩·디센딩 0.75"}[v], key="ev_fix_pat")
wave_ok = bc[2].checkbox("파동 근거 (0.5)", key="ev_fix_wave", help="채널→피보나치→절대규칙 검증을 거친 경우만")
vol_ok = bc[3].checkbox("캔들에 거래량 동반 (0.5)", value=bool(weekly.candle.vol_ratio and weekly.candle.vol_ratio >= 1.0), key="ev_fix_vol")

yes_w = sum(final.values())
score, parts = weighted_score({k: Check(final[k], "") for k in weekly.five}, sup_ok, vol_ok, pattern_pts, wave_ok)
lower_yes = max(daily.yes_count, h4.yes_count if h4 else 0)
vd = verdict(yes_w, lower_yes, score)
m = st.columns(4)
m[0].metric("주봉 예 개수", f"{yes_w}/5")
m[1].metric("근거가중치 합", f"{score:.2f}", "≥ 4 진입")
m[2].metric("하위 시간대 자동 예", f"{lower_yes}/5")
m[3].metric("판정", vd)
st.caption("가중치 구성: " + (", ".join(parts) if parts else "없음") + f" · 4분면(주봉): {weekly.channel.quadrant} · 일봉: {daily.channel.quadrant}")


# ---------------------------------------------------------------------------
# 수동 체크리스트 (설계서 §5 요약)
# ---------------------------------------------------------------------------
MANUAL: dict[str, tuple[str, ...]] = {
    "M-A 채널·추세선": (
        "로그차트로 봤다", "6개월봉·월봉 메인 채널 → 주봉 → 일봉 순으로 그렸다 (짧게 그리지 않았다)",
        "기준점이 변곡점(상장가 / 2번째 하락 다이버전스 고점 / 주봉 과매도+역배열 저점)이다", "IMF·리먼·코로나 구간을 기준점에서 뺐다",
        "고고저·저저고 둘 다 그려 유효 구간을 표시했다", "확장·중단의 중단으로 튀어나온 구간이 설명된다", "몸통 기준으로 그었고 ±2~5% 이탈은 접촉으로 봤다",
        "월봉·주봉 추세선을 함께 그었다", "4분면(상승+역배열 최선 / 하락+정배열 최악)을 적었다", "채널·200MA 리테스트가 없으면 관망으로 적었다"),
    "M-B 피보나치": (
        "기준점을 왜 거기로 잡았는지 적었다 (첫 돌파 / 스윕 / 하락장악형 / 365MA / 상장가)", "이동평균선에만 매몰되지 않았다",
        "방향이 맞다 (상승 중 매도=정방향, 하락 중 매수=리버스)", "지금 어느 구간인지 먼저 말했다 (2.618 위인가)",
        "기술적 반등(0.618) 여부에 따라 다음 행동을 정했다", "주봉 레벨과 하위 시간대 레벨이 겹치는지 봤다", "추세기반 확장 1:1 구간을 표시했다"),
    "M-C 매물대": (
        "몸통 기준 박스다", "전고점·상장가 일치 / 꼬리 존재 / 저항→지지 전환 중 무엇이 확인되는지 적었다",
        "소화 기간(형성 봉 수 : 소화 봉 수 ≈ 1:1)을 쟀다", "전고점 위 매물대를 하방 한계선으로 표시했다", "거래량 폭발로 뚫린 매물대 안에서 사지 않았다"),
    "M-D 캔들": (
        "캔들 전에 위치(채널·MA)를 먼저 적었다", "4시간봉 이상이고 최근 10개만 봤다", "꼬리 인정(몸통은 선 밖)·비율(1:2~3)·거래량을 적었다",
        "반전 캔들이 20MA 위 마감이다 / 과매수에서 모닝스타를 주장하지 않았다", "중단 도지면 포지션을 줄이고 다음 방향을 확인했다",
        "무효화 조건(백삼병 저점·흑삼병 고점·다람쥐 절반)을 추적했다", "이브닝스타·하락장악형은 마무리 음봉이 20MA 를 깼는지 봤다",
        "캔들과 패턴이 충돌하면 근거가 더 겹치는 쪽, 반반이면 매매 안 함"),
    "M-E 차트패턴": (
        "패턴을 배제 도구로 썼다 (불플래그 → 매도 계획)", "삼각수렴 3체크(거래량 감소·MA 수렴·추세선 준수)", "위치·배열로 확률을 조정했다",
        "돌파를 리테스트로 확인했다 (꼬리 이탈 = 트랩 가능)", "플래그는 깃발에 채널, 깃대 1:1 목표에 청산 주문", "베어플래그 3단계 컨펌(4h 역배열·모양·200MA 위 지지 없음)",
        "헤드앤숄더는 머리 거래량·넥라인 1:1, 주 근거는 MA 지지"),
    "M-F 파동": (
        "복합조정 → ABC → 충격파동 순서로 대입했다", "채널 준수 여부로 WXY / ABC X ABC / 3파 연장을 좁혔다", "피보나치 확장을 먼저 그었다 (하나만 확장, Y ≤ 1.618)",
        "절대규칙 3개를 확인했다", "1플랜·2플랜을 세웠고 방향이 엇갈리면 매매 안 함", "B파에서 매매하지 않는다", "손절을 절대규칙선에 두고 끌어올린다"),
    "M-G 종합": (
        "종목 전에 지수(코스피)를 먼저 봤다", "주봉 근거부터 적고 일봉·4시간봉을 더했다 (충돌하면 주봉)", "매도 계획은 MA 를 뺀 4요소로 따로 적었다",
        "근거 부족 시 패턴으로 보강했고, 합이 충분해도 좋은 캔들이 없으면 비중 축소"),
    "M-H 돈 관리": (
        "매수는 두 번 (1차 0.382 가볍게, 2차 근거 더 쌓인 자리)", "동시 보유 3~5개 미만", "추격매수 아님 (EMA20 과 괴리 작음, 다음 근거 자리 기다림)",
        "DCA 는 하락장에서만", "익절 계획: 1:1 에서 75% → 본전 스탑, 2.618+채널 상단", "손절선 = 근거선 몸통 이탈 (꼬리 X)", "애매하면 베팅 안 함", "매매일지에 근거·점수·결과 기록"),
}
st.subheader("수동 체크리스트 (설계서 §5)")
manual_counts: dict[str, str] = {}
for sec, items in MANUAL.items():
    with st.expander(sec, expanded=False):
        n_ok = 0
        for j, q in enumerate(items):
            if st.checkbox(q, key=f"ev_chk_{sec}_{j}"):
                n_ok += 1
        manual_counts[sec] = f"{n_ok}/{len(items)}"
st.caption("수동 완료: " + " · ".join(f"{k.split(' ')[0]} {v}" for k, v in manual_counts.items()))


# ---------------------------------------------------------------------------
# 기록 · 봉인 해제
# ---------------------------------------------------------------------------
st.subheader("기록 → 봉인 해제")
memo = st.text_area("근거 한 줄 (왜 이 판정인가)", height=70)
if "evidence_log" not in st.session_state:
    st.session_state["evidence_log"] = []
c1, c2 = st.columns([1, 1])
if c1.button("이 평가를 기록에 추가", type="primary"):
    st.session_state["evidence_log"].append({
        "티커": ticker, "기준일": as_of.isoformat(), "판정": vd, "주봉 예": yes_w, "가중치": round(score, 2),
        "하위 예": lower_yes, **{k: ("예" if v else "아니오") for k, v in final.items()},
        "매물대": sup_ok, "패턴": pattern_pts, "파동": wave_ok, **manual_counts, "메모": memo,
        "4주": None, "13주": None, "26주": None, "26주 MDD": None,
    })
    st.toast(f"{ticker} @ {as_of} 기록")
reveal = c2.button("봉인 해제 — 기준일 이후 성과 보기 (평가 기록 후에만)")
if reveal:
    fr = forward_result(daily_all, as_of)
    if not fr:
        st.warning("기준일 이후 데이터 없음")
    else:
        fc = st.columns(6)
        for i, (k, v) in enumerate(fr.items()):
            fc[i % 6].metric(k, "—" if v is None else f"{v:+.1%}")
        if st.session_state["evidence_log"]:
            last = st.session_state["evidence_log"][-1]
            if last["티커"] == ticker and last["기준일"] == as_of.isoformat():
                last.update({"4주": fr.get("4주 수익"), "13주": fr.get("13주 수익"), "26주": fr.get("26주 수익"), "26주 MDD": fr.get("26주 MDD")})
        fut = daily_all.copy()
        fut.index = pd.to_datetime(fut.index).tz_localize(None) if getattr(pd.to_datetime(fut.index), "tz", None) is not None else pd.to_datetime(fut.index)
        seg = fut[(fut.index.date >= as_of - timedelta(days=180)) & (fut.index.date <= as_of + timedelta(days=200))]
        fig = go.Figure(go.Scatter(x=seg.index, y=seg["Close"], name="종가"))
        fig.add_vline(x=pd.Timestamp(as_of), line=dict(color="red", dash="dash"))
        fig.update_layout(height=300, margin=dict(l=40, r=20, t=20, b=20), title="기준일(빨간 선) 전후 종가")
        st_candles(fig, use_container_width=True)

log = st.session_state["evidence_log"]
if log:
    ldf = pd.DataFrame(log)
    st.dataframe(ldf, use_container_width=True, hide_index=True)
    bet = ldf[ldf["판정"] == "베팅"]["26주"].dropna()
    no = ldf[ldf["판정"] == "매매 안 함"]["26주"].dropna()
    if len(bet) and len(no):
        st.caption(f"베팅 {len(bet)}건 평균 26주 {bet.mean():+.1%} vs 매매 안 함 {len(no)}건 {no.mean():+.1%} — 베팅이 뚜렷이 높아야 체크리스트가 작동하는 것.")
    st.download_button("기록 CSV", ldf.to_csv(index=False).encode("utf-8-sig"), file_name="evidence_checklist_log.csv", mime="text/csv")
else:
    st.info("기록 없음. ⚠️ 세션 메모리 저장 — 새로고침 전에 CSV 로 내려받을 것.")

_doc = Path(__file__).resolve().parent / "_shared" / "evidence_manual.md"
with st.expander("📖 설계서 원문 (용어·규칙·자동/수동 항목 전체)"):
    st.markdown(_doc.read_text(encoding="utf-8") if _doc.exists() else f"문서를 찾을 수 없음: {_doc}")
