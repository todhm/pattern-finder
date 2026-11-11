"""VWAP 지지·저항 돌파 (15분봉) — 종목·기간을 넣으면 강의 규칙대로 진입·손절선·익절을 차트에 그리고 손익을 낸다.

규칙 출처: docs/strategy_notes/images/vwapcourse/vwapcourse.pdf (Udemy VWAP 강의 메모). 설계서:
docs/strategy_notes/VWAP_지지저항_돌파_15분봉_설계_2026-11.md. 엔진: strategy/adapters/vwap_sr_breakout.py (순수 pandas).
"""

from __future__ import annotations

from dataclasses import fields
from datetime import date, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from data.adapters.alphavantage_adapter import AlphaVantageAdapter
from data.adapters.composed_market_data import build_default_market_data
from data.adapters.regular_session_filter import RegularSessionFilterAdapter
from strategy.adapters.vwap_sr_breakout import VwapSrParams, clean_wicks, current_setup, missed_breakouts, run_vwap_sr, structure_segments

st.set_page_config(page_title="VWAP 지지저항 돌파 15m", layout="wide")
st.title("VWAP 지지·저항 돌파 — 15분봉 손절선·익절 시뮬레이션")
st.caption("강의 규칙: 세션 VWAP 근처에서 **좁은 박스**(지지·저항)를 만들고 저항을 여러 번 두드린 뒤 **종가로 돌파**하면 진입, "
           "손절은 VWAP(또는 박스) 아래 **종가 이탈**, 익절은 VWAP 에서 멀어진 뒤 **거래량 스파이크·첫 역캔들·긴 윅·큰 봉**이 나온 봉 마감. "
           "한 번에 뻥 뚫는 큰 봉은 추격하지 않는다. 이 페이지는 **롱만** 본다. 규칙·가정은 설계서 참고.")

# ---------------------------------------------------------------------------
# 입력
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("종목·기간")
    ticker = st.text_input("티커", value="QQQ", help="미국 주식·ETF. 선물은 소스 지원 범위에 따라 다름(예: yfinance ES=F 는 최근 60일만)").strip().upper()
    c1, c2 = st.columns(2)
    start = c1.date_input("시작", value=date.today() - timedelta(days=30), max_value=date.today())
    end = c2.date_input("끝", value=date.today(), max_value=date.today())
    source = st.radio("15분봉 소스", ["Alpha Vantage (월별, 시간외 포함)", "Alpha Vantage (월별, 정규장)", "기본 스택 (EODHD→Massive→yfinance, 정규장)"], index=0,
                      help="시간외 포함이면 04:00~20:00 봉으로 VWAP·박스를 만들고 20:00 봉이 장 마감 청산 봉이 된다")
    capital = st.number_input("초기 자본 ($)", value=10_000, min_value=100, step=1_000)
    max_wick_pct = st.slider("꼬리 이상치 한도 (몸통에서 % 초과면 자름, 0 = 안 함)", 0.0, 5.0, 1.0, 0.25,
                             help="시간외 소량 체결의 bad tick(예: QQQ 2026-10-06 17:30 저가 610)이 꼬리로 찍혀 ATR·윅 익절·차트를 망친다. AV·EODHD·Massive 모두 같은 값을 준다")

    st.header("진입 ([가정] = 강의에 숫자가 없어 정한 값)")
    sr_mode = st.radio("지지·저항 모양", ["both", "trend", "box"], index=0, horizontal=True,
                       format_func=lambda k: {"both": "둘 다", "trend": "추세선(피벗 연결)", "box": "수평 박스"}[k],
                       help="추세선 = 창 안 최근 두 스윙 고점을 이은 저항선(평평·하락만)·두 스윙 저점을 이은 지지선(평평·상승만) — 강의의 하락 쐐기·상승 삼각형. 둘 다면 먼저 뚫리는 쪽")
    box_bars = st.slider("구조 탐색 창 (봉 수) [가정]", 6, 52, 16, help="직전 N 봉으로 박스·추세선을 만든다. 16 = 4시간. 강의: 횡보가 길수록 좋다")
    sr_price = st.radio("선을 긋는 가격", ["body", "wick"], index=0, horizontal=True, format_func=lambda k: {"body": "캔들 몸통", "wick": "꼬리(고가·저가)"}[k],
                        help="피벗·박스·터치·포락선 판정에 쓰는 가격. 몸통이면 긴 꼬리 하나가 선을 망치지 않는다")
    pivot_lr = st.slider("스윙 피벗 좌우 봉 수 [가정]", 1, 4, 2, help="추세선용. 고점이 좌우 N 봉보다 높아야 스윙 고점")
    trend_bars = st.slider("추세선 피벗 탐색 창 (봉 수) [가정]", 16, 390, 78, 2, help="이 창 안의 피벗 쌍 중 터치가 가장 많은(같으면 가장 긴) 포락선을 고른다. 길수록 긴 선. 78 = 정규장 3일 ≈ 시간외 포함 1.2일")
    max_box_atr = st.slider("구조 높이 ≤ ATR × [가정]", 0.5, 6.0, 3.0, 0.25, help="좁은 박스(압축)만 인정. 16봉 박스는 보통 ATR×3~4 — 1.5 이하로 조이면 거래가 거의 없다")
    min_touches = st.slider("저항(지지) 터치 봉 수 ≥", 1, 6, 2, help="강의: 여러 번 두드리고도 밀려나지 않으면 뚫릴 확률이 높다")
    touch_tol_atr = st.slider("터치 판정 폭 (ATR ×) [가정]", 0.05, 0.5, 0.15, 0.05)
    max_vwap_dist_atr = st.slider("직전 종가 ↔ VWAP 거리 ≤ ATR × [가정]", 0.25, 3.0, 1.5, 0.25, help="진입은 VWAP 에 최대한 가까이")
    max_entry_bar_atr = st.slider("돌파 봉 범위 ≤ ATR × (추격 금지) [가정]", 1.0, 5.0, 2.0, 0.25)
    breakout_volume_mult = st.slider("돌파 봉 거래량 ≥ 평균 × (0 = 미사용)", 0.0, 3.0, 1.0, 0.25)

    st.header("손절")
    stop_mode = st.selectbox("손절선", ["far", "box", "vwap"], index=0, format_func=lambda k: {"vwap": "VWAP 아래", "box": "지지(박스 아랫선·지지 추세선) 밖", "far": "둘 중 더 먼 곳"}[k])
    stop_buffer_atr = st.slider("손절 여유 (ATR ×) [가정]", 0.0, 0.5, 0.1, 0.05, help="강의: 지지선 정확히가 아니라 조금 밖에")
    stop_on_close = st.checkbox("종가 이탈 시만 손절 (윅 무시)", value=True)
    trail_vwap = st.checkbox("손절선을 VWAP 따라 끌어올림", value=False)

    st.header("익절")
    tp_min_dist_atr = st.slider("VWAP 에서 ATR × 이상 멀어진 뒤부터 [가정]", 0.0, 4.0, 1.5, 0.25)
    tp_min_dist_pct = st.slider("…또는 가격의 % 이상 (둘 중 큰 쪽) [가정]", 0.0, 2.0, 0.3, 0.1, help="시간외 봉은 ATR 이 작아 ATR 기준만으로는 2달러 위도 \"멀어짐\"이 된다")
    tp_min_bars = st.slider("진입 뒤 최소 보유 봉 (그 전엔 익절 신호 무시) [가정]", 0, 8, 2)
    tp_opp_body_atr = st.slider("역캔들 몸통 ≥ ATR × [가정]", 0.0, 2.0, 0.5, 0.1, help="0 이면 아무리 작은 음봉도 익절 신호")
    tp_volume_mult = st.slider("거래량 스파이크 ≥ 평균 × (0 = 미사용) [가정]", 0.0, 5.0, 2.5, 0.25)
    tp_opposite_candle = st.checkbox("첫 역캔들에서 익절", value=True)
    tp_wick_frac = st.slider("반대쪽 윅 ≥ 봉 범위 × (0 = 미사용) [가정]", 0.0, 0.9, 0.5, 0.05)
    tp_big_bar_atr = st.slider("큰 봉 ≥ ATR × (0 = 미사용) [가정]", 0.0, 5.0, 2.5, 0.25)
    session_flat = st.radio("강제 청산", ["weekend", "daily", "off"], index=0, horizontal=True,
                            format_func=lambda k: {"weekend": "주말·휴일 전만", "daily": "매일 장 마감", "off": "안 함"}[k],
                            help="시간외 데이터가 있으면 평일 밤은 들고 간다. 다음 봉까지 하루 넘게 비는(금요일·휴일 전) 마지막 봉에서만 청산")
    max_hold_bars = st.number_input("최대 보유 봉 수 (0 = 없음)", value=0, min_value=0, step=4)

    st.header("체결·비용")
    fill = st.radio("체결", ["close", "next_open"], index=0, format_func=lambda k: {"close": "신호 봉 종가", "next_open": "다음 봉 시가"}[k], horizontal=True)
    fee_bps = st.number_input("편도 비용 (bp)", value=5.0, min_value=0.0, step=1.0)
    risk_pct = st.number_input("1거래 위험 (자본 %) — 0 이면 전액", value=1.0, min_value=0.0, step=0.5,
                               help="포지션 크기 = 자본×위험% ÷ (진입↔손절 거리). 레버리지 없음(최대 전액)")

P = VwapSrParams(box_bars=box_bars, sr_mode=sr_mode, sr_price=sr_price, pivot_lr=pivot_lr, trend_bars=trend_bars, max_box_atr=max_box_atr, min_touches=min_touches, touch_tol_atr=touch_tol_atr,
                 max_vwap_dist_atr=max_vwap_dist_atr, max_entry_bar_atr=max_entry_bar_atr, breakout_volume_mult=breakout_volume_mult,
                 allow_long=True, allow_short=False, stop_mode=stop_mode, stop_buffer_atr=stop_buffer_atr,
                 stop_on_close=stop_on_close, trail_vwap=trail_vwap, tp_min_dist_atr=tp_min_dist_atr, tp_min_dist_pct=tp_min_dist_pct, tp_min_bars=int(tp_min_bars), tp_opp_body_atr=tp_opp_body_atr, tp_volume_mult=tp_volume_mult,
                 tp_opposite_candle=tp_opposite_candle, tp_wick_frac=tp_wick_frac, tp_big_bar_atr=tp_big_bar_atr,
                 session_flat=session_flat, max_hold_bars=int(max_hold_bars), fill=fill, fee_bps=fee_bps, risk_pct=risk_pct)


# ---------------------------------------------------------------------------
# 데이터
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner="15분봉 로딩…")
def _bars(sym: str, s: date, e: date, src: str) -> pd.DataFrame:
    if src.startswith("Alpha"):
        md = AlphaVantageAdapter()
        if "정규장" in src:
            md = RegularSessionFilterAdapter(md)
    else:
        md = RegularSessionFilterAdapter(build_default_market_data(bypass_today=e >= date.today()))
    df = md.fetch_ohlcv(sym, s, e, "15m")
    if df is None or df.empty:
        return pd.DataFrame()
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"]).sort_index()
    if df.index.tz is None:
        df.index = df.index.tz_localize("America/New_York")
    return df[~df.index.duplicated(keep="first")]


if start >= end:
    st.error("시작일이 끝보다 앞이어야 합니다."); st.stop()
if not ticker:
    st.stop()
try:
    bars = _bars(ticker, start, end, source)
except Exception as ex:  # noqa: BLE001
    st.error(f"데이터 오류: {type(ex).__name__}: {str(ex)[:200]}"); st.stop()
if bars.empty:
    st.warning("15분봉이 없습니다. 기간·소스를 바꿔 보세요 (yfinance 는 최근 60일만)."); st.stop()

bars, n_clipped = clean_wicks(bars, max_wick_pct)
res = run_vwap_sr(bars, P, capital=float(capital))
f, trades, S = res.frame, res.trades, res.stats

# ---------------------------------------------------------------------------
# 성적표
# ---------------------------------------------------------------------------
st.subheader(f"{ticker} {f.index[0]:%Y-%m-%d} ~ {f.index[-1]:%Y-%m-%d} · 15분봉 {S['봉 수']:,}개 · 거래일 {S['거래일 수']}"
             + (f" · 꼬리 이상치 {n_clipped}봉 정리" if n_clipped else ""))
m = st.columns(7)
m[0].metric("전략 수익률", f"{S['총 수익률']:+.2%}")
m[1].metric("단순 보유", f"{S['단순 보유']:+.2%}")
m[2].metric("최종 자본", f"${capital * (1 + S['총 수익률']):,.0f}")
m[3].metric("MDD", f"{S['MDD']:.2%}")
m[4].metric("거래 수", f"{S['거래 수']}")
m[5].metric("승률", f"{S['승률']:.0%}" if trades else "—")
m[6].metric("손익비 / 평균 R", (f"{S['손익비(PF)']:.2f} / {S['평균 R']:+.2f}" if trades and S['손익비(PF)'] != float('inf') else (f"∞ / {S['평균 R']:+.2f}" if trades else "—")))
if trades:
    st.caption("뚫은 구조: " + " · ".join(f"{k} {v}" for k, v in S["구조"].items()) + " | 청산 사유: " + " · ".join(f"{k} {v}" for k, v in S["청산 사유"].items())
               + f" · 거래당 평균 {S['거래당 평균']:+.2%} (중앙 {S['거래당 중앙']:+.2%}, 최악 {S['최악 거래']:+.2%}, 최고 {S['최고 거래']:+.2%}) · 평균 보유 {S['평균 보유 봉']:.1f}봉")
else:
    st.info("이 기간엔 규칙에 맞는 진입이 없었습니다. 박스 조건(높이·터치 수)이나 VWAP 거리를 완화해 보세요.")

# ---------------------------------------------------------------------------
# 차트
# ---------------------------------------------------------------------------
cA, cB, cC = st.columns([3, 1, 1])
max_bars = cA.slider("차트에 보일 최근 봉 수", 100, max(100, len(f)), min(len(f), 26 * 15), 26, help="너무 길면 느려진다. 거래 표에서 시각을 보고 범위를 좁혀도 된다")
show_all = cB.checkbox("감지된 모든 추세선·박스 그리기", value=False, help="진입하지 않은 후보까지. 실선 = 터치 수 조건 통과, 반투명 = 미달")
show_box_cand = cC.checkbox("좁은 박스 후보도", value=False, help="박스 후보는 거의 모든 봉에 있어 복잡하다. 추세선만 보려면 끈다")
# 거래 표에서 행을 선택하면(st.dataframe 선택 이벤트, 위젯 key 로 세션에 남는다) 그 거래 구간으로 차트를 옮긴다
_sel = st.session_state.get("trade_tbl")
def _sel_rows_of(st_obj):
    sel = st_obj.get("selection") if isinstance(st_obj, dict) else getattr(st_obj, "selection", None)
    rows = sel.get("rows") if isinstance(sel, dict) else getattr(sel, "rows", None)
    return list(rows or [])
_sel_rows = _sel_rows_of(_sel) if _sel is not None else []
sel_trade = trades[_sel_rows[0]] if _sel_rows and _sel_rows[0] < len(trades) else None
if sel_trade is not None:
    i_sig = f.index.get_loc(sel_trade["signal_time"]); i_ex = int(sel_trade["exit_i"])
    lo_i = max(0, min(i_sig - 3 * P.box_bars, int(sel_trade["res_geo"]["i0"]) - 4)); hi_i = min(len(f), i_ex + 2 * P.box_bars + 1)
    g = f.iloc[lo_i:hi_i]
    st.info(f"거래 #{_sel_rows[0] + 1} ({sel_trade['kind']} · {pd.Timestamp(sel_trade['entry_time']).tz_convert('America/New_York'):%m-%d %H:%M} 진입 → "
            f"{pd.Timestamp(sel_trade['exit_time']).tz_convert('America/New_York'):%m-%d %H:%M} {sel_trade['why']} {sel_trade['ret']:+.2%}) 구간을 보는 중 — "
            "거래 표에서 선택을 풀면 최근 봉 보기로 돌아간다.")
else:
    g = f.iloc[-max_bars:]
x = g.index.tz_convert("America/New_York").tz_localize(None)
fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.62, 0.18, 0.20], vertical_spacing=0.02,
                    subplot_titles=("15분봉 · VWAP(주황) · 저항선(파랑)/지지선(노랑: 수평 박스 또는 기울어진 추세선) · 손절선(빨강 점선) · 진입▲ · 청산✕", "거래량", "자본 곡선"))
fig.add_trace(go.Candlestick(x=x, open=g["Open"], high=g["High"], low=g["Low"], close=g["Close"], name="15m",
                             increasing_line_color="#26a69a", decreasing_line_color="#ef5350", showlegend=False), row=1, col=1)
fig.add_trace(go.Scatter(x=x, y=g["vwap"], mode="lines", name="세션 VWAP", line=dict(color="#ff9100", width=2.5)), row=1, col=1)
vol_color = ["#26a69a" if c >= o else "#ef5350" for o, c in zip(g["Open"], g["Close"])]
fig.add_trace(go.Bar(x=x, y=g["Volume"], marker_color=vol_color, name="거래량", showlegend=False), row=2, col=1)
fig.add_trace(go.Scatter(x=x, y=g["vol_avg"] * P.tp_volume_mult if P.tp_volume_mult > 0 else g["vol_avg"], mode="lines",
                         name="스파이크 기준", line=dict(color="orange", width=1, dash="dot")), row=2, col=1)
fig.add_trace(go.Scatter(x=x, y=g["equity"], mode="lines", name="자본", line=dict(color="#42a5f5", width=1.5)), row=3, col=1)
bh_curve = capital * g["Close"] / f["Close"].iloc[0]
fig.add_trace(go.Scatter(x=x, y=bh_curve, mode="lines", name="단순 보유", line=dict(color="gray", width=1, dash="dot")), row=3, col=1)

t0 = g.index[0]
_ny = lambda ts: pd.Timestamp(ts).tz_convert("America/New_York").tz_localize(None)
i_t0 = f.index.get_loc(t0); t1 = g.index[-1]; i_t1 = f.index.get_loc(t1)
if show_all:
    segs = [sg for sg in structure_segments(f, P) if sg["i1"] >= i_t0 and sg["i0"] <= i_t1 and (show_box_cand or sg["kind"] != "box")]
    buckets = {}
    for sg in segs:
        i0 = max(sg["i0"], i_t0); i1 = min(sg["i1"], i_t1); y0 = sg["y0"] + sg["slope"] * (i0 - sg["i0"]); y1 = sg["y0"] + sg["slope"] * (i1 - sg["i0"])
        xs = [_ny(f.index[i0]), _ny(f.index[i1]), None]
        ys_list = [[y0, y1, None]] + ([[sg["y_low"], sg["y_low"], None]] if sg["kind"] == "box" else [])
        for ys in ys_list:
            b = buckets.setdefault((sg["kind"], sg["ok"]), {"x": [], "y": []}); b["x"] += xs; b["y"] += ys
    style = {"res": ("#64b5f6", "저항 추세선 후보"), "sup": ("#fdd835", "지지 추세선 후보"), "box": ("#b0bec5", "좁은 박스 후보")}
    for (kind, ok), b in buckets.items():
        color, nm = style[kind]
        fig.add_trace(go.Scatter(x=b["x"], y=b["y"], mode="lines", name=nm + ("" if ok else " (터치 미달)"), opacity=0.9 if ok else 0.35,
                                 line=dict(color=color, width=1.2 if ok else 0.8, dash="solid" if kind != "box" else "dot"), hoverinfo="skip"), row=1, col=1)
    n_res = sum(sg["kind"] == "res" for sg in segs); n_sup = sum(sg["kind"] == "sup" for sg in segs); n_ok = sum(sg["ok"] and sg["kind"] != "box" for sg in segs)
    st.caption(f"표시 구간 후보: 저항 추세선 {n_res} · 지지 추세선 {n_sup} (터치 ≥{P.min_touches} 통과 {n_ok})"
               + (f" · 좁은 박스 {sum(sg['kind'] == 'box' for sg in segs)}" if show_box_cand else "") + f" · 실제 진입 {sum(t['exit_time'] >= t0 for t in trades)}")
for t in trades:
    if t["exit_time"] < t0 or t["signal_time"] > t1:
        continue
    for geo, color in ((t["res_geo"], "#64b5f6"), (t["sup_geo"], "#fdd835")):
        i0 = max(int(geo["i0"]), i_t0); i1 = min(int(t["exit_i"]), i_t1)
        y0 = geo["v0"] + geo["slope"] * (i0 - geo["i0"]); y1 = geo["v0"] + geo["slope"] * (i1 - geo["i0"])
        fig.add_trace(go.Scatter(x=[_ny(f.index[i0]), _ny(f.index[i1])], y=[y0, y1], mode="lines", line=dict(color=color, width=3),
                                 showlegend=False, hovertemplate=f"{t['kind']} {'저항' if color == '#64b5f6' else '지지'} {y1:.2f}<extra></extra>"), row=1, col=1)
    xe = [_ny(t["entry_time"]), _ny(min(t["exit_time"], t1))]
    fig.add_trace(go.Scatter(x=xe, y=[t["stop"], t["stop_last"]], mode="lines", line=dict(color="#ff1744", width=3, dash="dash"),
                             name="손절선", showlegend=False, hovertemplate=f"손절 {t['stop']:.2f}<extra></extra>"), row=1, col=1)
    up = t["side"] == "롱"
    fig.add_trace(go.Scatter(x=[xe[0]], y=[t["entry"]], mode="markers", marker=dict(symbol="triangle-up" if up else "triangle-down", size=13, color="#00e676" if up else "#ff1744"),
                             showlegend=False, hovertemplate=f"{t['side']} 진입 {t['entry']:.2f}<extra></extra>"), row=1, col=1)
    if t["exit_time"] > t1:
        continue
    fig.add_trace(go.Scatter(x=[xe[1]], y=[t["exit"]], mode="markers+text", marker=dict(symbol="x", size=11, color="#ffeb3b"),
                             text=[f"{t['kind']}·{t['why']} {t['ret']:+.1%}"], textposition="top center" if up else "bottom center", textfont=dict(size=10),
                             showlegend=False, hovertemplate=f"청산 {t['exit']:.2f} ({t['why']}) {t['ret']:+.2%}<extra></extra>"), row=1, col=1)

# 현재(마지막 봉) 박스
cur = current_setup(f)
if cur and len(g) > P.box_bars and t1 == f.index[-1]:
    xs = [x[-P.box_bars - 1], x[-1]]
    fig.add_trace(go.Scatter(x=xs, y=[cur["저항(박스 위)"]] * 2, mode="lines", line=dict(color="#64b5f6", width=1, dash="dot"), name="현재 박스 저항"), row=1, col=1)
    fig.add_trace(go.Scatter(x=xs, y=[cur["지지(박스 아래)"]] * 2, mode="lines", line=dict(color="#fdd835", width=1, dash="dot"), name="현재 박스 지지"), row=1, col=1)
    last = f.iloc[-1]; n_last = len(f) - 1
    for key, color, nm in (("res", "#64b5f6", "현재 저항 추세선"), ("sup", "#fdd835", "현재 지지 추세선")):
        if not pd.isna(last[key]):
            i0 = int(last[key + "_i"]); i0v = max(i0, f.index.get_loc(g.index[0]))
            y0 = last[key + "_v"] + last[key + "_slope"] * (i0v - i0)
            fig.add_trace(go.Scatter(x=[_ny(f.index[i0v]), _ny(f.index[n_last])], y=[y0, last[key]], mode="lines", line=dict(color=color, width=1.2, dash="dashdot"), name=nm), row=1, col=1)

# 세션 경계
days = pd.Series(x.normalize()).drop_duplicates()
for d in days.iloc[1:]:
    fig.add_vline(x=d, line=dict(color="rgba(255,255,255,0.15)", width=1, dash="dot"), row=1, col=1)
fig.update_layout(height=900, template="plotly_dark", xaxis_rangeslider_visible=False, hovermode="x unified", margin=dict(l=40, r=20, t=60, b=30),
                  legend=dict(orientation="h", y=1.02, x=0))
fig.update_xaxes(range=[x[0], x[-1]], rangebreaks=[dict(bounds=["sat", "mon"]), dict(bounds=[20, 4], pattern="hour") if "시간외" in source else dict(bounds=[16, 9.5], pattern="hour")])
st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------------------
# 현재 상태 · 거래 표
# ---------------------------------------------------------------------------
with st.expander("마지막 봉 기준 박스·VWAP 상태 (실시간 참고)", expanded=False):
    if cur:
        cc = st.columns(6)
        cc[0].metric("종가", f"{cur['종가']:.2f}"); cc[1].metric("VWAP", f"{cur['VWAP']:.2f}")
        cc[2].metric("저항(박스 위)", f"{cur['저항(박스 위)']:.2f}", f"터치 {cur['저항 터치']}")
        cc[3].metric("지지(박스 아래)", f"{cur['지지(박스 아래)']:.2f}", f"터치 {cur['지지 터치']}")
        cc[4].metric("좁은 박스", "예" if cur["좁은 박스"] else "아니오", f"ATR {cur['ATR']:.2f}")
        cc[5].metric("신호", "롱" if cur["롱 신호"] else "없음")
        st.caption(f"저항 추세선 {cur['저항 추세선']:.2f}" if cur["저항 추세선"] else "저항 추세선 없음(스윙 고점 2개 미만이거나 올라가는 선)"
                   + " · " + (f"지지 추세선 {cur['지지 추세선']:.2f}" if cur["지지 추세선"] else "지지 추세선 없음"))

with st.expander("돌파했는데 진입 안 한 봉 — 막은 조건 (표시 구간)", expanded=False):
    mb = missed_breakouts(f, P)
    if len(mb):
        mb = mb[mb["시각"] >= t0]
    if len(mb):
        mb["시각"] = pd.to_datetime(mb["시각"]).dt.tz_convert("America/New_York").dt.strftime("%m-%d %H:%M")
        st.caption("종가가 선택된 저항(추세선·박스)을 넘었지만 다른 조건에 걸린 봉. 포지션 보유 중이던 봉도 포함된다.")
        st.dataframe(mb.style.format({"종가": "{:.2f}", "저항": "{:.2f}", "VWAP": "{:.2f}", "ATR": "{:.3f}"}), width="stretch", height=min(400, 60 + 35 * len(mb)))
        st.caption("막은 조건 집계: " + " · ".join(f"{k} {v}" for k, v in pd.Series([w for ws in mb["막은 조건"] for w in ws.split(" · ")]).value_counts().items()))
    else:
        st.write("표시 구간에 돌파 봉이 없습니다.")

st.subheader("거래 목록")
if trades:
    tb = pd.DataFrame(trades)
    for k in ("signal_time", "entry_time", "exit_time"):
        tb[k] = pd.to_datetime(tb[k]).dt.tz_convert("America/New_York").dt.strftime("%m-%d %H:%M")
    show = tb[["kind", "entry_time", "entry", "stop", "box_high", "box_low", "vwap_entry", "exit_time", "exit", "why", "bars", "ret", "R", "frac", "pnl", "equity"]].rename(
        columns={"kind": "구조", "entry_time": "진입", "entry": "진입가", "stop": "손절선", "box_high": "저항(진입 시)", "box_low": "지지(진입 시)", "vwap_entry": "진입 VWAP",
                 "exit_time": "청산", "exit": "청산가", "why": "사유", "bars": "보유 봉", "ret": "수익률", "R": "R", "frac": "투입 비중", "pnl": "자본 손익", "equity": "자본"})
    st.dataframe(show.style.format({"진입가": "{:.2f}", "손절선": "{:.2f}", "저항(진입 시)": "{:.2f}", "지지(진입 시)": "{:.2f}", "진입 VWAP": "{:.2f}", "청산가": "{:.2f}",
                                    "수익률": "{:+.2%}", "R": "{:+.2f}", "투입 비중": "{:.0%}", "자본 손익": "{:+.2%}", "자본": "{:,.0f}"}),
                 width="stretch", height=min(600, 60 + 35 * len(show)), key="trade_tbl", on_select="rerun", selection_mode="single-row")
    st.caption("행 맨 왼쪽 체크박스를 누르면 위 차트가 그 거래 구간(진입 전 구조 → 청산 뒤)으로 이동한다. 다시 누르면 해제.")
    st.download_button("거래 CSV", tb.drop(columns=["res_geo", "sup_geo"]).to_csv(index=False).encode("utf-8-sig"), file_name=f"vwap_sr_{ticker}_{start}_{end}.csv")

with st.expander("쓰인 파라미터 (JSON)"):
    st.json({k.name: getattr(P, k.name) for k in fields(P)})
