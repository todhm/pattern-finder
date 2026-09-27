"""미너비니 트레이드 리플레이 — 스크리너가 뽑은 종목의 그 후.

44번 스크리너가 기준일에 뽑은 종목(또는 임의 티커)을 골라, 기준일
이후 주가가 실제로 어떻게 움직였고 **피봇 돌파 → 손절선/익절선 규칙을
그대로 따랐다면** 언제 진입·청산되어 얼마를 벌고 잃었는지 차트 위에
재생한다. 규칙 변형(익절 고정 / 50일선 러너 / 본전 스톱 / 존버)별
결과도 나란히 비교.

시뮬레이션 로직: ``strategy/adapters/minervini_screener.replay_trade``
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from data.adapters.composed_market_data import build_default_market_data
from pages._shared.growth_sources import fetch_growth_snapshot
from pages._shared.manual_evidence import load_manual_evidence
from pages._shared.market_regime import (
    combine_rows,
    fetch_market_verdicts,
    stage0_lines,
    verdicts_to_state,
)
from pages._shared.minervini_manual_checks import render_manual_checklist
from strategy.adapters.minervini_screener import (
    TradeReplay,
    annual_evidence,
    evaluate_annual,
    pivot_levels,
    replay_trade,
)

st.set_page_config(page_title="미너비니 트레이드 리플레이", layout="wide")
st.title("미너비니 트레이드 리플레이 — 뽑은 종목, 그 후")
st.caption(
    "기준일까지의 데이터로 피봇·손절선·익절선을 정하고(스크리너와 동일 "
    "로직), 기준일 **이후** 데이터로 돌파 매수를 재생한다: 언제 진입됐고, "
    "어느 선에 먼저 닿았고, 규칙 변형별로 결과가 어떻게 갈리는지. "
    "44번 스크리너를 먼저 돌렸다면 그 후보를 그대로 불러올 수 있다."
)

_screen = st.session_state.get("minervini_screen")

with st.sidebar:
    st.header("대상")
    candidates = []
    if _screen is not None and len(_screen.get("df", [])):
        candidates = list(_screen["df"]["티커"])
    source = st.radio(
        "종목 소스",
        ([f"44 스크린 후보 ({_screen['as_of']})"] if candidates else [])
        + ["직접 입력"],
        help="44번 페이지에서 Run Screen을 돌린 세션이면 후보가 나타난다.",
    )
    from_screen = source != "직접 입력"
    if from_screen:
        sel_ticker = st.selectbox("후보 종목", candidates)
        row = _screen["df"][_screen["df"]["티커"] == sel_ticker].iloc[0]
        ticker = sel_ticker
        as_of = _screen["as_of"]
        screen_pivot = float(row["피봇"])
        st.caption(f"스크린 피봇 ${screen_pivot:,.2f} · "
                   f"손절선 ${row['손절선']:,.2f} · 익절선 ${row['익절선']:,.2f}")
    else:
        ticker = st.text_input("티커", value="TSLA").strip().upper()
        as_of = st.date_input(
            "기준일 (스크린 날짜)", value=date(2020, 1, 2),
            min_value=date(2005, 1, 1), max_value=date.today(),
        )
        screen_pivot = None

    st.header("피봇 / 규칙")
    pivot_mode = st.radio(
        "피봇 결정",
        (["스크린 값 그대로"] if from_screen else []) + ["자동 (VCP 정밀)", "수동 입력"],
    )
    manual_pivot = (
        st.number_input("수동 피봇 ($)", value=0.0, min_value=0.0, format="%.2f")
        if pivot_mode == "수동 입력" else None
    )
    stop_pct = st.number_input("손절폭 (%)", value=8.0, min_value=1.0, max_value=10.0, step=0.5)
    target_pct = st.number_input("익절 목표 (%)", value=22.0, min_value=5.0, max_value=50.0, step=1.0)
    entry_window = st.number_input(
        "돌파 대기 (거래일)", value=20, min_value=1, max_value=60,
        help="기준일 이후 이 기간 안에 피봇을 돌파하지 못하면 '미돌파'.",
    )
    breakeven_at = st.number_input(
        "본전 스톱 트리거 (%, 변형 C)", value=10.0, min_value=3.0, max_value=30.0, step=1.0,
        help="종가가 진입가 +N%를 찍은 다음날부터 손절선을 본전으로 올림 (무료 롤).",
    )
    horizon_m = st.slider("관찰 기간 (개월)", 6, 24, 14)
    run_btn = st.button("Replay", type="primary", use_container_width=True)

VARIANT_DEFS = [
    ("A 기본", "손절 −{stop:g}% / 익절 +{target:g}% 선착"),
    ("B 러너", "손절 −{stop:g}% + 익절 없이 50일선 종가 이탈 청산"),
    ("C 본전스톱", "A + 종가 +{be:g}% 후 본전 스톱 (무료 롤)"),
    ("D 존버", "돌파 진입 후 규칙 없이 기간 끝까지 보유 (벤치마크)"),
]

if run_btn:
    md = build_default_market_data()
    fetch_end = min(as_of + timedelta(days=int(horizon_m * 30.5)), date.today())
    with st.spinner(f"{ticker} 데이터 수집..."):
        try:
            df_all = md.fetch_ohlcv(ticker, as_of - timedelta(days=460), fetch_end)
        except Exception as e:
            st.error(f"데이터 fetch 실패: {e}")
            st.stop()
    if df_all.index.tz is not None:
        df_all.index = df_all.index.tz_localize(None)
    df_all.index = df_all.index.normalize()
    before = df_all[df_all.index <= pd.Timestamp(as_of)]
    if len(before) < 60:
        st.error(f"{ticker}: 기준일 이전 데이터 {len(before)}봉 — 최소 60일 필요.")
        st.stop()

    pl = pivot_levels(before, stop_pct=stop_pct / 100.0, target_pct=target_pct / 100.0)
    if pivot_mode == "수동 입력" and manual_pivot:
        pivot = float(manual_pivot)
        pivot_src = "수동"
    elif pivot_mode == "스크린 값 그대로" and screen_pivot:
        pivot = screen_pivot
        pivot_src = "44 스크린"
    else:
        if pl is None:
            st.error("VCP 피봇 계산 불가 (데이터 부족) — 수동 피봇을 입력해줘.")
            st.stop()
        pivot = pl.pivot
        pivot_src = f"자동 [{pl.method}]"

    common = dict(df_all=df_all, as_of=as_of, pivot=pivot,
                  entry_window=int(entry_window))
    replays: dict[str, TradeReplay] = {
        "A 기본": replay_trade(**common, stop_pct=stop_pct / 100.0,
                              target_pct=target_pct / 100.0),
        "B 러너": replay_trade(**common, stop_pct=stop_pct / 100.0,
                              target_pct=None, ma_exit=50),
        "C 본전스톱": replay_trade(**common, stop_pct=stop_pct / 100.0,
                                target_pct=target_pct / 100.0,
                                breakeven_trigger=breakeven_at / 100.0),
        "D 존버": replay_trade(**common, stop_pct=None, target_pct=None),
    }
    st.session_state["minervini_replay"] = {
        "ticker": ticker, "as_of": as_of, "pivot": pivot,
        "pivot_src": pivot_src, "pl": pl, "df_all": df_all,
        "replays": replays, "stop_pct": stop_pct, "target_pct": target_pct,
        "breakeven_at": breakeven_at,
    }

_state = st.session_state.get("minervini_replay")
if _state is None:
    st.info("좌측에서 종목·기준일·규칙을 정하고 **Replay**. 44번 스크리너를 "
            "먼저 돌리면 후보를 드롭다운으로 바로 불러올 수 있다.")
    st.stop()

ticker = _state["ticker"]
as_of = _state["as_of"]
pivot = _state["pivot"]
pl = _state["pl"]
df_all = _state["df_all"]
replays = _state["replays"]
stop_line = pivot * (1 - _state["stop_pct"] / 100.0)
target_line = pivot * (1 + _state["target_pct"] / 100.0)

st.subheader(f"{ticker} @ {as_of} — 피봇 ${pivot:,.2f} ({_state['pivot_src']})")
if pl is not None:
    seq = " → ".join(f"−{d:.0%}" for d in pl.contractions) or "감지된 축소 없음"
    st.caption(f"기준일 시점 VCP: {pl.status} · 축소 {seq} · "
               f"조밀도10d {pl.tightness_10d:.1%} · 드라이업 {pl.volume_dryup:.0%}")

# --- 변형별 비교 테이블 -----------------------------------------------------
rows = []
for (name, desc_tpl) in VARIANT_DEFS:
    r = replays[name]
    desc = desc_tpl.format(stop=_state["stop_pct"], target=_state["target_pct"],
                           be=_state["breakeven_at"])
    rows.append({
        "규칙": name, "설명": desc,
        "결과": r.exit_reason if r.entered else "미돌파",
        "진입일": r.entry_date, "청산일": r.exit_date,
        "수익률": r.ret, "보유(거래일)": r.days_held,
        "최대상승(MFE)": r.mfe, "최대하락(MAE)": r.mae,
    })
st.dataframe(
    pd.DataFrame(rows), use_container_width=True, hide_index=True,
    column_config={c: st.column_config.NumberColumn(format="percent")
                   for c in ("수익률", "최대상승(MFE)", "최대하락(MAE)")},
)

base = replays["A 기본"]
if base.entered:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("진입", f"${base.entry_price:,.2f}",
              f"{base.entry_date}")
    if base.breakout_volume_ratio is not None:
        vol_ok = base.breakout_volume_ratio >= 1.4
        m2.metric("돌파일 거래량 (4-12)",
                  f"50일 평균 × {base.breakout_volume_ratio:.2f}",
                  "✅ 유효 돌파 (≥1.4)" if vol_ok else "⚠️ 거래량 미달 — 가짜 돌파 의심",
                  delta_color="normal" if vol_ok else "inverse")
    m3.metric("A 기본 결과", base.exit_reason,
              f"{base.ret:+.1%}" if base.ret is not None else None)
    m4.metric("최대상승 / 최대하락",
              f"{base.mfe:+.0%} / {base.mae:+.0%}")
else:
    st.warning(f"돌파 대기 {int(_state.get('entry_window', 20))}거래일 안에 피봇 미돌파 — "
               "진입 자체가 발생하지 않았다 (이것도 유효한 결과: 셋업 무산).")

# --- 차트: 기준일 전 맥락 + 이후 재생 ---------------------------------------
viz_name = st.selectbox("차트에 표시할 규칙", [n for n, _ in VARIANT_DEFS], index=0)
viz = replays[viz_name]

before_bars = 130
start_pos = max(0, len(df_all[df_all.index <= pd.Timestamp(as_of)]) - before_bars)
plot_df = df_all.iloc[start_pos:]
fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                    row_heights=[0.78, 0.22], vertical_spacing=0.02)
fig.add_trace(go.Candlestick(
    x=plot_df.index, open=plot_df["Open"], high=plot_df["High"],
    low=plot_df["Low"], close=plot_df["Close"], name="가격",
    increasing_line_color="#26A69A", decreasing_line_color="#EF5350",
), row=1, col=1)
sma50 = df_all["Close"].rolling(50).mean().reindex(plot_df.index)
fig.add_trace(go.Scatter(x=plot_df.index, y=sma50, name="SMA50",
                         line=dict(width=1.2, color="#FFA726")), row=1, col=1)
for level, name, color in (
    (pivot, f"피봇 {pivot:,.2f}", "#FFEE58"),
    (stop_line, f"손절 {stop_line:,.2f}", "#EF5350"),
    (target_line, f"익절 {target_line:,.2f}", "#26A69A"),
):
    fig.add_hline(y=level, line_dash="dot", line_color=color,
                  annotation_text=name, annotation_position="right", row=1, col=1)
fig.add_vline(x=pd.Timestamp(as_of), line_dash="dash", line_color="#B0BEC5",
              annotation_text="기준일", annotation_position="top left")
if viz.entered:
    fig.add_trace(go.Scatter(
        x=[pd.Timestamp(viz.entry_date)], y=[viz.entry_price],
        mode="markers+text", name="진입",
        marker=dict(symbol="triangle-up", size=13, color="#FFEE58"),
        text=["진입"], textposition="bottom center",
    ), row=1, col=1)
    if viz.exit_date is not None:
        fig.add_trace(go.Scatter(
            x=[pd.Timestamp(viz.exit_date)], y=[viz.exit_price],
            mode="markers+text", name=f"청산({viz.exit_reason})",
            marker=dict(symbol="x", size=12,
                        color="#26A69A" if (viz.ret or 0) > 0 else "#EF5350"),
            text=[f"{viz.exit_reason} {viz.ret:+.0%}"],
            textposition="top center",
        ), row=1, col=1)
vol_colors = ["#FFEE58" if viz.entered and ts.date() == viz.entry_date
              else "#78909C" for ts in plot_df.index]
fig.add_trace(go.Bar(x=plot_df.index, y=plot_df["Volume"], name="거래량",
                     marker_color=vol_colors, opacity=0.8), row=2, col=1)
vol50 = df_all["Volume"].rolling(50).mean().reindex(plot_df.index)
fig.add_trace(go.Scatter(x=plot_df.index, y=vol50, name="거래량 50일 평균",
                         line=dict(width=1, color="#EF5350", dash="dot")), row=2, col=1)
fig.update_layout(
    template="plotly_dark", height=620, xaxis_rangeslider_visible=False,
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    margin=dict(l=40, r=95, t=30, b=20),
)
st.plotly_chart(fig, use_container_width=True)

# --- 진입 후 일자별 테이블 ---------------------------------------------------
if viz.entered:
    st.subheader(f"진입 후 일자별 흐름 — {viz_name}")
    fwd = df_all[df_all.index >= pd.Timestamp(viz.entry_date)].head(90)
    day_rows = []
    for ts, row in fwd.iterrows():
        d = ts.date()
        event = ""
        if d == viz.entry_date:
            event = f"🎯 돌파 진입 ${viz.entry_price:,.2f}"
        if viz.exit_date is not None and d == viz.exit_date:
            event = f"🏁 {viz.exit_reason} ${viz.exit_price:,.2f}"
        day_rows.append({
            "날짜": d, "종가": float(row["Close"]),
            "진입가 대비": float(row["Close"]) / viz.entry_price - 1.0,
            "손절선까지": float(row["Low"]) / stop_line - 1.0,
            "익절선까지": float(row["High"]) / target_line - 1.0,
            "이벤트": event,
        })
        if viz.exit_date is not None and d >= viz.exit_date:
            break
    st.dataframe(
        pd.DataFrame(day_rows), use_container_width=True, hide_index=True,
        column_config={
            "종가": st.column_config.NumberColumn(format="dollar"),
            **{c: st.column_config.NumberColumn(format="percent")
               for c in ("진입가 대비", "손절선까지", "익절선까지")},
        },
    )
    st.caption(
        "'손절선까지'는 그날 저가 기준(음수 = 손절선 하회), '익절선까지'는 "
        "고가 기준(양수 = 익절선 상회). 청산일까지만 표시 (최대 90거래일)."
    )

# --- STAGE 0 시장 환경 (기준일 시점, SPY + QQQ) --------------------------------
st.divider()


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def _market_rows(d: date) -> tuple[list[dict], list[str]]:
    vs, errs = fetch_market_verdicts(build_default_market_data(), d)
    return verdicts_to_state(vs), errs


_mrows, _merrs = _market_rows(as_of)
if _mrows:
    _mok, _mfail = combine_rows(_mrows)
    _mlines, _, _ = stage0_lines(_mrows)
    _box = st.success if _mok else st.error
    _box(
        f"STAGE 0 시장 환경 @ {as_of} — "
        + ("통과 (신규 매수 허용 구간)" if _mok else
           f"**{' · '.join(_mfail)} → 매뉴얼상 신규 매수 중단 구간이었다. "
           "이 리플레이는 '규칙을 어기고 진입한 케이스'로 읽을 것.**")
        + "\n\n" + "\n\n".join(_mlines)
        + ("\n\n⚠️ " + " · ".join(_merrs) if _merrs else "")
    )
else:
    st.warning("STAGE 0 지수 조회 실패: " + " · ".join(_merrs))

# --- 연간 펀더멘털 자동 판정 (2-5 코드 33 · 2-6 · 2-10) ------------------------
st.divider()
st.subheader("연간 펀더멘털 자동 판정 — 2-5 코드 33 · 2-6 EPS 신고 · 2-10 감속 경고")


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def _annual_rows(tk: str, d: date) -> tuple[list[dict], list[dict]] | None:
    """(판정 행, 연간 근거 행). 소스 없으면 None. 발표일 ≤ 기준일 연도만."""
    snap = fetch_growth_snapshot(tk)
    if snap is None:
        return None
    a = evaluate_annual(snap.annuals, d)
    verdict = {True: "✅ 통과", False: "❌ 실패", None: "ℹ️ 미확인"}
    yoy = lambda ys: " → ".join(f"{y:+.0%}" for y in ys) if ys else "없음"
    rows = [
        {"항목": "2-5a 코드 33: 연간 EPS 3년 가속", "판정": verdict[a.checks.get("2-5a EPS 3년 가속")],
         "근거": f"EPS YoY {yoy(a.eps_yoy)}"},
        {"항목": "2-5b 코드 33: 연간 매출 3년 가속", "판정": verdict[a.checks.get("2-5b 매출 3년 가속")],
         "근거": f"매출 YoY {yoy(a.revenue_yoy)}"},
        {"항목": "2-5c 코드 33: 순이익률 3년 연속 상승", "판정": verdict[a.checks.get("2-5c 순이익률 3년 상승")],
         "근거": "순이익률 " + " → ".join(f"{m:.1%}" if m is not None else "?" for m in a.margins)},
        {"항목": "2-5 코드 33 종합 (보너스)", "판정": verdict[a.passed if a.data_available else None],
         "근거": f"{a.n_passed}/3 (3/3 = 초고수익 최상급 후보)"},
        {"항목": "2-6 연간 EPS 과거 고점 돌파 (보너스)", "판정": verdict[a.eps_breakout],
         "근거": (f"최신 ${a.eps[-1]:,.2f} vs 이전 최고 ${a.eps_prev_high:,.2f}"
                 if a.eps_prev_high is not None and a.eps and a.eps[-1] is not None
                 else "비교할 이전 연도 3개 미만")},
        {"항목": "2-10 연간 EPS 급감속 없음 (적색경보)",
         "판정": verdict[None if a.decel_warning is None else not a.decel_warning],
         "근거": f"EPS YoY {yoy(a.eps_yoy[-3:])}"
                 + (" — 델식 감속(80→65→28%) 경고, 신규 진입 금지" if a.decel_warning else "")},
    ]
    return rows, annual_evidence(snap.annuals, d)


_ann = _annual_rows(ticker, as_of)
if _ann is None:
    st.info("펀더멘털 소스 없음 (ALPHAVANTAGE_API_KEY / EODHD_API_KEY 미설정) — 연간 판정 생략.")
else:
    _rows, _evidence = _ann
    st.dataframe(pd.DataFrame(_rows), use_container_width=True, hide_index=True,
                 column_config={"근거": st.column_config.TextColumn(width="large")})
    if _evidence:
        with st.expander("연간 실적 근거 (기준일까지 발표된 회계연도만)"):
            st.dataframe(
                pd.DataFrame(_evidence), use_container_width=True, hide_index=True,
                column_config={
                    "EPS YoY": st.column_config.NumberColumn(format="percent"),
                    "매출 YoY": st.column_config.NumberColumn(format="percent"),
                    "순이익률": st.column_config.NumberColumn(format="percent"),
                    "매출": st.column_config.NumberColumn(format="compact"),
                },
            )
    st.caption("소스(Alpha Vantage) 연간 EPS는 GAAP/non-GAAP 혼재 가능 — 한 해만 튀면 "
               "일회성 손익일 수 있으니 보도자료로 재확인. 2-10 ❌면 매뉴얼상 즉시 탈락.")

st.divider()
render_manual_checklist(f"{as_of}_{ticker}", include_post_breakout=True,
                        ticker=ticker, as_of=as_of,
                        evidence=load_manual_evidence(ticker, as_of))
