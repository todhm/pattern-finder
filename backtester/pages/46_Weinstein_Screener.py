"""와인스타인 «최적의 타이밍» 전략 검증 — 기준일 하나로 유니버스에서 1→2단계 돌파 종목을 뽑고,
체크리스트·등급을 매긴 뒤 기준일 이후 데이터로 책의 등급 체계가 실제로 성과를 가르는지 본다.

체크리스트 원문: docs/strategy_notes/와인스타인_최적의타이밍_체크리스트_2026_10.md (137장 필기 스크린샷 요약)
TQQQ 검증 결론(§8-3): 책이 수익을 더하는 쪽은 지수가 아니라 **주도 업종의 1→2 전환 종목** → 이 페이지.
로직: strategy/adapters/weinstein_scan.py (깔때기·사후 검증), weinstein_stage.py (단계·베이스·체크리스트·등급)
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from data.adapters.composed_market_data import build_default_market_data
from data.adapters.wikipedia_universe import default_universe_provider
from pages._shared.growth_sources import build_growth_adapters
from strategy.adapters.weinstein_scan import (
    Candidate,
    ScanParams,
    check_stats,
    grade_stats,
    run_scan,
    simulate_forward,
)
from strategy.adapters.weinstein_stage import StageParams, checks_to_frame

st.set_page_config(page_title="와인스타인 단계 스크리너", layout="wide")
st.title("와인스타인 «최적의 타이밍» — 1→2단계 돌파 스크리너 & 등급 검증")
st.caption(
    "기준일(과거의 '오늘') 금요일 주봉으로 유니버스 전 종목의 30주 MA·단계·트레이딩 범위를 계산해 "
    "**이번 주 저항선을 돌파한 종목**(또는 돌파 대기 워치)을 뽑고 STAGE 0~5 체크리스트 → §7 등급(A+/A/B/B-)을 매긴다. "
    "사후 검증을 켜면 돌파 주 다음 거래일 시가 진입 → 7·8장 규칙 청산으로 **등급별 성과를 분리**한다 — "
    "책의 'A+는 대어, C-는 실패'가 데이터로 성립하는지가 이 페이지의 질문. "
    "⚠️ sp500/nasdaq100 유니버스는 현재 구성이라 과거 기준일엔 생존 편향이 있다 — 과거 검증은 `@기준일` 유니버스 권장."
)

_MANUAL_ITEMS = [  # 문서 §B — 코드가 못 보는 것. 종목+기준일 단위로 세션에 유지
    ("b1", "B-1 단계 분류가 눈으로 봐도 맞는가 (MA '평평' 판단, 3단계 vs 2단계 쉬어가기)"),
    ("b2", "B-2 저항·지지 수평선을 직접 그었는가 — 코드의 최댓값이 아니라 '여러 번 막힌 가격대'"),
    ("b2b", "B-2 10년 월봉에서 돌파가 위 거래 이력 확인 (팬암 4년 8 이상 = C- / 앨러지스 없음 = A+)"),
    ("b3", "B-3 돌파 주봉이 몸통·종가로 확실히 넘었는가, 거래량 막대가 베이스에서 눈에 띄게 큰가"),
    ("b4", "B-4 풀백이 돌파가·MA 근처에서 좁게 멈추고 거래량이 말라붙는가"),
    ("b5", "B-5 패턴(수평 박스 / 하락 추세선 돌파 / H&S 바닥 / 이중 바닥) — 추세선은 3접점인가"),
    ("b6", "B-6 RS선이 주가 횡보 중 저점을 높이며 올라왔는가, 돌파 주에 제로선을 분명히 넘는가"),
    ("b7", "B-7 같은 업종 3~5개 차트를 훑어 다수가 함께 돌파하는가 (혼재면 업종 신뢰 ↓)"),
    ("b9", "7-11 buy-stop 50% + 풀백 50%, 손절가(돌파가 −5% / 지지선 아래), 스윙 목표가를 숫자로 적었는가"),
]


@st.cache_data(ttl=86400, show_spinner=False)
def _universe_tickers(choice: str, as_of_date: date) -> list[str]:
    provider = default_universe_provider()
    if choice.endswith("@기준일"):
        return provider.get_tickers(f"{choice[:-len('@기준일')]}@{as_of_date.isoformat()}")
    if choice == "sp500+nasdaq100":
        merged = provider.get_tickers("sp500") + provider.get_tickers("nasdaq100")
        return list(dict.fromkeys(merged))
    return provider.get_tickers(choice)


with st.sidebar:
    st.header("기준일 / 유니버스")
    as_of = st.date_input("기준일", value=date.today(), min_value=date(2005, 1, 1), max_value=date.today(),
                          help="이날까지의 데이터만 본다. 금요일 마감 전이면 진행 중인 주봉이 포함된다(아래 토글).")
    universe_choice = st.selectbox(
        "유니버스", ["sp500", "sp500+nasdaq100", "nasdaq100", "nasdaq_full@기준일", "us_all@기준일"],
        help="저자 권장 시장 폭 유니버스 = S&P 500. `@기준일` = 그날 실제 상장 종목(생존 편향 없음, 느림).",
    )
    max_tickers = st.number_input("최대 종목 수 (0=전체)", value=0, min_value=0, step=50,
                                  help="스모크 테스트용. 종목당 11년 일봉을 받으므로 첫 실행은 느리고 이후엔 Mongo 캐시.")
    include_partial = st.toggle("진행 중인 주봉 포함", value=True,
                                help="끄면 마지막 금요일 마감분까지만 — 백테스트처럼 완결된 주봉으로 판정.")

    st.header("셋업")
    mode = st.radio("모드", ["breakout", "watch"], horizontal=True,
                    help="breakout = 최근 N주 안에 저항선·30주 MA 를 주봉 종가로 돌파 / watch = 저항 −5% 이내에서 대기 중(buy-stop 후보)")
    recent_weeks = st.number_input("돌파 탐색 범위 (주)", value=1, min_value=1, max_value=8,
                                   help="1 = 이번 주 돌파만. 늘리면 지난 몇 주의 돌파도 후보(진입은 각 돌파 주 다음 거래일).")
    min_grade = st.selectbox("표시 최소 등급", ["전부", "B-", "B", "A", "A+"], index=0)

    st.header("책의 임계값 (§6 제안값)")
    base_min_weeks = st.number_input("베이스 최소 길이 (주)", value=12, min_value=4, max_value=52)
    vol_pass = st.number_input("4-1 돌파 거래량 배수 (통과)", value=2.0, min_value=1.0, max_value=5.0, step=0.5)
    vol_required = st.toggle("4-1 거래량 필수", value=True, help="끄면 정보 항목 — ETF 처럼 거래량 규칙이 안 맞는 자산용")
    near_res = st.number_input("3-3 돌파가 위 매물대 탐색 범위 (%)", value=15, min_value=5, max_value=40, step=5)
    ma_ext = st.number_input("2-10 MA 이격 상한 (%)", value=15, min_value=5, max_value=60, step=5)
    min_dollar_vol = st.number_input("최소 일평균 거래대금 ($M)", value=5.0, min_value=0.0, step=1.0)

    st.header("검증 (기준일 이후 — 평가 뒤에만!)")
    show_forward = st.toggle("사후 시뮬레이션 (등급별 성과)", value=False,
                             help="돌파 주 다음 거래일 시가 진입, 7-5/7-7/8-3/8-4 청산, 최대 52주. 눈 판독 훈련 중이면 끄고 시작.")
    run_btn = st.button("Run Screen", type="primary", use_container_width=True)


def _chart(c: Candidate, weeks: int = 130):
    w = c.weekly.w.tail(weeks)
    rs = c.weekly.rs.reindex(w.index)
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.6, 0.2, 0.2], vertical_spacing=0.02)
    fig.add_trace(go.Candlestick(x=w.index, open=w.Open, high=w.High, low=w.Low, close=w.Close, name="주봉",
                                 increasing_line_color="#26A69A", decreasing_line_color="#EF5350"), row=1, col=1)
    fig.add_trace(go.Scatter(x=w.index, y=w.MA, name="30주 MA", line=dict(width=1.6, color="#FFA726")), row=1, col=1)
    fig.add_vrect(x0=c.base.start, x1=c.base.end, fillcolor="#42A5F5", opacity=0.12, line_width=0, row=1, col=1)
    for level, name, color in ((c.resistance, f"저항 {c.resistance:,.2f}", "#FFEE58"),
                               (c.support, f"지지 {c.support:,.2f}", "#AB47BC"),
                               (c.resistance * 0.95, f"스톱 {c.resistance * 0.95:,.2f}", "#EF5350")):
        fig.add_hline(y=level, line_dash="dot", line_color=color, annotation_text=name,
                      annotation_position="right", row=1, col=1)
    v4 = w["Volume"].rolling(4).mean().shift(1)
    colors = np.where(w["Volume"] >= 2 * v4, "#FFEE58", "#78909C")
    fig.add_trace(go.Bar(x=w.index, y=w.Volume, name="거래량 (노랑 = 4주 평균 2배↑)", marker_color=colors, opacity=0.8), row=2, col=1)
    fig.add_trace(go.Scatter(x=w.index, y=rs, name="맨스필드 RS (vs SPY)", line=dict(width=1.4, color="#26C6DA"),
                             fill="tozeroy"), row=3, col=1)
    fig.add_hline(y=0, line_color="#BDBDBD", line_width=1, row=3, col=1)
    fig.update_layout(template="plotly_dark", height=640, showlegend=True, xaxis_rangeslider_visible=False,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02), margin=dict(l=40, r=100, t=30, b=20))
    return fig


def _stage_strip(c: Candidate, weeks: int = 130) -> str:
    s = c.weekly.stages.tail(weeks)
    runs, cur, n = [], None, 0
    for v in s:
        if v == cur:
            n += 1
        else:
            if cur is not None:
                runs.append(f"{cur}단계×{n}주")
            cur, n = v, 1
    runs.append(f"{cur}단계×{n}주")
    return " → ".join(runs[-8:])


if run_btn:
    market_data = build_default_market_data()
    sp = StageParams(base_min_weeks=int(base_min_weeks), vol_pass=float(vol_pass), vol_required=bool(vol_required),
                     near_resistance_pct=near_res / 100.0, late_stage_ma_ext=ma_ext / 100.0)
    params = ScanParams(mode=mode, recent_weeks=int(recent_weeks), min_dollar_vol_m=float(min_dollar_vol),
                        include_partial_week=bool(include_partial), stage=sp)
    with st.spinner("유니버스 로드..."):
        try:
            tickers = _universe_tickers(universe_choice, as_of)
        except Exception as e:  # noqa: BLE001
            st.error(f"유니버스 로드 실패: {e}")
            st.stop()
    if max_tickers:
        tickers = tickers[: int(max_tickers)]
    prog = st.progress(0.0, text=f"일봉 수집·단계 분류 중... ({len(tickers)}종목, 종목당 11년)")
    logs: list[str] = []

    def _log(m: str) -> None:
        logs.append(m)
        prog.progress(min(0.95, len(logs) / max(len(tickers) / 100 + 2, 1)), text=m)

    res = run_scan(as_of, tickers, market_data, params, build_growth_adapters(), max_workers=4, log=_log)
    prog.empty()

    rows, outcomes = [], {}
    if show_forward and res.candidates:
        fprog = st.progress(0.0, text="사후 시뮬레이션...")
        for i, c in enumerate(res.candidates):
            fprog.progress((i + 1) / len(res.candidates), text=f"검증 {i + 1}/{len(res.candidates)} — {c.ticker}")
            try:
                outcomes[c.ticker] = simulate_forward(res.frames[c.ticker], c, sp)
            except Exception:  # noqa: BLE001
                outcomes[c.ticker] = None
        fprog.empty()
    for c in res.candidates:
        o = outcomes.get(c.ticker)
        rows.append({
            "티커": c.ticker, "진입": c.entry_note, "등급": c.grade, "종목등급": c.stock_grade, "A+플래그": "⭐" if c.aplus_flag else "",
            "유형": c.entry_type, "섹터": c.sector or "미확인", "섹터ETF": c.sector_etf or "", "단계": c.stage_now,
            "종가": c.close, "저항(돌파가)": c.resistance, "지지": c.support, "돌파폭": c.close / c.resistance - 1,
            "베이스(주)": c.base.weeks, "베이스폭": c.base.width, "거래량배수": c.vol_ratio, "RS": c.rs,
            "돌파주": c.week_end.date(),
            **({"결과": o.reason if o and o.entered else "미진입", "수익률": o.ret if o else None,
                "보유주": o.weeks if o else None, "MFE": o.mfe if o else None, "MAE": o.mae if o else None,
                "+13주": o.ret_13w if o else None, "+26주": o.ret_26w if o else None,
                "풀백2차": o.pullback_fill if o else None} if show_forward else {}),
        })
    st.session_state["weinstein_screen"] = {
        "as_of": as_of, "universe": universe_choice, "res": res, "df": pd.DataFrame(rows),
        "show_forward": show_forward, "outcomes": outcomes, "sp": sp,
    }

_state = st.session_state.get("weinstein_screen")
if _state is None:
    st.info("좌측에서 기준일·유니버스를 정하고 **Run Screen**. 처음엔 sp500 + 최대 종목 수 100 으로 스모크 테스트 권장 "
            "(첫 실행은 종목당 11년 일봉 fetch, 이후엔 Mongo 캐시).")
    st.stop()

res = _state["res"]
result_df: pd.DataFrame = _state["df"]
st.subheader(f"{_state['as_of']} 기준 — {_state['universe']} · 주봉 {res.week_end.date() if res.week_end is not None else '?'}"
             + (" (진행 중인 주 — 금요일 종가 미확정)" if res.partial_week else ""))

# --- STAGE 0 ---
if res.market_checks:
    lines = [f"{'✅' if c.passed else '❌' if c.passed is False else 'ℹ️'} **{c.code} {c.name}** — {c.value}"
             + (f" · {c.note}" if c.note else "") for c in res.market_checks]
    box = st.success if res.market_pass else st.error
    box(("STAGE 0 시장 환경 — " + ("상승세. 종목을 본다" if res.market_pass else
                                 "**하락세 → 매수 금지 규칙 1. 아래 후보는 관찰·대조군용**")) + "\n\n" + "\n\n".join(lines))
    st.caption("0-3~0-6 시장 폭은 선택한 유니버스로 계산(책은 NYSE 전체·S&P 500). 0-8 P/D 는 외부 배당 데이터라 생략. "
               "체크리스트 원칙: 0-1 실패면 나머지는 보지 않는다 — 단 이 페이지는 검증용이라 후보를 숨기지 않는다.")
if res.errors:
    st.warning(f"조회 실패 {len(res.errors)}건: " + " · ".join(res.errors[:8]) + (" …" if len(res.errors) > 8 else ""))

cols = st.columns(len(res.funnel))
for col, (name, n) in zip(cols, res.funnel.items()):
    col.metric(name, f"{n:,}")

if result_df.empty:
    st.warning("셋업 없음 — 이번 주 저항 돌파 종목이 없거나(돌파 탐색 범위를 늘려보라) 유니버스가 작다.")
    st.stop()

view = result_df
if min_grade != "전부":
    order = {"A+": 0, "A": 1, "B": 2, "B-": 3}
    view = view[view["등급"].map(lambda g: order.get(g, 99)) <= order[min_grade]]

pct_cols = {c: st.column_config.NumberColumn(format="percent") for c in
            ("돌파폭", "베이스폭", "수익률", "MFE", "MAE", "+13주", "+26주") if c in view.columns}
st.dataframe(
    view, use_container_width=True, hide_index=True,
    column_config={
        **pct_cols,
        "종가": st.column_config.NumberColumn(format="dollar"),
        "저항(돌파가)": st.column_config.NumberColumn(format="dollar"),
        "지지": st.column_config.NumberColumn(format="dollar"),
        "거래량배수": st.column_config.NumberColumn(format="%.2fx", help="돌파 주 거래량 ÷ 직전 4주 평균 (책: ≥2배, A+ ≥3배)"),
        "RS": st.column_config.NumberColumn(format="%.1f", help="맨스필드 RS vs SPY — 0 위 = 플러스 영역"),
        "등급": st.column_config.TextColumn(help="시장·업종 포함 최종. 탈락(코드) = 필수 항목 실패"),
        "종목등급": st.column_config.TextColumn(help="STAGE 0·1 을 뺀 종목 자체 등급 — 시장이 막힌 날의 대조군 분석용"),
    },
)
st.caption("매수 규칙(§7): 저항 +0.5% buy-stop 으로 **계획 투자액의 절반**, 돌파가 근처 풀백(거래량 급감)에서 나머지 절반. "
           "돌파 거래량 빈약이면 첫 반등에 매도, 돌파가 아래로 떨어지면 즉시 처분. 손절 = 돌파가 −5%(트레이더) 또는 지지선 아래(투자가).")

# --- 등급별 성과 (검증의 핵심) ---
if _state["show_forward"] and "수익률" in result_df.columns:
    st.divider()
    st.subheader("등급별 사후 성과 — 책의 등급 체계가 변별력이 있는가 (§E.3 대조군)")
    recs = []
    for c in res.candidates:
        o = _state["outcomes"].get(c.ticker)
        if o is None:
            continue
        recs.append({"grade": c.grade, "stock_grade": c.stock_grade, "entry_type": c.entry_type, "entered": o.entered,
                     "ret": o.ret, "weeks": o.weeks, "reason": o.reason, "mfe": o.mfe or 0.0, "mae": o.mae or 0.0,
                     "checks": {k.code: k.passed for k in c.checks}})
    for key, title in (("grade", "최종 등급(시장·업종 포함) — 탈락은 한 묶음"), ("stock_grade", "종목 자체 등급 (상세)"),
                       ("entry_type", "진입 유형 (1→2 vs 2단계 지속)")):
        gs = grade_stats(recs, key)
        if gs.empty:
            continue
        st.markdown(f"**{title}**")
        st.dataframe(gs, use_container_width=True, hide_index=True,
                     column_config={k: st.column_config.NumberColumn(format="percent") for k in ("승률", "평균", "중앙값", "평균 MFE", "평균 MAE")})
    cs = check_stats(recs)
    if not cs.empty:
        st.markdown("**항목별 예측력 — 그 항목을 통과한 거래 vs 미통과 거래** (차이 = 통과 평균 − 미통과 평균)")
        st.dataframe(cs, use_container_width=True, hide_index=True,
                     column_config={k: st.column_config.NumberColumn(format="percent") for k in ("통과 평균", "통과 승률", "미통과 평균", "미통과 승률", "차이")})
    st.caption("수익률 = 돌파 주 다음 거래일 시가(+0.5%) 진입 → 청산 사유 발생 주 다음 거래일 시가(−0.5%). "
               "청산 없이 52주 지나면 그 주 종가(time), 데이터가 끝나면 마지막 종가(open). 50/50 분할은 적용하지 않고 풀백 조건 발생 여부만 기록.")

# --- 종목별 상세 ---
st.divider()
st.subheader("종목별 상세 — 주봉·30주 MA·베이스·RS + 체크리스트 전 항목")
by_t = {c.ticker: c for c in res.candidates}
sel = st.selectbox("종목 선택", list(view["티커"]))
c = by_t[sel]
st.plotly_chart(_chart(c), use_container_width=True)
st.caption(f"단계 이력(최근): {_stage_strip(c)} · 베이스 {c.base.start.date()} ~ {c.base.end.date()} "
           f"({c.base.weeks}주, 폭 {c.base.width:.0%}, 상단 터치 {c.base.touches}회, MA 교차 {c.base.ma_crosses}회) · "
           f"스윙 목표가(7-8) ≈ {c.resistance + (c.resistance - c.support):,.2f} (저항 + 베이스 폭)")
st.dataframe(checks_to_frame(c.checks), use_container_width=True, hide_index=True,
             column_config={"code": st.column_config.TextColumn(width="small"), "판정": st.column_config.TextColumn(width="small"),
                            "값": st.column_config.TextColumn(width="large")})
o = _state["outcomes"].get(sel) if _state["show_forward"] else None
if o is not None and o.entered:
    st.info(f"사후: {o.entry_date} {o.entry_px:,.2f} 진입 → {o.exit_date} {o.exit_px:,.2f} ({o.reason}) · "
            f"**{o.ret:+.1%}** / {o.weeks}주 · MFE {o.mfe:+.0%} MAE {o.mae:+.0%} · 풀백 2차 조건 {'발생' if o.pullback_fill else '없음'}"
            + (f" · 돌파 후 2주 거래량 {o.vol_2w_ratio:.2f}x" if o.vol_2w_ratio is not None else ""))

st.markdown(f"**수동 체크 (문서 §B) — {sel} @ {_state['as_of']}**")
st.caption("코드가 후보를 뽑되, 아래는 차트를 열어 기준일 오른쪽을 가리고 직접 본다. 책의 방법론은 애초에 주봉 차트를 손으로 읽는 것.")
ctx = f"{sel}_{_state['as_of']}"
done = 0
for key, label in _MANUAL_ITEMS:
    done += int(st.checkbox(label, key=f"wmc_{ctx}_{key}"))
st.progress(done / len(_MANUAL_ITEMS), text=f"{done}/{len(_MANUAL_ITEMS)} 확인")

st.download_button("결과 CSV 다운로드", result_df.to_csv(index=False).encode("utf-8-sig"),
                   file_name=f"weinstein_screen_{_state['as_of']}.csv", mime="text/csv")
