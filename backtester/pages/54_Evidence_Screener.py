"""근거중심매매 스크리너 — 유니버스 전 종목에 체크리스트 엔진을 돌려 매수 후보만 추린다.

기준일까지의 일봉으로 주봉·일봉 5대 매매 근거를 자동 판정하고, 주봉 '예' 개수·가중치·4분면 조건을 통과한
종목만 표로 보여 준다. 후보를 고르면 53 체크리스트 페이지로 넘어가 사람이 채널·피보나치를 확정한다.

속도: ① 데이터는 기본이 **yfinance 일괄 다운로드**(200종목/요청, 종목당 0.05초)이고, 정확한 캐시 스택(AV→yfinance→
EODHD→Massive, 종목당 0.5~1초)은 옵션. ② 스캔 결과는 세션에 남고 **통과 조건은 다시 돌리지 않고 즉시 재적용**된다.
③ 같은 기준일·파라미터의 스캔은 6시간 캐시. 엔진: strategy/adapters/evidence_screen.py
"""

from __future__ import annotations

import time
from datetime import date, timedelta

import pandas as pd
import streamlit as st

from data.adapters.cache_frontier_fill import fill_recent_days
from data.adapters.composed_market_data import build_daily_cache_filler, build_default_market_data
from data.adapters.wikipedia_universe import default_universe_provider
from data.adapters.yfinance_bulk import download_many
from strategy.adapters.evidence_checklist import EvidenceParams
from strategy.adapters.evidence_screen import ScreenFilter, apply_filter, run_screen

st.set_page_config(page_title="근거중심매매 스크리너", layout="wide")
st.title("근거중심매매 스크리너 — 여러 종목 중 매수 후보만")
st.caption("유니버스 전 종목에 53 페이지의 자동 판정을 돌린다. 주봉 5대 매매 근거 '예' 개수가 기준 이상인 종목만 남기고, "
           "채널·피보나치는 근사치이므로 **후보**일 뿐이다. 확정은 종목을 골라 체크리스트 페이지에서 한다.")

with st.sidebar:
    st.header("기준일 / 유니버스")
    as_of = st.date_input("기준일", value=date.today() - timedelta(days=1), max_value=date.today())
    universe_choice = st.selectbox("유니버스", ["nasdaq100", "sp500", "sp500+nasdaq100", "nasdaq_full", "nasdaq_full@기준일", "직접 입력"],
                                   help="미국 종목 기본. 국내는 '직접 입력'에 005930.KS 처럼. "
                                        "**nasdaq_full** = 나스닥 전 종목(오늘 명단, 약 2,200개), **@기준일** = 그날 실제 상장돼 있던 종목(생존 편향 없음).")
    custom = st.text_area("직접 입력 (쉼표·줄바꿈)", value="AAPL, NVDA, MSFT, AMZN, GOOGL, META, TSLA, AVGO, QQQ, GC=F, CL=F, BTC-USD",
                          height=80, disabled=universe_choice != "직접 입력")
    max_tickers = st.number_input("최대 종목 수 (0=전체)", value=0, min_value=0, step=100)
    source = st.radio("데이터 소스", ["yfinance 일괄 (빠름)", "캐시 스택 (정확·느림)"], index=0,
                      help="일괄: 200종목씩 한 요청, 나스닥 전 종목 약 2~3분. 캐시에 기록되지 않음. "
                           "캐시 스택: 종목당 0.5~1초(처음), 폐지 종목·폴백 소스 지원, 이후 Mongo 캐시로 빠름.")
    st.header("작도 파라미터 ([가정])")
    pivot_lr = st.slider("스윙 피벗 좌우 봉 수 (일봉)", 3, 30, 10)
    years = st.slider("조회 기간 (년)", 2, 10, 5, help="주봉 200MA 에 약 4년 필요")
    workers = st.slider("동시 요청 수 (캐시 스택일 때)", 1, 4, 2)
    run_btn = st.button("스크리너 실행", type="primary", use_container_width=True)
    st.header("통과 조건 (스캔 없이 즉시 재적용)")
    min_yes = st.slider("주봉 '예' 최소 개수", 1, 5, 3, help="책: 4 이상 베팅. 3 은 '근거 쌓는 중' 후보까지.")
    min_score = st.slider("자동 가중치 최소", 0.0, 5.0, 0.0, 0.25)
    excl_worst = st.checkbox("하락채널+정배열(최악) 제외", value=True)
    above20 = st.checkbox("주봉 20MA 위만", value=False)
    min_price = st.number_input("최소 주가 (종가)", value=5.0, min_value=0.0, step=1.0, help="동전주·SPAC 제외. 0 = 무시")

P = EvidenceParams(pivot_lr=pivot_lr)


def _tickers() -> list[str]:
    if universe_choice == "직접 입력":
        raw = custom.replace("\n", ",").split(",")
        return [t.strip().upper() for t in raw if t.strip()]
    prov = default_universe_provider()
    if universe_choice == "nasdaq_full@기준일":
        return prov.get_tickers(f"nasdaq_full@{as_of.isoformat()}")
    if universe_choice == "sp500+nasdaq100":
        return list(dict.fromkeys(prov.get_tickers("sp500") + prov.get_tickers("nasdaq100")))
    return prov.get_tickers(universe_choice)


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def _scan_cached(tickers: tuple[str, ...], as_of_: date, pivot: int, years_: int, src: str, workers_: int) -> tuple[pd.DataFrame, str]:
    """같은 입력이면 다시 돌리지 않는다. 진행 막대는 캐시 미스 때만 보인다."""
    p = EvidenceParams(pivot_lr=pivot)
    note = []
    t0 = time.time()
    prog = st.progress(0.0, text="시작…")
    if src.startswith("yfinance"):
        prices = download_many(list(tickers), as_of_ - timedelta(days=365 * years_), as_of_,
                               progress=lambda i, n: prog.progress(0.6 * i / n, text=f"일괄 다운로드 {i}/{n}"))
        note.append(f"일괄 다운로드 {len(prices)}/{len(tickers)}종목 {time.time() - t0:.0f}초")
        t1 = time.time()
        _, all_df = run_screen(None, list(tickers), as_of_, p, ScreenFilter(years=years_), max_workers=4,
                               progress=lambda i, n, t: prog.progress(0.6 + 0.4 * i / n, text=f"판정 {i}/{n} — {t}"),
                               prices=prices)
        note.append(f"판정 {time.time() - t1:.0f}초")
    else:
        if any(t.startswith("nasdaq_full") for t in (src,)) or len(tickers) > 500:
            try:
                _filler = build_daily_cache_filler()
                if _filler is not None:
                    _rep = fill_recent_days(_filler[0], _filler[1], list(tickers), as_of_, marker_caches=_filler[2])
                    note.append("캐시 채우기: " + _rep.summary())
            except Exception as e:  # noqa: BLE001
                note.append(f"캐시 채우기 실패 ({type(e).__name__})")
        md = build_default_market_data()
        _, all_df = run_screen(md, list(tickers), as_of_, p, ScreenFilter(years=years_), max_workers=workers_,
                               progress=lambda i, n, t: prog.progress(i / n, text=f"{i}/{n} — {t}"))
        note.append(f"캐시 스택 조회+판정 {time.time() - t0:.0f}초")
    prog.empty()
    return all_df, " · ".join(note)


if run_btn:
    try:
        tickers = _tickers()
    except Exception as e:  # noqa: BLE001
        st.error(f"유니버스 로딩 실패: {e}")
        st.stop()
    if max_tickers:
        tickers = tickers[: int(max_tickers)]
    all_df, note = _scan_cached(tuple(tickers), as_of, pivot_lr, years, source, workers)
    st.session_state["ev_screen"] = {"as_of": as_of, "universe": universe_choice, "all": all_df, "note": note, "n": len(tickers)}

state = st.session_state.get("ev_screen")
if not state:
    st.info("왼쪽에서 기준일·유니버스를 정하고 **스크리너 실행**. 통과 조건은 실행 뒤에도 바꾸면 즉시 반영된다.")
    st.stop()

all_df: pd.DataFrame = state["all"]
F = ScreenFilter(min_weekly_yes=min_yes, min_score=min_score, exclude_worst_quadrant=excl_worst, require_above_ema20_weekly=above20, years=years)
hit = apply_filter(all_df, F)
if min_price > 0 and len(hit) and "종가" in hit:
    hit = hit[hit["종가"] >= float(min_price)].reset_index(drop=True)
for _c in ("주봉 예", "일봉 예"):
    if _c in hit:
        hit[_c] = hit[_c].astype(int)
n_err = int(all_df["오류"].notna().sum()) if "오류" in all_df else 0

st.subheader(f"{state['as_of']} 기준 · {state['universe']} — 후보 {len(hit)} / 전체 {len(all_df)} (오류 {n_err})")
st.caption(state["note"] + f" · 통과 조건: 주봉 예 ≥ {min_yes} · 가중치 ≥ {min_score} · "
           f"{'최악 4분면 제외' if excl_worst else '4분면 무관'} · {'주봉 20MA 위만' if above20 else '20MA 무관'} · 종가 ≥ {min_price:g}")
if "주봉 예" in all_df:
    dist = all_df["주봉 예"].dropna().astype(int).value_counts().sort_index(ascending=False)
    st.caption("주봉 예 분포(전체): " + " · ".join(f"{k}개 {v}" for k, v in dist.items()))

if len(hit):
    cols = ["티커", "주봉 예", "일봉 예", "자동 가중치", "①", "②", "③", "④", "⑤", "4분면(주봉)", "배열(주봉)", "주봉 MA 접촉",
            "피보나치(주봉)", "피보나치(일봉)", "캔들(주봉)", "캔들(일봉)", "RSI(주봉)", "다이버전스(주봉)", "매물대", "종가"]
    st.dataframe(hit[[c for c in cols if c in hit.columns]], use_container_width=True, hide_index=True, height=min(600, 60 + 35 * len(hit)))
    st.caption("① 채널 중단·하단 접촉(근사) ② 주봉 주요 MA ③ 캔들 상승 신호(20MA 위 마감) ④ 유리한 다이버전스·과매도 ⑤ 피보나치 주요 구간(근사). "
               "주봉 4 이상 = 베팅 후보, 3 = 근거 쌓는 중.")
    st.subheader("후보를 골라 체크리스트에서 확정")
    pick = st.selectbox("종목", hit["티커"].tolist())
    if st.button("53 체크리스트 페이지에서 열기 →"):
        st.session_state["ev_prefill_ticker"] = pick
        st.session_state["ev_prefill_date"] = state["as_of"]
        st.switch_page("pages/53_Evidence_Checklist.py")
else:
    st.warning("통과 종목 없음. 왼쪽 통과 조건을 낮추면(주봉 예 2) 스캔 없이 바로 반영된다.")

with st.expander(f"전체 결과 ({len(all_df)}종목, 오류 포함)"):
    st.dataframe(all_df, use_container_width=True, hide_index=True)
st.download_button("후보 CSV", hit.to_csv(index=False).encode("utf-8-sig"), file_name=f"evidence_screen_{state['as_of']}.csv", mime="text/csv")
