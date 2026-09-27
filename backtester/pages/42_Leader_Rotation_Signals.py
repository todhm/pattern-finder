import json
import os
from datetime import date, datetime

import pandas as pd
import streamlit as st

from strategy.adapters.leader_rotation_strategy import scan_leaders
from strategy.domain.models import LeaderRotationConfig, TossFeeSchedule

st.set_page_config(page_title="리더 로테이션 실행", layout="wide")
st.title("리더 로테이션 — 오늘의 매매 지시서")
st.caption(
    "41번 백테스트의 검증 설정(12-1 모멘텀 · 대형주 · 월간)을 **오늘 "
    "시점에 적용하면 뭘 사고 뭘 팔아야 하는지** 명시하는 실행 페이지. "
    "월 1회: ① 데이터 갱신 → ② 스캔 → ③ 지시서대로 매매 → ④ 저장."
)

CLOSE_PATH = "sweep_results/leader_rotation_close.parquet"
DVOL_PATH = "sweep_results/leader_rotation_dollarvol.parquet"
PORTFOLIO_PATH = "watchlists/leader_portfolio.json"


@st.cache_data(show_spinner="유니버스 로딩 중...")
def load_universe():
    close = pd.read_parquet(CLOSE_PATH)
    dvol = pd.read_parquet(DVOL_PATH)
    if close.index.tz is not None:
        close.index = close.index.tz_localize(None)
        dvol.index = dvol.index.tz_localize(None)
    return close, dvol


@st.cache_data(show_spinner=False)
def load_market(end: date) -> pd.Series:
    from data.adapters.eodhd_adapter import EODHDAdapter

    s = EODHDAdapter().fetch_ohlcv("QQQ", date(2017, 1, 1), end, "1d")["Close"]
    s = s.dropna()
    if s.index.tz is not None:
        s.index = s.index.tz_localize(None)
    s.index = s.index.normalize()
    return s


def load_portfolio() -> dict:
    if os.path.exists(PORTFOLIO_PATH):
        with open(PORTFOLIO_PATH) as f:
            return json.load(f)
    return {"history": []}


def save_portfolio(book: dict) -> None:
    os.makedirs(os.path.dirname(PORTFOLIO_PATH), exist_ok=True)
    with open(PORTFOLIO_PATH, "w") as f:
        json.dump(book, f, indent=1, ensure_ascii=False)


if not (os.path.exists(CLOSE_PATH) and os.path.exists(DVOL_PATH)):
    st.error(
        "유니버스 데이터가 없어. 먼저: "
        "`docker compose exec backtester python fetch_leader_universe.py`"
    )
    st.stop()

close, dvol = load_universe()
last_bar = close.index[-1].date()
staleness = (date.today() - last_bar).days
if staleness > 7:
    st.error(
        f"⚠️ 데이터가 {staleness}일 묵었다 (마지막 봉 {last_bar}). 갱신 후 "
        "스캔할 것:\n\n```\nrm -rf backtester/sweep_results/leader_parts\n"
        "docker compose exec backtester python fetch_leader_universe.py\n```"
        "\n(약 10분. 리밸런스가 월 1회라 월초 갱신이면 충분)"
    )
else:
    st.success(f"데이터 최신: 마지막 봉 {last_bar} ({staleness}일 전)")

with st.sidebar:
    st.header("규칙 (41번 검증 설정)")
    top_n = st.number_input("보유 종목 수", 1, 30, 8)
    lookback = st.selectbox("모멘텀 기간", [126, 252], index=1)
    skip = st.selectbox("최근 제외", [0, 21], index=1)
    min_dv = st.number_input("최소 거래대금 ($백만)", 50, 1000, 200) * 1e6
    market_filter = st.checkbox("시장 필터 (QQQ<200일선 → 현금)", value=True)
    st.divider()
    capital = st.number_input(
        "총 투입 자본 (만원)", 100, 1_000_000, 10_000, step=1000
    ) * 10_000
    tqqq_w = st.slider(
        "TQQQ 혼합 비중 (%)", 0, 70, 30,
        help="41번 결론: TQQQ 30% + 로테이션 70% 혼합이 존버보다 "
        "금액·위험효율 모두 우위 (Calmar 0.80).",
    ) / 100.0
    run_btn = st.button("오늘 기준 스캔", type="primary", use_container_width=True)

if run_btn:
    cfg = LeaderRotationConfig(
        start_date=date(2018, 11, 1), end_date=last_bar,
        lookback_days=int(lookback), skip_days=int(skip), top_n=int(top_n),
        min_dollar_volume=float(min_dv), market_filter=market_filter,
        fee_schedule=TossFeeSchedule(),
    )
    market = load_market(date.today()) if market_filter else None
    st.session_state["leader_scan"] = scan_leaders(close, dvol, cfg, market=market)
    st.session_state["leader_scan_params"] = dict(capital=capital, tqqq_w=tqqq_w)

snap = st.session_state.get("leader_scan")
if snap is None:
    st.info("좌측에서 **오늘 기준 스캔**을 눌러줘.")
    st.stop()
params = st.session_state.get("leader_scan_params", {})
capital = params.get("capital", capital)
tqqq_w = params.get("tqqq_w", tqqq_w)

book = load_portfolio()
held = set(book["history"][-1]["picks"]) if book["history"] else set()
picks = [r["symbol"] for r in snap["rows"]]

st.header(f"스캔 결과 — {snap['date']} 기준")
if snap["regime"] == "risk_off":
    st.error(
        "🔴 **위험회피 레짐** (QQQ가 200일선 아래). 규칙: 로테이션 슬리브는 "
        "**전량 현금**. 보유 중이면 아래 청산 목록대로 매도."
    )
else:
    st.success("🟢 진입 레짐 — 아래 지시서대로 리밸런스.")

sell_list = sorted(held - set(picks)) if snap["regime"] == "risk_on" else sorted(held)
buy_list = [s for s in picks if s not in held] if snap["regime"] == "risk_on" else []
keep_list = [s for s in picks if s in held] if snap["regime"] == "risk_on" else []

c1, c2, c3 = st.columns(3)
c1.metric("🔴 청산", f"{len(sell_list)}종목",
          ", ".join(sell_list) if sell_list else "없음", delta_color="off")
c2.metric("🟢 신규 매수", f"{len(buy_list)}종목",
          ", ".join(buy_list) if buy_list else "없음", delta_color="off")
c3.metric("⚪ 유지", f"{len(keep_list)}종목",
          ", ".join(keep_list) if keep_list else "없음", delta_color="off")

if snap["regime"] == "risk_on":
    rot_capital = capital * (1.0 - tqqq_w)
    per_slot = rot_capital / max(int(top_n), 1)
    st.subheader("오늘의 리더 랭킹 + 배분")
    rows = [
        {
            "순위": i + 1,
            "티커": r["symbol"],
            "12-1 모멘텀": r["momentum"],
            "현재가($)": r["price"],
            "20일 거래대금($M)": r["dollar_vol"] / 1e6,
            "52주 고점 대비": r["pct_from_high"],
            "배분 금액(원)": per_slot,
            "상태": "유지" if r["symbol"] in held else "신규 매수",
        }
        for i, r in enumerate(snap["rows"])
    ]
    st.dataframe(
        rows, use_container_width=True, hide_index=True,
        column_config={
            "12-1 모멘텀": st.column_config.NumberColumn(format="percent"),
            "현재가($)": st.column_config.NumberColumn(format="%.2f"),
            "20일 거래대금($M)": st.column_config.NumberColumn(format="localized"),
            "52주 고점 대비": st.column_config.NumberColumn(format="percent"),
            "배분 금액(원)": st.column_config.NumberColumn(format="localized"),
        },
    )
    if tqqq_w > 0:
        st.info(
            f"**TQQQ 슬리브**: {capital * tqqq_w:,.0f}원 "
            f"({tqqq_w:.0%}) — 로테이션과 함께 월 1회 이 비율로 원위치."
        )

if st.button("이 구성으로 리밸런스 확정 (저장)", use_container_width=True):
    book["history"].append({
        "date": str(snap["date"]),
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "regime": snap["regime"],
        "picks": picks if snap["regime"] == "risk_on" else [],
    })
    save_portfolio(book)
    st.success("저장됨 — 다음 스캔부터 이 구성 대비 차이를 보여준다.")

if book["history"]:
    last_rb = book["history"][-1]
    elapsed = (date.today() - date.fromisoformat(last_rb["date"])).days
    st.caption(
        f"마지막 확정 리밸런스: {last_rb['date']} ({elapsed}일 경과 · "
        f"{last_rb['regime']}) — 규칙은 **약 한 달(21거래일)마다** 재랭킹. "
        f"{'⏰ 리밸런스 시점이 지났다!' if elapsed >= 30 else ''}"
    )
    with st.expander(f"리밸런스 이력 ({len(book['history'])}회)"):
        st.dataframe(
            [
                {"날짜": h["date"], "레짐": h["regime"],
                 "구성": ", ".join(h["picks"]) if h["picks"] else "(현금)"}
                for h in reversed(book["history"])
            ],
            use_container_width=True, hide_index=True,
        )

st.markdown(
    """
---
**월간 운용 루틴** (기회를 놓치지 않는 법):

1. **매월 첫 주** 데이터 갱신 → 이 페이지에서 스캔
2. 지시서(청산/신규/유지)대로 매매 — 종목당 배분 금액 표 그대로
3. **리밸런스 확정 저장** — 다음 달 스캔이 이 구성과의 차이를 계산
4. 월중에는 아무것도 안 한다 (41번 검증: 빠른 손절선은 오히려 손해).
   단, QQQ가 200일선을 깨면 레짐이 🔴로 바뀌니 그때만 예외적으로 스캔
"""
)
