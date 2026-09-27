import json
import os
from datetime import date

import pandas as pd
import streamlit as st

from data.adapters.cached_market_data import CachedMarketDataAdapter
from data.adapters.yfinance_adapter import YFinanceAdapter
from pages._shared.coins import RECENT_COINS

st.set_page_config(page_title="신규 상장 감시", layout="wide")
st.title("신규 상장 감시 — '새 판'에 규율로 들어가기")
st.caption(
    "새 시장(신규 상장 코인)의 볼록성은 먹되 참사는 규율로 차단하는 "
    "감시탑. 검증된 규칙(2026-08, 22종): **상장 후 첫 20일 관망 → "
    "종가가 20일선 위 + 상장 후 전고점 돌파 시 진입 → 20일선 하향 "
    "이탈 시 전량 청산**. 무차별 매수는 중앙값 -72%·참사 14/22건, "
    "이 규칙은 **중앙값 0%·참사 0건·+100% 이상 6건** (AERO +1,090%). "
    "매일 이 페이지를 열면 각 코인의 현재 단계와 오늘의 신호가 보인다."
)
st.caption(
    "⚠️ 유동성 상위 코인만 실행할 것 — 소형 코인은 슬리피지가 엣지를 "
    "삼킨다 (35번 페이지 경고 참조). 신호는 참고용이며 자동 매매가 아니다."
)

WATCHLIST_PATH = "watchlists/new_listings.json"
WATCH_DAYS = 20
MA_DAYS = 20


def load_watchlist() -> list[str]:
    if os.path.exists(WATCHLIST_PATH):
        with open(WATCHLIST_PATH) as f:
            return json.load(f).get("symbols", [])
    return sorted(RECENT_COINS)


def save_watchlist(symbols: list[str]) -> None:
    os.makedirs(os.path.dirname(WATCHLIST_PATH), exist_ok=True)
    with open(WATCHLIST_PATH, "w") as f:
        json.dump({"symbols": sorted(set(symbols))}, f, indent=1)


def analyze(symbol: str, md) -> dict | None:
    """규칙 B 상태 기계 — 현재 단계와 오늘의 신호."""
    try:
        df = md.fetch_ohlcv(symbol, date(2023, 1, 1), date.today())
    except Exception:
        return None
    if df is None or df.empty:
        return None
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index = df.index.normalize()
    c = df["Close"]
    c = c[c > 0]
    if len(c) < 5:
        return None
    listed = c.index[0].date()
    days = len(c)
    px = float(c.iloc[-1])

    if days < WATCH_DAYS:
        return {
            "symbol": symbol, "listed": listed, "days": days, "price": px,
            "status": f"⏳ 관망 ({days}/{WATCH_DAYS}일)", "signal_rank": 3,
            "vs_ma": None, "vs_high": None, "rule_return": None,
            "position": "", "last_event": "",
        }

    ma = c.rolling(MA_DAYS).mean()
    hi = c.cummax().shift(1)
    pos = False
    entry = 0.0
    ret = 1.0
    last_event = ""
    cost = 0.0005 + 0.0010  # 수수료 + 슬리피지 (편도)
    for i in range(WATCH_DAYS, len(c)):
        p = float(c.iloc[i])
        if not pos:
            if p > float(ma.iloc[i]) and p >= float(hi.iloc[i]):
                pos = True
                entry = p * (1 + cost)
                last_event = f"진입 {c.index[i].date()}"
        else:
            if p < float(ma.iloc[i]):
                ret *= p * (1 - cost) / entry
                pos = False
                last_event = f"청산 {c.index[i].date()}"
    open_ret = px * (1 - cost) / entry - 1 if pos else None
    total_ret = ret * (px * (1 - cost) / entry) - 1 if pos else ret - 1

    ma_now = float(ma.iloc[-1])
    hi_now = float(hi.iloc[-1])
    vs_ma = px / ma_now - 1 if ma_now > 0 else None
    vs_high = px / hi_now - 1 if hi_now > 0 else None

    # 오늘의 신호 판정
    if pos and px < ma_now:
        status, rank = "🔴 청산 신호 (오늘)", 0
    elif not pos and px > ma_now and px >= hi_now:
        status, rank = "🔵 진입 신호 (오늘)", 0
    elif pos:
        status, rank = f"🟢 보유 중 ({open_ret:+.0%})", 1
    else:
        status, rank = "대기 (돌파 전)", 2
    return {
        "symbol": symbol, "listed": listed, "days": days, "price": px,
        "status": status, "signal_rank": rank,
        "vs_ma": vs_ma, "vs_high": vs_high, "rule_return": total_ret,
        "position": "보유" if pos else "", "last_event": last_event,
    }


with st.sidebar:
    st.header("워치리스트")
    current = load_watchlist()
    selected = st.multiselect(
        "감시 중인 코인",
        sorted(set(current) | set(RECENT_COINS)),
        default=current,
        accept_new_options=True,
        help="yfinance 심볼 (예: HYPE32196-USD). 새 심볼 타이핑으로 추가 "
        "— 저장 버튼을 눌러야 유지된다.",
    )
    if st.button("워치리스트 저장", use_container_width=True):
        save_watchlist([str(s).strip().upper() for s in selected])
        st.success(f"{len(selected)}개 저장됨")

    st.header("신규 코인 탐색")
    if st.button("코인게코에서 신규 상장 탐색", use_container_width=True):
        import httpx

        found, tried = [], []
        try:
            resp = httpx.get(
                "https://api.coingecko.com/api/v3/coins/list/new", timeout=20
            )
            resp.raise_for_status()
            md_probe = CachedMarketDataAdapter(YFinanceAdapter())
            for coin in resp.json()[:20]:
                sym = str(coin.get("symbol", "")).upper()
                if not sym or len(sym) > 10:
                    continue
                cand = f"{sym}-USD"
                tried.append(cand)
                try:
                    df = md_probe.fetch_ohlcv(cand, date(2025, 1, 1), date.today())
                    if df is not None and len(df) >= 3:
                        found.append(cand)
                except Exception:
                    continue
        except Exception as e:
            st.error(f"코인게코 조회 실패: {e}")
        if found:
            save_watchlist(list(selected) + found)
            st.success(f"yfinance 데이터 확인된 신규: {found} → 워치리스트에 추가됨. 새로고침.")
        else:
            st.info(
                f"탐색 {len(tried)}개 중 yfinance 데이터가 있는 신규 코인 "
                "없음 (상장 초기엔 yfinance 반영이 늦다 — 며칠 후 재시도)."
            )

    run_btn = st.button("감시 실행 / 새로고침", type="primary", use_container_width=True)

if run_btn:
    md = CachedMarketDataAdapter(YFinanceAdapter())
    rows = []
    progress = st.progress(0.0, text="코인 분석 중...")
    symbols = [str(s).strip().upper() for s in selected]
    for i, sym in enumerate(symbols):
        r = analyze(sym, md)
        if r is not None:
            rows.append(r)
        progress.progress((i + 1) / max(len(symbols), 1), text=f"코인 분석 중 ({i + 1}/{len(symbols)})...")
    progress.empty()
    st.session_state["listing_watch_rows"] = rows

rows = st.session_state.get("listing_watch_rows")
if not rows:
    st.info("좌측에서 **감시 실행**을 눌러줘.")
    st.stop()

signals = [r for r in rows if r["signal_rank"] == 0]
holding = [r for r in rows if r["signal_rank"] == 1]
m1, m2, m3 = st.columns(3)
m1.metric("🔔 오늘의 신호", f"{len(signals)}건")
m2.metric("🟢 보유 중", f"{len(holding)}건")
m3.metric("감시 코인", f"{len(rows)}개")
if signals:
    for r in signals:
        st.warning(f"**{r['symbol']}** — {r['status']}")

st.subheader("감시 현황")
table = [
    {
        "코인": r["symbol"],
        "상태": r["status"],
        "상장(데이터 시작)": r["listed"],
        "경과일": r["days"],
        "현재가": r["price"],
        "20일선 대비": r["vs_ma"],
        "전고점 대비": r["vs_high"],
        "규칙 B 누적수익": r["rule_return"],
        "마지막 이벤트": r["last_event"],
    }
    for r in sorted(rows, key=lambda x: (x["signal_rank"], x["symbol"]))
]
st.dataframe(
    table, use_container_width=True, hide_index=True,
    column_config={
        "현재가": st.column_config.NumberColumn(format="%.4g"),
        "20일선 대비": st.column_config.NumberColumn(format="percent"),
        "전고점 대비": st.column_config.NumberColumn(format="percent"),
        "규칙 B 누적수익": st.column_config.NumberColumn(format="percent"),
    },
)
st.caption(
    "**상태 읽는 법** — ⏳관망: 상장 20일 미만(진입 금지) · 대기: 돌파 "
    "조건 미충족 · 🔵진입 신호: 오늘 종가가 20일선 위 + 전고점 돌파 · "
    "🟢보유 중: 진입 후 20일선 위 유지 · 🔴청산 신호: 20일선 하향 이탈. "
    "**규칙 B 누적수익** = 상장 이후 이 규칙을 기계적으로 따랐을 때의 "
    "누적 성적 (수수료+슬리피지 반영)."
)
