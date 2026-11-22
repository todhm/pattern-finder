"""근거중심매매 AI — 오늘 진입할 종목: 주요 지수 구성 종목의 마지막 완성 일봉까지를 AI 가 하나씩 읽고 진입할 만한 종목을 고른다.

판단 기준은 68 페이지(Evidence_Based_AI)와 같은 패턴 원칙(`ai_patterns.PATTERNS["evidence"]`)·같은 자료 묶음(주봉·일봉 그림 + 봉 표).
1차(싼 모델)로 전 종목 → 진입 후보만 2차(상위 모델)로 재판정. 엔진 `strategy/adapters/ai_today_scan.py`.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

from strategy.adapters.ai_chart_judge import Judgement, prepare
from strategy.adapters.ai_patterns import PATTERNS
from strategy.adapters.ai_today_scan import completed_bars, plan, scan_today
from strategy.adapters.claude_chart_judge import MODELS, ClaudeJudge, has_credentials
from visualization.adapters.lightweight_charts import Pane, render as render_lwc

P = PATTERNS["evidence"]
ROOT = Path(__file__).resolve().parent.parent / "sweep_results" / "ai_today"
STEP = 5                                    # 조건부 진입(어느 가격 위 마감·눌림 매수)의 유효 기간 = 5 거래일
EST_IN, EST_OUT = 12000, 1800               # 종목 하나의 대략적인 (입력, 출력) 토큰
# 기본 대상은 개별 종목이 아니라 지수·원자재·레버리지 ETF (2026-11-19 사용자 지정). 인버스 ETF 는 넣지 않는다(롱만 판단).
GROUPS = {
    "지수 ETF": ["QQQ", "SPY", "DIA", "IWM", "SMH", "SOXX"],
    "레버리지 ETF": ["TQQQ", "SOXL", "UPRO", "SPXL", "QLD", "SSO", "TNA", "UDOW", "TECL", "USD"],
    "금·원자재": ["GLD", "SLV", "GDX", "USO", "CPER", "DBC"],
    "채권": ["TLT", "IEF", "TMF", "HYG"],
    "섹터 ETF": ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLU", "XLB", "XLC", "XLRE"],
    "해외 지수 ETF": ["EFA", "EEM", "EWY", "EWJ", "FXI", "INDA"],
    "비트코인 ETF": ["IBIT", "BITO"],
}
STOCK_INDEXES = {"나스닥 100 구성 종목": "nasdaq100", "S&P 500 구성 종목": "sp500"}
DEFAULT_GROUPS = ["지수 ETF", "레버리지 ETF", "금·원자재"]
ETF_NOTE = ("대상은 개별 주식이 아니라 지수·원자재 ETF 입니다. 레버리지 ETF(TQQQ·SOXL·UPRO 등)는 기초 지수의 2~3배로 움직이므로 손절 폭과 목표를 그만큼 넓게 잡고, "
            "긴 횡보에서는 값이 깎인다는 점을 감안하세요. 지수 ETF 는 실적·개별 악재가 없으니 차트 근거만으로 판단합니다.")

st.set_page_config(page_title="근거중심매매 AI — 오늘 진입", layout="wide")
st.title("근거중심매매 AI — 오늘 진입할 종목")
st.caption("**지수·레버리지·금/원자재 ETF**(QQQ·TQQQ·SOXL·GLD 등)를 하나씩, **마지막으로 마감된 일봉까지의 주봉·일봉 차트**를 Claude 에게 보여 주고 『근거중심매매』의 5대 근거로 진입 여부를 판단시킨다. "
           "숫자 조건으로 거르지 않는다 — 전부를 1차 모델이 읽고, 진입이라고 한 것만 2차 모델이 다시 읽는다. 개별 종목(나스닥 100·S&P 500 구성 종목)은 사이드바에서 따로 켤 수 있다. "
           "판단마다 API 비용이 든다. 결과는 매매 권유가 아니라 AI 의 차트 판독 기록이며, 이 저장소의 앞선 블라인드 검증에서는 같은 방식의 선별이 무작위와 차이가 없었다.")

with st.sidebar:
    st.header("대상")
    idx = st.multiselect("묶음", list(GROUPS) + list(STOCK_INDEXES), default=DEFAULT_GROUPS,
                         help="기본은 지수·레버리지·금/원자재 ETF. '구성 종목' 묶음을 켜면 개별 주식 100~500개가 추가된다(비용·시간이 크게 는다)")
    with st.expander("묶음에 든 티커"):
        st.caption(" / ".join(f"**{g}**: {', '.join(GROUPS[g])}" for g in GROUPS))
    extra = st.text_input("추가 티커 (쉼표로 구분, 선택)", value="", placeholder="예: FNGU, NVDL, URA")
    limit_n = st.number_input("최대 종목 수 (0 = 전부)", value=0, min_value=0, step=50, help="시험 삼아 일부만 돌릴 때. 알파벳 순 앞에서부터")
    st.header("모델")
    m1 = st.selectbox("1차 (전 종목)", list(MODELS), index=list(MODELS).index("claude-haiku-5-5"), format_func=lambda k: f"{MODELS[k][0]} (${MODELS[k][1]:g}/${MODELS[k][2]:g})")
    m2 = st.selectbox("2차 (진입 후보만)", ["(2차 없음)"] + list(MODELS), index=1 + list(MODELS).index("claude-opus-5-5"),
                      format_func=lambda k: k if k.startswith("(") else f"{MODELS[k][0]} (${MODELS[k][1]:g}/${MODELS[k][2]:g})")
    effort = st.select_slider("생각 깊이", ["low", "medium", "high", "xhigh"], value="medium")
    max_confirm = st.number_input("2차로 넘길 최대 종목 수", value=15, min_value=1, max_value=100, help="1차가 진입이라고 한 종목 중 확신이 높은 순서로")
    workers = st.slider("동시 호출 수", 1, 8, 4)


@st.cache_data(ttl=3600, show_spinner="지수 구성 종목 조회…")
def _tickers(names: tuple[str, ...]) -> list[str]:
    from data.adapters.wikipedia_universe import default_universe_provider

    prov = default_universe_provider(); out: list[str] = []
    for n in names:
        out += prov.get_tickers(n)
    return sorted(dict.fromkeys(t.replace(".", "-") for t in out))


@st.cache_data(ttl=3 * 3600, show_spinner="일봉 내려받는 중 (9년치, 종목 수에 따라 1~3분)…")
def _prices(tickers: tuple[str, ...], end: date) -> dict[str, pd.DataFrame]:
    from data.adapters.yfinance_bulk import download_many

    px = download_many(list(tickers), end - timedelta(days=365 * 9 + 30), end)
    out = {}
    for s, d in px.items():
        d = d[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])
        if d.index.tz is not None:
            d.index = d.index.tz_localize(None)
        out[s] = completed_bars(d)
    return out


etf_list = [t for g in idx if g in GROUPS for t in GROUPS[g]]
stock_names = tuple(STOCK_INDEXES[g] for g in idx if g in STOCK_INDEXES)
tickers = list(dict.fromkeys(etf_list + [t.strip().upper() for t in extra.split(",") if t.strip()] + (list(_tickers(stock_names)) if stock_names else [])))
if limit_n:
    tickers = tickers[: int(limit_n)]
if not tickers:
    st.info("사이드바에서 지수를 고르거나 티커를 넣어 주세요."); st.stop()

c1 = (EST_IN * MODELS[m1][1] + EST_OUT * MODELS[m1][2]) / 1e6
c2 = 0.0 if m2.startswith("(") else (EST_IN * MODELS[m2][1] + (EST_OUT + 1200) * MODELS[m2][2]) / 1e6
k = st.columns(4)
k[0].metric("대상 종목", f"{len(tickers):,}")
k[1].metric("1차 예상 비용", f"${c1 * len(tickers):.2f}", MODELS[m1][0], delta_color="off")
k[2].metric("2차 예상 비용 (최대)", f"${c2 * int(max_confirm):.2f}", "없음" if m2.startswith("(") else MODELS[m2][0], delta_color="off")
k[3].metric("예상 시간", f"{max(1, round(len(tickers) * (2 + 14 / int(workers)) / 60))}분 안팎")

have_key = has_credentials()
if not have_key:
    st.warning("**ANTHROPIC_API_KEY 가 컨테이너에 없습니다.** `secrets/secret.env` 에 넣고 컨테이너를 다시 만들어 주세요.")
skey = f"today|{'+'.join(idx)}|{extra}|{limit_n}|{m1}|{m2}|{effort}|{max_confirm}"
go = st.button("▶ 오늘 진입할 종목 찾기", type="primary", disabled=not have_key)
if go:
    prog = st.progress(0.0, text="일봉 내려받는 중…")
    try:
        px = _prices(tuple(tickers), date.today())
        mk = lambda m: ClaudeJudge(title=P.title, principles=P.principles, step=STEP, model=m, effort=effort, cache_dir=ROOT / "cache", workers=int(workers),
                                   note="" if stock_names else ETF_NOTE)  # noqa: E731
        j1 = mk(m1); j2 = None if m2.startswith("(") else mk(m2)
        res = scan_today(px, P.chart, j1.ask_entry, j2.ask_entry if j2 else None, max_confirm=int(max_confirm), chunk=int(workers) * 2,
                         progress=lambda f, m: prog.progress(min(1.0, f), text=m))
        res.update({"indexes": idx, "model1": m1, "model2": m2, "effort": effort, "n_tickers": len(px), "run_at": pd.Timestamp.now(tz="UTC").isoformat()})
        ROOT.mkdir(parents=True, exist_ok=True)
        (ROOT / f"scan_{res['as_of']}_{'etf' if not stock_names else '+'.join(stock_names)}_{pd.Timestamp.now():%H%M%S}.json").write_text(json.dumps(res, ensure_ascii=False, default=str))
        st.session_state[skey] = res; st.session_state["today_last"] = res
    except Exception as ex:  # noqa: BLE001
        st.error(f"실패: {type(ex).__name__}: {str(ex)[:400]} — 받은 답은 저장되어 다시 실행하면 이어집니다.")
    prog.empty()

res = st.session_state.get(skey) or st.session_state.get("today_last")
if res is None:
    files = sorted(ROOT.glob("scan_*.json"))
    if files:
        res = json.loads(files[-1].read_text()); st.info(f"가장 최근 저장된 스캔({files[-1].name})을 보여 줍니다. 새로 돌리려면 위 버튼을 누르세요.")
if res is None:
    st.caption("아직 스캔이 없습니다."); st.stop()

# ---------------------------------------------------------------------------
# 결과
# ---------------------------------------------------------------------------
tri, conf = res["triage"], res["confirmed"]
two_stage = not str(res.get("model2", "")).startswith("(")
final = [r for r in conf if r["judgement"]["decision"] == "enter_long"]
st.subheader(f"기준 봉 {res['as_of']} · {len(tri)}종목 판정 · 1차 진입 후보 {res['n_candidates']}개 · " + (f"2차 통과 {len(final)}개" if two_stage else "2차 없음")
             + f" · 비용 ${res['cost_usd']:.2f}")
vc = pd.Series([r["judgement"]["decision"] for r in tri]).value_counts()
st.caption("1차 판정 분포: " + " · ".join(f"{ {'enter_long': '진입', 'wait': '기다림', 'no_trade': '안 함'}[a]} {b}" for a, b in vc.items())
           + (f" · 자료 부족으로 건너뜀 {len(res['skipped'])}" if res.get("skipped") else ""))


def _table(rows: list[dict]) -> pd.DataFrame:
    out = []
    for r in rows:
        j = r["judgement"]; pl = plan(j, r["close"])
        out.append({"종목": r["symbol"], "확신": j["confidence"], "자리": j["setup"], "종가": r["close"], "진입": pl["entry_how"], "손절": j.get("stop"), "손절 %": pl["stop_pct"],
                    "목표1": j.get("target1"), "목표1 %": pl["t1_pct"], "목표2": j.get("target2"), "손익비": pl["rr"], "요약": j["summary"]})
    for c in ("손절", "손절 %", "목표1", "목표1 %", "목표2", "손익비"):
        if out:
            for o in out:
                o[c] = float("nan") if o[c] is None else o[c]
    return pd.DataFrame(out).sort_values("확신", ascending=False) if out else pd.DataFrame()


fmt = {"종가": "{:.2f}", "손절": "{:.2f}", "손절 %": "{:+.1%}", "목표1": "{:.2f}", "목표1 %": "{:+.1%}", "목표2": "{:.2f}", "손익비": "{:.1f}"}
if final:
    st.markdown("#### ✅ 오늘 진입할 만하다고 본 종목" + (" (1차·2차 모두 진입)" if two_stage else ""))
    tb = _table(final)
    st.dataframe(tb.style.format(fmt, na_rep="—"), width="stretch", hide_index=True, height=min(520, 60 + 35 * len(tb)))
else:
    st.info("오늘은 AI 가 진입할 만하다고 본 종목이 없습니다. '기다리는 것도 매매'입니다.")
rejected = [r for r in conf if r["judgement"]["decision"] != "enter_long"]
if rejected:
    with st.expander(f"1차는 진입이라 했지만 2차가 뒤집은 종목 {len(rejected)}개"):
        st.dataframe(pd.DataFrame([{"종목": r["symbol"], "2차 결정": {"wait": "기다림", "no_trade": "안 함"}[r["judgement"]["decision"]], "2차 이유": r["judgement"]["summary"],
                                    "1차 자리": r["triage"]["setup"]} for r in rejected]), width="stretch", hide_index=True)
not_sent = [r for r in tri if r["judgement"]["decision"] == "enter_long" and r["symbol"] not in {c["symbol"] for c in conf}]
if not_sent:
    with st.expander(f"1차 진입 후보였지만 2차로 넘기지 않은 종목 {len(not_sent)}개 (상한 초과)"):
        st.dataframe(_table(not_sent).style.format(fmt, na_rep="—"), width="stretch", hide_index=True)
waits = [r for r in tri if r["judgement"]["decision"] == "wait"]
if waits:
    with st.expander(f"기다림 {len(waits)}개 — 무엇이 나오면 진입인지"):
        st.dataframe(pd.DataFrame([{"종목": r["symbol"], "확신": r["judgement"]["confidence"], "자리": r["judgement"]["setup"], "기다리는 것": r["judgement"]["wait_for"]}
                                   for r in sorted(waits, key=lambda r: -r["judgement"]["confidence"])]), width="stretch", hide_index=True, height=360)

# 종목 상세 ------------------------------------------------------------------
pool = final + rejected + not_sent
if pool:
    st.markdown("#### 종목 상세")
    sym = st.selectbox("종목", [r["symbol"] for r in pool], format_func=lambda s: next(f"{r['symbol']} · {r['judgement']['setup'][:40]}" for r in pool if r["symbol"] == s))
    r = next(x for x in pool if x["symbol"] == sym)
    j = Judgement(**r["judgement"])
    try:
        d = _prices(tuple([sym, "QQQ"]), date.today()).get(sym)
    except Exception:  # noqa: BLE001
        d = None
    if d is not None and len(d):
        d = d[d.index <= pd.Timestamp(r["as_of"])]
        g = prepare(d, P.chart).iloc[-220:]
        x = pd.DatetimeIndex(g.index)
        lines = [dict(name=f"EMA{n}", data=g[f"ema{n}"], color=c, width=1) for n, c in zip(P.chart.emas, ("#42a5f5", "#ab47bc", "#8d6e63", "#26a69a"))]
        lines += [dict(name=f"SMA{n}", data=g[f"sma{n}"], color="#bdbdbd", width=1, style="dash") for n in P.chart.smas]
        segs = []; col = {"resistance": "#64b5f6", "trend_resistance": "#64b5f6", "support": "#fdd835", "trend_support": "#fdd835"}
        for lv in j.levels[:6]:
            i0 = max(0, len(x) - 1 - int(lv.bars_ago_start or 30))
            y0 = lv.price_start if (lv.kind.startswith("trend") and lv.price_start is not None) else lv.price
            if i0 < len(x) - 1:
                segs.append(dict(points=[(x[i0], y0), (x[-1], lv.price)], color=col[lv.kind], width=2))
        hl = []
        if j.entry_type in ("breakout_close", "limit") and j.entry_trigger:
            hl.append(dict(price=j.entry_trigger, color="#00e676", title="진입 조건", style="dot"))
        for v, t, c in ((j.stop, "손절", "#ff1744"), (j.target1, "목표1", "#26c6da"), (j.target2, "목표2", "#26c6da")):
            if v:
                hl.append(dict(price=v, color=c, title=t, style="dash"))
        vol_color = ["rgba(38,166,154,0.7)" if c >= o else "rgba(239,83,80,0.7)" for o, c in zip(g["Open"], g["Close"])]
        render_lwc([Pane(height=480, title=f"{sym} 일봉", candles=g, lines=lines, segments=segs, hlines=hl),
                    Pane(height=110, title="거래량", hist=[dict(name="거래량", data=g["Volume"], colors=vol_color)]),
                    Pane(height=130, title="RSI(14)", lines=[dict(name="RSI", data=g["rsi"], color="#ba68c8", width=1)],
                         hlines=[dict(price=70, color="#ef5350", style="dot"), dict(price=30, color="#26a69a", style="dot")])], index=x, intraday=False, theme="dark")
    st.info(j.summary)
    a, b = st.columns(2)
    a.markdown(f"**큰 그림(주봉)**  \n{j.market_read}\n\n**차트 구조 — 5대 근거**  \n{j.structure_read}\n\n**이동평균선·RSI**  \n{j.indicator_read}")
    b.markdown(f"**거래량**  \n{j.volume_read}\n\n**틀릴 수 있는 이유**  \n{j.risks}\n\n**계획을 버릴 조건**  \n{j.wait_for}")
    if two_stage and r.get("triage"):
        st.caption(f"1차({MODELS.get(res['model1'], [res['model1']])[0]})의 판단: {r['triage']['setup']} — {r['triage']['summary']}")

# 지난 스캔 --------------------------------------------------------------------
files = sorted(ROOT.glob("scan_*.json"))
if len(files) > 1 or (files and not go):
    with st.expander(f"지난 스캔 {len(files)}건 — 그때 고른 종목은 그 뒤 어땠나"):
        hist = []
        for fpath in files[-30:]:
            h = json.loads(fpath.read_text())
            for c in h["confirmed"]:
                if c["judgement"]["decision"] == "enter_long":
                    hist.append({"스캔 기준 봉": h["as_of"], "종목": c["symbol"], "그날 종가": c["close"], "손절": c["judgement"].get("stop"), "목표1": c["judgement"].get("target1")})
        if hist:
            H = pd.DataFrame(hist).drop_duplicates(["스캔 기준 봉", "종목"])
            try:
                px = _prices(tuple(sorted(set(H["종목"]) | {"QQQ", "SPY"})), date.today())
                H["지금 종가"] = H["종목"].map(lambda s: float(px[s]["Close"].iloc[-1]) if s in px else None)
                H["그 뒤 수익률"] = H["지금 종가"] / H["그날 종가"] - 1
                q = px["QQQ"]["Close"]
                H["같은 기간 QQQ"] = H["스캔 기준 봉"].map(lambda dd: float(q.iloc[-1] / q[q.index <= pd.Timestamp(dd)].iloc[-1] - 1))
            except Exception:  # noqa: BLE001
                pass
            st.dataframe(H.iloc[::-1].style.format({"그날 종가": "{:.2f}", "지금 종가": "{:.2f}", "손절": "{:.2f}", "목표1": "{:.2f}", "그 뒤 수익률": "{:+.1%}", "같은 기간 QQQ": "{:+.1%}"}, na_rep="—"),
                         width="stretch", hide_index=True)
            st.caption("단순히 '그날 종가에 사서 지금까지 들고 있었다면'의 수익률이다(손절·목표는 반영하지 않음). 고른 종목이 QQQ 를 꾸준히 넘는지를 여기서 쌓아 본다.")
        else:
            st.caption("지난 스캔에 진입 종목이 없습니다.")
