"""AI 패턴 백테스트 공용 페이지 — ``render_page("<pattern key>")`` 한 줄로 패턴별 페이지가 된다.

기간을 정하면 봉을 읽고, 기간 전체를 봉 순서대로 걸어가며 Claude 에게 진입·청산을 묻고(뒤 봉은 안 보여 줌) 손익을 낸다.
엔진 strategy/adapters/ai_pattern_backtest.py · 패턴 원칙 ai_patterns.py · Claude 호출 claude_chart_judge.py.
"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

from strategy.adapters.ai_chart_judge import NY, build_packet, prepare
from strategy.adapters.ai_pattern_backtest import BtConfig, checkpoints, run_backtest, run_scan
from strategy.adapters.ai_patterns import PATTERNS, PatternSpec
from strategy.adapters.claude_chart_judge import DEFAULT_MODEL, MODELS, ClaudeJudge, estimate_cost, has_credentials
from visualization.adapters.lightweight_charts import Pane, render as render_lwc

ROOT = Path(__file__).resolve().parent.parent.parent / "sweep_results" / "ai_pattern"
DECISION_KO = {"enter_long": "🟢 진입", "wait": "🟡 기다림", "no_trade": "⚪ 안 함"}
SOURCES = {"av_ext": "Alpha Vantage (시간외 포함)", "av_rth": "Alpha Vantage (정규장)", "stack_rth": "기본 스택 (정규장)", "stack_all": "기본 스택 (시간외 포함)",
           "crypto": "기본 스택 (24시간)", "daily": "yfinance 일봉"}


@st.cache_data(ttl=3600, show_spinner="봉 로딩…")
def _bars(sym: str, s: date, e: date, interval: str, src: str) -> pd.DataFrame:
    from data.adapters.alphavantage_adapter import AlphaVantageAdapter
    from data.adapters.composed_market_data import build_default_market_data
    from data.adapters.regular_session_filter import RegularSessionFilterAdapter
    from data.adapters.yfinance_adapter import YFinanceAdapter

    if src == "daily":
        df = YFinanceAdapter().fetch_ohlcv(sym, s, e)
    elif src in ("av_ext", "av_rth"):
        md = AlphaVantageAdapter()
        df = (RegularSessionFilterAdapter(md) if src == "av_rth" else md).fetch_ohlcv(sym, s, e, interval)
    else:
        md = build_default_market_data(bypass_today=e >= date.today())
        df = (RegularSessionFilterAdapter(md) if src == "stack_rth" else md).fetch_ohlcv(sym, s, e, interval)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"]).sort_index()
    if interval != "1d":
        df.index = df.index.tz_localize(NY) if df.index.tz is None else df.index.tz_convert(NY)
    elif df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df[~df.index.duplicated(keep="first")]


@st.cache_data(ttl=3600, show_spinner="일봉 로딩…")
def _daily(sym: str, s: date, e: date) -> pd.DataFrame:
    from data.adapters.composed_market_data import build_default_market_data

    df = build_default_market_data().fetch_ohlcv(sym, s, e, "1d")
    return pd.DataFrame() if df is None or df.empty else df[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"]).sort_index()


def _fmt_t(ts, intraday: bool) -> str:
    ts = pd.Timestamp(ts)
    if not intraday:
        return f"{ts:%Y-%m-%d}"
    return f"{(ts.tz_convert(NY) if ts.tzinfo else ts):%m-%d %H:%M}"


def render_page(key: str) -> None:
    P: PatternSpec = PATTERNS[key]
    spec = P.chart
    intraday = spec.intraday
    st.set_page_config(page_title=f"{P.title} AI", layout="wide")
    st.title(f"{P.title} — AI 판정 백테스트")
    st.caption(f"규칙 엔진 페이지(**{P.rule_page}**)는 숫자 조건으로 패턴을 끊는다. 이 페이지는 같은 패턴의 **원칙만** Claude 에게 주고, 설정한 기간을 봉 순서대로 걸어가며 "
               "**그 시점까지의 차트(그림 + 봉 표 + 큰 그림)** 를 보여 주고 진입·손절·목표·청산을 맡긴 뒤 손익을 낸다. 모델은 뒤 봉을 보지 못한다. "
               "기본은 **한 번에 훑기**(기간을 통째로 보여 주고 자리를 한꺼번에 찍게 함 — 호출 한두 번), 옵션으로 **한 봉씩 걸어가기**(뒤 봉을 가리고 시점마다 물음 — 느리고 비쌈). "
               "답은 디스크에 저장되어 같은 설정으로 다시 돌리면 무료다. 결과는 검증이 아니다."
               + (f" {P.note}" if P.note else ""))

    with st.sidebar:
        st.header("종목·기간")
        ticker = st.text_input("티커", value=P.ticker, key=f"{key}_tk").strip().upper()
        c1, c2 = st.columns(2)
        end_default = date.today() - timedelta(days=1)
        start = c1.date_input("시작", value=end_default - timedelta(days=P.days), max_value=date.today(), key=f"{key}_s")
        end = c2.date_input("끝", value=end_default, max_value=date.today(), key=f"{key}_e")
        src_opts = {"av_ext": ["av_ext", "av_rth", "stack_rth"], "stack_rth": ["stack_rth", "stack_all", "av_rth"], "stack_all": ["stack_all", "stack_rth", "av_ext"],
                    "crypto": ["crypto"], "daily": ["daily"]}[P.source]
        source = st.radio("데이터 소스", src_opts, index=0, format_func=lambda k: SOURCES[k], key=f"{key}_src") if len(src_opts) > 1 else src_opts[0]

        st.header("방식")
        mode = st.radio("AI 에게 묻는 방식", ["scan", "walk"], index=0, key=f"{key}_mode",
                        format_func=lambda k: {"scan": "한 번에 훑기 (빠름·저렴)", "walk": "한 봉씩 걸어가기 (느림·비쌈)"}[k],
                        help="훑기: 기간을 차트 한 장(길면 여러 장)으로 주고 패턴 자리를 한꺼번에 찍게 한다. 모델이 뒤 봉까지 보고 고르므로 수익률은 실전 성적이 아니라 "
                             "'AI 눈에 패턴이 어디서 보이나'를 그려 보는 용도. 걸어가기: 시점마다 그때까지의 차트만 주고 묻는다(미래 차단, 호출 수십~수백 번)")
        window = st.slider("훑기: 한 장에 넣을 봉 수", 100, 500, 300, 50, key=f"{key}_win", help="기간이 이보다 길면 여러 장으로 나눠 장마다 한 번 호출한다. 봉이 많을수록 한 봉이 작게 보인다")
        ai_exit = st.checkbox("훑기: AI 가 찍은 청산 봉도 쓰기", value=True, key=f"{key}_aix", help="끄면 손절·목표·장 마감·최대 보유로만 청산한다")

        st.header("모델")
        model = st.selectbox("Claude 모델", list(MODELS), index=list(MODELS).index(DEFAULT_MODEL), key=f"{key}_m",
                             format_func=lambda k: f"{MODELS[k][0]} (${MODELS[k][1]:g}/${MODELS[k][2]:g} per 1M)")
        effort = st.select_slider("생각 깊이 (effort)", ["low", "medium", "high", "xhigh"], value="medium", key=f"{key}_ef", help="높을수록 오래 생각하고 비용·시간이 는다")
        blind = st.checkbox("블라인드 (티커·날짜·가격 가림)", value=True, key=f"{key}_bl",
                            help="모델에게 '종목 X', D-1/D0(일봉은 T-3/T0), 지수화 가격만 준다. 과거 기간을 시험할 때 모델이 그날 시세를 기억해 맞히는 것을 막는다")
        note = st.text_area("모델에게 덧붙일 말 (선택)", value="", height=68, key=f"{key}_nt", help="바꾸면 캐시가 새로 쌓인다(다른 질문이므로)")

        st.header("걸어가기 설정" if mode == "walk" else "청산·비용 (걸어가기 전용 항목 포함)")
        step = st.number_input("몇 봉마다 물을까", value=P.step, min_value=1, max_value=64, key=f"{key}_st",
                               help="작을수록 촘촘히 보지만 호출이 는다. 조건부 진입(어느 가격 위 마감·눌림 매수)은 다음 판단 시점까지 유효하다")
        manage = st.checkbox("보유 중에도 같은 간격으로 묻기 (AI 가 청산·손절 올리기)", value=True, key=f"{key}_mg",
                             help="끄면 진입 때 정한 손절·목표로만 청산한다")
        regular_only = flat = False; ask_from = ask_until = ""
        if intraday and spec.session == "us":
            regular_only = st.checkbox("정규장 봉에서만 진입 판단", value=P.regular_only, key=f"{key}_ro")
            flat = st.checkbox("장 마감에 청산", value=P.flat_at_close, key=f"{key}_fl")
            a, b = st.columns(2)
            ask_from = a.text_input("진입 판단 시작 (HH:MM)", value=P.ask_from, key=f"{key}_af").strip()
            ask_until = b.text_input("진입 판단 끝 (HH:MM)", value=P.ask_until, key=f"{key}_au").strip()
        max_hold = st.number_input("최대 보유 봉 수 (0 = 없음)", value=P.max_hold, min_value=0, step=4, key=f"{key}_mh")
        fee_bps = st.number_input("편도 비용 (bp)", value=5.0, min_value=0.0, step=1.0, key=f"{key}_fee")
        max_calls = st.number_input("호출 상한", value=150, min_value=5, max_value=3000, step=25, key=f"{key}_mc", help="새로 내는 호출 수가 여기에 닿으면 멈춘다 (캐시된 답은 세지 않는다)")
        workers = st.slider("동시 호출 수", 1, 6, 3, key=f"{key}_wk", help="분당 한도(429)에 자주 걸리면 줄인다")
        capital = st.number_input("초기 자본 ($)", value=10_000, min_value=100, step=1_000, key=f"{key}_cap")

    if start >= end:
        st.error("시작일이 끝보다 앞이어야 합니다."); st.stop()
    if not ticker:
        st.stop()
    for hm in (ask_from, ask_until):
        if hm and not (len(hm) == 5 and hm[2] == ":" and hm.replace(":", "").isdigit()):
            st.error("진입 판단 시간대는 HH:MM 형식으로 적어 주세요 (예: 09:35)."); st.stop()
    try:
        bars = _bars(ticker, start - timedelta(days=P.warmup_days), end, spec.interval, source)
    except Exception as ex:  # noqa: BLE001
        st.error(f"데이터 오류: {type(ex).__name__}: {str(ex)[:200]}"); st.stop()
    if bars.empty:
        st.warning("봉이 없습니다. 기간·소스를 바꿔 보세요 (1분봉·5분봉은 소스에 따라 최근 며칠~몇 주만 있다)."); st.stop()
    context = pd.DataFrame()
    if intraday:
        if source in ("av_ext", "av_rth"):
            from strategy.adapters.vwap_sr_breakout import clean_wicks
            bars, _ = clean_wicks(bars, 1.0)             # 시간외 bad tick 꼬리 정리 (55 페이지와 같음)
        try:
            context = _daily(ticker, start - timedelta(days=220), end)
        except Exception:  # noqa: BLE001
            context = pd.DataFrame()

    cfg = BtConfig(step=int(step), max_hold=int(max_hold), fee_bps=float(fee_bps), blind=blind, manage=manage, flat_at_close=flat, regular_only=regular_only,
                   ask_from=ask_from, ask_until=ask_until, max_calls=int(max_calls), chunk=max(2, int(workers) * 2))
    t_start = pd.Timestamp(start); t_end = pd.Timestamp(end) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    if intraday:
        t_start, t_end = t_start.tz_localize(NY), t_end.tz_localize(NY)
    in_range = bars[(bars.index >= t_start) & (bars.index <= t_end)]
    scan = mode == "scan"
    if scan:
        n_win = max(1, -(-len(in_range) // int(window)))
        cps = list(range(n_win))
        per = min(len(in_range), int(window)) + 60                      # 한 장의 봉 수(맥락 60 포함) → 표 한 줄 ≈ 30 토큰 + 그림·원칙 ≈ 5천
        _, pin, pout = MODELS[model]
        est_lo = n_win * ((5000 + per * 30) * pin + 2500 * pout) / 1e6; est_hi = n_win * ((5000 + per * 30) * pin + 9000 * pout) / 1e6
    else:
        cps = checkpoints(bars, spec, cfg, t_start, t_end)
        est_lo = estimate_cost(model, len(cps)); est_hi = estimate_cost(model, int(len(cps) * (1.6 if manage else 1.0)))
    k = st.columns(5)
    k[0].metric("기간 안 봉 수", f"{len(in_range):,}", f"{spec.interval}", delta_color="off")
    k[1].metric("호출 수 (차트 장 수)" if scan else "판단 시점", f"{len(cps):,}", f"{int(window)}봉씩" if scan else f"{int(step)}봉마다", delta_color="off")
    k[2].metric("예상 비용 (전부 새 호출일 때)", f"${est_lo:.2f} – ${est_hi:.2f}", MODELS[model][0], delta_color="off")
    k[3].metric("호출 상한", f"{int(max_calls)}")
    k[4].metric("단순 보유 (기간)", f"{in_range['Close'].iloc[-1] / in_range['Close'].iloc[0] - 1:+.2%}" if len(in_range) > 1 else "—")
    if scan and len(in_range) < 10:
        st.warning("기간 안에 봉이 너무 적습니다."); st.stop()
    if not cps:
        st.warning("판단 시점이 없습니다. 기간 앞의 여유 봉이 부족하거나(40봉 필요) 진입 판단 시간대가 봉과 안 맞습니다."); st.stop()
    if len(cps) > int(max_calls):
        st.info(f"판단 시점({len(cps)})이 호출 상한({int(max_calls)})보다 많습니다. 상한에 닿으면 거기서 멈추고 그때까지의 결과를 보여 줍니다. "
                "다시 실행하면 앞부분은 캐시에서 무료로 읽고 이어서 갑니다.")

    have_key = has_credentials()
    if not have_key:
        st.warning("**ANTHROPIC_API_KEY 가 컨테이너에 없습니다.** `secrets/secret.env` 에 넣고 `docker compose up -d backtester` 로 다시 만들어 주세요.")
    rkey = f"ai_bt|{key}|{mode}|{window}|{ai_exit}|{ticker}|{start}|{end}|{source}|{model}|{effort}|{blind}|{note}|{cfg}"
    cache_dir = ROOT / "cache" / key
    b1, b2 = st.columns([1, 5])
    go = b1.button("▶ AI 로 패턴 찾기" if scan else "▶ AI 백테스트 실행", type="primary", disabled=not have_key, key=f"{key}_go")
    b2.caption("기간을 차트로 통째로 보여 주고 패턴 자리를 한꺼번에 받는다(장당 호출 1번, 보통 1–3분)." if scan else
               "실행하면 판단 시점마다 Claude 를 부른다(Opus 는 한 번에 20–60초, 여러 장 동시 호출). 중간에 멈춰도 받은 답은 저장되어 다음 실행에서 이어진다.")
    if go:
        judge = ClaudeJudge(title=P.title, principles=P.principles, step=int(step), model=model, effort=effort, note=note, cache_dir=cache_dir, workers=int(workers))
        prog = st.progress(0.0, text="시작…")
        try:
            if scan:
                res = run_scan(bars, spec, cfg, start=t_start, end=t_end, ask_scan=judge.ask_scan, context=context if len(context) else None, ticker=ticker,
                               window=int(window), use_ai_exit=ai_exit, progress=lambda f, m: prog.progress(min(1.0, f), text=m))
            else:
                res = run_backtest(bars, spec, cfg, start=t_start, end=t_end, ask_entry=judge.ask_entry, ask_manage=judge.ask_manage if manage else None,
                                   context=context if len(context) else None, ticker=ticker, progress=lambda f, m: prog.progress(min(1.0, f), text=m))
            res["mode"] = mode
            st.session_state[rkey] = res
            st.session_state[f"{key}_last"] = (res, f"{ticker} · {start} ~ {end} · {MODELS[model][0]}", ticker)
            _save_run(key, ticker, start, end, model, effort, blind, cfg, res, mode)
        except Exception as ex:  # noqa: BLE001
            st.error(f"실행 실패: {type(ex).__name__}: {str(ex)[:500]} — 이미 받은 답은 캐시에 남아 있어 다시 실행하면 이어집니다.")
        prog.empty()
    res = st.session_state.get(rkey)
    last = st.session_state.get(f"{key}_last")
    if not res and last:
        st.info(f"사이드바 설정이 바뀌어 **직전 실행 결과**({last[1].replace('~', '–')})를 그대로 보여 줍니다. 새 설정으로 보려면 위 버튼을 다시 누르세요.")
        _results(P, last[0], last[2], float(capital), cfg)
        _history(key)
        return
    if not res:
        st.caption("아직 실행하지 않았습니다. 아래는 모델에게 보낼 자료의 미리 보기입니다(호출 없음).")
        with st.expander("모델에게 보내는 자료 미리 보기 — 첫 판단 시점", expanded=True):
            try:
                i = bars.index.get_loc(in_range.index[min(len(in_range), int(window)) - 1]) if scan else cps[0]
                pk = build_packet(bars.iloc[: i + 1], bars.index[i], spec=(replace(spec, view_bars=min(len(in_range), int(window)) + 60) if scan else spec), context=context if len(context) else None, ticker=ticker, blind=blind)
                st.image(pk.chart_png, caption="차트 (모델이 보는 그대로)")
                if pk.context_png:
                    st.image(pk.context_png, caption="큰 그림")
                st.json(pk.facts)
            except Exception as ex:  # noqa: BLE001
                st.error(f"{type(ex).__name__}: {ex}")
        with st.expander("모델에게 주는 패턴 원칙"):
            st.text(P.principles)
        _history(key)
        return

    _results(P, res, ticker, float(capital), cfg)
    with st.expander("모델에게 주는 패턴 원칙"):
        st.text(P.principles)
    _history(key)


# ---------------------------------------------------------------------------
def _results(P: PatternSpec, res: dict, ticker: str, capital: float, cfg: BtConfig) -> None:
    spec = P.chart; intraday = spec.intraday
    S, trades, decisions, f = res["stats"], res["trades"], res["decisions"], res["frame"]
    scan = res.get("mode") == "scan"
    if S["중단"]:
        st.warning(S["중단"] + " — 호출 상한을 올리거나 다시 실행하면 이어집니다.")
    st.subheader(f"{ticker} {_fmt_t(S['기간'][0], intraday)} ~ {_fmt_t(S['기간'][1], intraday)} · {'찾은 자리' if scan else '판단'} {S['판단 수']}{'개' if scan else '회'} · 새 호출 {S['호출 수']}회 · 이번 실행 비용 ${S['비용($)']:.2f}")
    m = st.columns(7)
    m[0].metric("AI 전략 수익률", f"{S['총 수익률']:+.2%}")
    m[1].metric("단순 보유", f"{S['단순 보유']:+.2%}")
    m[2].metric("최종 자본", f"${capital * (1 + S['총 수익률']):,.0f}")
    m[3].metric("MDD", f"{S['MDD']:.2%}")
    m[4].metric("거래 수", f"{S['거래 수']}", f"노출 {S['시장 노출']:.0%}", delta_color="off")
    m[5].metric("승률", f"{S['승률']:.0%}" if trades else "—")
    pf = S["손익비(PF)"]
    m[6].metric("손익비 / 평균 R", (f"{'∞' if pf == float('inf') else f'{pf:.2f}'} / {S['평균 R']:+.2f}") if trades else "—")
    if trades:
        st.caption("청산 사유: " + " · ".join(f"{k} {v}" for k, v in S["청산 사유"].items())
                   + f" · 거래당 평균 {S['거래당 평균']:+.2%} · 평균 보유 {S['평균 보유 봉']:.1f}봉 · 진입 판단 {S['판단 수']}회 중 {S['거래 수']}회 체결")
    else:
        st.info("이 기간에 AI 가 찾은(진입한) 자리가 없습니다.")

    if scan:
        st.caption("⚠️ 한 번에 훑기: 모델이 뒤 봉까지 보고 자리를 골랐으므로 수익률은 실전 성적이 아니다 — AI 가 패턴을 어디서 보는지 그려 보는 용도.")

    # 차트 ------------------------------------------------------------------
    s_i, e_i = res["range"]
    view_opts = ["전체"] + [f"거래 #{n + 1} · {_fmt_t(t['entry_time'], intraday)} · {t['setup'][:24]} · {t['ret']:+.2%}" for n, t in enumerate(trades)]
    pick = st.selectbox("차트 보기", view_opts, index=0, key=f"{P.key}_view", help="거래를 고르면 그 진입~청산 구간을 확대하고 AI 가 그은 선을 굵게 보여 준다") if trades else "전체"
    pick_n = view_opts.index(pick) - 1 if pick in view_opts else -1
    tsel = st.session_state.get(f"{P.key}_trade_tbl"); dsel = st.session_state.get(f"{P.key}_dec_tbl")
    rows = lambda o: list(((o.get("selection") or {}).get("rows") if isinstance(o, dict) else getattr(getattr(o, "selection", None), "rows", None)) or []) if o is not None else []  # noqa: E731
    focus = None
    if pick_n >= 0:
        tr = trades[pick_n]; pad = max(12, spec.view_bars // 4)
        focus = (max(0, tr["decision_i"] - pad * 2), min(len(f), tr["exit_i"] + pad), tr["judgement"], tr["decision_i"])
    elif rows(tsel) and rows(tsel)[0] < len(trades):
        tr = trades[rows(tsel)[0]]; focus = (max(0, tr["decision_i"] - spec.view_bars // 2), min(len(f), tr["exit_i"] + spec.view_bars // 4), tr["judgement"], tr["decision_i"])
    elif rows(dsel) and rows(dsel)[0] < len(decisions):
        d = decisions[rows(dsel)[0]]; focus = (max(0, d["i"] - spec.view_bars), min(len(f), d["i"] + spec.view_bars // 3), d["judgement"], d["i"])
    lo, hi = (focus[0], focus[1]) if focus else (max(0, s_i - spec.view_bars // 3), min(len(f), e_i + 2))
    if hi - lo > 3000:
        lo = hi - 3000
        st.caption("봉이 많아 차트에는 마지막 3,000봉만 그립니다. 거래·판단 표에서 행을 고르면 그 구간으로 갑니다.")
    g = prepare(f, spec).iloc[lo:hi]
    x = pd.DatetimeIndex(g.index.tz_convert(NY).tz_localize(None) if intraday else g.index)
    gx = g.copy(); gx.index = x
    xt = lambda i: x[min(max(i - lo, 0), len(x) - 1)]  # noqa: E731
    lines = []
    if "vwap" in gx:
        lines.append(dict(name="세션 VWAP", data=gx["vwap"], color="#ff9100", width=2))
    for n, col in zip(spec.emas, ("#42a5f5", "#ab47bc", "#8d6e63", "#26a69a")):
        lines.append(dict(name=f"EMA{n}", data=gx[f"ema{n}"], color=col, width=1))
    segs: list[dict] = []; marks: list[dict] = []; boxes: list[dict] = []
    lvl_col = {"resistance": "#64b5f6", "trend_resistance": "#64b5f6", "support": "#fdd835", "trend_support": "#fdd835"}

    def draw_levels(jj, i_at: int, i_to: int, width: int) -> None:
        for lv in jj.levels[:4]:
            i0 = max(lo, i_at - int(lv.bars_ago_start or 20))
            y0 = lv.price_start if (lv.kind.startswith("trend") and lv.price_start is not None) else lv.price
            pts = [(xt(i0), y0), (xt(i_at), lv.price)]
            if not lv.kind.startswith("trend") and i_to > i_at:
                pts.append((xt(i_to), lv.price))
            if pts[-1][0] > pts[0][0]:
                segs.append(dict(points=pts, color=lvl_col[lv.kind], width=width))

    for n_tr, tr in enumerate(trades):
        if tr["exit_i"] < lo or tr["entry_i"] >= hi:
            continue
        win = tr["ret"] > 0
        last_px = tr["legs"][-1]["가격"]
        if min(tr["exit_i"], hi - 1) > max(tr["entry_i"], lo):   # 진입~청산 구간을 색 상자로 (이익 초록·손실 빨강)
            boxes.append(dict(t0=xt(max(tr["entry_i"], lo)), t1=xt(min(tr["exit_i"], hi - 1)), p0=min(tr["entry"], last_px), p1=max(tr["entry"], last_px),
                              fill="rgba(0,230,118,0.18)" if win else "rgba(255,23,68,0.18)", line="#00e676" if win else "#ff1744"))
        if lo <= tr["decision_i"] < hi:
            draw_levels(tr["judgement"], tr["decision_i"], min(tr["exit_i"], hi - 1), 1)
        a, b = max(tr["entry_i"], lo), min(tr["exit_i"], hi - 1)
        if b > a:
            segs.append(dict(points=[(xt(a), tr["stop0"]), (xt(b), tr["stop_last"])], color="#ff1744", width=2, style="dash"))
            jj = tr["judgement"]
            if jj.target1:
                segs.append(dict(points=[(xt(a), jj.target1), (xt(b), jj.target1)], color="#26c6da", width=1, style="dash"))
        tag = f"#{n_tr + 1}"
        if tr["entry_i"] >= lo:
            marks.append(dict(time=xt(tr["entry_i"]), position="below", shape="arrowUp", color="#00e676", text=f"{tag} 매수 {tr['entry']:.2f}"))
        for k_lg, lg in enumerate(tr["legs"]):
            if lo <= lg["i"] < hi:
                final = k_lg == len(tr["legs"]) - 1
                marks.append(dict(time=xt(lg["i"]), position="above", shape="arrowDown", color="#00e676" if (final and win) else "#ff5252" if final else "#ffeb3b",
                                  text=f"{tag} {lg['사유']} {lg['가격']:.2f}" + (f" ({tr['ret']:+.1%})" if final else "")))
    if focus:
        jf, i_f = focus[2], focus[3]
        marks.append(dict(time=xt(i_f), position="above", shape="circle", color="#90caf9", text="신호 봉"))
        draw_levels(jf, i_f, min(hi - 1, i_f + max(6, spec.view_bars // 8)), 3)
    marks.sort(key=lambda m_: m_["time"])
    vol_color = ["rgba(38,166,154,0.7)" if c >= o else "rgba(239,83,80,0.7)" for o, c in zip(g["Open"], g["Close"])]
    eq = res["equity"].iloc[lo:hi] * capital
    bh = capital * g["Close"] / f["Close"].iloc[s_i]
    st.caption("▲ 매수 · ▼ 청산(사유·수익률) · 색 상자 = 진입–청산 구간(초록 이익·빨강 손실) · 빨강 점선 = 손절 · 청록 점선 = 목표1 · 파랑/노랑 선 = AI 가 그은 저항/지지 "
               "— 위 '차트 보기'에서 거래를 고르면 그 구간을 확대한다")
    render_lwc([
        Pane(height=540, title=f"{ticker} {spec.interval}", candles=gx, lines=lines, segments=segs, markers=marks, boxes=boxes),
        Pane(height=120, title="거래량", hist=[dict(name="거래량", data=gx["Volume"], colors=vol_color)]),
        Pane(height=180, title="자본 곡선", lines=[dict(name="AI 전략", data=pd.Series(eq.to_numpy(), index=x), color="#42a5f5", width=2),
                                                dict(name="단순 보유", data=pd.Series(bh.to_numpy(), index=x), color="gray", width=1, style="dot")]),
    ], index=x, intraday=intraday, theme="dark")

    if scan and res.get("overview"):
        with st.expander("AI 의 구간 총평", expanded=True):
            st.write(res["overview"].replace("~", "–"))
    if focus:
        jf = focus[2]
        with st.expander(f"고른 판단의 근거 — {DECISION_KO[jf.decision]} · 확신 {jf.confidence} · {jf.setup}", expanded=True):
            st.info(jf.summary)
            a, b = st.columns(2)
            a.markdown(f"**큰 그림**  \n{jf.market_read}\n\n**차트 구조**  \n{jf.structure_read}\n\n**지표**  \n{jf.indicator_read}")
            b.markdown(f"**거래량**  \n{jf.volume_read}\n\n**틀릴 수 있는 이유**  \n{jf.risks}\n\n**조건**  \n{jf.wait_for}")

    # 거래 표 ----------------------------------------------------------------
    st.subheader("거래 목록")
    if trades:
        tb = pd.DataFrame([{"판단": _fmt_t(t["decision_time"], intraday), "자리": t["setup"], "확신": t["confidence"], "진입": _fmt_t(t["entry_time"], intraday), "진입가": t["entry"],
                            "손절": t["stop0"], "목표1": t["judgement"].target1 if t["judgement"].target1 is not None else float("nan"), "청산": _fmt_t(t["exit_time"], intraday), "사유": t["why"], "보유 봉": t["bars"],
                            "수익률": t["ret"], "R": t["R"], "자본": capital * t["equity"], "AI 관리": " / ".join(
                                f"{n['action']}{'→손절 %.2f' % n['new_stop'] if n['new_stop'] else ''}" for n in t["manage_notes"])} for t in trades])
        st.dataframe(tb.style.format({"진입가": "{:.2f}", "손절": "{:.2f}", "목표1": "{:.2f}", "수익률": "{:+.2%}", "R": "{:+.2f}", "자본": "{:,.0f}"}, na_rep="—"),
                     width="stretch", hide_index=True, height=min(520, 60 + 35 * len(tb)), key=f"{P.key}_trade_tbl", on_select="rerun", selection_mode="single-row")
        for n_t, t in enumerate(trades):
            if t["manage_notes"]:
                with st.expander(f"거래 #{n_t + 1} 보유 중 AI 판단 {len(t['manage_notes'])}회"):
                    for n in t["manage_notes"]:
                        st.markdown(f"- **{_fmt_t(f.index[n['t']], intraday)}** · {'계속 보유' if n['action'] == 'hold' else '청산'} (확신 {n['confidence']})"
                                    + (f" · 손절 → {n['new_stop']:.2f}" if n["new_stop"] else "") + (f" · 목표 → {n['new_target']:.2f}" if n["new_target"] else "") + f"  \n  {n['reason']}")
        st.download_button("거래 CSV", tb.to_csv(index=False).encode("utf-8-sig"), file_name=f"ai_{P.key}_{ticker}.csv", key=f"{P.key}_dl")

    st.subheader("AI 가 찾은 자리 전체" if scan else "진입 판단 전체")
    if not decisions:
        return
    db = pd.DataFrame([{"시각": _fmt_t(d["time"], intraday), "결정": DECISION_KO[d["judgement"].decision], "확신": d["judgement"].confidence, "자리": d["judgement"].setup,
                        "체결": "예" if d["filled"] else "—", "결과": d.get("why", ""), "수익률": d.get("ret", float("nan")), "종가": d["close"],
                        "캐시": "예" if d["usage"].get("cached") else "", "요약": d["judgement"].summary} for d in decisions])
    vc = db["결정"].value_counts()
    st.caption(" · ".join(f"{k_} {v}" for k_, v in vc.items()) + " — 행을 고르면 그 시점의 차트와 근거를 위에 보여 준다.")
    st.dataframe(db.style.format({"수익률": "{:+.2%}", "종가": "{:.2f}"}, na_rep="—"), width="stretch", hide_index=True, height=min(520, 60 + 35 * len(db)),
                 key=f"{P.key}_dec_tbl", on_select="rerun", selection_mode="single-row")


def _save_run(key, ticker, start, end, model, effort, blind, cfg: BtConfig, res: dict, mode: str = "walk") -> None:
    S = res["stats"]
    rec = {"at": pd.Timestamp.now(tz="UTC").isoformat(), "pattern": key, "ticker": ticker, "start": str(start), "end": str(end), "model": model, "effort": effort, "blind": blind,
           "mode": mode, "step": cfg.step, "manage": cfg.manage, **{k: (v if not isinstance(v, tuple) else [str(x) for x in v]) for k, v in S.items()}}
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT / "runs.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")


def _history(key: str) -> None:
    p = ROOT / "runs.jsonl"
    if not p.exists():
        return
    rows = [json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    df = pd.DataFrame([r for r in rows if r.get("pattern") == key])
    if df.empty:
        return
    with st.expander(f"이 패턴의 지난 실행 {len(df)}건 (전략 vs 단순 보유)"):
        cols = [c for c in ("at", "mode", "ticker", "start", "end", "model", "effort", "blind", "step", "총 수익률", "단순 보유", "MDD", "거래 수", "승률", "평균 R", "호출 수", "비용($)") if c in df]
        show = df[cols].iloc[::-1].copy(); show["at"] = show["at"].str[:16]
        st.dataframe(show.style.format({"총 수익률": "{:+.2%}", "단순 보유": "{:+.2%}", "MDD": "{:.2%}", "승률": "{:.0%}", "평균 R": "{:+.2f}"}, na_rep="—"), width="stretch", hide_index=True)
        st.caption("여러 종목·기간에서 AI 전략이 단순 보유를 꾸준히 넘는지, 거래 수가 충분한지를 본다. 한두 번의 결과는 운이다.")
