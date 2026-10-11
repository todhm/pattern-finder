"""미너비니 체크리스트 **전 단계 통과 종목만** 실제로 샀을 때의 수익률 — 샘플 기준일 백테스트.

두 단계로 나눠 돌린다 (사람 판독이 결과를 보기 **전에** 끝나야 하므로):

1) ``prepare`` — 기준일마다
   STAGE 0 게이트 → 스크린(워치 포함) → STAGE 1~4 자동 판정
   → 워치 종목을 기준일 이후 30거래일 동안 매일 재판정해 VCP 완성 + 거래량 돌파 진입
   → 진입 신호마다 **블라인드 판독 자료**(티커·날짜·이후 데이터 없음)를 만든다.
2) ``results`` — 판독 결과(BUY/PASS)를 읽어 포트폴리오(정상 5종목·경고 시 3종목,
   각 20%)를 진입일 순으로 채우고 손절 8%·익절 22% 로 청산해 계좌 수익률을 낸다.

    docker compose exec backtester python minervini_checklist_backtest_cli.py prepare \\
        '{"dates": ["2020-01-02", ...], "state": "/tmp/mcb_state.json", "dossiers": "/tmp/mcb_dossiers.md"}'
    docker compose exec backtester python minervini_checklist_backtest_cli.py results \\
        '{"state": "/tmp/mcb_state.json", "reviews": "/tmp/mcb_reviews.json"}'
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import asdict, replace
from datetime import date, timedelta

import pandas as pd

from data.adapters.alphavantage_adapter import AlphaVantageAdapter
from data.adapters.mongo_day_cache import MongoDayCacheAdapter
from data.adapters.wikipedia_universe import default_universe_provider
from pages._shared.growth_sources import build_growth_adapters
from strategy.adapters.minervini_full_checklist import (
    Entry,
    find_watch_entry,
    simulate_trade,
    stage2,
    stage3,
    stage4,
)
from strategy.adapters.minervini_manual_auto import industry_rank, manual_auto
from strategy.adapters.minervini_scan import ScreenParams, fetch_with_retry, run_screen
from strategy.adapters.minervini_screener import evaluate_balance, evaluate_surprise

MAX_POSITIONS = {"normal": 5, "caution": 3}
WEIGHT = 0.20


def _log(m: str) -> None:
    print(m, file=sys.stderr, flush=True)


def _clip(df: pd.DataFrame) -> pd.DataFrame:
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index = df.index.normalize()
    return df


def _news_titles(news, t: str, as_of: date) -> list[str] | None:
    if news is None:
        return None
    return [n.title for n in news.fetch_news(t, as_of - timedelta(days=120), as_of)]


def _blind(text: str) -> str:
    """판독 자료에서 절대 가격(티커 추정 단서)을 지운다 — % 만 남긴다."""
    return re.sub(r"\$?\b\d[\d,]*\.\d+\b(?!%)", "…", text)


def _dossier(did: str, df_full: pd.DataFrame, e: Entry, stages: dict) -> str:
    """블라인드 판독 자료 — 피봇=100 으로 정규화, 날짜는 셋업일 기준 상대 주/일."""
    hist = df_full.loc[: pd.Timestamp(e.setup_date)]
    k = 100.0 / e.pivot
    w = hist.tail(130).resample("W-FRI").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}).dropna()
    v50w = float(hist["Volume"].tail(50).mean()) * 5
    lines = [f"### {did}", "", "주봉 (피봇=100, 거래량=50일 평균 주간 대비 배수)", "",
             "| 주 | 고 | 저 | 종 | 거래량 |", "|---|---|---|---|---|"]
    for i, (_, r) in enumerate(w.tail(26).iterrows()):
        lines.append(f"| {i - len(w.tail(26)) + 1} | {r.High * k:.1f} | {r.Low * k:.1f} | "
                     f"{r.Close * k:.1f} | {r.Volume / v50w:.2f} |")
    v50 = float(hist["Volume"].tail(50).mean())
    lines += ["", "최근 15일 (일봉)", "", "| 일 | 시 | 고 | 저 | 종 | 거래량 |", "|---|---|---|---|---|---|"]
    for i, (_, r) in enumerate(hist.tail(15).iterrows()):
        lines.append(f"| {i - 14} | {r.Open * k:.1f} | {r.High * k:.1f} | {r.Low * k:.1f} | "
                     f"{r.Close * k:.1f} | {r.Volume / v50:.2f} |")
    sma = {n: float(hist["Close"].rolling(n).mean().iloc[-1]) * k for n in (50, 150, 200)}
    lines += ["", f"- SMA50/150/200: {sma[50]:.1f} / {sma[150]:.1f} / {sma[200]:.1f}",
              f"- 자동 STAGE 4: " + " · ".join(f"{key[1:]} {_blind(v)}" for key, v in e.stage4.notes.items() if v),
              f"- 돌파일(다음 날): 피봇 돌파, 거래량 {e.breakout_volume_ratio:.1f}배",
              f"- STAGE 2/3: " + stages["s23"], ""]
    return "\n".join(lines)


def _stage23(res, as_of: date, bal, news, counts: dict):
    """후보별 STAGE 2·3 판정 (prepare·live 공용). STAGE 3 와 2-8 이전 STAGE 2 를
    먼저 통과한 종목에만 재무상태표·뉴스를 조회한다 (API 절약).

    yield (candidate, stage2, stage3, surprise) — STAGE 2/3 둘 중 하나라도 사전 탈락이면 생략.
    """
    qqq = res.index_frames["QQQ"]
    ind_counts = pd.Series([c.industry for c in res.candidates if c.industry != "미분류"]).value_counts()
    leading = set(ind_counts[ind_counts >= 2].head(10).index)
    ranks = industry_rank([(c.ticker, c.industry, c.rs) for c in res.candidates])
    for c in res.candidates:
        ma = manual_auto(res.frames[c.ticker], qqq, c.pl)
        m = dict(ma.checks)
        m["m3_4"] = ranks.get(c.ticker, (None, ""))[0]
        s3 = stage3(m, c.industry, leading, liquidity_ok=True)
        quarters = c.snapshot.quarters if c.snapshot else []
        sr = evaluate_surprise(quarters, as_of)
        pre = stage2(c.growth, c.annual, sr, None, guidance_titles=[])
        counts["s3"] = counts.get("s3", 0) + s3.ok
        if not (s3.ok and pre.ok):
            continue
        br = None
        if bal is not None:
            try:
                br = evaluate_balance(bal.fetch_balance_sheet(c.ticker), quarters, as_of)
            except Exception:  # noqa: BLE001
                br = None
        s2 = stage2(c.growth, c.annual, sr, br, _news_titles(news, c.ticker, as_of))
        counts["s2"] = counts.get("s2", 0) + s2.ok
        yield c, s2, s3, sr


def _sources():
    md = MongoDayCacheAdapter(AlphaVantageAdapter(), source_name="alphavantage")
    growth = build_growth_adapters()
    bal = next((a for a in growth if hasattr(a, "fetch_balance_sheet")), None)
    try:
        from data.adapters.alphavantage_news import AlphaVantageNewsAdapter
        news = AlphaVantageNewsAdapter()
    except ValueError:
        news = None
    return md, growth, bal, news


def live(cfg: dict) -> None:
    """오늘(또는 ``as_of``) 기준 체크리스트 자동 판정 — 수동 검증 대상 목록.

    시장 상태와 무관하게 STAGE 1~3 을 판정하고 (시장이 막혀도 워치리스트는 관리),
    통과 종목에 STAGE 4 를 두 기준(핵심 전부 / 5개 이상)으로 붙인다.
    결과: ``out`` JSON + stdout 요약.
    """
    as_of = date.fromisoformat(cfg["as_of"]) if cfg.get("as_of") else date.today()
    md, growth, bal, news = _sources()
    tickers = default_universe_provider().get_tickers(f"nasdaq_full@{as_of.isoformat()}")
    params = replace(ScreenParams(), near_pivot_only=False, stage0_gate=False)
    res = run_screen(as_of, tickers, md, growth, None, params,
                     max_workers=int(cfg.get("workers", 6)), log=_log)
    market = {"failed": res.market_failed, "warns": res.market_warns,
              "nh": res.nh, "nl": res.nl, "breadth_ok": res.breadth_ok,
              "indexes": [asdict(v) for v in res.market]}
    rows = []
    counts: dict = {}
    for c, s2, s3, sr in _stage23(res, as_of, bal, news, counts):
        if not (s2.ok and s3.ok):
            continue
        df = res.frames[c.ticker]
        strict = stage4(df, c.pl)
        loose = stage4(df, c.pl, min_core=5)
        rows.append({
            "ticker": c.ticker, "industry": c.industry, "rs": c.rs, "score": c.score,
            "entry_note": c.entry_note, "close": c.pl.close, "pivot": c.pl.pivot,
            "dist": c.pl.dist_to_pivot, "stop": c.pl.pivot * (1 - params.stop_pct),
            "vol50": float(df["Volume"].tail(50).mean()),
            "stage4_strict": strict.ok, "stage4_core5": loose.ok,
            "s2": asdict(s2), "s3": asdict(s3), "s4": asdict(loose),
            "surprise_streak": sr.streak if sr.data_available else None,
        })
    rows.sort(key=lambda r: (r["stage4_core5"], r["rs"] or 0), reverse=True)
    out = {"as_of": str(as_of), "market": market, "counts": counts,
           "candidates": len(res.candidates), "passed": rows}
    with open(cfg.get("out", "/tmp/mcb_live.json"), "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=str)
    print(f"{as_of} 시장: 중단 {res.market_failed or '없음'} · 경고 {res.market_warns or '없음'} · "
          f"0-3 {res.nh}/{res.nl} {'✅' if res.breadth_ok else '❌'}")
    print(f"후보 {len(res.candidates)} → STAGE3 {counts.get('s3', 0)} → STAGE2+3 {len(rows)}")
    for r in rows:
        miss = [k for k, v in r["s4"]["items"].items() if v is not True]
        print(f"{r['ticker']:6s} RS{r['rs']:.0f} {r['industry'][:24]:24s} 종가 {r['close']:.2f} 피봇 {r['pivot']:.2f} "
              f"({r['dist']:+.1%}) S4엄격 {'✅' if r['stage4_strict'] else '❌'} S4(5/7) "
              f"{'✅' if r['stage4_core5'] else '❌'} 미통과 {miss}")


def prepare(cfg: dict) -> None:
    md, growth, bal, news = _sources()
    uni = default_universe_provider()
    params = replace(ScreenParams(), near_pivot_only=False)  # 워치(베이스 형성 중)까지 후보로
    state: dict = {"dates": [], "entries": {}}
    dossiers: list[str] = ["# 블라인드 판독 자료", "",
                           "각 항목: VCP 모양·털어내기(4-8)·손잡이 위치(4-9)·추격 여부(4-10) 눈판독 → BUY / PASS", ""]
    n_id = 0
    for d in cfg["dates"]:
        as_of = date.fromisoformat(d)
        tickers = uni.get_tickers(f"nasdaq_full@{as_of.isoformat()}")
        if cfg.get("max_tickers"):
            tickers = tickers[: int(cfg["max_tickers"])]
        res = run_screen(as_of, tickers, md, growth, None, params,
                         max_workers=int(cfg.get("workers", 6)), log=_log)
        nh_nl_ok = res.nh is not None and res.nh > res.nl
        regime = ("blocked" if res.skipped else "no_breadth" if not nh_nl_ok
                  else "caution" if res.market_warns else "normal")
        day = {"as_of": d, "regime": regime, "failed": res.market_failed, "warns": res.market_warns,
               "nh": res.nh, "nl": res.nl, "candidates": len(res.candidates), "stage_pass": {}}
        _log(f"[{d}] {regime} 후보 {len(res.candidates)}")
        if regime in ("blocked", "no_breadth"):
            state["dates"].append(day)
            continue
        counts = {"s2": 0, "s3": 0, "s23": 0, "entry": 0}
        for c, s2, s3, sr in _stage23(res, as_of, bal, news, counts):
            if not (s2.ok and s3.ok):
                continue
            counts["s23"] += 1
            try:
                df_full = _clip(fetch_with_retry(md, c.ticker, as_of - timedelta(days=460),
                                                 min(as_of + timedelta(days=400), date.today())))
            except Exception:  # noqa: BLE001
                continue
            e = find_watch_entry(df_full, as_of, min_core4=cfg.get("min_core4", 5))
            if e is None:
                continue
            counts["entry"] += 1
            n_id += 1
            did = f"S{n_id:03d}"
            s23 = (f"성장 {c.growth.n_passed}/4, 서프라이즈 {sr.latest_beat}, RS {c.rs:.0f}, "
                   f"산업군 주도 {s3.items['s3_5']}, 3-1/3-2/3-4 "
                   f"{s3.items['s3_1']}/{s3.items['s3_2']}/{s3.items['s3_4']}")
            dossiers.append(_dossier(did, df_full, e, {"s23": s23}))
            state["entries"][did] = {
                "as_of": d, "ticker": c.ticker, "industry": c.industry, "rs": c.rs,
                "entry": {**asdict(e), "stage4": asdict(e.stage4)},
                "s2": asdict(s2), "s3": asdict(s3),
            }
        day["stage_pass"] = counts
        _log(f"[{d}] STAGE3 {counts['s3']} · STAGE2+3 {counts['s23']} · 진입신호 {counts['entry']}")
        state["dates"].append(day)
        with open(cfg["state"], "w") as f:
            json.dump(state, f, ensure_ascii=False, default=str, indent=1)
        with open(cfg["dossiers"], "w") as f:
            f.write("\n".join(dossiers))
    with open(cfg["state"], "w") as f:
        json.dump(state, f, ensure_ascii=False, default=str, indent=1)
    with open(cfg["dossiers"], "w") as f:
        f.write("\n".join(dossiers))


def _bench(md, as_of: date, days: int = 126) -> float | None:
    df = _clip(fetch_with_retry(md, "QQQ", as_of - timedelta(days=10), min(as_of + timedelta(days=200), date.today())))
    base = df.loc[: pd.Timestamp(as_of)]
    fwd = df.loc[pd.Timestamp(as_of) + pd.Timedelta(days=1):]
    if base.empty or fwd.empty:
        return None
    return float(fwd["Close"].iloc[min(days, len(fwd)) - 1] / base["Close"].iloc[-1] - 1.0)


def results(cfg: dict) -> None:
    md = MongoDayCacheAdapter(AlphaVantageAdapter(), source_name="alphavantage")
    with open(cfg["state"]) as f:
        state = json.load(f)
    reviews: dict = {}
    if cfg.get("reviews"):
        with open(cfg["reviews"]) as f:
            reviews = json.load(f)
    rows = []
    for day in state["dates"]:
        d = day["as_of"]
        as_of = date.fromisoformat(d)
        bench = _bench(md, as_of)
        ents = sorted(((k, v) for k, v in state["entries"].items() if v["as_of"] == d),
                      key=lambda kv: kv[1]["entry"]["entry_date"])
        for mode in ("auto", "manual"):
            slots = MAX_POSITIONS.get(day["regime"], 0)
            taken = []
            for did, v in ents:
                if len(taken) >= slots:
                    break
                if mode == "manual" and reviews.get(did, {}).get("decision") != "BUY":
                    continue
                e = v["entry"]
                df_full = _clip(fetch_with_retry(md, v["ticker"], date.fromisoformat(e["entry_date"]) - timedelta(days=5),
                                                 min(date.fromisoformat(e["entry_date"]) + timedelta(days=200), date.today())))
                entry = Entry(setup_date=date.fromisoformat(e["setup_date"]),
                              entry_date=date.fromisoformat(e["entry_date"]),
                              entry_price=e["entry_price"], pivot=e["pivot"],
                              breakout_volume_ratio=e["breakout_volume_ratio"], stage4=None)  # type: ignore[arg-type]
                tr = simulate_trade(df_full, entry)
                taken.append({"id": did, "ticker": v["ticker"], "entry_date": e["entry_date"],
                              "exit": tr.exit_reason, "pnl": round(tr.pnl, 4), "days": tr.days_held})
            acct = sum(t["pnl"] for t in taken) * WEIGHT
            rows.append({"as_of": d, "regime": day["regime"], "mode": mode, "signals": len(ents),
                         "trades": taken, "account_return": round(acct, 4),
                         "qqq_6m": None if bench is None else round(bench, 4)})
    out = cfg.get("out", "/tmp/mcb_results.json")
    with open(out, "w") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
    for r in rows:
        t = r["trades"]
        print(f"{r['as_of']} {r['regime']:10s} {r['mode']:6s} 신호 {r['signals']:2d} 매수 {len(t)} "
              f"익절/손절/보유 {sum(x['exit'] == 'target' for x in t)}/{sum(x['exit'] == 'stop' for x in t)}/"
              f"{sum(x['exit'] in ('open', 'time') for x in t)} 계좌 {r['account_return']:+.2%} QQQ6M {r['qqq_6m']}"
              + "  " + " ".join(f"{x['ticker']}:{x['pnl']:+.0%}" for x in t))


if __name__ == "__main__":
    {"prepare": prepare, "results": results, "live": live}[sys.argv[1]](json.loads(sys.argv[2]))
