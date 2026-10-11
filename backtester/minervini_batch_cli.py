"""미너비니 스크리너 배치 검증 CLI — 기준일 여러 개를 44 페이지와 같은 설정으로 돌린다.

각 기준일마다:
  1. ``run_screen`` (STAGE 0 게이트 ON — 페이지 기본값)
  2. 게이트로 스캔이 생략되면 ``--gate-off`` 일 때 게이트 OFF 로 한 번 더
     (규칙을 어기고 샀다면 어땠나)
  3. 후보별 forward 시뮬레이션(피봇 돌파 → 손절 8% / 익절 22% 선착, 1/3/6M)
  4. 수동 체크리스트 자동 판정 (0-4·2-7·2-9·3-1~3-4·4-1·4-10)
  5. 같은 기간 SPY·QQQ 수익률 (벤치마크)

Run inside the docker container:
    docker compose exec backtester python minervini_batch_cli.py \\
        '{"dates": ["2020-01-02", "2022-06-16"], "gate_off": true,
          "out": "/tmp/minervini_batch.json"}'

진행 로그는 stderr, 결과 JSON 은 ``out`` 경로(기본 /tmp/minervini_batch.json)와
요약 표는 stdout.
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from datetime import date, timedelta

import pandas as pd

from data.adapters.composed_market_data import build_default_market_data
from data.adapters.wikipedia_universe import default_universe_provider
from pages._shared.growth_sources import build_growth_adapters
from strategy.adapters.minervini_manual_auto import industry_rank, manual_auto
from strategy.adapters.minervini_scan import (
    ScreenParams,
    ScreenResult,
    fetch_with_retry,
    run_screen,
)
from strategy.adapters.minervini_screener import (
    TRADING_DAYS,
    evaluate_balance,
    evaluate_market,
    evaluate_surprise,
    simulate_forward,
)


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _market_data(cfg: dict):
    """일봉 소스 선택 (composition root).

    기본 ``alphavantage`` = Mongo 캐시된 Alpha Vantage **단독** — 수천 종목
    배치에서 AV 에 없는 심볼(워런트·유닛 등)이 yfinance 로 넘어가 429 를
    유발하지 않게 한다. ``"default"`` 면 페이지와 같은 팩토리(AV → yfinance …).
    """
    if cfg.get("daily_source", "alphavantage") == "default":
        return build_default_market_data()
    from data.adapters.alphavantage_adapter import AlphaVantageAdapter
    from data.adapters.mongo_day_cache import MongoDayCacheAdapter
    return MongoDayCacheAdapter(AlphaVantageAdapter(), source_name="alphavantage")


def _splits_provider():
    try:
        from data.adapters.cached_fundamentals import CachedFundamentalsAdapter
        from data.adapters.eodhd_fundamentals import EODHDFundamentalsAdapter
        from data.adapters.fallback_fundamentals import FallbackFundamentalsAdapter
        from data.adapters.massive_fundamentals import MassiveFundamentalsAdapter
        try:
            src = FallbackFundamentalsAdapter(
                EODHDFundamentalsAdapter(), MassiveFundamentalsAdapter(),
                primary_label="eodhd", fallback_label="massive",
            )
        except ValueError:
            src = MassiveFundamentalsAdapter()
        return CachedFundamentalsAdapter(src)
    except Exception:  # noqa: BLE001
        return None


def _universe(provider, choice: str, as_of: date) -> list[str]:
    if choice.endswith("@기준일"):
        return provider.get_tickers(f"{choice[:-len('@기준일')]}@{as_of.isoformat()}")
    if choice == "sp500+nasdaq100":
        return list(dict.fromkeys(provider.get_tickers("sp500") + provider.get_tickers("nasdaq100")))
    return provider.get_tickers(choice)


def _forward_df(md, t: str, as_of: date) -> pd.DataFrame:
    end = min(as_of + timedelta(days=430), date.today())
    if end <= as_of:
        return pd.DataFrame()
    df = fetch_with_retry(md, t, as_of + timedelta(days=1), end)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df[df.index > pd.Timestamp(as_of)]


def _bench(md, sym: str, as_of: date) -> dict:
    try:
        base = fetch_with_retry(md, sym, as_of - timedelta(days=10), as_of)
        c0 = float(base["Close"].iloc[-1])
        fwd = _forward_df(md, sym, as_of)
    except Exception:  # noqa: BLE001
        return {}
    out = {}
    for k in ("1m", "3m", "6m"):
        n = TRADING_DAYS[k]
        out[k] = float(fwd["Close"].iloc[n - 1]) / c0 - 1.0 if len(fwd) >= n else None
    return out


def _r(v, nd=4):
    return None if v is None else round(float(v), nd)


def _b(v):
    """numpy.bool_ 은 json 이 str 로 떨어뜨린다 → 파이썬 bool 로."""
    return None if v is None else bool(v)


def _evaluate(res: ScreenResult, md, growth_adapters, params: ScreenParams) -> list[dict]:
    rows: list[dict] = []
    ranks = industry_rank([(c.ticker, c.industry, c.rs) for c in res.candidates])
    bal_adapter = next((a for a in (growth_adapters or []) if hasattr(a, "fetch_balance_sheet")), None)
    qqq = res.index_frames.get("QQQ")
    for c in res.candidates:
        try:
            out = simulate_forward(_forward_df(md, c.ticker, res.as_of), pivot=c.pl.pivot,
                                   stop_pct=params.stop_pct, target_pct=params.target_pct)
        except Exception:  # noqa: BLE001
            out = None
        ma = manual_auto(res.frames[c.ticker], qqq, c.pl) if qqq is not None else None
        checks = dict(ma.checks) if ma else {}
        notes = dict(ma.notes) if ma else {}
        checks["m3_4"], notes["m3_4"] = ranks.get(c.ticker, (None, ""))
        quarters = c.snapshot.quarters if c.snapshot else []
        sr = evaluate_surprise(quarters, res.as_of)
        checks["m2_7"] = sr.latest_beat if sr.data_available else None
        notes["m2_7"] = f"연속 비트 {sr.streak}회" if sr.data_available else "추정치 없음"
        if bal_adapter is not None and c.entry_ok:
            try:
                br = evaluate_balance(bal_adapter.fetch_balance_sheet(c.ticker), quarters, res.as_of)
                checks["m2_9"] = br.passed
                notes["m2_9"] = (f"매출 {_r(br.revenue_yoy, 2)} 재고 {_r(br.inventory_yoy, 2)} "
                                 f"매출채권 {_r(br.receivables_yoy, 2)}") if br.data_available else "데이터 부족"
            except Exception as e:  # noqa: BLE001
                checks["m2_9"], notes["m2_9"] = None, f"조회 실패 {type(e).__name__}"
        g, a, pl = c.growth, c.annual, c.pl
        rows.append({
            "ticker": c.ticker, "score": _r(c.score, 1), "rs": _r(c.rs, 0),
            "industry": c.industry, "entry_ok": c.entry_ok, "entry_note": c.entry_note,
            "industry_top": _b(c.industry_top),
            "close": _r(pl.close, 2), "pivot": _r(pl.pivot, 2), "dist_to_pivot": _r(pl.dist_to_pivot),
            "pivot_method": pl.method, "pivot_status": pl.status,
            "contractions": [_r(x, 3) for x in pl.contractions], "vcp_ok": _b(pl.contraction_ok),
            "base_depth": _r(pl.base_depth, 3), "base_weeks": _r(pl.base_weeks, 1),
            "tight10d": _r(pl.tightness_10d, 3), "vol_dryup": _r(pl.volume_dryup, 2),
            "pct_from_high": _r(c.tt.pct_from_high), "pct_above_low": _r(c.tt.pct_above_low),
            "eps_yoy": _r(getattr(g, "eps_yoy", None)), "eps_hist": [_r(x) for x in getattr(g, "eps_yoy_history", [])],
            "rev_yoy": _r(getattr(g, "revenue_yoy", None)),
            "growth_pass": (f"{g.n_passed}/4" if g and g.data_available else "미확인"),
            "code33": (f"{a.n_passed}/3" if a and a.data_available else "미확인"),
            "annual_decel": _b(a.decel_warning if a and a.data_available else None),
            "manual": {k: _b(checks[k]) for k in sorted(checks)},
            "manual_notes": {k: notes.get(k, "") for k in sorted(checks)},
            "fwd": None if out is None else {
                "entered": out.entered, "entry_date": str(out.entry_date) if out.entry_date else None,
                "entry_price": _r(out.entry_price, 2), "first_hit": out.first_hit,
                "hit_date": str(out.hit_date) if out.hit_date else None,
                "ret_1m": _r(out.ret_1m), "ret_3m": _r(out.ret_3m), "ret_6m": _r(out.ret_6m),
                "max_runup": _r(out.max_runup), "max_drawdown": _r(out.max_drawdown),
            },
        })
    return rows


def _summ(rows: list[dict], only_entry: bool) -> dict:
    rs = [r for r in rows if (r["entry_ok"] or not only_entry)]
    ent = [r for r in rs if r["fwd"] and r["fwd"]["entered"]]

    def mean(k):
        v = [r["fwd"][k] for r in ent if r["fwd"][k] is not None]
        return _r(sum(v) / len(v)) if v else None

    return {"n": len(rs), "entered": len(ent),
            "target": sum(r["fwd"]["first_hit"] == "target" for r in ent),
            "stop": sum(r["fwd"]["first_hit"] == "stop" for r in ent),
            "open": sum(r["fwd"]["first_hit"] == "none" for r in ent),
            "avg_1m": mean("ret_1m"), "avg_3m": mean("ret_3m"), "avg_6m": mean("ret_6m")}


def _market(res: ScreenResult) -> dict:
    slopes = {}
    for sym, idf in res.index_frames.items():
        for n in (10, 15, 20):
            v = evaluate_market(idf, symbol=sym, short_slope_days=n)
            if v is not None:
                slopes[f"{sym}_{n}"] = bool(v.short_ok)
    return {
        "failed": res.market_failed, "warns": res.market_warns, "nh": res.nh, "nl": res.nl,
        "nh_prev": res.nh_prev, "nl_prev": res.nl_prev, "short_ok_by_slope": slopes,
        "nh_liq": res.nh_liq, "nl_liq": res.nl_liq,
        "nh_liq_prev": res.nh_liq_prev, "nl_liq_prev": res.nl_liq_prev,
        "indexes": [{"symbol": v.symbol, "close": _r(v.close, 2), "sma50": _r(v.sma50, 2),
                     "sma50_prev": _r(v.sma50_prev, 2), "sma200": _r(v.sma200, 2),
                     "trend_ok": v.trend_ok, "short_ok": v.short_ok, "dist_days": v.dist_days}
                    for v in res.market],
    }


def prefetch(md, tickers: list[str], start: date, end: date, workers: int) -> int:
    """전 종목을 **한 번에 전체 구간**으로 조회해 Mongo 일 단위 캐시를 채운다.

    날짜별로 따로 조회하면 종목마다 기준일 수만큼 upstream 콜이 나간다
    (캐시는 요청 구간만 저장). Alpha Vantage 일봉은 1콜 = 전체 히스토리라
    먼저 한 번 받아두면 이후 기준일별 조회는 전부 캐시 히트.
    """
    from concurrent.futures import ThreadPoolExecutor

    def _one(t: str) -> bool:
        try:
            fetch_with_retry(md, t, start, end)
            return True
        except Exception:  # noqa: BLE001
            return False

    ok = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, r in enumerate(ex.map(_one, tickers)):
            ok += r
            if (i + 1) % 500 == 0:
                _log(f"  prefetch {i + 1}/{len(tickers)} (ok {ok})")
    return ok


def run(cfg: dict) -> dict:
    md = _market_data(cfg)
    uni = default_universe_provider()
    growth_adapters = build_growth_adapters()
    splits = _splits_provider()
    base = ScreenParams(**cfg.get("params", {}))
    choice = cfg.get("universe", "nasdaq_full@기준일")
    universes = {d: _universe(uni, choice, date.fromisoformat(d)) for d in cfg["dates"]}
    if cfg.get("max_tickers"):
        universes = {d: t[: int(cfg["max_tickers"])] for d, t in universes.items()}
    if cfg.get("prefetch", True):
        allt = sorted({t for ts in universes.values() for t in ts} | {"SPY", "QQQ"})
        first = min(date.fromisoformat(d) for d in cfg["dates"]) - timedelta(days=700)
        _log(f"prefetch {len(allt)} 종목 {first} ~ {date.today()}")
        ok = prefetch(md, allt, first, date.today(), int(cfg.get("workers", 2)))
        _log(f"prefetch 완료 ok {ok}/{len(allt)}")
    results = []
    for d in cfg["dates"]:
        as_of = date.fromisoformat(d)
        tickers = universes[d]
        _log(f"[{d}] 유니버스 {len(tickers)}")
        entry: dict = {"as_of": d, "universe_n": len(tickers),
                       "bench": {s: _bench(md, s, as_of) for s in ("SPY", "QQQ")}}
        runs = (("gate_on", base), ("gate_off", replace(base, stage0_gate=False)))
        if cfg.get("gate_off_only"):  # OOS: 게이트 판정은 저장된 지표로 사후 적용
            runs = runs[1:]
        for label, params in runs:
            if (label == "gate_off" and not cfg.get("gate_off_only")
                    and not (cfg.get("gate_off") and entry["gate_on"]["skipped"])):
                continue
            res = run_screen(as_of, tickers, md, growth_adapters, splits, params,
                             max_workers=int(cfg.get("workers", 2)), log=_log)
            rows = _evaluate(res, md, growth_adapters, params) if not res.skipped else []
            f = res.funnel
            fail_ratio = f.get("조회 실패", 0) / max(f.get("유니버스", 1), 1)
            entry[label] = {"skipped": res.skipped,
                            "valid": fail_ratio <= float(cfg.get("max_fail_ratio", 0.2)),
                            "fetch_fail_ratio": _r(fail_ratio, 3), "market": _market(res), "funnel": res.funnel,
                            "summary_all": _summ(rows, False), "summary_entry": _summ(rows, True),
                            "candidates": rows}
            _log(f"[{d}] {label}: skipped={res.skipped} 후보 {len(rows)} "
                 f"진입가능 {sum(r['entry_ok'] for r in rows)}")
        results.append(entry)
        with open(cfg.get("out", "/tmp/minervini_batch.json"), "w") as f:  # incremental dump
            json.dump(results, f, ensure_ascii=False, indent=1, default=str)
    return {"results": results}


def _print_table(results: list[dict]) -> None:
    print("기준일 | 게이트 | 시장 | 후보/진입가능 | 진입가능 돌파 · 익절/손절/보유 | 평균 3M · 6M | QQQ 3M · 6M")
    for e in results:
        for label in ("gate_on", "gate_off"):
            if label not in e:
                continue
            x = e[label]
            m = x["market"]
            mk = " · ".join(m["failed"] + m["warns"]) or "정상"
            s = x["summary_entry"]
            q = e["bench"].get("QQQ", {})
            mk += "" if x.get("valid", True) else f" ⚠️무효(조회실패 {x['fetch_fail_ratio']:.0%})"
            print(f"{e['as_of']} | {label} | {mk} | {x['summary_all']['n']}/{s['n']} | "
                  f"{s['entered']} · {s['target']}/{s['stop']}/{s['open']} | "
                  f"{s['avg_3m']} · {s['avg_6m']} | {q.get('3m')} · {q.get('6m')}")


_TRUE = (True, "True")  # 구버전 결과 파일은 bool 이 문자열로 저장돼 있다


def trades_frame(results: list[dict]) -> pd.DataFrame:
    """후보 → 거래 1행. 손익 = 익절 +target / 손절 −stop 선착, 미청산은 3M(없으면 1M) 수익률."""
    rows = []
    for e in results:
        for label in ("gate_on", "gate_off"):
            x = e.get(label)
            if not x or x["skipped"]:
                continue
            for i, c in enumerate(x["candidates"]):
                f = c["fwd"] or {}
                if not f.get("entered"):
                    continue
                pnl = {"target": 0.22, "stop": -0.08}.get(f["first_hit"])
                if pnl is None:
                    pnl = f["ret_3m"] if f["ret_3m"] is not None else f["ret_1m"]
                row = {"date": e["as_of"], "gate": label, "pos": i + 1,
                       "ticker": c["ticker"], "entry_ok": c["entry_ok"] in _TRUE,
                       "hit": f["first_hit"], "pnl": pnl, "rs": c["rs"] or 0.0,
                       "growth": c["growth_pass"], "vcp": c["vcp_ok"] in _TRUE,
                       "dist": c["dist_to_pivot"]}
                row.update({k: v in _TRUE for k, v in c["manual"].items()})
                rows.append(row)
    return pd.DataFrame(rows)


def quality_mask(df: pd.DataFrame) -> pd.Series:
    """품질 필터 — RS ≥ 90 · 분기 성장 4/4 · 3-4 산업군 1~2위 (샘플 백테스트 최고 조합)."""
    return (df["rs"] >= 90) & (df["growth"] == "4/4") & df.get("m3_4", False)


def analyze(results: list[dict]) -> None:
    """규칙 변형별 성과 + 항목별 예측력 (게이트 ON 날짜만)."""
    df = trades_frame(results)
    on = df[df["gate"] == "gate_on"]

    def line(name: str, s: pd.DataFrame, rest: pd.DataFrame | None = None) -> None:
        txt = (f"{name:30s} n={len(s):3d} 승률 {(s.hit == 'target').mean():.0%} "
               f"손절 {(s.hit == 'stop').mean():.0%} 거래당 {s.pnl.mean():+.2%}")
        if rest is not None:
            txt += f" | 나머지 n={len(rest):3d} 거래당 {rest.pnl.mean():+.2%}"
        print(txt)

    print("== 게이트 OFF (게이트가 막은 날에 샀다면) ==")
    for d, g in df[df["gate"] == "gate_off"].groupby("date"):
        line(d, g)
    print("== 진입 규칙 변형 (게이트 ON) ==")
    line("A 현재 규칙 (진입 가능만)", on[on.entry_ok])
    line("B 전체 후보", on)
    line("D 품질 필터", on[quality_mask(on)])
    print("== 항목별 예측력 (통과 vs 나머지) ==")
    checks = {"종합점수 상위3": on.pos <= 3, "RS≥90": on.rs >= 90, "성장 4/4": on.growth == "4/4",
              "VCP 실측": on.vcp, "피봇 아래(dist≤0)": on.dist <= 0}
    checks.update({k: on[k] for k in ("m0_4", "m2_7", "m2_9", "m3_1", "m3_2", "m3_3",
                                       "m3_4", "m4_1", "m4_10") if k in on})
    for name, m in checks.items():
        m = m.fillna(False).astype(bool)
        line(name, on[m], on[~m])


def oos_analyze(results: list[dict]) -> None:
    """변형 규칙 비교 — 게이트 OFF 로 저장된 후보에 시장·진입 규칙을 사후 적용.

    게이트: 0-1(200일선) 전 지수 통과 + 0-1b(50일선 기울기 ``slope``일) 전 지수 통과.
    0-3: level(신고가>신저가) | ratio_trend. 진입: old(경고 시 종합점수 상위 3, 정상은 전부)
    | quality(``entry_permissions``). 게이트에 막힌 날은 현금(거래 0).
    """
    from strategy.adapters.minervini_scan import EntryInput, breadth_ok, entry_permissions

    def day_trades(e: dict, slope: int, breadth_rule: str, entry_rule: str) -> list[float] | None:
        x = e["gate_off"]
        m = x["market"]
        trend = all(i["trend_ok"] for i in m["indexes"])
        short = all(m["short_ok_by_slope"].get(f"{i['symbol']}_{slope}", True) for i in m["indexes"])
        if not (trend and short):
            return None  # 게이트 → 현금
        b = breadth_ok(m["nh"], m["nl"], m.get("nh_prev"), m.get("nl_prev"), rule=breadth_rule)
        cands = x["candidates"]
        if entry_rule == "old":
            if not b:
                return []
            allowed = cands[:3] if m["warns"] else cands
        else:
            items = [EntryInput(rs=c["rs"], growth_full=c["growth_pass"] == "4/4",
                                verified=c["growth_pass"] != "미확인",
                                industry_top=c.get("industry_top", c["manual"].get("m3_4")) in _TRUE)
                     for c in cands]
            perms = entry_permissions(items, b, m["warns"], 3)
            allowed = [c for c, (ok, _) in zip(cands, perms) if ok]
        out = []
        for c in allowed:
            f = c["fwd"] or {}
            if not f.get("entered"):
                continue
            pnl = {"target": 0.22, "stop": -0.08}.get(f["first_hit"])
            out.append(pnl if pnl is not None else (f["ret_3m"] if f["ret_3m"] is not None else f["ret_1m"] or 0.0))
        return out

    print("변형 | 현금일 | 거래 | 승률 | 거래당 | 날짜당 평균(포트 5종목·20%) | 손실 날짜")
    for slope in (10, 15, 20):
        for br in ("level", "ratio_trend"):
            for er in ("old", "quality"):
                cash, allp, per_day, losing = 0, [], [], 0
                for e in results:
                    t = day_trades(e, slope, br, er)
                    if t is None:
                        cash += 1
                        per_day.append(0.0)
                        continue
                    allp += t
                    port = sum(t[:5]) * 0.20  # 진입 순서 근사: 정렬 순 상위 5
                    per_day.append(port)
                    losing += port < 0
                n = len(allp)
                win = sum(p >= 0.22 for p in allp) / n if n else 0
                print(f"slope{slope} {br:11s} {er:7s} | {cash:2d} | {n:4d} | {win:.0%} | "
                      f"{(sum(allp) / n if n else 0):+.2%} | {sum(per_day) / len(per_day):+.2%} | {losing}")


def breadth_analyze(results: list[dict]) -> None:
    """0-3 처리 방식 비교 — 게이트(0-1·0-1b 20일) 통과일에서 0-3 을 어떻게 쓸 것인가.

    집계 대상: all(조회 전 종목) / liq(가격·거래대금 프리필터 통과 종목, 기록이 있는 날짜만)
    처리: ignore(무시) / caution(0-2 경고처럼 RS 상위 3종목) / block(진입 0)
    진입 후보 = 품질 필터(``entry_permissions``). 포트 = 진입 순 상위 5(경고 시 3) × 20%.
    """
    from strategy.adapters.minervini_scan import EntryInput, entry_permissions

    days = []
    for e in results:
        x = e.get("gate_off")
        if not x:
            continue
        m = x["market"]
        trend = all(i["trend_ok"] for i in m["indexes"])
        short = all(m["short_ok_by_slope"].get(f"{i['symbol']}_20", True) for i in m["indexes"])
        cands = x["candidates"]
        items = [EntryInput(rs=c["rs"], growth_full=c["growth_pass"] == "4/4",
                            verified=c["growth_pass"] != "미확인",
                            industry_top=c.get("industry_top", c["manual"].get("m3_4")) in _TRUE)
                 for c in cands]
        pn = []
        for c in cands:
            f = c["fwd"] or {}
            if not f.get("entered"):
                pn.append(None)
                continue
            p = {"target": 0.22, "stop": -0.08}.get(f["first_hit"])
            pn.append(p if p is not None else (f["ret_3m"] if f["ret_3m"] is not None else f["ret_1m"] or 0.0))
        days.append({"date": e["as_of"], "gate": trend and short, "warns": bool(m["warns"]),
                     "b_all": m["nh"] > m["nl"],
                     "b_liq": (m["nh_liq"] > m["nl_liq"]) if m.get("nh_liq") is not None else None,
                     "items": items, "pnl": pn, "qqq3m": e["bench"].get("QQQ", {}).get("3m")})

    def run(universe: str, action: str, subset=None):
        trades, ports = [], []
        for d in days:
            if subset is not None and not subset(d):
                continue
            if not d["gate"]:
                ports.append(0.0)
                continue
            b = d["b_all"] if universe == "all" else d["b_liq"]
            if b is None:
                continue
            warn = d["warns"] or (not b and action == "caution")
            if not b and action == "block":
                ports.append(0.0)
                continue
            perms = entry_permissions(d["items"], True, ["w"] if warn else [], 3)
            t = [p for p, (ok, _) in zip(d["pnl"], perms) if ok and p is not None]
            trades += t
            ports.append(sum(t[: 3 if warn else 5]) * 0.20)
        n = len(trades)
        return (len(ports), n, (sum(trades) / n if n else 0.0),
                (sum(ports) / len(ports) if ports else 0.0), sum(p < 0 for p in ports))

    liq_days = [d for d in days if d["b_liq"] is not None]
    print(f"날짜 {len(days)} (유동성 기록 {len(liq_days)}) · 게이트 통과 {sum(d['gate'] for d in days)}")
    print("집계 | 처리 | 날짜 | 거래 | 거래당 | 날짜당 포트 | 손실 날짜")
    for uni in ("all", "liq"):
        for act in ("ignore", "caution", "block"):
            sub = (lambda d: d["b_liq"] is not None)
            nd, n, pt, pp, lose = run(uni, act, sub)
            print(f"{uni:3s} | {act:7s} | {nd:3d} | {n:4d} | {pt:+.2%} | {pp:+.2%} | {lose}")
    print("\n0-3 이 갈린 날 (게이트 통과, 유동성 기록 있음):")
    for d in liq_days:
        if d["gate"] and not (d["b_all"] and d["b_liq"]):
            t = [p for p, (ok, _) in zip(d["pnl"], entry_permissions(d["items"], True, ["w"] if d["warns"] else [], 3))
                 if ok and p is not None]
            print(f"  {d['date']} 전체 {'✅' if d['b_all'] else '❌'} 유동성 {'✅' if d['b_liq'] else '❌'} | "
                  f"품질 진입 {len(t)}건 평균 {(sum(t) / len(t) if t else 0):+.1%} | QQQ 3M {d['qqq3m']}")


if __name__ == "__main__":
    cfg = json.loads(sys.argv[1])
    if "analyze" in cfg:  # 기존 결과 파일만 분석: '{"analyze": "/tmp/minervini_batch.json"}'
        with open(cfg["analyze"]) as f:
            analyze(json.load(f))
    elif "breadth_analyze" in cfg:  # 여러 결과 파일 합쳐 0-3 처리 비교
        merged = []
        for path in cfg["breadth_analyze"]:
            with open(path) as f:
                merged += json.load(f)
        breadth_analyze(merged)
    elif "oos_analyze" in cfg:  # gate_off_only 결과의 변형 비교
        with open(cfg["oos_analyze"]) as f:
            oos_analyze(json.load(f))
    else:
        out = run(cfg)
        if cfg.get("gate_off_only"):
            oos_analyze(out["results"])
        else:
            _print_table(out["results"])
            analyze(out["results"])
