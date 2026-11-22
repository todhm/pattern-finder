"""필립 피셔 『위대한 기업에 투자하라』 15 포인트 — 1년 전 시점 나스닥 종목 선별 검증 배치.

체크리스트 `docs/strategy_notes/피셔_위대한기업_체크리스트_2026-11.md`. 단계(순서대로, 앞 단계 결과를 out 폴더에서 읽는다):

    python fisher_cli.py universe '{"as_of": "2025-11-19"}'     # 기준일 나스닥 상장 종목 → 유동성·재무제표·업종·매출 이력 거름 + 수치 포인트 점수
    python fisher_cli.py dossier  '{"top": 30, "control": 15}'  # 수치 상위 top + 무작위 대조 control 의 10-K·위임장·8-K 목록 수집(SEC EDGAR)
    python fisher_cli.py judge    '{"model": "claude-opus-5-5"}' # 종목마다 15 포인트 문서 판독(Claude, 디스크 캐시)
    python fisher_cli.py settle   '{}'                          # 통과 종목 상위 10 을 기준일에 사서 지금까지 보유한 수익률 + 대조군 비교 → md/json

엔진 `strategy/adapters/fisher_checklist.py`(수치·집계) · `strategy/adapters/fisher_judge.py`(판독) · `data/adapters/sec_edgar_filings.py`(공시 원문).
"""

from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from strategy.adapters import fisher_checklist as fc

ROOT = Path(__file__).resolve().parent / "sweep_results"
STMT_FILES = ("growth_statements_2026-11-09.parquet", "growth_statements_small_2026-11-09.parquet")
log = lambda *a: print(*a, file=sys.stderr, flush=True)  # noqa: E731


def _out(as_of: str) -> Path:
    p = ROOT / f"fisher_{as_of}"; p.mkdir(parents=True, exist_ok=True)
    return p


def _state(cfg: dict) -> tuple[str, Path]:
    as_of = cfg.get("as_of") or sorted(d.name.split("_", 1)[1] for d in ROOT.glob("fisher_20*"))[-1]
    return as_of, _out(as_of)


# ---------------------------------------------------------------------------
def cmd_universe(cfg: dict) -> None:
    as_of = cfg.get("as_of", "2025-11-19"); out = _out(as_of); T = pd.Timestamp(as_of)
    from data.adapters.alphavantage_listing_universe import AlphaVantageListingUniverseAdapter

    listed = AlphaVantageListingUniverseAdapter().get_tickers(f"nasdaq_full@{as_of}")
    log(f"기준일 {as_of} 나스닥 상장(ETF 제외): {len(listed)}")
    panel = pd.read_parquet(ROOT / "surge_panel.parquet", columns=["date", "symbol", "C", "raw_c", "p_dv20", "p_dd252"],
                            filters=[("date", ">=", T - pd.Timedelta(days=7)), ("date", "<=", T)])
    px = panel.sort_values("date").groupby("symbol").last()
    min_px, min_dv = float(cfg.get("min_price", 5.0)), float(cfg.get("min_dollar_volume", 5e6))
    liquid = px[(px["raw_c"] >= min_px) & (px["p_dv20"] >= min_dv)].index
    stm = pd.concat([pd.read_parquet(ROOT / f) for f in STMT_FILES], ignore_index=True).drop_duplicates(["symbol", "fiscal_date"], keep="first")
    stm_asof = fc.asof(stm, T)
    stm_asof = stm_asof[stm_asof["fiscal_date"] >= T - pd.Timedelta(days=365 * 7)]
    f = fc.quant_features(stm_asof)
    fresh = f["last_fiscal"] >= T - pd.Timedelta(days=200)          # 최근 분기가 반년 넘게 비어 있으면 자료가 낡은 것
    funnel = {"나스닥 상장": len(listed)}
    u = f[f.index.isin(listed)]; funnel["재무제표 있음"] = len(u)
    u = u[u.index.isin(liquid)]; funnel[f"주가 ≥ ${min_px:g}·거래대금 ≥ ${min_dv / 1e6:g}M"] = len(u)
    u = u[fresh.reindex(u.index).fillna(False)]; funnel["최근 분기 발표 있음"] = len(u)
    u = u[~u["sector"].isin(fc.EXCLUDED_SECTORS) & u["sector"].notna()]; funnel["금융·부동산 제외"] = len(u)
    u = u[(u["rev_ttm"] >= float(cfg.get("min_rev", 1e8))) & (u["n_quarters"] >= int(cfg.get("min_quarters", 16)))]; funnel["매출 ≥ $100M·이력 4년 이상(선전 단계 제외)"] = len(u)
    sc = fc.quant_scores(u)                                          # 섹터 내 순위는 걸러진 모집단 안에서
    u = u.join(sc).join(px[["raw_c", "p_dv20", "p_dd252"]])
    u = u[u["quant_n"] >= 4]; funnel["수치 포인트 4개 이상 계산됨"] = len(u)
    u = u.sort_values(["quant_avg", "rev_cagr3"], ascending=False)
    u.to_parquet(out / "universe.parquet")
    (out / "funnel.json").write_text(json.dumps({"as_of": as_of, "funnel": funnel}, ensure_ascii=False, indent=1))
    log(json.dumps(funnel, ensure_ascii=False))
    print(u.head(40)[["sector", "rev_ttm", "rev_cagr3", "rev_cagr5", "om_ttm", "om_chg3", "rnd_to_rev", "sh_chg3_ann", "p1", "p3", "p5", "p6", "p12", "p13", "quant_avg"]].round(3).to_string())


# ---------------------------------------------------------------------------
def _fmt(v, kind="pct"):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "자료 없음"
    return f"{v:+.1%}" if kind == "pct" else f"{v:.1%}" if kind == "share" else f"{v:,.2f}" if kind == "x" else f"${v / 1e6:,.0f}M"


def quant_block(r: pd.Series, ranks: dict[str, float]) -> str:
    pr = lambda k: f" (섹터 내 상위 {100 - ranks[k] * 100:.0f}%)" if k in ranks and np.isfinite(ranks[k]) else ""  # noqa: E731
    return "\n".join([
        f"- 섹터: {r['sector']} / 최근 12개월 매출 {_fmt(r['rev_ttm'], 'usd')} / 발표된 분기 수 {int(r['n_quarters'])}",
        f"- 매출 연평균 성장: 3년 {_fmt(r['rev_cagr3'])}, 5년 {_fmt(r['rev_cagr5'])}, 최근 1년 {_fmt(r['rev_yoy'])} / 최근 {int(r['growth_years_of'])}년 중 성장한 해 {int(r['growth_years'])}{pr('rev_cagr3')}",
        f"- 영업권 ÷ 총자산 {_fmt(r['goodwill_share'], 'share')} (인수로 큰 회사인지)",
        f"- 영업이익률: 최근 12개월 {_fmt(r['om_ttm'], 'share')}, 3년 평균 {_fmt(r['om_avg3'], 'share')}{pr('om_avg3')}, 3년 전 대비 {_fmt(r['om_chg3'])}p",
        f"- 매출총이익률 {_fmt(r['gm_ttm'], 'share')}, 3년 전 대비 {_fmt(r['gm_chg3'])}p",
        f"- R&D ÷ 매출 {_fmt(r['rnd_to_rev'], 'share')}, R&D 전년 대비 {_fmt(r['rnd_yoy'])}, 3년간 늘어난 매출총이익 ÷ 3년간 R&D = {_fmt(r['rnd_payoff'], 'x')}{pr('rnd_payoff')}",
        f"- 3년간 늘어난 매출 ÷ 늘어난 판관비 = {_fmt(r['rev_per_sga_chg'], 'x')}",
        f"- (설비투자 + R&D) ÷ 매출 {_fmt(r['reinvest_to_rev'], 'share')}{pr('reinvest_to_rev')} / 잉여현금흐름 ÷ 매출 {_fmt(r['fcf_margin'], 'share')}",
        f"- 주식 수 3년 연평균 변화 {_fmt(r['sh_chg3_ann'])} / 주식보상 ÷ 매출 {_fmt(r['sbc_to_rev'], 'share')} / 순현금 {_fmt(r['net_cash'], 'usd')} / 순부채 ÷ EBITDA {_fmt(r['netdebt_to_ebitda'], 'x')}",
    ])


def cmd_dossier(cfg: dict) -> None:
    as_of, out = _state(cfg); T = pd.Timestamp(as_of)
    from data.adapters.sec_edgar_filings import SecEdgarFilings, mask_company, split_items, window

    u = pd.read_parquet(out / "universe.parquet")
    top_n, ctl_n, seed = int(cfg.get("top", 30)), int(cfg.get("control", 15)), int(cfg.get("seed", 7))
    cik = pd.read_parquet(ROOT / "ind_sym_cik.parquet").drop_duplicates("symbol").set_index("symbol")["cik"]
    have = u[u.index.isin(cik.index)]
    have = have[~cik.reindex(have.index).duplicated(keep="first").to_numpy()]      # 같은 회사의 다른 주식 종류(GOOG/GOOGL)는 하나만
    top = list(have.index[:top_n])
    rest = have.iloc[top_n:]
    control = list(rest.sample(n=min(ctl_n, len(rest)), random_state=seed).index)
    picks = [(s, "top") for s in top] + [(s, "control") for s in control]
    ranks = {c: u.groupby("sector")[c].rank(pct=True) for c in ("rev_cagr3", "om_avg3", "rnd_payoff", "reinvest_to_rev")}
    sec = SecEdgarFilings(ROOT / "fisher_sec_cache")
    blind = bool(cfg.get("blind", True))
    meta = []
    ddir = out / "dossier"; ddir.mkdir(exist_ok=True)
    for n, (sym, grp) in enumerate(picks):
        try:
            info, rows = sec.filings(int(cik[sym]), since=(T - pd.Timedelta(days=365 * 3 + 30)).strftime("%Y-%m-%d"))
            before = [r for r in rows if r["filingDate"] <= as_of]
            annual = next((r for r in before if r["form"] in ("10-K", "10-K405", "10-KT")), None) or next((r for r in before if r["form"] in ("20-F", "40-F")), None)
            if annual is None:
                log(f"[{n + 1}/{len(picks)}] {sym}: 기준일 전 연차보고서 없음 — 건너뜀"); continue
            text = sec.document_text(int(cik[sym]), annual["accession"], annual["doc"])
            items = split_items(text)
            parts: list[tuple[str, str]] = []
            if annual["form"].startswith("10-K") and len(items.get("1", "")) > 3000:
                it1 = items["1"]
                parts.append(("Item 1. Business (앞부분)", it1[:34000]))
                hc = window(it1, r"human\s+capital", 5000)
                if hc and it1.find(hc[:200]) > 34000:
                    parts.append(("Item 1 중 Human Capital", hc))
                ex = window(text, r"(?:information\s+about\s+(?:our\s+)?executive\s+officers|executive\s+officers\s+of\s+the\s+(?:registrant|company))", 5000)
                if ex:
                    parts.append(("Executive Officers", ex))
                parts.append(("Item 1A. Risk Factors (앞부분만)", items.get("1a", "")[:9000]))
                parts.append(("Item 3. Legal Proceedings", items.get("3", "")[:5000]))
                parts.append(("Item 7. MD&A (앞부분)", items.get("7", "")[:34000]))
                parts.append(("Item 9A. Controls and Procedures", items.get("9a", "")[:5000]))
            else:                                                   # 20-F 등: 항목 구조가 달라 본문 앞쪽을 통째로
                import re as _re
                a0 = int(len(text) * 0.06)
                parts.append((f"{annual['form']} 본문 앞부분(사업·위험)", text[a0: a0 + 48000]))
                mds = [m.start() for m in _re.finditer(r"(?i)(?:management.{0,3}s\s+discussion\s+and\s+analysis\s+of\s+financial\s+condition|operating\s+and\s+financial\s+review)", text)]
                mds = [x for x in mds if x > a0 + 48000]
                if mds:
                    parts.append((f"{annual['form']} 중 경영진의 논의·분석(MD&A)", text[mds[0]: mds[0] + 40000]))
            proxy = next((r for r in before if r["form"] == "DEF 14A"), None)
            if proxy is not None:
                ptxt = sec.document_text(int(cik[sym]), proxy["accession"], proxy["doc"])
                rel = window(ptxt, r"(?:certain\s+relationships|related[\s\-]+(?:person|party)\s+transactions|transactions\s+with\s+related)", 7000)
                if rel:
                    parts.append((f"위임장(DEF 14A, {proxy['filingDate']}) 중 특수관계자 거래", rel))
            lo3 = (T - pd.Timedelta(days=365 * 3)).strftime("%Y-%m-%d")
            ev = [r for r in before if r["form"] == "8-K" and r["filingDate"] >= lo3]
            cnt = lambda code: [r["filingDate"] for r in ev if code in (r["items"] or "").split(",")]  # noqa: E731
            eightk = (f"- 임원·이사 선임·사임(5.02): {len(cnt('5.02'))}건 {cnt('5.02')[:12]}\n- 감사인 교체(4.01): {len(cnt('4.01'))}건 {cnt('4.01')}\n"
                      f"- 과거 재무제표 신뢰 불가(4.02, 재작성): {len(cnt('4.02'))}건 {cnt('4.02')}\n- 구조조정·사업 철수 비용(2.05): {len(cnt('2.05'))}건 {cnt('2.05')}\n"
                      f"- 중요한 손상(2.06): {len(cnt('2.06'))}건 {cnt('2.06')}")
            r = u.loc[sym]
            body = "\n\n".join(f"===== {title} =====\n{txt.strip()}" for title, txt in parts if txt and txt.strip())
            if blind:
                body = mask_company(body, [info.get("name", "")] + info.get("former", []), [sym] + [t for t in info.get("tickers", []) if t != sym])
            head = (f"기준일: {as_of}\n회사: {'[회사] (이름 가림)' if blind else info.get('name', sym) + ' (' + sym + ')'}\n업종(SEC 분류): {info.get('sic_desc', '')}\n"
                    f"연차보고서: {annual['form']} 제출일 {annual['filingDate']}\n\n"
                    f"[수치 요약 — 기준일까지 발표된 분기 재무제표로 도구가 계산]\n{quant_block(r, {k: float(v.get(sym, np.nan)) for k, v in ranks.items()})}\n\n"
                    f"[최근 3년 8-K 항목 — 기준일 전]\n{eightk}\n\n[공시 원문 발췌 — 모두 기준일 전에 제출됨]\n")
            (ddir / f"{sym}.txt").write_text(head + body, encoding="utf-8")
            meta.append({"symbol": sym, "group": grp, "name": info.get("name", ""), "sic_desc": info.get("sic_desc", ""), "annual_form": annual["form"], "annual_date": annual["filingDate"],
                         "proxy_date": proxy["filingDate"] if proxy else None, "chars": len(head) + len(body), "sections": [t for t, x in parts if x and x.strip()],
                         "n_502": len(cnt("5.02")), "n_402": len(cnt("4.02"))})
            log(f"[{n + 1}/{len(picks)}] {sym} ({grp}) {annual['form']} {annual['filingDate']} · {len(head) + len(body):,}자 · {len(parts)}부분")
        except Exception as ex:  # noqa: BLE001
            log(f"[{n + 1}/{len(picks)}] {sym}: 실패 {type(ex).__name__}: {str(ex)[:150]}")
    pd.DataFrame(meta).to_json(out / "dossier_meta.json", orient="records", force_ascii=False, indent=1)
    tot = sum(m["chars"] for m in meta)
    print(json.dumps({"dossiers": len(meta), "top": sum(m["group"] == "top" for m in meta), "control": sum(m["group"] == "control" for m in meta),
                      "total_chars": tot, "approx_input_tokens": int(tot / 3.6)}, ensure_ascii=False))


# ---------------------------------------------------------------------------
def cmd_judge(cfg: dict) -> None:
    as_of, out = _state(cfg)
    from strategy.adapters.fisher_judge import FisherJudge

    meta = json.loads((out / "dossier_meta.json").read_text())
    jd = FisherJudge(model=cfg.get("model", "claude-opus-5-5"), effort=cfg.get("effort", "medium"), cache_dir=ROOT / "fisher_judge_cache")
    only = set(cfg.get("only", []))
    todo = [m for m in meta if not only or m["symbol"] in only]

    def one(m):
        text = (out / "dossier" / f"{m['symbol']}.txt").read_text(encoding="utf-8")
        try:
            j, u = jd.judge(text)
            log(f"{m['symbol']}: {j.verdict} · {'캐시' if u.get('cached') else '$%.3f' % u['cost_usd']}")
            return m["symbol"], {"judgement": j.model_dump(), "usage": u}
        except Exception as ex:  # noqa: BLE001
            log(f"{m['symbol']}: 실패 {type(ex).__name__}: {str(ex)[:200]}")
            return m["symbol"], None

    with ThreadPoolExecutor(max_workers=int(cfg.get("workers", 3))) as ex:
        res = dict(ex.map(one, todo))
    prev = json.loads((out / "judgements.json").read_text()) if (out / "judgements.json").exists() else {}
    prev.update({k: v for k, v in res.items() if v is not None})
    (out / "judgements.json").write_text(json.dumps(prev, ensure_ascii=False, indent=1))
    paid = sum(v["usage"]["cost_usd"] for v in res.values() if v and not v["usage"].get("cached"))
    print(json.dumps({"judged": sum(v is not None for v in res.values()), "failed": [k for k, v in res.items() if v is None], "paid_usd": round(paid, 2)}, ensure_ascii=False))


# ---------------------------------------------------------------------------
def _returns_yf(symbols: list[str], start: date, end: date) -> tuple[pd.Series, pd.Timestamp]:
    from data.adapters.yfinance_bulk import download_many

    px = download_many(symbols, start - timedelta(days=5), end)
    rows = {}
    last = None
    for s, d in px.items():
        c = d["Close"].dropna()
        c0 = c[c.index <= pd.Timestamp(start)];
        if len(c0) and len(c) and c.index[-1] > pd.Timestamp(start):
            rows[s] = float(c.iloc[-1] / c0.iloc[-1] - 1); last = c.index[-1] if last is None else max(last, c.index[-1])
    return pd.Series(rows, dtype=float), last


def cmd_settle(cfg: dict) -> None:
    as_of, out = _state(cfg); T = pd.Timestamp(as_of)
    u = pd.read_parquet(out / "universe.parquet")
    meta = pd.DataFrame(json.loads((out / "dossier_meta.json").read_text())).set_index("symbol")
    J = json.loads((out / "judgements.json").read_text())
    n_pick = int(cfg.get("n_pick", 10))
    rows = []
    for sym, v in J.items():
        j = v["judgement"]
        sc = {p["point"]: p["score"] for p in j["points"]}
        vd = fc.fisher_verdict(sc, promotional=j["promotional"])
        rows.append({"symbol": sym, **{f"f{k}": sc.get(k) for k in range(1, 16)}, "fisher_avg": vd["avg"], "n_known": vd["n_known"], "n_zero": len(vd["zeros"]), "n_unknown": len(vd["unknown"]),
                     "passed": vd["passed"], "fail": "; ".join(vd["fail_reasons"]), "ai_verdict": j["verdict"], "promotional": j["promotional"], "tone_only": j["tone_only"],
                     "summary": j["summary"], "red_flags": j["red_flags"], "strengths": j["strengths"], "temporary_trouble": j["temporary_trouble"]})
    A = pd.DataFrame(rows).set_index("symbol").join(meta[["group", "name", "sic_desc", "annual_date"]]).join(u[["sector", "rev_ttm", "rev_cagr3", "om_ttm", "quant_avg", "p_dd252"]])
    # 수익률: 판독한 종목 + QQQ 는 yfinance 로 오늘까지, 모집단은 패널(상장폐지 포함)로 패널 끝까지
    today = date.fromisoformat(cfg["end"]) if cfg.get("end") else date.today()
    ret, last = _returns_yf(list(A.index) + ["QQQ"], T.date(), today)
    panel = pd.read_parquet(ROOT / "surge_panel.parquet", columns=["date", "symbol", "C"], filters=[("date", ">=", T - pd.Timedelta(days=7))])
    panel = panel[panel["symbol"].isin(u.index)].sort_values("date")
    p0 = panel[panel["date"] <= T].groupby("symbol")["C"].last(); p1 = panel.groupby("symbol")["C"].last(); pend = panel["date"].max()
    u_ret = (p1 / p0 - 1).reindex(u.index)
    A["ret"] = ret.reindex(A.index).fillna(u_ret.reindex(A.index))          # yfinance 에 없으면(상장폐지) 패널 마지막 가격
    qqq_last = float(ret.get("QQQ", np.nan))
    qq, _ = _returns_yf(["QQQ", "SPY"], T.date(), pend.date()); qqq_panel = float(qq.get("QQQ", np.nan))
    passed = A[A["passed"]].sort_values(["fisher_avg", "quant_avg"], ascending=False)
    picks = passed.head(n_pick)
    A["picked"] = A.index.isin(picks.index)

    def grp(name, s):
        s = s.dropna()
        return {"묶음": name, "종목 수": len(s), "평균": float(s.mean()) if len(s) else np.nan, "중앙": float(s.median()) if len(s) else np.nan,
                "오른 비율": float((s > 0).mean()) if len(s) else np.nan, "QQQ 이긴 비율": float((s > qqq_last).mean()) if len(s) else np.nan,
                "최고": float(s.max()) if len(s) else np.nan, "최악": float(s.min()) if len(s) else np.nan}

    table = [grp(f"피셔 통과 상위 {len(picks)} (최종 선택)", picks["ret"]), grp("피셔 통과 전체", passed["ret"]), grp("판독했으나 탈락", A[~A["passed"]]["ret"]),
             grp("수치 상위(top) 전체", A[A["group"] == "top"]["ret"]), grp("무작위 대조(control) 전체", A[A["group"] == "control"]["ret"]),
             grp("AI 결론 buy", A[A["ai_verdict"] == "buy"]["ret"]), grp("AI 결론 watch/pass", A[A["ai_verdict"] != "buy"]["ret"])]
    uq = u.assign(ret=u_ret)
    dec = uq.dropna(subset=["ret"]).assign(q=lambda d: pd.qcut(d["quant_avg"].rank(method="first"), 5, labels=["하위 20%", "20~40%", "40~60%", "60~80%", "상위 20%"]))
    quint = dec.groupby("q", observed=True)["ret"].agg(["count", "mean", "median"]).reset_index()
    rho_f, n_f = fc.spearman(A["fisher_avg"], A["ret"]); rho_q, n_q = fc.spearman(uq["quant_avg"], uq["ret"]); rho_qa, n_qa = fc.spearman(A["quant_avg"], A["ret"])
    per_point = []
    for k in range(1, 16):
        s = A[f"f{k}"]
        hi, lo = A[s >= 2]["ret"].dropna(), A[s <= 1]["ret"].dropna()
        per_point.append({"포인트": f"{k}. {fc.POINT_NAMES[k]}", "2~3점 수": len(hi), "2~3점 평균 수익": float(hi.mean()) if len(hi) else np.nan, "0~1점 수": len(lo),
                          "0~1점 평균 수익": float(lo.mean()) if len(lo) else np.nan, "모름": int(s.isna().sum()), "평균 점수": float(s.mean()) if s.notna().any() else np.nan})
    # 무작위 대조: 모집단에서 같은 수를 무작위로 2,000 번 뽑은 평균 수익률 분포 속 위치 (같은 가격 자료·같은 종료일 기준)
    rng = np.random.default_rng(int(cfg.get("seed", 7))); pool = uq["ret"].dropna().to_numpy()
    picks_panel = float(u_ret.reindex(picks.index).mean()) if len(picks) else np.nan
    rand = np.array([rng.choice(pool, size=max(1, len(picks)), replace=False).mean() for _ in range(2000)]) if len(pool) > len(picks) else np.array([np.nan])
    top_q = uq.sort_values("quant_avg", ascending=False).head(max(1, len(picks)))["ret"].mean()
    placebo = {"picks_mean_panel_basis": picks_panel, "random_mean": float(np.nanmean(rand)), "random_p5": float(np.nanpercentile(rand, 5)), "random_p95": float(np.nanpercentile(rand, 95)),
               "picks_percentile": float((rand < picks_panel).mean()) if np.isfinite(picks_panel) else np.nan, "top_quant_same_n_mean": float(top_q)}
    funnel = json.loads((out / "funnel.json").read_text())["funnel"]
    res = {"as_of": as_of, "end": str(last.date()) if last is not None else None, "panel_end": str(pend.date()), "qqq": qqq_last, "qqq_to_panel_end": qqq_panel,
           "universe_n": int(uq["ret"].notna().sum()), "universe_mean": float(uq["ret"].mean()), "universe_median": float(uq["ret"].median()),
           "universe_beat_qqq": float((uq["ret"] > qqq_panel).mean()), "funnel": funnel, "groups": table, "quant_quintiles": quint.to_dict("records"),
           "spearman": {"fisher_avg_vs_ret": [rho_f, n_f], "quant_avg_vs_ret_universe": [rho_q, n_q], "quant_avg_vs_ret_analysed": [rho_qa, n_qa]},
           "per_point": per_point, "placebo": placebo, "picks": list(picks.index), "cost_usd": round(sum(v["usage"].get("cost_usd", 0) for v in J.values()), 2)}
    (out / "settle.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    A.sort_values(["picked", "passed", "fisher_avg"], ascending=False).to_json(out / "analysed.json", orient="index", force_ascii=False, indent=1)
    pct = lambda v: "—" if v is None or not np.isfinite(v) else f"{v:+.1%}"  # noqa: E731
    L = [f"# 피셔 15 포인트 — {as_of} 선별, {res['end']} 까지 보유\n", f"QQQ {pct(qqq_last)} · 검토 모집단 {res['universe_n']}종목 평균 {pct(res['universe_mean'])}, 중앙 {pct(res['universe_median'])} "
         f"(모집단은 {res['panel_end']} 까지, 같은 기간 QQQ {pct(qqq_panel)})\n", "## 묶음별 수익률\n", "| 묶음 | 종목 수 | 평균 | 중앙 | 오른 비율 | QQQ 이긴 비율 | 최고 | 최악 |", "|---|---|---|---|---|---|---|---|"]
    L += [f"| {g['묶음']} | {g['종목 수']} | {pct(g['평균'])} | {pct(g['중앙'])} | {g['오른 비율']:.0%} | {g['QQQ 이긴 비율']:.0%} | {pct(g['최고'])} | {pct(g['최악'])} |" if g["종목 수"] else f"| {g['묶음']} | 0 | — | — | — | — | — | — |" for g in table]
    L += [f"\n무작위 대조(모집단에서 {len(picks)}종목을 2,000번 뽑음, {res['panel_end']} 까지): 평균 {pct(placebo['random_mean'])}, 5~95% 범위 {pct(placebo['random_p5'])} ~ {pct(placebo['random_p95'])}. "
          f"최종 선택은 같은 기준으로 {pct(placebo['picks_mean_panel_basis'])} → 무작위의 **{placebo['picks_percentile']:.0%}** 위치. 수치 점수 상위 {len(picks)}종목(AI 없음)은 {pct(placebo['top_quant_same_n_mean'])}."]
    L += ["\n## 최종 선택 종목\n", "| 종목 | 회사 | 섹터 | 피셔 평균 | 수치 평균 | 0점 | 모름 | 수익률 |", "|---|---|---|---|---|---|---|---|"]
    L += [f"| {s} | {r['name']} | {r['sector']} | {r['fisher_avg']:.2f} | {r['quant_avg']:.2f} | {r['n_zero']} | {r['n_unknown']} | {pct(r['ret'])} |" for s, r in picks.iterrows()]
    L += ["\n## 판독한 종목 전체\n", "| 종목 | 묶음 | 통과 | 피셔 평균 | AI 결론 | 수익률 | 탈락 이유 |", "|---|---|---|---|---|---|---|"]
    L += [f"| {s} | {r['group']} | {'✅' if r['passed'] else '—'} | {r['fisher_avg']:.2f} | {r['ai_verdict']} | {pct(r['ret'])} | {r['fail']} |" for s, r in A.sort_values('fisher_avg', ascending=False).iterrows()]
    L += ["\n## 점수와 수익률의 관계 (순위 상관)\n", f"- 피셔 평균 점수 vs 수익률 (판독 {n_f}종목): {rho_f:+.2f}", f"- 수치 점수 vs 수익률 (판독 {n_qa}종목): {rho_qa:+.2f}",
          f"- 수치 점수 vs 수익률 (모집단 {n_q}종목, AI 없음): {rho_q:+.2f}", "\n## 수치 점수 5분위별 수익률 (모집단 전체, AI 없음)\n", "| 분위 | 종목 수 | 평균 | 중앙 |", "|---|---|---|---|"]
    L += [f"| {q['q']} | {q['count']} | {pct(q['mean'])} | {pct(q['median'])} |" for q in res["quant_quintiles"]]
    L += ["\n## 포인트별 — 점수가 높았던 종목과 낮았던 종목의 수익률\n", "| 포인트 | 2~3점 | 평균 수익 | 0~1점 | 평균 수익 | 모름 | 평균 점수 |", "|---|---|---|---|---|---|---|"]
    L += [f"| {p['포인트']} | {p['2~3점 수']} | {pct(p['2~3점 평균 수익'])} | {p['0~1점 수']} | {pct(p['0~1점 평균 수익'])} | {p['모름']} | {p['평균 점수']:.2f} |" for p in per_point]
    (out / "settle.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L[:40]))


def cmd_figs(cfg: dict) -> None:
    """쉬운 설명판 그림 3장 → out/easy_fisher_*.png (한글 폰트 sweep_results/AppleGothic.ttf)."""
    as_of, out = _state(cfg)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    fp = ROOT / "AppleGothic.ttf"
    if fp.exists():
        font_manager.fontManager.addfont(str(fp)); plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(fp)).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    S = json.loads((out / "settle.json").read_text()); A = pd.read_json(out / "analysed.json", orient="index")
    # 1) 깔때기
    fig, ax = plt.subplots(figsize=(9, 4.6))
    steps = list(S["funnel"].items()) + [("수치 상위 30 + 무작위 15 = 문서 판독", len(A)), ("15 포인트 통과", int(A["passed"].sum())), ("최종 선택", len(S["picks"]))]
    names = [k.replace("$", "\\$") for k, _ in steps][::-1]; vals = [v for _, v in steps][::-1]      # $ 는 matplotlib 수식 기호라 막는다
    ax.barh(names, vals, color="#5c6bc0"); ax.set_xscale("log")
    for i, v in enumerate(vals):
        ax.text(v * 1.08, i, f"{v:,}", va="center", fontsize=10)
    ax.set_title(f"나스닥 {steps[0][1]:,}종목에서 {len(S['picks'])}종목까지 (기준일 {as_of})", loc="left"); ax.set_xlim(1, steps[0][1] * 4)
    ax.spines[["top", "right"]].set_visible(False); fig.tight_layout(); fig.savefig(out / "easy_fisher_funnel.png", dpi=140); plt.close(fig)
    # 2) 묶음별 수익률
    fig, ax = plt.subplots(figsize=(9, 4.6))
    g = {x["묶음"]: x["평균"] for x in S["groups"]}
    n_top, n_ctl = int((A["group"] == "top").sum()), int((A["group"] == "control").sum())
    bars = [(f"피셔 최종 선택 {len(S['picks'])}", g[next(k for k in g if k.startswith("피셔 통과 상위"))], "#ef5350"), (f"수치 점수 상위 {n_top}", g["수치 상위(top) 전체"], "#ffa726"),
            ("QQQ", S["qqq"], "#78909c"), (f"판독했으나 탈락 {int((~A['passed']).sum())}", g["판독했으나 탈락"], "#9ccc65"), (f"검토 모집단 {S['universe_n']} (평균)", S["universe_mean"], "#78909c"),
            (f"무작위 대조 {n_ctl}", g["무작위 대조(control) 전체"], "#66bb6a")]
    ax.bar([b[0] for b in bars], [b[1] * 100 for b in bars], color=[b[2] for b in bars])
    for i, b in enumerate(bars):
        ax.text(i, b[1] * 100 + 0.6, f"{b[1]:+.1%}", ha="center", fontsize=11)
    ax.set_ylabel("수익률 (%)"); ax.set_title(f"{as_of} 에 사서 {S['end']} 까지 들고 있었다면", loc="left")
    ax.spines[["top", "right"]].set_visible(False); plt.setp(ax.get_xticklabels(), rotation=15, ha="right"); fig.tight_layout(); fig.savefig(out / "easy_fisher_groups.png", dpi=140); plt.close(fig)
    # 3) 점수 vs 수익률
    fig, ax = plt.subplots(figsize=(9, 5))
    for grp, col, lab in (("top", "#5c6bc0", f"수치 상위 {n_top}"), ("control", "#66bb6a", f"무작위 대조 {n_ctl}")):
        d = A[A["group"] == grp]
        ax.scatter(d["fisher_avg"], d["ret"] * 100, s=np.where(d["passed"], 90, 40), color=col, alpha=0.8, label=lab, edgecolor="k", linewidth=np.where(d["passed"], 1.2, 0))
    for s_, r in A.iterrows():
        if r["passed"] or abs(r["ret"]) > 0.5:
            ax.annotate(s_, (r["fisher_avg"], r["ret"] * 100), xytext=(4, 3), textcoords="offset points", fontsize=8)
    ax.axhline(S["qqq"] * 100, color="#78909c", linestyle="--", linewidth=1); ax.text(A["fisher_avg"].min(), S["qqq"] * 100 + 2, f"QQQ {S['qqq']:+.1%}", color="#546e7a", fontsize=9)
    ax.axvline(2.0, color="#ef5350", linestyle=":", linewidth=1); ax.text(2.01, A["ret"].max() * 100, "통과선 2.0", color="#ef5350", fontsize=9, va="top")
    ax.set_xlabel("피셔 15 포인트 평균 점수 (0~3)"); ax.set_ylabel("수익률 (%)"); ax.legend(frameon=False, loc="upper left")
    ax.set_title(f"점수가 높을수록 수익률이 높았나? — 순위 상관 {S['spearman']['fisher_avg_vs_ret'][0]:+.2f} (큰 테두리 = 통과)", loc="left")
    ax.spines[["top", "right"]].set_visible(False); fig.tight_layout(); fig.savefig(out / "easy_fisher_scatter.png", dpi=140); plt.close(fig)
    print("figs ok")


def cmd_reasons(cfg: dict) -> None:
    """최종 선택 종목마다 15 포인트의 점수와 근거(판독 때 AI 가 적은 글 그대로) + 도구 수치 요약 → out/reasons.md."""
    as_of, out = _state(cfg)
    S = json.loads((out / "settle.json").read_text()); A = pd.read_json(out / "analysed.json", orient="index"); J = json.loads((out / "judgements.json").read_text())
    u = pd.read_parquet(out / "universe.parquet"); meta = pd.DataFrame(json.loads((out / "dossier_meta.json").read_text())).set_index("symbol")
    picks = S["picks"]
    conf = {"high": "높음", "medium": "중간", "low": "낮음"}
    sc_txt = lambda v: "모름" if v is None else f"{int(v)}점"  # noqa: E731
    pct = lambda v: f"{v:+.1%}"  # noqa: E731
    L = [f"# 피셔 15 포인트 — 최종 선택 {len(picks)}종목의 항목별 점수와 근거 (기준일 {as_of})\n",
         f"`{as_of}` 에 구할 수 있던 자료(그 전에 제출된 10-K·위임장·8-K 목록 + 그날까지 발표된 분기 재무제표)만으로 Claude Opus 5.5 가 매긴 점수와, 그때 적은 근거 글이다. "
         f"수익률은 {as_of} 종가 → {S['end']} (같은 기간 QQQ {pct(S['qqq'])}). 근거 글은 판독 당시 기록 그대로이며 그 뒤의 일은 반영돼 있지 않다.\n",
         "점수: **0** 반대 증거가 있다 · **1** 약하거나 평범 · **2** 충족 · **3** 뚜렷이 탁월 · **모름** 자료로 판단 불가. 확신 = AI 가 스스로 적은 판정의 확신(문서로 알기 어려운 항목은 낮다).",
         "'도구 점수' = 재무제표 숫자만으로 기계가 먼저 매긴 점수(포인트 1·3·5·6·12·13 만 있음, 12 는 2점이 상한). AI 점수와 다르면 AI 가 문서를 읽고 고친 것이다.\n",
         "## 한눈에 보기\n", "| 종목 | 수익률 | 평균 | " + " | ".join(str(k) for k in range(1, 16)) + " |", "|---|---|---|" + "---|" * 15]
    for s in picks:
        pts = {p["point"]: p for p in J[s]["judgement"]["points"]}
        L.append(f"| {s} | {pct(A.loc[s, 'ret'])} | {A.loc[s, 'fisher_avg']:.2f} | " + " | ".join("?" if pts[k]["score"] is None else str(pts[k]["score"]) for k in range(1, 16)) + " |")
    L.append("\n열 번호 = 포인트 번호: " + " · ".join(f"{k} {fc.POINT_NAMES[k]}" for k in range(1, 16)) + ". `?` = 모름.\n")
    for n, s in enumerate(picks, 1):
        j = J[s]["judgement"]; r = A.loc[s]; q = u.loc[s]; m = meta.loc[s]
        text = (out / "dossier" / f"{s}.txt").read_text(encoding="utf-8")
        a, b = text.find("[수치 요약"), text.find("[공시 원문 발췌")
        L += [f"\n---\n\n## {n}. {s} — {m['name']}\n",
              f"- 섹터 {q['sector']} · 읽은 연차보고서 {m['annual_form']} ({m['annual_date']} 제출) · 자료 {int(m['chars']):,}자",
              f"- **피셔 평균 {r['fisher_avg']:.2f}** (0점 {int(r['n_zero'])}개, 모름 {int(r['n_unknown'])}개) · 도구 수치 평균 {q['quant_avg']:.2f} · AI 결론 {j['verdict']}",
              f"- **{as_of} → {S['end']} 수익률 {pct(r['ret'])}** (QQQ {pct(S['qqq'])})\n",
              "**AI 가 받은 숫자 (도구 계산)**\n", text[a:b].strip().replace("[수치 요약 — 기준일까지 발표된 분기 재무제표로 도구가 계산]\n", "").replace("[최근 3년 8-K 항목 — 기준일 전]", "\n최근 3년 8-K:"), "",
              "**항목별 점수와 근거**\n", "| # | 포인트 | 점수 | 도구 점수 | 확신 | 근거 |", "|---|---|---|---|---|---|"]
        for p in j["points"]:
            k = p["point"]; qs = q.get(f"p{k}") if k in fc.QUANT_POINTS else None
            qtxt = "—" if k not in fc.QUANT_POINTS else ("해당 없음" if qs is None or not np.isfinite(qs) else f"{int(qs)}점")
            L.append(f"| {k} | {fc.POINT_NAMES[k]} | **{sc_txt(p['score'])}** | {qtxt} | {conf[p['confidence']]} | {p['evidence'].replace('|', '/').replace(chr(10), ' ')} |")
        L += ["", "**뛰어난 점**\n"] + [f"- {x}" for x in j["strengths"]] + ["", "**걸리는 점**\n"] + ([f"- {x}" for x in j["red_flags"]] or ["- (없음)"])
        if j["temporary_trouble"]:
            L += ["", f"**일시적 문제(책 5장 '사기 좋은 때')**: {j['temporary_trouble']}"]
        L += ["", f"**AI 의 종합**: {j['summary']}"]
    (out / "reasons.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"reasons ok · {len(picks)}종목 · {sum(len(x) for x in L):,}자")


def _mcap_asof(sym_shares: pd.Series, stm_all: pd.DataFrame, T: pd.Timestamp) -> pd.Series:
    """기준일 시가총액. Alpha Vantage 주식 수는 대개 지금 기준으로 분할 조정돼 있어 분할 조정 종가(C)와 곱한다.
    기준일 뒤 분할이 있었는데 주식 수 시계열에 그 배수만큼의 뜀이 남아 있으면(미조정) 원가격(raw_c)과 곱한다."""
    px = pd.read_parquet(ROOT / "surge_panel.parquet", columns=["date", "symbol", "C", "raw_c"], filters=[("date", ">=", T - pd.Timedelta(days=7)), ("date", "<=", T)])
    px = px[px["symbol"].isin(sym_shares.index)].sort_values("date").groupby("symbol").last()
    out = {}
    later = stm_all[pd.to_datetime(stm_all["fiscal_date"]) > T].sort_values("fiscal_date").groupby("symbol")["shares_outstanding"].last()
    for s, sh in sym_shares.items():
        if s not in px.index or not np.isfinite(sh) or sh <= 0:
            continue
        c, raw = float(px.at[s, "C"]), float(px.at[s, "raw_c"])
        f = raw / c if c > 0 else 1.0
        price = c
        if abs(f - 1) > 0.05 and s in later.index and np.isfinite(later[s]) and abs(later[s] / sh / f - 1) < 0.3:
            price = raw
        out[s] = price * sh
    return pd.Series(out, dtype=float)


def cmd_pie(cfg: dict) -> None:
    """모부신 『예측투자』 — 주가에 담긴 10년 매출 성장 기대를 역산해 기저율과 대조, 피셔 통과 종목에서 '지나친 고평가'를 빼고 다시 정산.
    체크리스트 `docs/strategy_notes/모부신_예측투자_고평가제외_체크리스트_2026-11.md`. settle 뒤에 돌린다."""
    as_of, out = _state(cfg); T = pd.Timestamp(as_of)
    from data.adapters.yfinance_bulk import download_many
    from strategy.adapters import expectations_pie as ep

    u = pd.read_parquet(out / "universe.parquet")
    A = pd.read_json(out / "analysed.json", orient="index"); S = json.loads((out / "settle.json").read_text())
    stm = pd.concat([pd.read_parquet(ROOT / f) for f in STMT_FILES], ignore_index=True).drop_duplicates(["symbol", "fiscal_date"], keep="first")
    stm = stm[stm["symbol"].isin(u.index)]
    sa = fc.asof(stm, T); sa = sa[sa["fiscal_date"] >= T - pd.Timedelta(days=365 * 7)]
    f = ep.fundamentals(sa)
    # 보고 통화가 달러가 아니면 매출·순차입금을 기준일 환율로 달러로 바꾼다(주가·시가총액은 달러). 환율을 못 구하면 계산에서 뺀다.
    cur_path = ROOT / "fisher_reported_currency.json"
    cur = pd.Series(json.loads(cur_path.read_text())) if cur_path.exists() else pd.Series(dtype=object)
    f["currency"] = cur.reindex(f.index).replace("?", np.nan).fillna("USD"); f["fx"] = 1.0
    foreign = sorted(set(f["currency"]) - {"USD"})
    if foreign:
        fxp = download_many([f"{c}USD=X" for c in foreign] + ["EURUSD=X", "JPYUSD=X"], T.date() - timedelta(days=10), T.date())
        for c in foreign:
            d = fxp.get(f"{c}USD=X")
            v = d["Close"].dropna() if d is not None and len(d) else pd.Series(dtype=float)
            v = v[v.index <= T]
            f.loc[f["currency"] == c, "fx"] = float(v.iloc[-1]) if len(v) else np.nan
        f["rev_ttm"] *= f["fx"]; f["net_debt"] *= f["fx"]
        log("달러 아닌 보고 통화:", f[f["currency"] != "USD"]["currency"].value_counts().to_dict(), "환율:", f[f["currency"] != "USD"].groupby("currency")["fx"].first().round(4).to_dict())
    mcap = _mcap_asof(f["shares"], stm, T)
    if cfg.get("wacc") is not None:
        y10, wacc = None, float(cfg["wacc"])
    else:
        tn = download_many(["^TNX", "^IRX"], T.date() - timedelta(days=10), T.date())["^TNX"]["Close"].dropna()
        y10 = float(tn[tn.index <= T].iloc[-1]) / 100.0; wacc = y10 + float(cfg.get("erp", 0.05))
    base = ep.PieParams(wacc=wacc, exclude_below=float(cfg.get("exclude_below", 0.05)))
    P = ep.pie_table(f, mcap, T.year, base)
    P.to_parquet(out / "pie.parquet")
    panel = pd.read_parquet(ROOT / "surge_panel.parquet", columns=["date", "symbol", "C"], filters=[("date", ">=", T - pd.Timedelta(days=7))])
    panel = panel[panel["symbol"].isin(u.index)].sort_values("date")
    u_ret = (panel.groupby("symbol")["C"].last() / panel[panel["date"] <= T].groupby("symbol")["C"].last() - 1).reindex(u.index)
    U = u.join(P[["mcap", "ev", "ev_to_sales", "margin_used", "margin_from_sector", "invest_used", "g_implied", "g_real", "bucket", "bucket_median", "base_rate_p", "excluded"]]).assign(ret=u_ret)
    U = U[U["base_rate_p"].notna()]
    n_pick = len(S["picks"]); qqq = S["qqq"]
    stat = lambda s: {"n": int(s.notna().sum()), "mean": float(s.mean()) if s.notna().any() else None, "median": float(s.median()) if s.notna().any() else None,  # noqa: E731
                      "beat_qqq": float((s.dropna() > S["qqq_to_panel_end"]).mean()) if s.notna().any() else None}

    def fisher_run(Pt: pd.DataFrame) -> dict:
        a = A.join(Pt[["g_implied", "g_real", "base_rate_p", "excluded", "bucket", "bucket_median", "ev_to_sales", "margin_used", "margin_from_sector"]])
        passed = a[a["passed"]].sort_values(["fisher_avg", "quant_avg"], ascending=False)
        keep = passed[~passed["excluded"].fillna(False).astype(bool)]
        new = keep.head(n_pick)
        return {"a": a, "passed": passed, "orig": passed.head(n_pick), "new": new, "dropped": passed[passed["excluded"].fillna(False).astype(bool)]}

    R = fisher_run(P)
    grp = lambda s: {"n": int(len(s)), "mean": float(s.mean()) if len(s) else None, "median": float(s.median()) if len(s) else None,  # noqa: E731
                     "beat_qqq": float((s > qqq).mean()) if len(s) else None}
    # 모집단 전체(AI 없음): 제외/유지, P 5분위, 순위 상관, 품질 상위 20% 안에서
    Uq = U.dropna(subset=["ret"])
    qs = Uq.assign(q=pd.qcut(Uq["base_rate_p"].rank(method="first"), 5, labels=["가장 비쌈(P 최저)", "2", "3", "4", "가장 쌈(P 최고)"])).groupby("q", observed=True)
    quint = [{"q": str(k), "n": int(len(g)), "p_median": float(g["base_rate_p"].median()), "g_real_median": float(g["g_real"].replace([np.inf, -np.inf], np.nan).median()),
              "mean": float(g["ret"].mean()), "median": float(g["ret"].median())} for k, g in qs]
    topq = Uq[Uq["quant_avg"] >= Uq["quant_avg"].quantile(0.8)]
    rho, n_rho = fc.spearman(Uq["base_rate_p"], Uq["ret"])
    sens = {}
    for name, prm in (("자본비용 8% 고정", ep.PieParams(wacc=0.08)), ("제외 문턱 P < 10%", ep.PieParams(wacc=wacc, exclude_below=0.10)),
                      ("자본비용 +2%p", ep.PieParams(wacc=wacc + 0.02))):
        Pt = ep.pie_table(f, mcap, T.year, prm); r = fisher_run(Pt)
        ux = Uq.join(Pt["excluded"].rename("ex2"))
        sens[name] = {"dropped": list(r["dropped"].index), "new_picks": list(r["new"].index), "new_mean": grp(r["new"]["ret"])["mean"],
                      "universe_excluded_n": int(ux["ex2"].sum()), "universe_excluded_mean": float(ux[ux["ex2"]]["ret"].mean()) if ux["ex2"].any() else None,
                      "universe_kept_mean": float(ux[~ux["ex2"]]["ret"].mean())}
    res = {"as_of": as_of, "end": S["end"], "panel_end": S["panel_end"], "y10": y10, "wacc": wacc, "qqq": qqq, "qqq_to_panel_end": S["qqq_to_panel_end"],
           "universe": {"n": int(len(Uq)), "excluded": stat(Uq[Uq["excluded"]]["ret"]), "kept": stat(Uq[~Uq["excluded"]]["ret"]), "all": stat(Uq["ret"]),
                        "spearman_p_vs_ret": [rho, n_rho], "quintiles": quint,
                        "top_quality": {"excluded": stat(topq[topq["excluded"]]["ret"]), "kept": stat(topq[~topq["excluded"]]["ret"])}},
           "fisher": {"passed_n": int(len(R["passed"])), "dropped": list(R["dropped"].index), "orig_picks": list(R["orig"].index), "new_picks": list(R["new"].index),
                      "orig": grp(R["orig"]["ret"]), "new": grp(R["new"]["ret"]), "dropped_stat": grp(R["dropped"]["ret"]),
                      "kept_all": grp(R["passed"][~R["passed"]["excluded"].fillna(False).astype(bool)]["ret"])},
           "sensitivity": sens}
    (out / "pie.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    pct = lambda v: "—" if v is None or not np.isfinite(v) else f"{v:+.1%}"  # noqa: E731
    gfmt = lambda v: "—" if v is None or pd.isna(v) else (">100%" if v == np.inf else ("<−20%" if v == -np.inf else f"{v:+.1%}"))  # noqa: E731
    L = [f"# 피셔 + 모부신 예측투자(고평가 제외) — {as_of} 선별, {S['end']} 까지\n",
         f"자본비용 {wacc:.1%}" + (f" (10년 국채 {y10:.2%} + 5%p)" if y10 is not None else "") + f" · 예측 10년 · 세율 21% · 물가 2% · 제외 = 기저율 P < {base.exclude_below:.0%} · QQQ {pct(qqq)}\n",
         "## 피셔 통과 종목의 내재 기대\n", "| 종목 | 피셔 평균 | 원래 선택 | EV/매출 | 쓴 이익률 | 내재 매출 성장(명목/실질, 10년) | 크기 구간(중앙 성장) | 기저율 P | 판정 | 수익률 |", "|---|---|---|---|---|---|---|---|---|---|"]
    for s, r in R["passed"].iterrows():
        L.append(f"| {s} | {r['fisher_avg']:.2f} | {'●' if s in R['orig'].index else ''} | {r['ev_to_sales']:.1f} | {r['margin_used']:.1%}{'*' if r['margin_from_sector'] else ''} | {gfmt(r['g_implied'])} / {gfmt(r['g_real'])} | "
                 f"{r['bucket']} ({r['bucket_median']:.1%}) | {r['base_rate_p']:.1%} | {'**제외**' if r['excluded'] else ('높은 기대' if r['base_rate_p'] < 0.25 else '무난')} | {pct(r['ret'])} |")
    L += ["\n`*` = 영업이익률이 5% 미만이라 섹터 중앙값을 씀.\n", "## 선택 비교\n", "| 묶음 | 종목 수 | 평균 | 중앙 | QQQ 이긴 비율 | 종목 |", "|---|---|---|---|---|---|"]
    for name, k, d in (("원래 피셔 선택", "orig", R["orig"]), ("고평가 제외 뒤 선택", "new", R["new"]), ("제외된 통과 종목", "dropped_stat", R["dropped"])):
        g = res["fisher"][k]
        L.append(f"| {name} | {g['n']} | {pct(g['mean'])} | {pct(g['median'])} | {'—' if g['beat_qqq'] is None else format(g['beat_qqq'], '.0%')} | {' '.join(d.index)} |")
    un = res["universe"]
    L += [f"\n## 모집단 {un['n']}종목 전체에 적용 (AI 없음, {S['panel_end']} 까지, 같은 기간 QQQ {pct(S['qqq_to_panel_end'])})\n", "| 묶음 | 종목 수 | 평균 | 중앙 | QQQ 이긴 비율 |", "|---|---|---|---|---|"]
    for name, g in (("제외(P < 문턱)", un["excluded"]), ("유지", un["kept"]), ("품질 상위 20% 중 제외", un["top_quality"]["excluded"]), ("품질 상위 20% 중 유지", un["top_quality"]["kept"])):
        L.append(f"| {name} | {g['n']} | {pct(g['mean'])} | {pct(g['median'])} | {'—' if g['beat_qqq'] is None else format(g['beat_qqq'], '.0%')} |")
    L += [f"\n기저율 P ↔ 수익률 순위 상관 {rho:+.2f} (n={n_rho}; 양수 = 기대가 낮았던 종목이 더 올랐다)\n", "| P 5분위 | 종목 수 | P 중앙 | 내재 실질 성장 중앙 | 평균 수익 | 중앙 수익 |", "|---|---|---|---|---|---|"]
    L += [f"| {q['q']} | {q['n']} | {q['p_median']:.1%} | {pct(q['g_real_median'])} | {pct(q['mean'])} | {pct(q['median'])} |" for q in quint]
    L += ["\n## 민감도\n", "| 설정 | 제외된 통과 종목 | 새 선택 평균 | 모집단 제외 수 | 모집단 제외 평균 | 모집단 유지 평균 |", "|---|---|---|---|---|---|"]
    L += [f"| {k} | {' '.join(v['dropped']) or '없음'} | {pct(v['new_mean'])} | {v['universe_excluded_n']} | {pct(v['universe_excluded_mean'])} | {pct(v['universe_kept_mean'])} |" for k, v in sens.items()]
    (out / "pie.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


def cmd_piefigs(cfg: dict) -> None:
    """예측투자 쉬운 설명용 그림 2장 → sweep_results/easy_pie/ (cfg dates = pie 를 돌린 기준일 목록)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    fp = ROOT / "AppleGothic.ttf"
    if fp.exists():
        font_manager.fontManager.addfont(str(fp)); plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(fp)).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    dates = cfg.get("dates", ["2021-11-19", "2025-11-19"]); od = ROOT / "easy_pie"; od.mkdir(exist_ok=True)
    R = {d: json.loads((_out(d) / "pie.json").read_text()) for d in dates}
    fig, axes = plt.subplots(1, len(dates), figsize=(6 * len(dates), 4.6))
    for ax, d in zip(np.atleast_1d(axes), dates):
        r = R[d]; f = r["fisher"]; u = r["universe"]
        bars = [(f"피셔 원래 선택 {f['orig']['n']}", f["orig"]["mean"], "#ef5350"), (f"고평가 제외 뒤 {f['new']['n']}", f["new"]["mean"], "#5c6bc0"), (f"제외된 통과 종목 {f['dropped_stat']['n']}", f["dropped_stat"]["mean"], "#bdbdbd"),
                ("QQQ", r["qqq"], "#78909c"), (f"모집단: 제외 {u['excluded']['n']}", u["excluded"]["mean"], "#ffa726"), (f"모집단: 유지 {u['kept']['n']}", u["kept"]["mean"], "#66bb6a")]
        ax.bar(range(len(bars)), [b[1] * 100 for b in bars], color=[b[2] for b in bars])
        for i, b in enumerate(bars):
            ax.text(i, b[1] * 100 + (1 if b[1] >= 0 else -4), f"{b[1]:+.1%}", ha="center", fontsize=10)
        ax.set_xticks(range(len(bars))); ax.set_xticklabels([b[0] for b in bars], rotation=20, ha="right", fontsize=9); ax.axhline(0, color="k", linewidth=0.6)
        ax.set_title(f"{d} 에 사서 {r['end']} 까지 (평균 수익률 %)", loc="left", fontsize=11); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(od / "easy_pie_groups.png", dpi=140); plt.close(fig)
    fig, axes = plt.subplots(1, len(dates), figsize=(6 * len(dates), 4.2))
    for ax, d in zip(np.atleast_1d(axes), dates):
        q = R[d]["universe"]["quintiles"]
        ax.bar(range(5), [x["median"] * 100 for x in q], color=["#ef5350", "#ffa726", "#bdbdbd", "#9ccc65", "#66bb6a"])
        for i, x in enumerate(q):
            ax.text(i, x["median"] * 100 + (1 if x["median"] >= 0 else -3), f"{x['median']:+.0%}", ha="center", fontsize=10)
        ax.set_xticks(range(5)); ax.set_xticklabels([f"{x['q']}\n요구 성장 {x['g_real_median']:+.0%}" for x in q], fontsize=8); ax.axhline(0, color="k", linewidth=0.6)
        ax.set_title(f"{d}: 주가가 요구한 성장이 클수록 그 뒤 수익률(중앙)은?", loc="left", fontsize=11); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(od / "easy_pie_quintiles.png", dpi=140); plt.close(fig)
    print("piefigs ok")


def cmd_piedetail(cfg: dict) -> None:
    """피셔 통과 종목마다 고평가 판정의 계산 과정(입력 → 역산 → 기저율 → 판정 → 선택에 든/빠진 이유)을 풀어 쓴 md → out/pie_detail.md. pie 뒤에 돌린다."""
    as_of, out = _state(cfg)
    from strategy.adapters import expectations_pie as ep

    P = pd.read_parquet(out / "pie.parquet"); A = pd.read_json(out / "analysed.json", orient="index"); S = json.loads((out / "settle.json").read_text()); R = json.loads((out / "pie.json").read_text())
    prm = ep.PieParams(wacc=R["wacc"]); n_pick = len(S["picks"]); cpi = ep.CPI_FROM_2015.get(pd.Timestamp(as_of).year, 1.0)
    passed = A[A["passed"]].sort_values(["fisher_avg", "quant_avg"], ascending=False)
    orig, new = R["fisher"]["orig_picks"], R["fisher"]["new_picks"]
    usd = lambda v: f"${v / 1e9:,.1f}B" if abs(v) >= 1e9 else f"${v / 1e6:,.0f}M"  # noqa: E731
    pc = lambda v, d=1: f"{v * 100:+.{d}f}%"  # noqa: E731
    edges = [-25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
    y10 = R.get("y10")
    L = [f"## 기준일 {as_of} (보유 → {S['end']}, QQQ {pc(S['qqq'])})\n",
         f"공통 가정: 자본비용 **{prm.wacc:.1%}**" + (f" (그날 10년 국채 {y10:.2%} + 5%p)" if y10 else "") + f" · 세율 {prm.tax:.0%} · 물가 {prm.inflation:.0%} · 예측 {prm.years}년 · 2015년 달러 환산 ÷{cpi:.2f} · 제외 문턱 P < {prm.exclude_below:.0%}\n",
         "### 선택이 어떻게 바뀌었나\n", "| 피셔 순위 | 종목 | 피셔 평균 | 요구 성장(실질) | P | 판정 | 원래 선택 | 새 선택 | 무슨 일이 있었나 | 수익률 |", "|---|---|---|---|---|---|---|---|---|---|"]
    kept_rank = 0
    for i, (s, a) in enumerate(passed.iterrows(), 1):
        r = P.loc[s]; ex = bool(r["excluded"])
        kept_rank += 0 if ex else 1
        if ex:
            why = f"P {r['base_rate_p']:.1%} < 5% → **빠짐**" + (" (원래 선택에서 탈락)" if s in orig else " (원래도 10위 밖)")
        elif s in orig:
            why = "그대로 남음"
        elif s in new:
            why = f"앞 순위 종목이 빠져 {i}위 → {kept_rank}위로 올라 **새로 들어옴**"
        else:
            why = f"제외는 아니지만 {kept_rank}위라 10위 밖"
        L.append(f"| {i} | {s} | {a['fisher_avg']:.2f} | {pc(r['g_real'])} | {r['base_rate_p']:.1%} | {'**제외**' if ex else '유지'} | {'●' if s in orig else ''} | {'●' if s in new else ''} | {why} | {pc(a['ret'])} |")
    fz = R["fisher"]
    L += [f"\n원래 선택 {fz['orig']['n']}종목 평균 **{pc(fz['orig']['mean'])}** → 새 선택 {fz['new']['n']}종목 평균 **{pc(fz['new']['mean'])}**. 빠진 종목: {' '.join(fz['dropped']) or '없음'}. "
          f"새로 들어온 종목: {' '.join(x for x in new if x not in orig) or '없음'}.\n", "### 종목별 계산\n"]
    for i, (s, a) in enumerate(passed.iterrows(), 1):
        r = P.loc[s]; g = float(r["g_implied"]); m = float(r["margin_used"]); k = float(r["invest_used"]); s0 = float(r["rev_ttm"]); ev = float(r["ev"])
        price = r["mcap"] / r["shares"]
        t = np.arange(1, prm.years + 1); sales = s0 * (1 + g) ** t; nopat = sales * m * (1 - prm.tax); fcf = nopat - (sales - s0 * (1 + g) ** (t - 1)) * k
        pv_f = float((fcf / (1 + prm.wacc) ** t).sum()); cv = float(nopat[-1] * (1 + prm.inflation) / (prm.wacc - prm.inflation)); pv_cv = cv / (1 + prm.wacc) ** prm.years
        v0 = float(ep.dcf_value(s0, 0.0, m, k, prm)[0])
        b = ep.size_bucket(r["sales_2015_musd"]); name, dist = ep.BASE_RATE_10Y[b][1], ep.BASE_RATE_10Y[b][2]
        gr = float(r["g_real"]) * 100; kk = int(np.searchsorted(edges, gr, side="right"))
        if gr < edges[0]:
            how = "요구 성장이 표의 가장 낮은 칸보다도 아래 → 거의 모든 회사가 해냄"
        elif gr >= edges[-1]:
            how = f"요구 성장이 표의 가장 높은 칸(>45%) 안 → 그 칸 {dist[-1]:.1f}% 의 절반"
        else:
            lo_e, hi_e = edges[kk - 1], edges[kk]; frac = (hi_e - gr) / (hi_e - lo_e); above = sum(dist[kk + 1:])
            how = f"{lo_e}~{hi_e}% 칸({dist[kk]:.1f}%) 중 {gr:.1f}% 보다 위쪽 부분 {frac:.0%} = {dist[kk] * frac:.1f}% + 그보다 높은 칸 합 {above:.1f}% = **{dist[kk] * frac + above:.1f}%**"
        ex = bool(r["excluded"])
        verdict = "**제외** — 같은 크기 회사 스무 곳 중 한 곳도 못 해낸 성장을 주가가 요구" if ex else ("유지 (높은 기대 — P 5~25%)" if r["base_rate_p"] < 0.25 else "유지 (무난 — P 25% 이상)")
        mtxt = f"{m:.1%}" + (f" (실제 {r['op_margin']:.1%} 가 5% 미만이라 섹터 중앙값 사용)" if r["margin_from_sector"] else " (최근 12개월 GAAP 영업이익률 그대로)")
        ktxt = f"{k:.1%}" + ("" if np.isfinite(r["invest_rate"]) and abs(np.clip(r["invest_rate"], 0, prm.max_invest) - k) < 1e-9 else " (계산 불가 → 섹터 중앙값)") + (f" (원래 값 {r['invest_rate']:.1%} 를 0~50% 로 자름)" if np.isfinite(r["invest_rate"]) and not (0 <= r["invest_rate"] <= prm.max_invest) else "")
        L += [f"#### {i}. {s} — {'제외' if ex else '유지'} · 피셔 {a['fisher_avg']:.2f} · 그 뒤 수익률 {pc(a['ret'])}\n",
              "| 단계 | 계산 | 값 |", "|---|---|---|",
              f"| ① 시가총액 | 종가 ${price:,.2f} × 주식 수 {r['shares'] / 1e6:,.0f}M (분할 조정 기준) | {usd(r['mcap'])} |",
              f"| ② 기업가치 EV | 시가총액 + 순차입금({usd(r['net_debt'])}) | **{usd(ev)}** |",
              f"| ③ 출발 매출 | 기준일까지 발표된 최근 4분기 합" + (f" ({r['currency']} 로 보고 → 그날 환율 {r['fx']:.4f} 로 달러 환산)" if r.get("currency", "USD") != "USD" else "") + f" | {usd(s0)} (EV 는 매출의 {ev / s0:.1f}배) |",
              f"| ④ 영업이익률 | {mtxt} | {m:.1%} → 세후 {m * (1 - prm.tax):.1%} |",
              f"| ⑤ 증분 투자율 | 매출이 $1 늘 때 드는 투자 | {ktxt} |",
              f"| ⑥ 성장이 0 이라면 | 지금 이익이 영원히 물가만큼만 는다고 볼 때의 가치 | {usd(v0)} (EV 의 {v0 / ev:.0%}) → 나머지 {1 - v0 / ev:.0%} 는 성장에 대한 기대 |",
              f"| ⑦ 역산 | EV 를 맞추는 10년 매출 성장률 | **연 {pc(g)}** (물가 빼면 **{pc(float(r['g_real']))}**) |",
              f"| ⑧ 그 뜻 | 10년 뒤 매출 = {usd(s0)} × (1{g:+.3f})¹⁰ | {usd(float(sales[-1]))} (지금의 {sales[-1] / s0:.1f}배) |",
              f"| ⑨ 검산 | 10년 현금흐름의 현재가치 {usd(pv_f)} + 잔존가치의 현재가치 {usd(pv_cv)} | {usd(pv_f + pv_cv)} ≈ EV |",
              f"| ⑩ 크기 구간 | 매출 {usd(s0)} ÷ {cpi:.2f} = 2015년 달러 ${r['sales_2015_musd']:,.0f}M | {name} (이 구간 10년 실질 성장 중앙 {r['bucket_median']:.1%}) |",
              f"| ⑪ 기저율 P | {how} | **{r['base_rate_p']:.1%}** |",
              f"| ⑫ 판정 | P {'<' if ex else '≥'} 5% | {verdict} |", ""]
    (out / "pie_detail.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"piedetail ok · {len(passed)}종목")


def cmd_currency(cfg: dict) -> None:
    """모집단 종목의 재무제표 보고 통화(Alpha Vantage INCOME_STATEMENT 의 reportedCurrency) → sweep_results/fisher_reported_currency.json (이어 받기).
    재무제표 parquet 에는 통화가 없어, 유로·위안 등으로 보고하는 회사의 매출을 달러 시가총액과 그대로 비교하면 틀린다."""
    import os
    import time

    import requests

    path = ROOT / "fisher_reported_currency.json"
    have = json.loads(path.read_text()) if path.exists() else {}
    syms = sorted({s for d in ROOT.glob("fisher_20*") if (d / "universe.parquet").exists() for s in pd.read_parquet(d / "universe.parquet").index})
    todo = [s for s in syms if s not in have]
    key = os.environ["ALPHAVANTAGE_API_KEY"]
    for n, sym in enumerate(todo):
        try:
            j = requests.get("https://www.alphavantage.co/query", params={"function": "INCOME_STATEMENT", "symbol": sym, "apikey": key}, timeout=60).json()
            q = j.get("quarterlyReports") or []
            if q:
                have[sym] = next((r.get("reportedCurrency") for r in q if r.get("reportedCurrency") not in (None, "", "None")), "USD")     # 최근 분기가 "None" 인 경우가 있다
            elif "Note" in j or "Information" in j:
                log(f"{sym}: 한도 응답 — 30초 쉼"); time.sleep(30); continue
            else:
                have[sym] = "?"
        except Exception as ex:  # noqa: BLE001
            log(f"{sym}: {type(ex).__name__}")
        if n % 50 == 0:
            path.write_text(json.dumps(have)); log(f"{n}/{len(todo)}")
        time.sleep(float(cfg.get("pace", 0.85)))
    path.write_text(json.dumps(have))
    print(json.dumps({"symbols": len(syms), "known": len(have), "non_usd": {k: v for k, v in have.items() if v not in ("USD", "?")}}, ensure_ascii=False))


if __name__ == "__main__":
    cmd = sys.argv[1]
    cfg = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
    {"universe": cmd_universe, "dossier": cmd_dossier, "judge": cmd_judge, "settle": cmd_settle, "figs": cmd_figs, "reasons": cmd_reasons, "pie": cmd_pie, "piefigs": cmd_piefigs, "piedetail": cmd_piedetail, "currency": cmd_currency}[cmd](cfg)
