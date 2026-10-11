"""와인스타인 1→2단계 돌파 스크린 — 기준일 여러 개 배치 검증 (46 페이지와 같은 깔때기).

문서 §E 절차 2~3: 기준일마다 '이번 주 돌파' 종목을 뽑아 등급을 매기고, 돌파 주 다음 거래일
진입 → 7·8장 청산으로 **등급별 성과를 분리**한다 (대조군 = B-/탈락 종목).

    docker compose exec backtester python weinstein_batch_cli.py \\
        '{"dates": ["2023-06-02", "2024-02-02"], "universe": "sp500", "out": "/tmp/weinstein_batch.json"}'
    docker compose exec backtester python weinstein_batch_cli.py '{"analyze": "/tmp/weinstein_batch.json"}'
    docker compose exec backtester python weinstein_batch_cli.py \\
        '{"analyze": "/tmp/weinstein_batch.json", "report_md": "/tmp/weinstein_report.md"}'   # 날짜별 계좌 표

기준일은 금요일을 권장(완결된 주봉). 진행 로그는 stderr, 요약 표는 stdout.

날짜별 계좌 수익률(``portfolio_report``)은 "그 날 그 규칙대로 샀으면" 을 그대로 재현한다:
후보를 등급·거래량 순으로 최대 5종목, 각 20% (남는 칸은 현금), 각 종목은 7·8장 규칙으로
청산 → 계좌 수익률 = Σ(종목 수익률 × 20%). 같은 기간 QQQ·SPY 를 옆에 둔다.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from datetime import date, timedelta

import pandas as pd

from data.adapters.alphavantage_adapter import AlphaVantageAdapter
from data.adapters.mongo_day_cache import MongoDayCacheAdapter
from data.adapters.wikipedia_universe import default_universe_provider
from pages._shared.growth_sources import build_growth_adapters
from strategy.adapters.minervini_scan import fetch_with_retry
from strategy.adapters.weinstein_scan import (
    ScanParams,
    check_stats,
    grade_stats,
    regrade_volume_optional,
    run_scan,
    simulate_forward,
)
from strategy.adapters.weinstein_stage import Check, StageParams

MAX_POS = 5
WEIGHT = 0.20

# 규칙 단계 — 느슨해지는 순서. 각 항목: (이름, 설명, 선택 함수(rec) -> bool)
RULES = [
    ("책 그대로", "시장·업종 통과 + 종목 A/A+ (거래량 2배 필수)", lambda r: r["grade"] in ("A+", "A")),
    ("거래량 완화", "시장·업종 통과 + 저항·RS 로만 A/A+ (4-1 정보 항목)", lambda r: r["grade_vol_opt"] in ("A+", "A")),
    ("종목만", "시장·업종 무시, 종목 A/A+ (거래량 완화)", lambda r: r["stock_grade_vol_opt"] in ("A+", "A")),
    ("전부", "돌파 셋업 전부 (대조군)", lambda r: True),
]


def _log(m: str) -> None:
    print(m, file=sys.stderr, flush=True)


def _universe(provider, choice: str, as_of: date) -> list[str]:
    if choice.endswith("@기준일"):
        return provider.get_tickers(f"{choice[:-len('@기준일')]}@{as_of.isoformat()}")
    if choice == "sp500+nasdaq100":
        return list(dict.fromkeys(provider.get_tickers("sp500") + provider.get_tickers("nasdaq100")))
    return provider.get_tickers(choice)


def _bench(md, sym: str, as_of: date) -> dict:
    """기준일 종가 → +13/26/52주 종가 수익률."""
    try:
        df = fetch_with_retry(md, sym, as_of - timedelta(days=10), min(as_of + timedelta(days=380), date.today()))
    except Exception:  # noqa: BLE001
        return {}
    idx = pd.to_datetime(df.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    df.index = idx.normalize()
    base = df[df.index <= pd.Timestamp(as_of)]
    fwd = df[df.index > pd.Timestamp(as_of)]
    if base.empty:
        return {}
    c0 = float(base["Close"].iloc[-1])
    out = {}
    for k, n in (("13w", 65), ("26w", 130), ("52w", 260)):
        out[k] = float(fwd["Close"].iloc[n - 1]) / c0 - 1 if len(fwd) >= n else None
    out["last"] = float(fwd["Close"].iloc[-1]) / c0 - 1 if len(fwd) else None
    return out


def run(cfg: dict) -> dict:
    md = MongoDayCacheAdapter(AlphaVantageAdapter(), source_name="alphavantage")
    uni = default_universe_provider()
    growth = build_growth_adapters()
    sp = StageParams(**cfg.get("stage", {}))
    params = ScanParams(mode=cfg.get("mode", "breakout"), recent_weeks=int(cfg.get("recent_weeks", 1)),
                        include_partial_week=False, stage=sp)
    results = []
    for d in cfg["dates"]:
        as_of = date.fromisoformat(d)
        tickers = _universe(uni, cfg.get("universe", "sp500"), as_of)
        if cfg.get("max_tickers"):
            tickers = tickers[: int(cfg["max_tickers"])]
        res = run_scan(as_of, tickers, md, params, growth, max_workers=int(cfg.get("workers", 4)), log=_log)
        day = {"as_of": d, "market_pass": res.market_pass, "funnel": res.funnel,
               "market": [asdict(c) for c in res.market_checks],
               "bench": {s: _bench(md, s, as_of) for s in ("SPY", "QQQ")}, "candidates": []}
        for c in res.candidates:
            try:
                o = simulate_forward(res.frames[c.ticker], c, sp)
            except Exception as e:  # noqa: BLE001
                _log(f"  {c.ticker} 시뮬 실패 {type(e).__name__}")
                continue
            stock_checks = [k for k in c.checks if k.code[0] not in "01"]
            rec = {"ticker": c.ticker, "grade": c.grade, "stock_grade": c.stock_grade,
                   "grade_vol_opt": regrade_volume_optional(c.checks),
                   "stock_grade_vol_opt": regrade_volume_optional(stock_checks),
                   "entry_type": c.entry_type, "sector": c.sector, "resistance": round(c.resistance, 2),
                   "base_weeks": c.base.weeks, "vol_ratio": round(c.vol_ratio, 2), "rs": round(c.rs, 2),
                   "aplus": c.aplus_flag, "checks": [asdict(k) for k in c.checks], **asdict(o)}
            day["candidates"].append(rec)
        results.append(day)
        _log(f"[{d}] 시장 {'✅' if res.market_pass else '❌'} 셋업 {len(res.candidates)} A이상 {res.funnel.get('A 이상', 0)}")
        with open(cfg.get("out", "/tmp/weinstein_batch.json"), "w") as f:
            json.dump(results, f, ensure_ascii=False, indent=1, default=str)
    analyze(results)
    print(portfolio_report(results))
    if cfg.get("report_md"):
        with open(cfg["report_md"], "w") as f:
            f.write(portfolio_report(results))
    return {"results": results}


def _recs(results: list[dict]) -> list[dict]:
    recs = []
    for day in results:
        for rec in day["candidates"]:
            r = dict(rec)
            if "grade_vol_opt" not in r:   # 구버전 JSON
                checks = [Check(**c) for c in r.get("checks", [])]
                r["grade_vol_opt"] = regrade_volume_optional(checks)
                r["stock_grade_vol_opt"] = regrade_volume_optional([c for c in checks if c.code[0] not in "01"])
            r["as_of"] = day["as_of"]
            r["check_map"] = {c["code"]: c["passed"] for c in r.get("checks", [])}
            recs.append(r)
    return recs


def analyze(results: list[dict]) -> None:
    """저장된 결과(JSON)로 등급·유형·항목별 사후 성과 표."""
    recs = [{**r, "checks": r["check_map"], "mfe": r.get("mfe") or 0.0, "mae": r.get("mae") or 0.0} for r in _recs(results)]
    pd.set_option("display.width", 220)
    fmt = lambda x: f"{x:+.1%}" if abs(x) < 5 else f"{x:.1f}"  # noqa: E731
    for key in ("grade", "grade_vol_opt", "entry_type"):
        gs = grade_stats(recs, key)
        if not gs.empty:
            print(f"\n== {key} 별 사후 성과 ({len(recs)}건) ==")
            print(gs.to_string(index=False, float_format=fmt))
    cs = check_stats(recs)
    if not cs.empty:
        print("\n== 항목별 예측력 (통과 vs 미통과 사후 수익률) ==")
        print(cs.to_string(index=False, float_format=fmt))


def _pct(x) -> str:
    return "—" if x is None else f"{x:+.1%}"


def portfolio_report(results: list[dict]) -> str:
    """날짜별 '그 규칙대로 샀으면' 계좌 수익률 — 마크다운."""
    recs = _recs(results)
    lines = ["# 와인스타인 샘플 날짜 백테스트 — 날짜별 계좌 수익률", "",
             f"규칙: 최대 {MAX_POS}종목 × {WEIGHT:.0%} (등급·거래량 순), 돌파 주 다음 거래일 시가 진입, "
             "7-5/7-7/8-3/8-4 청산(최대 52주). 계좌 수익률 = Σ 종목 수익률 × 20% (빈 칸은 현금). 진행 중 = 아직 청산 안 됨.", ""]
    header = "| 기준일 | 시장 | 셋업 | " + " | ".join(f"{n}: 매수·계좌" for n, _, _ in RULES) + " | QQQ 26주 | QQQ 52주 |"
    lines += [header, "|" + "---|" * (5 + len(RULES))]
    totals: dict[str, list[float]] = {n: [] for n, _, _ in RULES}
    picks_by_day: dict[str, dict[str, list[dict]]] = {}
    for day in results:
        d = day["as_of"]
        cands = [r for r in recs if r["as_of"] == d and r.get("entered")]
        cells = []
        picks_by_day[d] = {}
        for name, _, pick in RULES:
            chosen = [r for r in cands if pick(r)][:MAX_POS]
            picks_by_day[d][name] = chosen
            acct = sum((r["ret"] or 0.0) * WEIGHT for r in chosen)
            totals[name].append(acct)
            wins = sum((r["ret"] or 0) > 0 for r in chosen)
            cells.append(f"{len(chosen)}종목 ({wins}승) · **{acct:+.1%}**" if chosen else "0 · 0.0%")
        q = day.get("bench", {}).get("QQQ", {})
        lines.append(f"| {d} | {'✅' if day['market_pass'] else '❌ 차단'} | {len(day['candidates'])} | "
                     + " | ".join(cells) + f" | {_pct(q.get('26w'))} | {_pct(q.get('52w'))} |")
    lines += ["", "## 규칙별 요약 (날짜당 계좌 수익률)", "",
              "| 규칙 | 설명 | 날짜당 평균 | 날짜당 중앙값 | 이익 난 날짜 | 거래 수 | 거래 승률 |", "|---|---|---|---|---|---|---|"]
    for name, desc, _ in RULES:
        t = totals[name]
        trades = [r for d in picks_by_day for r in picks_by_day[d][name]]
        wr = sum((r["ret"] or 0) > 0 for r in trades) / len(trades) if trades else float("nan")
        lines.append(f"| {name} | {desc} | {sum(t) / len(t):+.2%} | {pd.Series(t).median():+.2%} | "
                     f"{sum(x > 0 for x in t)}/{sum(x != 0 for x in t)} (매수 있던 날 기준) | {len(trades)} | {wr:.0%} |")
    for rule_name, note in (("거래량 완화", "대형주에서 현실적인 책 규칙"), ("전부", "대조군 — 등급 무시, 셋업 상위 5")):
        lines += ["", f"## 날짜별 매수 내역 — 규칙 '{rule_name}' ({note})", ""]
        for day in results:
            d = day["as_of"]
            chosen = picks_by_day[d][rule_name]
            m01 = {c["code"]: c for c in day["market"]}.get("0-1", {})
            lines.append(f"### {d} — 시장 {'상승세 ✅' if day['market_pass'] else '하락세 ❌ (매수 금지 규칙 1)'} · {m01.get('value', '')}")
            if not chosen:
                why = "셋업 없음" if not day["candidates"] else "셋업은 있으나 A 등급 없음 (탈락 사유: " + ", ".join(
                    sorted({r["grade_vol_opt"] for r in recs if r["as_of"] == d})[:4]) + ")"
                lines += [f"- 매수 없음 — {why}", ""]
                continue
            lines += ["", "| 종목 | 등급 | 유형 | 섹터 | 진입일 · 가격 | 청산일 · 가격 | 사유 | 수익률 | 보유 |", "|---|---|---|---|---|---|---|---|---|"]
            for r in chosen:
                lines.append(f"| {r['ticker']} | {r['grade_vol_opt']} | {r['entry_type']} | {r.get('sector') or '?'} | "
                             f"{r['entry_date']} · {r['entry_px']:.2f} | {r['exit_date']} · {r['exit_px']:.2f} | "
                             f"{'진행 중' if r['reason'] == 'open' else r['reason'][:22]} | **{_pct(r['ret'])}** | {r['weeks']}주 |")
            acct = sum((r["ret"] or 0.0) * WEIGHT for r in chosen)
            lines.append(f"\n계좌: {len(chosen)}종목 × 20% → **{acct:+.1%}** (현금 {1 - WEIGHT * len(chosen):.0%}) · "
                         f"종목 평균 {sum(r['ret'] for r in chosen) / len(chosen):+.1%} · QQQ 26주 {_pct(day.get('bench', {}).get('QQQ', {}).get('26w'))}\n")
    return "\n".join(lines)


if __name__ == "__main__":
    cfg = json.loads(sys.argv[1])
    if "analyze" in cfg:
        with open(cfg["analyze"]) as f:
            results = json.load(f)
        analyze(results)
        rep = portfolio_report(results)
        print(rep)
        if cfg.get("report_md"):
            with open(cfg["report_md"], "w") as f:
                f.write(rep)
    else:
        run(cfg)
