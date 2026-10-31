"""근거중심매매 규칙 백테스트 — 레짐별 무작위 시작일, 매일 재판정, 플라세보 대조.

    docker compose exec backtester python evidence_backtest_cli.py '{"n_each": 3, "seed": 7, "horizon_days": 130,
        "universe": "sp500+nasdaq100", "out": "/app/sweep_results/evidence_bt_2026-10-28.json",
        "report_md": "/app/sweep_results/evidence_bt_2026-10-28.md", "placebo_seeds": 5, "workers": 4}'

cfg 추가 키: "preset": "book"(기본, 2026-10-29 책 재대조판) | "legacy"(첫 실행판), "rules": {SimRules 필드 덮어쓰기},
  "dates": ["YYYY-MM-DD", ...](무작위 대신 지정), "max_tickers": N.

엔진 strategy/adapters/evidence_backtest.py. 데이터 yfinance 일괄(오늘 명단 → 생존 편향 있음, 플라세보가 같은 편향을 공유).
"""

from __future__ import annotations

import json
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import date

import numpy as np
import pandas as pd

from data.adapters.wikipedia_universe import default_universe_provider
from data.adapters.yfinance_bulk import download_many
from dataclasses import asdict

from strategy.adapters.evidence_backtest import (
    SimRules, buy_hold, classify_regimes, legacy_rules, pick_dates, placebo_full, placebo_sel, precompute, run_sim,
)
from strategy.adapters.evidence_checklist import EvidenceParams

_DATA = None  # 프로세스별 전역 (fork 로 공유)


def _one(args):
    reg, start_iso, rules_d, placebo_seeds = args
    data, cal, qqq, spy, idx = _DATA
    rules = SimRules(**rules_d)
    start = pd.Timestamp(start_iso)
    t0 = time.time()
    cache: dict = {}
    res = run_sim(data, cal, start, rules, screen_cache=cache, index_data=idx)
    end = pd.Timestamp(res.end)
    universe = list(data)
    ps, pf = [], []
    for s in range(placebo_seeds):
        rng = np.random.default_rng(1000 + s)
        ps.append(run_sim(data, cal, start, rules, picker=placebo_sel(rng, universe), screen_cache=cache, index_data=idx).stats)
        rng2 = np.random.default_rng(2000 + s)
        pf.append(run_sim(data, cal, start, rules, picker=placebo_full(rng2, universe), screen_cache=cache, index_data=idx).stats)
    row = {
        "레짐": reg, "시작": res.start, "종료": res.end, **res.stats,
        "QQQ": buy_hold(qqq, start, end), "SPY": buy_hold(spy, start, end),
        "플라세보 선택 수익률": float(np.mean([x["수익률"] for x in ps])), "플라세보 선택 MDD": float(np.mean([x["MDD"] for x in ps])),
        "플라세보 선택 승률": float(np.nanmean([x["승률"] if x["승률"] is not None else np.nan for x in ps])),
        "플라세보 상시 수익률": float(np.mean([x["수익률"] for x in pf])), "플라세보 상시 MDD": float(np.mean([x["MDD"] for x in pf])),
        "주간 후보 수": {k: (len(v) if v != ["(지수 게이트)"] else "게이트") for k, v in res.signals.items()},
        "거래": [{"티커": tr.ticker, "신호": tr.signal, "진입": tr.entry, "진입가": round(tr.entry_px, 2), "손절": round(tr.stop, 2),
                 "2차": (round(tr.second_px, 2) if tr.second_px else False), "1:1": round(tr.target1, 2) if tr.tp1_done else False, "손익": round(tr.pnl(), 4),
                 "청산": [(d, round(px, 2), rs) for d, px, _, rs in tr.exits]} for tr in res.trades],
        "equity": {d.date().isoformat(): round(v, 5) for d, v in res.equity.items()},
        "소요초": round(time.time() - t0),
    }
    return row


def main(cfg: dict) -> None:
    global _DATA
    n_each = int(cfg.get("n_each", 3)); seed = int(cfg.get("seed", 7))
    over = dict(cfg.get("rules", {})); over.setdefault("horizon_days", int(cfg.get("horizon_days", 130)))
    rules = legacy_rules(**over) if cfg.get("preset") == "legacy" else SimRules(**over)
    prov = default_universe_provider()
    uni = cfg.get("universe", "sp500+nasdaq100")
    if uni == "sp500+nasdaq100":
        tickers = list(dict.fromkeys(prov.get_tickers("sp500") + prov.get_tickers("nasdaq100")))
    else:
        tickers = prov.get_tickers(uni)
    if cfg.get("max_tickers"):
        tickers = tickers[: int(cfg["max_tickers"])]
    print(f"universe {len(tickers)}", file=sys.stderr)
    t0 = time.time()
    prices = download_many(tickers + ["QQQ", "SPY"], date(2003, 1, 1), date.today())
    qqq, spy = prices.pop("QQQ"), prices.pop("SPY")
    print(f"download {len(prices)} tickers {time.time() - t0:.0f}s", file=sys.stderr)
    t0 = time.time()
    data = precompute({t: df for t, df in prices.items() if t in tickers}, EvidenceParams())
    idx = precompute({"QQQ": qqq}, EvidenceParams())["QQQ"]
    print(f"precompute {len(data)} {time.time() - t0:.0f}s", file=sys.stderr)
    cal = pd.to_datetime(qqq.index)
    regimes = classify_regimes(qqq)
    lo, hi = cfg.get("date_lo", "2006-01-01"), cfg.get("date_hi", "2026-03-31")
    dates = pick_dates(regimes, n_each, seed, lo, hi)
    if cfg.get("dates"):
        dates = [(regimes.loc[regimes.index <= pd.Timestamp(d)].regime.iloc[-1], pd.Timestamp(d)) for d in cfg["dates"]]
    print("dates", [(r, d.date().isoformat()) for r, d in dates], file=sys.stderr)
    _DATA = (data, cal, qqq, spy, idx)
    print("rules", asdict(rules), file=sys.stderr)
    jobs = [(reg, d.isoformat(), asdict(rules), int(cfg.get("placebo_seeds", 5))) for reg, d in dates]
    rows = []
    workers = int(cfg.get("workers", 4))
    # 3.14 기본 시작 방식(forkserver)은 전역 _DATA 를 물려주지 않는다 → fork 로 공유
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("fork")) as ex:
        for row in ex.map(_one, jobs):
            rows.append(row)
            print(f"done {row['레짐']} {row['시작']} 수익 {row['수익률']:+.1%} QQQ {row['QQQ']:+.1%} 플라세보선택 {row['플라세보 선택 수익률']:+.1%} ({row['소요초']}s)", file=sys.stderr)
    out = {"config": cfg, "rules": asdict(rules), "regime_counts": regimes.regime.value_counts().to_dict(), "rows": rows}
    with open(cfg["out"], "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=str)
    write_md(out, cfg["report_md"])


def write_md(out: dict, path: str) -> None:
    rows = out["rows"]
    L = ["# 근거중심매매 규칙 백테스트 (레짐별 무작위 시작일, 매일 재판정)", "",
         f"규칙: {out['rules']} · 레짐 월수: {out['regime_counts']}", "",
         "| 레짐 | 시작 | 종료 | 전략 | 세후 | MDD | 거래 | 승률 | 평균보유일 | 1:1 도달 | QQQ | SPY | 플라세보 선택 | 플라세보 상시 |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    f = lambda v: "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:+.1%}"
    for r in rows:
        L.append(f"| {r['레짐']} | {r['시작']} | {r['종료']} | {f(r['수익률'])} | {f(r['세후 수익률'])} | {f(r['MDD'])} | {r['거래 수']} | "
                 f"{'—' if r['승률'] is None else format(r['승률'], '.0%')} | {'—' if r['평균 보유일'] is None else format(r['평균 보유일'], '.0f')} | {r['1:1 도달']} | "
                 f"{f(r['QQQ'])} | {f(r['SPY'])} | {f(r['플라세보 선택 수익률'])} (MDD {f(r['플라세보 선택 MDD'])}) | {f(r['플라세보 상시 수익률'])} |")
    L.append("")
    df = pd.DataFrame(rows)
    for reg, g in df.groupby("레짐"):
        L.append(f"- **{reg}** ({len(g)}회): 전략 평균 {g['수익률'].mean():+.1%} (MDD {g['MDD'].mean():+.1%}) · QQQ {g['QQQ'].mean():+.1%} · "
                 f"플라세보 선택 {g['플라세보 선택 수익률'].mean():+.1%} · 플라세보 상시 {g['플라세보 상시 수익률'].mean():+.1%} · 거래 {g['거래 수'].mean():.1f}회")
    L.append(f"- **전체** ({len(df)}회): 전략 {df['수익률'].mean():+.1%} / 세후 {df['세후 수익률'].mean():+.1%} · QQQ {df['QQQ'].mean():+.1%} · SPY {df['SPY'].mean():+.1%} · "
             f"플라세보 선택 {df['플라세보 선택 수익률'].mean():+.1%} · 플라세보 상시 {df['플라세보 상시 수익률'].mean():+.1%}")
    L.append("")
    L.append("## 거래 기록")
    for r in rows:
        L.append(f"\n### {r['레짐']} {r['시작']} → {r['종료']} · 청산 사유 {r['청산 사유']} · 주간 후보 수 {list(r['주간 후보 수'].values())}")
        L.append("| 티커 | 신호(금) | 진입 | 진입가 | 손절 | 2차 | 1:1 | 손익 | 청산 |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for t in r["거래"]:
            L.append(f"| {t['티커']} | {t['신호']} | {t['진입']} | {t['진입가']} | {t['손절']} | {t['2차'] or '-'} | {t['1:1'] or '-'} | {t['손익']:+.2%} | {t['청산']} |")
    with open(path, "w") as fh:
        fh.write("\n".join(L))


if __name__ == "__main__":
    main(json.loads(sys.argv[1]))
