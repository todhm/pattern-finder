"""H18 — 직전 L* 로 배수 선택 그리드 → 관문 (설계 tqqq_search_runs/H18_설계.md)."""

from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from strategy.adapters.lstar_select import LstarParams, lstar_weights
from strategy.adapters.tqqq_gates import GATE_COSTS, gate_verdicts, load_baseline, verdict_markdown, window_table
from tqqq_gates_cli import BASELINE, PRICES, load_cash_rate
from tqqq_h11_cli import load_qld

FIRST = pd.Timestamp("1991-01-01")
GRID = {"years": [3, 5], "thr": [(3.0, 2.0), (2.5, 1.5)], "switch": [False, True]}


def neighbors(p):
    return [LstarParams(5 if p.years == 3 else 3, p.hi, p.lo, p.switch), LstarParams(p.years, *((2.5, 1.5) if p.hi == 3.0 else (3.0, 2.0)), p.switch)]


def main(cfg):
    t0 = time.time(); px = pd.read_parquet(PRICES); base = load_baseline(BASELINE); cash = load_cash_rate(); qld = load_qld(px.index, cash)
    prices = pd.DataFrame({"TQQQ": px["TQQQ"], "QLD": qld, "QQQ": px["QQQ"]})
    n_trials = int(cfg.get("n_trials", 1)); seeds = int(cfg.get("placebo_seeds", 30))
    combos = [LstarParams(y, hi, lo, sw) for y, (hi, lo), sw in itertools.product(GRID["years"], GRID["thr"], GRID["switch"])]
    rows, verdicts, tables = [], {}, {}
    for i, p in enumerate(combos):
        w, picks = lstar_weights(px["QQQ"], p, FIRST)
        tbl = window_table(px["TQQQ"], w, base, "dev", costs=GATE_COSTS, cash_rate=cash, prices=prices); v = gate_verdicts(tbl, n_trials)
        tables[p.name], verdicts[p.name] = tbl, v; a10, ainf = v["G1"]["A-10"], v["G1"]["A-∞"]; a = tbl[tbl.bundle == "A-10"]
        share = picks[(picks.index >= "2010-01-01") & (picks.index <= "2024-06-30")].value_counts(normalize=True).round(2).to_dict()
        rows.append({"name": p.name, "years": p.years, "hi": p.hi, "lo": p.lo, "switch": p.switch, "picks_2010": share,
                     "a10_win": a10["win_rate"], "a10_med": a10["ratio_median"], "a10_min": a10["ratio_min"], "ainf_med": ainf["ratio_median"], "worst_mdd": v["G2"]["worst_mdd"], "g3_fail": v["G3"]["n_fail"],
                     "b_beat_tqqq": v["G4"]["beat_tqqq_rate"], "b_beat_qqq": v["G4"]["beat_qqq_rate"], "b_worst_cagr": v["G4"]["worst_cagr"], "b_dotcom": v["G4"]["dotcom_strat_median"],
                     "a10_ret_med": float(a["strat_ret"].median()), "a10_mdd_med": float(a["strat_mdd"].median()), "first_fail": v["first_fail"], "passed_through": v["passed_through"]})
        print(f"[{i + 1}/8] {p.name}: 2010~ {share} · A-10 {a10['win_rate']:.0%}/{a10['ratio_median']:.2f}/{a10['ratio_min']:.2f} · A-∞ {ainf['ratio_median']:.2f} · MDD {v['G2']['worst_mdd']:.0%} · G3 미달 {v['G3']['n_fail']} · B {v['G4']['beat_tqqq_rate']:.0%}/{v['G4']['worst_cagr']:+.1%} 닷컴 {v['G4']['dotcom_strat_median']:+.0%} · 탈락 {v['first_fail']} ({time.time() - t0:.0f}s)", file=sys.stderr)
    for p in combos:
        v = verdicts[p.name]
        if v["first_fail"] not in (None, "G5", "G6"):
            continue
        meds = []
        for s in range(seeds):
            w2, _ = lstar_weights(px["QQQ"], p, FIRST, rng=np.random.default_rng(20261110 + s))
            meds.append(float(window_table(px["TQQQ"], w2, base, "dev", ("A-10", "A-∞"), GATE_COSTS, cash, prices)["ratio"].median()))
        nb = [verdicts[q.name] for q in neighbors(p) if q.name in verdicts]; v2 = gate_verdicts(tables[p.name], n_trials, meds, nb); verdicts[p.name] = v2
        for r in rows:
            if r["name"] == p.name:
                r.update({"first_fail": v2["first_fail"], "passed_through": v2["passed_through"], "g5_pct": v2["G5"]["percentile"], "g6": f"{v2['G6'].get('n_pass')}/{v2['G6'].get('n')}"})
    df = pd.DataFrame(rows); out = {"cfg": cfg, "rows": rows, "verdicts": verdicts, "elapsed_s": time.time() - t0}
    order = {"G1": 0, "G2": 1, "G3": 2, "G4": 3, "G5": 4, "G6": 5, None: 6}; d = df.assign(_o=df["first_fail"].map(order)).sort_values(["_o", "a10_med"], ascending=[False, False])
    L = ["# H18 — 직전 L* 배수 선택 (개발 판)", "", "| 조합 | 2010~ 선택 | A-10 승률·중앙·최악 | A-∞ 중앙 | A-10 세후 중앙·MDD | G2 최악 MDD | G3 미달 | B TQQQ 이김·최악 CAGR·닷컴 | G5 | G6 | 탈락 |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in d.itertuples(index=False):
        L.append(f"| {r.name} | {r.picks_2010} | {r.a10_win:.0%} · {r.a10_med:.2f} · {r.a10_min:.2f} | {r.ainf_med:.2f} | {r.a10_ret_med:+.0%} · {r.a10_mdd_med:.0%} | {r.worst_mdd:.0%} | {r.g3_fail} | {r.b_beat_tqqq:.0%}·{r.b_worst_cagr:+.1%}·{r.b_dotcom:+.0%} | {getattr(r, 'g5_pct', float('nan')):.0f}% | {getattr(r, 'g6', '—')} | {r.first_fail or '통과'} |")
    best = d.iloc[0]; L += ["", f"## 최고 — {best['name']}", "", verdict_markdown(best["name"], tables[best["name"]], verdicts[best["name"]], "dev")]
    md = "\n".join(L)
    if cfg.get("out"): Path(cfg["out"]).write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
    if cfg.get("md"): Path(cfg["md"]).write_text(md)
    print(md[:1200]); return out


if __name__ == "__main__":
    main(json.loads(sys.argv[1]) if len(sys.argv) > 1 else {})
