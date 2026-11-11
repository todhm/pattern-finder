"""H32 — 스티치 v2(한 규칙: 스위치 18 + 사다리, R-C 안은 월말 판정) → 관문 (설계 tqqq_search_runs/H32_설계.md)."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from strategy.adapters.ladder_reentry import LadderParams, ladder_switch
from strategy.adapters.regime_slots import RegimeParams, regime_labels, shuffle_labels
from strategy.adapters.tqqq_gates import GATE_COSTS, gate_verdicts, load_baseline, placebo_medians, verdict_markdown, window_table
from tqqq_gates_cli import BASELINE, PRICES, load_cash_rate

RUNGS = (0.50, 0.65, 0.80)


def target(px, labels, size_in_c, monthly_reentry):
    gate_c = labels == "C"
    return ladder_switch(px["QQQ"], px["TQQQ"], LadderParams(RUNGS, (1 / 3,) * 3), gate=gate_c, gate_size=size_in_c, monthly_gate=gate_c, monthly_reentry=monthly_reentry)


def main(cfg):
    t0 = time.time(); px = pd.read_parquet(PRICES); base = load_baseline(BASELINE); cash = load_cash_rate()
    n_trials = int(cfg.get("n_trials", 1)); seeds = int(cfg.get("placebo_seeds", 30))
    combos = [(0.30, 1.0, True), (0.30, 0.5, True), (0.25, 1.0, True), (0.25, 0.5, True), (0.30, 1.0, False)]
    labs = {dep: regime_labels(px["QQQ"], RegimeParams(c_depth=dep)) for dep in (0.30, 0.25)}
    rows, verdicts, tables, targets = [], {}, {}, {}
    for i, (dep, sz, mre) in enumerate(combos):
        name = f"stitch2 C{dep:.2f} ladder×{sz:g} {'월말 이탈+재진입' if mre else '월말 이탈만'}"
        t = target(px, labs[dep], sz, mre); targets[name] = t
        tbl = window_table(px["TQQQ"], t, base, "dev", costs=GATE_COSTS, cash_rate=cash); v = gate_verdicts(tbl, n_trials)
        tables[name], verdicts[name] = tbl, v; a10, ainf = v["G1"]["A-10"], v["G1"]["A-∞"]; a = tbl[tbl.bundle == "A-10"]
        ch = t[(t.index >= "2010-02-11") & (t.index <= "2024-06-30")]; buys = [f"{d.date()}:{x:.2f}" for d, x in ch[(ch.diff() > 0) & (ch < 1.0)].items()]
        rows.append({"name": name, "depth": dep, "size": sz, "mre": mre, "a10_win": a10["win_rate"], "a10_med": a10["ratio_median"], "a10_min": a10["ratio_min"], "ainf_med": ainf["ratio_median"], "worst_mdd": v["G2"]["worst_mdd"], "g3_fail": v["G3"]["n_fail"],
                     "b_beat_tqqq": v["G4"]["beat_tqqq_rate"], "b_beat_qqq": v["G4"]["beat_qqq_rate"], "b_worst_cagr": v["G4"]["worst_cagr"], "b_loss": v["G4"]["loss_windows"], "b_dotcom": v["G4"]["dotcom_strat_median"],
                     "a10_ret_med": float(a["strat_ret"].median()), "a10_mdd_med": float(a["strat_mdd"].median()), "buys": buys, "first_fail": v["first_fail"], "passed_through": v["passed_through"]})
        print(f"[{i + 1}/5] {name}: A-10 {a10['win_rate']:.0%}/{a10['ratio_median']:.2f}/{a10['ratio_min']:.2f} · A-∞ {ainf['ratio_median']:.2f} · MDD {v['G2']['worst_mdd']:.0%} · G3 미달 {v['G3']['n_fail']} · B {v['G4']['beat_tqqq_rate']:.0%}/QQQ {v['G4']['beat_qqq_rate']:.0%}/최악 {v['G4']['worst_cagr']:+.1%} 손실 {v['G4']['loss_windows']} 닷컴 {v['G4']['dotcom_strat_median']:+.0%} · 탈락 {v['first_fail']} · 2022 매수 {[b for b in buys if b.startswith('2022')]} ({time.time() - t0:.0f}s)", file=sys.stderr)
    for r in rows:
        if r["passed_through"] < 3:
            continue
        name = r["name"]; t = targets[name]
        meds_runs = placebo_medians(px["TQQQ"], t, base, seeds, 20261110, "runs", "dev", GATE_COSTS, cash)
        meds_lab = []
        for s_ in range(seeds):
            t2 = target(px, shuffle_labels(labs[r["depth"]], 20261110 + s_), r["size"], r["mre"])
            meds_lab.append(float(window_table(px["TQQQ"], t2, base, "dev", ("A-10", "A-∞"), GATE_COSTS, cash)["ratio"].median()))
        nb = [verdicts[q["name"]] for q in rows if q["name"] != name and q["mre"] == r["mre"] and (q["depth"] == r["depth"] or q["size"] == r["size"])]
        v_runs = gate_verdicts(tables[name], n_trials, meds_runs, nb); v_lab = gate_verdicts(tables[name], n_trials, meds_lab, nb)
        worse = v_runs if v_runs["G5"]["percentile"] <= v_lab["G5"]["percentile"] else v_lab
        worse["G5"]["runs_pct"] = v_runs["G5"]["percentile"]; worse["G5"]["label_pct"] = v_lab["G5"]["percentile"]; verdicts[name] = worse
        r.update({"first_fail": worse["first_fail"], "passed_through": worse["passed_through"], "g5_runs": v_runs["G5"]["percentile"], "g5_label": v_lab["G5"]["percentile"], "g6": f"{worse['G6'].get('n_pass')}/{worse['G6'].get('n')}"})
        print(f"  G5 {name}: 런 {v_runs['G5']['percentile']:.0f}% · 라벨 {v_lab['G5']['percentile']:.0f}% (라벨 플라세보 중앙 {np.median(meds_lab):.2f}, 전략 {v_lab['G5']['strat_median']:.2f}) · G6 {worse['G6']}", file=sys.stderr)
    d = pd.DataFrame(rows); out = {"cfg": cfg, "rows": rows, "verdicts": verdicts, "elapsed_s": time.time() - t0}
    order = {"G1": 0, "G2": 1, "G3": 2, "G4": 3, "G5": 4, "G6": 5, None: 6}; d = d.assign(_o=d["first_fail"].map(order)).sort_values(["_o", "a10_med"], ascending=[False, False])
    L = ["# H32 — 스티치 v2 (개발 판)", "", "| 조합 | A-10 승률·중앙·최악 | A-∞ 중앙 | A-10 세후 중앙·MDD | G2 최악 | G3 미달 | B TQQQ·QQQ 이김·최악 CAGR·손실·닷컴 | G5 런/라벨 | G6 | 탈락 |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in d.itertuples(index=False):
        L.append(f"| {r.name} | {r.a10_win:.0%} · {r.a10_med:.2f} · {r.a10_min:.2f} | {r.ainf_med:.2f} | {r.a10_ret_med:+.0%} · {r.a10_mdd_med:.0%} | {r.worst_mdd:.0%} | {r.g3_fail} | {r.b_beat_tqqq:.0%}·{r.b_beat_qqq:.0%}·{r.b_worst_cagr:+.1%}·{r.b_loss}·{r.b_dotcom:+.0%} | {getattr(r, 'g5_runs', float('nan')):.0f}%/{getattr(r, 'g5_label', float('nan')):.0f}% | {getattr(r, 'g6', '—')} | {r.first_fail or '통과'} |")
    best = d.iloc[0]; L += ["", f"## 최고 — {best['name']}", "", verdict_markdown(best["name"], tables[best["name"]], verdicts[best["name"]], "dev")]
    md = "\n".join(L)
    if cfg.get("out"): Path(cfg["out"]).write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
    if cfg.get("md"): Path(cfg["md"]).write_text(md)
    print(md[:1500]); return out


if __name__ == "__main__":
    main(json.loads(sys.argv[1]) if len(sys.argv) > 1 else {})
