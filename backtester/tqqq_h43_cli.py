"""H43 — 완성 패키지 (설계 tqqq_search_runs/H43_설계.md): 두 단계 이탈+사다리(그 외) / 월말 스위치 18(R-C) 스티치, 3축 G6."""

from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from strategy.adapters.ladder_reentry import LadderParams, ladder_switch
from strategy.adapters.regime_slots import as_frame, regime_labels, RegimeParams, shuffle_labels, stitch
from strategy.adapters.tqqq_gates import GATE_COSTS, gate_verdicts, load_baseline, placebo_medians, verdict_markdown, window_table
from strategy.adapters.whipsaw_switch import WhipsawParams, whipsaw_switch
from tqqq_gates_cli import BASELINE, PRICES, load_cash_rate
from tqqq_h11_cli import load_qld

DD = [(0.15, 0.21), (0.18, 0.24), (0.21, 0.27)]; DEPTH = [0.30, 0.35, 0.40]; R1 = [0.45, 0.50]


def name_of(dd, dep, r1):
    return f"pkg dd{dd[0]:.0%}/{dd[1]:.0%} depth{dep:.2f} r1={r1:.0%}"


def build(px, dd, dep, r1, labels_cache):
    idx = px.index; q = px["QQQ"]
    lab = labels_cache[dep]
    lad = as_frame(ladder_switch(q, px["TQQQ"], LadderParams((r1, r1 + 0.15, r1 + 0.30), (1 / 3,) * 3, exit_steps=((dd[0], 0.5), (dd[1], 0.0)))), idx)
    msw = as_frame(whipsaw_switch(q, WhipsawParams(0.0, True, False)).reindex(idx).ffill().fillna(1.0), idx)
    slots = {"A": lad, "B": lad, "D": lad, "C": msw}
    return lab, slots, stitch(slots, lab, idx)


def neighbors(dd, dep, r1):
    out = []
    i = DD.index(dd); out += [(DD[j], dep, r1) for j in (i - 1, i + 1) if 0 <= j < 3]
    i = DEPTH.index(dep); out += [(dd, DEPTH[j], r1) for j in (i - 1, i + 1) if 0 <= j < 3]
    out += [(dd, dep, x) for x in R1 if x != r1]
    return out


def main(cfg):
    t0 = time.time(); px = pd.read_parquet(PRICES); base = load_baseline(BASELINE); cash = load_cash_rate()
    prices = pd.DataFrame({"TQQQ": px["TQQQ"], "QLD": load_qld(px.index, cash), "QQQ": px["QQQ"]})
    n_trials = int(cfg.get("n_trials", 1)); seeds = int(cfg.get("placebo_seeds", 30))
    labels_cache = {dep: regime_labels(px["QQQ"], RegimeParams(c_depth=dep)) for dep in DEPTH}
    combos = list(itertools.product(DD, DEPTH, R1)); rows, verdicts, tables, built = [], {}, {}, {}
    for i, (dd, dep, r1) in enumerate(combos):
        name = name_of(dd, dep, r1); lab, slots, w = build(px, dd, dep, r1, labels_cache); built[name] = (lab, slots, w)
        tbl = window_table(px["TQQQ"], w, base, "dev", costs=GATE_COSTS, cash_rate=cash, prices=prices); v = gate_verdicts(tbl, n_trials)
        tables[name], verdicts[name] = tbl, v; a10, ainf = v["G1"]["A-10"], v["G1"]["A-∞"]; a = tbl[tbl.bundle == "A-10"]
        rows.append({"name": name, "dd": dd, "depth": dep, "r1": r1, "a10_win": a10["win_rate"], "a10_med": a10["ratio_median"], "a10_min": a10["ratio_min"], "ainf_med": ainf["ratio_median"], "worst_mdd": v["G2"]["worst_mdd"], "g3_fail": v["G3"]["n_fail"],
                     "b_beat_tqqq": v["G4"]["beat_tqqq_rate"], "b_beat_qqq": v["G4"]["beat_qqq_rate"], "b_worst_cagr": v["G4"]["worst_cagr"], "b_loss": v["G4"]["loss_windows"], "b_dotcom": v["G4"]["dotcom_strat_median"],
                     "a10_ret_med": float(a["strat_ret"].median()), "a10_mdd_med": float(a["strat_mdd"].median()), "first_fail": v["first_fail"], "passed_through": v["passed_through"]})
        print(f"[{i + 1}/{len(combos)}] {name}: A-10 {a10['win_rate']:.0%}/{a10['ratio_median']:.2f}/{a10['ratio_min']:.2f} · A-∞ {ainf['ratio_median']:.2f} · MDD {v['G2']['worst_mdd']:.0%} · G3 미달 {v['G3']['n_fail']} · B {v['G4']['beat_tqqq_rate']:.0%}/QQQ {v['G4']['beat_qqq_rate']:.0%}/최악 {v['G4']['worst_cagr']:+.1%} 손실 {v['G4']['loss_windows']} 닷컴 {v['G4']['dotcom_strat_median']:+.0%} · 탈락 {v['first_fail']} ({time.time() - t0:.0f}s)", file=sys.stderr)
    for r in rows:
        if r["passed_through"] < 4:
            continue
        name = r["name"]; lab, slots, w = built[name]
        meds = placebo_medians(px["TQQQ"], w, base, seeds, 20261110, "runs", "dev", GATE_COSTS, cash, prices=prices)
        nb = [verdicts[name_of(*q)] for q in neighbors(r["dd"], r["depth"], r["r1"])]
        v2 = gate_verdicts(tables[name], n_trials, meds, nb)
        meds_lab = [float(window_table(px["TQQQ"], stitch(slots, shuffle_labels(lab, 20261110 + s_), px.index), base, "dev", ("A-10", "A-∞"), GATE_COSTS, cash, prices)["ratio"].median()) for s_ in range(seeds)]
        lab_pct = float(100.0 * (np.asarray(meds_lab) < v2["G5"]["strat_median"]).mean()); v2["G5"]["label_pct_ref"] = lab_pct; verdicts[name] = v2
        r.update({"first_fail": v2["first_fail"], "passed_through": v2["passed_through"], "g5_pct": v2["G5"]["percentile"], "g5_label": lab_pct, "g6": f"{v2['G6'].get('n_pass')}/{v2['G6'].get('n')}"})
        print(f"  G5/G6 {name}: 런 {v2['G5']['percentile']:.0f}% (라벨 참고 {lab_pct:.0f}%) · 이웃 {v2['G6']} · 탈락 {v2['first_fail']}", file=sys.stderr)
    d = pd.DataFrame(rows); out = {"cfg": cfg, "rows": rows, "verdicts": verdicts, "elapsed_s": time.time() - t0}
    order = {"G1": 0, "G2": 1, "G3": 2, "G4": 3, "G5": 4, "G6": 5, None: 6}; d = d.assign(_o=d["first_fail"].map(order)).sort_values(["_o", "a10_med"], ascending=[False, False])
    L = ["# H43 — 완성 패키지 (개발 판)", "", "| 조합 | A-10 승률·중앙·최악 | A-∞ 중앙 | A-10 세후 중앙·MDD | G2 최악 | G3 미달 | B TQQQ·QQQ 이김·최악 CAGR·손실·닷컴 | G5 런(라벨 참고) | G6 | 탈락 |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in d.itertuples(index=False):
        g5 = f"{r.g5_pct:.0f}% ({r.g5_label:.0f}%)" if hasattr(r, "g5_pct") and r.g5_pct == r.g5_pct else "—"
        L.append(f"| {r.name} | {r.a10_win:.0%} · {r.a10_med:.2f} · {r.a10_min:.2f} | {r.ainf_med:.2f} | {r.a10_ret_med:+.0%} · {r.a10_mdd_med:.0%} | {r.worst_mdd:.0%} | {r.g3_fail} | {r.b_beat_tqqq:.0%}·{r.b_beat_qqq:.0%}·{r.b_worst_cagr:+.1%}·{r.b_loss}·{r.b_dotcom:+.0%} | {g5} | {getattr(r, 'g6', '—')} | {r.first_fail or '통과'} |")
    best = d.iloc[0]; L += ["", f"## 최고 — {best['name']}", "", verdict_markdown(best["name"], tables[best["name"]], verdicts[best["name"]], "dev")]
    md = "\n".join(L)
    if cfg.get("out"): Path(cfg["out"]).write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
    if cfg.get("md"): Path(cfg["md"]).write_text(md)
    print(md[:2000]); return out


if __name__ == "__main__":
    main(json.loads(sys.argv[1]) if len(sys.argv) > 1 else {})
