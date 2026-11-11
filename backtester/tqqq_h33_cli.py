"""H33 — 사다리 + 월말 재진입(전 구간) → 관문 (설계 tqqq_search_runs/H33_설계.md)."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

from strategy.adapters.ladder_reentry import LadderParams, ladder_switch
from strategy.adapters.tqqq_gates import GATE_COSTS, gate_verdicts, load_baseline, placebo_medians, verdict_markdown, window_table
from tqqq_gates_cli import BASELINE, PRICES, load_cash_rate

L3 = (1 / 3,) * 3


def combos():
    return [("50/65/80 ⅓ · 월말 재진입", (0.50, 0.65, 0.80), True, False), ("40/55/70 ⅓ · 월말 재진입", (0.40, 0.55, 0.70), True, False),
            ("50/65/80 ⅓ · 월말 이탈+재진입", (0.50, 0.65, 0.80), True, True), ("40/55/70 ⅓ · 월말 이탈+재진입", (0.40, 0.55, 0.70), True, True),
            ("사다리 없음 · 월말 재진입(대조)", (1.5, 1.6, 1.7), True, False), ("50/65/80 ⅓ · 매일(=R46 대조)", (0.50, 0.65, 0.80), False, False)]


def main(cfg):
    t0 = time.time(); px = pd.read_parquet(PRICES); base = load_baseline(BASELINE); cash = load_cash_rate()
    n_trials = int(cfg.get("n_trials", 1)); seeds = int(cfg.get("placebo_seeds", 30)); all_ = combos()
    allg = pd.Series(True, index=px.index)
    rows, verdicts, tables, targets = [], {}, {}, {}
    for i, (name, rungs, m_re, m_ex) in enumerate(all_):
        t = ladder_switch(px["QQQ"], px["TQQQ"], LadderParams(rungs, L3), monthly_gate=allg if m_re else None, monthly_reentry=True, monthly_exit=m_ex); targets[name] = t
        tbl = window_table(px["TQQQ"], t, base, "dev", costs=GATE_COSTS, cash_rate=cash); v = gate_verdicts(tbl, n_trials)
        tables[name], verdicts[name] = tbl, v; a10, ainf = v["G1"]["A-10"], v["G1"]["A-∞"]; a = tbl[tbl.bundle == "A-10"]
        rows.append({"name": name, "rungs": rungs, "m_re": m_re, "m_ex": m_ex, "a10_win": a10["win_rate"], "a10_med": a10["ratio_median"], "a10_min": a10["ratio_min"], "ainf_med": ainf["ratio_median"], "worst_mdd": v["G2"]["worst_mdd"], "g3_fail": v["G3"]["n_fail"],
                     "b_beat_tqqq": v["G4"]["beat_tqqq_rate"], "b_beat_qqq": v["G4"]["beat_qqq_rate"], "b_worst_cagr": v["G4"]["worst_cagr"], "b_loss": v["G4"]["loss_windows"], "b_dotcom": v["G4"]["dotcom_strat_median"],
                     "a10_ret_med": float(a["strat_ret"].median()), "a10_mdd_med": float(a["strat_mdd"].median()), "first_fail": v["first_fail"], "passed_through": v["passed_through"]})
        print(f"[{i + 1}/6] {name}: A-10 {a10['win_rate']:.0%}/{a10['ratio_median']:.2f}/{a10['ratio_min']:.2f} · A-∞ {ainf['ratio_median']:.2f} · MDD {v['G2']['worst_mdd']:.0%} · G3 미달 {v['G3']['n_fail']} · B {v['G4']['beat_tqqq_rate']:.0%}/QQQ {v['G4']['beat_qqq_rate']:.0%}/최악 {v['G4']['worst_cagr']:+.1%} 손실 {v['G4']['loss_windows']} 닷컴 {v['G4']['dotcom_strat_median']:+.0%} · 탈락 {v['first_fail']} ({time.time() - t0:.0f}s)", file=sys.stderr)
    for r in rows:
        if r["passed_through"] < 3:
            continue
        name = r["name"]; meds = placebo_medians(px["TQQQ"], targets[name], base, seeds, 20261110, "runs", "dev", GATE_COSTS, cash)
        nb = [verdicts[q["name"]] for q in rows if q["name"] != name and (q["rungs"] == r["rungs"] or (q["m_re"] == r["m_re"] and q["m_ex"] == r["m_ex"]))]
        v2 = gate_verdicts(tables[name], n_trials, meds, nb); verdicts[name] = v2
        r.update({"first_fail": v2["first_fail"], "passed_through": v2["passed_through"], "g5_pct": v2["G5"]["percentile"], "g6": f"{v2['G6'].get('n_pass')}/{v2['G6'].get('n')}"})
        print(f"  G5/G6 {name}: {v2['G5']['percentile']:.0f}% (max {v2['G5']['placebo_max']:.2f}) · 이웃 {v2['G6']}", file=sys.stderr)
    d = pd.DataFrame(rows); out = {"cfg": cfg, "rows": rows, "verdicts": verdicts, "elapsed_s": time.time() - t0}
    order = {"G1": 0, "G2": 1, "G3": 2, "G4": 3, "G5": 4, "G6": 5, None: 6}; d = d.assign(_o=d["first_fail"].map(order)).sort_values(["_o", "a10_med"], ascending=[False, False])
    L = ["# H33 — 사다리 + 월말 재진입 (개발 판)", "", "| 조합 | A-10 승률·중앙·최악 | A-∞ 중앙 | A-10 세후 중앙·MDD | G2 최악 | G3 미달 | B TQQQ·QQQ 이김·최악 CAGR·손실·닷컴 | G5 | G6 | 탈락 |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in d.itertuples(index=False):
        L.append(f"| {r.name} | {r.a10_win:.0%} · {r.a10_med:.2f} · {r.a10_min:.2f} | {r.ainf_med:.2f} | {r.a10_ret_med:+.0%} · {r.a10_mdd_med:.0%} | {r.worst_mdd:.0%} | {r.g3_fail} | {r.b_beat_tqqq:.0%}·{r.b_beat_qqq:.0%}·{r.b_worst_cagr:+.1%}·{r.b_loss}·{r.b_dotcom:+.0%} | {getattr(r, 'g5_pct', float('nan')):.0f}% | {getattr(r, 'g6', '—')} | {r.first_fail or '통과'} |")
    best = d.iloc[0]; L += ["", f"## 최고 — {best['name']}", "", verdict_markdown(best["name"], tables[best["name"]], verdicts[best["name"]], "dev")]
    md = "\n".join(L)
    if cfg.get("out"): Path(cfg["out"]).write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
    if cfg.get("md"): Path(cfg["md"]).write_text(md)
    print(md[:1500]); return out


if __name__ == "__main__":
    main(json.loads(sys.argv[1]) if len(sys.argv) > 1 else {})
