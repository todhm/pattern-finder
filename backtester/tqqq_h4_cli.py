"""H4 — 변동성 타기팅 TQQQ + 나머지 QQQ (+스위치 18) 그리드 → 관문 (설계 tqqq_search_runs/H4_설계.md).

    docker compose exec backtester python tqqq_h4_cli.py '{"out": "/app/sweep_results/tqqq_h4.json", "md": "/app/sweep_results/tqqq_h4.md", "n_trials": 5}'
"""

from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

import pandas as pd

from strategy.adapters.tqqq_gates import GATE_COSTS, gate_verdicts, load_baseline, placebo_medians, placebo_target, verdict_markdown, window_table
from strategy.adapters.vt_qqq_mix import VtMixParams, vt_mix_weights
from tqqq_gates_cli import BASELINE, PRICES, load_cash_rate

GRID = {"target": [0.36, 0.45, 0.54, 0.60, 0.72], "predictor": ["rv20", "rv60"], "switch": ["none", "cash", "qqq"]}


def neighbors(p: VtMixParams) -> list[VtMixParams]:
    out = []
    vals = GRID["target"]; i = vals.index(p.target)
    for j in (i - 1, i + 1):
        if 0 <= j < len(vals):
            out.append(VtMixParams(vals[j], p.predictor, p.switch))
    out.append(VtMixParams(p.target, "rv60" if p.predictor == "rv20" else "rv20", p.switch))
    return out


def main(cfg: dict) -> dict:
    t0 = time.time()
    px = pd.read_parquet(PRICES); base = load_baseline(BASELINE); cash = load_cash_rate()
    prices = px[["TQQQ", "QQQ"]]
    n_trials = int(cfg.get("n_trials", 1)); seeds = int(cfg.get("placebo_seeds", 30))
    combos = [VtMixParams(t, pr, sw) for t, pr, sw in itertools.product(*GRID.values())]
    if cfg.get("max_combos"):
        combos = combos[: int(cfg["max_combos"])]
    print(f"조합 {len(combos)} ({time.time() - t0:.0f}s)", file=sys.stderr)
    rows, verdicts, tables, weights = [], {}, {}, {}
    for i, p in enumerate(combos):
        w = vt_mix_weights(px["TQQQ"], px["QQQ"], p); weights[p.name] = w
        tbl = window_table(px["TQQQ"], w, base, "dev", costs=GATE_COSTS, cash_rate=cash, prices=prices)
        v = gate_verdicts(tbl, n_trials); tables[p.name], verdicts[p.name] = tbl, v
        a10, ainf = v["G1"]["A-10"], v["G1"]["A-∞"]; a = tbl[tbl.bundle == "A-10"]
        share = w[w.index >= "2010-02-11"].mean()
        rows.append({"name": p.name, **p.__dict__, "tqqq_w": float(share["TQQQ"]), "qqq_w": float(share["QQQ"]),
                     "a10_win": a10["win_rate"], "a10_med": a10["ratio_median"], "a10_min": a10["ratio_min"],
                     "ainf_win": ainf["win_rate"], "ainf_med": ainf["ratio_median"], "ainf_min": ainf["ratio_min"],
                     "g2_shallow": v["G2"]["shallower_rate"], "worst_mdd": v["G2"]["worst_mdd"], "g3_fail": v["G3"]["n_fail"],
                     "b_beat_tqqq": v["G4"]["beat_tqqq_rate"], "b_beat_qqq": v["G4"]["beat_qqq_rate"], "b_worst_cagr": v["G4"]["worst_cagr"],
                     "a10_ret_med": float(a["strat_ret"].median()), "a10_mdd_med": float(a["strat_mdd"].median()), "trades_a10": float(a["trades"].median()),
                     "tax_a10": float(a["tax"].median() / GATE_COSTS.initial_capital), "first_fail": v["first_fail"], "passed_through": v["passed_through"]})
        print(f"[{i + 1}/{len(combos)}] {p.name}: TQQQ 평균 {share['TQQQ']:.0%} · A-10 {a10['win_rate']:.0%}/{a10['ratio_median']:.2f}/{a10['ratio_min']:.2f} · A-∞ {ainf['win_rate']:.0%}/{ainf['ratio_median']:.2f} · "
              f"MDD {v['G2']['worst_mdd']:.0%} · G3 미달 {v['G3']['n_fail']} · 탈락 {v['first_fail']} ({time.time() - t0:.0f}s)", file=sys.stderr)
    for p in combos:
        v = verdicts[p.name]
        if v["first_fail"] not in (None, "G5", "G6"):
            continue
        w = weights[p.name]
        mode = "runs" if not placebo_target(w, 1, "runs").equals(w) else "shift"
        meds = placebo_medians(px["TQQQ"], w, base, seeds, 20261110, mode, "dev", GATE_COSTS, cash, prices=prices)
        nb = [verdicts[q.name] for q in neighbors(p) if q.name in verdicts]
        v2 = gate_verdicts(tables[p.name], n_trials, meds, nb); v2["G5"]["mode"] = mode; verdicts[p.name] = v2
        for r in rows:
            if r["name"] == p.name:
                r.update({"first_fail": v2["first_fail"], "passed_through": v2["passed_through"], "g5_pct": v2["G5"]["percentile"], "g6": f"{v2['G6'].get('n_pass')}/{v2['G6'].get('n')}"})
        print(f"  G5({mode})/G6 {p.name}: 백분위 {v2['G5']['percentile']:.0f}% · 이웃 {v2['G6']}", file=sys.stderr)
    df = pd.DataFrame(rows)
    out = {"cfg": cfg, "n_trials": n_trials, "grid": GRID, "rows": rows, "verdicts": verdicts, "elapsed_s": time.time() - t0}
    md = render(df, verdicts, tables)
    if cfg.get("out"):
        Path(cfg["out"]).write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
    if cfg.get("md"):
        Path(cfg["md"]).write_text(md)
    print(md[:2500])
    return out


def render(df, verdicts, tables) -> str:
    L = ["# H4 — 변동성 타기팅 TQQQ + 나머지 QQQ (+스위치 18) (개발 판, 끝 2024-06-30)", "", f"조합 {len(df)}", ""]
    order = {"G1": 0, "G2": 1, "G3": 2, "G4": 3, "G5": 4, "G6": 5, None: 6}
    d = df.assign(_o=df["first_fail"].map(order)).sort_values(["_o", "a10_med"], ascending=[False, False])
    L += ["| 목표σ | 예측 | 스위치 | 2010~ TQQQ·QQQ 평균 비중 | A-10 승률·중앙·최악 | A-∞ 승률·중앙·최악 | A-10 세후 중앙·MDD·매매·세금/자본 | G2 얕음·최악MDD | G3 미달 | B-10 TQQQ·QQQ 이김·최악 CAGR | G5 | G6 | 탈락 |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in d.itertuples(index=False):
        L.append(f"| {r.target:.2f} | {r.predictor} | {r.switch} | {r.tqqq_w:.0%}·{r.qqq_w:.0%} | {r.a10_win:.0%} · {r.a10_med:.2f} · {r.a10_min:.2f} | {r.ainf_win:.0%} · {r.ainf_med:.2f} · {r.ainf_min:.2f} | "
                 f"{r.a10_ret_med:+.0%} · {r.a10_mdd_med:.0%} · {r.trades_a10:.0f} · {r.tax_a10:.1f} | {r.g2_shallow:.0%}·{r.worst_mdd:.0%} | {r.g3_fail} | "
                 f"{r.b_beat_tqqq:.0%}·{r.b_beat_qqq:.0%}·{r.b_worst_cagr:+.1%} | {getattr(r, 'g5_pct', float('nan')):.0f}% | {getattr(r, 'g6', '—')} | {r.first_fail or '통과'} |")
    L.append("")
    best = d.iloc[0]
    L += [f"## 최고 조합 상세 — {best['name']}", "", verdict_markdown(best["name"], tables[best["name"]], verdicts[best["name"]], "dev")]
    return "\n".join(L)


if __name__ == "__main__":
    main(json.loads(sys.argv[1]) if len(sys.argv) > 1 else {})
