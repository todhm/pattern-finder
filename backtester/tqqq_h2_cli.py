"""H2 — 시장 폭 스위치 그리드 → 관문 G1~G6 (설계 tqqq_search_runs/H2_설계.md).

    docker compose exec backtester python tqqq_h2_cli.py '{"out": "/app/sweep_results/tqqq_h2.json", "md": "/app/sweep_results/tqqq_h2.md", "n_trials": 3}'
"""

from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

import pandas as pd

from strategy.adapters.breadth_switch import BreadthParams, breadth_series, breadth_switch
from strategy.adapters.tqqq_gates import GATE_COSTS, gate_verdicts, load_baseline, placebo_medians, verdict_markdown, window_table
from tqqq_gates_cli import BASELINE, PRICES, load_cash_rate

CLOSES = "/app/sweep_results/ndx100_current_close_1985_2026-10-24.parquet"
GRID = {"exit_pct": [0.30, 0.40, 0.50], "reenter_pct": [0.50, 0.60, 0.70], "off_fraction": [0.0, 0.33]}


def neighbors(p: BreadthParams) -> list[BreadthParams]:
    out = []
    for key in ("exit_pct", "reenter_pct"):
        vals = GRID[key]; i = vals.index(getattr(p, key))
        for j in (i - 1, i + 1):
            if 0 <= j < len(vals):
                out.append(BreadthParams(**{**p.__dict__, key: vals[j]}))
    return out


def main(cfg: dict) -> dict:
    t0 = time.time()
    px = pd.read_parquet(PRICES); base = load_baseline(BASELINE); cash = load_cash_rate()
    closes = pd.read_parquet(CLOSES)
    breadth = breadth_series(closes, 200, 30).reindex(px.index).ffill(limit=5)
    valid_from = breadth.dropna().index[0].date()
    n_trials = int(cfg.get("n_trials", 1)); seeds = int(cfg.get("placebo_seeds", 30))
    combos = [BreadthParams(exit_pct=x, reenter_pct=y, off_fraction=o) for x, y, o in itertools.product(*GRID.values())]
    if cfg.get("max_combos"):
        combos = combos[: int(cfg["max_combos"])]
    print(f"조합 {len(combos)}, 폭 유효 {valid_from}~ ({time.time() - t0:.0f}s)", file=sys.stderr)
    rows, verdicts, tables = [], {}, {}
    for i, p in enumerate(combos):
        tgt = breadth_switch(px["QQQ"], breadth, p)
        tbl = window_table(px["TQQQ"], tgt, base, "dev", costs=GATE_COSTS, cash_rate=cash)
        v = gate_verdicts(tbl, n_trials)
        tables[p.name], verdicts[p.name] = tbl, v
        a10, ainf = v["G1"]["A-10"], v["G1"]["A-∞"]
        rows.append({"name": p.name, **p.__dict__,
                     "a10_win": a10["win_rate"], "a10_med": a10["ratio_median"], "a10_min": a10["ratio_min"],
                     "ainf_win": ainf["win_rate"], "ainf_med": ainf["ratio_median"], "ainf_min": ainf["ratio_min"],
                     "g2_shallow": v["G2"]["shallower_rate"], "worst_mdd": v["G2"]["worst_mdd"], "g3_fail": v["G3"]["n_fail"],
                     "b_beat_tqqq": v["G4"]["beat_tqqq_rate"], "b_beat_qqq": v["G4"]["beat_qqq_rate"], "b_worst_cagr": v["G4"]["worst_cagr"],
                     "a10_ret_med": float(tbl[tbl.bundle == "A-10"]["strat_ret"].median()), "a10_mdd_med": float(tbl[tbl.bundle == "A-10"]["strat_mdd"].median()),
                     "trades_a10": float(tbl[tbl.bundle == "A-10"]["trades"].median()), "tim_a10": float(tbl[tbl.bundle == "A-10"]["time_in_market"].median()),
                     "first_fail": v["first_fail"], "passed_through": v["passed_through"]})
        print(f"[{i + 1}/{len(combos)}] {p.name}: A-10 {a10['win_rate']:.0%}/{a10['ratio_median']:.2f}/{a10['ratio_min']:.2f} · A-∞ {ainf['win_rate']:.0%}/{ainf['ratio_median']:.2f} · "
              f"G3 미달 {v['G3']['n_fail']} · 탈락 {v['first_fail']} ({time.time() - t0:.0f}s)", file=sys.stderr)
    for p in combos:
        v = verdicts[p.name]
        if v["first_fail"] not in (None, "G5", "G6"):
            continue
        tgt = breadth_switch(px["QQQ"], breadth, p)
        meds = placebo_medians(px["TQQQ"], tgt, base, seeds, 20261110, "runs", "dev", GATE_COSTS, cash)
        nb = [verdicts[q.name] for q in neighbors(p) if q.name in verdicts]
        v2 = gate_verdicts(tables[p.name], n_trials, meds, nb); verdicts[p.name] = v2
        for r in rows:
            if r["name"] == p.name:
                r.update({"first_fail": v2["first_fail"], "passed_through": v2["passed_through"], "g5_pct": v2["G5"]["percentile"], "g6": f"{v2['G6'].get('n_pass')}/{v2['G6'].get('n')}"})
        print(f"  G5/G6 {p.name}: 백분위 {v2['G5']['percentile']:.0f}% · 이웃 {v2['G6']}", file=sys.stderr)
    df = pd.DataFrame(rows)
    out = {"cfg": cfg, "n_trials": n_trials, "grid": GRID, "breadth_valid_from": str(valid_from), "rows": rows, "verdicts": verdicts, "elapsed_s": time.time() - t0}
    # 폭 자체의 요약(기록용): 연도별 최저 폭
    out["breadth_yearly_min"] = {int(k): float(v) for k, v in breadth.groupby(breadth.index.year).min().dropna().items()}
    md = render(df, verdicts, tables, valid_from, out["breadth_yearly_min"])
    if cfg.get("out"):
        Path(cfg["out"]).write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
    if cfg.get("md"):
        Path(cfg["md"]).write_text(md)
    print(md[:2500])
    return out


def render(df, verdicts, tables, valid_from, byear) -> str:
    L = ["# H2 — 시장 폭 스위치 (개발 판, 끝 2024-06-30)", "", f"조합 {len(df)} · 폭 유효 {valid_from}~ (현재 구성 101 종목, 유효 ≥30) · 연도별 최저 폭: "
         + ", ".join(f"{y} {v:.0%}" for y, v in byear.items() if y in (1987, 1990, 1998, 2000, 2001, 2002, 2008, 2011, 2015, 2018, 2020, 2022)), ""]
    order = {"G1": 0, "G2": 1, "G3": 2, "G4": 3, "G5": 4, "G6": 5, None: 6}
    d = df.assign(_o=df["first_fail"].map(order)).sort_values(["_o", "a10_med"], ascending=[False, False])
    L += ["| 이탈 X | 복귀 Y | off | A-10 승률·중앙·최악 | A-∞ 승률·중앙·최악 | A-10 세후 중앙 · MDD · 매매 · 체류 | G2 얕음·최악MDD | G3 미달 | B-10 TQQQ·QQQ 이김·최악 CAGR | G5 | G6 | 탈락 |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in d.itertuples(index=False):
        L.append(f"| {r.exit_pct:.0%} | {r.reenter_pct:.0%} | {r.off_fraction:g} | {r.a10_win:.0%} · {r.a10_med:.2f} · {r.a10_min:.2f} | {r.ainf_win:.0%} · {r.ainf_med:.2f} · {r.ainf_min:.2f} | "
                 f"{r.a10_ret_med:+.0%} · {r.a10_mdd_med:.0%} · {r.trades_a10:.0f} · {r.tim_a10:.0%} | {r.g2_shallow:.0%}·{r.worst_mdd:.0%} | {r.g3_fail} | "
                 f"{r.b_beat_tqqq:.0%}·{r.b_beat_qqq:.0%}·{r.b_worst_cagr:+.1%} | {getattr(r, 'g5_pct', float('nan')):.0f}% | {getattr(r, 'g6', '—')} | {r.first_fail or '통과'} |")
    L.append("")
    best = d.iloc[0]
    L += [f"## 최고 조합 상세 — {best['name']}", "", verdict_markdown(best["name"], tables[best["name"]], verdicts[best["name"]], "dev")]
    return "\n".join(L)


if __name__ == "__main__":
    main(json.loads(sys.argv[1]) if len(sys.argv) > 1 else {})
