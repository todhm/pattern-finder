"""H8 — 반도체 ↔ TQQQ 상대 모멘텀 그리드 → 관문 (설계 tqqq_search_runs/H8_설계.md)."""

from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

import pandas as pd

from strategy.adapters.sector_rotation import SectorRotParams, sector_rotation_weights
from strategy.adapters.tqqq_gates import GATE_COSTS, gate_verdicts, load_baseline, placebo_medians, verdict_markdown, window_table
from tqqq_gates_cli import BASELINE, PRICES, load_cash_rate

SOXX = "/app/sweep_results/soxx_spliced_1985_2026-10-24.parquet"
GRID = {"lookback_m": [3, 6, 12], "threshold": [0.0, 0.05], "base": ["qqq", "tqqq"]}
B_MIN_START = pd.Timestamp("1995-07-01")


def neighbors(p: SectorRotParams) -> list[SectorRotParams]:
    out = []; vals = GRID["lookback_m"]; i = vals.index(p.lookback_m)
    for j in (i - 1, i + 1):
        if 0 <= j < len(vals):
            out.append(SectorRotParams(vals[j], p.threshold, p.base))
    out.append(SectorRotParams(p.lookback_m, 0.05 if p.threshold == 0.0 else 0.0, p.base))
    return out


def main(cfg: dict) -> dict:
    t0 = time.time()
    px = pd.read_parquet(PRICES); base = load_baseline(BASELINE); cash = load_cash_rate()
    sox = pd.read_parquet(SOXX).iloc[:, 0].reindex(px.index).ffill()
    prices = pd.DataFrame({"TQQQ": px["TQQQ"], "SOXX": sox})
    n_trials = int(cfg.get("n_trials", 1)); seeds = int(cfg.get("placebo_seeds", 30))
    combos = [SectorRotParams(l, t, b) for l, t, b in itertools.product(*GRID.values())]
    rows, verdicts, tables, weights = [], {}, {}, {}

    def table_for(w):
        tA = window_table(px["TQQQ"], w, base, "dev", ("A-10", "A-∞", "S-5"), GATE_COSTS, cash, prices)
        tB = window_table(px["TQQQ"], w, base, "dev", ("B-10",), GATE_COSTS, cash, prices, min_start=B_MIN_START)
        return pd.concat([tA, tB], ignore_index=True)

    for i, p in enumerate(combos):
        w = sector_rotation_weights(px["TQQQ"], px["QQQ"], sox, p); weights[p.name] = w
        tbl = table_for(w); v = gate_verdicts(tbl, n_trials); tables[p.name], verdicts[p.name] = tbl, v
        a10, ainf = v["G1"]["A-10"], v["G1"]["A-∞"]; a = tbl[tbl.bundle == "A-10"]
        share = float(w.loc[w.index >= "2010-02-11", "SOXX"].mean())
        rows.append({"name": p.name, **p.__dict__, "soxx_share": share,
                     "a10_win": a10["win_rate"], "a10_med": a10["ratio_median"], "a10_min": a10["ratio_min"],
                     "ainf_win": ainf["win_rate"], "ainf_med": ainf["ratio_median"], "ainf_min": ainf["ratio_min"],
                     "g2_shallow": v["G2"]["shallower_rate"], "worst_mdd": v["G2"]["worst_mdd"], "g3_fail": v["G3"]["n_fail"],
                     "b_n": v["G4"].get("n"), "b_beat_tqqq": v["G4"].get("beat_tqqq_rate"), "b_beat_qqq": v["G4"].get("beat_qqq_rate"), "b_worst_cagr": v["G4"].get("worst_cagr"),
                     "a10_ret_med": float(a["strat_ret"].median()), "a10_mdd_med": float(a["strat_mdd"].median()), "trades_a10": float(a["trades"].median()),
                     "first_fail": v["first_fail"], "passed_through": v["passed_through"]})
        print(f"[{i + 1}/{len(combos)}] {p.name}: SOXX 체류 {share:.0%} · A-10 {a10['win_rate']:.0%}/{a10['ratio_median']:.2f}/{a10['ratio_min']:.2f} · A-∞ {ainf['win_rate']:.0%}/{ainf['ratio_median']:.2f} · "
              f"MDD {v['G2']['worst_mdd']:.0%} · G3 미달 {v['G3']['n_fail']} · B({v['G4'].get('n')}) {v['G4'].get('beat_tqqq_rate', 0):.0%}/{v['G4'].get('worst_cagr', 0):+.1%} · 탈락 {v['first_fail']} ({time.time() - t0:.0f}s)", file=sys.stderr)
    for p in combos:
        v = verdicts[p.name]
        if v["first_fail"] not in (None, "G5", "G6"):
            continue
        meds = placebo_medians(px["TQQQ"], weights[p.name], base, seeds, 20261110, "runs", "dev", GATE_COSTS, cash, prices=prices)
        nb = [verdicts[q.name] for q in neighbors(p) if q.name in verdicts]
        v2 = gate_verdicts(tables[p.name], n_trials, meds, nb); verdicts[p.name] = v2
        for r in rows:
            if r["name"] == p.name:
                r.update({"first_fail": v2["first_fail"], "passed_through": v2["passed_through"], "g5_pct": v2["G5"]["percentile"], "g6": f"{v2['G6'].get('n_pass')}/{v2['G6'].get('n')}"})
        print(f"  G5/G6 {p.name}: 백분위 {v2['G5']['percentile']:.0f}% · 이웃 {v2['G6']}", file=sys.stderr)
    df = pd.DataFrame(rows)
    out = {"cfg": cfg, "n_trials": n_trials, "grid": GRID, "b_min_start": str(B_MIN_START.date()), "rows": rows, "verdicts": verdicts, "elapsed_s": time.time() - t0}
    md = render(df, verdicts, tables)
    if cfg.get("out"):
        Path(cfg["out"]).write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
    if cfg.get("md"):
        Path(cfg["md"]).write_text(md)
    print(md[:2500])
    return out


def render(df, verdicts, tables) -> str:
    L = ["# H8 — 반도체(SOXX 1배) ↔ TQQQ 상대 모멘텀 (개발 판, 끝 2024-06-30)", "", f"조합 {len(df)} · ^SOX 1994-05~ + SOXX 2001-07~ · B-10 은 1995-07 이후 시작 창만", "",
         "| L | 문턱 | 기준 | 2010~ SOXX 체류 | A-10 승률·중앙·최악 | A-∞ 승률·중앙·최악 | A-10 세후 중앙·MDD·매매 | G2 얕음·최악MDD | G3 미달 | B(창)·TQQQ·QQQ 이김·최악 CAGR | G5 | G6 | 탈락 |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    order = {"G1": 0, "G2": 1, "G3": 2, "G4": 3, "G5": 4, "G6": 5, None: 6}
    d = df.assign(_o=df["first_fail"].map(order)).sort_values(["_o", "a10_med"], ascending=[False, False])
    for r in d.itertuples(index=False):
        L.append(f"| {r.lookback_m} | {r.threshold:+.0%} | {r.base} | {r.soxx_share:.0%} | {r.a10_win:.0%} · {r.a10_med:.2f} · {r.a10_min:.2f} | {r.ainf_win:.0%} · {r.ainf_med:.2f} · {r.ainf_min:.2f} | "
                 f"{r.a10_ret_med:+.0%} · {r.a10_mdd_med:.0%} · {r.trades_a10:.0f} | {r.g2_shallow:.0%}·{r.worst_mdd:.0%} | {r.g3_fail} | {r.b_n}·{r.b_beat_tqqq:.0%}·{r.b_beat_qqq:.0%}·{r.b_worst_cagr:+.1%} | "
                 f"{getattr(r, 'g5_pct', float('nan')):.0f}% | {getattr(r, 'g6', '—')} | {r.first_fail or '통과'} |")
    L.append("")
    best = d.iloc[0]
    L += [f"## 최고 조합 상세 — {best['name']}", "", verdict_markdown(best["name"], tables[best["name"]], verdicts[best["name"]], "dev")]
    return "\n".join(L)


if __name__ == "__main__":
    main(json.loads(sys.argv[1]) if len(sys.argv) > 1 else {})
