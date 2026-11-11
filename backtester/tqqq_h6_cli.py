"""H6 — 계절성·실적 시즌 노출 그리드 → 관문 (설계 tqqq_search_runs/H6_설계.md)."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

from strategy.adapters.seasonal_switch import SeasonParams, seasonal_weights
from strategy.adapters.tqqq_gates import GATE_COSTS, gate_verdicts, load_baseline, placebo_medians, verdict_markdown, window_table
from tqqq_gates_cli import BASELINE, PRICES, load_cash_rate

MONTH_SETS = [(10, 11, 12, 1, 2, 3), (11, 12, 1, 2, 3, 4), (12, 1, 2, 3, 4, 5)]
WEEK_SETS = [(2, 3, 4), (2, 3, 4, 5), (1, 2, 3, 4, 5)]


def combos_all() -> list[SeasonParams]:
    out = []
    for off in ("qqq", "cash"):
        out += [SeasonParams("halloween", m, (2, 3, 4), off) for m in MONTH_SETS]
        out += [SeasonParams("earnings", (11, 12, 1, 2, 3, 4), w, off) for w in WEEK_SETS]
    return out


def neighbors(p: SeasonParams) -> list[SeasonParams]:
    sets, key = (MONTH_SETS, "months") if p.kind == "halloween" else (WEEK_SETS, "weeks")
    i = sets.index(getattr(p, key)); out = []
    for j in (i - 1, i + 1):
        if 0 <= j < len(sets):
            out.append(SeasonParams(p.kind, sets[j] if key == "months" else p.months, sets[j] if key == "weeks" else p.weeks, p.off_asset))
    return out


def main(cfg: dict) -> dict:
    t0 = time.time()
    px = pd.read_parquet(PRICES); base = load_baseline(BASELINE); cash = load_cash_rate()
    prices = px[["TQQQ", "QQQ"]]
    n_trials = int(cfg.get("n_trials", 1)); seeds = int(cfg.get("placebo_seeds", 30))
    combos = combos_all()
    if cfg.get("max_combos"):
        combos = combos[: int(cfg["max_combos"])]
    rows, verdicts, tables, weights = [], {}, {}, {}
    for i, p in enumerate(combos):
        w = seasonal_weights(px.index, p); weights[p.name] = w
        tbl = window_table(px["TQQQ"], w, base, "dev", costs=GATE_COSTS, cash_rate=cash, prices=prices)
        v = gate_verdicts(tbl, n_trials); tables[p.name], verdicts[p.name] = tbl, v
        a10, ainf = v["G1"]["A-10"], v["G1"]["A-∞"]; a = tbl[tbl.bundle == "A-10"]
        rows.append({"name": p.name, "kind": p.kind, "months": p.months, "weeks": p.weeks, "off_asset": p.off_asset, "tqqq_share": float(w["TQQQ"].mean()),
                     "a10_win": a10["win_rate"], "a10_med": a10["ratio_median"], "a10_min": a10["ratio_min"],
                     "ainf_win": ainf["win_rate"], "ainf_med": ainf["ratio_median"], "ainf_min": ainf["ratio_min"],
                     "g2_shallow": v["G2"]["shallower_rate"], "worst_mdd": v["G2"]["worst_mdd"], "g3_fail": v["G3"]["n_fail"],
                     "b_beat_tqqq": v["G4"]["beat_tqqq_rate"], "b_beat_qqq": v["G4"]["beat_qqq_rate"], "b_worst_cagr": v["G4"]["worst_cagr"], "b_ret_med": v["G4"]["ret_median"],
                     "a10_ret_med": float(a["strat_ret"].median()), "a10_mdd_med": float(a["strat_mdd"].median()), "trades_a10": float(a["trades"].median()),
                     "first_fail": v["first_fail"], "passed_through": v["passed_through"]})
        print(f"[{i + 1}/{len(combos)}] {p.name}: TQQQ 체류 {w['TQQQ'].mean():.0%} · A-10 {a10['win_rate']:.0%}/{a10['ratio_median']:.2f}/{a10['ratio_min']:.2f} · A-∞ {ainf['median'] if 'median' in ainf else ainf['ratio_median']:.2f} · "
              f"MDD {v['G2']['worst_mdd']:.0%} · B TQQQ 이김 {v['G4']['beat_tqqq_rate']:.0%} · 탈락 {v['first_fail']} ({time.time() - t0:.0f}s)", file=sys.stderr)
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
    df = pd.DataFrame(rows)
    out = {"cfg": cfg, "n_trials": n_trials, "rows": rows, "verdicts": verdicts, "elapsed_s": time.time() - t0}
    md = render(df, verdicts, tables)
    if cfg.get("out"):
        Path(cfg["out"]).write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
    if cfg.get("md"):
        Path(cfg["md"]).write_text(md)
    print(md[:2500])
    return out


def render(df, verdicts, tables) -> str:
    L = ["# H6 — 계절성·실적 시즌 노출 (개발 판, 끝 2024-06-30)", "", f"조합 {len(df)}", "",
         "| 규칙 | 시즌 | 비시즌 | TQQQ 체류 | A-10 승률·중앙·최악 | A-∞ 승률·중앙·최악 | A-10 세후 중앙·MDD·매매 | G2 얕음·최악MDD | G3 미달 | B-10 TQQQ·QQQ 이김·최악 CAGR·세후 중앙 | G5 | G6 | 탈락 |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    order = {"G1": 0, "G2": 1, "G3": 2, "G4": 3, "G5": 4, "G6": 5, None: 6}
    d = df.assign(_o=df["first_fail"].map(order)).sort_values(["_o", "a10_med"], ascending=[False, False])
    for r in d.itertuples(index=False):
        season = f"{r.months[0]}~{r.months[-1]}월" if r.kind == "halloween" else f"{r.weeks[0]}~{r.weeks[-1]}주"
        L.append(f"| {r.kind} | {season} | {r.off_asset} | {r.tqqq_share:.0%} | {r.a10_win:.0%} · {r.a10_med:.2f} · {r.a10_min:.2f} | {r.ainf_win:.0%} · {r.ainf_med:.2f} · {r.ainf_min:.2f} | "
                 f"{r.a10_ret_med:+.0%} · {r.a10_mdd_med:.0%} · {r.trades_a10:.0f} | {r.g2_shallow:.0%}·{r.worst_mdd:.0%} | {r.g3_fail} | {r.b_beat_tqqq:.0%}·{r.b_beat_qqq:.0%}·{r.b_worst_cagr:+.1%}·{r.b_ret_med:+.0%} | "
                 f"{getattr(r, 'g5_pct', float('nan')):.0f}% | {getattr(r, 'g6', '—')} | {r.first_fail or '통과'} |")
    L.append("")
    best = d.iloc[0]
    L += [f"## 최고 조합 상세 — {best['name']}", "", verdict_markdown(best["name"], tables[best["name"]], verdicts[best["name"]], "dev")]
    return "\n".join(L)


if __name__ == "__main__":
    main(json.loads(sys.argv[1]) if len(sys.argv) > 1 else {})
