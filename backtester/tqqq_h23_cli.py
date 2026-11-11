"""H23 — 워크포워드 규칙 로테이션 그리드 → 관문 (설계 tqqq_search_runs/H23_설계.md)."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from strategy.adapters.regime_slots import as_frame
from strategy.adapters.tqqq_gates import GATE_COSTS, gate_verdicts, load_baseline, verdict_markdown, window_table
from strategy.adapters.walk_forward_select import WalkForwardParams, walk_forward
from tqqq_gates_cli import BASELINE, PRICES, load_cash_rate
from tqqq_h11_cli import load_qld
from tqqq_h22_cli import library

FIRST = pd.Timestamp("1991-01-01")


def main(cfg: dict) -> dict:
    t0 = time.time()
    px = pd.read_parquet(PRICES); base = load_baseline(BASELINE); cash = load_cash_rate()
    qld = load_qld(px.index, cash)
    prices = pd.DataFrame({"TQQQ": px["TQQQ"], "QLD": qld, "QQQ": px["QQQ"]})
    lib = library(px, prices); lib.pop("현금", None)
    try:
        from tqqq_h19_cli import OHLCV, SELL_SETS, signals
        from strategy.adapters.pattern_switch import PatternSwitchParams, pattern_switch
        df = pd.read_parquet(OHLCV); df.index = pd.to_datetime(df.index).tz_localize(None); df = df[df.index <= "2024-06-30"]
        sig = signals(df); sells = sorted({d for k in SELL_SETS["frequent"] for d in sig[k]})
        lib["스위치 18+패턴 10%"] = as_frame(pattern_switch(px["QQQ"], sells, sig["wedge_pop"], PatternSwitchParams("switch18_early", "frequent", "sma20", 0.10)), px.index)
    except Exception as e:  # noqa: BLE001
        print(f"패턴 부품 생략: {e}", file=sys.stderr)
    n_trials = int(cfg.get("n_trials", 1)); seeds = int(cfg.get("placebo_seeds", 30))
    combos = [WalkForwardParams(y, m, s) for y in (3, 5) for m in ("mult", "calmar") for s in (False, True)]
    print(f"라이브러리 {len(lib)}: {list(lib)} · 조합 {len(combos)}", file=sys.stderr)
    rows, verdicts, tables = [], {}, {}
    for i, p in enumerate(combos):
        w, picks = walk_forward(prices, lib, p, GATE_COSTS, cash, FIRST)
        tbl = window_table(px["TQQQ"], w, base, "dev", costs=GATE_COSTS, cash_rate=cash, prices=prices)
        v = gate_verdicts(tbl, n_trials); tables[p.name], verdicts[p.name] = tbl, v
        a10, ainf = v["G1"]["A-10"], v["G1"]["A-∞"]; a = tbl[tbl.bundle == "A-10"]
        share = picks[picks.index >= "2010-01-01"].value_counts(normalize=True).round(2).to_dict()
        rows.append({"name": p.name, **p.__dict__, "picks_2010": share, "switches": int((picks != picks.shift()).sum() - 1),
                     "a10_win": a10["win_rate"], "a10_med": a10["ratio_median"], "a10_min": a10["ratio_min"], "ainf_win": ainf["win_rate"], "ainf_med": ainf["ratio_median"],
                     "g2_shallow": v["G2"]["shallower_rate"], "worst_mdd": v["G2"]["worst_mdd"], "g3_fail": v["G3"]["n_fail"],
                     "b_beat_tqqq": v["G4"]["beat_tqqq_rate"], "b_beat_qqq": v["G4"]["beat_qqq_rate"], "b_worst_cagr": v["G4"]["worst_cagr"],
                     "a10_ret_med": float(a["strat_ret"].median()), "a10_mdd_med": float(a["strat_mdd"].median()), "first_fail": v["first_fail"], "passed_through": v["passed_through"]})
        print(f"[{i + 1}/{len(combos)}] {p.name}: 2010~ 선택 {share} · 교체 {rows[-1]['switches']} · A-10 {a10['win_rate']:.0%}/{a10['ratio_median']:.2f}/{a10['ratio_min']:.2f} · A-∞ {ainf['ratio_median']:.2f} · "
              f"MDD {v['G2']['worst_mdd']:.0%} · B {v['G4']['beat_tqqq_rate']:.0%}/{v['G4']['worst_cagr']:+.1%} · 탈락 {v['first_fail']} ({time.time() - t0:.0f}s)", file=sys.stderr)
    for p in combos:
        v = verdicts[p.name]
        if v["first_fail"] not in (None, "G5", "G6"):
            continue
        meds = []
        for s in range(seeds):
            w2, _ = walk_forward(prices, lib, p, GATE_COSTS, cash, FIRST, rng=np.random.default_rng(20261110 + s))
            meds.append(float(window_table(px["TQQQ"], w2, base, "dev", ("A-10", "A-∞"), GATE_COSTS, cash, prices)["ratio"].median()))
        nb = [verdicts[q.name] for q in (WalkForwardParams(5 if p.years == 3 else 3, p.metric, p.sticky), WalkForwardParams(p.years, "calmar" if p.metric == "mult" else "mult", p.sticky)) if q.name in verdicts]
        v2 = gate_verdicts(tables[p.name], n_trials, meds, nb); verdicts[p.name] = v2
        for r in rows:
            if r["name"] == p.name:
                r.update({"first_fail": v2["first_fail"], "passed_through": v2["passed_through"], "g5_pct": v2["G5"]["percentile"], "g6": f"{v2['G6'].get('n_pass')}/{v2['G6'].get('n')}"})
    df_rows = pd.DataFrame(rows)
    out = {"cfg": cfg, "library": list(lib), "rows": rows, "verdicts": verdicts, "elapsed_s": time.time() - t0}
    md = render(df_rows, verdicts, tables, out)
    if cfg.get("out"):
        Path(cfg["out"]).write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
    if cfg.get("md"):
        Path(cfg["md"]).write_text(md)
    print(md[:1500])
    return out


def render(df, verdicts, tables, out) -> str:
    L = ["# H23 — 워크포워드 규칙 로테이션 (개발 판, 끝 2024-06-30)", "", f"라이브러리 {len(out['library'])}: {', '.join(out['library'])}", "",
         "| 조합 | 2010~ 선택 비율 | 교체 수 | A-10 승률·중앙·최악 | A-∞ 승률·중앙 | A-10 세후 중앙·MDD | G2 얕음·최악MDD | G3 미달 | B-10 TQQQ·QQQ 이김·최악 CAGR | G5 | G6 | 탈락 |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    order = {"G1": 0, "G2": 1, "G3": 2, "G4": 3, "G5": 4, "G6": 5, None: 6}
    d = df.assign(_o=df["first_fail"].map(order)).sort_values(["_o", "a10_med"], ascending=[False, False])
    for r in d.itertuples(index=False):
        L.append(f"| {r.name} | {r.picks_2010} | {r.switches} | {r.a10_win:.0%} · {r.a10_med:.2f} · {r.a10_min:.2f} | {r.ainf_win:.0%} · {r.ainf_med:.2f} | {r.a10_ret_med:+.0%} · {r.a10_mdd_med:.0%} | "
                 f"{r.g2_shallow:.0%}·{r.worst_mdd:.0%} | {r.g3_fail} | {r.b_beat_tqqq:.0%}·{r.b_beat_qqq:.0%}·{r.b_worst_cagr:+.1%} | {getattr(r, 'g5_pct', float('nan')):.0f}% | {getattr(r, 'g6', '—')} | {r.first_fail or '통과'} |")
    L.append(""); best = d.iloc[0]
    L += [f"## 최고 조합 상세 — {best['name']}", "", verdict_markdown(best["name"], tables[best["name"]], verdicts[best["name"]], "dev")]
    return "\n".join(L)


if __name__ == "__main__":
    main(json.loads(sys.argv[1]) if len(sys.argv) > 1 else {})
