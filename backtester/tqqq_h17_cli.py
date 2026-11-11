"""H17 — 신호 지수 교체 그리드 → 관문 (설계 tqqq_search_runs/H17_설계.md)."""

from __future__ import annotations

import itertools
import json
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd

from strategy.adapters.signal_index_switch import SignalIdxParams, signal_index_switch
from strategy.adapters.tqqq_gates import GATE_COSTS, gate_verdicts, load_baseline, placebo_medians, verdict_markdown, window_table
from tqqq_gates_cli import BASELINE, PRICES, load_cash_rate

SPX = "/app/sweep_results/spx_1985_2026-10-24.parquet"
GRID = {"signal": ["spx", "both", "either"], "dd": [0.15, 0.18, 0.21]}


def load_spx(index):
    p = Path(SPX)
    if not p.exists():
        from data.adapters.composed_market_data import build_default_market_data
        from strategy.adapters.leveraged_etf_data import _close
        s = _close(build_default_market_data().fetch_ohlcv("^GSPC", date(1985, 1, 1), date(2026, 10, 24))).dropna().rename("SPX")
        s.index = pd.to_datetime(s.index).tz_localize(None) if getattr(s.index, "tz", None) is not None else s.index
        s.to_frame().to_parquet(p)
    return pd.read_parquet(p).iloc[:, 0].reindex(index).ffill()


def neighbors(p):
    out = []; vals = GRID["dd"]; i = vals.index(p.dd)
    for j in (i - 1, i + 1):
        if 0 <= j < 3:
            out.append(SignalIdxParams(p.signal, vals[j]))
    out += [SignalIdxParams(sg, p.dd) for sg in GRID["signal"] if sg != p.signal]
    return out


def main(cfg):
    t0 = time.time(); px = pd.read_parquet(PRICES); base = load_baseline(BASELINE); cash = load_cash_rate(); spx = load_spx(px.index)
    n_trials = int(cfg.get("n_trials", 1)); seeds = int(cfg.get("placebo_seeds", 30))
    combos = [SignalIdxParams(sg, d) for sg, d in itertools.product(*GRID.values())]
    rows, verdicts, tables, targets = [], {}, {}, {}
    for i, p in enumerate(combos):
        t = signal_index_switch(px["QQQ"], spx, p); targets[p.name] = t
        tbl = window_table(px["TQQQ"], t, base, "dev", costs=GATE_COSTS, cash_rate=cash); v = gate_verdicts(tbl, n_trials)
        tables[p.name], verdicts[p.name] = tbl, v; a10, ainf = v["G1"]["A-10"], v["G1"]["A-∞"]; a = tbl[tbl.bundle == "A-10"]
        rows.append({"name": p.name, "signal": p.signal, "dd": p.dd, "a10_win": a10["win_rate"], "a10_med": a10["ratio_median"], "a10_min": a10["ratio_min"], "ainf_med": ainf["ratio_median"],
                     "worst_mdd": v["G2"]["worst_mdd"], "g3_fail": v["G3"]["n_fail"], "b_beat_tqqq": v["G4"]["beat_tqqq_rate"], "b_worst_cagr": v["G4"]["worst_cagr"], "b_dotcom": v["G4"]["dotcom_strat_median"],
                     "a10_ret_med": float(a["strat_ret"].median()), "a10_mdd_med": float(a["strat_mdd"].median()), "trades_a10": float(a["trades"].median()), "first_fail": v["first_fail"], "passed_through": v["passed_through"]})
        print(f"[{i + 1}/9] {p.name}: A-10 {a10['win_rate']:.0%}/{a10['ratio_median']:.2f}/{a10['ratio_min']:.2f} · A-∞ {ainf['ratio_median']:.2f} · MDD {v['G2']['worst_mdd']:.0%} · G3 미달 {v['G3']['n_fail']} · B {v['G4']['beat_tqqq_rate']:.0%}/{v['G4']['worst_cagr']:+.1%} 닷컴 {v['G4']['dotcom_strat_median']:+.0%} · 매매 {rows[-1]['trades_a10']:.0f} · 탈락 {v['first_fail']} ({time.time() - t0:.0f}s)", file=sys.stderr)
    for p in combos:
        v = verdicts[p.name]
        if v["first_fail"] not in (None, "G5", "G6"):
            continue
        meds = placebo_medians(px["TQQQ"], targets[p.name], base, seeds, 20261110, "runs", "dev", GATE_COSTS, cash)
        nb = [verdicts[q.name] for q in neighbors(p) if q.name in verdicts]; v2 = gate_verdicts(tables[p.name], n_trials, meds, nb); verdicts[p.name] = v2
        for r in rows:
            if r["name"] == p.name:
                r.update({"first_fail": v2["first_fail"], "passed_through": v2["passed_through"], "g5_pct": v2["G5"]["percentile"], "g6": f"{v2['G6'].get('n_pass')}/{v2['G6'].get('n')}"})
        print(f"  G5/G6 {p.name}: {v2['G5']['percentile']:.0f}% · {v2['G6']}", file=sys.stderr)
    df = pd.DataFrame(rows); out = {"cfg": cfg, "rows": rows, "verdicts": verdicts, "elapsed_s": time.time() - t0}
    order = {"G1": 0, "G2": 1, "G3": 2, "G4": 3, "G5": 4, "G6": 5, None: 6}; d = df.assign(_o=df["first_fail"].map(order)).sort_values(["_o", "a10_med"], ascending=[False, False])
    L = ["# H17 — 신호 지수 교체 (개발 판)", "", "| 조합 | A-10 승률·중앙·최악 | A-∞ 중앙 | A-10 세후 중앙·MDD·매매 | G2 최악 MDD | G3 미달 | B TQQQ 이김·최악 CAGR·닷컴 | G5 | G6 | 탈락 |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in d.itertuples(index=False):
        L.append(f"| {r.name} | {r.a10_win:.0%} · {r.a10_med:.2f} · {r.a10_min:.2f} | {r.ainf_med:.2f} | {r.a10_ret_med:+.0%} · {r.a10_mdd_med:.0%} · {r.trades_a10:.0f} | {r.worst_mdd:.0%} | {r.g3_fail} | {r.b_beat_tqqq:.0%}·{r.b_worst_cagr:+.1%}·{r.b_dotcom:+.0%} | {getattr(r, 'g5_pct', float('nan')):.0f}% | {getattr(r, 'g6', '—')} | {r.first_fail or '통과'} |")
    best = d.iloc[0]; L += ["", f"## 최고 — {best['name']}", "", verdict_markdown(best["name"], tables[best["name"]], verdicts[best["name"]], "dev")]
    md = "\n".join(L)
    if cfg.get("out"): Path(cfg["out"]).write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
    if cfg.get("md"): Path(cfg["md"]).write_text(md)
    print(md[:1200]); return out


if __name__ == "__main__":
    main(json.loads(sys.argv[1]) if len(sys.argv) > 1 else {})
