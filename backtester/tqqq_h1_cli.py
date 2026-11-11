"""H1 — 거시 레짐 × 스위치 18 이탈 문턱 그리드 → 관문 G1~G6 (프로토콜 §2, 설계 tqqq_search_runs/H1_설계.md).

    docker compose exec backtester python tqqq_h1_cli.py '{"out": "/app/sweep_results/tqqq_h1.json", "md": "/app/sweep_results/tqqq_h1.md", "n_trials": 1}'

거시 점수는 FRED(T10Y3M·BAA10Y·UNRATE·VIXCLS) → `macro_regime` 스코어 → `composite_score`, 한 번 만들어 parquet 로 고정.
플라세보는 G1~G4 를 통과한 조합에만(관문 순서). G6 이웃 = dd_risk·dd_healthy 인접 그리드 값.
"""

from __future__ import annotations

import itertools
import json
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd

from strategy.adapters.macro_regime import composite_score, score_credit_spread, score_sahm_rule, score_vix, score_yield_curve
from strategy.adapters.macro_switch import MacroSwitchParams, macro_switch, regime_share
from strategy.adapters.tqqq_gates import GATE_COSTS, gate_verdicts, load_baseline, placebo_medians, verdict_markdown, window_table
from tqqq_gates_cli import BASELINE, PRICES, load_cash_rate

MACRO = "/app/sweep_results/macro_score_1985_2026-10-24.parquet"
GRID = {"dd_risk": [0.10, 0.12, 0.15], "dd_healthy": [0.18, 0.21, 0.24], "risk_thr": [0.5, 0.75], "healthy_thr": [0.75, 1.0]}
NEIGHBOR_STEP = {"dd_risk": [0.10, 0.12, 0.15], "dd_healthy": [0.18, 0.21, 0.24]}


def build_macro_score(index: pd.DatetimeIndex) -> pd.Series:
    p = Path(MACRO)
    if p.exists():
        return pd.read_parquet(p).iloc[:, 0]
    from data.adapters.fred_adapter import FredCsvAdapter
    f = FredCsvAdapter()
    s0, e = date(1984, 1, 1), date(2026, 10, 24)
    comps = {
        "curve": score_yield_curve(f.fetch_series("T10Y3M", s0, e)),
        "credit": score_credit_spread(f.fetch_series("BAA10Y", s0, e)),
        "vix": score_vix(f.fetch_series("VIXCLS", s0, e)),
        "sahm": score_sahm_rule(f.fetch_series("UNRATE", s0, e)),
    }
    for k, v in comps.items():
        print(f"  {k}: {v.index[0].date()}~{v.index[-1].date()} 건강 비율 {v.mean():.2f}", file=sys.stderr)
    m = composite_score(comps, {k: 1.0 for k in comps}, index).rename("macro")
    m.to_frame().to_parquet(p)
    return m


def neighbors(p: MacroSwitchParams) -> list[MacroSwitchParams]:
    out = []
    for key, vals in NEIGHBOR_STEP.items():
        i = vals.index(getattr(p, key))
        for j in (i - 1, i + 1):
            if 0 <= j < len(vals):
                out.append(MacroSwitchParams(**{**p.__dict__, key: vals[j]}))
    return out


def main(cfg: dict) -> dict:
    t0 = time.time()
    px = pd.read_parquet(PRICES)
    base = load_baseline(BASELINE)
    cash = load_cash_rate()
    macro = build_macro_score(px.index)
    n_trials = int(cfg.get("n_trials", 1))
    seeds = int(cfg.get("placebo_seeds", 30))
    combos = [MacroSwitchParams(dd_risk=a, dd_healthy=b, risk_thr=c, healthy_thr=d)
              for a, b, c, d in itertools.product(*GRID.values()) if c <= d]
    if cfg.get("max_combos"):
        combos = combos[: int(cfg["max_combos"])]
    print(f"조합 {len(combos)}, 거시 점수 {macro.dropna().index[0].date()}~ ({time.time() - t0:.0f}s)", file=sys.stderr)
    rows, verdicts, tables = [], {}, {}
    for i, p in enumerate(combos):
        tgt = macro_switch(px["QQQ"], macro, p)
        tbl = window_table(px["TQQQ"], tgt, base, "dev", costs=GATE_COSTS, cash_rate=cash)
        v = gate_verdicts(tbl, n_trials)
        tables[p.name], verdicts[p.name] = tbl, v
        a10, ainf = v["G1"]["A-10"], v["G1"]["A-∞"]
        rows.append({"name": p.name, **p.__dict__, "regime": regime_share(macro, px.index[px.index >= "2010-02-11"], p),
                     "a10_win": a10["win_rate"], "a10_med": a10["ratio_median"], "a10_min": a10["ratio_min"],
                     "ainf_win": ainf["win_rate"], "ainf_med": ainf["ratio_median"], "ainf_min": ainf["ratio_min"],
                     "g2_shallow": v["G2"]["shallower_rate"], "worst_mdd": v["G2"]["worst_mdd"], "g3_fail": v["G3"]["n_fail"],
                     "b_beat_tqqq": v["G4"]["beat_tqqq_rate"], "b_beat_qqq": v["G4"]["beat_qqq_rate"], "b_worst_cagr": v["G4"]["worst_cagr"],
                     "a10_ret_med": float(tbl[tbl.bundle == "A-10"]["strat_ret"].median()), "a10_mdd_med": float(tbl[tbl.bundle == "A-10"]["strat_mdd"].median()),
                     "trades_a10": float(tbl[tbl.bundle == "A-10"]["trades"].median()),
                     "first_fail": v["first_fail"], "passed_through": v["passed_through"]})
        print(f"[{i + 1}/{len(combos)}] {p.name}: A-10 승률 {a10['win_rate']:.0%} 중앙 {a10['ratio_median']:.2f} 최악 {a10['ratio_min']:.2f} · "
              f"A-∞ {ainf['win_rate']:.0%}/{ainf['ratio_median']:.2f} · 탈락 {v['first_fail']} ({time.time() - t0:.0f}s)", file=sys.stderr)
    # G5·G6: G1~G4 통과 조합만
    for p in combos:
        v = verdicts[p.name]
        if v["first_fail"] not in (None, "G5", "G6"):
            continue
        tgt = macro_switch(px["QQQ"], macro, p)
        meds = placebo_medians(px["TQQQ"], tgt, base, seeds, 20261110, "runs", "dev", GATE_COSTS, cash)
        nb = [verdicts[q.name] for q in neighbors(p) if q.name in verdicts]
        v2 = gate_verdicts(tables[p.name], n_trials, meds, nb)
        verdicts[p.name] = v2
        for r in rows:
            if r["name"] == p.name:
                r.update({"first_fail": v2["first_fail"], "passed_through": v2["passed_through"], "g5_pct": v2["G5"]["percentile"],
                          "g6": f"{v2['G6'].get('n_pass')}/{v2['G6'].get('n')}"})
        print(f"  G5/G6 {p.name}: 백분위 {v2['G5']['percentile']:.0f}% · 이웃 {v2['G6']}", file=sys.stderr)
    df = pd.DataFrame(rows)
    out = {"cfg": cfg, "n_trials": n_trials, "grid": GRID, "rows": rows, "verdicts": verdicts, "elapsed_s": time.time() - t0}
    md = render(df, verdicts, tables, macro, px)
    if cfg.get("out"):
        Path(cfg["out"]).write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
    if cfg.get("md"):
        Path(cfg["md"]).write_text(md)
    print(md[:3000])
    return out


def render(df: pd.DataFrame, verdicts: dict, tables: dict, macro: pd.Series, px: pd.DataFrame) -> str:
    L = ["# H1 — 거시 레짐 × 스위치 18 이탈 문턱 (개발 판, 끝 2024-06-30)", "",
         f"조합 {len(df)} · 거시 점수 {macro.dropna().index[0].date()}~ (커브·신용 BAA10Y·VIX·Sahm 동일 가중, 있는 지표만) · 2010-02~ 레짐 비율(명명 조합) "
         + ", ".join(f"{k} {v:.0%}" for k, v in df.iloc[0]["regime"].items()), ""]
    order = {"G1": 0, "G2": 1, "G3": 2, "G4": 3, "G5": 4, "G6": 5, None: 6}
    d = df.assign(_o=df["first_fail"].map(order)).sort_values(["_o", "a10_med"], ascending=[False, False])
    L += ["| dd 위험/중립/건강 | 문턱 | 레짐 위험·건강 | A-10 승률·중앙·최악 | A-∞ 승률·중앙·최악 | A-10 세후 중앙 · MDD · 매매 | G2 얕음·최악MDD | G3 미달 | B-10 TQQQ·QQQ 이김·최악 CAGR | G5 | G6 | 탈락 |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in d.itertuples(index=False):
        L.append(f"| {r.dd_risk:.2f}/{r.dd_mid:.2f}/{r.dd_healthy:.2f} | {r.risk_thr:g}–{r.healthy_thr:g} | {r.regime['risk']:.0%}·{r.regime['healthy']:.0%} | "
                 f"{r.a10_win:.0%} · {r.a10_med:.2f} · {r.a10_min:.2f} | {r.ainf_win:.0%} · {r.ainf_med:.2f} · {r.ainf_min:.2f} | "
                 f"{r.a10_ret_med:+.0%} · {r.a10_mdd_med:.0%} · {r.trades_a10:.0f} | {r.g2_shallow:.0%}·{r.worst_mdd:.0%} | {r.g3_fail} | "
                 f"{r.b_beat_tqqq:.0%}·{r.b_beat_qqq:.0%}·{r.b_worst_cagr:+.1%} | {getattr(r, 'g5_pct', float('nan')):.0f}% | {getattr(r, 'g6', '—')} | {r.first_fail or '통과'} |")
    L.append("")
    best = d.iloc[0]
    L.append(f"## 최고 조합 상세 — {best['name']}")
    L.append("")
    L.append(verdict_markdown(best["name"], tables[best["name"]], verdicts[best["name"]], "dev"))
    named = [n for n in tables if n.startswith("macro_sw r0.12/m0.18/h0.24 thr0.5-0.75")]
    if named and named[0] != best["name"]:
        L.append(f"## 대기열 명명 조합 — {named[0]}")
        L.append("")
        L.append(verdict_markdown(named[0], tables[named[0]], verdicts[named[0]], "dev"))
    return "\n".join(L)


if __name__ == "__main__":
    main(json.loads(sys.argv[1]) if len(sys.argv) > 1 else {})
