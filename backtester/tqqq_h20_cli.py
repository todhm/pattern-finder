"""H20 — 15분봉 장중 규칙 → 일간 승수 → 관문(A·S 만, B 없음) + §5 슬롯 (설계 tqqq_search_runs/H20_설계.md)."""

from __future__ import annotations

import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from strategy.adapters.intraday_rules import IntradayParams, daily_multiplier, placebo_multiplier, run_rule
from strategy.adapters.regime_slots import CODES, RegimeParams, episodes, placebo_slot_medians, regime_labels, slot_pass, slot_stats
from strategy.adapters.tqqq_gates import gate_verdicts, load_baseline, verdict_markdown, window_table_curve
from tqqq_gates_cli import BASELINE, PRICES

CACHE = "/app/sweep_results/av_intraday_cache"


def load_bars(sym: str) -> pd.DataFrame:
    fs = sorted(glob.glob(f"{CACHE}/{sym}_15m_*.parquet"))
    b = pd.concat([pd.read_parquet(f) for f in fs]).sort_index()
    return b[~b.index.duplicated()]


def combos_all():
    return ([IntradayParams("stop", x) for x in (0.03, 0.05, 0.08)] + [IntradayParams("stop", 0.05, 1), IntradayParams("stop_vwap", 0.05),
            IntradayParams("vwap"), IntradayParams("orb"), IntradayParams("overnight"), IntradayParams("intraday")])


def episode_ratios(mult: pd.Series, hold_mult: pd.Series, eps: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for e in eps[eps["eval"]].itertuples(index=False):
        m = mult[(mult.index > e.start) & (mult.index <= e.end)]; h = hold_mult.reindex(m.index)
        if len(m) < 5:
            continue
        rows.append({"code": e.code, "start": e.start, "end": e.end, "days": len(m), "comp": float(m.prod()), "hold": float(h.prod()), "ratio": float(m.prod() / h.prod())})
    return pd.DataFrame(rows)


def main(cfg: dict) -> dict:
    t0 = time.time()
    px = pd.read_parquet(PRICES); base = load_baseline(BASELINE)
    bars = load_bars("TQQQ")
    adj = px["TQQQ"]
    labels = regime_labels(px["QQQ"], RegimeParams()); eps = episodes(labels[(labels.index >= "2010-02-11") & (labels.index <= "2024-06-30")])
    hold_mult = (adj / adj.shift(1)).fillna(1.0)
    n_trials = int(cfg.get("n_trials", 1)); seeds = int(cfg.get("placebo_seeds", 30))
    rows, verdicts, tables, slot_rows = [], {}, {}, []
    for i, p in enumerate(combos_all()):
        rdf = run_rule(bars, p); mult = daily_multiplier(rdf, adj)
        tbl = window_table_curve(mult, base, "dev"); v = gate_verdicts(tbl, n_trials)
        # G4 자료 없음 → 명시
        v["G4"] = {"passed": False, "error": "B 표본 없음(장중 자료 2010~)"}
        if v["first_fail"] in (None, "G5", "G6"):
            v["first_fail"] = "G4(자료 없음)"
        # G5 (관문 순서와 무관하게 진단으로 계산 — 창 수가 적어 싸다)
        mode = "cash" if p.rule == "vwap" else "exit"
        if p.rule in ("overnight", "intraday"):
            g5 = {"passed": False, "error": "결정적 규칙 — 플라세보 없음"}
        else:
            a = tbl[tbl.bundle.isin(["A-10", "A-∞"])]; strat_med = float(a["ratio"].median()); meds = []
            for s in range(seeds):
                pm = placebo_multiplier(rdf, adj, 20261110 + s, mode)
                meds.append(float(window_table_curve(pm, base, "dev", ("A-10", "A-∞"))["ratio"].median()))
            pct = float(100.0 * (np.asarray(meds) < strat_med).mean())
            g5 = {"passed": pct >= 95.0, "percentile": pct, "strat_median": strat_med, "placebo_median": float(np.median(meds)), "placebo_max": float(np.max(meds))}
        v["G5_diag"] = g5
        tables[p.name], verdicts[p.name] = tbl, v
        a10, ainf = v["G1"]["A-10"], v["G1"]["A-∞"]; a = tbl[tbl.bundle == "A-10"]
        et = episode_ratios(mult, hold_mult, eps); slot_txt = []
        for code in CODES:
            st = slot_stats(et, code)
            if st.get("n", 0):
                pct_s = float(100.0 * (np.asarray(placebo_slot_medians(et, code, seeds)) < st["ratio_median"]).mean())
                ok = slot_pass(st, pct_s); slot_rows.append({"component": p.name, "code": code, **st, "placebo_pct": pct_s, "pass": ok})
                slot_txt.append(f"{code} {st['win_rate']:.0%}/{st['ratio_median']:.2f}{'✅' if ok else ''}")
        rows.append({"name": p.name, "rule": p.rule, "x": p.x, "lag": p.reenter_lag, "exit_days_per_year": float(rdf["exited"].sum() / (len(rdf) / 252)), "exposure": float(rdf["exposure"].mean()),
                     "a10_win": a10["win_rate"], "a10_med": a10["ratio_median"], "a10_min": a10["ratio_min"], "ainf_win": ainf["win_rate"], "ainf_med": ainf["ratio_median"],
                     "g2_shallow": v["G2"]["shallower_rate"], "worst_mdd": v["G2"]["worst_mdd"], "g3_fail": v["G3"]["n_fail"],
                     "a10_ret_med": float(a["strat_ret"].median()), "a10_mdd_med": float(a["strat_mdd"].median()), "g5": g5.get("percentile"),
                     "first_fail": v["first_fail"], "passed_through": v["passed_through"], "slots": " · ".join(slot_txt)})
        print(f"[{i + 1}/{len(combos_all())}] {p.name}: 이탈 {rows[-1]['exit_days_per_year']:.0f}/년 노출 {rows[-1]['exposure']:.2f} · A-10 {a10['win_rate']:.0%}/{a10['ratio_median']:.2f}/{a10['ratio_min']:.2f} · A-∞ {ainf['ratio_median']:.2f} · "
              f"MDD {v['G2']['worst_mdd']:.0%} · G3 미달 {v['G3']['n_fail']} · G5 {g5.get('percentile', float('nan')):.0f}% · 탈락 {v['first_fail']} · 슬롯 {rows[-1]['slots']} ({time.time() - t0:.0f}s)", file=sys.stderr)
    out = {"cfg": cfg, "rows": rows, "slot_rows": slot_rows, "verdicts": verdicts, "elapsed_s": time.time() - t0}
    md = render(pd.DataFrame(rows), verdicts, tables)
    if cfg.get("out"):
        Path(cfg["out"]).write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
    if cfg.get("md"):
        Path(cfg["md"]).write_text(md)
    print(md[:1500])
    return out


def render(df, verdicts, tables) -> str:
    L = ["# H20 — 15분봉 장중 규칙 (개발 판, A·S 만 — B 표본 없음)", "",
         "| 규칙 | 이탈/년 | 평균 노출 | A-10 승률·중앙·최악 | A-∞ 승률·중앙 | A-10 세후 중앙·MDD | G2 얕음·최악MDD | G3 미달 | G5 진단 | 슬롯 | 탈락 |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    order = {"G1": 0, "G2": 1, "G3": 2, "G4(자료 없음)": 3, "G4": 3, None: 6}
    d = df.assign(_o=df["first_fail"].map(order)).sort_values(["_o", "a10_med"], ascending=[False, False])
    for r in d.itertuples(index=False):
        g5 = f"{r.g5:.0f}%" if r.g5 == r.g5 and r.g5 is not None else "—"
        L.append(f"| {r.name} | {r.exit_days_per_year:.0f} | {r.exposure:.2f} | {r.a10_win:.0%} · {r.a10_med:.2f} · {r.a10_min:.2f} | {r.ainf_win:.0%} · {r.ainf_med:.2f} | {r.a10_ret_med:+.0%} · {r.a10_mdd_med:.0%} | {r.g2_shallow:.0%}·{r.worst_mdd:.0%} | {r.g3_fail} | {g5} | {r.slots} | {r.first_fail or '통과'} |")
    L.append(""); best = d.iloc[0]
    L += [f"## 최고 조합 상세 — {best['name']}", "", verdict_markdown(best["name"], tables[best["name"]], verdicts[best["name"]], "dev")]
    return "\n".join(L)


if __name__ == "__main__":
    main(json.loads(sys.argv[1]) if len(sys.argv) > 1 else {})
