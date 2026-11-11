"""H25 — 워크포워드 선택기 v2 (라이브러리 10·룩백 평균·top-2 혼합) → 관문 (설계 tqqq_search_runs/H25_설계.md)."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from strategy.adapters.bear_len_reentry import BearLenParams, bear_len_switch
from strategy.adapters.regime_slots import as_frame
from strategy.adapters.tqqq_gates import GATE_COSTS, gate_verdicts, load_baseline, verdict_markdown, window_table
from strategy.adapters.walk_forward_select import WalkForwardParams, walk_forward
from strategy.adapters.whipsaw_switch import WhipsawParams, whipsaw_switch
from tqqq_gates_cli import BASELINE, PRICES, load_cash_rate
from tqqq_h11_cli import load_qld
from tqqq_h22_cli import library
from tqqq_h23_cli import FIRST


def library_v2(px, prices):
    lib = library(px, prices); lib.pop("현금", None)
    idx = px.index; q = px["QQQ"]
    lib["L60 월말 재진입"] = as_frame(bear_len_switch(q, BearLenParams(60, "monthly20")), idx)
    msw = whipsaw_switch(q, WhipsawParams(0.0, True, False)).reindex(idx).ffill().fillna(1.0)
    qldm = pd.DataFrame({"TQQQ": 0.0, "QLD": msw.values, "QQQ": 0.0}, index=idx); lib["QLD+월말 스위치"] = qldm
    try:
        from tqqq_h19_cli import OHLCV, SELL_SETS, signals
        from strategy.adapters.pattern_switch import PatternSwitchParams, pattern_switch
        df = pd.read_parquet(OHLCV); df.index = pd.to_datetime(df.index).tz_localize(None); df = df[df.index <= "2024-06-30"]
        sig = signals(df); sells = sorted({d for k in SELL_SETS["frequent"] for d in sig[k]})
        lib["스위치 18+패턴 10%"] = as_frame(pattern_switch(q, sells, sig["wedge_pop"], PatternSwitchParams("switch18_early", "frequent", "sma20", 0.10)), idx)
    except Exception as e:  # noqa: BLE001
        print(f"패턴 부품 생략: {e}", file=sys.stderr)
    return lib


def combos():
    out = []
    for yrs in ((3, None), (5, None), (3, 5)):
        for sticky in (False, True):
            for k in (1, 2):
                out.append(WalkForwardParams(yrs[0], "mult", sticky, yrs[1], k))
    return out


def neighbors(p, all_):
    return [q for q in all_ if (q.sticky == p.sticky and q.top_k == p.top_k and (q.years, q.years2) != (p.years, p.years2)) or (q.sticky == p.sticky and (q.years, q.years2) == (p.years, p.years2) and q.top_k != p.top_k)]


def main(cfg):
    t0 = time.time(); px = pd.read_parquet(PRICES); base = load_baseline(BASELINE); cash = load_cash_rate(); qld = load_qld(px.index, cash)
    prices = pd.DataFrame({"TQQQ": px["TQQQ"], "QLD": qld, "QQQ": px["QQQ"]}); lib = library_v2(px, prices)
    n_trials = int(cfg.get("n_trials", 1)); seeds = int(cfg.get("placebo_seeds", 30)); all_ = combos()
    print(f"라이브러리 {len(lib)}: {list(lib)} · 조합 {len(all_)}", file=sys.stderr)
    rows, verdicts, tables = [], {}, {}
    for i, p in enumerate(all_):
        w, picks = walk_forward(prices, lib, p, GATE_COSTS, cash, FIRST)
        tbl = window_table(px["TQQQ"], w, base, "dev", costs=GATE_COSTS, cash_rate=cash, prices=prices); v = gate_verdicts(tbl, n_trials)
        tables[p.name], verdicts[p.name] = tbl, v; a10, ainf = v["G1"]["A-10"], v["G1"]["A-∞"]; a = tbl[tbl.bundle == "A-10"]
        pk = picks[(picks.index >= "2010-01-01") & (picks.index <= "2024-06-30")]
        rows.append({"name": p.name, "picks_2010": pk.value_counts(normalize=True).round(2).head(4).to_dict(), "switches": int((pk != pk.shift()).sum() - 1),
                     "a10_win": a10["win_rate"], "a10_med": a10["ratio_median"], "a10_min": a10["ratio_min"], "ainf_win": ainf["win_rate"], "ainf_med": ainf["ratio_median"],
                     "worst_mdd": v["G2"]["worst_mdd"], "g3_fail": v["G3"]["n_fail"], "b_beat_tqqq": v["G4"]["beat_tqqq_rate"], "b_beat_qqq": v["G4"]["beat_qqq_rate"], "b_worst_cagr": v["G4"]["worst_cagr"], "b_loss": v["G4"]["loss_windows"], "b_dotcom": v["G4"]["dotcom_strat_median"],
                     "a10_ret_med": float(a["strat_ret"].median()), "a10_mdd_med": float(a["strat_mdd"].median()), "first_fail": v["first_fail"], "passed_through": v["passed_through"]})
        print(f"[{i + 1}/{len(all_)}] {p.name}: A-10 {a10['win_rate']:.0%}/{a10['ratio_median']:.2f}/{a10['ratio_min']:.2f} · A-∞ {ainf['win_rate']:.0%}/{ainf['ratio_median']:.2f} · MDD {v['G2']['worst_mdd']:.0%} · G3 미달 {v['G3']['n_fail']} · B {v['G4']['beat_tqqq_rate']:.0%}/{v['G4']['worst_cagr']:+.1%} 손실 {v['G4']['loss_windows']} · 탈락 {v['first_fail']} ({time.time() - t0:.0f}s)", file=sys.stderr)
    # G5 진단은 3 관문 이상 통과 조합 전부에(관문 순서와 별개로 기록 가치)
    for p in all_:
        v = verdicts[p.name]
        if v["passed_through"] < 3:
            continue
        meds = []
        for s in range(seeds):
            w2, _ = walk_forward(prices, lib, p, GATE_COSTS, cash, FIRST, rng=np.random.default_rng(20261110 + s))
            meds.append(float(window_table(px["TQQQ"], w2, base, "dev", ("A-10", "A-∞"), GATE_COSTS, cash, prices)["ratio"].median()))
        nb = [verdicts[q.name] for q in neighbors(p, all_) if q.name in verdicts]
        v2 = gate_verdicts(tables[p.name], n_trials, meds, nb); v2["G5_diag"] = v2["G5"]; v2["G6_diag"] = v2["G6"]; verdicts[p.name] = v2
        for r in rows:
            if r["name"] == p.name:
                r.update({"first_fail": v2["first_fail"], "passed_through": v2["passed_through"], "g5_pct": v2["G5"]["percentile"], "g6": f"{v2['G6'].get('n_pass')}/{v2['G6'].get('n')}"})
        print(f"  G5/G6 진단 {p.name}: {v2['G5']['percentile']:.0f}% · 이웃 {v2['G6']}", file=sys.stderr)
    df = pd.DataFrame(rows); out = {"cfg": cfg, "library": list(lib), "rows": rows, "verdicts": verdicts, "elapsed_s": time.time() - t0}
    order = {"G1": 0, "G2": 1, "G3": 2, "G4": 3, "G5": 4, "G6": 5, None: 6}; d = df.assign(_o=df["first_fail"].map(order)).sort_values(["_o", "a10_med"], ascending=[False, False])
    L = ["# H25 — 워크포워드 선택기 v2 (개발 판)", "", f"라이브러리 {len(lib)}: {', '.join(lib)}", "", "| 조합 | 2010~ 선택(상위) | 교체 | A-10 승률·중앙·최악 | A-∞ 승률·중앙 | A-10 세후 중앙·MDD | G2 최악 | G3 미달 | B TQQQ·QQQ 이김·최악 CAGR·손실·닷컴 | G5 | G6 | 탈락 |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in d.itertuples(index=False):
        L.append(f"| {r.name} | {r.picks_2010} | {r.switches} | {r.a10_win:.0%} · {r.a10_med:.2f} · {r.a10_min:.2f} | {r.ainf_win:.0%} · {r.ainf_med:.2f} | {r.a10_ret_med:+.0%} · {r.a10_mdd_med:.0%} | {r.worst_mdd:.0%} | {r.g3_fail} | {r.b_beat_tqqq:.0%}·{r.b_beat_qqq:.0%}·{r.b_worst_cagr:+.1%}·{r.b_loss}·{r.b_dotcom:+.0%} | {getattr(r, 'g5_pct', float('nan')):.0f}% | {getattr(r, 'g6', '—')} | {r.first_fail or '통과'} |")
    best = d.iloc[0]; L += ["", f"## 최고 — {best['name']}", "", verdict_markdown(best["name"], tables[best["name"]], verdicts[best["name"]], "dev")]
    md = "\n".join(L)
    if cfg.get("out"): Path(cfg["out"]).write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
    if cfg.get("md"): Path(cfg["md"]).write_text(md)
    print(md[:1500]); return out


if __name__ == "__main__":
    main(json.loads(sys.argv[1]) if len(sys.argv) > 1 else {})
