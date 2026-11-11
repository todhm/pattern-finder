"""H19 — 차트 패턴 스위치 그리드 → 관문 + §5 슬롯 평가 (설계 tqqq_search_runs/H19_설계.md)."""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from pattern.adapters.base_n_break_downside import BaseNBreakDownsideDetector
from pattern.adapters.ema_crossback_downside import EmaCrossbackDownsideDetector
from pattern.adapters.exhaustion_extension_top import ExhaustionExtensionTopDetector
from pattern.adapters.reversal_extension import ReversalExtensionDetector
from pattern.adapters.wedge_drop import WedgeDropDetector
from pattern.adapters.wedge_pop import WedgePopDetector
from strategy.adapters.pattern_switch import PatternSwitchParams, pattern_switch
from strategy.adapters.regime_slots import CODES, RegimeParams, as_frame, episode_table, episodes, placebo_slot_medians, regime_labels, slot_pass, slot_stats
from strategy.adapters.tqqq_gates import GATE_COSTS, gate_verdicts, load_baseline, placebo_medians, verdict_markdown, window_table
from tqqq_gates_cli import BASELINE, PRICES, load_cash_rate

OHLCV = "/app/sweep_results/qqq_ohlcv_spliced_1985_2026-10-24.parquet"
SELL_SETS = {"all": ["wedge_drop", "ema_crossback", "reversal", "exhaustion", "base_break"], "frequent": ["wedge_drop", "ema_crossback"], "rare": ["reversal", "exhaustion", "base_break"]}
DETECTORS = {"wedge_drop": WedgeDropDetector, "ema_crossback": EmaCrossbackDownsideDetector, "reversal": ReversalExtensionDetector,
             "exhaustion": ExhaustionExtensionTopDetector, "base_break": BaseNBreakDownsideDetector, "wedge_pop": WedgePopDetector}


def signals(df: pd.DataFrame) -> dict[str, list]:
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for k, D in DETECTORS.items():
            out[k] = sorted({s.date for s in D().detect(df)})
    return out


def combos_all():
    out = [PatternSwitchParams("pattern", ss, re) for ss in SELL_SETS for re in ("wedge_pop", "sma20", "either")]
    out += [PatternSwitchParams("switch18_early", ss, "sma20", dd) for ss in ("all", "frequent") for dd in (0.05, 0.10)]
    return out


def neighbors(p: PatternSwitchParams):
    out = []
    for ss in SELL_SETS:
        if ss != p.sell_set:
            out.append(PatternSwitchParams(p.mode, ss, p.reenter, p.dd_alarm))
    if p.mode == "pattern":
        for re in ("wedge_pop", "sma20", "either"):
            if re != p.reenter:
                out.append(PatternSwitchParams(p.mode, p.sell_set, re))
    else:
        out.append(PatternSwitchParams(p.mode, p.sell_set, "sma20", 0.10 if p.dd_alarm == 0.05 else 0.05))
    return out


def main(cfg: dict) -> dict:
    t0 = time.time()
    px = pd.read_parquet(PRICES); base = load_baseline(BASELINE); cash = load_cash_rate()
    df = pd.read_parquet(OHLCV); df.index = pd.to_datetime(df.index).tz_localize(None); df = df[df.index <= "2024-06-30"]
    sig = signals(df)
    print("신호 수: " + ", ".join(f"{k} {len(v)}" for k, v in sig.items()) + f" ({time.time() - t0:.0f}s)", file=sys.stderr)
    labels = regime_labels(px["QQQ"], RegimeParams()); eps = episodes(labels[labels.index <= "2024-06-30"])
    prices = pd.DataFrame({"TQQQ": px["TQQQ"], "QLD": 0.0, "QQQ": px["QQQ"]})
    hold = as_frame(pd.Series(1.0, index=px.index), px.index)
    n_trials = int(cfg.get("n_trials", 1)); seeds = int(cfg.get("placebo_seeds", 30))
    rows, verdicts, tables, targets, slot_rows = [], {}, {}, {}, []
    for i, p in enumerate(combos_all()):
        sells = sorted({d for k in SELL_SETS[p.sell_set] for d in sig[k]})
        t = pattern_switch(px["QQQ"], sells, sig["wedge_pop"], p); targets[p.name] = t
        tbl = window_table(px["TQQQ"], t, base, "dev", costs=GATE_COSTS, cash_rate=cash)
        v = gate_verdicts(tbl, n_trials); tables[p.name], verdicts[p.name] = tbl, v
        a10, ainf = v["G1"]["A-10"], v["G1"]["A-∞"]; a = tbl[tbl.bundle == "A-10"]
        et = episode_table(prices, as_frame(t, px.index), hold, eps, GATE_COSTS, cash)
        slot_txt = []
        for code in CODES:
            st = slot_stats(et, code)
            if st.get("n", 0):
                pct = float(100.0 * (np.asarray(placebo_slot_medians(et, code, seeds)) < st["ratio_median"]).mean())
                slot_rows.append({"component": p.name, "code": code, **st, "placebo_pct": pct, "pass": slot_pass(st, pct)})
                slot_txt.append(f"{code} {st['win_rate']:.0%}/{st['ratio_median']:.2f}{'✅' if slot_pass(st, pct) else ''}")
        rows.append({"name": p.name, **p.__dict__, "tim": float(t[t.index >= "2010-02-11"].mean()),
                     "a10_win": a10["win_rate"], "a10_med": a10["ratio_median"], "a10_min": a10["ratio_min"], "ainf_win": ainf["win_rate"], "ainf_med": ainf["ratio_median"],
                     "g2_shallow": v["G2"]["shallower_rate"], "worst_mdd": v["G2"]["worst_mdd"], "g3_fail": v["G3"]["n_fail"],
                     "b_beat_tqqq": v["G4"]["beat_tqqq_rate"], "b_beat_qqq": v["G4"]["beat_qqq_rate"], "b_worst_cagr": v["G4"]["worst_cagr"],
                     "a10_ret_med": float(a["strat_ret"].median()), "a10_mdd_med": float(a["strat_mdd"].median()), "trades_a10": float(a["trades"].median()),
                     "first_fail": v["first_fail"], "passed_through": v["passed_through"], "slots": " · ".join(slot_txt)})
        print(f"[{i + 1}/13] {p.name}: 체류 {rows[-1]['tim']:.0%} · A-10 {a10['win_rate']:.0%}/{a10['ratio_median']:.2f}/{a10['ratio_min']:.2f} · A-∞ {ainf['ratio_median']:.2f} · MDD {v['G2']['worst_mdd']:.0%} · "
              f"B {v['G4']['beat_tqqq_rate']:.0%}/{v['G4']['worst_cagr']:+.1%} · 탈락 {v['first_fail']} · 슬롯 {rows[-1]['slots']} ({time.time() - t0:.0f}s)", file=sys.stderr)
    for p in combos_all():
        v = verdicts[p.name]
        if v["first_fail"] not in (None, "G5", "G6"):
            continue
        meds = placebo_medians(px["TQQQ"], targets[p.name], base, seeds, 20261110, "runs", "dev", GATE_COSTS, cash)
        nb = [verdicts[q.name] for q in neighbors(p) if q.name in verdicts]
        v2 = gate_verdicts(tables[p.name], n_trials, meds, nb); verdicts[p.name] = v2
        for r in rows:
            if r["name"] == p.name:
                r.update({"first_fail": v2["first_fail"], "passed_through": v2["passed_through"], "g5_pct": v2["G5"]["percentile"], "g6": f"{v2['G6'].get('n_pass')}/{v2['G6'].get('n')}"})
        print(f"  G5/G6 {p.name}: 백분위 {v2['G5']['percentile']:.0f}% · 이웃 {v2['G6']}", file=sys.stderr)
    df_rows = pd.DataFrame(rows)
    out = {"cfg": cfg, "n_trials": n_trials, "signal_counts": {k: len(v) for k, v in sig.items()}, "rows": rows, "slot_rows": slot_rows, "verdicts": verdicts, "elapsed_s": time.time() - t0}
    md = render(df_rows, verdicts, tables, out)
    if cfg.get("out"):
        Path(cfg["out"]).write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
    if cfg.get("md"):
        Path(cfg["md"]).write_text(md)
    print(md[:2000])
    return out


def render(df, verdicts, tables, out) -> str:
    L = ["# H19 — 차트 패턴 스위치 (개발 판, 끝 2024-06-30)", "", "신호 수: " + ", ".join(f"{k} {v}" for k, v in out["signal_counts"].items()), "",
         "| 조합 | 2010~ 체류 | A-10 승률·중앙·최악 | A-∞ 승률·중앙 | A-10 세후 중앙·MDD·매매 | G2 얕음·최악MDD | G3 미달 | B-10 TQQQ·QQQ 이김·최악 CAGR | 슬롯(승률/중앙) | G5 | G6 | 탈락 |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    order = {"G1": 0, "G2": 1, "G3": 2, "G4": 3, "G5": 4, "G6": 5, None: 6}
    d = df.assign(_o=df["first_fail"].map(order)).sort_values(["_o", "a10_med"], ascending=[False, False])
    for r in d.itertuples(index=False):
        L.append(f"| {r.name} | {r.tim:.0%} | {r.a10_win:.0%} · {r.a10_med:.2f} · {r.a10_min:.2f} | {r.ainf_win:.0%} · {r.ainf_med:.2f} | {r.a10_ret_med:+.0%} · {r.a10_mdd_med:.0%} · {r.trades_a10:.0f} | "
                 f"{r.g2_shallow:.0%}·{r.worst_mdd:.0%} | {r.g3_fail} | {r.b_beat_tqqq:.0%}·{r.b_beat_qqq:.0%}·{r.b_worst_cagr:+.1%} | {r.slots} | {getattr(r, 'g5_pct', float('nan')):.0f}% | {getattr(r, 'g6', '—')} | {r.first_fail or '통과'} |")
    L.append(""); best = d.iloc[0]
    L += [f"## 최고 조합 상세 — {best['name']}", "", verdict_markdown(best["name"], tables[best["name"]], verdicts[best["name"]], "dev")]
    return "\n".join(L)


if __name__ == "__main__":
    main(json.loads(sys.argv[1]) if len(sys.argv) > 1 else {})
