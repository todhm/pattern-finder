"""H11 — QLD(2×) ± 스위치, TQQQ·QQQ 혼합 ± 스위치 → 관문 (설계 tqqq_search_runs/H11_설계.md)."""

from __future__ import annotations

import json
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd

from strategy.adapters.lev_mix import LevMixParams, lev_mix_weights
from strategy.adapters.tqqq_gates import GATE_COSTS, gate_verdicts, load_baseline, placebo_medians, verdict_markdown, window_table
from tqqq_gates_cli import BASELINE, PRICES, load_cash_rate

QLD = "/app/sweep_results/qld_spliced_1985_2026-10-24.parquet"


def load_qld(index: pd.DatetimeIndex, cash: pd.Series | None) -> pd.Series:
    p = Path(QLD)
    if p.exists():
        return pd.read_parquet(p).iloc[:, 0].reindex(index).ffill()
    from data.adapters.composed_market_data import build_default_market_data
    from strategy.adapters.band_rebalance_strategy import CALIBRATED_DRAG, build_synthetic_leveraged, splice_series
    from strategy.adapters.leveraged_etf_data import _close
    md = build_default_market_data()
    ndx = _close(md.fetch_ohlcv("^NDX", date(1985, 1, 1), date(2026, 10, 24))).dropna()
    real = _close(md.fetch_ohlcv("QLD", date(2006, 6, 21), date(2026, 10, 24))).dropna()
    syn = build_synthetic_leveraged(ndx, 2.0, annual_expense=CALIBRATED_DRAG[2.0], financing_rate=cash)
    s = splice_series(syn, real).dropna().rename("QLD")
    s.to_frame().to_parquet(p)
    print(f"QLD 스플라이스 {s.index[0].date()}~{s.index[-1].date()} (실제 {real.index[0].date()}~) 저장", file=sys.stderr)
    return s.reindex(index).ffill()


def combos_all() -> list[LevMixParams]:
    out = [LevMixParams("qld", switch_dd=None)] + [LevMixParams("qld", switch_dd=d) for d in (0.15, 0.18, 0.21)]
    for w in (0.33, 0.5, 0.67):
        out += [LevMixParams("mix", w, None), LevMixParams("mix", w, 0.18)]
    return out


def neighbors(p: LevMixParams) -> list[LevMixParams]:
    if p.kind == "qld":
        if p.switch_dd is None:
            return []
        dds = [0.15, 0.18, 0.21]; i = dds.index(p.switch_dd)
        return [LevMixParams("qld", switch_dd=dds[j]) for j in (i - 1, i + 1) if 0 <= j < 3]
    ws = [0.33, 0.5, 0.67]; i = ws.index(p.w_tqqq)
    return [LevMixParams("mix", ws[j], p.switch_dd) for j in (i - 1, i + 1) if 0 <= j < 3]


def main(cfg: dict) -> dict:
    t0 = time.time()
    px = pd.read_parquet(PRICES); base = load_baseline(BASELINE); cash = load_cash_rate()
    qld = load_qld(px.index, cash)
    prices = pd.DataFrame({"TQQQ": px["TQQQ"], "QLD": qld, "QQQ": px["QQQ"]})
    n_trials = int(cfg.get("n_trials", 1)); seeds = int(cfg.get("placebo_seeds", 30))
    combos = combos_all()
    rows, verdicts, tables, weights = [], {}, {}, {}
    for i, p in enumerate(combos):
        w = lev_mix_weights(prices, px["QQQ"], p); weights[p.name] = w
        tbl = window_table(px["TQQQ"], w, base, "dev", costs=GATE_COSTS, cash_rate=cash, prices=prices)
        v = gate_verdicts(tbl, n_trials); tables[p.name], verdicts[p.name] = tbl, v
        a10, ainf = v["G1"]["A-10"], v["G1"]["A-∞"]; a = tbl[tbl.bundle == "A-10"]; b = tbl[tbl.bundle == "B-10"]
        rows.append({"name": p.name, "kind": p.kind, "w_tqqq": p.w_tqqq if p.kind == "mix" else None, "switch_dd": p.switch_dd,
                     "a10_win": a10["win_rate"], "a10_med": a10["ratio_median"], "a10_min": a10["ratio_min"],
                     "ainf_win": ainf["win_rate"], "ainf_med": ainf["ratio_median"], "ainf_min": ainf["ratio_min"],
                     "g2_shallow": v["G2"]["shallower_rate"], "worst_mdd": v["G2"]["worst_mdd"], "a10_mdd_med": float(a["strat_mdd"].median()), "g3_fail": v["G3"]["n_fail"],
                     "b_beat_tqqq": v["G4"]["beat_tqqq_rate"], "b_beat_qqq": v["G4"]["beat_qqq_rate"], "b_worst_cagr": v["G4"]["worst_cagr"], "b_loss": v["G4"]["loss_windows"],
                     "b_ret_med": v["G4"]["ret_median"], "b_dotcom": v["G4"]["dotcom_strat_median"], "b_dotcom_qqq": v["G4"]["dotcom_qqq_median"], "b_worst_mdd": v["G4"]["worst_mdd"],
                     "a10_ret_med": float(a["strat_ret"].median()), "trades_a10": float(a["trades"].median()),
                     "first_fail": v["first_fail"], "passed_through": v["passed_through"]})
        print(f"[{i + 1}/{len(combos)}] {p.name}: A-10 {a10['win_rate']:.0%}/{a10['ratio_median']:.2f}/{a10['ratio_min']:.2f} · A-∞ {ainf['ratio_median']:.2f} · MDD {v['G2']['worst_mdd']:.0%} · G3 미달 {v['G3']['n_fail']} · "
              f"B TQQQ {v['G4']['beat_tqqq_rate']:.0%} QQQ {v['G4']['beat_qqq_rate']:.0%} 최악 {v['G4']['worst_cagr']:+.1%} 손실 {v['G4']['loss_windows']} 닷컴 {v['G4']['dotcom_strat_median']:+.0%} · 탈락 {v['first_fail']} ({time.time() - t0:.0f}s)", file=sys.stderr)
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
    out = {"cfg": cfg, "n_trials": n_trials, "rows": rows, "verdicts": verdicts, "elapsed_s": time.time() - t0}
    md = render(df, verdicts, tables)
    if cfg.get("out"):
        Path(cfg["out"]).write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
    if cfg.get("md"):
        Path(cfg["md"]).write_text(md)
    print(md[:2500])
    return out


def render(df, verdicts, tables) -> str:
    L = ["# H11 — QLD(2×) ± 스위치 · TQQQ/QQQ 분기 혼합 ± 스위치 (개발 판, 끝 2024-06-30)", "", f"조합 {len(df)}", "",
         "| 조합 | A-10 승률·중앙·최악 | A-∞ 승률·중앙·최악 | A-10 세후 중앙·MDD 중앙·매매 | G2 얕음·최악MDD | G3 미달 | B-10 TQQQ·QQQ 이김·최악 CAGR·손실 창·세후 중앙·닷컴 중앙(QQQ)·최악 MDD | G5 | G6 | 탈락 |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    order = {"G1": 0, "G2": 1, "G3": 2, "G4": 3, "G5": 4, "G6": 5, None: 6}
    d = df.assign(_o=df["first_fail"].map(order)).sort_values(["_o", "a10_med"], ascending=[False, False])
    for r in d.itertuples(index=False):
        L.append(f"| {r.name} | {r.a10_win:.0%} · {r.a10_med:.2f} · {r.a10_min:.2f} | {r.ainf_win:.0%} · {r.ainf_med:.2f} · {r.ainf_min:.2f} | {r.a10_ret_med:+.0%} · {r.a10_mdd_med:.0%} · {r.trades_a10:.0f} | "
                 f"{r.g2_shallow:.0%}·{r.worst_mdd:.0%} | {r.g3_fail} | {r.b_beat_tqqq:.0%}·{r.b_beat_qqq:.0%}·{r.b_worst_cagr:+.1%}·{r.b_loss}·{r.b_ret_med:+.0%}·{r.b_dotcom:+.0%}({r.b_dotcom_qqq:+.0%})·{r.b_worst_mdd:.0%} | "
                 f"{getattr(r, 'g5_pct', float('nan')):.0f}% | {getattr(r, 'g6', '—')} | {r.first_fail or '통과'} |")
    L.append("")
    best = d.iloc[0]
    L += [f"## 최고 조합 상세 — {best['name']}", "", verdict_markdown(best["name"], tables[best["name"]], verdicts[best["name"]], "dev")]
    return "\n".join(L)


if __name__ == "__main__":
    main(json.loads(sys.argv[1]) if len(sys.argv) > 1 else {})
