"""H5 — 성장 ≥11 리스트 곡선 + TQQQ 혼합 → 예비 검사(§2-1) 또는 전체 관문 (설계 tqqq_search_runs/H5_설계.md).

    docker compose exec backtester python tqqq_h5_cli.py '{"dump": "/app/sweep_results/tqqq_h5_list150_dev_dump.json", "mode": "prelim", "out": ..., "md": ...}'

리스트 자산 가격 = growth_hold dump 의 세전 곡선(`equity`; 유리한 쪽) — `curve: "after_tax"` 면 세후 곡선(이중 과세, 불리한 쪽).
"""

from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

import pandas as pd

from strategy.adapters.list_tqqq_mix import ListMixParams, list_mix_weights
from strategy.adapters.tqqq_gates import GATE_COSTS, gate_verdicts, load_baseline, verdict_markdown, window_markdown, window_table
from tqqq_gates_cli import BASELINE, PRICES, load_cash_rate

PRELIM_A_STARTS = ["2010-02-11", "2011-04-01", "2012-07-01", "2013-10-01", "2014-04-01"]
PRELIM_B_STARTS = ["1986-01-01", "1989-01-01", "1992-01-01", "1995-01-01", "1997-01-01", "1999-10-01", "2001-01-01", "2003-01-01", "2006-01-01", "2008-01-01", "2010-01-01", "2013-01-01"]
GRID_PRELIM = {"w_tqqq": [0.3, 0.5, 0.7], "rebalance": ["Q"], "switch": [False, True]}


def load_list_curve(dump_path: str, curve: str = "equity") -> pd.Series:
    d = json.load(open(dump_path))
    c = d["combos"][0][curve]
    s = pd.Series({pd.Timestamp(k): float(v) for k, v in c.items()}).sort_index()
    return s.rename("LIST")


def prelim_baseline(base: pd.DataFrame, list_start: pd.Timestamp) -> pd.DataFrame:
    a = base[(base.edition == "dev") & base.bundle.isin(["A-10", "A-∞"]) & base.start.isin(pd.to_datetime(PRELIM_A_STARTS))]
    s = base[(base.edition == "dev") & (base.bundle == "S-5")]
    b = base[(base.edition == "dev") & (base.bundle == "B-10") & base.start.isin(pd.to_datetime(PRELIM_B_STARTS)) & (base.start >= list_start + pd.Timedelta(days=400))]
    return pd.concat([a, s, b])


def prelim_verdict(tbl: pd.DataFrame) -> dict:
    a = tbl[tbl.bundle.isin(["A-10", "A-∞"])]
    s5 = tbl[tbl.bundle == "S-5"]
    med = float(a["ratio"].median()); win = float((a["ratio"] > 1).mean())
    s5_bad = int((s5["strat_ret"] < 0.5 * s5["tqqq_ret"]).sum())
    return {"ratio_median": med, "win_rate": win, "s5_below_half": s5_bad, "n_a": int(len(a)),
            "pass": med >= 0.8 and win >= 0.4 and s5_bad <= 2}


def main(cfg: dict) -> dict:
    t0 = time.time()
    px = pd.read_parquet(PRICES); base = load_baseline(BASELINE); cash = load_cash_rate()
    curve = cfg.get("curve", "equity")
    lst = load_list_curve(cfg["dump"], curve).reindex(px.index)
    list_start = lst.dropna().index[0]
    prices = pd.DataFrame({"TQQQ": px["TQQQ"], "LIST": lst})
    mode = cfg.get("mode", "prelim")
    bl = prelim_baseline(base, list_start) if mode == "prelim" else base
    grid = cfg.get("grid", GRID_PRELIM)
    combos = [ListMixParams(w, r, s) for w, r, s in itertools.product(grid["w_tqqq"], grid["rebalance"], grid["switch"])]
    print(f"{mode}: 리스트 곡선 {curve} {list_start.date()}~{lst.dropna().index[-1].date()} · 창 {len(bl)} · 조합 {len(combos)}", file=sys.stderr)
    rows, tables, verdicts = [], {}, {}
    for i, p in enumerate(combos):
        w = list_mix_weights(prices, px["QQQ"], p)
        tbl = window_table(px["TQQQ"], w, bl, "dev", costs=GATE_COSTS, cash_rate=cash, prices=prices)
        tables[p.name] = tbl
        pv = prelim_verdict(tbl); gv = gate_verdicts(tbl, int(cfg.get("n_trials", 6)))
        verdicts[p.name] = {"prelim": pv, "gates": gv}
        a10 = tbl[tbl.bundle == "A-10"]
        rows.append({"name": p.name, **p.__dict__, **pv, "a10_ret_med": float(a10["strat_ret"].median()), "a10_mdd_med": float(a10["strat_mdd"].median()),
                     "a10_min_ratio": float(a10["ratio"].min()), "first_fail": gv["first_fail"],
                     "s5": [(str(r.start.date())[:7], round(float(r.strat_ret), 2), round(float(r.tqqq_ret), 2)) for r in tbl[tbl.bundle == "S-5"].itertuples()]})
        print(f"[{i + 1}/{len(combos)}] {p.name}: A 비율 중앙 {pv['ratio_median']:.2f} 승률 {pv['win_rate']:.0%} S-5 반토막 {pv['s5_below_half']} → 예비 {'통과' if pv['pass'] else '탈락'} · 관문 탈락 {gv['first_fail']} ({time.time() - t0:.0f}s)", file=sys.stderr)
    out = {"cfg": cfg, "mode": mode, "curve": curve, "list_start": str(list_start.date()), "rows": rows, "verdicts": verdicts, "elapsed_s": time.time() - t0,
           "prelim_pass_any": any(r["pass"] for r in rows)}
    md = render(out, tables)
    if cfg.get("out"):
        Path(cfg["out"]).write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
    if cfg.get("md"):
        Path(cfg["md"]).write_text(md)
    print(md[:2500])
    return out


def render(out, tables) -> str:
    L = [f"# H5 — 성장 ≥11 리스트 + TQQQ 혼합 ({out['mode']}, 리스트 곡선 {out['curve']}, 개발 판)", "",
         f"리스트 곡선 {out['list_start']}~ · 예비 통과 조합 {'있음' if out['prelim_pass_any'] else '없음'}", "",
         "| TQQQ w | 리밸 | 스위치 | A 표본 비율 중앙 | A 승률 | A-10 최악 비율 | A-10 세후 중앙 · MDD | S-5 (전략 / TQQQ) | S-5 반토막 | 예비 | 관문 탈락 |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in out["rows"]:
        s5 = " · ".join(f"{a} {b:+.0%}/{c:+.0%}" for a, b, c in r["s5"])
        L.append(f"| {r['w_tqqq']:.1f} | {r['rebalance']} | {'18' if r['switch'] else 'off'} | {r['ratio_median']:.2f} | {r['win_rate']:.0%} | {r['a10_min_ratio']:.2f} | {r['a10_ret_med']:+.0%} · {r['a10_mdd_med']:.0%} | {s5} | {r['s5_below_half']} | {'✅' if r['pass'] else '❌'} | {r['first_fail'] or '통과'} |")
    L.append("")
    best = max(out["rows"], key=lambda r: r["ratio_median"])
    L += [f"## 최고 조합 창별 — {best['name']}", "", window_markdown(tables[best["name"]], ("A-10", "A-∞", "S-5", "B-10"))]
    return "\n".join(L)


if __name__ == "__main__":
    main(json.loads(sys.argv[1]) if len(sys.argv) > 1 else {})
