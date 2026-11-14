"""급등주 탐색 S5 — 계단식(분할 계좌) 운용으로 시작일 운을 없앤 매끄러운 모멘텀 vs TQQQ.

실행: docker compose exec backtester python surge_s5_cli.py '{"out": "/app/sweep_results/surge_s5"}'
봉인 구간: '{"start": "2024-01-02", "end": "2026-11-10", "mid": "2025-06-30", "out": ".../surge_s5_sealed"}' — 장부에 열람 기록 후.
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from strategy.adapters.surge_events import load_symbol
from strategy.adapters.surge_rank_portfolio import RankRules, curve_stats, run_rank, scores, tranche_curve, wide

SR = Path("/app/sweep_results")


def main(cfg):
    out = Path(cfg.get("out", SR / "surge_s5")); start = cfg.get("start", "2018-02-01"); end = cfg.get("end", "2023-12-31"); mid = pd.Timestamp(cfg.get("mid", "2020-12-31"))
    n_tr = int(cfg.get("tranches", 9)); seeds = int(cfg.get("seeds", 10)); every = int(cfg.get("every", 63))
    grid = cfg.get("grid", {"thr": [None, 0.51, 0.53, 0.55], "k": [5, 10, 20]})
    panel = pd.read_parquet(SR / "surge_panel.parquet", columns=["date", "symbol", "C", "O", "raw_c", "dv"])
    w = wide(panel); del panel
    c = w["C"]; r = c.pct_change(fill_method=None)
    up = (r > 0).where(r.notna()).rolling(252, min_periods=200).mean()
    sc = scores(w, "mom12_1"); days = c.index[(c.index >= start) & (c.index <= end)]
    offs = [int(round(i * every / n_tr)) for i in range(n_tr)]
    tq = load_symbol(SR / "surge_av_cache" / "TQQQ.parquet"); qq = load_symbol(SR / "surge_av_cache" / "QQQ.parquet")

    def bench(f):
        e = f["C"].reindex(days).ffill() / f["O"].reindex(days).bfill().iloc[1]; e.iloc[0] = 1.0
        return e

    def halves(eq):
        m = eq[eq.index <= mid].iloc[-1]
        return float(m / eq.iloc[0]), float(eq.iloc[-1] / m)

    bt, bq = bench(tq), bench(qq); st, sq = curve_stats(bt), curve_stats(bq); ht = halves(bt)
    p = lambda x: f"{x:+.0%}"  # noqa: E731
    L = [f"# S5 — 계단식 {n_tr} 분할 운용 vs TQQQ ({start}~{end})", "", f"12-1 모멘텀 상위 K · 주가 $10·거래대금 $50M · 20일 +50% 초과 제외 · {every} 거래일 교체를 {n_tr} 계좌가 {every // n_tr} 거래일 간격으로 엇갈려 · 교체분 비용 0.5% · 세전 · 배당 미포함",
         f"**TQQQ** ×{st['mult']:.2f} (CAGR {p(st['cagr'])}, MDD {p(st['mdd'])}, 전반 ×{ht[0]:.2f}·후반 ×{ht[1]:.2f}) · **QQQ** ×{sq['mult']:.2f} (MDD {p(sq['mdd'])})", "",
         "| 오른 날 문턱 | K | 배수 | CAGR | MDD | 전반 | 후반 | TQQQ 대비 | 전/후반 대비 | 무작위 중앙 | 무작위 이긴% |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    res, curves = [], {"TQQQ": bt, "QQQ": bq}
    for thr, k in itertools.product(grid["thr"], grid["k"]):
        ex = None if thr is None else up >= thr
        rules = RankRules(k=k, min_price=10.0, min_dv20=5e7, max_ret20=0.5, every=every)
        eq = tranche_curve(w, [run_rank(w, rules, str(days[o].date()), end, score=sc, extra_ok=ex) for o in offs], days)
        s = curve_stats(eq); h = halves(eq)
        rm = []
        for sd in range(seeds):
            rr = RankRules(rank="random", k=k, min_price=10.0, min_dv20=5e7, max_ret20=0.5, every=every)
            rm.append(float(np.mean([(1 + run_rank(w, rr, str(days[o].date()), end, rng=np.random.default_rng(1000 * sd + o), extra_ok=ex)["ret"]).prod() for o in offs])))
        rm = np.array(rm) if rm else np.array([np.nan])
        row = {"thr": thr, "k": k, **s, "h1": h[0], "h2": h[1], "vs": s["mult"] / st["mult"], "vs1": h[0] / ht[0], "vs2": h[1] / ht[1], "rnd_med": float(np.median(rm)), "rnd_beat": float((s["mult"] > rm).mean())}
        res.append(row); curves[f"{thr}_{k}"] = eq
        L.append(f"| {'-' if thr is None else f'{thr:.0%}'} | {k} | ×{row['mult']:.2f} | {p(row['cagr'])} | {p(row['mdd'])} | ×{row['h1']:.2f} | ×{row['h2']:.2f} | {row['vs']:.2f} | {row['vs1']:.2f} / {row['vs2']:.2f} | ×{row['rnd_med']:.2f} | {row['rnd_beat']:.0%} |")
        print(L[-1], file=sys.stderr, flush=True)
    df = pd.DataFrame(res); ok = df[(df["vs"] > 1) & (df["vs1"] >= 0.7) & (df["vs2"] >= 0.7) & (df["rnd_beat"] >= 0.95)]
    L += ["", f"**관문 통과: {len(ok)} / {len(df)}**"]
    Path(f"{out}.md").write_text("\n".join(L)); df.to_json(f"{out}.json", orient="records", force_ascii=False, indent=1); pd.DataFrame(curves).to_parquet(f"{out}_curves.parquet"); print("\n".join(L))


if __name__ == "__main__":
    main(json.loads(sys.argv[1]) if len(sys.argv) > 1 else {})
