"""급등주 탐색 S4 — 12-1 모멘텀 뼈대 + 소거 조건(매끄러움·변동성 상한·시장 필터) vs TQQQ.

실행: docker compose exec backtester python surge_s4_cli.py '{"out": "/app/sweep_results/surge_s4"}'
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from strategy.adapters.surge_events import load_symbol
from strategy.adapters.surge_rank_portfolio import RankRules, bench_table, run_rank, scores, stats, wide

SR = Path("/app/sweep_results")


def main(cfg):
    out = Path(cfg.get("out", SR / "surge_s4")); start = cfg.get("start", "2018-02-01"); end = cfg.get("end", "2023-12-31"); mid = cfg.get("mid", "2020-12-31"); seeds = int(cfg.get("seeds", 30))
    panel = pd.read_parquet(SR / "surge_panel.parquet", columns=["date", "symbol", "C", "O", "raw_c", "dv"])
    w = wide(panel); del panel
    c = w["C"]; r = c.pct_change(fill_method=None)
    up = (r > 0).where(r.notna()).rolling(252, min_periods=200).mean()          # 지난 1년 오른 날 비율
    vol60 = r.rolling(60, min_periods=40).std()
    big = (r.abs() >= 0.15).rolling(252, min_periods=200).sum()                 # 지난 1년 ±15% 이상 움직인 날 수
    tqf = load_symbol(SR / "surge_av_cache" / "TQQQ.parquet"); qqf = load_symbol(SR / "surge_av_cache" / "QQQ.parquet")
    q = qqf["C"].reindex(c.index).ffill()
    mkt = {"-": None, "QQQ 200일선 위": q > q.rolling(200).mean(), "QQQ 고점 -18% 이내": q / q.rolling(252).max() - 1 > -0.18}
    elim = {"-": None, "오른 날 ≥ 53%": up >= 0.53, "60일 변동성 ≤ 4%": vol60 <= 0.04, "±15% 날 ≤ 2회": big <= 2,
            "오른 날 ≥ 53% & 변동성 ≤ 4%": (up >= 0.53) & (vol60 <= 0.04), "±15% 날 0회 & 변동성 ≤ 3%": (big == 0) & (vol60 <= 0.03)}
    sc = scores(w, "mom12_1")
    p = lambda x: f"{x:+.0%}"  # noqa: E731
    halves = lambda t: (float((1 + t.loc[t["date"] <= mid, "ret"]).prod()), float((1 + t.loc[t["date"] > mid, "ret"]).prod()))  # noqa: E731
    base = run_rank(w, RankRules(k=10), start, end, score=sc)
    bt = bench_table(tqf["O"], base); st_t = stats(bt); ht = halves(bt)
    L = ["# S4 — 12-1 모멘텀 + 소거 조건 vs TQQQ", "", f"{start}~{end} · S3 과 같은 체결·비용(교체분 0.5%)·세전 · 유니버스 주가 $10·거래대금 $50M · 전 조합 '직전 20일 +50% 초과 제외' 포함",
         f"**TQQQ** ×{st_t['mult']:.2f} (MDD {p(st_t['mdd'])}, 전반 ×{ht[0]:.2f}·후반 ×{ht[1]:.2f})", "",
         "| 소거 | 시장 필터 | K | 간격 | 배수 | CAGR | MDD | 전반 | 후반 | TQQQ 대비 | 전/후반 대비 | 현금 구간% | 무작위 이긴% |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    res = []
    for (en, ex), (mn, mk), k, every in itertools.product(elim.items(), mkt.items(), (5, 10, 20), (21, 63)):
        rules = RankRules(k=k, min_price=10.0, min_dv20=5e7, max_ret20=0.5, every=every)
        t = run_rank(w, rules, start, end, score=sc, extra_ok=ex, market_ok=mk); s = stats(t); h = halves(t)
        rm = np.array([stats(run_rank(w, RankRules(rank="random", k=k, min_price=10.0, min_dv20=5e7, max_ret20=0.5, every=every), start, end, rng=np.random.default_rng(sd), extra_ok=ex, market_ok=mk))["mult"] for sd in range(seeds)]) if cfg.get("placebo", True) else np.array([np.nan])
        row = {"elim": en, "market": mn, "k": k, "every": every, **s, "h1": h[0], "h2": h[1], "vs": s["mult"] / st_t["mult"], "vs1": h[0] / ht[0], "vs2": h[1] / ht[1], "cash": float((t["n"] == 0).mean()), "rnd_med": float(np.median(rm)), "rnd_beat": float((s["mult"] > rm).mean())}
        res.append(row)
        L.append(f"| {en} | {mn} | {k} | {every} | ×{row['mult']:.2f} | {p(row['cagr'])} | {p(row['mdd'])} | ×{row['h1']:.2f} | ×{row['h2']:.2f} | {row['vs']:.2f} | {row['vs1']:.2f} / {row['vs2']:.2f} | {row['cash']:.0%} | {row['rnd_beat']:.0%} |")
        print(L[-1], file=sys.stderr, flush=True)
    df = pd.DataFrame(res)
    ok = df[(df["vs"] > 1) & (df["vs1"] >= 0.7) & (df["vs2"] >= 0.7) & (df["rnd_beat"] >= 0.95)]
    L += ["", f"**관문(TQQQ 초과 & 전·후반 각 0.7 이상 & 무작위 95%)** 통과: {len(ok)} / {len(df)} · TQQQ 초과만: {(df['vs'] > 1).sum()}", ""]
    L += [f"- {r.elim} / {r.market} / K{r.k} / {r.every}일: ×{r.mult:.2f} (TQQQ 대비 {r.vs:.2f}, MDD {r.mdd:+.0%})" for r in ok.itertuples()]
    top = df.sort_values("vs", ascending=False).head(10)
    L += ["", "상위 10 (TQQQ 대비):", ""] + [f"- {r.elim} / {r.market} / K{r.k} / {r.every}일: ×{r.mult:.2f} · 대비 {r.vs:.2f} ({r.vs1:.2f}/{r.vs2:.2f}) · MDD {r.mdd:+.0%} · 무작위 {r.rnd_beat:.0%}" for r in top.itertuples()]
    Path(f"{out}.md").write_text("\n".join(L)); df.to_json(f"{out}.json", orient="records", force_ascii=False, indent=1)
    print("\n".join(L[-16:]))


if __name__ == "__main__":
    main(json.loads(sys.argv[1]) if len(sys.argv) > 1 else {})
