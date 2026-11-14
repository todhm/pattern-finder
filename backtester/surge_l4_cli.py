"""2차 탐색 L4′ — "모두가 거래하는 대형 주도주" 집중: 거래대금 최상위 N 종목 안에서 모멘텀 상위 K, 계단식 9 분할 vs TQQQ.

시가총액 자료 없이 60일 평균 거래대금 순위를 "시대의 주도 대형주" 대용으로 쓴다(그날까지의 자료만).
실행: docker compose exec backtester python surge_l4_cli.py '{"out": "/app/sweep_results/surge_l4"}'
봉인: '{"start": "2024-01-02", "end": "2026-11-10", "mid": "2025-06-30", "seeds": 0, "out": ".../surge_l4_sealed"}' — 장부에 열람 기록 후.
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
    out = Path(cfg.get("out", SR / "surge_l4")); start = cfg.get("start", "2018-02-01"); end = cfg.get("end", "2023-12-31"); mid = pd.Timestamp(cfg.get("mid", "2020-12-31"))
    n_tr = int(cfg.get("tranches", 9)); seeds = int(cfg.get("seeds", 5)); every = int(cfg.get("every", 63))
    grid = cfg.get("grid", {"top": [20, 30, 50, 100], "rank": ["mom12_1", "mom6", "mom3"], "k": [3, 5, 10], "trend": [False, True]})
    panel = pd.read_parquet(cfg.get("panel", SR / "surge_panel.parquet"), columns=["date", "symbol", "C", "O", "raw_c", "dv"]); panel = panel[panel["date"] <= end]
    w = wide(panel); del panel
    dv60 = w["dv"].rolling(60, min_periods=40).mean().where(w["alive"] & (w["raw_c"] >= 10.0)); rk = dv60.rank(axis=1, ascending=False, method="first")
    keep = (rk <= max(grid["top"])).any()
    w = {k: v.loc[:, keep] for k, v in w.items()}; rk = rk.loc[:, keep]; c = w["C"]
    above = c > c.rolling(200, min_periods=150).mean(); sc = {r: scores(w, r) for r in grid["rank"]}
    days = c.index[(c.index >= start) & (c.index <= end)]; offs = [int(round(i * every / n_tr)) for i in range(n_tr)]
    if cfg.get("bench"):                                   # 다른 표본용: TQQQ_C/TQQQ_O/QQQ_C/QQQ_O 열을 가진 parquet(합성 TQQQ 등)
        b = pd.read_parquet(cfg["bench"]); tq = b[["TQQQ_C", "TQQQ_O"]].set_axis(["C", "O"], axis=1); qq = b[["QQQ_C", "QQQ_O"]].set_axis(["C", "O"], axis=1)
    else:
        tq = load_symbol(SR / "surge_av_cache" / "TQQQ.parquet"); qq = load_symbol(SR / "surge_av_cache" / "QQQ.parquet")

    def bench(f):
        e = f["C"].reindex(days).ffill() / f["O"].reindex(days).bfill().iloc[1]; e.iloc[0] = 1.0
        return e

    def halves(eq):
        m = eq[eq.index <= mid].iloc[-1]
        return float(m / eq.iloc[0]), float(eq.iloc[-1] / m)

    def run(rules, ok, score=None, seed=None):
        tabs = [run_rank(w, rules, str(days[o].date()), end, score=score, extra_ok=ok, rng=None if seed is None else np.random.default_rng(1000 * seed + o)) for o in offs]
        return tranche_curve(w, tabs, days), [float((1 + t["ret"]).prod()) for t in tabs]

    bt, bq = bench(tq), bench(qq); st, sq = curve_stats(bt), curve_stats(bq); ht = halves(bt); pm = lambda x: f"{x:+.0%}"  # noqa: E731
    L = [f"# L4′ — 거래대금 최상위 대형 주도주 집중 ({start}~{end})", "",
         f"유니버스 = 주가 $10 이상 중 60일 평균 거래대금 상위 N. 모멘텀 상위 K 동일비중, {every} 거래일 교체를 {n_tr} 계좌가 엇갈려. 교체분 비용 0.5%, 세전, 배당 미포함.",
         f"**TQQQ** ×{st['mult']:.2f} (CAGR {pm(st['cagr'])}, MDD {pm(st['mdd'])}, 전반 ×{ht[0]:.2f}·후반 ×{ht[1]:.2f}) · **QQQ** ×{sq['mult']:.2f} (MDD {pm(sq['mdd'])})", "",
         "| 상위 N | 순위 | K | 200일선 위만 | 배수 | CAGR | MDD | 전반 | 후반 | TQQQ 대비 | 전/후반 대비 | 계좌별 배수(최소~최대) | 무작위 K 중앙 | 무작위 이긴% |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    res, curves = [], {"TQQQ": bt, "QQQ": bq}; rnd_cache = {}
    for top, k in itertools.product(grid["top"], grid["k"]):
        uni = rk <= top
        rr = RankRules(rank="random", k=k, min_price=10.0, min_dv20=0.0, every=every)
        rnd_cache[(top, k)] = np.array([curve_stats(run(rr, uni, seed=sd)[0])["mult"] for sd in range(seeds)]) if seeds else np.array([np.nan])
        eqa, _ = run(RankRules(k=top, min_price=10.0, min_dv20=0.0, every=every), uni, score=-rk) if k == grid["k"][0] else (None, None)
        if eqa is not None:
            s = curve_stats(eqa); h = halves(eqa)
            L.append(f"| {top} | (전부 보유) | {top} | - | ×{s['mult']:.2f} | {pm(s['cagr'])} | {pm(s['mdd'])} | ×{h[0]:.2f} | ×{h[1]:.2f} | {s['mult'] / st['mult']:.2f} | {h[0] / ht[0]:.2f} / {h[1] / ht[1]:.2f} | - | - | - |")
            res.append({"top": top, "rank": "all", "k": top, "trend": False, **s, "h1": h[0], "h2": h[1], "vs": s["mult"] / st["mult"]})
    for top, rank, k, trend in itertools.product(grid["top"], grid["rank"], grid["k"], grid["trend"]):
        uni = (rk <= top) & above if trend else rk <= top
        eq, ms = run(RankRules(k=k, min_price=10.0, min_dv20=0.0, every=every), uni, score=sc[rank]); s = curve_stats(eq); h = halves(eq); rm = rnd_cache[(top, k)]
        row = {"top": top, "rank": rank, "k": k, "trend": trend, **s, "h1": h[0], "h2": h[1], "vs": s["mult"] / st["mult"], "vs1": h[0] / ht[0], "vs2": h[1] / ht[1], "tr_min": min(ms), "tr_max": max(ms), "rnd_med": float(np.median(rm)), "rnd_beat": float((s["mult"] > rm).mean())}
        res.append(row); curves[f"{top}_{rank}_{k}_{int(trend)}"] = eq
        L.append(f"| {top} | {rank} | {k} | {'예' if trend else '-'} | ×{row['mult']:.2f} | {pm(row['cagr'])} | {pm(row['mdd'])} | ×{h[0]:.2f} | ×{h[1]:.2f} | {row['vs']:.2f} | {row['vs1']:.2f} / {row['vs2']:.2f} | ×{row['tr_min']:.2f}~×{row['tr_max']:.2f} | ×{row['rnd_med']:.2f} | {row['rnd_beat']:.0%} |")
        print(L[-1], file=sys.stderr, flush=True)
    df = pd.DataFrame(res); g = df[df["rank"] != "all"]
    L += ["", f"**TQQQ 를 넘은 칸 {int((g['vs'] > 1).sum())} / {len(g)}** · 0.9 배 이상 {int((g['vs'] >= 0.9).sum())} / {len(g)} · 전·후반 모두 0.9 배 이상 {int(((g['vs1'] >= 0.9) & (g['vs2'] >= 0.9)).sum())} / {len(g)}"]
    Path(f"{out}.md").write_text("\n".join(L)); df.to_json(f"{out}.json", orient="records", force_ascii=False, indent=1); pd.DataFrame(curves).to_parquet(f"{out}_curves.parquet"); print("\n".join(L))


if __name__ == "__main__":
    main(json.loads(sys.argv[1]) if len(sys.argv) > 1 else {})
