"""급등주 탐색 S3 — 생존 편향 없는 패널로 상위 K 월 교체(모멘텀·신고가 근처) vs TQQQ·무작위.

실행: docker compose exec backtester python surge_s3_cli.py '{"out": "/app/sweep_results/surge_s3"}'
개발 구간(2018-02~2023-12)만. cfg "start"/"end" 로 바꿀 수 있으나 봉인 구간은 장부에 적고 연다.
"""

from __future__ import annotations

import itertools
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from strategy.adapters.surge_events import load_symbol
from strategy.adapters.surge_rank_portfolio import RankRules, bench_table, run_rank, scores, stats, wide

SR = Path("/app/sweep_results")


def main(cfg):
    out = Path(cfg.get("out", SR / "surge_s3")); start = cfg.get("start", "2018-02-01"); end = cfg.get("end", "2023-12-31"); seeds = int(cfg.get("seeds", 30))
    mid = cfg.get("mid", "2020-12-31")
    panel = pd.read_parquet(SR / "surge_panel.parquet", columns=["date", "symbol", "C", "O", "raw_c", "dv"])
    w = wide(panel); del panel
    tq = load_symbol(SR / "surge_av_cache" / "TQQQ.parquet")["O"]; qq = load_symbol(SR / "surge_av_cache" / "QQQ.parquet")["O"]
    sc = {r: scores(w, r) for r in ("mom12_1", "mom6", "mom3", "near_high")}
    p = lambda x: f"{x:+.0%}"  # noqa: E731

    def halves(t):
        a = t[t["date"] <= mid]; b = t[t["date"] > mid]
        return float((1 + a["ret"]).prod()), float((1 + b["ret"]).prod())

    base = run_rank(w, RankRules(rank="mom6", k=10), start, end, score=sc["mom6"])
    bt, bq = bench_table(tq, base), bench_table(qq, base)
    st_t, st_q = stats(bt), stats(bq); ht, hq = halves(bt), halves(bq)
    L = ["# S3 — 생존 편향 없는 패널, 상위 K 월 교체 vs TQQQ", "", f"{start}~{end} · 신호 = 리밸런스일 종가, 체결 = 다음날 시가, 21 거래일마다 · 동일비중 · 교체분에 비용 0.5% · 세전 · 배당 미포함 · 폐지 = 마지막 종가 청산",
         f"전반 = ~{mid}, 후반 = 그 뒤. 무작위 = 같은 유니버스에서 K 종목 무작위 {seeds} 시드의 중앙과, 전략이 무작위보다 나은 시드 비율.", "",
         f"**기준선**: TQQQ ×{st_t['mult']:.2f} (CAGR {p(st_t['cagr'])}, MDD {p(st_t['mdd'])}, 전반 ×{ht[0]:.2f}·후반 ×{ht[1]:.2f}) · QQQ ×{st_q['mult']:.2f} (CAGR {p(st_q['cagr'])}, MDD {p(st_q['mdd'])})", "",
         "| 순위 | K | 유니버스 | 추가 조건 | 배수 | CAGR | MDD | 전반 | 후반 | TQQQ 대비 | 전반/후반 대비 | 교체율 | 무작위 중앙 | 무작위 이긴% |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    unis = {"$5·$20M": (5.0, 2e7), "$10·$50M": (10.0, 5e7), "$2·$5M": (2.0, 5e6)}
    extras = {"-": {}, "200일선 위": {"above_sma200": True}, "20일 +50% 이하": {"max_ret20": 0.5}}
    res = []
    for (uname, (mp, mdv)), k in itertools.product(unis.items(), (5, 10, 20)):
        rng_mults = []
        for sd in range(seeds):
            t = run_rank(w, RankRules(rank="random", k=k, min_price=mp, min_dv20=mdv), start, end, rng=np.random.default_rng(sd))
            rng_mults.append(stats(t)["mult"])
        rm = np.array(rng_mults)
        for rank, (ename, ekw) in itertools.product(sc, extras.items()):
            rules = RankRules(rank=rank, k=k, min_price=mp, min_dv20=mdv, **ekw)
            t = run_rank(w, rules, start, end, score=sc[rank]); s = stats(t); h = halves(t)
            res.append({"rank": rank, "k": k, "uni": uname, "extra": ename, **s, "h1": h[0], "h2": h[1], "vs": s["mult"] / st_t["mult"], "vs1": h[0] / ht[0], "vs2": h[1] / ht[1], "rnd_med": float(np.median(rm)), "rnd_beat": float((s["mult"] > rm).mean())})
            r = res[-1]
            L.append(f"| {rank} | {k} | {uname} | {ename} | ×{r['mult']:.2f} | {p(r['cagr'])} | {p(r['mdd'])} | ×{r['h1']:.2f} | ×{r['h2']:.2f} | {r['vs']:.2f} | {r['vs1']:.2f} / {r['vs2']:.2f} | {r['turnover']:.0%} | ×{r['rnd_med']:.2f} | {r['rnd_beat']:.0%} |")
            print(L[-1], file=sys.stderr, flush=True)
    df = pd.DataFrame(res)
    ok = df[(df["vs"] > 1) & (df["vs1"] >= 0.7) & (df["vs2"] >= 0.7) & (df["rnd_beat"] >= 0.95)]
    L += ["", f"**S1 관문(TQQQ 초과 & 전·후반 각 0.7 이상) + S2 관문(무작위 95%)** 통과: {len(ok)} / {len(df)}", ""]
    L += [f"- {r['rank']} K{r.k} {r.uni} {r.extra}: ×{r.mult:.2f} (TQQQ 대비 {r.vs:.2f})" for _, r in ok.iterrows()]
    md = "\n".join(L); Path(f"{out}.md").write_text(md); df.to_json(f"{out}.json", orient="records", force_ascii=False, indent=1); print(md)


if __name__ == "__main__":
    main(json.loads(sys.argv[1]) if len(sys.argv) > 1 else {})
