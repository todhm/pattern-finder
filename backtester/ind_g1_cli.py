"""산업군 성장 탐색 G1 — 49 산업을 월말 숫자로 줄 세워 상위 K 개를 들면 시장·무작위 산업을 이기는가 (1930~, 월간).

점수(industry_rank.signals) × K(3·5·10) × 교체 주기(1·12 달). 편도 비용 0.2%, 세전. 종목 수 5 미만 산업 제외.
기준선: 시장(전 종목 시총가중) · 49 산업 동일비중. 플라세보: 같은 날 같은 수의 산업을 무작위로 seeds 벌 → 창별 95 백분위.
통과 = 개발 창 전부에서 시장 초과 AND 플라세보 95 백분위 초과. 봉인(2024-01~)은 cfg end 를 명시해야만.
실행: docker compose exec backtester python ind_g1_cli.py '{}'
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from strategy.adapters.industry_rank import port_returns, signals, topk_weights, window_stats

SR = Path("/app/sweep_results")
WINDOWS = {"1930~1962": ("1930-01-31", "1962-12-31"), "1963~1989": ("1963-01-31", "1989-12-31"), "1990~2009": ("1990-01-31", "2009-12-31"), "2010~2023": ("2010-01-31", "2023-12-31")}


def load(end: str):
    d = pd.read_parquet(SR / "ind_ff49.parquet"); d = d[d["date"] <= end]; p = lambda c: d.pivot(index="date", columns="ind", values=c)  # noqa: E731
    mk = pd.read_parquet(SR / "ind_ff_mkt.parquet"); mk = mk[mk.index <= end]
    return p("vw"), p("ew"), p("n"), p("size"), p("bm"), mk


def main(cfg):
    out = Path(cfg.get("out", SR / "ind_g1")); end = cfg.get("end", "2023-12-31"); seeds = int(cfg.get("seeds", 200)); windows = cfg.get("windows", WINDOWS)
    vw, ew, n, size, bm, mk = load(end); rf = mk["rf"].reindex(vw.index).fillna(0.0); S = signals(vw, ew, n, size, bm)
    base = {"시장(시총가중)": window_stats(mk["mkt"].reindex(vw.index).fillna(0.0), windows), "49 산업 동일비중": window_stats(vw.mean(axis=1), windows)}; mkt = base["시장(시총가중)"]
    any_score = vw.notna().astype(float).where(vw.notna()); rng = np.random.default_rng(11); plc = {}
    for k, h in itertools.product((3, 5, 10), (1, 12)):
        m = np.array([[window_stats(port_returns(topk_weights(any_score, vw, k, h, rng=rng, min_n=n), vw, rf), windows)[wn]["mult"] for wn in windows] for _ in range(seeds)])
        plc[(k, h)] = {"p50": np.median(m, axis=0), "p95": np.percentile(m, 95, axis=0), "all": m}; print("placebo", k, h, np.round(plc[(k, h)]["p95"], 1), file=sys.stderr, flush=True)
    fmt = lambda v: " | ".join(f"×{x['mult']:,.1f} ({x['mdd']:+.0%})" for x in v.values())  # noqa: E731
    L = [f"# G1 — 49 산업 상위 K 개 보유 (개발 ~{end}, 월간, 세전, 편도 0.2%)", "", "값 = 배수(MDD). 굵게 = 그 창에서 시장과 무작위 산업 95 백분위를 둘 다 넘음.", "", "| 기준선 | " + " | ".join(windows) + " |", "|---|" + "---|" * len(windows)]
    L += [f"| {k} | {fmt(v)} |" for k, v in base.items()] + [f"| 무작위 {k} 개·{h} 달 교체 (중앙 / 95 백분위) | " + " | ".join(f"×{a:,.1f} / ×{b:,.1f}" for a, b in zip(p["p50"], p["p95"])) + " |" for (k, h), p in plc.items()]
    L += ["", "| 점수 | K | 교체(달) | " + " | ".join(windows) + " | 통과 창 | 플라세보 백분위(창별) | 연 교체율 |", "|---|---|---|" + "---|" * len(windows) + "---|---|---|"]; res = []
    for (sn, sc), k, h in itertools.product(S.items(), (3, 5, 10), (1, 12)):
        w = topk_weights(sc, vw, k, h, min_n=n); st = window_stats(port_returns(w, vw, rf), windows); p = plc[(k, h)]
        ok = [st[wn]["mult"] > mkt[wn]["mult"] and st[wn]["mult"] > p["p95"][i] for i, wn in enumerate(windows)]; pct = [float((p["all"][:, i] < st[wn]["mult"]).mean()) for i, wn in enumerate(windows)]
        turn = float(w.diff().abs().sum(axis=1).mean() * 6)
        res.append({"score": sn, "k": k, "hold": h, **{f"{wn}_mult": st[wn]["mult"] for wn in windows}, **{f"{wn}_mdd": st[wn]["mdd"] for wn in windows}, **{f"{wn}_pct": pc for wn, pc in zip(windows, pct)}, "n_ok": sum(ok), "turn": turn})
        L.append(f"| {sn} | {k} | {h} | " + " | ".join((f"**×{st[wn]['mult']:,.1f}**" if o else f"×{st[wn]['mult']:,.1f}") + f" ({st[wn]['mdd']:+.0%})" for wn, o in zip(windows, ok)) + f" | {sum(ok)}/{len(windows)} | " + " · ".join(f"{x:.0%}" for x in pct) + f" | {turn:.0%} |")
    df = pd.DataFrame(res); nw = len(windows)
    L += ["", f"전 창 통과 {int((df['n_ok'] == nw).sum())} / {len(df)} · 한 창만 빠짐 {int((df['n_ok'] == nw - 1).sum())} / {len(df)}", "", "## 점수별 요약 (6 칸 중 통과 창 수 평균, 창별 플라세보 백분위 평균)", "", "| 점수 | 통과 창 평균 | " + " | ".join(windows) + " |", "|---|---|" + "---|" * nw]
    g = df.groupby("score").agg({"n_ok": "mean", **{f"{wn}_pct": "mean" for wn in windows}}).sort_values("n_ok", ascending=False)
    L += [f"| {i} | {r['n_ok']:.1f} | " + " | ".join(f"{r[f'{wn}_pct']:.0%}" for wn in windows) + " |" for i, r in g.iterrows()]
    Path(f"{out}.md").write_text("\n".join(L)); df.to_json(f"{out}.json", orient="records", force_ascii=False, indent=1); print("\n".join(L[:14])); print("\n".join(L[-(len(g) + 6):]))


if __name__ == "__main__":
    main(json.loads(sys.argv[1]) if len(sys.argv) > 1 else {})
