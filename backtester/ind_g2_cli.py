"""산업군 성장 탐색 G2 — G1 에서 남은 '산업 모멘텀' 이 손잡이·비용·지연·10 년 단위에 버티는가.

룩백 L(3·6·9·12·18·24 달) × 건너뛰기(0·1 달) × K(3·5·10) × 교체(1·3 달) × 편도 비용(0.2%·0.5%) × 추가 지연(0·1 달).
10 년 단위(1930s~2010s + 2020~2023)로 시장·49 산업 동일비중 대비 승패. 세전. 봉인(2024-01~)은 cfg end 명시 때만.
실행: docker compose exec backtester python ind_g2_cli.py '{}'
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import pandas as pd

from ind_g1_cli import WINDOWS, load
from strategy.adapters.industry_rank import _cum, port_returns, topk_weights, window_stats

SR = Path("/app/sweep_results")


def decades(end: str) -> dict:
    d = {f"{y}s": (f"{y}-01-31", f"{y + 9}-12-31") for y in range(1930, 2020, 10)}; d["2020~"] = ("2020-01-31", end); return d


def main(cfg):
    out = Path(cfg.get("out", SR / "ind_g2")); end = cfg.get("end", "2023-12-31"); vw, ew, n, size, bm, mk = load(end); rf = mk["rf"].reindex(vw.index).fillna(0.0)
    DEC = decades(end); W = {**WINDOWS, **DEC} if end <= "2023-12-31" else {**WINDOWS, "봉인 2024~": ("2024-01-31", end), **DEC}
    mkt = window_stats(mk["mkt"].reindex(vw.index).fillna(0.0), W); eqw = window_stats(vw.mean(axis=1), W); res = []
    for L_, skip, k, h, cost, lag in itertools.product((3, 6, 9, 12, 18, 24), (0, 1), (3, 5, 10), (1, 3), (0.002, 0.005), (0, 1)):
        if skip >= L_:
            continue
        w = topk_weights(_cum(vw, L_, skip).shift(lag), vw, k, h, min_n=n); st = window_stats(port_returns(w, vw, rf, cost), W)
        res.append({"L": L_, "skip": skip, "k": k, "hold": h, "cost": cost, "lag": lag, **{f"{wn}": st[wn]["mult"] for wn in W}, **{f"{wn}_mdd": st[wn]["mdd"] for wn in WINDOWS},
                    "win_main": sum(st[wn]["mult"] > mkt[wn]["mult"] for wn in WINDOWS), "dec_mkt": sum(st[wn]["mult"] > mkt[wn]["mult"] for wn in DEC), "dec_eqw": sum(st[wn]["mult"] > eqw[wn]["mult"] for wn in DEC), "turn": float(w.diff().abs().sum(axis=1).mean() * 6)})
    df = pd.DataFrame(res); nm = len(WINDOWS); nd = len(DEC)
    Ls = [f"# G2 — 산업 모멘텀 강건성 (개발 ~{end}, 세전)", "", f"칸 {len(df)} 개. '큰 창' = {' · '.join(WINDOWS)} 에서 시장 초과한 창 수. '10 년' = {nd} 개 10 년 단위 중 시장을 넘은 수.", "",
          "| 기준선 | " + " | ".join(W) + " |", "|---|" + "---|" * len(W), "| 시장 | " + " | ".join(f"×{mkt[wn]['mult']:.1f}" for wn in W) + " |", "| 49 산업 동일비중 | " + " | ".join(f"×{eqw[wn]['mult']:.1f}" for wn in W) + " |", ""]
    for name, by in (("비용·지연별", ["cost", "lag"]), ("룩백·건너뛰기별", ["L", "skip"]), ("K·교체 주기별", ["k", "hold"])):
        g = df.groupby(by).agg(cells=("L", "size"), all_main=("win_main", lambda s: (s == nm).mean()), dec_mkt=("dec_mkt", "mean"), dec_eqw=("dec_eqw", "mean"), **{wn: (wn, "median") for wn in WINDOWS}, turn=("turn", "mean"))
        Ls += [f"## {name}", "", "| " + " | ".join(by) + " | 칸 | 큰 창 전부 시장 초과 | 10 년 중 시장 초과(평균) | 10 년 중 동일비중 초과 | " + " | ".join(f"{wn} 중앙" for wn in WINDOWS) + " | 연 교체율 |", "|" + "---|" * (len(by) + 5 + nm)]
        Ls += ["| " + " | ".join(str(x) for x in (i if isinstance(i, tuple) else (i,))) + f" | {int(r['cells'])} | {r['all_main']:.0%} | {r['dec_mkt']:.1f}/{nd} | {r['dec_eqw']:.1f}/{nd} | " + " | ".join(f"×{r[wn]:.1f}" for wn in WINDOWS) + f" | {r['turn']:.0%} |" for i, r in g.iterrows()] + [""]
    hard = df[(df["cost"] == 0.005) & (df["lag"] == 1)]
    Ls += [f"가장 불리한 설정(편도 0.5%·한 달 더 늦게)에서 큰 창 전부 시장 초과 {int((hard['win_main'] == nm).sum())} / {len(hard)}", "", "## 대표 칸 (L12·skip1·K5·매달) 10 년 단위", "", "| 비용 | 지연 | " + " | ".join(DEC) + " |", "|---|---|" + "---|" * nd]
    rep = df[(df["L"] == 12) & (df["skip"] == 1) & (df["k"] == 5) & (df["hold"] == 1)]
    Ls += [f"| {r['cost']:.1%} | {int(r['lag'])} | " + " | ".join((f"**×{r[wn]:.1f}**" if r[wn] > mkt[wn]["mult"] else f"×{r[wn]:.1f}") for wn in DEC) + " |" for _, r in rep.iterrows()]
    if "봉인 2024~" in W:
        s = df[(df["cost"] == 0.002) & (df["lag"] == 0) & (df["hold"] == 1)]; Ls += ["", f"## 봉인 2024-01~{end}", "", f"시장 ×{mkt['봉인 2024~']['mult']:.2f} · 동일비중 ×{eqw['봉인 2024~']['mult']:.2f} · 매달 교체·기본 비용 칸 {len(s)} 개 중 시장 초과 {int((s['봉인 2024~'] > mkt['봉인 2024~']['mult']).sum())}, 중앙 ×{s['봉인 2024~'].median():.2f}"]
    Path(f"{out}.md").write_text("\n".join(Ls)); df.to_json(f"{out}.json", orient="records", force_ascii=False, indent=1); print("\n".join(Ls))


if __name__ == "__main__":
    main(json.loads(sys.argv[1]) if len(sys.argv) > 1 else {})
