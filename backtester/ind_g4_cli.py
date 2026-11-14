"""산업군 성장 탐색 G4 — '실제로 성장 중인 산업' 을 SEC 제출 숫자로 골라 들면 시장·무작위를 이기는가 (2012-10~, 월간).

 build : sec_fs_cache → 제출물별 전년비(filing_growth) → 산업(French 49) 월말 지표 표 `ind_g4_panel.parquet`
 run   : 지표 × K(3·5·10) × 교체(1·3·12 달). 수익은 French 49 산업 시총가중(폐지 포함). 기준선 시장·동일비중, 플라세보 무작위 200 벌.
지표: g_agg(산업 합계 매출 전년비) · g_med(회사 중앙값) · hot_frac(25% 넘게 크는 회사 비율) · accel(g_agg − 6 달 전) · rnd_g · capex_g · margin_chg · filers_g(제출 회사 수 증가)
     + 반대 방향(−) + 가격 모멘텀 결합. 창: 2012-10~2017 · 2018~2023. 봉인(2024-01~)은 cfg end 명시 때만.
실행: docker compose exec backtester python ind_g4_cli.py build '{}' ; docker compose exec backtester python ind_g4_cli.py run '{}'
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ind_g1_cli import load
from strategy.adapters.industry_fundamentals import filing_growth, industry_panel, sic_to_ind
from strategy.adapters.industry_rank import _cum, port_returns, topk_weights, window_stats

SR = Path("/app/sweep_results")
WINDOWS = {"2012-10~2017": ("2012-10-31", "2017-12-31"), "2018~2023": ("2018-01-31", "2023-12-31")}


def build(cfg):
    parts = [filing_growth(pd.read_parquet(p)) for p in sorted((SR / "sec_fs_cache").glob("*.parquet")) if len(pd.read_parquet(p))]
    fg = pd.concat(parts, ignore_index=True); fg["ind"] = sic_to_ind(fg["sic"], pd.read_parquet(SR / "ind_sic49.parquet")); fg.to_parquet(SR / "ind_g4_filings.parquet")
    me = pd.date_range("2011-06-30", fg["filed"].max(), freq="ME"); p = industry_panel(fg, me)
    pd.concat({k: v.stack(future_stack=True) for k, v in p.items()}, axis=1).reset_index(names=["date", "ind"]).to_parquet(SR / "ind_g4_panel.parquet")
    print(len(fg), fg["cik"].nunique(), fg["ind"].value_counts().head(8).to_dict()); print(p["g_agg"].loc["2015-12-31":"2016-03-31"].round(2).T.head(20).to_string()); print(p["filers"].iloc[[6, 18, 60, -1]].T.describe().round(0).to_string())


def scores(end: str) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    vw, ew, n, size, bm, mk = load(end); d = pd.read_parquet(SR / "ind_g4_panel.parquet"); d = d[d["date"] <= end]; P = lambda c: d.pivot(index="date", columns="ind", values=c).reindex(index=vw.index, columns=vw.columns)  # noqa: E731
    g, gm, hot, shr, fl, rnd, cap, mar = (P(c) for c in ("g_agg", "g_med", "hot_frac", "shrink_frac", "filers", "rnd_g", "capex_g", "margin_chg")); r = lambda x: x.rank(axis=1, pct=True)  # noqa: E731
    mom = _cum(vw, 12, 1); S = {"g_agg": g, "g_med": gm, "hot_frac": hot, "few_shrink": -shr, "accel6": g - g.shift(6), "accel12": g - g.shift(12), "med_accel6": gm - gm.shift(6), "rnd_g": rnd, "capex_g": cap, "margin_chg": mar,
                                "filers_g": fl / fl.shift(12) - 1.0, "g_minus_capex": r(g) - r(cap), "g+accel": (r(g) + r(g - g.shift(6))) / 2, "g+margin": (r(g) + r(mar)) / 2, "g+mom": (r(g) + r(mom)) / 2, "accel+mom": (r(g - g.shift(6)) + r(mom)) / 2,
                                "g+accel+mom": (r(g) + r(g - g.shift(6)) + r(mom)) / 3, "mom12_1(대조)": mom.where(g.notna())}
    S.update({f"−{k}": -S[k] for k in ("g_agg", "g_med", "accel6", "capex_g", "hot_frac")})                  # 반대 방향: 쪼그라드는 산업
    return S, vw, n, mk["rf"].reindex(vw.index).fillna(0.0), mk["mkt"].reindex(vw.index).fillna(0.0)


def run(cfg):
    out = Path(cfg.get("out", SR / "ind_g4")); end = cfg.get("end", "2023-12-31"); seeds = int(cfg.get("seeds", 200)); S, vw, n, rf, mkt_r = scores(end)
    W = dict(WINDOWS) if end <= "2023-12-31" else {**WINDOWS, "봉인 2024~": ("2024-01-31", end)}; live = S["g_agg"].notna()
    base = {"시장": window_stats(mkt_r, W), "49 산업 동일비중": window_stats(vw.mean(axis=1), W)}; mkt = base["시장"]; rng = np.random.default_rng(3); ones = live.astype(float).where(live); plc = {}
    for k, h in itertools.product((3, 5, 10), (1, 3, 12)):
        plc[(k, h)] = np.array([[window_stats(port_returns(topk_weights(ones, vw, k, h, rng=rng, min_n=n), vw, rf), W)[wn]["mult"] for wn in W] for _ in range(seeds)])
    L = [f"# G4 — SEC 제출 숫자로 고른 '성장 산업' 보유 (개발 ~{end}, 세전, 편도 0.2%)", "", "값 = 배수(MDD). 굵게 = 그 창에서 시장과 무작위 95 백분위를 둘 다 넘음.", "", "| 기준선 | " + " | ".join(W) + " |", "|---|" + "---|" * len(W)]
    L += [f"| {k} | " + " | ".join(f"×{x['mult']:.2f} ({x['mdd']:+.0%})" for x in v.values()) + " |" for k, v in base.items()] + [f"| 무작위 {k} 개·{h} 달 (중앙 / 95) | " + " | ".join(f"×{np.median(m[:, i]):.2f} / ×{np.percentile(m[:, i], 95):.2f}" for i in range(len(W))) + " |" for (k, h), m in plc.items()]
    L += ["", "| 지표 | K | 교체 | " + " | ".join(W) + " | 통과 창 | 플라세보 백분위 |", "|---|---|---|" + "---|" * len(W) + "---|---|"]; res = []
    for (sn, sc), k, h in itertools.product(S.items(), (3, 5, 10), (1, 3, 12)):
        w = topk_weights(sc, vw, k, h, min_n=n); st = window_stats(port_returns(w, vw, rf), W); m = plc[(k, h)]
        ok = [st[wn]["mult"] > mkt[wn]["mult"] and st[wn]["mult"] > np.percentile(m[:, i], 95) for i, wn in enumerate(WINDOWS)]; pct = [float((m[:, i] < st[wn]["mult"]).mean()) for i, wn in enumerate(W)]
        res.append({"score": sn, "k": k, "hold": h, **{wn: st[wn]["mult"] for wn in W}, **{f"{wn}_mdd": st[wn]["mdd"] for wn in W}, **{f"{wn}_pct": p for wn, p in zip(W, pct)}, "n_ok": sum(ok), "beat_mkt": sum(st[wn]["mult"] > mkt[wn]["mult"] for wn in WINDOWS)})
        L.append(f"| {sn} | {k} | {h} | " + " | ".join((f"**×{st[wn]['mult']:.2f}**" if (i < len(ok) and ok[i]) else f"×{st[wn]['mult']:.2f}") + f" ({st[wn]['mdd']:+.0%})" for i, wn in enumerate(W)) + f" | {sum(ok)}/{len(WINDOWS)} | " + " · ".join(f"{x:.0%}" for x in pct) + " |")
    df = pd.DataFrame(res); nw = len(WINDOWS); g = df.groupby("score").agg(ok=("n_ok", lambda s: int((s == nw).sum())), bm=("beat_mkt", lambda s: int((s == nw).sum())), **{wn: (wn, "median") for wn in W}, **{f"{wn}_pct": (f"{wn}_pct", "mean") for wn in W}).sort_values(f"{list(WINDOWS)[0]}_pct", ascending=False)
    L += ["", f"전 창 통과 {int((df['n_ok'] == nw).sum())} / {len(df)}", "", "## 지표별 요약 (9 칸)", "", "| 지표 | 전 창 통과 | 전 창 시장 초과 | " + " | ".join(f"{wn} 중앙" for wn in W) + " | " + " | ".join(f"{wn} 백분위" for wn in W) + " |", "|---|---|---|" + "---|" * (2 * len(W))]
    L += [f"| {i} | {r['ok']}/9 | {r['bm']}/9 | " + " | ".join(f"×{r[wn]:.2f}" for wn in W) + " | " + " | ".join(f"{r[f'{wn}_pct']:.0%}" for wn in W) + " |" for i, r in g.iterrows()]
    Path(f"{out}.md").write_text("\n".join(L)); df.to_json(f"{out}.json", orient="records", force_ascii=False, indent=1); print("\n".join(L[:16])); print("\n".join(L[-(len(g) + 6):]))


if __name__ == "__main__":
    {"build": build, "run": run}[sys.argv[1]](json.loads(sys.argv[2]) if len(sys.argv) > 2 else {})
