"""산업군 성장 탐색 G6 — G5 에서 남은 '세분 산업(SIC 3 자리) 가격 모멘텀 → 그 산업 주식 전부 보유' 의 강건성.

월말마다 유동 종목(주가 ≥ $5·거래대금 ≥ $5M, SEC CIK·SIC 가 붙는 것)을 SIC 3 자리 산업으로 묶어, 산업 안 종목들의 지난 L 달 수익의 중앙값(또는 평균)으로 줄 세우고
상위 N 산업의 종목을 전부(산업 동일비중·종목 동일비중) hold 달 든다. 격자: L 3/6/9/12 × 건너뛰기 0/1 × 통계 × N 1/3/5/10 × hold 1/3 × 산업 최소 종목 수 3/5/8.
칸마다 편도 0.2%·0.5%, 한 달 늦게도 같이. 플라세보 = 같은 날 같은 수의 산업 무작위. 기준선 = 유니버스 동일비중·SPY·QQQ.
패널: main(2017~, 폐지 포함) 창 2018~2020·2021~2023 / long(1999~2017 근사 패널, SIC 는 2009 년 이후 제출 회사만) 창 2011~2014·2015~2017.
봉인(2024-01~)은 cfg end 명시 때만. 실행: docker compose run --rm --no-deps backtester python ind_g6_cli.py '{}'
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from strategy.adapters.industry_rank import port_returns, window_stats

SR = Path("/app/sweep_results")
PANELS = {"main": ("surge_panel.parquet", {"2018~2020": ("2018-02-28", "2020-12-31"), "2021~2023": ("2021-01-31", "2023-12-31")}), "long": ("surge_panel_long.parquet", {"2011~2014": ("2011-01-31", "2014-12-31"), "2015~2017": ("2015-01-31", "2017-11-30")})}


def monthly(path: str, start: str, end: str):
    cols = ["date", "symbol", "C", "raw_c", "dv"]; p = pd.read_parquet(SR / path, columns=cols); p = p[(p["date"] >= start) & (p["date"] <= end)]; p["m"] = p["date"] + pd.offsets.MonthEnd(0)
    g = p.sort_values("date").groupby(["m", "symbol"]); last = g[["C", "raw_c"]].last(); dv = g["dv"].median().unstack(); C = last["C"].unstack(); raw = last["raw_c"].unstack()
    return C, (raw >= 5) & (dv >= 5e6), dv


def sic_of(symbols) -> pd.Series:
    sm = pd.read_parquet(SR / "ind_sym_cik.parquet").set_index("symbol")["cik"]; fg = pd.read_parquet(SR / "ind_g4_filings.parquet", columns=["cik", "sic", "filed"]).dropna(subset=["sic"]).sort_values("filed").groupby("cik")["sic"].first()
    return sm.reindex(symbols).map(fg)


def main(cfg):
    out = Path(cfg.get("out", SR / "ind_g6")); end = cfg.get("end", "2023-12-31"); seeds = int(cfg.get("seeds", 100)); level = int(cfg.get("level", 3)); only = cfg.get("panels", list(PANELS))
    rfd = pd.read_parquet(SR / "irx_1985_2026-10-24.parquet")["irx"] / 252.0; res = []; L = [f"# G6 — 세분 산업(SIC {level} 자리) 모멘텀, 그 산업 주식 전부 보유 (개발 ~{end}, 세전)", ""]; notes = []
    for pn in only:
        path, W = PANELS[pn]; W = dict(W)
        if pn == "main" and end > "2023-12-31":
            W["봉인 2024~"] = ("2024-01-31", end)
        C, liq, dv = monthly(path, "2009-06-01" if pn == "long" else "2016-01-01", end if pn == "main" else "2017-12-31"); me = C.index; ret = C.pct_change(fill_method=None); nxt = ret.shift(-1).notna().to_numpy()
        sic = sic_of(C.columns); key = np.floor(sic.to_numpy(dtype=float) / 10 ** (4 - level)); has = ~np.isnan(key); elig = liq.to_numpy() & has[None, :] & nxt
        rf = ((1 + rfd).resample("ME").prod() - 1).reindex(me).ffill().fillna(0.0); etf = lambda s: pd.read_parquet(SR / "ind_etf_cache" / f"{s}.parquet")["adj"].resample("ME").last().pct_change().reindex(me).fillna(0.0)  # noqa: E731
        uw = elig / np.maximum(elig.sum(axis=1, keepdims=True), 1); base = {"유니버스 동일비중": window_stats(port_returns(pd.DataFrame(uw, index=me, columns=C.columns), ret, rf, 0.002), W), "SPY": window_stats(etf("SPY"), W), "QQQ": window_stats(etf("QQQ"), W)}; ref = base["유니버스 동일비중"]
        groups = [{k: np.flatnonzero(elig[i] & (key == k)) for k in np.unique(key[elig[i]])} for i in range(len(me))]          # 월별 산업 → 종목 열 위치
        mom = {(L_, s): (C.shift(s) / C.shift(L_) - 1.0).to_numpy() for L_ in (3, 6, 9, 12) for s in (0, 1)}

        def weights(score_fn, n_ind, hold, minstk, rng=None, log=None):
            w = np.zeros(C.shape); cj = cw = None
            for i in range(len(me)):
                if i % hold == 0:
                    g = {k: v for k, v in groups[i].items() if len(v) >= minstk}; cj = None
                    if rng is not None:
                        ks = list(rng.choice(list(g), n_ind, replace=False)) if len(g) >= n_ind else []
                    else:
                        sc = {k: score_fn(i, v) for k, v in g.items()}; sc = {k: x for k, x in sc.items() if np.isfinite(x)}; ks = sorted(sc, key=lambda k: -sc[k])[:n_ind] if len(sc) >= n_ind else []
                    if ks:
                        cj = np.concatenate([g[k] for k in ks]); cw = np.concatenate([np.full(len(g[k]), 1.0 / n_ind / len(g[k])) for k in ks])
                        if log is not None:
                            log.append((me[i], [(int(k), [C.columns[j] for j in g[k]]) for k in ks]))
                if cj is not None:
                    ok = nxt[i, cj]; np.add.at(w[i], cj[ok], cw[ok])
            return pd.DataFrame(w, index=me, columns=C.columns)

        def sfn(L_, s, stat, lag):
            m = mom[(L_, s)]
            def f(i, v):
                if i - lag < 0:
                    return np.nan
                x = m[i - lag, v]; x = x[np.isfinite(x)]
                return (np.median(x) if stat == "median" else np.mean(np.clip(x, None, 3.0))) if len(x) >= 2 else np.nan
            return f

        rng = np.random.default_rng(21); plc = {}
        for n_ind, hold, ms in itertools.product((1, 3, 5, 10), (1, 3), (3, 5, 8)):
            plc[(n_ind, hold, ms)] = np.array([[window_stats(port_returns(weights(None, n_ind, hold, ms, rng), ret, rf, 0.002), W)[wn]["mult"] for wn in W] for _ in range(seeds)]); print(pn, "placebo", n_ind, hold, ms, file=sys.stderr, flush=True)
        for L_, s, stat, n_ind, hold, ms in itertools.product((3, 6, 9, 12), (0, 1), ("median", "mean"), (1, 3, 5, 10), (1, 3), (3, 5, 8)):
            w = weights(sfn(L_, s, stat, 0), n_ind, hold, ms); st = window_stats(port_returns(w, ret, rf, 0.002), W); st5 = window_stats(port_returns(w, ret, rf, 0.005), W); stl = window_stats(port_returns(weights(sfn(L_, s, stat, 1), n_ind, hold, ms), ret, rf, 0.005), W)
            m = plc[(n_ind, hold, ms)]; row = {"panel": pn, "L": L_, "skip": s, "stat": stat, "n_ind": n_ind, "hold": hold, "minstk": ms, "turn": float(w.diff().abs().sum(axis=1).mean() * 6), "nstk": float((w > 0).sum(axis=1)[w.sum(axis=1) > 0].mean())}
            for i, wn in enumerate(W):
                row.update({f"{wn}": st[wn]["mult"], f"{wn}_mdd": st[wn]["mdd"], f"{wn}_c5": st5[wn]["mult"], f"{wn}_c5lag": stl[wn]["mult"], f"{wn}_pct": float((m[:, i] < st[wn]["mult"]).mean()), f"{wn}_ok": bool(st[wn]["mult"] > ref[wn]["mult"] and st[wn]["mult"] > np.percentile(m[:, i], 95)),
                            f"{wn}_ok_hard": bool(stl[wn]["mult"] > ref[wn]["mult"] and stl[wn]["mult"] > np.percentile(m[:, i], 95)), f"{wn}_qqq": bool(st[wn]["mult"] > base["QQQ"][wn]["mult"])})
            res.append(row)
        df = pd.DataFrame([r for r in res if r["panel"] == pn]); wn_ = list(W); dev = [w_ for w_ in wn_ if not w_.startswith("봉인")]; allok = df[[f"{w_}_ok" for w_ in dev]].all(axis=1); hard = df[[f"{w_}_ok_hard" for w_ in dev]].all(axis=1); qq = df[[f"{w_}_qqq" for w_ in dev]].all(axis=1)
        L += [f"## 패널 {pn} — 유니버스 종목 수 {int(elig[len(me) // 3].sum())}~{int(elig[-2].sum())}, 산업 수(종목 ≥3) {sum(len(v) >= 3 for v in groups[len(me) // 2].values())}", "", "| 기준선 | " + " | ".join(W) + " |", "|---|" + "---|" * len(W)]
        L += [f"| {k} | " + " | ".join(f"×{x['mult']:.2f} ({x['mdd']:+.0%})" for x in v.values()) + " |" for k, v in base.items()]
        L += ["", f"칸 {len(df)} 개 · 개발 창 전부 통과(유니버스·무작위 95 초과) **{int(allok.sum())}** · 편도 0.5%+한 달 지연에서도 통과 **{int(hard.sum())}** · 개발 창 전부 QQQ 초과 **{int(qq.sum())}**", ""]
        for by in ("L", "skip", "stat", "n_ind", "hold", "minstk"):
            g = df.assign(ok=allok, hard=hard, qq=qq).groupby(by); L += [f"| {by} | 칸 | 통과 | 불리한 설정 통과 | QQQ 초과 | " + " | ".join(f"{w_} 중앙" for w_ in wn_) + " | " + " | ".join(f"{w_} 백분위" for w_ in wn_) + " | 평균 종목 수 | 연 교체율 |", "|---|---|---|---|---|" + "---|" * (2 * len(wn_)) + "---|---|"]
            L += [f"| {k} | {len(x)} | {x['ok'].mean():.0%} | {x['hard'].mean():.0%} | {x['qq'].mean():.0%} | " + " | ".join(f"×{x[w_].median():.2f}" for w_ in wn_) + " | " + " | ".join(f"{x[f'{w_}_pct'].mean():.0%}" for w_ in wn_) + f" | {x['nstk'].mean():.0f} | {x['turn'].mean():.0%} |" for k, x in g] + [""]
        # 대표 칸: 무엇을 들었나, 해마다 어땠나
        log = []; w = weights(sfn(6, 0, "median", 0), 3, 1, 5, log=log); r = port_returns(w, ret, rf, 0.002); ur = port_returns(pd.DataFrame(uw, index=me, columns=C.columns), ret, rf, 0.002); q = etf("QQQ"); yr = lambda x: (1 + x).groupby(x.index.year).prod() - 1  # noqa: E731
        a0 = min(v[0] for v in W.values()); y = pd.DataFrame({"전략": yr(r[r.index >= a0]), "유니버스": yr(ur[ur.index >= a0]), "QQQ": yr(q[q.index >= a0])})
        L += ["### 대표 칸(L6·중앙값·3 산업·매달·종목 ≥5) 연도별", "", "| 연도 | 전략 | 유니버스 동일비중 | QQQ |", "|---|---|---|---|"] + [f"| {i} | {v['전략']:+.0%} | {v['유니버스']:+.0%} | {v['QQQ']:+.0%} |" for i, v in y.iterrows()]
        contrib = (w.shift(1).fillna(0.0) * ret.fillna(0.0)); top = contrib[contrib.index >= a0].sum().nlargest(8); L += ["", "기여 상위 종목(월 기여 합): " + ", ".join(f"{k} {v:+.0%}" for k, v in top.items()), "", "| 월 | 고른 산업(SIC) : 종목 |", "|---|---|"]
        L += [f"| {t:%Y-%m} | " + " / ".join(f"{k}: {' '.join(s[:8])}{'…' if len(s) > 8 else ''}" for k, s in ks) + " |" for t, ks in log if t >= pd.Timestamp(a0) and t.month in (6, 12)] + [""]
    Path(f"{out}.md").write_text("\n".join(L)); pd.DataFrame(res).to_json(f"{out}.json", orient="records", force_ascii=False, indent=1); print("\n".join(L))


if __name__ == "__main__":
    main(json.loads(sys.argv[1]) if len(sys.argv) > 1 else {})
