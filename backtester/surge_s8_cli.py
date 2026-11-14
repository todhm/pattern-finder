"""급등주 탐색 S8 — S7 수동 체크리스트의 기계 판을 개발 구간 전체에 적용.

S7 은 12 날짜·60 종목이라 결론을 못 냈다. 같은 항목(M1~M4·X1~X5)을 종가·거래대금 규칙으로 옮겨
 A) 항목 연구: 7 거래일마다 후보 풀(소거 4 개 통과·12-1 모멘텀 상위 n)에서 항목 참/거짓의 앞 63일 풀 내 초과 수익
 B) 계단식 운용: 풀 안에서 체크리스트 점수 상위 K vs 모멘텀 상위 K vs 풀 무작위 K vs TQQQ

실행: docker compose exec backtester python surge_s8_cli.py '{"out": "/app/sweep_results/surge_s8"}'
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from strategy.adapters.surge_checklist import AVOID, BUY, LABELS, checklist_features, checklist_score
from strategy.adapters.surge_events import load_symbol
from strategy.adapters.surge_rank_portfolio import RankRules, curve_stats, run_rank, scores, tranche_curve, wide

SR = Path("/app/sweep_results")
p = lambda x: "-" if pd.isna(x) else f"{x:+.1%}"  # noqa: E731


def pool_mask(w, sc, up, thr: float, n: int) -> pd.DataFrame:
    """S7 과 같은 후보 풀: 주가 $10·거래대금 $50M·20일 +50% 이하·오른 날 ≥ thr 중 12-1 모멘텀 상위 n."""
    c = w["C"]; dv20 = w["dv"].rolling(20, min_periods=10).mean()
    ok = w["alive"] & (w["raw_c"] >= 10.0) & (dv20 >= 5e7) & ((c / c.shift(20) - 1.0) <= 0.5) & (up >= thr)
    return sc.where(ok).rank(axis=1, ascending=False, method="first") <= n


def feature_study(w, pool, feats, score, qq_below: pd.Series, start: str, end: str, hold: int, step: int, mid: pd.Timestamp):
    idx = w["C"].index; last = np.flatnonzero(idx <= end)[-1]
    pos = [s for s in np.flatnonzero((idx >= start) & (idx <= end))[::step] if s + 1 + hold <= last]
    o = w["O_ff"]; rows = []
    for s in pos:
        syms = pool.columns[pool.iloc[s].to_numpy()]
        if len(syms) < 6:
            continue
        fwd = o.iloc[s + 1 + hold][syms] / o.iloc[s + 1][syms] - 1.0
        d = pd.DataFrame({"date": idx[s], "symbol": syms, "fwd": fwd.to_numpy(), "ex": (fwd - fwd.mean()).to_numpy(), "score": score.iloc[s][syms].to_numpy(),
                          "bear": bool(qq_below.iloc[s])})
        for k, f in feats.items():
            d[k] = f.iloc[s][syms].to_numpy()
        rows.append(d)
    ev = pd.concat(rows, ignore_index=True)
    n_eff = max(ev["date"].nunique() * step / hold, 2.0)       # 겹치는 63일 창 → 독립 표본 수 근사

    def diff_by_date(mask_col):
        g = ev.groupby(["date", mask_col])["ex"].mean().unstack()
        return (g[True] - g[False]).dropna() if True in g and False in g else pd.Series(dtype=float)

    out = []
    for k in list(feats) + ["score_hi", "score_lo"]:
        if k == "score_hi":
            ev[k] = ev["score"] >= ev.groupby("date")["score"].transform(lambda x: x.nlargest(5).min())
        elif k == "score_lo":
            ev[k] = ev["score"] <= ev.groupby("date")["score"].transform(lambda x: x.nsmallest(5).max())
        dd = diff_by_date(k); bear = ev.drop_duplicates("date").set_index("date")["bear"].reindex(dd.index)
        out.append({"item": k, "share": float(ev[k].mean()), "ex_true": float(ev.loc[ev[k], "ex"].mean()) if ev[k].any() else np.nan, "fwd_true": float(ev.loc[ev[k], "fwd"].mean()) if ev[k].any() else np.nan,
                    "diff": float(dd.mean()) if len(dd) else np.nan, "t": float(dd.mean() / (dd.std() / np.sqrt(n_eff))) if len(dd) > 2 else np.nan, "pos": float((dd > 0).mean()) if len(dd) else np.nan,
                    "h1": float(dd[dd.index <= mid].mean()), "h2": float(dd[dd.index > mid].mean()), "bear": float(dd[bear].mean()), "bull": float(dd[~bear].mean()), "n_dates": int(len(dd))})
    return ev, pd.DataFrame(out), n_eff


def main(cfg):
    out = Path(cfg.get("out", SR / "surge_s8")); start = cfg.get("start", "2018-02-01"); end = cfg.get("end", "2023-12-31"); mid = pd.Timestamp(cfg.get("mid", "2020-12-31"))
    n_tr = int(cfg.get("tranches", 9)); seeds = int(cfg.get("seeds", 10)); every = int(cfg.get("every", 63)); thr = float(cfg.get("thr", 0.51))
    pools = cfg.get("pools", [12, 16, 24]); ks = cfg.get("ks", [3, 5, 8]); main_cell = (16, 5)
    panel = pd.read_parquet(SR / "surge_panel.parquet", columns=["date", "symbol", "C", "O", "raw_c", "dv"])
    w = wide(panel); del panel
    c = w["C"]; r = c.pct_change(fill_method=None)
    up = (r > 0).where(r.notna()).rolling(252, min_periods=200).mean(); sc = scores(w, "mom12_1")
    masks = {n: pool_mask(w, sc, up, thr, n) for n in pools}
    keep = masks[max(pools)].loc[:end].any()                       # 한 번이라도 풀에 든 종목만 남겨 계산을 줄인다
    w = {k: v.loc[:, keep] for k, v in w.items()}; sc = sc.loc[:, keep]; up = up.loc[:, keep]; masks = {n: m.loc[:, keep] for n, m in masks.items()}
    c = w["C"]; days = c.index[(c.index >= start) & (c.index <= end)]
    feats = checklist_features(c, w["dv"]); chk = checklist_score(feats)
    tq = load_symbol(SR / "surge_av_cache" / "TQQQ.parquet"); qq = load_symbol(SR / "surge_av_cache" / "QQQ.parquet")
    qc = qq["C"].reindex(c.index).ffill(); qq_below = qc < qc.rolling(200, min_periods=150).mean()
    chk_regime = chk.where(~qq_below, checklist_score(feats, skip=("x3", "m1")), axis=0)   # 시장이 200일선 아래면 꺾임·추세 항목을 세지 않음(S7 사후 가설)

    # ── A) 항목 연구
    ev, ft, n_eff = feature_study(w, masks[16], feats, chk, qq_below, start, end, every, 7, mid)
    L = [f"# S8 — 체크리스트 기계 판 ({start}~{end})", "",
         "S7 의 사기(M1~M4)·피하기(X1~X5) 항목을 종가·거래대금 규칙으로 옮겨 개발 구간 전체에 적용. 후보 풀 = 주가 $10·거래대금 $50M·20일 +50% 이하·오른 날 ≥ 51% 중 12-1 모멘텀 상위 16.",
         "", "## 미리 정한 읽는 법", "",
         "- 항목에 정보가 있다 = 날짜별 (참 − 거짓) 풀 내 초과 수익 평균이 0 과 다르고(|t| ≥ 2, 겹침 보정) 전반·후반 부호가 같다.",
         "- 체크리스트가 보탬 = 계단식 운용에서 점수 상위 K 가 모멘텀 상위 K 와 풀 무작위 K(95% 이상)를 모두 이기고, 이웃 칸(풀 12/16/24 × K 3/5/8) 9 개 중 6 개 이상에서 같다.",
         "- TQQQ 초과 = 계획서 관문(배수 > TQQQ, 전·후반 각각 0.7 배 이상, 무작위 95%) + 이웃 칸 과반. 통과해야 봉인 구간을 연다.",
         "", f"## A. 항목별 앞 {every} 거래일 풀 내 초과 수익", "",
         f"7 거래일마다 {ev['date'].nunique()} 날짜 · {len(ev):,} 종목-날짜. 창이 겹치므로 t 는 독립 표본 약 {n_eff:.0f} 개로 계산. 풀 평균 앞 수익 {p(ev['fwd'].mean())}.", "",
         "| 항목 | 해당 비율 | 참일 때 초과 | 참 − 거짓 | t | 참이 이긴 날짜 | 전반 | 후반 | 시장 200일선 아래 | 위 |", "|---|---|---|---|---|---|---|---|---|---|"]
    names = {**LABELS, "score_hi": "점수 상위 5", "score_lo": "점수 하위 5"}
    for x in ft.itertuples():
        L.append(f"| {names[x.item]} | {x.share:.0%} | {p(x.ex_true)} | {p(x.diff)} | {x.t:+.1f} | {x.pos:.0%} | {p(x.h1)} | {p(x.h2)} | {p(x.bear)} | {p(x.bull)} |")
    g = ev.groupby("score").agg(n=("ex", "size"), ex=("ex", "mean"), fwd=("fwd", "mean"), win=("ex", lambda v: float((v > 0).mean())))
    L += ["", "점수(사기 항목 수 − 피하기 항목 수)별", "", "| 점수 | 표본 | 평균 초과 | 평균 앞 수익 | 풀 평균을 이긴 비율 |", "|---|---|---|---|---|"]
    L += [f"| {int(s):+d} | {int(x.n):,} | {p(x.ex)} | {p(x.fwd)} | {x.win:.0%} |" for s, x in g.iterrows()]
    print("\n".join(L), file=sys.stderr, flush=True)

    # ── B) 계단식 운용
    offs = [int(round(i * every / n_tr)) for i in range(n_tr)]

    def bench(f):
        e = f["C"].reindex(days).ffill() / f["O"].reindex(days).bfill().iloc[1]; e.iloc[0] = 1.0
        return e

    def halves(eq):
        m = eq[eq.index <= mid].iloc[-1]
        return float(m / eq.iloc[0]), float(eq.iloc[-1] / m)

    def run(k, score, ok):
        rules = RankRules(k=k, min_price=10.0, min_dv20=5e7, max_ret20=0.5, every=every)
        eq = tranche_curve(w, [run_rank(w, rules, str(days[o].date()), end, score=score, extra_ok=ok) for o in offs], days)
        return eq, {**curve_stats(eq), "h1": halves(eq)[0], "h2": halves(eq)[1]}

    def rnd(k, ok):
        rr = RankRules(rank="random", k=k, min_price=10.0, min_dv20=5e7, max_ret20=0.5, every=every)
        # 전략과 같은 자(계단식 곡선을 end 에서 자른 배수)로 잰다. 구간 수익의 곱은 마지막 구간이 end 뒤로 최대 63일 넘어간다.
        return np.array([curve_stats(tranche_curve(w, [run_rank(w, rr, str(days[o].date()), end, rng=np.random.default_rng(1000 * sd + o), extra_ok=ok) for o in offs], days))["mult"] for sd in range(seeds)])

    bt, bq = bench(tq), bench(qq); st, sq = curve_stats(bt), curve_stats(bq); ht = halves(bt)
    pm = lambda x: f"{x:+.0%}"  # noqa: E731
    L += ["", f"## B. 계단식 {n_tr} 분할 운용 ({every} 거래일 교체, 교체분 비용 0.5%, 세전)", "",
          f"**TQQQ** ×{st['mult']:.2f} (CAGR {pm(st['cagr'])}, MDD {pm(st['mdd'])}, 전반 ×{ht[0]:.2f}·후반 ×{ht[1]:.2f}) · **QQQ** ×{sq['mult']:.2f} (MDD {pm(sq['mdd'])})", "",
          "| 풀 | K | 고르는 법 | 배수 | CAGR | MDD | 전반 | 후반 | TQQQ 대비 | 전/후반 대비 | 풀 무작위 K 중앙 | 무작위 이긴% |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    res, curves = [], {"TQQQ": bt, "QQQ": bq}
    for n in pools:
        pool = masks[n]; pct = sc.where(pool).rank(axis=1, pct=True).fillna(0.0) * 0.9      # 동점은 모멘텀 순
        variants = {"모멘텀 상위": (sc, pool), "체크리스트 상위": (chk + pct, pool), "체크리스트 하위": (-(chk - pct), pool),
                    "피하기 0 개 중 모멘텀 상위": (sc, pool & (sum(feats[x].astype(int) for x in AVOID) == 0)),
                    "사기 3 개 이상 중 모멘텀 상위": (sc, pool & (sum(feats[x].astype(int) for x in BUY) >= 3)),
                    "레짐 판(약세장엔 꺾임 무시) 상위": (chk_regime + pct, pool)}
        for k in ks:
            rm = rnd(k, pool)
            for name, (score, ok) in variants.items():
                eq, s = run(k, score, ok)
                row = {"pool": n, "k": k, "pick": name, **s, "vs": s["mult"] / st["mult"], "vs1": s["h1"] / ht[0], "vs2": s["h2"] / ht[1], "rnd_med": float(np.median(rm)), "rnd_beat": float((s["mult"] > rm).mean())}
                res.append(row); curves[f"{n}_{k}_{name}"] = eq
                L.append(f"| {n} | {k} | {name} | ×{row['mult']:.2f} | {pm(row['cagr'])} | {pm(row['mdd'])} | ×{row['h1']:.2f} | ×{row['h2']:.2f} | {row['vs']:.2f} | {row['vs1']:.2f} / {row['vs2']:.2f} | ×{row['rnd_med']:.2f} | {row['rnd_beat']:.0%} |")
                print(L[-1], file=sys.stderr, flush=True)
        eq, s = run(n, sc, pool)
        res.append({"pool": n, "k": n, "pick": "풀 전부", **s, "vs": s["mult"] / st["mult"], "vs1": s["h1"] / ht[0], "vs2": s["h2"] / ht[1], "rnd_med": np.nan, "rnd_beat": np.nan})
        L.append(f"| {n} | {n} | 풀 전부 | ×{s['mult']:.2f} | {pm(s['cagr'])} | {pm(s['mdd'])} | ×{s['h1']:.2f} | ×{s['h2']:.2f} | {s['mult'] / st['mult']:.2f} | {s['h1'] / ht[0]:.2f} / {s['h2'] / ht[1]:.2f} | - | - |")
    df = pd.DataFrame(res); piv = df[df["pick"] != "풀 전부"].pivot(index=["pool", "k"], columns="pick", values="mult"); rb = df.pivot(index=["pool", "k"], columns="pick", values="rnd_beat")
    L += ["", "## 판정", ""]
    for name in [v for v in piv.columns if v != "모멘텀 상위"]:
        help_ = (piv[name] > piv["모멘텀 상위"]) & (rb[name].reindex(piv.index) >= 0.95)
        gate = df[(df["pick"] == name) & (df["vs"] > 1) & (df["vs1"] >= 0.7) & (df["vs2"] >= 0.7) & (df["rnd_beat"] >= 0.95)]
        L.append(f"- {name}: 모멘텀 상위·풀 무작위를 모두 이긴 칸 {int(help_.sum())}/{len(piv)} (기준 칸 풀 {main_cell[0]}·K {main_cell[1]}: {'예' if help_.get(main_cell, False) else '아니오'}) · TQQQ 관문 통과 칸 {len(gate)}/{len(piv)}")
    Path(f"{out}.md").write_text("\n".join(L)); df.to_json(f"{out}.json", orient="records", force_ascii=False, indent=1)
    ft.to_json(f"{out}_items.json", orient="records", force_ascii=False, indent=1); ev.to_parquet(f"{out}_events.parquet"); pd.DataFrame(curves).to_parquet(f"{out}_curves.parquet")
    print("\n".join(L))


if __name__ == "__main__":
    main(json.loads(sys.argv[1]) if len(sys.argv) > 1 else {})
