"""산업군 성장 탐색 G5 — '성장 산업을 고르고 그 산업의 주식을 산다' 를 종목 단위로 (폐지 종목이 든 패널 2017~, SEC 제출 숫자).

 map : 종목 ↔ SEC CIK (내부자 거래 자료의 발행사 CIK + SEC company_tickers.json) → sweep_results/ind_sym_cik.parquet
 run : 월말마다 ① 산업(분류 수준 ff49 | sic2 | sic3)을 지표로 줄 세워 상위 N 개 ② 그 안에서 종목 고르기(all | 자기 매출 성장 상위 5 | 12 달 모멘텀 상위 5 | 거래대금 상위 5)
       ③ 산업 동일비중·종목 동일비중, 1·3 달 보유. 플라세보 = 같은 날 같은 수의 산업을 무작위(종목 고르기는 같게). 기준선 = 유니버스 동일비중·SPY·QQQ.
       유니버스: CIK 가 붙고 주가 ≥ $5·20일 거래대금 ≥ $5M. 폐지 종목은 마지막 거래가까지 반영. 세전, 편도 0.2%. 봉인(2024-01~)은 cfg end 명시 때만.
실행: docker compose exec backtester python ind_g5_cli.py map ; docker compose exec backtester python ind_g5_cli.py run '{}'
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from strategy.adapters.industry_fundamentals import sic_to_ind
from strategy.adapters.industry_rank import port_returns, window_stats

SR = Path("/app/sweep_results")
WINDOWS = {"2018~2020": ("2018-02-28", "2020-12-31"), "2021~2023": ("2021-01-31", "2023-12-31")}
UA = {"User-Agent": "research-script/1.0 (academic backtest)"}


def build_map(cfg):
    import requests
    t = pd.read_parquet(SR / "surge_insider_tx.parquet", columns=["symbol", "issuer_cik"]).rename(columns={"issuer_cik": "cik"})
    j = pd.DataFrame(requests.get("https://www.sec.gov/files/company_tickers.json", headers=UA, timeout=60).json()).T.rename(columns={"ticker": "symbol", "cik_str": "cik"})[["symbol", "cik"]]
    m = pd.concat([t, j, j]); m["symbol"] = m["symbol"].astype(str).str.upper().str.strip(); m["cik"] = m["cik"].astype(int)
    m = m.groupby(["symbol", "cik"]).size().reset_index(name="n").sort_values("n").groupby("symbol").tail(1)[["symbol", "cik"]]; m.to_parquet(SR / "ind_sym_cik.parquet"); print(len(m))


def firm_table(fg: pd.DataFrame, month_ends, fresh_days=190) -> pd.DataFrame:
    """월말별 회사 최신 제출물(그날까지 제출된 것만): date, cik, sic, g(매출 전년비), rev_cur, rev_prev."""
    fg = fg.sort_values("filed"); out = []
    for t in month_ends:
        x = fg[(fg["filed"] <= t) & (fg["filed"] > t - pd.Timedelta(days=fresh_days))].groupby("cik").tail(1)[["cik", "sic", "rev_cur", "rev_prev"]].copy(); x["date"] = t; out.append(x)
    d = pd.concat(out); d["g"] = np.where(d["rev_prev"] > 0, d["rev_cur"] / d["rev_prev"] - 1.0, np.nan); return d


def ind_scores(ft: pd.DataFrame, key: str) -> dict[str, pd.DataFrame]:
    v = ft[ft["g"].notna()]; grp = v.groupby(["date", key]); cnt = grp["g"].size().unstack(); ok = cnt >= 5
    g = (grp["rev_cur"].sum() / grp["rev_prev"].sum() - 1.0).unstack().where(ok); med = grp["g"].median().unstack().where(ok); hot = grp["g"].apply(lambda s: (s > 0.25).mean()).unstack().where(ok)
    return {"g_agg": g, "g_med": med, "hot_frac": hot, "accel6": g - g.shift(6), "med_accel6": med - med.shift(6), "−g_med": -med}


def main(cfg):
    out = Path(cfg.get("out", SR / "ind_g5")); end = cfg.get("end", "2023-12-31"); seeds = int(cfg.get("seeds", 100)); cost = 0.002
    p = pd.read_parquet(SR / "surge_panel.parquet", columns=["date", "symbol", "C", "p_raw_c", "p_dv20"]); p = p[p["date"] <= end]; p["m"] = p["date"] + pd.offsets.MonthEnd(0)
    last = p.sort_values("date").groupby(["m", "symbol"]).tail(1); C = last.pivot(index="m", columns="symbol", values="C"); liq = (last.pivot(index="m", columns="symbol", values="p_raw_c") >= 5) & (last.pivot(index="m", columns="symbol", values="p_dv20") >= 5e6)
    dv = last.pivot(index="m", columns="symbol", values="p_dv20"); ret = C.pct_change(fill_method=None); mom12 = C / C.shift(12) - 1.0; mom6 = C / C.shift(6) - 1.0; me = C.index
    sm = pd.read_parquet(SR / "ind_sym_cik.parquet").set_index("symbol")["cik"]; cols = [c for c in C.columns if c in sm.index]; cik_of = sm.reindex(cols)
    fg = pd.read_parquet(SR / "ind_g4_filings.parquet"); fg = fg[fg["filed"] <= end]; ft = firm_table(fg, me); ft["ff49"] = sic_to_ind(ft["sic"], pd.read_parquet(SR / "ind_sic49.parquet")).to_numpy(); ft["sic2"] = ft["sic"] // 100; ft["sic3"] = ft["sic"] // 10
    rf = ((1 + pd.read_parquet(SR / "irx_1985_2026-10-24.parquet")["irx"] / 252.0).resample("ME").prod() - 1).reindex(me).ffill().fillna(0.0)
    etf = lambda s: pd.read_parquet(SR / "ind_etf_cache" / f"{s}.parquet")["adj"].resample("ME").last().pct_change().reindex(me).fillna(0.0)  # noqa: E731
    W = dict(WINDOWS) if end <= "2023-12-31" else {**WINDOWS, "봉인 2024~": ("2024-01-31", end)}
    # 월말별 종목 표(유니버스): symbol → cik → 산업 키·자기 성장
    per = {}; nxt_ok = ret.shift(-1).notna().to_numpy(); pos = {s: j for j, s in enumerate(C.columns)}; cj = np.array([pos[s] for s in cols]); liq_a = liq.to_numpy(); ti = {t: i for i, t in enumerate(me)}
    for i, t in enumerate(me):
        x = ft[ft["date"] == t].drop_duplicates("cik").set_index("cik"); ok = liq_a[i, cj] & nxt_ok[i, cj] & cik_of.isin(x.index).to_numpy(); u = [cols[k] for k in np.flatnonzero(ok)]
        if u:
            xi = x.loc[cik_of[u].to_numpy()]; per[t] = pd.DataFrame({"symbol": u, "j": cj[ok], "ff49": xi["ff49"].to_numpy(), "sic2": xi["sic2"].to_numpy(), "sic3": xi["sic3"].to_numpy(), "g": xi["g"].to_numpy(), "mom": mom12.loc[t, u].to_numpy(), "dv": dv.loc[t, u].to_numpy()})
    uni_w = pd.DataFrame(0.0, index=me, columns=C.columns)
    for t, x in per.items():
        uni_w.loc[t, x["symbol"]] = 1.0 / len(x)
    base = {"유니버스 동일비중": window_stats(port_returns(uni_w, ret, rf, cost), W), "SPY": window_stats(etf("SPY"), W), "QQQ": window_stats(etf("QQQ"), W)}; ref = base["유니버스 동일비중"]
    print({t.date(): len(x) for t, x in list(per.items())[::24]}, file=sys.stderr, flush=True)

    cache: dict = {}

    def picks(t, key, how) -> dict:
        """그 달 산업별 고른 종목(열 위치). 종목 3 개 미만 산업은 없음."""
        if (t, key, how) not in cache:
            o = {}
            for k_, x in per[t].groupby(key):
                if len(x) >= 3:
                    o[k_] = x["j"].to_numpy() if how == "all" else x.dropna(subset=[{"growth5": "g", "mom5": "mom", "big5": "dv"}[how]]).nlargest(5, {"growth5": "g", "mom5": "mom", "big5": "dv"}[how])["j"].to_numpy()
            cache[(t, key, how)] = {k_: v for k_, v in o.items() if len(v)}
        return cache[(t, key, how)]

    def weights(key, score, n_ind, how, hold, rng=None):
        w = np.zeros(C.shape); cj_, cw_ = np.array([], dtype=int), np.array([])
        for i, t in enumerate(me):
            if t not in per:
                continue
            if i % hold == 0:
                pk = picks(t, key, how); sc = score.loc[t].reindex(list(pk)).dropna() if t in score.index else pd.Series(dtype=float); cj_, cw_ = np.array([], dtype=int), np.array([])
                if len(sc) >= n_ind:
                    inds = rng.choice(sc.index.to_numpy(), n_ind, replace=False) if rng is not None else sc.nlargest(n_ind).index
                    cj_ = np.concatenate([pk[k_] for k_ in inds]); cw_ = np.concatenate([np.full(len(pk[k_]), 1.0 / n_ind / len(pk[k_])) for k_ in inds])
            if len(cj_):
                ok = nxt_ok[i, cj_]; np.add.at(w[i], cj_[ok], cw_[ok])
        return pd.DataFrame(w, index=me, columns=C.columns)

    L = [f"# G5 — 성장 산업의 주식 사기, 종목 단위 (개발 ~{end}, 세전, 편도 0.2%)", "", "값 = 배수(MDD). 굵게 = 그 창에서 유니버스 동일비중과 무작위 산업 95 백분위를 둘 다 넘음.", "", "| 기준선 | " + " | ".join(W) + " |", "|---|" + "---|" * len(W)]
    L += [f"| {k} | " + " | ".join(f"×{x['mult']:.2f} ({x['mdd']:+.0%})" for x in v.values()) + " |" for k, v in base.items()]
    L += ["", "| 분류 | 지표 | 산업 수 | 종목 고르기 | 보유 | " + " | ".join(W) + " | 통과 창 | 플라세보 백분위 | 무작위 중앙 |", "|---|---|---|---|---|" + "---|" * len(W) + "---|---|---|"]; res = []; rng = np.random.default_rng(9)
    for key in ("ff49", "sic2", "sic3"):
        S = ind_scores(ft, key)
        # 산업 가격 모멘텀(유니버스 종목 6 달 수익 중앙값) 과 결합
        im = pd.DataFrame({t: x.assign(m6=mom6.loc[t, x["symbol"]].to_numpy()).groupby(key)["m6"].median() for t, x in per.items()}).T; r = lambda d: d.rank(axis=1, pct=True)  # noqa: E731
        S["ind_mom6(대조)"] = im; S["g+mom"] = (r(S["g_agg"]) + r(im.reindex_like(S["g_agg"]))) / 2
        for n_ind, how, hold in itertools.product((3, 10), ("all", "growth5", "mom5", "big5"), (1, 3)):
            ones = S["g_agg"].notna().astype(float).where(S["g_agg"].notna()); m = np.array([[window_stats(port_returns(weights(key, ones, n_ind, how, hold, rng), ret, rf, cost), W)[wn]["mult"] for wn in W] for _ in range(seeds)])
            for sn, sc in S.items():
                st = window_stats(port_returns(weights(key, sc, n_ind, how, hold), ret, rf, cost), W); ok = [st[wn]["mult"] > ref[wn]["mult"] and st[wn]["mult"] > np.percentile(m[:, i], 95) for i, wn in enumerate(WINDOWS)]; pct = [float((m[:, i] < st[wn]["mult"]).mean()) for i, wn in enumerate(W)]
                res.append({"level": key, "score": sn, "n_ind": n_ind, "pick": how, "hold": hold, **{wn: st[wn]["mult"] for wn in W}, **{f"{wn}_mdd": st[wn]["mdd"] for wn in W}, **{f"{wn}_pct": q for wn, q in zip(W, pct)}, **{f"{wn}_p50": float(np.median(m[:, i])) for i, wn in enumerate(W)}, "n_ok": sum(ok)})
                L.append(f"| {key} | {sn} | {n_ind} | {how} | {hold} | " + " | ".join((f"**×{st[wn]['mult']:.2f}**" if (i < len(ok) and ok[i]) else f"×{st[wn]['mult']:.2f}") + f" ({st[wn]['mdd']:+.0%})" for i, wn in enumerate(W)) + f" | {sum(ok)}/{len(WINDOWS)} | " + " · ".join(f"{q:.0%}" for q in pct) + " | " + " · ".join(f"×{np.median(m[:, i]):.2f}" for i in range(len(W))) + " |")
            print(key, n_ind, how, hold, file=sys.stderr, flush=True)
    df = pd.DataFrame(res); nw = len(WINDOWS); L += ["", f"전 창 통과 {int((df['n_ok'] == nw).sum())} / {len(df)}", "", "## 지표별 요약", "", "| 지표 | 칸 | 전 창 통과 | " + " | ".join(f"{wn} 중앙" for wn in W) + " | " + " | ".join(f"{wn} 백분위 평균" for wn in W) + " |", "|---|---|---|" + "---|" * (2 * len(W))]
    for sn, g in df.groupby("score"):
        L.append(f"| {sn} | {len(g)} | {int((g['n_ok'] == nw).sum())} | " + " | ".join(f"×{g[wn].median():.2f}" for wn in W) + " | " + " | ".join(f"{g[f'{wn}_pct'].mean():.0%}" for wn in W) + " |")
    L += ["", "## 종목 고르기별 요약 (무작위 산업에서 같은 방식으로 고른 것의 중앙값 대비)", "", "| 종목 고르기 | " + " | ".join(f"{wn} 무작위 중앙" for wn in W) + " |", "|---|" + "---|" * len(W)]
    L += [f"| {h} | " + " | ".join(f"×{g[f'{wn}_p50'].median():.2f}" for wn in W) + " |" for h, g in df.groupby("pick")]
    Path(f"{out}.md").write_text("\n".join(L)); df.to_json(f"{out}.json", orient="records", force_ascii=False, indent=1); print("\n".join(L[:10])); print("\n".join(L[-(len(df['score'].unique()) + 14):]))


if __name__ == "__main__":
    {"map": build_map, "run": main}[sys.argv[1]](json.loads(sys.argv[2]) if len(sys.argv) > 2 else {})
