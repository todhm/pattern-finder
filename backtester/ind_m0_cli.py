"""산업 통찰 수동 검증 M0 — 산업 구조 카드와 정답표.

 build : sec_fs_cache2 → 제출물 요약 `ind_m_filings.parquet` → 매년 6 월 말(2012~) SIC 3 자리 산업 카드 `ind_m_cards.parquet`
         (카드 숫자는 그날까지 제출된 것만. 주가 줄 = 그 산업 유동 종목의 지난 12·36 달 수익 중앙값. 정답 = 그날 산 종목들을 그대로 든 앞 12·36 달 수익, 폐지는 마지막 가격)
 show  : '{"date": "2016-06-30", "sic3": 367}' → 이름이 보이는 카드 + 상위 회사(사례 공부용)
 pack  : '{"dates": [...], "out": "..."}' → 가린 카드 묶음 md + 정답 키 json(판정자에게는 md 만)
카드 조건: 제출 회사 ≥ 8, 유동 종목(주가 ≥ $5·거래대금 ≥ $5M, CIK 연결) ≥ 5. 정답이 2023-12 를 넘는 것은 비운다(봉인).
실행: docker compose run --rm --no-deps backtester python ind_m0_cli.py build
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from strategy.adapters.industry_dossier import card, filing_summary, firm_rows, render, render_firms, snapshot

SR = Path("/app/sweep_results"); SEAL = pd.Timestamp("2023-12-31")


def prices():
    """월말 종가·유동성·월수익: 2017-01 까지 long 패널, 그 뒤 main 패널."""
    out = []
    for path, a, z in (("surge_panel_long.parquet", "2008-06-01", "2017-01-31"), ("surge_panel.parquet", "2017-01-01", "2026-12-31")):
        p = pd.read_parquet(SR / path, columns=["date", "symbol", "C", "raw_c", "dv"]); p = p[(p["date"] >= a) & (p["date"] <= z)]; p["m"] = p["date"] + pd.offsets.MonthEnd(0)
        g = p.sort_values("date").groupby(["m", "symbol"]); C = g["C"].last().unstack(); liq = (g["raw_c"].last().unstack() >= 5) & (g["dv"].median().unstack() >= 5e6); out.append((C, liq, C.pct_change(fill_method=None)))
    (C1, l1, r1), (C2, l2, r2) = out; cols = C1.columns.union(C2.columns); cut = pd.Timestamp("2017-01-31")
    ret = pd.concat([r1[r1.index <= cut].reindex(columns=cols), r2[r2.index > cut].reindex(columns=cols)]); liq = pd.concat([l1[l1.index <= cut].reindex(columns=cols), l2[l2.index > cut].reindex(columns=cols)]).fillna(False).astype(bool)
    return ret, liq


def build(cfg):
    fsp = SR / "ind_m_filings.parquet"
    if not fsp.exists():
        pd.concat([filing_summary(pd.read_parquet(p)) for p in sorted((SR / "sec_fs_cache2").glob("*.parquet")) if len(pd.read_parquet(p))], ignore_index=True).to_parquet(fsp)
    fs = pd.read_parquet(fsp); fs["sic3"] = fs["sic"] // 10; ret, liq = prices(); lg = np.log1p(ret.fillna(0.0))
    sm = pd.read_parquet(SR / "ind_sym_cik.parquet"); sym_of = sm.groupby("cik")["symbol"].apply(list); cik_of = sm.set_index("symbol")["cik"].reindex(ret.columns)
    rows = []
    for y in range(2012, 2027):
        t = pd.Timestamp(f"{y}-06-30")
        if t not in ret.index:
            continue
        now, y1, y2 = snapshot(fs, t), snapshot(fs, t - pd.DateOffset(years=1)), snapshot(fs, t - pd.DateOffset(years=2)); i = ret.index.get_loc(t)
        live = liq.loc[t].to_numpy(dtype=bool) & cik_of.isin(now.index).to_numpy() & ret.iloc[i].notna().to_numpy(); syms = ret.columns[live]; sic_s = now["sic3"].reindex(cik_of[syms].to_numpy()).to_numpy()
        past = lambda k: np.expm1(lg.iloc[i - k + 1:i + 1].sum()).where(ret.iloc[i - k + 1].notna()) if i - k + 1 >= 0 else pd.Series(np.nan, index=ret.columns)  # noqa: E731
        fwd = lambda k: np.expm1(lg.iloc[i + 1:i + 1 + k].sum()) if (i + k < len(ret) and ret.index[i + k] <= SEAL) else pd.Series(np.nan, index=ret.columns)  # noqa: E731
        p12, p36, f12, f36 = past(12), past(36), fwd(12), fwd(36)
        for s3, g in now.groupby("sic3"):
            ss = syms[sic_s == s3]
            if len(g) < 8 or len(ss) < 5:
                continue
            c = card(g, y1[y1["sic3"] == s3], y2[y2["sic3"] == s3])
            if "rev_g" not in c:
                continue
            top = g["rev_cur"].nlargest(5)
            c.update({"date": t, "sic3": int(s3), "n_stocks": len(ss), "price_12m": float(p12[ss].median()), "price_36m": float(p36[ss].median()), "fwd12": float(f12[ss].mean()), "fwd36": float(f36[ss].mean()),
                      "u_fwd12": float(f12[syms].mean()), "u_fwd36": float(f36[syms].mean()), "leaders": ", ".join(g.loc[top.index, "name"].str.title()), "stocks": " ".join(ss), "firms": json.dumps(firm_rows(g, y1[y1["sic3"] == s3], 6), ensure_ascii=False)}); rows.append(c)
        print(y, sum(1 for r in rows if r["date"] == t), len(syms), file=sys.stderr, flush=True)
    d = pd.DataFrame(rows); d.to_parquet(SR / "ind_m_cards.parquet"); print(d.groupby("date").agg(n=("sic3", "size"), u12=("u_fwd12", "first"), u36=("u_fwd36", "first"), top36=("fwd36", "max"), bot36=("fwd36", "min")).round(2).to_string())


def show(cfg):
    d = pd.read_parquet(SR / "ind_m_cards.parquet"); r = d[(d["date"] == cfg["date"]) & (d["sic3"] == int(cfg["sic3"]))].iloc[0].to_dict()
    print(render(r, f"SIC {r['sic3']} @ {str(r['date'])[:10]}")); print(render_firms(json.loads(r["firms"]), names=True)); print("- 종목:", r["stocks"]); print(f"- 그 뒤: 12 달 {r['fwd12']:+.0%} (전체 {r['u_fwd12']:+.0%}) · 36 달 {r['fwd36']:+.0%} (전체 {r['u_fwd36']:+.0%})")


def pack(cfg):
    d = pd.read_parquet(SR / "ind_m_cards.parquet"); rng = np.random.default_rng(int(cfg.get("seed", 1))); out = Path(cfg["out"]); out.parent.mkdir(parents=True, exist_ok=True); key = {}; L = []
    for n, dt in enumerate(cfg["dates"]):
        x = d[d["date"] == dt].sample(frac=1.0, random_state=int(rng.integers(1 << 30))).reset_index(drop=True); L += [f"## 판정 묶음 {chr(65 + n)} — 산업 {len(x)} 곳 (모두 같은 날의 카드)", ""]
        for j, r in x.iterrows():
            lab = f"{chr(65 + n)}{j + 1:02d}"; L += [render(r.to_dict(), lab), ""]; key[lab] = {"date": str(r["date"])[:10], "sic3": int(r["sic3"]), "leaders": r["leaders"], "fwd12": r["fwd12"], "fwd36": r["fwd36"], "u_fwd12": r["u_fwd12"], "u_fwd36": r["u_fwd36"]}
    out.write_text("\n".join(L)); Path(str(out).replace(".md", "_key.json")).write_text(json.dumps(key, ensure_ascii=False, indent=1)); print(len(key), "cards →", out)


def machine_score(d: pd.DataFrame) -> pd.Series:
    """체크리스트 v1 을 공식으로: 답할 수 있는 A1~C2 중 '예' 비율(6 개 이상 답할 수 있을 때), D 에 걸리면 −0.2."""
    g = lambda c: d[c] if c in d else pd.Series(np.nan, index=d.index)  # noqa: E731
    items = {"A1": (g("capex_g") <= g("rev_g") + 0.10).where(g("capex_g").notna()), "A2": (d["n_firms"] <= d["n_firms_1y"]).where(d["n_firms_1y"] > 0), "A3": (g("top3_share") >= 0.5).where(g("top3_share").notna()),
             "A4": ((g("pct_diluting") < 0.25) & (g("issue_to_rev").fillna(0) < 0.05)).where(g("pct_diluting").notna()), "B1": (g("opm_chg") > 0).where(g("opm_chg").notna()), "B2": (g("incr_margin") >= 0.25).where(g("incr_margin").notna()),
             "B3": (g("gm_chg") >= 0).where(g("gm_chg").notna()), "B4": ((g("rev_g") > g("rev_g_1y_ago")) & (g("rev_g") > 0)).where(g("rev_g_1y_ago").notna()), "C1": (g("defrev_g") > g("rev_g")).where(g("defrev_g").notna()), "C2": (g("inv_g") <= g("rev_g") + 0.05).where(g("inv_g").notna())}
    m = pd.DataFrame(items).astype(float); sc = m.mean(axis=1).where(m.notna().sum(axis=1) >= 6); bad = ((g("pct_hot") >= 0.4) & (g("capex_g") >= 0.3)) | (g("issue_to_rev") >= 0.2)
    return sc - 0.2 * bad.astype(float)


def _row(name, x, pool, h, rng, k=5, n=2000):
    ex = x[f"fwd{h}"] - x[f"u_fwd{h}"]; allex = pool[f"fwd{h}"] - pool[f"u_fwd{h}"]; r = np.array([rng.choice(allex.to_numpy(), min(k, len(allex)), replace=False).mean() for _ in range(n)])
    return f"| {name} | {len(x)} | {x[f'fwd{h}'].mean():+.0%} | {ex.mean():+.0%}p | {(ex > 0).mean():.0%} | {(r < ex.mean()).mean():.0%} |"


def settle(cfg):
    """판정 json(scores·buy·avoid) + 키 → 사람 매수/회피, 기계 체크리스트 상위/하위 5, 12 달 모멘텀 상위 5 를 같은 풀에서 정산. cfg: {"sets": [[키 json, 판정 json], ...], "out": md}"""
    d = pd.read_parquet(SR / "ind_m_cards.parquet"); d["ms"] = machine_score(d); rng = np.random.default_rng(4); L = ["# M2 — 가린 판정 정산", "", "초과 = 그 산업 종목을 그날 사서 그대로 든 수익 − 같은 날 전체 유동 종목 평균. 백분위 = 같은 풀에서 5 곳을 무작위로 고른 2,000 벌 중 이보다 못한 비율.", ""]; tot = {}
    for keyf, resf in cfg["sets"]:
        key = json.load(open(keyf)); res = json.load(open(resf)); k = pd.DataFrame(key).T; k["score"] = pd.Series(res["scores"]).reindex(k.index).astype(float); dt = k["date"].iloc[0]; pool = d[d["date"] == dt].set_index("sic3")
        k = k.join(pool[["ms", "price_12m", "rev_g"]], on="sic3"); 
        for c in ("fwd12", "fwd36", "u_fwd12", "u_fwd36"):
            k[c] = k[c].astype(float)
        L += [f"## 판정 시점 {dt} — 산업 {len(k)} 곳, 전체 평균 12 달 {k['u_fwd12'].iloc[0]:+.0%} · 36 달 {k['u_fwd36'].iloc[0]:+.0%}", ""]
        groups = {"사람: 매수 5": k.loc[res["buy"]], "사람: 회피 5": k.loc[res["avoid"]], "사람: 점수 4~5 전부": k[k["score"] >= 4], "사람: 점수 1~2 전부": k[k["score"] <= 2], "기계 체크리스트 상위 5": k.nlargest(5, "ms"), "기계 체크리스트 하위 5": k.nsmallest(5, "ms"), "12 달 주가 상위 5": k.nlargest(5, "price_12m"), "매출 성장 상위 5": k.nlargest(5, "rev_g")}
        for h in (12, 36):
            L += [f"### 앞 {h} 달", "", "| 묶음 | 산업 수 | 평균 수익 | 초과 | 전체보다 나은 비율 | 무작위 5 곳 대비 백분위 |", "|---|---|---|---|---|---|"] + [_row(n_, x, k, h, rng) for n_, x in groups.items() if len(x)]
            ic = k["score"].rank().corr((k[f"fwd{h}"]).rank()); icm = k["ms"].rank().corr(k[f"fwd{h}"].rank()); L += ["", f"순위 상관: 사람 점수 {ic:+.2f} · 기계 점수 {icm:+.2f}", ""]
            for n_, x in groups.items():
                tot.setdefault((n_, h), []).append(float((x[f"fwd{h}"] - x[f"u_fwd{h}"]).mean()) if len(x) else np.nan)
            tot.setdefault(("ic사람", h), []).append(ic); tot.setdefault(("ic기계", h), []).append(icm)
        L += ["매수 5: " + " / ".join(f"{lab} = SIC {k.at[lab, 'sic3']} ({k.at[lab, 'leaders'][:40]}) 36 달 {k.at[lab, 'fwd36']:+.0%}" for lab in res["buy"]), "", "회피 5: " + " / ".join(f"{lab} = SIC {k.at[lab, 'sic3']} ({k.at[lab, 'leaders'][:40]}) 36 달 {k.at[lab, 'fwd36']:+.0%}" for lab in res["avoid"]), ""]
    L += ["## 전 시점 합산 (시점별 초과의 평균, + 였던 시점 수)", "", "| 묶음 | 12 달 초과 평균 | + 시점 | 36 달 초과 평균 | + 시점 |", "|---|---|---|---|---|"]
    for n_ in [g for g, h in tot if h == 12 and not g.startswith("ic")]:
        a, b = np.array(tot[(n_, 12)]), np.array(tot[(n_, 36)]); L.append(f"| {n_} | {np.nanmean(a):+.1%}p | {int((a > 0).sum())}/{len(a)} | {np.nanmean(b):+.1%}p | {int((b > 0).sum())}/{len(b)} |")
    L += ["", f"순위 상관 평균: 사람 12 달 {np.mean(tot[('ic사람', 12)]):+.2f} · 36 달 {np.mean(tot[('ic사람', 36)]):+.2f} / 기계 12 달 {np.mean(tot[('ic기계', 12)]):+.2f} · 36 달 {np.mean(tot[('ic기계', 36)]):+.2f}"]
    Path(cfg["out"]).write_text("\n".join(L)); print("\n".join(L))


if __name__ == "__main__":
    {"build": build, "show": show, "pack": pack, "settle": settle}[sys.argv[1]](json.loads(sys.argv[2]) if len(sys.argv) > 2 else {})
