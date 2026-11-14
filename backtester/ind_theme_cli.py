"""산업 통찰 수동 검증 M3 — 주제(테마)가 기업 공시에 퍼지는 모습을 읽는다 (SEC 전문 검색, 10-K 본문, 2001~).

"앞으로 클 산업" 은 분류표에 없다(클라우드·AI·셰일은 SIC 여러 개에 흩어짐). 대신 그 말이 연차보고서에 등장하고 퍼지는 과정은 제출일이 찍힌 채 남아 있다.
 fetch : 주제별·연도별(매년 6 월 말까지 12 달) 10-K 언급 문서 수, 언급 회사의 SIC 분포, 언급 회사 CIK 목록 → sweep_results/ind_theme_cache/{주제}_{연도}.json
주제 목록 THEMES 는 결과를 보기 전에 고정했다. 뜬 것과 꺼진 것을 일부러 섞었다(사후 선택 편향을 줄이려고) — 그래도 목록 자체가 오늘의 지식으로 고른 것이라는 한계는 남는다.
실행: docker compose run --rm --no-deps backtester python ind_theme_cli.py fetch '{}'
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import requests

SR = Path("/app/sweep_results"); CACHE = SR / "ind_theme_cache"
UA = {"User-Agent": "research-script/1.0 (academic backtest)"}
THEMES = {
    "cloud": '"cloud computing"', "saas": '"software as a service"', "bigdata": '"big data"', "ml": '"machine learning"', "ai": '"artificial intelligence"', "deep_learning": '"deep learning"', "gpu": '"graphics processing units"',
    "datacenter": '"hyperscale"', "cybersecurity": '"cybersecurity"', "iot": '"internet of things"', "blockchain": '"blockchain"', "crypto": '"cryptocurrency"', "vr": '"virtual reality"', "ar": '"augmented reality"', "wearable": '"wearable devices"',
    "printing3d": '"3D printing"', "drones": '"unmanned aerial"', "robotics": '"collaborative robots"', "autonomous": '"autonomous vehicles"', "ev": '"electric vehicles"', "lithium": '"lithium-ion"', "solar": '"solar modules"', "wind": '"wind turbines"',
    "fuelcell": '"fuel cells"', "hydrogen": '"green hydrogen"', "led": '"LED lighting"', "oled": '"OLED"', "sic_power": '"silicon carbide"', "g5": '"5G"', "edge": '"edge computing"', "streaming": '"video streaming"', "esports": '"esports"',
    "mobile_pay": '"mobile payments"', "fintech": '"fintech"', "bnpl": '"buy now, pay later"', "p2p_lending": '"marketplace lending"', "ecommerce": '"e-commerce fulfillment"', "dtc": '"direct-to-consumer"', "subscription": '"recurring revenue"',
    "shale": '"hydraulic fracturing"', "frac_sand": '"frac sand"', "lng": '"LNG export"', "cannabis": '"cannabis"', "vaping": '"e-cigarettes"', "plant_based": '"plant-based"', "telehealth": '"telehealth"', "immunotherapy": '"immuno-oncology"',
    "gene_therapy": '"gene therapy"', "crispr": '"CRISPR"', "car_t": '"CAR-T"', "mrna": '"mRNA"', "biosimilar": '"biosimilars"', "robotic_surgery": '"robotic surgery"', "glp1": '"GLP-1"', "aca": '"Medicaid expansion"', "sports_betting": '"sports betting"',
    "quantum": '"quantum computing"', "space": '"low earth orbit"', "metaverse": '"metaverse"', "genai": '"generative AI"', "nanotech": '"nanotechnology"', "rfid": '"RFID"', "ethanol": '"ethanol plants"', "smart_grid": '"smart grid"',
    "tablet": '"tablets"', "smartphone": '"smartphones"', "social": '"social media"', "mobile_app": '"mobile applications"', "_all": "",
}


def fetch(cfg):
    CACHE.mkdir(exist_ok=True); s = requests.Session(); s.headers.update(UA); y0, y1 = int(cfg.get("from", 2005)), int(cfg.get("to", 2026)); n = 0
    for slug, q in THEMES.items():
        for y in range(y0, y1 + 1):
            fn = CACHE / f"{slug}_{y}.json"
            if fn.exists():
                continue
            base = {"q": q, "forms": "10-K", "dateRange": "custom", "startdt": f"{y - 1}-07-01", "enddt": f"{y}-06-30"}; ciks: dict = {}; sic = {}; total = None; frm = 0
            while True:
                for attempt in range(12):
                    r = s.get("https://efts.sec.gov/LATEST/search-index", params={**base, "from": frm}, timeout=60)
                    if r.status_code == 200 and "hits" in r.json():
                        break
                    time.sleep(3 + 5 * attempt)
                else:
                    if frm == 0:
                        raise RuntimeError(f"{slug} {y} HTTP {r.status_code} {r.text[:200]}")
                    print("페이지 건너뜀", slug, y, frm, r.text[:120], file=sys.stderr, flush=True); break
                j = r.json(); h = j["hits"]; total = h["total"]["value"]
                if frm == 0:
                    sic = {b["key"]: b["doc_count"] for b in j.get("aggregations", {}).get("sic_filter", {}).get("buckets", [])}
                for x in h["hits"]:
                    for c in x["_source"].get("ciks", []):
                        ciks.setdefault(int(c), x["_source"].get("file_date"))
                frm += 100; time.sleep(0.12)
                if slug == "_all" or not h["hits"] or frm >= min(total, 1000):
                    break
            fn.write_text(json.dumps({"theme": slug, "q": q, "year": y, "total": total, "sic": sic, "ciks": ciks})); n += 1
        print(slug, n, file=sys.stderr, flush=True)


def cards(cfg):
    """주제 × 매년 6 월 말 카드: 확산(언급 문서 수·비중·증가), 언급 회사들의 재무 구조(industry_dossier.card), 주가, 정답(언급 회사 중 유동 종목을 그날 사서 든 앞 12·36 달)."""
    import numpy as np
    import pandas as pd

    from ind_m0_cli import SEAL, prices
    from strategy.adapters.industry_dossier import card, snapshot
    fs = pd.read_parquet(SR / "ind_m_filings.parquet"); ret, liq = prices(); lg = np.log1p(ret.fillna(0.0)); sm = pd.read_parquet(SR / "ind_sym_cik.parquet"); cik_of = sm.set_index("symbol")["cik"].reindex(ret.columns)
    J = {(f.stem.rsplit("_", 1)[0], int(f.stem.rsplit("_", 1)[1])): json.loads(f.read_text()) for f in CACHE.glob("*.json")}; rows = []
    for y in range(2012, 2027):
        t = pd.Timestamp(f"{y}-06-30")
        if t not in ret.index:
            continue
        now, y1, y2 = snapshot(fs, t), snapshot(fs, t - pd.DateOffset(years=1)), snapshot(fs, t - pd.DateOffset(years=2)); i = ret.index.get_loc(t); tot = lambda yy: max(J[("_all", yy)]["total"], 1)  # noqa: E731
        live = liq.loc[t].to_numpy(dtype=bool) & ret.iloc[i].notna().to_numpy() & cik_of.notna().to_numpy(); syms = ret.columns[live]; sc = cik_of[syms]
        past = lambda k: np.expm1(lg.iloc[i - k + 1:i + 1].sum()).where(ret.iloc[i - k + 1].notna())  # noqa: E731
        fwd = lambda k: np.expm1(lg.iloc[i + 1:i + 1 + k].sum()) if (i + k < len(ret) and ret.index[i + k] <= SEAL) else pd.Series(np.nan, index=ret.columns)  # noqa: E731
        p12, p36, f12, f36 = past(12), past(36), fwd(12), fwd(36)
        for slug in THEMES:
            if slug == "_all":
                continue
            j = J[(slug, y)]; n = j["total"]; hist = {k: J[(slug, y - k)]["total"] / tot(y - k) for k in range(0, 8) if (slug, y - k) in J}; ck = {int(c) for c in j["ciks"]}; prev = {int(c) for c in J[(slug, y - 1)]["ciks"]}
            first = next((yy for yy in range(2005, y + 1) if J[(slug, yy)]["total"] >= 5), None); sicn = sum(j["sic"].values()) or 1; top_sic = max(j["sic"].values()) / sicn if j["sic"] else np.nan; top_code = int(max(j["sic"], key=j["sic"].get)) if j["sic"] else -1
            m = now[now.index.isin(ck)]; c = card(m, y1[y1.index.isin(ck)], y2[y2.index.isin(ck)]) if len(m) >= 5 else {}; ss = syms[sc.isin(ck).to_numpy()]
            c.update({"date": t, "theme": slug, "q": j["q"], "docs": n, "share": hist[0], "share_1y": hist.get(1, np.nan), "share_3y": hist.get(3, np.nan), "share_5y": hist.get(5, np.nan), "age": (y - first) if first else np.nan, "new_mentioner_share": len(ck - prev) / max(len(ck), 1),
                      "n_ciks": len(ck), "top_sic_share": top_sic, "top_sic": top_code, "n_sics": len(j["sic"]), "n_stocks": len(ss), "price_12m": float(p12[ss].median()) if len(ss) else np.nan, "price_36m": float(p36[ss].median()) if len(ss) else np.nan,
                      "fwd12": float(f12[ss].mean()) if len(ss) >= 5 else np.nan, "fwd36": float(f36[ss].mean()) if len(ss) >= 5 else np.nan, "u_fwd12": float(f12[syms].mean()), "u_fwd36": float(f36[syms].mean()), "stocks": " ".join(ss[:60])}); rows.append(c)
        print(y, file=sys.stderr, flush=True)
    d = pd.DataFrame(rows); d.to_parquet(SR / "ind_theme_cards.parquet"); pd.set_option("display.width", 250)
    print(d.assign(y=d["date"].dt.year).pivot(index="theme", columns="y", values="docs").iloc[:, ::2].to_string())


def sector(sic: int) -> str:
    if sic < 0:
        return "알 수 없음"
    for lo, hi, name in ((2830, 2836, "제약·바이오"), (8731, 8734, "제약·바이오(연구)"), (3570, 3579, "컴퓨터·전자"), (3660, 3679, "통신 장비·반도체"), (7370, 7379, "소프트웨어·IT 서비스"), (100, 1499, "자원·에너지 채굴"), (1500, 1799, "건설"), (2900, 2999, "정유"),
                         (2000, 3999, "제조"), (4900, 4999, "전력·가스"), (4000, 4899, "운송·통신"), (5000, 5999, "유통·소매"), (6000, 6799, "금융·부동산"), (8000, 8099, "의료 서비스"), (7000, 8999, "서비스")):
        if lo <= sic <= hi:
            return name
    return "기타"


def render_theme(c: dict, label: str) -> str:
    import numpy as np

    from strategy.adapters.industry_dossier import _p, render
    g = lambda k: c.get(k, np.nan)  # noqa: E731
    chg = lambda k: _p(g("share") / g(k) - 1) if np.isfinite(g(k)) and g(k) > 0 else ("그때는 언급 없음" if np.isfinite(g(k)) else "자료 없음")  # noqa: E731
    age = "자료 없음" if not np.isfinite(g("age")) else ("6 년 이상" if g("age") >= 6 else f"{g('age'):.0f} 년")
    L = [f"### 주제 {label}", f"- 확산: 지난 12 달 연차보고서 언급 {g('docs'):.0f} 건 (전체 보고서의 {g('share'):.2%}) · 1 년 전 대비 {chg('share_1y')} · 3 년 전 대비 {chg('share_3y')} · 5 년 전 대비 {chg('share_5y')} · 언급이 5 건을 처음 넘은 지 {age} · 올해 처음 언급한 회사 비율 {_p(g('new_mentioner_share'), sign=False)}",
         f"- 누가 말하나: 가장 많은 업종 = {sector(int(g('top_sic')) if np.isfinite(g('top_sic')) else -1)} (언급의 {_p(g('top_sic_share'), sign=False)}) · 언급 업종 수 {g('n_sics'):.0f}", "- 아래는 이 주제를 언급한 회사들의 합산 재무:"]
    return "\n".join(L + render(c, label).split("\n")[1:]) if "rev_g" in c and np.isfinite(g("rev_g")) else "\n".join(L[:-1] + [f"- 언급 회사 재무: 자료 부족 · 유동 종목 {g('n_stocks'):.0f} 개 · 지난 12 달 주가 {_p(g('price_12m'))} · 36 달 {_p(g('price_36m'))}"])


def pack(cfg):
    import numpy as np
    import pandas as pd
    d = pd.read_parquet(SR / "ind_theme_cards.parquet"); x = d[(d["date"] == cfg["date"]) & (d["n_stocks"] >= 5)].sample(frac=1.0, random_state=int(cfg.get("seed", 1))).reset_index(drop=True); out = Path(cfg["out"]); out.parent.mkdir(parents=True, exist_ok=True)
    L = [f"## 판정 묶음 — 주제 {len(x)} 개 (모두 같은 날의 카드)", ""]; key = {}
    for j_, r in x.iterrows():
        lab = f"T{j_ + 1:02d}"; L += [render_theme(r.to_dict(), lab), ""]; key[lab] = {"date": str(r["date"])[:10], "theme": r["theme"], "n_stocks": int(r["n_stocks"]), "fwd12": r["fwd12"], "fwd36": r["fwd36"], "u_fwd12": r["u_fwd12"], "u_fwd36": r["u_fwd36"]}
    out.write_text("\n".join(L)); Path(str(out).replace(".md", "_key.json")).write_text(json.dumps(key, ensure_ascii=False, indent=1)); print(len(key), "cards →", out)


def emerging(d):
    """기계 대조판: '새로 퍼지는 말' = 5 건을 넘은 지 5 년 이하이거나 3 년 새 비중 3 배 이상, 언급 10 건 이상. 자금 과잉(급성장 회사 ≥ 40% 이면서 설비투자 ≥ +30%) 제외."""
    g3 = d["share"] / d["share_3y"].where(d["share_3y"] > 0); flood = (d.get("pct_hot", 0) >= 0.4) & (d.get("capex_g", 0) >= 0.3)
    return ((d["age"] <= 5) | (g3 >= 3) | (d["share_3y"] == 0)) & (d["docs"] >= 10) & ~flood


def settle(cfg):
    import numpy as np
    import pandas as pd
    d = pd.read_parquet(SR / "ind_theme_cards.parquet"); d["em"] = emerging(d); rng = np.random.default_rng(8); L = ["# M3 — 주제 가린 판정 정산", "", "초과 = 그 주제를 언급한 회사 중 유동 종목을 그날 사서 그대로 든 수익 − 같은 날 전체 유동 종목 평균. 백분위 = 같은 풀에서 주제 5 개를 무작위로 고른 2,000 벌 중 이보다 못한 비율.", ""]; tot = {}
    for keyf, resf in cfg["sets"]:
        key = json.load(open(keyf)); res = json.load(open(resf)); k = pd.DataFrame(key).T; k["score"] = pd.Series(res["scores"]).reindex(k.index).astype(float); dt = k["date"].iloc[0]
        for c in ("fwd12", "fwd36", "u_fwd12", "u_fwd36"):
            k[c] = k[c].astype(float)
        k = k.join(d[d["date"] == dt].set_index("theme")[["em", "price_12m", "docs"]], on="theme")
        L += [f"## 판정 시점 {dt} — 주제 {len(k)} 개, 전체 종목 평균 12 달 {k['u_fwd12'].iloc[0]:+.0%} · 36 달 {k['u_fwd36'].iloc[0]:+.0%}", ""]
        groups = {"사람: 매수 5": k.loc[res["buy"]], "사람: 회피 5": k.loc[res["avoid"]], "사람: 점수 4~5 전부": k[k["score"] >= 4], "사람: 점수 1~2 전부": k[k["score"] <= 2], "기계: 새로 퍼지는 말 전부": k[k["em"]], "기계: 그 밖 전부": k[~k["em"]], "주제 전부(평균 주제)": k, "12 달 주가 상위 5": k.nlargest(5, "price_12m")}
        for h in (12, 36):
            L += [f"### 앞 {h} 달", "", "| 묶음 | 주제 수 | 평균 수익 | 초과 | 전체보다 나은 비율 | 무작위 5 개 대비 백분위 |", "|---|---|---|---|---|---|"]
            ex_all = (k[f"fwd{h}"] - k[f"u_fwd{h}"]).dropna().to_numpy()
            for n_, x in groups.items():
                ex = (x[f"fwd{h}"] - x[f"u_fwd{h}"]).dropna(); r = np.array([rng.choice(ex_all, min(5, len(ex_all)), replace=False).mean() for _ in range(2000)])
                L.append(f"| {n_} | {len(ex)} | {x[f'fwd{h}'].mean():+.0%} | {ex.mean():+.0%}p | {(ex > 0).mean():.0%} | {(r < ex.mean()).mean():.0%} |" if len(ex) else f"| {n_} | 0 | - | - | - | - |"); tot.setdefault((n_, h), []).append(float(ex.mean()) if len(ex) else np.nan)
            ic = k["score"].rank().corr(k[f"fwd{h}"].rank()); tot.setdefault(("ic", h), []).append(ic); L += ["", f"순위 상관(사람 점수 vs 수익): {ic:+.2f}", ""]
        L += ["매수 5: " + " / ".join(f"{lab} = {k.at[lab, 'theme']} ({k.at[lab, 'n_stocks']} 종목) 36 달 {k.at[lab, 'fwd36']:+.0%}" for lab in res["buy"]), "", "회피 5: " + " / ".join(f"{lab} = {k.at[lab, 'theme']} 36 달 {k.at[lab, 'fwd36']:+.0%}" for lab in res["avoid"]), ""]
    L += ["## 전 시점 합산 (시점별 초과의 평균, + 였던 시점 수)", "", "| 묶음 | 12 달 초과 평균 | + 시점 | 36 달 초과 평균 | + 시점 |", "|---|---|---|---|---|"]
    for n_ in [g for g, h in tot if h == 12 and g != "ic"]:
        a, b = np.array(tot[(n_, 12)]), np.array(tot[(n_, 36)]); L.append(f"| {n_} | {np.nanmean(a):+.1%}p | {int((a > 0).sum())}/{len(a)} | {np.nanmean(b):+.1%}p | {int((b > 0).sum())}/{len(b)} |")
    L += ["", f"순위 상관 평균: 12 달 {np.mean(tot[('ic', 12)]):+.2f} · 36 달 {np.mean(tot[('ic', 36)]):+.2f}"]; Path(cfg["out"]).write_text("\n".join(L)); print("\n".join(L))


if __name__ == "__main__":
    {"fetch": fetch, "cards": cards, "pack": pack, "settle": settle}[sys.argv[1]](json.loads(sys.argv[2]) if len(sys.argv) > 2 else {})
