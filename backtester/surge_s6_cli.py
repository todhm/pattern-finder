"""급등주 탐색 S6 — 내부자 공개시장 매수(Form 4 코드 P) 군집 사건 연구 (개발 구간만).

공시일 종가까지 알려진 것으로 보고 다음 거래일 시가 매수 → 63·126·252 거래일째 종가. 폐지는 마지막 종가, 한 건 +500% 상한.
대조: 같은 날 TQQQ·QQQ 를 같은 방식으로 산 수익, 같은 날 같은 유동성 등급 전 종목 평균(= 무작위 종목의 기대값).
실행: docker compose exec backtester python surge_s6_cli.py '{"out": "/app/sweep_results/surge_s6"}'
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from strategy.adapters.insider_signals import attach, buy_windows, first_cross, forward_returns
from strategy.adapters.surge_events import load_symbol

SR = Path("/app/sweep_results")
HS = (63, 126, 252)
TIERS = {"체결 가능($1·$1M)": (1.0, 1e6), "$5·$5M": (5.0, 5e6), "$10·$50M": (10.0, 5e7)}
SIGNALS = {                                            # 이름: (매수자 수, 창 금액 합, 임원·이사만, 추가 조건)
    "매수자 1 명 이상·$10만": (1, 1e5, True, None),
    "매수자 1 명 이상·$100만": (1, 1e6, True, None),
    "매수자 2 명 이상·$10만": (2, 1e5, True, None),
    "매수자 3 명 이상·$10만": (3, 1e5, True, None),
    "매수자 3 명 이상·$100만": (3, 1e6, True, None),
    "매수자 5 명 이상": (5, 0.0, True, None),
    "대표·재무책임자 포함 2 명 이상": (2, 1e5, True, lambda e: e["ceo"]),
    "3 명 이상 & 52주 고점 -30% 이하": (3, 1e5, True, lambda e: e["p_dd252"] <= -0.3),
    "3 명 이상 & 52주 고점 -50% 이하": (3, 1e5, True, lambda e: e["p_dd252"] <= -0.5),
    "3 명 이상 & 고점 -15% 이내": (3, 1e5, True, lambda e: e["p_dd252"] >= -0.15),
    "3 명 이상 & 직전 20일 -20% 이하": (3, 1e5, True, lambda e: e["p_ret20"] <= -0.2),
    "3 명 이상 & 200일선 위": (3, 1e5, True, lambda e: e["p_vs_sma200"] > 0),
    "10% 주주 포함 3 명 이상·$100만": (3, 1e6, False, None),
}


def bench(sym: str, end: str, hs=HS) -> pd.DataFrame:
    t = load_symbol(SR / "surge_av_cache" / f"{sym}.parquet"); t = t[t.index <= end]
    out = pd.DataFrame({f"{sym.lower()}{n}": t["C"].shift(-n) / t["O"].shift(-1) - 1.0 for n in hs}); out.index.name = "date"
    return out.reset_index()


def main(cfg):
    out = Path(cfg.get("out", SR / "surge_s6")); start = cfg.get("start", "2017-03-01"); end = cfg.get("end", "2023-12-31"); mid = pd.Timestamp(cfg.get("mid", "2020-06-30"))
    tx = pd.read_parquet(SR / "surge_insider_tx.parquet")
    panel = pd.read_parquet(SR / "surge_panel.parquet", columns=["date", "symbol", "O", "C", "delisted", "p_raw_c", "p_dv20", "p_dd252", "p_ret20", "p_vs_sma200"])
    panel = panel[panel["date"] <= end]                      # 봉인 구간 가격은 읽지 않는다
    f = forward_returns(panel, HS); del panel
    f = f.merge(bench("TQQQ", end), on="date", how="left").merge(bench("QQQ", end), on="date", how="left")
    for n in HS:                                             # 보유가 개발 구간 안에서 끝나는 날만
        f.loc[f[f"tqqq{n}"].isna(), f"e{n}"] = np.nan
    tier_mean = {}
    for name, (mp, mdv) in TIERS.items():
        t = f[(f["p_raw_c"] >= mp) & (f["p_dv20"] >= mdv)]
        tier_mean[name] = t.groupby("date")[[f"e{n}" for n in HS]].mean().rename(columns={f"e{n}": f"u{n}" for n in HS}).reset_index()
    tx = tx[tx["symbol"].isin(set(f["symbol"].unique()))]
    bw = {True: buy_windows(tx, 30, 1e4, True), False: buy_windows(tx, 30, 1e4, False)}
    p = lambda x: "-" if pd.isna(x) else f"{x:+.1%}"  # noqa: E731
    L = [f"# S6 — 내부자 공개시장 매수 군집 ({start}~{end})", "",
         "SEC Form 4 비파생 거래 코드 P(공개시장 매수), 건당 $1만 이상, 직전 30일 안 서로 다른 매수자 수·금액 합. 같은 종목은 90일 안에 다시 세지 않음.",
         "공시일(또는 직전 거래일) 종가까지 알려진 것으로 보고 다음 거래일 시가 매수 → n 거래일째 종가. 폐지 종목은 마지막 종가, 한 건 +500% 상한, 비용 미반영.",
         f"자료: 매수 공시 {int((tx['code'] == 'P').sum()):,} 건(패널 종목과 맞는 것), 임원·이사 매수 {int(((tx['code'] == 'P') & tx['od']).sum()):,} 건. 폐지 회사 포함.", "",
         "## 미리 정한 읽는 법", "",
         "- 정보가 있다 = 같은 날 같은 등급 전 종목 평균 대비 초과가 126·252일 모두 플러스이고 전반·후반 부호가 같다.",
         "- TQQQ 초과 후보 = 252일 TQQQ 대비 평균 초과 > 0 이고 TQQQ 를 이긴 사건 비율 ≥ 50%, 전반·후반 모두. 사건 수 ≥ 100. 후보가 있어야 달력 시간 포트폴리오(계단식)로 넘어간다.", ""]
    res = []
    for tname, (mp, mdv) in TIERS.items():
        L += [f"## 유동성 등급: {tname}", "", "| 신호 | 건수 | 연 평균 | 63일 평균 | 126일 평균 | 252일 평균 | 252 중앙 | 252 이긴% | 등급 평균 대비 63/126/252 | 전반·후반(252 등급 대비) | TQQQ 대비 252 | TQQQ 이긴% | 전반·후반(TQQQ 이긴%) | QQQ 대비 252 |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for sname, (k, tot, od, extra) in SIGNALS.items():
            ev = attach(first_cross(bw[od], k, tot, 90), f).merge(tier_mean[tname], on="date", how="left")
            ev = ev[(ev["date"] >= start) & (ev["p_raw_c"] >= mp) & (ev["p_dv20"] >= mdv)]
            if extra is not None:
                ev = ev[extra(ev).fillna(False)]
            g = ev.dropna(subset=["e252", "tqqq252"]); g63 = ev.dropna(subset=["e63"]); g126 = ev.dropna(subset=["e126"])
            if len(g) < 30:
                L.append(f"| {sname} | {len(g)} | " + " | ".join(["-"] * 12) + " |"); continue
            yrs = max((g["date"].max() - g["date"].min()).days / 365.25, 0.1); h1, h2 = g[g["date"] <= mid], g[g["date"] > mid]
            row = {"tier": tname, "signal": sname, "n": len(g), "per_year": len(g) / yrs, "m63": g63["e63"].mean(), "m126": g126["e126"].mean(), "m252": g["e252"].mean(), "med252": g["e252"].median(), "win252": (g["e252"] > 0).mean(),
                   "x63": (g63["e63"] - g63["u63"]).mean(), "x126": (g126["e126"] - g126["u126"]).mean(), "x252": (g["e252"] - g["u252"]).mean(),
                   "x252_h1": (h1["e252"] - h1["u252"]).mean(), "x252_h2": (h2["e252"] - h2["u252"]).mean(),
                   "vs_tqqq": (g["e252"] - g["tqqq252"]).mean(), "beat_tqqq": (g["e252"] > g["tqqq252"]).mean(), "beat_h1": (h1["e252"] > h1["tqqq252"]).mean(), "beat_h2": (h2["e252"] > h2["tqqq252"]).mean(),
                   "vs_tqqq_h1": (h1["e252"] - h1["tqqq252"]).mean(), "vs_tqqq_h2": (h2["e252"] - h2["tqqq252"]).mean(), "vs_qqq": (g["e252"] - g["qqq252"]).mean(), "n_h1": len(h1), "n_h2": len(h2)}
            res.append({a: (float(b) if isinstance(b, (float, np.floating)) else b) for a, b in row.items()})
            L.append(f"| {sname} | {row['n']:,} | {row['per_year']:.0f} | {p(row['m63'])} | {p(row['m126'])} | {p(row['m252'])} | {p(row['med252'])} | {row['win252']:.0%} | {p(row['x63'])} / {p(row['x126'])} / {p(row['x252'])} | {p(row['x252_h1'])} · {p(row['x252_h2'])} | {p(row['vs_tqqq'])} | {row['beat_tqqq']:.0%} | {row['beat_h1']:.0%} · {row['beat_h2']:.0%} | {p(row['vs_qqq'])} |")
        L.append("")
    df = pd.DataFrame(res)
    info = df[(df["x126"] > 0) & (df["x252"] > 0) & (df["x252_h1"] > 0) & (df["x252_h2"] > 0)]
    cand = df[(df["n"] >= 100) & (df["vs_tqqq"] > 0) & (df["beat_tqqq"] >= 0.5) & (df["vs_tqqq_h1"] > 0) & (df["vs_tqqq_h2"] > 0) & (df["beat_h1"] >= 0.5) & (df["beat_h2"] >= 0.5)]
    L += ["## 판정", "", f"- 정보가 있는 칸(등급 평균 대비, 전·후반 같은 부호): {len(info)} / {len(df)}", f"- TQQQ 초과 후보: {len(cand)} / {len(df)}" + ("" if cand.empty else " — " + ", ".join(f"{r.tier} · {r.signal}" for r in cand.itertuples()))]
    Path(f"{out}.md").write_text("\n".join(L)); df.to_json(f"{out}.json", orient="records", force_ascii=False, indent=1); print("\n".join(L))


if __name__ == "__main__":
    main(json.loads(sys.argv[1]) if len(sys.argv) > 1 else {})
