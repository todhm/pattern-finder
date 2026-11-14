"""2차 탐색 L8 — H43 패키지가 현금으로 쉬는 몫에 다른 것을 넣으면 H43 을 넘는가 (네 창).

H43 목표 비중 w(전날 종가 기준) → TQQQ w, 나머지 1−w 를 X 에. X = 현금 | QQQ | L4′ 대형 주도주 곡선(상위 N·모멘텀·K 3).
세전, H43 비중 변경 비용 0.2%. 창: 2001~2009·2010~2017(근사 패널·합성 TQQQ), 2018~2023(개발), 2024~2026(이미 열람한 봉인 곡선).
실행: docker compose exec backtester python surge_l8_cli.py '{}'
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from strategy.adapters.surge_events import load_symbol
from strategy.adapters.surge_rank_portfolio import curve_stats
from strategy.adapters.tqqq_package import package_frame

SR = Path("/app/sweep_results")
WINDOWS = {"2001~2009": ("surge_l4_longA", True), "2010~2017": ("surge_l4_longB", True), "2018~2023 개발": ("surge_l4", False), "2024~2026 봉인": ("surge_l4_sealed", False)}
ALTS = ["20_mom3_3_1", "30_mom3_3_1", "30_mom6_3_1", "20_mom12_1_3_1", "30_mom12_1_5_1"]


def main(cfg):
    out = Path(cfg.get("out", SR / "surge_l8"))
    bl = pd.read_parquet(SR / "surge_bench_long.parquet"); tq_s = load_symbol(SR / "surge_av_cache" / "TQQQ.parquet")["C"]; qq_s = load_symbol(SR / "surge_av_cache" / "QQQ.parquet")["C"]
    ext = {"금(선물)": pd.read_parquet(SR / "gold_gcf_2000_2026-10-24.parquet")["GOLD"]}
    for sym, nm in (("TLT", "장기 국채 TLT"), ("IEF", "중기 국채 IEF")):               # 배당 포함 조정 종가(없던 기간은 현금)
        ext[nm] = pd.read_parquet(SR / "surge_av_cache_long" / f"{sym}.parquet")["adj"]
    irx = pd.read_parquet(SR / "irx_1985_2026-10-24.parquet")["irx"]; ext["현금 이자(^IRX)"] = (1 + irx / 252).cumprod()
    L = ["# L8 — H43 이 현금인 몫에 다른 것을 넣기", "", "H43 비중 w → TQQQ w + (1−w) × X. 세전. 2017 이전은 근사 패널·합성 TQQQ.", ""]
    res = []
    for wname, (tag, long_) in WINDOWS.items():
        curves = pd.read_parquet(SR / f"{tag}_curves.parquet"); days = curves.index
        tq, qq = (bl["TQQQ_C"], bl["QQQ_C"]) if long_ else (tq_s, qq_s)
        tq, qq = tq[tq.index <= days[-1]], qq[qq.index <= days[-1]]
        w = package_frame(qq, tq)["target"].astype(float); wp = w.shift(1).reindex(days).fillna(1.0)
        r_t = tq.pct_change().reindex(days).fillna(0.0); r_q = qq.pct_change().reindex(days).fillna(0.0); r_t.iloc[0] = r_q.iloc[0] = 0.0
        cost = 0.002 * w.diff().abs().shift(1).reindex(days).fillna(0.0)
        base = wp * r_t - cost; eq = {"TQQQ 보유": (1 + r_t).cumprod(), "H43 (현금)": (1 + base).cumprod(), "H43 + QQQ": (1 + base + (1 - wp) * r_q).cumprod()}
        for a in ALTS:
            if a in curves:
                ra = curves[a].pct_change().fillna(0.0); eq[f"H43 + 주도주 {a}"] = (1 + base + (1 - wp) * ra).cumprod(); eq[f"(참고) 주도주 {a} 단독"] = curves[a] / curves[a].iloc[0]
        for nm, px in ext.items():
            rx = px.pct_change().reindex(days).fillna(0.0); eq[f"H43 + {nm}"] = (1 + base + (1 - wp) * rx).cumprod()
        both = 0.5 * ext["금(선물)"].pct_change().reindex(days).fillna(0.0) + 0.5 * ext["중기 국채 IEF"].pct_change().reindex(days).fillna(0.0)
        eq["H43 + 금·IEF 반반"] = (1 + base + (1 - wp) * both).cumprod()
        h = curve_stats(eq["H43 (현금)"]); t = curve_stats(eq["TQQQ 보유"])
        L += [f"## {wname} ({days[0].date()}~{days[-1].date()}) — H43 현금 비중 평균 {(1 - wp).mean():.0%}", "", "| 방법 | 배수 | CAGR | MDD | TQQQ 대비 | H43 대비 |", "|---|---|---|---|---|---|"]
        for name, e in eq.items():
            s = curve_stats(e); res.append({"window": wname, "name": name, **s, "vs_tqqq": s["mult"] / t["mult"], "vs_h43": s["mult"] / h["mult"]})
            L.append(f"| {name} | ×{s['mult']:.2f} | {s['cagr']:+.0%} | {s['mdd']:+.0%} | {s['mult'] / t['mult']:.2f} | {s['mult'] / h['mult']:.2f} |")
        L.append("")
    df = pd.DataFrame(res); p = df[df["name"].str.startswith("H43 +")].pivot(index="name", columns="window", values="vs_h43")[list(WINDOWS)]
    L += ["## H43 대비 요약 (1 보다 크면 현금보다 나음)", "", "| X | " + " | ".join(WINDOWS) + " | 네 창 모두 ≥ 1 |", "|---|" + "---|" * (len(WINDOWS) + 1)]
    L += [f"| {n} | " + " | ".join(f"{v:.2f}" for v in r) + f" | {'예' if (r >= 1).all() else '-'} |" for n, r in zip(p.index, p.to_numpy())]
    Path(f"{out}.md").write_text("\n".join(L)); df.to_json(f"{out}.json", orient="records", force_ascii=False, indent=1); print("\n".join(L))


if __name__ == "__main__":
    main(json.loads(sys.argv[1]) if len(sys.argv) > 1 else {})
