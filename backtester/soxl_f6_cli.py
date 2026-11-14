"""SOXL 사이클 탐색 F6 — 공포가 치솟았다 가라앉을 때 SOXL 을 사서 M 개월 보유(그 밖에는 TQQQ H43 또는 현금).

 진입: 최근 20 거래일 안에 VIX 종가 ≥ X 가 있었고, 오늘 VIX 가 자기 10일 평균 아래(공포가 가라앉기 시작) — 전날 종가 기준.
 청산: 진입 뒤 M 개월. (재진입은 청산 뒤 새 공포 신호부터)
VIX = FRED VIXCLS(1990~, 당일 종가 그대로 발표·수정 없음). 세전, 비용 0.2%.
실행: docker compose exec backtester python soxl_f6_cli.py '{}'
"""

from __future__ import annotations

import itertools
import json
import os
import sys
from pathlib import Path

import pandas as pd
import requests

from strategy.adapters.surge_rank_portfolio import curve_stats
from strategy.adapters.tqqq_package import package_frame

SR = Path("/app/sweep_results")
WINDOWS = {"1999-11~2009": ("1999-11-02", "2009-12-31"), "2010~2017": ("2010-01-04", "2017-12-29"), "2018~2023": ("2018-01-02", "2023-12-29")}


def vix() -> pd.Series:
    p = SR / "vix_1990_2026-11-10.parquet"
    if not p.exists():
        j = requests.get("https://api.stlouisfed.org/fred/series/observations", params={"series_id": "VIXCLS", "api_key": os.environ["FRED_API_KEY"], "file_type": "json"}, timeout=120).json()
        pd.Series({pd.Timestamp(o["date"]): float(o["value"]) for o in j["observations"] if o["value"] != "."}).to_frame("vix").to_parquet(p)
    return pd.read_parquet(p)["vix"]


def main(cfg):
    out = Path(cfg.get("out", SR / "soxl_f6")); end = cfg.get("end", "2023-12-29")
    b = pd.read_parquet(SR / "soxl_bench_1994.parquet"); b = b[b.index <= end]; bl = pd.read_parquet(SR / "surge_bench_long.parquet"); bl = bl[bl.index <= end]
    days = b.index[b.index >= "1999-11-02"].intersection(bl.index); irx = pd.read_parquet(SR / "irx_1985_2026-10-24.parquet")["irx"].reindex(days).ffill().fillna(0.0) / 252.0
    rs = b["SOXL"].pct_change().reindex(days).fillna(0.0); rt = bl["TQQQ_C"].pct_change().reindex(days).fillna(0.0)
    hw = package_frame(bl["QQQ_C"], bl["TQQQ_C"])["target"].astype(float).shift(1).reindex(days).fillna(1.0); r_h43 = hw * rt + (1 - hw) * irx - 0.002 * hw.diff().abs().fillna(0.0)
    v = vix().reindex(days).ffill(); calm = v < v.rolling(10).mean()

    def stats(r):
        e = (1 + r).cumprod(); o = {}
        for wn, (a, z) in WINDOWS.items():
            s = e[(e.index >= a) & (e.index <= z)]; st = curve_stats(s / s.iloc[0]); o[wn] = (st["mult"], st["mdd"])
        return o

    h43 = stats(r_h43)
    L = [f"# F6 — 공포(VIX)가 치솟았다 가라앉을 때 SOXL (1999-11~{end})", "", "값 = 세전 배수(MDD). 굵게 = 그 창에서 TQQQ H43 이상.", "", "| 기준선 | " + " | ".join(WINDOWS) + " |", "|---|---|---|---|",
         "| TQQQ H43 패키지 | " + " | ".join(f"×{m:,.2f} ({d:+.0%})" for m, d in h43.values()) + " |", "",
         "| VIX 문턱 | 보유 개월 | 쉬는 동안 | " + " | ".join(WINDOWS) + " | SOXL 비중 | 진입 수 | 진입 뒤 평균 수익 | 이긴 진입 |", "|---|---|---|---|---|---|---|---|---|---|"]; res = []
    for x, m, idle in itertools.product((30, 35, 40, 45), (3, 6, 12), ("TQQQ H43", "현금")):
        sig = ((v.rolling(20).max() >= x) & calm).shift(1).fillna(False); w = pd.Series(0.0, index=days); until = pd.Timestamp("1900-01-01"); trades = []
        for d in days[sig.to_numpy()]:
            if d > until:
                until = d + pd.DateOffset(months=m); seg = (days >= d) & (days <= until); w[seg] = 1.0; trades.append(float((1 + rs[seg]).prod() - 1))
        r = w * rs + (1 - w) * (r_h43 if idle == "TQQQ H43" else irx) - 0.002 * w.diff().abs().fillna(0.0); st = stats(r); ok = [st[wn][0] >= h43[wn][0] for wn in WINDOWS]
        res.append({"vix": x, "months": m, "idle": idle, **{f"{wn}_mult": st[wn][0] for wn in WINDOWS}, **{f"{wn}_mdd": st[wn][1] for wn in WINDOWS}, "soxl": float(w.mean()), "n": len(trades), "avg": sum(trades) / max(len(trades), 1), "win": sum(t > 0 for t in trades) / max(len(trades), 1), "n_ok": sum(ok)})
        L.append(f"| {x} | {m} | {idle} | " + " | ".join((f"**×{st[wn][0]:,.2f}**" if o else f"×{st[wn][0]:,.2f}") + f" ({st[wn][1]:+.0%})" for wn, o in zip(WINDOWS, ok)) + f" | {w.mean():.0%} | {len(trades)} | {res[-1]['avg']:+.0%} | {res[-1]['win']:.0%} |")
    df = pd.DataFrame(res); L += ["", f"세 창 모두 TQQQ H43 이상 {int((df['n_ok'] == 3).sum())} / {len(df)} · 두 창 {int((df['n_ok'] == 2).sum())} / {len(df)}"]
    Path(f"{out}.md").write_text("\n".join(L)); df.to_json(f"{out}.json", orient="records", force_ascii=False, indent=1); print("\n".join(L))


if __name__ == "__main__":
    main(json.loads(sys.argv[1]) if len(sys.argv) > 1 else {})
