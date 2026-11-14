"""급등주 탐색 S0 — 패널 조립 + 급등 사건(+50%/+100%) 전수 집계.

실행: docker compose exec backtester python surge_s0_cli.py '{"out": "/app/sweep_results/surge_s0"}'
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from strategy.adapters.surge_events import build_panel, liquid, real_surge

SR = Path("/app/sweep_results")


def listing() -> pd.DataFrame:
    rows = []
    for name in ("surge_delisted_list.txt", "surge_active_list.txt"):
        for ln in (SR / name).read_text().splitlines():
            p = ln.split(",")
            if len(p) >= 4:
                rows.append({"symbol": p[0], "ipo_date": p[1], "delisting_date": p[2], "exchange": p[3]})
    return pd.DataFrame(rows).drop_duplicates("symbol")


def pct(x): return f"{x:+.1%}"


def main(cfg):
    out = Path(cfg.get("out", SR / "surge_s0")); panel_path = SR / "surge_panel.parquet"
    if cfg.get("rebuild", True) or not panel_path.exists():
        f = build_panel(SR / "surge_av_cache", listing()); f.to_parquet(panel_path)
    else:
        f = pd.read_parquet(panel_path)
    f["year"] = f["date"].dt.year
    L = [f"# S0 — 급등 사건 전수 집계", "", f"패널: {len(f):,} 행 · {f['symbol'].nunique():,} 종목(보통주) · 폐지 종목 {f.loc[f.delisted, 'symbol'].nunique():,} · {f['date'].min().date()}~{f['date'].max().date()}", ""]
    liq = liquid(f)
    L += [f"체결 가능(전날 원가격 ≥ $1 · 20일 평균 거래대금 ≥ $1M · 최근 분할 없음) 종목-일: {liq.sum():,} ({liq.mean():.0%})", ""]
    for thr in (0.5, 1.0):
        ev_all = f[f["ret"] >= thr]; is_ev = real_surge(f, thr); ev = f[is_ev & liq]
        L += [f"## 하루 종가 +{thr:.0%} 이상", "", f"전체 {len(ev_all):,} 건 → 체결 가능 {len(ev):,} 건 (종목 {ev['symbol'].nunique():,}, 폐지 종목 비중 {ev['delisted'].mean():.0%})", "",
              "| 연도 | 사건 | 체결 가능 종목-일 | 확률(1만 종목-일당) | 갭 중앙 | 다음날 시가 갭 중앙 | 다음날 시가→1일 | →5일 | →20일 | →60일 | 20일 내 최고 중앙 |", "|---|---|---|---|---|---|---|---|---|---|---|"]
        for y, g in ev.groupby("year"):
            den = int(liq[f["year"] == y].sum())
            L.append(f"| {y} | {len(g)} | {den:,} | {len(g) / den * 1e4:.2f} | {pct(g['gap'].median())} | {pct(g['n_open_gap'].median())} | {pct(g['fwd1'].median())} | {pct(g['fwd5'].median())} | {pct(g['fwd20'].median())} | {pct(g['fwd60'].median())} | {pct(g['fwd_max20'].median())} |")
        L += ["", "평균(한 건 +500% 상한): " + " · ".join(f"→{n}일 {pct(ev[f'fwd{n}'].clip(upper=5).mean())}" for n in (1, 5, 20, 60)) + f" · 이긴 비율 →20일 {(ev['fwd20'] > 0).mean():.0%}", ""]
        for col, bins, lab in (("p_raw_c", [1, 2, 5, 10, 20, 50, 1e9], "전날 주가($)"), ("p_dv20", [1e6, 3e6, 1e7, 3e7, 1e8, 1e12], "전날 20일 평균 거래대금($)"),
                               ("p_dd252", [-1, -0.9, -0.7, -0.5, -0.3, -0.1, 0.01], "전날 52주 고점 대비"), ("p_ret20", [-1, -0.3, -0.1, 0, 0.1, 0.3, 1, 100], "직전 20일 수익"),
                               ("p_vol_ratio5", [0, 0.5, 1, 2, 4, 1e9], "직전 5일 거래량 ÷ 60일 평균"), ("p_big_days_252", [-1, 0, 1, 3, 1e9], "지난 1년 +30% 날 수")):
            b = pd.cut(f.loc[liq, col], bins); e = is_ev[liq]
            t = pd.DataFrame({"n": e.groupby(b, observed=True).size(), "ev": e.groupby(b, observed=True).sum()})
            t["확률(1만당)"] = t["ev"] / t["n"] * 1e4
            L += [f"**{lab}** 구간별 급등 확률", "", "| 구간 | 종목-일 | 사건 | 확률(1만당) |", "|---|---|---|---|"] + [f"| {i} | {int(r.n):,} | {int(r.ev)} | {r['확률(1만당)']:.2f} |" for i, r in t.iterrows()] + [""]
        ev.sort_values("date").to_csv(f"{out}_events_{int(thr * 100)}.csv", index=False)
    md = "\n".join(L); Path(f"{out}.md").write_text(md); print(md[:6000])


if __name__ == "__main__":
    main(json.loads(sys.argv[1]) if len(sys.argv) > 1 else {})
