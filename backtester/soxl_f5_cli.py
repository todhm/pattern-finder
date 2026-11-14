"""SOXL 사이클 탐색 F5·F3 — TQQQ H43 을 기본으로 두고 "반도체가 주도할 때만" SOXL 로 갈아타기, 그리고 저점 확인 뒤 달력 보유.

F5 조건(전부 전날 종가 기준, 가격만):
 RS(n)   SOX/QQQ 비율이 자기 n 일 평균 위(반도체가 나스닥보다 강함)
 UP      SOX 가 200일선 위
 CALM(v) SOX 20일 실현 변동성(연율) < v
 ON      H43 이 TQQQ 전량 보유 중
조건이 참이면 SOXL 100%, 아니면 TQQQ H43 그대로. 판정 주기: 매일 | 월말(월말 조건을 다음 달 내내).
F3: SOX 저점이 확정(저점 대비 +43%)되면 SOXL 을 사서 M 개월 보유(도중에 H43 이 쉬라고 하면 쉼), 그 밖에는 TQQQ H43.
창은 TQQQ 자료가 있는 1999-11~. 세전, 갈아타기 비용 0.2%. 실행: docker compose exec backtester python soxl_f5_cli.py '{}'
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from soxl_f0_cli import zigzag
from strategy.adapters.surge_rank_portfolio import curve_stats
from strategy.adapters.tqqq_package import package_frame

SR = Path("/app/sweep_results")
WINDOWS = {"1999-11~2009": ("1999-11-02", "2009-12-31"), "2010~2017": ("2010-01-04", "2017-12-29"), "2018~2023": ("2018-01-02", "2023-12-29")}


def main(cfg):
    out = Path(cfg.get("out", SR / "soxl_f5")); end = cfg.get("end", "2023-12-29"); start = cfg.get("start", "1999-11-02")
    b = pd.read_parquet(SR / "soxl_bench_1994.parquet"); b = b[b.index <= end]; bl = pd.read_parquet(SR / "surge_bench_long.parquet"); bl = bl[bl.index <= end]
    days = b.index[b.index >= start].intersection(bl.index); sox_all, soxl = b["SOX"], b["SOXL"]; qq = bl["QQQ_C"].reindex(b.index).ffill()
    irx = pd.read_parquet(SR / "irx_1985_2026-10-24.parquet")["irx"].reindex(days).ffill().fillna(0.0) / 252.0
    rs_ = soxl.pct_change().reindex(days).fillna(0.0); rt = bl["TQQQ_C"].pct_change().reindex(days).fillna(0.0)
    hw = package_frame(bl["QQQ_C"], bl["TQQQ_C"])["target"].astype(float).shift(1).reindex(days).fillna(1.0); r_h43 = hw * rt + (1 - hw) * irx - 0.002 * hw.diff().abs().fillna(0.0)
    ratio = sox_all / qq; vol = sox_all.pct_change().rolling(20).std() * np.sqrt(252)
    C = {f"RS{n}": (ratio > ratio.rolling(n).mean()) for n in (50, 100, 200)}; C["UP"] = sox_all > sox_all.rolling(200).mean()
    C.update({f"CALM{int(v * 100)}": vol < v for v in (0.3, 0.4, 0.5)}); C = {k: v.shift(1).reindex(days).fillna(False).astype(bool) for k, v in C.items()}; C["ON"] = hw >= 1.0

    def stats(r):
        e = (1 + r).cumprod(); o = {}
        for wn, (a, z) in WINDOWS.items():
            s = e[(e.index >= a) & (e.index <= z)]; st = curve_stats(s / s.iloc[0]); o[wn] = (st["mult"], st["mdd"])
        return o

    def switch(cond: pd.Series) -> pd.Series:
        c = cond.astype(float); return c * rs_ + (1 - c) * r_h43 - 0.002 * c.diff().abs().fillna(0.0)

    monthly = lambda c: c.groupby([c.index.year, c.index.month]).transform("last").shift(1).fillna(False).groupby([c.index.year, c.index.month]).transform("first").astype(bool)  # noqa: E731  지난달 말 값
    base = {"SOXL 보유": stats(rs_), "TQQQ 보유": stats(rt), "TQQQ H43 패키지": stats(r_h43)}; h43 = base["TQQQ H43 패키지"]
    L = [f"# F5·F3 — 반도체가 주도할 때만 SOXL, 아니면 TQQQ H43 ({start}~{end})", "", "값 = 세전 배수(MDD). 굵게 = 그 창에서 TQQQ H43 이상.", "",
         "| 기준선 | " + " | ".join(WINDOWS) + " |", "|---|---|---|---|"] + [f"| {k} | " + " | ".join(f"×{m:,.2f} ({d:+.0%})" for m, d in v.values()) + " |" for k, v in base.items()]
    L += ["", "## F5 — 조건이 참이면 SOXL", "", "| 조건 | 주기 | " + " | ".join(WINDOWS) + " | SOXL 비중 | 갈아탄 횟수 |", "|---|---|---|---|---|---|---|"]; res = []
    combos = [("ON", r) for r in ("RS50", "RS100", "RS200")] + [("ON", r, "UP") for r in ("RS50", "RS100", "RS200")] + [("ON", "UP", c) for c in ("CALM30", "CALM40", "CALM50")] \
        + [("ON", r, "UP", c) for r, c in itertools.product(("RS100", "RS200"), ("CALM30", "CALM40", "CALM50"))] + [("ON", "UP")] + [("RS100", "UP"), ("RS200", "UP"), ("UP", "CALM40")]
    for names, per in itertools.product(combos, ("매일", "월말")):
        cond = pd.Series(True, index=days)
        for n in names:
            cond &= C[n]
        if per == "월말":
            cond = monthly(cond) & (C["ON"] if "ON" in names else True)        # 월말에 고르되 H43 이 쉬라고 하면 즉시 따른다
        st = stats(switch(cond)); ok = [st[wn][0] >= h43[wn][0] for wn in WINDOWS]
        res.append({"rule": " & ".join(names), "period": per, **{f"{wn}_mult": st[wn][0] for wn in WINDOWS}, **{f"{wn}_mdd": st[wn][1] for wn in WINDOWS}, "soxl": float(cond.mean()), "switches": int(cond.astype(int).diff().abs().sum()), "n_ok": sum(ok)})
        L.append(f"| {' & '.join(names)} | {per} | " + " | ".join((f"**×{st[wn][0]:,.2f}**" if o else f"×{st[wn][0]:,.2f}") + f" ({st[wn][1]:+.0%})" for wn, o in zip(WINDOWS, ok)) + f" | {cond.mean():.0%} | {res[-1]['switches']} |")
    # F3 — 저점 확정 뒤 달력 보유 (확정일은 그날까지의 가격만으로 알 수 있다)
    L += ["", "## F3 — SOX 저점 확정(저점 대비 반등) 뒤 M 개월 SOXL", "", "| 낙폭 기준 | 보유 개월 | " + " | ".join(WINDOWS) + " | SOXL 비중 | 사이클 수 |", "|---|---|---|---|---|---|---|"]
    for dd, m in itertools.product((0.25, 0.30, 0.35), (6, 12, 18, 24, 36)):
        up = 1.0 / (1.0 - dd) - 1.0; turns = zigzag(sox_all, dd); cond = pd.Series(False, index=days); n = 0
        for d, k, p in turns:
            if k != "저점":
                continue
            after = sox_all[(sox_all.index > d)]; hit = after[after >= p * (1 + up)]
            if hit.empty:
                continue
            c0 = hit.index[0]; cond[(days > c0) & (days <= c0 + pd.DateOffset(months=m))] = True; n += 1
        cond &= C["ON"]; st = stats(switch(cond)); ok = [st[wn][0] >= h43[wn][0] for wn in WINDOWS]
        res.append({"rule": f"F3 dd{dd}", "period": f"{m}m", **{f"{wn}_mult": st[wn][0] for wn in WINDOWS}, **{f"{wn}_mdd": st[wn][1] for wn in WINDOWS}, "soxl": float(cond.mean()), "switches": n, "n_ok": sum(ok)})
        L.append(f"| -{dd:.0%} | {m} | " + " | ".join((f"**×{st[wn][0]:,.2f}**" if o else f"×{st[wn][0]:,.2f}") + f" ({st[wn][1]:+.0%})" for wn, o in zip(WINDOWS, ok)) + f" | {cond.mean():.0%} | {n} |")
    df = pd.DataFrame(res); f5 = df[~df["rule"].str.startswith("F3")]; f3 = df[df["rule"].str.startswith("F3")]
    L += ["", f"세 창 모두 TQQQ H43 이상: F5 {int((f5['n_ok'] == 3).sum())} / {len(f5)} · F3 {int((f3['n_ok'] == 3).sum())} / {len(f3)}"]
    Path(f"{out}.md").write_text("\n".join(L)); df.to_json(f"{out}.json", orient="records", force_ascii=False, indent=1); print("\n".join(L))


if __name__ == "__main__":
    main(json.loads(sys.argv[1]) if len(sys.argv) > 1 else {})
