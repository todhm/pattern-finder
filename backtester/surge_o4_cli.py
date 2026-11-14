"""옵션 탐색 O4 — "H43 + 전량 보유 중 QQQ 콜 더 사기"(O3) 를 옵션 호가가 없는 2001~2010 에 모형 가격으로 대략 점검.

콜 값 = 블랙-숄즈(변동성 = VXN × iv_mult, 금리 ^IRX), 살 때 ×(1+spread)·팔 때 ×(1−spread). 행사가는 목표 델타에서 역산.
먼저 2011-03~2023-12 에서 실제 호가 결과(O3)와 모형 결과를 나란히 놓아 모형이 쓸 만한지 본다 → 그다음 2001~2009·2010~2017.
2010-02 이전 TQQQ 는 합성(surge_bench_long). 모형 점검은 참고용이다 — 판정은 실제 호가 구간으로.
실행: docker compose exec backtester python surge_o4_cli.py '{}'
"""

from __future__ import annotations

import itertools
import json
import math
import sys
from statistics import NormalDist
from pathlib import Path

import numpy as np
import pandas as pd

from strategy.adapters.surge_rank_portfolio import curve_stats
from strategy.adapters.tqqq_package import package_frame

SR = Path("/app/sweep_results")
_N = NormalDist()
WINDOWS = {"2001~2009": ("2001-02-01", "2009-12-31"), "2010~2017": ("2010-02-12", "2017-12-29"), "2018~2023": ("2018-01-02", "2023-12-29"), "실제 호가 구간 2011-03~2023": ("2011-03-23", "2023-12-29")}


def bs_call(s: float, k: float, t: float, r: float, v: float) -> float:
    if t <= 0:
        return max(s - k, 0.0)
    d1 = (math.log(s / k) + (r + 0.5 * v * v) * t) / (v * math.sqrt(t))
    return s * _N.cdf(d1) - k * math.exp(-r * t) * _N.cdf(d1 - v * math.sqrt(t))


def strike_for_delta(s: float, t: float, r: float, v: float, delta: float) -> float:
    return s * math.exp(-_N.inv_cdf(delta) * v * math.sqrt(t) + (r + 0.5 * v * v) * t)


def run(tq: pd.Series, qq: pd.Series, w: pd.Series, vxn: pd.Series, irx: pd.Series, delta: float, dte: int, roll: int, budget: float, iv_mult: float, spread: float, stock_cost: float = 0.002) -> pd.Series:
    """w = 전날 기준 H43 비중. 전량 보유(w ≥ 1)일 때만 자산의 budget 로 QQQ 콜을 사고 roll 일 남으면 갈아탄다."""
    days = tq.index; T, Q, W, V, R = tq.to_numpy(), qq.to_numpy(), w.to_numpy(), vxn.to_numpy() * iv_mult, irx.to_numpy()
    dn = days.to_numpy().astype("datetime64[D]").astype(np.int64); shares, cash = 0.0, 1.0; pos = None; eq = np.empty(len(days)); prev_w = np.nan
    for t in range(len(days)):
        if t:
            cash *= 1.0 + R[t] / 252.0
        mark = bs_call(Q[t], pos[0], (pos[1] - dn[t]) / 365.0, R[t], V[t]) if pos else 0.0
        event = t == 0 or W[t] != prev_w
        if pos and (W[t] < 1.0 or pos[1] - dn[t] <= roll):
            cash += pos[2] * mark * (1 - spread); pos = None; mark = 0.0; event = True
        total = shares * T[t] + cash + (pos[2] * mark if pos else 0.0)
        if pos is None and W[t] >= 1.0 and budget > 0:
            k = strike_for_delta(Q[t], dte / 365.0, R[t], V[t], delta); px = bs_call(Q[t], k, dte / 365.0, R[t], V[t]); q = budget * total / (px * (1 + spread))
            cash -= q * px * (1 + spread); pos = (k, dn[t] + dte, q); mark = px; event = True; total = shares * T[t] + cash + q * px
        if event:
            tgt = W[t] * max(total - (pos[2] * mark if pos else 0.0), 0.0) / T[t]
            if tgt > shares:
                tgt = shares + (tgt - shares) / (1 + stock_cost)
            cash -= (tgt - shares) * T[t] + abs(tgt - shares) * T[t] * stock_cost; shares = tgt
        prev_w = W[t]; eq[t] = shares * T[t] + cash + (pos[2] * mark if pos else 0.0)
    return pd.Series(eq, index=days)


def main(cfg):
    out = Path(cfg.get("out", SR / "surge_o4")); iv_mult = cfg.get("iv_mult", [0.9, 1.0, 1.1]); spread = cfg.get("spread", [0.02, 0.05])
    bl = pd.read_parquet(SR / "surge_bench_long.parquet"); vx = pd.read_parquet(SR / "vxn_2001_2026-11-10.parquet")["vxn"]; irx = pd.read_parquet(SR / "irx_1985_2026-10-24.parquet")["irx"]
    cells = cfg.get("cells", [[0.3, 90, 30, 0.02], [0.3, 90, 30, 0.05], [0.3, 90, 30, 0.10], [0.5, 90, 30, 0.02], [0.5, 90, 30, 0.05], [0.5, 90, 30, 0.10], [0.5, 365, 180, 0.05]])
    actual = {(r["name"]): r["vs_h43"] for r in json.load(open(SR / "surge_o3_boost_qqq.json")) if r["fill"] == 1.0} if (SR / "surge_o3_boost_qqq.json").exists() else {}
    L = ["# O4 — H43 + 보유 중 QQQ 콜 더 사기: 모형 가격으로 2001~ 점검", "",
         "콜 값 = 블랙-숄즈(변동성 VXN × 배수, 금리 ^IRX), 살 때·팔 때 한쪽 비용 spread. 값은 **H43 대비 배수 비율**(1 보다 크면 H43 을 이김). 2010-02 이전 TQQQ 는 합성.", ""]
    res = []
    for wname, (a, b) in WINDOWS.items():
        end = pd.Timestamp(b); tq = bl["TQQQ_C"][bl.index <= end]; qq = bl["QQQ_C"][bl.index <= end]
        w = package_frame(qq, tq)["target"].astype(float).shift(1); days = tq.index[tq.index >= a]
        args = (tq.reindex(days), qq.reindex(days), w.reindex(days).fillna(1.0), vx.reindex(days).ffill().bfill(), irx.reindex(days).ffill().fillna(0.0))
        h = run(*args, 0.5, 90, 30, 0.0, 1.0, 0.0); hs = curve_stats(h); ts = curve_stats(args[0] / args[0].iloc[0])
        L += [f"## {wname} — TQQQ ×{ts['mult']:.2f}(MDD {ts['mdd']:+.0%}) · H43 ×{hs['mult']:.2f}(MDD {hs['mdd']:+.0%})", "",
              "| 콜 | " + " | ".join(f"변동성 ×{m}·비용 {s:.0%}" for m, s in itertools.product(iv_mult, spread)) + " | MDD(가운데) | 실제 호가(O3) |", "|---|" + "---|" * (len(iv_mult) * len(spread) + 2)]
        for d, dte, roll, bud in cells:
            row = []; mdd = None
            for m, s in itertools.product(iv_mult, spread):
                e = run(*args, d, dte, roll, bud, m, s); st = curve_stats(e); row.append(st["mult"] / hs["mult"])
                res.append({"window": wname, "delta": d, "dte": dte, "budget": bud, "iv_mult": m, "spread": s, **st, "vs_h43": row[-1]})
                if m == 1.0 and s == spread[-1]:
                    mdd = st["mdd"]
            act = actual.get(f"U H43 + 보유 중 콜 Δ{d} {dte}일, 교체 때마다 자산의 {bud:.0%}") if wname.startswith("실제") else None
            L.append(f"| Δ{d}·{dte}일·{bud:.0%} | " + " | ".join(f"{x:.2f}" for x in row) + f" | {mdd:+.0%} | {'-' if act is None else f'{act:.2f}'} |")
            print(L[-1], file=sys.stderr, flush=True)
        L.append("")
    Path(f"{out}.md").write_text("\n".join(L)); pd.DataFrame(res).to_json(f"{out}.json", orient="records", force_ascii=False, indent=1); print("\n".join(L))


if __name__ == "__main__":
    main(json.loads(sys.argv[1]) if len(sys.argv) > 1 else {})
