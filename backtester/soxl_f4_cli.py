"""SOXL 사이클 탐색 F4 — "많이 빠졌을 때 나눠 사서, 회복하면 판다" (낙폭 사다리 + 회복 매도).

 매수: SOX 가 지난 3 년 고점 대비 −d1/−d2/−d3 에 닿을 때마다 1/n 씩 SOXL 매수(사이클당 칸마다 1 회).
 매도(전량): recover = SOX 가 사다리를 시작한 고점을 회복 · trend = 회복 뒤 200일선 이탈 · x3/x5 = 평균 매수가 대비 SOXL 3·5 배
            · ma200 = 첫 매수 뒤 SOX 가 200일선 위로 올라섰다가 다시 깨면
 쉬는 동안: 현금 | TQQQ H43 패키지.
신호는 전날 종가, 체결은 당일 종가. 세전, 비용 0.2%. 실행: docker compose exec backtester python soxl_f4_cli.py '{}'
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from strategy.adapters.surge_rank_portfolio import curve_stats
from strategy.adapters.tqqq_package import package_frame

SR = Path("/app/sweep_results")
WINDOWS = {"1995~2009": ("1995-01-03", "2009-12-31"), "2010~2017": ("2010-01-04", "2017-12-29"), "2018~2023": ("2018-01-02", "2023-12-29")}
LADDERS = {"30/40/50": (0.3, 0.4, 0.5), "40/50/60": (0.4, 0.5, 0.6), "50/60/70": (0.5, 0.6, 0.7), "25/35/45": (0.25, 0.35, 0.45), "40 한 번": (0.4,), "50 한 번": (0.5,), "60 한 번": (0.6,)}
EXITS = ("recover", "trend", "ma200", "x3", "x5")


def ladder_weights(sox: pd.Series, soxl: pd.Series, levels: tuple, exit_rule: str, look: int = 756) -> tuple[pd.Series, list]:
    """그날 종가 기준 목표 비중(0~1)과 거래 기록. 호출 쪽에서 shift(1) 해서 쓴다."""
    s, x = sox.to_numpy(), soxl.to_numpy(); hi = sox.rolling(look, min_periods=200).max().to_numpy(); ma = sox.rolling(200).mean().to_numpy()
    n = len(levels); w = np.zeros(len(s)); filled = 0; peak = np.nan; cost = 0.0; above_seen = False; recovered = False; log = []
    for t in range(len(s)):
        if np.isnan(hi[t]) or np.isnan(ma[t]):
            continue
        if filled == 0:
            peak = hi[t]
        dd = s[t] / peak - 1.0
        while filled < n and dd <= -levels[filled]:
            cost = (cost * filled + x[t]) / (filled + 1); filled += 1; log.append((sox.index[t], f"매수 {filled}/{n}", dd))
        if filled:
            above_seen |= s[t] > ma[t]; recovered |= s[t] >= peak
            out = {"recover": recovered, "trend": recovered and s[t] < ma[t], "ma200": above_seen and s[t] < ma[t], "x3": x[t] >= 3 * cost, "x5": x[t] >= 5 * cost}[exit_rule]
            if out:
                log.append((sox.index[t], "매도", x[t] / cost - 1.0)); filled = 0; above_seen = recovered = False
        w[t] = filled / n
    return pd.Series(w, index=sox.index), log


def main(cfg):
    out = Path(cfg.get("out", SR / "soxl_f4")); end = cfg.get("end", "2023-12-29")
    b = pd.read_parquet(SR / "soxl_bench_1994.parquet"); b = b[b.index <= end]; sox, soxl = b["SOX"], b["SOXL"]; days = b.index
    irx = pd.read_parquet(SR / "irx_1985_2026-10-24.parquet")["irx"].reindex(days).ffill().fillna(0.0) / 252.0; ret = soxl.pct_change().fillna(0.0)
    bl = pd.read_parquet(SR / "surge_bench_long.parquet"); bl = bl[bl.index <= end]; tq = bl["TQQQ_C"].reindex(days).ffill(); rt = tq.pct_change().fillna(0.0)
    hw = package_frame(bl["QQQ_C"], bl["TQQQ_C"])["target"].astype(float).shift(1).reindex(days).fillna(1.0); r_h43 = hw * rt + (1 - hw) * irx - 0.002 * hw.diff().abs().fillna(0.0)

    def stats(e):
        o = {}
        for wn, (a, z) in WINDOWS.items():
            s = e[(e.index >= a) & (e.index <= z)]; st = curve_stats(s / s.iloc[0]); o[wn] = (st["mult"], st["mdd"])
        return o

    base = {"SOXL 보유": stats((1 + ret).cumprod()), "TQQQ H43 패키지(1999-11~)": stats((1 + r_h43).cumprod())}; h43 = base["TQQQ H43 패키지(1999-11~)"]
    L = [f"# F4 — 낙폭 사다리 매수 + 회복 매도 (개발 ~{end})", "", "값 = 세전 배수(MDD). 굵게 = 그 창에서 TQQQ H43 을 넘음.", "",
         "| 기준선 | " + " | ".join(WINDOWS) + " |", "|---|---|---|---|"] + [f"| {k} | " + " | ".join(f"×{m:,.2f} ({d:+.0%})" for m, d in v.values()) + " |" for k, v in base.items()]
    L += ["", "| 사다리(SOX 낙폭 %) | 매도 | 쉬는 동안 | " + " | ".join(WINDOWS) + " | SOXL 비중 | 사이클 수 | 사이클 승률 |", "|---|---|---|---|---|---|---|---|---|"]; res = []
    for (ln, lv), ex, idle in itertools.product(LADDERS.items(), EXITS, ("현금", "TQQQ H43")):
        w0, log = ladder_weights(sox, soxl, lv, ex); w = w0.shift(1).fillna(0.0)
        r = w * ret + (1 - w) * (irx if idle == "현금" else r_h43) - 0.002 * w.diff().abs().fillna(0.0); st = stats((1 + r).cumprod())
        sells = [g for _, k, g in log if k == "매도"]; ok = [st[wn][0] > h43[wn][0] for wn in WINDOWS]
        res.append({"ladder": ln, "exit": ex, "idle": idle, **{f"{wn}_mult": st[wn][0] for wn in WINDOWS}, **{f"{wn}_mdd": st[wn][1] for wn in WINDOWS}, "hold": float(w.mean()), "cycles": len(sells), "win": float(np.mean([g > 0 for g in sells])) if sells else np.nan, "n_ok": sum(ok)})
        L.append(f"| {ln} | {ex} | {idle} | " + " | ".join((f"**×{st[wn][0]:,.2f}**" if o else f"×{st[wn][0]:,.2f}") + f" ({st[wn][1]:+.0%})" for wn, o in zip(WINDOWS, ok)) + f" | {w.mean():.0%} | {len(sells)} | {res[-1]['win']:.0%} |")
    df = pd.DataFrame(res); L += ["", f"세 창 모두 TQQQ H43 을 넘은 칸 {int((df['n_ok'] == 3).sum())} / {len(df)} · 두 창 {int((df['n_ok'] == 2).sum())} / {len(df)}"]
    Path(f"{out}.md").write_text("\n".join(L)); df.to_json(f"{out}.json", orient="records", force_ascii=False, indent=1); print("\n".join(L))


if __name__ == "__main__":
    main(json.loads(sys.argv[1]) if len(sys.argv) > 1 else {})
