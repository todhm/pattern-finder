"""SOXL 사이클 탐색 F0 — 반도체 사이클 지도(SOX 고점·저점)와 기준선.

 ① SOX 를 지그재그(하락 ≥ dd, 반등으로 되돌림)로 잘라 고점·저점 목록 → `soxl_turns.json`
 ② "미래를 아는 사람" 상한(저점 매수·고점 매도)과 그 n 개월 늦은 판
 ③ 가격만 쓰는 기준선: SOXL 보유, SOX 200일선, H43 규칙을 SOX/SOXL 에 그대로 적용, TQQQ 보유·H43
1994~2010-03 SOXL 은 합성(일간 3 배 − 조달 − 보수). 세전·현금 이자 ^IRX. 신호는 전날 종가, 체결은 당일 종가(전날 신호 shift 1).
실행: docker compose exec backtester python soxl_f0_cli.py '{}'
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from strategy.adapters.surge_rank_portfolio import curve_stats
from strategy.adapters.tqqq_package import package_frame

SR = Path("/app/sweep_results")
WINDOWS = {"1994~2009": ("1995-01-03", "2009-12-31"), "2010~2017": ("2010-01-04", "2017-12-29"), "2018~2023": ("2018-01-02", "2023-12-29"), "개발 전체 1995~2023": ("1995-01-03", "2023-12-29")}


def zigzag(px: pd.Series, dd: float) -> list[tuple[pd.Timestamp, str, float]]:
    """고점 대비 −dd 하락이 확정되면 그 고점을 '고점', 저점 대비 1/(1−dd)−1 상승이 확정되면 그 저점을 '저점' 으로 기록."""
    up = 1.0 / (1.0 - dd) - 1.0; out = []; mode = "up"; ext_i = 0; v = px.to_numpy()
    for i in range(1, len(v)):
        if mode == "up":
            if v[i] > v[ext_i]:
                ext_i = i
            elif v[i] <= v[ext_i] * (1 - dd):
                out.append((px.index[ext_i], "고점", float(v[ext_i]))); mode = "down"; ext_i = i
        else:
            if v[i] < v[ext_i]:
                ext_i = i
            elif v[i] >= v[ext_i] * (1 + up):
                out.append((px.index[ext_i], "저점", float(v[ext_i]))); mode = "up"; ext_i = i
    return out


def curve(ret: pd.Series, w: pd.Series, cash: pd.Series, cost: float = 0.002) -> pd.Series:
    """w = 그날 들고 있는 비중(이미 전날 신호로 shift 된 것)."""
    r = w * ret + (1 - w) * cash - cost * w.diff().abs().fillna(0.0)
    return (1 + r).cumprod()


def main(cfg):
    out = Path(cfg.get("out", SR / "soxl_f0")); dd = float(cfg.get("dd", 0.30))
    b = pd.read_parquet(SR / "soxl_bench_1994.parquet"); sox, soxl = b["SOX"], b["SOXL"]
    irx = pd.read_parquet(SR / "irx_1985_2026-10-24.parquet")["irx"].reindex(b.index).ffill().fillna(0.0) / 252.0
    bl = pd.read_parquet(SR / "surge_bench_long.parquet")
    turns = zigzag(sox, dd); json.dump([[str(d.date()), k, p] for d, k, p in turns], open(f"{out}_turns.json", "w"), ensure_ascii=False, indent=1)
    L = ["# F0 — 반도체 사이클 지도와 기준선", "", f"SOX(1994-05~, 2001-07 이후 SOXX 배당 포함)에서 하락 ≥ {dd:.0%} 로 확정되는 고점·저점. SOXL 은 2010-03 이전 합성.", "",
         "## ① 사이클 지도", "", "| 고점 | 저점 | SOX 하락 | 기간(달) | 같은 구간 SOXL | 다음 고점까지 SOX 상승 | 기간(달) | 같은 구간 SOXL |", "|---|---|---|---|---|---|---|---|"]
    pk = [(d, p) for d, k, p in turns if k == "고점"]; tr = [(d, p) for d, k, p in turns if k == "저점"]
    for i, (pd_, pp) in enumerate(pk):
        t = next(((d, p) for d, p in tr if d > pd_), None)
        if t is None:
            L.append(f"| {pd_.date()} | (진행 중) | | | | | | |"); continue
        nx = next(((d, p) for d, p in pk if d > t[0]), (sox.index[-1], float(sox.iloc[-1])))
        L.append(f"| {pd_.date()} | {t[0].date()} | {t[1] / pp - 1:+.0%} | {(t[0] - pd_).days / 30.4:.0f} | {soxl[t[0]] / soxl[pd_] - 1:+.0%} | {nx[1] / t[1] - 1:+.0%} | {(nx[0] - t[0]).days / 30.4:.0f} | ×{soxl[nx[0]] / soxl[t[0]]:.1f} |")
    # ② 상한: 저점~다음 고점 보유
    def oracle(lag_days: int) -> pd.Series:
        w = pd.Series(0.0, index=b.index)
        for d, k, _ in turns:
            w[b.index >= d + pd.Timedelta(days=lag_days)] = 1.0 if k == "저점" else 0.0
        if turns and turns[0][1] == "고점":
            w[b.index < turns[0][0] + pd.Timedelta(days=lag_days)] = 1.0
        return w
    ret = soxl.pct_change().fillna(0.0); rs = sox.pct_change().fillna(0.0)
    sma = lambda n: (sox > sox.rolling(n).mean()).astype(float).shift(1).fillna(0.0)  # noqa: E731
    h43_sox = package_frame(sox, soxl)["target"].astype(float).shift(1).fillna(1.0)
    W = {"SOXL 보유": pd.Series(1.0, index=b.index), "SOX 200일선 위일 때만 SOXL": sma(200), "SOX 100일선 위일 때만 SOXL": sma(100),
         "H43 규칙을 SOX·SOXL 에 그대로": h43_sox, "(상한) 저점 매수·고점 매도": oracle(0), "(상한) 1 달 늦게": oracle(30), "(상한) 2 달 늦게": oracle(61), "(상한) 3 달 늦게": oracle(91), "(상한) 6 달 늦게": oracle(182)}
    eq = {k: curve(ret, w, irx) for k, w in W.items()}; eq["SOX 지수(1 배) 보유"] = (1 + rs).cumprod()
    tq = bl["TQQQ_C"].reindex(b.index).ffill(); eq["TQQQ 보유"] = tq / tq.dropna().iloc[0]
    hw = package_frame(bl["QQQ_C"], bl["TQQQ_C"])["target"].astype(float).shift(1).reindex(b.index).fillna(1.0); eq["TQQQ H43 패키지"] = curve(tq.pct_change().fillna(0.0), hw, irx)
    L += ["", "## ② 기준선과 상한 (세전 배수 · MDD)", "", "| 방법 | " + " | ".join(WINDOWS) + " | SOXL 보유 비중(전체) |", "|---|" + "---|" * (len(WINDOWS) + 1)]
    res = []
    for k, e in eq.items():
        cells = []
        for wn, (a, z) in WINDOWS.items():
            s = e[(e.index >= a) & (e.index <= z)].dropna()
            if len(s) < 200:
                cells.append("-"); continue
            st = curve_stats(s / s.iloc[0]); cells.append(f"×{st['mult']:,.2f} ({st['mdd']:+.0%})"); res.append({"name": k, "window": wn, **st})
        L.append(f"| {k} | " + " | ".join(cells) + f" | {W[k][W[k].index <= '2023-12-29'].mean():.0%} |" if k in W else f"| {k} | " + " | ".join(cells) + " | - |")
    Path(f"{out}.md").write_text("\n".join(L)); pd.DataFrame(res).to_json(f"{out}.json", orient="records", force_ascii=False, indent=1); pd.DataFrame(eq).to_parquet(f"{out}_curves.parquet"); print("\n".join(L))


if __name__ == "__main__":
    main(json.loads(sys.argv[1]) if len(sys.argv) > 1 else {})
