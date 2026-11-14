"""SOXL 사이클 탐색 F2 — ① F1 의 무작위 대조 ② 역발상 합성 지수(나쁜 지표 비율·개선 지표 비율) × 발표 지연 0·1·2 개월.

① 각 F1 신호의 켜짐/꺼짐 구간 길이를 섞어(보유 비중·왕복 수 보존) 같은 결합으로 돌린 무작위 신호가 "세 창 모두 SOX 200일선 초과" 를 얼마나 자주 하나.
② 핵심 지표 8 종에서  bad = 전년비 < 0 인 비율,  imp = 전년비가 3 달 전보다 나아진 비율  → 규칙 몇 개(사전 고정) × 문턱 0.3/0.5/0.7.
실행: docker compose exec backtester python soxl_f2_cli.py '{}'
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from soxl_f0_cli import curve
from soxl_f1_cli import NAMES, WINDOWS, load_ind, monthly_state
from strategy.adapters.cycle_signals import RULES, features, to_daily
from strategy.adapters.surge_rank_portfolio import curve_stats
from strategy.adapters.tqqq_package import package_frame

SR = Path("/app/sweep_results")
CORE = ["kr_semi_px", "kr_it_exp", "kr_elec_prod", "kr_inv_ship", "us_semi_ip", "us_semi_util", "us_ce_orders", "us_semi_ppi"]


def shuffle_runs(s: np.ndarray, rng) -> np.ndarray:
    """0/1 월간 신호의 구간(run)을 켜짐·꺼짐 번갈아 유지한 채 길이만 섞는다."""
    ch = np.flatnonzero(np.diff(s) != 0) + 1; runs = np.split(s, ch); on = [r for r in runs if r[0] > 0]; off = [r for r in runs if r[0] == 0]
    rng.shuffle(on); rng.shuffle(off); a, b = (on, off) if rng.random() < 0.5 else (off, on); out = []
    for i in range(max(len(a), len(b))):
        out += ([a[i]] if i < len(a) else []) + ([b[i]] if i < len(b) else [])
    return np.concatenate(out)


def main(cfg):
    out = Path(cfg.get("out", SR / "soxl_f2")); end = cfg.get("end", "2023-12-29"); seeds = int(cfg.get("seeds", 40))
    b = pd.read_parquet(SR / "soxl_bench_1994.parquet"); b = b[b.index <= end]; sox, soxl = b["SOX"], b["SOXL"]; days = b.index
    irx = pd.read_parquet(SR / "irx_1985_2026-10-24.parquet")["irx"].reindex(days).ffill().fillna(0.0) / 252.0; ret = soxl.pct_change().fillna(0.0)
    above = (sox > sox.rolling(200).mean()).astype(float).shift(1).fillna(0.0); ind = load_ind()
    bl = pd.read_parquet(SR / "surge_bench_long.parquet"); bl = bl[bl.index <= end]; tq = bl["TQQQ_C"].reindex(days).ffill()
    hw = package_frame(bl["QQQ_C"], bl["TQQQ_C"])["target"].astype(float).shift(1).reindex(days).fillna(1.0)
    h43_soxl = package_frame(sox, soxl)["target"].astype(float).shift(1).fillna(1.0)

    def stats(e):
        o = {}
        for wn, (a, z) in WINDOWS.items():
            s = e[(e.index >= a) & (e.index <= z)].dropna(); st = curve_stats(s / s.iloc[0]); o[wn] = (st["mult"], st["mdd"])
        return o

    comb = {"단독": lambda s: s, "그리고 200일선": lambda s: s * above, "또는 200일선": lambda s: ((s + above) > 0).astype(float)}
    ref = stats(curve(ret, above, irx)); beat3 = lambda w: all(stats(curve(ret, w, irx))[wn][0] > ref[wn][0] for wn in WINDOWS)  # noqa: E731
    base = {"SOXL 보유": stats(curve(ret, above * 0 + 1, irx)), "SOX 200일선": ref, "H43 규칙을 SOXL 에": stats(curve(ret, h43_soxl, irx)), "TQQQ H43 패키지(1999-11~)": stats(curve(tq.pct_change().fillna(0.0), hw, irx))}
    # ① 무작위 대조
    rng = np.random.default_rng(7); real = 0; plc = np.zeros(seeds); n = 0
    for key, rule in itertools.product(NAMES, RULES):
        if key not in ind or (ind[key].dropna().index[0] > pd.Timestamp("2001-01-01") and key != "kr_semi_exp"):
            continue
        m = monthly_state(ind, key, rule)
        for cn, f in comb.items():
            n += 1; real += beat3(f(to_daily(m, days, 1).fillna(0.0).shift(1).fillna(0.0)))
            for sd in range(seeds):
                ms = pd.Series(shuffle_runs(m.to_numpy(), rng), index=m.index); plc[sd] += beat3(f(to_daily(ms, days, 1).fillna(0.0).shift(1).fillna(0.0)))
        print(key, rule, real, plc.mean(), file=sys.stderr, flush=True)
    L = ["# F2 — 무작위 대조와 역발상 합성 지수", "", "## ① F1 의 '세 창 모두 SOX 200일선 초과' 는 우연인가", "",
         f"실제 신호 {real} / {n} · 구간 길이를 섞은 무작위 신호 {seeds} 벌의 평균 {plc.mean():.1f} / {n} (최소 {plc.min():.0f} ~ 최대 {plc.max():.0f}, 실제 이상인 벌 {np.mean(plc >= real):.0%})", "",
         "## ② 역발상 합성 지수", "", f"핵심 지표 8 종: {', '.join(NAMES[k] for k in CORE)}.", "bad = 전년비 < 0 인 지표 비율 · imp = 전년비가 3 달 전보다 나아진 지표 비율. 값 = 세전 배수(MDD). 굵게 = TQQQ H43 과 SOX 200일선 둘 다 넘음.", "",
         "| 기준선 | " + " | ".join(WINDOWS) + " |", "|---|---|---|---|"] + [f"| {k} | " + " | ".join(f"×{m:,.2f} ({d:+.0%})" for m, d in v.values()) + " |" for k, v in base.items()]
    F = {k: features(ind[k]) for k in CORE}
    bad = pd.DataFrame({k: (f["yoy"] < 0).where(f["yoy"].notna()) for k, f in F.items()}).mean(axis=1, skipna=True)
    imp = pd.DataFrame({k: (f["accel3"] > 0).where(f["accel3"].notna()) for k, f in F.items()}).mean(axis=1, skipna=True)
    rules = {"bad ≥ θ (나쁠 때 보유)": lambda t: bad >= t, "bad ≥ θ 또는 200일선": lambda t: bad >= t, "imp ≥ θ (개선 중일 때 보유)": lambda t: imp >= t, "imp ≥ θ 또는 200일선": lambda t: imp >= t, "imp ≥ θ 그리고 200일선": lambda t: imp >= t,
             "(bad ≥ θ 이고 imp ≥ 0.5) 또는 200일선": lambda t: (bad >= t) & (imp >= 0.5), "bad < θ 이면 쉼(좋을 때 팔기), 아니면 200일선": lambda t: bad < t}
    L += ["", "| 규칙 | θ | 지연 | " + " | ".join(WINDOWS) + " | 보유 비중 |", "|---|---|---|---|---|---|---|"]; res = []; h43 = base["TQQQ H43 패키지(1999-11~)"]
    for (rn, fn), th, lag in itertools.product(rules.items(), (0.3, 0.5, 0.7), (0, 1, 2)):
        s = to_daily(fn(th).astype(float), days, lag).fillna(0.0).shift(1).fillna(0.0)
        if rn.startswith("bad < θ"):
            w = above * (1 - s)                                   # 지표가 대부분 좋은 때(bad < θ)는 쉬고, 그 밖엔 200일선
        elif "또는 200일선" in rn:
            w = ((s + above) > 0).astype(float)
        elif "그리고 200일선" in rn:
            w = s * above
        else:
            w = s
        st = stats(curve(ret, w, irx)); ok = [st[wn][0] > max(h43[wn][0], ref[wn][0]) for wn in WINDOWS]
        res.append({"rule": rn, "theta": th, "lag": lag, **{f"{wn}_mult": st[wn][0] for wn in WINDOWS}, **{f"{wn}_mdd": st[wn][1] for wn in WINDOWS}, "hold": float(w.mean()), "n_ok": sum(ok)})
        L.append(f"| {rn} | {th} | {lag} | " + " | ".join((f"**×{st[wn][0]:,.2f}**" if o else f"×{st[wn][0]:,.2f}") + f" ({st[wn][1]:+.0%})" for wn, o in zip(WINDOWS, ok)) + f" | {w.mean():.0%} |")
    df = pd.DataFrame(res); L += ["", f"세 창 모두 TQQQ H43·SOX 200일선을 넘은 칸 {int((df['n_ok'] == 3).sum())} / {len(df)} · 두 창 {int((df['n_ok'] == 2).sum())} / {len(df)}"]
    Path(f"{out}.md").write_text("\n".join(L)); df.to_json(f"{out}.json", orient="records", force_ascii=False, indent=1); print("\n".join(L))


if __name__ == "__main__":
    main(json.loads(sys.argv[1]) if len(sys.argv) > 1 else {})
