"""2차 탐색 L1 — 주도주 끝까지 들고 가기 (칸 포트폴리오, 추세 이탈 때만 매도) vs TQQQ.

적격 = 소거 4 개(주가 $10·거래대금 $50M·20일 +50% 이하·오른 날 ≥ 51%) + 200일선 위 + 52주 고점 near 이내, 순위 12-1 모멘텀.
시작일 운을 없애려 시작일을 7 거래일씩 9 번 밀어 돌리고 곡선을 평균(계단식)한다.
실행: docker compose exec backtester python surge_l1_cli.py '{"out": "/app/sweep_results/surge_l1"}'
봉인: '{"start": "2024-01-02", "end": "2026-11-10", "mid": "2025-06-30", "only": [...], "out": ".../surge_l1_sealed"}' — 장부에 열람 기록 후.
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from strategy.adapters.leader_hold import HoldRules, run_leader_hold
from strategy.adapters.surge_events import load_symbol
from strategy.adapters.surge_rank_portfolio import curve_stats, scores, wide

SR = Path("/app/sweep_results")
EXITS = {"200일선": ("ma", 200), "100일선": ("ma", 100), "50일선": ("ma", 50), "고점 -25%": ("trail", 0.25), "고점 -35%": ("trail", 0.35), "200일선 또는 -35%": ("both", 200, 0.35)}


def load(end: str, thr: float = 0.51, near: float = 0.10, min_dv: float = 5e7):
    panel = pd.read_parquet(SR / "surge_panel.parquet", columns=["date", "symbol", "C", "O", "raw_c", "dv"])
    panel = panel[panel["date"] <= end]                                  # end 뒤 가격은 읽지 않는다
    w = wide(panel); del panel
    c = w["C"]; r = c.pct_change(fill_method=None)
    up = (r > 0).where(r.notna()).rolling(252, min_periods=200).mean(); dv20 = w["dv"].rolling(20, min_periods=10).mean()
    hi = c.rolling(252, min_periods=200).max(); s200 = c.rolling(200, min_periods=150).mean()
    base = w["alive"] & (w["raw_c"] >= 10.0) & (dv20 >= min_dv) & ((c / c.shift(20) - 1.0) <= 0.5) & (up >= thr)
    elig = base & (c > s200) & (c >= hi * (1 - near))
    keep = elig.any()
    w = {k: v.loc[:, keep] for k, v in w.items()}; c = w["C"]
    ma = {n: c.rolling(n, min_periods=int(n * 0.75)).mean().to_numpy() for n in (50, 100, 200)}
    return w, elig.loc[:, keep].to_numpy(), base.loc[:, keep].to_numpy(), scores(w, "mom12_1").to_numpy(), ma


def main(cfg):
    out = Path(cfg.get("out", SR / "surge_l1")); start = cfg.get("start", "2018-02-01"); end = cfg.get("end", "2023-12-31"); mid = pd.Timestamp(cfg.get("mid", "2020-12-31"))
    n_st = int(cfg.get("starts", 9)); step = int(cfg.get("step", 7)); seeds = int(cfg.get("seeds", 5))
    grid = cfg.get("grid", {"slots": [3, 5, 10], "exit": list(EXITS)}); only = cfg.get("only")
    w, elig, base, sc, ma = load(end, near=float(cfg.get("near", 0.10)))
    idx = w["C"].index; c = w["C_ff"].to_numpy(); o = w["O_ff"].to_numpy(); alive = w["alive"].to_numpy()
    s0 = int(np.flatnonzero(idx >= start)[0]); e0 = len(idx) - 1; days = idx[s0:]
    tq = load_symbol(SR / "surge_av_cache" / "TQQQ.parquet"); qq = load_symbol(SR / "surge_av_cache" / "QQQ.parquet")

    def bench(f):
        e = f["C"].reindex(days).ffill() / f["O"].reindex(days).bfill().iloc[1]; e.iloc[0] = 1.0
        return e

    def halves(eq):
        m = eq[eq.index <= mid].iloc[-1]
        return float(m / eq.iloc[0]), float(eq.iloc[-1] / m)

    def run(slots, ex, rng_seed=None, el=elig):
        kind = EXITS[ex]; rules = HoldRules(slots=slots, trail=kind[1] if kind[0] == "trail" else (kind[2] if kind[0] == "both" else None))
        line = ma[kind[1]] if kind[0] in ("ma", "both") else None
        curves, mults, held = [], [], []
        for i in range(n_st):
            st = s0 + i * step
            eq, tr = run_leader_hold(c, o, alive, el, sc, rules, st, e0, exit_line=line, rng=None if rng_seed is None else np.random.default_rng(1000 * rng_seed + i))
            full = pd.Series(1.0, index=days); full.iloc[i * step:] = eq; curves.append(full)
            bt_i = bt.iloc[-1] / bt.iloc[i * step]; mults.append((eq[-1], eq[-1] / bt_i))
            held += [(x - en if x >= 0 else e0 - en) for _, en, x, _ in tr]
        avg = pd.concat(curves, axis=1).mean(axis=1)
        return avg, np.array(mults), float(np.mean(held)), len(held) / n_st

    bt, bq = bench(tq), bench(qq); st_, sq = curve_stats(bt), curve_stats(bq); ht = halves(bt)
    pm = lambda x: f"{x:+.0%}"  # noqa: E731
    L = [f"# L1 — 주도주 끝까지 들고 가기 ({start}~{end})", "",
         "적격 = 주가 $10·거래대금 $50M·20일 +50% 이하·지난 1년 오른 날 ≥ 51%·200일선 위·52주 고점 -10% 이내, 12-1 모멘텀 순. 칸이 비면 다음날 시가 매수, 이탈 조건이 뜨면 다음날 시가 매도. 한쪽 비용 0.25%, 세전, 배당 미포함.",
         f"시작일 {n_st} 개({step} 거래일 간격)의 곡선 평균. **TQQQ** ×{st_['mult']:.2f} (CAGR {pm(st_['cagr'])}, MDD {pm(st_['mdd'])}, 전반 ×{ht[0]:.2f}·후반 ×{ht[1]:.2f}) · **QQQ** ×{sq['mult']:.2f} (MDD {pm(sq['mdd'])})", "",
         "| 칸 | 매도 조건 | 배수 | CAGR | MDD | 전반 | 후반 | TQQQ 대비 | 전/후반 대비 | 시작일별 배수(최소~최대) | TQQQ 이긴 시작일 | 평균 보유일 | 거래 수 | 무작위 중앙 | 무작위 이긴% |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    res, curves = [], {"TQQQ": bt, "QQQ": bq}
    for slots, ex in itertools.product(grid["slots"], grid["exit"]):
        if only and [slots, ex] not in only:
            continue
        eq, m, hold, ntr = run(slots, ex); s = curve_stats(eq); h = halves(eq)
        rm = np.array([curve_stats(run(slots, ex, rng_seed=sd)[0])["mult"] for sd in range(seeds)]) if seeds else np.array([np.nan])
        row = {"slots": slots, "exit": ex, **s, "h1": h[0], "h2": h[1], "vs": s["mult"] / st_["mult"], "vs1": h[0] / ht[0], "vs2": h[1] / ht[1], "min": float(m[:, 0].min()), "max": float(m[:, 0].max()),
               "beat_starts": float((m[:, 1] > 1).mean()), "hold": hold, "trades": ntr, "rnd_med": float(np.median(rm)), "rnd_beat": float((s["mult"] > rm).mean())}
        res.append(row); curves[f"{slots}_{ex}"] = eq
        L.append(f"| {slots} | {ex} | ×{row['mult']:.2f} | {pm(row['cagr'])} | {pm(row['mdd'])} | ×{row['h1']:.2f} | ×{row['h2']:.2f} | {row['vs']:.2f} | {row['vs1']:.2f} / {row['vs2']:.2f} | ×{row['min']:.2f}~×{row['max']:.2f} | {row['beat_starts']:.0%} | {hold:.0f} | {ntr:.0f} | ×{row['rnd_med']:.2f} | {row['rnd_beat']:.0%} |")
        print(L[-1], file=sys.stderr, flush=True)
    df = pd.DataFrame(res); cand = df[(df["vs"] > 1) & (df["beat_starts"] > 0.5)]
    L += ["", f"**느슨한 후보선(배수 > TQQQ & 시작일 과반) 통과: {len(cand)} / {len(df)}** · TQQQ 의 0.9 배 이상인 칸 {int((df['vs'] >= 0.9).sum())} / {len(df)}"]
    Path(f"{out}.md").write_text("\n".join(L)); df.to_json(f"{out}.json", orient="records", force_ascii=False, indent=1); pd.DataFrame(curves).to_parquet(f"{out}_curves.parquet"); print("\n".join(L))


if __name__ == "__main__":
    main(json.loads(sys.argv[1]) if len(sys.argv) > 1 else {})
