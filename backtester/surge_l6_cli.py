"""2차 탐색 L6 — 내부자 군집 매수 × 낙폭 과대 × 시장 공포 구간, 1 년 보유 칸 포트폴리오 vs TQQQ.

신호 = 30일 안 임원·이사 k 명 이상 공개시장 매수(S6), 공시일 종가 기준 52주 고점 대비 dd 이하, (선택) QQQ 가 200일선 아래.
신호 뒤 5 거래일 동안 적격, 빈 칸이 있으면 다음날 시가 매수(매수자 수 많은 순), 252 거래일 뒤 매도. 남는 현금은 그대로 둔다.
곡선은 시작일 5 개 평균. 단독, 그리고 TQQQ 와 반반(분기 재조정)으로 비교.
실행: docker compose exec backtester python surge_l6_cli.py '{"out": "/app/sweep_results/surge_l6"}'
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from strategy.adapters.insider_signals import buy_windows, first_cross
from strategy.adapters.leader_hold import HoldRules, run_leader_hold
from strategy.adapters.surge_events import load_symbol
from strategy.adapters.surge_rank_portfolio import curve_stats, wide
from surge_l2_cli import mix_curve, roll_beat

SR = Path("/app/sweep_results")


def main(cfg):
    out = Path(cfg.get("out", SR / "surge_l6")); start = cfg.get("start", "2018-02-01"); end = cfg.get("end", "2023-12-31"); mid = pd.Timestamp(cfg.get("mid", "2020-12-31"))
    n_st, step = int(cfg.get("starts", 5)), int(cfg.get("step", 13)); hold = int(cfg.get("hold", 252))
    grid = cfg.get("grid", {"k": [2, 3], "dd": [None, -0.3, -0.5], "fear": [False, True], "slots": [10, 20]})
    tx = pd.read_parquet(SR / "surge_insider_tx.parquet"); tx = tx[tx["filing_date"] <= end]
    panel = pd.read_parquet(SR / "surge_panel.parquet", columns=["date", "symbol", "C", "O", "raw_c", "dv"]); panel = panel[(panel["date"] <= end) & panel["symbol"].isin(set(tx["symbol"]))]
    w = wide(panel); del panel
    c = w["C"]; idx = c.index; cols = {s: j for j, s in enumerate(c.columns)}
    dd = (c / c.rolling(252, min_periods=120).max() - 1.0).to_numpy(); dv20 = w["dv"].rolling(20, min_periods=10).mean().to_numpy()
    liquid = w["alive"].to_numpy() & (w["raw_c"].to_numpy() >= 5.0) & (dv20 >= 5e6)
    tqf = load_symbol(SR / "surge_av_cache" / "TQQQ.parquet"); qc = load_symbol(SR / "surge_av_cache" / "QQQ.parquet")["C"].reindex(idx).ffill()
    fear = (qc < qc.rolling(200, min_periods=150).mean()).to_numpy()
    bw = buy_windows(tx[tx["symbol"].isin(cols)], 30, 1e4, True)
    cc, oo, alive = w["C_ff"].to_numpy(), w["O_ff"].to_numpy(), w["alive"].to_numpy()
    s0 = int(np.flatnonzero(idx >= start)[0]); e0 = len(idx) - 1; days = idx[s0:]
    bt = tqf["C"].reindex(days).ffill(); bt = bt / bt.iloc[0]; r_t = bt.pct_change().fillna(0.0); st = curve_stats(bt)
    half = lambda e: (float(e[e.index <= mid].iloc[-1] / e.iloc[0]), float(e.iloc[-1] / e[e.index <= mid].iloc[-1]))  # noqa: E731
    ht = half(bt); pm = lambda x: f"{x:+.0%}"  # noqa: E731
    L = [f"# L6 — 내부자 군집 매수 × 낙폭 × 공포 구간 ({start}~{end})", "",
         f"30일 안 임원·이사 k 명 이상 매수(건당 $1만·합 $10만 이상), 주가 $5·거래대금 $5M 이상, 신호 뒤 5 거래일 안 빈 칸에 다음날 시가 매수, {hold} 거래일 보유. 한쪽 비용 0.25%, 세전. 시작일 {n_st} 개 평균. 남는 현금은 이자 없이 둔다.",
         f"**TQQQ** ×{st['mult']:.2f} (MDD {pm(st['mdd'])}, 전반 ×{ht[0]:.2f}·후반 ×{ht[1]:.2f})", "",
         "| k | 낙폭 조건 | 공포 구간만 | 칸 | 단독 배수 | MDD | 전반 | 후반 | TQQQ 대비 | 평균 투자 비중 | 거래 수 | TQQQ 50% 와 반반 | 반반 MDD | 반반 TQQQ 대비 | 반반 전/후반 대비 | 1년 창 이긴% |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    res = []
    for k, d, fr, slots in itertools.product(grid["k"], grid["dd"], grid["fear"], grid["slots"]):
        ev = first_cross(bw, k, 1e5, 90); elig = np.zeros(cc.shape, bool); score = np.full(cc.shape, np.nan)
        pos = np.searchsorted(idx.values, ev["date"].to_numpy(), side="right") - 1
        for sym, p, nb in zip(ev["symbol"], pos, ev["n_buyers"]):
            j = cols[sym]
            if p < 252 or not liquid[p, j] or (d is not None and not dd[p, j] <= d) or (fr and not fear[p]):
                continue
            elig[p: p + 5, j] = True; score[p: p + 5, j] = nb
        elig &= alive
        curves, inv, ntr = [], [], []
        for i in range(n_st):
            eq, tr = run_leader_hold(cc, oo, alive, elig, score, HoldRules(slots=slots, max_hold=hold), s0 + i * step, e0)
            full = pd.Series(1.0, index=days); full.iloc[i * step:] = eq; curves.append(full); ntr.append(len(tr))
            inv.append(sum((x if x >= 0 else e0) - en for _, en, x, _ in tr) / (slots * (e0 - s0 - i * step + 1)))
        eq = pd.concat(curves, axis=1).mean(axis=1); s = curve_stats(eq); h = half(eq)
        mx = mix_curve(r_t, eq.pct_change().fillna(0.0), 0.5); sm = curve_stats(mx); hm = half(mx); rb = roll_beat(mx, bt)
        row = {"k": k, "dd": d, "fear": fr, "slots": slots, **s, "h1": h[0], "h2": h[1], "vs": s["mult"] / st["mult"], "invested": float(np.mean(inv)), "trades": float(np.mean(ntr)),
               "mix": sm["mult"], "mix_mdd": sm["mdd"], "mix_vs": sm["mult"] / st["mult"], "mix_vs1": hm[0] / ht[0], "mix_vs2": hm[1] / ht[1], "mix_roll": rb[0]}
        res.append(row)
        L.append(f"| {k} | {'-' if d is None else pm(d)} | {'예' if fr else '-'} | {slots} | ×{s['mult']:.2f} | {pm(s['mdd'])} | ×{h[0]:.2f} | ×{h[1]:.2f} | {row['vs']:.2f} | {row['invested']:.0%} | {row['trades']:.0f} | ×{sm['mult']:.2f} | {pm(sm['mdd'])} | {row['mix_vs']:.2f} | {row['mix_vs1']:.2f} / {row['mix_vs2']:.2f} | {rb[0]:.0%} |")
        print(L[-1], file=sys.stderr, flush=True)
    df = pd.DataFrame(res)
    L += ["", f"단독으로 TQQQ 를 넘은 칸 {int((df['vs'] > 1).sum())} / {len(df)} · TQQQ 와 반반이 TQQQ 를 넘은 칸 {int((df['mix_vs'] > 1).sum())} / {len(df)} (L2 의 주도주 묶음 반반은 1.13~1.19)"]
    Path(f"{out}.md").write_text("\n".join(L)); df.to_json(f"{out}.json", orient="records", force_ascii=False, indent=1); print("\n".join(L))


if __name__ == "__main__":
    main(json.loads(sys.argv[1]) if len(sys.argv) > 1 else {})
