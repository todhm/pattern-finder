"""급등주 탐색 S4b — S4 후보의 리밸런스 시작일 민감도(분기 교체는 시작일 운이 큰지) + 문턱 이웃.

실행: docker compose exec backtester python surge_s4b_cli.py '{"out": "/app/sweep_results/surge_s4b"}'
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from strategy.adapters.surge_events import load_symbol
from strategy.adapters.surge_rank_portfolio import RankRules, bench_table, run_rank, scores, stats, wide

SR = Path("/app/sweep_results")


def main(cfg):
    out = Path(cfg.get("out", SR / "surge_s4b")); start = cfg.get("start", "2018-02-01"); end = cfg.get("end", "2023-12-31")
    panel = pd.read_parquet(SR / "surge_panel.parquet", columns=["date", "symbol", "C", "O", "raw_c", "dv"])
    w = wide(panel); del panel
    c = w["C"]; r = c.pct_change(fill_method=None)
    up = (r > 0).where(r.notna()).rolling(252, min_periods=200).mean()
    tq = load_symbol(SR / "surge_av_cache" / "TQQQ.parquet")["O"]; q = load_symbol(SR / "surge_av_cache" / "QQQ.parquet")["C"].reindex(c.index).ffill()
    mkt = {"-": None, "QQQ 고점 -18% 이내": q / q.rolling(252).max() - 1 > -0.18}
    sc = scores(w, "mom12_1"); days = c.index[c.index >= start]
    offs = list(range(0, 63, 7))
    L = ["# S4b — 시작일 민감도·문턱 이웃", "", f"{start}~{end} · 12-1 모멘텀 · $10·$50M · 20일 +50% 초과 제외 · 시작일을 7 거래일씩 밀어 {len(offs)} 번. 'TQQQ 대비' 는 같은 체결일들의 TQQQ 와의 배수 비.", "",
         "| 오른 날 문턱 | 시장 필터 | K | 간격 | 배수 중앙 | 최소 | 최대 | TQQQ 대비 중앙 | 최소 | 최대 | TQQQ 이긴 시작일 | MDD 최악 |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    res = []
    for thr, (mn, mk), k, every in itertools.product((None, 0.51, 0.53, 0.55), mkt.items(), (5, 10, 20), (63, 126)):
        ex = None if thr is None else up >= thr
        m, v, d = [], [], []
        for o in offs:
            t = run_rank(w, RankRules(k=k, min_price=10.0, min_dv20=5e7, max_ret20=0.5, every=every), str(days[o].date()), end, score=sc, extra_ok=ex, market_ok=mk)
            s = stats(t); b = stats(bench_table(tq, t)); m.append(s["mult"]); v.append(s["mult"] / b["mult"]); d.append(s["mdd"])
        row = {"thr": thr, "market": mn, "k": k, "every": every, "mult_med": float(np.median(m)), "mult_min": min(m), "mult_max": max(m), "vs_med": float(np.median(v)), "vs_min": min(v), "vs_max": max(v), "beat": float(np.mean(np.array(v) > 1)), "mdd_worst": min(d)}
        res.append(row)
        L.append(f"| {'-' if thr is None else f'{thr:.0%}'} | {mn} | {k} | {every} | ×{row['mult_med']:.2f} | ×{row['mult_min']:.2f} | ×{row['mult_max']:.2f} | {row['vs_med']:.2f} | {row['vs_min']:.2f} | {row['vs_max']:.2f} | {row['beat']:.0%} | {row['mdd_worst']:+.0%} |")
        print(L[-1], file=sys.stderr, flush=True)
    Path(f"{out}.md").write_text("\n".join(L)); pd.DataFrame(res).to_json(f"{out}.json", orient="records", force_ascii=False, indent=1); print("\n".join(L))


if __name__ == "__main__":
    main(json.loads(sys.argv[1]) if len(sys.argv) > 1 else {})
