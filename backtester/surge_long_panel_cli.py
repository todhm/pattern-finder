"""2차 탐색 — 1999~2017 근사 패널(제 3 표본) 조립.

`surge_delisted_cli.py` 로 받은 `sweep_results/surge_av_cache_long/`(2015~2017 거래대금 상위 900 + AV 가 아는 2017-10 이전 폐지 보통주)를
`surge_panel.parquet` 과 같은 열(date, symbol, C, O, raw_c, dv)로 묶는다. **근사치다**: AV 폐지 목록은 2015 이전이 드물어(연 40~190)
생존 편향이 남는다. 벤치는 QQQ 로 합성한 3 배(자금 조달 ^IRX·보정 드래그) + 2010-02 부터 실제 TQQQ.
실행: docker compose exec backtester python surge_long_panel_cli.py '{}'
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from strategy.adapters.band_rebalance_strategy import CALIBRATED_DRAG, build_synthetic_leveraged, splice_series
from strategy.adapters.surge_events import is_common, load_symbol

SR = Path("/app/sweep_results")


def main(cfg):
    cache = Path(cfg.get("cache", SR / "surge_av_cache_long")); end = pd.Timestamp(cfg.get("end", "2017-12-31"))
    dl = {}
    for f, names in ((SR / "surge_delisted_list.txt", None), (SR / "surge_long_delisted.csv", "csv")):
        if names is None:
            t = pd.read_csv(f, header=None, names=["symbol", "ipo", "del", "ex"]); dl.update(dict(zip(t["symbol"], pd.to_datetime(t["del"], errors="coerce"))))
        else:
            t = pd.read_csv(f); dl.update(dict(zip(t["symbol"], pd.to_datetime(t["delistingDate"], errors="coerce"))))
    parts = []; n_del = 0
    for fn in sorted(cache.glob("*.parquet")):
        sym = fn.stem
        if sym in ("TQQQ", "QQQ", "SPY") or not is_common(sym):
            continue
        d = load_symbol(fn, delisting_date=dl.get(sym) if pd.notna(dl.get(sym)) else None)
        if d is None:
            continue
        d = d[d.index <= end]
        if len(d) < 30:
            continue
        n_del += int(pd.notna(dl.get(sym)) and dl[sym] <= end)
        parts.append(d[["C", "O", "raw_c", "dv"]].assign(symbol=sym).rename_axis("date").reset_index())
    p = pd.concat(parts, ignore_index=True); p.to_parquet(SR / "surge_panel_long.parquet")
    q = load_symbol(cache / "QQQ.parquet"); t = load_symbol(cache / "TQQQ.parquet")
    irx = pd.read_parquet(SR / "irx_1985_2026-10-24.parquet")["irx"]
    drag = CALIBRATED_DRAG[3.0]
    syn = splice_series(build_synthetic_leveraged(q["C"], 3.0, annual_expense=drag, financing_rate=irx), t["C"])
    b = pd.DataFrame({"TQQQ_C": syn, "TQQQ_O": syn.shift(1), "QQQ_C": q["C"], "QQQ_O": q["O"]}).dropna(); b.to_parquet(SR / "surge_bench_long.parquet")
    print(f"종목 {p['symbol'].nunique():,} (그중 {end.date()} 이전 폐지 {n_del:,}) · {len(p):,} 행 · {p['date'].min().date()}~{p['date'].max().date()} · 벤치 {b.index[0].date()}~ · 합성 TQQQ 드래그 {drag:.4f}")
    y = p.assign(y=p["date"].dt.year).groupby("y")["symbol"].nunique(); print(y.to_string())


if __name__ == "__main__":
    main(json.loads(sys.argv[1]) if len(sys.argv) > 1 else {})
