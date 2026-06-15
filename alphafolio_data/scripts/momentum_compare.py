"""기존 momentum_score vs 새 momentum_score(52주 신고가 근접도) vs max(둘) 비교.

새 momentum_score = 날짜별 near52h(close/252일최고가) 퍼센타일 x 100 (0~100, 기존과 동일 스케일).
combined = max(기존, 새것).  각각 forward 수익률 rank-IC 비교 (기존이 있는 2019 grade 윈도우).
"""
import asyncio
import os
from datetime import timedelta

import pandas as pd
import asyncpg


async def main():
    c = await asyncpg.connect(os.getenv("DATABASE_URL"))
    g = await c.fetch("SELECT date,symbol,momentum_score FROM us_stock_grade")
    mn = await c.fetchval("SELECT min(date) FROM us_stock_grade")
    mx = await c.fetchval("SELECT max(date) FROM us_stock_grade")
    syms = [r["symbol"] for r in await c.fetch("SELECT DISTINCT symbol FROM us_stock_grade")]
    d = await c.fetch(
        "SELECT symbol,date,close FROM us_daily "
        "WHERE symbol=ANY($1::text[]) AND date>=$2 AND date<=$3",
        syms, mn - timedelta(days=400), mx + timedelta(days=130))
    await c.close()

    gdf = pd.DataFrame(g, columns=["date", "symbol", "old_mom"])
    gdf["date"] = pd.to_datetime(gdf["date"])
    gdf["old_mom"] = pd.to_numeric(gdf["old_mom"], errors="coerce")
    dd = pd.DataFrame(d, columns=["symbol", "date", "close"])
    dd["date"] = pd.to_datetime(dd["date"]); dd["close"] = dd["close"].astype(float)
    dd = dd.sort_values(["symbol", "date"]).reset_index(drop=True)
    gg = dd.groupby("symbol", sort=False)["close"]
    dd["near52h"] = dd["close"] / gg.transform(lambda s: s.rolling(252, min_periods=120).max())
    dd["fwd20"] = gg.shift(-20) / dd["close"] - 1
    dd["fwd60"] = gg.shift(-60) / dd["close"] - 1

    m = gdf.merge(dd[["symbol", "date", "near52h", "fwd20", "fwd60"]],
                  on=["symbol", "date"], how="left")
    m["new_mom"] = m.groupby("date")["near52h"].rank(pct=True) * 100
    m["max_mom"] = m[["old_mom", "new_mom"]].max(axis=1)

    v = m.dropna(subset=["fwd20"])

    def ic(col, fwd):
        x = v[[col, fwd]].dropna()
        return x[col].rank().corr(x[fwd].rank())

    print(f"rows={len(m)}, with fwd20={len(v)}\n")
    print(f"{'score':12} {'IC(fwd20)':>10} {'IC(fwd60)':>10}")
    for col in ["old_mom", "new_mom", "max_mom"]:
        print(f"{col:12} {ic(col,'fwd20'):>+10.4f} {ic(col,'fwd60'):>+10.4f}")

    print("\n=== 스케일 분포 ===")
    print(m[["old_mom", "new_mom", "max_mom"]].describe().round(1).T.to_string())

    print("\n=== max_mom 10분위별 평균 fwd20 (%) (9=최고) ===")
    vv = v.dropna(subset=["max_mom"]).copy()
    vv["dec"] = pd.qcut(vv["max_mom"], 10, labels=False, duplicates="drop")
    print((vv.groupby("dec")["fwd20"].mean() * 100).round(2).to_string())


asyncio.run(main())
