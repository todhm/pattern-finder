"""현재까지 적재된 us_stock_grade 기반 간단 backtest.

등급/final_score/momentum_score 별 향후 수익률(5/20/60d) + rank-IC.
"""
import asyncio
import os
from datetime import timedelta

import pandas as pd
import asyncpg


async def main():
    c = await asyncpg.connect(os.getenv("DATABASE_URL"))
    g = await c.fetch("SELECT date,symbol,final_grade,final_score,momentum_score FROM us_stock_grade")
    mn = await c.fetchval("SELECT min(date) FROM us_stock_grade")
    mx = await c.fetchval("SELECT max(date) FROM us_stock_grade")
    syms = [r["symbol"] for r in await c.fetch("SELECT DISTINCT symbol FROM us_stock_grade")]
    d = await c.fetch(
        "SELECT symbol,date,close FROM us_daily "
        "WHERE symbol=ANY($1::text[]) AND date>=$2 AND date<=$3",
        syms, mn, mx + timedelta(days=130))
    await c.close()

    gdf = pd.DataFrame(g, columns=["date", "symbol", "final_grade", "final_score", "momentum_score"])
    gdf["date"] = pd.to_datetime(gdf["date"])
    for col in ("final_score", "momentum_score"):
        gdf[col] = pd.to_numeric(gdf[col], errors="coerce")
    dd = pd.DataFrame(d, columns=["symbol", "date", "close"])
    dd["date"] = pd.to_datetime(dd["date"]); dd["close"] = dd["close"].astype(float)
    dd = dd.sort_values(["symbol", "date"]).reset_index(drop=True)
    gg = dd.groupby("symbol", sort=False)["close"]
    for n in (5, 20, 60):
        dd[f"f{n}"] = gg.shift(-n) / dd["close"] - 1
    m = gdf.merge(dd[["symbol", "date", "f5", "f20", "f60"]], on=["symbol", "date"], how="left")
    v = m.dropna(subset=["f20"])
    print(f"grade rows={len(m)}, with fwd20={len(v)}, dates={m['date'].nunique()} "
          f"({m['date'].min().date()}~{m['date'].max().date()})")

    order = ["강력 매수", "매수", "매수 고려", "중립", "매도 고려", "매도", "강력 매도"]
    res = (m.groupby("final_grade")[["f5", "f20", "f60"]].mean() * 100).reindex(order).round(2)
    res["n"] = m.groupby("final_grade").size().reindex(order)
    print("\n=== 등급별 평균 forward 수익률 (%) ===")
    print(res.to_string())

    def ic(col, fwd):
        x = v[[col, fwd]].dropna()
        return x[col].rank().corr(x[fwd].rank())
    print("\n=== rank-IC ===")
    print(f"            {'f5':>9} {'f20':>9} {'f60':>9}")
    for col in ("final_score", "momentum_score"):
        print(f"{col:11} {ic(col,'f5'):>+9.4f} {ic(col,'f20'):>+9.4f} {ic(col,'f60'):>+9.4f}")

    m2 = m.dropna(subset=["final_score", "f20"]).copy()
    m2["dec"] = pd.qcut(m2["final_score"], 10, labels=False, duplicates="drop")
    dec = (m2.groupby("dec")[["f20", "f60"]].mean() * 100).round(2)
    print("\n=== final_score 10분위별 forward 수익률 (%) (9=최고점) ===")
    print(dec.to_string())


asyncio.run(main())
