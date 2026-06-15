import asyncio
import os
from datetime import timedelta

import pandas as pd
import asyncpg


async def main():
    c = await asyncpg.connect(os.getenv("DATABASE_URL"))
    cols = "date,symbol,final_score,value_score,quality_score,momentum_score,growth_score"
    g = await c.fetch(f"SELECT {cols} FROM us_stock_grade")
    mn = await c.fetchval("SELECT min(date) FROM us_stock_grade")
    mx = await c.fetchval("SELECT max(date) FROM us_stock_grade")
    syms = [r["symbol"] for r in await c.fetch("SELECT DISTINCT symbol FROM us_stock_grade")]
    d = await c.fetch(
        "SELECT symbol,date,close FROM us_daily "
        "WHERE symbol=ANY($1::text[]) AND date>=$2 AND date<=$3",
        syms, mn, mx + timedelta(days=130))
    await c.close()

    gdf = pd.DataFrame(g, columns=cols.split(","))
    gdf["date"] = pd.to_datetime(gdf["date"])
    scores = ["final_score", "value_score", "quality_score", "momentum_score", "growth_score"]
    for s in scores:
        gdf[s] = pd.to_numeric(gdf[s], errors="coerce")
    dd = pd.DataFrame(d, columns=["symbol", "date", "close"])
    dd["date"] = pd.to_datetime(dd["date"]); dd["close"] = dd["close"].astype(float)
    dd = dd.sort_values(["symbol", "date"]).reset_index(drop=True)
    gg = dd.groupby("symbol", sort=False)["close"]
    for H in (5, 10, 20):
        dd[f"f{H}"] = gg.shift(-H) / dd["close"] - 1
    m = gdf.merge(dd[["symbol", "date", "f5", "f10", "f20"]], on=["symbol", "date"], how="left")
    gdates = sorted(m["date"].unique())

    res = []
    for score in scores:
        for direction in ("top", "bottom"):
            asc = (direction == "bottom")
            for N in (1, 3, 5, 10):
                for H in (5, 10, 20):
                    fcol = f"f{H}"
                    eq = 1.0
                    used = 0
                    for dt in gdates[::H]:
                        sub = m[m["date"] == dt].dropna(subset=[score, fcol])
                        if len(sub) < N:
                            continue
                        pick = sub.sort_values(score, ascending=asc).head(N)
                        eq *= (1 + pick[fcol].mean())
                        used += 1
                    if used < 3:
                        continue
                    days = used * H
                    ann = eq ** (252 / days) - 1
                    res.append((ann, eq - 1, score, direction, N, H, used, days))
    res.sort(reverse=True)
    print(f"{'annual%':>11} {'total%':>9} {'score':14} {'dir':7} {'N':>2} {'H':>3} {'rebal':>5} {'days':>5}")
    for ann, tot, score, direction, N, H, used, days in res[:18]:
        print(f"{ann*100:>11.0f} {tot*100:>9.1f} {score:14} {direction:7} {N:>2} {H:>3} {used:>5} {days:>5}")


asyncio.run(main())
