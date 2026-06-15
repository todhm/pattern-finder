import asyncio, os
from datetime import timedelta
import pandas as pd, asyncpg

async def main():
    c = await asyncpg.connect(os.getenv("DATABASE_URL"))
    g = await c.fetch("SELECT date,symbol,final_grade,final_score FROM us_stock_grade")
    mn = await c.fetchval("SELECT min(date) FROM us_stock_grade")
    mx = await c.fetchval("SELECT max(date) FROM us_stock_grade")
    syms = [r["symbol"] for r in await c.fetch("SELECT DISTINCT symbol FROM us_stock_grade")]
    d = await c.fetch("SELECT symbol,date,close FROM us_daily WHERE symbol=ANY($1::text[]) AND date>=$2 AND date<=$3", syms, mn, mx+timedelta(days=130))
    await c.close()
    gdf = pd.DataFrame(g, columns=["date","symbol","final_grade","final_score"])
    gdf["date"]=pd.to_datetime(gdf["date"]); gdf["final_score"]=pd.to_numeric(gdf["final_score"],errors="coerce")
    dd = pd.DataFrame(d, columns=["symbol","date","close"])
    dd["date"]=pd.to_datetime(dd["date"]); dd["close"]=dd["close"].astype(float)
    dd=dd.sort_values(["symbol","date"]).reset_index(drop=True)
    gg=dd.groupby("symbol",sort=False)["close"]
    for n in (5,20,60): dd[f"f{n}"]=gg.shift(-n)/dd["close"]-1
    m=gdf.merge(dd[["symbol","date","f5","f20","f60"]],on=["symbol","date"],how="left")
    fwds=["f5","f20","f60"]
    print(f"dates={m['date'].nunique()} ({m['date'].min().date()}~{m['date'].max().date()})\n")
    print("=== 강력매수 버킷 (%, 전체 rows) ===")
    sb=m[m["final_grade"]=="강력 매수"]
    for f in fwds:
        s=sb[f].dropna()*100
        print(f"  {f}: mean={s.mean():+6.2f}  median={s.median():+6.2f}  n={len(s)}")
    print("\n=== final_score 상위 N 집중 (매 등급일 동일가중 -> 일자평균) ===")
    mm=m.dropna(subset=["final_score"]).copy()
    for N in (3,5,10):
        print(f"  -- top {N} --")
        for f in fwds:
            pd_=mm.sort_values("final_score",ascending=False).groupby("date").head(N).dropna(subset=[f]).groupby("date")[f].mean()*100
            print(f"     {f}: mean={pd_.mean():+6.2f}  median={pd_.median():+6.2f}  ndates={len(pd_)}")
    print("\n=== baseline: 매 등급일 전체 평균 ===")
    for f in fwds:
        pd_=m.dropna(subset=[f]).groupby("date")[f].mean()*100
        print(f"  {f}: mean={pd_.mean():+6.2f}  median={pd_.median():+6.2f}")

asyncio.run(main())
