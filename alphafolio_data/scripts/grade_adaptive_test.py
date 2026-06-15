"""실제 final_score(us_stock_grade)에 adaptive 레짐전환 적용 시 static 대비 개선되나."""
import asyncio
import os
from datetime import timedelta
import numpy as np
import pandas as pd
import asyncpg


async def main():
    c = await asyncpg.connect(os.getenv("DATABASE_URL"))
    g = await c.fetch("SELECT date,symbol,final_score FROM us_stock_grade")
    mn = await c.fetchval("SELECT min(date) FROM us_stock_grade")
    mx = await c.fetchval("SELECT max(date) FROM us_stock_grade")
    syms = [r["symbol"] for r in await c.fetch("SELECT DISTINCT symbol FROM us_stock_grade")]
    d = await c.fetch(
        "SELECT symbol,date,close,volume FROM us_daily "
        "WHERE symbol=ANY($1::text[]) AND date>=$2 AND date<=$3",
        syms, mn - timedelta(days=60), mx + timedelta(days=60))
    spy = await c.fetch("SELECT date,close FROM us_daily_etf WHERE symbol='SPY' ORDER BY date")
    await c.close()

    gdf = pd.DataFrame(g, columns=["date", "symbol", "final_score"])
    gdf["date"] = pd.to_datetime(gdf["date"])
    gdf["final_score"] = pd.to_numeric(gdf["final_score"], errors="coerce")
    dd = pd.DataFrame(d, columns=["symbol", "date", "close", "volume"])
    dd["date"] = pd.to_datetime(dd["date"]); dd["close"] = dd["close"].astype(float)
    dd["volume"] = pd.to_numeric(dd["volume"], errors="coerce")
    dd = dd.sort_values(["symbol", "date"]).reset_index(drop=True)
    gg = dd.groupby("symbol", sort=False)
    dd["avol20"] = gg["volume"].transform(lambda s: s.rolling(20, min_periods=10).mean())
    cutoff = dd["date"].max() - pd.Timedelta(days=30)
    dd["exit"] = gg["close"].shift(-20)
    dd["last_close"] = gg["close"].transform("last")
    dd["last_date"] = gg["date"].transform("max")
    dd.loc[dd["exit"].isna() & (dd["last_date"] < cutoff), "exit"] = dd["last_close"]
    dd["ret"] = dd["exit"] / dd["close"] - 1
    dd["liquid"] = (dd["close"] >= 5) & (dd["close"] * dd["avol20"] >= 5_000_000)

    s = pd.DataFrame(spy, columns=["date", "spy"])
    s["date"] = pd.to_datetime(s["date"]); s["spy"] = s["spy"].astype(float)
    s = s.sort_values("date")
    s["bull"] = s["spy"] > s["spy"].rolling(200, min_periods=100).mean()
    reg = s.set_index("date")["bull"]

    m = gdf.merge(dd[["symbol", "date", "ret", "liquid"]], on=["symbol", "date"], how="left")
    m = m[m["liquid"]].dropna(subset=["final_score", "ret"])
    m["bull"] = m["date"].map(reg)
    m = m.dropna(subset=["bull"])

    rec = []
    for dt, day in m.groupby("date"):
        if len(day) < 30:
            continue
        bull = bool(day["bull"].iloc[0])
        day = day.sort_values("final_score")
        k = max(1, len(day) // 10)
        top = day.tail(k)["ret"].mean()
        bot = day.head(k)["ret"].mean()
        mkt = day["ret"].mean()
        rec.append((bull, top, (top if bull else bot), mkt))

    rec = pd.DataFrame(rec, columns=["bull", "static", "adaptive", "market"])
    nb = int(rec["bull"].sum()); nbear = len(rec) - nb
    print(f"등급일 {len(rec)}개 (bull={nb}, bear={nbear}), 보유 20일, 유동+생존편향 보정\n")
    print(f"{'strategy':30} {'평균 20d 수익%':>12}")
    print(f"{'static (항상 top decile)':30} {rec['static'].mean()*100:>12.2f}")
    print(f"{'adaptive (bull top/bear bot)':30} {rec['adaptive'].mean()*100:>12.2f}")
    print(f"{'market (전체 평균)':30} {rec['market'].mean()*100:>12.2f}")
    print("\n=== 레짐별 평균 20d 수익% (static / adaptive / market) ===")
    for b, lab in [(True, "bull"), (False, "bear")]:
        sub = rec[rec["bull"] == b]
        if len(sub):
            print(f"  {lab}({len(sub)}): {sub['static'].mean()*100:.2f} / {sub['adaptive'].mean()*100:.2f} / {sub['market'].mean()*100:.2f}")


asyncio.run(main())
