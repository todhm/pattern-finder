"""팩터별 차별화 레짐 규칙 (momentum만 flip, quality/value 유지, growth는 bull 전용)."""
import asyncio
import os
import numpy as np
import pandas as pd
import asyncpg

TRAIN_END = pd.Timestamp("2023-01-01")


async def main():
    c = await asyncpg.connect(os.getenv("DATABASE_URL"))
    d = await c.fetch("SELECT symbol,date,close,volume FROM us_daily WHERE close>0")
    spy = await c.fetch("SELECT date,close FROM us_daily_etf WHERE symbol='SPY' ORDER BY date")
    df = pd.DataFrame(d, columns=["symbol", "date", "close", "volume"])
    df["date"] = pd.to_datetime(df["date"]); df["close"] = df["close"].astype(float)
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    g = df.groupby("symbol", sort=False)
    df["near52h"] = df["close"] / g["close"].transform(lambda s: s.rolling(252, min_periods=120).max())
    df["avol20"] = g["volume"].transform(lambda s: s.rolling(20, min_periods=10).mean())
    cutoff = df["date"].max() - pd.Timedelta(days=30)
    df["exit"] = g["close"].shift(-21)
    df["last_close"] = g["close"].transform("last")
    df["last_date"] = g["date"].transform("max")
    df.loc[df["exit"].isna() & (df["last_date"] < cutoff), "exit"] = df["last_close"]
    df["ret"] = df["exit"] / df["close"] - 1
    df["liquid"] = (df["close"] >= 5) & (df["close"] * df["avol20"] >= 5_000_000)
    me = df.groupby(df["date"].dt.to_period("M"))["date"].max()
    rebal = [pd.Timestamp(x) for x in sorted(me.values)]
    fac = await c.fetch(
        "SELECT symbol,date,per,returnonequityttm,quarterlyearningsgrowthyoy "
        "FROM us_stock_basic WHERE source='computed' AND date = ANY($1::date[])",
        [r.date() for r in rebal])
    await c.close()
    fdf = pd.DataFrame(fac, columns=["symbol", "date", "per", "roe", "eg"])
    fdf["date"] = pd.to_datetime(fdf["date"])
    for col in ("per", "roe", "eg"):
        fdf[col] = pd.to_numeric(fdf[col], errors="coerce")

    r = df[df["date"].isin(rebal) & df["liquid"]].dropna(subset=["near52h", "ret"]).merge(
        fdf, on=["symbol", "date"], how="left")
    r["mom"] = r["near52h"]
    r["val"] = np.where(r["per"] > 0, 1.0 / r["per"], np.nan)
    r["qual"] = r["roe"]
    r["grw"] = r["eg"]
    for f in ("mom", "val", "qual", "grw"):
        r[f"r_{f}"] = r.groupby("date")[f].rank(pct=True).fillna(0.5)
    s = pd.DataFrame(spy, columns=["date", "spy"])
    s["date"] = pd.to_datetime(s["date"]); s["spy"] = s["spy"].astype(float)
    s = s.sort_values("date")
    s["bull"] = s["spy"] > s["spy"].rolling(200, min_periods=100).mean()
    r["bull"] = r["date"].map(s.set_index("date")["bull"])
    r = r.dropna(subset=["bull"])

    print("=== 팩터별 평균 21d 수익% (bull top/bot | bear top/bot) ===")
    for f in ("mom", "val", "qual", "grw"):
        out = []
        for b in (True, False):
            sub = r[r["bull"] == b].dropna(subset=[f])
            tops, bots = [], []
            for dt, day in sub.groupby("date"):
                if len(day) < 30:
                    continue
                day = day.sort_values(f); k = max(1, len(day) // 10)
                tops.append(day.tail(k)["ret"].mean()); bots.append(day.head(k)["ret"].mean())
            out.append((np.mean(tops) * 100, np.mean(bots) * 100))
        print(f"  {f:5}  bull {out[0][0]:5.2f}/{out[0][1]:5.2f}  |  bear {out[1][0]:5.2f}/{out[1][1]:5.2f}")

    def comp_for(day, mode):
        if mode == "static":
            return day["r_mom"] + day["r_val"] + day["r_qual"] + day["r_grw"]
        bull = bool(day["bull"].iloc[0])
        if mode == "uniform":
            base = day["r_mom"] + day["r_val"] + day["r_qual"] + day["r_grw"]
            return base if bull else (4 - base)
        if bull:
            return day["r_mom"] + day["r_val"] + day["r_qual"] + day["r_grw"]
        return (1 - day["r_mom"]) + day["r_val"] + day["r_qual"]

    def equity(mask, mode):
        dr = []
        for dt, day in r[mask].groupby("date"):
            if len(day) < 50:
                continue
            day = day.copy(); day["c"] = comp_for(day, mode)
            day = day.sort_values("c"); k = max(1, len(day) // 10)
            dr.append((dt, day.tail(k)["ret"].mean()))
        dr.sort()
        vals = [x[1] for x in dr]; dates = [x[0] for x in dr]
        eq = np.cumprod([1 + v for v in vals])
        span = (dates[-1] - dates[0]).days / 365.25
        cagr = eq[-1] ** (1 / span) - 1
        peak = np.maximum.accumulate(eq); mdd = ((eq - peak) / peak).min()
        return cagr, mdd

    print("\n=== 전략별 CAGR (train -> test), MDD(test) ===")
    tr = r["date"] < TRAIN_END; te = r["date"] >= TRAIN_END
    for mode in ("static", "uniform", "differentiated"):
        ctr, _ = equity(tr, mode); cte, mdde = equity(te, mode)
        print(f"  {mode:14} train {ctr*100:6.1f}%  ->  test {cte*100:6.1f}%  (MDD {mdde*100:5.1f}%)")
    mdr = [(dt, day["ret"].mean()) for dt, day in r[te].groupby("date") if len(day) >= 50]
    mdr.sort(); mv = np.cumprod([1 + x[1] for x in mdr])
    msp = (mdr[-1][0] - mdr[0][0]).days / 365.25
    print(f"  {'market':14}                          test {(mv[-1]**(1/msp)-1)*100:6.1f}%")


asyncio.run(main())
