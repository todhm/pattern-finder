"""레짐 적응형을 momentum 외 value/quality/growth/blend 신호로도 검증."""
import asyncio
import os
import numpy as np
import pandas as pd
import asyncpg


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
    panel_end = df["date"].max(); cutoff = panel_end - pd.Timedelta(days=30)
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

    rdf = df[df["date"].isin(rebal) & df["liquid"]].dropna(subset=["near52h", "ret"])
    rdf = rdf.merge(fdf, on=["symbol", "date"], how="left")
    rdf["mom"] = rdf["near52h"]
    rdf["val"] = np.where(rdf["per"] > 0, 1.0 / rdf["per"], np.nan)
    rdf["qual"] = rdf["roe"]
    rdf["grw"] = rdf["eg"]

    s = pd.DataFrame(spy, columns=["date", "spy"])
    s["date"] = pd.to_datetime(s["date"]); s["spy"] = s["spy"].astype(float)
    s = s.sort_values("date")
    s["spy200"] = s["spy"].rolling(200, min_periods=100).mean()
    regime = s.assign(bull=s["spy"] > s["spy200"]).set_index("date")["bull"]

    signals = ["mom", "val", "qual", "grw", "blend"]
    eq = {f"{sig}_{mode}": 1.0 for sig in signals for mode in ("adapt", "static")}
    eq["market"] = 1.0
    curve = {k: [] for k in eq}

    for dt in rebal:
        day = rdf[rdf["date"] == dt]
        if len(day) < 50 or dt not in regime.index or pd.isna(regime.loc[dt]):
            continue
        bull = bool(regime.loc[dt])
        day = day.copy()
        ranks = []
        for sig in ("mom", "val", "qual", "grw"):
            day[f"r_{sig}"] = day[sig].rank(pct=True)
            ranks.append(f"r_{sig}")
        day["blend"] = day[ranks].mean(axis=1)
        eq["market"] *= (1 + day["ret"].mean())
        curve["market"].append(eq["market"])
        for sig in signals:
            v = day.dropna(subset=[sig])
            if len(v) < 30:
                continue
            v = v.sort_values(sig)
            k = max(1, len(v) // 10)
            top = v.tail(k)["ret"].mean(); bot = v.head(k)["ret"].mean()
            eq[f"{sig}_static"] *= (1 + top)
            eq[f"{sig}_adapt"] *= (1 + (top if bull else bot))
            curve[f"{sig}_static"].append(eq[f"{sig}_static"])
            curve[f"{sig}_adapt"].append(eq[f"{sig}_adapt"])

    span = (rebal[-1] - rebal[0]).days / 365.25
    print(f"기간 {rebal[0].date()}~{rebal[-1].date()} ({span:.1f}y)\n")
    print(f"{'strategy':14} {'total%':>10} {'CAGR%':>8} {'MDD%':>8}")

    def stats(k):
        a = np.array(curve[k])
        if len(a) == 0:
            return
        cagr = a[-1] ** (1 / span) - 1
        peak = np.maximum.accumulate(a); mdd = ((a - peak) / peak).min()
        print(f"{k:14} {(a[-1]-1)*100:>10.1f} {cagr*100:>8.1f} {mdd*100:>8.1f}")

    for sig in signals:
        stats(f"{sig}_adapt"); stats(f"{sig}_static")
    stats("market")


asyncio.run(main())
