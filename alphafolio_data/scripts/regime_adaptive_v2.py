"""레짐 적응형 momentum (생존편향 제거 + 유동성 필터)."""
import asyncio
import os
import numpy as np
import pandas as pd
import asyncpg


async def main():
    c = await asyncpg.connect(os.getenv("DATABASE_URL"))
    d = await c.fetch("SELECT symbol,date,close,volume FROM us_daily WHERE close>0")
    spy = await c.fetch("SELECT date,close FROM us_daily_etf WHERE symbol='SPY' ORDER BY date")
    await c.close()

    df = pd.DataFrame(d, columns=["symbol", "date", "close", "volume"])
    df["date"] = pd.to_datetime(df["date"])
    df["close"] = df["close"].astype(float)
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    g = df.groupby("symbol", sort=False)
    df["near52h"] = df["close"] / g["close"].transform(lambda s: s.rolling(252, min_periods=120).max())
    df["avol20"] = g["volume"].transform(lambda s: s.rolling(20, min_periods=10).mean())

    panel_end = df["date"].max()
    cutoff = panel_end - pd.Timedelta(days=30)
    df["exit"] = g["close"].shift(-21)
    df["last_close"] = g["close"].transform("last")
    df["last_date"] = g["date"].transform("max")
    delisted = df["last_date"] < cutoff
    df.loc[df["exit"].isna() & delisted, "exit"] = df["last_close"]
    df["ret"] = df["exit"] / df["close"] - 1

    df["dvol"] = df["close"] * df["avol20"]
    df["liquid"] = (df["close"] >= 5) & (df["dvol"] >= 5_000_000)

    s = pd.DataFrame(spy, columns=["date", "spy"])
    s["date"] = pd.to_datetime(s["date"]); s["spy"] = s["spy"].astype(float)
    s = s.sort_values("date")
    s["spy200"] = s["spy"].rolling(200, min_periods=100).mean()
    regime = s.assign(bull=s["spy"] > s["spy200"]).set_index("date")["bull"]

    me = df.groupby(df["date"].dt.to_period("M"))["date"].max()
    rebal = [pd.Timestamp(x) for x in sorted(me.values)]

    eq = {k: 1.0 for k in ["adaptive", "always_momo", "always_rev", "market"]}
    curve = {k: [] for k in eq}
    nb = nbear = 0
    for dt in rebal:
        day = df[(df["date"] == dt) & df["liquid"]].dropna(subset=["near52h", "ret"])
        if len(day) < 50 or dt not in regime.index or pd.isna(regime.loc[dt]):
            continue
        bull = bool(regime.loc[dt]); nb += bull; nbear += (not bull)
        day = day.sort_values("near52h")
        k = max(1, len(day) // 10)
        bot = day.head(k)["ret"].mean()
        top = day.tail(k)["ret"].mean()
        mkt = day["ret"].mean()
        eq["always_momo"] *= (1 + top)
        eq["always_rev"] *= (1 + bot)
        eq["adaptive"] *= (1 + (top if bull else bot))
        eq["market"] *= (1 + mkt)
        for kk in eq:
            curve[kk].append(eq[kk])

    span = (rebal[-1] - rebal[0]).days / 365.25
    print(f"기간 {rebal[0].date()}~{rebal[-1].date()} ({span:.1f}y), "
          f"리밸 {len(curve['market'])}회 (bull={nb}, bear={nbear})")
    print(f"{'strategy':13} {'total%':>11} {'CAGR%':>8} {'MDD%':>8}")
    for kk in ["adaptive", "always_momo", "always_rev", "market"]:
        arr = np.array(curve[kk])
        tot = arr[-1] - 1
        cagr = arr[-1] ** (1 / span) - 1
        peak = np.maximum.accumulate(arr)
        mdd = ((arr - peak) / peak).min()
        print(f"{kk:13} {tot*100:>11.1f} {cagr*100:>8.1f} {mdd*100:>8.1f}")


asyncio.run(main())
