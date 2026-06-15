"""리밸 빈도(H)별 momentum(near52h) adaptive 전략 - 경량(메모리 안전)판."""
import asyncio
import os
import numpy as np
import pandas as pd
import asyncpg

HS = [21, 10, 5, 3, 2, 1]


async def main():
    c = await asyncpg.connect(os.getenv("DATABASE_URL"))
    d = await c.fetch("SELECT symbol,date,close,volume FROM us_daily WHERE close>0")
    spy = await c.fetch("SELECT date,close FROM us_daily_etf WHERE symbol='SPY' ORDER BY date")
    await c.close()
    df = pd.DataFrame(d, columns=["symbol", "date", "close", "volume"])
    del d
    df["date"] = pd.to_datetime(df["date"])
    df["close"] = df["close"].astype("float32")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").astype("float32")
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    g = df.groupby("symbol", sort=False)
    df["near52h"] = (df["close"] / g["close"].transform(lambda s: s.rolling(252, min_periods=120).max())).astype("float32")
    avol = g["volume"].transform(lambda s: s.rolling(20, min_periods=10).mean())
    df["liquid"] = (df["close"] >= 5) & (df["close"] * avol >= 5_000_000)
    del avol
    df.drop(columns=["volume"], inplace=True)
    lastc = g["close"].transform("last").astype("float32")
    lastd = g["date"].transform("max")
    cutoff = df["date"].max() - pd.Timedelta(days=30)
    for H in HS:
        ex = g["close"].shift(-H)
        ex = ex.where(~(ex.isna() & (lastd < cutoff)), lastc)
        df[f"ret{H}"] = (ex / df["close"] - 1).astype("float32")
    del lastc, lastd
    df.drop(columns=["close"], inplace=True)

    s = pd.DataFrame(spy, columns=["date", "spy"])
    s["date"] = pd.to_datetime(s["date"]); s["spy"] = s["spy"].astype(float)
    s = s.sort_values("date")
    s["bull"] = s["spy"] > s["spy"].rolling(200, min_periods=100).mean()
    bull_map = s.set_index("date")["bull"]

    df = df[df["liquid"]].dropna(subset=["near52h"])
    tdates = np.array(sorted(df["date"].unique()))
    by_date = {dt: day for dt, day in df.groupby("date")}
    del df

    print(f"{'H':>4} {'리밸수':>6} {'승률%':>6} {'평균%':>7} {'CAGR%':>7} {'MDD%':>7}")
    for H in HS:
        rcol = f"ret{H}"
        rets = []
        for dtx in tdates[::H]:
            dt = pd.Timestamp(dtx)
            if dt not in bull_map.index or pd.isna(bull_map.loc[dt]) or dt not in by_date:
                continue
            day = by_date[dt].dropna(subset=[rcol])
            if len(day) < 50:
                continue
            day = day.sort_values("near52h")
            k = max(1, len(day) // 10)
            bull = bool(bull_map.loc[dt])
            rets.append((day.tail(k) if bull else day.head(k))[rcol].mean())
        rets = np.array(rets, dtype=float)
        if len(rets) < 3:
            continue
        eq = np.cumprod(1 + rets)
        span = (pd.Timestamp(tdates[::H][-1]) - pd.Timestamp(tdates[::H][0])).days / 365.25
        cagr = eq[-1] ** (1 / span) - 1
        peak = np.maximum.accumulate(eq); mdd = ((eq - peak) / peak).min()
        print(f"{H:>4} {len(rets):>6} {(rets>0).mean()*100:>6.1f} {rets.mean()*100:>7.2f} {cagr*100:>7.1f} {mdd*100:>7.1f}")


asyncio.run(main())
