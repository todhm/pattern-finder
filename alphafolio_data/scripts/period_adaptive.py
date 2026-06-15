"""리밸 주기 H x 팩터별 adaptive 방식(static / all-flip / guru) 비교. 메모리 안전판."""
import asyncio
import os
import numpy as np
import pandas as pd
import asyncpg

HS = [21, 10, 5, 3]


async def main():
    c = await asyncpg.connect(os.getenv("DATABASE_URL"))
    d = await c.fetch("SELECT symbol,date,close,volume FROM us_daily WHERE close>0")
    spy = await c.fetch("SELECT date,close FROM us_daily_etf WHERE symbol='SPY' ORDER BY date")
    df = pd.DataFrame(d, columns=["symbol", "date", "close", "volume"]); del d
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

    tdates = np.array(sorted(df["date"].unique()))
    rebal_dates = sorted({pd.Timestamp(x) for H in HS for x in tdates[::H]})
    fac = await c.fetch(
        "SELECT symbol,date,per,returnonequityttm FROM us_stock_basic "
        "WHERE source='computed' AND date = ANY($1::date[])",
        [x.date() for x in rebal_dates])
    await c.close()
    fdf = pd.DataFrame(fac, columns=["symbol", "date", "per", "roe"])
    fdf["date"] = pd.to_datetime(fdf["date"])
    fdf["per"] = pd.to_numeric(fdf["per"], errors="coerce")
    fdf["roe"] = pd.to_numeric(fdf["roe"], errors="coerce")
    rsub = df[df["date"].isin(rebal_dates) & df["liquid"]].merge(fdf, on=["symbol", "date"], how="left")
    rsub["val"] = np.where(rsub["per"] > 0, 1 / rsub["per"], np.nan)
    del df

    s = pd.DataFrame(spy, columns=["date", "spy"])
    s["date"] = pd.to_datetime(s["date"]); s["spy"] = s["spy"].astype(float)
    s = s.sort_values("date")
    s["bull"] = s["spy"] > s["spy"].rolling(200, min_periods=100).mean()
    bull_map = s.set_index("date")["bull"]
    by_date = {dt: day for dt, day in rsub.groupby("date")}
    del rsub

    def comp(day, mode, bull):
        rm = day["near52h"].rank(pct=True)
        rv = day["val"].rank(pct=True).fillna(0.5)
        rq = day["roe"].rank(pct=True).fillna(0.5)
        if mode == "static":
            return rm + rv + rq
        if mode == "allflip":
            base = rm + rv + rq
            return base if bull else (3 - base)
        return (rm + rv + rq) if bull else ((1 - rm) + rv + rq)

    print(f"{'mode':9} {'H':>3} {'리밸':>5} {'승률%':>6} {'평균%':>7} {'CAGR%':>7} {'MDD%':>7}")
    for mode in ("static", "allflip", "guru"):
        for H in HS:
            rcol = f"ret{H}"; rets = []
            for dtx in tdates[::H]:
                dt = pd.Timestamp(dtx)
                if dt not in bull_map.index or pd.isna(bull_map.loc[dt]) or dt not in by_date:
                    continue
                day = by_date[dt].dropna(subset=["near52h", rcol])
                if len(day) < 50:
                    continue
                bull = bool(bull_map.loc[dt])
                day = day.assign(c=comp(day, mode, bull)).sort_values("c")
                k = max(1, len(day) // 10)
                rets.append(day.tail(k)[rcol].mean())
            rets = np.array(rets, dtype=float)
            if len(rets) < 3:
                continue
            eq = np.cumprod(1 + rets)
            span = (pd.Timestamp(tdates[::H][-1]) - pd.Timestamp(tdates[::H][0])).days / 365.25
            cagr = eq[-1] ** (1 / span) - 1
            peak = np.maximum.accumulate(eq); mdd = ((eq - peak) / peak).min()
            print(f"{mode:9} {H:>3} {len(rets):>5} {(rets>0).mean()*100:>6.1f} {rets.mean()*100:>7.2f} {cagr*100:>7.1f} {mdd*100:>7.1f}")


asyncio.run(main())
