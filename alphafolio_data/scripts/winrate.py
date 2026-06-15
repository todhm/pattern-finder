"""best 전략(uniform adaptive)의 승률/손익 통계."""
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
    r["mom"] = r["near52h"]; r["val"] = np.where(r["per"] > 0, 1 / r["per"], np.nan)
    r["qual"] = r["roe"]; r["grw"] = r["eg"]
    for f in ("mom", "val", "qual", "grw"):
        r[f"r_{f}"] = r.groupby("date")[f].rank(pct=True).fillna(0.5)
    s = pd.DataFrame(spy, columns=["date", "spy"])
    s["date"] = pd.to_datetime(s["date"]); s["spy"] = s["spy"].astype(float)
    s = s.sort_values("date")
    s["bull"] = s["spy"] > s["spy"].rolling(200, min_periods=100).mean()
    r["bull"] = r["date"].map(s.set_index("date")["bull"])
    r = r.dropna(subset=["bull"])

    rows = []
    for dt, day in r.groupby("date"):
        if len(day) < 50:
            continue
        bull = bool(day["bull"].iloc[0])
        base = day["r_mom"] + day["r_val"] + day["r_qual"] + day["r_grw"]
        comp = base if bull else (4 - base)
        day = day.assign(c=comp).sort_values("c")
        k = max(1, len(day) // 10)
        rows.append((dt, day.tail(k)["ret"].mean(), day["ret"].mean()))
    res = pd.DataFrame(rows, columns=["date", "strat", "market"]).sort_values("date")

    def stats(x, label):
        st = x["strat"]
        win = (st > 0).mean()
        beat = (st > x["market"]).mean()
        wins = st[st > 0]; losses = st[st < 0]
        wl = wins.mean() / abs(losses.mean()) if len(losses) else float("nan")
        print(f"[{label}] {len(st)}회 | 승률(>0) {win*100:.1f}% | 시장대비승률 {beat*100:.1f}% | "
              f"평균 {st.mean()*100:+.2f}% (승 {wins.mean()*100:+.2f} / 패 {losses.mean()*100:+.2f}, 손익비 {wl:.2f}) | "
              f"최고 {st.max()*100:+.1f} / 최악 {st.min()*100:+.1f}")

    stats(res, "전체")
    stats(res[res["date"] >= TRAIN_END], "test 2023~")
    stats(res[res["date"] < TRAIN_END], "train ~2022")


asyncio.run(main())
