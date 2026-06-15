"""승률 극대화: 저변동성 팩터 + 레짐 로테이션(공격/방어) + 추세필터 + 분산."""
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
    df["dret"] = g["close"].pct_change()
    df["vol60"] = g["dret"].transform(lambda s: s.rolling(60, min_periods=30).std())
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
    r["qual"] = r["roe"]; r["grw"] = r["eg"]; r["lv"] = -r["vol60"]
    for f in ("mom", "val", "qual", "grw", "lv"):
        r[f"r_{f}"] = r.groupby("date")[f].rank(pct=True).fillna(0.5)

    s = pd.DataFrame(spy, columns=["date", "spy"])
    s["date"] = pd.to_datetime(s["date"]); s["spy"] = s["spy"].astype(float)
    s = s.sort_values("date")
    s["bull"] = s["spy"] > s["spy"].rolling(200, min_periods=100).mean()
    s["sret"] = s["spy"].pct_change()
    s["svol"] = s["sret"].rolling(20, min_periods=10).std()
    s["hivol"] = s["svol"] > s["svol"].rolling(252, min_periods=60).median()
    reg = s.set_index("date")[["bull", "hivol"]]
    r = r.join(reg, on="date").dropna(subset=["bull"])

    def basket(day, sig, frac=0.1, bottom=False):
        v = day.sort_values(sig)
        k = max(1, int(len(day) * frac))
        return (v.head(k) if bottom else v.tail(k))["ret"].mean()

    strat = {}
    for dt, day in r.groupby("date"):
        if len(day) < 50:
            continue
        bull = bool(day["bull"].iloc[0])
        off = day["r_mom"] + day["r_val"]
        deff = day["r_qual"] + day["r_lv"]
        all4 = day["r_mom"] + day["r_val"] + day["r_qual"] + day["r_grw"]
        rec = strat.setdefault(dt, {})
        rec["base_adapt"] = basket(day.assign(c=all4), "c", bottom=not bull)
        rec["trend_cash"] = basket(day.assign(c=all4), "c") if bull else 0.0
        rec["defensive"] = basket(day.assign(c=deff), "c")
        rec["guru_rotate"] = basket(day.assign(c=(off if bull else deff)), "c")
        rec["guru_div30"] = basket(day.assign(c=(off if bull else deff)), "c", frac=0.30)
        rec["market"] = day["ret"].mean()

    res = pd.DataFrame(strat).T.sort_index()
    print(f"리밸 {len(res)}회\n")
    print(f"{'strategy':13} {'승률%':>6} {'시장대비%':>8} {'평균%':>7} {'CAGR%':>7} {'MDD%':>7}")
    span = (res.index[-1] - res.index[0]).days / 365.25
    for col in ["base_adapt", "trend_cash", "defensive", "guru_rotate", "guru_div30", "market"]:
        x = res[col].astype(float)
        win = (x > 0).mean() * 100
        beat = (x > res["market"].astype(float)).mean() * 100
        eq = np.cumprod(1 + x.values); cagr = eq[-1] ** (1 / span) - 1
        peak = np.maximum.accumulate(eq); mdd = ((eq - peak) / peak).min()
        print(f"{col:13} {win:>6.1f} {beat:>8.1f} {x.mean()*100:>7.2f} {cagr*100:>7.1f} {mdd*100:>7.1f}")


asyncio.run(main())
