"""final_score(4팩터 합성) 가중치 그리드 탐색 + train/test 분리 검증."""
import asyncio
import os
import itertools
import numpy as np
import pandas as pd
import asyncpg

TRAIN_END = pd.Timestamp("2023-01-01")


def perf(date_ret, dates):
    if len(date_ret) < 3:
        return None, None
    eq = (1 + pd.Series(date_ret)).cumprod()
    span = (dates[-1] - dates[0]).days / 365.25
    cagr = eq.iloc[-1] ** (1 / span) - 1
    peak = eq.cummax()
    mdd = ((eq - peak) / peak).min()
    return cagr, mdd


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
    reg = s.set_index("date")["bull"]
    r["bull"] = r["date"].map(reg)
    r = r.dropna(subset=["bull"])

    def run(weights, mask):
        wM, wV, wQ, wG = weights
        comp = wM * r["r_mom"] + wV * r["r_val"] + wQ * r["r_qual"] + wG * r["r_grw"]
        sub = r[mask].copy()
        sub["comp"] = comp[mask]
        sub["cp"] = sub.groupby("date")["comp"].rank(pct=True)
        sub["pick"] = np.where(sub["bull"], sub["cp"] >= 0.9, sub["cp"] <= 0.1)
        dr = sub[sub["pick"]].groupby("date")["ret"].mean()
        dates = sorted(dr.index)
        return perf(dr.reindex(dates).values, dates)

    train = r["date"] < TRAIN_END
    test = r["date"] >= TRAIN_END
    grid = [w for w in itertools.product([0, 1, 2], repeat=4) if sum(w) > 0]
    results = []
    for w in grid:
        ssum = sum(w)
        wn = tuple(x / ssum for x in w)
        ctr, _ = run(wn, train)
        if ctr is None:
            continue
        results.append((ctr, wn, w))
    results.sort(reverse=True)

    print(f"가중치(M,V,Q,G) | train CAGR -> test CAGR (train<{TRAIN_END.date()})")
    print("=== train CAGR 상위 8 ===")
    for ctr, wn, w in results[:8]:
        cte, mdde = run(wn, test)
        print(f"  M{w[0]} V{w[1]} Q{w[2]} G{w[3]}  train {ctr*100:6.1f}%  ->  test {cte*100:6.1f}%  (MDD {mdde*100:5.1f}%)")
    cte, mdde = run((0.25, 0.25, 0.25, 0.25), test)
    ctr, _ = run((0.25, 0.25, 0.25, 0.25), train)
    print(f"\n  동일가중  train {ctr*100:6.1f}%  ->  test {cte*100:6.1f}%  (MDD {mdde*100:5.1f}%)")


asyncio.run(main())
