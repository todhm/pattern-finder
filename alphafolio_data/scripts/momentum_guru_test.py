"""guru/학계 momentum 산정법을 us_daily 전체 패널에 구현 → 횡단면 rank-IC 검증.

방법:
  - JT 12-1     : close[-21] / close[-252] - 1  (Jegadeesh-Titman, skip 1개월)
  - MOM 6-1     : close[-21] / close[-126] - 1
  - IBD RS      : 0.4*(C/C65)+0.2*(C/C130)+0.2*(C/C195)+0.2*(C/C260)  (최근분기 2배)
  - NEAR52H     : close / rolling(252).max(close)  (George-Hwang 52주 신고가 근접도)
정석 IC: 각 월말 리밸일에 횡단면 rank-corr(metric, fwd) → 시계열 평균.
fwd: 20/60/120 거래일.
"""
import asyncio
import os
import numpy as np
import pandas as pd
import asyncpg


async def load():
    c = await asyncpg.connect(os.getenv("DATABASE_URL"))
    rows = await c.fetch("SELECT symbol, date, close FROM us_daily WHERE close > 0")
    await c.close()
    df = pd.DataFrame(rows, columns=["symbol", "date", "close"])
    df["date"] = pd.to_datetime(df["date"])
    df["close"] = df["close"].astype(float)
    return df.sort_values(["symbol", "date"]).reset_index(drop=True)


def build(df):
    g = df.groupby("symbol", sort=False)["close"]
    sh = lambda n: g.shift(n)
    df["jt_12_1"] = sh(21) / sh(252) - 1
    df["mom_6_1"] = sh(21) / sh(126) - 1
    c = df["close"]
    df["ibd_rs"] = (0.4 * (c / sh(65)) + 0.2 * (c / sh(130))
                    + 0.2 * (c / sh(195)) + 0.2 * (c / sh(260)))
    df["near52h"] = c / g.transform(lambda s: s.rolling(252, min_periods=120).max())
    for n in (20, 60, 120):
        df[f"fwd{n}"] = g.shift(-n) / c - 1
    return df


def xs_ic(df_rb, metric, fwd, min_n=50):
    def f(grp):
        x = grp[[metric, fwd]].dropna()
        if len(x) < min_n:
            return np.nan
        return x[metric].rank().corr(x[fwd].rank())
    return df_rb.groupby("date").apply(f).dropna()


async def main():
    df = await load()
    print(f"us_daily rows={len(df)}, symbols={df['symbol'].nunique()}, "
          f"{df['date'].min().date()}~{df['date'].max().date()}")
    df = build(df)
    # 월말 리밸일 (전역 거래일 기준)
    me = df.groupby(df["date"].dt.to_period("M"))["date"].transform("max")
    df_rb = df[df["date"] == me]
    print(f"rebalance dates(월말): {df_rb['date'].nunique()}, rows={len(df_rb)}\n")

    metrics = ["jt_12_1", "mom_6_1", "ibd_rs", "near52h"]
    print(f"{'metric':10} {'fwd':>5} {'meanIC':>9} {'IR':>7} {'%>0':>6} {'nDates':>7}")
    for fwd in ("fwd20", "fwd60", "fwd120"):
        for mt in metrics:
            ics = xs_ic(df_rb, mt, fwd)
            if len(ics) == 0:
                continue
            mean = ics.mean()
            ir = mean / ics.std() if ics.std() else float("nan")
            pos = (ics > 0).mean()
            print(f"{mt:10} {fwd:>5} {mean:>+9.4f} {ir:>7.2f} {pos:>6.0%} {len(ics):>7}")
        print()


asyncio.run(main())
