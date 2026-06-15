"""실험: wedge-pop 신호 기반 momentum 점수가 현재 momentum_score보다 예측력이 좋은지.

wedge_pop.py candidate_mask 로직을 us_daily(OHLCV)에 재현 → 종목별 발화일 →
grade 날짜에서 최근 발화 근접도 점수화 → forward 수익률 IC 를 momentum_score 와 비교.
"""
import asyncio
import os
from datetime import timedelta

import numpy as np
import pandas as pd
import asyncpg


def wedge_signals(df, lookback=10, ema_fast=10, ema_slow=20, consolidation_pct=0.6,
                  breakout_atr_mult=0.5, atr_period=14, cooldown=10,
                  sma_mid=50, sma_long=200):
    c = df["close"].to_numpy(float)
    o = df["open"].to_numpy(float)
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    n = len(df)
    if n < lookback + 1:
        return np.zeros(n, bool)
    fast = pd.Series(c).ewm(span=ema_fast, adjust=False).mean().to_numpy()
    slow = pd.Series(c).ewm(span=ema_slow, adjust=False).mean().to_numpy()
    smid = pd.Series(c).rolling(sma_mid).mean().to_numpy()
    slong = pd.Series(c).rolling(sma_long).mean().to_numpy()
    prev_c = np.roll(c, 1)
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    atr = pd.Series(tr).ewm(span=atr_period, adjust=False).mean().to_numpy()
    below = c < fast
    prior_below = pd.Series(below.astype(float)).rolling(lookback).sum().shift(1).to_numpy()
    cons_ratio = prior_below / lookback
    cons_ok = np.where(np.isnan(cons_ratio), False, cons_ratio >= consolidation_pct)
    prev_o = np.roll(o, 1)
    prev_fast = np.roll(fast, 1)
    primary = prev_c < prev_fast
    atr_valid = atr > 0
    resistance = np.maximum(fast, slow)
    with np.errstate(divide="ignore", invalid="ignore"):
        ema_dist = np.where(atr_valid, (c - resistance) / atr, -np.inf)
        daily_move = np.where(atr_valid, (c - prev_c) / atr, -np.inf)
    strength = np.maximum(ema_dist, daily_move)
    strength_ok = atr_valid & (strength >= breakout_atr_mult)
    sma_valid = ~(np.isnan(smid) | np.isnan(slong))
    above_sma = sma_valid & (c > smid) & (c > slong)
    cand = (cons_ok & (c > fast) & (c > slow) & (c > o) & primary
            & (c >= prev_o) & strength_ok & above_sma)
    cand[:lookback] = False
    out = np.zeros(n, bool)
    cooldown_until = -1
    for i in np.flatnonzero(cand):
        if i <= cooldown_until:
            continue
        out[i] = True
        cooldown_until = i + cooldown
    return out


async def main():
    c = await asyncpg.connect(os.getenv("DATABASE_URL"))
    g = await c.fetch("SELECT date,symbol,momentum_score,final_score FROM us_stock_grade")
    mn = await c.fetchval("SELECT min(date) FROM us_stock_grade")
    mx = await c.fetchval("SELECT max(date) FROM us_stock_grade")
    syms = [r["symbol"] for r in await c.fetch("SELECT DISTINCT symbol FROM us_stock_grade")]
    d = await c.fetch(
        "SELECT symbol,date,open,high,low,close FROM us_daily "
        "WHERE symbol = ANY($1::text[]) AND date >= $2 AND date <= $3",
        syms, mn - timedelta(days=400), mx + timedelta(days=130))
    await c.close()

    gdf = pd.DataFrame(g, columns=["date", "symbol", "momentum_score", "final_score"])
    gdf["date"] = pd.to_datetime(gdf["date"])
    dd = pd.DataFrame(d, columns=["symbol", "date", "open", "high", "low", "close"])
    dd["date"] = pd.to_datetime(dd["date"])
    for col in ("open", "high", "low", "close"):
        dd[col] = dd[col].astype(float)
    dd = dd.sort_values(["symbol", "date"]).reset_index(drop=True)
    dd["fwd20"] = dd.groupby("symbol")["close"].shift(-20) / dd["close"] - 1

    parts = []
    for sym, sub in dd.groupby("symbol", sort=False):
        sub = sub.copy()
        sig = wedge_signals(sub)
        bars_since = np.full(len(sub), 10000.0)
        last = -10000
        for j in range(len(sub)):
            if sig[j]:
                last = j
            bars_since[j] = j - last
        sub["bars_since"] = bars_since
        parts.append(sub[["symbol", "date", "fwd20", "bars_since"]])
    dd2 = pd.concat(parts, ignore_index=True)

    m = gdf.merge(dd2, on=["symbol", "date"], how="left")
    m["momentum_score"] = pd.to_numeric(m["momentum_score"], errors="coerce")
    m["wedge_recent10"] = (m["bars_since"] <= 10).astype(float)
    m["wedge_score"] = (100.0 / (1.0 + m["bars_since"])).where(m["bars_since"] < 10000, 0.0)

    valid = m.dropna(subset=["fwd20"])
    print(f"grade rows: {len(m)}, with fwd20: {len(valid)}, "
          f"wedge fired<=10d: {int(m['wedge_recent10'].sum())}")

    def ic(a, b):
        x = valid[[a, b]].dropna()
        return x[a].rank().corr(x[b].rank())

    print("\n=== forward-20d 수익률 예측 IC (rank corr) ===")
    print(f"  momentum_score      : {ic('momentum_score','fwd20'):+.4f}")
    print(f"  wedge_score(recency): {ic('wedge_score','fwd20'):+.4f}")
    print(f"  wedge_recent10(0/1) : {ic('wedge_recent10','fwd20'):+.4f}")

    print("\n=== 최근10일 wedge 발화 여부별 평균 forward-20d (%) ===")
    grp = valid.groupby("wedge_recent10")["fwd20"].agg(["mean", "count"])
    grp["mean"] = (grp["mean"] * 100).round(2)
    print(grp.to_string())

    print("\n=== momentum_score 분포 (변별력) ===")
    print(m["momentum_score"].describe().round(2).to_string())


asyncio.run(main())
