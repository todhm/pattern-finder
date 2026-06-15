# -*- coding: utf-8 -*-
"""개선 score 방식 vs 현행 final_score — 경량 백테스트 비교 (read-only).

현행 grade(us_stock_grade)는 현행 점수로 계산돼 있다. grade 를 다시 생성하지
않고, 이미 저장된 구성요소(value/quality/momentum/growth_score + 가격/유동성)
에서 '개선 composite' 를 즉석 계산해, 동일한 단순 백테스트 엔진으로 현행
final_score 선택과 직접 비교한다. (DB 쓰기 없음)

엔진(공정 비교용 단순화):
  - 매 5거래일 리밸런싱(비중첩 5일 보유), 선택점수 상위 N 종목 동일가중,
    5일 forward 수익률 실현 → NAV 누적.
  - grade 유니버스(=일자별 EM8 top-500)에서 점수 상위 N 선택. 현행/개선 동일
    유니버스라 선택 점수의 효과만 분리됨.
  - 거래비용 미반영(상대비교 목적). 실엔진(+commission/slippage)과 절대값은 다름.

개선 composite (검증된 부분만):
  IMP_full = z(quality)+z(momentum=near52h)+z(value_gated)+z(lowvol)
  변형으로 게이팅/저변동 효과 분리.
"""
from __future__ import annotations

import os
import sys
from datetime import date

import asyncpg
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import factor_ic_analysis as F  # noqa: E402

START = os.getenv("BT_START", "2019-01-02")
END = os.getenv("BT_END", "2021-03-18")
TOPN = [int(x) for x in os.getenv("BT_TOPN", "10,20").split(",")]
HOLD = 5  # 거래일 (리밸런싱 주기 = 보유기간)

SQL = """
WITH uni AS (SELECT DISTINCT symbol FROM us_stock_grade WHERE date BETWEEN $1 AND $2),
px AS (
    SELECT symbol, date, close::float8 AS close, volume::float8 AS volume,
        ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY date) AS rn,
        MAX(close) OVER (PARTITION BY symbol ORDER BY date
                         ROWS BETWEEN 251 PRECEDING AND CURRENT ROW)::float8 AS hi252,
        MIN(close) OVER (PARTITION BY symbol ORDER BY date
                         ROWS BETWEEN 251 PRECEDING AND CURRENT ROW)::float8 AS lo252
    FROM us_daily
    WHERE date BETWEEN ($1::date - INTERVAL '420 days') AND ($2::date + INTERVAL '30 days')
      AND symbol IN (SELECT symbol FROM uni)
),
g AS (
    SELECT symbol, date, value_score::float8 v, quality_score::float8 q,
           momentum_score::float8 m, growth_score::float8 gr, final_score::float8 f,
           volatility_annual::float8 volann
    FROM us_stock_grade WHERE date BETWEEN $1 AND $2
)
SELECT g.symbol, g.date, g.v, g.q, g.m, g.gr, g.f, g.volann,
       p0.close AS px0, p0.volume AS vol0, p0.hi252, p0.lo252,
       ph.close AS px_h
FROM g
JOIN px p0 ON p0.symbol=g.symbol AND p0.date=g.date
LEFT JOIN px ph ON ph.symbol=g.symbol AND ph.rn = p0.rn + %d
""" % HOLD

SPY_SQL = """
SELECT date, close::float8 AS close, ROW_NUMBER() OVER (ORDER BY date) AS rn
FROM us_daily_etf WHERE symbol='SPY' AND date BETWEEN $1 AND ($2::date + INTERVAL '30 days')
ORDER BY date
"""


async def load():
    url = os.environ["DATABASE_URL"].replace("postgresql+asyncpg://", "postgresql://")
    conn = await asyncpg.connect(url, timeout=180)
    d0, d1 = date.fromisoformat(START), date.fromisoformat(END)
    try:
        rows = await conn.fetch(SQL, d0, d1)
        spy = await conn.fetch(SPY_SQL, d0, d1)
    finally:
        await conn.close()
    return pd.DataFrame([dict(r) for r in rows]), pd.DataFrame([dict(r) for r in spy])


def zxs(df, col):
    g = df.groupby("date")[col]
    return (df[col] - g.transform("mean")) / g.transform("std")


def metrics(nav, per_ret, periods_per_year):
    nav = np.asarray(nav)
    total = nav[-1] / nav[0] - 1
    yrs = len(per_ret) / periods_per_year
    cagr = (nav[-1] / nav[0]) ** (1 / yrs) - 1 if yrs > 0 else np.nan
    r = np.asarray(per_ret)
    sharpe = (r.mean() / r.std() * np.sqrt(periods_per_year)) if r.std() > 0 else np.nan
    peak = np.maximum.accumulate(nav)
    mdd = ((nav - peak) / peak).min()
    win = (r > 0).mean()
    return dict(total=total, cagr=cagr, sharpe=sharpe, mdd=mdd, win=win, nper=len(r))


def backtest(df, score_col, topn):
    """비중첩 5일 리밸런싱, 점수 상위 topn 동일가중 long-only."""
    sub = df.dropna(subset=[score_col, "fwd_h"]).copy()
    dates = np.sort(sub["date"].unique())
    rebal = dates[::HOLD]
    rets = []
    for d in rebal:
        day = sub[sub["date"] == d]
        if len(day) < topn * 2:
            continue
        pick = day.nlargest(topn, score_col)
        rets.append(pick["fwd_h"].mean())
    nav = np.cumprod([1.0] + [1 + x for x in rets])
    ppy = 252 / HOLD
    return metrics(nav, rets, ppy)


def main():
    df, spy = F.asyncio.get_event_loop().run_until_complete(load())
    df["fwd_h"] = df["px_h"] / df["px0"] - 1.0
    rng = (df["hi252"] - df["lo252"]).replace(0, np.nan)
    df["pos52"] = (df["px0"] - df["lo252"]) / rng
    df["log_price"] = np.log(df["px0"].clip(lower=0.01))
    df["log_dvol"] = np.log((df["px0"] * df["vol0"]).clip(lower=1))

    # value 게이팅: 저가 OR 비유동 분위에서만 value 반영, 아니면 중립(z=0)
    pl = df.groupby("date")["log_price"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 3, labels=False, duplicates="drop"))
    dv = df.groupby("date")["log_dvol"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 3, labels=False, duplicates="drop"))
    small_illiq = (pl == 0) | (dv == 0)

    df["zq"], df["zm"], df["zv"] = zxs(df, "q"), zxs(df, "m"), zxs(df, "v")
    df["zg"], df["zlv"] = zxs(df, "gr"), zxs(df, "volann") * -1
    df["zv_gated"] = np.where(small_illiq, df["zv"], 0.0)

    df["IMP_full"] = df["zq"] + df["zm"] + df["zv_gated"] + df["zlv"]
    df["IMP_noGate"] = df["zq"] + df["zm"] + df["zv"] + df["zlv"]
    df["IMP_noLowVol"] = df["zq"] + df["zm"] + df["zv_gated"]
    df["IMP_QMonly"] = df["zq"] + df["zm"]

    strategies = [("baseline final_score", "f"),
                  ("IMP z(Q)+z(M)+z(Vgated)+z(lowvol)", "IMP_full"),
                  ("IMP (value NOT gated)", "IMP_noGate"),
                  ("IMP no lowvol", "IMP_noLowVol"),
                  ("IMP quality+momentum only", "IMP_QMonly")]

    # SPY benchmark (5일 비중첩)
    spy = spy.sort_values("date").reset_index(drop=True)
    spy["fwd_h"] = spy["close"].shift(-HOLD) / spy["close"] - 1.0
    gdates = np.sort(df["date"].unique())
    spy_b = spy[spy["date"].isin(gdates)].dropna(subset=["fwd_h"])
    spy_r = spy_b["fwd_h"].values[::HOLD]
    spy_nav = np.cumprod([1.0] + list(1 + spy_r))
    spy_m = metrics(spy_nav, spy_r, 252 / HOLD)

    print(f"\n기간 {START}~{END}, {df['date'].nunique()} dates, "
          f"리밸런싱 {HOLD}일 비중첩, long-only 동일가중")
    print(f"SPY 벤치마크: 총수익 {spy_m['total']:+.1%}  CAGR {spy_m['cagr']:+.1%}  "
          f"Sharpe {spy_m['sharpe']:.2f}  MDD {spy_m['mdd']:.1%}\n")

    for n in TOPN:
        print("=" * 92)
        print(f"top-{n} 선택  |  {'전략':<40}{'총수익':>9}{'CAGR':>8}{'Sharpe':>8}{'MDD':>8}{'승률':>7}")
        print("=" * 92)
        for label, col in strategies:
            m = backtest(df, col, n)
            print(f"{'':<13}{label:<40}{m['total']:>+8.1%}{m['cagr']:>+7.1%}"
                  f"{m['sharpe']:>8.2f}{m['mdd']:>8.1%}{m['win']:>7.0%}")
        print()

    print("[주의] 단일 레짐(COVID) + 거래비용 미반영 상대비교. 절대수치는 실엔진과")
    print("다르며, 개선안은 OOS(전체기간) 재검증 필요.\n")


if __name__ == "__main__":
    main()
