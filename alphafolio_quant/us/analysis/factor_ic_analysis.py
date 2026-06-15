# -*- coding: utf-8 -*-
"""Factor predictive-power (IC) analysis for us_stock_grade.

목적
----
us_stock_grade 의 각 score 구성요소가 미래 수익률을 얼마나 잘 예측하는지
(Information Coefficient) 측정하고, 약한 score 를 진단/개선하기 위한 근거를
만든다. momentum 을 한 방향으로만 보지 않고 가격대·최근 수익률·유동성·변동성·
52주 위치·시장국면(SPY 추세)·섹터 등으로 조건부 분해(conditional IC)한다.
또한 모델 score 가 아닌 "원시 신호"(과거 다기간 수익률, 52주 위치, 변동성,
유동성 등)의 IC 도 직접 측정해, 모델이 못 잡는 예측력이 있는지 본다.

IC 정의
-------
각 날짜의 횡단면(cross-section)에서 score 와 H일 forward 수익률의 Spearman
순위상관을 구하고, 전 기간 평균한다.
  - IC_mean : 예측 방향/강도 (부호 = 방향, 절대값 = 강도)
  - IC_IR   : IC_mean / IC_std (정보비율; 안정성 포함 품질지표)
  - t-stat  : IC_IR * sqrt(N_dates) (|t|>=2 면 유의)
  - hit%    : IC>0 인 날짜 비율 (방향 일관성)
분위 스프레드: score 5분위(date별)별 평균 forward 수익률, top-bottom 스프레드와
단조성(monotonic) 여부.

실행
----
docker compose exec -T alphafolio_quant python us/analysis/factor_ic_analysis.py
"""
from __future__ import annotations

import asyncio
import os
from datetime import date

import asyncpg
import numpy as np
import pandas as pd

START = os.getenv("IC_START", "2019-01-02")
END = os.getenv("IC_END", "2021-03-18")
MIN_NAMES = 20          # 횡단면 IC 계산에 필요한 최소 종목 수/날짜
HORIZONS = [20, 60]     # forward 수익률 horizon (거래일)


# ----------------------------------------------------------------- data load
DATASET_SQL = """
WITH uni AS (
    SELECT DISTINCT symbol FROM us_stock_grade WHERE date BETWEEN $1 AND $2
),
px AS (
    SELECT symbol, date, close::float8 AS close, volume::float8 AS volume,
        ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY date) AS rn,
        MAX(close) OVER (PARTITION BY symbol ORDER BY date
                         ROWS BETWEEN 251 PRECEDING AND CURRENT ROW)::float8 AS hi252,
        MIN(close) OVER (PARTITION BY symbol ORDER BY date
                         ROWS BETWEEN 251 PRECEDING AND CURRENT ROW)::float8 AS lo252
    FROM us_daily
    WHERE date BETWEEN ($1::date - INTERVAL '420 days') AND ($2::date + INTERVAL '120 days')
      AND symbol IN (SELECT symbol FROM uni)
),
basic AS (
    SELECT DISTINCT ON (symbol) symbol, sector
    FROM us_stock_basic WHERE sector IS NOT NULL AND sector <> ''
    ORDER BY symbol, date DESC
),
g AS (
    SELECT symbol, date,
           value_score::float8 v, quality_score::float8 q, momentum_score::float8 m,
           growth_score::float8 gr, final_score::float8 f, rs_value::float8 rs,
           volatility_annual::float8 volann, corr_spy::float8 beta
    FROM us_stock_grade WHERE date BETWEEN $1 AND $2
)
SELECT g.symbol, g.date, g.v, g.q, g.m, g.gr, g.f, g.rs, g.volann, g.beta,
       b.sector,
       p0.close AS px0, p0.volume AS vol0, p0.hi252, p0.lo252,
       pf20.close AS pf20, pf60.close AS pf60,
       pp20.close AS pp20, pp60.close AS pp60, pp126.close AS pp126
FROM g
JOIN px p0       ON p0.symbol = g.symbol AND p0.date = g.date
LEFT JOIN basic b ON b.symbol = g.symbol
LEFT JOIN px pf20  ON pf20.symbol = g.symbol  AND pf20.rn  = p0.rn + 20
LEFT JOIN px pf60  ON pf60.symbol = g.symbol  AND pf60.rn  = p0.rn + 60
LEFT JOIN px pp20  ON pp20.symbol = g.symbol  AND pp20.rn  = p0.rn - 20
LEFT JOIN px pp60  ON pp60.symbol = g.symbol  AND pp60.rn  = p0.rn - 60
LEFT JOIN px pp126 ON pp126.symbol = g.symbol AND pp126.rn = p0.rn - 126
"""

SPY_SQL = """
SELECT date, close::float8 AS close,
       ROW_NUMBER() OVER (ORDER BY date) AS rn
FROM us_daily_etf WHERE symbol='SPY'
  AND date BETWEEN ($1::date - INTERVAL '60 days') AND $2
ORDER BY date
"""


async def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    url = os.environ["DATABASE_URL"].replace("postgresql+asyncpg://", "postgresql://")
    conn = await asyncpg.connect(url, timeout=120)
    d0, d1 = date.fromisoformat(START), date.fromisoformat(END)
    try:
        rows = await conn.fetch(DATASET_SQL, d0, d1)
        spy = await conn.fetch(SPY_SQL, d0, d1)
    finally:
        await conn.close()
    df = pd.DataFrame([dict(r) for r in rows])
    spy = pd.DataFrame([dict(r) for r in spy])
    return df, spy


# ----------------------------------------------------------- feature engineer
def engineer(df: pd.DataFrame, spy: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # forward returns (예측 대상)
    df["fwd20"] = df["pf20"] / df["px0"] - 1.0
    df["fwd60"] = df["pf60"] / df["px0"] - 1.0
    # 원시 신호들 (예측 후보 rate)
    df["ret_p20"] = df["px0"] / df["pp20"] - 1.0      # 최근 1개월 수익률
    df["ret_p60"] = df["px0"] / df["pp60"] - 1.0      # 최근 3개월
    df["ret_p126"] = df["px0"] / df["pp126"] - 1.0    # 최근 6개월
    rng = (df["hi252"] - df["lo252"]).replace(0, np.nan)
    df["pos52"] = (df["px0"] - df["lo252"]) / rng     # 52주 가격 위치 0~1
    df["dist_high"] = df["px0"] / df["hi252"] - 1.0   # 52주 고점 대비 거리(<=0)
    df["dollar_vol"] = df["px0"] * df["vol0"]         # 유동성
    df["log_price"] = np.log(df["px0"].clip(lower=0.01))
    df["log_dvol"] = np.log(df["dollar_vol"].clip(lower=1))
    # 시장 국면: SPY 20거래일 수익률 부호 (그 날짜 기준)
    spy = spy.sort_values("date").reset_index(drop=True)
    spy["spy_ret20"] = spy["close"] / spy["close"].shift(20) - 1.0
    regime = spy.set_index("date")["spy_ret20"]
    df["spy_ret20"] = df["date"].map(regime)
    df["regime_up"] = df["spy_ret20"] > 0
    return df


# ------------------------------------------------------------------- IC core
def ic_series(df: pd.DataFrame, score: str, ret: str) -> pd.Series:
    """날짜별 횡단면 Spearman IC 시계열."""
    sub = df[["date", score, ret]].dropna()
    cnt = sub.groupby("date")[score].transform("count")
    sub = sub[cnt >= MIN_NAMES]
    if sub.empty:
        return pd.Series(dtype=float)
    g = sub.groupby("date")
    # Spearman = rank 후 Pearson
    ic = g.apply(lambda x: x[score].rank().corr(x[ret].rank()), include_groups=False)
    return ic.dropna()


def ic_stats(ic: pd.Series) -> dict:
    if len(ic) < 5:
        return dict(mean=np.nan, ir=np.nan, t=np.nan, hit=np.nan, n=len(ic))
    m, s = ic.mean(), ic.std(ddof=1)
    ir = m / s if s and not np.isnan(s) else np.nan
    t = ir * np.sqrt(len(ic)) if not np.isnan(ir) else np.nan
    return dict(mean=m, ir=ir, t=t, hit=(ic > 0).mean(), n=len(ic))


def quintile_spread(df: pd.DataFrame, score: str, ret: str, q: int = 5) -> tuple:
    sub = df[["date", score, ret]].dropna()
    cnt = sub.groupby("date")[score].transform("count")
    sub = sub[cnt >= q * 4].copy()
    if sub.empty:
        return [np.nan] * q, np.nan, False

    def _bucket(s):
        return pd.qcut(s.rank(method="first"), q, labels=False, duplicates="drop")

    sub["b"] = sub.groupby("date")[score].transform(_bucket)
    per = sub.groupby(["date", "b"])[ret].mean().reset_index()
    means = per.groupby("b")[ret].mean()
    vals = [means.get(i, np.nan) for i in range(q)]
    spread = vals[-1] - vals[0]
    mono = all(np.diff([v for v in vals if not np.isnan(v)]) > 0) or \
        all(np.diff([v for v in vals if not np.isnan(v)]) < 0)
    return vals, spread, mono


def conditional_ic(df: pd.DataFrame, score: str, ret: str, cond: str,
                   nb: int = 3) -> list:
    """조건변수 cond 의 date별 분위 버킷 안에서 score 의 IC 를 따로 계산."""
    sub = df[["date", score, ret, cond]].dropna()
    if sub.empty:
        return [dict(mean=np.nan, t=np.nan, n=0) for _ in range(nb)]

    def _b(s):
        return pd.qcut(s.rank(method="first"), nb, labels=False, duplicates="drop")

    sub["cb"] = sub.groupby("date")[cond].transform(_b)
    out = []
    for b in range(nb):
        d = sub[sub["cb"] == b]
        out.append(ic_stats(ic_series(d, score, ret)))
    return out


# ---------------------------------------------------------------- report
def fmt(x, p=4):
    return "   nan" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:+.{p}f}"


def main():
    df, spy = asyncio.get_event_loop().run_until_complete(load())
    print(f"\n로드: {len(df):,} obs, {df['date'].nunique()} dates "
          f"({df['date'].min()} ~ {df['date'].max()}), "
          f"{df['symbol'].nunique()} symbols")
    df = engineer(df, spy)

    MODEL = [("value", "v"), ("quality", "q"), ("momentum", "m"),
             ("growth", "gr"), ("final", "f"), ("rs_value", "rs")]
    RAW = [("ret_1m(20d)", "ret_p20"), ("ret_3m(60d)", "ret_p60"),
           ("ret_6m(126d)", "ret_p126"), ("pos_52w", "pos52"),
           ("dist_from_high", "dist_high"), ("volatility(-)", "volann"),
           ("liquidity_$vol", "log_dvol"), ("price_level", "log_price"),
           ("beta_corr_spy", "beta")]

    # ===== 1. 모델 score IC =====
    print("\n" + "=" * 78)
    print("1) 모델 SCORE 예측력 (Spearman IC)   [부호=방향, IR/t=품질, hit=일관성]")
    print("=" * 78)
    print(f"{'score':<12}{'H':>4}{'IC_mean':>10}{'IC_IR':>9}{'t':>8}{'hit%':>7}{'N':>6}")
    ic_cache = {}
    for name, col in MODEL:
        for h in HORIZONS:
            ic = ic_series(df, col, f"fwd{h}")
            ic_cache[(col, h)] = ic
            st = ic_stats(ic)
            print(f"{name:<12}{h:>4}{fmt(st['mean']):>10}{fmt(st['ir'],2):>9}"
                  f"{fmt(st['t'],1):>8}{('%.0f'%(st['hit']*100)) if not np.isnan(st['hit']) else 'nan':>7}{st['n']:>6}")

    # ===== 2. 분위 스프레드 (20일) =====
    print("\n" + "=" * 78)
    print("2) 모델 SCORE 5분위 평균 forward(20d) 수익률  [Q1=최저 ... Q5=최고 score]")
    print("=" * 78)
    print(f"{'score':<12}{'Q1':>9}{'Q2':>9}{'Q3':>9}{'Q4':>9}{'Q5':>9}{'Q5-Q1':>9}{'mono':>6}")
    for name, col in MODEL:
        vals, spread, mono = quintile_spread(df, col, "fwd20")
        cells = "".join(f"{fmt(v,3):>9}" for v in vals)
        print(f"{name:<12}{cells}{fmt(spread,3):>9}{('Y' if mono else 'N'):>6}")

    # ===== 3. 원시 신호 IC (모델 밖 rate) =====
    print("\n" + "=" * 78)
    print("3) 원시 신호 예측력 (모델 score 가 아닌 가격/거래/변동성 rate)")
    print("   volatility(-)/dist_from_high 은 음수일수록 좋다는 뜻이면 부호 반대로 해석")
    print("=" * 78)
    print(f"{'signal':<16}{'H':>4}{'IC_mean':>10}{'IC_IR':>9}{'t':>8}{'hit%':>7}{'N':>6}")
    for name, col in RAW:
        for h in HORIZONS:
            st = ic_stats(ic_series(df, col, f"fwd{h}"))
            print(f"{name:<16}{h:>4}{fmt(st['mean']):>10}{fmt(st['ir'],2):>9}"
                  f"{fmt(st['t'],1):>8}{('%.0f'%(st['hit']*100)) if not np.isnan(st['hit']) else 'nan':>7}{st['n']:>6}")

    # ===== 4. 조건부 IC: momentum 등을 국면별로 분해 =====
    CONDS = [("recent_ret_1m", "ret_p20", ["loser", "mid", "winner"]),
             ("price_level", "log_price", ["low", "mid", "high"]),
             ("liquidity", "log_dvol", ["illiq", "mid", "liquid"]),
             ("volatility", "volann", ["lowvol", "mid", "highvol"]),
             ("pos_52w", "pos52", ["near_low", "mid", "near_high"])]
    for score_name, score_col in [("momentum", "m"), ("final", "f"),
                                  ("value", "v"), ("growth", "gr")]:
        print("\n" + "=" * 78)
        print(f"4) 조건부 IC — {score_name}_score (forward 20d), 조건변수 3분위별")
        print("=" * 78)
        print(f"{'조건변수':<16}{'bucket0':>16}{'bucket1':>16}{'bucket2':>16}")
        for cname, ccol, labels in CONDS:
            res = conditional_ic(df, score_col, "fwd20", ccol, 3)
            cells = "".join(
                f"{labels[i]}:{fmt(res[i]['mean'],3)}(t{fmt(res[i]['t'],1)})".rjust(16)
                for i in range(3))
            print(f"{cname:<16}{cells}")

    # ===== 5. 시장 국면(SPY 추세)별 IC =====
    print("\n" + "=" * 78)
    print("5) 시장 국면별 IC (SPY 20일 수익률 부호) — momentum 의 양방향성 확인")
    print("=" * 78)
    print(f"{'score':<12}{'H':>4}{'UP_IC':>10}{'UP_t':>8}{'DOWN_IC':>10}{'DOWN_t':>8}")
    up_dates = set(df.loc[df["regime_up"], "date"].unique())
    for name, col in MODEL:
        for h in HORIZONS:
            ic = ic_cache[(col, h)]
            up = ic[ic.index.isin(up_dates)]
            dn = ic[~ic.index.isin(up_dates)]
            su, sd = ic_stats(up), ic_stats(dn)
            print(f"{name:<12}{h:>4}{fmt(su['mean'],3):>10}{fmt(su['t'],1):>8}"
                  f"{fmt(sd['mean'],3):>10}{fmt(sd['t'],1):>8}")

    # ===== 6. 섹터별 final IC (상위/하위) =====
    print("\n" + "=" * 78)
    print("6) 섹터별 final_score IC (forward 20d)  [예측 잘 되는/안 되는 섹터]")
    print("=" * 78)
    sec_rows = []
    for sec, d in df.dropna(subset=["sector"]).groupby("sector"):
        st = ic_stats(ic_series(d, "f", "fwd20"))
        if st["n"] >= 30:
            sec_rows.append((sec, st["mean"], st["t"], int(d["symbol"].nunique())))
    for sec, m, t, n in sorted(sec_rows, key=lambda r: (r[1] if not np.isnan(r[1]) else -9), reverse=True):
        print(f"  {sec:<28}{fmt(m,3):>10}  t={fmt(t,1):>7}  ({n} syms)")

    print("\n[해석 가이드] IC_mean 부호=예측방향, |IC_IR|>~0.3 또는 |t|>2 면 의미있음.")
    print("조건부 IC 에서 buckets 간 부호가 갈리면 그 score 는 '한 방향'이 아니라")
    print("국면 의존적 → 해당 조건을 게이팅/상호작용 항으로 넣으면 개선 여지.\n")


if __name__ == "__main__":
    main()
