# -*- coding: utf-8 -*-
"""value_score 가 소형/비유동에서 정말 예측력이 좋아지는지 검증 (read-only).

IC 뿐 아니라 분위 long-short 를 '중앙값' 기준으로 봐서 COVID 아웃라이어 영향을
제거한다. 유동성/가격(소형 proxy) 버킷별로 value 의 Q5-Q1(top-bottom) median
forward 수익률 스프레드를 비교 — 양(+)이고 클수록 "고value가 저value보다
잘 간다" = 예측력 좋음.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import factor_ic_analysis as F  # noqa: E402


def ls_median(d, score, ret, q=5):
    """date별 score 5분위 → 각 분위 median fwd ret → 날짜평균 → Q5-Q1."""
    sub = d[["date", score, ret]].dropna()
    cnt = sub.groupby("date")[score].transform("count")
    sub = sub[cnt >= q * 4].copy()
    if sub.empty:
        return np.nan, np.nan, np.nan
    sub["b"] = sub.groupby("date")[score].transform(
        lambda s: pd.qcut(s.rank(method="first"), q, labels=False, duplicates="drop"))
    per = sub.groupby(["date", "b"])[ret].median().reset_index()
    m = per.groupby("b")[ret].mean()
    q1, q5 = m.get(0, np.nan), m.get(q - 1, np.nan)
    return q1, q5, (q5 - q1)


def bucketize(df, col, n=3):
    return df.groupby("date")[col].transform(
        lambda s: pd.qcut(s.rank(method="first"), n, labels=False, duplicates="drop"))


def main():
    df, spy = F.asyncio.get_event_loop().run_until_complete(F.load())
    df = F.engineer(df, spy)
    print(f"\n로드: {len(df):,} obs, {df['date'].nunique()} dates\n")

    df["liq_b"] = bucketize(df, "log_dvol")     # 0=비유동 ... 2=유동
    df["prc_b"] = bucketize(df, "log_price")    # 0=저가(소형proxy) ... 2=고가

    for splitname, bcol, labels in [("유동성", "liq_b", ["비유동", "중", "유동"]),
                                    ("가격대(소형proxy)", "prc_b", ["저가", "중", "고가"])]:
        print("=" * 78)
        print(f"value_score 예측력 — {splitname} 버킷별  [IC + 중앙값 Q5-Q1 스프레드]")
        print("=" * 78)
        print(f"{'bucket':<8}{'IC20':>9}{'t20':>7}{'IC60':>9}{'LS20(median)':>15}{'LS60(median)':>15}{'N':>6}")
        for b in range(3):
            d = df[df[bcol] == b]
            ic20 = F.ic_stats(F.ic_series(d, "v", "fwd20"))
            ic60 = F.ic_stats(F.ic_series(d, "v", "fwd60"))
            _, _, ls20 = ls_median(d, "v", "fwd20")
            _, _, ls60 = ls_median(d, "v", "fwd60")
            print(f"{labels[b]:<8}{F.fmt(ic20['mean'],3):>9}{F.fmt(ic20['t'],1):>7}"
                  f"{F.fmt(ic60['mean'],3):>9}{F.fmt(ls20,4):>15}{F.fmt(ls60,4):>15}{ic20['n']:>6}")
        print()

    # 2x2: 소형&비유동 코너 vs 대형&유동 코너
    print("=" * 78)
    print("코너 비교 — value_score (저가+비유동) vs (고가+유동)")
    print("=" * 78)
    corner_small = df[(df["prc_b"] == 0) & (df["liq_b"] == 0)]
    corner_big = df[(df["prc_b"] == 2) & (df["liq_b"] == 2)]
    for name, d in [("저가+비유동(소형)", corner_small), ("고가+유동(대형)", corner_big)]:
        ic20 = F.ic_stats(F.ic_series(d, "v", "fwd20"))
        _, _, ls20 = ls_median(d, "v", "fwd20")
        _, _, ls60 = ls_median(d, "v", "fwd60")
        print(f"  {name:<18} IC20={F.fmt(ic20['mean'],3)} (t{F.fmt(ic20['t'],1)})  "
              f"Q5-Q1(median) 20d={F.fmt(ls20,4)} 60d={F.fmt(ls60,4)}  obs={len(d):,}")
    print("\n[판정] 소형/비유동 코너에서 IC·LS 가 양(+)이고 크며, 대형/유동에서 0/음(−)")
    print("이면 'value 는 소형/비유동에서 예측력이 좋다' 가 확정.\n")


if __name__ == "__main__":
    main()
