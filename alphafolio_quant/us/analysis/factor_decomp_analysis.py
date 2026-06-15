# -*- coding: utf-8 -*-
"""rs_value / value_score 분위별 차등 적용 + momentum 분해 효과 검증.

factor_ic_analysis.py 의 로더/IC 헬퍼를 재사용한다 (전부 read-only).

다루는 것
--------
A. rs_value(=IBD 12-1M 모멘텀) 가 어느 조건 버킷에서 +/- 로 갈리는지 (조건부 IC).
B. value_score 조건부 IC (가격대/유동성/변동성/52주위치/rs 버킷별).
C. momentum 분해: model momentum_score vs '추세'(52주 고점근접) vs '반등'(과거
   6M 음수=oversold) vs z-score 블렌드 — 어느 구성이 IC/IR 이 높은가.
D. 분위별 차등 적용 시제품(IN-SAMPLE 진단): rs 를 52주위치로 게이팅/부호반전,
   value 를 저가·비유동 구간에 한정 → raw 단방향 적용 대비 IC_IR 개선되는지.
   (※ in-sample 상한 추정치 — 실거래 적용 전 out-of-sample 검증 필요)

실행: docker compose exec -T alphafolio_quant python us/analysis/factor_decomp_analysis.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import factor_ic_analysis as F  # noqa: E402


def zscore_xs(df, col):
    """date별 횡단면 z-score (NaN 보존)."""
    g = df.groupby("date")[col]
    return (df[col] - g.transform("mean")) / g.transform("std")


def line(st, label):
    return (f"{label:<22}IC={F.fmt(st['mean'],4)}  IR={F.fmt(st['ir'],2)}  "
            f"t={F.fmt(st['t'],1)}  hit={('%.0f%%'%(st['hit']*100)) if not np.isnan(st['hit']) else 'nan':>4}  N={st['n']}")


def cond_table(df, score, cond, labels, ret="fwd20"):
    res = F.conditional_ic(df, score, ret, cond, 3)
    cells = "".join(
        f"{labels[i]}:{F.fmt(res[i]['mean'],3)}(t{F.fmt(res[i]['t'],1)})".rjust(22)
        for i in range(3))
    return cells


def main():
    df, spy = F.asyncio.get_event_loop().run_until_complete(F.load())
    df = F.engineer(df, spy)
    print(f"\n로드: {len(df):,} obs, {df['date'].nunique()} dates, "
          f"{df['symbol'].nunique()} symbols  ({df['date'].min()}~{df['date'].max()})")

    # 분해/블렌드용 합성 신호 (높을수록 매수쪽으로 정렬)
    df["sig_trend"] = df["pos52"]            # 52주 고점 근접 = 추세
    df["sig_rebound"] = -df["ret_p126"]      # 과거 6M 음수(oversold) = 반등
    df["sig_lowvol"] = -df["volann"]         # 저변동
    df["z_trend"] = zscore_xs(df, "sig_trend")
    df["z_rebound"] = zscore_xs(df, "sig_rebound")
    df["z_quality"] = zscore_xs(df, "q")
    df["z_mom"] = zscore_xs(df, "m")
    df["z_lowvol"] = zscore_xs(df, "sig_lowvol")
    df["blend_tr_rb"] = df["z_trend"] + df["z_rebound"]
    df["blend_q_tr"] = df["z_quality"] + df["z_trend"]
    df["blend_q_tr_lv"] = df["z_quality"] + df["z_trend"] + df["z_lowvol"]

    CONDS = [("ret_p20", ["loser", "mid", "winner"]),
             ("log_price", ["저가", "중", "고가"]),
             ("log_dvol", ["비유동", "중", "유동"]),
             ("volann", ["저변동", "중", "고변동"]),
             ("pos52", ["52w저점", "중", "52w고점"])]

    # ===== A. rs_value 조건부 IC =====
    print("\n" + "=" * 96)
    print("A) rs_value(12-1M 모멘텀) 조건부 IC (fwd20) — 어디서 +로 살고 어디서 -로 죽나")
    print("=" * 96)
    for cond, labels in CONDS:
        print(f"{cond:<12}{cond_table(df,'rs',cond,labels)}")
    print(f"{'value버킷별':<12}{cond_table(df,'rs','v',['저value','중','고value'])}")

    # ===== B. value_score 조건부 IC =====
    print("\n" + "=" * 96)
    print("B) value_score 조건부 IC (fwd20)")
    print("=" * 96)
    for cond, labels in CONDS:
        print(f"{cond:<12}{cond_table(df,'v',cond,labels)}")
    print(f"{'rs버킷별':<12}{cond_table(df,'v','rs',['저rs','중','고rs'])}")

    # ===== C. momentum 분해 — 어느 구성이 IC 높나 =====
    print("\n" + "=" * 96)
    print("C) momentum 분해 / 블렌드 IC (fwd20, fwd60)")
    print("=" * 96)
    sigs = [("model momentum_score", "m"), ("추세(52w근접 pos52)", "sig_trend"),
            ("반등(-과거6M ret)", "sig_rebound"), ("저변동(-vol)", "sig_lowvol"),
            ("z(추세)+z(반등)", "blend_tr_rb"),
            ("z(quality)+z(추세)", "blend_q_tr"),
            ("z(quality)+z(추세)+z(저변동)", "blend_q_tr_lv")]
    for h in (20, 60):
        print(f"-- horizon {h}d --")
        for label, col in sigs:
            print("  " + line(F.ic_stats(F.ic_series(df, col, f"fwd{h}")), label))

    # ===== D. 분위별 차등 적용 시제품 (IN-SAMPLE 진단) =====
    print("\n" + "=" * 96)
    print("D) 분위별 차등 적용 효과 (IN-SAMPLE 상한 — OOS 검증 필요)")
    print("=" * 96)
    # rs: 52주 고점 구간에서 부호 반전 (모멘텀이 거기서 리버설하므로)
    p = df.groupby("date")["pos52"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 3, labels=False, duplicates="drop"))
    df["rs_gated"] = np.where(p == 2, -df["rs"], df["rs"])      # 고점 버킷만 부호 반전
    df["rs_flip"] = -df["rs"]                                    # 전구간 반전(=리버설)
    print("  [rs_value]")
    print("  " + line(F.ic_stats(F.ic_series(df, "rs", "fwd20")), "raw rs (단방향)"))
    print("  " + line(F.ic_stats(F.ic_series(df, "rs_flip", "fwd20")), "rs 전구간 부호반전"))
    print("  " + line(F.ic_stats(F.ic_series(df, "rs_gated", "fwd20")), "rs 52w고점만 반전(게이팅)"))

    # value: 저가+비유동 구간에서만 적용(나머지는 횡단면 평균=중립), 차등 vs 단방향
    pl = df.groupby("date")["log_price"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 3, labels=False, duplicates="drop"))
    dv = df.groupby("date")["log_dvol"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 3, labels=False, duplicates="drop"))
    vmean = df.groupby("date")["v"].transform("mean")
    df["value_gated"] = np.where((pl == 0) | (dv == 0), df["v"], vmean)  # 저가/비유동만 살림
    print("  [value_score]")
    print("  " + line(F.ic_stats(F.ic_series(df, "v", "fwd20")), "raw value (단방향)"))
    print("  " + line(F.ic_stats(F.ic_series(df, "value_gated", "fwd20")), "value 저가/비유동만 적용"))

    # rs_value 를 quality/추세와 결합했을 때 (단독 vs 결합)
    df["z_v"] = zscore_xs(df, "v")
    df["blend_q_v"] = df["z_quality"] + df["z_v"]
    df["blend_q_v_tr"] = df["z_quality"] + df["z_v"] + df["z_trend"]
    print("  [결합]")
    print("  " + line(F.ic_stats(F.ic_series(df, "blend_q_v", "fwd20")), "z(quality)+z(value)"))
    print("  " + line(F.ic_stats(F.ic_series(df, "blend_q_v_tr", "fwd20")), "z(quality)+z(value)+z(추세)"))

    print("\n[해석] D 의 게이팅/반전 IC 가 raw 보다 크게 높으면, 그 score 를 '분위별로")
    print("다르게 적용'할 여지가 크다는 뜻. 단 in-sample 상한이므로 부호규칙은 구조적")
    print("(가격/유동성/52주위치)으로 고정하고 다음 기간에 OOS 재검증해야 함.\n")


if __name__ == "__main__":
    main()
