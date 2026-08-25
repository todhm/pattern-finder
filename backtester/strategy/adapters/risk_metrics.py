"""일별 평가액 곡선 → 리스크 지표 일괄 계산.

MDD 숫자 하나로는 안 보이는 것들 — 낙폭이 **얼마나 오래** 지속됐는지
(수면기간), 최대 낙폭에서 복구까지 걸린 시간, 변동성 대비 보상
(Sharpe/Calmar) — 을 함께 계산한다. 리스크 대시보드 페이지의 엔진.
"""

from __future__ import annotations

import math

import pandas as pd

# MDD 기반 안정성 등급 (통념적 구간).
GRADE_BANDS = [
    (-0.10, "매우 안정"),
    (-0.20, "안정적"),
    (-0.35, "시장 수준"),
    (-0.50, "공격적"),
]
SPECULATIVE = "투기적"


def grade_by_mdd(mdd: float) -> str:
    """MDD(음수) → 안정성 등급 문자열."""
    for threshold, label in GRADE_BANDS:
        if mdd >= threshold:
            return label
    return SPECULATIVE


def compute_risk_metrics(values: pd.Series) -> dict:
    """일별 평가액 Series → 리스크 지표 dict.

    반환 키:
      total_return, cagr, mdd, recovery_needed(원금 복구에 필요한
      수익률), ann_vol, sharpe(rf=0), calmar, longest_underwater_days
      (최장 수면기간, 달력일), max_dd_trough(최대 낙폭 저점 날짜),
      max_dd_recovery_days(최대 낙폭 고점→회복 달력일, 미회복이면
      None), worst_year(최악 연도 수익률), var95(일간 5% VaR), grade
    """
    values = values.dropna()
    if len(values) < 2:
        raise ValueError("need at least 2 points")
    initial = float(values.iloc[0])
    final = float(values.iloc[-1])
    n_days = (values.index[-1] - values.index[0]).days
    years = max(n_days / 365.25, 1e-9)
    cagr = math.exp(math.log(final / initial) / years) - 1.0 if final > 0 else -1.0

    peak = values.cummax()
    dd = values / peak - 1.0
    mdd = float(dd.min())

    # 최장 수면기간: 신고점 사이의 최대 간격 (진행 중 구간 포함).
    at_peak_dates = values.index[values >= peak * (1 - 1e-12)]
    longest = 0
    prev = values.index[0]
    for ts in at_peak_dates:
        longest = max(longest, (ts - prev).days)
        prev = ts
    longest = max(longest, (values.index[-1] - prev).days)

    # 최대 낙폭 구간: 저점 → 직전 고점, 그리고 회복 시점.
    trough_ts = dd.idxmin()
    pre = values.loc[:trough_ts]
    peak_ts = pre.idxmax()
    peak_value = float(values.loc[peak_ts])
    after = values.loc[trough_ts:]
    recovered = after[after >= peak_value]
    recovery_days = (
        int((recovered.index[0] - peak_ts).days) if len(recovered) else None
    )

    ret = values.pct_change().dropna()
    ann_vol = float(ret.std()) * math.sqrt(252) if len(ret) > 1 else 0.0
    sharpe = (
        float(ret.mean()) / float(ret.std()) * math.sqrt(252)
        if len(ret) > 1 and float(ret.std()) > 0
        else 0.0
    )
    yearly = values.groupby(values.index.year).agg(["first", "last"])
    worst_year = float((yearly["last"] / yearly["first"] - 1.0).min())
    var95 = float(ret.quantile(0.05)) if len(ret) > 1 else 0.0

    return {
        "total_return": final / initial - 1.0,
        "cagr": cagr,
        "mdd": mdd,
        "recovery_needed": (1.0 / (1.0 + mdd) - 1.0) if mdd > -1.0 else float("inf"),
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "calmar": cagr / abs(mdd) if mdd < 0 else float("inf"),
        "longest_underwater_days": int(longest),
        "max_dd_trough": trough_ts.date(),
        "max_dd_recovery_days": recovery_days,
        "worst_year": worst_year,
        "var95": var95,
        "grade": grade_by_mdd(mdd),
    }
