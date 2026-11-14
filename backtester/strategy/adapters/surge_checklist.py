"""S7 수동 체크리스트(M1~M4 사기 · X1~X5 피하기)의 기계 판 (순수 pandas).

사람이 차트에서 보던 항목을 종가·거래대금만으로 근사한다. 모든 값은 그날 종가까지의 정보다.
입력은 날짜×종목 행렬(분할 조정 종가 c, 거래대금 dv).
"""

from __future__ import annotations

import pandas as pd

BUY = ("m1", "m2", "m3", "m4")
AVOID = ("x1", "x2", "x3", "x4", "x5")
LABELS = {
    "m1": "M1 매끄러운 오름(종가>50일선>200일선, 50일선 상승)",
    "m2": "M2 과열 아님(50일선 +25% 안쪽)",
    "m3": "M3 고점 근처 조용한 쉼(25일 폭 ≤15%·고점 -10% 안·거래 줄어듦)",
    "m4": "M4 오르는 날 거래가 붙음(50일 상승일 거래대금 > 하락일)",
    "x1": "X1 인수 확정형(120일 안 +15% 갭 뒤 20일 변동 거의 없음)",
    "x2": "X2 분배(63일 안 -8% 이하 & 거래대금 3배 음봉)",
    "x3": "X3 꺾임(종가<50일선, 50일선 하락)",
    "x4": "X4 포물선(50일선 +40% 초과)",
    "x5": "X5 톱니(63일 안 하루 ±10% 3회 이상)",
}


def checklist_features(c: pd.DataFrame, dv: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """항목별 bool 행렬. 자료가 모자라 계산 못 한 칸은 False."""
    r = c.pct_change(fill_method=None)
    s50 = c.rolling(50, min_periods=30).mean()
    s200 = c.rolling(200, min_periods=120).mean()
    ext = c / s50 - 1.0
    dvr = dv / dv.rolling(50, min_periods=20).mean()
    up_dv = dv.where(r > 0).rolling(50, min_periods=10).mean()
    dn_dv = dv.where(r < 0).rolling(50, min_periods=10).mean()
    width25 = c.rolling(25).max() / c.rolling(25).min() - 1.0
    return {
        "m1": (c > s50) & (s50 > s200) & (s50 > s50.shift(20)),
        "m2": ext <= 0.25,
        "m3": (width25 <= 0.15) & (c >= 0.9 * c.rolling(252, min_periods=200).max()) & (dv.rolling(25).mean() < dv.rolling(50).mean().shift(25)),
        "m4": up_dv > dn_dv,
        "x1": (r.rolling(120, min_periods=60).max() >= 0.15) & (r.rolling(20).std() < 0.005),
        "x2": ((r <= -0.08) & (dvr >= 3.0)).astype(float).rolling(63, min_periods=1).max() > 0,
        "x3": (c < s50) & (s50 < s50.shift(20)),
        "x4": ext > 0.40,
        "x5": (r.abs() >= 0.10).astype(float).rolling(63, min_periods=1).sum() >= 3,
    }


def checklist_score(f: dict[str, pd.DataFrame], skip: tuple[str, ...] = ()) -> pd.DataFrame:
    """사기 항목 수 − 피하기 항목 수 (−5 ~ +4). skip 항목은 세지 않는다."""
    s = sum(f[k].astype(int) for k in BUY if k not in skip)
    return s - sum(f[k].astype(int) for k in AVOID if k not in skip)
