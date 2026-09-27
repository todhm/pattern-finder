"""바벨 포트폴리오 결합기 — 안전 슬리브 + 위험 슬리브.

탈레브식 바벨: 자본 대부분을 검증된 안전 전략에, 소량(5~10%)을
볼록성 큰 초고위험 전략에 배분하고 주기적으로 비중을 되돌린다.
리밸런싱이 "위험 슬리브 폭등 시 익절 → 안전으로, 폭락 시 재충전"을
자동 수행해, 위험 슬리브가 전멸해도 손실이 배분 비중으로 캡핑된다.

입력은 **각 슬리브의 일별 평가액 곡선**(각자 수수료·세금 반영 후) —
결합기는 곡선의 일수익률만 사용하므로 슬리브 전략의 구현과 독립적이다.
"""

from __future__ import annotations

import pandas as pd

REBALANCE_MODES = ("yearly", "quarterly", "never")


def combine_barbell_detailed(
    safe_values: pd.Series,
    risky_values: pd.Series,
    w_safe: float,
    rebalance: str = "yearly",
    initial_capital: float | None = None,
) -> tuple[pd.DataFrame, list[dict]]:
    """바벨 결합 — 슬리브별 경로와 리밸런싱 내역까지 반환.

    반환:
      - DataFrame(index=합집합 달력) columns: "total", "safe", "risky",
        "risky_weight"(위험 슬리브 비중).
      - 리밸런싱 이벤트 리스트: {"date", "risky_weight_before",
        "moved"} — ``moved`` > 0 이면 위험→안전(익절 회수),
        < 0 이면 안전→위험(재충전).
    """
    if rebalance not in REBALANCE_MODES:
        raise ValueError(f"rebalance must be one of {REBALANCE_MODES}")
    if not 0.0 <= w_safe <= 1.0:
        raise ValueError("w_safe must be in [0, 1]")

    idx = safe_values.index.union(risky_values.index).sort_values()
    rs = safe_values.reindex(idx).ffill().pct_change().fillna(0.0)
    rr = risky_values.reindex(idx).ffill().pct_change().fillna(0.0)

    capital = (
        float(initial_capital)
        if initial_capital is not None
        else float(safe_values.dropna().iloc[0])
    )
    a = capital * w_safe
    b = capital * (1.0 - w_safe)
    prev_year = idx[0].year
    prev_quarter = (idx[0].year, (idx[0].month - 1) // 3)
    rows = []
    events: list[dict] = []
    for ts, ra, rb in zip(idx, rs.values, rr.values):
        boundary = (
            (rebalance == "yearly" and ts.year != prev_year)
            or (
                rebalance == "quarterly"
                and (ts.year, (ts.month - 1) // 3) != prev_quarter
            )
        )
        if boundary:
            total = a + b
            target_b = total * (1.0 - w_safe)
            events.append(
                {
                    "date": ts.date(),
                    "risky_weight_before": b / total if total > 0 else 0.0,
                    "moved": b - target_b,
                }
            )
            a, b = total * w_safe, target_b
        prev_year = ts.year
        prev_quarter = (ts.year, (ts.month - 1) // 3)
        a *= 1.0 + ra
        b *= 1.0 + rb
        total = a + b
        rows.append((total, a, b, b / total if total > 0 else 0.0))
    df = pd.DataFrame(
        rows, index=idx, columns=["total", "safe", "risky", "risky_weight"]
    )
    return df, events


def combine_barbell(
    safe_values: pd.Series,
    risky_values: pd.Series,
    w_safe: float,
    rebalance: str = "yearly",
    initial_capital: float | None = None,
) -> pd.Series:
    """두 슬리브 평가액 곡선을 바벨로 결합한 일별 평가액을 반환.

    - 달력은 두 곡선의 합집합, 결측일은 직전값 유지(ffill).
    - ``rebalance``: "yearly"(연초) / "quarterly"(분기초) 비중 리셋,
      "never"는 초기 배분 후 방치.
    - ``initial_capital`` 미지정 시 안전 슬리브 첫 값 사용.
    """
    df, _ = combine_barbell_detailed(
        safe_values, risky_values, w_safe, rebalance, initial_capital
    )
    return df["total"]
