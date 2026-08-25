"""risk_metrics — 손으로 만든 곡선으로 지표 하나씩 검증."""

import pandas as pd
import pytest

from strategy.adapters.risk_metrics import compute_risk_metrics, grade_by_mdd


def _series(values: list[float]) -> pd.Series:
    idx = pd.bdate_range("2024-01-01", periods=len(values))
    return pd.Series(values, index=idx, dtype=float)


class TestGrade:
    def test_bands(self):
        assert grade_by_mdd(-0.05) == "매우 안정"
        assert grade_by_mdd(-0.15) == "안정적"
        assert grade_by_mdd(-0.30) == "시장 수준"
        assert grade_by_mdd(-0.45) == "공격적"
        assert grade_by_mdd(-0.80) == "투기적"


class TestMetrics:
    def test_mdd_and_recovery(self):
        # 100 → 120(고점) → 54(-55%) → 130(회복).
        m = compute_risk_metrics(_series([100, 120, 54, 90, 130]))
        assert m["mdd"] == pytest.approx(-0.55)
        assert m["recovery_needed"] == pytest.approx(1.0 / 0.45 - 1.0)
        assert m["grade"] == "투기적"
        # 고점(2일째)→회복(5일째): 달력일 기준 3일.
        assert m["max_dd_recovery_days"] == 3

    def test_grade_boundary_minus_50_is_aggressive(self):
        assert grade_by_mdd(-0.50) == "공격적"

    def test_unrecovered_drawdown_is_none(self):
        m = compute_risk_metrics(_series([100, 120, 60, 70]))
        assert m["max_dd_recovery_days"] is None

    def test_monotonic_growth_no_drawdown(self):
        m = compute_risk_metrics(_series([100, 101, 102, 103]))
        assert m["mdd"] == pytest.approx(0.0)
        # 매일 신고점 — 수면기간은 거래일 간 달력 간격(주말 최대 3일)뿐.
        assert m["longest_underwater_days"] <= 3
        assert m["grade"] == "매우 안정"

    def test_longest_underwater_includes_ongoing(self):
        # 2일째 고점 이후 끝까지 미회복 — 진행 중 수면기간 포함.
        vals = [100, 120] + [110] * 20
        m = compute_risk_metrics(_series(vals))
        idx = pd.bdate_range("2024-01-01", periods=len(vals))
        assert m["longest_underwater_days"] == (idx[-1] - idx[1]).days

    def test_needs_two_points(self):
        with pytest.raises(ValueError):
            compute_risk_metrics(_series([100.0]))
