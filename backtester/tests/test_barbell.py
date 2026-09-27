"""combine_barbell — 바벨 결합 수학 단위 테스트."""

import pandas as pd
import pytest

from strategy.adapters.barbell import combine_barbell


def _series(values, start="2024-01-01"):
    idx = pd.bdate_range(start, periods=len(values))
    return pd.Series(values, index=idx, dtype=float)


class TestCombine:
    def test_never_equals_weighted_normalized_sum(self):
        safe = _series([100, 110, 121])
        risky = _series([100, 200, 400])
        v = combine_barbell(safe, risky, w_safe=0.9, rebalance="never",
                            initial_capital=1000.0)
        # 방치: 900×(safe/100) + 100×(risky/100)
        assert v.iloc[-1] == pytest.approx(900 * 1.21 + 100 * 4.0)

    def test_yearly_rebalance_resets_weights(self):
        # 연말까지 위험이 4배 → 연초에 90/10으로 리셋 후 위험 반토막.
        idx = pd.to_datetime(["2024-12-30", "2024-12-31", "2025-01-02"])
        safe = pd.Series([100.0, 100.0, 100.0], index=idx)
        risky = pd.Series([100.0, 400.0, 200.0], index=idx)
        v = combine_barbell(safe, risky, 0.9, "yearly", initial_capital=1000.0)
        # 12/31: 900 + 400 = 1300 → 1/2 리셋: 안전 1170, 위험 130 → 위험 -50%
        assert v.iloc[1] == pytest.approx(1300.0)
        assert v.iloc[2] == pytest.approx(1170.0 + 65.0)

    def test_risky_wipeout_capped_by_allocation(self):
        safe = _series([100, 100, 100, 100])
        risky = _series([100, 50, 10, 0.01])
        v = combine_barbell(safe, risky, 0.95, "never", initial_capital=1000.0)
        assert v.iloc[-1] >= 950.0  # 손실이 위험 배분 5%로 캡

    def test_union_calendar_ffill(self):
        safe = _series([100, 101, 102, 103])  # 평일
        idx = pd.to_datetime(["2024-01-01", "2024-01-06", "2024-01-07"])
        risky = pd.Series([100.0, 110.0, 120.0], index=idx)  # 주말 포함
        v = combine_barbell(safe, risky, 0.5, "never", initial_capital=100.0)
        assert len(v) == len(safe.index.union(idx))

    def test_invalid_args(self):
        s = _series([100, 101])
        with pytest.raises(ValueError):
            combine_barbell(s, s, 0.5, "monthly")
        with pytest.raises(ValueError):
            combine_barbell(s, s, 1.5)


class TestDetailed:
    def test_components_sum_to_total_and_events(self):
        from strategy.adapters.barbell import combine_barbell_detailed

        idx = pd.to_datetime(["2024-12-30", "2024-12-31", "2025-01-02"])
        safe = pd.Series([100.0, 100.0, 100.0], index=idx)
        risky = pd.Series([100.0, 400.0, 200.0], index=idx)
        df, events = combine_barbell_detailed(
            safe, risky, 0.9, "yearly", initial_capital=1000.0
        )
        assert (df["safe"] + df["risky"]).round(6).equals(df["total"].round(6))
        # 연초 리셋 1회: 리셋 전 위험 비중 400/1300, 이동액 = 400 − 130 (익절 회수)
        assert len(events) == 1
        assert events[0]["risky_weight_before"] == pytest.approx(400 / 1300)
        assert events[0]["moved"] == pytest.approx(400 - 130.0)
