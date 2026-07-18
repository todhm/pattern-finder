"""macro_regime — 지표 정량화 / 합성 / 레짐 판정 단위 테스트."""

from datetime import date

import pandas as pd
import pytest

from strategy.adapters import macro_regime as mr
from strategy.adapters.band_rebalance_strategy import BandRebalanceStrategy
from strategy.domain.models import BandRebalanceConfig


def _series(values, start="2024-01-01", freq="B"):
    idx = pd.date_range(start, periods=len(values), freq=freq)
    return pd.Series(values, index=idx, dtype=float)


class TestScorers:
    def test_yield_curve_inversion_scores_zero_with_lag(self):
        # +0.5 → -0.3 역전. shift(1)로 다음 날부터 반영 (look-ahead 방지).
        spread = _series([0.5, 0.5, -0.3, -0.3])
        score = mr.score_yield_curve(spread)
        # index[1] = 전일(+0.5) 반영 → 1, index[3] = 전일(-0.3) → 0.
        assert score.loc[spread.index[1]] == 1.0
        assert score.loc[spread.index[3]] == 0.0
        # 첫날은 전일 값이 없어 제거됨.
        assert spread.index[0] not in score.index

    def test_credit_spread_widening_above_sma_is_risk(self):
        # 20일 평탄 후 급확대 — SMA(5) 위로 올라가면 0.
        oas = _series([3.0] * 10 + [5.0, 6.0])
        score = mr.score_credit_spread(oas, sma_days=5)
        assert score.iloc[-1] == 0.0  # 전일 6.0 > SMA
        assert score.loc[oas.index[6]] == 1.0  # 평탄 구간: 3.0 == SMA → not <...

    def test_vix_calm_vs_stressed(self):
        vix = _series([15.0] * 30 + [40.0] * 30)
        score = mr.score_vix(vix, smooth_days=5, calm_level=25.0)
        assert score.loc[vix.index[20]] == 1.0
        assert score.iloc[-1] == 0.0

    def test_sahm_rule_fires_on_unemployment_rise(self):
        # 실업률 3.5% 안정 → 4.3%로 상승: 3개월 평균이 12개월 저점
        # 대비 +0.5%p 초과 → 침체 신호(0).
        unrate = _series([3.5] * 15 + [4.0, 4.3, 4.5, 4.6], freq="MS")
        score = mr.score_sahm_rule(unrate, publish_lag_days=0)
        # 3개월 평균 + 12개월 저점 워밍업(14개월) 이후부터 산출.
        assert score.iloc[0] == 1.0  # 실업률 안정 구간
        assert score.iloc[-1] == 0.0  # 3개월 평균이 저점 +0.5%p 초과

    def test_sahm_publish_lag_shifts_index(self):
        unrate = _series([3.5] * 15, freq="MS")
        score = mr.score_sahm_rule(unrate, publish_lag_days=40)
        assert score.index[0] == unrate.index[13] + pd.Timedelta(days=40)


class TestComposite:
    def test_equal_weights_average(self):
        idx = pd.date_range("2024-01-01", periods=3, freq="B")
        comps = {
            "a": pd.Series([1.0, 1.0, 0.0], index=idx),
            "b": pd.Series([1.0, 0.0, 0.0], index=idx),
        }
        score = mr.composite_score(comps, {"a": 1.0, "b": 1.0}, idx)
        assert score.tolist() == [1.0, 0.5, 0.0]

    def test_missing_history_renormalizes_weights(self):
        # 'b'는 3일째부터만 존재 → 그 전엔 'a' 단독 100%.
        idx = pd.date_range("2024-01-01", periods=4, freq="B")
        comps = {
            "a": pd.Series([1.0, 0.0, 0.0, 0.0], index=idx),
            "b": pd.Series([1.0, 1.0], index=idx[2:]),
        }
        score = mr.composite_score(comps, {"a": 1.0, "b": 1.0}, idx)
        assert score.tolist() == [1.0, 0.0, 0.5, 0.5]

    def test_all_missing_is_nan(self):
        idx = pd.date_range("2024-01-01", periods=2, freq="B")
        comps = {"a": pd.Series([1.0], index=idx[1:])}
        score = mr.composite_score(comps, {"a": 1.0}, idx)
        assert pd.isna(score.iloc[0]) and score.iloc[1] == 1.0


class TestHysteresis:
    def test_off_immediate_on_needs_confirm(self):
        vals = [1.0, 0.3, 0.3, 0.7, 0.7, 0.7, 0.7]
        score = _series(vals)
        flags = mr.hysteresis_risk_on(
            score, on_threshold=0.6, off_threshold=0.4, confirm_days=2
        )
        # day1 0.3 < 0.4 → 즉시 off. day3~5 (0.7×3) streak 3 > 2 → day5 on.
        assert flags.tolist() == [True, False, False, False, False, True, True]

    def test_streak_resets_on_break(self):
        vals = [1.0, 0.3, 0.7, 0.3, 0.7, 0.7, 0.7]
        score = _series(vals)
        flags = mr.hysteresis_risk_on(score, 0.6, 0.4, confirm_days=2)
        # day2 streak1 → day3 붕괴 reset → day4~6 streak 3 → day6 on.
        assert flags.tolist() == [True, False, False, False, False, False, True]

    def test_nan_keeps_state(self):
        score = _series([float("nan"), 0.3, float("nan")])
        flags = mr.hysteresis_risk_on(score, 0.6, 0.4, 0)
        assert flags.tolist() == [True, False, False]


class TestHybrid:
    def test_trend_break_alone_is_not_enough(self):
        # 추세 붕괴(0)여도 거시 건강(0.8 ≥ veto 0.6) → 계속 risk-on.
        trend = _series([1.0, 0.0, 0.0, 0.0])
        macro = _series([0.8, 0.8, 0.8, 0.8])
        flags = mr.hybrid_risk_on(trend, macro, veto_threshold=0.6)
        assert flags.all()

    def test_trend_break_with_weak_macro_goes_off(self):
        trend = _series([1.0, 0.0, 0.0, 0.0])
        macro = _series([0.8, 0.4, 0.4, 0.4])
        flags = mr.hybrid_risk_on(trend, macro, veto_threshold=0.6)
        assert flags.tolist() == [True, False, False, False]

    def test_reentry_needs_trend_streak_only(self):
        # off 후 거시가 여전히 약해도(0.4) 추세 복귀 streak로 재진입.
        trend = _series([1.0, 0.0, 1.0, 1.0, 1.0, 1.0])
        macro = _series([0.8, 0.3, 0.4, 0.4, 0.4, 0.4])
        flags = mr.hybrid_risk_on(trend, macro, 0.6, confirm_days=2)
        # day2~4 streak 3 > 2 → day4 재진입.
        assert flags.tolist() == [True, False, False, False, True, True]

    def test_nan_macro_means_no_veto(self):
        # 거시 히스토리 이전(NaN) — 거부권 없음 → 추세 붕괴만으론 off 안 됨.
        trend = _series([1.0, 0.0, 0.0])
        macro = pd.Series(dtype=float)
        flags = mr.hybrid_risk_on(trend, macro, 0.6)
        assert flags.all()


class TestStrategyInjection:
    def test_risk_on_series_drives_transitions(self):
        idx = pd.bdate_range("2024-01-02", periods=6)
        agg = pd.Series([100.0] * 6, index=idx)
        dfn = pd.Series([50.0] * 6, index=idx)
        flags = pd.Series([True, True, False, False, True, True], index=idx)
        cfg = BandRebalanceConfig(
            start_date=date(2024, 1, 2),
            end_date=date(2024, 1, 9),
            initial_capital=100_000.0,
        )
        result = BandRebalanceStrategy().execute(
            agg, dfn, cfg, risk_on_series=flags
        )
        kinds = [e.kind for e in result.events]
        assert kinds == ["risk_off", "risk_on"]
        assert result.events[0].date == idx[2].date()
        assert result.events[1].date == idx[4].date()
        # 재진입 후 50:50 복원.
        assert result.events[1].aggressive_weight_after == pytest.approx(0.5)
