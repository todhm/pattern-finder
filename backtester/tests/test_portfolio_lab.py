"""portfolio_lab — 가중치 그리드 / 바스켓 합성 / 점수 테스트."""

import pandas as pd
import pytest

from strategy.adapters import portfolio_lab as pl


def _series(values, start="2024-01-01"):
    idx = pd.bdate_range(start, periods=len(values))
    return pd.Series(values, index=idx, dtype=float)


class TestWeightGrid:
    def test_two_assets_half_step(self):
        grid = pl.enumerate_weight_grid(["a", "b"], step=0.5)
        assert {frozenset(g.items()) for g in grid} == {
            frozenset({("a", 1.0)}),
            frozenset({("a", 0.5), ("b", 0.5)}),
            frozenset({("b", 1.0)}),
        }

    def test_grid_size_three_assets_20pct(self):
        # 5 유닛을 3자산에 배분: C(5+2, 2) = 21.
        grid = pl.enumerate_weight_grid(["a", "b", "c"], step=0.2)
        assert len(grid) == 21
        for g in grid:
            assert sum(g.values()) == pytest.approx(1.0)

    def test_invalid_step_rejected(self):
        with pytest.raises(ValueError, match="divide"):
            pl.enumerate_weight_grid(["a"], step=0.3)


class TestBasket:
    def test_single_asset_tracks_price(self):
        gold = _series([100.0, 110.0, 99.0])
        basket = pl.build_basket_series({"gold": gold}, {"gold": 1.0})
        assert basket.iloc[0] == pytest.approx(100.0)
        assert basket.iloc[1] == pytest.approx(110.0)
        assert basket.iloc[2] == pytest.approx(99.0)

    def test_5050_daily_rebalanced(self):
        a = _series([100.0, 120.0])  # +20%
        b = _series([100.0, 90.0])  # -10%
        basket = pl.build_basket_series({"a": a, "b": b}, {"a": 0.5, "b": 0.5})
        # 일수익률 = 0.5×20% + 0.5×(-10%) = +5%.
        assert basket.iloc[-1] == pytest.approx(105.0)

    def test_cash_leg_dampens_returns(self):
        a = _series([100.0, 120.0])
        basket = pl.build_basket_series(
            {"a": a, pl.CASH: None}, {"a": 0.5, pl.CASH: 0.5}
        )
        # 절반 현금 → +10%.
        assert basket.iloc[-1] == pytest.approx(110.0)

    def test_weights_renormalized(self):
        a = _series([100.0, 110.0])
        basket = pl.build_basket_series({"a": a}, {"a": 2.0})
        assert basket.iloc[-1] == pytest.approx(110.0)

    def test_all_cash_raises(self):
        with pytest.raises(ValueError, match="calendar"):
            pl.build_basket_series({pl.CASH: None}, {pl.CASH: 1.0})

    def test_misaligned_inner_join(self):
        a = _series([100.0, 110.0, 121.0])
        b = pd.Series(
            [50.0, 50.0], index=[a.index[0], a.index[2]], dtype=float
        )
        basket = pl.build_basket_series(
            {"a": a, "b": b}, {"a": 0.5, "b": 0.5}
        )
        assert len(basket) == 2  # 활성 자산들의 교집합 달력


class TestScores:
    def test_defense_score_anchors(self):
        # 무손실 = 100, 기준과 동일 = 0, 기준의 절반 손실 = 50.
        assert pl.defense_score(0.0, -0.5) == pytest.approx(100.0)
        assert pl.defense_score(-0.5, -0.5) == pytest.approx(0.0)
        assert pl.defense_score(-0.25, -0.5) == pytest.approx(50.0)
        # 기준보다 나쁘면 음수, 위기에 수익이면 100 초과.
        assert pl.defense_score(-0.75, -0.5) < 0
        assert pl.defense_score(0.10, -0.5) == pytest.approx(110.0)

    def test_defense_bonus_is_capped(self):
        # 위기 수익 +50%여도 120 상한 — 저성장 달러/현금 조합이
        # 위기 보너스만으로 종합 1위를 가져가는 왜곡 방지.
        assert pl.defense_score(0.50, -0.5) == pytest.approx(120.0)

    def test_mdd_score_anchors(self):
        assert pl.mdd_score(0.0, -0.4) == pytest.approx(100.0)
        assert pl.mdd_score(-0.4, -0.4) == pytest.approx(0.0)
        assert pl.mdd_score(-0.2, -0.4) == pytest.approx(50.0)
        assert pl.mdd_score(-0.6, -0.4) < 0

    def test_growth_score_relative_to_baseline(self):
        assert pl.growth_score(0.20, 0.20) == pytest.approx(100.0)
        assert pl.growth_score(0.30, 0.20) == pytest.approx(150.0)
        # 기준이 0 이하면 절대 스케일 fallback.
        assert pl.growth_score(0.10, -0.05) == pytest.approx(110.0)

    def test_window_return_out_of_range_is_none(self):
        v = _series([100.0, 110.0])
        assert pl.window_return(v, "1990-01-01", "1990-12-31") is None

    def test_label_weights_sorted_desc(self):
        label = pl.label_weights(
            {"GC=F": 0.4, "TLT": 0.6, "VOO": 0.0},
            {"GC=F": "금", "TLT": "장기국채"},
        )
        assert label == "장기국채 60% + 금 40%"
