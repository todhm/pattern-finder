"""TqqqP2pStrategy — 채권 사다리 + 현금흐름 리밸런싱 단위 테스트.

가격을 상수/계단으로 고정해 이자 지급·만기 상환·수수료·양도세
수학을 하나씩 검증한다.
"""

from datetime import date

import pandas as pd
import pytest

from strategy.adapters.tqqq_p2p_strategy import TqqqP2pStrategy
from strategy.domain.models import TossFeeSchedule, TqqqP2pConfig


def _series(values: list[float], start: str = "2024-01-02") -> pd.Series:
    idx = pd.bdate_range(start, periods=len(values))
    return pd.Series(values, index=idx, dtype=float)


def _flat_series(months: int, price: float = 100.0) -> pd.Series:
    """월 경계가 ``months``번 나오는 상수 가격 시계열 (영업일 기준)."""
    idx = pd.bdate_range("2024-01-02", periods=months * 23)
    return pd.Series(price, index=idx, dtype=float)


def _config(**overrides) -> TqqqP2pConfig:
    defaults = dict(
        start_date=date(2024, 1, 2),
        end_date=date(2025, 12, 31),
        initial_capital=100_000_000.0,
        tqqq_weight=0.5,
        bond_annual_rate=0.09,
        bond_maturity_months=6,
        allow_sell_rebalance=False,
        fee_schedule=TossFeeSchedule(),
        capital_gains_tax_pct=0.22,
        tax_deduction=2_500_000.0,
    )
    defaults.update(overrides)
    return TqqqP2pConfig(**defaults)


class TestInitialSplit:
    def test_day0_split_and_buy_fee(self):
        cfg = _config()
        result = TqqqP2pStrategy().execute(_flat_series(2), cfg)
        pt0 = result.curve[0]
        # TQQQ엔 5천만을 썼지만 수수료 0.1%가 빠진 노셔널만 평가액.
        expected_notional = 50_000_000 / 1.001
        assert pt0.tqqq_value == pytest.approx(expected_notional)
        assert pt0.bond_value == pytest.approx(50_000_000)
        assert pt0.cash == 0.0

    def test_needs_two_bars(self):
        with pytest.raises(ValueError):
            TqqqP2pStrategy().execute(_series([100.0]), _config())


class TestBondLadder:
    def test_first_month_interest(self):
        cfg = _config()
        result = TqqqP2pStrategy().execute(_flat_series(2), cfg)
        e0 = result.events[0]
        # 첫 경계: 채권 5천만 × 9%/12 이자.
        assert e0.interest == pytest.approx(50_000_000 * 0.09 / 12)
        assert e0.matured_principal == 0.0

    def test_principal_returns_at_maturity(self):
        cfg = _config()
        result = TqqqP2pStrategy().execute(_flat_series(8), cfg)
        # 6번째 월 경계에서 day0 채권 원금 5천만 상환.
        e5 = result.events[5]
        assert e5.matured_principal == pytest.approx(50_000_000)

    def test_interest_total_flat_price(self):
        # 가격이 안 움직이면 TQQQ 쪽은 초기 수수료만큼 목표에 미달
        # → 이자가 TQQQ 매수로 먼저 가고 나머지는 채권 재투자.
        cfg = _config()
        result = TqqqP2pStrategy().execute(_flat_series(4), cfg)
        assert result.liquidation.total_interest > 0
        # 모든 이벤트에서 이자는 채권 잔액 × 월리와 일치.
        prev_bond = 50_000_000.0
        for e in result.events:
            assert e.interest == pytest.approx(prev_bond * 0.09 / 12)
            prev_bond = e.bond_value_after


class TestCashflowRebalance:
    def test_flat_price_keeps_target_weight(self):
        cfg = _config()
        result = TqqqP2pStrategy().execute(_flat_series(12), cfg)
        # 상수 가격이면 이자가 계속 유입되며 50:50 근처 유지.
        final_w = result.events[-1].tqqq_weight_after
        assert final_w == pytest.approx(0.5, abs=0.01)

    def test_no_sell_when_disallowed(self):
        # 가격 급등 → TQQQ 초과. 현금흐름 모드에선 매도(음수 traded) 없음.
        values = [100.0] * 23 + [300.0] * 46
        cfg = _config()
        result = TqqqP2pStrategy().execute(_series(values), cfg)
        assert all(e.tqqq_traded >= 0 for e in result.events)
        # 초과 상태에선 이자 전액이 채권으로.
        overweight_events = [
            e for e in result.events if e.tqqq_weight_after > 0.5
        ]
        assert overweight_events
        assert all(e.tqqq_traded == 0 for e in overweight_events)

    def test_sell_rebalance_restores_target(self):
        values = [100.0] * 23 + [300.0] * 46
        cfg = _config(allow_sell_rebalance=True)
        result = TqqqP2pStrategy().execute(_series(values), cfg)
        sells = [e for e in result.events if e.tqqq_traded < 0]
        assert sells
        assert sells[0].tqqq_weight_after == pytest.approx(0.5, abs=0.005)


class TestFeesAndTax:
    def test_flat_price_no_tax(self):
        cfg = _config()
        result = TqqqP2pStrategy().execute(_flat_series(4), cfg)
        # 차익이 없으니 세금 0 (매도 수수료로 오히려 소폭 손실).
        assert result.liquidation.total_tax == 0.0

    def test_final_liquidation_tax(self):
        # 2배 상승 후 청산 → (차익 − 공제) × 22% 과세.
        values = [100.0] * 23 + [200.0] * 23
        cfg = _config()
        result = TqqqP2pStrategy().execute(_series(values), cfg)
        liq = result.liquidation
        assert liq.final_tax > 0
        expected = (
            max(0.0, liq.realized_gain_total - cfg.tax_deduction) * 0.22
        )
        assert liq.total_tax == pytest.approx(expected)
        assert liq.final_value_after_tax < liq.final_value_pre_tax

    def test_annual_tax_settlement_with_sells(self):
        # 매도 리밸런싱 + 연도 경계 → 연초 정산 이벤트에 세금 기록.
        idx = pd.bdate_range("2024-06-03", periods=200)
        values = pd.Series(
            [100.0] * 20 + [250.0] * 180, index=idx, dtype=float
        )
        cfg = _config(allow_sell_rebalance=True)
        result = TqqqP2pStrategy().execute(values, cfg)
        january = [e for e in result.events if e.date.year == 2025][0]
        assert january.tax_paid > 0

    def test_zero_fee_zero_tax_matches_gross(self):
        fee = TossFeeSchedule(
            buy_commission_pct=0.0, sell_commission_pct=0.0, sec_fee_pct=0.0
        )
        cfg = _config(
            fee_schedule=fee, capital_gains_tax_pct=0.0, tax_deduction=0.0
        )
        result = TqqqP2pStrategy().execute(_flat_series(2), cfg)
        assert result.liquidation.final_value_after_tax == pytest.approx(
            result.liquidation.final_value_pre_tax
        )


class TestBenchmarks:
    def test_bond_only_monthly_compounding(self):
        cfg = _config()
        result = TqqqP2pStrategy().execute(_flat_series(4), cfg)
        name = "P2P 채권 100% (연 9% 월복리)"
        points = result.benchmark_curves[name]
        n_boundaries = len(result.events)
        expected = 100_000_000 * (1 + 0.09 / 12) ** n_boundaries
        assert points[-1].equity == pytest.approx(expected)

    def test_benchmark_after_tax_present(self):
        cfg = _config()
        result = TqqqP2pStrategy().execute(_flat_series(2), cfg)
        assert set(result.benchmark_after_tax) == {
            b.name for b in result.benchmarks
        }
