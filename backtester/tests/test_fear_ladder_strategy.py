"""FearLadderStrategy — 공포 사다리 규칙 단위 테스트."""

from datetime import date

import pandas as pd
import pytest

from strategy.adapters.fear_ladder_strategy import FearLadderStrategy
from strategy.domain.models import FearLadderConfig, TossFeeSchedule

NO_FEE = TossFeeSchedule(
    buy_commission_pct=0.0, sell_commission_pct=0.0, sec_fee_pct=0.0
)


def _series(values: list[float], start: str = "2024-01-02") -> pd.Series:
    idx = pd.bdate_range(start, periods=len(values))
    return pd.Series(values, index=idx, dtype=float)


def _config(**overrides) -> FearLadderConfig:
    defaults = dict(
        start_date=date(2024, 1, 2),
        end_date=date(2025, 12, 31),
        initial_capital=100_000_000.0,
        base_stock_weight=0.5,
        cash_annual_rate=0.0,
        trigger_mode="ath",
        levels=[-0.20, -0.35, -0.50],
        deploy_fractions=[1 / 3, 1 / 2, 1.0],
        fee_schedule=NO_FEE,
        capital_gains_tax_pct=0.0,
        tax_deduction=0.0,
    )
    defaults.update(overrides)
    return FearLadderConfig(**defaults)


class TestLadder:
    def test_base_split_then_ladder_deploys(self):
        # 100 → -21% → -36% → -51%: 사다리 3단 순차 발동.
        vals = [100, 100, 79, 79, 64, 64, 49, 49]
        r = FearLadderStrategy().execute(_series(vals), _config())
        fear = [e for e in r.events if e.kind == "fear_buy"]
        assert [e.level for e in fear] == [1, 2, 3]
        # 1단: 현금 50M의 1/3, 2단: 남은 현금의 1/2, 3단: 전부.
        assert fear[0].notional == pytest.approx(50_000_000 / 3)
        assert fear[2].cash_after == pytest.approx(0.0)
        assert r.cycles[-1].levels_hit == 3
        assert r.cycles[-1].outcome == "open"

    def test_gap_crash_triggers_multiple_levels_same_day(self):
        vals = [100, 100, 45, 45]  # 하루에 -55%
        r = FearLadderStrategy().execute(_series(vals), _config())
        fear = [e for e in r.events if e.kind == "fear_buy"]
        assert [e.level for e in fear] == [1, 2, 3]
        assert fear[0].date == fear[2].date

    def test_recovery_rebalances_to_base_and_resets(self):
        # -21% 발동 후 신고점 회복 → 평시 50%로 익절, 사다리 리셋 후
        # 재하락 시 다시 발동.
        vals = [100, 100, 79, 90, 101, 101, 79, 79]
        r = FearLadderStrategy().execute(_series(vals), _config())
        recov = [e for e in r.events if e.kind == "recovery"]
        assert len(recov) == 1
        assert recov[0].stock_weight_after == pytest.approx(0.5, abs=0.01)
        assert r.cycles[0].outcome == "recovered"
        assert r.cycles[0].harvested > 0
        # 리셋 후 두 번째 사이클 발동.
        assert len([e for e in r.events if e.kind == "fear_buy"]) == 2

    def test_no_trigger_in_shallow_dip(self):
        vals = [100, 100, 85, 85]  # -15% — 1단(-20%) 미달
        r = FearLadderStrategy().execute(_series(vals), _config())
        assert not [e for e in r.events if e.kind == "fear_buy"]
        assert r.cycles == []


class TestMaMode:
    def test_ma_discount_triggers(self):
        # 5일 MA 기준 -25% 이탈 → 1단(-20%) 발동.
        cfg = _config(trigger_mode="ma", ma_days=5, levels=[-0.20],
                      deploy_fractions=[1.0])
        vals = [100] * 6 + [74, 74]
        r = FearLadderStrategy().execute(_series(vals), cfg)
        fear = [e for e in r.events if e.kind == "fear_buy"]
        assert len(fear) == 1
        assert fear[0].drawdown < -0.20

    def test_ma_recovery_needs_confirm_days(self):
        cfg = _config(trigger_mode="ma", ma_days=3, levels=[-0.20],
                      deploy_fractions=[1.0], recovery_confirm_days=2)
        # 회복 후 상승 지속 — MA가 가격을 따라잡으면(횡보) strict >가
        # 안 되므로 상승 꼬리로 확인일수 검증.
        vals = [100, 100, 100, 70, 120, 121, 122, 123, 124]
        r = FearLadderStrategy().execute(_series(vals), cfg)
        recov = [e for e in r.events if e.kind == "recovery"]
        assert len(recov) == 1


class TestVixGate:
    def test_vix_gate_blocks_without_fear(self):
        vals = [100, 100, 79, 79]
        vix = _series([12.0] * 4)  # 평온 → 발동 금지
        cfg = _config(vix_confirm=True, vix_threshold=30.0)
        r = FearLadderStrategy().execute(_series(vals), cfg, vix=vix)
        assert not [e for e in r.events if e.kind == "fear_buy"]


class TestCosts:
    def test_interest_fees_tax(self):
        vals = [100, 100, 79, 90, 101, 101]
        cfg = _config(
            cash_annual_rate=0.09,
            fee_schedule=TossFeeSchedule(),
            capital_gains_tax_pct=0.22,
        )
        r = FearLadderStrategy().execute(_series(vals), cfg)
        liq = r.liquidation
        assert liq.total_interest > 0
        assert liq.total_fees > 0
        assert liq.final_value_after_tax < liq.final_value_pre_tax
        assert set(r.benchmark_after_tax) == {b.name for b in r.benchmarks}
