"""DipBuyStrategy — 전일 급락 매수 규칙 단위 테스트."""

from datetime import date

import pandas as pd
import pytest

from strategy.adapters.dip_buy_strategy import DipBuyStrategy
from strategy.domain.models import DipBuyConfig, TossFeeSchedule

NO_FEE = TossFeeSchedule(
    buy_commission_pct=0.0, sell_commission_pct=0.0, sec_fee_pct=0.0
)


def _df(rows: list[tuple], start: str = "2024-01-02") -> pd.DataFrame:
    idx = pd.bdate_range(start, periods=len(rows))
    return pd.DataFrame(
        {
            "Open": [r[0] for r in rows],
            "High": [r[1] for r in rows],
            "Low": [0.0] * len(rows),
            "Close": [r[2] for r in rows],
        },
        index=idx,
    )


def _config(**overrides) -> DipBuyConfig:
    defaults = dict(
        start_date=date(2024, 1, 2),
        end_date=date(2025, 12, 31),
        initial_capital=90_000_000.0,
        drop_pct=0.05,
        target_pct=0.012,
        split=3,
        fee_schedule=NO_FEE,
        capital_gains_tax_pct=0.0,
        tax_deduction=0.0,
    )
    defaults.update(overrides)
    return DipBuyConfig(**defaults)


class TestEntry:
    def test_buys_one_third_after_drop_day(self):
        # day1 종가 -6% → day2 시가 매수 (총액 90M의 1/3 = 30M).
        rows = [(100, 100, 100), (95, 95, 94), (93, 93, 93), (93, 93, 93)]
        r = DipBuyStrategy().execute(_df(rows), _config())
        assert len(r.cycles) == 1
        c = r.cycles[0]
        assert c.n_buys >= 1
        assert c.invested == pytest.approx(30_000_000)
        assert c.start == date(2024, 1, 4)

    def test_no_entry_without_drop(self):
        rows = [(100, 100, 100), (99, 99, 98), (98, 98, 98)]
        r = DipBuyStrategy().execute(_df(rows), _config())
        assert r.cycles == []
        assert r.summary.final_value == pytest.approx(90_000_000)

    def test_stacks_up_to_split_times(self):
        # 연속 급락 3일 → 3번 스택 후 현금 소진.
        rows = [
            (100, 100, 100),
            (94, 94, 94),   # -6%
            (88, 88, 88),   # -6.4% → 매수1 (시가 88)
            (82, 82, 82),   # -6.8% → 매수2
            (77, 77, 77),   # 매수3
            (72, 72, 72),   # 현금 부족 → 스택 한계
        ]
        r = DipBuyStrategy().execute(_df(rows), _config())
        assert r.cycles[-1].n_buys == 3


class TestExit:
    def test_limit_sell_at_target(self):
        # 매수(시가 90) 후 고가가 평단+1.2% 도달 → 목표가 체결.
        rows = [
            (100, 100, 100),
            (94, 94, 94),          # -6%
            (90, 90, 90),          # 매수 @90
            (90.5, 91.2, 90.5),    # 고가 91.2 ≥ 90×1.012=91.08 → 익절
            (90, 90, 90),
        ]
        r = DipBuyStrategy().execute(_df(rows), _config())
        c = r.cycles[0]
        assert c.outcome == "profit"
        assert c.exit_price == pytest.approx(90 * 1.012)
        assert c.pnl == pytest.approx(30_000_000 * 0.012)

    def test_gap_open_above_target_fills_at_open(self):
        rows = [
            (100, 100, 100),
            (94, 94, 94),
            (90, 90, 90),
            (93, 94, 93),  # 시가 93 > 목표 91.08 → 93 체결
        ]
        r = DipBuyStrategy().execute(_df(rows), _config())
        assert r.cycles[0].exit_price == pytest.approx(93.0)

    def test_open_cycle_reported(self):
        rows = [(100, 100, 100), (94, 94, 94), (90, 90, 90), (89, 89, 89)]
        r = DipBuyStrategy().execute(_df(rows), _config())
        assert r.cycles[-1].outcome == "open"
        assert r.win_rate == 0.0


class TestCosts:
    def test_fees_and_tax(self):
        rows = [
            (100, 100, 100),
            (94, 94, 94),
            (90, 90, 90),
            (93, 94, 93),
        ]
        cfg = _config(
            fee_schedule=TossFeeSchedule(), capital_gains_tax_pct=0.22
        )
        r = DipBuyStrategy().execute(_df(rows), cfg)
        liq = r.liquidation
        assert liq.total_fees > 0
        assert liq.total_tax > 0
        assert liq.final_value_after_tax < liq.final_value_pre_tax

    def test_benchmark_after_tax_positive(self):
        rows = [(100, 100, 100), (94, 94, 94), (90, 90, 90), (93, 94, 93)]
        r = DipBuyStrategy().execute(_df(rows), _config())
        assert r.benchmark_after_tax > 0


class TestEvents:
    def test_buy_and_sell_events_recorded(self):
        rows = [
            (100, 100, 100),
            (94, 94, 94),          # -6% 트리거
            (90, 90, 90),          # 매수 @시가 90
            (90.5, 91.2, 90.5),    # 익절 @91.08
        ]
        r = DipBuyStrategy().execute(_df(rows), _config())
        kinds = [e.kind for e in r.events]
        assert kinds == ["buy", "sell"]
        buy, sell = r.events
        assert buy.stack_no == 1
        assert buy.trigger_ret == pytest.approx(-0.06)
        assert buy.price == pytest.approx(90.0)
        assert buy.target_price == pytest.approx(90 * 1.012)
        assert buy.cycle_no == 1
        assert sell.qty == pytest.approx(buy.qty)
        assert sell.cycle_no == 1

    def test_stack_numbers_increment(self):
        rows = [
            (100, 100, 100),
            (94, 94, 94),
            (88, 88, 88),
            (82, 82, 82),
            (77, 77, 77),
        ]
        r = DipBuyStrategy().execute(_df(rows), _config())
        buys = [e for e in r.events if e.kind == "buy"]
        assert [b.stack_no for b in buys] == [1, 2, 3]
        # 물타기로 평단·목표가가 계단식 하락.
        targets = [b.target_price for b in buys]
        assert targets[0] > targets[1] > targets[2]
