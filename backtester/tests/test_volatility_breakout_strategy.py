"""VolatilityBreakoutStrategy — 변동성 돌파 규칙 단위 테스트."""

from datetime import date

import pandas as pd
import pytest

from strategy.adapters.volatility_breakout_strategy import (
    VolatilityBreakoutStrategy,
)
from strategy.domain.models import TossFeeSchedule, VolBreakoutConfig

NO_COST = dict(
    fee_schedule=TossFeeSchedule(
        buy_commission_pct=0.0, sell_commission_pct=0.0, sec_fee_pct=0.0
    ),
    slippage_bp=0.0,
    capital_gains_tax_pct=0.0,
    tax_deduction=0.0,
    financing_annual_rate=0.0,
)


def _df(rows: list[tuple], start: str = "2024-01-02") -> pd.DataFrame:
    """rows = [(open, high, low, close), ...]"""
    idx = pd.bdate_range(start, periods=len(rows))
    return pd.DataFrame(
        {
            "Open": [r[0] for r in rows],
            "High": [r[1] for r in rows],
            "Low": [r[2] for r in rows],
            "Close": [r[3] for r in rows],
        },
        index=idx,
    )


def _config(**overrides) -> VolBreakoutConfig:
    defaults = dict(
        start_date=date(2024, 1, 2),
        end_date=date(2025, 12, 31),
        initial_capital=100_000_000.0,
        k=0.5,
        leverage=1.0,
        **NO_COST,
    )
    defaults.update(overrides)
    return VolBreakoutConfig(**defaults)


class TestTrigger:
    def test_breakout_fills_at_trigger_and_exits_at_close(self):
        # 전일 레인지 10 → 트리거 = 100 + 0.5×10 = 105. 고가 108 도달,
        # 종가 107 청산 → 수익률 (107/105 − 1).
        rows = [(100, 105, 95, 100), (100, 108, 99, 107), (107, 107, 107, 107)]
        r = VolatilityBreakoutStrategy().execute(_df(rows), _config())
        assert len(r.trades) == 1
        t = r.trades[0]
        assert t.entry == pytest.approx(105.0)
        assert t.exit == pytest.approx(107.0)
        assert t.pnl == pytest.approx(1e8 * (107 / 105 - 1))

    def test_trigger_anchors_to_todays_open_on_gap(self):
        # 갭 상승 시가 110 → 트리거 = 110 + 0.5×10 = 115 (시가 앵커,
        # 윌리엄스 원전). 고가 116 도달 → 115 체결.
        rows = [(100, 105, 95, 100), (110, 116, 108, 114), (114, 114, 114, 114)]
        r = VolatilityBreakoutStrategy().execute(_df(rows), _config())
        assert r.trades[0].entry == pytest.approx(115.0)

    def test_no_trade_below_trigger(self):
        rows = [(100, 105, 95, 100), (100, 104, 98, 103), (103, 104, 102, 103)]
        r = VolatilityBreakoutStrategy().execute(_df(rows), _config())
        assert r.trades == []
        assert r.summary.final_value == pytest.approx(1e8)

    def test_trend_filter_blocks(self):
        cfg = _config(trend_filter_days=2)
        # 종가가 2일선 아래 → 진입 금지.
        rows = [
            (100, 105, 95, 100), (100, 100, 90, 91),
            (91, 99, 90, 95), (95, 99, 94, 96),
        ]
        r = VolatilityBreakoutStrategy().execute(_df(rows), cfg)
        assert r.trades == []


class TestSizing:
    def test_leverage_scales_pnl(self):
        rows = [(100, 105, 95, 100), (100, 108, 99, 107), (107, 107, 107, 107)]
        r1 = VolatilityBreakoutStrategy().execute(_df(rows), _config(leverage=1.0))
        r3 = VolatilityBreakoutStrategy().execute(_df(rows), _config(leverage=3.0))
        assert r3.trades[0].pnl == pytest.approx(r1.trades[0].pnl * 3)

    def test_ruin_stops_trading(self):
        # 3배 레버리지로 -35% 일봉 → 자본 −105% → 파산, 이후 트레이드 없음.
        rows = [
            (100, 105, 95, 100),
            (103, 109, 60, 65),   # 트리거 108(=103+0.5×10) 체결 후 폭락
            (65, 75, 60, 70),
            (70, 80, 65, 75),
        ]
        r = VolatilityBreakoutStrategy().execute(
            _df(rows), _config(leverage=3.0)
        )
        assert r.ruined
        assert len(r.trades) == 1
        assert r.summary.final_value == 0.0

    def test_kelly_estimated_from_trades(self):
        rows = [(100, 105, 95, 100), (100, 108, 99, 107), (107, 107, 107, 107)]
        r = VolatilityBreakoutStrategy().execute(_df(rows), _config())
        assert r.win_rate == 1.0
        assert r.kelly_fraction == 0.0  # 손실 표본 없음 → 추정 불가(0)


class TestCosts:
    def test_fees_and_slippage_reduce_pnl(self):
        rows = [(100, 105, 95, 100), (100, 108, 99, 107), (107, 107, 107, 107)]
        base = VolatilityBreakoutStrategy().execute(_df(rows), _config())
        costed = VolatilityBreakoutStrategy().execute(
            _df(rows),
            _config(
                fee_schedule=TossFeeSchedule(),
                slippage_bp=5.0,
                capital_gains_tax_pct=0.0,
                tax_deduction=0.0,
            ),
        )
        assert costed.trades[0].pnl < base.trades[0].pnl
        assert costed.liquidation.total_fees > 0

    def test_yearly_returns_and_tax(self):
        idx_rows = [(100, 105, 95, 100), (100, 108, 99, 107)] + [
            (107, 107, 107, 107)
        ] * 3
        cfg = _config(capital_gains_tax_pct=0.22)
        r = VolatilityBreakoutStrategy().execute(_df(idx_rows), cfg)
        assert 2024 in r.yearly_returns
        assert r.liquidation.total_tax > 0
        assert (
            r.liquidation.final_value_after_tax
            < r.liquidation.final_value_pre_tax
        )


class TestFinancing:
    def test_no_financing_at_1x(self):
        rows = [(100, 105, 95, 100), (100, 108, 99, 107), (107, 107, 107, 107)]
        r = VolatilityBreakoutStrategy().execute(
            _df(rows), _config(leverage=1.0, financing_annual_rate=0.10)
        )
        assert r.total_financing == 0.0

    def test_financing_charged_on_borrowed_portion(self):
        rows = [(100, 105, 95, 100), (100, 108, 99, 107), (107, 107, 107, 107)]
        base = VolatilityBreakoutStrategy().execute(
            _df(rows), _config(leverage=2.0, financing_annual_rate=0.0)
        )
        fin = VolatilityBreakoutStrategy().execute(
            _df(rows), _config(leverage=2.0, financing_annual_rate=0.10)
        )
        # 빌린 부분 = 평가액×(2−1) → 1일치 비용 = 1e8 × 0.10/365.
        expected = 1e8 * 0.10 / 365
        assert fin.total_financing == pytest.approx(expected)
        assert fin.trades[0].pnl == pytest.approx(base.trades[0].pnl - expected)
