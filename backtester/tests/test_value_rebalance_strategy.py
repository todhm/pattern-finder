"""ValueRebalanceStrategy — 라오어 VR 규칙 단위 테스트.

가격 경로를 손으로 만들어 규칙을 하나씩 검증한다:
V 갱신(Pool/G) / 밴드 상단 매도 / 하단 매수(Pool 한도) /
실력공식 √보정 / 수수료·양도세.
"""

from datetime import date

import pandas as pd
import pytest

from strategy.adapters.value_rebalance_strategy import ValueRebalanceStrategy
from strategy.domain.models import TossFeeSchedule, ValueRebalanceConfig

NO_FEE = TossFeeSchedule(
    buy_commission_pct=0.0, sell_commission_pct=0.0, sec_fee_pct=0.0
)


def _series(values: list[float], start: str = "2024-01-02") -> pd.Series:
    idx = pd.bdate_range(start, periods=len(values))
    return pd.Series(values, index=idx, dtype=float)


def _config(**overrides) -> ValueRebalanceConfig:
    defaults = dict(
        start_date=date(2024, 1, 2),
        end_date=date(2025, 12, 31),
        initial_capital=100_000_000.0,
        stock_ratio=0.75,
        gradient=10.0,
        cycle_days=10,
        band_pct=0.15,
        check_daily=True,
        advanced_formula=False,
        fee_schedule=NO_FEE,
        capital_gains_tax_pct=0.0,
        tax_deduction=0.0,
    )
    defaults.update(overrides)
    return ValueRebalanceConfig(**defaults)


class TestSetup:
    def test_initial_split_and_value_path(self):
        result = ValueRebalanceStrategy().execute(
            _series([100.0] * 5), _config()
        )
        pt0 = result.curve[0]
        assert pt0.stock_value == pytest.approx(75_000_000)
        assert pt0.pool == pytest.approx(25_000_000)
        assert pt0.value_path == pytest.approx(75_000_000)

    def test_needs_two_bars(self):
        with pytest.raises(ValueError):
            ValueRebalanceStrategy().execute(_series([100.0]), _config())


class TestValuePath:
    def test_v_grows_by_pool_over_g_each_cycle(self):
        # 가격 고정 → 매매 없음 → V만 Pool/G씩 계단 상승.
        result = ValueRebalanceStrategy().execute(
            _series([100.0] * 25), _config()
        )
        # 10번째 봉(1사이클) 후 V = 75M + 25M/10 = 77.5M.
        assert result.curve[10].value_path == pytest.approx(77_500_000)
        # 2사이클 후: Pool 불변이므로 80M.
        assert result.curve[20].value_path == pytest.approx(80_000_000)

    def test_advanced_formula_slows_v_in_drawdown(self):
        # 밴드 안 완만한 하락(-12%) — 매수 없이 Pool 유지, E < V이므로
        # 실력공식의 √(E/V) < 1이 V 상승을 늦춘다.
        values = [100.0] + [88.0] * 24
        base = ValueRebalanceStrategy().execute(_series(values), _config())
        adv = ValueRebalanceStrategy().execute(
            _series(values), _config(advanced_formula=True)
        )
        assert (
            adv.curve[-1].value_path < base.curve[-1].value_path
        )


class TestBandTrades:
    def test_sell_excess_to_band_edge(self):
        # +20% 급등 → E = 90M > 상단 86.25M → 가장자리까지 3.75M 매도.
        values = [100.0, 120.0, 120.0]
        result = ValueRebalanceStrategy().execute(_series(values), _config())
        sells = [e for e in result.events if e.kind == "sell"]
        assert len(sells) == 1
        assert sells[0].traded == pytest.approx(-3_750_000)
        assert sells[0].stock_value_after == pytest.approx(86_250_000)

    def test_buy_deficit_to_band_edge(self):
        # -20% 급락 → E = 60M < 하단 63.75M → 가장자리까지 3.75M 매수.
        values = [100.0, 80.0, 80.0]
        result = ValueRebalanceStrategy().execute(_series(values), _config())
        buys = [e for e in result.events if e.kind == "buy"]
        assert len(buys) == 1
        assert buys[0].traded == pytest.approx(3_750_000)

    def test_graduated_buys_as_price_grids_down(self):
        # 연속 하락 → 매일 가장자리까지만 사서 분할 매수가 누적된다
        # (매수표 LOC 등가). 한 번에 사면 매수 1회로 끝났을 경로.
        values = [100.0, 80.0, 70.0, 60.0, 60.0]
        result = ValueRebalanceStrategy().execute(_series(values), _config())
        buys = [e for e in result.events if e.kind == "buy"]
        assert len(buys) >= 3
        # 뒤로 갈수록 낮은 가격에 체결 — 평균단가가 계단식으로 낮아짐.
        assert all(b.traded > 0 for b in buys)

    def test_center_mode_buys_full_deficit(self):
        # 공격적 모드: V까지 한 번에 — 초기 구현 호환.
        values = [100.0, 80.0, 80.0]
        result = ValueRebalanceStrategy().execute(
            _series(values), _config(rebalance_to_edge=False)
        )
        buys = [e for e in result.events if e.kind == "buy"]
        assert len(buys) == 1
        assert buys[0].traded == pytest.approx(15_000_000)
        assert buys[0].pool_after == pytest.approx(10_000_000)

    def test_buy_capped_by_pool(self):
        # -60% 폭락 → 하단까지 부족분 33.75M이지만 Pool 25M까지만 매수.
        values = [100.0, 40.0, 40.0]
        result = ValueRebalanceStrategy().execute(_series(values), _config())
        buys = [e for e in result.events if e.kind == "buy"]
        assert len(buys) == 1
        assert buys[0].traded == pytest.approx(25_000_000)
        assert buys[0].pool_after == pytest.approx(0.0)

    def test_cycle_only_check_skips_intra_cycle_moves(self):
        # 급등이 사이클 중간에만 있고 10일째엔 원위치 → 매매 없음.
        values = [100.0] + [120.0] * 5 + [100.0] * 10
        result = ValueRebalanceStrategy().execute(
            _series(values), _config(check_daily=False)
        )
        assert result.sell_count == 0


class TestFeesAndTax:
    def test_sell_realizes_gain_and_final_tax(self):
        values = [100.0, 120.0] + [120.0] * 3
        cfg = _config(
            fee_schedule=TossFeeSchedule(),
            capital_gains_tax_pct=0.22,
            tax_deduction=0.0,
        )
        result = ValueRebalanceStrategy().execute(_series(values), cfg)
        liq = result.liquidation
        assert liq.realized_gain_total > 0
        assert liq.total_tax == pytest.approx(
            liq.realized_gain_total * 0.22
        )
        assert liq.final_value_after_tax < liq.final_value_pre_tax
        assert liq.total_fees > 0

    def test_flat_price_no_tax(self):
        cfg = _config(
            fee_schedule=TossFeeSchedule(), capital_gains_tax_pct=0.22
        )
        result = ValueRebalanceStrategy().execute(
            _series([100.0] * 10), cfg
        )
        assert result.liquidation.total_tax == 0.0

    def test_benchmark_after_tax_present(self):
        result = ValueRebalanceStrategy().execute(
            _series([100.0] * 5), _config()
        )
        assert set(result.benchmark_after_tax) == {
            b.name for b in result.benchmarks
        }


class TestPoolInterest:
    def test_pool_earns_daily_interest(self):
        # 가격 고정 → 매매 없음 → Pool 25M이 연 9% 일할 복리로 증가.
        r = ValueRebalanceStrategy().execute(
            _series([100.0] * 30), _config(pool_annual_rate=0.09)
        )
        expected_pool = 25_000_000 * (1 + 0.09 / 252) ** 29
        assert r.curve[-1].pool == pytest.approx(expected_pool)
        assert r.liquidation.total_interest == pytest.approx(
            expected_pool - 25_000_000
        )

    def test_interest_beats_no_interest(self):
        base = ValueRebalanceStrategy().execute(
            _series([100.0] * 30), _config()
        )
        p2p = ValueRebalanceStrategy().execute(
            _series([100.0] * 30), _config(pool_annual_rate=0.09)
        )
        assert (
            p2p.liquidation.final_value_after_tax
            > base.liquidation.final_value_after_tax
        )
