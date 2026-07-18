"""BandRebalanceStrategy — 밴드 리밸런싱 규칙 단위 테스트.

가격 경로를 손으로 만들어 규칙 하나하나를 검증한다:
줍줍(-15%) / 수익 실현(+15%) / 기준가 갱신 / 무발동 드리프트 /
벤치마크·합성 레버리지 수학.
"""

from datetime import date

import pandas as pd
import pytest

from strategy.adapters.band_rebalance_strategy import (
    BandRebalanceStrategy,
    build_synthetic_leveraged,
    splice_series,
)
from strategy.domain.models import BandRebalanceConfig


def _series(values: list[float]) -> pd.Series:
    idx = pd.bdate_range("2024-01-02", periods=len(values))
    return pd.Series(values, index=idx, dtype=float)


def _config(**overrides) -> BandRebalanceConfig:
    defaults = dict(
        aggressive_ticker="TQQQ",
        defensive_ticker="VOO",
        start_date=date(2024, 1, 2),
        end_date=date(2024, 12, 31),
        initial_capital=100_000.0,
        band_pct=0.15,
        aggressive_weight=0.5,
        dip_sell_defensive_pct=0.15,
    )
    defaults.update(overrides)
    return BandRebalanceConfig(**defaults)


class TestInitialAllocation:
    def test_day0_splits_capital_at_target_weight(self):
        agg = _series([100.0, 100.0])
        dfn = _series([50.0, 50.0])
        result = BandRebalanceStrategy().execute(agg, dfn, _config())

        first = result.curve[0]
        assert first.total == pytest.approx(100_000.0)
        assert first.aggressive_value == pytest.approx(50_000.0)
        assert first.defensive_value == pytest.approx(50_000.0)
        assert first.reference_price == pytest.approx(100.0)

    def test_no_trigger_means_no_events_and_pure_drift(self):
        # ±15% 미만 등락만 → 트리거 없음, 평가액은 시장 드리프트 그대로.
        agg = _series([100.0, 110.0, 95.0, 105.0])
        dfn = _series([50.0, 51.0, 49.0, 52.0])
        result = BandRebalanceStrategy().execute(agg, dfn, _config())

        assert result.events == []
        last = result.curve[-1]
        # 초기 500주(agg) / 1000주(def) 그대로 보유.
        assert last.aggressive_value == pytest.approx(500 * 105.0)
        assert last.defensive_value == pytest.approx(1000 * 52.0)
        assert last.reference_price == pytest.approx(100.0)


class TestDipBuy:
    def test_dip_sells_defensive_and_buys_aggressive(self):
        # day1: TQQQ 100 → 85 (-15%) → 줍줍 발동.
        agg = _series([100.0, 85.0])
        dfn = _series([50.0, 50.0])
        result = BandRebalanceStrategy().execute(agg, dfn, _config())

        assert len(result.events) == 1
        e = result.events[0]
        assert e.kind == "dip_buy"
        assert e.reference_price_before == pytest.approx(100.0)
        assert e.aggressive_price == pytest.approx(85.0)
        # 방어 평가액 50,000의 15% = 7,500 매도 → 공격 매수.
        assert e.traded_amount == pytest.approx(7_500.0)
        assert e.defensive_value_after == pytest.approx(42_500.0)
        # 공격: 500주 × 85 + 7,500 = 50,000.
        assert e.aggressive_value_after == pytest.approx(50_000.0)
        # 기준가는 현재가로 갱신.
        assert result.curve[-1].reference_price == pytest.approx(85.0)

    def test_consecutive_dips_each_need_fresh_15pct(self):
        # 85(-15%) 발동 후 84~73 구간은 미발동, 72.25(85×0.85)에서 재발동.
        agg = _series([100.0, 85.0, 80.0, 72.25])
        dfn = _series([50.0, 50.0, 50.0, 50.0])
        result = BandRebalanceStrategy().execute(agg, dfn, _config())

        kinds = [e.kind for e in result.events]
        assert kinds == ["dip_buy", "dip_buy"]
        assert result.events[1].reference_price_before == pytest.approx(85.0)


class TestProfitTake:
    def test_rally_rebalances_back_to_target_weight(self):
        # day1: TQQQ 100 → 115 (+15%) → 완전 50:50 리밸런싱.
        agg = _series([100.0, 115.0])
        dfn = _series([50.0, 50.0])
        result = BandRebalanceStrategy().execute(agg, dfn, _config())

        assert len(result.events) == 1
        e = result.events[0]
        assert e.kind == "profit_take"
        # 총액 = 500×115 + 50,000 = 107,500 → 각 53,750.
        assert e.total_value_after == pytest.approx(107_500.0)
        assert e.aggressive_value_after == pytest.approx(53_750.0)
        assert e.defensive_value_after == pytest.approx(53_750.0)
        assert e.aggressive_weight_after == pytest.approx(0.5)
        # 공격 → 방어로 이동: traded_amount는 음수.
        assert e.traded_amount == pytest.approx(-3_750.0)
        assert result.curve[-1].reference_price == pytest.approx(115.0)

    def test_dip_then_recovery_beats_buy_and_hold(self):
        # V자 반등: 줍줍으로 싸게 산 물량 덕에 50:50 방치보다 우위 —
        # 보고서의 핵심 주장("리밸런싱의 복리 마법") 재현.
        agg = _series([100.0, 84.0, 70.0, 84.0, 101.0, 120.0])
        dfn = _series([50.0] * 6)
        result = BandRebalanceStrategy().execute(agg, dfn, _config())

        hold_5050 = next(
            b for b in result.benchmarks if "방치" in b.name
        )
        assert result.summary.final_value > hold_5050.final_value
        kinds = [e.kind for e in result.events]
        assert kinds[:2] == ["dip_buy", "dip_buy"]
        assert "profit_take" in kinds


class TestBenchmarks:
    def test_benchmark_final_values_match_price_ratios(self):
        agg = _series([100.0, 110.0])
        dfn = _series([50.0, 55.0])
        result = BandRebalanceStrategy().execute(agg, dfn, _config())

        by_name = {b.name: b for b in result.benchmarks}
        assert by_name["VOO 100%"].final_value == pytest.approx(110_000.0)
        assert by_name["TQQQ 100%"].final_value == pytest.approx(110_000.0)
        assert by_name["50:50 방치 (리밸런싱 없음)"].final_value == (
            pytest.approx(110_000.0)
        )

    def test_max_drawdown_is_peak_to_trough(self):
        # 100 → 120 → 60: 공격 100% 벤치마크 MDD = -50%.
        agg = _series([100.0, 120.0, 60.0])
        dfn = _series([50.0, 50.0, 50.0])
        result = BandRebalanceStrategy().execute(agg, dfn, _config())

        agg_bench = next(b for b in result.benchmarks if b.name == "TQQQ 100%")
        assert agg_bench.max_drawdown_pct == pytest.approx(-0.5)


class TestSyntheticLeverage:
    def test_triples_daily_returns_before_expense(self):
        base = _series([100.0, 110.0])  # +10% 하루
        lev = build_synthetic_leveraged(base, leverage=3.0, annual_expense=0.0)
        assert lev.iloc[0] == pytest.approx(100.0)
        assert lev.iloc[-1] == pytest.approx(130.0)

    def test_expense_drag_reduces_return(self):
        base = _series([100.0] * 253)  # 1년간 횡보
        lev = build_synthetic_leveraged(
            base, leverage=3.0, annual_expense=0.0252
        )
        # 보수율 2.52%/252 = 일 0.01% 차감 × 252일 ≈ -2.49% (복리).
        assert lev.iloc[-1] == pytest.approx(100.0 * (1 - 0.0001) ** 252)

    def test_volatility_decay_is_reproduced(self):
        # +10% → -9.09%로 기초지수 원위치. 3배는 +30% → -27.27%로
        # 원금 미회복 (음의 복리).
        base = _series([100.0, 110.0, 100.0])
        lev = build_synthetic_leveraged(base, leverage=3.0, annual_expense=0.0)
        assert base.iloc[-1] == pytest.approx(100.0)
        assert lev.iloc[-1] < 100.0


class TestFinancingCost:
    def test_financing_rate_drags_returns(self):
        # 횡보 1년 + 단기금리 5%: 3배 ETF는 (3-1)×5% = 연 10% 드래그.
        base = _series([100.0] * 253)
        fin = pd.Series(0.05, index=base.index)
        lev = build_synthetic_leveraged(
            base, leverage=3.0, annual_expense=0.0, financing_rate=fin
        )
        expected = 100.0 * (1 - 2 * 0.05 / 252) ** 252
        assert lev.iloc[-1] == pytest.approx(expected)

    def test_financing_series_is_ffilled_to_calendar(self):
        base = _series([100.0, 100.0, 100.0])
        # 금리 시계열이 첫날 하나뿐 — 이후 날짜는 ffill.
        fin = pd.Series([0.05], index=base.index[:1])
        lev = build_synthetic_leveraged(
            base, leverage=2.0, annual_expense=0.0, financing_rate=fin
        )
        assert lev.iloc[-1] == pytest.approx(100.0 * (1 - 0.05 / 252) ** 2)

    def test_no_financing_keeps_old_behavior(self):
        base = _series([100.0, 110.0])
        lev = build_synthetic_leveraged(base, leverage=3.0, annual_expense=0.0)
        assert lev.iloc[-1] == pytest.approx(130.0)


class TestSplice:
    def test_real_returns_used_after_anchor(self):
        # 합성: 매일 +10%. 실데이터는 3일째부터 시작해 +20%/일.
        early = _series([100.0, 110.0, 121.0, 133.1])
        late = pd.Series([50.0, 60.0], index=early.index[2:])
        spliced = splice_series(early, late)
        # anchor(3일째)까지는 합성 수익률, 그 뒤는 실데이터 +20%.
        assert spliced.iloc[0] == pytest.approx(100.0)
        assert spliced.iloc[2] == pytest.approx(121.0)
        assert spliced.iloc[3] == pytest.approx(121.0 * 1.2)
        assert len(spliced) == 4

    def test_no_real_data_returns_synthetic(self):
        early = _series([100.0, 110.0])
        assert splice_series(early, None).equals(early)
        assert splice_series(early, pd.Series(dtype=float)).equals(early)

    def test_real_covers_everything_returns_real(self):
        early = _series([100.0, 110.0])
        late = _series([50.0, 55.0])  # 같은 달력 — pre 구간 없음
        assert splice_series(early, late).equals(late)


class TestRegimeFilter:
    """레짐 필터: SMA 이탈 시 방어 전환 / 복귀 시 재진입.

    테스트 용이성을 위해 SMA 3일 사용 (로직은 기간과 무관).
    """

    def test_derisk_defensive_moves_everything_to_defensive(self):
        agg = _series([100.0] * 5)
        dfn = _series([50.0] * 5)
        # SMA3 확정 후 day3에 70 < SMA(90) → risk_off.
        regime = _series([100.0, 100.0, 100.0, 70.0, 70.0])
        result = BandRebalanceStrategy().execute(
            agg, dfn, _config(regime_sma_days=3), regime_close=regime
        )

        offs = [e for e in result.events if e.kind == "risk_off"]
        assert len(offs) == 1
        e = offs[0]
        assert e.aggressive_value_after == pytest.approx(0.0)
        assert e.defensive_value_after == pytest.approx(100_000.0)
        assert e.traded_amount == pytest.approx(-50_000.0)
        assert result.curve[-1].risk_on is False

    def test_no_dip_buys_while_risk_off(self):
        # risk_off 이후 공격 자산이 -50% 폭락해도 줍줍 금지 —
        # 원 전략의 "떨어지는 칼날 받기"가 차단되는지 확인.
        agg = _series([100.0, 100.0, 100.0, 70.0, 50.0, 30.0])
        dfn = _series([50.0] * 6)
        regime = _series([100.0, 100.0, 100.0, 70.0, 50.0, 30.0])
        result = BandRebalanceStrategy().execute(
            agg, dfn, _config(regime_sma_days=3), regime_close=regime
        )

        kinds = [e.kind for e in result.events]
        assert "dip_buy" not in kinds
        # 전환 시점(70)에 공격 50k→35k 손실은 확정되지만, 이후
        # -57% 추가 폭락은 방어 자산 대피로 전부 회피 → 85k 유지.
        assert result.curve[-1].total == pytest.approx(85_000.0)

    def test_derisk_cash_parks_everything_in_cash(self):
        agg = _series([100.0] * 5)
        dfn = _series([50.0] * 5)
        regime = _series([100.0, 100.0, 100.0, 70.0, 70.0])
        result = BandRebalanceStrategy().execute(
            agg,
            dfn,
            _config(regime_sma_days=3, risk_off_mode="derisk_cash"),
            regime_close=regime,
        )

        last = result.curve[-1]
        assert last.cash == pytest.approx(100_000.0)
        assert last.aggressive_value == pytest.approx(0.0)
        assert last.defensive_value == pytest.approx(0.0)

    def test_risk_on_reenters_at_target_weight_and_resets_ref(self):
        # day3 risk_off (70 < SMA 90) → day5 SMA3(=80) 위 120으로 복귀.
        agg = _series([100.0, 100.0, 100.0, 70.0, 70.0, 120.0, 120.0])
        dfn = _series([50.0] * 7)
        regime = _series([100.0, 100.0, 100.0, 70.0, 70.0, 120.0, 120.0])
        result = BandRebalanceStrategy().execute(
            agg, dfn, _config(regime_sma_days=3), regime_close=regime
        )

        ons = [e for e in result.events if e.kind == "risk_on"]
        assert len(ons) == 1
        assert ons[0].aggressive_weight_after == pytest.approx(0.5)
        # 기준가는 재진입 가격으로 리셋.
        assert result.curve[-1].reference_price == pytest.approx(120.0)
        assert result.curve[-1].risk_on is True

    def test_confirm_days_delays_reentry(self):
        # 복귀 조건 충족이 3일 연속 이어져야 재진입 (confirm=2 →
        # 3번째 충족일에 진입).
        agg = _series([100.0] * 9)
        dfn = _series([50.0] * 9)
        regime = _series(
            [100.0, 100.0, 100.0, 70.0, 70.0, 120.0, 125.0, 130.0, 130.0]
        )
        result = BandRebalanceStrategy().execute(
            agg,
            dfn,
            _config(regime_sma_days=3, regime_confirm_days=2),
            regime_close=regime,
        )

        ons = [e for e in result.events if e.kind == "risk_on"]
        assert len(ons) == 1
        # 충족일: day5(120), day6(125), day7(130) → day7 재진입.
        assert ons[0].date == regime.index[7].date()

    def test_pause_dip_keeps_holdings_but_allows_profit_take(self):
        # pause_dip: risk_off 전환 시 매도 없음(보유 유지), 줍줍만
        # 중단. +15% 도달 시 수익 실현은 계속 허용 (위험 축소 방향).
        agg = _series([100.0, 100.0, 100.0, 100.0, 116.0])
        dfn = _series([50.0] * 5)
        regime = _series([100.0, 100.0, 100.0, 70.0, 70.0])
        result = BandRebalanceStrategy().execute(
            agg,
            dfn,
            _config(regime_sma_days=3, risk_off_mode="pause_dip"),
            regime_close=regime,
        )

        offs = [e for e in result.events if e.kind == "risk_off"]
        assert len(offs) == 1
        assert offs[0].traded_amount == pytest.approx(0.0)
        assert offs[0].aggressive_value_after == pytest.approx(50_000.0)
        takes = [e for e in result.events if e.kind == "profit_take"]
        assert len(takes) == 1
        assert takes[0].aggressive_weight_after == pytest.approx(0.5)


class TestValidation:
    def test_rejects_insufficient_overlap(self):
        agg = _series([100.0])
        dfn = _series([50.0])
        with pytest.raises(ValueError, match="at least 2"):
            BandRebalanceStrategy().execute(agg, dfn, _config())

    def test_misaligned_series_are_inner_joined(self):
        agg = pd.Series(
            [100.0, 90.0, 84.0],
            index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
        )
        dfn = pd.Series(
            [50.0, 50.0],
            index=pd.to_datetime(["2024-01-02", "2024-01-04"]),
        )
        result = BandRebalanceStrategy().execute(agg, dfn, _config())
        # 겹치는 2일만 사용 — 01-04에 84(-16%)로 줍줍 1회.
        assert len(result.curve) == 2
        assert [e.kind for e in result.events] == ["dip_buy"]
