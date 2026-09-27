"""LeaderRotationStrategy — 리더 로테이션 규칙 단위 테스트."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from strategy.adapters.leader_rotation_strategy import LeaderRotationStrategy
from strategy.domain.models import LeaderRotationConfig, TossFeeSchedule

NO_FEE = TossFeeSchedule(
    buy_commission_pct=0.0, sell_commission_pct=0.0, sec_fee_pct=0.0
)


def _frame(cols: dict[str, list[float]], start="2020-01-01") -> pd.DataFrame:
    idx = pd.bdate_range(start, periods=len(next(iter(cols.values()))))
    return pd.DataFrame(cols, index=idx, dtype=float)


def _config(**overrides) -> LeaderRotationConfig:
    defaults = dict(
        start_date=date(2020, 2, 1),
        end_date=date(2021, 12, 31),
        initial_capital=100_000_000.0,
        lookback_days=10,
        top_n=1,
        rebalance_days=5,
        near_high_pct=1.0,  # 고점 필터 사실상 해제
        trend_ma_days=0,
        min_dollar_volume=0.0,
        market_filter=False,
        slippage_bp=0.0,
        fee_schedule=NO_FEE,
        capital_gains_tax_pct=0.0,
        tax_deduction=0.0,
    )
    defaults.update(overrides)
    return LeaderRotationConfig(**defaults)


def _dvol_like(px: pd.DataFrame, value: float = 1e9) -> pd.DataFrame:
    return pd.DataFrame(value, index=px.index, columns=px.columns)


class TestRotation:
    def test_picks_strongest_momentum(self):
        n = 60
        up = list(np.linspace(100, 300, n))     # 강한 상승
        flat = [100.0] * n
        px = _frame({"UP": up, "FLAT": flat})
        r = LeaderRotationStrategy().execute(px, _dvol_like(px), _config())
        risk_on = [rb for rb in r.rebalances if rb.picks]
        assert risk_on, "리밸런스에서 종목이 선택돼야 함"
        assert all(rb.picks == ["UP"] for rb in risk_on)
        assert r.summary.total_return_pct > 0.5  # UP을 계속 보유

    def test_rotates_when_leadership_changes(self):
        # 전반 A 상승/B 횡보 → 후반 A 횡보/B 급등: B로 갈아타야 함.
        n = 80
        a = list(np.linspace(100, 200, 40)) + [200.0] * 40
        b = [100.0] * 40 + list(np.linspace(100, 260, 40))
        px = _frame({"A": a, "B": b})
        r = LeaderRotationStrategy().execute(px, _dvol_like(px), _config())
        picked = [rb.picks[0] for rb in r.rebalances if rb.picks]
        assert "A" in picked and "B" in picked
        assert picked[-1] == "B"

    def test_liquidity_filter_excludes_thin_names(self):
        n = 60
        px = _frame({"THIN": list(np.linspace(100, 400, n)),
                     "LIQ": list(np.linspace(100, 150, n))})
        dv = _dvol_like(px)
        dv["THIN"] = 1e5  # 거래대금 미달
        cfg = _config(min_dollar_volume=1e6)
        r = LeaderRotationStrategy().execute(px, dv, cfg)
        for rb in r.rebalances:
            assert "THIN" not in rb.picks

    def test_market_filter_goes_cash(self):
        n = 60
        px = _frame({"UP": list(np.linspace(100, 300, n))})
        market = pd.Series(
            np.linspace(300, 100, 300),
            index=pd.bdate_range("2019-01-01", periods=300),
        )  # 시장 하락 → 200MA 아래
        cfg = _config(market_filter=True)
        r = LeaderRotationStrategy().execute(
            px, _dvol_like(px), cfg, market=market
        )
        assert all(rb.regime == "risk_off" for rb in r.rebalances)
        assert r.summary.total_return_pct == pytest.approx(0.0)

    def test_delisted_position_force_sold(self):
        # UP이 중간에 데이터 종료 → 마지막가로 강제 청산, 이후 FLAT 보유.
        n = 60
        up = list(np.linspace(100, 200, 30)) + [np.nan] * 30
        flat = [100.0] * n
        px = _frame({"UP": up, "FLAT": flat})
        r = LeaderRotationStrategy().execute(px, _dvol_like(px), _config())
        assert r.equity_curve[-1].equity > 0
        # 상폐 후 리밸런스에서 UP이 다시 선택되면 안 됨.
        for rb in r.rebalances[-3:]:
            assert "UP" not in rb.picks

    def test_fees_and_tax_reduce_result(self):
        n = 60
        px = _frame({"UP": list(np.linspace(100, 300, n))})
        cfg = _config(
            fee_schedule=TossFeeSchedule(), slippage_bp=20.0,
            capital_gains_tax_pct=0.22, tax_deduction=0.0,
        )
        free = LeaderRotationStrategy().execute(px, _dvol_like(px), _config())
        cost = LeaderRotationStrategy().execute(px, _dvol_like(px), cfg)
        assert cost.liquidation.total_fees > 0
        assert cost.liquidation.total_tax > 0
        assert (
            cost.liquidation.final_value_after_tax
            < free.liquidation.final_value_after_tax
        )


class TestFastStop:
    def test_stop_ma_sells_before_next_rebalance(self):
        # 급등 후 급락: 20일선 이탈 즉시 매도 → 월말 리밸런스보다 빠름.
        n = 80
        up = list(np.linspace(100, 200, 40)) + list(np.linspace(200, 80, 40))
        px = _frame({"UP": up})
        cfg = _config(rebalance_days=40, stop_ma_days=20)
        stopped = LeaderRotationStrategy().execute(px, _dvol_like(px), cfg)
        naive = LeaderRotationStrategy().execute(
            px, _dvol_like(px), _config(rebalance_days=40)
        )
        assert (
            stopped.summary.total_return_pct > naive.summary.total_return_pct
        )

    def test_vol_adj_prefers_steady_over_parabolic(self):
        # 같은 총수익이면 변동성 낮은 쪽을 선택.
        n = 60
        steady = list(np.linspace(100, 200, n))
        wild = []
        p = 100.0
        for i in range(n):  # 급등/급락 반복하며 200 도달
            p *= 1.25 if i % 2 == 0 else 0.83
            wild.append(p * (200 / 110) ** (i / n))
        px = _frame({"STEADY": steady, "WILD": wild})
        cfg = _config(rank_mode="vol_adj")
        r = LeaderRotationStrategy().execute(px, _dvol_like(px), cfg)
        picked = [rb.picks[0] for rb in r.rebalances if rb.picks]
        assert picked and all(s == "STEADY" for s in picked)


class TestScanLeaders:
    def test_scan_matches_backtest_rebalance_picks(self):
        # 시그널 스캔과 백테스트 리밸런스 랭킹이 같은 로직인지 드리프트 가드.
        from strategy.adapters.leader_rotation_strategy import scan_leaders

        rng = np.random.default_rng(7)
        n, syms = 120, [f"S{i}" for i in range(12)]
        data = {
            s: 100 * np.cumprod(1 + rng.normal(0.002 * (i % 5), 0.02, n))
            for i, s in enumerate(syms)
        }
        px = _frame({s: list(v) for s, v in data.items()})
        dv = _dvol_like(px)
        cfg = _config(top_n=3, rebalance_days=5)
        r = LeaderRotationStrategy().execute(px, dv, cfg)
        last_rb = [rb for rb in r.rebalances if rb.picks][-1]
        snap = scan_leaders(px, dv, cfg, asof=pd.Timestamp(last_rb.date))
        assert [row["symbol"] for row in snap["rows"]] == last_rb.picks

    def test_scan_risk_off(self):
        from strategy.adapters.leader_rotation_strategy import scan_leaders

        px = _frame({"UP": list(np.linspace(100, 300, 60))})
        market = pd.Series(
            np.linspace(300, 100, 300),
            index=pd.bdate_range("2019-01-01", periods=300),
        )
        snap = scan_leaders(
            px, _dvol_like(px), _config(market_filter=True), market=market
        )
        assert snap["regime"] == "risk_off"
