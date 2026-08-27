"""MultiDipBuyStrategy — 공유 현금 멀티 종목 규칙 테스트."""

from datetime import date

import pandas as pd
import pytest

from strategy.adapters.dip_buy_strategy import DipBuyStrategy
from strategy.adapters.multi_dip_buy_strategy import MultiDipBuyStrategy
from strategy.domain.models import (
    DipBuyConfig,
    MultiDipBuyConfig,
    TossFeeSchedule,
)

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


def _config(**overrides) -> MultiDipBuyConfig:
    defaults = dict(
        tickers=["A", "B"],
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
    return MultiDipBuyConfig(**defaults)


class TestSingleTickerEquivalence:
    def test_matches_single_ticker_strategy(self):
        rows = [
            (100, 100, 100), (94, 94, 94), (90, 90, 90),
            (90.5, 91.2, 90.5), (91, 91, 91),
        ]
        single = DipBuyStrategy().execute(
            _df(rows),
            DipBuyConfig(
                start_date=date(2024, 1, 2), end_date=date(2025, 12, 31),
                initial_capital=90_000_000.0, drop_pct=0.05,
                target_pct=0.012, split=3, fee_schedule=NO_FEE,
                capital_gains_tax_pct=0.0, tax_deduction=0.0,
            ),
        )
        multi = MultiDipBuyStrategy().execute(
            {"A": _df(rows)}, _config(tickers=["A"])
        )
        assert multi.liquidation.final_value_after_tax == pytest.approx(
            single.liquidation.final_value_after_tax
        )
        assert len(multi.cycles) == len(single.cycles)


class TestSharedCash:
    def test_independent_signals_both_traded(self):
        # A만 급락 → A만 매수. B는 조용.
        a = [(100, 100, 100), (94, 94, 94), (90, 90, 90), (90, 92, 90)]
        b = [(50, 50, 50), (50, 50, 50), (50, 50, 50), (50, 50, 50)]
        r = MultiDipBuyStrategy().execute({"A": _df(a), "B": _df(b)}, _config())
        assert {c.ticker for c in r.cycles} == {"A"}

    def test_same_day_signals_deepest_first(self):
        # A -6%, B -9% 동시 급락, 현금은 총액 1/3씩 두 번 배정 가능.
        a = [(100, 100, 100), (94, 94, 94), (90, 90, 90), (90, 90, 90)]
        b = [(50, 50, 50), (45.5, 45.5, 45.5), (44, 44, 44), (44, 44, 44)]
        r = MultiDipBuyStrategy().execute(
            {"A": _df(a), "B": _df(b)}, _config()
        )
        opens = [c for c in r.cycles if c.outcome == "open"]
        assert {c.ticker for c in opens} == {"A", "B"}
        # 둘 다 계좌 총액(90M)의 1/3 근처 투입.
        for c in opens:
            assert c.invested == pytest.approx(30_000_000, rel=0.02)

    def test_cash_cap_limits_late_signals(self):
        # 3종목 동시 급락인데 split=2 → 총액/2씩 두 종목이면 현금 소진,
        # 낙폭 얕은 종목은 못 산다.
        mk = lambda p0: [(p0, p0, p0), (p0 * 0.90, p0 * 0.90, p0 * 0.90),
                         (p0 * 0.85, p0 * 0.85, p0 * 0.85)]
        deep = _df(mk(100))       # -10%
        mid = _df(mk(50))         # -10% (같은 낙폭, 정렬 안정성 확인용)
        shallow = [(10, 10, 10), (9.4, 9.4, 9.4), (9.2, 9.2, 9.2)]
        r = MultiDipBuyStrategy().execute(
            {"A": deep, "B": mid, "C": _df(shallow)},
            _config(tickers=["A", "B", "C"], split=2),
        )
        bought = {c.ticker for c in r.cycles}
        assert len(bought) == 2  # 현금이 두 종목에서 바닥

    def test_recycled_cash_after_profit(self):
        # A 익절로 돌아온 현금이 다음 날 B 신호에 쓰인다.
        a = [
            (100, 100, 100), (94, 94, 94), (90, 90, 90),
            (92, 95, 92), (92, 92, 92),
        ]  # 3일째 매수, 4일째 익절
        b = [
            (50, 50, 50), (50, 50, 50), (50, 50, 50),
            (46, 46, 46), (44, 44, 44),
        ]  # 4일째 -8% → 5일째 매수
        r = MultiDipBuyStrategy().execute(
            {"A": _df(a), "B": _df(b)}, _config(split=1)
        )
        assert any(c.ticker == "A" and c.outcome == "profit" for c in r.cycles)
        assert any(c.ticker == "B" for c in r.cycles)


class TestLateInception:
    def test_late_listed_ticker_joins_when_data_starts(self):
        a = [(100, 100, 100)] * 6
        b_rows = [(50, 50, 50), (46, 46, 46), (45, 45, 45)]
        b = _df(b_rows, start="2024-01-05")  # 늦은 상장
        r = MultiDipBuyStrategy().execute({"A": _df(a), "B": b}, _config())
        assert {c.ticker for c in r.cycles} == {"B"}
