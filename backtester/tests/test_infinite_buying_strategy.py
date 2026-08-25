"""InfiniteBuyingStrategy — 무한매수법 규칙 단위 테스트.

일봉 OHLC를 손으로 만들어 검증한다: 첫날 1T / 큰수·평단 LOC /
지정가 익절(갭·터치) / v2.2 쿼터매도 / 소진 홀드·손절 / 15m 체결.
"""

from datetime import date

import pandas as pd
import pytest

from strategy.adapters.infinite_buying_strategy import InfiniteBuyingStrategy
from strategy.domain.models import InfiniteBuyingConfig, TossFeeSchedule

NO_FEE = TossFeeSchedule(
    buy_commission_pct=0.0, sell_commission_pct=0.0, sec_fee_pct=0.0
)


def _df(rows: list[tuple], start: str = "2024-01-02") -> pd.DataFrame:
    """rows = [(open, high, close), ...] — Low는 미사용이라 0."""
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


def _config(**overrides) -> InfiniteBuyingConfig:
    defaults = dict(
        start_date=date(2024, 1, 2),
        end_date=date(2025, 12, 31),
        initial_capital=100_000_000.0,
        divisions=40,
        version="v2.1",
        target_profit_pct=0.10,
        depletion_mode="hold",
        fee_schedule=NO_FEE,
        capital_gains_tax_pct=0.0,
        tax_deduction=0.0,
    )
    defaults.update(overrides)
    return InfiniteBuyingConfig(**defaults)


class TestBuying:
    def test_first_day_buys_one_tranche(self):
        r = InfiniteBuyingStrategy().execute(_df([(100, 100, 100)] * 3), _config())
        e0 = r.events[0]
        assert e0.kind == "start_buy"
        assert e0.notional == pytest.approx(2_500_000)  # 1억/40

    def test_flat_price_buys_full_tranche_daily(self):
        # 종가 == 평단 → 큰수 T/2 + 평단 LOC T/2 = 매일 1T.
        r = InfiniteBuyingStrategy().execute(_df([(100, 100, 100)] * 5), _config())
        day2 = [e for e in r.events if e.ts.date() == date(2024, 1, 3)]
        assert [e.kind for e in day2] == ["big_buy", "avg_buy"]
        assert sum(e.notional for e in day2) == pytest.approx(2_500_000)

    def test_avg_loc_skipped_when_close_above_avg(self):
        # 종가가 평단 위 → 큰수만 체결.
        r = InfiniteBuyingStrategy().execute(
            _df([(100, 100, 100), (104, 104, 104), (104, 104, 104)]), _config()
        )
        day2 = [e for e in r.events if e.ts.date() == date(2024, 1, 3)]
        assert [e.kind for e in day2] == ["big_buy"]


class TestSelling:
    def test_limit_sell_at_target_touch(self):
        # 평단 100, 고가 115 ≥ 110 → 110에 전량 익절 (v2.1).
        r = InfiniteBuyingStrategy().execute(
            _df([(100, 100, 100), (105, 115, 105), (105, 105, 105)]), _config()
        )
        sells = [e for e in r.events if e.kind == "limit_sell"]
        assert len(sells) == 1
        assert sells[0].price == pytest.approx(110.0)
        assert r.cycles[0].outcome == "profit"
        # 투입 1T(2.5M)의 +10% = 250k → 계좌(1억) 기준 +0.25%.
        assert r.cycles[0].pnl == pytest.approx(250_000)
        assert r.cycles[0].pnl_pct == pytest.approx(0.0025)

    def test_gap_open_fills_at_open(self):
        r = InfiniteBuyingStrategy().execute(
            _df([(100, 100, 100), (120, 125, 118), (118, 118, 118)]), _config()
        )
        sells = [e for e in r.events if e.kind == "limit_sell"]
        assert sells[0].price == pytest.approx(120.0)

    def test_sell_day_cancels_loc_buys(self):
        r = InfiniteBuyingStrategy().execute(
            _df([(100, 100, 100), (105, 115, 90), (90, 90, 90)]), _config()
        )
        sell_day = [e for e in r.events if e.ts.date() == date(2024, 1, 3)]
        assert [e.kind for e in sell_day] == ["limit_sell"]

    def test_new_cycle_starts_next_day_with_compounded_capital(self):
        r = InfiniteBuyingStrategy().execute(
            _df([(100, 100, 100), (105, 115, 105), (105, 105, 105)]), _config()
        )
        starts = [e for e in r.events if e.kind == "start_buy"]
        assert len(starts) == 2
        # 사이클 1 수익(2.5M×10%)이 합산된 원금의 1/40.
        assert starts[1].notional == pytest.approx(100_250_000 / 40)

    def test_v22_quarter_sell_then_hold(self):
        # 종가 106 ≥ 평단×1.05, 고가 108 < 110 → 25%만 LOC 매도.
        cfg = _config(version="v2.2")
        r = InfiniteBuyingStrategy().execute(
            _df([(100, 100, 100), (106, 108, 106), (106, 106, 106)]), cfg
        )
        quarters = [e for e in r.events if e.kind == "quarter_sell"]
        assert len(quarters) == 1
        assert quarters[0].price == pytest.approx(106.0)
        assert r.cycles[0].outcome == "open"  # 잔량 75% 유지

    def test_v22_full_exit_when_both_orders_fill(self):
        cfg = _config(version="v2.2")
        r = InfiniteBuyingStrategy().execute(
            _df([(100, 100, 100), (105, 112, 111), (111, 111, 111)]), cfg
        )
        kinds = [e.kind for e in r.events if e.ts.date() == date(2024, 1, 3)]
        assert kinds == ["limit_sell", "quarter_sell"]
        assert r.cycles[0].outcome == "profit"


class TestDepletion:
    def test_hold_mode_stops_buying(self):
        # 41일 하락 횡보 — 40분할 소진 후 매수 이벤트 없음.
        rows = [(100, 100, 100)] + [(90, 90, 90)] * 45
        r = InfiniteBuyingStrategy().execute(_df(rows), _config())
        buys = [e for e in r.events if "buy" in e.kind]
        total_spent = sum(e.notional for e in buys)
        assert total_spent == pytest.approx(100_000_000)
        assert r.cycles[-1].depleted

    def test_stop_loss_mode_restarts(self):
        rows = [(100, 100, 100)] + [(90, 90, 90)] * 50
        r = InfiniteBuyingStrategy().execute(
            _df(rows), _config(depletion_mode="stop_loss")
        )
        assert any(e.kind == "stop_loss" for e in r.events)
        assert any(c.outcome == "stop_loss" for c in r.cycles)
        # 손절 후 다음 날 새 사이클 시작.
        assert len([e for e in r.events if e.kind == "start_buy"]) >= 2


class TestIntraday:
    def test_intraday_fill_uses_bar_sequence(self):
        daily = _df([(100, 100, 100), (105, 115, 105), (105, 105, 105)])
        bars = pd.DataFrame(
            {
                "Open": [105.0, 109.0, 112.0],
                "High": [106.0, 111.0, 115.0],
                "Low": [104.0, 108.0, 111.0],
                "Close": [105.5, 110.5, 114.0],
            },
            index=pd.DatetimeIndex(
                ["2024-01-03 09:30", "2024-01-03 10:00", "2024-01-03 10:30"]
            ),
        )
        r = InfiniteBuyingStrategy().execute(daily, _config(), intraday=bars)
        sells = [e for e in r.events if e.kind == "limit_sell"]
        assert sells[0].intraday
        # 두 번째 봉(고가 111 ≥ 110)에서 110 체결.
        assert sells[0].price == pytest.approx(110.0)
        assert sells[0].ts.hour == 10
        assert r.intraday_days == 1

    def test_fees_and_tax_applied(self):
        cfg = _config(
            fee_schedule=TossFeeSchedule(),
            capital_gains_tax_pct=0.22,
        )
        r = InfiniteBuyingStrategy().execute(
            _df([(100, 100, 100), (105, 115, 105), (105, 105, 105)]), cfg
        )
        liq = r.liquidation
        assert liq.total_fees > 0
        assert liq.total_tax > 0
        assert liq.final_value_after_tax < liq.final_value_pre_tax


class TestEventDetail:
    def test_buy_events_carry_qty_tranche_and_target(self):
        r = InfiniteBuyingStrategy().execute(
            _df([(100, 100, 100)] * 3), _config()
        )
        buys = [e for e in r.events if "buy" in e.kind]
        # 시작 1T + (큰수, 평단)×2일 = 5건, 회차 1~5.
        assert [e.tranche_no for e in buys] == [1, 2, 3, 4, 5]
        assert all(e.qty > 0 for e in buys)
        # 평단 100 → 익절 목표 110, 투입률은 누적 증가.
        assert buys[-1].target_price == pytest.approx(110.0)
        assert buys[0].spent_pct == pytest.approx(1 / 40)
        assert buys[-1].spent_pct == pytest.approx(3 / 40)

    def test_sell_event_qty_matches_holdings(self):
        r = InfiniteBuyingStrategy().execute(
            _df([(100, 100, 100), (105, 115, 105), (105, 105, 105)]), _config()
        )
        sell = [e for e in r.events if e.kind == "limit_sell"][0]
        assert sell.qty == pytest.approx(25_000)  # 2.5M / 100
        assert sell.tranche_no == 0
