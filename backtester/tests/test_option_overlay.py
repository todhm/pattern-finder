import numpy as np
import pandas as pd
import pytest

from strategy.adapters.option_overlay import Chain, Leg, Spec, run_overlay

DAYS = pd.bdate_range("2020-01-13", periods=6)
EXP = "2020-01-18"                                       # 토요일 만기 → 5 번째 거래일(금) 종가로 정산


def _chain(rows):
    return Chain.from_frame(pd.DataFrame(rows, columns=["date", "expiration", "type", "strike", "bid", "ask", "delta"]))


def _px(vals, split=None):
    c = pd.Series(vals, index=DAYS, dtype=float); s = pd.Series(1.0, index=DAYS)
    if split:
        s.iloc[split[0]] = split[1]
    return c, s, pd.Series(0.0, index=DAYS)


def test_no_legs_equals_holding():
    c, s, r = _px([100, 110, 121, 100, 90, 99])
    eq, log = run_overlay(_chain([[DAYS[0], EXP, "call", 100.0, 1.0, 1.2, 0.5]]), c, s, r, Spec(w=pd.Series(1.0, index=DAYS), stock_cost=0.0))
    assert log.empty and np.allclose(eq.to_numpy(), c.to_numpy() / 100.0)


def test_covered_call_expiry_payoff_and_no_negative_cash():
    c, s, r = _px([100, 104, 108, 112, 110, 110])
    rows = [[d, EXP, "call", 105.0, 1.9, 2.1, 0.3] for d in DAYS[:5]]
    spec = Spec(w=pd.Series(1.0, index=DAYS), legs=[Leg(call=True, side=-1, delta=0.3, dte=5)], fill=0.0, stock_cost=0.0, fee=0.0)
    eq, log = run_overlay(_chain(rows), c, s, r, spec)
    # 첫날: 주식 0.01 주(자산 전부), 콜 −0.01 주, 받은 프리미엄 0.02 는 현금으로 둔다. 만기(110) 내재가치 5.
    assert eq.iloc[0] == pytest.approx(1.0)
    assert eq.iloc[4] == pytest.approx(0.01 * 110 + 0.02 - 0.01 * 5, abs=1e-9)
    assert list(log["what"]) == ["열기", "만기"] and log["qty"].iloc[0] == pytest.approx(-0.01)


def test_fill_crosses_spread():
    c, s, r = _px([100, 100, 100, 100, 100, 100])
    rows = [[d, EXP, "put", 95.0, 1.0, 2.0, -0.2] for d in DAYS[:5]]
    mk = lambda f: run_overlay(_chain(rows), c, s, r, Spec(w=pd.Series(1.0, index=DAYS), legs=[Leg(call=False, side=1, delta=0.2, dte=5)], fill=f, stock_cost=0.0, fee=0.0))[0]  # noqa: E731
    mid, full = mk(0.0), mk(1.0)
    assert mid.iloc[0] == pytest.approx(1.0) and full.iloc[0] < mid.iloc[0]          # 호가를 건너면 사자마자 평가손
    assert mid.iloc[4] == pytest.approx(1.0 - 0.01 * 1.5, rel=1e-3)                   # 풋은 무가치 만기(주식을 조금 덜 산 효과 포함)


def test_split_adjusts_strike_and_qty():
    c, s, r = _px([100, 100, 50, 50, 40, 40], split=(2, 2.0))                          # 셋째 날 2:1 분할
    rows = [[DAYS[0], EXP, "put", 90.0, 0.9, 1.1, -0.2], [DAYS[1], EXP, "put", 90.0, 0.9, 1.1, -0.2]]
    rows += [[d, EXP, "put", 45.0, 0.45, 0.55, -0.2] for d in DAYS[2:4]] + [[DAYS[4], EXP, "put", 45.0, 4.9, 5.1, -0.9]]
    spec = Spec(w=pd.Series(1.0, index=DAYS), legs=[Leg(call=False, side=1, delta=0.2, dte=5)], fill=0.0, stock_cost=0.0, fee=0.0)
    eq, log = run_overlay(_chain(rows), c, s, r, spec)
    assert eq.iloc[1] == pytest.approx(eq.iloc[2])                                     # 분할 당일 자산 연속
    last = log[log["what"] == "만기"].iloc[0]
    assert last["strike"] == pytest.approx(45.0) and last["qty"] == pytest.approx(log["qty"].iloc[0] * 2) and last["price"] == pytest.approx(5.0)


def test_cash_secured_put_is_fully_collateralised():
    c, s, r = _px([100, 90, 80, 70, 60, 60])
    rows = [[d, EXP, "put", 95.0, 2.9, 3.1, -0.3] for d in DAYS[:5]]
    spec = Spec(w=pd.Series(0.0, index=DAYS), legs=[Leg(call=False, side=-1, delta=0.3, dte=5, basis="cash")], fill=0.0, fee=0.0)
    eq, log = run_overlay(_chain(rows), c, s, r, spec)
    q = -log["qty"].iloc[0]
    assert q * 95.0 <= 1.0 + 1e-9                                                      # 행사가 × 수량 ≤ 가진 현금
    assert eq.iloc[4] == pytest.approx(1.0 + q * 3.0 - q * 35.0)                       # 프리미엄 − 내재가치(95−60)
