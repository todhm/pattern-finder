"""H13 — 한국 세제: 연 통산·250만 공제·이월 없음(기존)·배당세 드래그(신규)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from strategy.adapters.leveraged_etf_lab import CostParams, simulate_exposure
from strategy.adapters.multi_asset_sim import simulate_weights


def _two_year():
    idx = pd.bdate_range("2020-01-01", "2021-12-31")
    y1 = idx.year == 2020
    px = pd.Series(0.0, index=idx)
    px[y1] = np.linspace(100, 90, y1.sum()); px[~y1] = np.linspace(90, 100, (~y1).sum())
    t = pd.Series(1.0, index=idx); t[idx[y1][-1]] = 0.0          # 2020 마지막 날 손실 실현
    return px, t


def test_no_loss_carry_and_deduction():
    px, t = _two_year()
    c = CostParams(buy_fee=0, sell_fee=0, tax_rate=0.22, tax_deduction=2.5e6, initial_capital=1e8, rebalance_band=0, exec_lag=0)
    r = simulate_exposure(px, t, c, None)
    assert abs(r.final_after_liquidation / 1e8 - (1.0 - (10e6 - 2.5e6) * 0.22 / 1e8)) < 1e-9     # 이월 없음 + 공제
    c0 = CostParams(buy_fee=0, sell_fee=0, tax_rate=0.22, tax_deduction=0.0, initial_capital=1e8, rebalance_band=0, exec_lag=0)
    assert simulate_exposure(px, t, c0, None).final_after_liquidation < r.final_after_liquidation


def test_dividend_tax_drag_single_and_multi():
    idx = pd.bdate_range("2015-01-01", periods=252 * 4)
    px = pd.Series(100.0, index=idx)
    base = CostParams(buy_fee=0, sell_fee=0, tax_rate=0, initial_capital=1.0, rebalance_band=0, exec_lag=0)
    kr = CostParams(buy_fee=0, sell_fee=0, tax_rate=0, initial_capital=1.0, rebalance_band=0, exec_lag=0, dividend_yield=0.01, dividend_tax_rate=0.154)
    hold = pd.Series(1.0, index=idx)
    a = simulate_exposure(px, hold, base); b = simulate_exposure(px, hold, kr)
    assert abs(a.final_after_liquidation - 1.0) < 1e-9
    assert abs((1.0 - b.final_after_liquidation) - 4 * 0.01 * 0.154) < 2e-4 and b.tax > 0   # 4년 × 1% × 15.4%
    prices = pd.DataFrame({"TQQQ": px, "GOLD": px})
    w = pd.DataFrame({"TQQQ": 0.5, "GOLD": 0.5}, index=idx)
    m = simulate_weights(prices, w, kr, None, "m", dividend_yields={"TQQQ": 0.02, "GOLD": 0.0})
    assert abs((1.0 - m.final_after_liquidation) - 4 * 0.5 * 0.02 * 0.154) < 2e-4
    m0 = simulate_weights(prices, w, base, None, "m0")
    assert abs(m0.final_after_liquidation - 1.0) < 1e-9
