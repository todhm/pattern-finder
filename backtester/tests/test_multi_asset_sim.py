"""multi_asset_sim — 단일 자산이면 simulate_exposure 와 동치, 두 자산 전환 시 세금·수수료, 듀얼 모멘텀 비중."""

from __future__ import annotations

import numpy as np
import pandas as pd

from strategy.adapters.dual_momentum import DualMomParams, dual_momentum_weights, month_end_index
from strategy.adapters.leveraged_etf_lab import CostParams, simulate_exposure
from strategy.adapters.multi_asset_sim import simulate_weights
from strategy.adapters.tqqq_gates import placebo_target


def _px(seed=1, n=1000):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n)
    a = 100 * np.cumprod(1 + rng.normal(0.001, 0.03, n)); b = 50 * np.cumprod(1 + rng.normal(0.0003, 0.01, n))
    return pd.DataFrame({"TQQQ": a, "GOLD": b}, index=idx)


def test_single_asset_matches_simulate_exposure():
    px = _px()
    t = pd.Series(1.0, index=px.index); t.iloc[300:400] = 0.0; t.iloc[700:720] = 0.0
    costs = CostParams(exec_lag=1, rebalance_band=0.05)
    a = simulate_exposure(px["TQQQ"], t, costs, 0.02, "a")
    b = simulate_weights(px[["TQQQ"]], t.to_frame("TQQQ"), costs, 0.02, "b")
    assert np.allclose(a.equity.values, b.equity.values, rtol=1e-9)
    assert abs(a.final_after_liquidation - b.final_after_liquidation) < 1e-6 and a.trades == b.trades


def test_two_assets_rotation_pays_tax_and_fees():
    px = _px()
    w = pd.DataFrame(0.0, index=px.index, columns=["TQQQ", "GOLD"])
    w.iloc[:500, 0] = 1.0; w.iloc[500:, 1] = 1.0
    nocost = CostParams(buy_fee=0, sell_fee=0, tax_rate=0, initial_capital=1.0, rebalance_band=0, exec_lag=0)
    r0 = simulate_weights(px, w, nocost)
    expected = (px["TQQQ"].iloc[500] / px["TQQQ"].iloc[0]) * (px["GOLD"].iloc[-1] / px["GOLD"].iloc[500])   # 500 일째 종가에 교체
    assert abs(r0.final_after_liquidation - expected) < 1e-9
    r1 = simulate_weights(px, w, CostParams(initial_capital=1e8, rebalance_band=0, exec_lag=0))
    assert r1.final_after_liquidation < r0.final_after_liquidation * 1e8 and r1.fees > 0 and r1.trades >= 2
    assert 0 <= r1.exposure.max() <= 1.0 + 1e-9


def test_dual_momentum_weights_and_placebo():
    px = _px(2, 1500)
    qqq = px["TQQQ"] ** (1 / 3)
    w = dual_momentum_weights(px["TQQQ"], qqq, px["GOLD"], None, DualMomParams("rel", 6, 1, "tqqq"))
    assert set(w.columns) == {"TQQQ", "GOLD"} and (w.sum(axis=1) <= 1.0 + 1e-9).all()
    me = month_end_index(px.index)
    assert w.loc[me[:6], "TQQQ"].eq(1.0).all()          # 워밍업은 TQQQ 보유
    w2 = dual_momentum_weights(px["TQQQ"], qqq, px["GOLD"], None, DualMomParams("abs", 3, 0, "qqq"))
    assert len(w2) == len(px)
    pt = placebo_target(w, 7)
    assert pt.shape == w.shape and abs(pt["TQQQ"].sum() - w["TQQQ"].sum()) < 1e-9 and abs(pt["GOLD"].sum() - w["GOLD"].sum()) < 1e-9
