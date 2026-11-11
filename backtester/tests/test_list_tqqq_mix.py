"""list_tqqq_mix — 리밸런스일 비중 복원·사이 드리프트·스위치는 TQQQ 몫만."""

from __future__ import annotations

import numpy as np
import pandas as pd

from strategy.adapters.list_tqqq_mix import ListMixParams, list_mix_weights, rebalance_days
from strategy.adapters.multi_asset_sim import simulate_weights
from strategy.adapters.leveraged_etf_lab import CostParams


def _px():
    rng = np.random.default_rng(11); n = 800
    idx = pd.bdate_range("2015-01-01", periods=n)
    q = pd.Series(100 * np.cumprod(1 + rng.normal(0.0004, 0.012, n)), index=idx)
    q.iloc[500:560] *= np.linspace(1, 0.72, 60)
    t = 100 * (1 + 3 * q.pct_change().fillna(0)).cumprod()
    l = pd.Series(100 * np.cumprod(1 + rng.normal(0.0005, 0.01, n)), index=idx)
    return pd.DataFrame({"TQQQ": t, "LIST": l}), q


def test_rebalance_and_drift_no_trades_between():
    px, q = _px()
    w = list_mix_weights(px, q, ListMixParams(0.5, "Q", False))
    rd = rebalance_days(px.index, "Q")
    assert np.allclose(w.loc[rd, "TQQQ"], 0.5) and np.allclose(w.sum(axis=1), 1.0)
    nocost = CostParams(buy_fee=0, sell_fee=0, tax_rate=0, initial_capital=1.0, rebalance_band=0.001, exec_lag=0)
    r = simulate_weights(px, w, nocost)
    assert r.trades <= 2 * len(rd) + 2                    # 리밸런스일에만 거래
    assert abs(r.final_after_liquidation - r.equity.iloc[-1]) < 1e-9


def test_switch_only_hits_tqqq_leg():
    px, q = _px()
    a = list_mix_weights(px, q, ListMixParams(0.5, "Q", False))
    b = list_mix_weights(px, q, ListMixParams(0.5, "Q", True))
    assert (b["LIST"] == a["LIST"]).all() and (b["TQQQ"] <= a["TQQQ"] + 1e-12).all() and (b["TQQQ"] == 0).any()
    px2 = px.copy(); px2.loc[px2.index[:100], "LIST"] = np.nan
    c = list_mix_weights(px2, q, ListMixParams(0.7, "A", False))
    assert (c.iloc[:100]["LIST"] == 0).all() and (c.iloc[:100]["TQQQ"] == 0.7).all()
