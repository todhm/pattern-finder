from __future__ import annotations

import numpy as np
import pandas as pd

from strategy.adapters.leveraged_etf_lab import CostParams
from strategy.adapters.regime_slots import as_frame
from strategy.adapters.walk_forward_select import WalkForwardParams, quarter_starts, walk_forward


def test_walk_forward_picks_best_and_placebo_random():
    idx = pd.bdate_range("2012-01-02", periods=1500); rng = np.random.default_rng(1)
    q = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, 1500))), index=idx)
    prices = pd.DataFrame({"TQQQ": 100 * (1 + 3 * q.pct_change().fillna(0)).cumprod(), "QLD": 100 * (1 + 2 * q.pct_change().fillna(0)).cumprod(), "QQQ": q})
    lib = {"hold": as_frame(pd.Series(1.0, index=idx), idx), "cash": as_frame(pd.Series(0.0, index=idx), idx)}
    costs = CostParams(buy_fee=0, sell_fee=0, tax_rate=0, initial_capital=1.0, rebalance_band=0, exec_lag=0)
    first = idx[0] + pd.DateOffset(years=2)
    w, picks = walk_forward(prices, lib, WalkForwardParams(2, "mult"), costs, None, first)
    assert len(picks) == len(quarter_starts(idx, first)) and set(picks.unique()) <= {"hold", "cash"}
    assert picks.value_counts().get("hold", 0) >= picks.value_counts().get("cash", 0)   # 상승 세계에선 보유가 더 자주
    assert np.allclose(w.sum(axis=1).loc[idx >= first].max(), 1.0)
    w2, picks2 = walk_forward(prices, lib, WalkForwardParams(2, "calmar", True), costs, None, first, rng=np.random.default_rng(3))
    assert len(picks2) == len(picks)
