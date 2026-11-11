from __future__ import annotations

import numpy as np
import pandas as pd

from strategy.adapters.bear_len_reentry import BearLenParams, bear_len_switch
from strategy.adapters.leveraged_etf_lab import deep_correction_filter


def test_huge_L_equals_switch18_and_slow_modes_hold_cash_longer():
    rng = np.random.default_rng(7); n = 2500
    r = rng.normal(0.0003, 0.012, n); r[900:1100] = -0.004; r[1100:1200] = 0.004
    px = pd.Series(100 * np.exp(np.cumsum(r)), index=pd.bdate_range("2010-01-04", periods=n))
    base = deep_correction_filter(px, 200, 0.18, 20)
    a = bear_len_switch(px, BearLenParams(10**6, "sma50"))
    assert (a.values == base.values).all()
    for mode in ("sma50", "monthly20"):
        s = bear_len_switch(px, BearLenParams(20, mode))
        assert s.sum() <= base.sum()
