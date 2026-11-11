from __future__ import annotations

import numpy as np
import pandas as pd

from strategy.adapters.leveraged_etf_lab import deep_correction_filter
from strategy.adapters.vol_scaled_switch import VolScaledParams, vol_scaled_switch


def test_fixed_band_equals_switch_and_k_monotone():
    rng = np.random.default_rng(2); n = 2000
    r = rng.normal(0.0003, 0.012, n); r[800:950] = -0.006
    c = pd.Series(100 * np.exp(np.cumsum(r)), index=pd.bdate_range("2011-01-03", periods=n))
    fixed = vol_scaled_switch(c, VolScaledParams(k=0.0, lo=0.18, hi=0.18))
    assert (fixed.values == deep_correction_filter(c, 200, 0.18, 20).values).all()
    low = vol_scaled_switch(c, VolScaledParams(0.6, 60)); high = vol_scaled_switch(c, VolScaledParams(0.9, 60))
    assert low.sum() <= high.sum()
