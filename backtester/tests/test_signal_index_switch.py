from __future__ import annotations

import numpy as np
import pandas as pd

from strategy.adapters.leveraged_etf_lab import deep_correction_filter
from strategy.adapters.signal_index_switch import SignalIdxParams, signal_index_switch


def test_same_series_equals_switch_and_both_is_conservative():
    rng = np.random.default_rng(3); n = 2000
    r = rng.normal(0.0003, 0.012, n); r[800:950] = -0.006
    q = pd.Series(100 * np.exp(np.cumsum(r)), index=pd.bdate_range("2010-01-04", periods=n))
    s = q * (1 + rng.normal(0, 0.002, n)).cumprod()
    assert (signal_index_switch(q, q, SignalIdxParams("spx")).values == deep_correction_filter(q, 200, 0.18, 20).values).all()
    both = signal_index_switch(q, s, SignalIdxParams("both")); either = signal_index_switch(q, s, SignalIdxParams("either"))
    assert both.sum() >= either.sum()
