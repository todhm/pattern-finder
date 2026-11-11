from __future__ import annotations

import numpy as np
import pandas as pd

from strategy.adapters.calm_exit_switch import CalmExitParams, calm_exit_switch
from strategy.adapters.leveraged_etf_lab import deep_correction_filter


def test_huge_r_equals_switch18_and_delay_bounded():
    rng = np.random.default_rng(9); n = 2000
    r = rng.normal(0.0003, 0.012, n); r[700:800] = -0.006
    c = pd.Series(100 * np.exp(np.cumsum(r)), index=pd.bdate_range("2011-01-03", periods=n))
    df = pd.DataFrame({"Open": c.shift(1).fillna(c.iloc[0]), "High": c, "Low": c, "Close": c, "Volume": 1.0})
    base = deep_correction_filter(c, 200, 0.18, 20)
    a = calm_exit_switch(df, CalmExitParams(9.0, 5))
    assert (a.values == base.values).all()
    b = calm_exit_switch(df, CalmExitParams(0.005, 3))
    # 이탈은 기본보다 늦되 3일 이내
    ex_a = np.flatnonzero(np.diff(base.values) < 0); ex_b = np.flatnonzero(np.diff(b.values) < 0)
    assert len(ex_b) >= 1 and all(any(0 <= eb - ea <= 3 for ea in ex_a) for eb in ex_b)
