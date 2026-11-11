from __future__ import annotations

import numpy as np
import pandas as pd

from strategy.adapters.leveraged_etf_lab import deep_correction_filter
from strategy.adapters.reentry_switch import ReentryParams, reentry_switch


def _px():
    rng = np.random.default_rng(4); n = 1500
    r = rng.normal(0.0004, 0.012, n); r[700:760] = -0.012; r[760:780] = 0.02; r[780:800] = -0.015
    return pd.Series(100 * np.exp(np.cumsum(r)), index=pd.bdate_range("2012-01-02", periods=n))


def test_default_equals_switch18_and_variants_monotone():
    px = _px()
    base = reentry_switch(px, None, ReentryParams())
    ref = deep_correction_filter(px, 200, 0.18, 20)
    assert (base.values == ref.values).all()
    b = pd.Series(0.3, index=px.index)                       # 폭 낮음 → 재진입 막힘
    gated = reentry_switch(px, b, ReentryParams(breadth_min=0.5))
    assert gated.sum() <= base.sum()
    nan_b = pd.Series(np.nan, index=px.index)                # 폭 무효 → 조건 없음
    assert (reentry_switch(px, nan_b, ReentryParams(breadth_min=0.5)).values == base.values).all()
    locked = reentry_switch(px, None, ReentryParams(lock_days=60))
    assert locked.sum() >= base.sum()
    flips = lambda s: int((s.diff().abs() > 0).sum())
    assert flips(locked) <= flips(base)
