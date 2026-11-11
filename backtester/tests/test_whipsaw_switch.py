from __future__ import annotations

import numpy as np
import pandas as pd

from strategy.adapters.leveraged_etf_lab import deep_correction_filter
from strategy.adapters.whipsaw_switch import WhipsawParams, round_trips, whipsaw_switch


def _px():
    rng = np.random.default_rng(6); n = 1800
    r = rng.normal(0.0004, 0.013, n); r[600:700] = -0.01; r[700:720] = 0.02; r[720:760] = -0.012; r[760:800] = 0.015
    return pd.Series(100 * np.exp(np.cumsum(r)), index=pd.bdate_range("2011-01-03", periods=n))


def test_base_equals_switch18_and_variants():
    px = _px()
    base = whipsaw_switch(px, WhipsawParams())
    assert (base.values == deep_correction_filter(px, 200, 0.18, 20).values).all()
    band = whipsaw_switch(px, WhipsawParams(band=0.03))
    monthly = whipsaw_switch(px, WhipsawParams(monthly=True))
    two = whipsaw_switch(px, WhipsawParams(two_stage=True))
    s, e = str(px.index[0].date()), str(px.index[-1].date())
    assert round_trips(band, s, e) <= round_trips(base, s, e)
    assert round_trips(monthly, s, e) <= round_trips(base, s, e)
    assert set(np.unique(two.values)) <= {0.0, 0.5, 1.0} and (two == 0.5).any()
    # 월말 판정: 상태 변화(워밍업 끝 제외)는 그 달 마지막 거래일에만
    chg = monthly.diff().fillna(0) != 0
    last_days = set(pd.Series(monthly.index, index=monthly.index).groupby(monthly.index.to_period("M")).last().values)
    assert all(d in last_days for d in monthly.index[chg][1:])
