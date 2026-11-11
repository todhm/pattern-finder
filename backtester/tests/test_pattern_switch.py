from __future__ import annotations

import numpy as np
import pandas as pd

from strategy.adapters.leveraged_etf_lab import deep_correction_filter
from strategy.adapters.pattern_switch import PatternSwitchParams, pattern_switch


def test_pattern_switch_modes():
    idx = pd.bdate_range("2012-01-02", periods=900); rng = np.random.default_rng(5)
    c = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0004, 0.01, 900))), index=idx)
    sells = [idx[300].date(), idx[600].date()]; buys = [idx[320].date()]
    a = pattern_switch(c, sells, buys, PatternSwitchParams("pattern", "all", "wedge_pop"))
    assert a.iloc[301] == 0.0 and a.iloc[321] == 1.0 and a.iloc[601] == 0.0 and a.iloc[-1] == 0.0   # 두 번째 매도 뒤 매수 신호 없음
    b = pattern_switch(c, sells, buys, PatternSwitchParams("pattern", "all", "sma20"))
    assert b.iloc[301] == 0.0 and (b.iloc[302:400] == 1.0).any()
    e = pattern_switch(c, [], [], PatternSwitchParams("switch18_early", "all", dd_alarm=0.05))
    assert (e.values == deep_correction_filter(c, 200, 0.18, 20).values).all()        # 매도 신호 없으면 스위치 18
    e2 = pattern_switch(c, sells, [], PatternSwitchParams("switch18_early", "all", dd_alarm=0.0))
    assert e2.sum() <= e.sum()
