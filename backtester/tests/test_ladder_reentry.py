from __future__ import annotations

import numpy as np
import pandas as pd

from strategy.adapters.ladder_reentry import LadderParams, ladder_switch
from strategy.adapters.leveraged_etf_lab import deep_correction_filter


def test_no_rungs_hit_equals_switch_and_ladder_adds_partial_exposure():
    rng = np.random.default_rng(12); n = 2200
    r = rng.normal(0.0003, 0.012, n); r[900:1050] = -0.01
    q = pd.Series(100 * np.exp(np.cumsum(r)), index=pd.bdate_range("2010-01-04", periods=n))
    t = 100 * (1 + 3 * q.pct_change().fillna(0)).cumprod()
    base = deep_correction_filter(q, 200, 0.18, 20)
    far = ladder_switch(q, t, LadderParams((1.5, 1.6, 1.7)))   # 닿을 수 없는 사다리
    assert (far.values == base.values).all()
    lad = ladder_switch(q, t, LadderParams((0.3, 0.45, 0.6)))
    assert lad.sum() >= base.sum() and set(np.round(lad.unique(), 3)) <= {0.0, 0.333, 0.667, 1.0}
    # 한 낙폭 에피소드에서 사다리는 한 번만: 재이탈 직후 같은 칸으로 다시 사지 않는다
    lad2 = ladder_switch(q, t, LadderParams((0.3, 0.45, 0.6)))
    ups = (lad2.diff() > 0) & (lad2 < 1.0); downs = lad2.diff() < 0
    assert ups.sum() <= 3 * (downs.sum() + 1)
    # 게이트가 전부 True 면 사다리 매수 없음 = 스위치 18
    gated = ladder_switch(q, t, LadderParams((0.3, 0.45, 0.6)), gate=pd.Series(True, index=q.index), gate_size=0.0)
    assert (gated.values == base.values).all()
    # 월말 게이트가 전부 True 면 이탈·재진입이 월말에만
    mg = ladder_switch(q, t, LadderParams((1.5, 1.6, 1.7)), monthly_gate=pd.Series(True, index=q.index))
    chg = mg.diff().fillna(0) != 0
    last_days = set(pd.Series(q.index, index=q.index).groupby(q.index.to_period("M")).last().values)
    assert all(d in last_days for d in q.index[chg][1:])
