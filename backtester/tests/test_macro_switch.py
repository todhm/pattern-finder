"""macro_switch — 레짐 매핑, 스위치 18 과의 동치(점수 중립), 위험 레짐에서 더 일찍 이탈."""

from __future__ import annotations

import numpy as np
import pandas as pd

from strategy.adapters.leveraged_etf_lab import deep_correction_filter
from strategy.adapters.macro_switch import MacroSwitchParams, dd_threshold_series, macro_switch, regime_series, regime_share


def _px():
    rng = np.random.default_rng(5)
    n = 252 * 6
    r = rng.normal(0.0005, 0.012, n)
    r[800:900] = -0.006
    return pd.Series(100 * np.cumprod(1 + r), index=pd.bdate_range("2012-01-02", periods=n))


def test_regime_mapping_and_neutral_when_missing():
    idx = pd.bdate_range("2012-01-02", periods=10)
    m = pd.Series([0.0, 0.25, 0.5, 0.75, 1.0], index=idx[2:7])
    reg = regime_series(m, idx, 0.5, 0.75)
    assert reg.iloc[:2].tolist() == [0, 0]                   # 점수 이전 = 중립
    assert reg.loc[idx[2:7]].tolist() == [-1, -1, 0, 1, 1]
    assert reg.iloc[7:].tolist() == [1, 1, 1]                # ffill
    p = MacroSwitchParams(dd_risk=0.1, dd_mid=0.18, dd_healthy=0.24)
    dd = dd_threshold_series(m, idx, p)
    assert dd.loc[idx[2]] == 0.1 and dd.loc[idx[4]] == 0.18 and dd.loc[idx[6]] == 0.24
    sh = regime_share(m, idx, p)
    assert abs(sum(sh.values()) - 1.0) < 1e-9


def test_equals_switch18_when_all_thresholds_equal():
    px = _px()
    m = pd.Series(np.random.default_rng(1).random(len(px)), index=px.index)
    p = MacroSwitchParams(dd_risk=0.18, dd_mid=0.18, dd_healthy=0.18)
    a = macro_switch(px, m, p)
    b = deep_correction_filter(px, 200, 0.18, 20)
    assert (a.values == b.values).all()


def test_risk_regime_exits_earlier_than_healthy():
    px = _px()
    risky = pd.Series(0.0, index=px.index)
    healthy = pd.Series(1.0, index=px.index)
    p = MacroSwitchParams(dd_risk=0.08, dd_mid=0.18, dd_healthy=0.30)
    a = macro_switch(px, risky, p); b = macro_switch(px, healthy, p)
    assert a.sum() <= b.sum()
    assert set(a.unique()) <= {0.0, 1.0}
