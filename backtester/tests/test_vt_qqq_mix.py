"""vt_qqq_mix — 비중 합 1(또는 스위치 현금), 목표 높으면 TQQQ 100%, 스위치 변형."""

from __future__ import annotations

import numpy as np
import pandas as pd

from strategy.adapters.vt_qqq_mix import VtMixParams, vt_mix_weights


def _px():
    rng = np.random.default_rng(9)
    n = 1200
    q = pd.Series(100 * np.cumprod(1 + rng.normal(0.0004, 0.012, n)), index=pd.bdate_range("2014-01-01", periods=n))
    q.iloc[600:700] *= np.linspace(1, 0.7, 100)
    t = 100 * (1 + 3 * q.pct_change().fillna(0)).cumprod()
    return t, q


def test_weights_sum_and_bounds():
    t, q = _px()
    w = vt_mix_weights(t, q, VtMixParams(0.45, "rv20", "none"))
    assert np.allclose(w.sum(axis=1), 1.0) and (w >= 0).all().all()
    hi = vt_mix_weights(t, q, VtMixParams(5.0, "rv20", "none"))
    assert (hi["TQQQ"] == 1.0).all() and (hi["QQQ"] == 0.0).all()


def test_switch_variants():
    t, q = _px()
    cash = vt_mix_weights(t, q, VtMixParams(0.45, "rv60", "cash"))
    qqq = vt_mix_weights(t, q, VtMixParams(0.45, "rv60", "qqq"))
    off = cash.sum(axis=1) < 1e-9
    assert off.any()                                    # 급락 구간에 이탈이 있다
    assert np.allclose(qqq.sum(axis=1), 1.0) and (qqq.loc[off, "QQQ"] == 1.0).all()
    assert (cash["TQQQ"] == qqq["TQQQ"]).all()
