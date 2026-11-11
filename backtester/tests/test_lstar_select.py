from __future__ import annotations

import numpy as np
import pandas as pd

from strategy.adapters.lstar_select import LstarParams, lstar_weights, trailing_lstar


def test_lstar_and_weights():
    rng = np.random.default_rng(4); n = 2500
    q = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0008, 0.012, n))), index=pd.bdate_range("2010-01-04", periods=n))
    L = trailing_lstar(q, q.index[1500], 3)
    assert np.isfinite(L) and L > 0
    w, picks = lstar_weights(q, LstarParams(3, 3.0, 2.0, False), q.index[800])
    assert np.allclose(w.sum(axis=1), 1.0) and set(picks.unique()) <= {"TQQQ", "QLD", "QQQ"}
    w2, _ = lstar_weights(q, LstarParams(3, 3.0, 2.0, True), q.index[800])
    assert (w2.sum(axis=1) <= 1.0 + 1e-9).all()
    w3, p3 = lstar_weights(q, LstarParams(3, 3.0, 2.0, False), q.index[800], rng=np.random.default_rng(1))
    assert len(p3) == len(picks)
