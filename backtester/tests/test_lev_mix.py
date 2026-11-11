from __future__ import annotations

import numpy as np
import pandas as pd

from strategy.adapters.lev_mix import LevMixParams, lev_mix_weights


def test_qld_and_mix_weights():
    idx = pd.bdate_range("2015-01-01", periods=900)
    rng = np.random.default_rng(2)
    q = pd.Series(100 * np.cumprod(1 + rng.normal(0.0004, 0.012, 900)), index=idx)
    q.iloc[500:580] *= np.linspace(1, 0.7, 80)
    px = pd.DataFrame({"TQQQ": 100 * (1 + 3 * q.pct_change().fillna(0)).cumprod(), "QLD": 100 * (1 + 2 * q.pct_change().fillna(0)).cumprod(), "QQQ": q})
    a = lev_mix_weights(px, q, LevMixParams("qld"))
    assert (a["QLD"] == 1.0).all() and (a[["TQQQ", "QQQ"]] == 0).all().all()
    b = lev_mix_weights(px, q, LevMixParams("qld", switch_dd=0.18))
    assert (b["QLD"] == 0).any() and b["QLD"].max() == 1.0
    c = lev_mix_weights(px, q, LevMixParams("mix", 0.5, 0.18))
    on = c["TQQQ"] > 0                                       # 스위치 on 인 날만 합 1(off 면 TQQQ 몫이 현금)
    assert on.any() and np.allclose(c.loc[on, "TQQQ"] + c.loc[on, "QQQ"], 1.0) and (c["TQQQ"] == 0).any() and (c["QQQ"] > 0).all()
    d = lev_mix_weights(px, q, LevMixParams("mix", 0.67))
    assert np.allclose(d[["TQQQ", "QQQ"]].sum(axis=1), 1.0)
