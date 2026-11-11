from __future__ import annotations

import numpy as np
import pandas as pd

from strategy.adapters.sector_rotation import SectorRotParams, sector_rotation_weights


def test_rotation_picks_sector_only_when_relative_momentum_positive():
    idx = pd.bdate_range("2015-01-01", periods=800)
    rng = np.random.default_rng(3)
    q = pd.Series(100 * np.cumprod(1 + rng.normal(0.0004, 0.01, 800)), index=idx)
    t = 100 * (1 + 3 * q.pct_change().fillna(0)).cumprod()
    strong = q * np.linspace(1, 3, 800)                      # 섹터가 꾸준히 더 강함
    w = sector_rotation_weights(t, q, strong, SectorRotParams(6, 0.0, "qqq"))
    assert (w.sum(axis=1) == 1.0).all() and w["SOXX"].iloc[-200:].mean() > 0.9
    weak = q * np.linspace(1, 0.5, 800)
    w2 = sector_rotation_weights(t, q, weak, SectorRotParams(6, 0.0, "qqq"))
    assert w2["SOXX"].iloc[-200:].mean() < 0.1
    nan_sector = pd.Series(np.nan, index=idx)
    w3 = sector_rotation_weights(t, q, nan_sector, SectorRotParams(3, 0.05, "tqqq"))
    assert (w3["TQQQ"] == 1.0).all()
