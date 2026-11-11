from __future__ import annotations

import pandas as pd

from strategy.adapters.seasonal_switch import SeasonParams, in_season, seasonal_weights


def test_halloween_and_earnings_masks():
    idx = pd.bdate_range("2020-01-01", "2020-12-31")
    h = in_season(idx, SeasonParams("halloween", (11, 12, 1, 2, 3, 4)))
    assert h[idx.month == 1].all() and not h[idx.month == 7].any()
    e = in_season(idx, SeasonParams("earnings", weeks=(2, 3, 4)))
    assert e[pd.Timestamp("2020-01-15")] and not e[pd.Timestamp("2020-01-02")] and not e[pd.Timestamp("2020-02-12")]
    assert not e[pd.Timestamp("2020-01-30")]           # 다섯째 주
    w = seasonal_weights(idx, SeasonParams("halloween", off_asset="qqq"))
    assert (w.sum(axis=1) == 1.0).all()
    wc = seasonal_weights(idx, SeasonParams("earnings", off_asset="cash"))
    assert (wc["QQQ"] == 0).all() and 0 < wc["TQQQ"].mean() < 0.35
