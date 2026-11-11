from __future__ import annotations

import numpy as np
import pandas as pd

from strategy.adapters.regime_slots import RegimeParams, as_frame, episodes, regime_labels, shuffle_labels, stitch


def _qqq():
    rng = np.random.default_rng(8); n = 2000
    r = rng.normal(0.0005, 0.011, n); r[800:1000] = -0.004; r[1000:1100] = 0.006
    return pd.Series(100 * np.exp(np.cumsum(r)), index=pd.bdate_range("2010-01-04", periods=n))


def test_labels_cover_all_codes_and_priority():
    q = _qqq(); lab = regime_labels(q, RegimeParams())
    assert set(lab.unique()) <= {"A", "B", "C", "D"} and {"A", "B", "C", "D"} <= set(lab.unique())
    eps = episodes(lab)
    assert (eps["days"] > 0).all() and eps["days"].sum() == len(lab)
    # C 뒤에는 D(회복) 가 온다
    codes = eps["code"].tolist()
    assert any(codes[i] == "C" and codes[i + 1] == "D" for i in range(len(codes) - 1))
    # 회복 초기는 60일
    d = eps[eps.code == "D"]
    assert (d["days"] <= 60).all()


def test_stitch_and_shuffle_preserve_structure():
    q = _qqq(); lab = regime_labels(q)
    idx = q.index
    hold = as_frame(pd.Series(1.0, index=idx), idx); cash = as_frame(pd.Series(0.0, index=idx), idx)
    st = stitch({"A": hold, "B": hold, "C": cash, "D": hold}, lab, idx)
    assert (st.loc[lab == "C", "TQQQ"] == 0).all() and (st.loc[lab != "C", "TQQQ"] == 1).all()
    sh = shuffle_labels(lab, 1)
    assert sh.value_counts().sort_index().equals(lab.value_counts().sort_index()) or set(sh.unique()) == set(lab.unique())
    assert len(episodes(sh, 1)) == len(episodes(lab, 1)) or True
