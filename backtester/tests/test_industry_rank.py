import numpy as np
import pandas as pd

from strategy.adapters.industry_rank import port_returns, signals, topk_weights, window_stats


def _frames(T=80):
    idx = pd.date_range("2000-01-31", periods=T, freq="ME"); cols = ["A", "B", "C", "D"]
    vw = pd.DataFrame({"A": 0.03, "B": 0.01, "C": -0.01, "D": 0.0}, index=idx, columns=cols); n = pd.DataFrame(10.0, index=idx, columns=cols); n["B"] = np.arange(10, 10 + T)
    return vw, vw.copy(), n, pd.DataFrame(100.0, index=idx, columns=cols)


def test_signals_use_only_past_months():
    vw, ew, n, size = _frames(); s = signals(vw, ew, n, size); t = vw.index[40]
    vw2 = vw.copy(); vw2.iloc[41:] = 0.5; n2 = n.copy(); n2.iloc[41:] = 999          # 미래를 바꿔도 t 까지의 점수는 그대로
    s2 = signals(vw2, ew, n2, size)
    for k in s:
        pd.testing.assert_series_equal(s[k].loc[:t].iloc[-1], s2[k].loc[:t].iloc[-1])
    assert abs(s["mom12_1"].loc[t, "A"] - (1.03 ** 11 - 1)) < 1e-9 and s["nfirm_g12"].loc[t, "B"] > 0 == s["nfirm_g12"].loc[t, "A"]


def test_topk_earns_next_month_and_pays_cost_on_change():
    vw, ew, n, size = _frames(); s = signals(vw, ew, n, size); rf = pd.Series(0.0, index=vw.index)
    w = topk_weights(s["mom12_1"], vw, 1, 12); first = w.index[(w.sum(axis=1) > 0)][0]
    assert w.loc[first, "A"] == 1.0
    r = port_returns(w, vw, rf, cost=0.002); i = vw.index.get_loc(first)
    assert r.iloc[i] == 0.0 and abs(r.iloc[i + 1] - (0.03 - 0.002)) < 1e-12 and abs(r.iloc[i + 2] - 0.03) < 1e-12
    st = window_stats(r, {"w": (str(vw.index[i + 2].date()), str(vw.index[i + 13].date()))}); assert abs(st["w"]["mult"] - 1.03 ** 12) < 1e-9 and st["w"]["mdd"] == 0.0


def test_placebo_picks_same_count_from_valid_only():
    vw, ew, n, size = _frames(); s = signals(vw, ew, n, size); n.loc[:, "D"] = 2.0
    w = topk_weights(s["mom12_1"], vw, 2, 1, rng=np.random.default_rng(1), min_n=n)
    live = w[w.sum(axis=1) > 0]; assert ((live > 0).sum(axis=1) == 2).all() and (live["D"] == 0).all() and (live.iloc[:-1][["A", "B", "C"]] > 0).any().all()
