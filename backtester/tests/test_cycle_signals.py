import numpy as np
import pandas as pd

from strategy.adapters.cycle_signals import features, flips, rule_state, to_daily


def test_to_daily_never_uses_a_month_before_it_is_published():
    m = pd.Series([1.0, 2.0, 3.0], index=pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01"]))
    days = pd.bdate_range("2020-01-01", "2020-06-30"); d = to_daily(m, days, lag=1)
    assert np.isnan(d["2020-02-28"])                       # 1월 값은 3월 1일 전에는 모른다
    assert d["2020-03-02"] == 1.0 and d["2020-03-31"] == 1.0
    assert d["2020-04-01"] == 2.0 and d["2020-05-01"] == 3.0
    assert to_daily(m, days, lag=0)["2020-02-03"] == 1.0


def test_features_yoy_and_accel():
    idx = pd.date_range("2015-01-01", periods=30, freq="MS"); x = pd.Series(100.0 * 1.01 ** np.arange(30), index=idx)
    f = features(x)
    assert np.isnan(f["yoy"].iloc[11]) and abs(f["yoy"].iloc[12] - (1.01 ** 12 - 1)) < 1e-9
    assert abs(f["accel3"].iloc[20]) < 1e-9                 # 성장률이 일정하면 가속 0


def test_trough_rule_buys_low_and_sells_high_only():
    idx = pd.date_range("2010-01-01", periods=8, freq="MS")
    f = pd.DataFrame({"yoy": [-.3, -.2, 0, .2, .4, .5, .4, .2], "accel3": [-1, 1, 1, 1, 1, 1, -1, -1], "accel6": 0.0, "pct": [.1, .2, .5, .6, .8, .9, .8, .5]}, index=idx)
    s = rule_state(f, "trough")
    assert list(s) == [0, 1, 1, 1, 1, 1, 0, 0]              # 하위 30% 에서 돌아설 때 매수, 상위 30% 에서 꺾일 때 매도
    assert list(rule_state(f, "yoy_pos")) == [0, 0, 0, 1, 1, 1, 1, 1]
    fl = flips(s); assert list(fl["side"]) == ["매수", "매도"]
