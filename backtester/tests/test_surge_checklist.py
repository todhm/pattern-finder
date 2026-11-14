import numpy as np
import pandas as pd

from strategy.adapters.surge_checklist import checklist_features, checklist_score


def _frames(n=400):
    idx = pd.bdate_range("2020-01-01", periods=n)
    t = np.arange(n)
    smooth = 100 * 1.002 ** t                                   # 꾸준히 오름
    broken = np.where(t < 300, 100 * 1.003 ** t, 100 * 1.003 ** 300 * 0.99 ** (t - 300))   # 300일 뒤 꺾임
    para = np.where(t < 370, 100 * 1.001 ** t, 100 * 1.001 ** 370 * 1.03 ** (t - 370))      # 마지막 30일 수직
    c = pd.DataFrame({"SMOOTH": smooth, "BROKEN": broken, "PARA": para}, index=idx)
    dv = pd.DataFrame(1e8, index=idx, columns=c.columns)
    return c, dv


def test_smooth_uptrend_passes_buy_items_and_no_avoid():
    c, dv = _frames(); f = checklist_features(c, dv); last = {k: v.iloc[-1] for k, v in f.items()}
    assert last["m1"]["SMOOTH"] and last["m2"]["SMOOTH"]
    assert not any(last[k]["SMOOTH"] for k in ("x1", "x2", "x3", "x4", "x5"))


def test_broken_and_parabolic_are_flagged():
    c, dv = _frames(); f = checklist_features(c, dv)
    assert f["x3"]["BROKEN"].iloc[-1] and not f["m1"]["BROKEN"].iloc[-1]
    assert f["x4"]["PARA"].iloc[-1] and not f["m2"]["PARA"].iloc[-1]
    s = checklist_score(f).iloc[-1]
    assert s["SMOOTH"] > s["BROKEN"] and s["SMOOTH"] > s["PARA"]


def test_distribution_day_and_skip():
    c, dv = _frames(); c = c.copy(); dv = dv.copy()
    c.iloc[380:, 0] *= 0.9; dv.iloc[380, 0] = 5e8                # 거래 5배 -10% 음봉
    f = checklist_features(c, dv)
    assert f["x2"]["SMOOTH"].iloc[-1] and not f["x2"]["SMOOTH"].iloc[379]
    assert (checklist_score(f, skip=("x2",)) - checklist_score(f)).iloc[-1]["SMOOTH"] == 1


def test_no_lookahead():
    c, dv = _frames(); a = checklist_score(checklist_features(c, dv)).iloc[350]
    b = checklist_score(checklist_features(c.iloc[:351], dv.iloc[:351])).iloc[-1]
    assert (a == b).all()
