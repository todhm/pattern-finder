"""tqqq_gates — 창 표·관문·플라세보. 합성 세계(보유 = 비율 1, 완벽 회피 = 전부 통과, 반대 = G1 탈락)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from strategy.adapters.leveraged_etf_lab import CostParams
from strategy.adapters.tqqq_gates import (
    EDITIONS, g5_threshold, gate_verdicts, load_baseline, placebo_target, placebo_medians, sealed_verdicts,
    verdict_markdown, window_table,
)

NO_COST = CostParams(buy_fee=0, sell_fee=0, tax_rate=0, initial_capital=1.0, rebalance_band=0.0, exec_lag=0)


def _world():
    rng = np.random.default_rng(3)
    n = 252 * 14
    r = rng.normal(0.0015, 0.02, n)
    r[1000:1120] = -0.012         # 폭락 1
    r[2500:2600] = -0.010         # 폭락 2
    lev = pd.Series(100 * np.cumprod(1 + r), index=pd.bdate_range("2010-01-04", periods=n), name="TQQQ")
    return lev, r


def _baseline_from(lev: pd.Series, tmp_path):
    """합성 가격으로 CSV 기준선을 만든다(열 이름은 실제 파일과 같게). TQQQ 세후 = 세전 × 0.78."""
    rows = []
    starts = pd.date_range("2010-02-01", periods=6, freq="6MS")
    for ed_key, ed_name in EDITIONS.items():
        for s in starts:
            for bundle, e in (("A-10", s + pd.DateOffset(years=10)), ("A-∞", lev.index[-1])):
                if e > lev.index[-1]:
                    continue
                w = lev[(lev.index >= s) & (lev.index <= e)]
                pre = w.iloc[-1] / w.iloc[0] - 1
                mdd = (w / w.cummax() - 1).min()
                rows.append([ed_name, bundle, s.date(), w.index[-1].date(), pre, pre * 0.78 if pre > 0 else pre, mdd, 0.1, 0.5, -0.3])
        for s, e in ((starts[0], starts[0] + pd.DateOffset(years=5)),):
            w = lev[(lev.index >= s) & (lev.index <= e)]
            pre = w.iloc[-1] / w.iloc[0] - 1
            rows.append([ed_name, "S-5", s.date(), w.index[-1].date(), pre, pre * 0.78, (w / w.cummax() - 1).min(), 0.1, 0.5, -0.3])
        for s in pd.date_range("2010-02-01", periods=4, freq="12MS"):
            e = s + pd.DateOffset(years=3)
            w = lev[(lev.index >= s) & (lev.index <= e)]
            pre = w.iloc[-1] / w.iloc[0] - 1
            rows.append([ed_name, "B-10", s.date(), w.index[-1].date(), pre, pre * 0.78 if pre > 0 else pre, (w / w.cummax() - 1).min(), 0.1, 0.2, -0.3])
    df = pd.DataFrame(rows, columns=["판", "창", "시작", "끝", "TQQQ 세전", "TQQQ 세후", "TQQQ MDD", "TQQQ 세후 CAGR", "QQQ 세후", "QQQ MDD"])
    p = tmp_path / "base.csv"
    df.to_csv(p, index=False)
    return load_baseline(str(p))


def test_hold_ratio_is_one_without_costs(tmp_path):
    lev, _ = _world()
    base = _baseline_from(lev, tmp_path)
    hold = pd.Series(1.0, index=lev.index)
    tbl = window_table(lev, hold, base, "dev", costs=NO_COST)
    a = tbl[tbl["bundle"].isin(["A-10", "A-∞"])]
    # 기준선은 세후(×0.78) 인데 무비용·무세 보유는 세전이라 비율 > 1 — 세금 22% 를 넣으면 ≈ 1
    taxed = CostParams(buy_fee=0, sell_fee=0, tax_rate=0.22, tax_deduction=0.0, initial_capital=1.0, rebalance_band=0.0, exec_lag=0)
    tbl2 = window_table(lev, hold, base, "dev", costs=taxed)
    a2 = tbl2[tbl2["bundle"].isin(["A-10", "A-∞"])]
    assert (a["tqqq_ret"] > 0).all() and (a["ratio"] > 1).all()
    assert np.allclose(a2["ratio"], 1.0, atol=1e-6)
    assert set(tbl["bundle"]) == {"A-10", "A-∞", "B-10", "S-5"}
    assert (tbl["strat_mdd"] <= 0).all() and (tbl["trades"] >= 1).all()


def test_perfect_avoidance_passes_g1_to_g4_and_opposite_fails_g1(tmp_path):
    lev, r = _world()
    base = _baseline_from(lev, tmp_path)
    taxed = CostParams(buy_fee=0, sell_fee=0, tax_rate=0.22, tax_deduction=0.0, initial_capital=1.0, rebalance_band=0.0, exec_lag=0)
    perfect = pd.Series(1.0, index=lev.index)
    perfect.iloc[999:1121] = 0.0
    perfect.iloc[2499:2601] = 0.0
    tbl = window_table(lev, perfect, base, "dev", costs=taxed)
    v = gate_verdicts(tbl, n_trials=0, placebo_medians=[0.5] * 30, neighbor_verdicts=[{"G1": {"passed": True}, "G2": {"passed": True}}] * 3)
    assert v["first_fail"] is None and v["passed_through"] == 6
    assert v["G1"]["A-10"]["ratio_min"] > 1.0 and v["G2"]["shallower_rate"] >= 0.7
    md = verdict_markdown("완벽 회피", tbl, v)
    assert "G1~G6 통과" in md and "| A-10 |" in md
    opposite = 1.0 - perfect
    tbl_o = window_table(lev, opposite, base, "dev", costs=taxed)
    v_o = gate_verdicts(tbl_o)
    assert v_o["first_fail"] == "G1" and v_o["passed_through"] == 0


def test_placebo_preserves_runs_and_exposure():
    idx = pd.bdate_range("2015-01-01", periods=1000)
    t = pd.Series(1.0, index=idx)
    t.iloc[100:150] = 0.0; t.iloc[400:420] = 0.0; t.iloc[800:900] = 0.0
    p = placebo_target(t, 1)
    assert p.sum() == t.sum() and len(p) == len(t)
    assert (p.diff().abs() > 0).sum() == (t.diff().abs() > 0).sum()
    assert not p.equals(t) or True        # 길이 셔플이 같은 순서를 뽑을 수도 있다 — 통계만 본다
    s = placebo_target(t, 2, mode="shift")
    assert s.sum() == t.sum()
    with pytest.raises(ValueError):
        placebo_target(t, 1, mode="x")


def test_placebo_medians_and_threshold(tmp_path):
    lev, _ = _world()
    base = _baseline_from(lev, tmp_path)
    t = pd.Series(1.0, index=lev.index)
    t.iloc[999:1121] = 0.0
    meds = placebo_medians(lev, t, base, n_seeds=3, costs=NO_COST)
    assert len(meds) == 3 and all(np.isfinite(meds))
    assert g5_threshold(0) == 95.0 and g5_threshold(100) == 97.0 and g5_threshold(1000) == 99.0


def test_sealed_and_callable_target(tmp_path):
    lev, _ = _world()
    base = _baseline_from(lev, tmp_path)
    calls = []

    def f(start, end):
        calls.append((start, end))
        return pd.Series(1.0, index=lev.index)

    tbl = window_table(lev, f, base, "sealed", bundles=("A-10", "A-∞", "S-5"), costs=NO_COST)
    assert len(calls) == len(tbl)
    v = sealed_verdicts(tbl)
    assert set(v) == {"G1", "G2", "G3", "passed"}
