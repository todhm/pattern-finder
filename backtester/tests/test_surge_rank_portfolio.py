"""상위 K 교체 포트폴리오 — 미래 비참조·폐지 처리·비용·계단식 곡선."""

from __future__ import annotations

import numpy as np
import pandas as pd

from strategy.adapters.surge_rank_portfolio import RankRules, bench_table, curve_stats, daily_curve, run_rank, scores, stats, tranche_curve, wide


def _panel(n=520, m=30, seed=5, dead=("S03", "S07")):
    rng = np.random.default_rng(seed); idx = pd.bdate_range("2018-01-02", periods=n); rows = []
    for j in range(m):
        s = f"S{j:02d}"; c = 30 * np.exp(np.cumsum(rng.normal(0.0003 * (j - m / 2) / m, 0.02, n))); o = c * (1 + rng.normal(0, 0.003, n))
        d = pd.DataFrame({"date": idx, "symbol": s, "C": c, "O": o, "raw_c": c, "dv": c * 3e6})
        rows.append(d.iloc[:400] if s in dead else d)
    return pd.concat(rows, ignore_index=True)


def test_wide_fills_delisted_with_last_close():
    w = wide(_panel())
    assert not w["alive"]["S03"].iloc[450] and np.isclose(w["C_ff"]["S03"].iloc[450], w["C"]["S03"].iloc[399]) and np.isclose(w["O_ff"]["S03"].iloc[450], w["C"]["S03"].iloc[399])


def test_run_rank_no_lookahead_and_costs():
    p = _panel(); w = wide(p); rules = RankRules(rank="mom6", k=5, min_dv20=1e6, every=21)
    t = run_rank(w, rules, "2018-09-03", "2019-12-31")
    assert len(t) > 5 and (t["n"] == 5).all() and t["turnover"].iloc[0] == 1.0
    # 마지막 신호일 뒤 가격을 바꿔도 그 전 구간의 선택은 같다
    cut = t["date"].iloc[-2]; p2 = p.copy(); late = p2["date"] > cut; p2.loc[late, ["C", "O", "raw_c"]] *= 5.0
    t2 = run_rank(wide(p2), rules, "2018-09-03", "2019-12-31")
    assert (t["picks"].iloc[:-2] == t2["picks"].iloc[:-2]).all()
    free = run_rank(w, RankRules(rank="mom6", k=5, min_dv20=1e6, every=21, cost=0.0), "2018-09-03", "2019-12-31")
    assert (free["ret"] >= t["ret"] - 1e-12).all() and stats(free)["mult"] > stats(t)["mult"]
    # 폐지된 종목은 폐지 뒤 후보가 아니다
    assert not any("S03" in x.split(",") for x in t.loc[t["date"] > w["C"].index[405], "picks"])
    # 시장 필터가 꺼지면 현금
    off = run_rank(w, rules, "2018-09-03", "2019-12-31", market_ok=pd.Series(False, index=w["C"].index))
    assert (off["n"] == 0).all() and np.allclose(off["ret"], 0.0)
    assert scores(w, "near_high").max().max() <= 1e-12


def test_curves_match_tables():
    w = wide(_panel()); idx = w["C"].index[(w["C"].index >= "2018-09-03")]
    rules = RankRules(rank="mom6", k=5, min_dv20=1e6, every=21)
    ts = [run_rank(w, rules, str(idx[o].date()), "2019-12-31") for o in (0, 7, 14)]
    eq = daily_curve(w, ts[0], idx)
    assert np.isclose(eq.loc[ts[0]["exit"].iloc[-1]], (1 + ts[0]["ret"]).prod()) and eq.iloc[0] == 1.0
    tr = tranche_curve(w, ts, idx)
    assert np.isclose(tr.iloc[-1], np.mean([daily_curve(w, t, idx).iloc[-1] for t in ts])) and -1 < curve_stats(tr)["mdd"] <= 0
    b = bench_table(w["O"]["S00"], ts[0])
    assert np.isclose(b["ret"].iloc[0], w["O"]["S00"].loc[ts[0]["exit"].iloc[0]] / w["O"]["S00"].loc[ts[0]["date"].iloc[0]] - 1)
