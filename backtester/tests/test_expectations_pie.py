"""예측투자 역산: DCF 왕복, 단조성, 기저율 꼬리, 표 합계."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from strategy.adapters import expectations_pie as ep


def test_implied_growth_round_trip_and_monotone():
    p = ep.PieParams(wacc=0.08)
    for g in (-0.05, 0.0, 0.07, 0.25, 0.6):
        ev = float(ep.dcf_value(1000.0, g, 0.25, 0.2, p)[0])
        assert ep.implied_growth(ev, 1000.0, 0.25, 0.2, p) == pytest.approx(g, abs=2e-4)
    cheap, dear = ep.implied_growth(3000, 1000, 0.25, 0.2, p), ep.implied_growth(30000, 1000, 0.25, 0.2, p)
    assert dear > cheap
    assert ep.implied_growth(30000, 1000, 0.25, 0.2, ep.PieParams(wacc=0.10)) > dear          # 자본비용이 높으면 같은 주가가 더 큰 성장을 요구
    assert ep.implied_growth(1e12, 1000, 0.25, 0.2, p) == float("inf") and ep.implied_growth(1.0, 1000, 0.25, 0.2, p) == float("-inf")
    assert np.isnan(ep.implied_growth(5000, 1000, -0.1, 0.2, p))


def test_zero_growth_value_is_perpetuity():
    p = ep.PieParams(wacc=0.08, inflation=0.0)
    assert float(ep.dcf_value(100.0, 0.0, 0.2, 0.3, p)[0]) == pytest.approx(100 * 0.2 * 0.79 / 0.08, rel=1e-9)


def test_base_rate_tables_and_tail():
    for _, name, dist in ep.BASE_RATE_10Y:
        assert len(dist) == 16 and sum(dist) == pytest.approx(89.2 if name == "$325~700M" else 100, abs=0.6)      # 원문 표의 그 열은 합이 89.2 로 인쇄돼 있다
    assert ep.base_rate_tail(0.20, 100) == pytest.approx(0.181, abs=0.002)                    # 최소 구간: 20% 넘게 10년 = 18.1%
    assert ep.base_rate_tail(0.20, 30000) == pytest.approx(0.002, abs=0.0005)                 # $25B 초과: 0.2%
    assert ep.base_rate_tail(0.225, 100) == pytest.approx(0.181 - 0.031, abs=0.002)           # 칸 안 직선 보간
    assert ep.base_rate_tail(-0.5, 5000) == 1.0 and ep.base_rate_tail(float("inf"), 5000) == 0.0
    assert ep.base_rate_tail(0.05, 5000) > ep.base_rate_tail(0.10, 5000) > ep.base_rate_tail(0.30, 5000)
    assert ep.size_bucket(324) == 0 and ep.size_bucket(325) == 1 and ep.size_bucket(1e6) == 9


def test_pie_table_flags_expensive():
    f = pd.DataFrame({"rev_ttm": [5e9, 5e9, 5e9], "op_margin": [0.25, 0.25, 0.01], "invest_rate": [0.2, 0.2, np.nan], "net_debt": [0.0, 0.0, 0.0],
                      "shares": [1e9, 1e9, 1e9], "sbc_to_rev": [0.05] * 3, "sector": ["TECHNOLOGY"] * 3}, index=["CHEAP", "DEAR", "LOWM"])
    t = ep.pie_table(f, pd.Series({"CHEAP": 20e9, "DEAR": 400e9, "LOWM": 20e9}), 2021, ep.PieParams(wacc=0.08))
    assert not t.loc["CHEAP", "excluded"] and t.loc["DEAR", "excluded"]
    assert t.loc["DEAR", "g_implied"] > t.loc["CHEAP", "g_implied"] and t.loc["DEAR", "base_rate_p"] < 0.05
    assert t.loc["LOWM", "margin_from_sector"] and t.loc["LOWM", "margin_used"] == pytest.approx(0.25)
    assert t.loc["CHEAP", "sales_2015_musd"] == pytest.approx(5000 / 1.17)
