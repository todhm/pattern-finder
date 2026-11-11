"""breadth_switch — 폭 계산(분모·무효), 이탈/재진입 히스테리시스, off_fraction."""

from __future__ import annotations

import numpy as np
import pandas as pd

from strategy.adapters.breadth_switch import BreadthParams, breadth_series, breadth_switch


def test_breadth_series_denominator_and_min_names():
    idx = pd.bdate_range("2020-01-01", periods=30)
    closes = pd.DataFrame({f"s{i}": np.linspace(1, 2, 30) * (1 if i < 3 else -1) + 10 for i in range(5)}, index=idx)
    closes.iloc[:, 4] = np.nan                              # 데이터 없는 종목은 분모 제외
    b = breadth_series(closes, sma=5, min_names=3)
    assert b.iloc[:4].isna().all()                          # SMA 워밍업
    assert abs(b.iloc[-1] - 3 / 4) < 1e-9                   # 상승 3 / 유효 4
    assert breadth_series(closes, sma=5, min_names=5).isna().all()


def test_switch_hysteresis_and_off_fraction():
    idx = pd.bdate_range("2020-01-01", periods=200)
    px = pd.Series(np.r_[np.linspace(100, 120, 100), np.linspace(120, 90, 50), np.linspace(90, 130, 50)], index=idx)
    b = pd.Series(0.8, index=idx); b.iloc[110:140] = 0.2; b.iloc[140:160] = 0.55; b.iloc[160:] = 0.7
    p = BreadthParams(exit_pct=0.4, reenter_pct=0.6, price_sma=20)
    s = breadth_switch(px, b, p)
    assert s.iloc[:110].min() == 1.0
    assert s.iloc[139] == 0.0 and s.iloc[150] == 0.0            # 0.55 는 재진입 아님(히스테리시스)
    assert s.iloc[165] == 1.0
    s2 = breadth_switch(px, b, BreadthParams(exit_pct=0.4, reenter_pct=0.6, price_sma=20, off_fraction=0.33))
    assert s2.min() == 0.33 and s2.max() == 1.0
    b_nan = pd.Series(np.nan, index=idx)
    assert breadth_switch(px, b_nan, p).min() == 1.0            # 폭 무효면 보유
