from __future__ import annotations

import numpy as np
import pandas as pd

from strategy.adapters.intraday_rules import IntradayParams, after_tax_curve, daily_multiplier, placebo_multiplier, run_rule


def _bars(n_days=60, seed=2):
    rng = np.random.default_rng(seed); rows = []
    px = 100.0
    for d in pd.bdate_range("2021-01-04", periods=n_days):
        for t in pd.date_range(d + pd.Timedelta(hours=9, minutes=30), d + pd.Timedelta(hours=15, minutes=45), freq="15min"):
            o = px; c = px * (1 + rng.normal(0, 0.004)); h = max(o, c) * 1.001; l = min(o, c) * 0.999
            rows.append({"t": t, "Open": o, "High": h, "Low": l, "Close": c, "Volume": 1000.0}); px = c
    df = pd.DataFrame(rows).set_index("t"); df.index = df.index.tz_localize("America/New_York")
    return df


def test_hold_equivalence_and_rules():
    bars = _bars()
    adj = run_rule(bars, IntradayParams("stop", 9.99))["hold"].cumprod() * 100        # 수정 종가 = 장중 종가 경로
    for rule in ("stop", "vwap", "orb", "overnight", "intraday", "stop_vwap", "gap", "firsthour"):
        df = run_rule(bars, IntradayParams(rule, 0.03))
        m = daily_multiplier(df, adj)
        assert len(m) == 60 and (m > 0).all()
    never = daily_multiplier(run_rule(bars, IntradayParams("stop", 9.99)), adj)
    assert np.allclose(never.cumprod().iloc[-1], adj.iloc[-1] / adj.iloc[0])              # 이탈 없는 손절 = 보유
    on = run_rule(bars, IntradayParams("overnight")); intra = run_rule(bars, IntradayParams("intraday"))
    assert np.allclose((on["mult"] * intra["mult"]).iloc[1:] / (1 - 0.0015) ** 4, on["hold"].iloc[1:])   # 야간×장중 = 하루
    eq, final = after_tax_curve(pd.Series([1.1] * 10 + [0.95] * 5, index=pd.bdate_range("2021-12-20", periods=15)))
    assert final < 1.1 ** 10 * 0.95 ** 5 and final > 0
    pm = placebo_multiplier(run_rule(bars, IntradayParams("stop", 0.03)), adj, 1, "exit")
    assert len(pm) == 60


def test_gap_rule_never_fires_on_small_gaps():
    bars = _bars()
    adj = run_rule(bars, IntradayParams("stop", 9.99))["hold"].cumprod() * 100
    g = run_rule(bars, IntradayParams("gap", 0.5)); assert not g["exited"].any()
    m = daily_multiplier(g, adj); assert np.allclose(m.cumprod().iloc[-1], adj.iloc[-1] / adj.iloc[0])
