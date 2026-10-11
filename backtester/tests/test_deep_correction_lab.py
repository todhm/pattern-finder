"""deep_correction_lab — 롤링 연구·회피 구간·격자·현재 상태."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from strategy.adapters.deep_correction_lab import (
    DeepParams,
    avoided_segments,
    current_state,
    param_grid,
    rolling_study,
    rolling_summary,
    run_pair,
    signal,
)
from strategy.adapters.leveraged_etf_lab import CostParams

NO_COST = CostParams(buy_fee=0, sell_fee=0, tax_rate=0, initial_capital=1_000.0)


def _world():
    rng = np.random.default_rng(7)
    n = 252 * 12
    r = rng.normal(0.0004, 0.012, n)
    r[1500:1620] -= 0.012                   # 깊은 하락 구간
    sig = pd.Series(100 * np.exp(np.cumsum(r)), index=pd.bdate_range("2012-01-02", periods=n))
    lev = 100 * (1 + 3 * sig.pct_change().fillna(0)).cumprod()
    return sig, lev


def test_run_pair_and_avoided_segments():
    sig, lev = _world()
    s = signal(sig, DeepParams(dd=0.15))
    hold, strat = run_pair(lev, s, date(2013, 1, 2), None, NO_COST)
    assert hold.equity.index[0] == strat.equity.index[0] and strat.exposure.min() == 0.0
    seg = avoided_segments(strat, lev, sig)
    assert len(seg) >= 1 and {"이탈", "재진입", "회피 구간 ETF"} <= set(seg.columns)
    assert (seg["일수"] > 0).all()


def test_rolling_study_shapes_and_summary():
    sig, lev = _world()
    s = signal(sig, DeepParams(dd=0.15))
    df = rolling_study(lev, s, years=3, costs=NO_COST, step_months=6)
    assert len(df) >= 10 and (df["end"] > df["start"]).all()
    assert np.allclose(df["ratio"], df["strat_mult"] / df["hold_mult"])
    sm = rolling_summary(df)
    assert 0 <= sm["win_rate"] <= 1 and sm["ratio_min"] <= sm["ratio_median"] <= sm["ratio_max"]
    assert sm["windows"] == len(df)


def test_param_grid_and_current_state():
    sig, lev = _world()
    g = param_grid(lev, sig, date(2013, 1, 2), NO_COST, dds=(0.10, 0.20), reenters=(20,))
    assert len(g) == 2 and {"dd", "reenter", "vs_hold", "mdd"} <= set(g.columns)
    st = current_state(sig, DeepParams())
    assert st["exit_level"] < st["close"] * 1.0 + 1e9 and isinstance(st["holding"], bool)
    assert abs(st["drawdown"]) <= 1.0


def test_decision_hold_sell_and_reenter_paths():
    from strategy.adapters.deep_correction_lab import decision
    up = pd.Series(np.linspace(100, 200, 300), index=pd.bdate_range("2023-01-02", periods=300))
    d = decision(up, DeepParams())
    assert d["action"] == "보유" and d["prev_holding"] and not d["cond_deep_dd"] and d["sell_trigger"] < d["close"]
    crash = pd.concat([up, pd.Series(np.linspace(198, 150, 40), index=pd.bdate_range(up.index[-1] + pd.offsets.BDay(1), periods=40))])
    d2 = decision(crash, DeepParams(dd=0.15))
    assert d2["holding_after"] is False and d2["action"] in ("매도", "현금 유지") and d2["cond_below_sma"] and d2["cond_deep_dd"]
    # 가정 종가: 현금 상태에서 20일선 위로 뛰면 재진입
    still_deep = decision(crash, DeepParams(dd=0.15), hypothetical_close=float(crash.rolling(20).mean().iloc[-1] * 1.02))
    assert still_deep["action"] == "현금 유지" and still_deep["cond_above_reenter"]      # 20일선 위지만 아직 200일선 아래·낙폭 안
    d3 = decision(crash, DeepParams(dd=0.15), hypothetical_close=float(d2["buy_trigger"] * 1.01))
    assert d3["hypothetical"] and d3["action"] == "재진입 매수" and d3["holding_after"] is True
    # 보유 중 가정 종가가 이탈선·200일선 아래면 매도
    d4 = decision(up, DeepParams(dd=0.15), hypothetical_close=float(up.iloc[-1] * 0.5))
    assert d4["action"] == "매도" and d4["holding_after"] is False


def test_monthly_sma_signal_decision_and_grid():
    from strategy.adapters.deep_correction_lab import MonthlySmaParams, decision_monthly, param_grid_monthly, signal_monthly_sma
    sig, lev = _world()
    p = MonthlySmaParams(n=100, off_fraction=0.33)
    s = signal_monthly_sma(sig, p)
    assert set(np.round(s.unique(), 2)) <= {0.0, 0.33, 1.0}
    # 한 달 안에서는 값이 바뀌지 않는다 (월말에만 판정)
    per_month = s.groupby(s.index.to_period("M")).nunique()
    assert (per_month[1:] <= 2).all()
    d = decision_monthly(sig, p)
    assert d["action"] in ("보유", "잔여 33% 유지") and d["next_decision"] >= d["as_of"] and isinstance(d["would_change"], bool)
    g = param_grid_monthly(lev, sig, date(2013, 1, 2), NO_COST, ns=(50, 100), offs=(0.0, 0.33))
    assert len(g) == 4 and {"n", "off", "vs_hold"} <= set(g.columns)
