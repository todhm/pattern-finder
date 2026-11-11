"""H43 패키지 운용 판정 — 사다리 추적이 엔진과 같고, 프레임·판정·이력이 일관된지."""

from __future__ import annotations

import numpy as np
import pandas as pd

from strategy.adapters.ladder_reentry import ladder_switch
from strategy.adapters.regime_slots import as_frame, regime_labels, stitch
from strategy.adapters.tqqq_package import PACKAGE_PARAMS, change_table, decision, ladder_trace, package_frame, run_pair
from strategy.adapters.whipsaw_switch import whipsaw_switch


def _prices(seed=7, n=3000):
    rng = np.random.default_rng(seed)
    r = rng.normal(0.0004, 0.013, n); r[800:1000] = -0.012; r[1900:2300] = -0.006; r[2300:2400] = 0.012
    q = pd.Series(100 * np.exp(np.cumsum(r)), index=pd.bdate_range("2008-01-02", periods=n))
    t = 100 * (1 + 3 * q.pct_change().fillna(0)).cumprod()
    return q, t


def test_trace_matches_ladder_switch_and_stitch():
    q, t = _prices(); p = PACKAGE_PARAMS
    tr = ladder_trace(q, t, p)
    assert np.allclose(tr["state"].to_numpy(), ladder_switch(q, t, p.ladder).to_numpy())
    f = package_frame(q, t, p)
    lab = regime_labels(q, p.regime)
    lad = as_frame(ladder_switch(q, t, p.ladder), q.index); msw = as_frame(whipsaw_switch(q, p.monthly).reindex(q.index).ffill().fillna(1.0), q.index)
    expect = stitch({"A": lad, "B": lad, "D": lad, "C": msw}, lab, q.index)["TQQQ"]
    assert np.allclose(f["target"].to_numpy(), expect.to_numpy())
    # 사다리 칸·이탈 단계는 0~3 / 0~2, 이벤트가 있는 날은 비중이 바뀐다(월말 모드 밖에서)
    assert f["rung_i"].between(0, 3).all() and f["step_i"].between(0, 2).all()
    dw = f["ladder_w"].diff().fillna(0).abs()
    ev = (f["ladder_event"] != "")
    assert ev.sum() > 0 and (dw[ev] > 1e-9).all()
    assert set(f.loc[f["target"].diff().abs() > 1e-9, "reason"]) - {""} and (f.loc[f["target"].diff().abs() <= 1e-9, "reason"] == "").all()


def test_decision_and_change_table_consistent():
    q, t = _prices()
    d = decision(q, t)
    f = package_frame(q, t)
    assert d["target"] == float(f["target"].iloc[-1]) and d["as_of"] == f.index[-1].date()
    assert d["action"] and len(d["triggers"]) >= 1
    assert d["half_level"] > d["full_level"] > d["depth_level"]
    ct = change_table(f)
    assert len(ct) == int((f["target"].diff().fillna(0).abs() > 1e-9).sum()) and (ct["이유"] != "").all()
    # 가정 종가: QQQ 를 −30% 로 두면 보유 중이라면 매도, 모드가 어떻든 목표 비중은 1 아래
    h = decision(q, t, hypothetical_close=float(q.iloc[-1] * 0.7))
    assert h["hypothetical"] and h["target"] <= 1.0 and h["close"] == float(q.iloc[-1] * 0.7)


def test_run_pair_shapes():
    q, t = _prices(); f = package_frame(q, t)
    hold, strat = run_pair(t, f["target"], q.index[300], None)
    assert len(hold.equity) == len(strat.equity) and strat.trades >= 1 and hold.trades <= 2


def test_crash_switch_page_renders_package(monkeypatch):
    from pathlib import Path

    from streamlit.testing.v1 import AppTest

    import data.adapters.composed_market_data as cmd
    from tests.test_leveraged_etf_page import FakeMarketData

    monkeypatch.setattr(cmd, "build_default_market_data", lambda **kw: FakeMarketData())
    at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "pages" / "52_TQQQ_Crash_Switch.py"), default_timeout=600)
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    assert any("오늘의 행동" in h.value for h in at.subheader)
    assert any("비중 변경 이력" in h.value for h in at.subheader)
    banners = [b.value for b in list(at.success) + list(at.error) + list(at.warning)]
    assert any("목표 비중" in b for b in banners)
    assert any("사다리" in m.label for m in at.metric)
