"""minervini_full_checklist — 단계 규칙·가이던스 판정·진입/청산 시뮬레이션."""

from datetime import date

import numpy as np
import pandas as pd

from strategy.adapters.minervini_full_checklist import (
    Entry,
    _stage,
    guidance_cut,
    simulate_trade,
)


def test_stage_rule_guard_all_core_min():
    items = {"g": True, "a": True, "b": True, "c": False}
    assert _stage(items, guard={"g"}, bonus=set(), min_core=2, notes={}).ok
    assert not _stage(items, guard={"g"}, bonus=set(), min_core=None, notes={}).ok   # core 전부 필요
    assert not _stage({**items, "g": False}, guard={"g"}, bonus=set(), min_core=1, notes={}).ok
    # 측정 불가(None) guard 는 통과, core 는 미통과로 센다
    assert _stage({"g": None, "a": True}, guard={"g"}, bonus=set(), min_core=1, notes={}).ok
    assert not _stage({"g": True, "a": None}, guard={"g"}, bonus=set(), min_core=1, notes={}).ok


def test_stage_rule_handles_numpy_bools():
    items = {"g": np.bool_(False), "a": np.bool_(True)}
    assert not _stage(items, guard={"g"}, bonus=set(), min_core=1, notes={}).ok
    r = _stage({"g": np.bool_(True), "a": np.bool_(True)}, guard={"g"}, bonus=set(), min_core=1, notes={})
    assert r.ok and r.items["a"] is True


def test_guidance_cut_detects_downgrades_only():
    assert guidance_cut(["Acme lowers full-year guidance amid weak demand"])
    assert guidance_cut(["Acme cuts 2024 outlook"])
    assert guidance_cut(["Acme guidance cut sends shares lower"])
    assert not guidance_cut(["Acme raises outlook after record quarter"])
    assert not guidance_cut(["Acme reaffirms guidance", "Acme beats estimates"])
    assert not guidance_cut([])


def _bars(highs, lows, closes):
    idx = pd.bdate_range("2024-01-01", periods=len(closes))
    return pd.DataFrame({"Open": closes, "High": highs, "Low": lows, "Close": closes,
                         "Volume": np.full(len(closes), 1e6)}, index=idx)


def _entry(d) -> Entry:
    return Entry(setup_date=d, entry_date=d, entry_price=100.0, pivot=100.0,
                 breakout_volume_ratio=2.0, stage4=None)  # type: ignore[arg-type]


def test_simulate_trade_stop_target_and_open():
    df = _bars([101, 105, 123], [99, 100, 110], [100, 104, 121])
    r = simulate_trade(df, _entry(date(2024, 1, 1)))
    assert r.exit_reason == "target" and r.pnl == 0.22 and r.days_held == 2
    df = _bars([101, 102, 103], [99, 91, 95], [100, 92, 96])
    r = simulate_trade(df, _entry(date(2024, 1, 1)))
    assert r.exit_reason == "stop" and r.pnl == -0.08
    df = _bars([101, 102], [99, 98], [100, 105])
    r = simulate_trade(df, _entry(date(2024, 1, 1)))
    assert r.exit_reason == "open" and abs(r.pnl - 0.05) < 1e-9
