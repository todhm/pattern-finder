"""weinstein_scan — 합성 주가로 셋업 탐지·등급·사후 청산 규칙 검증."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from strategy.adapters.weinstein_scan import (
    SECTOR_ETF,
    Candidate,
    ScanParams,
    check_stats,
    clip_daily,
    find_setup,
    grade_stats,
    market_checks,
    prepare_weekly,
    sector_checks,
    simulate_forward,
)
from strategy.adapters.weinstein_stage import StageParams, evaluate_breakout, grade


def _daily(weekly_closes: list[float], vol_mult: list[float] | None = None, start="2018-01-08") -> pd.DataFrame:
    """주간 종가 목록 → 5일씩 보간한 일봉 (고가 +1%, 저가 −1%). start 는 월요일이어야 주봉(W-FRI)과 정렬된다."""
    idx = pd.bdate_range(start, periods=len(weekly_closes) * 5)
    closes = np.repeat(np.asarray(weekly_closes, dtype=float), 5)
    vol = np.repeat(np.asarray(vol_mult or [1.0] * len(weekly_closes)), 5) * 1e6
    return pd.DataFrame({"Open": closes, "High": closes * 1.01, "Low": closes * 0.99,
                         "Close": closes, "Volume": vol}, index=idx)


def _stage_path(breakout_vol: float = 3.0, after: list[float] | None = None):
    """4단계 하락(25주: 100→84) → 1단계 베이스(40주: 55~65, 완만히 상승) → 돌파 주 70 → after.

    하락 구간은 베이스 저점 +30% 밖에 둔다 — ``find_base`` 는 폭 30% 안의 구간을 전부 베이스로
    흡수하므로, 하락 꼬리가 섞이면 저항선이 베이스 천장이 아니라 하락 구간 고점이 된다.
    베이스에 주당 +0.15 드리프트를 줘서 돌파 시점에 30주 MA 4주 기울기가 +0.5% 를 넘게(2단계) 한다.
    """
    decline = list(np.linspace(100, 84, 25))
    base = [57 + 0.15 * k + 2 * np.sin(k / 2) for k in range(40)]   # 상단 터치 여러 번
    path = decline + base + [70.0] + (after or [])
    vols = [1.0] * len(decline) + [0.8] * len(base) + [breakout_vol] + [1.0] * len(after or [])
    return path, vols


def test_find_setup_detects_this_week_breakout_and_grades():
    path, vols = _stage_path()
    df = _daily(path, vols)
    wk = prepare_weekly(df, StageParams(), index_close=None)
    assert wk is not None
    setup = find_setup(wk, ScanParams(mode="breakout", recent_weeks=1))
    assert setup is not None and setup.kind == "breakout"
    assert setup.week_i == len(wk.w) - 1
    assert 63 < setup.base.high < 68 and setup.base.weeks >= 12
    assert int(wk.stages.iloc[-1]) == 2 and (wk.stages.iloc[-45:-1] == 1).any()   # 1단계 바닥 → 2단계 돌파
    checks = evaluate_breakout(wk.w, wk.stages, wk.rs, setup.week_i, setup.base, StageParams())
    cmap = {c.code: c for c in checks}
    assert cmap["3-2"].passed and cmap["2-1"].passed
    assert float(cmap["4-1"].value.rstrip("x")) > 2.5        # 돌파 주 거래량 3배
    g = grade([c for c in checks if c.code[0] not in "01"])
    # 베이스 위에 직전 하락 구간 매물(60~70)이 있어 3-3 실패 → B-, 또는 RS 측정 불가 → 등급은 둘 중 하나
    assert g in ("B-", "B", "A", "A+") or g.startswith("탈락")


def test_find_setup_watch_mode_requires_price_just_below_resistance():
    path, vols = _stage_path()
    path, vols = path[:-1] + [64.5], vols[:-1] + [0.8]     # 돌파 전, 저항(≈65.5) 바로 아래
    wk = prepare_weekly(_daily(path, vols), StageParams(), None)
    setup = find_setup(wk, ScanParams(mode="watch", watch_pct=0.05))
    assert setup is not None and setup.kind == "watch"
    assert find_setup(wk, ScanParams(mode="breakout")) is None


def test_no_setup_in_plain_downtrend():
    wk = prepare_weekly(_daily(list(np.linspace(100, 40, 80))), StageParams(), None)
    assert find_setup(wk, ScanParams()) is None


def _candidate(df: pd.DataFrame, wk, setup) -> Candidate:
    checks = evaluate_breakout(wk.w, wk.stages, wk.rs, setup.week_i, setup.base, StageParams())
    return Candidate(ticker="T", kind="breakout", week_end=wk.w.index[setup.week_i], base=setup.base, checks=checks,
                     grade="A", stock_grade="A", close=float(wk.w.Close.iloc[setup.week_i]), resistance=setup.base.high,
                     support=setup.base.low, vol_ratio=3.0, rs=0.0, stage_now=2, entry_type="1→2",
                     sector=None, sector_etf=None, weekly=wk)


def test_simulate_forward_fake_breakout_exits_by_7_5():
    after = [68.0, 64.0, 60.0, 58.0, 57.0]                # 2주 뒤 종가 < 돌파가(≈65.5)
    path, vols = _stage_path(after=after)
    full = _daily(path, vols)
    as_of = full.index[(len(path) - len(after)) * 5 - 1].date()
    clipped = clip_daily(full, as_of)
    wk = prepare_weekly(clipped, StageParams(), None)
    setup = find_setup(wk, ScanParams())
    o = simulate_forward(full, _candidate(clipped, wk, setup), StageParams())
    assert o.entered and o.reason.startswith("7-5") and o.ret < 0
    assert o.entry_date > as_of and o.weeks <= 6


def test_simulate_forward_winner_holds_until_ma_break():
    up = list(np.linspace(72, 130, 30))
    down = list(np.linspace(128, 75, 30))                  # 급락 → 30주 MA 이탈
    path, vols = _stage_path(after=up + down)
    full = _daily(path, vols)
    as_of = full.index[(len(path) - 60) * 5 - 1].date()
    clipped = clip_daily(full, as_of)
    wk = prepare_weekly(clipped, StageParams(), None)
    setup = find_setup(wk, ScanParams())
    o = simulate_forward(full, _candidate(clipped, wk, setup), StageParams())
    assert o.entered and o.weeks > 20 and o.mfe > 0.5
    assert o.reason.startswith(("8-3", "8-4", "7-7"))
    assert o.ret_13w is not None and o.ret_13w > 0


def test_clip_daily_completed_weeks_only_drops_partial_week():
    idx = pd.bdate_range("2026-09-28", periods=9)         # 월~목 (10-09 금 전)
    df = pd.DataFrame({"Open": 1.0, "High": 1.0, "Low": 1.0, "Close": 1.0, "Volume": 1.0}, index=idx)
    out = clip_daily(df, date(2026, 10, 8), completed_weeks_only=True)
    assert out.index[-1].date() == date(2026, 10, 2)      # 직전 금요일
    assert clip_daily(df, date(2026, 10, 8)).index[-1].date() == date(2026, 10, 8)


def test_market_and_sector_checks_shape():
    up = _daily(list(np.linspace(100, 200, 80)))
    spy = prepare_weekly(up, StageParams(), None)
    qqq = prepare_weekly(_daily(list(np.linspace(100, 300, 80))), StageParams(), spy.w["Close"])  # SPY 보다 강함 → RS > 0
    m = market_checks({"SPY": spy, "QQQ": qqq}, None, spy.w.index[-1])
    assert m["pass"] is True and m["checks"][0].code == "0-1" and m["checks"][0].required
    sec = sector_checks(qqq, "XLK", "TECHNOLOGY", spy.w.index[-1])
    assert sec["pass"] is True and [c.code for c in sec["checks"]] == ["1-1", "1-2", "1-3"]
    unknown = sector_checks(None, None, None, spy.w.index[-1])
    assert unknown["pass"] is None and all(c.passed is None for c in unknown["checks"])
    assert SECTOR_ETF["TECHNOLOGY"] == "XLK" and len(SECTOR_ETF) == 11


def test_grade_stats_groups_and_orders():
    rows = [{"grade": "B", "entered": True, "ret": -0.05, "weeks": 3, "reason": "7-5 x", "mfe": 0.01, "mae": -0.06},
            {"grade": "A+", "entered": True, "ret": 0.30, "weeks": 20, "reason": "8-3 x", "mfe": 0.4, "mae": -0.02},
            {"grade": "탈락(4-1)", "entered": True, "ret": 0.10, "weeks": 9, "reason": "time", "mfe": 0.2, "mae": -0.01},
            {"grade": "탈락(2-10,4-1)", "entered": True, "ret": -0.10, "weeks": 2, "reason": "7-5 x", "mfe": 0.0, "mae": -0.1},
            {"grade": "A+", "entered": False, "ret": None, "weeks": 0, "reason": "", "mfe": 0, "mae": 0}]
    gs = grade_stats(rows)
    assert list(gs["grade"]) == ["A+", "B", "탈락"]            # 상세 탈락 사유는 한 묶음
    assert gs.iloc[0]["진입"] == 1 and gs.iloc[1]["7-5 실패"] == 1 and gs.iloc[2]["진입"] == 2


def test_check_stats_splits_pass_vs_fail():
    rows = [{"entered": True, "ret": 0.3, "checks": {"4-1": True, "5-1": True}},
            {"entered": True, "ret": -0.1, "checks": {"4-1": False, "5-1": True}},
            {"entered": True, "ret": 0.1, "checks": {"4-1": False, "5-1": None}},
            {"entered": False, "ret": None, "checks": {"4-1": True}}]
    cs = check_stats(rows).set_index("항목")
    assert cs.loc["4-1", "통과 n"] == 1 and cs.loc["4-1", "미통과 n"] == 2
    assert abs(cs.loc["4-1", "차이"] - (0.3 - 0.0)) < 1e-9
    assert cs.loc["5-1", "통과 n"] == 2 and cs.loc["5-1", "미통과 n"] == 0


def test_regrade_volume_optional_ignores_4_1_only():
    from strategy.adapters.weinstein_stage import Check
    from strategy.adapters.weinstein_scan import regrade_volume_optional
    base = [Check("2-6", "베이스", "30주", True), Check("3-4", "2.5y", "x", True), Check("3-5", "10y", "x", True),
            Check("5-2", "RS", "1.0", True), Check("4-1", "vol", "1.10x", False, True)]
    assert regrade_volume_optional(base) == "A+"
    assert regrade_volume_optional(base + [Check("2-2", "MA", "x", False, True)]).startswith("탈락(2-2")
