"""lifecycle_trade — 합성 주가로 단계·패턴 분류, 매수 지점, 매도 규칙, 6장 통계 검증."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from data.domain.models import IpoEvent
from strategy.adapters.lifecycle_trade import (
    LifecycleParams,
    analyze_ticker,
    classify_lifecycle,
    cohort_tables,
    find_bases,
    find_buy_points,
    ipo_cohort,
    ipo_stats,
    prepare,
    rule_summary,
    screen_ticker,
    simulate_rule,
    trades_frame,
)

P = LifecycleParams()
START = "2019-01-07"   # 월요일 — 주봉(W-FRI) 정렬


def _daily(weekly: list[float], start: str = START, vol: float = 2e6, wobble: float = 0.01) -> pd.DataFrame:
    """주간 종가 → 5일 일봉 (주 안에서는 그 주 종가로 고정 — 주봉 고/저가 다음 주 움직임을 끌어오지 않게), 고/저는 ±wobble."""
    c = np.repeat(np.asarray(weekly, dtype=float), 5)
    idx = pd.bdate_range(start, periods=len(c))
    return pd.DataFrame({"Open": c, "High": c * (1 + wobble), "Low": c * (1 - wobble), "Close": c,
                         "Volume": np.full(len(c), vol)}, index=idx)


def _flat(level: float, weeks: int, amp: float = 0.03) -> list[float]:
    return [level * (1 + amp * np.sin(k)) for k in range(weeks)]


def _ramp(a: float, b: float, weeks: int) -> list[float]:
    return list(np.linspace(a, b, weeks + 1)[1:])


def test_find_bases_detects_flat_box_and_breakout():
    path = _flat(100, 6) + [112, 115, 118]
    fr = prepare(_daily(path), date(2019, 1, 7))
    bases = find_bases(fr.w, 0, len(fr.w) - 1, P)
    assert bases and bases[0].start_i == 0 and bases[0].weeks >= 4
    assert bases[0].breakout_i is not None and 100 < bases[0].high < 106


def test_stair_stepper_has_ipo_base_buy_point_and_no_failure():
    path = _flat(100, 5) + _ramp(100, 130, 6) + _flat(130, 8) + _ramp(130, 170, 6) + _flat(170, 8) + _ramp(170, 210, 6)
    fr = prepare(_daily(path), date(2019, 1, 7))
    lc = classify_lifecycle(fr.w, P)
    assert lc.ipo_base is not None and lc.ap_exists and lc.af_i is None
    assert lc.pattern == "Stair Stepper"
    kinds = {b.kind for b in find_buy_points(fr, lc, P)}
    assert "IPO base" in kinds


def test_rocket_ship_doubles_fast_without_undercut():
    path = _flat(100, 4) + _ramp(100, 260, 9) + _flat(250, 10)
    fr = prepare(_daily(path), date(2019, 1, 7))
    lc = classify_lifecycle(fr.w, P)
    assert lc.pattern == "Rocket Ship" and lc.ap_gain >= 1.0


def test_one_hit_wonder_then_plus_after_mature_base():
    path = _flat(100, 4) + _ramp(100, 220, 8) + _ramp(220, 70, 10) + _flat(70, 30)
    fr = prepare(_daily(path), date(2019, 1, 7))
    lc = classify_lifecycle(fr.w, P)
    assert lc.ap_exists and lc.af_i is not None and lc.pattern == "One-Hit Wonder"
    assert lc.phase_now == "I-DDP"
    # 성숙 베이스 돌파 → Plus, 매수 지점 'first mature base'
    path2 = path + _ramp(70, 95, 6) + _flat(95, 10) + _ramp(95, 130, 4)
    fr2 = prepare(_daily(path2), date(2019, 1, 7))
    lc2 = classify_lifecycle(fr2.w, P)
    assert lc2.pattern == "One-Hit Wonder Plus" and lc2.iap_start_i is not None
    assert any(b.kind == "first mature base" for b in find_buy_points(fr2, lc2, P))


def test_pump_and_dump_and_disappointment():
    tank = _ramp(100, 55, 8) + _flat(55, 50)
    fr = prepare(_daily(tank), date(2019, 1, 7))
    lc = classify_lifecycle(fr.w, P)
    assert not lc.ap_exists and lc.pattern == "Disappointment"
    recover = tank + _ramp(55, 62, 4) + _flat(62, 10) + _ramp(62, 80, 4) + _flat(80, 4)
    fr2 = prepare(_daily(recover), date(2019, 1, 7))
    lc2 = classify_lifecycle(fr2.w, P)
    assert lc2.pattern == "Pump and Dump" and lc2.phase_now == "I-AP"


def test_late_bloomer_requires_long_iddp():
    path = _flat(100, 4) + _ramp(100, 140, 6) + _ramp(140, 80, 8) + _flat(80, 45) + _ramp(80, 100, 6) + _flat(100, 10) + _ramp(100, 130, 4)
    fr = prepare(_daily(path), date(2019, 1, 7))
    lc = classify_lifecycle(fr.w, P)
    assert lc.pattern == "Late Bloomer" and lc.iap_start_i is not None and lc.iap_start_i - lc.af_i >= 39


def test_stop_loss_exits_next_open():
    path = _flat(100, 4) + [112, 85, 80, 78]   # 돌파 다음 주 급락 → 금요일 전에 −10%
    fr = prepare(_daily(path), date(2019, 1, 7))
    lc = classify_lifecycle(fr.w, P)
    bp = find_buy_points(fr, lc, P)[0]
    t = simulate_rule(fr, bp.signal_date, "40-Week", bp.kind, P)
    assert t is not None and t.reason.startswith("손절") and t.pct_gain < -0.08 and not t.open_position


def test_forty_week_rule_holds_trend_then_exits_below_sma40():
    path = _flat(100, 4) + _ramp(100, 300, 60) + _ramp(300, 150, 14)
    fr = prepare(_daily(path), date(2019, 1, 7))
    lc = classify_lifecycle(fr.w, P)
    bp = next(b for b in find_buy_points(fr, lc, P) if b.kind == "IPO base")
    t = simulate_rule(fr, bp.signal_date, "40-Week", bp.kind, P)
    assert t.reason.startswith("40-Week") and t.pct_gain > 0.5 and t.drawdown_k > 0
    assert t.peak_retained is not None and 0 < t.peak_retained < 1


def test_ascender_partial_sells_sum_to_full_position():
    path = _flat(100, 4) + _ramp(100, 200, 10) + _ramp(200, 95, 12)
    fr = prepare(_daily(path), date(2019, 1, 7))
    lc = classify_lifecycle(fr.w, P)
    bp = find_buy_points(fr, lc, P)[0]
    t = simulate_rule(fr, bp.signal_date, "Ascender", bp.kind, P)
    assert abs(sum(f for _, _, f, _ in t.sells) - 1.0) < 1e-6
    assert any("½" in why for *_, why in t.sells)


def test_mcp_stop_retains_at_least_mcp_share_of_peak_profit():
    path = _flat(100, 4) + _ramp(100, 200, 10) + _ramp(200, 120, 10)
    fr = prepare(_daily(path), date(2019, 1, 7))
    lc = classify_lifecycle(fr.w, P)
    bp = find_buy_points(fr, lc, P)[0]
    t = simulate_rule(fr, bp.signal_date, "MCP", bp.kind, P)
    assert t.reason.startswith("MCP") and t.peak_retained is not None and t.peak_retained >= 0.40


def test_mcp_not_armed_before_cushion():
    """이익이 +10% 나기 전엔 MCP 스톱이 켜지지 않는다 — 작은 흔들림에 바로 털리지 않게."""
    path = _flat(100, 4) + [104, 102, 101, 103, 102, 104, 103]
    fr = prepare(_daily(path), date(2019, 1, 7))
    lc = classify_lifecycle(fr.w, P)
    bp = find_buy_points(fr, lc, P)[0]
    t = simulate_rule(fr, bp.signal_date, "MCP", bp.kind, P)
    assert not t.reason.startswith("MCP") and t.open_position


def test_ipo_stats_and_cohort_tables():
    rows = []
    paths = {"DBL": _flat(100, 2) + _ramp(100, 210, 8) + _ramp(210, 90, 20) + _flat(90, 30),
             "DIS": _ramp(100, 60, 10) + _flat(60, 50),
             "SLOW": _flat(100, 20) + _ramp(100, 150, 40)}
    for sym, path in paths.items():
        fr = prepare(_daily(path, vol=3e5), date(2019, 1, 7))
        s = ipo_stats(fr, sym)
        assert s is not None
        rows.append(s)
    by = {s.symbol: s for s in rows}
    assert by["DBL"].weeks_to_double is not None and by["DBL"].weeks_to_double <= 13
    assert by["DBL"].corrected50_after_double is True
    assert by["DIS"].weeks_to_double is None and by["DIS"].weeks_undercut_low is not None
    tabs = cohort_tables(rows)
    assert set(tabs) >= {"table_6_1", "fig_6_1", "fig_6_2", "table_6_2", "summary"}
    t61 = tabs["table_6_1"].set_index("첫날 종가 대비 +100% 달성")
    assert abs(t61.loc["13주", "비율"] - 1 / 3) < 1e-9


def test_screen_flags_ipo_alert_and_breakout_week():
    path = _ramp(100, 60, 3) + _flat(70, 10) + [92]   # 급락 → 횡보 → 이번 주 돌파 (왼쪽 고점은 베이스 밖)
    daily = _daily(path, vol=1e6)                     # 거래대금 ≈ $90M
    fr = prepare(daily, date(2019, 1, 7))
    lc = classify_lifecycle(fr.w, P)
    row = screen_ticker("X", fr, lc, revenue_yoy=0.8, p=P)
    assert row.ipo_alert and row.breakout_this_week and row.liquidity_matters is True
    assert row.rare_jewel is False


def test_analyze_ticker_end_to_end_and_summary():
    path = _flat(100, 4) + _ramp(100, 180, 12) + _flat(180, 8) + _ramp(180, 260, 10) + _ramp(260, 200, 8)
    daily = _daily(path)
    ipo = IpoEvent(symbol="T", name="Test Inc", ipo_date=date(2019, 1, 7), exchange="NASDAQ Global", offer_price=20.0)
    res = analyze_ticker("T", daily, ipo, P, as_of=date(2019, 6, 28))
    assert res is not None and res.trades and res.screen is not None
    tf = trades_frame([res])
    rs = rule_summary(tf)
    assert set(rs["rule"]) == set(t.rule for t in res.trades) and "책 중앙값 수익" in rs.columns


def test_ipo_cohort_filters_spac_and_exchange():
    evs = [IpoEvent(symbol="A", name="Alpha Inc", ipo_date=date(2021, 1, 5), exchange="NASDAQ Global", offer_price=15),
           IpoEvent(symbol="B", name="Beta Acquisition Corp", ipo_date=date(2021, 1, 6), exchange="NYSE", offer_price=10),
           IpoEvent(symbol="C", name="Gamma", ipo_date=date(2021, 1, 7), exchange="OTCBB", offer_price=4),
           IpoEvent(symbol="D", name="Delta", ipo_date=date(2021, 1, 8), exchange="New York Stock Exchange", offer_price=3)]
    out = ipo_cohort(evs)
    assert [e.symbol for e in out] == ["A"]
    assert evs[1].looks_like_spac and evs[3].exchange_group == "NYSE"
