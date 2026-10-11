"""minervini_manual_auto · minervini_scan.entry_permissions — 합성 데이터 판정 검증."""

import numpy as np
import pandas as pd

from strategy.adapters.minervini_manual_auto import (
    index_correction,
    industry_rank,
    leadership_checks,
    new_bull_window,
    not_chasing,
    prior_uptrend,
)
from strategy.adapters.minervini_scan import EntryInput, breadth_ok, entry_permissions
from strategy.adapters.minervini_screener import PivotLevels


def _df(closes) -> pd.DataFrame:
    c = np.asarray(closes, dtype=float)
    idx = pd.bdate_range("2021-01-01", periods=len(c))
    return pd.DataFrame({"Open": c, "High": c * 1.01, "Low": c * 0.99,
                         "Close": c, "Volume": np.full(len(c), 1e6)}, index=idx)


# 지수: 100 → 120 (150봉) → 96 (−20%, 50봉) → 115 (30봉 반등)
INDEX = _df(np.concatenate([np.linspace(100, 120, 150), np.linspace(120, 96, 50),
                            np.linspace(96, 115, 30)]))


def _pl(**kw) -> PivotLevels:
    base = dict(pivot=100.0, close=99.0, dist_to_pivot=-0.01, tightness_10d=0.05,
                volume_dryup=0.6, stop=92.0, target=122.0, base_weeks=6.0)
    base.update(kw)
    return PivotLevels(**base)


def test_index_correction_and_new_bull_window():
    corr = index_correction(INDEX)
    assert corr is not None
    assert abs(corr.depth - 0.20) < 0.01
    assert corr.weeks_since_trough == 30 / 5
    assert new_bull_window(corr) is True              # −20% 뒤 6주, 반등 +20%
    late = index_correction(_df(np.concatenate([INDEX["Close"].values, np.linspace(115, 130, 30)])))
    assert new_bull_window(late) is False             # 저점 후 12주 — 초입 지남
    assert new_bull_window(None) is None


def test_leadership_checks_leader_vs_laggard():
    corr = index_correction(INDEX)
    # 리더: 지수 조정 중 −8%만 빠지고 지수보다 먼저 바닥, 지수 저점일 고가 근처
    leader = _df(np.concatenate([np.linspace(50, 80, 150), np.linspace(80, 73.6, 20),
                                 np.linspace(73.6, 79, 30), np.linspace(79, 90, 30)]))
    r = leadership_checks(leader, corr)
    assert r["m3_1"][0] is True and r["m3_2"][0] is True and r["m3_3"][0] is True
    # 낙오주: 지수 저점 이후까지 −70%
    lag = _df(np.concatenate([np.linspace(50, 80, 150), np.linspace(80, 24, 60), np.full(20, 25.0)]))
    r = leadership_checks(lag, corr)
    assert r["m3_1"][0] is False and r["m3_2"][0] is False and r["m3_3"][0] is False


def test_leadership_checks_none_without_correction():
    flat_up = index_correction(_df(np.linspace(100, 130, 200)))
    r = leadership_checks(_df(np.linspace(10, 20, 200)), flat_up)
    assert all(v[0] is None for v in r.values())


def test_prior_uptrend_and_not_chasing():
    # 60 → 100 (+67%) 상승 후 6주 베이스
    df = _df(np.concatenate([np.linspace(60, 100, 120), np.full(30, 95.0)]))
    ok, note = prior_uptrend(df, _pl(base_weeks=6.0))
    assert ok is True and "+" in note
    flat = _df(np.concatenate([np.linspace(90, 100, 120), np.full(30, 95.0)]))
    assert prior_uptrend(flat, _pl(base_weeks=6.0))[0] is False
    assert prior_uptrend(df, _pl(base_weeks=None))[0] is None
    assert not_chasing(_pl(dist_to_pivot=0.03))[0] is True
    assert not_chasing(_pl(dist_to_pivot=0.12))[0] is False


def test_industry_rank_top2_within_group():
    r = industry_rank([("A", "Semis", 99), ("B", "Semis", 95), ("C", "Semis", 90),
                       ("D", "Software", 80), ("E", "미분류", 85)])
    assert r["A"][0] and r["B"][0] and r["C"][0] is False
    assert r["D"][0] is True and r["E"][0] is None


def _ei(rs=95, growth=True, verified=True, top=True):
    return EntryInput(rs=rs, growth_full=growth, verified=verified, industry_top=top)


def test_entry_permissions_quality_filter():
    items = [_ei(), _ei(rs=80), _ei(growth=False), _ei(top=False), _ei(verified=False), _ei(top=None)]
    oks = [ok for ok, _ in entry_permissions(items, breadth=True, warns=[], max_entries=3)]
    assert oks == [True, False, False, False, False, False]
    notes = [n for _, n in entry_permissions(items, breadth=True, warns=[], max_entries=3)]
    assert "RS<90" in notes[1] and "성장<4/4" in notes[2] and "산업군" in notes[3]
    assert "미확인" in notes[4]


def test_entry_permissions_breadth_and_caution_cap_by_rs():
    items = [_ei(rs=91), _ei(rs=99), _ei(rs=95), _ei(rs=93)]
    assert not any(ok for ok, _ in entry_permissions(items, breadth=False, warns=[], max_entries=3))
    p = entry_permissions(items, breadth=True, warns=["0-2"], max_entries=2)
    assert [ok for ok, _ in p] == [False, True, True, False]   # RS 99·95 만
    assert all(ok for ok, _ in entry_permissions(items, breadth=True, warns=[], max_entries=2))


def test_breadth_ok_level_and_ratio_trend():
    assert breadth_ok(100, 50) and not breadth_ok(50, 50)
    assert breadth_ok(70, 30, 75, 25, rule="ratio_trend")          # 70% vs 75% — 소폭 하락
    assert not breadth_ok(55, 45, 85, 15, rule="ratio_trend")      # 85% → 55% 급락
    assert not breadth_ok(40, 60, rule="ratio_trend")              # 50% 미만


def test_flat_box_excluded_before_everything_else():
    """박스 고정(M&A·우선주·ETN) 종목은 시장·펀더멘털과 무관하게 제외 — 2026-10-10 SMTI/DSGR/STRC/STRF/USOI."""
    from strategy.adapters.minervini_scan import EntryInput, entry_permissions, is_flat_box

    assert is_flat_box(0.016, 0.02) and is_flat_box(0.012, 0.008)          # SMTI · DSGR
    assert not is_flat_box(0.047, 0.047) and not is_flat_box(0.016, 0.10)   # NDSN(조밀하지만 베이스 있음) · 깊은 베이스
    assert not is_flat_box(None, 0.01) and not is_flat_box(0.01, None)
    items = [EntryInput(rs=95, growth_full=True, verified=True, industry_top=True, flat_box=True),
             EntryInput(rs=95, growth_full=True, verified=True, industry_top=True)]
    good = entry_permissions(items, breadth=True, warns=[], max_entries=3)
    assert good[0] == (False, "⛔ 제외 (가격 고정 — 상승 여지 없음: M&A 확정·우선주·ETN)") and good[1][0] is True
    bad_market = entry_permissions(items, breadth=False, warns=[], max_entries=3)
    assert bad_market[0][1].startswith("⛔") and bad_market[1][1].startswith("👀 관찰 (0-3")
