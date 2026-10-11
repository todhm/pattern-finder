"""intraday_lab — ORB 규칙(방향·스톱·목표·EoD·사이징·수수료)과 오버나이트 곡선을 합성 5분봉으로 검증."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from strategy.adapters.intraday_lab import (
    OrbParams,
    alpha_beta,
    atr14_from_bars,
    curve_summary,
    overnight_curves,
    run_orb,
)


def _day(date: str, path: list[tuple[float, float, float, float]], pre: bool = False) -> pd.DataFrame:
    """09:30 부터 5분 간격 (O,H,L,C) 목록 → 봉. pre=True 면 09:00 프리마켓 봉 하나를 앞에 붙인다(필터 확인용)."""
    idx = pd.date_range(f"{date} 09:30", periods=len(path), freq="5min", tz="America/New_York")
    df = pd.DataFrame(path, columns=["Open", "High", "Low", "Close"], index=idx)
    df["Volume"] = 1000.0
    if pre:
        pm = pd.DataFrame([[999, 999, 999, 999, 1.0]], columns=df.columns,
                          index=pd.DatetimeIndex([pd.Timestamp(f"{date} 09:00", tz="America/New_York")]))
        df = pd.concat([pm, df])
    return df


NO_COST = OrbParams(commission_per_share=0.0)


def test_long_trade_held_to_eod_sized_by_risk():
    # 첫 봉 양봉 100→101 (저가 99.5). 둘째 봉 시가 101 진입, R=1.5, 종가 104 → +3 × shares
    bars = _day("2024-01-02", [(100, 101.2, 99.5, 101), (101, 102, 100.8, 101.5), (101.5, 104, 101, 104)], pre=True)
    r = run_orb(bars, p=NO_COST)
    assert len(r.trades) == 1
    t = r.trades[0]
    assert t.side == 1 and t.entry == 101 and t.stop == 99.5 and t.exit_reason == "eod" and t.exit == 104
    assert t.shares == int(25_000 * 0.01 / 1.5) == 166 and not t.capped
    assert t.pnl == pytest.approx(166 * 3) and t.r_multiple == pytest.approx(2.0)
    assert r.equity.iloc[-1] == pytest.approx(25_000 + 498)


def test_short_trade_stopped_at_first_bar_high():
    bars = _day("2024-01-03", [(100, 100.4, 99, 99.2), (99.2, 99.5, 98.8, 99.3), (99.3, 100.6, 99.2, 100.5)])
    r = run_orb(bars, p=NO_COST)
    t = r.trades[0]
    assert t.side == -1 and t.stop == 100.4 and t.exit_reason == "stop" and t.exit == 100.4
    assert t.r_multiple == pytest.approx(-1.0) and r.equity.iloc[-1] == pytest.approx(25_000 * 0.99, rel=1e-3)


def test_target_10r_and_doji_and_leverage_cap():
    # 도지 날 → 거래 없음
    doji = _day("2024-01-04", [(100, 100.5, 99.5, 100), (100, 101, 99, 100.5)])
    assert run_orb(doji, p=NO_COST).trades == []
    # 10R 목표: R=0.2 → 목표 102.0 → 세 번째 봉 고가 102.5 에서 체결
    tgt = _day("2024-01-05", [(100, 100.1, 99.8, 100.1), (100.0, 100.2, 99.9, 100.1), (100.1, 102.5, 100, 102.4)])
    r = run_orb(tgt, p=NO_COST)
    t = r.trades[0]
    assert t.exit_reason == "target" and t.exit == pytest.approx(100.0 + 10 * 0.2) and t.r_multiple == pytest.approx(10.0)
    # 레버리지 상한: R 이 아주 작으면 주수가 4×A/P 로 잘린다
    assert t.shares == int(4 * 25_000 / 100.0) == 1000 and t.capped


def test_commissions_per_share_and_pct_and_tax():
    bars = _day("2024-12-31", [(100, 101.2, 99.5, 101), (101, 102, 100.8, 101.5), (101.5, 104, 101, 104)])
    paper = run_orb(bars, p=OrbParams(commission_per_share=0.0005))
    assert paper.trades[0].pnl == pytest.approx(166 * 3 - 2 * 166 * 0.0005)
    toss = run_orb(bars, p=OrbParams(commission_per_share=0.0, commission_pct=0.001, tax_rate=0.22))
    gross = 166 * 3; comm = 0.001 * 166 * (101 + 104)
    assert toss.trades[0].pnl == pytest.approx(gross - comm)
    assert toss.tax_paid == pytest.approx((gross - comm) * 0.22) and toss.equity.iloc[-1] == pytest.approx(25_000 + (gross - comm) * 0.78)


def test_atr_stop_mode_uses_previous_days_atr():
    days = []
    for i, d in enumerate(pd.bdate_range("2024-02-01", periods=16)):
        base = 100 + i
        days.append(_day(d.strftime("%Y-%m-%d"), [(base, base + 2, base - 2, base + 0.5), (base + 0.5, base + 1, base, base + 0.8),
                                                   (base + 0.8, base + 3, base + 0.5, base + 2.5)]))
    bars = pd.concat(days)
    atr = atr14_from_bars(bars)
    assert np.isnan(atr.iloc[0]) and atr.iloc[-1] > 3.5                        # TR ≈ 4 (고저) 또는 ≈5 (갭 포함)
    r = run_orb(bars, p=OrbParams(stop_mode="atr", atr_frac=0.05, commission_per_share=0.0, target_r=None))
    last = r.trades[-1]
    assert last.r_value == pytest.approx(0.05 * float(atr.iloc[-1])) and last.exit_reason in ("eod", "stop")


def test_signal_and_trade_symbols_can_differ_and_alpha_beta_shape():
    sig = _day("2024-03-01", [(100, 101, 99.5, 101), (101, 102, 100.8, 101.5), (101.5, 104, 101, 104)])
    trd = _day("2024-03-01", [(50, 50.5, 49.5, 50.4), (50.4, 51, 50.2, 50.6), (50.6, 52, 50.5, 52)])
    r = run_orb(sig, trd, p=NO_COST)
    t = r.trades[0]
    assert t.side == 1 and t.entry == 50.4 and t.stop == 49.5 and t.exit == 52
    idx = pd.bdate_range("2024-01-01", periods=60)
    rng = np.random.default_rng(0)
    bench = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 60))), index=idx)
    strat = bench * 1.0001 ** np.arange(60) * 2
    ab = alpha_beta(strat, bench)
    assert abs(ab["beta"] - 1.0) < 0.05 and ab["n"] == 59


def test_overnight_curves_split_night_and_day_returns():
    idx = pd.bdate_range("2024-01-01", periods=4)
    d = pd.DataFrame({"Open": [100, 102, 103, 105], "Close": [101, 101, 104, 104]}, index=idx)
    c = overnight_curves(d, commission_pct=0.0, initial_capital=100.0)
    assert c["장기보유"].iloc[-1] == pytest.approx(100 * 104 / 101)
    night = (102 / 101) * (103 / 101) * (105 / 104)
    day = (101 / 100) * (101 / 102) * (104 / 103) * (104 / 105)
    assert c["밤만 (종가→시가)"].iloc[-1] == pytest.approx(100 * night)
    assert c["낮만 (시가→종가)"].iloc[-1] == pytest.approx(100 * day)
    taxed = overnight_curves(d, commission_pct=0.001, tax_rate=0.22, initial_capital=100.0)
    assert taxed["밤만 (종가→시가)"].iloc[-1] < c["밤만 (종가→시가)"].iloc[-1]
    s = curve_summary(c)
    assert list(s["전략"]) == list(c.columns) and {"CAGR", "MDD"} <= set(s.columns)


def test_tax_nets_losses_against_gains_within_year_and_applies_deduction():
    """연 단위 순이익 과세: 손실일이 이익을 상쇄하고, 공제 이하면 세금 0."""
    win = _day("2024-06-03", [(100, 101.2, 99.5, 101), (101, 102, 100.8, 101.5), (101.5, 104, 101, 104)])   # +498
    loss = _day("2024-06-04", [(100, 101.2, 99.5, 101), (101, 101.5, 100.8, 101.2), (101.2, 101.3, 98, 98.5)])  # 스톱 −1R ≈ −250
    bars = pd.concat([win, loss])
    r = run_orb(bars, p=OrbParams(commission_per_share=0.0, tax_rate=0.22))
    net = sum(t.pnl for t in r.trades)
    assert r.trades[1].pnl < 0 < net and r.tax_paid == pytest.approx(net * 0.22)
    r2 = run_orb(bars, p=OrbParams(commission_per_share=0.0, tax_rate=0.22, tax_deduction=1_800.0))
    assert r2.tax_paid == 0.0
    only_loss = run_orb(loss, p=OrbParams(commission_per_share=0.0, tax_rate=0.22))
    assert only_loss.tax_paid == 0.0


def test_split_factor_scales_per_share_costs_to_unadjusted_shares():
    from strategy.adapters.intraday_lab import split_factor_series
    idx = pd.DatetimeIndex(["2016-06-01", "2017-06-01", "2018-06-01", "2021-06-01", "2022-06-01"])
    f = split_factor_series(idx, [("2017-01-12", 2), ("2018-05-24", 3), ("2021-01-21", 2), ("2022-01-13", 2)])
    assert list(f.values) == [24.0, 12.0, 4.0, 2.0, 1.0]
    bars = _day("2016-06-01", [(4, 4.05, 3.98, 4.04), (4.04, 4.08, 4.03, 4.06), (4.06, 4.16, 4.04, 4.16)])
    p = OrbParams(commission_per_share=0.0035, slippage_per_share=0.01)
    raw = run_orb(bars, p=p)
    adj = run_orb(bars, p=p, unadjusted_factor=pd.Series(24.0, index=pd.DatetimeIndex(["2016-06-01"])))
    sh = raw.trades[0].shares
    assert raw.trades[0].pnl == pytest.approx(sh * 0.12 - 2 * sh * 0.0135)
    assert adj.trades[0].pnl == pytest.approx(sh * 0.12 - 2 * sh * 0.0135 / 24)
