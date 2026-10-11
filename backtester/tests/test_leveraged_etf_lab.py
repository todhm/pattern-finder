"""leveraged_etf_lab — 신호·원장·9Sig 규칙을 손으로 만든 가격으로 검증."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from strategy.adapters.leveraged_etf_lab import (
    CRISIS_WINDOWS,
    CostParams,
    LabParams,
    combine,
    crisis_table,
    current_signal,
    realized_vol,
    run_lab,
    simulate_exposure,
    simulate_nine_sig,
    sma_filter,
    summary_table,
    vol_target,
    yearly_returns,
)

NO_COST = CostParams(buy_fee=0.0, sell_fee=0.0, tax_rate=0.0, initial_capital=1_000.0)


def _series(values, start="2020-01-02") -> pd.Series:
    return pd.Series(np.asarray(values, dtype=float), index=pd.bdate_range(start, periods=len(values)))


def test_sma_filter_hysteresis_and_warmup():
    s = _series([10] * 5 + [12, 12, 9, 9, 10.3, 10.3])
    f = sma_filter(s, n=5, enter_band=0.05, exit_band=0.05)
    assert f.iloc[:4].sum() == 0                       # SMA 미계산 구간은 현금
    assert f.iloc[5] == 1.0                            # 12 > 10×1.05 진입
    assert f.iloc[7] == 0.0                            # 9 < SMA×0.95 이탈
    assert f.iloc[9] == 0.0                            # 밴드 안(SMA≈10.4 → 10.3 은 −1%)이면 유지


def test_sma_filter_weekly_decides_only_on_week_end():
    # 월요일에 급등해도 금요일까지는 신호가 안 바뀐다
    idx = pd.bdate_range("2024-01-01", periods=15)   # 월요일 시작
    s = pd.Series([10.0] * 10 + [20.0, 21.0, 22.0, 23.0, 24.0], index=idx)
    daily = sma_filter(s, n=5)
    weekly = sma_filter(s, n=5, weekly=True)
    assert daily.iloc[10] == 1.0                       # 세 번째 주 월요일 즉시
    assert weekly.iloc[10] == 0.0 and weekly.iloc[14] == 1.0   # 금요일에만


def test_vol_target_scales_inverse_to_realized_vol():
    rng = np.random.default_rng(0)
    calm = 100 * np.exp(np.cumsum(rng.normal(0, 0.005, 60)))
    wild = calm[-1] * np.exp(np.cumsum(rng.normal(0, 0.05, 60)))
    s = _series(np.concatenate([calm, wild]))
    w = vol_target(s, target_vol=0.45, lookback=20, max_exposure=1.0)
    assert w.iloc[:19].sum() == 0 and w.iloc[55] == 1.0            # 조용할 땐 상한
    assert 0.3 < w.iloc[-1] < 0.8                                   # 변동성 ~80% → 비중 ~0.55
    assert realized_vol(s, 20).iloc[-1] > 0.5


def test_simulate_exposure_buy_hold_matches_price_without_costs():
    s = _series([10, 11, 12, 9, 13])
    r = simulate_exposure(s, pd.Series(1.0, index=s.index), NO_COST)
    assert r.equity.iloc[-1] == pytest.approx(1_000 * 13 / 10)
    assert r.trades == 1 and r.time_in_market == 1.0


def test_simulate_exposure_cash_when_signal_off_and_fees_tax_applied():
    s = _series([10, 20, 20, 10, 10], start="2023-12-27")       # 연말 걸침 → 세금 정산
    tgt = pd.Series([1, 0, 0, 0, 0], index=s.index, dtype=float)
    c = CostParams(buy_fee=0.001, sell_fee=0.001, tax_rate=0.22, tax_deduction=0.0, initial_capital=1_000.0)
    r = simulate_exposure(s, tgt, c)
    # 매수 1000/1.001 → 가격 2배 → 매도 ~1998, 수수료 ~2, 실현이익 ~998 → 세금 ~219.6
    assert r.trades == 2 and r.fees == pytest.approx(0.999 + 1.998, rel=1e-3)
    assert r.tax == pytest.approx(0.22 * (1998.0 - 999.0), rel=1e-2)
    assert r.equity.iloc[-1] == pytest.approx(1000 + 998 - 2.997 - r.tax, rel=1e-2)
    assert r.exposure.iloc[-1] == 0.0


def test_exec_lag_shifts_entry_one_day():
    s = _series([10, 10, 20, 20])
    tgt = pd.Series([0, 1, 1, 1], index=s.index, dtype=float)
    same = simulate_exposure(s, tgt, NO_COST)
    lag = simulate_exposure(s, tgt, CostParams(buy_fee=0, sell_fee=0, tax_rate=0, initial_capital=1_000.0, exec_lag=1))
    assert same.equity.iloc[-1] == pytest.approx(2_000) and lag.equity.iloc[-1] == pytest.approx(1_000)


def test_rebalance_band_suppresses_small_trades():
    s = _series([10] * 10)
    tgt = pd.Series(np.linspace(0.5, 0.54, 10), index=s.index)   # 누적 차이 40 < 밴드 50
    assert simulate_exposure(s, tgt, NO_COST).trades == 10
    assert simulate_exposure(s, tgt, CostParams(buy_fee=0, sell_fee=0, tax_rate=0, initial_capital=1_000.0,
                                                rebalance_band=0.05)).trades == 1


def test_cash_rate_accrues_only_on_cash():
    s = _series([10] * 253)
    r = simulate_exposure(s, pd.Series(0.0, index=s.index), NO_COST, cash_rate=0.05)
    assert r.equity.iloc[-1] == pytest.approx(1_000 * (1 + 0.05 / 252) ** 252, rel=1e-6)


def test_nine_sig_rebalances_quarterly_to_growing_target():
    # 2분기: 1분기 끝까지 가격 불변 → 2분기 첫날 V=600×1.09=654 → 54 매수. 이후 가격 2배 → 다음 분기 초과분 매도.
    idx = pd.bdate_range("2024-01-02", "2024-07-05")
    px = pd.Series(10.0, index=idx)
    px[idx >= "2024-04-02"] = 20.0
    r = simulate_nine_sig(px, NO_COST, stock_weight=0.6, quarterly_growth=0.09)
    q2_first = idx[idx >= "2024-04-01"][0]
    assert r.exposure.loc[q2_first] == pytest.approx(654 / 1000, rel=1e-6)   # 가격은 아직 10
    q3_first = idx[idx >= "2024-07-01"][0]
    V3 = 654 * 1.09                   # 712.86 → 1308 − 712.86 = 595.14 매도
    cash_q3 = 400 - 54 + (1308 - V3)  # 941.14
    assert r.exposure.loc[q3_first] == pytest.approx(V3 / (cash_q3 + V3), rel=1e-6)
    assert r.trades == 3


def test_run_lab_and_tables_shapes():
    rng = np.random.default_rng(1)
    sig = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, 900))),
                    index=pd.bdate_range("2019-01-02", periods=900))
    lev = 100 * (1 + 3 * sig.pct_change().fillna(0)).cumprod()
    res = run_lab(lev, sig, LabParams(sma_n=50), NO_COST, start=date(2019, 6, 1))
    assert set(res) == {"① 장기보유", "② 추세 필터", "③ 변동성 타기팅", "④ 필터+타기팅", "⑤ 9Sig", "⑩ 급락 회피"}
    assert res["① 장기보유"].equity.index[0] >= pd.Timestamp("2019-06-01")
    assert res["② 추세 필터"].exposure.max() <= 1.0 and 0 < res["② 추세 필터"].time_in_market < 1
    assert (res["④ 필터+타기팅"].target <= res["② 추세 필터"].target + 1e-12).all()
    st = summary_table(res)
    assert list(st["전략"]) == list(res) and {"CAGR", "MDD", "매매 횟수", "시장 노출"} <= set(st.columns)
    yr = yearly_returns(res)
    assert list(yr.index) == [2019, 2020, 2021, 2022] and yr.shape[1] == 6
    ct = crisis_table(res, {"코로나": CRISIS_WINDOWS["코로나"], "닷컴 붕괴": CRISIS_WINDOWS["닷컴 붕괴"]})
    assert ct.loc[ct["구간"] == "코로나", "① 장기보유"].notna().all()
    assert ct.loc[ct["구간"] == "닷컴 붕괴", "① 장기보유"].isna().all()          # 데이터 밖


def test_combine_and_current_signal():
    s = _series(np.linspace(100, 200, 300))
    lev = s.copy()
    f = sma_filter(s, 50)
    v = vol_target(lev, 0.45, 20)
    c = combine(f, v)
    assert (c <= f).all() and (c <= v.reindex(c.index) + 1e-12).all()
    cs = current_signal(lev, s, LabParams(sma_n=50))
    assert cs["filter_on"] is True and cs["dist_to_sma"] > 0 and 0 <= cs["combined_weight"] <= 1


def test_deep_correction_filter_exits_only_on_deep_drawdown_and_reenters_fast():
    from strategy.adapters.leveraged_etf_lab import deep_correction_filter, month_end_only
    up = list(np.linspace(100, 200, 260))
    shallow = list(np.linspace(200, 180, 10)) + list(np.linspace(180, 210, 10))          # −10%: 이탈 안 함
    deep = list(np.linspace(210, 160, 30)) + list(np.linspace(160, 215, 40))             # −24%: 이탈 후 20일 SMA 회복 시 재진입
    s = _series(up + shallow + deep)
    f = deep_correction_filter(s, n=200, dd=0.15, reenter_n=20)
    assert f.iloc[:199].sum() == 0 and f.iloc[200:280].min() == 1.0                      # 워밍업 전 0, 얕은 조정은 보유
    assert f.iloc[280:310].min() == 0.0 and f.iloc[-1] == 1.0                            # 깊은 조정에서 이탈, 반등 후 복귀
    off = deep_correction_filter(s, 200, 0.15, 20, off_fraction=0.33)
    assert off.min() == pytest.approx(0.33)
    m = month_end_only(f)
    assert set(m.unique()) <= {0.0, 1.0} and (m.groupby(m.index.to_period("M")).nunique() <= 2).all()


def test_run_lab_includes_deep_correction_strategy():
    rng = np.random.default_rng(2)
    sig = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, 900))), index=pd.bdate_range("2019-01-02", periods=900))
    lev = 100 * (1 + 3 * sig.pct_change().fillna(0)).cumprod()
    res = run_lab(lev, sig, LabParams(sma_n=50, dd_n=50, sma_monthly=True), NO_COST, start=date(2019, 6, 1))
    assert "⑩ 급락 회피" in res and res["⑩ 급락 회피"].exposure.max() <= 1.0
    assert len(summary_table(res)) == 6
