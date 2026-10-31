"""근거중심매매 백테스트 엔진 — 합성 데이터로 레벨·레짐·시뮬 뼈대 검증."""
from datetime import date

import numpy as np
import pandas as pd

from strategy.adapters import evidence_backtest as eb
from strategy.adapters import evidence_checklist as ec


def _ohlc(closes, start="2015-01-01"):
    c = np.asarray(closes, dtype=float)
    o = np.r_[c[0], c[:-1]]
    idx = pd.bdate_range(start, periods=len(c))
    return pd.DataFrame({"Open": o, "High": np.maximum(o, c) * 1.003, "Low": np.minimum(o, c) * 0.997, "Close": c, "Volume": 1e6}, index=idx)


def _rw(seed, n=1500, drift=0.0004):
    rng = np.random.default_rng(seed)
    return _ohlc(100 * np.exp(np.cumsum(rng.normal(drift, 0.015, n))))


def _pullback_report():
    # 100 → 140 상승 뒤 120(0.5) 부근으로 되돌림
    closes = [95, 92, 90] + list(np.linspace(91, 140, 60)) + list(np.linspace(139, 120, 20)) + [121, 122]
    df = _ohlc(closes); df["Open"] = df["Close"]; df["High"] = df["Close"]; df["Low"] = df["Close"]
    d = ec.add_indicators(df)
    return eb.fast_report("일봉", d, None, ec.EvidenceParams(), 5)


def test_entry_levels_book_rules_second_at_0618_and_extension_target():
    wr = _pullback_report()
    fb = wr.fib
    assert fb is not None and fb.kind.startswith("상승")
    lv = eb.entry_levels(wr, 122.0, eb.SimRules(), pullback_low=120.0)
    assert lv is not None
    stop, second, t1, fin = lv
    # 2차는 0.618 이상 깊은 레벨, 손절은 그 아래 레벨(0.786)
    assert second == fb.levels[0.618] and stop == fb.levels[0.786]
    # 1:1 = 되돌림 저점 C + 앞 상승분 (B−A), 최종 = C + 1.618 (B−A)
    leg = fb.hi - fb.lo
    assert abs(t1 - (120.0 + leg)) < 1e-6 and abs(fin - (120.0 + 1.618 * leg)) < 1e-6


def test_entry_levels_legacy_uses_first_level_and_swing_high():
    wr = _pullback_report()
    fb = wr.fib
    lv = eb.entry_levels(wr, 122.0, eb.legacy_rules())
    assert lv is not None
    stop, second, t1, fin = lv
    assert second == fb.levels[0.382]          # 첫 실행판: 진입가 아래 첫 레벨
    assert t1 == fb.hi and fin > t1
    assert eb.legacy_rules().second_fill == "next_open" and eb.SimRules().second_fill == "limit"


def test_classify_and_pick_dates():
    qqq = _rw(1, 2500, 0.0005)
    reg = eb.classify_regimes(qqq)
    assert set(reg.regime) <= {"상승장", "하락장", "평범"} and len(reg) > 60
    dates = eb.pick_dates(reg, 2, 0, str(reg.index[10].date()), str(reg.index[-10].date()))
    assert 1 <= len(dates) <= 6 and all(isinstance(d, pd.Timestamp) for _, d in dates)


def test_run_sim_executes_and_books_trades():
    prices = {f"T{i}": _rw(10 + i) for i in range(12)}
    data = eb.precompute(prices)
    cal = pd.to_datetime(prices["T0"].index)
    rules = eb.SimRules(horizon_days=80, weekly_min_yes=2, require_daily_candle=False, max_ext_above_ema20=None)
    res = eb.run_sim(data, cal, cal[900], rules, index_data=data["T0"])
    assert len(res.equity) == 81 and res.equity.iloc[0] == 1.0
    assert res.stats["거래 수"] >= 0 and "청산 사유" in res.stats
    for tr in res.trades:
        assert tr.stop < tr.entry_px * 1.0001 and tr.target1 > tr.entry_px
        assert abs(tr.open_qty) < 1e-9        # 구간 종료 시 전량 평가 청산
    # 플라세보 picker 도 같은 뼈대로 돈다
    rng = np.random.default_rng(0)
    res2 = eb.run_sim(data, cal, cal[900], rules, picker=eb.placebo_full(rng, list(data)), screen_cache={})
    assert len(res2.equity) == 81
    # 지정가 2차는 체결가가 레벨 이하
    for tr in res.trades + res2.trades:
        if tr.second_px is not None:
            assert tr.second_px <= tr.second_lvl * 1.0001


def test_legacy_rules_still_run():
    prices = {f"L{i}": _rw(30 + i) for i in range(8)}
    data = eb.precompute(prices)
    cal = pd.to_datetime(prices["L0"].index)
    rules = eb.legacy_rules(horizon_days=60, weekly_min_yes=2, require_daily_candle=False)
    res = eb.run_sim(data, cal, cal[900], rules)
    assert len(res.equity) == 61 and all(tr.target1 > tr.entry_px for tr in res.trades)
