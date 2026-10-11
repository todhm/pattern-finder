"""short_term_lab — 코너스·라시케 신호 규칙을 합성 일봉으로 검증."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from strategy.adapters.leveraged_etf_lab import CostParams
from strategy.adapters.short_term_lab import (
    SHORT_TERM_ORDER,
    ShortTermParams,
    adx,
    connors_double7,
    connors_rsi2,
    raschke_holy_grail,
    raschke_turtle_soup,
    rsi,
    run_short_term,
    trade_stats,
)


def _ohlc(close, start="2020-01-02", rng_pct=0.01) -> pd.DataFrame:
    c = pd.Series(np.asarray(close, dtype=float), index=pd.bdate_range(start, periods=len(close)))
    return pd.DataFrame({"Open": c, "High": c * (1 + rng_pct), "Low": c * (1 - rng_pct), "Close": c, "Volume": 1e6})


def test_rsi2_buys_after_two_down_days_in_uptrend_and_exits_above_sma5():
    up = list(np.linspace(100, 160, 230))                       # 200일선 위 상승
    path = up + [158, 156, 154, 155, 159, 161]                 # 3일 하락 → RSI2 급락 → 반등
    s = connors_rsi2(pd.Series(path, index=pd.bdate_range("2020-01-02", periods=len(path))), ShortTermParams(sma_trend=200))
    assert s.iloc[229] == 0 and s.iloc[232] == 1.0             # 154 에서 매수 (RSI2 < 10)
    assert s.iloc[-1] == 0.0 and s.iloc[-2] == 0.0             # 159 가 5일선 위 → 청산
    r = rsi(pd.Series([100, 99, 98, 97, 96], dtype=float), 2)
    assert r.iloc[-1] < 5


def test_double7_entry_at_7day_low_close_and_exit_at_7day_high():
    up = list(np.linspace(100, 160, 230))
    path = up + [159, 158, 157, 156, 158, 160, 161, 162, 163]
    s = connors_double7(pd.Series(path, index=pd.bdate_range("2020-01-02", periods=len(path))))
    assert s.iloc[:230].sum() == 0 and s.iloc[231:234].max() == 1.0   # 158~156 구간에서 7일 최저 종가 → 매수
    assert s.iloc[233] == 1.0 and s.iloc[-3:].sum() == 0               # 160/161 이 7일 최고 → 청산, 이후 0
    assert 2 <= s.sum() <= 6


def test_holy_grail_requires_adx_trend_and_ema_touch_then_breakout():
    strong = np.linspace(100, 200, 80) + 1.5 * np.sin(np.arange(80))   # 강한 추세(약간의 흔들림 → ADX 100 포화 방지)
    pull = [200, 194, 190, 186, 188, 193, 199, 206, 212, 220]           # 되돌림 → 20EMA 터치 → 돌파 → 전고 돌파
    df = _ohlc(list(strong) + pull, rng_pct=0.005)
    a = adx(df)
    assert 30 < a.iloc[79] < 100
    s = raschke_holy_grail(df)
    assert s.iloc[:80].sum() == 0 and s.max() == 1.0
    held = s[s == 1.0]
    assert 1 <= len(held) <= 10                                 # 진입 후 목표(전고 200) 또는 최대 보유일에 청산
    flat = _ohlc(list(100 + 2 * np.sin(np.arange(120) / 3)))
    assert raschke_holy_grail(flat).sum() == 0                  # 추세 없으면 거래 없음


def test_turtle_soup_buys_false_breakdown_and_plus_one_variant():
    base = [100 + (i % 5) for i in range(30)]                   # 20일 저점 100 이 4일 이상 전에 있음
    path = base + [101]                                         # 오늘: 저가 99.9 (< 100) 인데 종가 101 → 복귀
    df = _ohlc(path, rng_pct=0.0)
    df.loc[df.index[-1], "Low"] = 99.9
    s = raschke_turtle_soup(df)
    assert s.iloc[-1] == 1.0 and s.iloc[-2] == 0.0
    after = pd.concat([df, _ohlc([102, 103, 104, 105, 106, 107], start=str((df.index[-1] + pd.offsets.BDay(1)).date()), rng_pct=0.0)])
    s2 = raschke_turtle_soup(after, ShortTermParams(ts_max_hold=5))
    assert s2.iloc[-7] == 1.0 and s2.iloc[-1] == 0.0            # 5일 보유 후 익절
    plus = raschke_turtle_soup(after, ShortTermParams(ts_plus_one=True))
    assert plus.iloc[-7] == 0.0 and plus.iloc[-6] == 1.0         # 다음날 종가 102 > 100 → 진입


def test_run_short_term_and_trade_stats():
    rng = np.random.default_rng(3)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.012, 700)))
    df = _ohlc(close, start="2019-01-02")
    lev = df["Close"] ** 1.0 * 3 - 200                          # 대충 레버리지 흉내 (양수 유지)
    lev = lev.clip(lower=1.0)
    res = run_short_term(lev, df, ShortTermParams(), CostParams(buy_fee=0, sell_fee=0, tax_rate=0, initial_capital=1_000.0))
    assert list(res) == SHORT_TERM_ORDER
    for k, r in res.items():
        assert set(r.exposure.round().unique()) <= {0.0, 1.0}
        st = trade_stats(r, lev)
        assert st["trades"] >= 0 and (st["trades"] == 0 or 0 <= st["win_rate"] <= 1)
    assert trade_stats(res[SHORT_TERM_ORDER[0]], lev)["trades"] > 0
