"""근거중심매매 체크리스트 엔진 — 합성 데이터로 규칙 판정을 검증."""
from datetime import date

import numpy as np
import pandas as pd
import pytest

from strategy.adapters import evidence_checklist as ec


def _ohlc(closes, vol=1000.0, wick=0.0):
    c = np.asarray(closes, dtype=float)
    o = np.r_[c[0], c[:-1]]
    h = np.maximum(o, c) * (1 + wick)
    l = np.minimum(o, c) * (1 - wick)
    idx = pd.bdate_range("2020-01-01", periods=len(c))
    return pd.DataFrame({"Open": o, "High": h, "Low": l, "Close": c, "Volume": vol}, index=idx)


def test_rsi_bounds_and_direction():
    up = ec.rsi(pd.Series(np.linspace(100, 200, 60)))
    dn = ec.rsi(pd.Series(np.linspace(200, 100, 60)))
    assert up.iloc[-1] > 70 and dn.iloc[-1] < 30
    assert (up.dropna().between(0, 100)).all()


def test_alignment_uptrend_is_bull():
    df = ec.add_indicators(_ohlc(np.linspace(100, 300, 400)))
    st = ec.ma_state(df)
    assert st.alignment == "정배열" and st.above_ema20 and not st.below_200


def test_candle_patterns():
    # 큰 음봉 → 도지 → 큰 양봉 = 모닝스타, 20MA 위 마감
    base = list(np.linspace(100, 120, 40))
    closes = base + [110.0, 110.2, 121.0]
    df = _ohlc(closes)
    df.loc[df.index[-2], ["Open", "High", "Low"]] = [110.0, 111.0, 109.0]   # 도지 (몸통 ≪ 범위)
    df.loc[df.index[-3], "Open"] = 120.0        # 장대음봉 120→110
    df.loc[df.index[-1], "Open"] = 110.2        # 장대양봉
    df = ec.add_indicators(df)
    cs = ec.candle_state(df)
    assert "모닝스타" in cs.patterns and cs.bullish

    # 긴 아래꼬리 (몸통 1, 꼬리 3) = 강한 신호
    df2 = _ohlc(list(np.linspace(100, 110, 30)) + [111.0])
    df2.loc[df2.index[-1], ["Open", "Low", "High"]] = [110.0, 107.0, 111.1]
    df2 = ec.add_indicators(df2)
    cs2 = ec.candle_state(df2)
    assert any(p.startswith("아래꼬리 강") for p in cs2.patterns)


def test_fvg_detection_and_fill():
    closes = [100, 101, 102, 110, 111, 112, 113, 112, 111]
    df = _ohlc(closes)
    # 3번째→4번째 캔들에서 갭: High[2]=102 < Low[4]=110 되도록 Low 조정
    df.loc[df.index[3], "Low"] = 108.0
    df.loc[df.index[4], "Low"] = 110.0
    fv = ec.fvg_list(df, lookback=20)
    hit = [f for f in fv if f.bullish and f.bottom == pytest.approx(102.0) and f.top == pytest.approx(110.0)]
    assert hit and 0.0 <= hit[0].fill <= 1.0


def test_fib_retracement_zone():
    # 100 → 200 상승 뒤 0.5 되돌림(150) 근처
    # 피벗은 "엄격한" 극값만 잡으므로 같은 값이 연속되지 않게 만든다
    closes = [105, 102, 100] + list(np.linspace(101.5, 200, 40)) + list(np.linspace(198, 150, 25)) + [151, 152]
    df = _ohlc(closes)
    df["Open"] = df["Close"]; df["High"] = df["Close"]; df["Low"] = df["Close"]   # 꼬리 없는 봉 (고점·저점이 겹치지 않게)
    df = ec.add_indicators(df)
    sw = ec.swing_points(df, 5)
    fb = ec.fib_state(df, sw)
    assert fb is not None and fb.kind.startswith("상승 되돌림")
    assert fb.zone.startswith("정석 매수")


def test_supply_box_found_on_sideways():
    rng = np.random.default_rng(0)
    closes = 100 + np.cumsum(rng.normal(0, 0.3, 200))
    closes = 100 + (closes - closes.mean()) * 0.5          # 좁은 횡보
    df = ec.add_indicators(_ohlc(closes))
    assert ec.supply_zones(df)


def test_evaluate_and_verdict_shapes():
    rng = np.random.default_rng(1)
    closes = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.02, 3000)))
    df = _ohlc(closes)
    reps = ec.evaluate(df, df.index[2500].date())
    assert set(reps) >= {"주봉", "일봉"}
    for r in reps.values():
        assert len(r.five) == 5 and 0 <= r.yes_count <= 5
    pts, parts = ec.weighted_score(reps["주봉"].five, True, True, 0.5, True)
    assert pts >= 0.75 + 0.5 + 0.5
    assert ec.verdict(4, 0, 0) == "베팅" and ec.verdict(1, 4, 0).startswith("단기") and ec.verdict(0, 0, 0) == "매매 안 함"
    fr = ec.forward_result(df, df.index[2500].date())
    assert fr["26주 수익"] is not None and fr["26주 MDD"] <= 0


def test_screen_summarize_and_filter():
    from strategy.adapters import evidence_screen as es
    rng = np.random.default_rng(2)
    closes = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.02, 2000)))
    df = _ohlc(closes)
    reps = ec.evaluate(df, df.index[-1].date())
    row = es.summarize(reps)
    assert {"주봉 예", "일봉 예", "자동 가중치", "①", "⑤", "4분면(주봉)"} <= set(row)
    assert es.passes({**row, "주봉 예": 5, "자동 가중치": 4.0, "4분면(주봉)": "x", "주봉 20MA 위": True}, es.ScreenFilter())
    assert not es.passes({**row, "주봉 예": 1}, es.ScreenFilter(min_weekly_yes=3))
    assert not es.passes({**row, "주봉 예": 5, "4분면(주봉)": "하락채널+정배열 (최악)"}, es.ScreenFilter())


def test_screen_apply_filter_and_prices_path():
    from datetime import date as _d
    from strategy.adapters import evidence_screen as es
    rng = np.random.default_rng(3)
    prices = {}
    for t in ("AAA", "BBB"):
        closes = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.02, 1500)))
        prices[t] = _ohlc(closes)
    as_of = prices["AAA"].index[-1].date()
    hit, all_df = es.run_screen(None, ["AAA", "BBB", "ZZZ"], as_of, prices=prices, f=es.ScreenFilter(min_weekly_yes=0, exclude_worst_quadrant=False))
    assert len(all_df) == 3 and all_df.loc[all_df["티커"] == "ZZZ", "오류"].iloc[0] == "데이터 없음"
    assert set(hit["티커"]) == {"AAA", "BBB"}
    assert len(es.apply_filter(all_df, es.ScreenFilter(min_weekly_yes=6))) == 0
