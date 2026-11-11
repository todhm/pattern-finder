"""VWAP 지지·저항 돌파 엔진 — 합성 15분봉으로 VWAP 리셋·박스·진입·손절·익절 검증."""
import numpy as np
import pandas as pd

from strategy.adapters import vwap_sr_breakout as vs


def _bars(closes, vols=None, day="2026-03-02", spread=0.2, start_time="09:30", hl="oc"):
    """종가 목록 → 15분봉 프레임(한 거래일). 고가/저가는 max/min(시가, 종가) ± spread (hl="c" 면 종가 ± spread — 스윙 피벗 테스트용)."""
    n = len(closes)
    idx = pd.date_range(f"{day} {start_time}", periods=n, freq="15min", tz="America/New_York")
    c = np.asarray(closes, dtype=float)
    o = np.r_[c[0], c[:-1]]
    v = np.full(n, 1000.0) if vols is None else np.asarray(vols, dtype=float)
    hi, lo = (np.maximum(o, c), np.minimum(o, c)) if hl == "oc" else (c, c)
    return pd.DataFrame({"Open": o, "High": hi + spread, "Low": lo - spread, "Close": c, "Volume": v}, index=idx)


def test_session_vwap_resets_each_day():
    a = _bars([10, 11, 12], day="2026-03-02"); b = _bars([20, 21], day="2026-03-03")
    df = pd.concat([a, b])
    v = vs.session_vwap(df)
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    assert abs(v.iloc[0] - tp.iloc[0]) < 1e-9
    assert abs(v.iloc[3] - tp.iloc[3]) < 1e-9            # 둘째 날 첫 봉 = 그 봉의 typical price (리셋)
    assert abs(v.iloc[2] - tp.iloc[:3].mean()) < 1e-9    # 같은 거래량이면 단순 평균


def _box_then_breakout(n_box=16, tail=None, vols_tail=None):
    """박스(100 ± 0.3 왕복, 저항 100.3 을 여러 번 터치) 뒤 돌파·추세."""
    box = [100.0 + (0.3 if i % 2 else -0.3) for i in range(n_box)]
    tail = tail or [101.0, 101.5, 102.0, 102.6, 103.2, 103.0]
    closes = [100.0] * 4 + box + tail                  # 앞 4봉은 ATR 워밍업
    vols = [1000.0] * (4 + n_box) + (vols_tail or [1500.0] * len(tail))
    return _bars(closes, vols)


def test_long_breakout_enters_and_takes_profit_on_opposite_candle():
    df = _box_then_breakout()
    p = vs.VwapSrParams(box_bars=16, atr_bars=8, vol_avg_bars=8, min_touches=2, max_box_atr=3.0, max_vwap_dist_atr=3.0, sr_mode="box",
                        tp_min_dist_atr=1.0, tp_min_dist_pct=0, tp_opp_body_atr=0, tp_volume_mult=0, tp_wick_frac=0, tp_big_bar_atr=0, session_flat=False, fee_bps=0, risk_pct=0)
    r = vs.run_vwap_sr(df, p)
    assert len(r.trades) == 1
    t = r.trades[0]
    assert t["side"] == "롱" and t["kind"] == "박스" and t["entry"] == 101.0 and t["stop"] < t["vwap_entry"]
    assert t["why"] == "익절·역캔들" and t["exit"] == 103.0 and t["ret"] > 0
    assert abs(r.stats["총 수익률"] - (103.0 / 101.0 - 1)) < 1e-9


def test_stop_loss_on_close_below_vwap_only():
    # 돌파 뒤 윅만 손절선 아래로 찍고 종가는 위 → 손절 안 함, 그 다음 종가 이탈 → 손절
    df = _box_then_breakout(tail=[101.0, 101.2, 97.0])
    df.loc[df.index[-2], "Low"] = 95.0                  # 윅
    p = vs.VwapSrParams(box_bars=16, atr_bars=8, vol_avg_bars=8, max_box_atr=3.0, max_vwap_dist_atr=3.0, max_entry_bar_atr=50,
                        tp_min_dist_atr=99, session_flat=False, fee_bps=0, risk_pct=0, stop_on_close=True)
    r = vs.run_vwap_sr(df, p)
    assert len(r.trades) == 1 and r.trades[0]["why"] == "손절" and r.trades[0]["exit"] == 97.0
    p2 = vs.VwapSrParams(**{**p.to_dict(), "stop_on_close": False})
    r2 = vs.run_vwap_sr(df, p2)
    assert r2.trades[0]["exit_time"] == df.index[-2]   # 윅 터치로 한 봉 먼저 손절


def test_volume_spike_exit_and_no_chase_big_bar():
    df = _box_then_breakout(tail=[101.0, 101.5, 102.5, 103.0], vols_tail=[1500, 1500, 9000, 1500])
    p = vs.VwapSrParams(box_bars=16, atr_bars=8, vol_avg_bars=8, max_box_atr=3.0, max_vwap_dist_atr=3.0, tp_min_dist_atr=0.5,
                        tp_volume_mult=3.0, tp_opposite_candle=False, tp_wick_frac=0, tp_big_bar_atr=0, session_flat=False, fee_bps=0, risk_pct=0)
    r = vs.run_vwap_sr(df, p)
    assert r.trades and r.trades[0]["why"] == "익절·거래량 스파이크" and r.trades[0]["exit"] == 102.5
    # 돌파 봉이 ATR 의 50배짜리 큰 봉이면 추격하지 않는다
    df2 = _box_then_breakout(tail=[130.0, 131.0, 132.0])
    r2 = vs.run_vwap_sr(df2, vs.VwapSrParams(**{**p.to_dict(), "max_entry_bar_atr": 2.0}))
    assert not r2.trades


def test_short_symmetry_and_session_flat():
    box = [100.0 + (0.3 if i % 2 else -0.3) for i in range(16)]
    closes = [100.0] * 4 + box + [99.0, 98.5, 98.0]
    df = _bars(closes)
    p = vs.VwapSrParams(box_bars=16, atr_bars=8, vol_avg_bars=8, max_box_atr=3.0, max_vwap_dist_atr=3.0, tp_min_dist_atr=99,
                        session_flat=True, fee_bps=0, risk_pct=0)
    r = vs.run_vwap_sr(df, p)
    assert len(r.trades) == 1 and r.trades[0]["side"] == "숏" and r.trades[0]["why"] == "장 마감" and r.trades[0]["ret"] > 0   # True → daily
    assert r.trades[0]["stop"] > r.trades[0]["vwap_entry"]
    r_long_only = vs.run_vwap_sr(df, vs.VwapSrParams(**{**p.to_dict(), "allow_short": False}))
    assert not r_long_only.trades


def test_risk_sizing_caps_at_full_capital_and_equity_curve_is_mtm():
    df = _box_then_breakout()
    p = vs.VwapSrParams(box_bars=16, atr_bars=8, vol_avg_bars=8, max_box_atr=3.0, max_vwap_dist_atr=3.0, tp_min_dist_atr=1.0, tp_min_dist_pct=0, tp_opp_body_atr=0,
                        tp_volume_mult=0, tp_wick_frac=0, tp_big_bar_atr=0, session_flat=False, fee_bps=0, risk_pct=0.5)
    r = vs.run_vwap_sr(df, p, capital=10_000)
    t = r.trades[0]
    assert 0 < t["frac"] <= 1 and abs(t["pnl"] - t["frac"] * t["ret"]) < 1e-12
    eq = r.frame["equity"]
    assert abs(eq.iloc[-1] - 10_000 * (1 + t["pnl"])) < 1e-6 and (eq.iloc[:4] == 10_000).all()


def test_descending_trendline_breaks_before_horizontal_box():
    # 고점이 내려오는 쐐기: 스윙 고점 102.0 → 101.4 → 100.8, 저점은 99 근처 평평. 종가 101.2 는 수평 박스 윗선(102.0)엔 못 미치지만 내려온 저항선은 넘는다
    closes = [98.0] * 4                      # 아침에 낮게 시작해 VWAP 이 쐐기 아래(≈100)에 있게 한다
    for peak in (102.0, 101.4, 100.8):
        closes += [100.0, 100.6, peak, 100.3, 99.3]
    closes += [100.2, 101.2, 101.6, 101.4]
    df = _bars(closes, spread=0.05, hl="c")
    base = dict(box_bars=15, pivot_lr=2, atr_bars=8, vol_avg_bars=8, min_touches=2, max_box_atr=6.0, max_vwap_dist_atr=5.0, max_entry_bar_atr=50, sr_price="wick",
                breakout_volume_mult=0, tp_min_dist_atr=99, session_flat=False, fee_bps=0, risk_pct=0, allow_short=False)
    r_tr = vs.run_vwap_sr(df, vs.VwapSrParams(sr_mode="trend", **base))
    r_bx = vs.run_vwap_sr(df, vs.VwapSrParams(sr_mode="box", **base))
    assert r_tr.trades and r_tr.trades[0]["kind"] == "추세선" and r_tr.trades[0]["entry"] == 101.2
    assert r_tr.trades[0]["res_geo"]["slope"] < 0 and r_tr.trades[0]["box_high"] < 102.0   # 기울어진 선, 진입 시 저항값은 내려온 값
    assert not r_bx.trades                                                                 # 수평 박스(102.0)는 안 뚫렸다
    f = vs.signal_frame(df, vs.VwapSrParams(sr_mode="both", **base))
    assert f["long_kind"].iloc[-3] == "추세선"


def test_structure_segments_group_consecutive_bars():
    closes = [98.0] * 4
    for peak in (102.0, 101.4, 100.8):
        closes += [100.0, 100.6, peak, 100.3, 99.3]
    closes += [100.2, 101.2, 101.6, 101.4]
    df = _bars(closes, spread=0.05, hl="c")
    p = vs.VwapSrParams(box_bars=15, pivot_lr=2, atr_bars=8, vol_avg_bars=8, min_touches=2, max_box_atr=6.0, allow_short=False, sr_price="wick")
    segs = vs.structure_segments(vs.signal_frame(df, p), p)
    res = [sg for sg in segs if sg["kind"] == "res"]
    # 피벗 6·11·16 이 한 직선 위라 터치 3 인 (6,11) 선이 끝까지 하나로 이어진다 (돌파 봉 20 까지)
    assert len(res) == 1 and res[0]["i0"] == 6 and res[0]["i1"] == 20 and res[0]["slope"] < 0 and res[0]["touches"] == 3 and res[0]["ok"]
    assert all(sg["i1"] >= sg["i0"] for sg in segs)


def test_overlapping_lines_prefer_more_retests():
    # 같은 높이(≈102)를 세 번 두드린 평평한 저항 + 그 바로 아래를 두 번 두드린 선: 두 선이 tol 안에서 겹치면 터치 3 인 선을 고른다
    closes = [98.0] * 4
    for peak in (102.0, 101.95, 102.05, 101.9, 101.85):
        closes += [100.0, 100.6, peak, 100.3, 99.3]
    closes += [100.2, 100.5]
    df = _bars(closes, spread=0.05, hl="c")
    p = vs.VwapSrParams(box_bars=15, pivot_lr=2, atr_bars=8, vol_avg_bars=8, min_touches=2, touch_tol_atr=0.3, trend_bars=60, allow_short=False, sr_price="wick")
    f = vs.signal_frame(df, p)
    last = f.iloc[-1]
    assert not np.isnan(last["res"]) and last["res_touch"] >= 4                 # 다섯 고점이 모두 tol 안이라 한 선으로 묶여 리테스트 4~5
    assert int(last["res_i"]) == 6                                              # 가장 긴(첫 피벗부터) 선


def test_weekend_flat_only_before_gap_and_body_lines_ignore_wicks():
    # 목(03-05)·금(03-06)·월(03-09): 목요일 마지막 봉엔 안 팔고 금요일 마지막 봉에서 "주말 전 청산"
    thu = _box_then_breakout(tail=[101.0, 101.2, 101.3]); thu.index = thu.index + pd.Timedelta(days=3)      # 2026-03-05 목
    fri = _bars([101.4, 101.5, 101.6], day="2026-03-06"); mon = _bars([101.7, 101.8], day="2026-03-09")
    df = pd.concat([thu, fri, mon])
    p = vs.VwapSrParams(box_bars=16, atr_bars=8, vol_avg_bars=8, max_box_atr=3.0, max_vwap_dist_atr=3.0, max_entry_bar_atr=50, sr_mode="box",
                        tp_min_dist_atr=99, tp_min_dist_pct=99, stop_mode="box", session_flat="weekend", fee_bps=0, risk_pct=0, allow_short=False)
    r = vs.run_vwap_sr(df, p)
    assert r.trades and r.trades[0]["why"] == "주말 전 청산" and r.trades[0]["exit_time"] == fri.index[-1]
    r2 = vs.run_vwap_sr(df, vs.VwapSrParams(**{**p.to_dict(), "session_flat": "daily"}))
    assert r2.trades[0]["why"] == "장 마감" and r2.trades[0]["exit_time"] == thu.index[-1]
    # 몸통 기준: 박스 안 한 봉에 긴 윗꼬리를 달아도 박스 윗선(몸통)은 그대로, 꼬리 기준이면 올라간다
    df2 = _box_then_breakout(); df2.loc[df2.index[10], "High"] = 110.0
    fb = vs.signal_frame(df2, vs.VwapSrParams(box_bars=16, atr_bars=8, sr_price="body")); fw = vs.signal_frame(df2, vs.VwapSrParams(box_bars=16, atr_bars=8, sr_price="wick"))
    assert fb["box_high"].iloc[20] < 103 and fw["box_high"].iloc[20] == 110.0


def test_clean_wicks_clips_only_outliers():
    df = _box_then_breakout()
    df.loc[df.index[5], "Low"] = 50.0; df.loc[df.index[6], "High"] = 200.0; df.loc[df.index[7], "High"] = df["Close"].iloc[7] + 0.5   # 0.5% 안은 그대로
    out, n = vs.clean_wicks(df, 1.0)
    assert n == 2 and out["Low"].iloc[5] == min(df["Open"].iloc[5], df["Close"].iloc[5]) and out["High"].iloc[6] == max(df["Open"].iloc[6], df["Close"].iloc[6])
    assert out["High"].iloc[7] == df["High"].iloc[7] and vs.clean_wicks(df, 0)[1] == 0
