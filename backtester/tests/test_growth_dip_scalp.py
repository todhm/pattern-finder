"""성장주 당일 눌림 엔진 — 합성 데이터로 체결 경계·리스트·플라세보 뼈대 검증."""
from datetime import date

import numpy as np
import pandas as pd

from strategy.adapters import growth_dip_scalp as gs


def _row(o, h, l, c, pc, sma20=np.nan):
    return pd.Series({"Open": o, "High": h, "Low": l, "Close": c, "prev_close": pc, "sma20": sma20})


def test_day_trade_three_fill_bounds():
    r = gs.ScalpRules(trigger="prev_close", x_pct=3.0, tp_pct=1.0, sl_pct=1.0, fee_side=0.0)
    # 전일 100, 오늘 시가 100, 저가 96(손절·익절 모두 가능 범위), 고가 99, 종가 97.5 → 진입 97
    row = _row(100, 99, 96, 97.5, 100)
    opt = gs.day_trade(row, gs.ScalpRules(**{**r.__dict__, "fill": "optimistic"}))
    neu = gs.day_trade(row, gs.ScalpRules(**{**r.__dict__, "fill": "neutral"}))
    pes = gs.day_trade(row, gs.ScalpRules(**{**r.__dict__, "fill": "pessimistic"}))
    assert opt[1] == "익절" and abs(opt[0] - 0.01) < 1e-9          # 고가 99 ≥ 97.97
    assert neu[1] == "손절" and abs(neu[0] + 0.01) < 1e-9          # 저가 96 ≤ 96.03 → 손절 먼저
    assert pes[1] == "손절"
    # 손절 안 닿고 종가 청산: 저가 96.5, 종가 97.2
    row2 = _row(100, 97.8, 96.5, 97.2, 100)
    pes2 = gs.day_trade(row2, gs.ScalpRules(**{**r.__dict__, "fill": "pessimistic"}))
    assert pes2[1] == "종가" and abs(pes2[0] - (97.2 / 97 - 1)) < 1e-9
    # 트리거 미달
    assert gs.day_trade(_row(100, 101, 97.5, 100, 100), r) is None
    # 갭 하락: 시가가 트리거 아래면 시가 진입, skip_gap_below 면 건너뜀
    gap = _row(95, 97, 94.5, 96, 100)      # 손절가 94.05 는 안 닿고 익절가 95.95 는 닿음
    assert gs.day_trade(gap, r)[1] == "익절"
    assert gs.day_trade(gap, gs.ScalpRules(**{**r.__dict__, "skip_gap_below": True})) is None
    # 비용 반영
    rc = gs.day_trade(row, gs.ScalpRules(**{**r.__dict__, "fill": "optimistic", "fee_side": 0.0015}))
    assert abs(rc[0] - (0.01 - 0.003)) < 1e-9


def _synthetic(seed, n=400):
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.02, n)))
    o = np.r_[c[0], c[:-1]] * (1 + rng.normal(0, 0.004, n))
    idx = pd.bdate_range("2022-01-03", periods=n)
    return pd.DataFrame({"Open": o, "High": np.maximum(o, c) * 1.01, "Low": np.minimum(o, c) * 0.98, "Close": c, "Volume": 2e6}, index=idx)


def test_run_sim_lists_and_placebos():
    prices = {f"T{i}": _synthetic(i) for i in range(10)}
    cal = pd.to_datetime(prices["T0"].index)
    data = gs.precompute(prices, cal)
    idx = gs.precompute({"Q": _synthetic(99)}, cal)["Q"]
    # 합성 400일은 200일선이 늦게 생기므로 추세 필터(기본 sma200+20)는 끈다
    rules = gs.ScalpRules(trigger="prev_close", x_pct=2.0, min_dollar_vol_m=1.0, min_price=1.0, index_filter=False, trend_filter=None)
    start, end = cal[60], cal[-1]
    lists = gs.weekly_lists(list(data), cal, lambda t, d: t in ("T0", "T1", "T2", "T3"), data, rules, start, end)
    assert lists and all(v <= {"T0", "T1", "T2", "T3"} for v in lists.values())
    res = gs.run_sim(data, cal, idx, lists, rules, start, end)
    assert len(res.equity) == len(cal[(cal >= start) & (cal <= end)])
    assert res.stats["거래 수"] > 0 and all(t["ticker"] in {"T0", "T1", "T2", "T3"} for t in res.trades)
    assert res.stats["세후"] <= res.stats["수익률"] + 1e-12
    rng = np.random.default_rng(0)
    pa = gs.run_sim(data, cal, idx, lists, rules, start, end, picker=gs.placebo_a_picker(rng, data, rules))
    assert all(t["ticker"] not in {"T0", "T1", "T2", "T3"} for t in pa.trades)
    pb = gs.run_sim(data, cal, idx, lists, rules, start, end, placebo_b=np.random.default_rng(1))
    assert pb.stats["거래 수"] > 0
    combos = gs.expand_grid(rules, {"x_pct": [2, 3], "sl_pct": [1.0, None]})
    assert len(combos) == 4 and {c.sl_pct for c in combos} == {1.0, None}


def test_day_trade_intraday_orders_fills_correctly():
    idx = pd.date_range("2026-10-20 09:30", "2026-10-20 15:55", freq="5min", tz="America/New_York")
    tm = list(idx.strftime("%H:%M")); n = len(idx)
    k_trig, k_tp = tm.index("10:30"), tm.index("11:00")
    px = np.where(np.arange(n) >= k_trig, 97.5, 100.0)          # 트리거 뒤에는 97.5 부근 횡보 (고가 97.7 < 익절 97.97, 저가 97.3 > 손절 96.03)
    bars = pd.DataFrame({"Open": px, "High": px + 0.2, "Low": px - 0.2, "Close": px, "Volume": 1e4}, index=idx)
    bars.iloc[k_trig, bars.columns.get_loc("Low")] = 97.0          # 10:30 저가 97 = 전일 100 의 −3% 트리거 → 지정가 97 체결
    bars.iloc[k_tp, bars.columns.get_loc("High")] = 98.5           # 11:00 고가 98.5 ≥ 97.97 익절
    r = gs.ScalpRules(trigger="prev_close", x_pct=3.0, tp_pct=1.0, sl_pct=1.0, fee_side=0.0)
    ret, why = gs.day_trade_intraday(bars, prev_close=100.0, rules=r)
    assert why == "익절" and abs(ret - 0.01) < 1e-9
    # 익절 전에 손절(96.03)이 먼저 오면 손절
    bars2 = bars.copy(); bars2.iloc[k_trig + 2, bars2.columns.get_loc("Low")] = 95.9
    ret2, why2 = gs.day_trade_intraday(bars2, 100.0, r)
    assert why2 == "손절" and abs(ret2 + 0.01) < 1e-9
    # 09:30 봉의 급락은 min_time 전이라 무시 → 트리거 없음
    bars3 = bars.copy(); bars3.iloc[k_trig, bars3.columns.get_loc("Low")] = 97.3; bars3.iloc[0, bars3.columns.get_loc("Low")] = 95.0
    assert gs.day_trade_intraday(bars3, 100.0, r) is None
    # 둘 다 안 오면 15:55 봉 시가(97.5) 청산
    bars4 = bars.copy(); bars4.iloc[k_tp, bars4.columns.get_loc("High")] = 97.7
    ret4, why4 = gs.day_trade_intraday(bars4, 100.0, r)
    assert why4 == "종가" and abs(ret4 - (97.5 / 97 - 1)) < 1e-9


def test_fast_sim_matches_loop_sim():
    prices = {f"T{i}": _synthetic(50 + i) for i in range(12)}
    cal = pd.to_datetime(prices["T0"].index)
    data = gs.precompute(prices, cal)
    idx = gs.precompute({"Q": _synthetic(77)}, cal)["Q"]
    start, end = cal[60], cal[-1]
    for rules in (gs.ScalpRules(trigger="prev_close", x_pct=2.0, min_dollar_vol_m=1.0, min_price=1.0, index_filter=True, fill="neutral"),
                  gs.ScalpRules(trigger="open", x_pct=1.5, sl_pct=None, min_dollar_vol_m=1.0, min_price=1.0, index_filter=True, fill="pessimistic"),
                  gs.ScalpRules(trigger="ma20", min_dollar_vol_m=1.0, min_price=1.0, index_filter=False, fill="optimistic")):
        lists = gs.weekly_lists(list(data), cal, lambda t, d: t in ("T0", "T1", "T2", "T3", "T4", "T5"), data, rules, start, end)
        slow = gs.run_sim(data, cal, idx, lists, rules, start, end)
        arr = gs.build_arrays(data, cal, idx, start, end)
        fast = gs.run_sim_fast(arr, gs.list_mask(arr, lists), gs.earnings_mask(arr, None), rules)
        assert slow.stats["거래 수"] == fast.stats["거래 수"] > 0
        assert abs(slow.stats["수익률"] - fast.stats["수익률"]) < 1e-9 and abs(slow.stats["세후"] - fast.stats["세후"]) < 1e-9
        assert [(t["ticker"], round(t["ret"], 9)) for t in slow.trades] == [(t["ticker"], round(t["ret"], 9)) for t in fast.trades]
    pa = gs.run_sim_fast(arr, gs.list_mask(arr, lists), gs.earnings_mask(arr, None), rules, mode="placebo_a", rng=np.random.default_rng(0))
    pb = gs.run_sim_fast(arr, gs.list_mask(arr, lists), gs.earnings_mask(arr, None), rules, mode="placebo_b", rng=np.random.default_rng(1))
    assert pa.stats["거래 수"] >= 0 and pb.stats["거래 수"] > 0


def test_intraday_fast_matches_loop_replay():
    idx = pd.date_range("2026-10-20 09:30", "2026-10-20 15:55", freq="5min", tz="America/New_York")
    tm = list(idx.strftime("%H:%M")); n = len(idx)
    k_trig, k_tp = tm.index("10:30"), tm.index("11:00")
    rng = np.random.default_rng(3)
    days = pd.DatetimeIndex([pd.Timestamp("2026-10-20")])
    tickers = [f"T{i}" for i in range(6)]
    intraday = {}; data = {}
    for j, t in enumerate(tickers):
        px = 100 + np.cumsum(rng.normal(0, 0.15, n))
        bars = pd.DataFrame({"Open": px, "High": px + 0.3, "Low": px - 0.3, "Close": px, "Volume": 1e4}, index=idx)
        if j % 2 == 0:
            bars.iloc[k_trig, bars.columns.get_loc("Low")] = 96.5
        intraday[t] = {days[0].date(): bars}
        data[t] = pd.DataFrame({"Open": 100.0, "High": 101.0, "Low": 96.5, "Close": 99.0, "prev_close": 100.0, "sma20": 98.0, "sma50": 97.0, "dvol20": 1e9}, index=days)
    arr = gs.Arrays(days, tickers, *[np.array([[float(data[t][c].iloc[0]) for t in tickers]]) for c in ("Open", "High", "Low", "Close", "prev_close", "sma20", "sma50", "dvol20")], np.array([np.nan]), np.array([np.nan]))
    b3 = gs.build_bars3d(arr, intraday, "5m")
    for rules in (gs.ScalpRules(trigger="prev_close", x_pct=3.0, tp_pct=1.0, sl_pct=1.0, fee_side=0.0, index_filter=False, min_dollar_vol_m=1, min_price=1),
                  gs.ScalpRules(trigger="open", x_pct=3.0, tp_pct=0.5, sl_pct=None, fee_side=0.0, index_filter=False, min_dollar_vol_m=1, min_price=1),
                  gs.ScalpRules(trigger="intraday_high", x_pct=2.0, tp_pct=0.5, sl_pct=1.0, fee_side=0.0, index_filter=False, min_dollar_vol_m=1, min_price=1)):
        trig, entry, ret, why = gs.intraday_outcomes(b3, arr.PC, rules)
        for j, t in enumerate(tickers):
            loop = gs.day_trade_intraday(intraday[t][days[0].date()], 100.0, rules)
            if loop is None:
                assert not trig[0, j]
            else:
                assert trig[0, j] and abs(loop[0] - ret[0, j]) < 1e-5 and loop[1] == gs._WHY[int(why[0, j])]


def test_swing_mode_holds_until_tp2_earnings_or_limit():
    days = pd.bdate_range("2024-01-02", periods=40)
    tickers = ["A", "B", "C"]
    T, N = len(days), 3
    O = np.full((T, N), 100.0); H = O + 0.5; L = O - 0.5; C = O.copy(); PC = O.copy()
    # 전부 첫날(인덱스 1) 저가 96 → −4% 트리거, 당일 익절 없음(고가 100.5 < 97×1.01=97.97? 아니, 100.5 ≥ 97.97 → 낙관이면 익절) → pessimistic 로 종가 기준
    L[1, :] = 96.0; C[1, :] = 97.0
    # A: 10일째 고가가 +10% 도달 / B: 실적 전 청산 / C: 기한 청산
    H[11, 0] = 110.0
    earn = {"B": [days[8]]}
    arr = gs.Arrays(days, tickers, O, H, L, C, PC, np.full((T, N), np.nan), np.full((T, N), np.nan), np.full((T, N), 1e9), np.full(T, np.nan), np.full(T, np.nan))
    rules = gs.ScalpRules(trigger="prev_close", x_pct=3.0, tp_pct=1.0, sl_pct=None, fill="pessimistic", index_filter=False, min_dollar_vol_m=1, min_price=1,
                          hold_mode="swing", tp2_pct=10.0, max_hold_days=20, earnings_exit_days_before=1, fee_side=0.0)
    ee = gs.next_earnings_exit(arr, earn, 1)
    assert ee[1, 1] == 7 and ee[1, 0] == T
    lm = np.ones((T, N), dtype=bool); em = np.zeros((T, N), dtype=bool)
    res = gs.run_swing(arr, lm, em, rules, ee)
    by = {t["ticker"]: t for t in res.trades}
    assert by["A"]["why"] == "익절2" and abs(by["A"]["ret"] - 0.10) < 1e-9 and by["A"]["hold"] == 10
    assert by["B"]["why"] == "실적 전" and by["B"]["exit"] == days[7]
    assert by["C"]["why"] == "기한" and by["C"]["hold"] == 20
    # day 모드는 전부 당일 종가 청산
    res_d = gs.run_swing(arr, lm, em, gs.ScalpRules(**{**rules.__dict__, "hold_mode": "day"}), ee)
    assert all(t["hold"] == 0 for t in res_d.trades)


def test_open_trigger_and_swing_cash_accounting():
    days = pd.bdate_range("2024-01-02", periods=6)
    O = np.array([[100.0]] * 6); H = O + 3; L = O - 4; C = O - 1; PC = O.copy()   # 매일 시가 100 → 저가 96 (−4%) → 익절 97×1.01 = 97.97 ≤ 고가 103
    arr = gs.Arrays(days, ["A"], O, H, L, C, PC, np.full((6, 1), np.nan), np.full((6, 1), np.nan), np.full((6, 1), 1e9), np.full(6, np.nan), np.full(6, np.nan), np.full(6, np.nan))
    rules = gs.ScalpRules(trigger="open", x_pct=3.0, tp_pct=1.0, sl_pct=None, fill="optimistic", index_filter=False, min_dollar_vol_m=1, min_price=1, fee_side=0.0, position_frac=0.5)
    lm = np.ones((6, 1), dtype=bool); em = np.zeros((6, 1), dtype=bool)
    res = gs.run_swing(arr, lm, em, rules, np.full((6, 1), 6))
    assert len(res.trades) == 6 and all(t["why"] == "익절1" for t in res.trades)
    # 하루 +1% × 비중 0.5 = +0.5%/일, 6일 복리 — 원금이 두 번 더해지지 않는다
    assert abs(res.equity.iloc[-1] - 1.005 ** 6) < 1e-9


def test_bars3d_drops_mismatched_price_basis():
    """티커 재사용 종목(GEN=옛 SYMC)처럼 분봉이 일봉과 다른 가격 기준이면 그 날 분봉은 버린다."""
    days = pd.bdate_range("2024-01-02", periods=3)
    def daily(o):
        return pd.DataFrame({"Open": o, "High": [x * 1.02 for x in o], "Low": [x * 0.98 for x in o], "Close": o, "Volume": 1e6,
                             "prev_close": o, "sma20": o, "sma50": o, "dvol20": 50.0}, index=days)
    data = {"AAA": daily([100.0, 100.0, 100.0])}
    arr = gs.build_arrays(data, days, None, days[0], days[-1])
    def bars(px):
        idx = pd.date_range("2024-01-02 09:30", periods=78, freq="5min", tz="America/New_York")
        return pd.DataFrame({"Open": px, "High": px * 1.001, "Low": px * 0.999, "Close": px}, index=idx)
    good = bars(100.0); bad = bars(5.0)                      # 5.0 = 다른 회사 가격
    bad.index = bad.index + pd.Timedelta(days=1)
    b3 = gs.build_bars3d(arr, {"AAA": {days[0].date(): good, days[1].date(): bad}}, "5m")
    assert b3.has[0, 0] and not b3.has[1, 0] and b3.dropped == 1


def _mk_arr(days, o, h, l, c, vol=None, vol20=None):
    n = len(days)
    df = pd.DataFrame({"Open": o, "High": h, "Low": l, "Close": c, "Volume": vol if vol is not None else [1e6] * n,
                       "prev_close": [o[0]] + list(c[:-1]), "sma20": c, "sma50": c, "dvol20": 1e9, "sma200": c,
                       "vol20": vol20 if vol20 is not None else [1e6] * n}, index=days)
    return gs.build_arrays({"AAA": df}, days, None, days[0], days[-1])


def test_close_pos_stop_exits_same_day_at_close():
    """당일 +1% 미도달 + 종가가 범위 하위 1/3 → 그날 종가 손절 (why=종가 손절, hold 0)."""
    days = pd.bdate_range("2024-01-02", periods=4)
    # 1일차: 시가 100, 저가 95(트리거 open −3% = 97 터치), 고가 100.5(익절 98×1.01=98.98 도달 → 익절). 2일차: 다시 트리거, 고가 97.5 < 익절, 종가 95.5 (close_pos = 0.5/5 = 0.1)
    arr = _mk_arr(days, o=[100, 100, 100, 100], h=[100.5, 97.5, 101, 101], l=[95, 95, 99, 99], c=[99, 95.5, 100, 100])
    rules = gs.ScalpRules(trigger="open", x_pct=3, index_filter=False, fill="pessimistic", hold_mode="swing", close_pos_stop=0.33,
                          min_dollar_vol_m=0, min_price=0)
    lm = np.ones((4, 1), bool); em = np.zeros((4, 1), bool); ee = np.full((4, 1), 4)
    res = gs.run_swing(arr, lm, em, rules, ee)
    whys = [t["why"] for t in res.trades]
    assert "종가 손절" in whys
    t = [t for t in res.trades if t["why"] == "종가 손절"][0]
    assert t["hold"] == 0 and abs(t["ret"] - (95.5 / 97 - 1 - 0.003)) < 1e-9
    # 필터 없으면 보유로 넘어간다
    res2 = gs.run_swing(arr, lm, em, gs.ScalpRules(**{**gs.asdict(rules), "close_pos_stop": None}), ee)
    assert "종가 손절" not in [t["why"] for t in res2.trades]


def test_max_vol_ratio_blocks_high_volume_day():
    days = pd.bdate_range("2024-01-02", periods=3)
    arr = _mk_arr(days, o=[100, 100, 100], h=[100.5, 100.5, 100.5], l=[95, 95, 95], c=[99, 99, 99], vol=[1e6, 3e6, 1e6], vol20=[1e6] * 3)
    base = dict(trigger="open", x_pct=3, index_filter=False, fill="pessimistic", hold_mode="swing", min_dollar_vol_m=0, min_price=0)
    lm = np.ones((3, 1), bool); em = np.zeros((3, 1), bool); ee = np.full((3, 1), 3)
    n_all = len(gs.run_swing(arr, lm, em, gs.ScalpRules(**base, max_vol_ratio=None), ee).trades)
    n_f = len(gs.run_swing(arr, lm, em, gs.ScalpRules(**base, max_vol_ratio=1.5), ee).trades)
    assert n_all == 3 and n_f == 2


def test_trend_ok_sma200_plus():
    row = pd.Series({"Close": 125.0, "sma50": 100.0, "sma200": 100.0})
    assert gs._trend_ok(row, "sma200") and gs._trend_ok(row, "sma200+20") and not gs._trend_ok(row, "sma200+30")


def test_per_columns_point_in_time_and_per_own_trigger_hold():
    days = pd.bdate_range("2024-01-01", periods=60)
    T = len(days)
    close = np.full(T, 100.0)
    df = pd.DataFrame({"Open": close, "High": close + 0.5, "Low": close - 0.5, "Close": close, "Volume": 1e6}, index=days)
    # 분기 EPS 1.0 씩, 발표일은 모두 첫날 이전 → TTM 4.0 → PER 25. 다섯 번째 분기(EPS 2.0)가 30일째 발표 → TTM 5.0 → PER 20 (가격은 그대로)
    rep = np.array([pd.Timestamp(d).value for d in ("2023-01-01", "2023-04-01", "2023-07-01", "2023-10-01", str(days[30].date()))])
    gs.per_columns(df, rep, np.array([1.0, 1.0, 1.0, 1.0, 2.0]), lookback=20)
    assert abs(df["ttm_eps"].iloc[0] - 4.0) < 1e-9 and abs(df["per"].iloc[0] - 25.0) < 1e-9
    assert abs(df["ttm_eps"].iloc[30] - 5.0) < 1e-9 and abs(df["per"].iloc[30] - 20.0) < 1e-9
    assert np.isnan(df["per_ref"].iloc[9]) and abs(df["per_ref"].iloc[11] - 25.0) < 1e-9     # 전날까지 창 10 개 이상부터, 하루 밀림
    # 레벨 = 25 × 0.8 × 4 = 80: 40일째 저가 79 (전일 종가 100 > 80) → per_own 교차 트리거. 이후 PER 이 회복(종가 ≥ 25 × 4 = 100)되는 50일째 청산
    O = np.full((T, 1), 100.0); H = O + 0.5; L = O - 0.5; C = O.copy(); PC = O.copy()
    L[40, 0] = 79.0; C[40:50, 0] = 85.0; PC[41:51, 0] = 85.0; O[41:50, 0] = 85.0; H[41:50, 0] = 85.5; L[41:50, 0] = 84.5
    EPS = np.full((T, 1), 4.0); PREF = np.full((T, 1), 25.0)
    arr = gs.Arrays(days, ["A"], O, H, L, C, PC, np.full((T, 1), np.nan), np.full((T, 1), np.nan), np.full((T, 1), 1e9), np.full(T, np.nan), np.full(T, np.nan),
                    np.full(T, np.nan), None, None, None, EPS, PREF)
    rules = gs.ScalpRules(trigger="per_own", x_pct=20.0, hold_mode="per", per_exit_pct=0.0, max_hold_days=0, fill="pessimistic", index_filter=False,
                          min_dollar_vol_m=1, min_price=1, fee_side=0.0, max_vol_ratio=None, min_days_to_earnings=0)
    lvl, trig = gs._daily_level(arr, rules, np.ones((T, 1), dtype=bool))
    assert abs(lvl[40, 0] - 80.0) < 1e-9 and trig[40, 0] and not trig[41, 0]                    # 41일째는 전일 종가 85 > 80 이지만 저가 84.5 > 80
    lm = np.ones((T, 1), dtype=bool); em = np.zeros((T, 1), dtype=bool)
    res = gs.run_swing(arr, lm, em, rules, np.full((T, 1), T))
    assert len(res.trades) == 1
    tr = res.trades[0]
    assert tr["day"] == days[40] and tr["why"] == "PER 회복" and tr["exit"] == days[50] and abs(tr["ret"] - (100 / 80 - 1)) < 1e-9
    # per_exit_pct=20 → 목표 PER 20 → 종가 85 (PER 21.25) 로 다음날 바로 회복
    r2 = gs.ScalpRules(**{**rules.__dict__, "per_exit_pct": 20.0})
    assert gs.run_swing(arr, lm, em, r2, np.full((T, 1), T)).trades[0]["exit"] == days[41]
    # EPS 소멸(적자 전환) → 그날 종가 청산
    EPS2 = EPS.copy(); EPS2[45:, 0] = np.nan
    arr2 = gs.Arrays(**{**arr.__dict__, "EPS": EPS2})
    assert gs.run_swing(arr2, lm, em, rules, np.full((T, 1), T)).trades[0]["why"] == "PER 소멸"
    # per_own_any 는 레벨 아래 모든 날 트리거 (교차 무관)
    L3 = L.copy(); L3[41, 0] = 79.0
    arr3 = gs.Arrays(**{**arr.__dict__, "L": L3})
    _, t3 = gs._daily_level(arr3, gs.ScalpRules(**{**rules.__dict__, "trigger": "per_own_any"}), np.ones((T, 1), dtype=bool))
    assert t3[40, 0] and t3[41, 0]


def test_checklist_filters_block_entries():
    T = 30; days = pd.bdate_range("2024-01-01", periods=T)
    O = np.full((T, 2), 100.0); H = O + 0.5; L = O - 0.5; C = O.copy(); PC = O.copy()
    L[10, :] = 79.0                                                   # 둘 다 레벨 80 아래로 교차
    base = dict(days=days, tickers=["A", "B"], O=O, H=H, L=L, C=C, PC=PC, S20=np.full((T, 2), np.nan), S50=np.full((T, 2), np.nan), DV=np.full((T, 2), 1e9),
                idx_low=np.full(T, np.nan), idx_pc=np.full(T, np.nan), idx_open=np.full(T, np.nan), EPS=np.full((T, 2), 4.0), PREF=np.full((T, 2), 25.0),
                PXMED=np.column_stack([np.full(T, 120.0), np.full(T, 100.0)]),      # A: 전일 종가 100 ≤ 120×0.9 통과 / B: 100 > 90 탈락
                S200=np.column_stack([np.full(T, 120.0), np.full(T, 100.0)]),
                PWMAX=np.column_stack([np.full(T, 30.0), np.full(T, 60.0)]),         # B: 60/25 = 2.4 ≥ 1.5 탈락
                LYOY=np.column_stack([np.full(T, 0.5), np.full(T, 3.0)]), DSR=np.column_stack([np.full(T, 20.0), np.full(T, 5.0)]))
    arr = gs.Arrays(**base)
    lm = np.ones((T, 2), dtype=bool); em = np.zeros((T, 2), dtype=bool)
    r0 = gs.ScalpRules(trigger="per_own", x_pct=20.0, hold_mode="per", max_hold_days=0, fill="pessimistic", index_filter=False, min_dollar_vol_m=1, min_price=1,
                       fee_side=0.0, max_vol_ratio=None, min_days_to_earnings=0)
    assert {t["ticker"] for t in gs.run_swing(arr, lm, em, r0, np.full((T, 2), T)).trades} == {"A", "B"}
    for kw in ({"px_below_med_pct": 10.0}, {"below_sma200_pct": 10.0}, {"per_ref_max_ratio": 1.5}, {"max_last_yoy": 1.0}, {"min_days_since_report": 13}):
        r = gs.ScalpRules(**{**r0.__dict__, **kw})
        assert {t["ticker"] for t in gs.run_swing(arr, lm, em, r, np.full((T, 2), T)).trades} == {"A"}, kw
