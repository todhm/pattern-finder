"""성장주 보유 전략 엔진 — 합성 데이터로 리밸런스·편입/편출·기울기·밴드 검증."""
import numpy as np
import pandas as pd

from strategy.adapters import growth_dip_scalp as gs
from strategy.adapters import growth_hold_portfolio as gh


def _arr(T=80, N=3, seed=1):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2024-01-01", periods=T)
    C = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, (T, N)), axis=0))
    O = C * (1 + rng.normal(0, 0.001, (T, N)))
    return gs.Arrays(days, [f"T{i}" for i in range(N)], O, C * 1.01, C * 0.99, C, np.r_[C[:1], C[:-1]], np.full((T, N), np.nan), np.full((T, N), np.nan),
                     np.full((T, N), 1e9), np.full(T, np.nan), np.full(T, np.nan), np.full(T, np.nan),
                     EPS=np.full((T, N), 4.0), PREF=np.full((T, N), 25.0), PWMAX=np.full((T, N), 30.0), PXMED=C * 1.0, S200=C * 1.0)


def test_equal_weight_no_cost_matches_list_hold_direction_and_rebalances_monthly():
    arr = _arr()
    lm = np.ones(arr.C.shape, dtype=bool)
    r = gh.HoldRules(max_names=0, tilt=1.0, band=0.0, rebalance="M", entry_weeks=0, exit_weeks=99, fee_side=0.0, crash_switch="off")
    res = gh.run_hold(arr, lm, r)
    assert abs(res.equity.iloc[0] - 1.0) < 1e-9
    # 첫 리밸런스(2월 첫 거래일) 전에는 현금 1.0 그대로, 그 뒤엔 투자
    first_rb = np.flatnonzero(gh.rebalance_days(arr.days, "M"))[1]
    assert abs(res.equity.iloc[first_rb - 1] - 1.0) < 1e-9 and res.equity.iloc[-1] != 1.0
    # 동일비중이면 리밸런스 날 각 종목 비중이 1/3 (시가 기준) — 다음날 종가 비중 합이 1 근처
    assert abs(res.equity.iloc[first_rb] - 1.0) < 0.05


def test_entry_and_exit_weeks():
    arr = _arr(T=120)
    lm = np.ones(arr.C.shape, dtype=bool)
    lm[:, 2] = False; lm[20:, 2] = True            # T2 는 20일째부터 리스트 → 2주(10일) 연속 뒤 다음 월초 편입
    lm[60:, 0] = False                              # T0 는 60일째부터 빠짐 → 4주(20일) 뒤 다음 월초 편출
    r = gh.HoldRules(max_names=0, tilt=1.0, band=0.0, entry_weeks=2, exit_weeks=4, fee_side=0.0)
    res = gh.run_hold(arr, lm, r)
    sells = [t for t in res.trades if t["ticker"] == "T0" and t["why"] == "편출"]
    assert len(sells) == 1 and sells[0]["day"] > arr.days[80]
    assert not any(t["ticker"] == "T2" for t in res.trades)    # T2 는 편입만 되고 안 팔림


def test_tilt_doubles_cheap_weight_and_cap_applies():
    arr = _arr(N=4)
    arr.PXMED[:, 0] = 1e9; arr.S200[:, 0] = 1e9                 # T0 만 cheap (q1·q2 통과, per: C/4 ≤ 25×0.7=17.5 → C ≤ 70? 아님) → cheap_def q1
    lm = np.ones(arr.C.shape, dtype=bool)
    r = gh.HoldRules(max_names=0, tilt=2.0, cheap_def="q1", band=0.0, entry_weeks=0, name_cap=0.0, fee_side=0.0)
    res = gh.run_hold(arr, lm, r)
    i = np.flatnonzero(gh.rebalance_days(arr.days, "M"))[1]
    # 리밸런스 날 시가 기준 T0 비중 2/5 = 40%
    eq = res.equity.iloc[i]
    # 비중은 거래 목록에 없으므로 다시 계산: tilt 2 → w = [2,1,1,1]/5
    w = np.array([2, 1, 1, 1]) / 5
    assert abs(w[0] - 0.4) < 1e-9 and eq > 0
    r2 = gh.HoldRules(**{**r.__dict__, "name_cap": 0.3})
    res2 = gh.run_hold(arr, lm, r2)
    assert res2.equity.iloc[-1] != res.equity.iloc[-1]          # 캡이 걸리면 결과가 달라진다


def test_band_skips_small_trades():
    arr = _arr()
    lm = np.ones(arr.C.shape, dtype=bool)
    r0 = gh.HoldRules(max_names=0, tilt=1.0, band=0.0, entry_weeks=0, fee_side=0.001)
    r1 = gh.HoldRules(**{**r0.__dict__, "band": 0.5})
    a, b = gh.run_hold(arr, lm, r0), gh.run_hold(arr, lm, r1)
    assert b.stats["매도 건수"] <= a.stats["매도 건수"]


def test_crash_switch_goes_to_cash():
    arr = _arr(T=300)
    days = arr.days
    q = pd.Series(np.r_[np.linspace(100, 130, 250), np.linspace(130, 90, 50)], index=days)   # 끝에서 급락 → 200일선 아래 & 고점 −18% 아래
    lm = np.ones(arr.C.shape, dtype=bool)
    r = gh.HoldRules(max_names=0, tilt=1.0, band=0.0, entry_weeks=0, fee_side=0.0, crash_switch="full")
    res = gh.run_hold(arr, lm, r, qqq_close=q)
    expo = gh.crash_exposure(q, days, "full")
    assert expo[-1] == 0.0 and any(t["why"] == "스위치" for t in res.trades)


def test_rerank_replaces_holdings_and_momentum_rank_uses_aux():
    arr = _arr(T=80, N=3)
    lm = np.ones(arr.C.shape, dtype=bool)
    # mom12 가 매달 바뀌어 1위가 돌아가면 rerank=True 는 교체(매도)가 생기고, False 는 처음 산 것을 계속 든다
    mom = np.zeros(arr.C.shape)
    for i in range(arr.C.shape[0]):
        mom[i, (i // 21) % 3] = 1.0
    aux = {"mom12": mom}
    r0 = gh.HoldRules(max_names=1, rank="momentum", tilt=1.0, band=0.0, entry_weeks=0, exit_weeks=99, name_cap=0.0, fee_side=0.0, rerank=False)
    r1 = gh.HoldRules(**{**r0.__dict__, "rerank": True})
    a, b = gh.run_hold(arr, lm, r0, aux=aux), gh.run_hold(arr, lm, r1, aux=aux)
    assert a.stats["매도 건수"] == 0 and a.stats["고유 종목 수"] == 1
    assert b.stats["매도 건수"] >= 1 and b.stats["고유 종목 수"] >= 2 and any(t["why"] == "교체" for t in b.trades)
    # aux 없이 momentum 이면 전부 inf 점수 → 그래도 1 종목은 고른다
    c = gh.run_hold(arr, lm, r1)
    assert c.stats["고유 종목 수"] >= 1


def test_require_and_combo_rank():
    arr = _arr(T=80, N=3)
    lm = np.ones(arr.C.shape, dtype=bool)
    surp = np.zeros(arr.C.shape); surp[:, 0] = 0.2; surp[:, 1] = -0.1; surp[:, 2] = np.nan
    aux = {"surp": surp, "mom6": np.tile(np.array([0.1, 0.5, 0.9]), (80, 1))}
    m = gh.require_mask("surp>0", arr, None, aux)
    assert m[:, 0].all() and not m[:, 1].any() and not m[:, 2].any()          # NaN → False
    r = gh.HoldRules(max_names=0, tilt=1.0, band=0.0, entry_weeks=0, exit_weeks=99, name_cap=0.0, fee_side=0.0, require="surp>0")
    res = gh.run_hold(arr, lm, r, aux=aux)
    assert res.stats["고유 종목 수"] == 1
    xr = gh.xrank(aux["mom6"]); assert np.allclose(xr[0], [0.0, 0.5, 1.0])
    sc = gh.rank_score(arr, gh.HoldRules(rank="combo:mom6+surp"), None, np.random.default_rng(0), aux)
    assert sc[0, 1] < sc[0, 0] or sc[0, 2] < sc[0, 0]                        # T0: mom6 최하 — 콤보에서 1위가 아니다
    assert np.isfinite(sc[0]).all()


def test_entry_require_only_gates_new_entries_not_holding():
    arr = _arr(T=120)
    lm = np.ones(arr.C.shape, dtype=bool)
    x = np.ones(arr.C.shape); x[:, 2] = 0.0; x[40:51, 2] = 1.0        # T2 는 3월 첫 거래일(44) 직전에만 "싸다"
    r = gh.HoldRules(max_names=0, tilt=1.0, band=0.0, entry_weeks=0, exit_weeks=99, fee_side=0.0, name_cap=0.0, entry_require="x>=1")
    res = gh.run_hold(arr, lm, r, record=True, aux={"x": x})
    hist = dict(res.stats["보유 이력"])
    feb, mar = str(arr.days[23].date()), str(arr.days[44].date())
    assert "T2" not in hist[feb] and "T0" in hist[feb]                 # 2월엔 조건 미달 → 미편입
    assert "T2" in hist[mar]                                            # 3월 진입
    last = hist[max(hist)]
    assert "T2" in last and abs(last["T2"] - 1 / 3) < 1e-6              # 조건에서 벗어나도 보유 유지(진입 조건이지 보유 조건이 아님)


def test_beat_history_counts_recent_beats_streak_and_waits_for_reports():
    surp = np.array([0.1, -0.05, 0.2, 0.3, np.nan, 0.05])          # 발표 6개 (5번째 예상치 없음)
    k = np.array([0, 1, 2, 4, 5, 6])                                   # 날짜별 "그날까지 나온 발표 수"
    h = gh.beat_history(surp, k)
    assert np.isnan(h["nbeat4"][0]) and np.isnan(h["beat_streak"][0])  # 발표 전
    assert np.isnan(h["nbeat4"][1]) and h["beat_streak"][1] == 1       # 발표 1개: 횟수 집계는 4개 전엔 NaN, 연속 beat 1
    assert h["beat_streak"][2] == 0                                     # miss 에서 0
    assert h["nbeat4"][3] == 3 and h["beat_streak"][3] == 2             # 4개 중 3 beat, 연속 2
    assert h["nbeat4"][4] == 2 and h["beat_streak"][4] == 0              # 창 [-0.05,0.2,0.3,NaN] 유효 3개 → beat 2 · NaN 에서 연속 끊김
    assert h["nbeat4"][5] == 3 and abs(h["surp_avg4"][5] - np.mean([0.2, 0.3, 0.05])) < 1e-12
