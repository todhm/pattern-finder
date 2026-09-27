"""minervini_screener 순수 로직 테스트 — 합성 데이터로 판정 검증."""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from data.domain.models import QuarterlyFinancials
from strategy.adapters.minervini_screener import (
    MIN_BARS,
    COMPOSITE_WEIGHTS,
    GrowthResult,
    PivotLevels,
    composite_score,
    distribution_days,
    evaluate_growth,
    evaluate_trend_template,
    momentum_components,
    momentum_score,
    pivot_levels,
    quarterly_evidence,
    replay_trade,
    rs_percentiles,
    simulate_forward,
    split_factor,
)


def _daily(closes: np.ndarray, start: str = "2019-01-01") -> pd.DataFrame:
    idx = pd.bdate_range(start, periods=len(closes))
    return pd.DataFrame(
        {
            "Open": closes * 0.999,
            "High": closes * 1.01,
            "Low": closes * 0.99,
            "Close": closes,
            "Volume": np.full(len(closes), 1_000_000.0),
        },
        index=idx,
    )


# ---- 추세 템플릿 ----------------------------------------------------------

def test_trend_template_passes_on_steady_uptrend():
    closes = np.linspace(10.0, 30.0, 300)
    tt = evaluate_trend_template(_daily(closes))
    assert tt is not None
    assert tt.passed, tt.checks
    assert tt.pct_above_low >= 0.30
    assert tt.pct_from_high >= -0.25


def test_trend_template_fails_on_downtrend():
    closes = np.linspace(30.0, 10.0, 300)
    tt = evaluate_trend_template(_daily(closes))
    assert tt is not None
    assert not tt.passed
    assert not tt.checks["1-1 주가>150·200일선"]


def test_trend_template_fails_deep_below_high():
    # 상승 후 40% 급락 — 1-7 (신고가 −25% 이내) 위반
    up = np.linspace(10.0, 30.0, 280)
    down = np.linspace(30.0, 18.0, 20)
    tt = evaluate_trend_template(_daily(np.concatenate([up, down])))
    assert tt is not None
    assert not tt.checks["1-7 신고가-25%내"]


def test_trend_template_requires_min_bars():
    closes = np.linspace(10.0, 20.0, MIN_BARS - 1)
    assert evaluate_trend_template(_daily(closes)) is None


# ---- RS -------------------------------------------------------------------

def test_momentum_score_sign_and_rank():
    up = momentum_score(_daily(np.linspace(10.0, 30.0, 300)))
    flat = momentum_score(_daily(np.full(300, 20.0)))
    down = momentum_score(_daily(np.linspace(30.0, 10.0, 300)))
    assert up > flat > down
    pct = rs_percentiles({"UP": up, "FLAT": flat, "DOWN": down})
    assert pct["UP"] > pct["FLAT"] > pct["DOWN"]
    assert pct["UP"] == pytest.approx(100.0)


# ---- 펀더멘털 (point-in-time) --------------------------------------------

def _quarters() -> list[QuarterlyFinancials]:
    """2017~2019 12개 분기 — EPS YoY 가속(40→50→60→110%), 매출 ~20%."""
    eps = [1.00, 1.00, 1.00, 1.00,          # 2017
           1.10, 1.20, 1.30, 1.40,          # 2018: YoY 10/20/30/40%
           1.54, 1.80, 2.08, 2.94]          # 2019: YoY 40/50/60/110%
    rev = [100, 100, 100, 100,
           110, 115, 120, 125,
           132, 139, 146, 154]
    fiscals = [date(2017 + i // 4, [3, 6, 9, 12][i % 4], 28) for i in range(12)]
    out = []
    for i, (f, e, r) in enumerate(zip(fiscals, eps, rev)):
        # 마지막 분기(2019-12)만 2020-02-15 발표 — point-in-time 테스트용
        report = date(2020, 2, 15) if i == 11 else f + timedelta(days=45)
        out.append(QuarterlyFinancials(
            symbol="T", fiscal_date=f, report_date=report,
            eps_actual=e, revenue=float(r), net_income=e * 10.0,
        ))
    return out


def test_growth_point_in_time_excludes_unreported_quarter():
    g = evaluate_growth(_quarters(), as_of=date(2020, 1, 2))
    # 2019-12 분기(발표 2020-02-15)는 제외 → 최신 YoY는 2019-09의 60%
    assert g.eps_yoy == pytest.approx(0.60)
    assert g.eps_accelerating is True
    assert g.checks["2-1 EPS YoY"] and g.checks["2-2 EPS 가속"]
    assert g.checks["2-3 매출"]      # 146/120 − 1 ≈ 21.7% ≥ 20%
    assert g.checks["2-4 마진 개선"]  # EPS(순익)가 매출보다 빠르게 성장
    assert g.passed and g.n_passed == 4


def test_growth_includes_quarter_after_report_date():
    g = evaluate_growth(_quarters(), as_of=date(2020, 3, 1))
    assert g.eps_yoy == pytest.approx(1.10)  # 2.94/1.40 − 1


def test_growth_no_data():
    g = evaluate_growth([], as_of=date(2020, 1, 2))
    assert not g.data_available and not g.passed


def test_growth_decelerating_fails_accel_check():
    qs = []
    eps = [1.0, 1.0, 1.0, 1.0, 2.0, 1.8, 1.5, 1.2]  # YoY 100→80→50→20%
    for i, e in enumerate(eps):
        f = date(2018 + i // 4, [3, 6, 9, 12][i % 4], 28)
        qs.append(QuarterlyFinancials(
            symbol="T", fiscal_date=f, report_date=f + timedelta(days=45),
            eps_actual=e, revenue=100.0, net_income=e,
        ))
    g = evaluate_growth(qs, as_of=date(2020, 3, 1))
    assert g.eps_accelerating is False
    assert not g.checks["2-2 EPS 가속"]


def test_momentum_components_match_score():
    df = _daily(np.linspace(10.0, 30.0, 300))
    comps = momentum_components(df)
    assert set(comps) == {"3m", "6m", "9m", "12m"}
    assert momentum_score(df) == pytest.approx(
        2 * comps["3m"] + comps["6m"] + comps["9m"] + comps["12m"]
    )


def test_quarterly_evidence_point_in_time():
    rows = quarterly_evidence(_quarters(), as_of=date(2020, 1, 2), n=4)
    # 2019-12 분기(발표 2020-02-15)는 제외 — 마지막 행은 2019-09
    assert rows[-1]["분기말"] == "2019-09-28"
    assert rows[-1]["EPS YoY"] == pytest.approx(0.60)
    assert rows[-1]["순이익률"] == pytest.approx(2.08 * 10 / 146)
    assert len(rows) == 4


# ---- 분산일 (STAGE 0-2) ----------------------------------------------------

def test_distribution_days_counts_down_days_on_rising_volume():
    n = 30
    closes = np.full(n, 100.0)
    vols = np.full(n, 1e6)
    # 마지막 25일 창 안에 분산일 3개 심기: -0.5% 하락 + 거래량 증가
    for i in (10, 15, 20):
        closes[i:] *= 0.995
        vols[i] = vols[i - 1] * 1.5
    idx = pd.bdate_range("2020-01-01", periods=n)
    df = pd.DataFrame({"Open": closes, "High": closes, "Low": closes,
                       "Close": closes, "Volume": vols}, index=idx)
    assert distribution_days(df) == 3


def test_distribution_days_ignores_low_volume_declines_and_short_data():
    n = 30
    closes = np.linspace(100.0, 90.0, n)          # 매일 하락
    vols = np.linspace(2e6, 1e6, n)               # 거래량은 매일 감소
    idx = pd.bdate_range("2020-01-01", periods=n)
    df = pd.DataFrame({"Open": closes, "High": closes, "Low": closes,
                       "Close": closes, "Volume": vols}, index=idx)
    assert distribution_days(df) == 0             # 거래량 증가 조건 미충족
    assert distribution_days(df.head(10)) is None  # 봉 부족


# ---- 분할 환산 (당시 실제 가격 복원) ---------------------------------------

def test_split_factor_counts_only_future_splits():
    # NVDA 사례: 2021-07 4:1, 2024-06 10:1 → 2020년 기준일이면 ×40
    splits = pd.Series(
        [1.5, 4.0, 10.0],
        index=pd.to_datetime(["2007-09-11", "2021-07-20", "2024-06-10"]),
    )
    assert split_factor(splits, date(2020, 1, 2)) == pytest.approx(40.0)
    # 기준일이 모든 분할 이후면 조정가 = 실제가
    assert split_factor(splits, date(2025, 1, 2)) == pytest.approx(1.0)
    # 기준일 당일 발효 분할은 이미 가격에 반영된 것으로 취급 (> 비교)
    assert split_factor(splits, date(2024, 6, 10)) == pytest.approx(1.0)


def test_split_factor_none_and_empty():
    assert split_factor(None, date(2020, 1, 2)) == 1.0
    assert split_factor(pd.Series(dtype=float), date(2020, 1, 2)) == 1.0


def test_split_factor_tz_aware_index():
    splits = pd.Series(
        [4.0], index=pd.to_datetime(["2021-07-20"]).tz_localize("America/New_York")
    )
    assert split_factor(splits, date(2020, 1, 2)) == pytest.approx(4.0)


# ---- 피봇 / 손익선 --------------------------------------------------------

def _vcp_frame() -> pd.DataFrame:
    """상승 → T1 −20% → T2 −10% → T3 −5% → 조밀 선반: 교과서 VCP."""
    segs = [
        np.linspace(10.0, 30.0, 120),   # 선행 상승
        np.linspace(30.0, 24.0, 12),    # T1 ≈ −20%
        np.linspace(24.0, 29.4, 12),
        np.linspace(29.4, 26.5, 10),    # T2 ≈ −10%
        np.linspace(26.5, 29.0, 10),
        np.linspace(29.0, 27.6, 8),     # T3 ≈ −5%
        np.full(8, 27.9),               # 최종 조밀 선반
    ]
    closes = np.concatenate(segs)
    idx = pd.bdate_range("2019-01-01", periods=len(closes))
    return pd.DataFrame({
        "Open": closes, "High": closes * 1.004, "Low": closes * 0.996,
        "Close": closes, "Volume": np.full(len(closes), 1e6),
    }, index=idx)


def test_vcp_pivot_measures_contractions_and_shelf():
    pl = pivot_levels(_vcp_frame())
    # 축소 3회, 깊이 대략 −20% → −10% → −5%, 체감 통과
    assert len(pl.contractions) == 3
    assert pl.contractions[0] == pytest.approx(0.20, abs=0.02)
    assert pl.contractions[1] == pytest.approx(0.10, abs=0.02)
    assert pl.contractions[2] == pytest.approx(0.05, abs=0.02)
    assert pl.contraction_ok is True
    # 마지막 조밀 선반(6% 이내)이 마지막 스윙 고점보다 뚜렷이 아래 →
    # 속임수(cheat) 피봇 = 선반 고가
    assert pl.method == "tight-shelf(cheat)"
    assert pl.pivot == pytest.approx(27.9 * 1.004, rel=1e-3)
    assert "피봇 근접" in pl.status
    assert 0.10 <= pl.base_depth <= 0.35


def test_vcp_pivot_contraction_high_when_no_shelf():
    pl = pivot_levels(_vcp_frame(), shelf_max_range=0.0)  # 선반 감지 끔
    assert pl.method == "contraction-high"
    assert pl.pivot == pytest.approx(29.0 * 1.004, rel=1e-3)  # 마지막 스윙 고점


def test_vcp_pivot_straight_rally_falls_back():
    pl = pivot_levels(_daily(np.linspace(10.0, 30.0, 300)))
    assert pl.method == "recent-high"
    assert "압축 없음" in pl.status
    assert pl.contraction_ok is None


def test_vcp_pivot_flags_deep_base():
    segs = [np.linspace(10.0, 30.0, 150),
            np.linspace(30.0, 13.0, 40),    # −57% 깊은 조정
            np.linspace(13.0, 16.0, 40)]
    closes = np.concatenate(segs)
    idx = pd.bdate_range("2019-01-01", periods=len(closes))
    df = pd.DataFrame({"Open": closes, "High": closes * 1.004,
                       "Low": closes * 0.996, "Close": closes,
                       "Volume": np.full(len(closes), 1e6)}, index=idx)
    pl = pivot_levels(df)
    assert pl.base_depth > 0.35  # 4-3 위반 (10~35% 밖)


def test_pivot_levels_math():
    closes = np.linspace(10.0, 30.0, 300)
    df = _daily(closes)
    pl = pivot_levels(df, lookback=25, stop_pct=0.08, target_pct=0.22)
    assert pl is not None
    assert pl.pivot == pytest.approx(float(df["High"].tail(25).max()))
    assert pl.stop == pytest.approx(pl.pivot * 0.92)
    assert pl.target == pytest.approx(pl.pivot * 1.22)
    assert pl.dist_to_pivot <= 0.0 or pl.dist_to_pivot == pytest.approx(
        pl.close / pl.pivot - 1.0
    )


# ---- 기준일 이후 시뮬레이션 ------------------------------------------------

def _forward(rows: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    idx = pd.bdate_range("2020-01-03", periods=len(rows))
    o, h, l, c = zip(*rows)
    return pd.DataFrame(
        {"Open": o, "High": h, "Low": l, "Close": c,
         "Volume": np.full(len(rows), 1e6)},
        index=idx,
    )


def test_forward_target_first():
    pivot = 100.0
    rows = [(98, 99, 97, 98)] * 3 + [(99, 101, 98, 100.5)]  # 4일째 돌파
    rows += [(101, 112, 100, 111)] * 30                      # 익절선(122) 미달
    rows += [(115, 125, 114, 124)]                           # 122 도달
    out = simulate_forward(_forward(rows), pivot, stop_pct=0.08, target_pct=0.22)
    assert out.entered
    assert out.entry_price == pytest.approx(100.0)  # 시가 99 < 피봇 → 피봇 체결
    assert out.first_hit == "target"


def test_forward_stop_first_and_gap_entry():
    pivot = 100.0
    rows = [(103, 104, 102, 103)]            # 갭 상승 → 시가 103 체결
    rows += [(101, 102, 94.5, 95)]           # 저가 94.5 ≤ 103×0.92=94.76 → 손절
    out = simulate_forward(_forward(rows), pivot, stop_pct=0.08, target_pct=0.22)
    assert out.entered
    assert out.entry_price == pytest.approx(103.0)
    assert out.first_hit == "stop"


def test_forward_no_breakout():
    pivot = 100.0
    rows = [(90, 95, 88, 92)] * 30
    out = simulate_forward(_forward(rows), pivot)
    assert not out.entered
    assert out.first_hit == "none"


# ---- 트레이드 리플레이 (규칙 변형) -----------------------------------------

def _replay_frame(fwd_rows: list[tuple[float, float, float, float]],
                  as_of: str = "2020-01-02") -> tuple[pd.DataFrame, date]:
    """기준일 이전 60일(가격 95 부근) + 이후 fwd_rows 를 이어붙인 일봉."""
    n_before = 60
    before_close = np.full(n_before, 95.0)
    idx_before = pd.bdate_range(end=as_of, periods=n_before)
    before = pd.DataFrame({
        "Open": before_close, "High": before_close * 1.005,
        "Low": before_close * 0.995, "Close": before_close,
        "Volume": np.full(n_before, 1e6),
    }, index=idx_before)
    o, h, l, c = zip(*fwd_rows)
    idx_fwd = pd.bdate_range(
        pd.Timestamp(as_of) + pd.Timedelta(days=1), periods=len(fwd_rows)
    )
    fwd = pd.DataFrame({"Open": o, "High": h, "Low": l, "Close": c,
                        "Volume": np.full(len(fwd_rows), 2e6)}, index=idx_fwd)
    return pd.concat([before, fwd]), date.fromisoformat(as_of)


def test_replay_basic_target():
    rows = [(99, 101, 98, 100.5)] + [(101, 125, 100, 124)]  # 돌파 → 익절선 122
    df, as_of = _replay_frame(rows)
    r = replay_trade(df, as_of, pivot=100.0, stop_pct=0.08, target_pct=0.22)
    assert r.entered and r.exit_reason == "익절"
    assert r.ret == pytest.approx(0.22)
    assert r.breakout_volume_ratio == pytest.approx(2.0)  # 2e6 / 1e6


def test_replay_ma_exit_runner():
    # 돌파 후 상승 유지하다 손절선(92)은 안 건드리고 50일선(~107)만
    # 종가로 하회 → 러너 청산 (손절이 먼저면 손절 — 별도 케이스)
    rows = [(99, 101, 98, 100.5)]
    rows += [(101, 116, 100, 115)] * 30       # 상승 유지 (익절 없음 규칙)
    rows += [(105, 106, 100, 100.5)]          # 저가 100 > 손절 92, 종가 < SMA50
    df, as_of = _replay_frame(rows)
    r = replay_trade(df, as_of, pivot=100.0, stop_pct=0.08,
                     target_pct=None, ma_exit=50)
    assert r.entered and r.exit_reason == "50일선 이탈"
    assert r.exit_price == pytest.approx(100.5)


def test_replay_breakeven_stop():
    # +10% 종가 마감으로 본전 스톱 장전 → 이후 하락해도 본전청산 (−8% 손절 아님)
    rows = [(99, 101, 98, 100.5)]
    rows += [(101, 112, 100, 111)]            # +11% 종가 → 트리거
    rows += [(110, 110, 99.0, 99.5)]          # 저가 99 ≤ 본전(100)
    df, as_of = _replay_frame(rows)
    r = replay_trade(df, as_of, pivot=100.0, stop_pct=0.08, target_pct=0.50,
                     breakeven_trigger=0.10)
    assert r.entered and r.exit_reason == "본전청산"
    assert r.ret == pytest.approx(0.0)


def test_replay_hold_variant_and_no_breakout():
    rows = [(99, 101, 98, 100.5)] + [(101, 108, 100, 107)] * 10
    df, as_of = _replay_frame(rows)
    hold = replay_trade(df, as_of, pivot=100.0, stop_pct=None, target_pct=None)
    assert hold.entered and hold.exit_reason == "보유 중"
    assert hold.ret == pytest.approx(0.07)
    none_df, as_of2 = _replay_frame([(90, 95, 88, 92)] * 30)
    r = replay_trade(none_df, as_of2, pivot=100.0)
    assert not r.entered and r.exit_reason == "미돌파"


def test_forward_returns_from_entry():
    pivot = 100.0
    rows = [(100, 101, 99, 100)]                      # 1일째 돌파, 체결 100
    rows += [(100, 106, 99, 105)] * 30                # 이후 105 유지
    out = simulate_forward(_forward(rows), pivot, stop_pct=0.08, target_pct=0.50)
    assert out.entered and out.first_hit == "none"
    assert out.ret_1m == pytest.approx(0.05)
    assert out.max_runup == pytest.approx(0.06)


# ---- 종합 점수 (정렬용) ---------------------------------------------------

def _pl(dist=-0.02, method="contraction-high", contraction_ok=True,
        tightness=0.05, dryup=0.6) -> PivotLevels:
    pivot = 100.0
    return PivotLevels(
        pivot=pivot, close=pivot * (1 + dist), dist_to_pivot=dist,
        tightness_10d=tightness, volume_dryup=dryup,
        stop=92.0, target=122.0, method=method,
        contraction_ok=contraction_ok,
    )


def _growth(n_passed: int, available: bool = True) -> GrowthResult:
    return GrowthResult(
        eps_yoy=0.5, eps_yoy_history=[], eps_accelerating=True,
        revenue_yoy=0.3, revenue_accelerating=True,
        margin_now=0.2, margin_year_ago=0.1,
        n_passed=n_passed, passed=n_passed >= 3, data_available=available,
    )


def test_composite_perfect_setup_scores_100():
    cs = composite_score(100.0, _growth(4), _pl(), industry_survivors=3)
    assert cs.total == pytest.approx(100.0)
    assert cs.parts["코드33"] is None                    # 연간 미제공 → 제외
    assert all(v == 100.0 for k, v in cs.parts.items() if k != "코드33")


def test_composite_weights_sum_to_100():
    assert sum(COMPOSITE_WEIGHTS.values()) == pytest.approx(100.0)


def test_composite_penalises_extended_and_no_vcp_over_pure_rs():
    # RS 99지만 피봇 위 +8% 확장 + 직선 랠리(축소 無) 종목보다
    # RS 90에 피봇 근접 + VCP 통과 종목이 위로 와야 한다.
    extended = composite_score(
        99.0, _growth(4), _pl(dist=0.08, method="recent-high", contraction_ok=None),
        industry_survivors=3,
    )
    ready = composite_score(90.0, _growth(4), _pl(), industry_survivors=3)
    assert ready.total > extended.total
    assert extended.parts["VCP구조"] == 0.0
    assert extended.parts["피봇위치"] == 20.0


def test_composite_pivot_position_bands():
    assert composite_score(50, None, _pl(dist=-0.03)).parts["피봇위치"] == 100.0
    assert composite_score(50, None, _pl(dist=0.02)).parts["피봇위치"] == 70.0
    assert composite_score(50, None, _pl(dist=0.05)).parts["피봇위치"] == 20.0
    # −5% 아래는 −20%까지 선형 감소: −12.5% → 50, −20% 이하 → 0
    assert composite_score(50, None, _pl(dist=-0.125)).parts["피봇위치"] == pytest.approx(50.0)
    assert composite_score(50, None, _pl(dist=-0.30)).parts["피봇위치"] == 0.0


def test_composite_renormalises_when_parts_missing():
    # 성장·산업군 없음 → 나머지 가중치(75)로 재정규화. 나머지가 전부 100이면 100.
    cs = composite_score(100.0, None, _pl(), industry_survivors=None)
    assert cs.parts["성장"] is None and cs.parts["산업군"] is None
    assert cs.total == pytest.approx(100.0)
    # data_available=False 도 제외 처리
    cs2 = composite_score(100.0, _growth(0, available=False), _pl(), industry_survivors=None)
    assert cs2.parts["성장"] is None and cs2.total == pytest.approx(100.0)


def test_composite_growth_and_industry_scaling():
    cs = composite_score(0.0, _growth(2), _pl(contraction_ok=False, tightness=0.25, dryup=1.5),
                         industry_survivors=2)
    assert cs.parts["성장"] == 50.0
    assert cs.parts["VCP구조"] == 20.0
    assert cs.parts["조밀도"] == 0.0 and cs.parts["거래량"] == 0.0
    assert cs.parts["산업군"] == 60.0
    expected = (0 * 30 + 50 * 20 + 20 * 15 + 100 * 15 + 0 + 0 + 60 * 5) / 95
    assert cs.total == pytest.approx(round(expected, 1))


def test_composite_nan_hints_are_excluded():
    pl = _pl(tightness=float("nan"), dryup=float("nan"))
    cs = composite_score(80.0, _growth(4), pl, industry_survivors=1)
    assert cs.parts["조밀도"] is None and cs.parts["거래량"] is None
    expected = (80 * 30 + 100 * 20 + 100 * 15 + 100 * 15 + 30 * 5) / 85
    assert cs.total == pytest.approx(round(expected, 1))


# ---- 연간 판정 (2-5 코드 33 · 2-6 · 2-10) ---------------------------------

from data.domain.models import AnnualFinancials
from strategy.adapters.minervini_screener import annual_evidence, evaluate_annual


def _annuals(eps, rev, ni, start_year=2015, report_lag_days=60):
    out = []
    for i, (e, r, n) in enumerate(zip(eps, rev, ni)):
        f = date(start_year + i, 12, 31)
        out.append(AnnualFinancials(
            symbol="T", fiscal_date=f, report_date=f + timedelta(days=report_lag_days),
            eps=e, revenue=float(r), net_income=float(n),
        ))
    return out


def test_annual_code33_monster_passes_all_three():
    # 책의 몬스터 사례를 흉내: EPS +75→+214→+195%, 매출 +20→+63→+93%, NPM 3.3→11.3→18%
    eps = [1.00, 1.75, 5.50, 16.2]
    rev = [100, 120, 196, 378]
    ni = [3.3, 13.6, 35.3, 75.0]            # 마진 3.3% → 11.3% → 18.0% → 19.8%
    a = evaluate_annual(_annuals(eps, rev, ni), as_of=date(2019, 6, 1))
    assert a.data_available
    assert a.checks["2-5a EPS 3년 가속"] is True      # 75→214→195: 마지막 > 첫 해
    assert a.checks["2-5b 매출 3년 가속"] is True
    assert a.checks["2-5c 순이익률 3년 상승"] is True
    assert a.n_passed == 3 and a.passed
    assert a.eps_breakout is True and a.eps_prev_high == pytest.approx(5.50)
    assert a.decel_warning is False


def test_annual_point_in_time_excludes_unreported_year():
    eps = [1.0, 1.5, 2.5, 5.0]
    rev = [100, 130, 180, 300]
    ni = [5, 10, 20, 45]
    ann = _annuals(eps, rev, ni)                     # FY2018 발표일 = 2019-03-01
    early = evaluate_annual(ann, as_of=date(2019, 2, 1))
    late = evaluate_annual(ann, as_of=date(2019, 3, 1))
    assert early.years[-1] == date(2017, 12, 31) and len(early.eps_yoy) == 2
    assert early.checks["2-5a EPS 3년 가속"] is None  # YoY 2개뿐 → 미확인
    assert late.years[-1] == date(2018, 12, 31) and late.n_passed == 3


def test_annual_decel_warning_dell_style():
    # EPS YoY 80 → 65 → 28% (마지막 < 첫 해의 절반) → 2-10 경고
    eps = [1.0, 1.8, 2.97, 3.80]
    rev = [100, 150, 200, 240]
    ni = [10, 18, 30, 38]
    a = evaluate_annual(_annuals(eps, rev, ni), as_of=date(2019, 6, 1))
    assert a.decel_warning is True
    assert a.checks["2-5a EPS 3년 가속"] is False
    # 완만한 둔화(30→26→24%)는 경고 아님
    rev2 = [100, 130, 164, 203]
    b = evaluate_annual(_annuals(rev2, rev2, [10, 13, 17, 21]), as_of=date(2019, 6, 1))
    assert b.decel_warning is False


def test_annual_breakout_needs_three_prior_years_and_positive():
    a = evaluate_annual(_annuals([0.5, 0.4, 0.45], [1, 1, 1], [1, 1, 1]), as_of=date(2019, 6, 1))
    assert a.eps_breakout is None                    # 비교 대상 2년뿐
    b = evaluate_annual(_annuals([-0.1, -0.2, -0.3, -0.05], [1, 1, 1, 1], [1, 1, 1, 1]),
                        as_of=date(2019, 6, 1))
    assert b.eps_breakout is False                   # 적자 → 신고 아님
    c = evaluate_annual(_annuals([0.3, 0.2, 0.25, 1.3], [1, 1, 1, 1], [1, 1, 1, 1]),
                        as_of=date(2019, 6, 1))
    assert c.eps_breakout is True and c.eps_prev_high == pytest.approx(0.3)


def test_annual_empty_and_evidence_rows():
    a = evaluate_annual([], as_of=date(2019, 6, 1))
    assert not a.data_available and a.n_passed == 0 and a.eps_breakout is None
    rows = annual_evidence(_annuals([1.0, 2.0, 3.0, 4.0], [10, 20, 30, 40], [1, 2, 3, 4]),
                           as_of=date(2019, 6, 1), n=3)
    assert [r["회계연도"] for r in rows] == ["2016-12-31", "2017-12-31", "2018-12-31"]
    assert rows[-1]["EPS YoY"] == pytest.approx(1 / 3)
    assert rows[-1]["순이익률"] == pytest.approx(0.1)


def test_composite_includes_code33_when_available():
    full = evaluate_annual(_annuals([1, 1.75, 5.5, 16.2], [100, 120, 196, 378], [3.3, 13.6, 35.3, 75]),
                           as_of=date(2019, 6, 1))
    cs = composite_score(100.0, _growth(4), _pl(), industry_survivors=3, annual=full)
    assert cs.parts["코드33"] == 100.0 and cs.total == pytest.approx(100.0)
    none_ = composite_score(100.0, _growth(4), _pl(), industry_survivors=3, annual=None)
    assert none_.parts["코드33"] is None and none_.total == pytest.approx(100.0)
    partial = evaluate_annual(_annuals([1.0, 1.8, 2.97, 3.8], [100, 150, 200, 240], [10, 18, 30, 38]),
                              as_of=date(2019, 6, 1))
    cs2 = composite_score(100.0, _growth(4), _pl(), industry_survivors=3, annual=partial)
    assert cs2.parts["코드33"] == pytest.approx(100.0 * partial.n_passed / 3)


# ---- STAGE 0 시장 환경 — SPY + QQQ 이중 판정 -------------------------------

from strategy.adapters.minervini_screener import MarketVerdict, combine_market, evaluate_market


def _index(closes: np.ndarray, vols=None) -> pd.DataFrame:
    idx = pd.bdate_range("2019-01-01", periods=len(closes))
    v = np.full(len(closes), 1e6) if vols is None else vols
    return pd.DataFrame({"Open": closes, "High": closes * 1.005, "Low": closes * 0.995,
                         "Close": closes, "Volume": v}, index=idx)


def test_evaluate_market_uptrend_passes_and_short_data_none():
    v = evaluate_market(_index(np.linspace(100, 150, 300)), symbol="SPY")
    assert v is not None and v.trend_ok and v.dist_days == 0 and v.dist_ok and v.ok
    assert "SPY" in v.note and "상승" in v.note
    assert evaluate_market(_index(np.linspace(100, 150, 200))) is None


def test_evaluate_market_below_sma200_fails_trend():
    up = np.linspace(100, 150, 250); down = np.linspace(150, 110, 50)
    v = evaluate_market(_index(np.concatenate([up, down])), symbol="QQQ")
    assert v is not None and not v.trend_ok and not v.ok


def test_evaluate_market_distribution_days_flag():
    closes = np.linspace(100, 150, 300).copy(); vols = np.full(300, 1e6)
    for i in (280, 284, 288, 292, 296):        # 마지막 25일 안에 분산일 5개
        closes[i:] *= 0.99; vols[i] = vols[i - 1] * 1.5
    v = evaluate_market(_index(closes, vols), symbol="SPY")
    assert v is not None and v.trend_ok and v.dist_days == 5 and v.dist_ok is False and not v.ok


def _mv(sym, trend_ok, dist_days):
    return MarketVerdict(symbol=sym, close=100, sma200=90, sma200_prev21=85, trend_ok=trend_ok,
                         dist_days=dist_days, dist_ok=None if dist_days is None else dist_days < 5)


def test_combine_market_rules():
    ok, failed = combine_market([_mv("SPY", True, 2), _mv("QQQ", True, 3)])
    assert ok is True and failed == []
    # 0-1: 한 지수라도 추세 실패면 실패
    ok, failed = combine_market([_mv("SPY", True, 2), _mv("QQQ", False, 3)])
    assert ok is False and failed[0].startswith("0-1") and "QQQ" in failed[0] and "SPY" not in failed[0]
    # 0-2: 한 지수라도 분산일 5회↑면 경고 (IBD)
    ok, failed = combine_market([_mv("SPY", True, 6), _mv("QQQ", True, 1)])
    assert ok is False and failed == ["0-2 분산일 (SPY 5회↑)"]
    # 분산일 측정 불가(None)는 실패로 치지 않는다
    ok, failed = combine_market([_mv("SPY", True, None)])
    assert ok is True
    assert combine_market([]) == (None, [])


# ---- 2-7 서프라이즈 · 2-9 재고/매출채권 (수동 항목 데이터 보조) ---------------

from data.domain.models import QuarterlyBalance
from strategy.adapters.minervini_screener import evaluate_balance, evaluate_surprise


def _q(f, rep, actual, est, rev):
    return QuarterlyFinancials(symbol="T", fiscal_date=f, report_date=rep,
                               eps_actual=actual, eps_estimate=est, revenue=rev, net_income=1.0)


def test_surprise_point_in_time_and_streak():
    qs = [
        _q(date(2018, 12, 31), date(2019, 2, 6), 0.23, 0.22, 100),
        _q(date(2019, 3, 31), date(2019, 5, 2), 0.60, 0.57, 140),
        _q(date(2019, 6, 30), date(2019, 8, 8), 0.34, 0.25, 120),
        _q(date(2019, 9, 30), date(2019, 10, 30), 0.36, 0.26, 127),
        _q(date(2019, 12, 31), date(2020, 2, 6), 0.10, 0.30, 130),   # 기준일 이후 → 제외
    ]
    r = evaluate_surprise(qs, date(2020, 1, 2))
    assert r.data_available and r.latest_beat is True and r.streak == 4
    assert [x["분기말"] for x in r.rows][-1] == "2019-09-30"
    assert r.rows[-1]["서프라이즈"] == pytest.approx((0.36 - 0.26) / 0.26)
    later = evaluate_surprise(qs, date(2020, 3, 1))
    assert later.latest_beat is False and later.streak == 0
    assert not evaluate_surprise([], date(2020, 1, 2)).data_available


def _bs(f, rep, inv, rec):
    return QuarterlyBalance(symbol="T", fiscal_date=f, report_date=rep, inventory=inv, receivables=rec)


def test_balance_inventory_red_flag_and_service_company():
    fis = [date(2018, 3, 31), date(2018, 6, 30), date(2018, 9, 30), date(2018, 12, 31),
           date(2019, 3, 31), date(2019, 6, 30), date(2019, 9, 30)]
    reps = [f + timedelta(days=40) for f in fis]
    qs = [_q(f, r, 1.0, 0.9, rev) for f, r, rev in zip(fis, reps, [100, 100, 100, 100, 111, 111, 111])]
    # 제조업: 재고 +79% vs 매출 +11% → 탈락, 채권 +30% > 매출 → 경고
    bal = [_bs(f, r, 100_000, 50_000) for f, r in zip(fis[:4], reps[:4])] + \
          [_bs(f, r, 179_000, 65_000) for f, r in zip(fis[4:], reps[4:])]
    b = evaluate_balance(bal, qs, date(2020, 1, 2))
    assert b.data_available and b.fiscal_date == date(2019, 9, 30)
    assert b.revenue_yoy == pytest.approx(0.11) and b.inventory_yoy == pytest.approx(0.79)
    assert b.inventory_flag is True and b.receivables_flag is True and b.passed is False
    # 서비스업: 재고 없음 → 재고 플래그 None, 통과
    bal2 = [_bs(f, r, None, 50_000) for f, r in zip(fis, reps)]
    # 소스 아티팩트('1' 같은 값)도 재고 없음으로 취급
    bal2[-1] = _bs(fis[-1], reps[-1], 1, 65_000)
    b2 = evaluate_balance(bal2, qs, date(2020, 1, 2))
    assert b2.inventory_absent and b2.inventory_flag is None and b2.passed is True
    # point-in-time: 최신 분기 발표 전이면 한 분기 전 기준
    b3 = evaluate_balance(bal, qs, date(2019, 10, 15))
    assert b3.fiscal_date == date(2019, 6, 30)
    # 분기 5개 미만 → 미확인
    assert not evaluate_balance(bal[:4], qs, date(2020, 1, 2)).data_available
