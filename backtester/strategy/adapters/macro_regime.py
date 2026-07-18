"""거시 경제지표 기반 선행 레짐 스코어러.

각 지표를 일별 0/1 건강도 스코어로 **정량화**하고, 가중 평균으로
합성 스코어(0..1)를 만든 뒤, 히스테리시스 + 재진입 확인으로
risk-on/off 불리언 시계열을 생성한다. 이 시계열을
:class:`BandRebalanceStrategy`의 ``risk_on_series``로 주입하면
가격 SMA 대신 거시 지표가 방어 전환을 결정한다.

설계 원칙 — **look-ahead 방지**:
- 일별 지표(금리커브, VIX, HY 스프레드)는 당일 종가/마감 후 발표
  → 다음 거래일부터 사용 (`shift_days=1`).
- 월별 지표(실업률)는 다음 달 초 발표 → 관측월 + ``publish_lag_days``
  이후부터 사용.

각 스코어 함수는 지표가 "위험 신호가 아님"일 때 1, 위험일 때 0.
전쟁·팬데믹 같은 이벤트 자체가 아니라 그 이벤트가 신용·변동성·
고용·수익률곡선에 남기는 **정량 흔적**에 반응하므로, 시장이
소화해버리는 지정학 이벤트(예: 2022 우크라이나 침공 직후 신용
스프레드 안정)에는 과잉 반응하지 않는다.
"""

from __future__ import annotations

import pandas as pd

# ---------------------------------------------------------------------------
# 개별 지표 정량화 (healthy=1 / risk=0)
# ---------------------------------------------------------------------------


def score_yield_curve(t10y3m: pd.Series, shift_days: int = 1) -> pd.Series:
    """장단기 금리커브 (10Y − 3M, %p): 역전(<0)이면 위험.

    커브 역전은 6~18개월 선행하는 고전적 침체 선행지표. 단독으로는
    너무 이르게 꺼지는 경향이 있어(2006-07, 2019) 합성 스코어의 한
    표로만 쓴다.
    """
    healthy = (t10y3m > 0).astype(float).where(t10y3m.notna())
    return healthy.shift(shift_days).dropna()


def score_credit_spread(
    hy_oas: pd.Series, sma_days: int = 200, shift_days: int = 1
) -> pd.Series:
    """하이일드 OAS 스프레드: 200일 평균 위로 **확대**되면 위험.

    신용 시장은 주식보다 스트레스를 먼저 반영하는 대표 선행지표
    (2000, 2007-08, 2020, 2022 모두 선행 확대). 레벨이 아니라
    추세 비교라 저금리/고금리 시대 모두에서 동작한다.
    """
    sma = hy_oas.rolling(sma_days).mean()
    # SMA 워밍업 NaN 구간은 "위험(0)"이 아니라 판단 불가(NaN)로 제외.
    healthy = (hy_oas <= sma).astype(float).where(sma.notna())
    return healthy.shift(shift_days).dropna()


def score_vix(
    vix: pd.Series,
    smooth_days: int = 21,
    calm_level: float = 25.0,
    shift_days: int = 1,
) -> pd.Series:
    """VIX 21일 평균이 calm_level 미만이면 건강.

    스무딩으로 하루짜리 스파이크(이벤트 헤드라인)에는 반응하지
    않고, 지속적 변동성 체제 전환에만 반응한다.
    """
    smooth = vix.rolling(smooth_days).mean()
    healthy = (smooth < calm_level).astype(float).where(smooth.notna())
    return healthy.shift(shift_days).dropna()


def score_sahm_rule(
    unrate_monthly: pd.Series,
    trigger: float = 0.5,
    publish_lag_days: int = 40,
) -> pd.Series:
    """Sahm Rule: 실업률 3개월 평균이 직전 12개월 저점 대비
    +0.5%p 이상이면 침체 시작 신호.

    실시간 침체 판정 지표로 고안된 규칙 (Claudia Sahm, 2019).
    관측월 실업률은 다음 달 초 발표되므로 ``publish_lag_days``만큼
    인덱스를 밀어 look-ahead를 차단한다.
    """
    avg3 = unrate_monthly.rolling(3).mean()
    low12 = avg3.rolling(12).min()
    sahm = avg3 - low12
    score = (sahm < trigger).astype(float).where(sahm.notna()).dropna()
    score.index = score.index + pd.Timedelta(days=publish_lag_days)
    return score


def score_ratio_trend(
    numerator: pd.Series,
    denominator: pd.Series,
    sma_days: int = 200,
    shift_days: int = 1,
) -> pd.Series:
    """비율(예: XLY/XLP, 구리/금)이 자기 200일 평균 위면 건강.

    XLY/XLP = 경기민감 소비 대 필수 소비 — 소비자의 리스크 선호를
    실거래 가격으로 정량화한 선행 지표. 구리/금 = 실물 성장 대
    안전자산 선호.
    """
    ratio = (numerator / denominator).dropna()
    sma = ratio.rolling(sma_days).mean()
    healthy = (ratio > sma).astype(float).where(sma.notna())
    return healthy.shift(shift_days).dropna()


def score_price_trend(
    close: pd.Series, sma_days: int = 200, shift_days: int = 1
) -> pd.Series:
    """가격 추세 (레짐 지수 > 200SMA) — 시장 내부 확인용 지표."""
    sma = close.rolling(sma_days).mean()
    healthy = (close > sma).astype(float).where(sma.notna())
    return healthy.shift(shift_days).dropna()


# ---------------------------------------------------------------------------
# 합성 + 히스테리시스
# ---------------------------------------------------------------------------


def composite_score(
    components: dict[str, pd.Series],
    weights: dict[str, float],
    index: pd.DatetimeIndex,
) -> pd.Series:
    """가중 평균 합성 스코어 (0..1)를 백테스트 달력에 정렬해 반환.

    지표마다 히스토리 시작일이 달라서(NaN 구간), 날짜별로 **값이
    존재하는 지표만으로 가중치를 재정규화**한다 — 1999년엔 커브+
    VIX+고용만으로, 2001년부터는 HY 스프레드까지 합류하는 식.
    전 지표 NaN인 날은 NaN (소비자가 기본 risk-on 처리).
    """
    aligned = pd.DataFrame(
        {k: s.reindex(index, method="ffill") for k, s in components.items()}
    )
    w = pd.Series({k: weights.get(k, 1.0) for k in aligned.columns})
    mask = aligned.notna()
    weight_sum = mask.mul(w, axis=1).sum(axis=1)
    weighted = aligned.fillna(0.0).mul(w, axis=1).sum(axis=1)
    return (weighted / weight_sum).where(weight_sum > 0)


def hybrid_risk_on(
    trend_score: pd.Series,
    macro_score: pd.Series,
    veto_threshold: float = 0.6,
    confirm_days: int = 10,
) -> pd.Series:
    """가격 추세 × 거시 확인 하이브리드 risk-on 시계열.

    - **risk-off**: 가격 추세 붕괴(trend=0) **AND** 거시 합성 스코어
      < ``veto_threshold`` — 두 조건이 모두 맞아야 방어 전환.
      거시가 건강한 단순 기술적 조정(2011, 2015-16, 2018 등)은
      거시 거부권(veto)이 살려서 그대로 보유 → 가격 필터의 휩쏘
      비용을 제거해 상승률을 높인다.
    - **risk-on**: 가격 추세 복귀(trend=1)가 ``confirm_days``일
      연속 유지되면 재진입 — 가격 회복이 거시 회복을 선행하므로
      재진입은 추세만 본다.
    - 거시 스코어 NaN(지표 히스토리 이전)은 거부권 없음으로 간주
      (= 순수 추세 필터로 동작).
    """
    if macro_score.empty:
        macro = pd.Series(float("nan"), index=trend_score.index)
    else:
        macro = macro_score.reindex(trend_score.index, method="ffill")
    risk_on = True
    streak = 0
    out = []
    for t, m in zip(trend_score.values, macro.values):
        if pd.isna(t):
            out.append(risk_on)
            continue
        if risk_on:
            macro_weak = (not pd.isna(m)) and m < veto_threshold
            if t == 0 and macro_weak:
                risk_on = False
                streak = 0
        else:
            if t == 1:
                streak += 1
                if streak > confirm_days:
                    risk_on = True
            else:
                streak = 0
        out.append(risk_on)
    return pd.Series(out, index=trend_score.index, dtype=bool)


def hysteresis_risk_on(
    score: pd.Series,
    on_threshold: float = 0.6,
    off_threshold: float = 0.4,
    confirm_days: int = 10,
) -> pd.Series:
    """합성 스코어 → risk-on 불리언 (히스테리시스 + 재진입 확인).

    - risk-on 중 스코어가 ``off_threshold`` 미만이면 **즉시** off.
    - risk-off 중 스코어가 ``confirm_days``일 연속 ``on_threshold``
      이상이면 on 복귀 (베어랠리 휩쏘 필터 — 방어는 빠르게, 복귀는
      신중하게).
    - 스코어 NaN(지표 히스토리 이전)은 risk-on 취급.
    """
    risk_on = True
    streak = 0
    out = []
    for v in score.values:
        if pd.isna(v):
            out.append(risk_on)
            continue
        if risk_on:
            if v < off_threshold:
                risk_on = False
                streak = 0
        else:
            if v >= on_threshold:
                streak += 1
                if streak > confirm_days:
                    risk_on = True
            else:
                streak = 0
        out.append(risk_on)
    return pd.Series(out, index=score.index, dtype=bool)
