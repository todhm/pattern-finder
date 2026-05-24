"""
US Quant Weight Adjustments Configuration

Phase 3.1 IC 분석 결과 기반 가중치 조정값 정의

- 거래소별 가중치 조정 (NASDAQ IC 개선용)
- 섹터별 가중치 조정 (섹터 특성 반영)
- ORS (Outlier Risk Score) 임계값 및 위험 등급

Usage:
    from weight_adjustments import (
        EXCHANGE_ADJUSTMENTS,
        SECTOR_ADJUSTMENTS,
        ORS_THRESHOLDS,
        ORS_RISK_LEVELS
    )

File: us/weight_adjustments.py
Created: 2025-11-29
"""

from typing import Dict, List, Tuple

# ============================================================================
# 1. Exchange Weight Adjustments (거래소별 가중치 조정)
# ============================================================================
# IC 분석 결과:
#   - NYSE: final_score IC 0.050 (기준)
#   - NASDAQ: final_score IC -0.008 (Growth만 양의 IC)
#
# NASDAQ에서 Growth IC가 상대적으로 높고, Momentum/Quality IC가 낮음
# 따라서 NASDAQ 종목은 Growth 비중 높이고, Momentum/Quality 비중 낮춤

EXCHANGE_ADJUSTMENTS: Dict[str, Dict[str, int]] = {
    'NYSE': {
        'growth': 0,
        'momentum': 0,
        'quality': 0,
        'value': 0
    },
    # Phase 3.1.6 NASDAQ IC 분석 기반 가중치 강화
    # IC 비교: NYSE vs NASDAQ
    #   growth:   0.077 vs 0.035 → NASDAQ에서도 최고 IC, 강화
    #   momentum: 0.062 vs 0.029 → 상대적 약화
    #   quality:  0.026 vs 0.019 → 가장 낮은 IC, 대폭 감소
    #   value:    0.026 vs 0.024 → 유사하나 약간 감소
    # 목표 가중치: growth 33%, momentum 27%, value 22%, quality 18%
    'NASDAQ': {
        'growth': 12,     # 5 → 12 (Growth IC 최고, 비중 대폭 증가)
        'momentum': -3,   # -5 → -3 (Momentum IC도 양수, 완화)
        'quality': -8,    # -3 → -8 (Quality IC 가장 낮음, 대폭 감소)
        'value': -1       # 3 → -1 (Value IC 낮은 편)
    },
    'AMEX': {
        'growth': 0,
        'momentum': 0,
        'quality': 0,
        'value': 0
    },
}


# ============================================================================
# 2. Sector Weight Adjustments (섹터별 가중치 조정)
# ============================================================================
# IC 분석 결과 기반 섹터별 최적 팩터 가중치 조정
#
# 섹터별 최고 IC 팩터:
#   - HEALTHCARE: Growth (0.034)
#   - FINANCIAL SERVICES: EM7/Momentum (0.415)
#   - TECHNOLOGY: Growth (0.052)
#   - CONSUMER DEFENSIVE: Quality (0.169)
#   - ENERGY: Quality IC 음수 (-0.052)
#   - BASIC MATERIALS: 모든 팩터 균형 (조정 없음)

SECTOR_ADJUSTMENTS: Dict[str, Dict[str, int]] = {
    'HEALTHCARE': {
        'growth': 10,     # Growth만 양의 IC
        'momentum': -5,
        'quality': -5,
        'value': 0
    },
    'FINANCIAL SERVICES': {
        'growth': 0,
        'momentum': 10,   # EM7 IC 0.415로 최고
        'quality': 0,
        'value': -5
    },
    'TECHNOLOGY': {
        'growth': 5,      # Growth IC 0.052
        'momentum': 0,
        'quality': 0,
        'value': -5
    },
    'CONSUMER DEFENSIVE': {
        'growth': 0,
        'momentum': -5,
        'quality': 10,    # Quality IC 0.169
        'value': -5
    },
    'ENERGY': {
        'growth': 5,
        'momentum': 0,
        'quality': -10,   # Quality IC -0.052 (음수)
        'value': 5
    },
    'BASIC MATERIALS': {
        'growth': 0,      # 모든 팩터 IC 0.06+ (균형)
        'momentum': 0,
        'quality': 0,
        'value': 0
    },
    'INDUSTRIALS': {
        'growth': 0,
        'momentum': 5,    # EM7 IC 0.329
        'quality': 0,
        'value': -5
    },
    'CONSUMER CYCLICAL': {
        'growth': 5,
        'momentum': 0,
        'quality': 0,
        'value': -5
    },
    'UTILITIES': {
        'growth': 0,
        'momentum': -5,
        'quality': 5,
        'value': 0
    },
    'REAL ESTATE': {
        'growth': 0,
        'momentum': 0,
        'quality': 5,
        'value': -5
    },
    'COMMUNICATION SERVICES': {
        'growth': 5,
        'momentum': 0,
        'quality': 0,
        'value': -5
    },
}


# ============================================================================
# 3. ORS (Outlier Risk Score) Thresholds
# ============================================================================
# Mean-Median 괴리 원인 종목 식별용 임계값
#
# ORS 구성요소 (각 25점, 총 100점):
#   - Penny Stock: 저가주 위험
#   - Low Volume: 유동성 부족 위험
#   - High Volatility: 변동성 위험
#   - Small Cap: 소형주 위험

# 가격 기준 점수 (threshold, score)
# 가격이 threshold 미만이면 해당 score 부여
ORS_PRICE_SCORES: List[Tuple[float, int]] = [
    (1.0, 25),    # $1 미만: 25점 (최고 위험)
    (3.0, 20),    # $3 미만: 20점
    (5.0, 15),    # $5 미만: 15점 (페니주식)
    (10.0, 5),    # $10 미만: 5점
]

# 거래량 기준 점수 (threshold, score)
# 일평균 거래량이 threshold 미만이면 해당 score 부여
ORS_VOLUME_SCORES: List[Tuple[int, int]] = [
    (50000, 25),     # 5만주 미만: 25점
    (100000, 20),    # 10만주 미만: 20점
    (500000, 10),    # 50만주 미만: 10점
]

# 변동성 기준 점수 (threshold, score)
# 연환산 변동성(%)이 threshold 초과이면 해당 score 부여
ORS_VOLATILITY_SCORES: List[Tuple[float, int]] = [
    (150.0, 25),   # 150% 초과: 25점
    (100.0, 20),   # 100% 초과: 20점
    (75.0, 10),    # 75% 초과: 10점
]

# 시가총액 기준 점수 (threshold, score)
# 시가총액이 threshold 미만이면 해당 score 부여
ORS_MKTCAP_SCORES: List[Tuple[float, int]] = [
    (100_000_000, 25),     # $100M 미만: 25점 (마이크로캡)
    (300_000_000, 20),     # $300M 미만: 20점
    (1_000_000_000, 10),   # $1B 미만: 10점 (스몰캡)
]

# 통합 임계값 딕셔너리 (하위 호환용)
ORS_THRESHOLDS: Dict[str, List[Tuple]] = {
    'price': ORS_PRICE_SCORES,
    'volume': ORS_VOLUME_SCORES,
    'volatility': ORS_VOLATILITY_SCORES,
    'mktcap': ORS_MKTCAP_SCORES,
}


# ============================================================================
# 4. ORS Risk Levels (위험 등급)
# ============================================================================
# ORS 점수에 따른 위험 등급 및 포지션 조정 배수

ORS_RISK_LEVELS: Dict[str, Dict] = {
    '극단적 위험': {
        'min_score': 80,
        'max_score': 100,
        'position_multiplier': 0.25,  # 포지션 75% 축소
        'description': '극단적 이상치 위험 - 포지션 대폭 축소 필요'
    },
    '고위험': {
        'min_score': 60,
        'max_score': 79,
        'position_multiplier': 0.50,  # 포지션 50% 축소
        'description': '높은 이상치 위험 - 포지션 적정 축소 필요'
    },
    '중위험': {
        'min_score': 40,
        'max_score': 59,
        'position_multiplier': 0.75,  # 포지션 25% 축소
        'description': '중간 이상치 위험 - 포지션 소폭 축소 권고'
    },
    '정상': {
        'min_score': 0,
        'max_score': 39,
        'position_multiplier': 1.0,   # 포지션 유지
        'description': '정상 위험 수준 - 조정 불필요'
    },
}


# ============================================================================
# Helper Functions
# ============================================================================

def get_exchange_adjustment(exchange: str) -> Dict[str, int]:
    """
    거래소별 가중치 조정값 반환

    Args:
        exchange: 거래소 코드 (NYSE, NASDAQ, AMEX)

    Returns:
        Dict with factor adjustments (growth, momentum, quality, value)
    """
    return EXCHANGE_ADJUSTMENTS.get(exchange, {
        'growth': 0, 'momentum': 0, 'quality': 0, 'value': 0
    })


def get_sector_adjustment(sector: str) -> Dict[str, int]:
    """
    섹터별 가중치 조정값 반환

    Args:
        sector: 섹터명

    Returns:
        Dict with factor adjustments (growth, momentum, quality, value)
    """
    return SECTOR_ADJUSTMENTS.get(sector, {
        'growth': 0, 'momentum': 0, 'quality': 0, 'value': 0
    })


def get_risk_level(ors_score: float) -> str:
    """
    ORS 점수에 따른 위험 등급 반환

    Args:
        ors_score: ORS 점수 (0-100)

    Returns:
        Risk level string (극단적 위험, 고위험, 중위험, 정상)
    """
    if ors_score >= 80:
        return '극단적 위험'
    elif ors_score >= 60:
        return '고위험'
    elif ors_score >= 40:
        return '중위험'
    else:
        return '정상'


def get_position_multiplier(ors_score: float) -> float:
    """
    ORS 점수에 따른 포지션 조정 배수 반환

    Args:
        ors_score: ORS 점수 (0-100)

    Returns:
        Position multiplier (0.25 ~ 1.0)
    """
    risk_level = get_risk_level(ors_score)
    return ORS_RISK_LEVELS[risk_level]['position_multiplier']


def calculate_adjusted_weights(
    base_weights: Dict[str, float],
    exchange: str = None,
    sector: str = None
) -> Dict[str, float]:
    """
    기본 가중치에 거래소/섹터 조정 적용 후 정규화

    Args:
        base_weights: 기본 가중치 (예: {'growth': 25, 'momentum': 25, ...})
        exchange: 거래소 코드 (optional)
        sector: 섹터명 (optional)

    Returns:
        정규화된 조정 가중치 (합계 100)
    """
    adjusted = base_weights.copy()

    # 거래소 조정 적용
    if exchange:
        ex_adj = get_exchange_adjustment(exchange)
        for factor in adjusted:
            adjusted[factor] += ex_adj.get(factor, 0)

    # 섹터 조정 적용
    if sector:
        sec_adj = get_sector_adjustment(sector)
        for factor in adjusted:
            adjusted[factor] += sec_adj.get(factor, 0)

    # 음수 방지 (최소 5)
    for factor in adjusted:
        adjusted[factor] = max(adjusted[factor], 5)

    # 합계 100으로 정규화
    total = sum(adjusted.values())
    if total > 0:
        adjusted = {k: v * 100 / total for k, v in adjusted.items()}

    return adjusted


# ============================================================================
# Module Test
# ============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("Weight Adjustments Configuration Test")
    print("=" * 60)

    # Test exchange adjustments
    print("\n[Exchange Adjustments]")
    for exchange in ['NYSE', 'NASDAQ', 'AMEX']:
        adj = get_exchange_adjustment(exchange)
        print(f"  {exchange}: {adj}")

    # Test sector adjustments
    print("\n[Sector Adjustments]")
    for sector in ['HEALTHCARE', 'TECHNOLOGY', 'FINANCIAL SERVICES']:
        adj = get_sector_adjustment(sector)
        print(f"  {sector}: {adj}")

    # Test ORS risk levels
    print("\n[ORS Risk Levels]")
    for score in [0, 35, 45, 65, 85]:
        level = get_risk_level(score)
        mult = get_position_multiplier(score)
        print(f"  Score {score}: {level} (position x{mult})")

    # Test adjusted weights
    print("\n[Adjusted Weights Example]")
    base = {'growth': 25, 'momentum': 25, 'quality': 25, 'value': 25}

    print(f"  Base: {base}")

    nasdaq_adj = calculate_adjusted_weights(base, exchange='NASDAQ')
    print(f"  NASDAQ: {nasdaq_adj}")

    healthcare_adj = calculate_adjusted_weights(base, sector='HEALTHCARE')
    print(f"  HEALTHCARE: {healthcare_adj}")

    combined = calculate_adjusted_weights(base, exchange='NASDAQ', sector='HEALTHCARE')
    print(f"  NASDAQ + HEALTHCARE: {combined}")
