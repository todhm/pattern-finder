"""
US Sector Dynamic Weights

Phase 3.1.3 섹터별 동적 가중치 최적화

IC 분석 결과 기반 섹터별 최적 팩터 가중치:
- HEALTHCARE: Growth IC 0.034 (유일한 양의 IC)
- FINANCIAL SERVICES: EM7/Momentum IC 0.415 (최고)
- TECHNOLOGY: Growth IC 0.052
- CONSUMER DEFENSIVE: Quality IC 0.169
- ENERGY: Quality IC -0.052 (음수)
- BASIC MATERIALS: 모든 팩터 균형

Usage:
    optimizer = USSectorDynamicWeights()

    # 단일 종목 가중치 조정
    weights = optimizer.get_adjusted_weights(base_weights, 'HEALTHCARE')

    # 거래소 + 섹터 통합 조정
    weights = optimizer.get_combined_weights(base_weights, 'NASDAQ', 'HEALTHCARE')

    # 배치 처리 (DB 연동)
    await optimizer.apply_sector_adjustments(db_manager, target_date)

File: us/us_sector_dynamic_weights.py
Created: 2025-11-29
"""

import logging
from typing import Dict, List, Optional, Tuple
from datetime import date

from weight_adjustments import (
    SECTOR_ADJUSTMENTS,
    EXCHANGE_ADJUSTMENTS,
    get_sector_adjustment,
    get_exchange_adjustment,
    calculate_adjusted_weights,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# 섹터 목록
SECTORS = [
    'TECHNOLOGY',
    'HEALTHCARE',
    'FINANCIAL SERVICES',
    'CONSUMER CYCLICAL',
    'CONSUMER DEFENSIVE',
    'INDUSTRIALS',
    'ENERGY',
    'UTILITIES',
    'REAL ESTATE',
    'BASIC MATERIALS',
    'COMMUNICATION SERVICES'
]


class USSectorDynamicWeights:
    """
    섹터별 동적 팩터 가중치 최적화

    IC 분석 결과 기반으로 각 섹터의 팩터 가중치를 조정하여
    예측력을 개선합니다.
    """

    # 기본 가중치 (레짐 미적용 시)
    DEFAULT_BASE_WEIGHTS = {
        'growth': 25.0,
        'momentum': 25.0,
        'quality': 25.0,
        'value': 25.0
    }

    def __init__(self):
        """Initialize optimizer"""
        self.sector_adjustments = SECTOR_ADJUSTMENTS
        self.exchange_adjustments = EXCHANGE_ADJUSTMENTS
        self._cache: Dict[str, Dict] = {}

    def get_adjusted_weights(
        self,
        base_weights: Dict[str, float],
        sector: str
    ) -> Dict[str, float]:
        """
        섹터별 조정된 가중치 반환

        Args:
            base_weights: 기본 가중치 (레짐 기반)
            sector: 섹터명

        Returns:
            정규화된 조정 가중치 (합계 100)
        """
        return calculate_adjusted_weights(base_weights, sector=sector)

    def get_combined_weights(
        self,
        base_weights: Dict[str, float],
        exchange: str,
        sector: str
    ) -> Dict[str, float]:
        """
        거래소 + 섹터 통합 조정된 가중치 반환

        Args:
            base_weights: 기본 가중치
            exchange: 거래소 코드
            sector: 섹터명

        Returns:
            정규화된 통합 조정 가중치 (합계 100)
        """
        return calculate_adjusted_weights(base_weights, exchange=exchange, sector=sector)

    def get_adjustment_delta(self, sector: str) -> Dict[str, int]:
        """
        섹터별 조정값 (delta) 반환

        Args:
            sector: 섹터명

        Returns:
            조정값 딕셔너리
        """
        return get_sector_adjustment(sector)

    def calculate_final_score(
        self,
        factor_scores: Dict[str, float],
        base_weights: Dict[str, float],
        exchange: str,
        sector: str
    ) -> Tuple[float, Dict[str, float]]:
        """
        거래소 + 섹터 조정 적용한 최종 점수 계산

        Args:
            factor_scores: 팩터별 점수
            base_weights: 기본 가중치
            exchange: 거래소 코드
            sector: 섹터명

        Returns:
            Tuple of (final_score, applied_weights)
        """
        # 통합 가중치 조정
        adjusted_weights = self.get_combined_weights(base_weights, exchange, sector)

        # 최종 점수 계산
        final_score = sum(
            factor_scores.get(f'{factor}_score', 0) * (weight / 100)
            for factor, weight in adjusted_weights.items()
        )

        return round(final_score, 2), adjusted_weights

    async def apply_sector_adjustments(
        self,
        db_manager,
        target_date: Optional[date] = None,
        base_weights: Optional[Dict[str, float]] = None,
        include_exchange: bool = True
    ) -> int:
        """
        전체 종목에 섹터별 가중치 조정 적용 (배치 처리)

        Args:
            db_manager: AsyncDatabaseManager instance
            target_date: 대상 날짜
            base_weights: 기본 가중치
            include_exchange: 거래소 조정도 포함할지 여부

        Returns:
            업데이트된 레코드 수
        """
        if base_weights is None:
            base_weights = self.DEFAULT_BASE_WEIGHTS

        target_date = target_date or date.today()

        logger.info(f"Applying sector adjustments for {target_date}")
        logger.info(f"Include exchange adjustment: {include_exchange}")

        # SQL CASE WHEN 생성
        sql_case = self._generate_sector_sql_case(base_weights, include_exchange)

        update_query = f"""
        UPDATE us_stock_grade
        SET
            final_score = {sql_case['final_score']},
            weight_growth = {sql_case['weight_growth']},
            weight_momentum = {sql_case['weight_momentum']},
            weight_quality = {sql_case['weight_quality']},
            weight_value = {sql_case['weight_value']}
        WHERE base_date = $1
        """

        try:
            await db_manager.execute(update_query, target_date)

            # 업데이트된 행 수 조회
            count_query = """
            SELECT COUNT(*) as cnt FROM us_stock_grade WHERE base_date = $1
            """
            result = await db_manager.execute_query(count_query, target_date)
            updated_count = result[0]['cnt'] if result else 0

            logger.info(f"Sector adjustments applied to {updated_count} stocks")
            return updated_count

        except Exception as e:
            logger.error(f"Failed to apply sector adjustments: {e}")
            raise

    def _generate_sector_sql_case(
        self,
        base_weights: Dict[str, float],
        include_exchange: bool = True
    ) -> Dict[str, str]:
        """
        섹터별 SQL CASE WHEN 문 생성

        Args:
            base_weights: 기본 가중치
            include_exchange: 거래소 조정 포함 여부

        Returns:
            Dict with SQL statements for each column
        """
        # 모든 섹터 x 거래소 조합의 가중치 계산
        weights_map = {}

        exchanges = ['NYSE', 'NASDAQ', 'AMEX'] if include_exchange else [None]

        for sector in SECTORS + [None]:  # None = 기타 섹터
            for exchange in exchanges:
                key = (exchange, sector)

                if include_exchange and exchange:
                    weights = self.get_combined_weights(
                        base_weights,
                        exchange,
                        sector if sector else ''
                    )
                else:
                    weights = self.get_adjusted_weights(
                        base_weights,
                        sector if sector else ''
                    )

                weights_map[key] = weights

        # SQL CASE 문 생성
        factors = ['growth', 'momentum', 'quality', 'value']
        sql_cases = {}

        for factor in factors:
            case_parts = []

            for (exchange, sector), weights in weights_map.items():
                weight_val = weights[factor] / 100

                if include_exchange and exchange and sector:
                    condition = f"exchange = '{exchange}' AND sector = '{sector}'"
                elif include_exchange and exchange:
                    condition = f"exchange = '{exchange}' AND (sector IS NULL OR sector NOT IN ({self._sector_list_sql()}))"
                elif sector:
                    condition = f"sector = '{sector}'"
                else:
                    continue

                case_parts.append(f"WHEN {condition} THEN {weight_val}")

            # 기본값
            default_weight = base_weights[factor] / 100
            case_sql = f"CASE\n            " + "\n            ".join(case_parts) + f"\n            ELSE {default_weight}\n        END"

            sql_cases[f'weight_{factor}'] = f"({case_sql}) * 100"

        # final_score 계산
        final_score_parts = []
        for factor in factors:
            factor_case = sql_cases[f'weight_{factor}'].replace(' * 100', '')
            final_score_parts.append(f"{factor}_score * {factor_case}")

        sql_cases['final_score'] = "(" + " + ".join(final_score_parts) + ")"

        return sql_cases

    def _sector_list_sql(self) -> str:
        """섹터 목록을 SQL IN 절용 문자열로 변환"""
        return ", ".join(f"'{s}'" for s in SECTORS)

    async def get_sector_distribution(
        self,
        db_manager,
        target_date: Optional[date] = None
    ) -> Dict[str, int]:
        """
        섹터별 종목 분포 조회

        Args:
            db_manager: AsyncDatabaseManager instance
            target_date: 대상 날짜

        Returns:
            Dict[sector, count]
        """
        target_date = target_date or date.today()

        query = """
        SELECT sector, COUNT(*) as cnt
        FROM us_stock_grade
        WHERE base_date = $1
        GROUP BY sector
        ORDER BY cnt DESC
        """

        try:
            result = await db_manager.execute_query(query, target_date)
            return {row['sector']: row['cnt'] for row in result}
        except Exception as e:
            logger.error(f"Failed to get sector distribution: {e}")
            return {}

    def get_sector_weight_summary(
        self,
        base_weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        모든 섹터의 조정된 가중치 요약

        Args:
            base_weights: 기본 가중치

        Returns:
            Dict[sector, weights]
        """
        if base_weights is None:
            base_weights = self.DEFAULT_BASE_WEIGHTS

        summary = {}
        for sector in SECTORS:
            summary[sector] = self.get_adjusted_weights(base_weights, sector)

        return summary


# ============================================================================
# Standalone Functions
# ============================================================================

def get_all_sector_weights(
    base_weights: Optional[Dict[str, float]] = None
) -> Dict[str, Dict[str, float]]:
    """
    모든 섹터의 가중치 조정 반환

    Args:
        base_weights: 기본 가중치

    Returns:
        Dict[sector, weights]
    """
    optimizer = USSectorDynamicWeights()
    return optimizer.get_sector_weight_summary(base_weights)


def calculate_score_with_sector(
    growth_score: float,
    momentum_score: float,
    quality_score: float,
    value_score: float,
    exchange: str,
    sector: str,
    base_weights: Optional[Dict[str, float]] = None
) -> float:
    """
    거래소 + 섹터 조정 적용한 최종 점수 계산 (함수형)

    Args:
        growth_score: Growth 팩터 점수
        momentum_score: Momentum 팩터 점수
        quality_score: Quality 팩터 점수
        value_score: Value 팩터 점수
        exchange: 거래소 코드
        sector: 섹터명
        base_weights: 기본 가중치

    Returns:
        최종 점수
    """
    optimizer = USSectorDynamicWeights()

    if base_weights is None:
        base_weights = optimizer.DEFAULT_BASE_WEIGHTS

    weights = optimizer.get_combined_weights(base_weights, exchange, sector)

    return (
        growth_score * (weights['growth'] / 100) +
        momentum_score * (weights['momentum'] / 100) +
        quality_score * (weights['quality'] / 100) +
        value_score * (weights['value'] / 100)
    )


# ============================================================================
# Test
# ============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("US Sector Dynamic Weights Test")
    print("=" * 70)

    optimizer = USSectorDynamicWeights()
    base = optimizer.DEFAULT_BASE_WEIGHTS

    print(f"\nBase Weights: {base}")

    print("\n[Sector Adjustments]")
    for sector in ['HEALTHCARE', 'FINANCIAL SERVICES', 'TECHNOLOGY', 'ENERGY', 'BASIC MATERIALS']:
        delta = optimizer.get_adjustment_delta(sector)
        adjusted = optimizer.get_adjusted_weights(base, sector)
        print(f"\n  {sector}:")
        print(f"    Delta: {delta}")
        print(f"    Adjusted: g={adjusted['growth']:.1f}, m={adjusted['momentum']:.1f}, "
              f"q={adjusted['quality']:.1f}, v={adjusted['value']:.1f}")

    print("\n[Combined: Exchange + Sector]")
    test_cases = [
        ('NYSE', 'FINANCIAL SERVICES'),
        ('NASDAQ', 'HEALTHCARE'),
        ('NASDAQ', 'TECHNOLOGY'),
    ]

    for exchange, sector in test_cases:
        combined = optimizer.get_combined_weights(base, exchange, sector)
        print(f"\n  {exchange} + {sector}:")
        print(f"    Weights: g={combined['growth']:.1f}, m={combined['momentum']:.1f}, "
              f"q={combined['quality']:.1f}, v={combined['value']:.1f}")

    print("\n[Score Calculation Example]")
    factor_scores = {
        'growth_score': 80,
        'momentum_score': 60,
        'quality_score': 70,
        'value_score': 50
    }
    print(f"  Factor Scores: {factor_scores}")

    for exchange, sector in [('NYSE', 'TECHNOLOGY'), ('NASDAQ', 'HEALTHCARE')]:
        score, weights = optimizer.calculate_final_score(
            factor_scores, base, exchange, sector
        )
        print(f"\n  {exchange} + {sector}:")
        print(f"    Final Score: {score}")
        print(f"    Weights: g={weights['growth']:.1f}, m={weights['momentum']:.1f}, "
              f"q={weights['quality']:.1f}, v={weights['value']:.1f}")

    print("\n" + "=" * 70)
    print("Test Complete")
    print("=" * 70)
