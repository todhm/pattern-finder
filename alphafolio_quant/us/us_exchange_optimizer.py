"""
US Exchange Optimizer

Phase 3.1.2 거래소별 가중치 최적화

IC 분석 결과:
- NYSE: final_score IC 0.050 (기준)
- NASDAQ: final_score IC -0.008 (Growth만 양의 IC)

NASDAQ에서는 Growth 비중을 높이고 Momentum/Quality 비중을 낮춰서
예측력을 개선합니다.

Usage:
    optimizer = USExchangeOptimizer()

    # 단일 종목 가중치 조정
    weights = optimizer.get_adjusted_weights(base_weights, 'NASDAQ')

    # 배치 처리 (DB 연동)
    await optimizer.apply_exchange_adjustments(db_manager, target_date)

File: us/us_exchange_optimizer.py
Created: 2025-11-29
"""

import logging
from typing import Dict, List, Optional, Tuple
from datetime import date

from weight_adjustments import (
    EXCHANGE_ADJUSTMENTS,
    get_exchange_adjustment,
    calculate_adjusted_weights,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class USExchangeOptimizer:
    """
    거래소별 팩터 가중치 최적화

    IC 분석 결과 기반으로 NYSE/NASDAQ/AMEX 각 거래소의
    팩터 가중치를 조정하여 예측력을 개선합니다.
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
        self.adjustments = EXCHANGE_ADJUSTMENTS
        self._cache: Dict[str, Dict] = {}

    def get_adjusted_weights(
        self,
        base_weights: Dict[str, float],
        exchange: str
    ) -> Dict[str, float]:
        """
        거래소별 조정된 가중치 반환

        Args:
            base_weights: 기본 가중치 (레짐 기반)
            exchange: 거래소 코드 (NYSE, NASDAQ, AMEX)

        Returns:
            정규화된 조정 가중치 (합계 100)
        """
        return calculate_adjusted_weights(base_weights, exchange=exchange)

    def get_adjustment_delta(self, exchange: str) -> Dict[str, int]:
        """
        거래소별 조정값 (delta) 반환

        Args:
            exchange: 거래소 코드

        Returns:
            조정값 딕셔너리
        """
        return get_exchange_adjustment(exchange)

    def calculate_final_score(
        self,
        factor_scores: Dict[str, float],
        base_weights: Dict[str, float],
        exchange: str
    ) -> Tuple[float, Dict[str, float]]:
        """
        거래소 조정 적용한 최종 점수 계산

        Args:
            factor_scores: 팩터별 점수 (growth_score, momentum_score, etc.)
            base_weights: 기본 가중치
            exchange: 거래소 코드

        Returns:
            Tuple of (final_score, applied_weights)
        """
        # 가중치 조정
        adjusted_weights = self.get_adjusted_weights(base_weights, exchange)

        # 최종 점수 계산
        final_score = sum(
            factor_scores.get(f'{factor}_score', 0) * (weight / 100)
            for factor, weight in adjusted_weights.items()
        )

        return round(final_score, 2), adjusted_weights

    async def apply_exchange_adjustments(
        self,
        db_manager,
        target_date: Optional[date] = None,
        base_weights: Optional[Dict[str, float]] = None
    ) -> int:
        """
        전체 종목에 거래소별 가중치 조정 적용 (배치 처리)

        Args:
            db_manager: AsyncDatabaseManager instance
            target_date: 대상 날짜
            base_weights: 기본 가중치 (None이면 DEFAULT_BASE_WEIGHTS 사용)

        Returns:
            업데이트된 레코드 수
        """
        if base_weights is None:
            base_weights = self.DEFAULT_BASE_WEIGHTS

        target_date = target_date or date.today()

        logger.info(f"Applying exchange adjustments for {target_date}")
        logger.info(f"Base weights: {base_weights}")

        # 거래소별 조정된 가중치 계산
        exchange_weights = {}
        for exchange in ['NYSE', 'NASDAQ', 'AMEX']:
            adjusted = self.get_adjusted_weights(base_weights, exchange)
            exchange_weights[exchange] = adjusted
            logger.info(f"  {exchange}: {adjusted}")

        # SQL CASE WHEN으로 배치 업데이트
        # 한 번의 쿼리로 모든 거래소 처리
        update_query = f"""
        UPDATE us_stock_grade usg
        SET
            final_score = (
                growth_score * (
                    CASE exchange
                        WHEN 'NASDAQ' THEN {exchange_weights['NASDAQ']['growth'] / 100}
                        WHEN 'AMEX' THEN {exchange_weights['AMEX']['growth'] / 100}
                        ELSE {exchange_weights['NYSE']['growth'] / 100}
                    END
                ) +
                momentum_score * (
                    CASE exchange
                        WHEN 'NASDAQ' THEN {exchange_weights['NASDAQ']['momentum'] / 100}
                        WHEN 'AMEX' THEN {exchange_weights['AMEX']['momentum'] / 100}
                        ELSE {exchange_weights['NYSE']['momentum'] / 100}
                    END
                ) +
                quality_score * (
                    CASE exchange
                        WHEN 'NASDAQ' THEN {exchange_weights['NASDAQ']['quality'] / 100}
                        WHEN 'AMEX' THEN {exchange_weights['AMEX']['quality'] / 100}
                        ELSE {exchange_weights['NYSE']['quality'] / 100}
                    END
                ) +
                value_score * (
                    CASE exchange
                        WHEN 'NASDAQ' THEN {exchange_weights['NASDAQ']['value'] / 100}
                        WHEN 'AMEX' THEN {exchange_weights['AMEX']['value'] / 100}
                        ELSE {exchange_weights['NYSE']['value'] / 100}
                    END
                )
            ),
            weight_growth = CASE exchange
                WHEN 'NASDAQ' THEN {exchange_weights['NASDAQ']['growth']}
                WHEN 'AMEX' THEN {exchange_weights['AMEX']['growth']}
                ELSE {exchange_weights['NYSE']['growth']}
            END,
            weight_momentum = CASE exchange
                WHEN 'NASDAQ' THEN {exchange_weights['NASDAQ']['momentum']}
                WHEN 'AMEX' THEN {exchange_weights['AMEX']['momentum']}
                ELSE {exchange_weights['NYSE']['momentum']}
            END,
            weight_quality = CASE exchange
                WHEN 'NASDAQ' THEN {exchange_weights['NASDAQ']['quality']}
                WHEN 'AMEX' THEN {exchange_weights['AMEX']['quality']}
                ELSE {exchange_weights['NYSE']['quality']}
            END,
            weight_value = CASE exchange
                WHEN 'NASDAQ' THEN {exchange_weights['NASDAQ']['value']}
                WHEN 'AMEX' THEN {exchange_weights['AMEX']['value']}
                ELSE {exchange_weights['NYSE']['value']}
            END
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

            logger.info(f"Exchange adjustments applied to {updated_count} stocks")
            return updated_count

        except Exception as e:
            logger.error(f"Failed to apply exchange adjustments: {e}")
            raise

    async def get_exchange_distribution(
        self,
        db_manager,
        target_date: Optional[date] = None
    ) -> Dict[str, int]:
        """
        거래소별 종목 분포 조회

        Args:
            db_manager: AsyncDatabaseManager instance
            target_date: 대상 날짜

        Returns:
            Dict[exchange, count]
        """
        target_date = target_date or date.today()

        query = """
        SELECT exchange, COUNT(*) as cnt
        FROM us_stock_grade
        WHERE base_date = $1
        GROUP BY exchange
        ORDER BY cnt DESC
        """

        try:
            result = await db_manager.execute_query(query, target_date)
            return {row['exchange']: row['cnt'] for row in result}
        except Exception as e:
            logger.error(f"Failed to get exchange distribution: {e}")
            return {}

    def generate_sql_case_weights(
        self,
        base_weights: Dict[str, float]
    ) -> str:
        """
        SQL CASE WHEN 문 생성 (외부 사용용)

        Args:
            base_weights: 기본 가중치

        Returns:
            SQL CASE WHEN statement
        """
        # 거래소별 가중치 계산
        nasdaq_w = self.get_adjusted_weights(base_weights, 'NASDAQ')
        nyse_w = self.get_adjusted_weights(base_weights, 'NYSE')
        amex_w = self.get_adjusted_weights(base_weights, 'AMEX')

        sql = f"""
        -- Exchange-adjusted final_score calculation
        (
            growth_score * (
                CASE exchange
                    WHEN 'NASDAQ' THEN {nasdaq_w['growth'] / 100}
                    WHEN 'AMEX' THEN {amex_w['growth'] / 100}
                    ELSE {nyse_w['growth'] / 100}
                END
            ) +
            momentum_score * (
                CASE exchange
                    WHEN 'NASDAQ' THEN {nasdaq_w['momentum'] / 100}
                    WHEN 'AMEX' THEN {amex_w['momentum'] / 100}
                    ELSE {nyse_w['momentum'] / 100}
                END
            ) +
            quality_score * (
                CASE exchange
                    WHEN 'NASDAQ' THEN {nasdaq_w['quality'] / 100}
                    WHEN 'AMEX' THEN {amex_w['quality'] / 100}
                    ELSE {nyse_w['quality'] / 100}
                END
            ) +
            value_score * (
                CASE exchange
                    WHEN 'NASDAQ' THEN {nasdaq_w['value'] / 100}
                    WHEN 'AMEX' THEN {amex_w['value'] / 100}
                    ELSE {nyse_w['value'] / 100}
                END
            )
        )
        """
        return sql


# ============================================================================
# Standalone Functions
# ============================================================================

def get_exchange_weight_summary() -> Dict[str, Dict]:
    """
    모든 거래소의 가중치 조정 요약

    Returns:
        Dict[exchange, weights]
    """
    optimizer = USExchangeOptimizer()
    base = optimizer.DEFAULT_BASE_WEIGHTS

    return {
        exchange: optimizer.get_adjusted_weights(base, exchange)
        for exchange in ['NYSE', 'NASDAQ', 'AMEX']
    }


def calculate_score_with_exchange(
    growth_score: float,
    momentum_score: float,
    quality_score: float,
    value_score: float,
    exchange: str,
    base_weights: Optional[Dict[str, float]] = None
) -> float:
    """
    거래소 조정 적용한 최종 점수 계산 (함수형)

    Args:
        growth_score: Growth 팩터 점수
        momentum_score: Momentum 팩터 점수
        quality_score: Quality 팩터 점수
        value_score: Value 팩터 점수
        exchange: 거래소 코드
        base_weights: 기본 가중치 (optional)

    Returns:
        최종 점수
    """
    optimizer = USExchangeOptimizer()

    if base_weights is None:
        base_weights = optimizer.DEFAULT_BASE_WEIGHTS

    weights = optimizer.get_adjusted_weights(base_weights, exchange)

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
    print("=" * 60)
    print("US Exchange Optimizer Test")
    print("=" * 60)

    optimizer = USExchangeOptimizer()
    base = optimizer.DEFAULT_BASE_WEIGHTS

    print(f"\nBase Weights: {base}")

    print("\n[Exchange Adjustments]")
    for exchange in ['NYSE', 'NASDAQ', 'AMEX']:
        delta = optimizer.get_adjustment_delta(exchange)
        adjusted = optimizer.get_adjusted_weights(base, exchange)
        print(f"\n  {exchange}:")
        print(f"    Delta: {delta}")
        print(f"    Adjusted: {adjusted}")

    print("\n[Score Calculation Example]")
    factor_scores = {
        'growth_score': 80,
        'momentum_score': 60,
        'quality_score': 70,
        'value_score': 50
    }
    print(f"  Factor Scores: {factor_scores}")

    for exchange in ['NYSE', 'NASDAQ']:
        score, weights = optimizer.calculate_final_score(factor_scores, base, exchange)
        print(f"\n  {exchange}:")
        print(f"    Final Score: {score}")
        print(f"    Applied Weights: {weights}")

    print("\n[SQL CASE Statement]")
    sql = optimizer.generate_sql_case_weights(base)
    print(sql[:500] + "...")

    print("\n" + "=" * 60)
    print("Test Complete")
    print("=" * 60)
