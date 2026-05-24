"""
US IC Monitor - Lagged IC & Factor Spread Monitoring (v2.0)

별도 검증/모니터링 도구 (메인 프로세스와 독립)

Features:
1. Lagged IC Monitor: 60일 지연 IC로 모델 유효성 검증 (Look-ahead Bias 없음)
2. Factor Spread Monitor: 상하위 분위 스프레드로 Factor Crowding 감지

Usage:
    python us_ic_monitor.py

References:
- arXiv:2010.08601 - IC Volatility (std 3-11x mean)
- arXiv:2410.14841v1 - Factor Timing "Siren Song"

File: us/us_ic_monitor.py
"""

import asyncio
import logging
from datetime import date, timedelta
from typing import Dict, Optional, List
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# IC Thresholds (학술 연구 기반 - v2.0)
# ============================================================================
# 기존 v1.0: HEALTHY >= 0.15 (과도하게 높음)
# v2.0: 학술 기준 완화 (IC >= 0.05면 유의미)

IC_THRESHOLDS = {
    'HEALTHY': 0.08,      # 0.08 이상: 양호
    'DEGRADED': 0.05,     # 0.05-0.08: 주의
    'CRITICAL': 0.02,     # 0.02-0.05: 위험
    'FAILED': 0.00        # 0.02 미만: 무효
}

IC_STATUS_DESCRIPTION = {
    'HEALTHY': '모델 정상 작동',
    'DEGRADED': '모델 효과 감소 - 모니터링 강화',
    'CRITICAL': '모델 효과 미미 - 가중치 조정 검토',
    'FAILED': '모델 무효화 - 긴급 점검 필요'
}


# ============================================================================
# Factor Spread Thresholds
# ============================================================================
# 상위 5분위 - 하위 5분위 점수 차이
# 정상: 30-40점, 압축: <25점 (Crowding), 확대: >45점

SPREAD_THRESHOLDS = {
    'WIDE': 45,           # 45 이상: Factor 분산 효과 강함
    'NORMAL': 30,         # 30-45: 정상
    'NARROW': 25,         # 25-30: 약간 압축
    'COMPRESSED': 0       # 25 미만: Factor Crowding 신호
}


class LaggedICMonitor:
    """
    지연된 IC 모니터링 시스템 (v2.0)

    핵심 아이디어:
    - 실시간 IC 계산 불가 (미래 수익률 필요)
    - 대신 60일 전 시점의 IC를 현재 판단에 사용
    - 60일 IC = 당시 점수와 이후 60일 수익률의 상관관계

    장점:
    - Look-ahead Bias 없음
    - 실제 구현 가능

    단점:
    - 60일 지연된 정보
    - 급격한 변화에 늦은 반응
    """

    def __init__(self, db_manager):
        """
        Args:
            db_manager: AsyncDatabaseManager instance
        """
        self.db = db_manager

    async def get_lagged_ic(
        self,
        lag_days: int = 60,
        lookback_window: int = 63,
        analysis_date: Optional[date] = None
    ) -> Dict:
        """
        지연된 IC 조회 (Look-ahead Bias 없음)

        Args:
            lag_days: IC 계산 완료까지 필요한 기간 (60일)
            lookback_window: IC 계산에 사용할 과거 데이터 기간
            analysis_date: 분석 기준일 (None이면 오늘)

        Returns:
            {
                'total_ic': float,
                'factor_ic': {'value': float, 'quality': float, ...},
                'status': str,
                'n_samples': int,
                'lag_days': int,
                'note': str
            }
        """
        analysis_date = analysis_date or date.today()

        query = """
        WITH lagged_scores AS (
            -- lag_days 이전 시점의 점수
            SELECT
                g.symbol,
                g.date as score_date,
                g.final_score,
                g.value_score,
                g.quality_score,
                g.momentum_score,
                g.growth_score
            FROM us_stock_grade g
            WHERE g.date <= $1::DATE - $2
              AND g.date >= $1::DATE - $2 - $3
              AND g.final_score IS NOT NULL
        ),
        forward_returns AS (
            -- 해당 시점 이후 lag_days 동안의 수익률
            SELECT
                s.symbol,
                s.score_date,
                s.final_score,
                s.value_score,
                s.quality_score,
                s.momentum_score,
                s.growth_score,
                (d_end.close - d_start.close) / NULLIF(d_start.close, 0) * 100 as forward_return
            FROM lagged_scores s
            JOIN us_daily d_start
                ON s.symbol = d_start.symbol
                AND d_start.date = (
                    SELECT MIN(date) FROM us_daily
                    WHERE symbol = s.symbol AND date >= s.score_date
                )
            JOIN us_daily d_end
                ON s.symbol = d_end.symbol
                AND d_end.date = (
                    SELECT MIN(date) FROM us_daily
                    WHERE symbol = s.symbol AND date >= s.score_date + $2
                )
            WHERE d_start.close > 0
        )
        SELECT
            CORR(final_score, forward_return) as total_ic,
            CORR(value_score, forward_return) as value_ic,
            CORR(quality_score, forward_return) as quality_ic,
            CORR(momentum_score, forward_return) as momentum_ic,
            CORR(growth_score, forward_return) as growth_ic,
            COUNT(*) as n_samples,
            AVG(forward_return) as avg_return,
            STDDEV(forward_return) as std_return
        FROM forward_returns
        WHERE forward_return IS NOT NULL
        """

        try:
            result = await self.db.execute_query(
                query, analysis_date, lag_days, lookback_window
            )

            if not result or not result[0].get('total_ic'):
                return self._default_response(lag_days)

            row = result[0]
            total_ic = float(row['total_ic']) if row['total_ic'] else 0

            return {
                'total_ic': round(total_ic, 4),
                'factor_ic': {
                    'value': round(float(row['value_ic'] or 0), 4),
                    'quality': round(float(row['quality_ic'] or 0), 4),
                    'momentum': round(float(row['momentum_ic'] or 0), 4),
                    'growth': round(float(row['growth_ic'] or 0), 4)
                },
                'status': self._classify_status(total_ic),
                'status_description': IC_STATUS_DESCRIPTION[self._classify_status(total_ic)],
                'n_samples': int(row['n_samples'] or 0),
                'avg_return': round(float(row['avg_return'] or 0), 2),
                'std_return': round(float(row['std_return'] or 0), 2),
                'lag_days': lag_days,
                'lookback_window': lookback_window,
                'analysis_date': str(analysis_date),
                'note': f'IC calculated {lag_days} days ago (no look-ahead bias)'
            }

        except Exception as e:
            logger.error(f"Lagged IC calculation failed: {e}")
            return self._default_response(lag_days)

    def _classify_status(self, ic: float) -> str:
        """IC 상태 분류 (v2.0 임계값)"""
        if ic >= IC_THRESHOLDS['HEALTHY']:
            return 'HEALTHY'
        elif ic >= IC_THRESHOLDS['DEGRADED']:
            return 'DEGRADED'
        elif ic >= IC_THRESHOLDS['CRITICAL']:
            return 'CRITICAL'
        else:
            return 'FAILED'

    def _default_response(self, lag_days: int) -> Dict:
        """기본 응답"""
        return {
            'total_ic': None,
            'factor_ic': {
                'value': None,
                'quality': None,
                'momentum': None,
                'growth': None
            },
            'status': 'UNKNOWN',
            'status_description': '데이터 부족',
            'n_samples': 0,
            'lag_days': lag_days,
            'note': 'Insufficient data for IC calculation'
        }

    async def get_ic_trend(
        self,
        periods: int = 6,
        lag_days: int = 60,
        lookback_window: int = 63
    ) -> List[Dict]:
        """
        IC 추이 조회 (여러 기간)

        Args:
            periods: 조회할 기간 수
            lag_days: 각 기간의 lag
            lookback_window: 각 기간의 lookback

        Returns:
            List of IC results for each period
        """
        results = []
        base_date = date.today()

        for i in range(periods):
            period_date = base_date - timedelta(days=i * 30)  # 월별
            ic_result = await self.get_lagged_ic(
                lag_days=lag_days,
                lookback_window=lookback_window,
                analysis_date=period_date
            )
            ic_result['period'] = i
            ic_result['period_date'] = str(period_date)
            results.append(ic_result)

        return results


class FactorSpreadMonitor:
    """
    Factor Spread 모니터링 (v2.0)

    IC 대신 사용할 수 있는 실시간 지표:
    - High Score 종목 vs Low Score 종목의 점수 갭
    - Factor Crowding 프록시
    - 실시간 계산 가능 (Look-ahead Bias 없음)
    """

    def __init__(self, db_manager):
        """
        Args:
            db_manager: AsyncDatabaseManager instance
        """
        self.db = db_manager

    async def calculate_factor_spread(
        self,
        analysis_date: Optional[date] = None
    ) -> Dict:
        """
        High Score vs Low Score 종목 간 스프레드 계산

        스프레드가 좁아지면 → Factor Crowding 신호
        스프레드가 넓어지면 → Factor 효과 유효

        Args:
            analysis_date: 분석 날짜 (None이면 최신)

        Returns:
            {
                'top_quintile_avg': float,
                'bottom_quintile_avg': float,
                'score_spread': float,
                'spread_status': str,
                'factor_signal': str
            }
        """
        query = """
        WITH scored_stocks AS (
            SELECT
                symbol,
                final_score,
                NTILE(5) OVER (ORDER BY final_score) as quintile
            FROM us_stock_grade
            WHERE date = COALESCE($1::DATE, (
                SELECT MAX(date) FROM us_stock_grade
            ))
            AND final_score IS NOT NULL
        ),
        quintile_stats AS (
            SELECT
                quintile,
                AVG(final_score) as avg_score,
                COUNT(*) as count,
                STDDEV(final_score) as std_score
            FROM scored_stocks
            GROUP BY quintile
        )
        SELECT
            (SELECT avg_score FROM quintile_stats WHERE quintile = 5) as top_quintile_avg,
            (SELECT avg_score FROM quintile_stats WHERE quintile = 1) as bottom_quintile_avg,
            (SELECT count FROM quintile_stats WHERE quintile = 5) as top_count,
            (SELECT count FROM quintile_stats WHERE quintile = 1) as bottom_count,
            (SELECT std_score FROM quintile_stats WHERE quintile = 5) as top_std,
            (SELECT std_score FROM quintile_stats WHERE quintile = 1) as bottom_std
        """

        try:
            result = await self.db.execute_query(query, analysis_date)

            if not result or not result[0].get('top_quintile_avg'):
                return self._default_response()

            row = result[0]
            top_avg = float(row['top_quintile_avg'])
            bottom_avg = float(row['bottom_quintile_avg'])
            spread = top_avg - bottom_avg

            # 스프레드 상태 분류
            if spread >= SPREAD_THRESHOLDS['WIDE']:
                spread_status = 'WIDE'
                factor_signal = 'STRONG'
                recommendation = '팩터 효과 강함 - 현재 전략 유지'
            elif spread >= SPREAD_THRESHOLDS['NORMAL']:
                spread_status = 'NORMAL'
                factor_signal = 'NORMAL'
                recommendation = '팩터 효과 정상 - 모니터링 지속'
            elif spread >= SPREAD_THRESHOLDS['NARROW']:
                spread_status = 'NARROW'
                factor_signal = 'WEAK'
                recommendation = '팩터 효과 약화 - 주의 필요'
            else:
                spread_status = 'COMPRESSED'
                factor_signal = 'CROWDING_ALERT'
                recommendation = 'Factor Crowding 의심 - 전략 검토 필요'

            return {
                'top_quintile_avg': round(top_avg, 2),
                'bottom_quintile_avg': round(bottom_avg, 2),
                'score_spread': round(spread, 2),
                'spread_status': spread_status,
                'factor_signal': factor_signal,
                'recommendation': recommendation,
                'top_count': int(row['top_count'] or 0),
                'bottom_count': int(row['bottom_count'] or 0),
                'top_std': round(float(row['top_std'] or 0), 2),
                'bottom_std': round(float(row['bottom_std'] or 0), 2),
                'analysis_date': str(analysis_date) if analysis_date else 'latest'
            }

        except Exception as e:
            logger.error(f"Factor spread calculation failed: {e}")
            return self._default_response()

    def _default_response(self) -> Dict:
        """기본 응답"""
        return {
            'top_quintile_avg': None,
            'bottom_quintile_avg': None,
            'score_spread': None,
            'spread_status': 'UNKNOWN',
            'factor_signal': 'UNKNOWN',
            'recommendation': '데이터 부족',
            'analysis_date': None
        }

    async def get_spread_by_factor(
        self,
        analysis_date: Optional[date] = None
    ) -> Dict:
        """
        개별 팩터별 스프레드 계산

        Args:
            analysis_date: 분석 날짜

        Returns:
            Dict with spread for each factor
        """
        factors = ['value_score', 'quality_score', 'momentum_score', 'growth_score']
        results = {}

        for factor in factors:
            query = f"""
            WITH scored_stocks AS (
                SELECT
                    symbol,
                    {factor},
                    NTILE(5) OVER (ORDER BY {factor}) as quintile
                FROM us_stock_grade
                WHERE date = COALESCE($1::DATE, (
                    SELECT MAX(date) FROM us_stock_grade
                ))
                AND {factor} IS NOT NULL
            )
            SELECT
                (SELECT AVG({factor}) FROM scored_stocks WHERE quintile = 5) as top_avg,
                (SELECT AVG({factor}) FROM scored_stocks WHERE quintile = 1) as bottom_avg
            """

            try:
                result = await self.db.execute_query(query, analysis_date)
                if result and result[0].get('top_avg'):
                    top_avg = float(result[0]['top_avg'])
                    bottom_avg = float(result[0]['bottom_avg'])
                    spread = top_avg - bottom_avg

                    factor_name = factor.replace('_score', '')
                    results[factor_name] = {
                        'top_avg': round(top_avg, 2),
                        'bottom_avg': round(bottom_avg, 2),
                        'spread': round(spread, 2)
                    }
            except Exception as e:
                logger.warning(f"Factor {factor} spread calculation failed: {e}")

        return results


class ICMonitor:
    """
    통합 IC 모니터링 클래스

    Lagged IC + Factor Spread 통합 리포트 생성
    """

    def __init__(self, db_manager):
        """
        Args:
            db_manager: AsyncDatabaseManager instance
        """
        self.db = db_manager
        self.lagged_ic_monitor = LaggedICMonitor(db_manager)
        self.factor_spread_monitor = FactorSpreadMonitor(db_manager)

    async def generate_report(
        self,
        analysis_date: Optional[date] = None,
        lag_days: int = 60
    ) -> Dict:
        """
        통합 모니터링 리포트 생성

        Args:
            analysis_date: 분석 날짜
            lag_days: Lagged IC 계산용 lag

        Returns:
            통합 리포트 Dict
        """
        # Lagged IC
        lagged_ic = await self.lagged_ic_monitor.get_lagged_ic(
            lag_days=lag_days,
            analysis_date=analysis_date
        )

        # Factor Spread
        factor_spread = await self.factor_spread_monitor.calculate_factor_spread(
            analysis_date=analysis_date
        )

        # Factor별 Spread
        factor_spreads = await self.factor_spread_monitor.get_spread_by_factor(
            analysis_date=analysis_date
        )

        # 종합 판단
        overall_status = self._determine_overall_status(lagged_ic, factor_spread)

        return {
            'analysis_date': str(analysis_date) if analysis_date else str(date.today()),
            'lagged_ic': lagged_ic,
            'factor_spread': factor_spread,
            'factor_spreads_detail': factor_spreads,
            'overall_status': overall_status,
            'recommendations': self._generate_recommendations(lagged_ic, factor_spread)
        }

    def _determine_overall_status(self, lagged_ic: Dict, factor_spread: Dict) -> str:
        """종합 상태 판단"""
        ic_status = lagged_ic.get('status', 'UNKNOWN')
        spread_status = factor_spread.get('spread_status', 'UNKNOWN')

        if ic_status == 'HEALTHY' and spread_status in ['WIDE', 'NORMAL']:
            return 'HEALTHY'
        elif ic_status in ['HEALTHY', 'DEGRADED'] and spread_status in ['NORMAL', 'NARROW']:
            return 'MODERATE'
        elif ic_status in ['CRITICAL', 'FAILED'] or spread_status == 'COMPRESSED':
            return 'WARNING'
        else:
            return 'UNKNOWN'

    def _generate_recommendations(self, lagged_ic: Dict, factor_spread: Dict) -> List[str]:
        """권고 사항 생성"""
        recommendations = []

        # IC 기반 권고
        ic_status = lagged_ic.get('status', 'UNKNOWN')
        if ic_status == 'HEALTHY':
            recommendations.append("IC 양호 - 현재 모델 유지")
        elif ic_status == 'DEGRADED':
            recommendations.append("IC 하락 추세 - 분기 리밸런싱 시 검토")
        elif ic_status == 'CRITICAL':
            recommendations.append("IC 위험 수준 - 가중치 조정 필요")
        elif ic_status == 'FAILED':
            recommendations.append("IC 무효 - 모델 재검토 긴급")

        # Factor Spread 기반 권고
        spread_status = factor_spread.get('spread_status', 'UNKNOWN')
        if spread_status == 'COMPRESSED':
            recommendations.append("Factor Crowding 의심 - 포지션 분산 검토")
        elif spread_status == 'NARROW':
            recommendations.append("Factor 효과 약화 - 모니터링 강화")

        # 개별 팩터 IC 권고
        factor_ic = lagged_ic.get('factor_ic', {})
        for factor, ic_value in factor_ic.items():
            if ic_value is not None and ic_value < 0:
                recommendations.append(f"{factor.upper()} IC 음수 ({ic_value:.3f}) - 가중치 축소 검토")

        return recommendations


# ============================================================================
# Main Execution
# ============================================================================

async def main():
    """메인 실행"""
    from us_db_async import AsyncDatabaseManager

    print("\n" + "=" * 80)
    print("US IC Monitor - 모델 검증 리포트")
    print("=" * 80)

    # DB 연결
    db_manager = AsyncDatabaseManager()
    await db_manager.initialize()

    try:
        # 통합 모니터 생성
        monitor = ICMonitor(db_manager)

        # 리포트 생성
        report = await monitor.generate_report()

        # 결과 출력
        print(f"\n분석 날짜: {report['analysis_date']}")
        print(f"종합 상태: {report['overall_status']}")

        print("\n" + "-" * 40)
        print("[Lagged IC (60일 지연)]")
        print("-" * 40)
        ic = report['lagged_ic']
        print(f"  Total IC: {ic.get('total_ic', 'N/A')}")
        print(f"  상태: {ic.get('status', 'N/A')} - {ic.get('status_description', '')}")
        print(f"  샘플 수: {ic.get('n_samples', 0):,}")

        if ic.get('factor_ic'):
            print(f"\n  Factor별 IC:")
            for factor, value in ic['factor_ic'].items():
                if value is not None:
                    indicator = "+" if value > 0 else ""
                    print(f"    {factor:12}: {indicator}{value:.4f}")

        print("\n" + "-" * 40)
        print("[Factor Spread]")
        print("-" * 40)
        spread = report['factor_spread']
        print(f"  상위 5분위 평균: {spread.get('top_quintile_avg', 'N/A')}")
        print(f"  하위 5분위 평균: {spread.get('bottom_quintile_avg', 'N/A')}")
        print(f"  스프레드: {spread.get('score_spread', 'N/A')}")
        print(f"  상태: {spread.get('spread_status', 'N/A')}")
        print(f"  신호: {spread.get('factor_signal', 'N/A')}")

        print("\n" + "-" * 40)
        print("[권고 사항]")
        print("-" * 40)
        for i, rec in enumerate(report['recommendations'], 1):
            print(f"  {i}. {rec}")

        print("\n" + "=" * 80)

    finally:
        await db_manager.close()


if __name__ == '__main__':
    asyncio.run(main())
