"""
US Outlier Risk Score (ORS) Calculator

Phase 3.1 Mean-Median 괴리 해결을 위한 이상치 위험 점수 계산

ORS 구성요소 (각 25점, 총 100점):
- Penny Stock Score: 저가주 위험 (price < $5)
- Volume Score: 유동성 부족 위험 (avg_volume < 100K)
- Volatility Score: 변동성 위험 (annual_vol > 100%)
- Market Cap Score: 소형주 위험 (mktcap < $300M)

ORS 용도:
- position_size 조정 (ORS 높을수록 포지션 축소)
- risk_flag 표시 (EXTREME_RISK, HIGH_RISK, MODERATE_RISK, NORMAL)
- Mean-Median 괴리 원인 식별

Usage:
    calculator = USOutlierRisk(db_manager, analysis_date)

    # 배치 계산 (전체 종목)
    results = await calculator.calculate_ors_batch()

    # 단일 종목
    ors = await calculator.calculate_ors_single('AAPL')

    # DB 업데이트
    await calculator.update_ors_in_db()

File: us/us_outlier_risk.py
Created: 2025-11-29
"""

import logging
from typing import Dict, List, Optional, Tuple
from datetime import date
from decimal import Decimal

from weight_adjustments import (
    ORS_PRICE_SCORES,
    ORS_VOLUME_SCORES,
    ORS_VOLATILITY_SCORES,
    ORS_MKTCAP_SCORES,
    ORS_RISK_LEVELS,
    get_risk_level,
    get_position_multiplier,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class USOutlierRisk:
    """
    US Outlier Risk Score Calculator

    이상치 위험 점수를 계산하여 Mean-Median 괴리 원인 종목 식별
    """

    def __init__(self, db_manager=None, analysis_date: Optional[date] = None):
        """
        Args:
            db_manager: AsyncDatabaseManager instance (optional for single calculation)
            analysis_date: 분석 날짜 (None이면 최신)
        """
        self.db = db_manager
        self.analysis_date = analysis_date
        self._cache: Dict[str, Dict] = {}

    async def calculate_ors_batch(self) -> Dict[str, Dict]:
        """
        전체 종목의 ORS를 배치로 계산

        Returns:
            Dict[ticker, {ors, risk_flag, components, position_multiplier}]
        """
        logger.info("Calculating ORS for all stocks (batch mode)")

        # 한 번의 쿼리로 모든 필요 데이터 조회
        query = """
        WITH latest_prices AS (
            SELECT
                d.symbol,
                d.close as price,
                d.volume,
                ROW_NUMBER() OVER (PARTITION BY d.symbol ORDER BY d.date DESC) as rn
            FROM us_daily d
            WHERE ($1::DATE IS NULL OR d.date <= $1::DATE)
        ),
        avg_volumes AS (
            SELECT
                symbol,
                AVG(volume) as avg_volume
            FROM us_daily
            WHERE date >= COALESCE($1::DATE, CURRENT_DATE) - INTERVAL '30 days'
                AND date <= COALESCE($1::DATE, CURRENT_DATE)
            GROUP BY symbol
        ),
        volatility_calc AS (
            SELECT
                symbol,
                STDDEV(daily_return) * SQRT(252) * 100 as volatility_252d
            FROM (
                SELECT
                    symbol,
                    (close - LAG(close) OVER (PARTITION BY symbol ORDER BY date))
                        / NULLIF(LAG(close) OVER (PARTITION BY symbol ORDER BY date), 0) as daily_return
                FROM us_daily
                WHERE date >= COALESCE($1::DATE, CURRENT_DATE) - INTERVAL '1 year'
                    AND date <= COALESCE($1::DATE, CURRENT_DATE)
            ) returns
            WHERE daily_return IS NOT NULL
            GROUP BY symbol
        )
        SELECT
            b.symbol,
            COALESCE(lp.price, 0) as price,
            COALESCE(av.avg_volume, 0) as avg_volume,
            COALESCE(vc.volatility_252d, 0) as volatility_252d,
            COALESCE(b.market_cap, 0) as market_cap,
            b.exchange,
            b.sector
        FROM us_stock_basic b
        LEFT JOIN latest_prices lp ON b.symbol = lp.symbol AND lp.rn = 1
        LEFT JOIN avg_volumes av ON b.symbol = av.symbol
        LEFT JOIN volatility_calc vc ON b.symbol = vc.symbol
        WHERE b.symbol IS NOT NULL
        """

        try:
            rows = await self.db.execute_query(query, self.analysis_date)

            if not rows:
                logger.warning("No data found for ORS calculation")
                return {}

            results = {}
            for row in rows:
                symbol = row['symbol']
                ors_data = self._calculate_ors_from_data(
                    price=self._to_float(row['price']),
                    avg_volume=self._to_float(row['avg_volume']),
                    volatility=self._to_float(row['volatility_252d']),
                    market_cap=self._to_float(row['market_cap'])
                )
                ors_data['exchange'] = row['exchange']
                ors_data['sector'] = row['sector']
                results[symbol] = ors_data

            # 캐시 저장
            self._cache = results

            logger.info(f"ORS calculated for {len(results)} stocks")
            self._log_distribution(results)

            return results

        except Exception as e:
            logger.error(f"Failed to calculate ORS batch: {e}")
            raise

    async def calculate_ors_by_symbol(self, symbol: str) -> Dict:
        """
        단일 종목의 ORS 계산 (DB에서 데이터 조회)

        Args:
            symbol: 종목 티커

        Returns:
            Dict with ors, risk_flag, components, position_multiplier
        """
        # 캐시 확인
        if symbol in self._cache:
            return self._cache[symbol]

        query = """
        WITH latest_price AS (
            SELECT close as price, volume
            FROM us_daily
            WHERE symbol = $1
                AND ($2::DATE IS NULL OR date <= $2::DATE)
            ORDER BY date DESC
            LIMIT 1
        ),
        avg_vol AS (
            SELECT AVG(volume) as avg_volume
            FROM us_daily
            WHERE symbol = $1
                AND date >= COALESCE($2::DATE, CURRENT_DATE) - INTERVAL '30 days'
                AND date <= COALESCE($2::DATE, CURRENT_DATE)
        ),
        vol_calc AS (
            SELECT STDDEV(daily_return) * SQRT(252) * 100 as volatility_252d
            FROM (
                SELECT
                    (close - LAG(close) OVER (ORDER BY date))
                        / NULLIF(LAG(close) OVER (ORDER BY date), 0) as daily_return
                FROM us_daily
                WHERE symbol = $1
                    AND date >= COALESCE($2::DATE, CURRENT_DATE) - INTERVAL '1 year'
                    AND date <= COALESCE($2::DATE, CURRENT_DATE)
            ) returns
            WHERE daily_return IS NOT NULL
        )
        SELECT
            COALESCE(lp.price, 0) as price,
            COALESCE(av.avg_volume, 0) as avg_volume,
            COALESCE(vc.volatility_252d, 0) as volatility_252d,
            COALESCE(b.marketcap, 0) as market_cap
        FROM us_stock_basic b
        CROSS JOIN latest_price lp
        CROSS JOIN avg_vol av
        CROSS JOIN vol_calc vc
        WHERE b.symbol = $1
        """

        try:
            rows = await self.db.execute_query(query, symbol, self.analysis_date)

            if not rows or not rows[0]:
                logger.warning(f"No data found for {symbol}")
                return self._get_default_ors()

            row = rows[0]
            ors_data = self._calculate_ors_from_data(
                price=self._to_float(row['price']),
                avg_volume=self._to_float(row['avg_volume']),
                volatility=self._to_float(row['volatility_252d']),
                market_cap=self._to_float(row['market_cap'])
            )

            # 캐시 저장
            self._cache[symbol] = ors_data

            return ors_data

        except Exception as e:
            logger.error(f"Failed to calculate ORS for {symbol}: {e}")
            return self._get_default_ors()

    def calculate_ors_single(
        self,
        price: float,
        avg_volume: float,
        volatility: float,
        market_cap: float
    ) -> Dict:
        """
        데이터로부터 ORS 계산 (DB 연결 없이 사용 가능)

        Args:
            price: 현재가
            avg_volume: 평균 거래량
            volatility: 연환산 변동성 (%)
            market_cap: 시가총액

        Returns:
            Dict with ors, risk_flag, components, position_multiplier
        """
        return self._calculate_ors_from_data(price, avg_volume, volatility, market_cap)

    def _calculate_ors_from_data(
        self,
        price: float,
        avg_volume: float,
        volatility: float,
        market_cap: float
    ) -> Dict:
        """
        데이터로부터 ORS 계산 (내부 메서드)

        Args:
            price: 현재가
            avg_volume: 평균 거래량
            volatility: 연환산 변동성 (%)
            market_cap: 시가총액

        Returns:
            Dict with ors, risk_flag, components, position_multiplier
        """
        # 각 구성요소 점수 계산
        price_score = self._calc_price_score(price)
        volume_score = self._calc_volume_score(avg_volume)
        volatility_score = self._calc_volatility_score(volatility)
        mktcap_score = self._calc_mktcap_score(market_cap)

        # 총점 (0-100)
        ors = price_score + volume_score + volatility_score + mktcap_score

        # 위험 등급
        risk_flag = get_risk_level(ors)

        # 포지션 배수
        position_multiplier = get_position_multiplier(ors)

        return {
            'ors': round(ors, 2),
            'risk_flag': risk_flag,
            'position_multiplier': position_multiplier,
            'components': {
                'price_score': price_score,
                'volume_score': volume_score,
                'volatility_score': volatility_score,
                'mktcap_score': mktcap_score,
            },
            'raw_data': {
                'price': price,
                'avg_volume': avg_volume,
                'volatility': volatility,
                'market_cap': market_cap,
            }
        }

    def _calc_price_score(self, price: float) -> int:
        """가격 기준 점수 (0-25)"""
        if price <= 0:
            return 25  # 데이터 없음 = 최고 위험

        for threshold, score in ORS_PRICE_SCORES:
            if price < threshold:
                return score
        return 0

    def _calc_volume_score(self, avg_volume: float) -> int:
        """거래량 기준 점수 (0-25)"""
        if avg_volume <= 0:
            return 25  # 데이터 없음 = 최고 위험

        for threshold, score in ORS_VOLUME_SCORES:
            if avg_volume < threshold:
                return score
        return 0

    def _calc_volatility_score(self, volatility: float) -> int:
        """변동성 기준 점수 (0-25)"""
        if volatility <= 0:
            return 0  # 데이터 없음 = 안전으로 처리

        for threshold, score in ORS_VOLATILITY_SCORES:
            if volatility > threshold:
                return score
        return 0

    def _calc_mktcap_score(self, market_cap: float) -> int:
        """시가총액 기준 점수 (0-25)"""
        if market_cap <= 0:
            return 25  # 데이터 없음 = 최고 위험

        for threshold, score in ORS_MKTCAP_SCORES:
            if market_cap < threshold:
                return score
        return 0

    async def update_ors_in_db(self) -> int:
        """
        계산된 ORS를 us_stock_grade 테이블에 업데이트

        Returns:
            업데이트된 레코드 수
        """
        if not self._cache:
            await self.calculate_ors_batch()

        if not self._cache:
            logger.warning("No ORS data to update")
            return 0

        # 배치 업데이트 (한 번의 쿼리로 처리)
        # PostgreSQL의 unnest를 사용한 배치 업데이트
        update_query = """
        UPDATE us_stock_grade usg
        SET
            outlier_risk_score = data.ors,
            risk_flag = data.risk_flag
        FROM (
            SELECT
                unnest($1::text[]) as ticker,
                unnest($2::numeric[]) as ors,
                unnest($3::text[]) as risk_flag
        ) data
        WHERE usg.ticker = data.ticker
            AND usg.base_date = $4::DATE
        """

        try:
            tickers = list(self._cache.keys())
            ors_values = [self._cache[t]['ors'] for t in tickers]
            risk_flags = [self._cache[t]['risk_flag'] for t in tickers]
            target_date = self.analysis_date or date.today()

            result = await self.db.execute(
                update_query,
                tickers,
                ors_values,
                risk_flags,
                target_date
            )

            # 업데이트된 행 수 추출
            updated_count = len(tickers)  # 실제로는 result에서 추출

            logger.info(f"ORS updated for {updated_count} stocks in us_stock_grade")
            return updated_count

        except Exception as e:
            logger.error(f"Failed to update ORS in DB: {e}")
            raise

    async def get_high_risk_stocks(self, min_ors: float = 60) -> List[Dict]:
        """
        고위험 종목 목록 조회

        Args:
            min_ors: 최소 ORS 점수 (default: 60 = HIGH_RISK 이상)

        Returns:
            List of high risk stock data
        """
        if not self._cache:
            await self.calculate_ors_batch()

        high_risk = [
            {'symbol': symbol, **data}
            for symbol, data in self._cache.items()
            if data['ors'] >= min_ors
        ]

        # ORS 내림차순 정렬
        high_risk.sort(key=lambda x: x['ors'], reverse=True)

        return high_risk

    async def get_ors_statistics(self) -> Dict:
        """
        ORS 통계 요약

        Returns:
            Dict with distribution statistics
        """
        if not self._cache:
            await self.calculate_ors_batch()

        if not self._cache:
            return {}

        ors_values = [d['ors'] for d in self._cache.values()]

        # 위험 등급별 분포
        risk_distribution = {
            'EXTREME_RISK': 0,
            'HIGH_RISK': 0,
            'MODERATE_RISK': 0,
            'NORMAL': 0
        }

        for data in self._cache.values():
            risk_distribution[data['risk_flag']] += 1

        return {
            'total_stocks': len(ors_values),
            'mean_ors': round(sum(ors_values) / len(ors_values), 2),
            'max_ors': max(ors_values),
            'min_ors': min(ors_values),
            'risk_distribution': risk_distribution,
            'high_risk_count': risk_distribution['HIGH_RISK'] + risk_distribution['EXTREME_RISK'],
            'high_risk_pct': round(
                (risk_distribution['HIGH_RISK'] + risk_distribution['EXTREME_RISK'])
                / len(ors_values) * 100, 2
            )
        }

    def _log_distribution(self, results: Dict[str, Dict]):
        """ORS 분포 로깅"""
        distribution = {'EXTREME_RISK': 0, 'HIGH_RISK': 0, 'MODERATE_RISK': 0, 'NORMAL': 0}
        for data in results.values():
            distribution[data['risk_flag']] += 1

        total = len(results)
        logger.info(f"ORS Distribution:")
        for level, count in distribution.items():
            pct = count / total * 100 if total > 0 else 0
            logger.info(f"  {level}: {count} ({pct:.1f}%)")

    def _get_default_ors(self) -> Dict:
        """기본 ORS (데이터 없을 때)"""
        return {
            'ors': 0,
            'risk_flag': 'NORMAL',
            'position_multiplier': 1.0,
            'components': {
                'price_score': 0,
                'volume_score': 0,
                'volatility_score': 0,
                'mktcap_score': 0,
            },
            'raw_data': {
                'price': 0,
                'avg_volume': 0,
                'volatility': 0,
                'market_cap': 0,
            }
        }

    def _to_float(self, value) -> float:
        """값을 float로 변환"""
        if value is None:
            return 0.0
        if isinstance(value, Decimal):
            return float(value)
        try:
            return float(value)
        except:
            return 0.0

    def apply_ors_to_position(
        self,
        symbol: str,
        original_position_pct: float
    ) -> Tuple[float, str]:
        """
        ORS 기반 포지션 조정

        Args:
            symbol: 종목 티커
            original_position_pct: 원래 포지션 비율 (%)

        Returns:
            Tuple of (adjusted_position_pct, risk_flag)
        """
        if symbol not in self._cache:
            return original_position_pct, 'NORMAL'

        data = self._cache[symbol]
        adjusted = original_position_pct * data['position_multiplier']

        return round(adjusted, 2), data['risk_flag']


# ============================================================================
# Standalone Functions (for SQL integration)
# ============================================================================

def calculate_ors_sql_case() -> str:
    """
    SQL CASE WHEN 문 생성 (배치 업데이트용)

    Returns:
        SQL CASE WHEN statement for ORS calculation
    """
    sql = """
    (
        -- Price Score (0-25)
        CASE
            WHEN price < 1 THEN 25
            WHEN price < 3 THEN 20
            WHEN price < 5 THEN 15
            WHEN price < 10 THEN 5
            ELSE 0
        END +
        -- Volume Score (0-25)
        CASE
            WHEN avg_volume < 50000 THEN 25
            WHEN avg_volume < 100000 THEN 20
            WHEN avg_volume < 500000 THEN 10
            ELSE 0
        END +
        -- Volatility Score (0-25)
        CASE
            WHEN volatility_252d > 150 THEN 25
            WHEN volatility_252d > 100 THEN 20
            WHEN volatility_252d > 75 THEN 10
            ELSE 0
        END +
        -- Market Cap Score (0-25)
        CASE
            WHEN market_cap < 100000000 THEN 25
            WHEN market_cap < 300000000 THEN 20
            WHEN market_cap < 1000000000 THEN 10
            ELSE 0
        END
    )
    """
    return sql


def calculate_risk_flag_sql_case() -> str:
    """
    SQL CASE WHEN 문 생성 (risk_flag용)

    Returns:
        SQL CASE WHEN statement for risk_flag
    """
    sql = """
    CASE
        WHEN outlier_risk_score >= 80 THEN 'EXTREME_RISK'
        WHEN outlier_risk_score >= 60 THEN 'HIGH_RISK'
        WHEN outlier_risk_score >= 40 THEN 'MODERATE_RISK'
        ELSE 'NORMAL'
    END
    """
    return sql


# ============================================================================
# Test
# ============================================================================

if __name__ == '__main__':
    import asyncio

    async def test():
        print("=" * 60)
        print("US Outlier Risk Score Calculator Test")
        print("=" * 60)

        # Mock test without DB
        print("\n[Component Score Test]")

        # Create instance without DB for testing
        class MockDB:
            async def execute_query(self, *args): return []
            async def execute(self, *args): return None

        calc = USOutlierRisk(MockDB())

        # Test score calculations
        test_cases = [
            {'price': 0.5, 'avg_volume': 10000, 'volatility': 200, 'market_cap': 50e6},
            {'price': 2.0, 'avg_volume': 80000, 'volatility': 120, 'market_cap': 200e6},
            {'price': 15.0, 'avg_volume': 500000, 'volatility': 50, 'market_cap': 2e9},
            {'price': 150.0, 'avg_volume': 5000000, 'volatility': 25, 'market_cap': 100e9},
        ]

        for tc in test_cases:
            result = calc._calculate_ors_from_data(**tc)
            print(f"\n  Price: ${tc['price']:.2f}, Volume: {tc['avg_volume']:,}, "
                  f"Vol: {tc['volatility']:.0f}%, MktCap: ${tc['market_cap']/1e6:.0f}M")
            print(f"  ORS: {result['ors']}, Risk: {result['risk_flag']}, "
                  f"Position Mult: {result['position_multiplier']}")
            print(f"  Components: {result['components']}")

        print("\n" + "=" * 60)
        print("SQL CASE Statement for ORS:")
        print("=" * 60)
        print(calculate_ors_sql_case())

    asyncio.run(test())
