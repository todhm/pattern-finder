# -*- coding: utf-8 -*-
"""
Korean Outlier Risk Score (ORS) Calculator

Phase Portfolio: Mean-Median 괴리 해결을 위한 이상치 위험 점수 계산

ORS 구성요소 (각 25점, 총 100점):
- Price Score: 저가주 위험 (price < 2,000원)
- Liquidity Score: 유동성 부족 위험 (avg_trading_value < 5억원)
- Volatility Score: 변동성 위험 (annual_vol > 60%)
- Market Cap Score: 소형주 위험 (mktcap < 2,000억원)

ORS 용도:
- position_size 조정 (ORS 높을수록 포지션 축소)
- risk_flag 표시 (EXTREME_RISK, HIGH_RISK, MODERATE_RISK, NORMAL)
- Mean-Median 괴리 원인 식별

Usage:
    calculator = KROutlierRisk(db_manager, analysis_date)

    # 배치 계산 (전체 종목)
    results = await calculator.calculate_ors_batch()

    # 단일 종목
    ors = await calculator.calculate_ors_by_symbol('005930')

File: kr/kr_outlier_risk.py
Created: 2025-12-23
"""

import logging
from typing import Dict, List, Optional, Tuple
from datetime import date
from decimal import Decimal

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Korean ORS Thresholds (한국 시장 특화)
# ============================================================================

# 가격 기준 점수 (threshold, score) - 가격이 threshold 미만이면 해당 score 부여
# 설계 근거: 한국 시장 가격 구조 반영 (액면가 기준, 관리종목 분포)
KR_ORS_PRICE_SCORES: List[Tuple[int, int]] = [
    (500, 25),      # 500원 미만: 25점 (극위험 - 액면가 이하, 자본잠식 위험)
    (1000, 20),     # 1,000원 미만: 20점 (고위험 - 투기성 종목)
    (2000, 15),     # 2,000원 미만: 15점 (중위험 - 저가주)
    (5000, 5),      # 5,000원 미만: 5점 (주의 - 관찰 필요)
]

# 거래대금 기준 점수 (threshold, score) - 일평균 거래대금이 threshold 미만이면 해당 score 부여
# 설계 근거: 기획안 최소 기준 5억원, batch_weight.py 유동성 분류 참조
KR_ORS_LIQUIDITY_SCORES: List[Tuple[int, int]] = [
    (100_000_000, 25),       # 1억원 미만: 25점 (극저유동성 - 매매 사실상 불가)
    (500_000_000, 20),       # 5억원 미만: 20점 (저유동성 - 기획안 최소 기준 미달)
    (2_000_000_000, 10),     # 20억원 미만: 10점 (중저유동성 - 슬리피지 위험)
]

# 변동성 기준 점수 (threshold, score) - 연환산 변동성(%)이 threshold 초과이면 해당 score 부여
# 설계 근거: 한국 ±30% 일일 가격 제한 반영, 미국 대비 0.6배 조정
KR_ORS_VOLATILITY_SCORES: List[Tuple[float, int]] = [
    (80.0, 25),    # 80% 초과: 25점 (극고변동성)
    (60.0, 20),    # 60% 초과: 20점 (고변동성)
    (45.0, 10),    # 45% 초과: 10점 (중상 변동성)
]

# 시가총액 기준 점수 (threshold, score) - 시가총액이 threshold 미만이면 해당 score 부여
# 설계 근거: batch_weight.py 시가총액 분류 (SMALL < 2,000억원)
KR_ORS_MKTCAP_SCORES: List[Tuple[int, int]] = [
    (50_000_000_000, 25),       # 500억원 미만: 25점 (마이크로캡)
    (200_000_000_000, 20),      # 2,000억원 미만: 20점 (초소형주 - batch_weight SMALL)
    (500_000_000_000, 10),      # 5,000억원 미만: 10점 (소형주)
]

# 위험 등급 정의
KR_ORS_RISK_LEVELS: Dict[str, Dict] = {
    'EXTREME_RISK': {
        'min_score': 80,
        'max_score': 100,
        'position_multiplier': 0.25,  # 포지션 75% 축소
        'description': '극단적 이상치 위험 - 포지션 대폭 축소 권장'
    },
    'HIGH_RISK': {
        'min_score': 60,
        'max_score': 79,
        'position_multiplier': 0.50,  # 포지션 50% 축소
        'description': '높은 이상치 위험 - 포지션 축소 권장'
    },
    'MODERATE_RISK': {
        'min_score': 40,
        'max_score': 59,
        'position_multiplier': 0.75,  # 포지션 25% 축소
        'description': '중간 이상치 위험 - 주의 필요'
    },
    'NORMAL': {
        'min_score': 0,
        'max_score': 39,
        'position_multiplier': 1.00,  # 포지션 유지
        'description': '정상 - 이상치 위험 낮음'
    }
}


def get_risk_level(ors: float) -> str:
    """ORS 점수로 위험 등급 반환"""
    if ors >= 80:
        return 'EXTREME_RISK'
    elif ors >= 60:
        return 'HIGH_RISK'
    elif ors >= 40:
        return 'MODERATE_RISK'
    else:
        return 'NORMAL'


def get_position_multiplier(ors: float) -> float:
    """ORS 점수로 포지션 배수 반환"""
    risk_level = get_risk_level(ors)
    return KR_ORS_RISK_LEVELS[risk_level]['position_multiplier']


class KROutlierRisk:
    """
    Korean Outlier Risk Score Calculator

    이상치 위험 점수를 계산하여 Mean-Median 괴리 원인 종목 식별
    """

    def __init__(self, db_manager=None, analysis_date: Optional[date] = None):
        """
        Args:
            db_manager: AsyncDatabaseManager instance
            analysis_date: 분석 날짜 (None이면 최신)
        """
        self.db = db_manager
        self.analysis_date = analysis_date
        self._cache: Dict[str, Dict] = {}

    async def calculate_ors_batch(self) -> Dict[str, Dict]:
        """
        전체 종목의 ORS를 배치로 계산

        Returns:
            Dict[symbol, {ors, risk_flag, components, position_multiplier}]
        """
        logger.info("Calculating ORS for all Korean stocks (batch mode)")

        # 한 번의 쿼리로 모든 필요 데이터 조회
        query = """
        WITH latest_prices AS (
            SELECT
                t.symbol,
                t.close as price,
                t.market_cap,
                ROW_NUMBER() OVER (PARTITION BY t.symbol ORDER BY t.date DESC) as rn
            FROM kr_intraday_total t
            WHERE ($1::DATE IS NULL OR t.date <= $1::DATE)
        ),
        avg_trading_values AS (
            SELECT
                symbol,
                AVG(trading_value) as avg_trading_value
            FROM kr_intraday_total
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
                FROM kr_intraday_total
                WHERE date >= COALESCE($1::DATE, CURRENT_DATE) - INTERVAL '1 year'
                    AND date <= COALESCE($1::DATE, CURRENT_DATE)
            ) returns
            WHERE daily_return IS NOT NULL
            GROUP BY symbol
        )
        SELECT
            lp.symbol,
            COALESCE(lp.price, 0) as price,
            COALESCE(atv.avg_trading_value, 0) as avg_trading_value,
            COALESCE(vc.volatility_252d, 0) as volatility_252d,
            COALESCE(lp.market_cap, 0) as market_cap
        FROM latest_prices lp
        LEFT JOIN avg_trading_values atv ON lp.symbol = atv.symbol
        LEFT JOIN volatility_calc vc ON lp.symbol = vc.symbol
        WHERE lp.rn = 1
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
                    avg_trading_value=self._to_float(row['avg_trading_value']),
                    volatility=self._to_float(row['volatility_252d']),
                    market_cap=self._to_float(row['market_cap'])
                )
                results[symbol] = ors_data

            # 캐시 저장
            self._cache = results

            logger.info(f"ORS calculated for {len(results)} Korean stocks")
            self._log_distribution(results)

            return results

        except Exception as e:
            logger.error(f"Failed to calculate ORS batch: {e}")
            raise

    async def calculate_ors_by_symbol(self, symbol: str) -> Dict:
        """
        단일 종목의 ORS 계산 (DB에서 데이터 조회)

        Args:
            symbol: 종목 코드

        Returns:
            Dict with ors, risk_flag, components, position_multiplier
        """
        # 캐시 확인
        if symbol in self._cache:
            return self._cache[symbol]

        query = """
        WITH latest_price AS (
            SELECT close as price, market_cap
            FROM kr_intraday_total
            WHERE symbol = $1
                AND ($2::DATE IS NULL OR date <= $2::DATE)
            ORDER BY date DESC
            LIMIT 1
        ),
        avg_trading AS (
            SELECT AVG(trading_value) as avg_trading_value
            FROM kr_intraday_total
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
                FROM kr_intraday_total
                WHERE symbol = $1
                    AND date >= COALESCE($2::DATE, CURRENT_DATE) - INTERVAL '1 year'
                    AND date <= COALESCE($2::DATE, CURRENT_DATE)
            ) returns
            WHERE daily_return IS NOT NULL
        )
        SELECT
            COALESCE(lp.price, 0) as price,
            COALESCE(at.avg_trading_value, 0) as avg_trading_value,
            COALESCE(vc.volatility_252d, 0) as volatility_252d,
            COALESCE(lp.market_cap, 0) as market_cap
        FROM latest_price lp
        CROSS JOIN avg_trading at
        CROSS JOIN vol_calc vc
        """

        try:
            rows = await self.db.execute_query(query, symbol, self.analysis_date)

            if not rows or not rows[0]:
                logger.warning(f"No data found for {symbol}")
                return self._get_default_ors()

            row = rows[0]
            ors_data = self._calculate_ors_from_data(
                price=self._to_float(row['price']),
                avg_trading_value=self._to_float(row['avg_trading_value']),
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
        avg_trading_value: float,
        volatility: float,
        market_cap: float
    ) -> Dict:
        """
        데이터로부터 ORS 계산 (DB 연결 없이 사용 가능)

        Args:
            price: 현재가 (원)
            avg_trading_value: 일평균 거래대금 (원)
            volatility: 연환산 변동성 (%)
            market_cap: 시가총액 (원)

        Returns:
            Dict with ors, risk_flag, components, position_multiplier
        """
        return self._calculate_ors_from_data(price, avg_trading_value, volatility, market_cap)

    def _calculate_ors_from_data(
        self,
        price: float,
        avg_trading_value: float,
        volatility: float,
        market_cap: float
    ) -> Dict:
        """
        데이터로부터 ORS 계산 (내부 메서드)

        Args:
            price: 현재가 (원)
            avg_trading_value: 일평균 거래대금 (원)
            volatility: 연환산 변동성 (%)
            market_cap: 시가총액 (원)

        Returns:
            Dict with ors, risk_flag, components, position_multiplier
        """
        # 각 구성요소 점수 계산
        price_score = self._calc_price_score(price)
        liquidity_score = self._calc_liquidity_score(avg_trading_value)
        volatility_score = self._calc_volatility_score(volatility)
        mktcap_score = self._calc_mktcap_score(market_cap)

        # 총점 (0-100)
        ors = price_score + liquidity_score + volatility_score + mktcap_score

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
                'liquidity_score': liquidity_score,
                'volatility_score': volatility_score,
                'mktcap_score': mktcap_score,
            },
            'raw_data': {
                'price': price,
                'avg_trading_value': avg_trading_value,
                'volatility': volatility,
                'market_cap': market_cap,
            }
        }

    def _calc_price_score(self, price: float) -> int:
        """가격 기준 점수 (0-25)"""
        if price <= 0:
            return 25  # 데이터 없음 = 최고 위험

        for threshold, score in KR_ORS_PRICE_SCORES:
            if price < threshold:
                return score
        return 0

    def _calc_liquidity_score(self, avg_trading_value: float) -> int:
        """거래대금 기준 점수 (0-25)"""
        if avg_trading_value <= 0:
            return 25  # 데이터 없음 = 최고 위험

        for threshold, score in KR_ORS_LIQUIDITY_SCORES:
            if avg_trading_value < threshold:
                return score
        return 0

    def _calc_volatility_score(self, volatility: float) -> int:
        """변동성 기준 점수 (0-25)"""
        if volatility <= 0:
            return 0  # 데이터 없음 = 안전으로 처리

        for threshold, score in KR_ORS_VOLATILITY_SCORES:
            if volatility > threshold:
                return score
        return 0

    def _calc_mktcap_score(self, market_cap: float) -> int:
        """시가총액 기준 점수 (0-25)"""
        if market_cap <= 0:
            return 25  # 데이터 없음 = 최고 위험

        for threshold, score in KR_ORS_MKTCAP_SCORES:
            if market_cap < threshold:
                return score
        return 0

    def _to_float(self, value) -> float:
        """안전하게 float 변환"""
        if value is None:
            return 0.0
        if isinstance(value, Decimal):
            return float(value)
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    def _get_default_ors(self) -> Dict:
        """기본 ORS 결과 (데이터 없을 때)"""
        return {
            'ors': 0,
            'risk_flag': 'NORMAL',
            'position_multiplier': 1.0,
            'components': {
                'price_score': 0,
                'liquidity_score': 0,
                'volatility_score': 0,
                'mktcap_score': 0,
            },
            'raw_data': {
                'price': 0,
                'avg_trading_value': 0,
                'volatility': 0,
                'market_cap': 0,
            }
        }

    def _log_distribution(self, results: Dict[str, Dict]) -> None:
        """ORS 분포 로깅"""
        if not results:
            return

        risk_distribution = {
            'EXTREME_RISK': 0,
            'HIGH_RISK': 0,
            'MODERATE_RISK': 0,
            'NORMAL': 0
        }

        for data in results.values():
            risk_distribution[data['risk_flag']] += 1

        total = len(results)
        logger.info(f"ORS Distribution:")
        for level, count in risk_distribution.items():
            pct = count / total * 100 if total > 0 else 0
            logger.info(f"  {level}: {count} ({pct:.1f}%)")


# Test function
if __name__ == '__main__':
    print("=" * 60)
    print("Korean ORS Calculator - Test Cases")
    print("=" * 60)

    calc = KROutlierRisk()

    # Test case 1: Normal stock (Samsung Electronics-like)
    test1 = calc.calculate_ors_single(
        price=70000,
        avg_trading_value=500_000_000_000,  # 5000억원
        volatility=25.0,
        market_cap=400_000_000_000_000  # 400조원
    )
    print("\nTest 1 - Large Cap Normal Stock:")
    print(f"  ORS: {test1['ors']}, Risk: {test1['risk_flag']}")
    print(f"  Components: {test1['components']}")

    # Test case 2: Risky stock (low price, low liquidity)
    test2 = calc.calculate_ors_single(
        price=800,
        avg_trading_value=50_000_000,  # 5000만원
        volatility=70.0,
        market_cap=30_000_000_000  # 300억원
    )
    print("\nTest 2 - High Risk Small Cap:")
    print(f"  ORS: {test2['ors']}, Risk: {test2['risk_flag']}")
    print(f"  Components: {test2['components']}")

    # Test case 3: Medium risk stock
    test3 = calc.calculate_ors_single(
        price=3000,
        avg_trading_value=1_000_000_000,  # 10억원
        volatility=50.0,
        market_cap=150_000_000_000  # 1500억원
    )
    print("\nTest 3 - Medium Risk Stock:")
    print(f"  ORS: {test3['ors']}, Risk: {test3['risk_flag']}")
    print(f"  Components: {test3['components']}")

    # Test case 4: Extreme risk (penny stock equivalent)
    test4 = calc.calculate_ors_single(
        price=300,
        avg_trading_value=30_000_000,  # 3000만원
        volatility=90.0,
        market_cap=20_000_000_000  # 200억원
    )
    print("\nTest 4 - Extreme Risk Penny Stock:")
    print(f"  ORS: {test4['ors']}, Risk: {test4['risk_flag']}")
    print(f"  Components: {test4['components']}")
    print(f"  Position Multiplier: {test4['position_multiplier']}")
