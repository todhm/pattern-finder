"""
US Growth Factor v2.0 - Future Growth

핵심: "과거 성장"보다 "미래 성장 가능성" 중시

Strategies:
- FG1: Revenue Growth (20%) - 매출 성장률 (YoY)
- FG2: EPS Growth (20%) - EPS 성장률 (YoY)
- FG3: Forward Growth (20%) - 애널리스트 예상 성장률
- FG4: Growth Consistency (15%) - 성장 일관성 (분기별 변동)
- FG5: Revenue Acceleration (15%) - 성장 가속화 여부
- FG6: Profit Growth (10%) - 이익 성장 (수익성 개선)

File: us/us_growth_factor.py
"""

import logging
from typing import Dict, Optional, List
from datetime import date
from decimal import Decimal
import math

from us_volatility_adjustment import USVolatilityAdjustment

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ========================================================================
# Strategy Weights
# ========================================================================

# 일반 거래소 가중치
STRATEGY_WEIGHTS = {
    'FG1': 0.20,  # Revenue Growth
    'FG2': 0.20,  # EPS Growth
    'FG3': 0.20,  # Forward Growth
    'FG4': 0.15,  # Growth Consistency
    'FG5': 0.15,  # Revenue Acceleration
    'FG6': 0.10   # Profit Growth
}

# NASDAQ 전용 가중치 (NQ1~NQ4 NASDAQ 특화 전략 포함)
# 기존 FG1~FG6 가중치 축소 + NQ1~NQ4 35% 추가 = 합계 100%
# NASDAQ은 성장주 중심이므로 Growth Factor에 특화 전략 추가
NASDAQ_STRATEGY_WEIGHTS = {
    'FG1': 0.12,  # Revenue Growth (20% → 12%)
    'FG2': 0.12,  # EPS Growth (20% → 12%)
    'FG3': 0.15,  # Forward Growth (20% → 15%)
    'FG4': 0.08,  # Growth Consistency (15% → 8%)
    'FG5': 0.10,  # Revenue Acceleration (15% → 10%)
    'FG6': 0.08,  # Profit Growth (10% → 8%)
    # NASDAQ 특화 전략 (35%)
    'NQ1': 0.10,  # Institutional Quality Score
    'NQ2': 0.10,  # Analyst Revision Momentum
    'NQ3': 0.08,  # Growth Sustainability Index
    'NQ4': 0.07   # Options Sentiment Score
}
# Total: 12+12+15+8+10+8+10+10+8+7 = 100%

# 섹터별 성장률 기준 (고성장 섹터는 기준이 높음)
SECTOR_GROWTH_THRESHOLDS = {
    'TECHNOLOGY': {'high': 0.30, 'good': 0.15, 'moderate': 0.08},
    'HEALTHCARE': {'high': 0.25, 'good': 0.12, 'moderate': 0.06},
    'COMMUNICATION SERVICES': {'high': 0.25, 'good': 0.12, 'moderate': 0.06},
    'CONSUMER CYCLICAL': {'high': 0.20, 'good': 0.10, 'moderate': 0.05},
    'FINANCIAL SERVICES': {'high': 0.15, 'good': 0.08, 'moderate': 0.04},
    'DEFAULT': {'high': 0.20, 'good': 0.10, 'moderate': 0.05}
}


class USGrowthFactorV2:
    """Growth Factor v2.0 - Future Growth"""

    def __init__(self, symbol: str, db_manager, sector_benchmarks: Dict,
                 analysis_date: Optional[date] = None, exchange: Optional[str] = None,
                 volatility_engine: Optional['USVolatilityAdjustment'] = None,
                 prefetched_data: Optional[Dict] = None):
        """
        Args:
            symbol: 종목 코드
            db_manager: AsyncDatabaseManager
            sector_benchmarks: 섹터 기준값 (from USSectorBenchmarks)
            analysis_date: 분석 날짜
            exchange: 거래소 (NYSE, NASDAQ, AMEX)
            volatility_engine: Phase 3.4.2 - Shared volatility engine (optional)
            prefetched_data: Phase 3.4.3 - Prefetched data from USDataPrefetcher (optional)
        """
        self.symbol = symbol
        self.db = db_manager
        self.benchmarks = sector_benchmarks
        self.analysis_date = analysis_date
        self.exchange = exchange
        self.stock_data = {}
        self.sector = None
        self.quarterly_data = []
        # Phase 3.4.2: Shared Volatility Engine (can be passed from main)
        self.volatility_engine: Optional[USVolatilityAdjustment] = volatility_engine
        self.price_data: List[Dict] = []
        # Phase 3.4.3: Prefetched data for query optimization
        self.prefetched_data: Optional[Dict] = prefetched_data

    async def calculate(self) -> Dict:
        """
        Growth Factor 점수 계산

        Returns:
            {
                'growth_score': float (0-100),
                'strategies': {
                    'FG1': {'score': float, 'raw': float, ...},
                    ...
                }
            }
        """
        try:
            # 1. 데이터 로드 (Phase 3.4.3: prefetched_data 우선 사용)
            if self.prefetched_data:
                self._load_from_prefetched()
            else:
                await self._load_stock_data()
                await self._load_quarterly_data()

            # Phase 3.4.2: Shared Volatility Engine (use passed engine or create new)
            if self.volatility_engine is None:
                await self._load_price_data()
                self.volatility_engine = USVolatilityAdjustment(self.db)
                await self.volatility_engine.load_data(self.symbol, self.analysis_date, self.price_data)

            if not self.stock_data:
                logger.warning(f"{self.symbol}: No data available")
                return {'growth_score': 50.0, 'strategies': {}}

            # 2. 각 전략별 점수 계산
            strategies = {}
            strategies['FG1'] = self._calc_fg1_revenue_growth()
            strategies['FG2'] = self._calc_fg2_eps_growth()
            strategies['FG3'] = self._calc_fg3_forward_growth()
            strategies['FG4'] = self._calc_fg4_growth_consistency()
            strategies['FG5'] = self._calc_fg5_revenue_acceleration()
            strategies['FG6'] = self._calc_fg6_profit_growth()

            # 3. NASDAQ일 때 NQ1~NQ4 추가 계산
            is_nasdaq = self.exchange == 'NASDAQ'
            if is_nasdaq:
                await self._load_nasdaq_data()
                strategies['NQ1'] = self._calc_nq1_institutional_quality()
                strategies['NQ2'] = self._calc_nq2_analyst_revision()
                strategies['NQ3'] = self._calc_nq3_growth_sustainability()
                strategies['NQ4'] = await self._calc_nq4_options_sentiment()

            # 4. 가중 평균 계산 (거래소에 따라 다른 가중치 사용)
            weights = NASDAQ_STRATEGY_WEIGHTS if is_nasdaq else STRATEGY_WEIGHTS
            total_weight = 0
            weighted_sum = 0

            for strategy_id, result in strategies.items():
                if result['score'] is not None and strategy_id in weights:
                    weight = weights[strategy_id]
                    weighted_sum += result['score'] * weight
                    total_weight += weight

            growth_score = weighted_sum / total_weight if total_weight > 0 else 50.0

            # 5. Phase 3.4.2: Volatility Adjustment (sensitivity=0.7 for Growth)
            vol_adjustment = self.volatility_engine.get_adjustment('growth')
            original_score = growth_score
            growth_score = max(0, min(100, growth_score + vol_adjustment['modifier']))

            if vol_adjustment['modifier'] != 0:
                logger.info(f"{self.symbol}: Growth {original_score:.1f} -> {growth_score:.1f} "
                           f"(Vol Adj: {vol_adjustment['modifier']:+.1f}, NASDAQ: {is_nasdaq})")
            else:
                logger.info(f"{self.symbol}: Growth Score = {growth_score:.1f} (NASDAQ: {is_nasdaq})")

            return {
                'growth_score': round(growth_score, 1),
                'growth_original': round(original_score, 1),
                'strategies': strategies,
                # Phase 3.4.2: Volatility data
                'iv_percentile': self.volatility_engine.iv_percentile if self.volatility_engine else None,
                'hv_20d': self.volatility_engine.hv_20d if self.volatility_engine else None,
                'vol_adjustment': vol_adjustment
            }

        except Exception as e:
            logger.error(f"{self.symbol}: Growth calculation failed - {e}")
            return {'growth_score': 50.0, 'strategies': {}}

    async def _load_stock_data(self):
        """종목 기본 데이터 로드"""

        query = """
        SELECT
            symbol,
            sector,
            quarterlyrevenuegrowthyoy as revenue_growth,
            quarterlyearningsgrowthyoy as earnings_growth,
            peg,
            forwardpe,
            per,
            revenuettm as revenue,
            grossprofitttm as gross_profit,
            operatingmarginttm as operating_margin,
            profitmargin as net_margin,
            dilutedepsttm as eps,
            ebitda
        FROM us_stock_basic
        WHERE symbol = $1
        """

        try:
            result = await self.db.execute_query(query, self.symbol)

            if result and result[0]:
                row = result[0]
                self.sector = row['sector']
                self.stock_data = {
                    'revenue_growth': self._to_float(row['revenue_growth']),
                    'earnings_growth': self._to_float(row['earnings_growth']),
                    'peg': self._to_float(row['peg']),
                    'forward_pe': self._to_float(row['forwardpe']),
                    'per': self._to_float(row['per']),
                    'revenue': self._to_float(row['revenue']),
                    'gross_profit': self._to_float(row['gross_profit']),
                    'operating_margin': self._to_float(row['operating_margin']),
                    'net_margin': self._to_float(row['net_margin']),
                    'eps': self._to_float(row['eps']),
                    'ebitda': self._to_float(row['ebitda'])
                }

        except Exception as e:
            logger.error(f"{self.symbol}: Failed to load stock data - {e}")

    async def _load_quarterly_data(self):
        """분기별 재무 데이터 로드"""

        query = """
        SELECT
            fiscal_date_ending,
            total_revenue,
            gross_profit,
            operating_income,
            net_income
        FROM us_income_statement
        WHERE symbol = $1
        ORDER BY fiscal_date_ending DESC
        LIMIT 12
        """

        try:
            result = await self.db.execute_query(query, self.symbol)

            if result:
                # EPS는 us_stock_basic 테이블의 dilutedepsttm 사용
                eps_ttm = self.stock_data.get('dilutedepsttm')
                self.quarterly_data = [
                    {
                        'date': r['fiscal_date_ending'],
                        'revenue': self._to_float(r['total_revenue']),
                        'gross_profit': self._to_float(r['gross_profit']),
                        'operating_income': self._to_float(r['operating_income']),
                        'net_income': self._to_float(r['net_income']),
                        'eps': self._to_float(eps_ttm) if i == 0 else None
                    }
                    for i, r in enumerate(result)
                ]

        except Exception as e:
            logger.warning(f"{self.symbol}: Failed to load quarterly data - {e}")

    async def _load_price_data(self):
        """Phase 3.4.2: Load price data for HV calculation"""
        query = """
        SELECT date, close
        FROM us_daily
        WHERE symbol = $1
          AND ($2::DATE IS NULL OR date <= $2::DATE)
        ORDER BY date DESC
        LIMIT 30
        """

        try:
            result = await self.db.execute_query(query, self.symbol, self.analysis_date)
            if result:
                self.price_data = [{'date': r['date'], 'close': float(r['close']) if r['close'] else None}
                                   for r in result]
        except Exception as e:
            logger.warning(f"{self.symbol}: Price data load failed - {e}")

    def _load_from_prefetched(self):
        """
        Phase 3.4.3: Load data from prefetched data (no DB query).
        Maps prefetched_data to stock_data and quarterly_data format.
        """
        if not self.prefetched_data:
            return

        stock_basic = self.prefetched_data.get('stock_basic', {})
        financials = self.prefetched_data.get('financials', {})

        if not stock_basic:
            return

        # Set sector
        self.sector = stock_basic.get('sector')

        # Map stock_basic to stock_data
        self.stock_data = {
            'revenue_growth': self._to_float(stock_basic.get('quarterlyrevenuegrowthyoy')),
            'earnings_growth': self._to_float(stock_basic.get('quarterlyearningsgrowthyoy')),
            'peg': self._to_float(stock_basic.get('peg')),
            'forward_pe': self._to_float(stock_basic.get('forwardpe')),
            'per': self._to_float(stock_basic.get('per')),
            'revenue': self._to_float(stock_basic.get('revenuettm')),
            'gross_profit': self._to_float(stock_basic.get('grossprofitttm')),
            'operating_margin': self._to_float(stock_basic.get('operatingmarginttm')),
            'net_margin': self._to_float(stock_basic.get('profitmargin')),
            'eps': self._to_float(stock_basic.get('dilutedepsttm')),
            'ebitda': self._to_float(stock_basic.get('ebitda'))
        }

        # Extract quarterly data from income statements
        income = financials.get('income', []) if financials else []
        eps_ttm = self.stock_data.get('eps')
        self.quarterly_data = []
        for i, r in enumerate(income[:12]):
            self.quarterly_data.append({
                'date': r.get('fiscal_date_ending'),
                'revenue': self._to_float(r.get('total_revenue')),
                'gross_profit': self._to_float(r.get('gross_profit')),
                'operating_income': self._to_float(r.get('operating_income')),
                'net_income': self._to_float(r.get('net_income')),
                'eps': eps_ttm if i == 0 else None
            })

    def _calc_fg1_revenue_growth(self) -> Dict:
        """FG1: Revenue Growth (YoY)"""

        revenue_growth = self.stock_data.get('revenue_growth')

        # 분기 데이터에서 직접 계산 시도
        if revenue_growth is None and len(self.quarterly_data) >= 5:
            recent_rev = self.quarterly_data[0]['revenue']
            yoy_rev = self.quarterly_data[4]['revenue'] if len(self.quarterly_data) > 4 else None

            if recent_rev and yoy_rev and yoy_rev > 0:
                revenue_growth = (recent_rev - yoy_rev) / yoy_rev

        if revenue_growth is None:
            return {'score': None, 'raw': None, 'reason': 'No revenue growth data'}

        # 섹터별 기준 적용
        thresholds = SECTOR_GROWTH_THRESHOLDS.get(self.sector, SECTOR_GROWTH_THRESHOLDS['DEFAULT'])

        # 점수 계산
        if revenue_growth >= thresholds['high']:
            score = 85 + min(15, (revenue_growth - thresholds['high']) / 0.20 * 15)
        elif revenue_growth >= thresholds['good']:
            diff = thresholds['high'] - thresholds['good']
            score = 65 + (revenue_growth - thresholds['good']) / diff * 20
        elif revenue_growth >= thresholds['moderate']:
            diff = thresholds['good'] - thresholds['moderate']
            score = 50 + (revenue_growth - thresholds['moderate']) / diff * 15
        elif revenue_growth >= 0:
            score = 35 + revenue_growth / thresholds['moderate'] * 15
        elif revenue_growth >= -0.10:
            score = 25 + (revenue_growth + 0.10) / 0.10 * 10
        else:
            score = max(10, 25 + (revenue_growth + 0.10) * 100)

        return {
            'score': min(100, max(0, score)),
            'raw': revenue_growth,
            'thresholds': thresholds
        }

    def _calc_fg2_eps_growth(self) -> Dict:
        """FG2: EPS Growth (YoY)"""

        earnings_growth = self.stock_data.get('earnings_growth')

        # 분기 데이터에서 직접 계산 시도
        if earnings_growth is None and len(self.quarterly_data) >= 5:
            recent_eps = self.quarterly_data[0]['eps']
            yoy_eps = self.quarterly_data[4]['eps'] if len(self.quarterly_data) > 4 else None

            if recent_eps and yoy_eps and yoy_eps > 0:
                earnings_growth = (recent_eps - yoy_eps) / abs(yoy_eps)

        if earnings_growth is None:
            return {'score': None, 'raw': None, 'reason': 'No earnings growth data'}

        # 점수 계산 (EPS 성장은 매출 성장보다 기준이 유연함)
        if earnings_growth >= 0.50:
            score = 90 + min(10, (earnings_growth - 0.50) / 0.50 * 10)
        elif earnings_growth >= 0.25:
            score = 75 + (earnings_growth - 0.25) / 0.25 * 15
        elif earnings_growth >= 0.10:
            score = 60 + (earnings_growth - 0.10) / 0.15 * 15
        elif earnings_growth >= 0:
            score = 45 + earnings_growth / 0.10 * 15
        elif earnings_growth >= -0.15:
            score = 30 + (earnings_growth + 0.15) / 0.15 * 15
        elif earnings_growth >= -0.30:
            score = 20 + (earnings_growth + 0.30) / 0.15 * 10
        else:
            score = max(10, 20 + (earnings_growth + 0.30) * 30)

        return {
            'score': min(100, max(0, score)),
            'raw': earnings_growth
        }

    def _calc_fg3_forward_growth(self) -> Dict:
        """FG3: Forward Growth (PEG에서 추정)"""

        peg = self.stock_data.get('peg')
        forward_pe = self.stock_data.get('forward_pe')
        trailing_pe = self.stock_data.get('per')

        forward_growth = None

        # PEG에서 역산
        if peg and forward_pe and peg > 0 and forward_pe > 0:
            # PEG = PE / (Growth * 100)
            # Growth = PE / (PEG * 100)
            forward_growth = forward_pe / (peg * 100)

        # Forward PE vs Trailing PE에서 추정
        if forward_growth is None and forward_pe and trailing_pe and forward_pe > 0:
            if trailing_pe > 0:
                # Forward PE가 낮으면 성장 예상
                pe_ratio = trailing_pe / forward_pe
                if pe_ratio > 1:
                    forward_growth = pe_ratio - 1  # 대략적 추정

        if forward_growth is None:
            # 현재 성장률을 미래 성장률로 대체
            revenue_growth = self.stock_data.get('revenue_growth')
            earnings_growth = self.stock_data.get('earnings_growth')

            if revenue_growth is not None and earnings_growth is not None:
                forward_growth = (revenue_growth + earnings_growth) / 2 * 0.8  # 보수적 할인
            elif revenue_growth is not None:
                forward_growth = revenue_growth * 0.8
            elif earnings_growth is not None:
                forward_growth = earnings_growth * 0.8

        if forward_growth is None:
            return {'score': None, 'raw': None, 'reason': 'Cannot estimate forward growth'}

        # 점수 계산
        if forward_growth >= 0.40:
            score = 90 + min(10, (forward_growth - 0.40) / 0.20 * 10)
        elif forward_growth >= 0.25:
            score = 75 + (forward_growth - 0.25) / 0.15 * 15
        elif forward_growth >= 0.15:
            score = 60 + (forward_growth - 0.15) / 0.10 * 15
        elif forward_growth >= 0.08:
            score = 50 + (forward_growth - 0.08) / 0.07 * 10
        elif forward_growth >= 0:
            score = 40 + forward_growth / 0.08 * 10
        elif forward_growth >= -0.10:
            score = 30 + (forward_growth + 0.10) / 0.10 * 10
        else:
            score = max(15, 30 + (forward_growth + 0.10) * 100)

        return {
            'score': min(100, max(0, score)),
            'raw': forward_growth,
            'source': 'peg' if peg else 'pe_ratio' if trailing_pe else 'current_growth'
        }

    def _calc_fg4_growth_consistency(self) -> Dict:
        """FG4: Growth Consistency (분기별 변동성)"""

        if len(self.quarterly_data) < 6:
            return {'score': None, 'raw': None, 'reason': 'Insufficient quarterly data'}

        # 분기별 매출 성장률 계산 (YoY)
        revenue_growths = []

        for i in range(min(4, len(self.quarterly_data) - 4)):
            recent = self.quarterly_data[i]['revenue']
            yoy = self.quarterly_data[i + 4]['revenue']

            if recent and yoy and yoy > 0:
                growth = (recent - yoy) / yoy
                revenue_growths.append(growth)

        if len(revenue_growths) < 3:
            return {'score': None, 'raw': None, 'reason': 'Cannot calculate growth consistency'}

        # 성장률 평균 및 표준편차
        avg_growth = sum(revenue_growths) / len(revenue_growths)
        variance = sum((g - avg_growth) ** 2 for g in revenue_growths) / len(revenue_growths)
        std_growth = math.sqrt(variance)

        # Coefficient of Variation (변동계수)
        if avg_growth != 0:
            cv = abs(std_growth / avg_growth)
        else:
            cv = float('inf') if std_growth > 0 else 0

        # 일관된 양의 성장 확인
        positive_quarters = sum(1 for g in revenue_growths if g > 0)
        positive_ratio = positive_quarters / len(revenue_growths)

        # 점수 계산
        # 낮은 변동성 + 높은 양의 성장 비율 = 높은 점수
        if cv <= 0.3 and positive_ratio >= 0.75:
            base_score = 85 + min(15, (0.3 - cv) / 0.3 * 10 + positive_ratio * 5)
        elif cv <= 0.5 and positive_ratio >= 0.5:
            base_score = 65 + (0.5 - cv) / 0.2 * 10 + positive_ratio * 10
        elif cv <= 0.8 and positive_ratio >= 0.25:
            base_score = 45 + (0.8 - cv) / 0.3 * 10 + positive_ratio * 10
        elif positive_ratio > 0:
            base_score = 30 + positive_ratio * 15
        else:
            base_score = 20

        # 성장 추세 보너스 (최근 성장률 > 이전 성장률)
        if len(revenue_growths) >= 2:
            if revenue_growths[0] > revenue_growths[-1]:
                trend_bonus = 5
            elif revenue_growths[0] < revenue_growths[-1]:
                trend_bonus = -5
            else:
                trend_bonus = 0
        else:
            trend_bonus = 0

        final_score = base_score + trend_bonus

        return {
            'score': min(100, max(0, final_score)),
            'raw': cv,
            'avg_growth': avg_growth,
            'std_growth': std_growth,
            'positive_ratio': positive_ratio,
            'revenue_growths': revenue_growths
        }

    def _calc_fg5_revenue_acceleration(self) -> Dict:
        """FG5: Revenue Acceleration (성장 가속화)"""

        if len(self.quarterly_data) < 8:
            return {'score': None, 'raw': None, 'reason': 'Insufficient data'}

        # 최근 4분기 매출 성장률 (YoY)
        recent_growths = []
        older_growths = []

        for i in range(min(4, len(self.quarterly_data) - 4)):
            recent_rev = self.quarterly_data[i]['revenue']
            yoy_rev = self.quarterly_data[i + 4]['revenue']

            if recent_rev and yoy_rev and yoy_rev > 0:
                recent_growths.append((recent_rev - yoy_rev) / yoy_rev)

        # 4분기 전의 성장률 (비교용)
        if len(self.quarterly_data) >= 8:
            for i in range(4, min(8, len(self.quarterly_data) - 4)):
                recent_rev = self.quarterly_data[i]['revenue']
                yoy_rev = self.quarterly_data[i + 4]['revenue'] if i + 4 < len(self.quarterly_data) else None

                if recent_rev and yoy_rev and yoy_rev > 0:
                    older_growths.append((recent_rev - yoy_rev) / yoy_rev)

        if not recent_growths:
            return {'score': None, 'raw': None, 'reason': 'Cannot calculate acceleration'}

        recent_avg = sum(recent_growths) / len(recent_growths)
        older_avg = sum(older_growths) / len(older_growths) if older_growths else 0

        # 가속도 = 최근 성장률 - 이전 성장률
        acceleration = recent_avg - older_avg

        # 점수 계산
        if acceleration >= 0.10:  # 10%p 이상 가속
            score = 90 + min(10, (acceleration - 0.10) / 0.10 * 10)
        elif acceleration >= 0.05:
            score = 75 + (acceleration - 0.05) / 0.05 * 15
        elif acceleration >= 0:
            score = 55 + acceleration / 0.05 * 20
        elif acceleration >= -0.05:
            score = 40 + (acceleration + 0.05) / 0.05 * 15
        elif acceleration >= -0.10:
            score = 25 + (acceleration + 0.10) / 0.05 * 15
        else:
            score = max(10, 25 + (acceleration + 0.10) * 100)

        # 절대 성장률 보너스
        if recent_avg > 0.20:
            abs_bonus = 5
        elif recent_avg > 0.10:
            abs_bonus = 3
        elif recent_avg < 0:
            abs_bonus = -5
        else:
            abs_bonus = 0

        final_score = score + abs_bonus

        return {
            'score': min(100, max(0, final_score)),
            'raw': acceleration,
            'recent_avg_growth': recent_avg,
            'older_avg_growth': older_avg
        }

    def _calc_fg6_profit_growth(self) -> Dict:
        """FG6: Profit Growth (이익 성장)"""

        if len(self.quarterly_data) < 5:
            return {'score': None, 'raw': None, 'reason': 'Insufficient data'}

        # Operating Income 성장률
        recent_oi = self.quarterly_data[0]['operating_income']
        yoy_oi = self.quarterly_data[4]['operating_income'] if len(self.quarterly_data) > 4 else None

        oi_growth = None
        if recent_oi and yoy_oi and yoy_oi != 0:
            oi_growth = (recent_oi - yoy_oi) / abs(yoy_oi)

        # Net Income 성장률
        recent_ni = self.quarterly_data[0]['net_income']
        yoy_ni = self.quarterly_data[4]['net_income'] if len(self.quarterly_data) > 4 else None

        ni_growth = None
        if recent_ni and yoy_ni and yoy_ni != 0:
            ni_growth = (recent_ni - yoy_ni) / abs(yoy_ni)

        # 두 지표 평균 (가능한 경우)
        growths = [g for g in [oi_growth, ni_growth] if g is not None]

        if not growths:
            # Operating Margin 변화로 대체
            op_margin = self.stock_data.get('operating_margin')
            net_margin = self.stock_data.get('net_margin')

            if op_margin and net_margin:
                # 마진율 기반 점수
                avg_margin = (op_margin + net_margin) / 2
                if avg_margin >= 0.20:
                    score = 70
                elif avg_margin >= 0.10:
                    score = 55
                elif avg_margin >= 0:
                    score = 45
                else:
                    score = 30

                return {
                    'score': score,
                    'raw': avg_margin,
                    'source': 'margin_level'
                }

            return {'score': None, 'raw': None, 'reason': 'No profit growth data'}

        profit_growth = sum(growths) / len(growths)

        # 점수 계산
        if profit_growth >= 0.50:
            score = 90 + min(10, (profit_growth - 0.50) / 0.50 * 10)
        elif profit_growth >= 0.25:
            score = 75 + (profit_growth - 0.25) / 0.25 * 15
        elif profit_growth >= 0.10:
            score = 60 + (profit_growth - 0.10) / 0.15 * 15
        elif profit_growth >= 0:
            score = 45 + profit_growth / 0.10 * 15
        elif profit_growth >= -0.20:
            score = 30 + (profit_growth + 0.20) / 0.20 * 15
        else:
            score = max(10, 30 + (profit_growth + 0.20) * 50)

        # 흑자 전환 보너스
        if recent_oi and yoy_oi:
            if yoy_oi < 0 and recent_oi > 0:  # 흑자 전환
                score = min(100, score + 10)
            elif yoy_oi > 0 and recent_oi < 0:  # 적자 전환
                score = max(0, score - 10)

        return {
            'score': min(100, max(0, score)),
            'raw': profit_growth,
            'oi_growth': oi_growth,
            'ni_growth': ni_growth
        }

    def _to_float(self, value) -> Optional[float]:
        """Decimal/None을 float로 변환"""
        if value is None:
            return None
        if isinstance(value, Decimal):
            return float(value)
        try:
            return float(value)
        except:
            return None

    # ====================================================================
    # NASDAQ 특화 전략 (NQ1~NQ4)
    # ====================================================================

    async def _load_nasdaq_data(self):
        """NASDAQ 특화 데이터 로드 (Institutional, Analyst, Options)"""

        # 1. Institutional & Insider 데이터 (us_stock_basic)
        inst_query = """
        SELECT
            percentinstitutions,
            percentinsiders,
            analysttargetprice,
            analystratingstrongbuy,
            analystratingbuy,
            analystratinghold,
            analystratingsell,
            analystratingstrongsell
        FROM us_stock_basic
        WHERE symbol = $1
        """

        # 2. EPS Estimates 데이터 (us_earnings_estimates)
        eps_query = """
        SELECT
            eps_estimate_average,
            eps_estimate_average_7_days_ago,
            eps_estimate_average_30_days_ago,
            eps_estimate_revision_up_trailing_7_days,
            eps_estimate_revision_down_trailing_7_days,
            revenue_estimate_average
        FROM us_earnings_estimates
        WHERE symbol = $1 AND horizon = 'next fiscal quarter'
        ORDER BY estimate_date DESC
        LIMIT 1
        """

        try:
            # Institutional 데이터
            inst_result = await self.db.execute_query(inst_query, self.symbol)
            if inst_result and inst_result[0]:
                row = inst_result[0]
                # percentinstitutions는 이미 백분율 형태로 저장되어 있음 (0.xx 형태)
                inst_pct = self._to_float(row['percentinstitutions'])
                insider_pct = self._to_float(row['percentinsiders'])

                # 값이 100 이상이면 이미 백분율로 저장된 것이므로 /100
                if inst_pct and inst_pct > 1:
                    inst_pct = inst_pct / 100
                if insider_pct and insider_pct > 1:
                    insider_pct = insider_pct / 100

                self.stock_data['nasdaq'] = {
                    'institutional_pct': inst_pct,
                    'insider_pct': insider_pct,
                    'analyst': {
                        'target_price': self._to_float(row['analysttargetprice']),
                        'strong_buy': row['analystratingstrongbuy'] or 0,
                        'buy': row['analystratingbuy'] or 0,
                        'hold': row['analystratinghold'] or 0,
                        'sell': row['analystratingsell'] or 0,
                        'strong_sell': row['analystratingstrongsell'] or 0
                    }
                }

            # EPS Estimates 데이터
            eps_result = await self.db.execute_query(eps_query, self.symbol)
            if eps_result and eps_result[0]:
                row = eps_result[0]
                if 'nasdaq' not in self.stock_data:
                    self.stock_data['nasdaq'] = {}

                self.stock_data['nasdaq']['eps_estimates'] = {
                    'current': self._to_float(row['eps_estimate_average']),
                    '7d_ago': self._to_float(row['eps_estimate_average_7_days_ago']),
                    '30d_ago': self._to_float(row['eps_estimate_average_30_days_ago']),
                    'revision_up_7d': row['eps_estimate_revision_up_trailing_7_days'] or 0,
                    'revision_down_7d': row['eps_estimate_revision_down_trailing_7_days'] or 0,
                    'revenue_estimate': self._to_float(row['revenue_estimate_average'])
                }

        except Exception as e:
            logger.warning(f"{self.symbol}: NASDAQ data load failed - {e}")

    def _calc_nq1_institutional_quality(self) -> Dict:
        """NQ1: Institutional Quality Score

        NASDAQ 종목의 기관 보유 비율 및 품질 평가
        - 기관 보유율 적정 범위 (30-80%)가 이상적
        - 너무 낮으면 (< 20%) 관심 부족
        - 너무 높으면 (> 90%) 과밀 위험

        점수 범위: 0-100
        """
        nasdaq_data = self.stock_data.get('nasdaq', {})
        inst_pct = nasdaq_data.get('institutional_pct')
        insider_pct = nasdaq_data.get('insider_pct')

        if inst_pct is None:
            return {'score': None, 'raw': None, 'reason': 'No institutional data'}

        # 기관 보유율 점수 (적정 범위 선호)
        if 0.40 <= inst_pct <= 0.75:
            # 황금 범위
            inst_score = 85 + (1 - abs(inst_pct - 0.575) / 0.175) * 15
        elif 0.30 <= inst_pct < 0.40:
            inst_score = 70 + (inst_pct - 0.30) / 0.10 * 15
        elif 0.75 < inst_pct <= 0.85:
            inst_score = 70 + (0.85 - inst_pct) / 0.10 * 15
        elif 0.20 <= inst_pct < 0.30:
            inst_score = 50 + (inst_pct - 0.20) / 0.10 * 20
        elif 0.85 < inst_pct <= 0.95:
            inst_score = 50 + (0.95 - inst_pct) / 0.10 * 20
        elif inst_pct < 0.20:
            inst_score = max(20, inst_pct / 0.20 * 50)
        else:  # > 0.95
            inst_score = max(30, 50 - (inst_pct - 0.95) / 0.05 * 20)

        # 내부자 보유 보너스 (5-15%가 이상적)
        insider_bonus = 0
        if insider_pct:
            if 0.05 <= insider_pct <= 0.15:
                insider_bonus = 10
            elif 0.02 <= insider_pct < 0.05:
                insider_bonus = 5
            elif insider_pct > 0.25:  # 너무 높으면 유동성 문제
                insider_bonus = -5

        final_score = inst_score + insider_bonus

        return {
            'score': min(100, max(0, final_score)),
            'raw': inst_pct,
            'institutional_pct': inst_pct,
            'insider_pct': insider_pct,
            'insider_bonus': insider_bonus
        }

    def _calc_nq2_analyst_revision(self) -> Dict:
        """NQ2: Analyst Revision Momentum

        애널리스트 추정치 변화 및 컨센서스 강도
        - EPS 추정치 상향/하향
        - 애널리스트 컨센서스 (Buy/Hold/Sell)

        점수 범위: 0-100
        """
        nasdaq_data = self.stock_data.get('nasdaq', {})
        eps_data = nasdaq_data.get('eps_estimates')
        analyst_data = nasdaq_data.get('analyst')

        scores = []

        # 1. EPS Revision (50% 비중)
        if eps_data:
            eps_current = eps_data.get('current')
            eps_30d_ago = eps_data.get('30d_ago')
            revision_up = eps_data.get('revision_up_7d', 0)
            revision_down = eps_data.get('revision_down_7d', 0)

            # revision 값 정규화
            if revision_up > 1000:
                revision_up = revision_up / 1000
            if revision_down > 1000:
                revision_down = revision_down / 1000

            if eps_current and eps_30d_ago and eps_30d_ago != 0:
                eps_change = (eps_current - eps_30d_ago) / abs(eps_30d_ago)

                if eps_change >= 0.10:
                    eps_score = 90
                elif eps_change >= 0.05:
                    eps_score = 75 + (eps_change - 0.05) / 0.05 * 15
                elif eps_change >= 0:
                    eps_score = 55 + eps_change / 0.05 * 20
                elif eps_change >= -0.05:
                    eps_score = 35 + (eps_change + 0.05) / 0.05 * 20
                else:
                    eps_score = max(15, 35 + eps_change * 200)

                scores.append(('eps_revision', eps_score, 0.5))
            elif revision_up > 0 or revision_down > 0:
                net_revision = revision_up - revision_down
                if net_revision >= 2:
                    eps_score = 80
                elif net_revision >= 0:
                    eps_score = 50 + net_revision * 15
                else:
                    eps_score = max(25, 50 + net_revision * 12)
                scores.append(('revision_count', eps_score, 0.5))

        # 2. Analyst Consensus (50% 비중)
        if analyst_data:
            strong_buy = analyst_data.get('strong_buy', 0)
            buy = analyst_data.get('buy', 0)
            hold = analyst_data.get('hold', 0)
            sell = analyst_data.get('sell', 0)
            strong_sell = analyst_data.get('strong_sell', 0)

            total = strong_buy + buy + hold + sell + strong_sell
            if total > 0:
                # Buy ratio
                buy_ratio = (strong_buy + buy) / total
                sell_ratio = (sell + strong_sell) / total

                if buy_ratio >= 0.8:
                    consensus_score = 90
                elif buy_ratio >= 0.6:
                    consensus_score = 70 + (buy_ratio - 0.6) / 0.2 * 20
                elif buy_ratio >= 0.4:
                    consensus_score = 50 + (buy_ratio - 0.4) / 0.2 * 20
                elif sell_ratio >= 0.4:
                    consensus_score = max(20, 50 - sell_ratio * 50)
                else:
                    consensus_score = 45

                scores.append(('consensus', consensus_score, 0.5))

        if not scores:
            return {'score': None, 'raw': None, 'reason': 'No analyst revision data'}

        # 가중 평균
        total_weight = sum(w for _, _, w in scores)
        final_score = sum(s * w for _, s, w in scores) / total_weight

        return {
            'score': min(100, max(0, final_score)),
            'raw': {name: score for name, score, _ in scores},
            'components': scores
        }

    def _calc_nq3_growth_sustainability(self) -> Dict:
        """NQ3: Growth Sustainability Index

        성장의 지속 가능성 평가
        - FG4 (Growth Consistency)와 FG5 (Acceleration) 조합
        - 매출 성장과 이익 성장의 동조화

        점수 범위: 0-100
        """
        # 기존 FG4, FG5 결과 활용
        revenue_growth = self.stock_data.get('revenue_growth')
        earnings_growth = self.stock_data.get('earnings_growth')

        if len(self.quarterly_data) < 5:
            return {'score': None, 'raw': None, 'reason': 'Insufficient data'}

        scores = []

        # 1. 매출-이익 동조화 (둘 다 성장하면 높은 점수)
        if revenue_growth is not None and earnings_growth is not None:
            if revenue_growth > 0 and earnings_growth > 0:
                # 둘 다 양수
                sync_score = 70 + min(30, (revenue_growth + earnings_growth) * 50)
            elif revenue_growth > 0 or earnings_growth > 0:
                # 하나만 양수
                sync_score = 45 + max(revenue_growth, earnings_growth) * 50
            else:
                # 둘 다 음수
                sync_score = max(20, 45 + (revenue_growth + earnings_growth) * 30)

            scores.append(('revenue_earnings_sync', sync_score, 0.4))

        # 2. 분기별 성장 추세 (최근 > 과거)
        if len(self.quarterly_data) >= 8:
            recent_revs = [self.quarterly_data[i]['revenue'] for i in range(4) if self.quarterly_data[i]['revenue']]
            older_revs = [self.quarterly_data[i]['revenue'] for i in range(4, 8) if i < len(self.quarterly_data) and self.quarterly_data[i]['revenue']]

            if recent_revs and older_revs:
                recent_avg = sum(recent_revs) / len(recent_revs)
                older_avg = sum(older_revs) / len(older_revs)

                if older_avg > 0:
                    trend = (recent_avg - older_avg) / older_avg

                    if trend >= 0.15:
                        trend_score = 85
                    elif trend >= 0.05:
                        trend_score = 65 + (trend - 0.05) / 0.10 * 20
                    elif trend >= 0:
                        trend_score = 50 + trend / 0.05 * 15
                    elif trend >= -0.10:
                        trend_score = 35 + (trend + 0.10) / 0.10 * 15
                    else:
                        trend_score = max(15, 35 + trend * 100)

                    scores.append(('growth_trend', trend_score, 0.35))

        # 3. Forward/Trailing Growth 비율
        forward_pe = self.stock_data.get('forward_pe')
        trailing_pe = self.stock_data.get('per')

        if forward_pe and trailing_pe and forward_pe > 0 and trailing_pe > 0:
            pe_ratio = trailing_pe / forward_pe
            # Forward PE < Trailing PE = 성장 기대
            if pe_ratio >= 1.3:
                fwd_score = 85
            elif pe_ratio >= 1.1:
                fwd_score = 65 + (pe_ratio - 1.1) / 0.2 * 20
            elif pe_ratio >= 0.95:
                fwd_score = 50 + (pe_ratio - 0.95) / 0.15 * 15
            else:
                fwd_score = max(25, pe_ratio / 0.95 * 50)

            scores.append(('forward_growth_expectation', fwd_score, 0.25))

        if not scores:
            return {'score': None, 'raw': None, 'reason': 'Cannot calculate sustainability'}

        total_weight = sum(w for _, _, w in scores)
        final_score = sum(s * w for _, s, w in scores) / total_weight

        return {
            'score': min(100, max(0, final_score)),
            'raw': {name: score for name, score, _ in scores},
            'revenue_growth': revenue_growth,
            'earnings_growth': earnings_growth
        }

    async def _calc_nq4_options_sentiment(self) -> Dict:
        """NQ4: Options Sentiment Score

        옵션 시장 심리 지표
        - Put/Call Ratio (낮을수록 강세)
        - Implied Volatility

        점수 범위: 0-100
        """
        # Options 데이터 조회
        options_query = """
        SELECT
            total_call_volume,
            total_put_volume,
            avg_implied_volatility,
            avg_call_iv,
            avg_put_iv
        FROM us_option_daily_summary
        WHERE symbol = $1
        ORDER BY date DESC
        LIMIT 5
        """

        try:
            result = await self.db.execute_query(options_query, self.symbol)

            if not result:
                return {'score': None, 'raw': None, 'reason': 'No options data'}

            # 최근 5일 평균
            total_calls = sum(r['total_call_volume'] or 0 for r in result)
            total_puts = sum(r['total_put_volume'] or 0 for r in result)

            if total_calls == 0:
                return {'score': 50.0, 'raw': None, 'reason': 'No call volume'}

            # Put/Call Ratio
            pcr = total_puts / total_calls

            # PCR 기반 점수
            # 낮은 PCR = 강세 심리 = 높은 점수
            # 높은 PCR = 약세 심리 / 헤지 = 낮은 점수
            if pcr <= 0.5:
                pcr_score = 85 + (0.5 - pcr) / 0.5 * 15  # Very bullish
            elif pcr <= 0.7:
                pcr_score = 70 + (0.7 - pcr) / 0.2 * 15  # Bullish
            elif pcr <= 1.0:
                pcr_score = 50 + (1.0 - pcr) / 0.3 * 20  # Neutral
            elif pcr <= 1.3:
                pcr_score = 35 + (1.3 - pcr) / 0.3 * 15  # Bearish
            else:
                pcr_score = max(15, 35 - (pcr - 1.3) * 20)  # Very bearish

            # IV Skew (Put IV vs Call IV)
            iv_bonus = 0
            if result[0].get('avg_put_iv') and result[0].get('avg_call_iv'):
                put_iv = self._to_float(result[0]['avg_put_iv'])
                call_iv = self._to_float(result[0]['avg_call_iv'])

                if put_iv and call_iv and call_iv > 0:
                    iv_skew = put_iv / call_iv
                    # Put IV > Call IV = 하락 헤지 수요 = 약간 부정적
                    if iv_skew <= 0.9:
                        iv_bonus = 5
                    elif iv_skew >= 1.2:
                        iv_bonus = -5

            final_score = pcr_score + iv_bonus

            return {
                'score': min(100, max(0, final_score)),
                'raw': pcr,
                'put_call_ratio': pcr,
                'total_calls': total_calls,
                'total_puts': total_puts,
                'iv_bonus': iv_bonus
            }

        except Exception as e:
            logger.warning(f"{self.symbol}: Options data query failed - {e}")
            return {'score': None, 'raw': None, 'reason': f'Query failed: {e}'}


# ========================================================================
# Convenience Function
# ========================================================================

async def calculate_growth_score(symbol: str, db_manager, sector_benchmarks: Dict,
                                  analysis_date: Optional[date] = None,
                                  exchange: Optional[str] = None,
                                  volatility_engine: Optional['USVolatilityAdjustment'] = None,
                                  prefetched_data: Optional[Dict] = None) -> Dict:
    """Growth Factor 점수 계산 (편의 함수)

    Args:
        symbol: 종목 코드
        db_manager: AsyncDatabaseManager
        sector_benchmarks: 섹터 기준값
        analysis_date: 분석 날짜
        exchange: 거래소 (NYSE, NASDAQ, AMEX) - NASDAQ일 때 NQ1~NQ4 추가 계산
        volatility_engine: Phase 3.4.2 - Shared volatility engine (optional)
        prefetched_data: Phase 3.4.3 - Prefetched data from USDataPrefetcher (optional)
    """
    calculator = USGrowthFactorV2(symbol, db_manager, sector_benchmarks, analysis_date, exchange,
                                  volatility_engine, prefetched_data)
    return await calculator.calculate()


# ========================================================================
# Test
# ========================================================================

if __name__ == '__main__':
    import asyncio
    from us_db_async import AsyncDatabaseManager
    from us_sector_benchmarks import USSectorBenchmarks

    async def test():
        db = AsyncDatabaseManager()
        await db.initialize()

        # 섹터 기준값 로드
        sector_calc = USSectorBenchmarks(db)
        benchmarks = await sector_calc.get_sector_benchmarks('TECHNOLOGY')

        # 테스트 종목
        test_symbols = ['AAPL', 'NVDA', 'MSFT']

        print("\n" + "="*60)
        print("Growth Factor v2.0 Test")
        print("="*60)

        for symbol in test_symbols:
            result = await calculate_growth_score(symbol, db, benchmarks)
            print(f"\n{symbol}: Growth Score = {result['growth_score']}")

            for strat_id, strat_data in result['strategies'].items():
                if strat_data['score'] is not None:
                    raw = strat_data.get('raw', 'N/A')
                    if isinstance(raw, dict):
                        raw = 'complex'
                    print(f"  {strat_id}: {strat_data['score']:.1f} (raw: {raw})")

        await db.close()

    asyncio.run(test())
