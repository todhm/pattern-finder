"""
US Momentum Factor v3.0 - Long-term Enhanced Momentum (Phase 3)

핵심: 장기(3개월~1년) 예측 정확도 향상을 위한 모멘텀 팩터 재설계

Phase 3 변경사항:
- EM1: Risk-Adjusted Momentum (15%) - 126일 기준으로 장기화
- EM2: Sector Relative Strength (15%) - 126일 중심으로 장기화
- EM3: EPS Estimate Revision (20%) - 비중 증가
- EM4: Revenue Estimate Revision (20%) - 비중 증가
- EM5: 제외 (단기 20일 Volume Confirmation - 장기 예측에 부적합)
- EM6: Earnings Momentum (15%) - 비중 증가
- EM7: 제외 (단기 target price 기반 - 장기 예측에 부적합)
- EM8: Long-term Price Momentum (15%) - 신규 (IBD RS Style, 252일)

Phase 2 분석 결과:
- Momentum 3일 IC: -0.047 (단기 역작동)
- Momentum 252일 IC: +0.064 (장기 유효)
→ 단기 지표 제외, 장기 지표 강화

File: us/us_momentum_factor.py
"""

import logging
from typing import Dict, Optional, List, Tuple
from datetime import date, timedelta
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

# Phase 3.1.5 가중치 (IC 분석 결과 반영)
# ========================================================================
# IC Analysis Results (30d return):
#   EM1: -0.030 (reversed) -> Energy/Basic Materials 섹터에만 적용
#   EM2: -0.078 (reversed) -> 주석 처리 (전체 계산에서 제외)
#   EM3: +0.036 (normal)   -> increase weight
#   EM4: +0.041 (normal)   -> increase weight
#   EM5: Volume Confirmation -> Basic Materials 섹터에만 적용
#   EM6: +0.062 (normal)   -> increase weight
#   EM7: +0.178 (normal)   -> highest IC, increase weight
#
# Sector-specific strategies:
# - EM1: Energy/Basic Materials 섹터에만 적용
# - EM5: Basic Materials 섹터에만 적용
# - EM2: 주석 처리 (전체 계산에서 제외)
# ========================================================================
STRATEGY_WEIGHTS = {
    'EM1': 0.05,  # Risk-Adjusted Momentum - Energy/Basic Materials only (IC -0.030, reversed)
    # 'EM2': 0.07,  # Sector Relative Strength - disabled
    'EM3': 0.10,  # EPS Estimate Revision (IC +0.036) - reduced from 22%
    'EM4': 0.10,  # Revenue Estimate Revision (IC +0.041) - reduced from 22%
    'EM5': 0.05,  # Volume Confirmation - Basic Materials only
    'EM6': 0.20,  # Earnings Momentum (IC +0.062) - reduced from 22%
    'EM8': 0.50   # Long-term Price Momentum (IC +0.178) - increased from 24%
}
# Total: 5 + 10 + 10 + 5 + 20 + 50 = 100%
# (EM1, EM5는 섹터 제한이 있어 해당 섹터가 아니면 자동으로 제외되고 나머지로 재분배)

# ========================================================================
# NASDAQ-Specific Strategy Weights (Phase 3.2)
# ========================================================================
# IC Analysis Results (20d forward return, NASDAQ only):
#   EM8: +0.0565*** (best)     -> 40% weight
#   EM6: +0.0411*** (2nd)      -> 35% weight
#   EM3: -0.0104   (weak)      -> 10% weight (reduced from 22%)
#   EM4: -0.0562*** (reversed) -> 5% weight  (reduced from 22%)
#   EM1/EM5: sector-specific   -> 5% each
#
# Key insight: EM4 (Revenue Growth) has NEGATIVE IC on NASDAQ
# because high growth is already priced in for NASDAQ stocks
# ========================================================================
NASDAQ_STRATEGY_WEIGHTS = {
    'EM1': 0.05,  # Risk-Adjusted Momentum - sector-specific
    'EM3': 0.05,  # EPS Estimate Revision (IC -0.0104) - reduced from 10%
    'EM4': 0.05,  # Revenue Estimate Revision (IC -0.0562) - kept low
    'EM5': 0.05,  # Volume Confirmation - sector-specific
    'EM6': 0.25,  # Earnings Momentum (IC +0.0411) - reduced from 35%
    'EM8': 0.55   # Long-term Price Momentum (IC +0.0565) - increased from 40%
}
# Total: 5 + 5 + 5 + 5 + 25 + 55 = 100%

# Sector-specific strategy constraints
# EM1: Energy/Basic Materials 섹터에만 적용
EM1_APPLICABLE_SECTORS = ['ENERGY', 'BASIC MATERIALS']
# EM5: Basic Materials (원자재) 섹터에만 적용
EM5_APPLICABLE_SECTORS = ['BASIC MATERIALS']

# Phase 3.1.4 역방향 전략 (음의 IC → 점수 역전으로 양의 IC 기대)
# IC 분석 결과:
#   - EM1: IC -0.030 → 역방향 적용
REVERSED_STRATEGIES = ['EM1']


class USMomentumFactorV2:
    """Momentum Factor v3.0 - Long-term Enhanced Momentum (Phase 3)

    Phase 3 변경사항:
    - EM1: Risk-Adjusted Momentum (15%, 126일)
    - EM2: Sector Relative Strength (15%, 126일 중심)
    - EM3: EPS Estimate Revision (20%)
    - EM4: Revenue Estimate Revision (20%)
    - EM6: Earnings Momentum (15%)
    - EM8: Long-term Price Momentum (15%, 252일 IBD RS Style) - 신규

    제외된 전략:
    - EM5: Volume Confirmation (단기 20일, 장기 예측에 부적합)
    - EM7: Analyst Revisions (target price 기반, 단기 지표)
    """

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
            exchange: 거래소 ('NASDAQ', 'NYSE' 등) - NASDAQ이면 전용 가중치 사용
            volatility_engine: Phase 3.4.2 - Shared volatility engine (optional)
            prefetched_data: Phase 3.4.3 - Prefetched data from USDataPrefetcher (optional)
        """
        self.symbol = symbol
        self.db = db_manager
        self.benchmarks = sector_benchmarks
        self.analysis_date = analysis_date or date.today()
        self.exchange = exchange
        self.sector = None
        self.price_data = []
        self.stock_data = {}
        self.sector_returns = {}
        self.earnings_estimates = {}
        # Phase 3.4.2: Shared Volatility Engine (can be passed from main)
        self.volatility_engine: Optional[USVolatilityAdjustment] = volatility_engine
        # Phase 3.4.3: Prefetched data for query optimization
        self.prefetched_data: Optional[Dict] = prefetched_data

    async def calculate(self) -> Dict:
        """
        Momentum Factor 점수 계산

        Returns:
            {
                'momentum_score': float (0-100),
                'strategies': {
                    'EM1': {'score': float, 'raw': float, ...},
                    ...
                }
            }
        """
        try:
            # 1. 데이터 로드 (Phase 3.4.3: prefetched_data 우선 사용)
            if self.prefetched_data:
                self._load_from_prefetched()
                await self._load_sector_returns()  # Sector returns still needs DB query
            else:
                await self._load_price_data()
                await self._load_stock_data()
                await self._load_sector_returns()
                await self._load_earnings_estimates()

            # Phase 3.4.2: Shared Volatility Engine (use passed engine or create new)
            if self.volatility_engine is None:
                self.volatility_engine = USVolatilityAdjustment(self.db)
                await self.volatility_engine.load_data(self.symbol, self.analysis_date, self.price_data)

            if not self.price_data:
                logger.warning(f"{self.symbol}: No price data available")
                return {'momentum_score': 50.0, 'strategies': {}}

            # 2. 각 전략별 점수 계산
            # Phase 3.1.6: 섹터별 전략 적용
            # - EM1: Energy/Basic Materials 섹터에만 적용
            # - EM2: 주석 처리 (전체 계산에서 제외)
            # - EM5: Basic Materials 섹터에만 적용
            strategies = {}

            # EM1: Energy/Basic Materials 섹터에만 적용
            if self.sector and self.sector.upper() in EM1_APPLICABLE_SECTORS:
                strategies['EM1'] = self._calc_em1_risk_adjusted_momentum()
            else:
                strategies['EM1'] = {'score': None, 'raw': None, 'reason': f'Not applicable sector (only {EM1_APPLICABLE_SECTORS})'}

            # EM2: 주석 처리 - 전체 계산에서 제외
            # strategies['EM2'] = self._calc_em2_sector_relative_strength()

            strategies['EM3'] = self._calc_em3_eps_revision()
            strategies['EM4'] = self._calc_em4_revenue_revision()

            # EM5: Basic Materials (원자재) 섹터에만 적용
            if self.sector and self.sector.upper() in EM5_APPLICABLE_SECTORS:
                strategies['EM5'] = self._calc_em5_volume_confirmation()
            else:
                strategies['EM5'] = {'score': None, 'raw': None, 'reason': f'Not applicable sector (only {EM5_APPLICABLE_SECTORS})'}

            strategies['EM6'] = self._calc_em6_earnings_momentum()
            # EM7 제외: Analyst Revisions (target price) - 단기 지표
            strategies['EM8'] = self._calc_em8_long_term_momentum()  # 252일 IBD RS Style

            # 3. 역방향 전략 적용 (Phase 3.1.4)
            for strategy_id, result in strategies.items():
                if result['score'] is not None and strategy_id in REVERSED_STRATEGIES:
                    original_score = result['score']
                    reversed_score = 100 - original_score
                    result['original_score'] = original_score
                    result['score'] = reversed_score
                    result['reversed'] = True

            # 4. 가중 평균 계산
            # Phase 3.2: NASDAQ이면 전용 가중치 사용
            if self.exchange and self.exchange.upper() == 'NASDAQ':
                weights = NASDAQ_STRATEGY_WEIGHTS
                logger.debug(f"{self.symbol}: Using NASDAQ-specific strategy weights")
            else:
                weights = STRATEGY_WEIGHTS

            total_weight = 0
            weighted_sum = 0

            for strategy_id, result in strategies.items():
                if result['score'] is not None:
                    weight = weights[strategy_id]
                    weighted_sum += result['score'] * weight
                    total_weight += weight

            momentum_score = weighted_sum / total_weight if total_weight > 0 else 50.0

            # 5. Phase 3.4.2: Volatility Adjustment (Shared Module)
            vol_adjustment = self.volatility_engine.get_adjustment('momentum')
            original_score = momentum_score
            momentum_score = max(0, min(100, momentum_score + vol_adjustment['modifier']))

            if vol_adjustment['modifier'] != 0:
                logger.info(f"{self.symbol}: Momentum {original_score:.1f} -> {momentum_score:.1f} "
                           f"(Vol Adj: {vol_adjustment['modifier']:+.1f}, IV%: {self.volatility_engine.iv_percentile})")
            else:
                logger.info(f"{self.symbol}: Momentum Score = {momentum_score:.1f}")

            return {
                'momentum_score': round(momentum_score, 1),
                'momentum_original': round(original_score, 1),
                'strategies': strategies,
                # Phase 3.4.2: Volatility data (from shared module)
                'iv_percentile': self.volatility_engine.iv_percentile if self.volatility_engine else None,
                'hv_20d': self.volatility_engine.hv_20d if self.volatility_engine else None,
                'vol_adjustment': vol_adjustment
            }

        except Exception as e:
            logger.error(f"{self.symbol}: Momentum calculation failed - {e}")
            return {'momentum_score': 50.0, 'strategies': {}}

    async def _load_price_data(self):
        """가격 데이터 로드 (최근 260 거래일)"""

        query = """
        SELECT
            date,
            open,
            high,
            low,
            close,
            volume
        FROM us_daily
        WHERE symbol = $1
          AND date <= $2
        ORDER BY date DESC
        LIMIT 260
        """

        try:
            result = await self.db.execute_query(query, self.symbol, self.analysis_date)

            if result:
                self.price_data = [
                    {
                        'date': r['date'],
                        'open': self._to_float(r['open']),
                        'high': self._to_float(r['high']),
                        'low': self._to_float(r['low']),
                        'close': self._to_float(r['close']),
                        'volume': self._to_float(r['volume'])
                    }
                    for r in result
                ]

        except Exception as e:
            logger.error(f"{self.symbol}: Failed to load price data - {e}")

    async def _load_stock_data(self):
        """종목 기본 데이터 로드"""

        query = """
        SELECT
            symbol,
            sector,
            week52high,
            week52low,
            day50movingaverage as ma50,
            day200movingaverage as ma200,
            analysttargetprice,
            quarterlyearningsgrowthyoy as earnings_growth,
            quarterlyrevenuegrowthyoy as revenue_growth
        FROM us_stock_basic
        WHERE symbol = $1
        """

        try:
            result = await self.db.execute_query(query, self.symbol)

            if result and result[0]:
                row = result[0]
                self.sector = row['sector']
                self.stock_data = {
                    'high_52w': self._to_float(row['week52high']),
                    'low_52w': self._to_float(row['week52low']),
                    'ma50': self._to_float(row['ma50']),
                    'ma200': self._to_float(row['ma200']),
                    'target_price': self._to_float(row['analysttargetprice']),
                    'earnings_growth': self._to_float(row['earnings_growth']),
                    'revenue_growth': self._to_float(row['revenue_growth'])
                }

        except Exception as e:
            logger.error(f"{self.symbol}: Failed to load stock data - {e}")

    async def _load_sector_returns(self):
        """섹터 평균 수익률 조회 (벤치마크에서 가져오거나 계산)"""

        # 벤치마크에서 섹터 수익률 가져오기
        self.sector_returns = {
            'return_1m': self.benchmarks.get('return_1m_median'),
            'return_3m': self.benchmarks.get('return_3m_median'),
            'return_6m': self.benchmarks.get('return_6m_median')
        }

    async def _load_earnings_estimates(self):
        """EPS/Revenue 추정치 데이터 로드 (us_earnings_estimates)"""

        query = """
        SELECT
            estimate_date,
            horizon,
            eps_estimate_average,
            eps_estimate_average_7_days_ago,
            eps_estimate_average_30_days_ago,
            eps_estimate_revision_up_trailing_7_days,
            eps_estimate_revision_down_trailing_7_days,
            revenue_estimate_average
        FROM us_earnings_estimates
        WHERE symbol = $1
            AND horizon IN ('next fiscal quarter', 'next fiscal year')
            AND eps_estimate_average IS NOT NULL
        ORDER BY estimate_date DESC
        LIMIT 2
        """

        try:
            result = await self.db.execute_query(query, self.symbol)

            if result and result[0]:
                row = result[0]
                self.earnings_estimates = {
                    'eps_current': self._to_float(row['eps_estimate_average']),
                    'eps_7d_ago': self._to_float(row['eps_estimate_average_7_days_ago']),
                    'eps_30d_ago': self._to_float(row['eps_estimate_average_30_days_ago']),
                    'revision_up_7d': row['eps_estimate_revision_up_trailing_7_days'],
                    'revision_down_7d': row['eps_estimate_revision_down_trailing_7_days'],
                    'revenue_current': self._to_float(row['revenue_estimate_average']),
                    'horizon': row['horizon']
                }

        except Exception as e:
            logger.warning(f"{self.symbol}: Failed to load earnings estimates - {e}")

    def _load_from_prefetched(self):
        """
        Phase 3.4.3: Load data from prefetched data (no DB query).
        Maps prefetched_data to price_data, stock_data, and earnings_estimates.
        """
        if not self.prefetched_data:
            return

        stock_basic = self.prefetched_data.get('stock_basic', {})
        price_data = self.prefetched_data.get('price_data', {})
        estimates = self.prefetched_data.get('estimates', {})

        # Set sector
        self.sector = stock_basic.get('sector') if stock_basic else None

        # Map price_data (needed for momentum calculations)
        if price_data and 'raw' in price_data:
            self.price_data = [
                {'date': r['date'], 'close': float(r['close']) if r.get('close') else None,
                 'volume': int(r['volume']) if r.get('volume') else None}
                for r in price_data['raw']
            ]

        # Map stock_data
        if stock_basic:
            self.stock_data = {
                'market_cap': self._to_float(stock_basic.get('market_cap')),
                'beta': self._to_float(stock_basic.get('beta'))
            }

        # Map earnings_estimates
        if estimates:
            est = estimates.get('next_quarter') or estimates.get('next_year')
            if est:
                self.earnings_estimates = {
                    'eps_current': self._to_float(est.get('eps_estimate_average')),
                    'eps_7d_ago': self._to_float(est.get('eps_estimate_average_7_days_ago')),
                    'eps_30d_ago': self._to_float(est.get('eps_estimate_average_30_days_ago')),
                    'revision_up_7d': est.get('eps_estimate_revision_up_trailing_7_days'),
                    'revision_down_7d': est.get('eps_estimate_revision_down_trailing_7_days'),
                    'revenue_current': self._to_float(est.get('revenue_estimate_average')),
                    'horizon': est.get('horizon')
                }

    def _calc_em1_risk_adjusted_momentum(self) -> Dict:
        """EM1: Risk-Adjusted Momentum (변동성 조정 수익률)"""

        if len(self.price_data) < 60:
            return {'score': None, 'raw': None, 'reason': 'Insufficient price data'}

        # 6개월 수익률
        prices = [p['close'] for p in self.price_data[:126] if p['close']]
        if len(prices) < 60:
            return {'score': None, 'raw': None, 'reason': 'Insufficient close prices'}

        current = prices[0]
        price_6m = prices[-1] if len(prices) >= 126 else prices[min(len(prices)-1, 125)]
        return_6m = (current - price_6m) / price_6m

        # 일별 수익률 계산
        daily_returns = []
        for i in range(len(prices) - 1):
            if prices[i+1] > 0:
                ret = (prices[i] - prices[i+1]) / prices[i+1]
                daily_returns.append(ret)

        if len(daily_returns) < 20:
            return {'score': None, 'raw': None, 'reason': 'Cannot calculate volatility'}

        # 변동성 (표준편차)
        mean_ret = sum(daily_returns) / len(daily_returns)
        variance = sum((r - mean_ret) ** 2 for r in daily_returns) / len(daily_returns)
        volatility = math.sqrt(variance) * math.sqrt(252)  # 연율화

        if volatility <= 0:
            return {'score': None, 'raw': None, 'reason': 'Zero volatility'}

        # Risk-Adjusted Return (Sharpe-like, 무위험 이자율 0 가정)
        risk_adj_return = return_6m / volatility

        # 점수 계산
        if risk_adj_return >= 2.0:
            score = 95
        elif risk_adj_return >= 1.5:
            score = 80 + (risk_adj_return - 1.5) / 0.5 * 15
        elif risk_adj_return >= 1.0:
            score = 65 + (risk_adj_return - 1.0) / 0.5 * 15
        elif risk_adj_return >= 0.5:
            score = 50 + (risk_adj_return - 0.5) / 0.5 * 15
        elif risk_adj_return >= 0:
            score = 35 + risk_adj_return / 0.5 * 15
        elif risk_adj_return >= -0.5:
            score = 25 + (risk_adj_return + 0.5) / 0.5 * 10
        else:
            score = max(10, 25 + (risk_adj_return + 0.5) * 15)

        return {
            'score': min(100, max(0, score)),
            'raw': risk_adj_return,
            'return_6m': return_6m,
            'volatility': volatility
        }

    def _calc_em2_sector_relative_strength(self) -> Dict:
        """EM2: Sector Relative Strength (섹터 대비 상대 강도)

        Phase 3 변경: 126일(6개월) 중심으로 장기화
        - 기존: 1m(40%), 3m(35%), 6m(25%) 가중평균
        - 변경: 3m(30%), 6m(50%), 9m(20%) 가중평균 → 장기 추세 강조
        """

        if len(self.price_data) < 126:
            return {'score': None, 'raw': None, 'reason': 'Insufficient price data (need 126+ days)'}

        prices = [p['close'] for p in self.price_data if p['close']]

        # Phase 3: 장기 기간 수익률 계산 (126일 중심)
        periods = {
            '3m': 63,
            '6m': 126,
            '9m': 189
        }

        relative_strengths = []

        for period_name, days in periods.items():
            if len(prices) > days:
                stock_return = (prices[0] - prices[days]) / prices[days]

                # 섹터 수익률: 6m만 벤치마크에서 가져옴, 나머지는 비율로 추정
                if period_name == '6m':
                    sector_return = self.sector_returns.get('return_6m_median')
                elif period_name == '3m':
                    sector_return = self.sector_returns.get('return_3m_median')
                else:  # 9m
                    # 9m 섹터 수익률은 6m의 1.5배로 추정
                    sector_6m = self.sector_returns.get('return_6m_median')
                    sector_return = sector_6m * 1.5 if sector_6m else None

                if sector_return is not None:
                    relative = stock_return - sector_return
                    relative_strengths.append((period_name, relative, stock_return, sector_return))

        if not relative_strengths:
            return {'score': None, 'raw': None, 'reason': 'No sector comparison data'}

        # Phase 3: 장기 가중치 (6개월 중심)
        weights = {'3m': 0.30, '6m': 0.50, '9m': 0.20}
        weighted_rs = sum(
            rs[1] * weights.get(rs[0], 0.33)
            for rs in relative_strengths
        )

        # 점수 계산 (장기 기준으로 범위 조정)
        if weighted_rs >= 0.30:  # 섹터 대비 30%+ 초과 (장기 기준 상향)
            score = 90 + min(10, (weighted_rs - 0.30) / 0.15 * 10)
        elif weighted_rs >= 0.15:
            score = 75 + (weighted_rs - 0.15) / 0.15 * 15
        elif weighted_rs >= 0.05:
            score = 60 + (weighted_rs - 0.05) / 0.10 * 15
        elif weighted_rs >= 0:
            score = 50 + weighted_rs / 0.05 * 10
        elif weighted_rs >= -0.15:
            score = 35 + (weighted_rs + 0.15) / 0.15 * 15
        else:
            score = max(10, 35 + (weighted_rs + 0.15) * 80)

        return {
            'score': min(100, max(0, score)),
            'raw': weighted_rs,
            'lookback': '126d_centered',
            'details': {rs[0]: {'stock': round(rs[2], 4), 'sector': round(rs[3], 4), 'diff': round(rs[1], 4)}
                        for rs in relative_strengths}
        }

    def _calc_em3_eps_revision(self) -> Dict:
        """EM3: EPS Estimate Revision (30-day EPS estimate change rate)

        Fallback strategy:
        1. Primary: EPS estimate 30-day change from us_earnings_estimates
        2. Fallback: EPS YoY growth from us_stock_basic
        3. Final: Neutral score (50)
        """

        # 1. Primary: us_earnings_estimates EPS revision
        eps_current = self.earnings_estimates.get('eps_current')
        eps_30d_ago = self.earnings_estimates.get('eps_30d_ago')

        if eps_current and eps_30d_ago and eps_30d_ago != 0:
            eps_revision = (eps_current - eps_30d_ago) / abs(eps_30d_ago)

            # Score calculation: Higher revision = Higher score
            if eps_revision >= 0.10:  # 10%+ upward revision
                score = 90 + min(10, (eps_revision - 0.10) / 0.05 * 10)
            elif eps_revision >= 0.05:
                score = 75 + (eps_revision - 0.05) / 0.05 * 15
            elif eps_revision >= 0.02:
                score = 60 + (eps_revision - 0.02) / 0.03 * 15
            elif eps_revision >= 0:
                score = 50 + eps_revision / 0.02 * 10
            elif eps_revision >= -0.02:
                score = 40 + (eps_revision + 0.02) / 0.02 * 10
            elif eps_revision >= -0.05:
                score = 25 + (eps_revision + 0.05) / 0.03 * 15
            else:
                score = max(10, 25 + (eps_revision + 0.05) * 150)

            return {
                'score': min(100, max(0, score)),
                'raw': eps_revision,
                'eps_current': eps_current,
                'eps_30d_ago': eps_30d_ago,
                'source': 'earnings_estimates'
            }

        # 2. Fallback: us_stock_basic earnings_growth (YoY)
        earnings_growth = self.stock_data.get('earnings_growth')

        if earnings_growth is not None:
            # YoY growth scoring (different scale than revision)
            if earnings_growth >= 0.50:  # 50%+ YoY growth
                score = 85
            elif earnings_growth >= 0.25:
                score = 70 + (earnings_growth - 0.25) / 0.25 * 15
            elif earnings_growth >= 0.10:
                score = 55 + (earnings_growth - 0.10) / 0.15 * 15
            elif earnings_growth >= 0:
                score = 45 + earnings_growth / 0.10 * 10
            elif earnings_growth >= -0.20:
                score = 30 + (earnings_growth + 0.20) / 0.20 * 15
            else:
                score = max(15, 30 + (earnings_growth + 0.20) * 50)

            return {
                'score': min(100, max(0, score)),
                'raw': earnings_growth,
                'source': 'earnings_growth_yoy'
            }

        # 3. Final fallback: No data (exclude from weighted average)
        return {'score': None, 'raw': None, 'source': 'no_data'}

    def _calc_em4_revenue_revision(self) -> Dict:
        """EM4: Revenue Estimate Revision

        Note: us_earnings_estimates does not have revenue_estimate_average_30_days_ago column,
        so we use revenue YoY growth from us_stock_basic as primary source.

        Fallback strategy:
        1. Primary: Revenue YoY growth from us_stock_basic
        2. Final: Neutral score (50)
        """

        # 1. Primary: us_stock_basic revenue_growth (YoY)
        revenue_growth = self.stock_data.get('revenue_growth')

        if revenue_growth is not None:
            # Revenue growth scoring
            if revenue_growth >= 0.50:  # 50%+ YoY growth
                score = 90
            elif revenue_growth >= 0.30:
                score = 75 + (revenue_growth - 0.30) / 0.20 * 15
            elif revenue_growth >= 0.15:
                score = 60 + (revenue_growth - 0.15) / 0.15 * 15
            elif revenue_growth >= 0.05:
                score = 50 + (revenue_growth - 0.05) / 0.10 * 10
            elif revenue_growth >= 0:
                score = 45 + revenue_growth / 0.05 * 5
            elif revenue_growth >= -0.10:
                score = 35 + (revenue_growth + 0.10) / 0.10 * 10
            elif revenue_growth >= -0.25:
                score = 20 + (revenue_growth + 0.25) / 0.15 * 15
            else:
                score = max(10, 20 + (revenue_growth + 0.25) * 40)

            return {
                'score': min(100, max(0, score)),
                'raw': revenue_growth,
                'source': 'revenue_growth_yoy'
            }

        # 2. Final fallback: No data (exclude from weighted average)
        return {'score': None, 'raw': None, 'source': 'no_data'}

    def _calc_em5_volume_confirmation(self) -> Dict:
        """EM5: Volume Confirmation (거래량 동반 여부)"""

        if len(self.price_data) < 50:
            return {'score': None, 'raw': None, 'reason': 'Insufficient data'}

        # 최근 20일 vs 50일 평균 거래량
        volumes_20 = [p['volume'] for p in self.price_data[:20] if p['volume']]
        volumes_50 = [p['volume'] for p in self.price_data[:50] if p['volume']]

        if not volumes_20 or not volumes_50:
            return {'score': None, 'raw': None, 'reason': 'No volume data'}

        avg_vol_20 = sum(volumes_20) / len(volumes_20)
        avg_vol_50 = sum(volumes_50) / len(volumes_50)

        if avg_vol_50 <= 0:
            return {'score': None, 'raw': None, 'reason': 'Invalid volume'}

        # 거래량 증가율
        vol_ratio = avg_vol_20 / avg_vol_50

        # 가격 방향 확인 (최근 20일)
        prices = [p['close'] for p in self.price_data[:20] if p['close']]
        if len(prices) >= 2:
            price_change = (prices[0] - prices[-1]) / prices[-1]
        else:
            price_change = 0

        # 가격 상승 + 거래량 증가 = 강한 모멘텀
        # 가격 하락 + 거래량 증가 = 매도 압력 (부정적)
        # 가격 상승 + 거래량 감소 = 약한 모멘텀

        if price_change > 0.05:  # 가격 상승 중
            if vol_ratio >= 1.5:
                score = 90  # 강한 매수세
            elif vol_ratio >= 1.2:
                score = 75 + (vol_ratio - 1.2) / 0.3 * 15
            elif vol_ratio >= 1.0:
                score = 60 + (vol_ratio - 1.0) / 0.2 * 15
            elif vol_ratio >= 0.8:
                score = 50 + (vol_ratio - 0.8) / 0.2 * 10
            else:
                score = 40  # 상승하지만 거래량 감소 (약한 모멘텀)
        elif price_change > 0:  # 소폭 상승
            if vol_ratio >= 1.3:
                score = 70
            elif vol_ratio >= 1.0:
                score = 55 + (vol_ratio - 1.0) / 0.3 * 15
            else:
                score = 50
        elif price_change > -0.05:  # 보합 ~ 소폭 하락
            if vol_ratio >= 1.3:
                score = 40  # 하락 + 높은 거래량 = 매도 압력
            else:
                score = 45
        else:  # 하락 중
            if vol_ratio >= 1.5:
                score = 25  # 강한 매도세
            elif vol_ratio >= 1.2:
                score = 30
            elif vol_ratio >= 1.0:
                score = 35
            else:
                score = 40  # 하락하지만 거래량 감소 (매도세 둔화)

        return {
            'score': min(100, max(0, score)),
            'raw': vol_ratio,
            'avg_vol_20': avg_vol_20,
            'avg_vol_50': avg_vol_50,
            'price_change': price_change
        }

    def _calc_em6_earnings_momentum(self) -> Dict:
        """EM6: Earnings Momentum (EPS Estimate Revisions)

        Uses us_earnings_estimates data:
        - revision_up_7d / revision_down_7d: Net revision direction
        - eps_30d change rate: Estimate momentum

        Fallback strategy:
        1. Primary: us_earnings_estimates revision data
        2. Fallback: us_stock_basic earnings_growth
        3. Final: Neutral score (50)
        """

        # 1. Primary: us_earnings_estimates revision data
        revision_up = self.earnings_estimates.get('revision_up_7d')
        revision_down = self.earnings_estimates.get('revision_down_7d')
        eps_current = self.earnings_estimates.get('eps_current')
        eps_30d_ago = self.earnings_estimates.get('eps_30d_ago')

        has_revision_data = (revision_up is not None or revision_down is not None)
        has_eps_change = (eps_current and eps_30d_ago and eps_30d_ago != 0)

        if has_revision_data or has_eps_change:
            score_components = []

            # Component 1: Net Revision Direction (7-day)
            if has_revision_data:
                rev_up = revision_up or 0
                rev_down = revision_down or 0

                # Normalize revision counts (typically in thousands)
                if rev_up > 1000:
                    rev_up = rev_up / 1000
                if rev_down > 1000:
                    rev_down = rev_down / 1000

                net_revision = rev_up - rev_down

                if net_revision >= 5:
                    rev_score = 90
                elif net_revision >= 3:
                    rev_score = 75 + (net_revision - 3) / 2 * 15
                elif net_revision >= 1:
                    rev_score = 60 + (net_revision - 1) / 2 * 15
                elif net_revision >= 0:
                    rev_score = 50 + net_revision * 10
                elif net_revision >= -2:
                    rev_score = 35 + (net_revision + 2) / 2 * 15
                else:
                    rev_score = max(15, 35 + (net_revision + 2) * 5)

                score_components.append(('revision', rev_score, net_revision))

            # Component 2: EPS Estimate Change (30-day)
            if has_eps_change:
                eps_change = (eps_current - eps_30d_ago) / abs(eps_30d_ago)

                if eps_change >= 0.05:
                    change_score = 85 + min(15, (eps_change - 0.05) / 0.05 * 15)
                elif eps_change >= 0.02:
                    change_score = 70 + (eps_change - 0.02) / 0.03 * 15
                elif eps_change >= 0:
                    change_score = 55 + eps_change / 0.02 * 15
                elif eps_change >= -0.02:
                    change_score = 40 + (eps_change + 0.02) / 0.02 * 15
                elif eps_change >= -0.05:
                    change_score = 25 + (eps_change + 0.05) / 0.03 * 15
                else:
                    change_score = max(10, 25 + (eps_change + 0.05) * 150)

                score_components.append(('eps_change', change_score, eps_change))

            if score_components:
                # Weighted average (revision 60%, eps_change 40%)
                if len(score_components) == 2:
                    final_score = score_components[0][1] * 0.6 + score_components[1][1] * 0.4
                else:
                    final_score = score_components[0][1]

                raw_data = {comp[0]: comp[2] for comp in score_components}

                return {
                    'score': min(100, max(0, final_score)),
                    'raw': raw_data,
                    'source': 'earnings_estimates'
                }

        # 2. Fallback: us_stock_basic earnings_growth
        earnings_growth = self.stock_data.get('earnings_growth')

        if earnings_growth is not None:
            if earnings_growth >= 0.50:
                score = 85
            elif earnings_growth >= 0.25:
                score = 70 + (earnings_growth - 0.25) / 0.25 * 15
            elif earnings_growth >= 0.10:
                score = 55 + (earnings_growth - 0.10) / 0.15 * 15
            elif earnings_growth >= 0:
                score = 45 + earnings_growth / 0.10 * 10
            elif earnings_growth >= -0.20:
                score = 30 + (earnings_growth + 0.20) / 0.20 * 15
            else:
                score = max(15, 30 + (earnings_growth + 0.20) * 50)

            return {
                'score': min(100, max(0, score)),
                'raw': earnings_growth,
                'source': 'earnings_growth_yoy'
            }

        # 3. Final fallback: No data (exclude from weighted average)
        return {'score': None, 'raw': None, 'source': 'no_data'}

    def _calc_em7_analyst_revisions(self) -> Dict:
        """EM7: Analyst Revisions (목표가 대비)"""

        current_price = self.price_data[0]['close'] if self.price_data else None
        target_price = self.stock_data.get('target_price')

        if current_price is None or target_price is None:
            return {'score': None, 'raw': None, 'reason': 'No target price data'}

        # 업사이드 계산
        upside = (target_price - current_price) / current_price

        # 점수 계산
        if upside >= 0.40:
            score = 90 + min(10, (upside - 0.40) / 0.20 * 10)
        elif upside >= 0.25:
            score = 75 + (upside - 0.25) / 0.15 * 15
        elif upside >= 0.15:
            score = 60 + (upside - 0.15) / 0.10 * 15
        elif upside >= 0.05:
            score = 45 + (upside - 0.05) / 0.10 * 15
        elif upside >= 0:
            score = 35 + upside / 0.05 * 10
        elif upside >= -0.10:
            score = 25 + (upside + 0.10) / 0.10 * 10
        else:
            score = max(10, 25 + (upside + 0.10) * 50)

        return {
            'score': min(100, max(0, score)),
            'raw': upside,
            'current_price': current_price,
            'target_price': target_price
        }

    def _calc_em8_long_term_momentum(self) -> Dict:
        """EM8: Long-term Price Momentum (IBD RS Style + Jegadeesh-Titman 12-1M)

        Phase 3 전략: 252일 기반 장기 모멘텀 + 1개월 리버설 제외

        공식: RS = (3-1M)*0.4 + (6-1M)*0.2 + (9-1M)*0.2 + (12-1M)*0.2
        - 각 기간에서 최근 1개월(21 거래일) 제외하여 단기 리버설 효과 제거

        논리:
        - Jegadeesh-Titman (1993): 12-1M 모멘텀이 가장 유효
        - 최근 1개월은 평균 회귀(mean reversion) 경향 -> skip
        - IBD Relative Strength 스타일 가중치 유지

        IC 분석 결과:
        - EM8 IC: +0.178 (전체 전략 중 최고)
        - 1개월 리버설 제외로 추가 IC 개선 기대
        """

        required_days = 252

        if len(self.price_data) < required_days:
            return {'score': None, 'raw': None, 'reason': f'Insufficient price data (need {required_days}+ days)'}

        prices = [p['close'] for p in self.price_data if p['close']]

        if len(prices) < required_days:
            return {'score': None, 'raw': None, 'reason': 'Insufficient close prices'}

        current_price = prices[0]

        # 기간별 수익률 계산 (12-1M 방식: 최근 1개월 skip으로 단기 리버설 제거)
        ret_3m = self._get_period_return_skip1m(prices, 63)    # 3-1M: 63일전 대비 21일전
        ret_6m = self._get_period_return_skip1m(prices, 126)   # 6-1M: 126일전 대비 21일전
        ret_9m = self._get_period_return_skip1m(prices, 189)   # 9-1M: 189일전 대비 21일전
        ret_12m = self._get_period_return_skip1m(prices, 252)  # 12-1M: 252일전 대비 21일전

        if any(r is None for r in [ret_3m, ret_6m, ret_9m, ret_12m]):
            # 일부 데이터만 있어도 계산 가능하도록 처리
            available_returns = [(r, w) for r, w in
                                 [(ret_3m, 0.4), (ret_6m, 0.2), (ret_9m, 0.2), (ret_12m, 0.2)]
                                 if r is not None]

            if not available_returns:
                return {'score': None, 'raw': None, 'reason': 'Cannot calculate any period return'}

            # 가용 데이터로 가중평균 재계산
            total_weight = sum(w for _, w in available_returns)
            rs_value = sum(r * w / total_weight for r, w in available_returns)
        else:
            # IBD RS Style 가중 평균
            rs_value = ret_3m * 0.4 + ret_6m * 0.2 + ret_9m * 0.2 + ret_12m * 0.2

        # 점수 계산 (장기 기준: -50% ~ +100% → 0 ~ 100)
        # rs_value는 % 단위 (예: 50 = 50%)
        if rs_value >= 80:
            score = 95 + min(5, (rs_value - 80) / 40 * 5)
        elif rs_value >= 50:
            score = 85 + (rs_value - 50) / 30 * 10
        elif rs_value >= 25:
            score = 70 + (rs_value - 25) / 25 * 15
        elif rs_value >= 10:
            score = 55 + (rs_value - 10) / 15 * 15
        elif rs_value >= 0:
            score = 45 + rs_value / 10 * 10
        elif rs_value >= -15:
            score = 30 + (rs_value + 15) / 15 * 15
        elif rs_value >= -30:
            score = 20 + (rs_value + 30) / 15 * 10
        else:
            score = max(5, 20 + (rs_value + 30) / 20 * 10)

        return {
            'score': min(100, max(0, score)),
            'raw': round(rs_value, 2),
            'lookback': '252d_ibd_rs',
            'ret_3m': round(ret_3m, 2) if ret_3m else None,
            'ret_6m': round(ret_6m, 2) if ret_6m else None,
            'ret_9m': round(ret_9m, 2) if ret_9m else None,
            'ret_12m': round(ret_12m, 2) if ret_12m else None
        }

    def _get_period_return(self, prices: List[float], days: int) -> Optional[float]:
        """기간 수익률 계산 (%)

        Args:
            prices: 종가 리스트 (최신순, index 0이 최신)
            days: 기간 (거래일 수)

        Returns:
            수익률 (%) 또는 None
        """
        if len(prices) < days:
            return None

        current = prices[0]
        past = prices[days - 1]

        if past and past > 0:
            return (current / past - 1) * 100
        return None

    def _get_period_return_skip1m(self, prices: List[float], days: int, skip: int = 21) -> Optional[float]:
        """기간 수익률 계산 - 최근 1개월(21 거래일) skip (Jegadeesh-Titman 12-1M 방식)

        단기 리버설 효과 제거를 위해 최근 1개월을 제외하고 수익률 계산
        계산: (price_skip일전 - price_days일전) / price_days일전 * 100

        Args:
            prices: 종가 리스트 (최신순, index 0이 최신)
            days: 기간 (거래일 수, 예: 63=3개월, 126=6개월, 252=12개월)
            skip: 제외할 최근 기간 (기본값: 21 거래일 = 약 1개월)

        Returns:
            수익률 (%) 또는 None

        Example:
            days=252, skip=21 -> 12-1M momentum
            (21일 전 가격 - 252일 전 가격) / 252일 전 가격
        """
        if len(prices) < days:
            return None

        # skip일 전 가격 (최근 1개월 제외 시점)
        recent = prices[skip - 1] if len(prices) > skip else prices[0]
        # days일 전 가격
        past = prices[days - 1]

        if past and past > 0 and recent:
            return (recent / past - 1) * 100
        return None

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

# ========================================================================
# Convenience Function
# ========================================================================

async def calculate_momentum_score(symbol: str, db_manager, sector_benchmarks: Dict,
                                    analysis_date: Optional[date] = None,
                                    exchange: Optional[str] = None,
                                    volatility_engine: Optional['USVolatilityAdjustment'] = None,
                                    prefetched_data: Optional[Dict] = None) -> Dict:
    """Momentum Factor 점수 계산 (편의 함수)

    Args:
        symbol: 종목 코드
        db_manager: AsyncDatabaseManager
        sector_benchmarks: 섹터 기준값
        analysis_date: 분석 날짜
        exchange: 거래소 ('NASDAQ'이면 전용 가중치 사용)
        volatility_engine: Phase 3.4.2 - Shared volatility engine (optional)
        prefetched_data: Phase 3.4.3 - Prefetched data from USDataPrefetcher (optional)

    Returns:
        {
            'momentum_score': float (0-100),
            'strategies': {...}
        }
    """
    calculator = USMomentumFactorV2(symbol, db_manager, sector_benchmarks, analysis_date, exchange,
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
        print("Momentum Factor v3.0 Test (Phase 3)")
        print("Strategies: EM1, EM2, EM3, EM4, EM6, EM8")
        print("Excluded: EM5 (Volume), EM7 (Analyst)")
        print("="*60)

        for symbol in test_symbols:
            result = await calculate_momentum_score(symbol, db, benchmarks)
            print(f"\n{symbol}: Momentum Score = {result['momentum_score']}")

            for strat_id, strat_data in result['strategies'].items():
                if strat_data['score'] is not None:
                    raw = strat_data.get('raw', 'N/A')
                    if isinstance(raw, dict):
                        raw = 'complex'
                    print(f"  {strat_id}: {strat_data['score']:.1f} (raw: {raw})")

        await db.close()

    asyncio.run(test())
