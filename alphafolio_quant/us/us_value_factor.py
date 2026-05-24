"""
US Value Factor v2.0 - Reasonable Valuation

핵심 변경: "저평가 = 좋다" → "성장 대비 합리적 가격 = 좋다"

Strategies:
- RV1: PEG Ratio (25%) - 성장 대비 밸류에이션
- RV2: Forward PE Relative (20%) - 섹터 대비 Forward PE
- RV3: EV/Revenue to Growth (20%) - EV/매출 대비 성장률
- RV4: Rule of 40 (15%) - 성장률 + 마진 (Tech/SaaS)
- RV5: FCF Yield Adjusted (10%) - 조정 FCF 수익률
- RV6: Price to Target (10%) - 애널리스트 목표가 대비

File: us/us_value_factor.py
"""

import logging
from typing import Dict, Optional, List
from datetime import date
from decimal import Decimal

from us_volatility_adjustment import USVolatilityAdjustment

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ========================================================================
# Strategy Weights
# ========================================================================

STRATEGY_WEIGHTS = {
    'RV1': 0.25,  # PEG
    'RV2': 0.20,  # Forward PE Relative
    'RV3': 0.20,  # EV/Rev to Growth
    'RV4': 0.15,  # Rule of 40
    'RV5': 0.10,  # FCF Yield
    'RV6': 0.10   # Price to Target
}

# Phase 3.1.4 역방향 전략 (음의 IC → 점수 역전으로 양의 IC 기대)
# IC 분석 결과:
#   - RV2: IC -0.139 → 역방향 적용
REVERSED_STRATEGIES = ['RV2']

# 섹터별 PEG 기준 (성장주 섹터는 기준 완화)
SECTOR_PEG_THRESHOLDS = {
    'TECHNOLOGY': {'excellent': 1.3, 'good': 2.0, 'fair': 3.0},
    'HEALTHCARE': {'excellent': 1.4, 'good': 2.2, 'fair': 3.2},
    'COMMUNICATION SERVICES': {'excellent': 1.2, 'good': 1.8, 'fair': 2.5},
    'CONSUMER CYCLICAL': {'excellent': 1.1, 'good': 1.7, 'fair': 2.3},
    'DEFAULT': {'excellent': 1.0, 'good': 1.5, 'fair': 2.0}
}

# Rule of 40 적용 섹터
RULE_OF_40_SECTORS = ['TECHNOLOGY', 'COMMUNICATION SERVICES']

# RV5 FCF Yield 적용 섹터 (Utilities / Financial Services 섹터에만 적용)
RV5_APPLICABLE_SECTORS = ['UTILITIES', 'FINANCIAL SERVICES']


class USValueFactorV2:
    """Value Factor v2.0 - Reasonable Valuation"""

    def __init__(self, symbol: str, db_manager, sector_benchmarks: Dict,
                 analysis_date: Optional[date] = None,
                 volatility_engine: Optional['USVolatilityAdjustment'] = None,
                 prefetched_data: Optional[Dict] = None):
        """
        Args:
            symbol: 종목 코드
            db_manager: AsyncDatabaseManager
            sector_benchmarks: 섹터 기준값 (from USSectorBenchmarks)
            analysis_date: 분석 날짜
            volatility_engine: Phase 3.4.2 - Shared volatility engine (optional)
            prefetched_data: Phase 3.4.3 - Prefetched data from USDataPrefetcher (optional)
        """
        self.symbol = symbol
        self.db = db_manager
        self.benchmarks = sector_benchmarks
        self.analysis_date = analysis_date
        self.stock_data = {}
        self.sector = None
        # Phase 3.4.2: Shared Volatility Engine (can be passed from main)
        self.volatility_engine: Optional[USVolatilityAdjustment] = volatility_engine
        self.price_data: List[Dict] = []
        # Phase 3.4.3: Prefetched data for query optimization
        self.prefetched_data: Optional[Dict] = prefetched_data

    async def calculate(self) -> Dict:
        """
        Value Factor 점수 계산

        Returns:
            {
                'value_score': float (0-100),
                'strategies': {
                    'RV1': {'score': float, 'raw': float, ...},
                    ...
                }
            }
        """
        try:
            # 1. 종목 데이터 조회 (Phase 3.4.3: prefetched_data 우선 사용)
            if self.prefetched_data:
                self._load_from_prefetched()
            else:
                await self._load_stock_data()

            # Phase 3.4.2: Shared Volatility Engine (use passed engine or create new)
            if self.volatility_engine is None:
                await self._load_price_data()
                self.volatility_engine = USVolatilityAdjustment(self.db)
                await self.volatility_engine.load_data(self.symbol, self.analysis_date, self.price_data)

            if not self.stock_data:
                logger.warning(f"{self.symbol}: No data available")
                return {'value_score': 50.0, 'strategies': {}}

            # 2. 각 전략별 점수 계산
            # Phase 3.1.6: 섹터별 전략 적용
            # - RV5: Utilities/Financial Services 섹터에만 적용
            strategies = {}
            strategies['RV1'] = self._calc_rv1_peg()
            strategies['RV2'] = self._calc_rv2_forward_pe_relative()
            strategies['RV3'] = self._calc_rv3_ev_rev_growth()
            strategies['RV4'] = self._calc_rv4_rule_of_40()

            # RV5: Utilities/Financial Services 섹터에만 적용
            if self.sector and self.sector.upper() in RV5_APPLICABLE_SECTORS:
                strategies['RV5'] = self._calc_rv5_fcf_yield()
            else:
                strategies['RV5'] = {'score': None, 'raw': None, 'reason': f'Not applicable sector (only {RV5_APPLICABLE_SECTORS})'}

            strategies['RV6'] = self._calc_rv6_price_to_target()

            # 3. 역방향 전략 적용 (Phase 3.1.4)
            for strategy_id, result in strategies.items():
                if result['score'] is not None and strategy_id in REVERSED_STRATEGIES:
                    original_score = result['score']
                    reversed_score = 100 - original_score
                    result['original_score'] = original_score
                    result['score'] = reversed_score
                    result['reversed'] = True

            # 4. 가중 평균 계산
            total_weight = 0
            weighted_sum = 0

            for strategy_id, result in strategies.items():
                if result['score'] is not None:
                    weight = STRATEGY_WEIGHTS[strategy_id]
                    weighted_sum += result['score'] * weight
                    total_weight += weight

            value_score = weighted_sum / total_weight if total_weight > 0 else 50.0

            # 5. Phase 3.4.2: Volatility Adjustment (sensitivity=0.3 for Value)
            vol_adjustment = self.volatility_engine.get_adjustment('value')
            original_score = value_score
            value_score = max(0, min(100, value_score + vol_adjustment['modifier']))

            if vol_adjustment['modifier'] != 0:
                logger.info(f"{self.symbol}: Value {original_score:.1f} -> {value_score:.1f} "
                           f"(Vol Adj: {vol_adjustment['modifier']:+.1f}, IV%: {self.volatility_engine.iv_percentile})")
            else:
                logger.info(f"{self.symbol}: Value Score = {value_score:.1f}")

            return {
                'value_score': round(value_score, 1),
                'value_original': round(original_score, 1),
                'strategies': strategies,
                # Phase 3.4.2: Volatility data
                'iv_percentile': self.volatility_engine.iv_percentile if self.volatility_engine else None,
                'hv_20d': self.volatility_engine.hv_20d if self.volatility_engine else None,
                'vol_adjustment': vol_adjustment
            }

        except Exception as e:
            logger.error(f"{self.symbol}: Value calculation failed - {e}")
            return {'value_score': 50.0, 'strategies': {}}

    async def _load_stock_data(self):
        """종목 기본 데이터 로드"""

        query = """
        SELECT
            b.symbol,
            b.sector,
            b.per,
            b.forwardpe,
            b.peg,
            b.pricetosalesratiottm as ps_ratio,
            b.evtorevenue,
            b.evtoebitda,
            b.market_cap,
            b.ebitda,
            b.revenuettm as revenue,
            b.grossprofitttm as gross_profit,
            b.operatingmarginttm as operating_margin,
            b.quarterlyrevenuegrowthyoy as revenue_growth,
            b.quarterlyearningsgrowthyoy as earnings_growth,
            b.analysttargetprice as target_price,
            b.profitmargin,
            d.close as current_price
        FROM us_stock_basic b
        LEFT JOIN (
            SELECT symbol, close
            FROM us_daily
            WHERE ($2::DATE IS NULL OR date <= $2::DATE)
            ORDER BY date DESC
            LIMIT 1
        ) d ON b.symbol = d.symbol
        WHERE b.symbol = $1
        """

        try:
            result = await self.db.execute_query(query, self.symbol, self.analysis_date)

            if result and result[0]:
                row = result[0]
                self.sector = row['sector']
                self.stock_data = {
                    'per': self._to_float(row['per']),
                    'forward_pe': self._to_float(row['forwardpe']),
                    'peg': self._to_float(row['peg']),
                    'ps_ratio': self._to_float(row['ps_ratio']),
                    'ev_revenue': self._to_float(row['evtorevenue']),
                    'ev_ebitda': self._to_float(row['evtoebitda']),
                    'market_cap': self._to_float(row['market_cap']),
                    'ebitda': self._to_float(row['ebitda']),
                    'revenue': self._to_float(row['revenue']),
                    'gross_profit': self._to_float(row['gross_profit']),
                    'operating_margin': self._to_float(row['operating_margin']),
                    'revenue_growth': self._to_float(row['revenue_growth']),
                    'earnings_growth': self._to_float(row['earnings_growth']),
                    'target_price': self._to_float(row['target_price']),
                    'current_price': self._to_float(row['current_price']),
                    'profit_margin': self._to_float(row['profitmargin'])
                }

            # FCF 데이터 추가 조회
            await self._load_fcf_data()

        except Exception as e:
            logger.error(f"{self.symbol}: Failed to load stock data - {e}")

    async def _load_fcf_data(self):
        """FCF 데이터 조회"""

        query = """
        SELECT
            operating_cashflow,
            capital_expenditures
        FROM us_cash_flow
        WHERE symbol = $1
        ORDER BY fiscal_date_ending DESC
        LIMIT 4
        """

        try:
            result = await self.db.execute_query(query, self.symbol)

            if result:
                total_ocf = sum(self._to_float(r['operating_cashflow']) or 0 for r in result)
                total_capex = sum(abs(self._to_float(r['capital_expenditures']) or 0) for r in result)
                self.stock_data['fcf_ttm'] = total_ocf - total_capex

        except Exception as e:
            logger.warning(f"{self.symbol}: FCF data not available - {e}")
            self.stock_data['fcf_ttm'] = None

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
        Maps prefetched_data to stock_data format.
        """
        if not self.prefetched_data:
            return

        stock_basic = self.prefetched_data.get('stock_basic', {})
        price_data = self.prefetched_data.get('price_data', {})
        financials = self.prefetched_data.get('financials', {})

        if not stock_basic:
            return

        # Set sector
        self.sector = stock_basic.get('sector')

        # Get current price from price_data
        closes = price_data.get('closes', []) if price_data else []
        current_price = closes[0] if closes and closes[0] else None

        # Map stock_basic to stock_data
        self.stock_data = {
            'per': self._to_float(stock_basic.get('per')),
            'forward_pe': self._to_float(stock_basic.get('forwardpe')),
            'peg': self._to_float(stock_basic.get('peg')),
            'ps_ratio': self._to_float(stock_basic.get('pricetosalesratiottm')),
            'ev_revenue': self._to_float(stock_basic.get('evtorevenue')),
            'ev_ebitda': self._to_float(stock_basic.get('evtoebitda')),
            'market_cap': self._to_float(stock_basic.get('market_cap')),
            'ebitda': self._to_float(stock_basic.get('ebitda')),
            'revenue': self._to_float(stock_basic.get('revenuettm')),
            'gross_profit': self._to_float(stock_basic.get('grossprofitttm')),
            'operating_margin': self._to_float(stock_basic.get('operatingmarginttm')),
            'revenue_growth': self._to_float(stock_basic.get('quarterlyrevenuegrowthyoy')),
            'earnings_growth': self._to_float(stock_basic.get('quarterlyearningsgrowthyoy')),
            'target_price': self._to_float(stock_basic.get('analysttargetprice')),
            'current_price': current_price,
            'profit_margin': self._to_float(stock_basic.get('profitmargin'))
        }

        # Extract FCF from financials (cash_flow)
        cashflow = financials.get('cashflow', []) if financials else []
        if cashflow and len(cashflow) >= 4:
            total_ocf = sum(self._to_float(r.get('operating_cashflow')) or 0 for r in cashflow[:4])
            total_capex = sum(abs(self._to_float(r.get('capital_expenditures')) or 0) for r in cashflow[:4])
            self.stock_data['fcf_ttm'] = total_ocf - total_capex
        else:
            self.stock_data['fcf_ttm'] = None

    def _calc_rv1_peg(self) -> Dict:
        """RV1: PEG Ratio 점수"""

        peg = self.stock_data.get('peg')

        if peg is None or peg <= 0:
            # PEG 계산 불가 시 PE와 Growth로 추정
            pe = self.stock_data.get('forward_pe') or self.stock_data.get('per')
            growth = self.stock_data.get('earnings_growth')

            if pe and growth and growth > 0:
                peg = pe / (growth * 100)  # growth는 소수점 형태
            else:
                return {'score': None, 'raw': None, 'reason': 'No PEG data'}

        # 섹터별 기준 적용
        thresholds = SECTOR_PEG_THRESHOLDS.get(self.sector, SECTOR_PEG_THRESHOLDS['DEFAULT'])

        if peg < 0:
            score = 20  # 역성장 또는 적자
        elif peg <= thresholds['excellent']:
            score = 85 + (thresholds['excellent'] - peg) / thresholds['excellent'] * 15
        elif peg <= thresholds['good']:
            score = 65 + (thresholds['good'] - peg) / (thresholds['good'] - thresholds['excellent']) * 20
        elif peg <= thresholds['fair']:
            score = 45 + (thresholds['fair'] - peg) / (thresholds['fair'] - thresholds['good']) * 20
        else:
            score = max(15, 45 - (peg - thresholds['fair']) * 8)

        return {
            'score': min(100, max(0, score)),
            'raw': peg,
            'thresholds': thresholds
        }

    def _calc_rv2_forward_pe_relative(self) -> Dict:
        """RV2: 섹터 대비 Forward PE"""

        forward_pe = self.stock_data.get('forward_pe')
        sector_median = self.benchmarks.get('forward_pe_median')

        if forward_pe is None or sector_median is None or sector_median <= 0:
            return {'score': None, 'raw': None, 'reason': 'No Forward PE data'}

        # 섹터 대비 비율 (1.0 = 섹터 평균)
        ratio = forward_pe / sector_median

        # 점수 계산 (섹터 평균 대비)
        if ratio <= 0.6:
            score = 90  # 섹터 대비 40% 이상 할인
        elif ratio <= 0.8:
            score = 75 + (0.8 - ratio) / 0.2 * 15
        elif ratio <= 1.0:
            score = 55 + (1.0 - ratio) / 0.2 * 20
        elif ratio <= 1.2:
            score = 40 + (1.2 - ratio) / 0.2 * 15
        elif ratio <= 1.5:
            score = 25 + (1.5 - ratio) / 0.3 * 15
        else:
            score = max(10, 25 - (ratio - 1.5) * 10)

        return {
            'score': min(100, max(0, score)),
            'raw': ratio,
            'forward_pe': forward_pe,
            'sector_median': sector_median
        }

    def _calc_rv3_ev_rev_growth(self) -> Dict:
        """RV3: EV/Revenue to Growth"""

        ev_rev = self.stock_data.get('ev_revenue')
        rev_growth = self.stock_data.get('revenue_growth')

        if ev_rev is None or rev_growth is None or rev_growth <= 0:
            return {'score': None, 'raw': None, 'reason': 'No EV/Rev or Growth data'}

        # EV/Rev ÷ 성장률 (낮을수록 좋음)
        # 예: EV/Rev=5, Growth=20% → 5/20 = 0.25
        growth_pct = rev_growth * 100 if rev_growth < 1 else rev_growth
        ratio = ev_rev / growth_pct if growth_pct > 0 else 999

        # 점수 계산
        if ratio <= 0.2:
            score = 90
        elif ratio <= 0.4:
            score = 70 + (0.4 - ratio) / 0.2 * 20
        elif ratio <= 0.7:
            score = 50 + (0.7 - ratio) / 0.3 * 20
        elif ratio <= 1.0:
            score = 35 + (1.0 - ratio) / 0.3 * 15
        else:
            score = max(10, 35 - (ratio - 1.0) * 15)

        return {
            'score': min(100, max(0, score)),
            'raw': ratio,
            'ev_revenue': ev_rev,
            'revenue_growth': growth_pct
        }

    def _calc_rv4_rule_of_40(self) -> Dict:
        """RV4: Rule of 40 (성장률 + 이익률 >= 40)"""

        # Tech/SaaS 섹터만 적용
        if self.sector not in RULE_OF_40_SECTORS:
            # 다른 섹터는 Operating Margin만 평가
            op_margin = self.stock_data.get('operating_margin')
            if op_margin is None:
                return {'score': None, 'raw': None, 'reason': 'Not applicable sector'}

            margin_pct = op_margin * 100 if abs(op_margin) < 1 else op_margin
            if margin_pct >= 20:
                score = 70 + min(30, (margin_pct - 20) / 20 * 30)
            elif margin_pct >= 10:
                score = 50 + (margin_pct - 10) / 10 * 20
            elif margin_pct >= 0:
                score = 30 + margin_pct / 10 * 20
            else:
                score = max(10, 30 + margin_pct)

            return {
                'score': min(100, max(0, score)),
                'raw': margin_pct,
                'type': 'operating_margin_only'
            }

        # Rule of 40 계산
        rev_growth = self.stock_data.get('revenue_growth')
        op_margin = self.stock_data.get('operating_margin')

        if rev_growth is None or op_margin is None:
            return {'score': None, 'raw': None, 'reason': 'No growth or margin data'}

        # 백분율로 변환
        growth_pct = rev_growth * 100 if abs(rev_growth) < 1 else rev_growth
        margin_pct = op_margin * 100 if abs(op_margin) < 1 else op_margin

        rule_of_40 = growth_pct + margin_pct

        # 점수 계산
        if rule_of_40 >= 60:
            score = 95
        elif rule_of_40 >= 50:
            score = 80 + (rule_of_40 - 50) / 10 * 15
        elif rule_of_40 >= 40:
            score = 65 + (rule_of_40 - 40) / 10 * 15
        elif rule_of_40 >= 30:
            score = 50 + (rule_of_40 - 30) / 10 * 15
        elif rule_of_40 >= 20:
            score = 35 + (rule_of_40 - 20) / 10 * 15
        else:
            score = max(10, 35 - (20 - rule_of_40) * 1.5)

        return {
            'score': min(100, max(0, score)),
            'raw': rule_of_40,
            'revenue_growth': growth_pct,
            'operating_margin': margin_pct
        }

    def _calc_rv5_fcf_yield(self) -> Dict:
        """RV5: FCF Yield (조정)"""

        fcf = self.stock_data.get('fcf_ttm')
        market_cap = self.stock_data.get('market_cap')

        if fcf is None or market_cap is None or market_cap <= 0:
            return {'score': None, 'raw': None, 'reason': 'No FCF or Market Cap data'}

        fcf_yield = fcf / market_cap * 100  # %

        # 점수 계산
        if fcf_yield >= 10:
            score = 90 + min(10, (fcf_yield - 10) / 5 * 10)
        elif fcf_yield >= 6:
            score = 70 + (fcf_yield - 6) / 4 * 20
        elif fcf_yield >= 3:
            score = 50 + (fcf_yield - 3) / 3 * 20
        elif fcf_yield >= 0:
            score = 30 + fcf_yield / 3 * 20
        else:
            # 음수 FCF (성장 투자 중일 수 있음 - 섹터에 따라 패널티 완화)
            if self.sector in RULE_OF_40_SECTORS:
                score = max(20, 30 + fcf_yield * 2)  # 완화된 패널티
            else:
                score = max(10, 30 + fcf_yield * 3)

        return {
            'score': min(100, max(0, score)),
            'raw': fcf_yield,
            'fcf': fcf,
            'market_cap': market_cap
        }

    def _calc_rv6_price_to_target(self) -> Dict:
        """RV6: 현재가 vs 애널리스트 목표가"""

        current = self.stock_data.get('current_price')
        target = self.stock_data.get('target_price')

        if current is None or target is None or target <= 0:
            return {'score': None, 'raw': None, 'reason': 'No price or target data'}

        # 업사이드 계산
        upside = (target - current) / current * 100  # %

        # 점수 계산
        if upside >= 40:
            score = 90 + min(10, (upside - 40) / 20 * 10)
        elif upside >= 25:
            score = 75 + (upside - 25) / 15 * 15
        elif upside >= 15:
            score = 60 + (upside - 15) / 10 * 15
        elif upside >= 5:
            score = 45 + (upside - 5) / 10 * 15
        elif upside >= 0:
            score = 35 + upside / 5 * 10
        elif upside >= -10:
            score = 25 + (upside + 10) / 10 * 10
        else:
            score = max(10, 25 + (upside + 10) * 1.5)

        return {
            'score': min(100, max(0, score)),
            'raw': upside,
            'current_price': current,
            'target_price': target
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


# ========================================================================
# Convenience Function
# ========================================================================

async def calculate_value_score(symbol: str, db_manager, sector_benchmarks: Dict,
                                 analysis_date: Optional[date] = None,
                                 volatility_engine: Optional['USVolatilityAdjustment'] = None,
                                 prefetched_data: Optional[Dict] = None) -> Dict:
    """Value Factor 점수 계산 (편의 함수)

    Args:
        volatility_engine: Phase 3.4.2 - Shared volatility engine (optional)
        prefetched_data: Phase 3.4.3 - Prefetched data from USDataPrefetcher (optional)
    """

    calculator = USValueFactorV2(symbol, db_manager, sector_benchmarks, analysis_date,
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
        print("Value Factor v2.0 Test")
        print("="*60)

        for symbol in test_symbols:
            result = await calculate_value_score(symbol, db, benchmarks)
            print(f"\n{symbol}: Value Score = {result['value_score']}")

            for strat_id, strat_data in result['strategies'].items():
                if strat_data['score'] is not None:
                    print(f"  {strat_id}: {strat_data['score']:.1f} (raw: {strat_data.get('raw', 'N/A')})")

        await db.close()

    asyncio.run(test())
