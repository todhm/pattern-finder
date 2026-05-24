"""
US Quality Factor v2.0 - Management Quality

핵심: 지속 가능한 고품질 기업 (일시적 수익이 아닌 구조적 우위)

Strategies:
- MQ1: Gross Margin (20%) - 총이익률 (섹터 대비 경쟁 우위)
- MQ2: ROIC (20%) - 투하자본수익률 (자본 효율성)
- MQ3: Operating Leverage (15%) - 영업 레버리지 (확장성)
- MQ4: Earnings Quality (15%) - 이익의 질 (현금 vs 발생주의)
- MQ5: Balance Sheet (15%) - 재무건전성 (부채/유동성)
- MQ6: Margin Trend (15%) - 마진 추세 (개선 vs 악화)

File: us/us_quality_factor.py
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

# 일반 섹터 가중치
STRATEGY_WEIGHTS = {
    'MQ1': 0.20,  # Gross Margin
    'MQ2': 0.20,  # ROIC
    'MQ3': 0.15,  # Operating Leverage
    'MQ4': 0.15,  # Earnings Quality
    'MQ5': 0.15,  # Balance Sheet
    'MQ6': 0.15   # Margin Trend
}

# Healthcare 섹터 전용 가중치 (HC1~HC4 Healthcare 특화 전략 포함)
# 기존 MQ1~MQ6 가중치 축소 + HC1~HC4 40% 추가 = 합계 100%
HEALTHCARE_STRATEGY_WEIGHTS = {
    'MQ1': 0.12,  # Gross Margin (20% → 12%)
    'MQ2': 0.12,  # ROIC (20% → 12%)
    'MQ3': 0.08,  # Operating Leverage (15% → 8%)
    'MQ4': 0.10,  # Earnings Quality (15% → 10%)
    'MQ5': 0.10,  # Balance Sheet (15% → 10%)
    'MQ6': 0.08,  # Margin Trend (15% → 8%)
    # Healthcare 특화 전략 (40%)
    'HC1': 0.10,  # Analyst Consensus Score
    'HC2': 0.10,  # EPS Revision Momentum
    'HC3': 0.10,  # R&D Intensity Score
    'HC4': 0.10   # Cash Runway Score
}
# Total: 12+12+8+10+10+8+10+10+10+10 = 100%

# 섹터별 Gross Margin 기준 (구조적으로 마진이 다름)
SECTOR_GROSS_MARGIN_THRESHOLDS = {
    'TECHNOLOGY': {'excellent': 0.65, 'good': 0.50, 'fair': 0.35},
    'HEALTHCARE': {'excellent': 0.60, 'good': 0.45, 'fair': 0.30},
    'CONSUMER DEFENSIVE': {'excellent': 0.40, 'good': 0.30, 'fair': 0.20},
    'FINANCIAL SERVICES': {'excellent': 0.50, 'good': 0.35, 'fair': 0.20},
    'INDUSTRIALS': {'excellent': 0.35, 'good': 0.25, 'fair': 0.15},
    'ENERGY': {'excellent': 0.40, 'good': 0.30, 'fair': 0.20},
    'UTILITIES': {'excellent': 0.45, 'good': 0.35, 'fair': 0.25},
    'DEFAULT': {'excellent': 0.45, 'good': 0.35, 'fair': 0.25}
}

# ROIC 기준 (WACC 대비 - 일반적으로 8-10% 가정)
ROIC_THRESHOLDS = {
    'excellent': 0.20,  # 20%+ ROIC = 명확한 경쟁 우위
    'good': 0.12,       # 12-20% = 양호
    'fair': 0.08,       # 8-12% = WACC 수준
    'poor': 0.05        # 5-8% = WACC 하회
}


class USQualityFactorV2:
    """Quality Factor v2.0 - Management Quality"""

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
        self.historical_data = []
        # Phase 3.4.2: Shared Volatility Engine (can be passed from main)
        self.volatility_engine: Optional[USVolatilityAdjustment] = volatility_engine
        self.price_data: List[Dict] = []
        # Phase 3.4.3: Prefetched data for query optimization
        self.prefetched_data: Optional[Dict] = prefetched_data

    async def calculate(self) -> Dict:
        """
        Quality Factor 점수 계산

        Returns:
            {
                'quality_score': float (0-100),
                'strategies': {
                    'MQ1': {'score': float, 'raw': float, ...},
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
                await self._load_historical_data()

            # Phase 3.4.2: Shared Volatility Engine (use passed engine or create new)
            if self.volatility_engine is None:
                await self._load_price_data()
                self.volatility_engine = USVolatilityAdjustment(self.db)
                await self.volatility_engine.load_data(self.symbol, self.analysis_date, self.price_data)

            if not self.stock_data:
                logger.warning(f"{self.symbol}: No data available")
                return {'quality_score': 50.0, 'strategies': {}}

            # 2. 각 전략별 점수 계산
            strategies = {}
            strategies['MQ1'] = self._calc_mq1_gross_margin()
            strategies['MQ2'] = self._calc_mq2_roic()
            strategies['MQ3'] = self._calc_mq3_operating_leverage()
            strategies['MQ4'] = self._calc_mq4_earnings_quality()
            strategies['MQ5'] = self._calc_mq5_balance_sheet()
            strategies['MQ6'] = self._calc_mq6_margin_trend()

            # 3. Healthcare 섹터일 때 HC1~HC4 추가 계산
            is_healthcare = self.sector == 'HEALTHCARE'
            if is_healthcare:
                await self._load_healthcare_data()
                strategies['HC1'] = self._calc_hc1_analyst_consensus()
                strategies['HC2'] = self._calc_hc2_eps_revision_momentum()
                strategies['HC3'] = self._calc_hc3_rd_intensity()
                strategies['HC4'] = self._calc_hc4_cash_runway()

            # 4. 가중 평균 계산 (섹터에 따라 다른 가중치 사용)
            weights = HEALTHCARE_STRATEGY_WEIGHTS if is_healthcare else STRATEGY_WEIGHTS
            total_weight = 0
            weighted_sum = 0

            for strategy_id, result in strategies.items():
                if result['score'] is not None and strategy_id in weights:
                    weight = weights[strategy_id]
                    weighted_sum += result['score'] * weight
                    total_weight += weight

            quality_score = weighted_sum / total_weight if total_weight > 0 else 50.0

            # 5. Phase 3.4.2: Volatility Adjustment (sensitivity=0.1 for Quality)
            vol_adjustment = self.volatility_engine.get_adjustment('quality')
            original_score = quality_score
            quality_score = max(0, min(100, quality_score + vol_adjustment['modifier']))

            if vol_adjustment['modifier'] != 0:
                logger.info(f"{self.symbol}: Quality {original_score:.1f} -> {quality_score:.1f} "
                           f"(Vol Adj: {vol_adjustment['modifier']:+.1f}, Healthcare: {is_healthcare})")
            else:
                logger.info(f"{self.symbol}: Quality Score = {quality_score:.1f} (Healthcare: {is_healthcare})")

            return {
                'quality_score': round(quality_score, 1),
                'quality_original': round(original_score, 1),
                'strategies': strategies,
                # Phase 3.4.2: Volatility data
                'iv_percentile': self.volatility_engine.iv_percentile if self.volatility_engine else None,
                'hv_20d': self.volatility_engine.hv_20d if self.volatility_engine else None,
                'vol_adjustment': vol_adjustment
            }

        except Exception as e:
            logger.error(f"{self.symbol}: Quality calculation failed - {e}")
            return {'quality_score': 50.0, 'strategies': {}}

    async def _load_stock_data(self):
        """종목 기본 데이터 로드"""

        query = """
        SELECT
            b.symbol,
            b.sector,
            b.grossprofitttm as gross_profit,
            b.revenuettm as revenue,
            b.operatingmarginttm as operating_margin,
            b.profitmargin as net_margin,
            b.returnonequityttm as roe,
            b.returnonassetsttm as roa,
            b.ebitda,
            b.market_cap,
            b.beta,
            b.sharesoutstanding as shares_outstanding,
            b.bookvalue as book_value,
            b.dilutedepsttm as eps
        FROM us_stock_basic b
        WHERE b.symbol = $1
        """

        try:
            result = await self.db.execute_query(query, self.symbol)

            if result and result[0]:
                row = result[0]
                self.sector = row['sector']
                self.stock_data = {
                    'gross_profit': self._to_float(row['gross_profit']),
                    'revenue': self._to_float(row['revenue']),
                    'operating_margin': self._to_float(row['operating_margin']),
                    'net_margin': self._to_float(row['net_margin']),
                    'roe': self._to_float(row['roe']),
                    'roa': self._to_float(row['roa']),
                    'ebitda': self._to_float(row['ebitda']),
                    'market_cap': self._to_float(row['market_cap']),
                    'beta': self._to_float(row['beta']),
                    'shares_outstanding': self._to_float(row['shares_outstanding']),
                    'book_value': self._to_float(row['book_value']),
                    'eps': self._to_float(row['eps'])
                }

            # Gross Margin 계산
            if self.stock_data.get('gross_profit') and self.stock_data.get('revenue'):
                self.stock_data['gross_margin'] = self.stock_data['gross_profit'] / self.stock_data['revenue']

            # Balance Sheet 데이터 추가
            await self._load_balance_sheet_data()

            # Cash Flow 데이터 추가 (Earnings Quality용)
            await self._load_cash_flow_data()

        except Exception as e:
            logger.error(f"{self.symbol}: Failed to load stock data - {e}")

    async def _load_balance_sheet_data(self):
        """Balance Sheet 데이터 조회"""

        query = """
        SELECT
            total_assets,
            total_liabilities,
            total_shareholder_equity,
            total_current_assets,
            total_current_liabilities,
            long_term_debt,
            short_term_debt,
            cash_and_cash_equivalents_at_carrying_value,
            short_long_term_debt_total
        FROM us_balance_sheet
        WHERE symbol = $1
        ORDER BY fiscal_date_ending DESC
        LIMIT 1
        """

        try:
            result = await self.db.execute_query(query, self.symbol)

            if result and result[0]:
                row = result[0]
                self.stock_data.update({
                    'total_assets': self._to_float(row['total_assets']),
                    'total_liabilities': self._to_float(row['total_liabilities']),
                    'shareholder_equity': self._to_float(row['total_shareholder_equity']),
                    'current_assets': self._to_float(row['total_current_assets']),
                    'current_liabilities': self._to_float(row['total_current_liabilities']),
                    'long_term_debt': self._to_float(row['long_term_debt']),
                    'short_term_debt': self._to_float(row['short_term_debt']),
                    'cash': self._to_float(row['cash_and_cash_equivalents_at_carrying_value']),
                    'total_debt': self._to_float(row['short_long_term_debt_total'])
                })

        except Exception as e:
            logger.warning(f"{self.symbol}: Balance sheet data not available - {e}")

    async def _load_cash_flow_data(self):
        """Cash Flow 데이터 조회"""

        query = """
        SELECT
            operating_cashflow,
            net_income,
            depreciation_depletion_and_amortization,
            capital_expenditures
        FROM us_cash_flow
        WHERE symbol = $1
        ORDER BY fiscal_date_ending DESC
        LIMIT 4
        """

        try:
            result = await self.db.execute_query(query, self.symbol)

            if result:
                # TTM 합계
                total_ocf = sum(self._to_float(r['operating_cashflow']) or 0 for r in result)
                total_ni = sum(self._to_float(r['net_income']) or 0 for r in result)
                total_da = sum(self._to_float(r['depreciation_depletion_and_amortization']) or 0 for r in result)
                total_capex = sum(abs(self._to_float(r['capital_expenditures']) or 0) for r in result)

                self.stock_data.update({
                    'operating_cashflow_ttm': total_ocf,
                    'net_income_ttm': total_ni,
                    'depreciation_ttm': total_da,
                    'capex_ttm': total_capex,
                    'fcf_ttm': total_ocf - total_capex
                })

        except Exception as e:
            logger.warning(f"{self.symbol}: Cash flow data not available - {e}")

    async def _load_historical_data(self):
        """과거 분기 데이터 로드 (Margin Trend용)"""

        query = """
        SELECT
            fiscal_date_ending,
            gross_profit,
            total_revenue,
            operating_income
        FROM us_income_statement
        WHERE symbol = $1
        ORDER BY fiscal_date_ending DESC
        LIMIT 8
        """

        try:
            result = await self.db.execute_query(query, self.symbol)

            if result:
                self.historical_data = []
                for row in result:
                    gross_profit = self._to_float(row['gross_profit'])
                    revenue = self._to_float(row['total_revenue'])
                    operating_income = self._to_float(row['operating_income'])

                    gross_margin = gross_profit / revenue if gross_profit and revenue else None
                    op_margin = operating_income / revenue if operating_income and revenue else None

                    self.historical_data.append({
                        'date': row['fiscal_date_ending'],
                        'gross_margin': gross_margin,
                        'operating_margin': op_margin,
                        'revenue': revenue
                    })

        except Exception as e:
            logger.warning(f"{self.symbol}: Historical data not available - {e}")

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
        Maps prefetched_data to stock_data and historical_data format.
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
        gross_profit = self._to_float(stock_basic.get('grossprofitttm'))
        revenue = self._to_float(stock_basic.get('revenuettm'))

        self.stock_data = {
            'gross_profit': gross_profit,
            'revenue': revenue,
            'operating_margin': self._to_float(stock_basic.get('operatingmarginttm')),
            'net_margin': self._to_float(stock_basic.get('profitmargin')),
            'roe': self._to_float(stock_basic.get('returnonequityttm')),
            'roa': self._to_float(stock_basic.get('returnonassetsttm')),
            'ebitda': self._to_float(stock_basic.get('ebitda')),
            'market_cap': self._to_float(stock_basic.get('market_cap')),
            'beta': self._to_float(stock_basic.get('beta')),
            'shares_outstanding': self._to_float(stock_basic.get('sharesoutstanding')),
            'book_value': self._to_float(stock_basic.get('bookvalue')),
            'eps': self._to_float(stock_basic.get('dilutedepsttm'))
        }

        # Calculate gross_margin
        if gross_profit and revenue:
            self.stock_data['gross_margin'] = gross_profit / revenue

        # Extract balance sheet data from financials
        balance = financials.get('balance', []) if financials else []
        if balance:
            bs = balance[0]
            self.stock_data.update({
                'total_assets': self._to_float(bs.get('total_assets')),
                'total_liabilities': self._to_float(bs.get('total_liabilities')),
                'shareholder_equity': self._to_float(bs.get('total_shareholder_equity')),
                'current_assets': self._to_float(bs.get('total_current_assets')),
                'current_liabilities': self._to_float(bs.get('total_current_liabilities')),
                'long_term_debt': self._to_float(bs.get('long_term_debt')),
                'short_term_debt': self._to_float(bs.get('short_term_debt')),
                'cash': self._to_float(bs.get('cash_and_cash_equivalents_at_carrying_value')),
                'total_debt': self._to_float(bs.get('short_long_term_debt_total'))
            })

        # Extract cash flow data from financials (4 quarters TTM)
        cashflow = financials.get('cashflow', []) if financials else []
        if cashflow and len(cashflow) >= 4:
            total_ocf = sum(self._to_float(r.get('operating_cashflow')) or 0 for r in cashflow[:4])
            total_ni = sum(self._to_float(r.get('net_income')) or 0 for r in cashflow[:4])
            total_da = sum(self._to_float(r.get('depreciation_depletion_and_amortization')) or 0 for r in cashflow[:4])
            total_capex = sum(abs(self._to_float(r.get('capital_expenditures')) or 0) for r in cashflow[:4])

            self.stock_data.update({
                'operating_cashflow_ttm': total_ocf,
                'net_income_ttm': total_ni,
                'depreciation_ttm': total_da,
                'capex_ttm': total_capex,
                'fcf_ttm': total_ocf - total_capex
            })

        # Extract historical data from income statements (for Margin Trend)
        income = financials.get('income', []) if financials else []
        self.historical_data = []
        for row in income[:8]:
            gross_profit_q = self._to_float(row.get('gross_profit'))
            revenue_q = self._to_float(row.get('total_revenue'))
            operating_income_q = self._to_float(row.get('operating_income'))

            gross_margin_q = gross_profit_q / revenue_q if gross_profit_q and revenue_q else None
            op_margin_q = operating_income_q / revenue_q if operating_income_q and revenue_q else None

            self.historical_data.append({
                'date': row.get('fiscal_date_ending'),
                'gross_margin': gross_margin_q,
                'operating_margin': op_margin_q,
                'revenue': revenue_q
            })

    def _calc_mq1_gross_margin(self) -> Dict:
        """MQ1: Gross Margin (섹터 대비)"""

        gross_margin = self.stock_data.get('gross_margin')
        sector_median = self.benchmarks.get('gross_margin_median')

        if gross_margin is None:
            return {'score': None, 'raw': None, 'reason': 'No gross margin data'}

        # 섹터별 절대 기준
        thresholds = SECTOR_GROSS_MARGIN_THRESHOLDS.get(
            self.sector, SECTOR_GROSS_MARGIN_THRESHOLDS['DEFAULT']
        )

        # 절대 기준 점수 (60%)
        if gross_margin >= thresholds['excellent']:
            abs_score = 90 + min(10, (gross_margin - thresholds['excellent']) / 0.1 * 10)
        elif gross_margin >= thresholds['good']:
            abs_score = 70 + (gross_margin - thresholds['good']) / (thresholds['excellent'] - thresholds['good']) * 20
        elif gross_margin >= thresholds['fair']:
            abs_score = 50 + (gross_margin - thresholds['fair']) / (thresholds['good'] - thresholds['fair']) * 20
        else:
            abs_score = max(20, 50 - (thresholds['fair'] - gross_margin) / thresholds['fair'] * 30)

        # 섹터 대비 점수 (40%)
        relative_score = 50  # 기본값
        if sector_median and sector_median > 0:
            ratio = gross_margin / sector_median
            if ratio >= 1.3:
                relative_score = 90
            elif ratio >= 1.1:
                relative_score = 70 + (ratio - 1.1) / 0.2 * 20
            elif ratio >= 0.9:
                relative_score = 50 + (ratio - 0.9) / 0.2 * 20
            elif ratio >= 0.7:
                relative_score = 30 + (ratio - 0.7) / 0.2 * 20
            else:
                relative_score = max(10, 30 - (0.7 - ratio) / 0.2 * 20)

        # 가중 평균
        score = abs_score * 0.6 + relative_score * 0.4

        return {
            'score': min(100, max(0, score)),
            'raw': gross_margin,
            'sector_median': sector_median,
            'thresholds': thresholds
        }

    def _calc_mq2_roic(self) -> Dict:
        """MQ2: ROIC (투하자본수익률)"""

        # ROIC 계산 시도
        # ROIC = NOPAT / Invested Capital
        # NOPAT ≈ Operating Income * (1 - Tax Rate) 또는 EBIT * (1 - 21%)
        # Invested Capital = Total Equity + Total Debt - Cash

        ebitda = self.stock_data.get('ebitda')
        equity = self.stock_data.get('shareholder_equity')
        total_debt = self.stock_data.get('total_debt')
        cash = self.stock_data.get('cash')

        # 대안: ROE 사용
        roe = self.stock_data.get('roe')

        roic = None

        if ebitda and equity:
            # 간이 ROIC 계산
            invested_capital = (equity or 0) + (total_debt or 0) - (cash or 0)
            if invested_capital > 0:
                # NOPAT ≈ EBITDA * 0.75 (세금 및 D&A 조정 근사)
                nopat_approx = ebitda * 0.75
                roic = nopat_approx / invested_capital

        if roic is None and roe:
            # ROE를 ROIC 대용으로 사용 (약간 할인)
            roic = roe * 0.85 if isinstance(roe, (int, float)) else self._to_float(roe) * 0.85

        if roic is None:
            return {'score': None, 'raw': None, 'reason': 'Cannot calculate ROIC'}

        # 점수 계산
        if roic >= ROIC_THRESHOLDS['excellent']:
            score = 90 + min(10, (roic - ROIC_THRESHOLDS['excellent']) / 0.1 * 10)
        elif roic >= ROIC_THRESHOLDS['good']:
            score = 70 + (roic - ROIC_THRESHOLDS['good']) / (ROIC_THRESHOLDS['excellent'] - ROIC_THRESHOLDS['good']) * 20
        elif roic >= ROIC_THRESHOLDS['fair']:
            score = 50 + (roic - ROIC_THRESHOLDS['fair']) / (ROIC_THRESHOLDS['good'] - ROIC_THRESHOLDS['fair']) * 20
        elif roic >= ROIC_THRESHOLDS['poor']:
            score = 30 + (roic - ROIC_THRESHOLDS['poor']) / (ROIC_THRESHOLDS['fair'] - ROIC_THRESHOLDS['poor']) * 20
        elif roic >= 0:
            score = 20 + roic / ROIC_THRESHOLDS['poor'] * 10
        else:
            score = max(5, 20 + roic * 100)  # 음수 ROIC 패널티

        return {
            'score': min(100, max(0, score)),
            'raw': roic,
            'roe': roe
        }

    def _calc_mq3_operating_leverage(self) -> Dict:
        """MQ3: Operating Leverage (확장성)"""

        # Operating Leverage = 매출 성장 대비 영업이익 성장
        # 높으면 규모의 경제 효과

        if len(self.historical_data) < 5:
            return {'score': None, 'raw': None, 'reason': 'Insufficient historical data'}

        # 최근 4분기 vs 이전 4분기 비교
        recent = self.historical_data[:4]
        older = self.historical_data[4:8] if len(self.historical_data) >= 8 else self.historical_data[4:]

        if len(older) < 2:
            return {'score': None, 'raw': None, 'reason': 'Insufficient comparison data'}

        # 매출 성장
        recent_rev = sum(r['revenue'] or 0 for r in recent)
        older_rev = sum(r['revenue'] or 0 for r in older)

        if older_rev <= 0:
            return {'score': None, 'raw': None, 'reason': 'Invalid revenue data'}

        rev_growth = (recent_rev - older_rev) / older_rev

        # Operating Margin 변화
        recent_op_margins = [r['operating_margin'] for r in recent if r['operating_margin'] is not None]
        older_op_margins = [r['operating_margin'] for r in older if r['operating_margin'] is not None]

        if not recent_op_margins or not older_op_margins:
            return {'score': None, 'raw': None, 'reason': 'No margin data'}

        recent_avg_margin = sum(recent_op_margins) / len(recent_op_margins)
        older_avg_margin = sum(older_op_margins) / len(older_op_margins)
        margin_change = recent_avg_margin - older_avg_margin

        # Operating Leverage 계산
        # 매출이 늘면서 마진도 개선 = 높은 레버리지
        if rev_growth > 0:
            # 매출 성장 중 마진 개선
            leverage_score = margin_change / rev_growth if rev_growth != 0 else 0
        else:
            # 매출 감소 중 마진 방어
            leverage_score = margin_change  # 마진 방어력

        # 점수 계산
        if rev_growth > 0.05 and margin_change > 0.02:
            # 매출 성장 + 마진 개선 = 최고
            score = 80 + min(20, margin_change * 200)
        elif rev_growth > 0 and margin_change >= 0:
            # 매출 성장 + 마진 유지
            score = 60 + rev_growth * 100 + margin_change * 100
        elif rev_growth > 0 and margin_change > -0.02:
            # 매출 성장, 소폭 마진 하락 (투자기)
            score = 50 + rev_growth * 50
        elif margin_change >= 0:
            # 매출 정체/하락, 마진 개선
            score = 45 + margin_change * 150
        else:
            # 마진 악화
            score = max(15, 45 + margin_change * 200)

        return {
            'score': min(100, max(0, score)),
            'raw': leverage_score,
            'revenue_growth': rev_growth,
            'margin_change': margin_change,
            'recent_margin': recent_avg_margin,
            'older_margin': older_avg_margin
        }

    def _calc_mq4_earnings_quality(self) -> Dict:
        """MQ4: Earnings Quality (OCF vs Net Income)"""

        ocf = self.stock_data.get('operating_cashflow_ttm')
        net_income = self.stock_data.get('net_income_ttm')

        if ocf is None or net_income is None:
            return {'score': None, 'raw': None, 'reason': 'No cash flow data'}

        if net_income <= 0:
            # 적자 기업
            if ocf > 0:
                # 적자지만 영업현금흐름은 양수 (감가상각 등)
                score = 45
                quality_ratio = None
            else:
                score = 25
                quality_ratio = None
        else:
            # OCF / Net Income 비율
            quality_ratio = ocf / net_income

            # 이상적: OCF > Net Income (1.0 이상)
            if quality_ratio >= 1.5:
                score = 95  # 매우 높은 현금 창출
            elif quality_ratio >= 1.2:
                score = 85 + (quality_ratio - 1.2) / 0.3 * 10
            elif quality_ratio >= 1.0:
                score = 70 + (quality_ratio - 1.0) / 0.2 * 15
            elif quality_ratio >= 0.8:
                score = 55 + (quality_ratio - 0.8) / 0.2 * 15
            elif quality_ratio >= 0.5:
                score = 35 + (quality_ratio - 0.5) / 0.3 * 20
            else:
                # 발생주의 이익 과다 (위험 신호)
                score = max(10, 35 - (0.5 - quality_ratio) * 50)

        return {
            'score': min(100, max(0, score)),
            'raw': quality_ratio,
            'operating_cashflow': ocf,
            'net_income': net_income
        }

    def _calc_mq5_balance_sheet(self) -> Dict:
        """MQ5: Balance Sheet 건전성"""

        total_debt = self.stock_data.get('total_debt') or 0
        cash = self.stock_data.get('cash') or 0
        equity = self.stock_data.get('shareholder_equity')
        current_assets = self.stock_data.get('current_assets')
        current_liabilities = self.stock_data.get('current_liabilities')
        ebitda = self.stock_data.get('ebitda')

        scores = []

        # 1. Net Debt / EBITDA (레버리지)
        if ebitda and ebitda > 0:
            net_debt = total_debt - cash
            net_debt_ebitda = net_debt / ebitda

            if net_debt_ebitda <= 0:  # 순현금
                debt_score = 95
            elif net_debt_ebitda <= 1:
                debt_score = 80 + (1 - net_debt_ebitda) * 15
            elif net_debt_ebitda <= 2:
                debt_score = 60 + (2 - net_debt_ebitda) * 20
            elif net_debt_ebitda <= 3:
                debt_score = 40 + (3 - net_debt_ebitda) * 20
            elif net_debt_ebitda <= 5:
                debt_score = 20 + (5 - net_debt_ebitda) / 2 * 20
            else:
                debt_score = max(5, 20 - (net_debt_ebitda - 5) * 3)

            scores.append(('debt_score', debt_score, net_debt_ebitda))

        # 2. Current Ratio (유동비율)
        if current_assets and current_liabilities and current_liabilities > 0:
            current_ratio = current_assets / current_liabilities

            if current_ratio >= 2.5:
                liquidity_score = 90
            elif current_ratio >= 2.0:
                liquidity_score = 75 + (current_ratio - 2.0) / 0.5 * 15
            elif current_ratio >= 1.5:
                liquidity_score = 60 + (current_ratio - 1.5) / 0.5 * 15
            elif current_ratio >= 1.0:
                liquidity_score = 40 + (current_ratio - 1.0) / 0.5 * 20
            else:
                liquidity_score = max(10, current_ratio * 40)

            scores.append(('liquidity_score', liquidity_score, current_ratio))

        # 3. Debt / Equity (자본 구조)
        if equity and equity > 0:
            de_ratio = total_debt / equity

            if de_ratio <= 0.2:
                de_score = 90
            elif de_ratio <= 0.5:
                de_score = 70 + (0.5 - de_ratio) / 0.3 * 20
            elif de_ratio <= 1.0:
                de_score = 50 + (1.0 - de_ratio) / 0.5 * 20
            elif de_ratio <= 2.0:
                de_score = 30 + (2.0 - de_ratio) / 1.0 * 20
            else:
                de_score = max(10, 30 - (de_ratio - 2.0) * 10)

            scores.append(('de_score', de_score, de_ratio))

        if not scores:
            return {'score': None, 'raw': None, 'reason': 'No balance sheet data'}

        # 가중 평균
        final_score = sum(s[1] for s in scores) / len(scores)

        return {
            'score': min(100, max(0, final_score)),
            'raw': {name: value for name, score, value in scores},
            'components': {name: score for name, score, value in scores}
        }

    def _calc_mq6_margin_trend(self) -> Dict:
        """MQ6: Margin Trend (개선 vs 악화)"""

        if len(self.historical_data) < 4:
            return {'score': None, 'raw': None, 'reason': 'Insufficient historical data'}

        # Gross Margin 추세
        gross_margins = [r['gross_margin'] for r in self.historical_data if r['gross_margin'] is not None]
        op_margins = [r['operating_margin'] for r in self.historical_data if r['operating_margin'] is not None]

        if len(gross_margins) < 3 and len(op_margins) < 3:
            return {'score': None, 'raw': None, 'reason': 'No margin trend data'}

        scores = []

        # Gross Margin 추세 분석
        if len(gross_margins) >= 3:
            gm_trend = self._calculate_trend(gross_margins)
            # 최근값 기준 개선/악화 판단
            gm_recent = gross_margins[0]
            gm_older = gross_margins[-1]
            gm_change = gm_recent - gm_older

            if gm_trend > 0 and gm_change > 0.02:
                gm_score = 80 + min(20, gm_change * 200)
            elif gm_trend > 0 or gm_change > 0:
                gm_score = 60 + min(20, max(gm_change, 0) * 150)
            elif gm_change > -0.02:
                gm_score = 50 + gm_change * 100
            else:
                gm_score = max(20, 50 + gm_change * 150)

            scores.append(gm_score)

        # Operating Margin 추세 분석
        if len(op_margins) >= 3:
            op_trend = self._calculate_trend(op_margins)
            op_recent = op_margins[0]
            op_older = op_margins[-1]
            op_change = op_recent - op_older

            if op_trend > 0 and op_change > 0.02:
                op_score = 80 + min(20, op_change * 200)
            elif op_trend > 0 or op_change > 0:
                op_score = 60 + min(20, max(op_change, 0) * 150)
            elif op_change > -0.02:
                op_score = 50 + op_change * 100
            else:
                op_score = max(20, 50 + op_change * 150)

            scores.append(op_score)

        if not scores:
            return {'score': None, 'raw': None, 'reason': 'Cannot calculate trend'}

        final_score = sum(scores) / len(scores)

        return {
            'score': min(100, max(0, final_score)),
            'raw': {
                'gross_margin_change': gm_change if len(gross_margins) >= 3 else None,
                'operating_margin_change': op_change if len(op_margins) >= 3 else None
            },
            'gross_margins': gross_margins[:4] if gross_margins else None,
            'op_margins': op_margins[:4] if op_margins else None
        }

    def _calculate_trend(self, values: List[float]) -> float:
        """간단한 선형 추세 계산 (양수 = 상승 추세)"""
        if len(values) < 2:
            return 0

        # 시간 역순이므로 뒤집어서 계산
        values = values[::-1]
        n = len(values)
        x_mean = (n - 1) / 2
        y_mean = sum(values) / n

        numerator = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
        denominator = sum((i - x_mean) ** 2 for i in range(n))

        return numerator / denominator if denominator > 0 else 0

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
    # Healthcare 특화 전략 (HC1~HC4)
    # ====================================================================

    async def _load_healthcare_data(self):
        """Healthcare 특화 데이터 로드 (Analyst, EPS Estimates, R&D, Cash)"""

        # 1. Analyst Rating 데이터 (us_stock_basic)
        analyst_query = """
        SELECT
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
            eps_estimate_revision_down_trailing_7_days
        FROM us_earnings_estimates
        WHERE symbol = $1 AND horizon = 'next fiscal quarter'
        ORDER BY estimate_date DESC
        LIMIT 1
        """

        # 3. R&D 데이터 (us_income_statement)
        rd_query = """
        SELECT
            research_and_development,
            total_revenue
        FROM us_income_statement
        WHERE symbol = $1 AND research_and_development IS NOT NULL
        ORDER BY fiscal_date_ending DESC
        LIMIT 4
        """

        # 4. Cash Runway 데이터 (us_balance_sheet + us_cash_flow)
        # cash는 이미 _load_balance_sheet_data에서 로드됨
        # operating_cashflow는 이미 _load_cash_flow_data에서 로드됨

        try:
            # Analyst 데이터
            analyst_result = await self.db.execute_query(analyst_query, self.symbol)
            if analyst_result and analyst_result[0]:
                row = analyst_result[0]
                self.stock_data['analyst'] = {
                    'target_price': self._to_float(row['analysttargetprice']),
                    'strong_buy': row['analystratingstrongbuy'] or 0,
                    'buy': row['analystratingbuy'] or 0,
                    'hold': row['analystratinghold'] or 0,
                    'sell': row['analystratingsell'] or 0,
                    'strong_sell': row['analystratingstrongsell'] or 0
                }

            # EPS Estimates 데이터
            eps_result = await self.db.execute_query(eps_query, self.symbol)
            if eps_result and eps_result[0]:
                row = eps_result[0]
                self.stock_data['eps_estimates'] = {
                    'current': self._to_float(row['eps_estimate_average']),
                    '7d_ago': self._to_float(row['eps_estimate_average_7_days_ago']),
                    '30d_ago': self._to_float(row['eps_estimate_average_30_days_ago']),
                    'revision_up_7d': row['eps_estimate_revision_up_trailing_7_days'] or 0,
                    'revision_down_7d': row['eps_estimate_revision_down_trailing_7_days'] or 0
                }

            # R&D 데이터 (TTM 합계)
            rd_result = await self.db.execute_query(rd_query, self.symbol)
            if rd_result:
                rd_total = sum(self._to_float(r['research_and_development']) or 0 for r in rd_result)
                rev_total = sum(self._to_float(r['total_revenue']) or 0 for r in rd_result)
                self.stock_data['rd'] = {
                    'rd_ttm': rd_total,
                    'revenue_ttm': rev_total,
                    'rd_intensity': rd_total / rev_total if rev_total > 0 else None
                }

        except Exception as e:
            logger.warning(f"{self.symbol}: Healthcare data load failed - {e}")

    def _calc_hc1_analyst_consensus(self) -> Dict:
        """HC1: Analyst Consensus Score

        공식: (StrongBuy×2 + Buy×1.5 + Hold×1 + Sell×0.5 + StrongSell×0) / Total Ratings
        점수 범위: 0-100

        해석:
        - 2.0 = 모두 Strong Buy → 100점
        - 1.5 = 모두 Buy → 85점
        - 1.0 = 모두 Hold → 50점
        - 0.5 = 모두 Sell → 25점
        - 0.0 = 모두 Strong Sell → 0점
        """
        analyst = self.stock_data.get('analyst')
        if not analyst:
            return {'score': None, 'raw': None, 'reason': 'No analyst data'}

        strong_buy = analyst['strong_buy']
        buy = analyst['buy']
        hold = analyst['hold']
        sell = analyst['sell']
        strong_sell = analyst['strong_sell']

        total = strong_buy + buy + hold + sell + strong_sell
        if total == 0:
            return {'score': None, 'raw': None, 'reason': 'No analyst ratings'}

        # 가중 평균 계산 (0~2 범위)
        weighted_sum = (strong_buy * 2.0 + buy * 1.5 + hold * 1.0 +
                        sell * 0.5 + strong_sell * 0.0)
        consensus = weighted_sum / total

        # 점수 변환 (0~2 → 0~100)
        score = consensus / 2.0 * 100

        return {
            'score': min(100, max(0, score)),
            'raw': consensus,
            'strong_buy': strong_buy,
            'buy': buy,
            'hold': hold,
            'sell': sell,
            'strong_sell': strong_sell,
            'total_ratings': total
        }

    def _calc_hc2_eps_revision_momentum(self) -> Dict:
        """HC2: EPS Revision Momentum

        공식: (EPS_now - EPS_30d_ago) / |EPS_30d_ago| × 100
        + 추가 보정: 7일간 상향/하향 수정 건수

        점수 범위: 0-100
        """
        eps_data = self.stock_data.get('eps_estimates')
        if not eps_data:
            return {'score': None, 'raw': None, 'reason': 'No EPS estimates data'}

        eps_current = eps_data['current']
        eps_30d_ago = eps_data['30d_ago']
        revision_up = eps_data['revision_up_7d']
        revision_down = eps_data['revision_down_7d']

        # EPS 변화율 계산
        if eps_current is None or eps_30d_ago is None or eps_30d_ago == 0:
            # Revision 데이터만으로 점수 계산
            if revision_up == 0 and revision_down == 0:
                return {'score': 50.0, 'raw': None, 'reason': 'No revision data', 'source': 'neutral'}

            # Net revision
            # revision 값이 매우 큰 경우 정규화 (천 단위 이상이면 /1000)
            if revision_up > 1000:
                revision_up = revision_up / 1000
            if revision_down > 1000:
                revision_down = revision_down / 1000

            net_revision = revision_up - revision_down

            if net_revision >= 3:
                score = 85
            elif net_revision >= 1:
                score = 65 + net_revision * 10
            elif net_revision >= 0:
                score = 50 + net_revision * 15
            elif net_revision >= -1:
                score = 35 + (net_revision + 1) * 15
            else:
                score = max(10, 35 + net_revision * 10)

            return {
                'score': min(100, max(0, score)),
                'raw': net_revision,
                'revision_up': revision_up,
                'revision_down': revision_down,
                'source': 'revision_only'
            }

        # EPS 변화율 계산
        eps_change = (eps_current - eps_30d_ago) / abs(eps_30d_ago)

        # 변화율 기반 점수 (70% 비중)
        if eps_change >= 0.10:
            change_score = 90 + min(10, (eps_change - 0.10) / 0.05 * 10)
        elif eps_change >= 0.05:
            change_score = 75 + (eps_change - 0.05) / 0.05 * 15
        elif eps_change >= 0.02:
            change_score = 60 + (eps_change - 0.02) / 0.03 * 15
        elif eps_change >= 0:
            change_score = 50 + eps_change / 0.02 * 10
        elif eps_change >= -0.02:
            change_score = 40 + (eps_change + 0.02) / 0.02 * 10
        elif eps_change >= -0.05:
            change_score = 25 + (eps_change + 0.05) / 0.03 * 15
        else:
            change_score = max(10, 25 + (eps_change + 0.05) * 150)

        # Revision 방향 보정 (30% 비중)
        if revision_up > 1000:
            revision_up = revision_up / 1000
        if revision_down > 1000:
            revision_down = revision_down / 1000

        net_revision = revision_up - revision_down
        if net_revision >= 2:
            revision_score = 80
        elif net_revision >= 0:
            revision_score = 50 + net_revision * 15
        else:
            revision_score = max(20, 50 + net_revision * 15)

        # 가중 평균
        score = change_score * 0.7 + revision_score * 0.3

        return {
            'score': min(100, max(0, score)),
            'raw': eps_change,
            'eps_current': eps_current,
            'eps_30d_ago': eps_30d_ago,
            'revision_up': revision_up,
            'revision_down': revision_down
        }

    def _calc_hc3_rd_intensity(self) -> Dict:
        """HC3: R&D Intensity Score

        공식: R&D / Revenue (TTM)

        Healthcare 기준:
        - >30%: 매우 높은 R&D 투자 (Biotech 수준) → 높은 점수
        - 20-30%: 높은 R&D 투자 (대형 제약사 수준)
        - 10-20%: 보통 수준
        - <10%: 낮은 R&D 투자 (비제약 Healthcare)

        점수 범위: 0-100
        """
        rd_data = self.stock_data.get('rd')
        if not rd_data or rd_data['rd_intensity'] is None:
            return {'score': None, 'raw': None, 'reason': 'No R&D data'}

        rd_intensity = rd_data['rd_intensity']

        # Healthcare R&D Intensity 기준 점수
        if rd_intensity >= 0.50:  # 50%+ (고집중 Biotech)
            score = 95
        elif rd_intensity >= 0.35:  # 35-50%
            score = 85 + (rd_intensity - 0.35) / 0.15 * 10
        elif rd_intensity >= 0.25:  # 25-35% (대형 제약사 수준)
            score = 75 + (rd_intensity - 0.25) / 0.10 * 10
        elif rd_intensity >= 0.15:  # 15-25%
            score = 60 + (rd_intensity - 0.15) / 0.10 * 15
        elif rd_intensity >= 0.10:  # 10-15%
            score = 50 + (rd_intensity - 0.10) / 0.05 * 10
        elif rd_intensity >= 0.05:  # 5-10%
            score = 35 + (rd_intensity - 0.05) / 0.05 * 15
        else:  # <5%
            score = max(20, rd_intensity / 0.05 * 35)

        return {
            'score': min(100, max(0, score)),
            'raw': rd_intensity,
            'rd_ttm': rd_data['rd_ttm'],
            'revenue_ttm': rd_data['revenue_ttm']
        }

    def _calc_hc4_cash_runway(self) -> Dict:
        """HC4: Cash Runway Score

        공식: Cash / |Quarterly Burn Rate| (분기 수)

        적자 기업 (Operating Cashflow < 0):
        - 12분기+ (3년+): 안전 → 높은 점수
        - 8-12분기 (2-3년): 양호
        - 4-8분기 (1-2년): 주의
        - <4분기 (1년 미만): 위험

        흑자 기업 (Operating Cashflow >= 0):
        - Cash Runway 불필요, 자동 높은 점수

        점수 범위: 0-100
        """
        cash = self.stock_data.get('cash') or 0
        ocf_ttm = self.stock_data.get('operating_cashflow_ttm')

        if ocf_ttm is None:
            return {'score': None, 'raw': None, 'reason': 'No cashflow data'}

        # 흑자 기업 (OCF >= 0)
        if ocf_ttm >= 0:
            # Cash 보유량에 따른 추가 점수
            if cash > 0:
                # Cash / Revenue 비율로 추가 평가
                revenue = self.stock_data.get('revenue') or self.stock_data.get('rd', {}).get('revenue_ttm') or 1
                cash_ratio = cash / revenue if revenue > 0 else 0

                if cash_ratio >= 0.3:  # 30%+ Cash
                    score = 95
                elif cash_ratio >= 0.15:
                    score = 85 + (cash_ratio - 0.15) / 0.15 * 10
                else:
                    score = 75 + cash_ratio / 0.15 * 10
            else:
                score = 70

            return {
                'score': min(100, max(0, score)),
                'raw': 'positive_cashflow',
                'cash': cash,
                'operating_cashflow_ttm': ocf_ttm,
                'status': 'profitable'
            }

        # 적자 기업 (OCF < 0)
        quarterly_burn = abs(ocf_ttm) / 4  # TTM을 분기로 환산
        if quarterly_burn <= 0:
            return {'score': 50.0, 'raw': None, 'reason': 'Invalid burn rate'}

        # Cash Runway (분기 수)
        runway_quarters = cash / quarterly_burn

        # Runway 기반 점수
        if runway_quarters >= 16:  # 4년+
            score = 90 + min(10, (runway_quarters - 16) / 8 * 10)
        elif runway_quarters >= 12:  # 3-4년
            score = 80 + (runway_quarters - 12) / 4 * 10
        elif runway_quarters >= 8:  # 2-3년
            score = 65 + (runway_quarters - 8) / 4 * 15
        elif runway_quarters >= 4:  # 1-2년
            score = 45 + (runway_quarters - 4) / 4 * 20
        elif runway_quarters >= 2:  # 6개월-1년
            score = 25 + (runway_quarters - 2) / 2 * 20
        else:  # 6개월 미만
            score = max(5, runway_quarters / 2 * 25)

        return {
            'score': min(100, max(0, score)),
            'raw': runway_quarters,
            'cash': cash,
            'quarterly_burn': quarterly_burn,
            'operating_cashflow_ttm': ocf_ttm,
            'status': 'cash_burning'
        }


# ========================================================================
# Convenience Function
# ========================================================================

async def calculate_quality_score(symbol: str, db_manager, sector_benchmarks: Dict,
                                   analysis_date: Optional[date] = None,
                                   volatility_engine: Optional['USVolatilityAdjustment'] = None,
                                   prefetched_data: Optional[Dict] = None) -> Dict:
    """Quality Factor 점수 계산 (편의 함수)

    Args:
        volatility_engine: Phase 3.4.2 - Shared volatility engine (optional)
        prefetched_data: Phase 3.4.3 - Prefetched data from USDataPrefetcher (optional)
    """

    calculator = USQualityFactorV2(symbol, db_manager, sector_benchmarks, analysis_date,
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
        print("Quality Factor v2.0 Test")
        print("="*60)

        for symbol in test_symbols:
            result = await calculate_quality_score(symbol, db, benchmarks)
            print(f"\n{symbol}: Quality Score = {result['quality_score']}")

            for strat_id, strat_data in result['strategies'].items():
                if strat_data['score'] is not None:
                    raw = strat_data.get('raw', 'N/A')
                    if isinstance(raw, dict):
                        raw = 'complex'
                    print(f"  {strat_id}: {strat_data['score']:.1f} (raw: {raw})")

        await db.close()

    asyncio.run(test())
