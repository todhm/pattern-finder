"""
Korean Stock Quality Factor System
Implements 15 quality factor strategies to score stocks on a 100-point scale
File: kr_quality_factor.py
"""

import os
import logging
import asyncio
from dotenv import load_dotenv
from datetime import datetime, timedelta
import statistics

# Import market classifier
try:
    from market_classifier import MarketClassifier
except ImportError:
    from kr.market_classifier import MarketClassifier

# Load environment variables
load_dotenv()

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ========================================================================
# Market State-based Quality Strategy Weights (19 market states × 17 strategies)
# ========================================================================

QUALITY_STRATEGY_WEIGHTS = {
    # Large Cap Group (6)
    # Q18-Q21: Phase 3.8 Sustainable Growth Paradigm
    'KOSPI대형-확장과열-공격형': {
        'Q1': 0.8, 'Q2': 1.0, 'Q3': 0.5, 'Q4': 0.6, 'Q5': 1.2, 'Q6': 1.0,
        'Q7': 0.7, 'Q8': 0.5, 'Q9': 1.0, 'Q10': 0.8, 'Q11': 0.9, 'Q12': 1.0,
        'Q13': 0.7, 'Q14': 0.5, 'Q15': 1.0, 'Q16': 0.6, 'Q17': 0.9,
        'Q18': 1.3, 'Q19': 0.8, 'Q20': 0.9, 'Q21': 1.0  # Phase 3.8: SGR 중시
    },
    'KOSPI대형-확장중립-성장형': {
        'Q1': 1.5, 'Q2': 1.5, 'Q3': 1.2, 'Q4': 1.0, 'Q5': 1.4, 'Q6': 1.3,
        'Q7': 1.3, 'Q8': 1.2, 'Q9': 1.4, 'Q10': 1.3, 'Q11': 1.4, 'Q12': 1.3,
        'Q13': 1.2, 'Q14': 1.1, 'Q15': 1.4, 'Q16': 1.2, 'Q17': 1.3,
        'Q18': 1.6, 'Q19': 1.3, 'Q20': 1.2, 'Q21': 1.4  # Phase 3.8
    },
    'KOSPI대형-둔화공포-방어형': {
        'Q1': 2.0, 'Q2': 1.8, 'Q3': 2.0, 'Q4': 1.8, 'Q5': 1.5, 'Q6': 1.7,
        'Q7': 2.0, 'Q8': 2.0, 'Q9': 1.8, 'Q10': 1.8, 'Q11': 1.9, 'Q12': 1.6,
        'Q13': 1.5, 'Q14': 2.0, 'Q15': 1.9, 'Q16': 2.0, 'Q17': 1.8,
        'Q18': 1.5, 'Q19': 2.0, 'Q20': 1.4, 'Q21': 1.8  # Phase 3.8: Accrual 중시
    },
    'KOSPI대형-침체패닉-초방어형': {
        'Q1': 2.0, 'Q2': 2.0, 'Q3': 2.0, 'Q4': 2.0, 'Q5': 1.3, 'Q6': 1.8,
        'Q7': 2.0, 'Q8': 2.0, 'Q9': 2.0, 'Q10': 2.0, 'Q11': 2.0, 'Q12': 1.5,
        'Q13': 1.8, 'Q14': 2.0, 'Q15': 2.0, 'Q16': 2.0, 'Q17': 2.0,
        'Q18': 1.4, 'Q19': 2.0, 'Q20': 1.5, 'Q21': 2.0  # Phase 3.8
    },
    'KOSPI대형-회복탐욕-밸류형': {
        'Q1': 1.6, 'Q2': 1.5, 'Q3': 1.5, 'Q4': 1.3, 'Q5': 1.5, 'Q6': 1.5,
        'Q7': 1.6, 'Q8': 1.5, 'Q9': 1.5, 'Q10': 1.5, 'Q11': 1.6, 'Q12': 1.4,
        'Q13': 1.4, 'Q14': 1.4, 'Q15': 1.6, 'Q16': 1.5, 'Q17': 1.5,
        'Q18': 1.5, 'Q19': 1.5, 'Q20': 1.3, 'Q21': 1.5  # Phase 3.8
    },
    'KOSPI대형-중립안정-균형형': {
        'Q1': 1.5, 'Q2': 1.4, 'Q3': 1.4, 'Q4': 1.2, 'Q5': 1.3, 'Q6': 1.3,
        'Q7': 1.5, 'Q8': 1.5, 'Q9': 1.4, 'Q10': 1.4, 'Q11': 1.5, 'Q12': 1.3,
        'Q13': 1.3, 'Q14': 1.3, 'Q15': 1.5, 'Q16': 1.4, 'Q17': 1.4,
        'Q18': 1.5, 'Q19': 1.4, 'Q20': 1.2, 'Q21': 1.4  # Phase 3.8
    },

    # Mid Cap Group (6)
    'KOSPI중형-확장과열-모멘텀형': {
        'Q1': 0.7, 'Q2': 0.9, 'Q3': 0.6, 'Q4': 0.7, 'Q5': 1.3, 'Q6': 1.1,
        'Q7': 0.5, 'Q8': 0.6, 'Q9': 0.9, 'Q10': 0.7, 'Q11': 0.8, 'Q12': 1.2,
        'Q13': 0.8, 'Q14': 0.6, 'Q15': 0.9, 'Q16': 0.7, 'Q17': 1.0,
        'Q18': 1.2, 'Q19': 0.7, 'Q20': 0.8, 'Q21': 0.9  # Phase 3.8
    },
    'KOSPI중형-회복중립-성장형': {
        'Q1': 1.3, 'Q2': 1.4, 'Q3': 1.1, 'Q4': 1.0, 'Q5': 1.5, 'Q6': 1.4,
        'Q7': 1.0, 'Q8': 1.1, 'Q9': 1.3, 'Q10': 1.2, 'Q11': 1.3, 'Q12': 1.4,
        'Q13': 1.1, 'Q14': 1.0, 'Q15': 1.3, 'Q16': 1.1, 'Q17': 1.3,
        'Q18': 1.5, 'Q19': 1.2, 'Q20': 1.1, 'Q21': 1.3  # Phase 3.8
    },
    'KOSPI중형-둔화공포-혼조형': {
        'Q1': 1.6, 'Q2': 1.5, 'Q3': 1.6, 'Q4': 1.5, 'Q5': 1.2, 'Q6': 1.3,
        'Q7': 1.5, 'Q8': 1.6, 'Q9': 1.5, 'Q10': 1.5, 'Q11': 1.6, 'Q12': 1.2,
        'Q13': 1.3, 'Q14': 1.6, 'Q15': 1.5, 'Q16': 1.6, 'Q17': 1.4,
        'Q18': 1.3, 'Q19': 1.6, 'Q20': 1.3, 'Q21': 1.5  # Phase 3.8
    },
    'KOSDAQ중형-확장탐욕-공격성장형': {
        'Q1': 0.5, 'Q2': 0.7, 'Q3': 0.4, 'Q4': 0.5, 'Q5': 1.0, 'Q6': 0.8,
        'Q7': 0.3, 'Q8': 0.4, 'Q9': 0.7, 'Q10': 0.5, 'Q11': 0.6, 'Q12': 0.9,
        'Q13': 0.6, 'Q14': 0.4, 'Q15': 0.7, 'Q16': 0.5, 'Q17': 0.8,
        'Q18': 1.4, 'Q19': 0.6, 'Q20': 0.7, 'Q21': 0.8  # Phase 3.8: SGR 최고 가중치
    },
    'KOSDAQ중형-회복중립-성장테마형': {
        'Q1': 1.0, 'Q2': 1.2, 'Q3': 0.8, 'Q4': 0.8, 'Q5': 1.3, 'Q6': 1.2,
        'Q7': 0.6, 'Q8': 0.8, 'Q9': 1.1, 'Q10': 0.9, 'Q11': 1.0, 'Q12': 1.2,
        'Q13': 0.9, 'Q14': 0.7, 'Q15': 1.1, 'Q16': 0.8, 'Q17': 1.1,
        'Q18': 1.4, 'Q19': 1.0, 'Q20': 0.9, 'Q21': 1.1  # Phase 3.8
    },
    'KOSDAQ중형-침체공포-역발상형': {
        'Q1': 1.5, 'Q2': 1.4, 'Q3': 1.5, 'Q4': 1.6, 'Q5': 1.0, 'Q6': 1.2,
        'Q7': 1.2, 'Q8': 1.5, 'Q9': 1.4, 'Q10': 1.4, 'Q11': 1.5, 'Q12': 1.0,
        'Q13': 1.2, 'Q14': 1.5, 'Q15': 1.4, 'Q16': 1.5, 'Q17': 1.3,
        'Q18': 1.2, 'Q19': 1.5, 'Q20': 1.2, 'Q21': 1.4  # Phase 3.8
    },

    # Small Cap Group (4) - 소형주는 성장 잠재력 중시
    'KOSDAQ소형-핫섹터-초고위험형': {
        'Q1': 0.3, 'Q2': 0.5, 'Q3': 0.2, 'Q4': 0.8, 'Q5': 0.8, 'Q6': 0.6,
        'Q7': 0.1, 'Q8': 0.2, 'Q9': 0.5, 'Q10': 0.3, 'Q11': 0.4, 'Q12': 0.7,
        'Q13': 0.4, 'Q14': 0.2, 'Q15': 0.5, 'Q16': 0.3, 'Q17': 0.6,
        'Q18': 1.5, 'Q19': 0.5, 'Q20': 0.6, 'Q21': 0.7  # Phase 3.8: SGR 최고
    },
    'KOSDAQ소형-성장테마-고위험형': {
        'Q1': 0.5, 'Q2': 0.7, 'Q3': 0.3, 'Q4': 0.9, 'Q5': 1.0, 'Q6': 0.8,
        'Q7': 0.2, 'Q8': 0.3, 'Q9': 0.7, 'Q10': 0.5, 'Q11': 0.6, 'Q12': 0.9,
        'Q13': 0.6, 'Q14': 0.3, 'Q15': 0.7, 'Q16': 0.4, 'Q17': 0.8,
        'Q18': 1.4, 'Q19': 0.6, 'Q20': 0.7, 'Q21': 0.8  # Phase 3.8
    },
    'KOSDAQ소형-침체-극단역발상형': {
        'Q1': 1.2, 'Q2': 1.0, 'Q3': 1.3, 'Q4': 1.8, 'Q5': 0.8, 'Q6': 1.0,
        'Q7': 0.8, 'Q8': 1.2, 'Q9': 1.0, 'Q10': 1.0, 'Q11': 1.2, 'Q12': 0.8,
        'Q13': 1.0, 'Q14': 1.3, 'Q15': 1.0, 'Q16': 1.3, 'Q17': 1.1,
        'Q18': 1.1, 'Q19': 1.3, 'Q20': 1.0, 'Q21': 1.2  # Phase 3.8
    },
    'KOSDAQ소형-회복-모멘텀형': {
        'Q1': 0.8, 'Q2': 0.9, 'Q3': 0.6, 'Q4': 1.0, 'Q5': 1.2, 'Q6': 1.0,
        'Q7': 0.4, 'Q8': 0.6, 'Q9': 0.9, 'Q10': 0.7, 'Q11': 0.8, 'Q12': 1.1,
        'Q13': 0.8, 'Q14': 0.5, 'Q15': 0.9, 'Q16': 0.6, 'Q17': 1.0,
        'Q18': 1.3, 'Q19': 0.8, 'Q20': 0.8, 'Q21': 0.9  # Phase 3.8
    },

    # Special Situation Group (2)
    '전시장-극저유동성-고위험형': {
        'Q1': 1.8, 'Q2': 1.6, 'Q3': 2.0, 'Q4': 2.0, 'Q5': 0.8, 'Q6': 1.5,
        'Q7': 1.5, 'Q8': 1.8, 'Q9': 1.6, 'Q10': 1.7, 'Q11': 1.8, 'Q12': 0.8,
        'Q13': 1.3, 'Q14': 2.0, 'Q15': 1.7, 'Q16': 2.0, 'Q17': 1.6,
        'Q18': 1.2, 'Q19': 1.8, 'Q20': 1.3, 'Q21': 1.6  # Phase 3.8
    },
    '테마특화-모멘텀폭발형': {
        'Q1': 0.4, 'Q2': 0.6, 'Q3': 0.3, 'Q4': 0.6, 'Q5': 1.0, 'Q6': 0.8,
        'Q7': 0.2, 'Q8': 0.3, 'Q9': 0.6, 'Q10': 0.4, 'Q11': 0.5, 'Q12': 0.9,
        'Q13': 0.5, 'Q14': 0.3, 'Q15': 0.6, 'Q16': 0.4, 'Q17': 0.7,
        'Q18': 1.3, 'Q19': 0.5, 'Q20': 0.6, 'Q21': 0.7  # Phase 3.8
    },

    # Others (fallback)
    '기타': {
        'Q1': 1.0, 'Q2': 1.0, 'Q3': 1.0, 'Q4': 1.0, 'Q5': 1.0, 'Q6': 1.0,
        'Q7': 1.0, 'Q8': 1.0, 'Q9': 1.0, 'Q10': 1.0, 'Q11': 1.0, 'Q12': 1.0,
        'Q13': 1.0, 'Q14': 1.0, 'Q15': 1.0, 'Q16': 1.0, 'Q17': 1.0,
        'Q18': 1.0, 'Q19': 1.0, 'Q20': 1.0, 'Q21': 1.0  # Phase 3.8
    }
}


class QualityFactorCalculator:
    """Calculate quality factor scores using 17 different strategies"""

    def __init__(self, symbol, db_manager, market_state=None, analysis_date=None):
        self.symbol = symbol
        self.db_manager = db_manager
        self.market_state = market_state
        self.analysis_date = analysis_date
        self.strategies_scores = {}

    async def execute_query(self, query, *params):
        """Execute SQL query and return results"""
        try:
            result = await self.db_manager.execute_query(query, *params)
            return result
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            logger.error(f"Query: {query}")
            raise

    # ========================================================================
    # Q1. ROE Consistency Strategy
    # ========================================================================

    async def calculate_q1(self):
        """
        Q1. ROE Consistency Strategy (Modified: 2-year comparison)
        Description: Companies maintaining consistently high ROE
        ROE = (Net Income / Total Equity) × 100 for 2 years
        Average ROE + Low volatility = High score
        """
        query = """
        SELECT
            MAX(CASE WHEN sj_div = 'IS' AND thstrm_amount > 100000000
                THEN thstrm_amount END) as net_income_t0,
            MAX(CASE WHEN sj_div = 'IS' AND frmtrm_amount > 100000000
                THEN frmtrm_amount END) as net_income_t1,
            MAX(CASE WHEN sj_div = 'BS'
                THEN thstrm_amount END) as total_t0,
            MAX(CASE WHEN sj_div = 'BS'
                THEN frmtrm_amount END) as total_t1
        FROM kr_financial_position
        WHERE symbol = $1
            AND rcept_dt <= COALESCE($2::date, CURRENT_DATE)
        GROUP BY bsns_year
        ORDER BY bsns_year DESC, MAX(rcept_dt) DESC
        LIMIT 1
        """

        result = await self.execute_query(query, self.symbol, self.analysis_date)

        if not result or not result[0]['net_income_t0']:
            return None

        row = result[0]

        # Calculate ROE for 2 years
        roes = []

        # Current year ROE
        if row['net_income_t0'] and row['total_t0'] and float(row['total_t0']) > 0:
            roe_t0 = (float(row['net_income_t0']) / float(row['total_t0'])) * 100
            roes.append(roe_t0)

        # Previous year ROE
        if row['net_income_t1'] and row['total_t1'] and float(row['total_t1']) > 0:
            roe_t1 = (float(row['net_income_t1']) / float(row['total_t1'])) * 100
            roes.append(roe_t1)

        if len(roes) < 1:
            return None

        # Calculate average ROE
        avg_roe = statistics.mean(roes)
        avg_score = min(100, max(0, avg_roe * 5))  # ROE 20% = 100점

        # Stability score (if 2 years data available)
        if len(roes) == 2:
            roe_diff = abs(roes[0] - roes[1])
            stability_score = max(0, 100 - roe_diff * 5)  # 차이 작을수록 높은 점수
            final_score = (avg_score * 0.7) + (stability_score * 0.3)
        else:
            # Only 1 year data - just use ROE level
            final_score = avg_score * 0.7

        return final_score

    # ========================================================================
    # Q2. Operating Margin Excellence Strategy
    # ========================================================================

    async def calculate_q2(self):
        """
        Q2. Operating Margin Excellence Strategy
        Description: High profitability relative to industry peers
        Operating Margin = (Operating Income / Sales) × 100
        Relative to industry average
        """
        # Get sales (largest CIS amount)
        sales_query = """
        SELECT thstrm_amount as sales
        FROM kr_financial_position
        WHERE symbol = $1
            AND sj_div = 'IS'
            AND thstrm_amount > 0
            AND rcept_dt <= COALESCE($2::date, CURRENT_DATE)
        ORDER BY
            bsns_year DESC,
            rcept_dt DESC,
            CASE
                WHEN report_code = '11011' THEN 1
                WHEN report_code = '11012' THEN 2
                WHEN report_code = '11013' THEN 3
                WHEN report_code = '11014' THEN 4
                ELSE 5
            END,
            thstrm_amount DESC
        LIMIT 1
        """

        sales_result = await self.execute_query(sales_query, self.symbol, self.analysis_date)

        if not sales_result or not sales_result[0]['sales']:
            return None

        sales = float(sales_result[0]['sales'])

        # Get operating income (second largest CIS amount, typically)
        # Due to encoding issues, use all CIS data and pick second
        op_income_query = """
        SELECT thstrm_amount
        FROM kr_financial_position
        WHERE symbol = $1
            AND sj_div = 'IS'
            AND thstrm_amount > 0
            AND rcept_dt <= COALESCE($2::date, CURRENT_DATE)
        ORDER BY
            bsns_year DESC,
            rcept_dt DESC,
            CASE
                WHEN report_code = '11011' THEN 1
                WHEN report_code = '11012' THEN 2
                WHEN report_code = '11013' THEN 3
                WHEN report_code = '11014' THEN 4
                ELSE 5
            END,
            thstrm_amount DESC
        LIMIT 3
        """

        op_results = await self.execute_query(op_income_query, self.symbol, self.analysis_date)

        if not op_results or len(op_results) < 2:
            return None

        # Second largest is typically operating income
        operating_income = float(op_results[1]['thstrm_amount'])
        operating_margin = (operating_income / sales) * 100

        # Get industry
        industry_query = """
        SELECT industry
        FROM kr_stock_detail
        WHERE symbol = $1
        """

        industry_result = await self.execute_query(industry_query, self.symbol)

        if not industry_result or not industry_result[0]['industry']:
            # No industry data, use absolute score only
            absolute_score = min(100, max(0, operating_margin * 5))
            return absolute_score

        industry = industry_result[0]['industry']

        # Note: Industry average calculation has encoding issues
        # Using absolute score only for reliability
        # Assume industry average is ~10% for relative scoring
        industry_avg = 10.0
        relative_score = min(100, max(0, (operating_margin / industry_avg) * 50))

        # Absolute score
        absolute_score = min(100, max(0, operating_margin * 5))

        final_score = (relative_score * 0.6) + (absolute_score * 0.4)

        return final_score

    # ========================================================================
    # Q3. Debt Stability Strategy
    # ========================================================================

    async def calculate_q3(self):
        """
        Q3. Debt Stability Strategy (Enhanced with BS fallback)
        Description: Stable financial structure with good debt management
        Debt Ratio, Current Ratio, Interest Coverage Ratio
        """
        # Get equity and debt from SCE
        sce_query = """
        SELECT
            MAX(CASE WHEN account_nm IN ('기말자본', '자본총계') THEN thstrm_amount END) as total_equity,
            MAX(CASE WHEN account_nm IN ('부채총계', '총부채') THEN thstrm_amount END) as total_debt
        FROM kr_financial_position
        WHERE symbol = $1
            AND sj_div = 'SCE'
            AND rcept_dt <= COALESCE($2::date, CURRENT_DATE)
        GROUP BY bsns_year
        ORDER BY bsns_year DESC, MAX(rcept_dt) DESC
        LIMIT 1
        """

        sce_result = await self.execute_query(sce_query, self.symbol, self.analysis_date)

        scores = []
        total_equity = None
        total_debt = None

        # Try SCE first
        if sce_result and sce_result[0]['total_equity'] and sce_result[0]['total_debt']:
            total_equity = float(sce_result[0]['total_equity'])
            total_debt = float(sce_result[0]['total_debt'])

        # Fallback: Use BS to estimate equity and debt
        if not total_equity or not total_debt:
            bs_query = """
            SELECT
                MAX(CASE WHEN account_nm IN ('자산총계', '자산') THEN thstrm_amount END) as total_assets,
                MAX(CASE WHEN account_nm IN ('자본총계', '자본', '기말자본') THEN thstrm_amount END) as total_equity,
                MAX(CASE WHEN account_nm IN ('부채총계', '부채', '총부채') THEN thstrm_amount END) as total_debt
            FROM kr_financial_position
            WHERE symbol = $1
                AND sj_div = 'BS'
                AND rcept_dt <= COALESCE($2::date, CURRENT_DATE)
                AND account_nm IN ('자산총계', '자산', '자본총계', '자본', '기말자본', '부채총계', '부채', '총부채')
            GROUP BY bsns_year
            ORDER BY bsns_year DESC, MAX(rcept_dt) DESC
            LIMIT 1
            """

            bs_result = await self.execute_query(bs_query, self.symbol, self.analysis_date)

            if bs_result and bs_result[0]['total_assets']:
                total_assets = float(bs_result[0]['total_assets']) if bs_result[0]['total_assets'] else None
                bs_equity = float(bs_result[0]['total_equity']) if bs_result[0]['total_equity'] else None
                bs_debt = float(bs_result[0]['total_debt']) if bs_result[0]['total_debt'] else None

                # Use direct values if available
                if bs_equity:
                    total_equity = bs_equity
                if bs_debt:
                    total_debt = bs_debt

                # If one is missing, calculate from assets
                if total_assets and total_equity and not total_debt:
                    total_debt = total_assets - total_equity
                elif total_assets and total_debt and not total_equity:
                    total_equity = total_assets - total_debt

        # Debt ratio score
        if total_equity and total_debt and total_equity > 0:
            debt_ratio = (total_debt / total_equity) * 100
            debt_score = 100 - min(100, max(0, (debt_ratio - 50) * 2))
            scores.append(('debt', debt_score, 0.5))

        # Get operating income from IS
        cis_query = """
        SELECT thstrm_amount
        FROM kr_financial_position
        WHERE symbol = $1
            AND sj_div = 'IS'
            AND account_nm IN ('영업이익', '영업이익(손실)')
            AND rcept_dt <= COALESCE($2::date, CURRENT_DATE)
        ORDER BY
            bsns_year DESC,
            rcept_dt DESC,
            CASE
                WHEN report_code = '11011' THEN 1
                WHEN report_code = '11012' THEN 2
                WHEN report_code = '11013' THEN 3
                WHEN report_code = '11014' THEN 4
                ELSE 5
            END
        LIMIT 1
        """

        cis_result = await self.execute_query(cis_query, self.symbol, self.analysis_date)

        # Interest coverage score (if data available)
        # Note: Interest expense data often unavailable, so this may not contribute
        if cis_result and cis_result[0]['thstrm_amount']:
            operating_income = float(cis_result[0]['thstrm_amount'])

            # Try to find interest expense (usually small negative or positive amount in CIS)
            interest_query = """
            SELECT thstrm_amount
            FROM kr_financial_position
            WHERE symbol = $1
                AND sj_div = 'IS'
                AND ABS(thstrm_amount) < $2
                AND thstrm_amount != 0
                AND rcept_dt <= COALESCE($3::date, CURRENT_DATE)
            ORDER BY
                bsns_year DESC,
                rcept_dt DESC,
                CASE
                    WHEN report_code = '11011' THEN 1
                    WHEN report_code = '11012' THEN 2
                    WHEN report_code = '11013' THEN 3
                    WHEN report_code = '11014' THEN 4
                    ELSE 5
                END,
                ABS(thstrm_amount) ASC
            LIMIT 1
            """

            interest_result = await self.execute_query(interest_query, self.symbol, operating_income * 0.1, self.analysis_date)

            if interest_result and interest_result[0]['thstrm_amount']:
                interest_expense = abs(float(interest_result[0]['thstrm_amount']))
                if interest_expense > 0:
                    interest_coverage = operating_income / interest_expense
                    interest_score = min(100, max(0, interest_coverage * 10))
                    scores.append(('interest', interest_score, 0.5))

        if not scores:
            return None

        # Weighted average
        total_weight = sum(s[2] for s in scores)
        weighted_score = sum(s[1] * s[2] for s in scores) / total_weight

        return weighted_score

    # ========================================================================
    # Q4. Cash Reserve Adequacy Strategy
    # ========================================================================

    async def calculate_q4(self):
        """
        Q4. Cash Reserve Adequacy Strategy (position-based, LIMIT bug fixed)
        Description: 현금 보유 비율 (위치 기반, 버그 수정)
        BS에서 가장 작은 양수 = 현금성자산일 가능성
        LIMIT 10 버그 수정: ROW_NUMBER를 최종 결과에 적용
        """
        query = """
        SELECT
            MAX(CASE WHEN account_nm IN ('현금및현금성자산', '현금및예금', '현금', '당좌자산')
                THEN thstrm_amount END) as cash_estimate,
            MAX(CASE WHEN account_nm IN ('현금및현금성자산', '현금및예금', '현금', '당좌자산')
                THEN frmtrm_amount END) as cash_prev,
            MAX(CASE WHEN account_nm IN ('자산총계', '자산총액')
                THEN thstrm_amount END) as total_assets
        FROM kr_financial_position
        WHERE symbol = $1
            AND sj_div = 'BS'
            AND rcept_dt <= COALESCE($2::date, CURRENT_DATE)
        GROUP BY bsns_year
        ORDER BY bsns_year DESC, MAX(rcept_dt) DESC
        LIMIT 1
        """

        result = await self.execute_query(query, self.symbol, self.analysis_date)

        if not result or not result[0]['cash_estimate'] or not result[0]['total_assets']:
            return None

        row = result[0]
        cash = float(row['cash_estimate'])
        total_assets = float(row['total_assets'])

        if total_assets <= 0:
            return None

        # Cash ratio
        cash_ratio = (cash / total_assets) * 100

        # Score based on cash ratio
        # 30% 이상: 100점
        # 20~30%: 70~100점
        # 10~20%: 40~70점
        # 10% 미만: 비례
        if cash_ratio >= 30:
            ratio_score = 100.0
        elif cash_ratio >= 20:
            ratio_score = 70 + (cash_ratio - 20) * 3
        elif cash_ratio >= 10:
            ratio_score = 40 + (cash_ratio - 10) * 3
        else:
            ratio_score = cash_ratio * 4

        # Bonus for cash growth
        growth_bonus = 0
        if row['cash_prev'] and float(row['cash_prev']) > 0:
            cash_prev = float(row['cash_prev'])
            cash_growth = ((cash - cash_prev) / cash_prev) * 100
            if cash_growth > 0:
                growth_bonus = min(20, cash_growth * 2)

        final_score = min(100, ratio_score + growth_bonus)

        return final_score

    # ========================================================================
    # Q5. Asset Efficiency Strategy
    # ========================================================================

    async def calculate_q5_original(self):
        """
        Q5. Asset Efficiency Strategy (ORIGINAL - For IC Comparison)

        IC: -0.0203 (Negative correlation)

        Description: Efficient use of assets to generate sales (Turnover only)
        Total asset turnover = Sales / Total Assets

        This is the original version for IC comparison purposes.
        """
        query = """
        SELECT
            MAX(CASE WHEN sj_div = 'IS' AND account_nm = '매출액'
                THEN thstrm_amount END) as sales,
            MAX(CASE WHEN sj_div = 'BS' AND account_nm IN ('자산총계', '자산총액')
                THEN thstrm_amount END) as total_assets
        FROM kr_financial_position
        WHERE symbol = $1
            AND rcept_dt <= COALESCE($2::date, CURRENT_DATE)
            AND ((sj_div = 'IS' AND account_nm = '매출액')
                 OR (sj_div = 'BS' AND account_nm IN ('자산총계', '자산총액')))
        GROUP BY bsns_year
        ORDER BY bsns_year DESC, MAX(rcept_dt) DESC
        LIMIT 1
        """

        result = await self.execute_query(query, self.symbol, self.analysis_date)

        if not result or not result[0]['sales'] or not result[0]['total_assets']:
            return None

        row = result[0]
        sales = float(row['sales'])
        total_assets = float(row['total_assets'])

        if total_assets <= 0:
            return None

        # Total asset turnover
        asset_turnover = sales / total_assets

        # Score calculation
        # IT/Electronics: 0.3~1.0 range (0.5 = 60점, 1.0 = 100점)
        # Manufacturing: 0.5~1.5 range (1.0 = 70점, 1.5 = 100점)
        # Use general scaling: 0.5 = 50점, 1.0 = 70점, 1.5+ = 100점
        if asset_turnover >= 1.5:
            score = 100.0
        elif asset_turnover >= 1.0:
            score = 70.0 + (asset_turnover - 1.0) * 60  # 1.0~1.5 사이 선형
        elif asset_turnover >= 0.5:
            score = 50.0 + (asset_turnover - 0.5) * 40  # 0.5~1.0 사이 선형
        else:
            score = asset_turnover * 100  # 0.5 이하는 비례

        return min(100, max(0, score))

    async def calculate_q5(self):
        """
        Q5 V3. Asset Efficiency Trend Strategy (Trend-Based REDESIGN)

        Problem with V2:
        - ROA Level-based approach failed (IC -0.0138)
        - Korean market rewards IMPROVEMENT over absolute levels

        New Approach - Focus on Trends:
        1. ROA Trend (50%): 3-year ROA improvement trajectory
           - Linear regression slope of ROA over 3 years
           - Positive slope = improving efficiency

        2. ROA Consistency (30%): ROA stability
           - Lower standard deviation = more reliable
           - Inverse of coefficient of variation

        3. Asset Turnover (20%): Sales / Assets
           - Maintained from original Q5

        Target IC: > +0.03 (current: -0.0138)

        Data Sources:
        - 당기순이익, 자산총계, 매출액 (최근 3개년)
        """
        # Get 3-year financial data
        query = """
        WITH recent_years AS (
            SELECT DISTINCT bsns_year, rcept_dt
            FROM kr_financial_position
            WHERE symbol = $1
                AND rcept_dt <= COALESCE($2::date, CURRENT_DATE)
            ORDER BY bsns_year DESC, rcept_dt DESC
            LIMIT 3
        )
        SELECT
            ry.bsns_year,
            MAX(CASE WHEN sj_div = 'IS' AND account_nm = '매출액'
                THEN thstrm_amount END) as sales,
            MAX(CASE WHEN sj_div = 'IS' AND account_nm IN ('당기순이익', '당기순이익(손실)')
                THEN thstrm_amount END) as net_income,
            MAX(CASE WHEN sj_div = 'BS' AND account_nm IN ('자산총계', '자산총액')
                THEN thstrm_amount END) as total_assets
        FROM kr_financial_position fp
        INNER JOIN recent_years ry
            ON fp.bsns_year = ry.bsns_year
            AND fp.rcept_dt = ry.rcept_dt
        WHERE fp.symbol = $1
        GROUP BY ry.bsns_year
        ORDER BY ry.bsns_year DESC
        """

        results = await self.execute_query(query, self.symbol, self.analysis_date)

        if not results or len(results) < 2:
            # Need at least 2 years for trend
            return None

        # Extract ROA for each year
        roa_data = []
        latest_sales = None
        latest_assets = None

        for i, row in enumerate(results):
            net_income = float(row['net_income']) if row['net_income'] else None
            total_assets = float(row['total_assets']) if row['total_assets'] else None
            sales = float(row['sales']) if row['sales'] else None

            if i == 0:  # Latest year
                latest_sales = sales
                latest_assets = total_assets

            if net_income is not None and total_assets and total_assets > 0:
                roa = (net_income / total_assets) * 100
                roa_data.append(roa)

        if len(roa_data) < 2 or latest_assets is None or latest_assets <= 0:
            return None

        # Component 1: ROA Trend (50%)
        trend_score = 50.0  # Default neutral

        if len(roa_data) >= 2:
            # Calculate linear regression slope
            # y = ROA values (most recent first)
            # x = time periods (0, 1, 2, ... from most recent)
            n = len(roa_data)
            x = list(range(n))
            y = roa_data

            # Linear regression: slope = Σ((x-x̄)(y-ȳ)) / Σ((x-x̄)²)
            x_mean = sum(x) / n
            y_mean = sum(y) / n

            numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n))
            denominator = sum((x[i] - x_mean) ** 2 for i in range(n))

            if denominator > 0:
                slope = numerator / denominator

                # Slope Scoring (percentage points per year)
                # +5%p/year = 100점 (excellent improvement)
                # +2%p/year = 80점 (good improvement)
                # 0%p/year = 50점 (flat)
                # -2%p/year = 20점 (deteriorating)
                # -5%p/year = 0점 (bad decline)

                if slope >= 5:
                    trend_score = 100.0
                elif slope >= 2:
                    trend_score = 80.0 + (slope - 2) * (20 / 3)  # 2~5: 80~100
                elif slope >= 0:
                    trend_score = 50.0 + slope * 15  # 0~2: 50~80
                elif slope >= -2:
                    trend_score = 20.0 + (slope + 2) * 15  # -2~0: 20~50
                else:
                    trend_score = max(0.0, 20.0 + (slope + 2) * 10)  # < -2: 0~20

        # Component 2: ROA Consistency (30%)
        consistency_score = 50.0  # Default neutral

        if len(roa_data) >= 2:
            import statistics
            roa_std = statistics.stdev(roa_data)
            roa_mean = statistics.mean(roa_data)

            if abs(roa_mean) > 0.1:  # Avoid division by very small numbers
                # Coefficient of Variation (CV) = std / |mean|
                cv = roa_std / abs(roa_mean)

                # Lower CV = more consistent = better
                # CV < 0.2 = 100점 (very stable)
                # CV = 0.5 = 70점 (moderate)
                # CV = 1.0 = 40점 (volatile)
                # CV > 2.0 = 0점 (very volatile)

                if cv <= 0.2:
                    consistency_score = 100.0
                elif cv <= 0.5:
                    consistency_score = 70.0 + (0.5 - cv) * 100  # 0.2~0.5: 70~100
                elif cv <= 1.0:
                    consistency_score = 40.0 + (1.0 - cv) * 60  # 0.5~1.0: 40~70
                elif cv <= 2.0:
                    consistency_score = (2.0 - cv) * 40  # 1.0~2.0: 0~40
                else:
                    consistency_score = 0.0
            else:
                # Mean too close to zero, use std directly
                if roa_std <= 2:
                    consistency_score = 100.0
                elif roa_std <= 5:
                    consistency_score = 70.0 + (5 - roa_std) * 10
                elif roa_std <= 10:
                    consistency_score = 40.0 + (10 - roa_std) * 6
                else:
                    consistency_score = max(0.0, 40.0 - (roa_std - 10) * 4)

        # Component 3: Asset Turnover (20%)
        turnover_score = 50.0  # Default neutral

        if latest_sales and latest_assets > 0:
            asset_turnover = latest_sales / latest_assets

            # Same scoring as original Q5
            if asset_turnover >= 1.5:
                turnover_score = 100.0
            elif asset_turnover >= 1.0:
                turnover_score = 70.0 + (asset_turnover - 1.0) * 60
            elif asset_turnover >= 0.5:
                turnover_score = 40.0 + (asset_turnover - 0.5) * 60
            else:
                turnover_score = asset_turnover * 80

        # Final Score: Weighted average
        final_score = (
            trend_score * 0.5 +
            consistency_score * 0.3 +
            turnover_score * 0.2
        )

        return max(0.0, min(100.0, final_score))

    async def calculate_q5_v2(self):
        """
        Q5 V2. ROA-Centered Asset Efficiency Strategy (BACKUP - FAILED IC: -0.0138)

        Target IC: > +0.03 (improved from -0.0203)
        Result: FAILED - IC still negative, inconsistent performance

        Components:
        1. ROA (60%): Net Income / Total Assets - Primary efficiency metric
        2. Asset Turnover Quality (25%): (Sales/Assets) × Operating Margin
        3. Asset Turnover Level (15%): Sales / Assets

        Data Sources (100% available):
        - 당기순이익, 자산총계, 매출액, 영업이익

        Status: Replaced by Trend-Based Q5 V3
        """
        query = """
        WITH latest_report AS (
            SELECT bsns_year, rcept_dt
            FROM kr_financial_position
            WHERE symbol = $1
                AND rcept_dt <= COALESCE($2::date, CURRENT_DATE)
            ORDER BY bsns_year DESC, rcept_dt DESC
            LIMIT 1
        )
        SELECT
            MAX(CASE WHEN sj_div = 'IS' AND account_nm = '매출액'
                THEN thstrm_amount END) as sales,
            MAX(CASE WHEN sj_div = 'IS' AND account_nm IN ('당기순이익', '당기순이익(손실)')
                THEN thstrm_amount END) as net_income,
            MAX(CASE WHEN sj_div = 'IS' AND account_nm IN ('영업이익', '영업이익(손실)')
                THEN thstrm_amount END) as operating_profit,
            MAX(CASE WHEN sj_div = 'BS' AND account_nm IN ('자산총계', '자산총액')
                THEN thstrm_amount END) as total_assets
        FROM kr_financial_position fp
        INNER JOIN latest_report lr
            ON fp.bsns_year = lr.bsns_year
            AND fp.rcept_dt = lr.rcept_dt
        WHERE fp.symbol = $1
        """

        result = await self.execute_query(query, self.symbol, self.analysis_date)

        if not result or not result[0]['total_assets']:
            return None

        row = result[0]
        sales = float(row['sales']) if row['sales'] else None
        net_income = float(row['net_income']) if row['net_income'] else None
        operating_profit = float(row['operating_profit']) if row['operating_profit'] else None
        total_assets = float(row['total_assets'])

        if total_assets <= 0:
            return None

        # Component 1: ROA (60%) - Primary metric
        roa_score = 50.0  # Default neutral
        if net_income is not None:
            roa = (net_income / total_assets) * 100

            # ROA Scoring
            # 15% = 100점, 10% = 80점, 5% = 60점, 0% = 50점, -10% = 0점
            if roa >= 15:
                roa_score = 100.0
            elif roa >= 10:
                roa_score = 80.0 + (roa - 10) * 4  # 10~15%: 80~100점
            elif roa >= 5:
                roa_score = 60.0 + (roa - 5) * 4   # 5~10%: 60~80점
            elif roa >= 0:
                roa_score = 50.0 + (roa / 5) * 10  # 0~5%: 50~60점
            else:
                # Negative ROA: Penalty
                roa_score = max(0.0, 50.0 + roa * 5)  # -10% = 0점

        # Component 2: Asset Turnover Quality (25%)
        quality_score = 50.0  # Default neutral
        if sales and operating_profit and sales > 0:
            asset_turnover = sales / total_assets
            operating_margin = (operating_profit / sales) * 100

            # Turnover Quality = Turnover × (Operating Margin / 100)
            # Example: Turnover 1.5 × Margin 10% = 0.15
            # Example: Turnover 0.6 × Margin 25% = 0.15 (same quality!)
            turnover_quality = asset_turnover * (operating_margin / 100)

            # Quality Scoring
            # 0.30 = 100점 (예: 1.5 × 20% or 3.0 × 10%)
            # 0.15 = 70점 (예: 1.5 × 10% or 0.6 × 25%)
            # 0.05 = 40점
            # 0 = 0점
            if turnover_quality >= 0.30:
                quality_score = 100.0
            elif turnover_quality >= 0.15:
                quality_score = 70.0 + (turnover_quality - 0.15) * 200
            elif turnover_quality >= 0.05:
                quality_score = 40.0 + (turnover_quality - 0.05) * 300
            else:
                quality_score = turnover_quality * 800

        # Component 3: Asset Turnover Level (15%)
        turnover_score = 50.0  # Default neutral
        if sales:
            asset_turnover = sales / total_assets

            # Same scoring as original Q5
            if asset_turnover >= 1.5:
                turnover_score = 100.0
            elif asset_turnover >= 1.0:
                turnover_score = 70.0 + (asset_turnover - 1.0) * 60
            elif asset_turnover >= 0.5:
                turnover_score = 40.0 + (asset_turnover - 0.5) * 60
            else:
                turnover_score = asset_turnover * 80

        # Final Score: Weighted average
        final_score = (
            roa_score * 0.6 +
            quality_score * 0.25 +
            turnover_score * 0.15
        )

        return max(0.0, min(100.0, final_score))

    # ========================================================================
    # Q6. Cash Generation Ability Strategy
    # ========================================================================

    async def calculate_q6(self):
        """
        Q6. Cash Generation Ability Strategy
        Description: Cash flow generation and quality assessment
        Primary: OCF / Sales + FCF / Market Cap (from CF statement)
        Fallback: CCR (Cash Conversion Ratio) + ROA (from BS + IS)
        """
        # Get market cap
        market_cap_query = """
        SELECT market_cap
        FROM kr_intraday_total
        WHERE symbol = $1
            AND ($2::date IS NULL OR date = $2)
        ORDER BY date DESC
        LIMIT 1
        """

        mc_result = await self.execute_query(market_cap_query, self.symbol, self.analysis_date)

        if not mc_result or not mc_result[0]['market_cap']:
            return None

        market_cap = float(mc_result[0]['market_cap'])

        # Primary: Try to get Operating Cash Flow from CF statement
        ocf_query = """
        SELECT thstrm_amount
        FROM kr_financial_position
        WHERE symbol = $1
            AND sj_div = 'CF'
            AND (
                account_nm LIKE '%영업활동%현금흐름%'
                OR account_nm LIKE '%영업활동으로%현금%'
                OR account_nm = '영업활동현금흐름'
                OR account_nm = '영업활동으로인한현금흐름'
            )
            AND thstrm_amount IS NOT NULL
            AND rcept_dt <= COALESCE($2::date, CURRENT_DATE)
        ORDER BY bsns_year DESC, rcept_dt DESC, ABS(thstrm_amount) DESC
        LIMIT 1
        """

        ocf_result = await self.execute_query(ocf_query, self.symbol, self.analysis_date)

        # Primary method: Use CF data if available
        if ocf_result and ocf_result[0]['thstrm_amount']:
            operating_cash = float(ocf_result[0]['thstrm_amount'])

            # Get Sales and Capex
            sales_capex_query = """
            WITH latest_report AS (
                SELECT bsns_year, rcept_dt
                FROM kr_financial_position
                WHERE symbol = $1
                    AND rcept_dt <= COALESCE($2::date, CURRENT_DATE)
                ORDER BY bsns_year DESC, rcept_dt DESC
                LIMIT 1
            )
            SELECT
                MAX(CASE WHEN sj_div = 'IS' AND thstrm_amount > 0
                    THEN thstrm_amount END) as sales,
                MAX(CASE WHEN sj_div = 'CF' AND (
                        account_nm LIKE '%유형자산%취득%'
                        OR account_nm LIKE '%유형자산의취득%'
                        OR account_nm = '유형자산의 취득'
                    )
                    THEN ABS(thstrm_amount) END) as capex
            FROM kr_financial_position fp
            INNER JOIN latest_report lr
                ON fp.bsns_year = lr.bsns_year
                AND fp.rcept_dt = lr.rcept_dt
            WHERE fp.symbol = $1
            """

            sales_result = await self.execute_query(sales_capex_query, self.symbol, self.analysis_date)

            if sales_result and sales_result[0]['sales']:
                sales = float(sales_result[0]['sales'])
                capex = float(sales_result[0]['capex']) if sales_result[0]['capex'] else 0

                # Calculate scores
                cash_generation_rate = (operating_cash / sales) * 100

                # Check for negative operating cash flow
                if operating_cash <= 0:
                    generation_score = 0
                else:
                    generation_score = min(100, max(0, cash_generation_rate * 5))

                # Free Cash Flow
                free_cash_flow = operating_cash - capex
                fcf_rate = (free_cash_flow / market_cap) * 100
                fcf_score = min(100, max(0, fcf_rate * 10))

                final_score = (generation_score * 0.5) + (fcf_score * 0.5)

                return final_score

        # Fallback: Use BS cash change and CCR calculation
        ccr_query = """
        WITH latest_report AS (
            SELECT bsns_year, rcept_dt
            FROM kr_financial_position
            WHERE symbol = $1
                AND sj_div = 'BS'
                AND rcept_dt <= COALESCE($2::date, CURRENT_DATE)
            ORDER BY bsns_year DESC, rcept_dt DESC
            LIMIT 1
        ),
        net_income AS (
            SELECT fp.thstrm_amount
            FROM kr_financial_position fp
            INNER JOIN latest_report lr
                ON fp.bsns_year = lr.bsns_year
                AND fp.rcept_dt = lr.rcept_dt
            WHERE fp.symbol = $1
                AND fp.sj_div = 'IS'
                AND fp.account_nm IN ('당기순이익(손실)', '당기순이익')
            ORDER BY ABS(fp.thstrm_amount) DESC
            LIMIT 1
        ),
        cash_change AS (
            SELECT
                SUM(fp.thstrm_amount - COALESCE(fp.frmtrm_amount, 0)) as cash_delta
            FROM kr_financial_position fp
            INNER JOIN latest_report lr
                ON fp.bsns_year = lr.bsns_year
                AND fp.rcept_dt = lr.rcept_dt
            WHERE fp.symbol = $1
                AND fp.sj_div = 'BS'
                AND (
                    fp.account_nm IN (
                        '현금및현금성자산',
                        '현금',
                        '현금및예치금',
                        '단기금융상품',
                        '단기투자자산',
                        '현금성자산'
                    )
                    OR fp.account_nm LIKE '%현금%자산%'
                    OR fp.account_nm LIKE '%단기금융%'
                )
        ),
        total_assets AS (
            SELECT fp.thstrm_amount
            FROM kr_financial_position fp
            INNER JOIN latest_report lr
                ON fp.bsns_year = lr.bsns_year
                AND fp.rcept_dt = lr.rcept_dt
            WHERE fp.symbol = $1
                AND fp.sj_div = 'BS'
                AND fp.account_nm IN ('자산총계', '자산총액')
            LIMIT 1
        )
        SELECT
            ni.thstrm_amount as net_income,
            cc.cash_delta,
            ta.thstrm_amount as total_assets
        FROM net_income ni
        CROSS JOIN cash_change cc
        CROSS JOIN total_assets ta
        """

        ccr_result = await self.execute_query(ccr_query, self.symbol, self.analysis_date)

        if not ccr_result or not ccr_result[0]['net_income'] or not ccr_result[0]['total_assets']:
            return None

        row = ccr_result[0]
        net_income = float(row['net_income'])
        cash_delta = float(row['cash_delta']) if row['cash_delta'] else 0
        total_assets = float(row['total_assets'])

        # Exclude loss-making companies
        if net_income <= 0:
            return None

        if total_assets <= 0:
            return None

        # Calculate CCR (Cash Conversion Ratio)
        ccr = cash_delta / net_income if net_income != 0 else 0

        # CCR Scoring
        # 0.8 ~ 1.2: 이상적 (이익이 현금으로 잘 전환)
        # 0.5 ~ 0.8: 보통 (재고/외상 일부 증가)
        # < 0.5: 나쁨 (이익의 절반도 현금 안됨)
        # > 1.5: 비정상 (자산 처분 등 일회성)
        if 0.8 <= ccr <= 1.2:
            ccr_score = 100.0
        elif 0.5 <= ccr < 0.8:
            # 0.5 → 60점, 0.8 → 100점
            ccr_score = 60.0 + (ccr - 0.5) * 133.33
        elif ccr < 0.5:
            # 0 → 0점, 0.5 → 60점
            ccr_score = ccr * 120
        elif 1.2 < ccr <= 1.5:
            # 1.2 → 100점, 1.5 → 70점
            ccr_score = 100.0 - (ccr - 1.2) * 100
        else:  # ccr > 1.5
            # 비정상적으로 높음 (일회성 가능성)
            ccr_score = 40.0

        # ROA (Return on Assets)
        roa = (net_income / total_assets) * 100
        if roa >= 10:
            roa_score = 100.0
        elif roa >= 5:
            roa_score = 70 + (roa - 5) * 6
        elif roa >= 2:
            roa_score = 40 + (roa - 2) * 10
        else:
            roa_score = roa * 20

        # Final Score: CCR 70%, ROA 30%
        final_score = (ccr_score * 0.7) + (roa_score * 0.3)

        return min(100, max(0, final_score))

    # ========================================================================
    # Q7. Dividend Sustainability Strategy
    # ========================================================================

    async def calculate_q7(self):
        """
        Q7. Dividend Sustainability Strategy
        Description: Stable and continuous dividend policy
        Years of dividends, growth, payout ratio
        """
        # Get dividend years count
        years_query = """
        SELECT COUNT(DISTINCT EXTRACT(YEAR FROM stlm_dt)) as dividend_years
        FROM kr_dividends
        WHERE symbol = $1
            AND thstrm > 0
            AND stlm_dt IS NOT NULL
        """

        years_result = await self.execute_query(years_query, self.symbol)

        if not years_result:
            return None

        dividend_years = int(years_result[0]['dividend_years']) if years_result[0]['dividend_years'] else 0

        if dividend_years == 0:
            return None

        # Continuity score
        continuity_score = min(100, dividend_years * 20)

        # Get dividend growth
        growth_query = """
        SELECT thstrm, lwfr
        FROM kr_dividends
        WHERE symbol = $1
            AND thstrm > 0
            AND stlm_dt IS NOT NULL
        ORDER BY stlm_dt DESC
        LIMIT 1
        """

        growth_result = await self.execute_query(growth_query, self.symbol)

        growth_score = 25  # Default
        if growth_result and growth_result[0]['thstrm'] and growth_result[0]['lwfr']:
            current_div = float(growth_result[0]['thstrm'])
            previous_div = float(growth_result[0]['lwfr'])

            if previous_div > 0:
                growth_rate = ((current_div - previous_div) / previous_div) * 100
                growth_score = min(50, max(0, growth_rate * 2))

        # Get payout ratio
        payout_query = """
        SELECT eps
        FROM kr_intraday_detail
        WHERE symbol = $1
        """

        eps_result = await self.execute_query(payout_query, self.symbol)

        stability_score = 30  # Default
        if eps_result and eps_result[0]['eps'] and growth_result:
            eps = float(eps_result[0]['eps'])
            dividend_per_share = float(growth_result[0]['thstrm'])

            if eps > 0:
                payout_ratio = (dividend_per_share / eps) * 100

                if 20 <= payout_ratio <= 50:
                    stability_score = 100
                elif 10 <= payout_ratio <= 60:
                    stability_score = 70
                else:
                    stability_score = 30

        final_score = (continuity_score * 0.4) + (growth_score * 0.2) + (stability_score * 0.4)

        return final_score

    # ========================================================================
    # Q8. Sales Stability Strategy
    # ========================================================================

    async def calculate_q8(self):
        """
        Q8. Sales Stability Strategy (Modified: 2-year comparison)
        Description: Low volatility and predictable sales
        Sales growth rate and absolute level
        """
        query = """
        SELECT
            MAX(CASE WHEN sj_div = 'IS' AND thstrm_amount > 0
                THEN thstrm_amount END) as sales_t0,
            MAX(CASE WHEN sj_div = 'IS' AND frmtrm_amount > 0
                THEN frmtrm_amount END) as sales_t1
        FROM kr_financial_position
        WHERE symbol = $1
            AND rcept_dt <= COALESCE($2::date, CURRENT_DATE)
        GROUP BY bsns_year
        ORDER BY bsns_year DESC, MAX(rcept_dt) DESC
        LIMIT 1
        """

        result = await self.execute_query(query, self.symbol, self.analysis_date)

        if not result or not result[0]['sales_t0']:
            return None

        row = result[0]
        sales_t0 = float(row['sales_t0'])

        # If previous year data available, calculate growth
        if row['sales_t1'] and float(row['sales_t1']) > 0:
            sales_t1 = float(row['sales_t1'])
            growth_rate = ((sales_t0 - sales_t1) / sales_t1) * 100

            # Score based on growth rate
            # > 10%: 80~100점
            # 3~10%: 60~80점
            # 0~3%: 40~60점
            # < 0%: 0~40점
            if growth_rate >= 10:
                growth_score = min(100, max(0, 80 + (growth_rate - 10) * 2))
            elif growth_rate >= 3:
                growth_score = 60 + (growth_rate - 3) * (20 / 7)
            elif growth_rate >= 0:
                growth_score = 40 + growth_rate * (20 / 3)
            else:
                growth_score = max(0, 40 + growth_rate * 2)

            final_score = growth_score
        else:
            # Only current year - give neutral score based on absolute sales level
            # Large sales (> 1 trillion) = 60점
            if sales_t0 > 1000000000000:
                final_score = 60.0
            else:
                final_score = 50.0

        return final_score

    # ========================================================================
    # Q9. Profitability Improvement Trend Strategy
    # ========================================================================

    async def calculate_q9(self):
        """
        Q9. Profitability Excellence Strategy (Enhanced with relaxed conditions)
        Description: Current profitability level assessment with fallback logic
        GPM (Gross Profit Margin) and NPM (Net Profit Margin) absolute levels
        """
        query = """
        SELECT
            MAX(CASE WHEN account_nm = '매출액' THEN thstrm_amount END) as sales,
            MAX(CASE WHEN account_nm IN ('매출총이익', '매출총손익') THEN thstrm_amount END) as gross_profit,
            MAX(CASE WHEN account_nm IN ('당기순이익(손실)', '당기순이익') THEN thstrm_amount END) as net_income
        FROM kr_financial_position
        WHERE symbol = $1
            AND sj_div = 'IS'
            AND rcept_dt <= COALESCE($2::date, CURRENT_DATE)
            AND account_nm IN ('매출액', '매출총이익', '매출총손익', '당기순이익(손실)', '당기순이익')
        GROUP BY bsns_year
        ORDER BY bsns_year DESC, MAX(rcept_dt) DESC
        LIMIT 1
        """

        result = await self.execute_query(query, self.symbol, self.analysis_date)

        if not result or not result[0]['sales']:
            return None

        row = result[0]
        sales = float(row['sales'])
        gross_profit = float(row['gross_profit']) if row['gross_profit'] else None
        net_income = float(row['net_income']) if row['net_income'] else None

        scores = []

        # GPM score
        if gross_profit:
            gpm = (gross_profit / sales) * 100
            # > 30%: 50점, > 40%: 70점, > 50%: 100점
            if gpm >= 50:
                gpm_score = 100.0
            elif gpm >= 40:
                gpm_score = 70 + (gpm - 40) * 3
            elif gpm >= 30:
                gpm_score = 50 + (gpm - 30) * 2
            elif gpm >= 20:
                gpm_score = 30 + (gpm - 20) * 2
            else:
                gpm_score = gpm * 1.5

            scores.append(min(100, max(0, gpm_score)))

        # NPM score
        if net_income:
            npm = (net_income / sales) * 100
            # > 10%: 50점, > 15%: 70점, > 20%: 100점
            if npm >= 20:
                npm_score = 100.0
            elif npm >= 15:
                npm_score = 70 + (npm - 15) * 6
            elif npm >= 10:
                npm_score = 50 + (npm - 10) * 4
            elif npm >= 5:
                npm_score = 30 + (npm - 5) * 4
            else:
                npm_score = npm * 6

            scores.append(min(100, max(0, npm_score)))

        # Accept single metric (GPM or NPM alone is valid)
        if not scores:
            return None

        final_score = statistics.mean(scores)

        return final_score

    # ========================================================================
    # Q10. Executive Stability Strategy
    # ========================================================================

    async def calculate_q10(self):
        """
        Q10. Governance Quality Strategy (Modified: Using audit opinions as proxy)
        Description: Estimate management stability through audit reliability
        Clean audit opinions over multiple years indicate stable governance
        """
        audit_query = """
        SELECT
            COUNT(*) as total_years,
            COUNT(CASE WHEN adt_opinion IN ('적정', '정상', '적정의견') THEN 1 END) as clean_years,
            COUNT(CASE WHEN adt_reprt_spcmnt_matter IS NULL OR adt_reprt_spcmnt_matter = '' THEN 1 END) as no_special_matters
        FROM (
            SELECT adt_opinion, adt_reprt_spcmnt_matter
            FROM kr_audit
            WHERE symbol = $1
            ORDER BY bsns_year DESC
            LIMIT 5
        ) AS recent_audits
        """

        result = await self.execute_query(audit_query, self.symbol)

        if not result or not result[0]['total_years']:
            # No audit data - return neutral score
            return 50.0

        row = result[0]
        total_years = int(row['total_years'])
        clean_years = int(row['clean_years']) if row['clean_years'] else 0
        no_special_matters = int(row['no_special_matters']) if row['no_special_matters'] else 0

        # Clean opinion ratio
        clean_ratio = (clean_years / total_years) * 100 if total_years > 0 else 0

        # Special matters ratio
        clean_matters_ratio = (no_special_matters / total_years) * 100 if total_years > 0 else 0

        # Score calculation
        # 5년 연속 적정: 100점
        # 4년 적정: 80점
        # 3년 적정: 60점
        opinion_score = min(100, clean_ratio)

        # No special matters bonus
        matters_score = min(100, clean_matters_ratio)

        final_score = (opinion_score * 0.7) + (matters_score * 0.3)

        return final_score

    # ========================================================================
    # Q11. Audit Reliability Strategy
    # ========================================================================

    async def calculate_q11(self):
        """
        Q11. Audit Reliability Strategy
        Description: Clean audit opinion with no special matters
        Audit opinion, special matters, emphasis matters
        """
        audit_query = """
        SELECT
            adt_opinion,
            adt_reprt_spcmnt_matter,
            emphs_matter
        FROM kr_audit
        WHERE symbol = $1
            AND adt_opinion IS NOT NULL
        ORDER BY bsns_year DESC
        LIMIT 1
        """

        result = await self.execute_query(audit_query, self.symbol)

        if not result:
            return None

        row = result[0]

        # Audit opinion score (encoding issues - use pattern matching)
        opinion_score = 70  # Default
        if row['adt_opinion']:
            opinion = row['adt_opinion']
            if '무적정' in opinion or '적정' in opinion or 'ǰ' in opinion:  # Corrupted text pattern
                opinion_score = 100
            elif '한정' in opinion:
                opinion_score = 50
            elif '부적정' in opinion or '거절' in opinion:
                opinion_score = 0

        # Special matters score
        special_score = 100
        if row['adt_reprt_spcmnt_matter'] and len(row['adt_reprt_spcmnt_matter']) > 0:
            special_score = 50

        # Emphasis matters score
        emphasis_score = 100
        if row['emphs_matter']:
            emphasis = row['emphs_matter']
            if '계속기업' in emphasis:
                emphasis_score = 0
            else:
                emphasis_score = 70

        final_score = (opinion_score * 0.5) + (special_score * 0.25) + (emphasis_score * 0.25)

        return final_score

    # ========================================================================
    # Q12. Working Capital Management Efficiency Strategy
    # ========================================================================

    async def calculate_q12(self):
        """
        Q12. Working Capital Management Efficiency Strategy
        Description: Efficient working capital with fast cash conversion
        Cash Conversion Cycle (CCC)
        """
        query = """
        SELECT
            MAX(CASE WHEN account_nm IN ('매출채권', '매출채권및기타채권', '단기매출채권') THEN thstrm_amount END) as receivables,
            MAX(CASE WHEN account_nm IN ('재고자산', '상품', '제품') THEN thstrm_amount END) as inventory,
            MAX(CASE WHEN account_nm IN ('매입채무', '매입채무및기타채무', '단기매입채무') THEN thstrm_amount END) as payables,
            MAX(CASE WHEN account_nm = '매출액' THEN thstrm_amount END) as sales,
            MAX(CASE WHEN account_nm IN ('매출원가', '상품매출원가', '제품매출원가') THEN thstrm_amount END) as cogs,
            MIN(CASE
                WHEN report_code = '11011' THEN 1
                WHEN report_code = '11012' THEN 2
                WHEN report_code = '11013' THEN 3
                WHEN report_code = '11014' THEN 4
                ELSE 5
            END) as report_priority
        FROM kr_financial_position
        WHERE symbol = $1
            AND rcept_dt <= COALESCE($2::date, CURRENT_DATE)
        GROUP BY bsns_year
        ORDER BY bsns_year DESC, MAX(rcept_dt) DESC, report_priority
        LIMIT 1
        """

        result = await self.execute_query(query, self.symbol, self.analysis_date)

        if not result or not result[0]['sales']:
            return None

        row = result[0]

        days_sales_outstanding = 0
        days_inventory_outstanding = 0
        days_payables_outstanding = 0

        # Calculate DSO
        if row['receivables'] and float(row['sales']) > 0:
            dso = (float(row['receivables']) / float(row['sales'])) * 365
            days_sales_outstanding = dso

        # Calculate DIO
        if row['inventory'] and row['cogs'] and float(row['cogs']) > 0:
            dio = (float(row['inventory']) / float(row['cogs'])) * 365
            days_inventory_outstanding = dio

        # Calculate DPO
        if row['payables'] and row['cogs'] and float(row['cogs']) > 0:
            dpo = (float(row['payables']) / float(row['cogs'])) * 365
            days_payables_outstanding = dpo

        # Cash Conversion Cycle
        ccc = days_sales_outstanding + days_inventory_outstanding - days_payables_outstanding

        # Score: Lower CCC is better
        score = 100 - min(100, max(0, ccc * 0.5))

        return score

    # ========================================================================
    # Q13. Supply-Demand Quality Strategy
    # ========================================================================

    async def calculate_q13(self):
        """
        Q13. Supply-Demand Quality Strategy
        Description: Continuous net buying by institutions and foreigners
        Institutional and foreign investor buying
        """
        # Get market cap
        market_cap_query = """
        SELECT market_cap
        FROM kr_intraday_total
        WHERE symbol = $1
            AND ($2::date IS NULL OR date = $2)
        ORDER BY date DESC
        LIMIT 1
        """

        mc_result = await self.execute_query(market_cap_query, self.symbol, self.analysis_date)

        if not mc_result or not mc_result[0]['market_cap']:
            return None

        market_cap = float(mc_result[0]['market_cap'])

        # Get institutional net buying (60 days)
        inst_query = """
        SELECT SUM(inst_net_value) as inst_cumulative
        FROM kr_individual_investor_daily_trading
        WHERE symbol = $1
            AND date >= CURRENT_DATE - INTERVAL '60 days'
        """

        inst_result = await self.execute_query(inst_query, self.symbol)

        inst_cumulative = 0
        if inst_result and inst_result[0]['inst_cumulative']:
            inst_cumulative = float(inst_result[0]['inst_cumulative'])

        inst_ratio = (inst_cumulative / market_cap) * 100
        inst_score = min(50, max(0, inst_ratio * 10 + 25))

        # Get foreign ownership
        foreign_query = """
        SELECT foreign_rate
        FROM kr_foreign_ownership
        WHERE symbol = $1
            AND ($2::date IS NULL OR date = $2)
        ORDER BY date DESC
        LIMIT 1
        """

        foreign_result = await self.execute_query(foreign_query, self.symbol, self.analysis_date)

        foreign_score = 25  # Default
        if foreign_result and foreign_result[0]['foreign_rate']:
            foreign_rate = float(foreign_result[0]['foreign_rate'])
            foreign_score = min(50, foreign_rate * 2)

        final_score = inst_score + foreign_score

        return final_score

    # ========================================================================
    # Q14. Equity Ratio Strategy
    # ========================================================================

    async def calculate_q14(self):
        """
        Q14. Equity Ratio Strategy (Enhanced with BS fallback)
        Description: SCE에서 자본 정보 추출, 실패시 BS 위치 기반 추정
        SCE 최대값 = 자본일 가능성 (자본변동표의 특성)
        """
        query = """
        SELECT
            MAX(CASE WHEN sj_div = 'SCE' AND account_nm IN ('기말자본', '자본총계', '당기말자본')
                THEN thstrm_amount END) as sce_equity,
            MAX(CASE WHEN sj_div = 'BS' AND account_nm IN ('자산총계', '자산총액')
                THEN thstrm_amount END) as total_assets
        FROM kr_financial_position
        WHERE symbol = $1
            AND rcept_dt <= COALESCE($2::date, CURRENT_DATE)
            AND ((sj_div = 'SCE' AND account_nm IN ('기말자본', '자본총계', '당기말자본'))
                 OR (sj_div = 'BS' AND account_nm IN ('자산총계', '자산총액')))
        GROUP BY bsns_year
        ORDER BY bsns_year DESC, MAX(rcept_dt) DESC
        LIMIT 1
        """

        result = await self.execute_query(query, self.symbol, self.analysis_date)

        if not result or not result[0]['total_assets']:
            return None

        row = result[0]
        total_assets = float(row['total_assets'])
        equity = None

        # Try SCE equity first
        if row['sce_equity'] and float(row['sce_equity']) > 0:
            sce_equity = float(row['sce_equity'])
            equity_ratio_test = (sce_equity / total_assets) * 100

            # Sanity check: equity should be 20-80% of assets
            if 20 <= equity_ratio_test <= 80:
                equity = sce_equity

        # Fallback: Use BS to get equity directly
        if not equity:
            bs_query = """
            SELECT
                MAX(CASE WHEN account_nm IN ('자본총계', '자본', '기말자본') THEN thstrm_amount END) as bs_equity
            FROM kr_financial_position
            WHERE symbol = $1
                AND sj_div = 'BS'
                AND rcept_dt <= COALESCE($2::date, CURRENT_DATE)
                AND account_nm IN ('자본총계', '자본', '기말자본')
            GROUP BY bsns_year
            ORDER BY bsns_year DESC, MAX(rcept_dt) DESC
            LIMIT 1
            """

            bs_result = await self.execute_query(bs_query, self.symbol, self.analysis_date)

            if bs_result and bs_result[0]['bs_equity']:
                bs_equity = float(bs_result[0]['bs_equity'])
                equity_ratio_test = (bs_equity / total_assets) * 100
                # Sanity check: equity should be 20-80% of assets
                if 20 <= equity_ratio_test <= 80:
                    equity = bs_equity

        if not equity or equity <= 0:
            return None

        # Equity ratio
        equity_ratio = (equity / total_assets) * 100

        # Score calculation
        # > 70%: 100점
        # 50~70%: 80~100점
        # 30~50%: 50~80점
        # < 30%: 비례
        if equity_ratio >= 70:
            score = 100.0
        elif equity_ratio >= 50:
            score = 80 + (equity_ratio - 50) * 1.0
        elif equity_ratio >= 30:
            score = 50 + (equity_ratio - 30) * 1.5
        else:
            score = equity_ratio * (50 / 30)

        return min(100, max(0, score))

    # ========================================================================
    # Q15. Operating Profit Continuity Strategy
    # ========================================================================

    async def calculate_q15(self):
        """
        Q15. Operating Profit Continuity Strategy (Modified)
        Description: CIS에서 매출 다음으로 큰 양수 = 영업이익 추정
        Consecutive years of positive operating profit
        """
        query = """
        SELECT bsns_year,
               MAX(thstrm_amount) as operating_profit_estimate
        FROM kr_financial_position
        WHERE symbol = $1
            AND sj_div = 'IS'
            AND account_nm IN ('영업이익', '영업이익(손실)')
            AND rcept_dt <= COALESCE($2::date, CURRENT_DATE)
        GROUP BY bsns_year
        ORDER BY bsns_year DESC
        LIMIT 5
        """

        result = await self.execute_query(query, self.symbol, self.analysis_date)

        if not result:
            return None

        # Count consecutive profitable years
        consecutive_years = 0

        for row in result:
            if row['operating_profit_estimate'] and float(row['operating_profit_estimate']) > 0:
                consecutive_years += 1
            else:
                break

        if consecutive_years == 0:
            return None

        # Score calculation
        # 5년 연속 흑자: 100점
        # 4년 연속 흑자: 80점
        # 3년 연속 흑자: 60점
        # 2년 연속 흑자: 40점
        # 1년 흑자: 20점
        if consecutive_years >= 5:
            score = 100.0
        elif consecutive_years == 4:
            score = 80.0
        elif consecutive_years == 3:
            score = 60.0
        elif consecutive_years == 2:
            score = 40.0
        elif consecutive_years >= 1:
            score = 20.0
        else:
            score = 0.0

        return score

    # ========================================================================
    # Q16. Interest Coverage Ratio Strategy
    # ========================================================================

    async def calculate_q16(self):
        """
        Q16. Interest Coverage Ratio Strategy (Modified)
        Description: 영업이익(추정) 대비 부채 수준으로 이자 지급 능력 추정
        Estimated interest coverage using debt-based calculation
        """
        # Get operating income and total assets
        query = """
        SELECT
            MAX(CASE WHEN sj_div = 'IS' AND account_nm IN ('영업이익', '영업이익(손실)')
                THEN thstrm_amount END) as operating_income_est,
            MAX(CASE WHEN sj_div = 'BS' AND account_nm IN ('자산총계', '자산총액')
                THEN thstrm_amount END) as total_assets
        FROM kr_financial_position
        WHERE symbol = $1
            AND rcept_dt <= COALESCE($2::date, CURRENT_DATE)
            AND ((sj_div = 'IS' AND account_nm IN ('영업이익', '영업이익(손실)'))
                 OR (sj_div = 'BS' AND account_nm IN ('자산총계', '자산총액')))
        GROUP BY bsns_year
        ORDER BY bsns_year DESC, MAX(rcept_dt) DESC
        LIMIT 1
        """

        result = await self.execute_query(query, self.symbol, self.analysis_date)

        if not result or not result[0]['operating_income_est']:
            return None

        row = result[0]
        op_income = float(row['operating_income_est'])

        # Get debt info from SCE and BS
        debt_ratio_query = """
        SELECT
            MAX(CASE WHEN sj_div = 'SCE' AND account_nm IN ('기말자본', '자본총계', '당기말자본')
                THEN thstrm_amount END) as equity_est,
            MAX(CASE WHEN sj_div = 'BS' AND account_nm IN ('자산총계', '자산총액')
                THEN thstrm_amount END) as total_assets
        FROM kr_financial_position
        WHERE symbol = $1
            AND rcept_dt <= COALESCE($2::date, CURRENT_DATE)
            AND ((sj_div = 'SCE' AND account_nm IN ('기말자본', '자본총계', '당기말자본'))
                 OR (sj_div = 'BS' AND account_nm IN ('자산총계', '자산총액')))
        GROUP BY bsns_year
        ORDER BY bsns_year DESC, MAX(rcept_dt) DESC
        LIMIT 1
        """

        debt_result = await self.execute_query(debt_ratio_query, self.symbol, self.analysis_date)

        if debt_result and debt_result[0]['equity_est'] and debt_result[0]['total_assets']:
            equity = float(debt_result[0]['equity_est'])
            assets = float(debt_result[0]['total_assets'])
            debt = assets - equity

            if debt <= 0:
                # 부채 없음 = 이자 부담 없음
                return 100.0

            # 추정 이자비용 = 부채 * 3% (가정)
            estimated_interest = debt * 0.03

            if estimated_interest <= 0:
                return 100.0

            coverage_ratio = op_income / estimated_interest

            # Score calculation
            if coverage_ratio >= 20:
                score = 100.0
            elif coverage_ratio >= 10:
                score = 80 + (coverage_ratio - 10) * 2
            elif coverage_ratio >= 5:
                score = 60 + (coverage_ratio - 5) * 4
            elif coverage_ratio >= 3:
                score = 40 + (coverage_ratio - 3) * 10
            else:
                score = coverage_ratio * 13.3

            return min(100, max(0, score))

        # 부채 정보 없으면 영업이익 수준만으로 점수
        if row['total_assets']:
            assets = float(row['total_assets'])
            op_margin = (op_income / assets) * 100
            return min(100, op_margin * 10)

        return 50.0  # 기본 중립

    # ========================================================================
    # Q17. Operating Cash Flow Continuity Strategy
    # ========================================================================

    async def calculate_q17(self):
        """
        Q17. Operating Cash Flow Continuity Strategy (Enhanced with net income fallback)
        Description: CF 데이터 우선, 없으면 순이익 연속성으로 대체
        Primary: 영업활동현금흐름 계정명 직접 매칭
        Fallback: 순이익 연속성 (80% 점수)
        """
        # Primary: Try to get Operating Cash Flow from CF statement
        cf_query = """
        SELECT bsns_year,
               thstrm_amount
        FROM kr_financial_position
        WHERE symbol = $1
            AND sj_div = 'CF'
            AND (
                account_nm LIKE '%영업활동%현금흐름%'
                OR account_nm LIKE '%영업활동으로%현금%'
                OR account_nm = '영업활동현금흐름'
                OR account_nm = '영업활동으로인한현금흐름'
            )
            AND thstrm_amount IS NOT NULL
            AND rcept_dt <= COALESCE($2::date, CURRENT_DATE)
        ORDER BY bsns_year DESC, rcept_dt DESC
        LIMIT 5
        """

        result = await self.execute_query(cf_query, self.symbol, self.analysis_date)

        consecutive_years = 0
        use_fallback = False

        if result and len(result) > 0:
            # Count consecutive positive CF years
            for row in result:
                if row['thstrm_amount'] and float(row['thstrm_amount']) > 0:
                    consecutive_years += 1
                else:
                    break

        # Fallback: Use net income continuity if CF data insufficient
        if consecutive_years == 0:
            use_fallback = True
            net_income_query = """
            WITH cis_ranked AS (
                SELECT bsns_year,
                       thstrm_amount,
                       ROW_NUMBER() OVER (
                       PARTITION BY bsns_year
                       ORDER BY
                           CASE
                               WHEN report_code = '11011' THEN 1
                               WHEN report_code = '11012' THEN 2
                               WHEN report_code = '11013' THEN 3
                               WHEN report_code = '11014' THEN 4
                               ELSE 5
                           END,
                           thstrm_amount DESC
                   ) as rn
                FROM kr_financial_position
                WHERE symbol = $1
                    AND sj_div = 'IS'
                    AND thstrm_amount > 0
                    AND rcept_dt <= COALESCE($2::date, CURRENT_DATE)
            )
            SELECT bsns_year,
                   thstrm_amount as net_income_estimate
            FROM cis_ranked
            WHERE rn IN (2, 3, 4)  -- 매출 다음 큰 값들
            ORDER BY bsns_year DESC
            """

            ni_result = await self.execute_query(net_income_query, self.symbol, self.analysis_date)

            if ni_result and len(ni_result) > 0:
                # Group by year and pick largest per year
                years_dict = {}
                for row in ni_result:
                    year = row['bsns_year']
                    amount = float(row['net_income_estimate'])
                    if year not in years_dict or amount > years_dict[year]:
                        years_dict[year] = amount

                # Sort by year descending and count consecutive positives
                sorted_years = sorted(years_dict.items(), key=lambda x: x[0], reverse=True)[:5]

                for year, amount in sorted_years:
                    if amount > 0:
                        consecutive_years += 1
                    else:
                        break

        if consecutive_years == 0:
            return None

        # Score calculation
        # 5년 연속 양수: 100점
        # 4년 연속 양수: 80점
        # 3년 연속 양수: 60점
        # 2년 연속 양수: 40점
        # 1년 양수: 20점
        if consecutive_years >= 5:
            score = 100.0
        elif consecutive_years == 4:
            score = 80.0
        elif consecutive_years == 3:
            score = 60.0
        elif consecutive_years == 2:
            score = 40.0
        elif consecutive_years >= 1:
            score = 20.0
        else:
            score = 0.0

        # If using fallback (net income), reduce score to 80%
        if use_fallback:
            score = score * 0.8

        return score

    # ========================================================================
    # Q18. Sustainable Growth Rate (지속가능성장률) - Phase 3.8
    # ========================================================================

    async def calculate_q18_sustainable_growth_rate(self):
        """
        Q18. Sustainable Growth Rate Strategy (Phase 3.8 - Paradigm Shift)

        핵심 개념: "재무건전성" → "지속성장성" 패러다임 전환
        - QuantPedia 연구 기반: Quality Factor SGR 6.7% 연간 수익률
        - 지속가능성장률 = ROE × (1 - 배당성향)
        - 외부 자금 조달 없이 자체적으로 성장할 수 있는 비율

        Score calculation:
        1. ROE (50%): 수익성 기반
        2. Retention Rate (30%): 재투자 비율 = 1 - 배당성향
        3. SGR Value (20%): ROE × Retention Rate

        Higher SGR = Higher Score (성장 잠재력)

        Data: kr_financial_position, kr_stock_dividends
        """
        # Step 1: Get ROE (Net Income / Total Equity)
        roe_query = """
        SELECT
            bsns_year,
            MAX(CASE WHEN account_nm LIKE '%당기순이익%' OR account_nm = '당기순이익(손실)'
                THEN thstrm_amount END) as net_income,
            MAX(CASE WHEN account_nm LIKE '%자본총계%' OR account_nm = '자본총계'
                THEN thstrm_amount END) as total_equity
        FROM kr_financial_position
        WHERE symbol = $1
            AND sj_div IN ('CIS', 'BS')
            AND rcept_dt <= COALESCE($2::date, CURRENT_DATE)
        GROUP BY bsns_year
        ORDER BY bsns_year DESC
        LIMIT 3
        """

        roe_result = await self.execute_query(roe_query, self.symbol, self.analysis_date)

        if not roe_result:
            return None

        # Calculate ROE
        roe_values = []
        for row in roe_result:
            if row['net_income'] and row['total_equity'] and float(row['total_equity']) > 0:
                roe = (float(row['net_income']) / float(row['total_equity'])) * 100
                roe_values.append(roe)

        if not roe_values:
            return None

        current_roe = roe_values[0]
        avg_roe = sum(roe_values) / len(roe_values)

        # Step 2: Get Payout Ratio from kr_dividends (DART data)
        payout_query = """
        SELECT thstrm as payout_ratio
        FROM kr_dividends
        WHERE symbol = $1
            AND se = '(연결)현금배당성향(%)'
        ORDER BY stlm_dt DESC
        LIMIT 1
        """

        payout_result = await self.execute_query(payout_query, self.symbol)

        payout_ratio = 0.0
        if payout_result and payout_result[0].get('payout_ratio'):
            payout_ratio = float(payout_result[0]['payout_ratio'])

        # Retention Rate = 1 - Payout Ratio (cap at 100%)
        retention_rate = max(0, min(100, 100 - payout_ratio))

        # Step 3: Calculate Sustainable Growth Rate
        sgr = (current_roe / 100) * (retention_rate / 100) * 100  # As percentage

        # Step 4: Calculate component scores

        # ROE Score (50%): 0% = 0, 10% = 70, 20%+ = 100
        if current_roe >= 20:
            roe_score = 100
        elif current_roe <= 0:
            roe_score = 0
        else:
            roe_score = min(100, current_roe * 5)

        # Retention Rate Score (30%): 0% = 0, 50% = 70, 70%+ = 100
        # Too high retention (>90%) could mean no dividend = slight penalty
        if 50 <= retention_rate <= 80:
            retention_score = 100  # Sweet spot
        elif retention_rate >= 90:
            retention_score = 80  # Too much retention, no shareholder return
        elif retention_rate >= 30:
            retention_score = 50 + retention_rate  # Linear scale
        elif retention_rate > 0:
            retention_score = retention_rate * 1.67
        else:
            retention_score = 0

        # SGR Score (20%): 0% = 0, 8% = 70, 15%+ = 100
        if sgr >= 15:
            sgr_score = 100
        elif sgr <= 0:
            sgr_score = 0
        else:
            sgr_score = min(100, sgr * 6.67)

        # Base score
        score = roe_score * 0.50 + retention_score * 0.30 + sgr_score * 0.20

        logger.info(f"Q18: SGR (ROE: {current_roe:.1f}%, Retention: {retention_rate:.1f}%, SGR: {sgr:.1f}%, Score: {score:.1f})")

        return min(100, max(0, score))

    # ========================================================================
    # Q19. Accrual Quality (발생액 품질) - Phase 3.8
    # ========================================================================

    async def calculate_q19_accrual_quality(self):
        """
        Q19. Accrual Quality Strategy (Phase 3.8 - Paradigm Shift)

        핵심 개념: 이익의 질 측정
        - 영업현금흐름 / 당기순이익 비율
        - 높을수록 이익이 현금으로 뒷받침됨 (고품질)
        - 낮으면 회계상 이익만 있고 현금 유입 없음 (저품질)

        Score calculation:
        1. Accrual Ratio = Operating CF / Net Income
        2. >1.0: 100점 (현금이 이익보다 많음 = 보수적 회계)
        3. 0.8-1.0: 80점 (정상 범위)
        4. 0.5-0.8: 60점 (주의 필요)
        5. <0.5: 40점 이하 (저품질 이익)
        6. <0 or Net Income <0: 특수 처리

        Data: kr_financial_position (CIS: 당기순이익, CF: 영업활동현금흐름)
        """
        # Step 1: Get Net Income
        ni_query = """
        SELECT
            bsns_year,
            thstrm_amount as net_income
        FROM kr_financial_position
        WHERE symbol = $1
            AND sj_div = 'CIS'
            AND (account_nm LIKE '%당기순이익%' OR account_nm = '당기순이익(손실)')
            AND rcept_dt <= COALESCE($2::date, CURRENT_DATE)
        ORDER BY bsns_year DESC, rcept_dt DESC
        LIMIT 1
        """

        ni_result = await self.execute_query(ni_query, self.symbol, self.analysis_date)

        if not ni_result or not ni_result[0]['net_income']:
            return None

        net_income = float(ni_result[0]['net_income'])

        # Step 2: Get Operating Cash Flow
        cf_query = """
        SELECT thstrm_amount as operating_cf
        FROM kr_financial_position
        WHERE symbol = $1
            AND sj_div = 'CF'
            AND (
                account_nm LIKE '%영업활동%현금흐름%'
                OR account_nm LIKE '%영업활동으로%현금%'
                OR account_nm = '영업활동현금흐름'
            )
            AND rcept_dt <= COALESCE($2::date, CURRENT_DATE)
        ORDER BY bsns_year DESC, rcept_dt DESC
        LIMIT 1
        """

        cf_result = await self.execute_query(cf_query, self.symbol, self.analysis_date)

        if not cf_result or cf_result[0]['operating_cf'] is None:
            # No CF data = assume neutral
            return 50

        operating_cf = float(cf_result[0]['operating_cf'])

        # Step 3: Calculate Accrual Ratio and Score
        if net_income <= 0:
            # Net Income is negative
            if operating_cf > 0:
                # Loss but positive CF = turnaround potential
                score = 60
                logger.info(f"Q19: Loss but +CF (NI: {net_income/1e8:.1f}억, CF: {operating_cf/1e8:.1f}억, Score: {score})")
            else:
                # Both negative = poor quality
                score = 20
                logger.info(f"Q19: Both negative (NI: {net_income/1e8:.1f}억, CF: {operating_cf/1e8:.1f}억, Score: {score})")
        else:
            # Net Income is positive
            accrual_ratio = operating_cf / net_income

            if accrual_ratio >= 1.2:
                score = 100  # Excellent: CF > NI significantly
            elif accrual_ratio >= 1.0:
                score = 90  # Very good: CF >= NI
            elif accrual_ratio >= 0.8:
                score = 80  # Good: Normal range
            elif accrual_ratio >= 0.5:
                score = 60  # Caution: Some accruals
            elif accrual_ratio >= 0.2:
                score = 40  # Warning: High accruals
            elif accrual_ratio >= 0:
                score = 25  # Poor: Very low CF
            else:
                score = 10  # Negative CF with positive NI = red flag

            logger.info(f"Q19: Accrual (NI: {net_income/1e8:.1f}억, CF: {operating_cf/1e8:.1f}억, Ratio: {accrual_ratio:.2f}, Score: {score})")

        return min(100, max(0, score))

    # ========================================================================
    # Q20. Inventory Efficiency Change (재고효율성 변화) - Phase 3.8
    # ========================================================================

    async def calculate_q20_inventory_efficiency(self):
        """
        Q20. Inventory Efficiency Change Strategy (Phase 3.8 - User Suggestion)

        핵심 개념: 재고자산회전율 YoY 변화
        - 재고자산회전율 = 매출원가 / 평균재고자산
        - 증가 = 운영 효율성 개선 (고점수)
        - 감소 = 재고 누적, 판매 부진 (저점수)

        Score calculation:
        1. Current turnover vs Previous turnover
        2. Improvement = Higher score
        3. No inventory = Neutral (서비스업 등)

        Data: kr_financial_position (BS: 재고자산, CIS: 매출원가)
        """
        # Step 1: Get Inventory and COGS for 2 years
        data_query = """
        SELECT
            bsns_year,
            MAX(CASE WHEN sj_div = 'BS' AND account_nm LIKE '%재고자산%'
                THEN thstrm_amount END) as inventory,
            MAX(CASE WHEN sj_div = 'CIS' AND (account_nm LIKE '%매출원가%' OR account_nm = '매출원가')
                THEN thstrm_amount END) as cogs
        FROM kr_financial_position
        WHERE symbol = $1
            AND rcept_dt <= COALESCE($2::date, CURRENT_DATE)
        GROUP BY bsns_year
        ORDER BY bsns_year DESC
        LIMIT 3
        """

        result = await self.execute_query(data_query, self.symbol, self.analysis_date)

        if not result or len(result) < 2:
            # Not enough data for comparison
            return 50  # Neutral

        # Calculate inventory turnover for each year
        turnover_values = []
        for row in result:
            inventory = float(row['inventory']) if row['inventory'] else 0
            cogs = float(row['cogs']) if row['cogs'] else 0

            if inventory > 0 and cogs > 0:
                turnover = cogs / inventory
                turnover_values.append({
                    'year': row['bsns_year'],
                    'turnover': turnover
                })

        if len(turnover_values) < 2:
            # No inventory (service company) = neutral
            logger.info(f"Q20: No inventory data (service industry?), Score: 50")
            return 50

        # Calculate YoY change
        current_turnover = turnover_values[0]['turnover']
        previous_turnover = turnover_values[1]['turnover']

        if previous_turnover == 0:
            return 50  # Cannot calculate change

        turnover_change_pct = ((current_turnover - previous_turnover) / previous_turnover) * 100

        # Step 2: Calculate Score
        # +20%+ improvement = 100
        # +10% improvement = 80
        # 0% (stable) = 60
        # -10% deterioration = 40
        # -20%+ deterioration = 20

        if turnover_change_pct >= 20:
            score = 100  # Significant improvement
        elif turnover_change_pct >= 10:
            score = 80 + (turnover_change_pct - 10) * 2
        elif turnover_change_pct >= 0:
            score = 60 + turnover_change_pct * 2
        elif turnover_change_pct >= -10:
            score = 60 + turnover_change_pct * 2  # 40-60 range
        elif turnover_change_pct >= -20:
            score = 40 + (turnover_change_pct + 10) * 2
        else:
            score = 20  # Significant deterioration

        logger.info(f"Q20: Inventory (Turnover: {current_turnover:.2f} vs {previous_turnover:.2f}, Change: {turnover_change_pct:.1f}%, Score: {score:.1f})")

        return min(100, max(0, score))

    # ========================================================================
    # Q21. 5-Year ROA Trend (장기 수익성 추세) - Phase 3.8
    # ========================================================================

    async def calculate_q21_roa_trend(self):
        """
        Q21. 5-Year ROA Trend Strategy (Phase 3.8 - QuantPedia Research)

        핵심 개념: 장기 ROA 추세로 지속적 수익성 평가
        - QuantPedia: ROA 5yr 연간 수익률 4.7%
        - 단기 변동보다 장기 추세 중시
        - ROA = 순이익 / 총자산

        Score calculation:
        1. 5년간 ROA 기울기 (추세)
        2. 상승 추세 = 고점수
        3. 하락 추세 = 저점수
        4. 최근 ROA 레벨도 함께 고려

        Data: kr_financial_position
        """
        # Step 1: Get 5 years of ROA data
        roa_query = """
        SELECT
            bsns_year,
            MAX(CASE WHEN sj_div IN ('CIS', 'IS') AND
                (account_nm LIKE '%당기순이익%' OR account_nm = '당기순이익(손실)')
                THEN thstrm_amount END) as net_income,
            MAX(CASE WHEN sj_div = 'BS' AND
                (account_nm LIKE '%자산총계%' OR account_nm = '자산총계')
                THEN thstrm_amount END) as total_assets
        FROM kr_financial_position
        WHERE symbol = $1
            AND rcept_dt <= COALESCE($2::date, CURRENT_DATE)
        GROUP BY bsns_year
        ORDER BY bsns_year DESC
        LIMIT 5
        """

        result = await self.execute_query(roa_query, self.symbol, self.analysis_date)

        if not result or len(result) < 2:
            return None

        # Calculate ROA for each year
        roa_data = []
        for row in result:
            if row['net_income'] and row['total_assets'] and float(row['total_assets']) > 0:
                roa = (float(row['net_income']) / float(row['total_assets'])) * 100
                roa_data.append({
                    'year': int(row['bsns_year']),
                    'roa': roa
                })

        if len(roa_data) < 2:
            return None

        # Sort by year ascending for trend calculation
        roa_data.sort(key=lambda x: x['year'])

        # Step 2: Calculate trend (simple linear regression slope)
        n = len(roa_data)
        x_values = list(range(n))  # 0, 1, 2, ...
        y_values = [d['roa'] for d in roa_data]

        x_mean = sum(x_values) / n
        y_mean = sum(y_values) / n

        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values))
        denominator = sum((x - x_mean) ** 2 for x in x_values)

        if denominator == 0:
            slope = 0
        else:
            slope = numerator / denominator  # ROA change per year

        # Current ROA (most recent)
        current_roa = roa_data[-1]['roa']

        # Step 3: Calculate scores

        # Trend Score (60%): -2%/yr = 0, 0%/yr = 50, +2%/yr = 100
        if slope >= 2:
            trend_score = 100
        elif slope <= -2:
            trend_score = 0
        else:
            trend_score = 50 + (slope * 25)

        # Level Score (40%): Current ROA level
        # 0% = 0, 5% = 50, 10%+ = 100
        if current_roa >= 10:
            level_score = 100
        elif current_roa <= 0:
            level_score = 0
        else:
            level_score = current_roa * 10

        # Base score
        score = trend_score * 0.60 + level_score * 0.40

        # Bonus: Consistently improving (all positive changes)
        is_improving = all(roa_data[i]['roa'] <= roa_data[i+1]['roa'] for i in range(len(roa_data)-1))
        if is_improving and len(roa_data) >= 3:
            score = min(100, score * 1.1)  # 10% bonus

        logger.info(f"Q21: ROA Trend (Current: {current_roa:.1f}%, Slope: {slope:.2f}%/yr, Score: {score:.1f})")

        return min(100, max(0, score))


    # ========================================================================
    # Calculate All Quality Factor Scores
    # ========================================================================

    async def calculate_all_strategies(self):
        """
        Calculate all quality factor strategies (Phase 3.8 Upgrade - 20 strategies)

        Q5 (Asset Efficiency) REMOVED on 2025-11-19:
        - Multiple redesign attempts failed (V1, V2, V3)
        - V2 (ROA-Centered): IC -0.0081
        - V3 (Trend-Based): IC -0.0174 (worse)
        - No predictive power across all versions

        NEW Phase 3.8 Strategies (Sustainable Growth Paradigm):
        - Q18: Sustainable Growth Rate (SGR = ROE × Retention)
        - Q19: Accrual Quality (Operating CF / Net Income)
        - Q20: Inventory Efficiency Change (YoY Turnover)
        - Q21: 5-Year ROA Trend (Long-term Profitability)

        Returns: dict of {strategy_name: score}
        """
        logger.info(f"Calculating all quality factor strategies for {self.symbol}")

        strategies = {
            'Q1_ROE_Consistency': await self.calculate_q1(),
            'Q2_Operating_Margin_Excellence': await self.calculate_q2(),
            'Q3_Debt_Stability': await self.calculate_q3(),
            'Q4_Cash_Reserve_Adequacy': await self.calculate_q4(),
            'Q5_Asset_Efficiency': None,  # REMOVED - Failed IC validation (multiple attempts)
            'Q6_Cash_Generation': await self.calculate_q6(),
            'Q7_Dividend_Sustainability': await self.calculate_q7(),
            'Q8_Sales_Stability': await self.calculate_q8(),
            'Q9_Profitability_Excellence': await self.calculate_q9(),
            'Q10_Governance_Quality': await self.calculate_q10(),
            'Q11_Audit_Reliability': await self.calculate_q11(),
            'Q12_Working_Capital_Efficiency': await self.calculate_q12(),
            'Q13_Supply_Demand_Quality': await self.calculate_q13(),
            'Q14_Equity_Ratio': await self.calculate_q14(),
            'Q15_Operating_Profit_Continuity': await self.calculate_q15(),
            'Q16_Interest_Coverage_Ratio': await self.calculate_q16(),
            'Q17_Operating_CF_Continuity': await self.calculate_q17(),
            # NEW Phase 3.8: Sustainable Growth Paradigm
            'Q18_Sustainable_Growth_Rate': await self.calculate_q18_sustainable_growth_rate(),
            'Q19_Accrual_Quality': await self.calculate_q19_accrual_quality(),
            'Q20_Inventory_Efficiency': await self.calculate_q20_inventory_efficiency(),
            'Q21_ROA_Trend': await self.calculate_q21_roa_trend(),
        }

        self.strategies_scores = strategies

        return strategies

    async def calculate_comprehensive_score(self):
        """
        Calculate comprehensive quality factor score
        Formula: Comprehensive Score = Σ(Strategy Score)

        Returns:
            float: Comprehensive score (sum of all valid strategy scores)
        """
        if not self.strategies_scores:
            await self.calculate_all_strategies()

        # Filter out None scores
        valid_scores = {k: v for k, v in self.strategies_scores.items() if v is not None}

        if not valid_scores:
            logger.warning(f"No valid strategy scores for {self.symbol}")
            return None

        # Calculate simple sum
        comprehensive_score = sum(valid_scores.values())

        return comprehensive_score

    async def calculate_weighted_score(self, market_state=None):
        """
        Calculate weighted average score based on market state

        Formula:
            Weighted Average = Σ(Strategy Score × Weight) / Σ(Weight)

        Example (KOSPI대형-확장과열-공격형):
            Weights: Q1=0.8, Q2=1.0, Q3=0.5, ..., Q17=0.9 (Sum=13.6)
            Scores: Q1=85, Q2=72, Q3=65, ..., Q17=68
            Weighted Average = (85×0.8 + 72×1.0 + ... + 68×0.9) / 13.6 = 72.2

        Args:
            market_state: Market state classification (one of 19 states)
                         If None, uses self.market_state

        Returns:
            dict: {
                'weighted_score': float (0~100),
                'weight_sum': float,
                'market_state': str,
                'valid_strategies': int
            }
            Returns None if no valid scores or weights
        """
        # Use provided market_state or instance variable
        if market_state is None:
            market_state = self.market_state

        # Default to '기타' if no market state provided
        if market_state is None:
            market_state = '기타'
            logger.warning(f"No market state provided for {self.symbol}, using '기타'")

        # Calculate strategy scores if not already done
        if not self.strategies_scores:
            await self.calculate_all_strategies()

        # Get weights for this market state
        weights = QUALITY_STRATEGY_WEIGHTS.get(market_state)

        if weights is None:
            logger.error(f"Invalid market state: {market_state}, using '기타'")
            weights = QUALITY_STRATEGY_WEIGHTS['기타']
            market_state = '기타'

        # Calculate weighted average
        weighted_sum = 0.0
        weight_sum = 0.0
        valid_count = 0

        for strategy_name, score in self.strategies_scores.items():
            if score is not None:
                # Validate strategy score range
                if abs(score) > 100:
                    logger.error(f"[{self.symbol}] ABNORMAL STRATEGY: {strategy_name} = {score:.2f} (expected: 0-100)")
                elif score < 0:
                    logger.warning(f"[{self.symbol}] NEGATIVE STRATEGY: {strategy_name} = {score:.2f}")

                # Extract strategy key (e.g., 'Q1_ROE' -> 'Q1')
                strategy_key = strategy_name.split('_')[0]
                weight = weights.get(strategy_key, 1.0)

                weighted_sum += float(score) * weight
                weight_sum += weight
                valid_count += 1

        # Normalize by weight sum
        if weight_sum > 0 and valid_count > 0:
            final_score = weighted_sum / weight_sum

            # Calculate base score (simple average without weights - for Refactoring)
            valid_scores = [score for score in self.strategies_scores.values() if score is not None]
            base_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0

            result = {
                'weighted_score': round(final_score, 1),
                'base_score': round(base_score, 1),  # Added for Refactoring
                'weight_sum': round(weight_sum, 2),
                'market_state': market_state,
                'valid_strategies': valid_count,
                'strategies': self.strategies_scores
            }

            logger.info(f"Weighted quality score for {self.symbol}: {final_score:.2f} "
                       f"(market_state: {market_state}, weight_sum: {weight_sum:.2f})")

            return result
        else:
            logger.warning(f"No valid scores for weighted calculation: {self.symbol}")
            return None



async def main():
    """Main execution function"""
    print("\n" + "="*80)
    print("Korean Stock Quality Factor System (Enhanced with Market State Weighting)")
    print("="*80 + "\n")

    # Get symbol input
    symbol = input("Enter stock symbol: ").strip()

    if not symbol:
        print("No symbol entered.")
        return

    try:
        # Step 1: Determine market state using weight.py
        print("\nStep 1: Analyzing market conditions...")
        try:
            from weight import ConditionAnalyzer
            analyzer = ConditionAnalyzer(symbol)
            conditions, _ = analyzer.analyze()
            market_state = conditions.get('market_state', '기타')
            print(f"Market State: {market_state}")
        except Exception as e:
            logger.warning(f"Could not determine market state: {e}")
            market_state = '기타'
            print(f"Market State: {market_state} (default)")

        # Step 2: Create calculator with market state
        print("\nStep 2: Calculating quality factor strategies...")
        from db_async import AsyncDatabaseManager
        db_manager = AsyncDatabaseManager()
        await db_manager.initialize()

        calculator = QualityFactorCalculator(symbol, db_manager, market_state=market_state)

        # Calculate all strategies
        strategies = await calculator.calculate_all_strategies()

        # Get weights for current market state
        weights = QUALITY_STRATEGY_WEIGHTS.get(market_state, QUALITY_STRATEGY_WEIGHTS['기타'])

        # Display results
        print("\n" + "="*80)
        print(f"Quality Factor Strategies for {symbol}")
        print(f"Market State: {market_state}")
        print("="*80 + "\n")

        print(f"{'Strategy':<40s} {'Score':>8s} {'Weight':>8s} {'Weighted':>10s}")
        print("-"*80)

        for strategy_name, score in strategies.items():
            strategy_key = strategy_name.split('_')[0]
            weight = weights.get(strategy_key, 1.0)

            if score is not None:
                weighted = score * weight
                print(f"{strategy_name:<40s} {score:8.2f} {weight:8.2f} {weighted:10.2f}")
            else:
                print(f"{strategy_name:<40s} {'N/A':>8s} {weight:8.2f} {'N/A':>10s}")

        # Calculate scores
        comprehensive_score = await calculator.calculate_comprehensive_score()
        weighted_result = await calculator.calculate_weighted_score()

        # Display final scores
        print("\n" + "="*80)
        print("Final Scores")
        print("="*80)
        if comprehensive_score:
            print(f"Simple Sum (unweighted):        {comprehensive_score:.2f} points")
        else:
            print("Simple Sum (unweighted):        N/A")

        if weighted_result:
            print(f"\n[Market State-based Weighted Score]")
            print(f"Market State:                   {weighted_result['market_state']}")
            print(f"Weight Sum:                     {weighted_result['weight_sum']:.2f}")
            print(f"Valid Strategies:               {weighted_result['valid_strategies']}/17")
            print(f"Weighted Average Score:         {weighted_result['weighted_score']:.2f} / 100")
        else:
            print("\nWeighted score calculation failed.")

        print("="*80 + "\n")

        await db_manager.close()

    except Exception as e:
        logger.error(f"Error occurred: {e}")
        print(f"\nError: {e}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
