"""
Korean Stock Growth Factor Calculator

성장(Growth) 팩터 분석을 위한 계산기
18개 성장 전략 구현
"""

import os
import logging
from datetime import datetime, timedelta

# MarketClassifier import
try:
    from market_classifier import MarketClassifier
except ImportError:
    from kr.market_classifier import MarketClassifier

# 로거 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Growth Strategy Weights by Market State
GROWTH_STRATEGY_WEIGHTS = {
    'KOSPI대형-확장과열-공격형': {
        'G1': 1.8, 'G2': 1.6, 'G3': 1.7, 'G4': 1.4, 'G5': 1.5, 'G6': 1.3,
        'G7': 1.6, 'G8': 1.5, 'G9': 1.4, 'G10': 1.3, 'G11': 0.8, 'G12': 1.2,
        'G13': 1.6, 'G14': 1.5, 'G15': 1.4, 'G16': 1.3, 'G17': 1.2, 'G18': 0.9
    },
    'KOSPI대형-확장과열-방어형': {
        'G1': 1.3, 'G2': 1.2, 'G3': 1.4, 'G4': 1.3, 'G5': 1.2, 'G6': 1.1,
        'G7': 1.0, 'G8': 1.0, 'G9': 1.3, 'G10': 1.1, 'G11': 1.5, 'G12': 1.2,
        'G13': 0.9, 'G14': 1.2, 'G15': 1.0, 'G16': 1.3, 'G17': 1.1, 'G18': 1.0
    },
    'KOSPI대형-침체조정-가치형': {
        'G1': 0.9, 'G2': 1.0, 'G3': 1.1, 'G4': 1.6, 'G5': 1.0, 'G6': 1.1,
        'G7': 0.8, 'G8': 0.8, 'G9': 1.2, 'G10': 0.9, 'G11': 1.4, 'G12': 1.0,
        'G13': 0.7, 'G14': 1.3, 'G15': 0.8, 'G16': 1.3, 'G17': 1.1, 'G18': 1.7
    },
    'KOSPI대형-침체조정-배당형': {
        'G1': 0.8, 'G2': 0.9, 'G3': 1.0, 'G4': 1.4, 'G5': 0.9, 'G6': 1.0,
        'G7': 0.7, 'G8': 0.7, 'G9': 1.1, 'G10': 0.8, 'G11': 1.8, 'G12': 1.1,
        'G13': 0.6, 'G14': 1.2, 'G15': 0.7, 'G16': 1.2, 'G17': 1.0, 'G18': 1.5
    },
    'KOSPI대형-금융위기-방어형': {
        'G1': 0.7, 'G2': 0.8, 'G3': 0.9, 'G4': 1.5, 'G5': 0.8, 'G6': 0.9,
        'G7': 0.6, 'G8': 0.6, 'G9': 1.3, 'G10': 0.7, 'G11': 1.9, 'G12': 1.2,
        'G13': 0.5, 'G14': 1.1, 'G15': 0.6, 'G16': 1.4, 'G17': 1.1, 'G18': 1.6
    },
    'KOSPI대형-금융위기-초가치형': {
        'G1': 0.6, 'G2': 0.7, 'G3': 0.8, 'G4': 1.8, 'G5': 0.7, 'G6': 0.8,
        'G7': 0.5, 'G8': 0.5, 'G9': 1.4, 'G10': 0.6, 'G11': 2.0, 'G12': 1.3,
        'G13': 0.5, 'G14': 1.2, 'G15': 0.5, 'G16': 1.5, 'G17': 1.2, 'G18': 1.9
    },
    'KOSPI중형-확장과열-모멘텀형': {
        'G1': 1.6, 'G2': 1.5, 'G3': 1.6, 'G4': 1.3, 'G5': 1.4, 'G6': 1.2,
        'G7': 1.5, 'G8': 1.4, 'G9': 0.4, 'G10': 1.5, 'G11': 0.9, 'G12': 0.8,
        'G13': 1.7, 'G14': 1.6, 'G15': 1.7, 'G16': 1.8, 'G17': 1.7, 'G18': 1.0
    },
    'KOSPI중형-확장과열-성장형': {
        'G1': 1.7, 'G2': 1.6, 'G3': 1.8, 'G4': 1.4, 'G5': 1.5, 'G6': 1.4,
        'G7': 1.6, 'G8': 1.5, 'G9': 0.4, 'G10': 1.4, 'G11': 1.0, 'G12': 0.8,
        'G13': 1.8, 'G14': 1.7, 'G15': 1.5, 'G16': 1.6, 'G17': 1.5, 'G18': 1.1
    },
    'KOSPI중형-침체조정-가치형': {
        'G1': 1.0, 'G2': 1.1, 'G3': 1.2, 'G4': 1.6, 'G5': 1.1, 'G6': 1.2,
        'G7': 0.9, 'G8': 0.9, 'G9': 0.7, 'G10': 1.0, 'G11': 1.5, 'G12': 1.0,
        'G13': 0.8, 'G14': 1.4, 'G15': 0.9, 'G16': 1.4, 'G17': 1.2, 'G18': 1.7
    },
    'KOSPI중형-침체조정-배당형': {
        'G1': 0.9, 'G2': 1.0, 'G3': 1.1, 'G4': 1.5, 'G5': 1.0, 'G6': 1.1,
        'G7': 0.8, 'G8': 0.8, 'G9': 0.6, 'G10': 0.9, 'G11': 1.7, 'G12': 1.0,
        'G13': 0.7, 'G14': 1.3, 'G15': 0.8, 'G16': 1.3, 'G17': 1.1, 'G18': 1.6
    },
    'KOSPI중형-핫섹터-고위험형': {
        'G1': 1.5, 'G2': 1.4, 'G3': 1.5, 'G4': 1.2, 'G5': 1.6, 'G6': 1.3,
        'G7': 1.4, 'G8': 1.3, 'G9': 0.3, 'G10': 1.5, 'G11': 0.8, 'G12': 0.5,
        'G13': 1.6, 'G14': 1.5, 'G15': 1.8, 'G16': 1.7, 'G17': 1.6, 'G18': 1.2
    },
    'KOSPI중형-정상주기-균형형': {
        'G1': 1.2, 'G2': 1.2, 'G3': 1.3, 'G4': 1.3, 'G5': 1.2, 'G6': 1.2,
        'G7': 1.1, 'G8': 1.1, 'G9': 0.4, 'G10': 1.1, 'G11': 1.2, 'G12': 0.8,
        'G13': 1.1, 'G14': 1.3, 'G15': 1.2, 'G16': 1.3, 'G17': 1.2, 'G18': 1.2
    },
    'KOSDAQ소형-핫섹터-초고위험형': {
        'G1': 1.4, 'G2': 1.3, 'G3': 1.4, 'G4': 1.1, 'G5': 1.7, 'G6': 1.2,
        'G7': 1.5, 'G8': 1.4, 'G9': 0.2, 'G10': 1.6, 'G11': 0.7, 'G12': 0.3,
        'G13': 1.8, 'G14': 1.6, 'G15': 1.9, 'G16': 1.8, 'G17': 1.7, 'G18': 1.3
    },
    'KOSDAQ소형-핫섹터-고성장형': {
        'G1': 1.8, 'G2': 1.7, 'G3': 1.9, 'G4': 1.3, 'G5': 1.6, 'G6': 1.5,
        'G7': 1.7, 'G8': 1.6, 'G9': 0.3, 'G10': 1.5, 'G11': 0.9, 'G12': 0.5,
        'G13': 1.9, 'G14': 1.8, 'G15': 1.6, 'G16': 1.7, 'G17': 1.6, 'G18': 1.2
    },
    'KOSDAQ소형-정상주기-저유동성형': {
        'G1': 1.1, 'G2': 1.1, 'G3': 1.2, 'G4': 1.4, 'G5': 1.2, 'G6': 1.1,
        'G7': 1.0, 'G8': 1.0, 'G9': 0.3, 'G10': 1.2, 'G11': 1.3, 'G12': 0.7,
        'G13': 1.0, 'G14': 1.3, 'G15': 1.1, 'G16': 1.4, 'G17': 1.3, 'G18': 1.4
    },
    'KOSDAQ소형-정상주기-기술형': {
        'G1': 1.5, 'G2': 1.4, 'G3': 1.6, 'G4': 1.3, 'G5': 1.4, 'G6': 1.3,
        'G7': 1.7, 'G8': 1.6, 'G9': 0.4, 'G10': 1.3, 'G11': 1.0, 'G12': 0.8,
        'G13': 1.5, 'G14': 1.5, 'G15': 1.3, 'G16': 1.5, 'G17': 1.4, 'G18': 1.2
    },
    '전시장-극저유동성-고위험형': {
        'G1': 0.8, 'G2': 0.9, 'G3': 1.0, 'G4': 1.5, 'G5': 0.9, 'G6': 0.9,
        'G7': 0.7, 'G8': 0.7, 'G9': 0.8, 'G10': 0.8, 'G11': 1.6, 'G12': 1.2,
        'G13': 0.6, 'G14': 1.2, 'G15': 0.7, 'G16': 1.5, 'G17': 1.4, 'G18': 1.6
    },
    '테마특화-모멘텀폭발형': {
        'G1': 1.6, 'G2': 1.5, 'G3': 1.6, 'G4': 1.2, 'G5': 1.7, 'G6': 1.3,
        'G7': 1.5, 'G8': 1.4, 'G9': 0.2, 'G10': 1.7, 'G11': 0.8, 'G12': 0.3,
        'G13': 1.7, 'G14': 1.7, 'G15': 1.9, 'G16': 1.9, 'G17': 1.8, 'G18': 1.3
    },
    '기타': {
        'G1': 1.0, 'G2': 1.0, 'G3': 1.0, 'G4': 1.0, 'G5': 1.0, 'G6': 1.0,
        'G7': 1.0, 'G8': 1.0, 'G9': 1.0, 'G10': 1.0, 'G11': 1.0, 'G12': 1.0,
        'G13': 1.0, 'G14': 1.0, 'G15': 1.0, 'G16': 1.0, 'G17': 1.0, 'G18': 1.0
    }
}


class GrowthFactorCalculator:
    """
    Growth factor strategy calculator for Korean stocks

    18 growth strategies implemented:
    G1-G18 covering revenue, profit, asset, market growth, liquidity, and valuation metrics
    """

    def __init__(self, symbol, db_manager, market_state=None, analysis_date=None):
        """
        Initialize Growth Factor Calculator

        Args:
            symbol: Stock symbol (e.g., '005930' for Samsung)
            db_manager: AsyncDatabaseManager instance
            market_state: Market state classification (optional)
            analysis_date: Specific date for analysis (optional, defaults to latest)
        """
        self.symbol = symbol
        self.db_manager = db_manager
        self.market_state = market_state
        self.analysis_date = analysis_date

    async def execute_query(self, query, *params):
        """
        Execute SQL query using async database manager

        Args:
            query: SQL query string
            *params: Query parameters (unpacked)

        Returns:
            List of dict rows or None if error
        """
        try:
            return await self.db_manager.execute_query(query, *params)
        except Exception as e:
            logger.error(f"Query execution error: {e}")
            return None

    async def _get_market_cap_category(self):
        """
        Get market cap category for current symbol

        Returns:
            str: '대형', '중형', '소형', or '기타'
        """
        query = """
        SELECT market_cap
        FROM kr_intraday_total
        WHERE symbol = $1
        ORDER BY date DESC
        LIMIT 1
        """
        result = await self.execute_query(query, self.symbol)
        if not result or not result[0]['market_cap']:
            return '기타'

        market_cap = float(result[0]['market_cap'])

        if market_cap >= 10_000_000_000_000:  # 10조
            return '대형'
        elif market_cap >= 1_000_000_000_000:  # 1조
            return '중형'
        else:
            return '소형'

    async def calculate_g1(self):
        """
        G1. Revenue Growth Strategy (with fallback)
        Description: Sustained and accelerating revenue growth trend
        Fallback: Uses trading value growth if financial data unavailable
        """
        # Try financial position data first
        query_financial = """
        WITH revenue_data AS (
            SELECT
                thstrm_amount as current_revenue,
                frmtrm_amount as prev_revenue,
                bfefrmtrm_amount as prev2_revenue
            FROM kr_financial_position
            WHERE symbol = $1
                AND sj_div = 'IS'
                AND account_nm = '매출액'
                AND frmtrm_amount IS NOT NULL
                AND bfefrmtrm_amount IS NOT NULL
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
        )
        SELECT
            current_revenue,
            prev_revenue,
            prev2_revenue,
            CASE
                WHEN prev_revenue != 0
                THEN ((current_revenue::NUMERIC - prev_revenue) / ABS(prev_revenue) * 100)
                ELSE NULL
            END as growth_1y,
            CASE
                WHEN prev2_revenue != 0 AND prev_revenue != 0
                THEN ((prev_revenue::NUMERIC - prev2_revenue) / ABS(prev2_revenue) * 100)
                ELSE NULL
            END as growth_2y,
            CASE
                WHEN prev2_revenue > 0 AND current_revenue > 0
                THEN (POWER(current_revenue::NUMERIC / prev2_revenue::NUMERIC, 0.5) - 1) * 100
                ELSE NULL
            END as cagr_3y
        FROM revenue_data
        """

        result = await self.execute_query(query_financial, self.symbol, self.analysis_date)

        # Fallback to trading value growth if financial data is invalid
        # (negative revenue causes cagr_3y to be NULL)
        if not result or not result[0] or result[0]['growth_1y'] is None or result[0]['cagr_3y'] is None:
            query_fallback = """
            WITH trading_data AS (
                SELECT
                    AVG(trading_value) as avg_trading
                FROM kr_intraday_total
                WHERE symbol = $1
                    AND date >= CURRENT_DATE - INTERVAL '90 days'
                    AND trading_value IS NOT NULL
            ),
            trading_90d AS (
                SELECT
                    AVG(trading_value) as avg_trading_90d
                FROM kr_intraday_total
                WHERE symbol = $2
                    AND date >= CURRENT_DATE - INTERVAL '180 days'
                    AND date < CURRENT_DATE - INTERVAL '90 days'
                    AND trading_value IS NOT NULL
            ),
            trading_180d AS (
                SELECT
                    AVG(trading_value) as avg_trading_180d
                FROM kr_intraday_total
                WHERE symbol = $3
                    AND date >= CURRENT_DATE - INTERVAL '270 days'
                    AND date < CURRENT_DATE - INTERVAL '180 days'
                    AND trading_value IS NOT NULL
            )
            SELECT
                td.avg_trading,
                t90.avg_trading_90d,
                t180.avg_trading_180d,
                CASE WHEN t90.avg_trading_90d > 0
                    THEN ((td.avg_trading - t90.avg_trading_90d)::NUMERIC / t90.avg_trading_90d * 100)
                    ELSE NULL
                END as growth_recent,
                CASE WHEN t180.avg_trading_180d > 0
                    THEN ((t90.avg_trading_90d - t180.avg_trading_180d)::NUMERIC / t180.avg_trading_180d * 100)
                    ELSE NULL
                END as growth_prev,
                CASE WHEN t180.avg_trading_180d > 0 AND td.avg_trading > 0
                    THEN ((td.avg_trading - t180.avg_trading_180d)::NUMERIC / t180.avg_trading_180d * 100)
                    ELSE NULL
                END as growth_overall
            FROM trading_data td
            CROSS JOIN trading_90d t90
            CROSS JOIN trading_180d t180
            """

            result = await self.execute_query(query_fallback, self.symbol, self.symbol, self.symbol)

            if not result or not result[0] or result[0]['growth_recent'] is None:
                return None

            row = result[0]
            growth_recent = float(row['growth_recent'] or 0)
            growth_prev = float(row['growth_prev'] or 0)
            growth_overall = float(row['growth_overall'] or 0)

            # Acceleration
            acceleration = 20 if growth_recent > growth_prev else 0

            # Growth score (overall growth based)
            growth_score = min(100, max(0, growth_overall / 2 + 50))

            # Acceleration score
            accel_score = min(100, max(0, (growth_recent - growth_prev) + 50))

            score = (growth_score * 0.7) + (accel_score * 0.3)
            return round(score, 2)

        row = result[0]
        growth_1y = float(row['growth_1y'])  # Guaranteed non-NULL by Line 250 check
        growth_2y = float(row['growth_2y']) if row['growth_2y'] is not None else 0
        cagr_3y = float(row['cagr_3y'])  # Guaranteed non-NULL by Line 250 check

        # Acceleration bonus
        acceleration = 20 if growth_1y > growth_2y else 0

        # Growth score (3Y CAGR based)
        growth_score = min(100, max(0, cagr_3y * 3))

        # Acceleration score
        accel_score = min(100, max(0, (growth_1y - growth_2y) * 2 + 50))

        score = (growth_score * 0.7) + (accel_score * 0.3)

        return round(score, 2)

    async def calculate_g2(self):
        """
        G2. Market Cap Growth Strategy (Modified - replaces Operating Profit Growth)
        Description: Market capitalization growth as proxy for business value growth
        Operating profit data unavailable, using market cap as alternative
        """
        query = """
        WITH market_cap_data AS (
            SELECT
                market_cap,
                LAG(market_cap, 90) OVER (ORDER BY date) as mc_90d,
                LAG(market_cap, 180) OVER (ORDER BY date) as mc_180d,
                ROW_NUMBER() OVER (ORDER BY date DESC) as rn
            FROM kr_intraday_total
            WHERE symbol = $1
                AND market_cap IS NOT NULL
        )
        SELECT
            market_cap as current_mc,
            mc_90d,
            mc_180d,
            CASE
                WHEN mc_90d != 0
                THEN ((market_cap::NUMERIC - mc_90d) / ABS(mc_90d) * 100)
                ELSE NULL
            END as growth_90d,
            CASE
                WHEN mc_180d != 0
                THEN ((market_cap::NUMERIC - mc_180d) / ABS(mc_180d) * 100)
                ELSE NULL
            END as growth_180d,
            CASE
                WHEN mc_180d != 0
                THEN ((market_cap::NUMERIC - mc_180d) / ABS(mc_180d) * 100) * 2
                ELSE NULL
            END as growth_annualized
        FROM market_cap_data
        WHERE rn = 1
        """

        result = await self.execute_query(query, self.symbol)

        if not result or result[0]['growth_180d'] is None:
            return None

        row = result[0]
        current_mc = float(row['current_mc'] or 0)
        mc_90d = float(row['mc_90d'] or 0)
        mc_180d = float(row['mc_180d'] or 0)
        growth_annualized = float(row['growth_annualized'] or 0)

        # Acceleration: recent growth vs earlier growth
        acceleration = 0
        if current_mc > mc_90d and mc_90d > mc_180d:
            acceleration = 100  # Accelerating growth
        elif current_mc > mc_90d:
            acceleration = 60  # Positive growth
        elif current_mc > mc_180d:
            acceleration = 30  # Slower growth

        # Growth rate score (annualized)
        growth_score = min(100, max(0, growth_annualized * 1.5))

        # Acceleration score
        accel_score = acceleration

        score = (growth_score * 0.7) + (accel_score * 0.3)

        return round(score, 2)

    async def calculate_g3(self):
        """
        G3. EPS Growth Strategy (Modified for available data)
        Description: Sustained EPS growth and consistency using 90d and 180d data
        Uses annualized growth rate based on 180-day period
        """
        query = """
        WITH eps_data AS (
            SELECT
                date,
                eps,
                LAG(eps, 90) OVER (ORDER BY date) as eps_90d_ago,
                LAG(eps, 180) OVER (ORDER BY date) as eps_180d_ago,
                ROW_NUMBER() OVER (ORDER BY date DESC) as rn
            FROM kr_intraday_total
            WHERE symbol = $1
                AND eps IS NOT NULL
        )
        SELECT
            eps as current_eps,
            eps_90d_ago,
            eps_180d_ago,
            CASE
                WHEN eps_90d_ago != 0
                THEN ((eps::NUMERIC - eps_90d_ago) / ABS(eps_90d_ago) * 100)
                ELSE NULL
            END as growth_quarterly,
            CASE
                WHEN eps_180d_ago != 0
                THEN ((eps::NUMERIC - eps_180d_ago) / ABS(eps_180d_ago) * 100)
                ELSE NULL
            END as growth_half_yearly,
            CASE
                WHEN eps_180d_ago != 0
                THEN ((eps::NUMERIC - eps_180d_ago) / ABS(eps_180d_ago) * 100) * 2
                ELSE NULL
            END as growth_yearly_annualized
        FROM eps_data
        WHERE rn = 1
        """

        result = await self.execute_query(query, self.symbol)

        # Fallback to financial statement net profit if eps data unavailable
        if not result or result[0]['growth_half_yearly'] is None:
            query_fallback = """
            WITH net_profit_data AS (
                SELECT
                    thstrm_amount as current_profit,
                    frmtrm_amount as prev_profit,
                    bfefrmtrm_amount as prev2_profit
                FROM kr_financial_position
                WHERE symbol = $1
                    AND sj_div = 'IS'
                    AND account_nm = '당기순이익(손실)'
                    AND frmtrm_amount IS NOT NULL
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
            )
            SELECT
                current_profit,
                prev_profit,
                prev2_profit,
                CASE
                    WHEN prev_profit != 0
                    THEN ((current_profit::NUMERIC - prev_profit) / ABS(prev_profit) * 100)
                    ELSE NULL
                END as growth_1y,
                CASE
                    WHEN prev2_profit != 0
                    THEN ((prev_profit::NUMERIC - prev2_profit) / ABS(prev2_profit) * 100)
                    ELSE NULL
                END as growth_2y
            FROM net_profit_data
            """

            fallback_result = await self.execute_query(query_fallback, self.symbol, self.analysis_date)

            if not fallback_result or fallback_result[0]['growth_1y'] is None:
                return None

            fb_row = fallback_result[0]
            current_profit = float(fb_row['current_profit'] or 0)
            prev_profit = float(fb_row['prev_profit'] or 0)
            prev2_profit = float(fb_row['prev2_profit'] or 0) if fb_row['prev2_profit'] else 0
            growth_1y = float(fb_row['growth_1y'])
            growth_2y = float(fb_row['growth_2y']) if fb_row['growth_2y'] is not None else 0

            # Consecutive growth
            consecutive_growth = 0
            if prev2_profit and current_profit > prev_profit and prev_profit > prev2_profit:
                consecutive_growth = 100
            elif current_profit > prev_profit:
                consecutive_growth = 60
            elif prev2_profit and current_profit > prev2_profit:
                consecutive_growth = 30
            else:
                consecutive_growth = 0

            # Growth rate score (1Y growth as proxy)
            growth_score = min(100, max(0, growth_1y * 2))

            # Consistency score
            consistency_score = consecutive_growth

            score = (growth_score * 0.6) + (consistency_score * 0.4)

            return round(score, 2)

        row = result[0]
        current_eps = float(row['current_eps'] or 0)
        eps_90d = float(row['eps_90d_ago'] or 0)
        eps_180d = float(row['eps_180d_ago'] or 0)
        growth_yearly = float(row['growth_yearly_annualized'] or 0)

        # Consecutive growth (comparing current -> 90d -> 180d)
        consecutive_growth = 0
        if current_eps > eps_90d and eps_90d > eps_180d:
            consecutive_growth = 100
        elif current_eps > eps_90d:
            consecutive_growth = 60
        elif current_eps > eps_180d:
            consecutive_growth = 30
        else:
            consecutive_growth = 0

        # Growth rate score (annualized growth)
        growth_score = min(100, max(0, growth_yearly * 2))

        # Consistency score
        consistency_score = consecutive_growth

        score = (growth_score * 0.6) + (consistency_score * 0.4)

        return round(score, 2)

    async def calculate_g4(self):
        """
        G4. PEG Ratio Strategy (Modified for available data)
        Description: Valuation attractiveness relative to growth
        Uses 180-day EPS growth annualized for calculation
        """
        query = """
        WITH latest_data AS (
            SELECT
                per,
                eps,
                LAG(eps, 180) OVER (ORDER BY date) as eps_180d_ago,
                ROW_NUMBER() OVER (ORDER BY date DESC) as rn
            FROM kr_intraday_total
            WHERE symbol = $1
                AND per IS NOT NULL
                AND eps IS NOT NULL
        )
        SELECT
            per,
            eps,
            eps_180d_ago,
            CASE
                WHEN eps_180d_ago != 0
                THEN ((eps::NUMERIC - eps_180d_ago) / ABS(eps_180d_ago) * 100)
                ELSE NULL
            END as eps_growth_180d,
            CASE
                WHEN eps_180d_ago != 0
                THEN ((eps::NUMERIC - eps_180d_ago) / ABS(eps_180d_ago) * 100) * 2
                ELSE NULL
            END as eps_growth_annualized
        FROM latest_data
        WHERE rn = 1
        """

        result = await self.execute_query(query, self.symbol)

        # Fallback to financial statement net profit if eps data unavailable
        if not result or result[0]['eps_growth_annualized'] is None:
            # Get net profit growth from financial statements
            net_profit_query = """
            WITH net_profit_data AS (
                SELECT
                    thstrm_amount as current_profit,
                    frmtrm_amount as prev_profit
                FROM kr_financial_position
                WHERE symbol = $1
                    AND sj_div = 'IS'
                    AND account_nm = '당기순이익(손실)'
                    AND frmtrm_amount IS NOT NULL
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
            )
            SELECT
                CASE
                    WHEN prev_profit != 0
                    THEN ((current_profit::NUMERIC - prev_profit) / ABS(prev_profit) * 100)
                    ELSE NULL
                END as profit_growth_1y
            FROM net_profit_data
            """

            net_profit_result = await self.execute_query(net_profit_query, self.symbol, self.analysis_date)

            if not net_profit_result or net_profit_result[0]['profit_growth_1y'] is None:
                return None

            profit_growth = float(net_profit_result[0]['profit_growth_1y'])

            # Get PER from kr_intraday_total
            per_query = """
            SELECT per
            FROM kr_intraday_total
            WHERE symbol = $1
                AND per IS NOT NULL
            ORDER BY date DESC
            LIMIT 1
            """

            per_result = await self.execute_query(per_query, self.symbol)

            if not per_result or per_result[0]['per'] is None:
                return None

            per = float(per_result[0]['per'])

            # Get revenue CAGR (reuse existing query structure)
            revenue_query_fb = """
            SELECT
                CASE
                    WHEN bfefrmtrm_amount > 0 AND thstrm_amount > 0
                    THEN (POWER(thstrm_amount::NUMERIC / bfefrmtrm_amount::NUMERIC, 0.5) - 1) * 100
                    ELSE NULL
                END as revenue_cagr_3y
            FROM kr_financial_position
            WHERE symbol = $1
                AND sj_div = 'IS'
                AND account_nm = '매출액'
                AND frmtrm_amount IS NOT NULL
                AND bfefrmtrm_amount IS NOT NULL
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

            revenue_result_fb = await self.execute_query(revenue_query_fb, self.symbol, self.analysis_date)
            revenue_cagr = float(revenue_result_fb[0]['revenue_cagr_3y'] or 0) if revenue_result_fb and revenue_result_fb[0]['revenue_cagr_3y'] else 0

            # Expected growth rate (using net profit growth instead of EPS growth)
            expected_growth = (profit_growth * 0.6) + ((revenue_cagr or 0) * 0.4)

            if expected_growth <= 0 or per <= 0:
                return 0

            # PEG ratio
            peg = per / expected_growth

            # PEG score
            if peg < 0 or expected_growth < 0:
                peg_score = 0
            elif peg <= 1:
                peg_score = 100
            elif peg <= 1.5:
                peg_score = 80
            elif peg <= 2:
                peg_score = 60
            else:
                peg_score = max(0, 100 - (peg - 2) * 20)

            # Growth rate score
            growth_score = min(50, max(0, expected_growth))

            score = (peg_score * 0.7) + (growth_score * 0.3)

            return round(score, 2)

        row = result[0]
        per = float(row['per'] or 0)
        eps_growth = float(row['eps_growth_annualized'] or 0)

        # Get revenue 3Y CAGR
        revenue_query = """
        SELECT
            CASE
                WHEN bfefrmtrm_amount > 0 AND thstrm_amount > 0
                THEN (POWER(thstrm_amount::NUMERIC / bfefrmtrm_amount::NUMERIC, 0.5) - 1) * 100
                ELSE NULL
            END as revenue_cagr_3y
        FROM kr_financial_position
        WHERE symbol = $1
            AND sj_div = 'IS'
            AND account_nm = '매출액'
                AND frmtrm_amount IS NOT NULL
                AND bfefrmtrm_amount IS NOT NULL
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

        revenue_result = await self.execute_query(revenue_query, self.symbol, self.analysis_date)
        revenue_cagr = float(revenue_result[0]['revenue_cagr_3y'] or 0) if revenue_result and revenue_result[0]['revenue_cagr_3y'] else 0

        # Expected growth rate
        expected_growth = (eps_growth * 0.6) + ((revenue_cagr or 0) * 0.4)

        if expected_growth <= 0 or per <= 0:
            return 0

        # PEG ratio
        peg = per / expected_growth

        # PEG score
        if peg < 0 or expected_growth < 0:
            peg_score = 0
        elif peg <= 1:
            peg_score = 100
        elif peg <= 1.5:
            peg_score = 80
        elif peg <= 2:
            peg_score = 60
        else:
            peg_score = max(0, 100 - (peg - 2) * 20)

        # Growth rate score
        growth_score = min(50, max(0, expected_growth))

        score = (peg_score * 0.7) + (growth_score * 0.3)

        return round(score, 2)

    async def calculate_g5(self):
        """
        G5. Market Share Expansion Strategy (with fallback)
        Description: Growing market share within industry
        Fallback: Uses market cap share if revenue data unavailable
        """
        # Try revenue-based market share first
        query_revenue = """
        WITH company_revenue AS (
            SELECT
                fp.symbol,
                sd.industry,
                fp.thstrm_amount as current_revenue,
                fp.frmtrm_amount as prev_revenue
            FROM kr_financial_position fp
            JOIN kr_stock_detail sd ON fp.symbol = sd.symbol
            WHERE fp.sj_div = 'IS'
                AND fp.account_nm = '매출액'
                AND fp.thstrm_amount IS NOT NULL
                AND fp.rcept_dt <= COALESCE($2::date, CURRENT_DATE)
        ),
        target_company AS (
            SELECT industry, current_revenue, prev_revenue
            FROM company_revenue
            WHERE symbol = $1
        ),
        industry_totals AS (
            SELECT
                SUM(cr.current_revenue) as industry_current,
                SUM(cr.prev_revenue) as industry_prev
            FROM company_revenue cr
            CROSS JOIN target_company tc
            WHERE cr.industry = tc.industry
        )
        SELECT
            tc.current_revenue,
            tc.prev_revenue,
            it.industry_current,
            it.industry_prev,
            (tc.current_revenue::NUMERIC / NULLIF(it.industry_current, 0) * 100) as current_share,
            (tc.prev_revenue::NUMERIC / NULLIF(it.industry_prev, 0) * 100) as prev_share
        FROM target_company tc
        CROSS JOIN industry_totals it
        """

        result = await self.execute_query(query_revenue, self.symbol, self.analysis_date)

        # Fallback to market cap based share
        if not result or not result[0] or result[0]['current_share'] is None:
            query_fallback = """
            WITH company_market_cap AS (
                SELECT
                    it.symbol,
                    sd.industry,
                    AVG(it.market_cap) as avg_market_cap
                FROM kr_intraday_total it
                JOIN kr_stock_detail sd ON it.symbol = sd.symbol
                WHERE it.date >= CURRENT_DATE - INTERVAL '30 days'
                    AND it.market_cap IS NOT NULL
                    AND sd.industry IS NOT NULL
                GROUP BY it.symbol, sd.industry
            ),
            target_company AS (
                SELECT industry, avg_market_cap as current_mc
                FROM company_market_cap
                WHERE symbol = $1
            ),
            prev_market_cap AS (
                SELECT
                    it.symbol,
                    sd.industry,
                    AVG(it.market_cap) as avg_market_cap
                FROM kr_intraday_total it
                JOIN kr_stock_detail sd ON it.symbol = sd.symbol
                WHERE it.date >= CURRENT_DATE - INTERVAL '90 days'
                    AND it.date < CURRENT_DATE - INTERVAL '30 days'
                    AND it.market_cap IS NOT NULL
                    AND sd.industry IS NOT NULL
                GROUP BY it.symbol, sd.industry
            ),
            target_prev_mc AS (
                SELECT avg_market_cap as prev_mc
                FROM prev_market_cap
                WHERE symbol = $2
            ),
            industry_totals AS (
                SELECT
                    SUM(cmc.avg_market_cap) as industry_current
                FROM company_market_cap cmc
                CROSS JOIN target_company tc
                WHERE cmc.industry = tc.industry
            ),
            industry_prev_totals AS (
                SELECT
                    SUM(pmc.avg_market_cap) as industry_prev
                FROM prev_market_cap pmc
                CROSS JOIN target_company tc
                WHERE pmc.industry = tc.industry
            )
            SELECT
                tc.current_mc,
                tpm.prev_mc,
                it.industry_current,
                ipt.industry_prev,
                (tc.current_mc::NUMERIC / NULLIF(it.industry_current, 0) * 100) as current_share,
                (tpm.prev_mc::NUMERIC / NULLIF(ipt.industry_prev, 0) * 100) as prev_share
            FROM target_company tc
            CROSS JOIN target_prev_mc tpm
            CROSS JOIN industry_totals it
            CROSS JOIN industry_prev_totals ipt
            """

            result = await self.execute_query(query_fallback, self.symbol, self.symbol)

            if not result or not result[0] or result[0]['current_share'] is None:
                return None

        row = result[0]
        current_share = float(row['current_share'] or 0)
        prev_share = float(row['prev_share'] or 0)

        # Market share change
        share_change = current_share - prev_share
        share_growth_rate = ((current_share - prev_share) / prev_share * 100) if prev_share > 0 else 0

        # Absolute market share score
        absolute_score = min(100, current_share * 3)

        # Market share increase score
        increase_score = min(100, max(0, share_growth_rate * 5 + 50))

        score = (absolute_score * 0.4) + (increase_score * 0.6)

        return round(score, 2)

    async def calculate_g6(self):
        """
        G6. BPS Growth Strategy (Modified - replaces Asset Growth)
        Description: Book value per share growth as proxy for asset growth
        Asset total data unavailable, using BPS as perfect alternative
        """
        query = """
        WITH bps_data AS (
            SELECT
                bps,
                pbr,
                LAG(bps, 90) OVER (ORDER BY date) as bps_90d,
                LAG(bps, 180) OVER (ORDER BY date) as bps_180d,
                LAG(pbr, 180) OVER (ORDER BY date) as pbr_180d,
                ROW_NUMBER() OVER (ORDER BY date DESC) as rn
            FROM kr_intraday_total
            WHERE symbol = $1
                AND bps IS NOT NULL
        )
        SELECT
            bps as current_bps,
            bps_90d,
            bps_180d,
            pbr as current_pbr,
            pbr_180d,
            CASE
                WHEN bps_90d != 0
                THEN ((bps::NUMERIC - bps_90d) / ABS(bps_90d) * 100)
                ELSE NULL
            END as growth_90d,
            CASE
                WHEN bps_180d != 0
                THEN ((bps::NUMERIC - bps_180d) / ABS(bps_180d) * 100)
                ELSE NULL
            END as growth_180d,
            CASE
                WHEN bps_180d != 0
                THEN ((bps::NUMERIC - bps_180d) / ABS(bps_180d) * 100) * 2
                ELSE NULL
            END as growth_annualized
        FROM bps_data
        WHERE rn = 1
        """

        result = await self.execute_query(query, self.symbol)

        # Fallback to financial statement asset total if BPS data unavailable
        if not result or result[0]['growth_180d'] is None:
            query_fallback = """
            WITH asset_data AS (
                SELECT
                    thstrm_amount as current_asset,
                    frmtrm_amount as prev_asset,
                    bfefrmtrm_amount as prev2_asset
                FROM kr_financial_position
                WHERE symbol = $1
                    AND sj_div = 'BS'
                    AND account_nm = '자산총계'
                    AND frmtrm_amount IS NOT NULL
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
            )
            SELECT
                current_asset,
                prev_asset,
                prev2_asset,
                CASE
                    WHEN prev_asset != 0
                    THEN ((current_asset::NUMERIC - prev_asset) / ABS(prev_asset) * 100)
                    ELSE NULL
                END as growth_1y,
                CASE
                    WHEN prev2_asset != 0
                    THEN ((prev_asset::NUMERIC - prev2_asset) / ABS(prev2_asset) * 100)
                    ELSE NULL
                END as growth_2y
            FROM asset_data
            """

            fallback_result = await self.execute_query(query_fallback, self.symbol, self.analysis_date)

            if not fallback_result or fallback_result[0]['growth_1y'] is None:
                return None

            fb_row = fallback_result[0]
            current_asset = float(fb_row['current_asset'] or 0)
            prev_asset = float(fb_row['prev_asset'] or 0)
            prev2_asset = float(fb_row['prev2_asset'] or 0) if fb_row['prev2_asset'] else 0
            growth_1y = float(fb_row['growth_1y'])
            growth_2y = float(fb_row['growth_2y']) if fb_row['growth_2y'] is not None else 0

            # Asset growth score
            asset_score = min(100, max(0, growth_1y * 5))

            # Consistency score
            consistency = 0
            if prev2_asset and current_asset > prev_asset and prev_asset > prev2_asset:
                consistency = 100  # Consistent growth
            elif current_asset > prev_asset:
                consistency = 60  # Recent growth

            # No PBR improvement check for fallback (no market data)
            score = (asset_score * 0.7) + (consistency * 0.3)

            return round(score, 2)

        row = result[0]
        current_bps = float(row['current_bps'] or 0)
        bps_90d = float(row['bps_90d'] or 0)
        bps_180d = float(row['bps_180d'] or 0)
        growth_annualized = float(row['growth_annualized'] or 0)
        current_pbr = float(row['current_pbr'] or 1)
        pbr_180d = float(row['pbr_180d'] or 1)

        # Quality check: PBR improvement = market recognizing asset value
        pbr_improvement = 20 if current_pbr > pbr_180d else 0

        # BPS growth score (annualized)
        bps_score = min(100, max(0, growth_annualized * 5))

        # Consistency score
        consistency = 0
        if current_bps > bps_90d and bps_90d > bps_180d:
            consistency = 100  # Consistent growth
        elif current_bps > bps_90d:
            consistency = 60  # Recent growth

        score = (bps_score * 0.6) + (consistency * 0.3) + (pbr_improvement * 0.1)

        return round(score, 2)

    async def calculate_g7(self):
        """
        G7. PBR Trend Strategy (Modified - replaces R&D Investment Growth)
        Description: PBR uptrend as proxy for innovation/R&D value recognition
        R&D data unavailable, using PBR trend as market's valuation of intangible value
        """
        query = """
        WITH pbr_data AS (
            SELECT
                pbr,
                per,
                LAG(pbr, 30) OVER (ORDER BY date) as pbr_30d,
                LAG(pbr, 90) OVER (ORDER BY date) as pbr_90d,
                LAG(pbr, 180) OVER (ORDER BY date) as pbr_180d,
                ROW_NUMBER() OVER (ORDER BY date DESC) as rn
            FROM kr_intraday_total
            WHERE symbol = $1
                AND pbr IS NOT NULL
        )
        SELECT
            pbr as current_pbr,
            per as current_per,
            pbr_30d,
            pbr_90d,
            pbr_180d,
            CASE
                WHEN pbr_180d != 0
                THEN ((pbr::NUMERIC - pbr_180d) / ABS(pbr_180d) * 100)
                ELSE NULL
            END as pbr_change_180d
        FROM pbr_data
        WHERE rn = 1
        """

        result = await self.execute_query(query, self.symbol)

        if not result or result[0]['pbr_change_180d'] is None:
            return None

        row = result[0]
        current_pbr = float(row['current_pbr'] or 0)
        pbr_30d = float(row['pbr_30d'] or 0)
        pbr_90d = float(row['pbr_90d'] or 0)
        pbr_180d = float(row['pbr_180d'] or 0)
        pbr_change = float(row['pbr_change_180d'] or 0)

        # Uptrend consistency: continuous PBR increase = market values innovation
        trend_strength = 0
        if current_pbr > pbr_30d and pbr_30d > pbr_90d and pbr_90d > pbr_180d:
            trend_strength = 100  # Strong uptrend
        elif current_pbr > pbr_90d and pbr_90d > pbr_180d:
            trend_strength = 70  # Moderate uptrend
        elif current_pbr > pbr_180d:
            trend_strength = 40  # Weak uptrend

        # PBR growth score
        growth_score = min(100, max(0, pbr_change * 3 + 50))

        # Trend score
        trend_score = trend_strength

        score = (growth_score * 0.5) + (trend_score * 0.5)

        return round(score, 2)

    async def calculate_g8(self):
        """
        G8. PBR Growth Strategy (Modified - replaces Intangible Asset Growth)
        Description: PBR growth as proxy for intangible value (brand, tech) recognition
        Intangible asset data unavailable, PBR reflects market's valuation of intangibles
        """
        query = """
        WITH pbr_data AS (
            SELECT
                pbr,
                bps,
                close,
                LAG(pbr, 90) OVER (ORDER BY date) as pbr_90d,
                LAG(pbr, 180) OVER (ORDER BY date) as pbr_180d,
                ROW_NUMBER() OVER (ORDER BY date DESC) as rn
            FROM kr_intraday_total
            WHERE symbol = $1
                AND pbr IS NOT NULL
                AND bps IS NOT NULL
        )
        SELECT
            pbr as current_pbr,
            bps as current_bps,
            close as current_price,
            pbr_90d,
            pbr_180d,
            CASE
                WHEN pbr_90d != 0
                THEN ((pbr::NUMERIC - pbr_90d) / ABS(pbr_90d) * 100)
                ELSE NULL
            END as pbr_growth_90d,
            CASE
                WHEN pbr_180d != 0
                THEN ((pbr::NUMERIC - pbr_180d) / ABS(pbr_180d) * 100)
                ELSE NULL
            END as pbr_growth_180d,
            CASE
                WHEN pbr_180d != 0
                THEN ((pbr::NUMERIC - pbr_180d) / ABS(pbr_180d) * 100) * 2
                ELSE NULL
            END as pbr_growth_annualized
        FROM pbr_data
        WHERE rn = 1
        """

        result = await self.execute_query(query, self.symbol)

        if not result or result[0]['pbr_growth_180d'] is None:
            return None

        row = result[0]
        current_pbr = float(row['current_pbr'] or 0)
        pbr_90d = float(row['pbr_90d'] or 0)
        pbr_180d = float(row['pbr_180d'] or 0)
        pbr_growth_annualized = float(row['pbr_growth_annualized'] or 0)

        # PBR level: higher PBR = market recognizes more intangible value
        pbr_level_score = min(100, (current_pbr - 1) * 50) if current_pbr > 1 else 0

        # PBR growth: increasing PBR = growing intangible value recognition
        growth_score = min(100, max(0, pbr_growth_annualized * 2 + 50))

        # Consistency: continuous growth
        consistency = 0
        if current_pbr > pbr_90d and pbr_90d > pbr_180d:
            consistency = 100
        elif current_pbr > pbr_180d:
            consistency = 50

        score = (growth_score * 0.5) + (pbr_level_score * 0.3) + (consistency * 0.2)

        return round(score, 2)

    async def calculate_g9(self):
        """
        G9. Institutional Interest Growth (Modified - replaces New Business Growth)
        Description: Institutional buying increase as proxy for new growth potential
        New business data unavailable, institutions detect fundamental changes early
        """
        query = """
        WITH inst_data AS (
            SELECT
                date,
                inst_net_value,
                SUM(inst_net_value) OVER (
                    ORDER BY date
                    ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
                ) as inst_net_30d,
                SUM(inst_net_value) OVER (
                    ORDER BY date
                    ROWS BETWEEN 89 PRECEDING AND CURRENT ROW
                ) as inst_net_90d,
                ROW_NUMBER() OVER (ORDER BY date DESC) as rn
            FROM kr_individual_investor_daily_trading
            WHERE symbol = $1
        )
        SELECT
            inst_net_30d,
            inst_net_90d,
            inst_net_value as latest_net
        FROM inst_data
        WHERE rn = 1
        """

        result = await self.execute_query(query, self.symbol)

        if not result:
            return None

        row = result[0]
        inst_net_30d = float(row['inst_net_30d'] or 0)
        inst_net_90d = float(row['inst_net_90d'] or 0)
        latest_net = float(row['latest_net'] or 0)

        # Recent 30d vs previous 60d (acceleration)
        inst_net_60d_prev = inst_net_90d - inst_net_30d

        # Get market cap category
        market_cap_category = await self._get_market_cap_category()

        # For small/mid cap stocks: check threshold
        if market_cap_category in ['중형', '소형']:
            # Threshold: if inst_net_30d is too small, exclude from calculation
            # 0.2% of 1 billion won = 2 million won per day avg
            threshold = 2_000_000 * 30  # 60 million won for 30 days
            if abs(inst_net_30d) < threshold:
                return None  # Exclude from calculation (N/A)

        # Buying pressure score
        buying_score = 0
        if inst_net_30d > 0:
            # Apply 10x scaling for small/mid cap
            if market_cap_category in ['중형', '소형']:
                buying_score = min(100, abs(inst_net_30d) / 100_000_000 * 10)  # 10x scaling
            else:
                buying_score = min(100, abs(inst_net_30d) / 1_000_000_000 * 10)  # Original

        # Acceleration score: recent activity stronger than before
        acceleration = 0
        if inst_net_30d > inst_net_60d_prev and inst_net_30d > 0:
            acceleration = 100  # Accelerating buying
        elif inst_net_30d > 0:
            acceleration = 60  # Steady buying
        elif inst_net_90d > 0:
            acceleration = 30  # Cumulative positive

        # Latest activity
        recent_activity = 50 if latest_net > 0 else 0

        score = (buying_score * 0.4) + (acceleration * 0.4) + (recent_activity * 0.2)

        return round(score, 2)

    async def calculate_g10(self):
        """
        G10. Customer Base Expansion Strategy
        Description: Volume growth indicating customer base expansion
        """
        query = """
        WITH volume_stats AS (
            SELECT
                AVG(volume) as avg_volume_90d
            FROM kr_intraday_total
            WHERE symbol = $1
                AND date >= CURRENT_DATE - INTERVAL '90 days'
        ),
        volume_stats_1y AS (
            SELECT
                AVG(volume) as avg_volume_365d
            FROM kr_intraday_total
            WHERE symbol = $2
                AND date >= CURRENT_DATE - INTERVAL '365 days'
        ),
        retail_participation AS (
            SELECT
                SUM(retail_net_volume) as retail_net_90d,
                SUM(ABS(retail_net_volume)) as retail_total_90d
            FROM kr_individual_investor_daily_trading
            WHERE symbol = $3
                AND date >= CURRENT_DATE - INTERVAL '90 days'
        )
        SELECT
            vs.avg_volume_90d,
            vs1y.avg_volume_365d,
            rp.retail_net_90d,
            rp.retail_total_90d,
            ((vs.avg_volume_90d - vs1y.avg_volume_365d)::NUMERIC /
             NULLIF(vs1y.avg_volume_365d, 0) * 100) as volume_growth_rate,
            (rp.retail_net_90d::NUMERIC / NULLIF(rp.retail_total_90d, 0) * 100) as retail_participation_rate
        FROM volume_stats vs
        CROSS JOIN volume_stats_1y vs1y
        CROSS JOIN retail_participation rp
        """

        result = await self.execute_query(query, self.symbol, self.symbol, self.symbol)

        if not result:
            return None

        row = result[0]
        volume_growth = float(row['volume_growth_rate'] or 0)
        retail_participation = float(row['retail_participation_rate'] or 0)

        # Volume growth score
        volume_score = min(100, max(0, volume_growth + 50))

        # Retail participation score
        retail_score = min(100, max(0, retail_participation * 5 + 50))

        # Shareholder expansion score (estimated)
        shareholder_score = min(100, max(0, volume_growth * 0.5 + 50))

        score = (volume_score * 0.3) + (retail_score * 0.3) + (shareholder_score * 0.4)

        return round(score, 2)

    async def calculate_g11(self):
        """
        G11. Dividend Growth Strategy (with fallback)
        Description: Sustained dividend growth showing confidence
        Fallback: Uses dps trend from kr_intraday_total if dividend data unavailable
        """
        # Try kr_dividends first
        query_dividends = """
        WITH dividend_data AS (
            SELECT
                thstrm as current_dividend,
                frmtrm as prev_dividend,
                lwfr as prev2_dividend
            FROM kr_dividends
            WHERE symbol = $1
            ORDER BY stlm_dt DESC
            LIMIT 1
        ),
        eps_data AS (
            SELECT eps, dps
            FROM kr_intraday_total
            WHERE symbol = $2
                AND ($3::date IS NULL OR date = $3)
            ORDER BY date DESC
            LIMIT 1
        )
        SELECT
            dd.current_dividend,
            dd.prev_dividend,
            dd.prev2_dividend,
            ed.eps,
            ed.dps,
            CASE
                WHEN dd.prev_dividend != 0
                THEN ((dd.current_dividend - dd.prev_dividend)::NUMERIC / ABS(dd.prev_dividend) * 100)
                ELSE NULL
            END as dividend_growth,
            CASE
                WHEN dd.prev2_dividend > 0 AND dd.current_dividend > 0
                THEN (POWER(dd.current_dividend::NUMERIC / dd.prev2_dividend::NUMERIC, 0.5) - 1) * 100
                ELSE NULL
            END as dividend_cagr_3y,
            (ed.dps::NUMERIC / NULLIF(ed.eps, 0) * 100) as payout_ratio
        FROM dividend_data dd
        CROSS JOIN eps_data ed
        """

        result = await self.execute_query(query_dividends, self.symbol, self.symbol, self.analysis_date)

        # Fallback to dps/dividend_yield from kr_intraday_total
        if not result or not result[0] or result[0]['dividend_cagr_3y'] is None:
            query_fallback = """
            WITH dps_data AS (
                SELECT
                    dps,
                    dividend_yield,
                    eps,
                    LAG(dps, 90) OVER (ORDER BY date) as dps_90d,
                    LAG(dps, 180) OVER (ORDER BY date) as dps_180d,
                    LAG(dividend_yield, 90) OVER (ORDER BY date) as yield_90d,
                    LAG(dividend_yield, 180) OVER (ORDER BY date) as yield_180d,
                    ROW_NUMBER() OVER (ORDER BY date DESC) as rn
                FROM kr_intraday_total
                WHERE symbol = $1
                    AND dps IS NOT NULL
                    AND dividend_yield IS NOT NULL
                    AND ($2::date IS NULL OR date <= $2)
            )
            SELECT
                dps,
                dps_90d,
                dps_180d,
                dividend_yield,
                yield_90d,
                yield_180d,
                eps,
                CASE WHEN dps_90d > 0
                    THEN ((dps - dps_90d)::NUMERIC / dps_90d * 100) * 4
                    ELSE NULL
                END as dividend_growth_annualized,
                CASE WHEN dps_180d > 0
                    THEN ((dps - dps_180d)::NUMERIC / dps_180d * 100) * 2
                    ELSE NULL
                END as dividend_cagr,
                (dps::NUMERIC / NULLIF(eps, 0) * 100) as payout_ratio
            FROM dps_data
            WHERE rn = 1
            """

            result = await self.execute_query(query_fallback, self.symbol, self.analysis_date)

            if not result or not result[0] or result[0]['dividend_cagr'] is None:
                return None

            row = result[0]
            dps = float(row['dps'] or 0)
            dps_90d = float(row['dps_90d'] or 0)
            dps_180d = float(row['dps_180d'] or 0)
            dividend_cagr = float(row['dividend_cagr'] or 0)
            payout_ratio = float(row['payout_ratio'] or 0)

            # Consecutive increase
            consecutive_increase = 0
            if dps > dps_90d and dps_90d > dps_180d:
                consecutive_increase = 100
            elif dps > dps_90d:
                consecutive_increase = 60
            else:
                consecutive_increase = 20

            # Payout ratio appropriateness
            if 20 <= payout_ratio <= 50:
                payout_score = 100
            elif 10 <= payout_ratio <= 60:
                payout_score = 70
            elif payout_ratio > 0:
                payout_score = 40
            else:
                payout_score = 0

            # Growth score
            growth_score = min(100, max(0, dividend_cagr * 2 + 50))

            score = (growth_score * 0.4) + (payout_score * 0.3) + (consecutive_increase * 0.3)

            return round(score, 2)

        row = result[0]
        current_div = float(row['current_dividend'] or 0)
        prev_div = float(row['prev_dividend'] or 0)
        prev2_div = float(row['prev2_dividend'] or 0)
        dividend_cagr = float(row['dividend_cagr_3y'] or 0)
        payout_ratio = float(row['payout_ratio'] or 0)

        # Consecutive increase
        consecutive_increase = 0
        if current_div > prev_div and prev_div > prev2_div:
            consecutive_increase = 100
        elif current_div > prev_div:
            consecutive_increase = 60
        else:
            consecutive_increase = 20

        # Payout ratio appropriateness
        if 20 <= payout_ratio <= 50:
            payout_score = 100
        elif 10 <= payout_ratio <= 60:
            payout_score = 70
        else:
            payout_score = 30

        # Growth score
        growth_score = min(100, max(0, dividend_cagr * 5))

        score = (growth_score * 0.4) + (payout_score * 0.3) + (consecutive_increase * 0.3)

        return round(score, 2)

    async def calculate_g12(self):
        """
        G12. Foreign Ownership Growth (Modified - replaces Export Growth)
        Description: Foreign investor interest as proxy for global competitiveness
        Export data unavailable, foreign ownership reflects global recognition
        """
        query = """
        WITH foreign_data AS (
            SELECT
                foreign_rate,
                LAG(foreign_rate, 30) OVER (ORDER BY date) as foreign_30d,
                LAG(foreign_rate, 90) OVER (ORDER BY date) as foreign_90d,
                ROW_NUMBER() OVER (ORDER BY date DESC) as rn
            FROM kr_foreign_ownership
            WHERE symbol = $1
                AND foreign_rate IS NOT NULL
        )
        SELECT
            foreign_rate as current_rate,
            foreign_30d,
            foreign_90d,
            CASE
                WHEN foreign_30d IS NOT NULL
                THEN foreign_rate - foreign_30d
                ELSE 0
            END as change_30d,
            CASE
                WHEN foreign_90d IS NOT NULL
                THEN foreign_rate - foreign_90d
                ELSE 0
            END as change_90d
        FROM foreign_data
        WHERE rn = 1
        """

        result = await self.execute_query(query, self.symbol)

        if not result:
            return None

        row = result[0]
        current_rate = float(row['current_rate'] or 0)
        foreign_30d = float(row['foreign_30d'] or 0)
        foreign_90d = float(row['foreign_90d'] or 0)
        change_30d = float(row['change_30d'] or 0)
        change_90d = float(row['change_90d'] or 0)

        # Get market cap category
        market_cap_category = await self._get_market_cap_category()

        # For small/mid cap stocks: check threshold (0.2% or less = N/A)
        if market_cap_category in ['중형', '소형']:
            if current_rate <= 0.2:
                return None  # Exclude from calculation (N/A)
            # 10x scaling for mid/small cap: 10% = 100 points
            level_score = min(100, current_rate * 10)
        else:
            # Original scaling for large cap: 50% = 100 points
            level_score = min(100, current_rate * 2)

        # Increasing trend
        trend_score = 0
        if change_30d > 0 and change_90d > 0:
            trend_score = 100  # Consistent increase
        elif change_30d > 0:
            trend_score = 60  # Recent increase
        elif change_90d > 0:
            trend_score = 30  # Historical increase

        # Growth magnitude
        growth_score = min(100, abs(change_90d) * 10)

        score = (level_score * 0.4) + (trend_score * 0.4) + (growth_score * 0.2)

        return round(score, 2)

    async def calculate_g13(self):
        """
        G13. Growth Aggressiveness (Modified - replaces Leverage Strategy)
        Description: PBR/PER ratio as indicator of growth vs value characteristics
        Debt data unavailable, valuation ratios show market's growth expectations
        """
        query = """
        WITH valuation_data AS (
            SELECT
                pbr,
                per,
                LAG(pbr, 90) OVER (ORDER BY date) as pbr_90d,
                LAG(per, 90) OVER (ORDER BY date) as per_90d,
                LAG(pbr, 180) OVER (ORDER BY date) as pbr_180d,
                LAG(per, 180) OVER (ORDER BY date) as per_180d,
                ROW_NUMBER() OVER (ORDER BY date DESC) as rn
            FROM kr_intraday_total
            WHERE symbol = $1
                AND pbr IS NOT NULL
                AND per IS NOT NULL
        )
        SELECT
            pbr as current_pbr,
            per as current_per,
            pbr_90d,
            per_90d,
            pbr_180d,
            per_180d,
            CASE
                WHEN per > 0 THEN pbr / per
                ELSE NULL
            END as growth_ratio,
            CASE
                WHEN per_180d > 0 THEN pbr_180d / per_180d
                ELSE NULL
            END as growth_ratio_180d
        FROM valuation_data
        WHERE rn = 1
        """

        result = await self.execute_query(query, self.symbol)

        if not result or result[0]['growth_ratio'] is None:
            return None

        row = result[0]
        current_pbr = float(row['current_pbr'] or 0)
        current_per = float(row['current_per'] or 0)
        pbr_90d = float(row['pbr_90d'] or 0)
        per_90d = float(row['per_90d'] or 0)
        growth_ratio = float(row['growth_ratio'] or 0)
        growth_ratio_180d = float(row['growth_ratio_180d'] or 1)

        # Growth characteristic: PBR/PER > 1 = growth stock (aggressive)
        # PBR/PER < 0.5 = value stock (conservative)
        if growth_ratio > 1:
            growth_type_score = 100  # Strong growth stock
        elif growth_ratio > 0.7:
            growth_type_score = 70  # Moderate growth
        elif growth_ratio > 0.5:
            growth_type_score = 50  # Balanced
        else:
            growth_type_score = 30  # Value stock

        # Increasing aggressiveness (ratio trend)
        ratio_trend = ((growth_ratio - growth_ratio_180d) / growth_ratio_180d * 100) if growth_ratio_180d > 0 else 0
        trend_score = min(100, max(0, ratio_trend * 2 + 50))

        # PER expansion (market willing to pay more for growth)
        per_expansion = 0
        if current_per > per_90d:
            per_expansion = 50  # Market expects growth

        score = (growth_type_score * 0.5) + (trend_score * 0.3) + (per_expansion * 0.2)

        return round(score, 2)

    async def calculate_g14(self):
        """
        G14. Earnings Surprise Strategy (with fallback)
        Description: Upward earnings revision and surprise
        Fallback: Works with partial data if any EPS metric available
        """
        query = """
        WITH eps_trend AS (
            SELECT
                eps,
                LAG(eps, 90) OVER (ORDER BY date) as eps_90d_ago,
                per,
                LAG(per, 90) OVER (ORDER BY date) as per_90d_ago,
                ROW_NUMBER() OVER (ORDER BY date DESC) as rn
            FROM kr_intraday_total
            WHERE symbol = $1
                AND eps IS NOT NULL
        ),
        eps_history AS (
            SELECT
                AVG(eps) as avg_eps_historical
            FROM kr_intraday_total
            WHERE symbol = $2
                AND date BETWEEN CURRENT_DATE - INTERVAL '365 days'
                            AND CURRENT_DATE - INTERVAL '90 days'
                AND eps IS NOT NULL
        )
        SELECT
            et.eps as current_eps,
            et.eps_90d_ago,
            et.per as current_per,
            et.per_90d_ago,
            eh.avg_eps_historical,
            CASE
                WHEN et.eps_90d_ago != 0
                THEN ((et.eps - et.eps_90d_ago)::NUMERIC / ABS(et.eps_90d_ago) * 100)
                ELSE NULL
            END as eps_growth,
            CASE
                WHEN eh.avg_eps_historical != 0
                THEN ((et.eps - eh.avg_eps_historical)::NUMERIC / eh.avg_eps_historical * 100)
                ELSE NULL
            END as eps_surprise,
            CASE
                WHEN et.per_90d_ago != 0
                THEN ((et.per - et.per_90d_ago)::NUMERIC / et.per_90d_ago * 100)
                ELSE NULL
            END as per_expansion
        FROM eps_trend et
        CROSS JOIN eps_history eh
        WHERE et.rn = 1
        """

        result = await self.execute_query(query, self.symbol, self.symbol)

        # Accept result if ANY metric is available (not just eps_growth)
        if not result or not result[0]:
            return None

        row = result[0]

        # Check if at least one metric is available
        has_eps_growth = row['eps_growth'] is not None
        has_eps_surprise = row['eps_surprise'] is not None
        has_per_expansion = row['per_expansion'] is not None

        if not (has_eps_growth or has_eps_surprise or has_per_expansion):
            return None

        eps_growth = float(row['eps_growth'] or 0)
        eps_surprise = float(row['eps_surprise'] or 0)
        per_expansion = float(row['per_expansion'] or 0)

        # EPS growth score
        growth_score = min(100, max(0, eps_growth)) if has_eps_growth else 50

        # Surprise score
        surprise_score = min(100, max(0, eps_surprise * 2 + 50)) if has_eps_surprise else 50

        # Expectation upgrade score
        expectation_score = min(100, max(0, per_expansion + 50)) if has_per_expansion else 50

        # Adjust weights based on available data
        if has_eps_growth and has_eps_surprise and has_per_expansion:
            score = (growth_score * 0.4) + (surprise_score * 0.3) + (expectation_score * 0.3)
        elif has_eps_growth and has_eps_surprise:
            score = (growth_score * 0.6) + (surprise_score * 0.4)
        elif has_eps_growth and has_per_expansion:
            score = (growth_score * 0.6) + (expectation_score * 0.4)
        elif has_eps_surprise and has_per_expansion:
            score = (surprise_score * 0.5) + (expectation_score * 0.5)
        elif has_eps_growth:
            score = growth_score
        elif has_eps_surprise:
            score = surprise_score
        else:
            score = expectation_score

        return round(score, 2)

    async def calculate_g15(self):
        """
        G15. Liquidity Growth Strategy
        Description: Trading liquidity expansion as growth signal
        Rationale: Increasing turnover indicates growing market interest
        """
        query = """
        WITH liquidity_data AS (
            SELECT
                trading_value,
                market_cap,
                CASE WHEN market_cap > 0
                    THEN (trading_value::NUMERIC / market_cap * 100)
                    ELSE NULL
                END as turnover_ratio,
                LAG(trading_value, 90) OVER (ORDER BY date) as trading_90d,
                LAG(market_cap, 90) OVER (ORDER BY date) as mc_90d,
                LAG(trading_value, 180) OVER (ORDER BY date) as trading_180d,
                LAG(market_cap, 180) OVER (ORDER BY date) as mc_180d,
                ROW_NUMBER() OVER (ORDER BY date DESC) as rn
            FROM kr_intraday_total
            WHERE symbol = $1
                AND trading_value IS NOT NULL
                AND market_cap IS NOT NULL
        )
        SELECT
            trading_value,
            market_cap,
            turnover_ratio as current_turnover,
            CASE WHEN mc_90d > 0
                THEN (trading_90d::NUMERIC / mc_90d * 100)
                ELSE NULL
            END as turnover_90d,
            CASE WHEN mc_180d > 0
                THEN (trading_180d::NUMERIC / mc_180d * 100)
                ELSE NULL
            END as turnover_180d,
            CASE WHEN trading_90d > 0
                THEN ((trading_value::NUMERIC - trading_90d) / trading_90d * 100)
                ELSE NULL
            END as trading_growth_90d,
            CASE WHEN trading_180d > 0
                THEN ((trading_value::NUMERIC - trading_180d) / trading_180d * 100)
                ELSE NULL
            END as trading_growth_180d
        FROM liquidity_data
        WHERE rn = 1
        """

        result = await self.execute_query(query, self.symbol)

        if not result or result[0]['current_turnover'] is None:
            return None

        row = result[0]
        current_turnover = float(row['current_turnover'] or 0)
        turnover_90d = float(row['turnover_90d'] or 0)
        turnover_180d = float(row['turnover_180d'] or 0)
        trading_growth_90d = float(row['trading_growth_90d'] or 0)
        trading_growth_180d = float(row['trading_growth_180d'] or 0)

        # Turnover level score (higher liquidity = higher score)
        turnover_score = min(100, current_turnover * 10)

        # Turnover growth score
        turnover_growth = 0
        if turnover_180d > 0:
            turnover_growth = (current_turnover - turnover_180d) / turnover_180d * 100
        growth_score = min(100, max(0, turnover_growth * 2 + 50))

        # Acceleration score (recent 90d vs previous 90d)
        acceleration = 0
        if turnover_90d > turnover_180d:
            acceleration = 100  # Accelerating
        elif turnover_90d > turnover_180d * 0.9:
            acceleration = 60  # Stable
        else:
            acceleration = 30  # Decelerating

        # Trading value growth
        trading_score = min(100, max(0, trading_growth_180d / 2 + 50))

        score = (turnover_score * 0.3) + (growth_score * 0.3) + (acceleration * 0.2) + (trading_score * 0.2)

        return round(score, 2)

    async def calculate_g16(self):
        """
        G16. Smart Money Concentration Strategy
        Description: Institutional and foreign investor buying concentration
        Rationale: Smart money simultaneous buying indicates strong growth potential
        """
        query = """
        WITH smart_money AS (
            SELECT
                inst_net_value,
                foreign_net_value,
                inst_buy_value,
                foreign_buy_value,
                total_buy_value,
                (inst_net_value + foreign_net_value) as smart_money_net,
                CASE WHEN total_buy_value > 0
                    THEN ((inst_buy_value + foreign_buy_value)::NUMERIC / total_buy_value * 100)
                    ELSE NULL
                END as smart_money_concentration,
                SUM(inst_net_value + foreign_net_value) OVER (
                    ORDER BY date ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
                ) as smart_net_30d,
                SUM(inst_net_value + foreign_net_value) OVER (
                    ORDER BY date ROWS BETWEEN 89 PRECEDING AND CURRENT ROW
                ) as smart_net_90d,
                AVG(
                    CASE WHEN total_buy_value > 0
                        THEN ((inst_buy_value + foreign_buy_value)::NUMERIC / total_buy_value * 100)
                        ELSE NULL
                    END
                ) OVER (
                    ORDER BY date ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
                ) as avg_concentration_30d,
                ROW_NUMBER() OVER (ORDER BY date DESC) as rn
            FROM kr_individual_investor_daily_trading
            WHERE symbol = $1
        )
        SELECT
            inst_net_value,
            foreign_net_value,
            smart_money_net,
            smart_money_concentration,
            smart_net_30d,
            smart_net_90d,
            avg_concentration_30d,
            CASE WHEN smart_net_30d > 0 AND smart_net_90d > 0 THEN 1 ELSE 0 END as both_positive
        FROM smart_money
        WHERE rn = 1
        """

        result = await self.execute_query(query, self.symbol)

        if not result or result[0]['smart_money_concentration'] is None:
            return None

        row = result[0]
        inst_net = float(row['inst_net_value'] or 0)
        foreign_net = float(row['foreign_net_value'] or 0)
        smart_net = float(row['smart_money_net'] or 0)
        concentration = float(row['smart_money_concentration'] or 0)
        smart_net_30d = float(row['smart_net_30d'] or 0)
        smart_net_90d = float(row['smart_net_90d'] or 0)
        avg_concentration_30d = float(row['avg_concentration_30d'] or 0)
        both_positive = int(row['both_positive'] or 0)

        # Simultaneous buying score
        if inst_net > 0 and foreign_net > 0:
            simultaneous_score = 100  # Both buying
        elif smart_net > 0:
            simultaneous_score = 60  # Net positive but not both
        else:
            simultaneous_score = 20  # Net selling

        # Concentration level score
        concentration_score = min(100, concentration * 2)

        # Sustained buying score (30d and 90d both positive)
        if smart_net_30d > 0 and smart_net_90d > 0:
            sustained_score = 100
        elif smart_net_30d > 0:
            sustained_score = 60
        else:
            sustained_score = 30

        # Increasing concentration
        concentration_trend = 0
        if concentration > avg_concentration_30d:
            concentration_trend = 100  # Increasing concentration
        elif concentration > avg_concentration_30d * 0.9:
            concentration_trend = 60  # Stable
        else:
            concentration_trend = 30  # Decreasing

        score = (simultaneous_score * 0.3) + (concentration_score * 0.25) + (sustained_score * 0.25) + (concentration_trend * 0.2)

        return round(score, 2)

    async def calculate_g17(self):
        """
        G17. Block Trade Growth Strategy
        Description: Large block trading activity as institutional interest
        Rationale: Increasing block trades signal significant institutional accumulation
        """
        query = """
        WITH block_data AS (
            SELECT
                block_volume_rate,
                block_volume,
                volume,
                LAG(block_volume_rate, 30) OVER (ORDER BY date) as block_rate_30d,
                LAG(block_volume_rate, 90) OVER (ORDER BY date) as block_rate_90d,
                AVG(block_volume_rate) OVER (
                    ORDER BY date ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
                ) as avg_block_rate_30d,
                COUNT(*) FILTER (WHERE block_volume_rate > 10) OVER (
                    ORDER BY date ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
                ) as high_block_days_30d,
                ROW_NUMBER() OVER (ORDER BY date DESC) as rn
            FROM kr_blocktrades
            WHERE symbol = $1
                AND block_volume_rate IS NOT NULL
        )
        SELECT
            block_volume_rate as current_block_rate,
            block_rate_30d,
            block_rate_90d,
            avg_block_rate_30d,
            high_block_days_30d,
            CASE WHEN block_rate_90d > 0
                THEN ((block_volume_rate - block_rate_90d)::NUMERIC / block_rate_90d * 100)
                ELSE NULL
            END as block_rate_growth_90d,
            CASE WHEN volume > 0
                THEN (block_volume::NUMERIC / volume * 100)
                ELSE NULL
            END as block_ratio
        FROM block_data
        WHERE rn = 1
        """

        result = await self.execute_query(query, self.symbol)

        if not result or result[0]['current_block_rate'] is None:
            return None

        row = result[0]
        current_block_rate = float(row['current_block_rate'] or 0)
        block_rate_30d = float(row['block_rate_30d'] or 0)
        block_rate_90d = float(row['block_rate_90d'] or 0)
        avg_block_rate_30d = float(row['avg_block_rate_30d'] or 0)
        high_block_days_30d = int(row['high_block_days_30d'] or 0)
        block_rate_growth = float(row['block_rate_growth_90d'] or 0)

        # Block rate level score (higher = more institutional activity)
        level_score = min(100, current_block_rate * 5)

        # Growth trend score
        if block_rate_growth > 20:
            growth_score = 100  # Strong increase
        elif block_rate_growth > 0:
            growth_score = 70  # Moderate increase
        elif block_rate_growth > -20:
            growth_score = 40  # Stable
        else:
            growth_score = 20  # Decreasing

        # Increasing trend (30d > 90d)
        if current_block_rate > block_rate_30d > block_rate_90d:
            trend_score = 100  # Accelerating
        elif current_block_rate > block_rate_90d:
            trend_score = 60  # Growing
        else:
            trend_score = 30  # Declining

        # Frequency score (how often block trades occur)
        frequency_score = min(100, (high_block_days_30d / 30) * 100 * 2)

        # Above average activity
        above_avg = 100 if current_block_rate > avg_block_rate_30d else 50

        score = (level_score * 0.25) + (growth_score * 0.25) + (trend_score * 0.2) + (frequency_score * 0.15) + (above_avg * 0.15)

        return round(score, 2)

    async def calculate_g18(self):
        """
        G18. Valuation Normalization Strategy
        Description: Undervalued stock reverting to industry average
        Rationale: PBR discount closing indicates revaluation opportunity
        """
        query = """
        WITH stock_pbr AS (
            SELECT
                pbr,
                LAG(pbr, 90) OVER (ORDER BY date) as pbr_90d,
                LAG(pbr, 180) OVER (ORDER BY date) as pbr_180d,
                ROW_NUMBER() OVER (ORDER BY date DESC) as rn
            FROM kr_intraday_total
            WHERE symbol = $1 AND pbr IS NOT NULL
        ),
        stock_industry AS (
            SELECT industry
            FROM kr_stock_detail
            WHERE symbol = $2 AND industry IS NOT NULL
        ),
        industry_avg AS (
            SELECT AVG(it.pbr) as industry_avg_pbr
            FROM kr_intraday_total it
            INNER JOIN kr_stock_detail sd ON it.symbol = sd.symbol
            INNER JOIN stock_industry si ON sd.industry = si.industry
            WHERE it.date = (SELECT MAX(date) FROM kr_intraday_total)
                AND it.pbr IS NOT NULL
                AND it.pbr > 0
        )
        SELECT
            sp.pbr as current_pbr,
            sp.pbr_90d,
            sp.pbr_180d,
            ia.industry_avg_pbr,
            CASE WHEN ia.industry_avg_pbr > 0
                THEN ((sp.pbr - ia.industry_avg_pbr)::NUMERIC / ia.industry_avg_pbr * 100)
                ELSE NULL
            END as pbr_discount,
            CASE WHEN sp.pbr_90d > 0 AND ia.industry_avg_pbr > 0
                THEN (((sp.pbr - ia.industry_avg_pbr) - (sp.pbr_90d - ia.industry_avg_pbr))::NUMERIC / ia.industry_avg_pbr * 100)
                ELSE NULL
            END as gap_closing_90d
        FROM stock_pbr sp
        CROSS JOIN industry_avg ia
        WHERE sp.rn = 1
        """

        result = await self.execute_query(query, self.symbol, self.symbol)

        if not result or result[0]['current_pbr'] is None or result[0]['industry_avg_pbr'] is None:
            return None

        row = result[0]
        current_pbr = float(row['current_pbr'] or 0)
        pbr_90d = float(row['pbr_90d'] or 0)
        pbr_180d = float(row['pbr_180d'] or 0)
        industry_avg = float(row['industry_avg_pbr'] or 0)
        pbr_discount = float(row['pbr_discount'] or 0)
        gap_closing = float(row['gap_closing_90d'] or 0)

        # Discount level score (undervalued = higher potential)
        if pbr_discount < -30:
            discount_score = 100  # Deeply undervalued
        elif pbr_discount < -15:
            discount_score = 80  # Moderately undervalued
        elif pbr_discount < 0:
            discount_score = 60  # Slightly undervalued
        elif pbr_discount < 15:
            discount_score = 40  # Fair value
        else:
            discount_score = 20  # Overvalued

        # Gap closing trend (discount reducing = revaluation)
        if gap_closing > 5:
            gap_score = 100  # Strong revaluation
        elif gap_closing > 0:
            gap_score = 70  # Gradual revaluation
        elif gap_closing > -5:
            gap_score = 40  # Stable
        else:
            gap_score = 20  # Widening gap

        # PBR trend (increasing PBR = market recognition)
        pbr_trend = 0
        if pbr_180d > 0:
            pbr_trend = (current_pbr - pbr_180d) / pbr_180d * 100

        if pbr_trend > 20:
            trend_score = 100  # Strong appreciation
        elif pbr_trend > 0:
            trend_score = 70  # Appreciation
        elif pbr_trend > -10:
            trend_score = 40  # Stable
        else:
            trend_score = 20  # Declining

        # Momentum score (accelerating revaluation)
        pbr_change_90d = 0
        if pbr_90d > 0:
            pbr_change_90d = (current_pbr - pbr_90d) / pbr_90d * 100

        if pbr_change_90d > pbr_trend / 2:
            momentum_score = 100  # Accelerating
        elif pbr_change_90d > 0:
            momentum_score = 60  # Positive
        else:
            momentum_score = 30  # Slowing

        score = (discount_score * 0.3) + (gap_score * 0.3) + (trend_score * 0.25) + (momentum_score * 0.15)

        return round(score, 2)

    async def calculate_all_strategies(self):
        """
        Calculate all 18 growth factor strategies

        Returns:
            dict: Dictionary with strategy names as keys and scores as values
        """
        strategies = {
            'G1_Revenue_Growth': await self.calculate_g1(),
            'G2_Operating_Profit_Growth': await self.calculate_g2(),
            'G3_EPS_Growth': await self.calculate_g3(),
            'G4_PEG_Ratio': await self.calculate_g4(),
            'G5_Market_Share_Expansion': await self.calculate_g5(),
            'G6_Asset_Growth': await self.calculate_g6(),
            'G7_RD_Investment_Growth': await self.calculate_g7(),
            'G8_Intangible_Asset_Growth': await self.calculate_g8(),
            'G9_Institutional_Interest': await self.calculate_g9(),
            'G10_Customer_Base_Expansion': await self.calculate_g10(),
            'G11_Dividend_Growth': await self.calculate_g11(),
            'G12_Foreign_Ownership': await self.calculate_g12(),
            'G13_Growth_Aggressiveness': await self.calculate_g13(),
            'G14_Earnings_Surprise': await self.calculate_g14(),
            'G15_Liquidity_Growth': await self.calculate_g15(),
            'G16_Smart_Money_Concentration': await self.calculate_g16(),
            'G17_Block_Trade_Growth': await self.calculate_g17(),
            'G18_Valuation_Normalization': await self.calculate_g18()
        }

        # Store strategies for weighted score calculation
        self.strategies_scores = strategies

        return strategies

    async def calculate_comprehensive_score(self):
        """
        Calculate comprehensive growth score (simple sum of valid strategies)

        Returns:
            float: Sum of all valid strategy scores
        """
        if not hasattr(self, 'strategies_scores'):
            await self.calculate_all_strategies()

        total_score = sum(score for score in self.strategies_scores.values() if score is not None)
        return round(total_score, 2)

    async def calculate_weighted_score(self, market_state=None):
        """
        Calculate market state-based weighted score for growth strategies

        Formula:
            Weighted Average = Σ(Strategy Score × Weight) / Σ(Weight)

        This normalizes the score to 0-100 scale based on market state-specific weights.

        Args:
            market_state: Market state classification (uses self.market_state if not provided)

        Returns:
            dict: {
                'weighted_score': float (0-100 normalized score),
                'weight_sum': float (sum of weights used),
                'market_state': str (market state used),
                'valid_strategies': int (number of strategies with valid scores)
            }
            or None if calculation fails
        """
        if not hasattr(self, 'strategies_scores'):
            await self.calculate_all_strategies()

        # Use provided market_state or instance's market_state
        state = market_state or self.market_state or '기타'

        # Get weights for this market state
        weights = GROWTH_STRATEGY_WEIGHTS.get(state, GROWTH_STRATEGY_WEIGHTS['기타'])

        weighted_sum = 0.0
        weight_sum = 0.0
        valid_count = 0

        # Calculate weighted sum
        for strategy_name, score in self.strategies_scores.items():
            if score is not None:
                # Validate strategy score range
                if abs(score) > 100:
                    logger.error(f"[{self.symbol}] ABNORMAL STRATEGY: {strategy_name} = {score:.2f} (expected: 0-100)")
                elif score < 0:
                    logger.warning(f"[{self.symbol}] NEGATIVE STRATEGY: {strategy_name} = {score:.2f}")

                # Extract strategy key (e.g., 'G1' from 'G1_Revenue_Growth')
                strategy_key = strategy_name.split('_')[0]
                weight = weights.get(strategy_key, 1.0)

                weighted_sum += float(score) * weight
                weight_sum += weight
                valid_count += 1

        # Avoid division by zero
        if weight_sum == 0:
            logger.warning(f"Weight sum is 0 for market state: {state}")
            return None

        # Calculate weighted average (normalized to 0-100)
        final_score = weighted_sum / weight_sum

        # Calculate base score (simple average without weights - for Refactoring)
        valid_scores = [score for score in self.strategies_scores.values() if score is not None]
        base_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0

        return {
            'weighted_score': round(final_score, 1),
            'base_score': round(base_score, 1),  # Added for Refactoring
            'weight_sum': round(weight_sum, 2),
            'market_state': state,
            'valid_strategies': valid_count,
            'strategies': self.strategies_scores
        }



async def main():
    """
    Main function demonstrating Growth Factor analysis with market state weighting
    """
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

    try:
        from kr.weight import ConditionAnalyzer
    except ImportError:
        from weight import ConditionAnalyzer

    from dotenv import load_dotenv
    load_dotenv()

    # Test symbol
    symbol = '005930'  # Samsung Electronics

    print(f"\n{'='*80}")
    print(f"Growth Factor Analysis for {symbol}")
    print(f"{'='*80}\n")

    # Step 1: Determine market state
    print("Step 1: Determining market state...")
    try:
        analyzer = ConditionAnalyzer(symbol)
        conditions, _ = analyzer.analyze()
        market_state = conditions.get('market_state', '기타')
        print(f"  Market State: {market_state}")
    except Exception as e:
        logger.warning(f"Could not determine market state: {e}")
        market_state = '기타'
        print(f"  Market State: {market_state} (default)")

    # Step 2: Calculate growth strategies
    print("\nStep 2: Calculating growth strategies...")
    calc = GrowthFactorCalculator(symbol, market_state=market_state)
    strategies = calc.calculate_all_strategies()

    # Get weights for display
    weights = GROWTH_STRATEGY_WEIGHTS.get(market_state, GROWTH_STRATEGY_WEIGHTS['기타'])
    weight_sum = sum(weights.values())
    print(f"  Weight Sum for {market_state}: {weight_sum:.2f}")

    # Step 3: Calculate scores
    print("\nStep 3: Calculating final scores...")
    comprehensive_score = calc.calculate_comprehensive_score()
    weighted_result = calc.calculate_weighted_score()

    # Display results
    print("\n" + "="*80)
    print("Results")
    print("="*80)

    # Show top 5 strategies
    print("\nTop 5 Strategies:")
    sorted_strategies = sorted(
        [(k, v) for k, v in strategies.items() if v is not None],
        key=lambda x: x[1],
        reverse=True
    )[:5]

    for strategy_name, score in sorted_strategies:
        strategy_key = strategy_name.split('_')[0]
        weight = weights.get(strategy_key, 1.0)
        weighted = score * weight
        print(f"  {strategy_name:<35s}: {score:6.2f} pts (weight: {weight:.2f}, weighted: {weighted:7.2f})")

    # Show all strategies (abbreviated)
    print("\n" + "-"*80)
    print("All Strategies:")
    for strategy_name, score in strategies.items():
        status = "[OK]" if score is not None else "[--]"
        score_str = f"{score:6.2f}" if score is not None else "  None"
        strategy_key = strategy_name.split('_')[0]
        weight = weights.get(strategy_key, 1.0)
        print(f"{status} {strategy_key:4s} {strategy_name:35s} Score: {score_str} (weight: {weight:.2f})")

    # Final scores
    print("\n" + "-"*80)
    print("Final Scores:")
    print(f"  Simple Sum (unweighted):     {comprehensive_score:.2f}")

    if weighted_result:
        print(f"  Weighted Average (0-100):    {weighted_result['weighted_score']:.2f}")
        print(f"  Market State:                {weighted_result['market_state']}")
        print(f"  Weight Sum:                  {weighted_result['weight_sum']:.2f}")
        print(f"  Valid Strategies:            {weighted_result['valid_strategies']}/18")
    else:
        print("  Weighted score calculation failed.")

    print("="*80 + "\n")

    calc.close()


if __name__ == '__main__':
    main()
