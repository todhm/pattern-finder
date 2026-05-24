"""
Korean Stock Value Factor System (Async Version)
Implements 16 value factor strategies to score stocks on a 100-point scale
File: kr_value_factor.py

Active Strategies: 14 (V2_Magic_Formula, V5_PSR deprecated due to negative IC)
- V2_Magic_Formula: IC -0.032 (60:40 optimization failed, 0/5 dates passed)
- V5_PSR: IC -0.0401 (negative correlation)

ASYNC CONVERSION: Uses asyncpg for database operations
"""

import os
import logging
import asyncio
from dotenv import load_dotenv
from datetime import datetime, timedelta

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
# Market State-based Value Strategy Weights (19 market states x 11 strategies)
# Phase 3.7: Active strategies - V2, V3, V4, V13, V14, V21~V26
# Deprecated: V1, V5~V12, V15~V20 (negative IC)
# ========================================================================

VALUE_STRATEGY_WEIGHTS = {
    # Large Cap Group (6)
    # V27-V30: Phase 3.8 Shareholder Value Paradigm - Higher weights in defensive/stable states
    'KOSPI대형-확장과열-공격형': {
        'V2': 0.5, 'V3': 1.2, 'V4': 0.8, 'V13': 0.8, 'V14': 0.9,
        'V21': 0.8, 'V22': 0.6, 'V23': 0.5, 'V24': 1.0, 'V25': 0.7, 'V26': 1.2,
        'V27': 0.8, 'V28': 1.2, 'V29': 0.6, 'V30': 0.9  # Phase 3.8
    },
    'KOSPI대형-확장중립-성장형': {
        'V2': 1.2, 'V3': 1.5, 'V4': 1.3, 'V13': 1.5, 'V14': 1.4,
        'V21': 1.0, 'V22': 1.0, 'V23': 0.8, 'V24': 1.2, 'V25': 0.9, 'V26': 1.3,
        'V27': 1.3, 'V28': 1.4, 'V29': 1.0, 'V30': 1.2  # Phase 3.8
    },
    'KOSPI대형-둔화공포-방어형': {
        'V2': 2.0, 'V3': 1.8, 'V4': 2.0, 'V13': 1.7, 'V14': 2.0,
        'V21': 1.5, 'V22': 1.8, 'V23': 1.3, 'V24': 0.8, 'V25': 1.6, 'V26': 1.0,
        'V27': 2.0, 'V28': 1.8, 'V29': 1.5, 'V30': 1.8  # Phase 3.8: 방어형에서 주주환원 중시
    },
    'KOSPI대형-침체패닉-초방어형': {
        'V2': 2.0, 'V3': 2.0, 'V4': 2.0, 'V13': 1.8, 'V14': 2.0,
        'V21': 1.8, 'V22': 2.0, 'V23': 1.5, 'V24': 0.6, 'V25': 2.0, 'V26': 0.8,
        'V27': 2.0, 'V28': 2.0, 'V29': 1.8, 'V30': 2.0  # Phase 3.8: 초방어형 최대 가중치
    },
    'KOSPI대형-회복탐욕-밸류형': {
        'V2': 1.7, 'V3': 1.6, 'V4': 1.5, 'V13': 1.8, 'V14': 1.6,
        'V21': 1.3, 'V22': 1.5, 'V23': 1.2, 'V24': 1.2, 'V25': 1.4, 'V26': 1.5,
        'V27': 1.6, 'V28': 1.5, 'V29': 1.2, 'V30': 1.5  # Phase 3.8
    },
    'KOSPI대형-중립안정-균형형': {
        'V2': 1.3, 'V3': 1.4, 'V4': 1.5, 'V13': 1.4, 'V14': 1.5,
        'V21': 1.0, 'V22': 1.2, 'V23': 1.0, 'V24': 1.0, 'V25': 1.0, 'V26': 1.0,
        'V27': 1.5, 'V28': 1.3, 'V29': 1.0, 'V30': 1.4  # Phase 3.8
    },

    # Mid Cap Group (6)
    'KOSPI중형-확장과열-모멘텀형': {
        'V2': 0.7, 'V3': 1.0, 'V4': 0.5, 'V13': 1.0, 'V14': 0.6,
        'V21': 0.7, 'V22': 0.5, 'V23': 0.5, 'V24': 1.1, 'V25': 0.6, 'V26': 1.3,
        'V27': 0.6, 'V28': 1.0, 'V29': 0.5, 'V30': 0.7  # Phase 3.8: 모멘텀형 낮은 가중치
    },
    'KOSPI중형-회복중립-성장형': {
        'V2': 1.5, 'V3': 1.4, 'V4': 1.0, 'V13': 1.6, 'V14': 1.2,
        'V21': 1.2, 'V22': 1.0, 'V23': 0.9, 'V24': 1.3, 'V25': 1.0, 'V26': 1.4,
        'V27': 1.2, 'V28': 1.3, 'V29': 0.9, 'V30': 1.1  # Phase 3.8
    },
    'KOSPI중형-둔화공포-혼조형': {
        'V2': 1.6, 'V3': 1.5, 'V4': 1.3, 'V13': 1.5, 'V14': 1.4,
        'V21': 1.4, 'V22': 1.5, 'V23': 1.2, 'V24': 0.9, 'V25': 1.4, 'V26': 1.1,
        'V27': 1.5, 'V28': 1.6, 'V29': 1.3, 'V30': 1.4  # Phase 3.8
    },
    'KOSDAQ중형-확장탐욕-공격성장형': {
        'V2': 0.5, 'V3': 0.8, 'V4': 0.3, 'V13': 0.8, 'V14': 0.4,
        'V21': 0.6, 'V22': 0.4, 'V23': 0.4, 'V24': 1.2, 'V25': 0.5, 'V26': 1.5,
        'V27': 0.4, 'V28': 0.9, 'V29': 0.5, 'V30': 0.6  # Phase 3.8: 공격성장형 낮은 가중치
    },
    'KOSDAQ중형-회복중립-성장테마형': {
        'V2': 1.0, 'V3': 1.2, 'V4': 0.5, 'V13': 1.3, 'V14': 0.6,
        'V21': 1.0, 'V22': 0.7, 'V23': 0.8, 'V24': 1.3, 'V25': 0.8, 'V26': 1.4,
        'V27': 0.7, 'V28': 1.1, 'V29': 0.8, 'V30': 0.9  # Phase 3.8
    },
    'KOSDAQ중형-침체공포-역발상형': {
        'V2': 1.8, 'V3': 1.7, 'V4': 1.0, 'V13': 1.6, 'V14': 0.8,
        'V21': 1.6, 'V22': 1.3, 'V23': 1.4, 'V24': 0.7, 'V25': 1.7, 'V26': 0.9,
        'V27': 1.4, 'V28': 1.8, 'V29': 1.4, 'V30': 1.5  # Phase 3.8
    },

    # Small Cap Group (4) - 소형주는 주주환원보다 성장 중시
    'KOSDAQ소형-핫섹터-초고위험형': {
        'V2': 0.3, 'V3': 0.5, 'V4': 0.1, 'V13': 0.6, 'V14': 0.2,
        'V21': 0.5, 'V22': 0.3, 'V23': 0.3, 'V24': 1.3, 'V25': 0.4, 'V26': 1.5,
        'V27': 0.3, 'V28': 0.8, 'V29': 0.4, 'V30': 0.5  # Phase 3.8: 소형 핫섹터 최저
    },
    'KOSDAQ소형-성장테마-고위험형': {
        'V2': 0.5, 'V3': 0.7, 'V4': 0.2, 'V13': 0.8, 'V14': 0.3,
        'V21': 0.6, 'V22': 0.4, 'V23': 0.4, 'V24': 1.2, 'V25': 0.5, 'V26': 1.4,
        'V27': 0.4, 'V28': 0.9, 'V29': 0.5, 'V30': 0.6  # Phase 3.8
    },
    'KOSDAQ소형-침체-극단역발상형': {
        'V2': 1.8, 'V3': 1.5, 'V4': 0.5, 'V13': 1.4, 'V14': 0.3,
        'V21': 1.7, 'V22': 1.0, 'V23': 1.5, 'V24': 0.6, 'V25': 1.8, 'V26': 0.7,
        'V27': 1.0, 'V28': 1.7, 'V29': 1.3, 'V30': 1.2  # Phase 3.8
    },
    'KOSDAQ소형-회복-모멘텀형': {
        'V2': 1.0, 'V3': 1.0, 'V4': 0.3, 'V13': 1.2, 'V14': 0.4,
        'V21': 1.0, 'V22': 0.6, 'V23': 0.8, 'V24': 1.3, 'V25': 0.8, 'V26': 1.4,
        'V27': 0.5, 'V28': 1.0, 'V29': 0.7, 'V30': 0.8  # Phase 3.8
    },

    # Special Situation Group (2)
    '전시장-극저유동성-고위험형': {
        'V2': 1.5, 'V3': 2.0, 'V4': 1.8, 'V13': 1.3, 'V14': 1.5,
        'V21': 1.5, 'V22': 1.6, 'V23': 1.3, 'V24': 0.7, 'V25': 1.8, 'V26': 0.8,
        'V27': 1.8, 'V28': 1.6, 'V29': 1.5, 'V30': 1.7  # Phase 3.8: 저유동성에서 안전 중시
    },
    '테마특화-모멘텀폭발형': {
        'V2': 0.3, 'V3': 0.6, 'V4': 0.2, 'V13': 0.7, 'V14': 0.3,
        'V21': 0.5, 'V22': 0.3, 'V23': 0.4, 'V24': 1.3, 'V25': 0.4, 'V26': 1.5,
        'V27': 0.3, 'V28': 0.7, 'V29': 0.4, 'V30': 0.5  # Phase 3.8: 테마폭발 최저
    },

    # Others (fallback)
    '기타': {
        'V2': 1.0, 'V3': 1.0, 'V4': 1.0, 'V13': 1.0, 'V14': 1.0,
        'V21': 1.0, 'V22': 1.0, 'V23': 1.0, 'V24': 1.0, 'V25': 1.0, 'V26': 1.0,
        'V27': 1.0, 'V28': 1.0, 'V29': 1.0, 'V30': 1.0  # Phase 3.8
    }
}


class ValueFactorCalculator:
    """Calculate value factor scores using 16 different strategies (Async version)"""

    def __init__(self, symbol, db_manager, market_state=None, analysis_date=None):
        """
        Initialize Value Factor Calculator

        Args:
            symbol: Stock symbol
            db_manager: AsyncDatabaseManager instance
            market_state: Market state classification (optional)
            analysis_date: Specific date for analysis (optional, defaults to latest)
        """
        self.symbol = symbol
        self.db_manager = db_manager
        self.market_state = market_state
        self.analysis_date = analysis_date
        self.strategies_scores = {}

    async def execute_query(self, query, *params):
        """
        Execute SQL query and return results

        Args:
            query: SQL query string (asyncpg format with $1, $2, ...)
            params: Query parameters

        Returns:
            List of dict results
        """
        try:
            result = await self.db_manager.execute_query(query, *params)
            return result
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            logger.error(f"Query: {query}")
            raise

    # ========================================================================
    # Sector Health Check (Phase 2)
    # ========================================================================

    async def calculate_sector_health_score(self):
        """
        섹터 건강도 평가 (MV 최적화 버전)

        OPTIMIZED: mv_sector_daily_performance 사용
        - 기존: 복잡한 CTE 쿼리 (2,748종목 × 60일 JOIN) → 연결 타임아웃 발생
        - 변경: MV에서 사전 계산된 섹터 수익률 조회 (단일 행)

        Returns:
            int: 0-100 (0=재앙, 100=최고)
        """

        # Step 1: 종목의 섹터 조회
        sector_query = """
        SELECT theme
        FROM kr_stock_detail
        WHERE symbol = $1
        """

        sector_result = await self.execute_query(sector_query, self.symbol)

        if not sector_result or not sector_result[0]['theme']:
            logger.info(f"Sector Health: No sector data for {self.symbol}, returning neutral")
            return 50  # 중립

        sector = sector_result[0]['theme']

        # Step 2: MV에서 섹터 수익률과 시장 평균 조회 (단일 쿼리)
        mv_query = """
        SELECT
            mv.avg_return_30d as sector_return,
            (SELECT AVG(avg_return_30d) FROM mv_sector_daily_performance WHERE date = $2) as market_return
        FROM mv_sector_daily_performance mv
        WHERE mv.sector_code = $1 AND mv.date = $2
        """

        mv_result = await self.execute_query(mv_query, sector, self.analysis_date)

        if not mv_result or mv_result[0]['sector_return'] is None:
            logger.info(f"Sector Health: No MV data for sector {sector} on {self.analysis_date}, returning neutral")
            return 50

        sector_return_30d = float(mv_result[0]['sector_return'])
        market_return_30d = float(mv_result[0]['market_return']) if mv_result[0]['market_return'] else 0

        # Step 3: 섹터 알파
        sector_alpha = sector_return_30d - market_return_30d

        # Step 4: 건강도 점수
        health_score = 50  # 기본값

        if sector_return_30d < -10:
            health_score = 20  # 섹터 폭락
            logger.info(f"Sector Health: {health_score} - Sector crash ({sector}: {sector_return_30d:.1f}%)")
        elif sector_alpha < -5:
            health_score = 40  # 섹터 부진
            logger.info(f"Sector Health: {health_score} - Sector underperforming ({sector}: α={sector_alpha:.1f}%)")
        elif sector_return_30d > 5 and sector_alpha > 0:
            health_score = 80  # 섹터 강세
            logger.info(f"Sector Health: {health_score} - Sector outperforming ({sector}: α=+{sector_alpha:.1f}%)")
        else:
            health_score = 50  # 중립
            logger.info(f"Sector Health: {health_score} - Sector neutral ({sector}: {sector_return_30d:.1f}%)")

        return health_score


    # ========================================================================
    # V1. Low PER Strategy (Sector-Health-Weighted)
    # ========================================================================

    async def calculate_v1(self):
        """
        V1. Low PER Strategy (Enhanced with fallback)
        Description: Find undervalued stocks with low price-to-earnings ratio
        Score = 100 - MIN(100, MAX(0, (PER / 15) × 50))
        Condition: PER > 0 (exclude negative PER)
        Interpretation: PER <= 15 gets 100 points, PER >= 30 gets 0 points
        """
        query = """
        SELECT per
        FROM kr_intraday_detail
        WHERE symbol = $1
        """

        result = await self.execute_query(query, self.symbol)
        per = None

        # Try to get PER from kr_intraday_detail first
        if result and result[0]['per'] is not None and result[0]['per'] > 0:
            per = float(result[0]['per'])

        # Fallback: Calculate PER from market cap and net income
        if per is None:
            fallback_query = """
            SELECT
                kit.market_cap,
                fp.thstrm_amount as net_income
            FROM kr_intraday_total kit
            LEFT JOIN (
                SELECT symbol, thstrm_amount
                FROM kr_financial_position
                WHERE symbol = $1
                    AND sj_div = 'IS'
                    AND account_nm IN ('당기순이익(손실)', '당기순이익')
                    AND thstrm_amount > 0
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
                    thstrm_amount DESC
                LIMIT 1
            ) fp ON kit.symbol = fp.symbol
            WHERE kit.symbol = $2
                AND ($3::date IS NULL OR kit.date = $3)
            ORDER BY kit.date DESC
            LIMIT 1
            """

            fallback_result = await self.execute_query(fallback_query, self.symbol, self.symbol, self.analysis_date)

            if fallback_result and fallback_result[0]['market_cap'] and fallback_result[0]['net_income']:
                mktcap = float(fallback_result[0]['market_cap'])
                net_income = float(fallback_result[0]['net_income'])
                if net_income > 0 and mktcap > 0:
                    # PER = Market Cap / Net Income
                    per = mktcap / net_income

        # 적자 기업 처리: V17로 폴백
        if per is None or per <= 0:
            logger.info(f"V1: PER invalid ({per}), falling back to V17 (Growth-Adjusted Value)")
            return await self.calculate_v17_growth_adjusted_value()

        # 흑자 기업: Sector-Health-Weighted Peer-Relative 평가

        # Step 1: 섹터 건강도 체크
        sector_health = await self.calculate_sector_health_score()

        # Step 2: 섹터 중앙값 PER
        sector_per_query = """
        WITH sector_stocks AS (
            SELECT sd.symbol, sd.theme
            FROM kr_stock_detail sd
            WHERE sd.symbol = $1
        ),
        peer_pers AS (
            SELECT per
            FROM kr_intraday_detail kid
            JOIN kr_stock_detail sd ON kid.symbol = sd.symbol
            WHERE sd.theme = (SELECT theme FROM sector_stocks)
                AND kid.per > 0
                AND kid.per < 100
        )
        SELECT PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY per) as median_per
        FROM peer_pers
        """

        sector_per_result = await self.execute_query(sector_per_query, self.symbol)

        # Step 3: 전체 시장 중앙값 PER
        market_per_query = """
        SELECT PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY per) as median_per
        FROM kr_intraday_detail
        WHERE per > 0 AND per < 100
        """

        market_per_result = await self.execute_query(market_per_query)

        sector_median_per = 15  # 기본값
        market_median_per = 15  # 기본값

        if sector_per_result and sector_per_result[0]['median_per']:
            sector_median_per = float(sector_per_result[0]['median_per'])

        if market_per_result and market_per_result[0]['median_per']:
            market_median_per = float(market_per_result[0]['median_per'])

        # Step 4: Discount 계산
        peer_discount = (per - sector_median_per) / sector_median_per
        market_discount = (per - market_median_per) / market_median_per

        # Step 5: 섹터 건강도 기반 점수
        score = 50

        if sector_health >= 70:
            # 건강한 섹터: Peer-Relative 신뢰
            if peer_discount < -0.30:
                score = 90
            elif peer_discount < -0.15:
                score = 75
            elif peer_discount < 0:
                score = 60
            else:
                score = 40
            logger.info(f"V1: Healthy sector peer-relative (PER: {per:.1f}, Sector median: {sector_median_per:.1f}, Discount: {peer_discount*100:.1f}%)")

        elif sector_health >= 40:
            # 중립 섹터: Peer + Absolute 혼합
            peer_score = 60 if peer_discount < -0.30 else 50 if peer_discount < 0 else 35
            absolute_score = 100 - min(100, max(0, (per / 15) * 50))
            score = peer_score * 0.7 + absolute_score * 0.3
            logger.info(f"V1: Neutral sector mixed (PER: {per:.1f}, Peer: {peer_score}, Absolute: {absolute_score:.0f}, Health: {sector_health})")

        else:
            # 망한 섹터: 절대 평가 + 페널티
            absolute_score = 100 - min(100, max(0, (per / 15) * 50))

            if market_discount < -0.50 and sector_health >= 20:
                # 망한 섹터지만 절대적으로 초저평가
                score = absolute_score * 0.8  # 페널티 20%
                logger.info(f"V1: Weak sector but ultra-cheap (PER: {per:.1f}, Market discount: {market_discount*100:.1f}%, Penalty: 20%)")
            else:
                score = absolute_score * 0.6  # 페널티 40%
                logger.info(f"V1: Weak sector penalty applied (PER: {per:.1f}, Health: {sector_health}, Penalty: 40%)")

        return score

    # ========================================================================
    # V2. Magic Formula Strategy (DEPRECATED - IC -0.032, 0/5 dates passed)
    # ========================================================================

    async def calculate_v2(self):
        """
        V2. Magic Formula Strategy (Korean Market Adapted) - DEPRECATED

        Previous IC: -0.1645 (ROE Trend version failed)
        Target IC: > 0

        Components:
        1. Earnings Yield (60%): Operating Profit / Market Cap
           - Positive: Normal scoring (10% = 100 points)
           - Negative: Turnaround scoring (improvement vs prev year, max 30 points)
        2. Quality Score (40%): ROE (60%) + Operating Margin (40%)

        Magic Formula Logic:
        - Earnings Yield = measure of undervaluation
        - Quality = combination of profitability (ROE) and efficiency (margin)
        - Independent from V3 (correlation 0.2-0.3, no CF overlap)

        Data Coverage: 97-100% (IS + BS only, no CF dependency)
        """
        # Step 1: Get Market Cap from kr_intraday_total
        market_cap_query = """
        SELECT market_cap
        FROM kr_intraday_total
        WHERE symbol = $1
            AND date = COALESCE($2::date, CURRENT_DATE)
        """

        market_cap_result = await self.execute_query(market_cap_query, self.symbol, self.analysis_date)

        if not market_cap_result or market_cap_result[0]['market_cap'] is None or market_cap_result[0]['market_cap'] <= 0:
            return None

        market_cap = float(market_cap_result[0]['market_cap'])

        # Step 2: Get Financial Data (Operating Profit, Net Income, Equity, Sales)
        financial_query = """
        WITH latest_report AS (
            SELECT bsns_year, rcept_dt
            FROM kr_financial_position
            WHERE symbol = $1
                AND rcept_dt <= COALESCE($2::date, CURRENT_DATE)
            ORDER BY bsns_year DESC, rcept_dt DESC
            LIMIT 1
        ),
        prev_report AS (
            SELECT bsns_year, rcept_dt
            FROM kr_financial_position
            WHERE symbol = $1
                AND rcept_dt <= COALESCE($2::date, CURRENT_DATE)
                AND bsns_year < (SELECT bsns_year FROM latest_report)
            ORDER BY bsns_year DESC, rcept_dt DESC
            LIMIT 1
        )
        SELECT
            (SELECT thstrm_amount FROM kr_financial_position, latest_report
             WHERE symbol = $1 AND sj_div = 'IS'
             AND account_nm IN ('영업이익(손실)', '영업이익')
             AND kr_financial_position.bsns_year = latest_report.bsns_year
             AND kr_financial_position.rcept_dt = latest_report.rcept_dt
             LIMIT 1) as operating_profit,

            (SELECT thstrm_amount FROM kr_financial_position, prev_report
             WHERE symbol = $1 AND sj_div = 'IS'
             AND account_nm IN ('영업이익(손실)', '영업이익')
             AND kr_financial_position.bsns_year = prev_report.bsns_year
             AND kr_financial_position.rcept_dt = prev_report.rcept_dt
             LIMIT 1) as prev_operating_profit,

            (SELECT thstrm_amount FROM kr_financial_position, latest_report
             WHERE symbol = $1 AND sj_div = 'IS'
             AND account_nm IN ('당기순이익(손실)', '당기순이익')
             AND kr_financial_position.bsns_year = latest_report.bsns_year
             AND kr_financial_position.rcept_dt = latest_report.rcept_dt
             LIMIT 1) as net_income,

            (SELECT thstrm_amount FROM kr_financial_position, latest_report
             WHERE symbol = $1 AND sj_div = 'BS'
             AND account_nm IN ('기말자본', '자본총계')
             AND thstrm_amount > 0
             AND kr_financial_position.bsns_year = latest_report.bsns_year
             AND kr_financial_position.rcept_dt = latest_report.rcept_dt
             ORDER BY thstrm_amount DESC
             LIMIT 1) as total_equity,

            (SELECT thstrm_amount FROM kr_financial_position, latest_report
             WHERE symbol = $1 AND sj_div = 'IS'
             AND account_nm IN ('매출액', '수익(매출액)')
             AND kr_financial_position.bsns_year = latest_report.bsns_year
             AND kr_financial_position.rcept_dt = latest_report.rcept_dt
             LIMIT 1) as sales
        """

        financial_result = await self.execute_query(financial_query, self.symbol, self.analysis_date)

        if not financial_result:
            return None

        operating_profit = financial_result[0]['operating_profit']
        prev_operating_profit = financial_result[0]['prev_operating_profit']
        net_income = financial_result[0]['net_income']
        total_equity = financial_result[0]['total_equity']
        sales = financial_result[0]['sales']

        # Validate required data
        if operating_profit is None or net_income is None or total_equity is None or sales is None:
            return None

        if total_equity <= 0 or sales <= 0:
            return None

        operating_profit = float(operating_profit)
        net_income = float(net_income)
        total_equity = float(total_equity)
        sales = float(sales)

        # Component 1: Earnings Yield (50%)
        earnings_yield = (operating_profit / market_cap) * 100

        if earnings_yield >= 0:
            # Positive: 10% earnings yield = 100 points
            ey_score = min(100.0, earnings_yield * 10)
        else:
            # Negative: Option A - Check turnaround (YoY improvement)
            if prev_operating_profit is not None:
                prev_operating_profit = float(prev_operating_profit)

                # Check if improved from previous year
                if operating_profit > prev_operating_profit:
                    # Calculate improvement rate
                    improvement_rate = ((operating_profit - prev_operating_profit) / abs(prev_operating_profit)) * 100
                    # Give partial credit: max 30 points for turnaround situations
                    ey_score = min(30.0, max(0.0, 30 + improvement_rate * 2))
                else:
                    # No improvement: 0 points
                    ey_score = 0.0
            else:
                # No previous data: 0 points for negative operating profit
                ey_score = 0.0

        # Component 2: Quality Score (50%)
        # Sub-component 2a: ROE (60%)
        roe = (net_income / total_equity) * 100

        if roe >= 20:
            roe_score = 100.0
        elif roe >= 15:
            roe_score = 80.0 + (roe - 15) * 4
        elif roe >= 10:
            roe_score = 60.0 + (roe - 10) * 4
        elif roe >= 5:
            roe_score = 40.0 + (roe - 5) * 4
        elif roe >= 0:
            roe_score = roe * 8
        else:
            roe_score = 0.0

        # Sub-component 2b: Operating Margin (40%)
        operating_margin = (operating_profit / sales) * 100

        if operating_margin >= 20:
            margin_score = 100.0
        elif operating_margin >= 15:
            margin_score = 80.0 + (operating_margin - 15) * 4
        elif operating_margin >= 10:
            margin_score = 60.0 + (operating_margin - 10) * 4
        elif operating_margin >= 5:
            margin_score = 40.0 + (operating_margin - 5) * 4
        elif operating_margin >= 0:
            margin_score = operating_margin * 8
        else:
            margin_score = 0.0

        # Quality Score = ROE (60%) + Operating Margin (40%)
        quality_score = (roe_score * 0.6) + (margin_score * 0.4)

        # Base Score: Earnings Yield (60%) + Quality Score (40%)
        base_score = (ey_score * 0.6) + (quality_score * 0.4)

        # Sector-Health-Weighted 평가
        sector_health = await self.calculate_sector_health_score()

        # 섹터 중앙값 Earnings Yield 계산
        sector_ey_query = """
        WITH sector_stocks AS (
            SELECT sd.symbol, sd.theme
            FROM kr_stock_detail sd
            WHERE sd.symbol = $1
        ),
        peer_ey AS (
            SELECT
                (fp.thstrm_amount / kit.market_cap) * 100 as earnings_yield
            FROM kr_intraday_total kit
            JOIN kr_stock_detail sd ON kit.symbol = sd.symbol
            LEFT JOIN (
                SELECT DISTINCT ON (symbol)
                    symbol, thstrm_amount
                FROM kr_financial_position
                WHERE sj_div = 'IS'
                    AND account_nm IN ('영업이익(손실)', '영업이익')
                    AND rcept_dt <= COALESCE($2::date, CURRENT_DATE)
                ORDER BY symbol, bsns_year DESC, rcept_dt DESC
            ) fp ON kit.symbol = fp.symbol
            WHERE sd.theme = (SELECT theme FROM sector_stocks)
                AND ($2::date IS NULL OR kit.date = $2)
                AND kit.market_cap > 0
                AND fp.thstrm_amount > 0
        )
        SELECT PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY earnings_yield) as median_ey
        FROM peer_ey
        WHERE earnings_yield > 0
        """

        sector_ey_result = await self.execute_query(sector_ey_query, self.symbol, self.analysis_date)

        sector_median_ey = 5.0  # 기본값
        if sector_ey_result and sector_ey_result[0]['median_ey']:
            sector_median_ey = float(sector_ey_result[0]['median_ey'])

        # Earnings Yield 상대 평가
        if earnings_yield > 0 and sector_median_ey > 0:
            ey_premium = (earnings_yield - sector_median_ey) / sector_median_ey
        else:
            ey_premium = 0

        # 섹터 건강도 기반 최종 점수
        if sector_health >= 70:
            # 건강한 섹터: 상대평가 가중
            if ey_premium > 0.30:  # 섹터 대비 30% 이상 높음
                final_score = base_score * 1.2
            elif ey_premium > 0:
                final_score = base_score * 1.1
            else:
                final_score = base_score * 0.9
            logger.info(f"V3: Healthy sector (EY: {earnings_yield:.1f}%, Sector median: {sector_median_ey:.1f}%, Premium: {ey_premium*100:.1f}%)")

        elif sector_health >= 40:
            # 중립 섹터: 기본 점수 유지
            final_score = base_score
            logger.info(f"V3: Neutral sector (EY: {earnings_yield:.1f}%, Base score: {base_score:.1f})")

        else:
            # 망한 섹터: 페널티 적용
            if earnings_yield > 10:  # 절대적으로 높은 EY
                final_score = base_score * 0.8  # 페널티 20%
                logger.info(f"V3: Weak sector but high EY ({earnings_yield:.1f}%), Penalty: 20%)")
            else:
                final_score = base_score * 0.6  # 페널티 40%
                logger.info(f"V3: Weak sector penalty (EY: {earnings_yield:.1f}%, Health: {sector_health})")

        return min(100, final_score)

    async def calculate_v2_original(self):
        """
        V2. Low PBR + High ROE Strategy (ORIGINAL Level-based version)

        Original IC: -0.0400 (Negative)
        This is the original version for IC comparison purposes

        Components:
        1. PBR Score (60%): 100 - MIN(100, MAX(0, (PBR - 0.5) × 50))
        2. ROE Level Score (40%): MIN(100, MAX(0, ROE × 5))

        Final Score = (PBR × 0.6) + (ROE Level × 0.4)
        """
        # Get PBR
        pbr_query = """
        SELECT pbr
        FROM kr_intraday_detail
        WHERE symbol = $1
        """

        pbr_result = await self.execute_query(pbr_query, self.symbol)

        if not pbr_result or pbr_result[0]['pbr'] is None or pbr_result[0]['pbr'] <= 0:
            return None

        pbr = float(pbr_result[0]['pbr'])

        # Get Net Income and Total Equity
        roe_query = """
        SELECT
            fp1.thstrm_amount as net_income,
            fp2.thstrm_amount as total_equity
        FROM kr_financial_position fp1
        JOIN kr_financial_position fp2
            ON fp1.symbol = fp2.symbol
            AND fp1.bsns_year = fp2.bsns_year
            AND fp1.rcept_dt = fp2.rcept_dt
        WHERE fp1.symbol = $1
            AND fp1.account_nm IN ('당기순이익(손실)', '당기순이익')
            AND fp2.account_nm IN ('기말자본', '자본총계')
            AND fp1.sj_div = 'IS'
            AND fp2.sj_div = 'BS'
            AND fp1.rcept_dt <= COALESCE($2::date, CURRENT_DATE)
        ORDER BY fp1.bsns_year DESC, fp1.rcept_dt DESC
        LIMIT 1
        """

        roe_result = await self.execute_query(roe_query, self.symbol, self.analysis_date)

        net_income = None
        total_equity = None

        if roe_result and roe_result[0]['net_income'] and roe_result[0]['total_equity']:
            net_income = float(roe_result[0]['net_income'])
            total_equity = float(roe_result[0]['total_equity'])

        if not net_income or not total_equity or total_equity <= 0:
            return None

        current_roe = (net_income / total_equity) * 100

        if current_roe <= 0:
            return None

        # Calculate component scores (ORIGINAL version - no trend)
        pbr_score = 100 - min(100, max(0, (pbr - 0.5) * 50))
        roe_level_score = min(100, max(0, current_roe * 5))

        # Final score: PBR (60%) + ROE Level (40%) - ORIGINAL weights
        final_score = (pbr_score * 0.6) + (roe_level_score * 0.4)

        return final_score

    # ========================================================================
    # V3. Net Cash Flow Yield Strategy
    # ========================================================================

    async def calculate_v3_original(self):
        """
        V3. Net Cash Flow Yield Strategy (ORIGINAL - For IC Comparison)

        IC: -0.0180 (Negative correlation)

        Description: Operating cash flow based valuation
        Primary: Operating Cash Flow / Market Cap × 100 (from CF statement)
        Fallback: FCF Proxy (영업이익 × 0.7 - Capex) / Market Cap × 100 (from IS + BS)
        Score = MIN(100, MAX(0, Cash Flow Yield × 10))
        Interpretation: 10%+ cash flow yield yields 100 points

        This is the original version for IC comparison purposes.
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

        if not mc_result or mc_result[0]['market_cap'] is None:
            return None

        market_cap = float(mc_result[0]['market_cap'])

        cash_flow_value = None

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

        if ocf_result and ocf_result[0]['thstrm_amount']:
            cash_flow_value = float(ocf_result[0]['thstrm_amount'])

        # Fallback: Use FCF Proxy if CF data not available
        if cash_flow_value is None:
            fcf_proxy_query = """
            WITH latest_report AS (
                SELECT bsns_year, rcept_dt
                FROM kr_financial_position
                WHERE symbol = $1
                    AND sj_div = 'IS'
                    AND rcept_dt <= COALESCE($2::date, CURRENT_DATE)
                ORDER BY bsns_year DESC, rcept_dt DESC
                LIMIT 1
            ),
            operating_profit AS (
                SELECT fp.thstrm_amount
                FROM kr_financial_position fp
                INNER JOIN latest_report lr
                    ON fp.bsns_year = lr.bsns_year
                    AND fp.rcept_dt = lr.rcept_dt
                WHERE fp.symbol = $1
                    AND fp.sj_div = 'IS'
                    AND fp.account_nm IN ('영업이익', '영업이익(손실)')
                LIMIT 1
            ),
            tangible_assets AS (
                SELECT
                    fp.thstrm_amount as current_assets,
                    fp.frmtrm_amount as prev_assets
                FROM kr_financial_position fp
                INNER JOIN latest_report lr
                    ON fp.bsns_year = lr.bsns_year
                    AND fp.rcept_dt = lr.rcept_dt
                WHERE fp.symbol = $1
                    AND fp.sj_div = 'BS'
                    AND fp.account_nm = '유형자산'
                LIMIT 1
            )
            SELECT
                op.thstrm_amount as operating_profit,
                COALESCE(ta.current_assets - COALESCE(ta.prev_assets, 0), 0) as capex_proxy
            FROM operating_profit op
            LEFT JOIN tangible_assets ta ON true
            """

            fcf_result = await self.execute_query(fcf_proxy_query, self.symbol, self.analysis_date)

            if fcf_result and fcf_result[0]['operating_profit']:
                operating_profit = float(fcf_result[0]['operating_profit'])
                capex_proxy = float(fcf_result[0]['capex_proxy']) if fcf_result[0]['capex_proxy'] else 0

                # Calculate FCF Proxy
                # After-tax operating profit (assume 30% tax rate)
                after_tax_op_profit = operating_profit * 0.7

                # FCF Proxy = After-tax operating profit - Capex
                cash_flow_value = after_tax_op_profit - capex_proxy

        if cash_flow_value is None:
            return None

        # Calculate Cash Flow Yield
        cf_yield = (cash_flow_value / market_cap) * 100

        # Scoring (동일 기준 적용)
        # Positive CF: 0~10% → 0~100점
        # Negative CF: 0점 (현금 소진 중 or 대규모 투자 중)
        if cf_yield < 0:
            score = 0
        else:
            score = min(100, cf_yield * 10)

        return score

    async def calculate_v3(self):
        """
        V3. Cash Flow Sustainability Strategy (REDESIGNED)

        Target IC: > +0.02 (improved from -0.0180)

        Components:
        1. OCF Stability (40%): 3-year OCF volatility (Coefficient of Variation)
        2. OCF Growth (30%): 3-year OCF CAGR
        3. FCF Yield (20%): (OCF - CapEx) / Market Cap
        4. Reinvestment Rate (10%): CapEx / OCF ratio

        Data Sources (CF Statement ONLY):
        - 영업활동현금흐름 (OCF) - 86% coverage
        - 유형자산의취득 (CapEx) - 90% coverage

        Improvements:
        - Uses actual CF statement data (not proxy)
        - Considers stability and growth trend
        - Allows negative FCF (for growth companies)
        """
        # Get market cap for FCF Yield calculation
        market_cap_query = """
        SELECT market_cap
        FROM kr_intraday_total
        WHERE symbol = $1
            AND ($2::date IS NULL OR date = $2)
        ORDER BY date DESC
        LIMIT 1
        """

        mc_result = await self.execute_query(market_cap_query, self.symbol, self.analysis_date)
        if not mc_result or mc_result[0]['market_cap'] is None:
            return None

        market_cap = float(mc_result[0]['market_cap'])

        # Component 1 & 2: Get 3-year OCF history
        ocf_history_query = """
        WITH yearly_ocf AS (
            SELECT
                bsns_year,
                thstrm_amount as ocf,
                ROW_NUMBER() OVER (
                    PARTITION BY bsns_year
                    ORDER BY rcept_dt DESC
                ) as rn
            FROM kr_financial_position
            WHERE symbol = $1
                AND sj_div = 'CF'
                AND (
                    account_nm LIKE '%영업활동%현금%'
                    OR account_nm = '영업활동현금흐름'
                    OR account_nm = '영업활동으로인한현금흐름'
                )
                AND rcept_dt <= COALESCE($2::date, CURRENT_DATE)
                AND thstrm_amount IS NOT NULL
        )
        SELECT bsns_year, ocf
        FROM yearly_ocf
        WHERE rn = 1
        ORDER BY bsns_year DESC
        LIMIT 3
        """

        ocf_history = await self.execute_query(ocf_history_query, self.symbol, self.analysis_date)

        # Component 1: OCF Stability (Coefficient of Variation)
        stability_score = 50.0  # Default neutral
        if ocf_history and len(ocf_history) >= 2:
            ocf_values = [float(row['ocf']) for row in ocf_history]
            ocf_mean = sum(ocf_values) / len(ocf_values)

            if ocf_mean > 0:
                # Calculate standard deviation
                variance = sum([(x - ocf_mean) ** 2 for x in ocf_values]) / len(ocf_values)
                ocf_std = variance ** 0.5
                cv = ocf_std / ocf_mean

                # CV Scoring: Lower CV = More stable = Higher score
                if cv < 0.2:
                    stability_score = 100.0
                elif cv < 0.5:
                    stability_score = 100.0 - (cv - 0.2) * 167
                elif cv < 1.0:
                    stability_score = 50.0 - (cv - 0.5) * 100
                else:
                    stability_score = max(0.0, 50.0 - (cv - 1.0) * 50)
            else:
                stability_score = 20.0  # Negative average OCF

        # Component 2: OCF Growth
        growth_score = 50.0  # Default neutral
        if ocf_history and len(ocf_history) >= 2:
            ocf_sorted = sorted(ocf_history, key=lambda x: x['bsns_year'])
            growth_rates = []

            for i in range(1, len(ocf_sorted)):
                prev_ocf = float(ocf_sorted[i-1]['ocf'])
                curr_ocf = float(ocf_sorted[i]['ocf'])

                if prev_ocf != 0:
                    growth = ((curr_ocf - prev_ocf) / abs(prev_ocf)) * 100
                    growth_rates.append(growth)

            if growth_rates:
                avg_growth = sum(growth_rates) / len(growth_rates)
                # +20% = 100점, 0% = 50점, -20% = 0점
                growth_score = max(0.0, min(100.0, 50.0 + (avg_growth * 2.5)))

        # Component 3 & 4: Latest OCF and CapEx for FCF Yield
        latest_cf_query = """
        WITH latest_ocf AS (
            SELECT thstrm_amount as ocf
            FROM kr_financial_position
            WHERE symbol = $1
                AND sj_div = 'CF'
                AND (
                    account_nm LIKE '%영업활동%현금%'
                    OR account_nm = '영업활동현금흐름'
                    OR account_nm = '영업활동으로인한현금흐름'
                )
                AND rcept_dt <= COALESCE($2::date, CURRENT_DATE)
                AND thstrm_amount IS NOT NULL
            ORDER BY bsns_year DESC, rcept_dt DESC
            LIMIT 1
        ),
        latest_capex AS (
            SELECT thstrm_amount as capex
            FROM kr_financial_position
            WHERE symbol = $1
                AND sj_div = 'CF'
                AND (
                    account_nm LIKE '%유형자산%취득%'
                    OR account_nm = '유형자산의취득'
                )
                AND rcept_dt <= COALESCE($2::date, CURRENT_DATE)
                AND thstrm_amount IS NOT NULL
            ORDER BY bsns_year DESC, rcept_dt DESC
            LIMIT 1
        )
        SELECT
            (SELECT ocf FROM latest_ocf) as ocf,
            (SELECT capex FROM latest_capex) as capex
        """

        latest_cf = await self.execute_query(latest_cf_query, self.symbol, self.analysis_date)

        fcf_yield_score = 50.0  # Default neutral
        reinvest_score = 50.0   # Default neutral

        if latest_cf and latest_cf[0]['ocf'] is not None:
            ocf = float(latest_cf[0]['ocf'])
            capex = abs(float(latest_cf[0]['capex'])) if latest_cf[0]['capex'] else 0

            # Component 3: FCF Yield
            fcf = ocf - capex
            fcf_yield = (fcf / market_cap) * 100

            # FCF Yield Scoring: 10% = 100점, 0% = 50점
            if fcf_yield >= 0:
                fcf_yield_score = min(100.0, fcf_yield * 10)
            else:
                # Negative FCF: Allow some score for growth companies
                # -5% FCF yield = 0점, 0% = 50점
                fcf_yield_score = max(0.0, 50.0 + (fcf_yield * 10))

            # Component 4: Reinvestment Rate
            if ocf > 0 and capex > 0:
                reinvest_rate = capex / ocf

                # Optimal: 30-70% reinvestment
                if 0.3 <= reinvest_rate <= 0.7:
                    reinvest_score = 100.0
                elif reinvest_rate < 0.3:
                    # Too little reinvestment
                    reinvest_score = (reinvest_rate / 0.3) * 100.0
                else:
                    # Too much reinvestment (>70%)
                    reinvest_score = max(0.0, 100.0 - (reinvest_rate - 0.7) * 100)

        # Final Score: Weighted average
        final_score = (
            stability_score * 0.4 +
            growth_score * 0.3 +
            fcf_yield_score * 0.2 +
            reinvest_score * 0.1
        )

        return max(0.0, min(100.0, final_score))

    # ========================================================================
    # Extreme Risk Penalty Helpers (for V4, V25)
    # ========================================================================

    async def _calculate_volatility_penalty(self, volatility_annual):
        """
        Extreme volatility penalty for V4/V25

        Based on empirical analysis (v4_v25_failure_analysis.csv):
        - Vol > 100%: Average -47.3% return (16 stocks)
        - Vol > 200%: Extreme speculative risk
        """
        if volatility_annual is None:
            return 0

        if volatility_annual > 200:
            return -50
        elif volatility_annual > 100:
            return -30
        elif volatility_annual > 60:
            return -15
        elif volatility_annual > 40:
            return -5
        else:
            return 0

    async def _calculate_beta_penalty(self, beta):
        """
        Beta-based risk penalty for V4/V25

        Based on empirical analysis:
        - Average beta of failed stocks: 15.5
        - Beta > 5.0: Extreme market sensitivity
        """
        if beta is None:
            return 0

        if beta > 5.0:
            return -40
        elif beta > 3.0:
            return -25
        elif beta > 2.0:
            return -10
        elif beta > 1.5:
            return -5
        else:
            return 0

    async def _calculate_var_penalty(self, var_95):
        """
        VaR-based risk penalty for V4/V25

        VaR(95%): Daily maximum loss at 95% confidence level
        """
        if var_95 is None or var_95 >= 0:
            return 0

        var_abs = abs(var_95)

        if var_abs > 8.0:
            return -30
        elif var_abs > 5.0:
            return -15
        elif var_abs > 3.0:
            return -5
        else:
            return 0

    async def _get_volatility_for_risk_adjustment(self):
        """Get volatility_annual for risk adjustment"""
        query = """
        SELECT volatility_annual
        FROM kr_stock_grade
        WHERE symbol = $1
            AND ($2::date IS NULL OR date = $2)
        ORDER BY date DESC
        LIMIT 1
        """
        result = await self.execute_query(query, self.symbol, self.analysis_date)
        if result and result[0]['volatility_annual']:
            return float(result[0]['volatility_annual'])
        return None

    async def _get_beta_for_risk_adjustment(self):
        """Get beta for risk adjustment"""
        query = """
        SELECT beta
        FROM kr_stock_grade
        WHERE symbol = $1
            AND ($2::date IS NULL OR date = $2)
        ORDER BY date DESC
        LIMIT 1
        """
        result = await self.execute_query(query, self.symbol, self.analysis_date)
        if result and result[0]['beta']:
            return float(result[0]['beta'])
        return None

    async def _get_var_for_risk_adjustment(self):
        """Get VaR(95%) for risk adjustment"""
        query = """
        SELECT var_95
        FROM kr_stock_grade
        WHERE symbol = $1
            AND ($2::date IS NULL OR date = $2)
        ORDER BY date DESC
        LIMIT 1
        """
        result = await self.execute_query(query, self.symbol, self.analysis_date)
        if result and result[0]['var_95']:
            return float(result[0]['var_95'])
        return None

    # ========================================================================
    # V4. Sustainable Dividend Strategy (재설계)
    # ========================================================================

    async def calculate_v4(self):
        """
        V4. Sustainable Dividend Strategy (재설계)

        Description: 지속 가능한 배당 평가

        Components:
        1. Payout Ratio (배당성향): >80% = 위험
        2. FCF Coverage (현금흐름 커버리지): FCF < Dividend = 위험
        3. Dividend Growth (배당 성장): 3년 CAGR
        4. Dividend Yield: 최종 평가

        Scoring Logic:
        - 위험 신호 탐지: Payout >80%, FCF < Div, Div 감소 → 10-20점
        - 우수 배당: 성장 >5%, Payout <50%, FCF >1.5x → 75-90점
        - 안정 배당: 유지, Payout <60%, FCF >1.2x → 40-70점
        """

        # Step 1: 배당 데이터 수집
        dividend_query = """
        WITH latest_dividend AS (
            SELECT
                thstrm as current_dividend,
                frmtrm as prev_dividend,
                lwfr as prev2_dividend,
                stlm_dt
            FROM kr_dividends
            WHERE symbol = $1
                AND se LIKE '%배당%'
                AND stock_knd LIKE '%보통주%'
                AND stlm_dt <= COALESCE($2::date, CURRENT_DATE)
            ORDER BY stlm_dt DESC
            LIMIT 1
        )
        SELECT * FROM latest_dividend
        """

        div_result = await self.execute_query(dividend_query, self.symbol, self.analysis_date)

        if not div_result or div_result[0]['current_dividend'] is None:
            return None  # 배당 데이터 없음

        current_dividend = float(div_result[0]['current_dividend']) if div_result[0]['current_dividend'] else 0
        prev_dividend = float(div_result[0]['prev_dividend']) if div_result[0]['prev_dividend'] else 0
        prev2_dividend = float(div_result[0]['prev2_dividend']) if div_result[0]['prev2_dividend'] else 0

        if current_dividend <= 0:
            return None  # 무배당

        # Step 2: 주가 및 배당 수익률
        price_query = """
        SELECT close
        FROM kr_intraday_total
        WHERE symbol = $1
            AND ($2::date IS NULL OR date = $2)
        ORDER BY date DESC
        LIMIT 1
        """

        price_result = await self.execute_query(price_query, self.symbol, self.analysis_date)

        if not price_result or not price_result[0]['close']:
            return None

        price = float(price_result[0]['close'])
        dividend_yield = (current_dividend / price) * 100 if price > 0 else 0

        # Step 3: Payout Ratio (배당성향)
        payout_query = """
        WITH latest_financials AS (
            SELECT
                ni.thstrm_amount as net_income,
                eps.close as eps,
                eps.listed_shares
            FROM kr_financial_position ni
            LEFT JOIN kr_intraday_total eps ON ni.symbol = eps.symbol
            WHERE ni.symbol = $1
                AND ni.sj_div = 'IS'
                AND ni.account_nm IN ('당기순이익(손실)', '당기순이익')
                AND ni.rcept_dt <= COALESCE($2::date, CURRENT_DATE)
                AND ($2::date IS NULL OR eps.date = $2)
            ORDER BY ni.bsns_year DESC, ni.rcept_dt DESC, eps.date DESC
            LIMIT 1
        )
        SELECT net_income, listed_shares
        FROM latest_financials
        """

        payout_result = await self.execute_query(payout_query, self.symbol, self.analysis_date)

        payout_ratio = 999  # 기본값: 알 수 없음 (보수적)
        if payout_result and payout_result[0]['net_income'] and payout_result[0]['listed_shares']:
            net_income = float(payout_result[0]['net_income'])
            listed_shares = float(payout_result[0]['listed_shares'])
            if net_income > 0 and listed_shares > 0:
                total_dividend = current_dividend * listed_shares
                payout_ratio = (total_dividend / net_income) * 100

        # Step 4: FCF Coverage (잉여현금흐름 커버리지)
        fcf_query = """
        WITH latest_cf AS (
            SELECT
                ocf.thstrm_amount as operating_cash_flow,
                inv.thstrm_amount as capex
            FROM kr_financial_position ocf
            LEFT JOIN kr_financial_position inv
                ON ocf.symbol = inv.symbol
                AND ocf.bsns_year = inv.bsns_year
                AND ocf.rcept_dt = inv.rcept_dt
                AND inv.sj_div = 'CF'
                AND inv.account_nm LIKE '%투자활동%현금흐름%'
            WHERE ocf.symbol = $1
                AND ocf.sj_div = 'CF'
                AND ocf.account_nm LIKE '%영업활동%현금흐름%'
                AND ocf.rcept_dt <= COALESCE($2::date, CURRENT_DATE)
            ORDER BY ocf.bsns_year DESC, ocf.rcept_dt DESC
            LIMIT 1
        )
        SELECT operating_cash_flow, capex
        FROM latest_cf
        """

        fcf_result = await self.execute_query(fcf_query, self.symbol, self.analysis_date)

        fcf_coverage = 0  # 기본값: 커버 안됨 (보수적)
        if fcf_result and fcf_result[0]['operating_cash_flow']:
            ocf = float(fcf_result[0]['operating_cash_flow'])
            capex = float(fcf_result[0]['capex']) if fcf_result[0]['capex'] else 0
            fcf = ocf - abs(capex)  # FCF = OCF - CapEx

            # 총 배당금 계산
            if payout_result and payout_result[0]['listed_shares']:
                listed_shares = float(payout_result[0]['listed_shares'])
                total_dividend = current_dividend * listed_shares
                if total_dividend > 0:
                    fcf_coverage = fcf / total_dividend

        # Step 5: 배당 성장률 (3년 CAGR)
        dividend_growth = 0
        if prev2_dividend > 0:
            dividend_growth = (current_dividend / prev2_dividend) ** (1/3) - 1
        elif prev_dividend > 0:
            dividend_growth = (current_dividend / prev_dividend) - 1

        # Step 6: 점수 계산
        score = 0

        # === 위험 신호 탐지 ===

        # 위험 1: 과도한 배당 (Payout Ratio > 80%)
        if payout_ratio > 80:
            logger.info(f"V4 WARNING: High payout ratio {payout_ratio:.1f}%")
            return 10  # 지속 불가능

        # 위험 2: FCF < 배당 (빚내서 배당)
        if fcf_coverage < 1.0:
            logger.info(f"V4 WARNING: FCF coverage {fcf_coverage:.2f}x < 1.0")
            return 15  # 위험

        # 위험 3: 배당 감소 추세
        if dividend_growth < -0.05:
            logger.info(f"V4 WARNING: Dividend declining {dividend_growth*100:.1f}%")
            return 20  # 배당 컷 위험

        # === 긍정 신호 평가 ===

        # 우수: 배당 증가 + 여유 있음
        if (dividend_growth > 0.05 and
            payout_ratio < 50 and
            fcf_coverage > 1.5):

            # 배당 수익률 기반 점수
            if dividend_yield > 4.0:      # 4% 이상
                score = 90
            elif dividend_yield > 3.0:    # 3% 이상
                score = 75
            else:
                score = 60

            logger.info(f"V4: Excellent dividend (Yield: {dividend_yield:.2f}%, Growth: {dividend_growth*100:.1f}%)")

        # 양호: 안정적
        elif (payout_ratio < 60 and
              fcf_coverage > 1.2 and
              dividend_growth > -0.02):

            if dividend_yield > 3.0:
                score = 70
            elif dividend_yield > 2.0:
                score = 55
            else:
                score = 40

            logger.info(f"V4: Stable dividend (Yield: {dividend_yield:.2f}%, Payout: {payout_ratio:.1f}%)")

        # 보통
        else:
            if dividend_yield > 3.0:
                score = 50
            else:
                score = 30

            logger.info(f"V4: Moderate dividend (Yield: {dividend_yield:.2f}%)")

        # Apply extreme risk penalty
        if score >= 40:
            volatility = await self._get_volatility_for_risk_adjustment()
            beta = await self._get_beta_for_risk_adjustment()
            var_95 = await self._get_var_for_risk_adjustment()

            vol_penalty = await self._calculate_volatility_penalty(volatility)
            beta_penalty = await self._calculate_beta_penalty(beta)
            var_penalty = await self._calculate_var_penalty(var_95)

            total_penalty = vol_penalty + beta_penalty + var_penalty

            if total_penalty < 0:
                logger.info(f"V4: Extreme risk penalty applied: {total_penalty} (Vol: {vol_penalty}, Beta: {beta_penalty}, VaR: {var_penalty})")
                score = score + total_penalty

        return max(0, score)

    # ========================================================================
    # V5. PSR (Price to Sales Ratio) Strategy
    # ========================================================================

    async def calculate_v5(self):
        """
        V5. PSR Strategy (DEPRECATED)

        IC: -0.0401 (Negative correlation with returns)

        Reason for deprecation:
        - PSR showed consistent negative IC across all market conditions
        - Particularly poor performance in KOSDAQ GLOBAL segment
        - Sales-based valuation not reliable in Korean market

        Status: Excluded from Value factor calculation
        Total active strategies: 15 (V1~V16 excluding V5)

        Original logic (commented out):
        PSR = Market Cap / Annual Sales
        Score = 100 - MIN(100, MAX(0, (PSR - 0.5) × 50))
        """
        # Strategy deprecated - return None to exclude from weighted average
        return None

    # ========================================================================
    # V6. 52-Week Relative Price Strategy
    # ========================================================================

    async def calculate_v6(self):
        """
        V6. 52-Week Relative Price Strategy (Enhanced with fallback)
        Description: Determine undervaluation based on position vs 52-week range
        52-Week High = MAX(high in last 365 days)
        52-Week Low = MIN(low in last 365 days)
        Current Price = close
        Relative Position = (Current - 52W Low) / (52W High - 52W Low) × 100
        Score = 100 - MIN(100, MAX(0, Relative Position))
        Interpretation: Closer to 52-week low gets higher score
        """
        query = """
        WITH price_range AS (
            SELECT
                MAX(high) as week52_high,
                MIN(low) as week52_low
            FROM kr_intraday_total
            WHERE symbol = $1
                AND date >= COALESCE($3::date, CURRENT_DATE) - INTERVAL '365 days'
                AND date <= COALESCE($3::date, CURRENT_DATE)
        ),
        current_price AS (
            SELECT close
            FROM kr_intraday_total
            WHERE symbol = $2
                AND ($3::date IS NULL OR date = $3)
            ORDER BY date DESC
            LIMIT 1
        )
        SELECT
            pr.week52_high,
            pr.week52_low,
            cp.close as current_price
        FROM price_range pr, current_price cp
        """

        result = await self.execute_query(query, self.symbol, self.symbol, self.analysis_date)

        week52_high = None
        week52_low = None
        current_price = None

        if result and result[0]['week52_high'] and result[0]['week52_low'] and result[0]['current_price']:
            week52_high = float(result[0]['week52_high'])
            week52_low = float(result[0]['week52_low'])
            current_price = float(result[0]['current_price'])

        # Fallback: Use shorter period (180 days) or available data
        if not week52_high or not week52_low or not current_price:
            fallback_query = """
            WITH price_range AS (
                SELECT
                    MAX(high) as max_high,
                    MIN(low) as min_low,
                    COUNT(*) as days_count
                FROM kr_intraday_total
                WHERE symbol = $1
                    AND date >= COALESCE($3::date, CURRENT_DATE) - INTERVAL '180 days'
                    AND date <= COALESCE($3::date, CURRENT_DATE)
            ),
            current_price AS (
                SELECT close
                FROM kr_intraday_total
                WHERE symbol = $2
                    AND ($3::date IS NULL OR date = $3)
                ORDER BY date DESC
                LIMIT 1
            )
            SELECT
                pr.max_high as week52_high,
                pr.min_low as week52_low,
                cp.close as current_price,
                pr.days_count
            FROM price_range pr, current_price cp
            """

            fallback_result = await self.execute_query(fallback_query, self.symbol, self.symbol, self.analysis_date)

            if fallback_result and fallback_result[0]['week52_high'] and fallback_result[0]['days_count'] and fallback_result[0]['days_count'] >= 30:
                week52_high = float(fallback_result[0]['week52_high'])
                week52_low = float(fallback_result[0]['week52_low'])
                current_price = float(fallback_result[0]['current_price'])

        if not week52_high or not week52_low or not current_price or week52_high == week52_low:
            return None

        relative_position = ((current_price - week52_low) / (week52_high - week52_low)) * 100

        score = 100 - min(100, max(0, relative_position))

        return score

    # ========================================================================
    # V7. Audit Opinion Discount Value Strategy
    # ========================================================================

    async def calculate_v7(self):
        """
        V7. Audit Opinion Discount Value Strategy
        Description: Undervalued stocks with high financial reliability
        Audit Score = CASE
            WHEN '적정' THEN 100
            WHEN '한정' THEN 50
            ELSE 0
        PBR Discount = 100 - MIN(100, PBR × 50)
        Final Score = (Audit Score × 0.3) + (PBR Discount × 0.7)
        Interpretation: Clean opinion + PBR <= 1
        """
        # Get Audit Opinion
        audit_query = """
        SELECT adt_opinion
        FROM kr_audit
        WHERE symbol = $1
            AND adt_opinion IS NOT NULL
        ORDER BY bsns_year DESC
        LIMIT 1
        """

        audit_result = await self.execute_query(audit_query, self.symbol)

        if not audit_result or not audit_result[0]['adt_opinion']:
            return None

        audit_opinion = audit_result[0]['adt_opinion']

        # Note: kr_audit data has encoding issues. Using pattern matching on corrupted text
        # '�����ǰ�' represents clean opinion (적정) - 97.6% of records
        # '����' likely represents qualified opinion (한정)
        if '�����ǰ�' in audit_opinion:
            audit_score = 100
        elif '����' in audit_opinion:
            audit_score = 50
        else:
            audit_score = 0

        # Get PBR
        pbr_query = """
        SELECT pbr
        FROM kr_intraday_detail
        WHERE symbol = $1
        """

        pbr_result = await self.execute_query(pbr_query, self.symbol)

        if not pbr_result or pbr_result[0]['pbr'] is None:
            return None

        pbr = float(pbr_result[0]['pbr'])

        pbr_discount = 100 - min(100, pbr * 50)

        base_score = (audit_score * 0.3) + (pbr_discount * 0.7)

        # Sector-Health-Weighted 평가
        sector_health = await self.calculate_sector_health_score()

        # 섹터 건강도 기반 최종 점수
        if sector_health >= 70:
            # 건강한 섹터: 감사의견 신뢰도 높음
            if audit_score == 100:
                final_score = base_score * 1.1  # 보너스 10%
            else:
                final_score = base_score
            logger.info(f"V7: Healthy sector (Audit: {audit_score}, PBR: {pbr:.2f})")

        elif sector_health >= 40:
            # 중립 섹터: 기본 점수 유지
            final_score = base_score
            logger.info(f"V7: Neutral sector (Audit: {audit_score}, Base score: {base_score:.1f})")

        else:
            # 망한 섹터: 감사의견 신뢰도 낮음 (섹터가 망하면 재무 신뢰도도 의미 없음)
            if audit_score == 100 and pbr < 0.5:  # 적정 의견 + 초저PBR
                final_score = base_score * 0.7  # 페널티 30%
                logger.info(f"V7: Weak sector but clean audit + ultra-low PBR ({pbr:.2f}), Penalty: 30%")
            elif audit_score == 100:
                final_score = base_score * 0.6  # 페널티 40%
                logger.info(f"V7: Weak sector penalty (Health: {sector_health}), Penalty: 40%")
            else:
                final_score = base_score * 0.4  # 페널티 60% (한정/부적정 의견)
                logger.info(f"V7: Weak sector + poor audit, Penalty: 60%")

        return min(100, final_score)

    # ========================================================================
    # V8. Treasury Stock Cancellation Value Strategy
    # ========================================================================

    async def calculate_v8(self):
        """
        V8. Treasury Stock Cancellation Value Strategy (Enhanced with fallback)
        Description: Companies enhancing shareholder value through buybacks
        Treasury Stock Change = SUM(acquisitions - disposals)
        Treasury Stock Ratio = (Change / Listed Shares) × 100
        PER Discount = 100 - MIN(100, PER / 10 × 50)
        Score = (Treasury Stock Ratio × 20) + (PER Discount × 0.5)
        Interpretation: 5% buyback + low PER
        """
        # Get Listed Shares
        shares_query = """
        SELECT listed_shares
        FROM kr_intraday_total
        WHERE symbol = $1
        ORDER BY date DESC
        LIMIT 1
        """

        shares_result = await self.execute_query(shares_query, self.symbol)

        if not shares_result or not shares_result[0]['listed_shares']:
            return None

        listed_shares = float(shares_result[0]['listed_shares'])

        # Get Treasury Stock Change
        treasury_query = """
        SELECT
            SUM(COALESCE(change_qy_acqs, 0) - COALESCE(change_qy_dsps, 0)) as stock_change
        FROM kr_stockacquisitiondisposal
        WHERE symbol = $1
        """

        treasury_result = await self.execute_query(treasury_query, self.symbol)

        stock_change = float(treasury_result[0]['stock_change']) if treasury_result and treasury_result[0]['stock_change'] else 0

        treasury_ratio = (stock_change / listed_shares) * 100

        # Get PER
        per_query = """
        SELECT per
        FROM kr_intraday_detail
        WHERE symbol = $1
        """

        per_result = await self.execute_query(per_query, self.symbol)
        per = None

        if per_result and per_result[0]['per'] is not None and per_result[0]['per'] > 0:
            per = float(per_result[0]['per'])

        # Fallback for PER: Calculate from market cap and net income
        if per is None:
            per_fallback_query = """
            SELECT kit.market_cap,
                   fp.thstrm_amount as net_income
            FROM kr_intraday_total kit
            LEFT JOIN (
                SELECT symbol, thstrm_amount
                FROM kr_financial_position
                WHERE symbol = $1
                    AND sj_div = 'IS'
                    AND account_nm IN ('당기순이익(손실)', '당기순이익')
                    AND thstrm_amount > 0
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
                    thstrm_amount DESC
                LIMIT 1
            ) fp ON kit.symbol = fp.symbol
            WHERE kit.symbol = $2
                AND ($3::date IS NULL OR kit.date = $3)
            ORDER BY kit.date DESC
            LIMIT 1
            """

            per_fb_result = await self.execute_query(per_fallback_query, self.symbol, self.symbol, self.analysis_date)

            if per_fb_result and per_fb_result[0]['market_cap'] and per_fb_result[0]['net_income']:
                mktcap = float(per_fb_result[0]['market_cap'])
                net_income = float(per_fb_result[0]['net_income'])
                if net_income > 0 and mktcap > 0:
                    per = mktcap / net_income

        if per is None or per <= 0:
            return None

        per_discount = 100 - min(100, per / 10 * 50)

        # 자사주 순매도는 0점, 매입은 비례 점수
        # 가중치 조정: 자사주 70점 + PER 할인 30점 = 최대 100점
        if treasury_ratio <= 0:
            treasury_score = 0
        else:
            # 5% 자사주 매입 = 70점 만점
            treasury_score = min(70, treasury_ratio * 14)

        # PER 할인 기여도: 최대 30점
        per_contribution = min(30, per_discount * 0.3)

        score = treasury_score + per_contribution

        return score

    # ========================================================================
    # V9. Foreign Ownership Contrarian Strategy
    # ========================================================================

    async def calculate_v9(self):
        """
        V9. Foreign Ownership Contrarian Strategy
        Description: Quality undervalued stocks with low foreign ownership
        Foreign Ownership = foreign_rate
        Foreign Contrarian = 100 - MIN(100, Foreign Ownership × 2)
        EPS Growth = (Current EPS - Previous EPS) / Previous EPS × 100
        Growth Score = MIN(50, MAX(0, EPS Growth))
        Score = (Foreign Contrarian × 0.6) + (Growth Score × 0.4)
        Interpretation: Low foreign ownership but growing earnings
        """
        # Get Foreign Ownership
        foreign_query = """
        SELECT foreign_rate
        FROM kr_foreign_ownership
        WHERE symbol = $1
        ORDER BY date DESC
        LIMIT 1
        """

        foreign_result = await self.execute_query(foreign_query, self.symbol)

        if not foreign_result or foreign_result[0]['foreign_rate'] is None:
            return None

        foreign_rate = float(foreign_result[0]['foreign_rate'])

        foreign_contrarian = 100 - min(100, foreign_rate * 2)

        # Get EPS Growth - kr_intraday_detail has only one record per symbol, so get from kr_intraday_total
        eps_query = """
        SELECT eps
        FROM kr_intraday_total
        WHERE symbol = $1
            AND eps IS NOT NULL
        ORDER BY date DESC
        LIMIT 2
        """

        eps_result = await self.execute_query(eps_query, self.symbol)

        if not eps_result or len(eps_result) < 2 or not eps_result[0]['eps'] or not eps_result[1]['eps']:
            # If no EPS growth data, just use foreign contrarian score
            return foreign_contrarian * 0.6

        current_eps = float(eps_result[0]['eps'])
        previous_eps = float(eps_result[1]['eps'])

        if previous_eps == 0:
            return foreign_contrarian * 0.6

        eps_growth = ((current_eps - previous_eps) / previous_eps) * 100

        growth_score = min(50, max(0, eps_growth))

        score = (foreign_contrarian * 0.6) + (growth_score * 0.4)

        return score

    # ========================================================================
    # V10. PCR (Price to Cash Flow Ratio) Strategy
    # ========================================================================

    async def calculate_v10(self):
        """
        V10. Net Cash Change Ratio Strategy
        Description: Undervaluation vs net cash change per share
        Primary: Net Cash Change from CF statement (현금및현금성자산의순증가)
        Fallback: BS cash change (현금및현금성자산 증감)
        Cash Change per Share = Net Cash Change / Listed Shares
        Cash Change Ratio = Close Price / Cash Change per Share
        Score = 100 - MIN(100, MAX(0, (Ratio - 5) × 10))
        Interpretation: Ratio <= 5 gets 100 points, >= 15 gets 0 points
        """
        net_cash_change = None

        # Primary: Get Net Cash Change from CF statement
        cf_query = """
        SELECT thstrm_amount
        FROM kr_financial_position
        WHERE symbol = $1
            AND sj_div = 'CF'
            AND (
                account_nm LIKE '%현금%현금성자산%순증%'
                OR account_nm LIKE '%현금성자산%순증%'
                OR account_nm LIKE '%현금%증감%'
                OR account_nm = '현금및현금성자산의순증가(감소)'
                OR account_nm = '현금및현금성자산의 증가(감소)'
            )
            AND thstrm_amount IS NOT NULL
            AND rcept_dt <= COALESCE($2::date, CURRENT_DATE)
        ORDER BY bsns_year DESC, rcept_dt DESC, ABS(thstrm_amount) DESC
        LIMIT 1
        """

        cf_result = await self.execute_query(cf_query, self.symbol, self.analysis_date)

        if cf_result and cf_result[0]['thstrm_amount']:
            net_cash_change = float(cf_result[0]['thstrm_amount'])

        # Fallback: Calculate from BS cash change
        if net_cash_change is None:
            bs_query = """
            WITH latest_report AS (
                SELECT bsns_year, rcept_dt
                FROM kr_financial_position
                WHERE symbol = $1
                    AND sj_div = 'BS'
                    AND rcept_dt <= COALESCE($2::date, CURRENT_DATE)
                ORDER BY bsns_year DESC, rcept_dt DESC
                LIMIT 1
            )
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
                        '단기금융상품'
                    )
                    OR fp.account_nm LIKE '%현금%자산%'
                )
            """

            bs_result = await self.execute_query(bs_query, self.symbol, self.analysis_date)

            if bs_result and bs_result[0]['cash_delta']:
                net_cash_change = float(bs_result[0]['cash_delta'])

        if net_cash_change is None:
            return None

        # Only proceed if cash increased (positive change)
        if net_cash_change <= 0:
            return 0  # Return 0 points for negative cash change

        # Get Listed Shares and Close Price
        price_query = """
        SELECT listed_shares, close
        FROM kr_intraday_total
        WHERE symbol = $1
        ORDER BY date DESC
        LIMIT 1
        """

        price_result = await self.execute_query(price_query, self.symbol)

        if not price_result or not price_result[0]['listed_shares'] or not price_result[0]['close']:
            return None

        listed_shares = float(price_result[0]['listed_shares'])
        close_price = float(price_result[0]['close'])

        cash_change_per_share = net_cash_change / listed_shares

        cash_change_ratio = close_price / cash_change_per_share

        score = 100 - min(100, max(0, (cash_change_ratio - 5) * 10))

        return score

    # ========================================================================
    # V11. Largest Shareholder Stake Value Strategy
    # ========================================================================

    async def calculate_v11(self):
        """
        V11. Largest Shareholder Stake Value Strategy (Enhanced with fallback)
        Description: Undervalued stocks with high controlling shareholder stake (aligned interests)
        Largest Shareholder Stake = trmend_posesn_stock_qota_rt
        Stake Score = MIN(50, Largest Shareholder Stake)
        PER Score = 100 - MIN(100, PER / 10 × 50)
        PBR Score = 100 - MIN(100, PBR × 50)
        Score = (Stake Score × 0.3) + (PER Score × 0.35) + (PBR Score × 0.35)
        Interpretation: 50% stake + low PER + low PBR
        """
        # Get Largest Shareholder Stake
        shareholder_query = """
        SELECT trmend_posesn_stock_qota_rt
        FROM kr_largest_shareholder
        WHERE symbol = $1
        ORDER BY stlm_dt DESC
        LIMIT 1
        """

        shareholder_result = await self.execute_query(shareholder_query, self.symbol)

        stake = None
        if shareholder_result and shareholder_result[0]['trmend_posesn_stock_qota_rt'] is not None:
            stake = float(shareholder_result[0]['trmend_posesn_stock_qota_rt'])

        # Fallback: Use average stake if no data (neutral score)
        if stake is None:
            stake = 25.0  # Assume average 25% stake

        stake_score = min(50, stake)

        # Get PER and PBR
        valuation_query = """
        SELECT per, pbr
        FROM kr_intraday_detail
        WHERE symbol = $1
        """

        valuation_result = await self.execute_query(valuation_query, self.symbol)

        per = None
        pbr = None

        if valuation_result:
            if valuation_result[0]['per'] is not None and valuation_result[0]['per'] > 0:
                per = float(valuation_result[0]['per'])
            if valuation_result[0]['pbr'] is not None and valuation_result[0]['pbr'] > 0:
                pbr = float(valuation_result[0]['pbr'])

        # Fallback for PER
        if per is None:
            per_fallback_query = """
            SELECT kit.market_cap,
                   fp.thstrm_amount as net_income
            FROM kr_intraday_total kit
            LEFT JOIN (
                SELECT symbol, thstrm_amount
                FROM kr_financial_position
                WHERE symbol = $1
                    AND sj_div = 'IS'
                    AND account_nm IN ('당기순이익(손실)', '당기순이익')
                    AND thstrm_amount > 0
                ORDER BY
                    bsns_year DESC,
                    CASE
                        WHEN report_code = '11011' THEN 1
                        WHEN report_code = '11012' THEN 2
                        WHEN report_code = '11013' THEN 3
                        WHEN report_code = '11014' THEN 4
                        ELSE 5
                    END,
                    thstrm_amount DESC
                LIMIT 1
            ) fp ON kit.symbol = fp.symbol
            WHERE kit.symbol = $2
                AND ($3::date IS NULL OR kit.date = $3)
            ORDER BY kit.date DESC
            LIMIT 1
            """

            per_fb_result = await self.execute_query(per_fallback_query, self.symbol, self.symbol, self.analysis_date)

            if per_fb_result and per_fb_result[0]['market_cap'] and per_fb_result[0]['net_income']:
                mktcap = float(per_fb_result[0]['market_cap'])
                net_income = float(per_fb_result[0]['net_income'])
                if net_income > 0 and mktcap > 0:
                    per = mktcap / net_income

        if per is None or per <= 0 or pbr is None or pbr <= 0:
            return None

        per_score = 100 - min(100, per / 10 * 50)
        pbr_score = 100 - min(100, pbr * 50)

        score = (stake_score * 0.3) + (per_score * 0.35) + (pbr_score * 0.35)

        return score

    # ========================================================================
    # V12. Cash Cushion Strategy (NEW)
    # ========================================================================

    async def calculate_v12(self):
        """
        V12. Cash Cushion Strategy
        Description: Financial stability through high cash reserves vs market cap
        Primary: Ending cash balance from CF statement (기말현금및현금성자산)
        Fallback: Cash & equivalents from BS (현금및현금성자산)
        Cash Holding Ratio = (Cash & Cash Equivalents / Market Cap) × 100
        Score = MIN(100, Cash Holding Ratio × 10)
        Interpretation: 10%+ cash ratio gets 100 points
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

        if not mc_result or mc_result[0]['market_cap'] is None:
            return None

        market_cap = float(mc_result[0]['market_cap'])

        cash_balance = None

        # Primary: Get ending cash balance from CF statement
        cf_query = """
        SELECT thstrm_amount
        FROM kr_financial_position
        WHERE symbol = $1
            AND sj_div = 'CF'
            AND (
                account_nm LIKE '%기말%현금%'
                OR account_nm LIKE '%기말%자산%'
                OR account_nm = '기말현금및현금성자산'
                OR account_nm = '기말의 현금및현금성자산'
            )
            AND thstrm_amount > 0
            AND rcept_dt <= COALESCE($2::date, CURRENT_DATE)
        ORDER BY bsns_year DESC, rcept_dt DESC, thstrm_amount DESC
        LIMIT 1
        """

        cf_result = await self.execute_query(cf_query, self.symbol, self.analysis_date)

        if cf_result and cf_result[0]['thstrm_amount']:
            cash_balance = float(cf_result[0]['thstrm_amount'])

        # Fallback: Get cash from BS
        if cash_balance is None:
            bs_query = """
            SELECT thstrm_amount
            FROM kr_financial_position
            WHERE symbol = $1
                AND sj_div = 'BS'
                AND (
                    account_nm IN (
                        '현금및현금성자산',
                        '현금',
                        '현금및예치금'
                    )
                    OR account_nm LIKE '%현금%자산%'
                )
                AND thstrm_amount > 0
                AND rcept_dt <= COALESCE($2::date, CURRENT_DATE)
            ORDER BY bsns_year DESC, rcept_dt DESC, thstrm_amount DESC
            LIMIT 1
            """

            bs_result = await self.execute_query(bs_query, self.symbol, self.analysis_date)

            if bs_result and bs_result[0]['thstrm_amount']:
                cash_balance = float(bs_result[0]['thstrm_amount'])

        if cash_balance is None or cash_balance <= 0:
            return 0

        # Calculate cash holding ratio
        cash_ratio = (cash_balance / market_cap) * 100

        # Score: 10% cash ratio = 100 points
        score = min(100, cash_ratio * 10)

        return score


    # ========================================================================
    # V13. Magic Formula Strategy (NEW)
    # ========================================================================

    async def calculate_v13(self):
        """
        V13. Magic Formula Strategy (Enhanced with fallback)
        Description: Joel Greenblatt's proven strategy combining quality and value
        Formula: High ROE (Quality) + High Earnings Yield (Value)
        ROE = (Net Income / Total Equity) × 100
        Earnings Yield = (Net Income / Market Cap) × 100
        ROE Score = MIN(50, ROE × 2.5)
        EY Score = MIN(50, Earnings Yield × 5)
        Final Score = ROE Score + EY Score
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

        if not mc_result or mc_result[0]['market_cap'] is None:
            return None

        market_cap = float(mc_result[0]['market_cap'])

        # Get Net Income and Total Equity
        magic_query = """
        SELECT
            fp1.thstrm_amount as net_income,
            fp2.thstrm_amount as total_equity
        FROM kr_financial_position fp1
        JOIN kr_financial_position fp2
            ON fp1.symbol = fp2.symbol
            AND fp1.bsns_year = fp2.bsns_year
            AND fp1.rcept_dt = fp2.rcept_dt
        WHERE fp1.symbol = $1
            AND fp1.account_nm IN ('당기순이익(손실)', '당기순이익')
            AND fp2.account_nm IN ('기말자본', '자본총계', '당기말자본')
            AND fp1.sj_div = 'IS'
            AND fp2.sj_div IN ('SCE', 'BS')
            AND fp1.rcept_dt <= COALESCE($2::date, CURRENT_DATE)
        ORDER BY fp1.bsns_year DESC, fp1.rcept_dt DESC
        LIMIT 1
        """

        result = await self.execute_query(magic_query, self.symbol, self.analysis_date)

        net_income = None
        total_equity = None

        if result and result[0]['net_income'] and result[0]['total_equity']:
            net_income = float(result[0]['net_income'])
            total_equity = float(result[0]['total_equity'])

        # Fallback: Estimate from position-based selection
        if not net_income or not total_equity:
            fallback_query = """
            WITH latest_report AS (
                SELECT bsns_year, rcept_dt
                FROM kr_financial_position
                WHERE symbol = $1
                    AND rcept_dt <= COALESCE($2::date, CURRENT_DATE)
                ORDER BY bsns_year DESC, rcept_dt DESC
                LIMIT 1
            ),
            cis_data AS (
                SELECT thstrm_amount
                FROM kr_financial_position, latest_report
                WHERE symbol = $1
                    AND sj_div = 'IS'
                    AND account_nm IN ('당기순이익(손실)', '당기순이익')
                    AND thstrm_amount > 0
                    AND kr_financial_position.bsns_year = latest_report.bsns_year
                    AND kr_financial_position.rcept_dt = latest_report.rcept_dt
                ORDER BY
                    CASE
                        WHEN report_code = '11011' THEN 1
                        WHEN report_code = '11012' THEN 2
                        WHEN report_code = '11013' THEN 3
                        WHEN report_code = '11014' THEN 4
                        ELSE 5
                    END,
                    thstrm_amount DESC
                LIMIT 1
            ),
            sce_data AS (
                SELECT thstrm_amount
                FROM kr_financial_position, latest_report
                WHERE symbol = $1
                    AND sj_div = 'SCE'
                    AND account_nm IN ('기말자본', '자본총계', '당기말자본')
                    AND thstrm_amount > 0
                    AND kr_financial_position.bsns_year = latest_report.bsns_year
                    AND kr_financial_position.rcept_dt = latest_report.rcept_dt
                ORDER BY thstrm_amount DESC
                LIMIT 1
            )
            SELECT
                (SELECT thstrm_amount FROM cis_data) as net_income_est,
                (SELECT thstrm_amount FROM sce_data) as equity_est
            """

            fb_result = await self.execute_query(fallback_query, self.symbol, self.analysis_date)

            if fb_result:
                if not net_income and fb_result[0]['net_income_est']:
                    net_income = float(fb_result[0]['net_income_est'])
                if not total_equity and fb_result[0]['equity_est']:
                    total_equity = float(fb_result[0]['equity_est'])

        if not net_income or not total_equity or total_equity <= 0:
            return None

        # Calculate ROE and Earnings Yield
        roe = (net_income / total_equity) * 100
        earnings_yield = (net_income / market_cap) * 100

        # Check for loss-making companies
        if net_income <= 0:
            return 0  # Loss-making companies get 0 score

        # Score components (with max protection for extreme negative values)
        roe_score = max(0, min(50, roe * 2.5))
        ey_score = max(0, min(50, earnings_yield * 5))

        base_score = roe_score + ey_score

        # Sector-Health-Weighted 평가
        sector_health = await self.calculate_sector_health_score()

        # 섹터 중앙값 ROE 계산
        sector_roe_query = """
        WITH sector_stocks AS (
            SELECT sd.symbol, sd.theme
            FROM kr_stock_detail sd
            WHERE sd.symbol = $1
        ),
        peer_roe AS (
            SELECT
                (fp1.thstrm_amount / fp2.thstrm_amount) * 100 as roe
            FROM kr_financial_position fp1
            JOIN kr_financial_position fp2
                ON fp1.symbol = fp2.symbol
                AND fp1.bsns_year = fp2.bsns_year
                AND fp1.rcept_dt = fp2.rcept_dt
            JOIN kr_stock_detail sd ON fp1.symbol = sd.symbol
            WHERE sd.theme = (SELECT theme FROM sector_stocks)
                AND fp1.sj_div = 'IS'
                AND fp2.sj_div IN ('SCE', 'BS')
                AND fp1.account_nm IN ('당기순이익(손실)', '당기순이익')
                AND fp2.account_nm IN ('기말자본', '자본총계', '당기말자본')
                AND fp1.rcept_dt <= COALESCE($2::date, CURRENT_DATE)
                AND fp1.thstrm_amount > 0
                AND fp2.thstrm_amount > 0
        )
        SELECT PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY roe) as median_roe
        FROM peer_roe
        WHERE roe > 0 AND roe < 100
        """

        sector_roe_result = await self.execute_query(sector_roe_query, self.symbol, self.analysis_date)

        sector_median_roe = 10.0  # 기본값
        if sector_roe_result and sector_roe_result[0]['median_roe']:
            sector_median_roe = float(sector_roe_result[0]['median_roe'])

        # ROE 상대 평가
        if roe > 0 and sector_median_roe > 0:
            roe_premium = (roe - sector_median_roe) / sector_median_roe
        else:
            roe_premium = 0

        # 섹터 건강도 기반 최종 점수
        if sector_health >= 70:
            # 건강한 섹터: 상대평가 가중
            if roe_premium > 0.50:  # 섹터 대비 50% 이상 높음
                final_score = base_score * 1.2
            elif roe_premium > 0:
                final_score = base_score * 1.1
            else:
                final_score = base_score * 0.95
            logger.info(f"V13: Healthy sector (ROE: {roe:.1f}%, Sector median: {sector_median_roe:.1f}%, Premium: {roe_premium*100:.1f}%)")

        elif sector_health >= 40:
            # 중립 섹터: 기본 점수 유지
            final_score = base_score
            logger.info(f"V13: Neutral sector (ROE: {roe:.1f}%, EY: {earnings_yield:.1f}%, Base: {base_score:.1f})")

        else:
            # 망한 섹터: 페널티 적용
            if roe > 20 and earnings_yield > 8:  # 절대적으로 우수한 지표
                final_score = base_score * 0.8  # 페널티 20%
                logger.info(f"V13: Weak sector but excellent metrics (ROE: {roe:.1f}%, EY: {earnings_yield:.1f}%), Penalty: 20%")
            else:
                final_score = base_score * 0.6  # 페널티 40%
                logger.info(f"V13: Weak sector penalty (Health: {sector_health}), Penalty: 40%")

        return min(100, final_score)

    # ========================================================================
    # V14. Dividend Growth Strategy (NEW)
    # ========================================================================

    async def calculate_v14(self):
        """
        V14. Dividend Growth Strategy
        Description: Companies consistently increasing dividends
        Dividend Growth Rate = (This Year Dividend - Last Year Dividend) / Last Year × 100
        Growth Consistency = COUNT(years with dividend increase) / total years
        Score = MIN(50, Growth Rate) + (Consistency × 50)
        """
        # Get dividend history (last 4 years)
        div_query = """
        SELECT EXTRACT(YEAR FROM stlm_dt) as year,
               SUM(thstrm) as total_dividend
        FROM kr_dividends
        WHERE symbol = $1
            AND stlm_dt >= CURRENT_DATE - INTERVAL '4 years'
            AND thstrm > 0
        GROUP BY EXTRACT(YEAR FROM stlm_dt)
        ORDER BY year DESC
        """

        div_result = await self.execute_query(div_query, self.symbol)

        if not div_result or len(div_result) < 2:
            return None

        dividends = [(int(row['year']), float(row['total_dividend'])) for row in div_result]
        dividends.sort(key=lambda x: x[0])  # Sort by year ascending

        # Calculate growth rate (latest vs previous)
        if len(dividends) >= 2:
            latest_div = dividends[-1][1]
            previous_div = dividends[-2][1]

            if previous_div <= 0:
                return None

            growth_rate = ((latest_div - previous_div) / previous_div) * 100
        else:
            growth_rate = 0

        # Calculate consistency
        increases = 0
        for i in range(1, len(dividends)):
            if dividends[i][1] > dividends[i-1][1]:
                increases += 1

        consistency = increases / (len(dividends) - 1) if len(dividends) > 1 else 0

        # Calculate score
        growth_score = min(50, max(0, growth_rate))
        consistency_score = consistency * 50

        final_score = growth_score + consistency_score

        return final_score

    # ========================================================================
    # V15. Growth At Reasonable Price (GARP) Strategy (NEW)
    # ========================================================================

    async def calculate_v15(self):
        """
        V15. GARP Strategy (Enhanced with fallback)
        Description: Peter Lynch's strategy - growing companies at reasonable prices
        Sales Growth Rate = (Current Sales - Previous Sales) / Previous Sales × 100
        PEG Ratio = PER / Sales Growth Rate
        Score = 100 - MIN(100, MAX(0, PEG × 10))
        Condition: Sales Growth > 0, PER > 0
        """
        # Get PER
        per_query = """
        SELECT per
        FROM kr_intraday_detail
        WHERE symbol = $1
        """

        per_result = await self.execute_query(per_query, self.symbol)
        per = None

        if per_result and per_result[0]['per'] is not None and per_result[0]['per'] > 0:
            per = float(per_result[0]['per'])

        # Fallback for PER: Calculate from market cap and net income
        if per is None:
            per_fallback_query = """
            SELECT kit.market_cap,
                   fp.thstrm_amount as net_income
            FROM kr_intraday_total kit
            LEFT JOIN (
                SELECT symbol, thstrm_amount
                FROM kr_financial_position
                WHERE symbol = $1
                    AND sj_div = 'IS'
                    AND account_nm IN ('당기순이익(손실)', '당기순이익')
                    AND thstrm_amount > 0
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
                    thstrm_amount DESC
                LIMIT 1
            ) fp ON kit.symbol = fp.symbol
            WHERE kit.symbol = $2
                AND ($3::date IS NULL OR kit.date = $3)
            ORDER BY kit.date DESC
            LIMIT 1
            """

            per_fb_result = await self.execute_query(per_fallback_query, self.symbol, self.symbol, self.analysis_date)

            if per_fb_result and per_fb_result[0]['market_cap'] and per_fb_result[0]['net_income']:
                mktcap = float(per_fb_result[0]['market_cap'])
                net_income = float(per_fb_result[0]['net_income'])
                if net_income > 0 and mktcap > 0:
                    per = mktcap / net_income

        if per is None or per <= 0:
            return None

        # Get sales for last 2 years (improved query - get largest value per year)
        sales_query = """
        WITH sales_by_year AS (
            SELECT bsns_year,
                   MAX(thstrm_amount) as sales,
                   MIN(CASE
                       WHEN report_code = '11011' THEN 1
                       WHEN report_code = '11012' THEN 2
                       WHEN report_code = '11013' THEN 3
                       WHEN report_code = '11014' THEN 4
                       ELSE 5
                   END) as report_priority,
                   MAX(rcept_dt) as latest_rcept_dt
            FROM kr_financial_position
            WHERE symbol = $1
                AND sj_div = 'IS'
                AND account_nm = '매출액'
                AND thstrm_amount > 0
                AND rcept_dt <= COALESCE($2::date, CURRENT_DATE)
            GROUP BY bsns_year
            ORDER BY bsns_year DESC, report_priority
            LIMIT 2
        )
        SELECT bsns_year, sales
        FROM sales_by_year
        ORDER BY bsns_year DESC
        """

        sales_result = await self.execute_query(sales_query, self.symbol, self.analysis_date)

        if not sales_result or len(sales_result) < 2:
            return None

        current_sales = float(sales_result[0]['sales'])
        previous_sales = float(sales_result[1]['sales'])

        if previous_sales <= 0:
            return None

        # Calculate sales growth rate
        sales_growth = ((current_sales - previous_sales) / previous_sales) * 100

        # Relaxed condition: Allow low growth (>-10%) with penalty
        if sales_growth < -10:
            return None

        # For negative or very low growth, use minimum growth of 1% to avoid division issues
        if sales_growth <= 0:
            sales_growth = 1.0

        # Calculate PEG ratio
        peg = per / sales_growth

        # Score: Lower PEG is better
        score = 100 - min(100, max(0, peg * 10))

        return score

    # ========================================================================
    # V16. Small Cap Value Strategy
    # ========================================================================

    async def calculate_v16_original(self):
        """
        V16. Small Cap Value Strategy (ORIGINAL - For IC Comparison)

        Original IC: -0.0422 (Negative correlation)

        Original logic:
        - Market cap score (50%): Smaller is better
        - PER score (50%): Lower is better
        - No quality filters, no risk management

        This is the original version for IC comparison purposes.
        """
        # Get market cap and PER
        async with self.db_manager.connection_pool.acquire() as conn:
            result = await conn.fetchrow("""
                SELECT market_cap, per
                FROM kr_intraday_total
                WHERE symbol = $1
                    AND date = $2
                    AND market_cap IS NOT NULL
                    AND per IS NOT NULL
                ORDER BY date DESC
                LIMIT 1
            """, self.symbol, self.analysis_date)

        if not result or result['market_cap'] is None or result['per'] is None:
            return None

        market_cap = float(result['market_cap'])
        per = float(result['per'])

        # Market cap score (작을수록 높은 점수)
        # 10조 = 0점, 0원 = 100점
        market_cap_score = 100 - min(100, (market_cap / 10_000_000_000_000) * 100)

        # PER score (낮을수록 높은 점수)
        # PER 15 = 50점, 0 = 100점
        if per > 0:
            per_score = 100 - min(100, (per / 15) * 50)
        else:
            per_score = 0

        # Final score (50:50)
        final_score = (market_cap_score * 0.5) + (per_score * 0.5)

        return final_score

    async def calculate_v16(self):
        """
        V16. Small Cap Risk Avoidance Strategy (Redesigned)
        Description: Identify and avoid high-risk small cap stocks
        Target: Market cap < 200 billion won (SMALL category per batch_weight.py)
        Method: 100 - Sum(risk penalties)

        Risk Penalties (based on analysis results):
        - Operating profit negative: -40 (IC -0.16 validated)
        - ROE negative: -30 (IC -0.16 validated)
        - Extreme low PBR (< 0.5): -25 (value trap signal)
        - Extreme PER (< 3 or > 50): -20 (unreliable)
        - High debt ratio (> 200%): -20 (financial distress)
        - Sharp decline (-30% in 3M): -25 (momentum risk)
        - Low liquidity (< 30% avg): -15 (illiquidity risk)

        Returns: 0-100 (higher score = lower risk = safer small cap)
                 None for mid/large caps (>= 200 billion won)
        """
        # Step 1: Get market cap - Filter for small caps only
        market_cap_query = """
        SELECT market_cap, pbr, per, close
        FROM kr_intraday_total
        WHERE symbol = $1
            AND ($2::date IS NULL OR date = $2)
        ORDER BY date DESC
        LIMIT 1
        """

        mc_result = await self.execute_query(market_cap_query, self.symbol, self.analysis_date)

        if not mc_result or mc_result[0]['market_cap'] is None:
            return None

        market_cap = float(mc_result[0]['market_cap'])

        # Filter: Only small caps (< 200 billion won)
        SMALL_CAP_THRESHOLD = 200_000_000_000  # 2000억
        if market_cap >= SMALL_CAP_THRESHOLD:
            return None  # Not a small cap, strategy not applicable

        # Get valuation metrics
        pbr = float(mc_result[0]['pbr']) if mc_result[0]['pbr'] else None
        per = float(mc_result[0]['per']) if mc_result[0]['per'] else None
        current_price = float(mc_result[0]['close']) if mc_result[0]['close'] else None

        # Step 2: Get financial health indicators
        financial_query = """
        WITH latest_financial AS (
            SELECT
                symbol,
                bsns_year,
                rcept_dt,
                MAX(CASE WHEN sj_div = 'IS' AND account_nm IN ('영업이익', '영업이익(손실)')
                    THEN thstrm_amount END) as operating_profit,
                MAX(CASE WHEN sj_div = 'IS' AND account_nm IN ('당기순이익', '당기순이익(손실)')
                    THEN thstrm_amount END) as net_income,
                MAX(CASE WHEN sj_div = 'BS' AND account_nm IN ('자본총계')
                    THEN thstrm_amount END) as total_equity,
                MAX(CASE WHEN sj_div = 'BS' AND account_nm IN ('부채총계')
                    THEN thstrm_amount END) as total_debt
            FROM kr_financial_position
            WHERE symbol = $1
                AND rcept_dt <= COALESCE($2::date, CURRENT_DATE)
            GROUP BY symbol, bsns_year, rcept_dt
            ORDER BY bsns_year DESC, rcept_dt DESC
            LIMIT 1
        )
        SELECT
            operating_profit,
            net_income,
            total_equity,
            total_debt,
            CASE
                WHEN total_equity > 0 AND total_equity IS NOT NULL
                THEN (net_income / total_equity * 100)
                ELSE NULL
            END as roe,
            CASE
                WHEN total_equity > 0 AND total_equity IS NOT NULL
                THEN (total_debt / total_equity * 100)
                ELSE NULL
            END as debt_ratio
        FROM latest_financial
        """

        fin_result = await self.execute_query(financial_query, self.symbol, self.analysis_date)

        operating_profit = None
        roe = None
        debt_ratio = None

        if fin_result and fin_result[0]:
            operating_profit = float(fin_result[0]['operating_profit']) if fin_result[0]['operating_profit'] else None
            roe = float(fin_result[0]['roe']) if fin_result[0]['roe'] else None
            debt_ratio = float(fin_result[0]['debt_ratio']) if fin_result[0]['debt_ratio'] else None

        # Step 3: Get momentum (3-month return)
        momentum_query = """
        WITH price_data AS (
            SELECT
                close,
                date,
                ROW_NUMBER() OVER (ORDER BY date DESC) as rn
            FROM kr_intraday_total
            WHERE symbol = $1
                AND ($2::date IS NULL OR date <= $2)
                AND close IS NOT NULL
            ORDER BY date DESC
            LIMIT 65
        )
        SELECT
            MAX(CASE WHEN rn = 1 THEN close END) as current_price,
            MAX(CASE WHEN rn = 63 THEN close END) as price_3m_ago
        FROM price_data
        """

        mom_result = await self.execute_query(momentum_query, self.symbol, self.analysis_date)

        return_3m = None
        if mom_result and mom_result[0]:
            curr_p = float(mom_result[0]['current_price']) if mom_result[0]['current_price'] else None
            past_p = float(mom_result[0]['price_3m_ago']) if mom_result[0]['price_3m_ago'] else None
            if curr_p and past_p and past_p > 0:
                return_3m = ((curr_p - past_p) / past_p) * 100

        # Step 4: Get liquidity (volume trend)
        liquidity_query = """
        WITH recent_volume AS (
            SELECT
                volume,
                ROW_NUMBER() OVER (ORDER BY date DESC) as rn
            FROM kr_intraday_total
            WHERE symbol = $1
                AND ($2::date IS NULL OR date <= $2)
                AND volume IS NOT NULL
            ORDER BY date DESC
            LIMIT 30
        )
        SELECT
            AVG(volume) as avg_volume,
            MAX(CASE WHEN rn <= 5 THEN volume END) as recent_avg_volume
        FROM recent_volume
        """

        liq_result = await self.execute_query(liquidity_query, self.symbol, self.analysis_date)

        volume_ratio = None
        if liq_result and liq_result[0]:
            avg_vol = float(liq_result[0]['avg_volume']) if liq_result[0]['avg_volume'] else None
            recent_vol = float(liq_result[0]['recent_avg_volume']) if liq_result[0]['recent_avg_volume'] else None
            if avg_vol and recent_vol and avg_vol > 0:
                volume_ratio = recent_vol / avg_vol

        # Step 5: Calculate risk penalties
        risk_score = 100.0

        # Penalty 1: Operating profit negative (-40)
        if operating_profit is not None and operating_profit < 0:
            risk_score -= 40

        # Penalty 2: ROE negative (-30)
        if roe is not None and roe < 0:
            risk_score -= 30

        # Penalty 3: Extreme low PBR (-25)
        if pbr is not None and pbr < 0.5:
            risk_score -= 25

        # Penalty 4: Extreme PER (-20)
        if per is not None and (per < 3 or per > 50):
            risk_score -= 20

        # Penalty 5: High debt ratio (-20)
        if debt_ratio is not None and debt_ratio > 200:
            risk_score -= 20

        # Penalty 6: Sharp decline (-25)
        if return_3m is not None and return_3m < -30:
            risk_score -= 25

        # Penalty 7: Low liquidity (-15)
        if volume_ratio is not None and volume_ratio < 0.3:
            risk_score -= 15

        # Final score: max(0, risk_score)
        final_score = max(0.0, risk_score)

        return final_score

    # ========================================================================
    # V17. Growth-Adjusted Value for Loss-Making Companies (신규)
    # ========================================================================

    async def calculate_v17_growth_adjusted_value(self):
        """
        V17. Growth-Adjusted Value Strategy (적자 성장 기업용)

        Description: 적자 기업을 성장성과 PSR로 평가

        Logic:
        - 매출 성장률 (3년 CAGR, 1년 성장률)
        - 영업 마진 개선 추세
        - 현금 소진율 (생존 가능성)
        - PSR (섹터 대비)

        Scoring:
        - High Growth (CAGR >50%, 1Y >30%, 마진 개선, 생존 >12개월): 85점
        - Mid Growth (CAGR >20%, 1Y >10%, 마진 유지, 생존 >6개월): 65점
        - Low Growth / Troubled: 0-10점
        """

        # Step 1: 적자 여부 확인
        net_income_query = """
        SELECT thstrm_amount as net_income
        FROM kr_financial_position
        WHERE symbol = $1
            AND sj_div = 'IS'
            AND account_nm IN ('당기순이익(손실)', '당기순이익')
            AND rcept_dt <= COALESCE($2::date, CURRENT_DATE)
        ORDER BY bsns_year DESC, rcept_dt DESC
        LIMIT 1
        """

        ni_result = await self.execute_query(net_income_query, self.symbol, self.analysis_date)

        if not ni_result or ni_result[0]['net_income'] is None:
            return None

        net_income = float(ni_result[0]['net_income'])

        # 흑자면 이 전략 사용 안함 (V1~V16 사용)
        if net_income > 0:
            return None

        # Step 2: 매출 데이터 수집 (최근 3년)
        sales_query = """
        WITH sales_data AS (
            SELECT
                bsns_year,
                thstrm_amount as sales,
                ROW_NUMBER() OVER (PARTITION BY bsns_year ORDER BY rcept_dt DESC) as rn
            FROM kr_financial_position
            WHERE symbol = $1
                AND sj_div = 'IS'
                AND account_nm IN ('매출액', '수익(매출액)')
                AND rcept_dt <= COALESCE($2::date, CURRENT_DATE)
                AND thstrm_amount > 0
            ORDER BY bsns_year DESC
        )
        SELECT bsns_year, sales
        FROM sales_data
        WHERE rn = 1
        ORDER BY bsns_year DESC
        LIMIT 3
        """

        sales_result = await self.execute_query(sales_query, self.symbol, self.analysis_date)

        if not sales_result or len(sales_result) < 2:
            return 0  # 매출 데이터 부족

        # Step 3: 성장률 계산
        sales_current = float(sales_result[0]['sales'])
        sales_1y_ago = float(sales_result[1]['sales']) if len(sales_result) > 1 else None
        sales_3y_ago = float(sales_result[2]['sales']) if len(sales_result) > 2 else None

        # 1년 성장률
        if sales_1y_ago and sales_1y_ago > 0:
            sales_growth_1y = (sales_current - sales_1y_ago) / sales_1y_ago
        else:
            sales_growth_1y = 0

        # 3년 CAGR
        if sales_3y_ago and sales_3y_ago > 0:
            sales_growth_3y = (sales_current / sales_3y_ago) ** (1/3) - 1
        else:
            sales_growth_3y = sales_growth_1y  # 3년 데이터 없으면 1년으로 대체

        # Step 4: 영업 마진 추세
        margin_query = """
        WITH margin_data AS (
            SELECT
                op.bsns_year,
                op.thstrm_amount as operating_profit,
                sales.thstrm_amount as sales,
                (op.thstrm_amount::DECIMAL / NULLIF(sales.thstrm_amount, 0)) * 100 as operating_margin
            FROM kr_financial_position op
            JOIN kr_financial_position sales
                ON op.symbol = sales.symbol
                AND op.bsns_year = sales.bsns_year
                AND op.rcept_dt = sales.rcept_dt
            WHERE op.symbol = $1
                AND op.sj_div = 'IS'
                AND sales.sj_div = 'IS'
                AND op.account_nm IN ('영업이익(손실)', '영업이익')
                AND sales.account_nm IN ('매출액', '수익(매출액)')
                AND op.rcept_dt <= COALESCE($2::date, CURRENT_DATE)
            ORDER BY op.bsns_year DESC
            LIMIT 2
        )
        SELECT operating_margin
        FROM margin_data
        ORDER BY bsns_year DESC
        """

        margin_result = await self.execute_query(margin_query, self.symbol, self.analysis_date)

        operating_margin_trend = 0
        if margin_result and len(margin_result) >= 2:
            margin_current = float(margin_result[0]['operating_margin']) if margin_result[0]['operating_margin'] else -999
            margin_prev = float(margin_result[1]['operating_margin']) if margin_result[1]['operating_margin'] else -999
            if margin_current > -999 and margin_prev > -999:
                operating_margin_trend = margin_current - margin_prev

        # Step 5: 현금 소진율 (생존 가능성)
        cash_query = """
        WITH latest_cf AS (
            SELECT
                cf.thstrm_amount as operating_cash_flow,
                cash.thstrm_amount as cash_and_equivalents
            FROM kr_financial_position cf
            JOIN kr_financial_position cash
                ON cf.symbol = cash.symbol
                AND cf.bsns_year = cash.bsns_year
                AND cf.rcept_dt = cash.rcept_dt
            WHERE cf.symbol = $1
                AND cf.sj_div = 'CF'
                AND cash.sj_div = 'BS'
                AND cf.account_nm LIKE '%영업활동%현금흐름%'
                AND cash.account_nm LIKE '%현금%'
                AND cf.rcept_dt <= COALESCE($2::date, CURRENT_DATE)
            ORDER BY cf.bsns_year DESC, cf.rcept_dt DESC
            LIMIT 1
        )
        SELECT operating_cash_flow, cash_and_equivalents
        FROM latest_cf
        """

        cash_result = await self.execute_query(cash_query, self.symbol, self.analysis_date)

        months_of_runway = 999  # 기본값: 충분
        if cash_result and cash_result[0]['operating_cash_flow'] and cash_result[0]['cash_and_equivalents']:
            ocf = float(cash_result[0]['operating_cash_flow'])
            cash = float(cash_result[0]['cash_and_equivalents'])
            if ocf < 0 and cash > 0:
                monthly_burn = abs(ocf) / 12
                if monthly_burn > 0:
                    months_of_runway = cash / monthly_burn

        # Step 6: PSR (Price-to-Sales Ratio)
        psr_query = """
        SELECT
            kit.market_cap,
            fp.thstrm_amount as sales
        FROM kr_intraday_total kit
        LEFT JOIN (
            SELECT symbol, thstrm_amount
            FROM kr_financial_position
            WHERE symbol = $1
                AND sj_div = 'IS'
                AND account_nm IN ('매출액', '수익(매출액)')
                AND rcept_dt <= COALESCE($2::date, CURRENT_DATE)
            ORDER BY bsns_year DESC, rcept_dt DESC
            LIMIT 1
        ) fp ON kit.symbol = fp.symbol
        WHERE kit.symbol = $1
            AND ($2::date IS NULL OR kit.date = $2)
        ORDER BY kit.date DESC
        LIMIT 1
        """

        psr_result = await self.execute_query(psr_query, self.symbol, self.analysis_date)

        psr = None
        if psr_result and psr_result[0]['market_cap'] and psr_result[0]['sales']:
            market_cap = float(psr_result[0]['market_cap'])
            sales = float(psr_result[0]['sales'])
            if sales > 0:
                psr = market_cap / sales

        # Step 7: 섹터 중앙값 PSR 계산
        sector_psr_query = """
        WITH sector_stocks AS (
            SELECT sd.symbol, sd.theme
            FROM kr_stock_detail sd
            WHERE sd.symbol = $1
        ),
        peer_psr AS (
            SELECT
                kit.symbol,
                kit.market_cap / NULLIF(fp.thstrm_amount, 0) as psr
            FROM kr_intraday_total kit
            JOIN kr_stock_detail sd ON kit.symbol = sd.symbol
            LEFT JOIN (
                SELECT DISTINCT ON (symbol)
                    symbol, thstrm_amount
                FROM kr_financial_position
                WHERE sj_div = 'IS'
                    AND account_nm IN ('매출액', '수익(매출액)')
                    AND rcept_dt <= COALESCE($2::date, CURRENT_DATE)
                ORDER BY symbol, bsns_year DESC, rcept_dt DESC
            ) fp ON kit.symbol = fp.symbol
            WHERE sd.theme = (SELECT theme FROM sector_stocks)
                AND ($2::date IS NULL OR kit.date = $2)
                AND kit.market_cap > 0
                AND fp.thstrm_amount > 0
        )
        SELECT PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY psr) as median_psr
        FROM peer_psr
        WHERE psr > 0 AND psr < 100
        """

        sector_psr_result = await self.execute_query(sector_psr_query, self.symbol, self.analysis_date)

        sector_median_psr = 2.0  # 기본값
        if sector_psr_result and sector_psr_result[0]['median_psr']:
            sector_median_psr = float(sector_psr_result[0]['median_psr'])

        # Step 8: 점수 계산
        score = 0

        # High Growth (초고성장)
        if (sales_growth_3y > 0.50 and          # 연 50% 이상 성장
            sales_growth_1y > 0.30 and          # 최근도 30% 이상
            operating_margin_trend > 0 and      # 마진 개선 중
            months_of_runway > 12):             # 1년 이상 생존 가능

            if psr and psr < sector_median_psr * 0.7:   # 섹터 대비 30% 할인
                score = 85  # 고성장 + 저평가 = 최고 점수
            elif psr and psr < sector_median_psr:
                score = 70
            else:
                score = 50  # 성장하지만 고평가

        # Mid Growth (중성장)
        elif (sales_growth_3y > 0.20 and
              sales_growth_1y > 0.10 and
              operating_margin_trend > -0.05 and
              months_of_runway > 6):

            if psr and psr < sector_median_psr * 0.8:
                score = 65
            elif psr and psr < sector_median_psr:
                score = 45
            else:
                score = 25

        # Low Growth / Troubled (저성장 / 위험)
        else:
            if months_of_runway < 6:
                score = 0   # 생존 위험
            elif sales_growth_1y < 0:
                score = 0   # 매출 역성장
            else:
                score = 10  # 위험하지만 일단 생존

        logger.info(f"V17 Score: {score:.1f} (Sales Growth 3Y: {sales_growth_3y*100:.1f}%, 1Y: {sales_growth_1y*100:.1f}%, Runway: {months_of_runway:.0f}M)")

        return score

    # ========================================================================
    # V18. EV/Sales with Growth Adjustment (신규)
    # ========================================================================

    async def calculate_v18_ev_sales_growth(self):
        """
        V18. EV/Sales/Growth Strategy (신규)

        Description: 성장률로 조정된 EV/Sales (PEG와 유사한 개념)

        Logic:
        - EV/Sales/Growth = (EV/Sales) / (Sales Growth Rate)
        - 성장 1%당 EV/Sales = "성장 보정 밸류에이션"
        - 낮을수록 좋음 (고성장 저평가)

        Scoring:
        - 섹터 내 백분위 기반 평가
        - 하위 25% = 90점 (최고)
        - 상위 25% = 30점 (낮음)
        """

        # Step 1: Enterprise Value 계산
        ev_query = """
        WITH latest_data AS (
            SELECT
                kit.market_cap,
                fp_debt.thstrm_amount as total_debt,
                fp_cash.thstrm_amount as cash
            FROM kr_intraday_total kit
            LEFT JOIN (
                SELECT symbol, thstrm_amount
                FROM kr_financial_position
                WHERE symbol = $1
                    AND sj_div = 'BS'
                    AND account_nm LIKE '%부채총계%'
                    AND rcept_dt <= COALESCE($2::date, CURRENT_DATE)
                ORDER BY bsns_year DESC, rcept_dt DESC
                LIMIT 1
            ) fp_debt ON kit.symbol = fp_debt.symbol
            LEFT JOIN (
                SELECT symbol, thstrm_amount
                FROM kr_financial_position
                WHERE symbol = $1
                    AND sj_div = 'BS'
                    AND account_nm LIKE '%현금%'
                    AND rcept_dt <= COALESCE($2::date, CURRENT_DATE)
                ORDER BY bsns_year DESC, rcept_dt DESC
                LIMIT 1
            ) fp_cash ON kit.symbol = fp_cash.symbol
            WHERE kit.symbol = $1
                AND ($2::date IS NULL OR kit.date = $2)
            ORDER BY kit.date DESC
            LIMIT 1
        )
        SELECT market_cap, total_debt, cash
        FROM latest_data
        """

        ev_result = await self.execute_query(ev_query, self.symbol, self.analysis_date)

        if not ev_result or not ev_result[0]['market_cap']:
            return None

        market_cap = float(ev_result[0]['market_cap'])
        total_debt = float(ev_result[0]['total_debt']) if ev_result[0]['total_debt'] else 0
        cash = float(ev_result[0]['cash']) if ev_result[0]['cash'] else 0

        enterprise_value = market_cap + total_debt - cash

        if enterprise_value <= 0:
            return None

        # Step 2: Sales 및 성장률
        sales_query = """
        WITH sales_history AS (
            SELECT
                bsns_year,
                thstrm_amount as sales,
                ROW_NUMBER() OVER (PARTITION BY bsns_year ORDER BY rcept_dt DESC) as rn
            FROM kr_financial_position
            WHERE symbol = $1
                AND sj_div = 'IS'
                AND account_nm IN ('매출액', '수익(매출액)')
                AND rcept_dt <= COALESCE($2::date, CURRENT_DATE)
            ORDER BY bsns_year DESC
        )
        SELECT bsns_year, sales
        FROM sales_history
        WHERE rn = 1
        ORDER BY bsns_year DESC
        LIMIT 3
        """

        sales_result = await self.execute_query(sales_query, self.symbol, self.analysis_date)

        if not sales_result or len(sales_result) < 2:
            return None

        sales_current = float(sales_result[0]['sales'])
        sales_3y_ago = float(sales_result[2]['sales']) if len(sales_result) > 2 else float(sales_result[1]['sales'])

        if sales_current <= 0 or sales_3y_ago <= 0:
            return None

        # 3년 CAGR
        years = 3 if len(sales_result) > 2 else 2
        sales_growth_rate = (sales_current / sales_3y_ago) ** (1/years) - 1

        # Step 3: EV/Sales 계산
        ev_sales = enterprise_value / sales_current

        # Step 4: EV/Sales/Growth 계산
        if sales_growth_rate > 0.05:  # 최소 5% 성장
            ev_sales_growth = ev_sales / (sales_growth_rate * 100)
        else:
            # 저성장 (<5%): 페널티
            ev_sales_growth = ev_sales * 10  # 높은 값 = 나쁨

        # Step 5: 섹터 내 백분위 계산
        sector_query = """
        WITH sector_stocks AS (
            SELECT sd.symbol, sd.theme
            FROM kr_stock_detail sd
            WHERE sd.symbol = $1
        ),
        peer_ev_sales_growth AS (
            SELECT
                kit.symbol,
                (kit.market_cap + COALESCE(debt.thstrm_amount, 0) - COALESCE(cash.thstrm_amount, 0)) /
                NULLIF(sales_current.thstrm_amount, 0) as ev_sales,
                CASE
                    WHEN sales_growth.growth_rate > 0.05 THEN
                        ((kit.market_cap + COALESCE(debt.thstrm_amount, 0) - COALESCE(cash.thstrm_amount, 0)) /
                         NULLIF(sales_current.thstrm_amount, 0)) / (sales_growth.growth_rate * 100)
                    ELSE
                        ((kit.market_cap + COALESCE(debt.thstrm_amount, 0) - COALESCE(cash.thstrm_amount, 0)) /
                         NULLIF(sales_current.thstrm_amount, 0)) * 10
                END as ev_sales_growth_metric
            FROM kr_intraday_total kit
            JOIN kr_stock_detail sd ON kit.symbol = sd.symbol
            LEFT JOIN (
                SELECT DISTINCT ON (symbol) symbol, thstrm_amount
                FROM kr_financial_position
                WHERE sj_div = 'BS' AND account_nm LIKE '%부채총계%'
                    AND rcept_dt <= COALESCE($2::date, CURRENT_DATE)
                ORDER BY symbol, bsns_year DESC, rcept_dt DESC
            ) debt ON kit.symbol = debt.symbol
            LEFT JOIN (
                SELECT DISTINCT ON (symbol) symbol, thstrm_amount
                FROM kr_financial_position
                WHERE sj_div = 'BS' AND account_nm LIKE '%현금%'
                    AND rcept_dt <= COALESCE($2::date, CURRENT_DATE)
                ORDER BY symbol, bsns_year DESC, rcept_dt DESC
            ) cash ON kit.symbol = cash.symbol
            LEFT JOIN (
                SELECT DISTINCT ON (symbol) symbol, thstrm_amount
                FROM kr_financial_position
                WHERE sj_div = 'IS' AND account_nm IN ('매출액', '수익(매출액)')
                    AND rcept_dt <= COALESCE($2::date, CURRENT_DATE)
                ORDER BY symbol, bsns_year DESC, rcept_dt DESC
            ) sales_current ON kit.symbol = sales_current.symbol
            LEFT JOIN (
                SELECT
                    symbol,
                    CASE
                        WHEN MAX(thstrm_amount) > 0 AND MIN(thstrm_amount) > 0 THEN
                            POWER(MAX(thstrm_amount)::DECIMAL / NULLIF(MIN(thstrm_amount), 0), 1.0/2) - 1
                        ELSE
                            NULL
                    END as growth_rate
                FROM kr_financial_position
                WHERE sj_div = 'IS' AND account_nm IN ('매출액', '수익(매출액)')
                    AND rcept_dt <= COALESCE($2::date, CURRENT_DATE)
                    AND bsns_year >= EXTRACT(YEAR FROM COALESCE($2::date, CURRENT_DATE))::INTEGER - 3
                GROUP BY symbol
                HAVING COUNT(*) >= 2
            ) sales_growth ON kit.symbol = sales_growth.symbol
            WHERE sd.theme = (SELECT theme FROM sector_stocks)
                AND ($2::date IS NULL OR kit.date = $2)
                AND kit.market_cap > 0
                AND sales_current.thstrm_amount > 0
        )
        SELECT PERCENTILE_CONT(ARRAY[0.25, 0.5, 0.75]) WITHIN GROUP (ORDER BY ev_sales_growth_metric) as percentiles
        FROM peer_ev_sales_growth
        WHERE ev_sales_growth_metric > 0 AND ev_sales_growth_metric < 100
        """

        percentile_result = await self.execute_query(sector_query, self.symbol, self.analysis_date)

        # 백분위 기반 점수
        if percentile_result and percentile_result[0]['percentiles']:
            percentiles = percentile_result[0]['percentiles']
            p25, p50, p75 = percentiles[0], percentiles[1], percentiles[2]

            if ev_sales_growth < p25:
                score = 90  # 하위 25% = 최고 (낮을수록 좋음)
            elif ev_sales_growth < p50:
                score = 70  # 중앙값 이하
            elif ev_sales_growth < p75:
                score = 50  # 중앙값 이상
            else:
                score = 30  # 상위 25%
        else:
            # 절대 평가 폴백
            if ev_sales_growth < 2:
                score = 80
            elif ev_sales_growth < 5:
                score = 60
            elif ev_sales_growth < 10:
                score = 40
            else:
                score = 20

        logger.info(f"V18: EV/Sales/Growth = {ev_sales_growth:.2f} (EV/Sales: {ev_sales:.2f}, Growth: {sales_growth_rate*100:.1f}%), Score: {score:.1f}")

        return score

    # ========================================================================
    # V20. ROIC-Based Value (신규)
    # ========================================================================

    async def calculate_v20_roic_value(self):
        """
        V20. ROIC-Based Value Strategy (신규)

        Description: 자본 효율성 기반 밸류 평가

        Components:
        1. ROIC (Return on Invested Capital): NOPAT / Invested Capital
        2. EV/IC (Enterprise Value / Invested Capital)
        3. Magic Ratio: ROIC / (EV/IC)

        Logic:
        - ROIC: 투입 자본 대비 수익성 (ROE보다 정확)
        - EV/IC: 시장이 자본을 얼마로 평가하는가
        - Magic Ratio 높을수록 좋음 = 효율적인데 저평가

        Scoring:
        - ROIC >15% & EV/IC <1.5: 90점 (진짜 저평가)
        - ROIC >15% & EV/IC >2.5: 40점 (좋지만 비쌈)
        - ROIC <5% & EV/IC <1.5: 20점 (Value Trap)
        """

        # Step 1: NOPAT (Net Operating Profit After Tax)
        nopat_query = """
        WITH latest_is AS (
            SELECT
                op.thstrm_amount as operating_profit,
                tax.thstrm_amount as tax_expense,
                ni.thstrm_amount as net_income
            FROM kr_financial_position op
            LEFT JOIN kr_financial_position tax
                ON op.symbol = tax.symbol
                AND op.bsns_year = tax.bsns_year
                AND op.rcept_dt = tax.rcept_dt
                AND tax.sj_div = 'IS'
                AND tax.account_nm LIKE '%법인세%'
            LEFT JOIN kr_financial_position ni
                ON op.symbol = ni.symbol
                AND op.bsns_year = ni.bsns_year
                AND op.rcept_dt = ni.rcept_dt
                AND ni.sj_div = 'IS'
                AND ni.account_nm IN ('당기순이익(손실)', '당기순이익')
            WHERE op.symbol = $1
                AND op.sj_div = 'IS'
                AND op.account_nm IN ('영업이익(손실)', '영업이익')
                AND op.rcept_dt <= COALESCE($2::date, CURRENT_DATE)
            ORDER BY op.bsns_year DESC, op.rcept_dt DESC
            LIMIT 1
        )
        SELECT operating_profit, tax_expense, net_income
        FROM latest_is
        """

        nopat_result = await self.execute_query(nopat_query, self.symbol, self.analysis_date)

        if not nopat_result or not nopat_result[0]['operating_profit']:
            return None

        operating_profit = float(nopat_result[0]['operating_profit'])
        tax_expense = float(nopat_result[0]['tax_expense']) if nopat_result[0]['tax_expense'] else 0
        net_income = float(nopat_result[0]['net_income']) if nopat_result[0]['net_income'] else 0

        # Tax Rate 계산
        if net_income != 0:
            tax_rate = abs(tax_expense) / (net_income + abs(tax_expense))
        else:
            tax_rate = 0.25  # 기본 법인세율

        # NOPAT = Operating Profit × (1 - Tax Rate)
        nopat = operating_profit * (1 - tax_rate)

        # Step 2: Invested Capital = Total Assets - Current Liabilities
        ic_query = """
        WITH latest_bs AS (
            SELECT
                assets.thstrm_amount as total_assets,
                cl.thstrm_amount as current_liabilities
            FROM kr_financial_position assets
            LEFT JOIN kr_financial_position cl
                ON assets.symbol = cl.symbol
                AND assets.bsns_year = cl.bsns_year
                AND assets.rcept_dt = cl.rcept_dt
                AND cl.sj_div = 'BS'
                AND cl.account_nm LIKE '%유동부채%'
            WHERE assets.symbol = $1
                AND assets.sj_div = 'BS'
                AND assets.account_nm LIKE '%자산총계%'
                AND assets.rcept_dt <= COALESCE($2::date, CURRENT_DATE)
            ORDER BY assets.bsns_year DESC, assets.rcept_dt DESC
            LIMIT 1
        )
        SELECT total_assets, current_liabilities
        FROM latest_bs
        """

        ic_result = await self.execute_query(ic_query, self.symbol, self.analysis_date)

        if not ic_result or not ic_result[0]['total_assets']:
            return None

        total_assets = float(ic_result[0]['total_assets'])
        current_liabilities = float(ic_result[0]['current_liabilities']) if ic_result[0]['current_liabilities'] else 0

        invested_capital = total_assets - current_liabilities

        if invested_capital <= 0:
            return None

        # Step 3: ROIC 계산
        roic = (nopat / invested_capital) * 100

        # Step 4: Enterprise Value
        ev_query = """
        WITH latest_data AS (
            SELECT
                kit.market_cap,
                debt.thstrm_amount as total_debt,
                cash.thstrm_amount as cash
            FROM kr_intraday_total kit
            LEFT JOIN (
                SELECT symbol, thstrm_amount
                FROM kr_financial_position
                WHERE symbol = $1
                    AND sj_div = 'BS'
                    AND account_nm LIKE '%부채총계%'
                    AND rcept_dt <= COALESCE($2::date, CURRENT_DATE)
                ORDER BY bsns_year DESC, rcept_dt DESC
                LIMIT 1
            ) debt ON kit.symbol = debt.symbol
            LEFT JOIN (
                SELECT symbol, thstrm_amount
                FROM kr_financial_position
                WHERE symbol = $1
                    AND sj_div = 'BS'
                    AND account_nm LIKE '%현금%'
                    AND rcept_dt <= COALESCE($2::date, CURRENT_DATE)
                ORDER BY bsns_year DESC, rcept_dt DESC
                LIMIT 1
            ) cash ON kit.symbol = cash.symbol
            WHERE kit.symbol = $1
                AND ($2::date IS NULL OR kit.date = $2)
            ORDER BY kit.date DESC
            LIMIT 1
        )
        SELECT market_cap, total_debt, cash
        FROM latest_data
        """

        ev_result = await self.execute_query(ev_query, self.symbol, self.analysis_date)

        if not ev_result or not ev_result[0]['market_cap']:
            return None

        market_cap = float(ev_result[0]['market_cap'])
        total_debt = float(ev_result[0]['total_debt']) if ev_result[0]['total_debt'] else 0
        cash = float(ev_result[0]['cash']) if ev_result[0]['cash'] else 0

        enterprise_value = market_cap + total_debt - cash

        if enterprise_value <= 0:
            return None

        # Step 5: EV/IC 계산
        ev_ic = enterprise_value / invested_capital

        # Step 6: Magic Ratio
        if ev_ic > 0:
            magic_ratio = roic / ev_ic
        else:
            magic_ratio = 0

        # Step 7: 점수 계산
        score = 0

        # A. 높은 ROIC + 낮은 EV/IC = "진짜 저평가"
        if roic > 15 and ev_ic < 1.5:
            score = 90
            logger.info(f"V20: Excellent (ROIC: {roic:.1f}%, EV/IC: {ev_ic:.2f})")

        # B. 높은 ROIC + 높은 EV/IC = "좋지만 비쌈"
        elif roic > 15 and ev_ic > 2.5:
            score = 40
            logger.info(f"V20: Good but expensive (ROIC: {roic:.1f}%, EV/IC: {ev_ic:.2f})")

        # C. 낮은 ROIC + 낮은 EV/IC = "Value Trap"
        elif roic < 5 and ev_ic < 1.5:
            score = 20
            logger.info(f"V20: Value Trap (ROIC: {roic:.1f}%, EV/IC: {ev_ic:.2f})")

        # D. 중간 범위: Magic Ratio 기반
        else:
            # Magic Ratio 백분위 점수
            if magic_ratio > 10:
                score = 85
            elif magic_ratio > 5:
                score = 70
            elif magic_ratio > 2:
                score = 55
            elif magic_ratio > 1:
                score = 40
            else:
                score = 25

            logger.info(f"V20: Magic Ratio {magic_ratio:.2f} (ROIC: {roic:.1f}%, EV/IC: {ev_ic:.2f})")

        return score


    # ========================================================================
    # V19. Value Momentum Strategy
    # ========================================================================

    async def calculate_v19_value_momentum(self):
        """
        V19. Value Momentum Strategy

        Description: 밸류에이션 개선 추세 평가

        Logic:
        - PER/PBR이 하락하면서 (밸류 개선)
        - 주가는 상승하면 (시장이 인정)
        - = 진짜 밸류

        Scenarios:
        A. 밸류 개선 + 주가 상승 = 90점 (시장이 인정 시작)
        B. 밸류 개선 + 주가 정체 = 80점 (아직 발견 안됨, 기회!)
        C. 밸류 악화 + 주가 하락 = 10점 (Value Trap)
        D. 밸류 악화 + 주가 상승 = 30점 (버블 조짐)
        """

        # Step 1: 현재 PER
        per_current_query = """
        SELECT per
        FROM kr_intraday_detail
        WHERE symbol = $1
        LIMIT 1
        """

        per_current_result = await self.execute_query(per_current_query, self.symbol)

        if not per_current_result or not per_current_result[0]['per']:
            logger.info(f"V19: No current PER data for {self.symbol}")
            return None

        per_current = float(per_current_result[0]['per'])

        if per_current <= 0:
            logger.info(f"V19: Invalid PER ({per_current}) for {self.symbol}")
            return None

        # Step 2: 60일 전 PER (역산)
        per_60d_query = """
        WITH price_60d AS (
            SELECT close as price_60d_ago
            FROM kr_intraday_total
            WHERE symbol = $1
                AND date <= COALESCE($2::date, CURRENT_DATE) - INTERVAL '60 days'
            ORDER BY date DESC
            LIMIT 1
        ),
        latest_eps AS (
            SELECT eps
            FROM kr_intraday_detail
            WHERE symbol = $1
            LIMIT 1
        )
        SELECT
            p.price_60d_ago,
            e.eps,
            CASE
                WHEN e.eps > 0 THEN p.price_60d_ago / e.eps
                ELSE NULL
            END as per_60d_ago
        FROM price_60d p, latest_eps e
        """

        per_60d_result = await self.execute_query(per_60d_query, self.symbol, self.analysis_date)

        if not per_60d_result or not per_60d_result[0]['per_60d_ago']:
            logger.info(f"V19: No 60d PER data for {self.symbol}")
            return None

        per_60d_ago = float(per_60d_result[0]['per_60d_ago'])

        if per_60d_ago <= 0:
            logger.info(f"V19: Invalid 60d PER ({per_60d_ago}) for {self.symbol}")
            return None

        # Step 3: PER 개선율 (양수 = 개선 = PER 하락)
        per_improvement = (per_60d_ago - per_current) / per_60d_ago

        # Step 4: 주가 수익률 (60일)
        price_return_query = """
        WITH prices AS (
            SELECT
                (SELECT close FROM kr_intraday_total
                 WHERE symbol = $1 AND ($2::date IS NULL OR date = $2)
                 ORDER BY date DESC LIMIT 1) as price_now,
                (SELECT close FROM kr_intraday_total
                 WHERE symbol = $1 AND date <= COALESCE($2::date, CURRENT_DATE) - INTERVAL '60 days'
                 ORDER BY date DESC LIMIT 1) as price_60d_ago
        )
        SELECT
            price_now,
            price_60d_ago,
            (price_now - price_60d_ago) / NULLIF(price_60d_ago, 0) as price_return
        FROM prices
        """

        price_return_result = await self.execute_query(price_return_query, self.symbol, self.analysis_date)

        if not price_return_result or price_return_result[0]['price_return'] is None:
            logger.info(f"V19: No price return data for {self.symbol}")
            return None

        price_return_60d = float(price_return_result[0]['price_return'])

        # Step 5: 시나리오 분석
        score = 50  # 기본값

        # A. 밸류 개선 + 주가 상승 = "시장이 인정하기 시작"
        if per_improvement > 0.10 and price_return_60d > 0.05:
            score = 90
            logger.info(f"V19: Market recognizing value (PER ↓{per_improvement*100:.1f}%, Price ↑{price_return_60d*100:.1f}%)")

        # B. 밸류 개선 + 주가 정체 = "아직 발견 안됨" (기회!)
        elif per_improvement > 0.10 and -0.05 <= price_return_60d <= 0.05:
            score = 80
            logger.info(f"V19: Undiscovered value opportunity (PER ↓{per_improvement*100:.1f}%, Price flat {price_return_60d*100:+.1f}%)")

        # C. 밸류 악화 + 주가 하락 = "실적 악화" (Value Trap)
        elif per_improvement < -0.10 and price_return_60d < -0.05:
            score = 10
            logger.info(f"V19: Value Trap (PER ↑{abs(per_improvement)*100:.1f}%, Price ↓{abs(price_return_60d)*100:.1f}%)")

        # D. 밸류 악화 + 주가 상승 = "버블 조짐"
        elif per_improvement < -0.10 and price_return_60d > 0.05:
            score = 30
            logger.info(f"V19: Potential bubble (PER ↑{abs(per_improvement)*100:.1f}%, Price ↑{price_return_60d*100:.1f}%)")

        # E. 소폭 변화
        else:
            score = 50
            logger.info(f"V19: Neutral momentum (PER {per_improvement*100:+.1f}%, Price {price_return_60d*100:+.1f}%)")

        return score


    # ========================================================================
    # V21. Korea Adjusted PBR Strategy (Phase 3.7 - New)
    # ========================================================================

    async def calculate_v21_korea_adjusted_pbr(self):
        """
        V21. Korea Adjusted PBR Strategy

        Korean-style adjusted PBR strategy:
        - Not just low PBR, but low PBR with smart money support
        - Filter out value traps using foreign/institutional flow

        Score calculation:
        1. PBR score (40%): Lower PBR = higher score
        2. Foreign net buy score (30%): 20-day foreign net buy / market cap
        3. Institutional net buy score (30%): 20-day inst net buy / market cap

        Condition:
        - PBR < 1.5 (below market average)
        - Foreign or Institutional net buy > 0
        """
        # Step 1: Get PBR
        pbr_query = """
        SELECT pbr, market_cap
        FROM kr_intraday_total
        WHERE symbol = $1
            AND ($2::date IS NULL OR date = $2)
        ORDER BY date DESC
        LIMIT 1
        """

        pbr_result = await self.execute_query(pbr_query, self.symbol, self.analysis_date)

        if not pbr_result or pbr_result[0]['pbr'] is None:
            return None

        pbr = float(pbr_result[0]['pbr'])
        market_cap = float(pbr_result[0]['market_cap']) if pbr_result[0]['market_cap'] else 0

        if pbr <= 0 or market_cap <= 0:
            return None

        # Step 2: Get foreign/institutional net buy (last 20 days)
        investor_query = """
        SELECT
            SUM(foreign_net_value) as foreign_net_20d,
            SUM(inst_net_value) as inst_net_20d
        FROM kr_individual_investor_daily_trading
        WHERE symbol = $1
            AND date >= COALESCE($2::date, CURRENT_DATE) - INTERVAL '20 days'
            AND date <= COALESCE($2::date, CURRENT_DATE)
        """

        investor_result = await self.execute_query(investor_query, self.symbol, self.analysis_date)

        foreign_net = 0
        inst_net = 0

        if investor_result and investor_result[0]['foreign_net_20d']:
            foreign_net = float(investor_result[0]['foreign_net_20d'])
        if investor_result and investor_result[0]['inst_net_20d']:
            inst_net = float(investor_result[0]['inst_net_20d'])

        # Step 3: Calculate scores
        # PBR score (40%): PBR 0.5 or below = 100, PBR 1.5 or above = 0
        if pbr <= 0.5:
            pbr_score = 100
        elif pbr >= 1.5:
            pbr_score = 0
        else:
            pbr_score = 100 - ((pbr - 0.5) / 1.0) * 100

        # Foreign net buy score (30%): Calculate ratio to market cap
        foreign_ratio = (foreign_net / market_cap) * 100 if market_cap > 0 else 0
        if foreign_ratio >= 1.0:
            foreign_score = 100
        elif foreign_ratio <= -1.0:
            foreign_score = 0
        else:
            foreign_score = 50 + foreign_ratio * 50

        # Institutional net buy score (30%)
        inst_ratio = (inst_net / market_cap) * 100 if market_cap > 0 else 0
        if inst_ratio >= 1.0:
            inst_score = 100
        elif inst_ratio <= -1.0:
            inst_score = 0
        else:
            inst_score = 50 + inst_ratio * 50

        # Final score (weighted average)
        score = pbr_score * 0.4 + foreign_score * 0.3 + inst_score * 0.3

        # Check smart money condition: at least one of foreign or inst must be net buyer
        if foreign_net <= 0 and inst_net <= 0:
            score = score * 0.7  # 30% penalty
            logger.info(f"V21: No smart money support (PBR: {pbr:.2f}, Foreign: {foreign_net/1e8:.1f}B, Inst: {inst_net/1e8:.1f}B), Penalty applied")
        else:
            logger.info(f"V21: Korea Adjusted PBR (PBR: {pbr:.2f}, Foreign: {foreign_net/1e8:.1f}B, Inst: {inst_net/1e8:.1f}B, Score: {score:.1f})")

        return min(100, max(0, score))


    # ========================================================================
    # V22. Quality Dividend Strategy (Phase 3.7 - New)
    # ========================================================================

    async def calculate_v22_quality_dividend(self):
        """
        V22. Quality Dividend Strategy

        Quality dividend strategy:
        - Not just high dividend, but sustainable dividend stocks
        - Avoid dividend cut risk using ROE and dividend growth

        Score calculation:
        1. Dividend yield score (30%): Dividend yield vs market average
        2. ROE score (35%): Return on equity
        3. Dividend sustainability score (35%): 3-year dividend history

        Condition:
        - Dividend yield > 0%
        - ROE > 0%
        """
        # Step 1: Get dividend yield and ROE
        basic_query = """
        SELECT
            kit.dividend_yield,
            kit.market_cap,
            kit.eps,
            kit.bps
        FROM kr_intraday_total kit
        WHERE kit.symbol = $1
            AND ($2::date IS NULL OR kit.date = $2)
        ORDER BY kit.date DESC
        LIMIT 1
        """

        basic_result = await self.execute_query(basic_query, self.symbol, self.analysis_date)

        if not basic_result:
            return None

        dividend_yield = float(basic_result[0]['dividend_yield']) if basic_result[0]['dividend_yield'] else 0
        eps = float(basic_result[0]['eps']) if basic_result[0]['eps'] else 0
        bps = float(basic_result[0]['bps']) if basic_result[0]['bps'] else 0

        # Calculate ROE: EPS / BPS * 100
        roe = (eps / bps * 100) if bps > 0 and eps > 0 else 0

        # Step 2: Get dividend history (last 3 years)
        dividend_history_query = """
        SELECT
            thstrm as current_dividend,
            frmtrm as prev_dividend,
            lwfr as prev2_dividend
        FROM kr_dividends
        WHERE symbol = $1
            AND se = '현금배당'
            AND stock_knd = '보통주'
        ORDER BY stlm_dt DESC
        LIMIT 1
        """

        dividend_result = await self.execute_query(dividend_history_query, self.symbol)

        has_dividend_history = False
        dividend_growth = 0

        if dividend_result and dividend_result[0]['current_dividend']:
            current_div = float(dividend_result[0]['current_dividend']) if dividend_result[0]['current_dividend'] else 0
            prev_div = float(dividend_result[0]['prev_dividend']) if dividend_result[0]['prev_dividend'] else 0
            prev2_div = float(dividend_result[0]['prev2_dividend']) if dividend_result[0]['prev2_dividend'] else 0

            # Check 3 consecutive years of dividend
            if current_div > 0 and prev_div > 0 and prev2_div > 0:
                has_dividend_history = True
                # Dividend growth rate (2-year average)
                if prev2_div > 0:
                    dividend_growth = ((current_div / prev2_div) ** 0.5 - 1) * 100

        # Step 3: Calculate scores
        # Dividend yield score (30%): 0% = 0 points, 5% or above = 100 points
        if dividend_yield <= 0:
            yield_score = 0
        elif dividend_yield >= 5:
            yield_score = 100
        else:
            yield_score = (dividend_yield / 5) * 100

        # ROE score (35%): 0% or below = 0 points, 15% or above = 100 points
        if roe <= 0:
            roe_score = 0
        elif roe >= 15:
            roe_score = 100
        else:
            roe_score = (roe / 15) * 100

        # Dividend sustainability score (35%)
        if has_dividend_history:
            if dividend_growth >= 5:  # 5% or more annual growth
                sustainability_score = 100
            elif dividend_growth >= 0:  # Growing or stable
                sustainability_score = 70
            else:  # Declining
                sustainability_score = 40
        else:
            sustainability_score = 20  # No dividend history

        # Final score
        score = yield_score * 0.30 + roe_score * 0.35 + sustainability_score * 0.35

        logger.info(f"V22: Quality Dividend (Yield: {dividend_yield:.2f}%, ROE: {roe:.1f}%, DivGrowth: {dividend_growth:.1f}%, Score: {score:.1f})")

        return min(100, max(0, score))


    # ========================================================================
    # V23. Asset Growth Value Strategy (Phase 3.7 - New)
    # ========================================================================

    async def calculate_v23_asset_growth_value(self):
        """
        V23. Asset Growth Value Strategy

        Asset growth contrarian strategy:
        - Companies that rapidly expand assets tend to have lower returns (Asset Growth Anomaly)
        - Select undervalued companies with stable growth

        Score calculation:
        1. Asset growth reverse score (50%): Lower asset growth = higher score
        2. PBR score (30%): Lower = higher score
        3. ROE stability (20%): Positive ROE = bonus

        Data: kr_financial_position (BS - Total Assets)
        """
        # Step 1: Get asset data (current and previous period)
        asset_query = """
        WITH latest_bs AS (
            SELECT
                thstrm_amount as current_assets,
                frmtrm_amount as prev_assets,
                bsns_year
            FROM kr_financial_position
            WHERE symbol = $1
                AND sj_div = 'BS'
                AND account_nm IN ('자산총계', '자산 총계')
                AND thstrm_amount > 0
                AND rcept_dt <= COALESCE($2::date, CURRENT_DATE)
            ORDER BY bsns_year DESC, rcept_dt DESC
            LIMIT 1
        )
        SELECT current_assets, prev_assets
        FROM latest_bs
        """

        asset_result = await self.execute_query(asset_query, self.symbol, self.analysis_date)

        if not asset_result or not asset_result[0]['current_assets']:
            return None

        current_assets = float(asset_result[0]['current_assets'])
        prev_assets = float(asset_result[0]['prev_assets']) if asset_result[0]['prev_assets'] else current_assets

        # Calculate asset growth rate
        if prev_assets > 0:
            asset_growth = ((current_assets / prev_assets) - 1) * 100
        else:
            asset_growth = 0

        # Step 2: Get PBR
        pbr_query = """
        SELECT pbr
        FROM kr_intraday_total
        WHERE symbol = $1
            AND ($2::date IS NULL OR date = $2)
        ORDER BY date DESC
        LIMIT 1
        """

        pbr_result = await self.execute_query(pbr_query, self.symbol, self.analysis_date)
        pbr = float(pbr_result[0]['pbr']) if pbr_result and pbr_result[0]['pbr'] else 1.0

        # Step 3: Get ROE
        roe_query = """
        SELECT
            ni.thstrm_amount as net_income,
            eq.thstrm_amount as equity
        FROM kr_financial_position ni
        JOIN kr_financial_position eq ON ni.symbol = eq.symbol AND ni.bsns_year = eq.bsns_year
        WHERE ni.symbol = $1
            AND ni.sj_div = 'IS'
            AND ni.account_nm IN ('당기순이익(손실)', '당기순이익')
            AND eq.sj_div = 'BS'
            AND eq.account_nm IN ('자본총계', '기말자본')
            AND eq.thstrm_amount > 0
            AND ni.rcept_dt <= COALESCE($2::date, CURRENT_DATE)
        ORDER BY ni.bsns_year DESC, ni.rcept_dt DESC
        LIMIT 1
        """

        roe_result = await self.execute_query(roe_query, self.symbol, self.analysis_date)
        roe = 0
        if roe_result and roe_result[0]['net_income'] and roe_result[0]['equity']:
            net_income = float(roe_result[0]['net_income'])
            equity = float(roe_result[0]['equity'])
            if equity > 0:
                roe = (net_income / equity) * 100

        # Step 4: Calculate scores
        # Asset growth reverse score (50%): -10% ~ +30% range, lower = higher score
        if asset_growth <= 0:
            asset_score = 100  # Asset reduction or stable = best
        elif asset_growth >= 30:
            asset_score = 20  # Rapid asset expansion = risky
        else:
            asset_score = 100 - (asset_growth / 30) * 80

        # PBR score (30%): 0.5 or below = 100, 2.0 or above = 0
        if pbr <= 0.5:
            pbr_score = 100
        elif pbr >= 2.0:
            pbr_score = 0
        else:
            pbr_score = 100 - ((pbr - 0.5) / 1.5) * 100

        # ROE stability score (20%): Positive ROE = bonus
        if roe >= 10:
            roe_score = 100
        elif roe >= 0:
            roe_score = 50 + roe * 5
        else:
            roe_score = 30  # Loss-making

        # Final score
        score = asset_score * 0.50 + pbr_score * 0.30 + roe_score * 0.20

        logger.info(f"V23: Asset Growth Value (AssetGrowth: {asset_growth:.1f}%, PBR: {pbr:.2f}, ROE: {roe:.1f}%, Score: {score:.1f})")

        return min(100, max(0, score))


    # ========================================================================
    # V24. Operating Leverage Value Strategy (Phase 3.7 - New)
    # ========================================================================

    async def calculate_v24_operating_leverage(self):
        """
        V24. Operating Leverage Value Strategy

        Operating leverage value strategy:
        - Companies where operating profit increases more than sales growth
        - High fixed cost ratio leads to rapid profit improvement in turnaround

        Score calculation:
        1. Operating leverage (40%): (Operating profit growth / Sales growth)
        2. Operating margin improvement (30%): YoY margin change
        3. Low PER score (30%): Lower PER = higher score

        Data: kr_financial_position (CIS - Operating profit, Sales)
        """
        # Step 1: Get operating profit and sales (current and previous)
        financial_query = """
        WITH latest_report AS (
            SELECT bsns_year, rcept_dt
            FROM kr_financial_position
            WHERE symbol = $1
                AND rcept_dt <= COALESCE($2::date, CURRENT_DATE)
            ORDER BY bsns_year DESC, rcept_dt DESC
            LIMIT 1
        )
        SELECT
            (SELECT thstrm_amount FROM kr_financial_position, latest_report lr
             WHERE symbol = $1 AND sj_div = 'IS'
             AND account_nm IN ('영업이익(손실)', '영업이익')
             AND kr_financial_position.bsns_year = lr.bsns_year
             LIMIT 1) as op_current,

            (SELECT frmtrm_amount FROM kr_financial_position, latest_report lr
             WHERE symbol = $1 AND sj_div = 'IS'
             AND account_nm IN ('영업이익(손실)', '영업이익')
             AND kr_financial_position.bsns_year = lr.bsns_year
             LIMIT 1) as op_prev,

            (SELECT thstrm_amount FROM kr_financial_position, latest_report lr
             WHERE symbol = $1 AND sj_div = 'IS'
             AND account_nm IN ('매출액', '수익(매출액)')
             AND kr_financial_position.bsns_year = lr.bsns_year
             LIMIT 1) as sales_current,

            (SELECT frmtrm_amount FROM kr_financial_position, latest_report lr
             WHERE symbol = $1 AND sj_div = 'IS'
             AND account_nm IN ('매출액', '수익(매출액)')
             AND kr_financial_position.bsns_year = lr.bsns_year
             LIMIT 1) as sales_prev
        """

        financial_result = await self.execute_query(financial_query, self.symbol, self.analysis_date)

        if not financial_result:
            return None

        op_current = float(financial_result[0]['op_current']) if financial_result[0]['op_current'] else 0
        op_prev = float(financial_result[0]['op_prev']) if financial_result[0]['op_prev'] else 0
        sales_current = float(financial_result[0]['sales_current']) if financial_result[0]['sales_current'] else 0
        sales_prev = float(financial_result[0]['sales_prev']) if financial_result[0]['sales_prev'] else 0

        if sales_current <= 0:
            return None

        # Calculate operating margin
        op_margin_current = (op_current / sales_current) * 100 if sales_current > 0 else 0
        op_margin_prev = (op_prev / sales_prev) * 100 if sales_prev > 0 else 0

        # Calculate growth rates
        sales_growth = ((sales_current / sales_prev) - 1) * 100 if sales_prev > 0 else 0
        op_growth = ((op_current / op_prev) - 1) * 100 if op_prev > 0 and op_current > 0 else 0

        # Operating leverage
        operating_leverage = op_growth / sales_growth if sales_growth > 0 and op_growth != 0 else 1.0

        # Step 2: Get PER
        per_query = """
        SELECT per
        FROM kr_intraday_total
        WHERE symbol = $1
            AND ($2::date IS NULL OR date = $2)
        ORDER BY date DESC
        LIMIT 1
        """

        per_result = await self.execute_query(per_query, self.symbol, self.analysis_date)
        per = float(per_result[0]['per']) if per_result and per_result[0]['per'] and per_result[0]['per'] > 0 else 15

        # Step 3: Calculate scores
        # Operating leverage score (40%): Leverage > 1 means high op profit elasticity
        if op_current > 0:  # Profitable companies only
            if operating_leverage >= 2.0:
                leverage_score = 100
            elif operating_leverage >= 1.0:
                leverage_score = 50 + (operating_leverage - 1.0) * 50
            else:
                leverage_score = operating_leverage * 50
        else:
            leverage_score = 30  # Loss-making

        # Operating margin improvement score (30%)
        margin_improvement = op_margin_current - op_margin_prev
        if margin_improvement >= 5:
            margin_score = 100
        elif margin_improvement >= 0:
            margin_score = 50 + margin_improvement * 10
        elif margin_improvement >= -5:
            margin_score = 50 + margin_improvement * 10
        else:
            margin_score = 0

        # PER score (30%): PER 5 or below = 100, PER 30 or above = 0
        if per <= 5:
            per_score = 100
        elif per >= 30:
            per_score = 0
        else:
            per_score = 100 - ((per - 5) / 25) * 100

        # Final score
        score = leverage_score * 0.40 + margin_score * 0.30 + per_score * 0.30

        logger.info(f"V24: Operating Leverage (OPLev: {operating_leverage:.2f}, MarginChg: {margin_improvement:.1f}%p, PER: {per:.1f}, Score: {score:.1f})")

        return min(100, max(0, score))


    # ========================================================================
    # V25. Cash Rich Undervalued Strategy (Phase 3.7 - New)
    # ========================================================================

    async def calculate_v25_cash_rich_undervalued(self):
        """
        V25. Cash Rich Undervalued Strategy

        Cash-rich undervalued strategy:
        - Companies with high net cash (cash - debt) relative to market cap
        - Extreme undervaluation where liquidation value exceeds market cap

        Score calculation:
        1. Net cash ratio (50%): (Cash - Total Debt) / Market Cap
        2. PBR score (30%): Lower = higher score
        3. Current ratio score (20%): Short-term solvency

        Data: kr_financial_position (BS - Cash, Total Debt)
        """
        # Step 1: Get financial data
        financial_query = """
        WITH latest_bs AS (
            SELECT bsns_year, rcept_dt
            FROM kr_financial_position
            WHERE symbol = $1
                AND sj_div = 'BS'
                AND rcept_dt <= COALESCE($2::date, CURRENT_DATE)
            ORDER BY bsns_year DESC, rcept_dt DESC
            LIMIT 1
        )
        SELECT
            (SELECT thstrm_amount FROM kr_financial_position, latest_bs lb
             WHERE symbol = $1 AND sj_div = 'BS'
             AND account_nm LIKE '%현금및현금성자산%'
             AND kr_financial_position.bsns_year = lb.bsns_year
             LIMIT 1) as cash,

            (SELECT thstrm_amount FROM kr_financial_position, latest_bs lb
             WHERE symbol = $1 AND sj_div = 'BS'
             AND account_nm IN ('부채총계', '부채 총계')
             AND kr_financial_position.bsns_year = lb.bsns_year
             LIMIT 1) as total_debt,

            (SELECT thstrm_amount FROM kr_financial_position, latest_bs lb
             WHERE symbol = $1 AND sj_div = 'BS'
             AND account_nm IN ('유동자산', '유동 자산')
             AND kr_financial_position.bsns_year = lb.bsns_year
             LIMIT 1) as current_assets,

            (SELECT thstrm_amount FROM kr_financial_position, latest_bs lb
             WHERE symbol = $1 AND sj_div = 'BS'
             AND account_nm IN ('유동부채', '유동 부채')
             AND kr_financial_position.bsns_year = lb.bsns_year
             LIMIT 1) as current_liabilities
        """

        financial_result = await self.execute_query(financial_query, self.symbol, self.analysis_date)

        cash = 0
        total_debt = 0
        current_assets = 0
        current_liabilities = 0

        if financial_result:
            cash = float(financial_result[0]['cash']) if financial_result[0]['cash'] else 0
            total_debt = float(financial_result[0]['total_debt']) if financial_result[0]['total_debt'] else 0
            current_assets = float(financial_result[0]['current_assets']) if financial_result[0]['current_assets'] else 0
            current_liabilities = float(financial_result[0]['current_liabilities']) if financial_result[0]['current_liabilities'] else 1

        # Step 2: Get market cap and PBR
        market_query = """
        SELECT market_cap, pbr
        FROM kr_intraday_total
        WHERE symbol = $1
            AND ($2::date IS NULL OR date = $2)
        ORDER BY date DESC
        LIMIT 1
        """

        market_result = await self.execute_query(market_query, self.symbol, self.analysis_date)

        if not market_result or not market_result[0]['market_cap']:
            return None

        market_cap = float(market_result[0]['market_cap'])
        pbr = float(market_result[0]['pbr']) if market_result[0]['pbr'] else 1.0

        if market_cap <= 0:
            return None

        # Step 3: Calculate ratios
        net_cash = cash - total_debt
        net_cash_ratio = (net_cash / market_cap) * 100  # Net cash ratio vs market cap
        current_ratio = (current_assets / current_liabilities) * 100 if current_liabilities > 0 else 100

        # Step 4: Calculate scores
        # Net cash ratio score (50%): -50% = 0, +50% = 100
        if net_cash_ratio >= 50:
            cash_score = 100
        elif net_cash_ratio <= -50:
            cash_score = 0
        else:
            cash_score = 50 + net_cash_ratio

        # PBR score (30%): 0.3 or below = 100, 1.5 or above = 0
        if pbr <= 0.3:
            pbr_score = 100
        elif pbr >= 1.5:
            pbr_score = 0
        else:
            pbr_score = 100 - ((pbr - 0.3) / 1.2) * 100

        # Current ratio score (20%): 200% or above = 100, 100% or below = 0
        if current_ratio >= 200:
            liquidity_score = 100
        elif current_ratio <= 100:
            liquidity_score = 0
        else:
            liquidity_score = (current_ratio - 100)

        # Final score
        score = cash_score * 0.50 + pbr_score * 0.30 + liquidity_score * 0.20

        logger.info(f"V25: Cash Rich (NetCash/MktCap: {net_cash_ratio:.1f}%, PBR: {pbr:.2f}, CurRatio: {current_ratio:.0f}%, Score: {score:.1f})")

        # Apply extreme risk penalty
        if score >= 40:
            volatility = await self._get_volatility_for_risk_adjustment()
            beta = await self._get_beta_for_risk_adjustment()
            var_95 = await self._get_var_for_risk_adjustment()

            vol_penalty = await self._calculate_volatility_penalty(volatility)
            beta_penalty = await self._calculate_beta_penalty(beta)
            var_penalty = await self._calculate_var_penalty(var_95)

            total_penalty = vol_penalty + beta_penalty + var_penalty

            if total_penalty < 0:
                logger.info(f"V25: Extreme risk penalty applied: {total_penalty} (Vol: {vol_penalty}, Beta: {beta_penalty}, VaR: {var_penalty})")
                score = score + total_penalty

        return min(100, max(0, score))


    # ========================================================================
    # V26. Smart Money Value Strategy (Phase 3.7 - New)
    # ========================================================================

    async def calculate_v26_smart_money_value(self):
        """
        V26. Smart Money Value Strategy

        Smart money value strategy:
        - Undervalued stocks where both foreign and institutional investors are buying
        - Leveraging collective wisdom of information-advantaged investors

        Score calculation:
        1. Foreign net buy (35%): 20-day foreign net buy / market cap
        2. Institutional net buy (35%): 20-day inst net buy / market cap
        3. PBR undervaluation (30%): Lower PBR = higher score

        Condition:
        - Foreign net buy > 0 AND Institutional net buy > 0 = bonus
        - PBR < 1.5

        Data: kr_individual_investor_daily_trading
        """
        # Step 1: Get market cap and PBR
        market_query = """
        SELECT market_cap, pbr
        FROM kr_intraday_total
        WHERE symbol = $1
            AND ($2::date IS NULL OR date = $2)
        ORDER BY date DESC
        LIMIT 1
        """

        market_result = await self.execute_query(market_query, self.symbol, self.analysis_date)

        if not market_result or not market_result[0]['market_cap']:
            return None

        market_cap = float(market_result[0]['market_cap'])
        pbr = float(market_result[0]['pbr']) if market_result[0]['pbr'] else 1.0

        if market_cap <= 0:
            return None

        # Step 2: Get foreign/institutional net buy (last 20 days)
        investor_query = """
        SELECT
            SUM(foreign_net_value) as foreign_net_20d,
            SUM(inst_net_value) as inst_net_20d,
            COUNT(*) as trading_days
        FROM kr_individual_investor_daily_trading
        WHERE symbol = $1
            AND date >= COALESCE($2::date, CURRENT_DATE) - INTERVAL '20 days'
            AND date <= COALESCE($2::date, CURRENT_DATE)
        """

        investor_result = await self.execute_query(investor_query, self.symbol, self.analysis_date)

        foreign_net = 0
        inst_net = 0

        if investor_result and investor_result[0]['foreign_net_20d']:
            foreign_net = float(investor_result[0]['foreign_net_20d'])
        if investor_result and investor_result[0]['inst_net_20d']:
            inst_net = float(investor_result[0]['inst_net_20d'])

        # Step 3: Calculate ratios (% of market cap)
        foreign_ratio = (foreign_net / market_cap) * 100 if market_cap > 0 else 0
        inst_ratio = (inst_net / market_cap) * 100 if market_cap > 0 else 0

        # Step 4: Calculate scores
        # Foreign net buy score (35%): -2% ~ +2% range
        if foreign_ratio >= 2.0:
            foreign_score = 100
        elif foreign_ratio <= -2.0:
            foreign_score = 0
        else:
            foreign_score = 25 + (foreign_ratio + 2) * 18.75

        # Institutional net buy score (35%)
        if inst_ratio >= 2.0:
            inst_score = 100
        elif inst_ratio <= -2.0:
            inst_score = 0
        else:
            inst_score = 25 + (inst_ratio + 2) * 18.75

        # PBR score (30%): 0.5 or below = 100, 2.0 or above = 0
        if pbr <= 0.5:
            pbr_score = 100
        elif pbr >= 2.0:
            pbr_score = 0
        else:
            pbr_score = 100 - ((pbr - 0.5) / 1.5) * 100

        # Base score
        score = foreign_score * 0.35 + inst_score * 0.35 + pbr_score * 0.30

        # Smart money aligned bonus
        if foreign_net > 0 and inst_net > 0:
            score = min(100, score * 1.15)  # 15% bonus
            logger.info(f"V26: Smart Money ALIGNED! (Foreign: {foreign_ratio:.2f}%, Inst: {inst_ratio:.2f}%, PBR: {pbr:.2f}, Score: {score:.1f})")
        elif foreign_net > 0 or inst_net > 0:
            logger.info(f"V26: Partial Smart Money (Foreign: {foreign_ratio:.2f}%, Inst: {inst_ratio:.2f}%, PBR: {pbr:.2f}, Score: {score:.1f})")
        else:
            score = score * 0.8  # 20% penalty
            logger.info(f"V26: No Smart Money support (Foreign: {foreign_ratio:.2f}%, Inst: {inst_ratio:.2f}%, Score: {score:.1f})")

        return min(100, max(0, score))

    # ========================================================================
    # V27. Total Shareholder Return (총주주수익률) - Phase 3.8
    # ========================================================================

    async def calculate_v27_total_shareholder_return(self):
        """
        V27. Total Shareholder Return Strategy (Phase 3.8 - Paradigm Shift)

        핵심 개념: "저평가"가 아닌 "주주환원" 중심 가치 평가
        - 한국 시장의 코리아 디스카운트는 낮은 주주환원에서 기인
        - Value-Up 프로그램 정신 반영: 배당 + 자사주매입 = 총주주수익률

        Score calculation:
        1. Dividend Yield (40%): 현금배당수익률
        2. Treasury Stock Buyback Rate (35%): 자사주매입금액 / 시가총액
        3. Payout Consistency (25%): 연속 배당 년수

        Condition:
        - 최근 배당 or 자사주매입 실적이 있어야 함
        - 둘 다 있으면 보너스

        Data: kr_stock_dividends, kr_treasury_stocks, kr_intraday_total
        """
        # Step 1: Get current market cap and price
        market_query = """
        SELECT market_cap, close as current_price
        FROM kr_intraday_total
        WHERE symbol = $1
            AND ($2::date IS NULL OR date <= $2)
        ORDER BY date DESC
        LIMIT 1
        """

        market_result = await self.execute_query(market_query, self.symbol, self.analysis_date)

        if not market_result or not market_result[0]['market_cap']:
            return None

        market_cap = float(market_result[0]['market_cap'])
        current_price = float(market_result[0]['current_price']) if market_result[0]['current_price'] else 0

        if market_cap <= 0:
            return None

        # Step 2: Get dividend data from kr_intraday_total (real-time dividend data)
        dividend_query = """
        SELECT
            dps as cash_dividend_per_share,
            dividend_yield
        FROM kr_intraday_total
        WHERE symbol = $1
            AND date <= COALESCE($2::date, CURRENT_DATE)
            AND dps IS NOT NULL AND dps > 0
        ORDER BY date DESC
        LIMIT 1
        """

        dividend_result = await self.execute_query(dividend_query, self.symbol, self.analysis_date)

        dividend_yield = 0.0
        consecutive_dividend_years = 0

        if dividend_result:
            # Latest dividend yield from kr_intraday_total
            if dividend_result[0]['dividend_yield']:
                dividend_yield = float(dividend_result[0]['dividend_yield'])
            elif dividend_result[0]['cash_dividend_per_share'] and current_price > 0:
                dividend_yield = (float(dividend_result[0]['cash_dividend_per_share']) / current_price) * 100

            # If dividend exists, assume at least 1 year of consecutive dividends
            if dividend_yield > 0:
                consecutive_dividend_years = 1

        # Step 3: Get treasury stock buyback data from kr_stockacquisitiondisposal
        # Uses change_qy_acqs (acquired quantity) as proxy for buyback activity
        treasury_query = """
        SELECT
            SUM(change_qy_acqs) as total_buyback_volume
        FROM kr_stockacquisitiondisposal
        WHERE symbol = $1
            AND stlm_dt >= COALESCE($2::date, CURRENT_DATE) - INTERVAL '2 years'
            AND stlm_dt <= COALESCE($2::date, CURRENT_DATE)
            AND change_qy_acqs > 0
        """

        treasury_result = await self.execute_query(treasury_query, self.symbol, self.analysis_date)

        buyback_rate = 0.0
        has_buyback = False

        # Note: kr_stockacquisitiondisposal only has volume, not amount
        # If company has buyback activity, give a moderate score
        if treasury_result and treasury_result[0]['total_buyback_volume']:
            buyback_volume = float(treasury_result[0]['total_buyback_volume'])
            if buyback_volume > 0:
                # Estimate buyback rate based on volume vs listed shares
                # Use a proxy: if any buyback exists, assume 1% rate (moderate activity)
                buyback_rate = 1.0  # Base rate for having buyback activity
                has_buyback = True

        # Step 4: Calculate component scores

        # Dividend Yield Score (40%): 0% = 0, 3% = 70, 5%+ = 100
        if dividend_yield >= 5.0:
            dividend_score = 100
        elif dividend_yield <= 0:
            dividend_score = 0
        else:
            dividend_score = min(100, dividend_yield * 20)

        # Treasury Buyback Score (35%): 0% = 0, 1% = 70, 2%+ = 100
        if buyback_rate >= 2.0:
            buyback_score = 100
        elif buyback_rate <= 0:
            buyback_score = 0
        else:
            buyback_score = min(100, buyback_rate * 50)

        # Payout Consistency Score (25%): 0년 = 0, 1년 = 50, 2년 = 80, 3년+ = 100
        if consecutive_dividend_years >= 3:
            consistency_score = 100
        elif consecutive_dividend_years == 2:
            consistency_score = 80
        elif consecutive_dividend_years == 1:
            consistency_score = 50
        else:
            consistency_score = 0

        # Base score
        score = dividend_score * 0.40 + buyback_score * 0.35 + consistency_score * 0.25

        # Bonus: Both dividend AND buyback = +15%
        has_dividend = dividend_yield > 0
        if has_dividend and has_buyback:
            score = min(100, score * 1.15)
            logger.info(f"V27: FULL Shareholder Return! (Div: {dividend_yield:.2f}%, Buyback: {buyback_rate:.2f}%, Score: {score:.1f})")
        elif has_dividend or has_buyback:
            logger.info(f"V27: Partial Shareholder Return (Div: {dividend_yield:.2f}%, Buyback: {buyback_rate:.2f}%, Score: {score:.1f})")
        else:
            # No shareholder return = heavy penalty
            score = score * 0.5
            logger.info(f"V27: No Shareholder Return (Score: {score:.1f})")

        return min(100, max(0, score))

    # ========================================================================
    # V28. Value Trap Filter (가치함정 회피) - Phase 3.8
    # ========================================================================

    async def calculate_v28_value_trap_filter(self):
        """
        V28. Value Trap Filter Strategy (Phase 3.8 - Paradigm Shift)

        핵심 개념: 한국 시장의 "싸면 더 싸지는" 가치함정 회피
        - 3년 연속 PBR < 0.5 + ROE < 5% = 구조적 저평가 (함정)
        - 단순 저PBR이 아닌, 저평가 해소 가능성 평가

        Score calculation:
        1. PBR Level Check: 3년 평균 PBR (역상관)
        2. ROE Trend: 3년간 ROE 추세 (개선 = 고점수)
        3. Value Trap Detection: PBR < 0.5 + ROE < 5% = 패널티

        Logic:
        - 저PBR + 고ROE = 진정한 저평가 (고점수)
        - 저PBR + 저ROE = 가치함정 (저점수)
        - 고PBR + 고ROE = 성장주 (중간)
        - 고PBR + 저ROE = 고평가 (저점수)

        Data: kr_intraday_total, kr_financial_position
        """
        # Step 1: Get 3-year PBR history
        pbr_query = """
        SELECT
            EXTRACT(YEAR FROM date) as year,
            AVG(pbr) as avg_pbr
        FROM kr_intraday_total
        WHERE symbol = $1
            AND date >= COALESCE($2::date, CURRENT_DATE) - INTERVAL '3 years'
            AND date <= COALESCE($2::date, CURRENT_DATE)
            AND pbr > 0
        GROUP BY EXTRACT(YEAR FROM date)
        ORDER BY year DESC
        LIMIT 3
        """

        pbr_result = await self.execute_query(pbr_query, self.symbol, self.analysis_date)

        if not pbr_result:
            return None

        pbr_values = [float(row['avg_pbr']) for row in pbr_result if row['avg_pbr']]
        if not pbr_values:
            return None

        avg_pbr_3yr = sum(pbr_values) / len(pbr_values)
        current_pbr = pbr_values[0] if pbr_values else avg_pbr_3yr

        # Step 2: Get 3-year ROE data
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
            AND bsns_year >= EXTRACT(YEAR FROM COALESCE($2::date, CURRENT_DATE)) - 3
        GROUP BY bsns_year
        ORDER BY bsns_year DESC
        LIMIT 3
        """

        roe_result = await self.execute_query(roe_query, self.symbol, self.analysis_date)

        roe_values = []
        if roe_result:
            for row in roe_result:
                if row['net_income'] and row['total_equity'] and float(row['total_equity']) > 0:
                    roe = (float(row['net_income']) / float(row['total_equity'])) * 100
                    roe_values.append(roe)

        avg_roe_3yr = sum(roe_values) / len(roe_values) if roe_values else 0
        current_roe = roe_values[0] if roe_values else 0

        # Step 3: ROE Trend (improvement = positive)
        roe_trend = 0
        if len(roe_values) >= 2:
            roe_trend = roe_values[0] - roe_values[-1]  # Most recent - oldest

        # Step 4: Value Trap Detection
        is_value_trap = False
        if avg_pbr_3yr < 0.5 and avg_roe_3yr < 5:
            is_value_trap = True

        is_genuine_value = False
        if current_pbr < 1.0 and current_roe >= 10:
            is_genuine_value = True

        # Step 5: Calculate component scores

        # ROE Score (50%): 0% = 0, 10% = 70, 20%+ = 100
        if current_roe >= 20:
            roe_score = 100
        elif current_roe <= 0:
            roe_score = 0
        else:
            roe_score = min(100, current_roe * 5)

        # ROE Trend Score (30%): -5% = 0, 0% = 50, +5%+ = 100
        if roe_trend >= 5:
            trend_score = 100
        elif roe_trend <= -5:
            trend_score = 0
        else:
            trend_score = 50 + (roe_trend * 10)

        # PBR Reasonability Score (20%): Inverted U-shape
        # Too low (< 0.3) or too high (> 3) = risky
        if 0.5 <= current_pbr <= 1.5:
            pbr_score = 100  # Sweet spot
        elif current_pbr < 0.3:
            pbr_score = 30  # Potential value trap
        elif current_pbr < 0.5:
            pbr_score = 60  # Caution
        elif current_pbr <= 2.0:
            pbr_score = 80  # Acceptable
        elif current_pbr <= 3.0:
            pbr_score = 50  # Expensive
        else:
            pbr_score = 20  # Very expensive

        # Base score
        score = roe_score * 0.50 + trend_score * 0.30 + pbr_score * 0.20

        # Value Trap Penalty: -30%
        if is_value_trap:
            score = score * 0.7
            logger.info(f"V28: VALUE TRAP DETECTED! (PBR: {avg_pbr_3yr:.2f}, ROE: {avg_roe_3yr:.1f}%, Score: {score:.1f})")
        # Genuine Value Bonus: +20%
        elif is_genuine_value:
            score = min(100, score * 1.2)
            logger.info(f"V28: Genuine Value! (PBR: {current_pbr:.2f}, ROE: {current_roe:.1f}%, Score: {score:.1f})")
        else:
            logger.info(f"V28: Neutral (PBR: {current_pbr:.2f}, ROE: {current_roe:.1f}%, Score: {score:.1f})")

        return min(100, max(0, score))

    # ========================================================================
    # V29. Accounting Freshness (회계 시의성) - Phase 3.8
    # ========================================================================

    async def calculate_v29_accounting_freshness(self):
        """
        V29. Accounting Freshness Strategy (Phase 3.8 - User Suggestion)

        핵심 개념: 한국 시장 회계 시차 문제 반영
        - 분기보고서 미공시 시 패널티 (10-25% 차감)
        - 최신 재무정보가 없으면 신뢰도 하락

        Score calculation:
        1. Report Timeliness: 최근 분기보고서 공시 여부
        2. Data Completeness: 주요 재무항목 완성도

        report_code:
        - 11011: 1분기보고서
        - 11012: 반기보고서
        - 11013: 3분기보고서
        - 11014: 사업보고서 (연간)

        Data: kr_financial_position
        """
        # Step 1: Check most recent report
        report_query = """
        SELECT
            report_code,
            bsns_year,
            rcept_dt,
            COUNT(DISTINCT account_nm) as item_count
        FROM kr_financial_position
        WHERE symbol = $1
        GROUP BY report_code, bsns_year, rcept_dt
        ORDER BY rcept_dt DESC
        LIMIT 4
        """

        report_result = await self.execute_query(report_query, self.symbol)

        if not report_result:
            return None

        # Get current date context
        from datetime import datetime
        analysis_date = self.analysis_date if self.analysis_date else datetime.now().date()
        current_year = analysis_date.year
        current_quarter = (analysis_date.month - 1) // 3 + 1

        # Expected report code based on current date
        expected_reports = {
            1: '11014',  # Q1: Should have annual report from prev year
            2: '11011',  # Q2: Should have Q1 report
            3: '11012',  # Q3: Should have semi-annual report
            4: '11013',  # Q4: Should have Q3 report
        }
        expected_report = expected_reports.get(current_quarter, '11014')

        # Check latest report
        latest_report = report_result[0]
        latest_report_code = latest_report['report_code']
        latest_report_date = latest_report['rcept_dt']
        item_count = int(latest_report['item_count'])

        # Calculate days since last report
        if latest_report_date:
            days_since_report = (analysis_date - latest_report_date).days if hasattr(latest_report_date, '__sub__') else 0
        else:
            days_since_report = 365  # Default to 1 year if unknown

        # Step 2: Calculate Timeliness Score (60%)
        # Report within 90 days = 100, 180 days = 50, 365+ days = 0
        if days_since_report <= 90:
            timeliness_score = 100
        elif days_since_report <= 180:
            timeliness_score = 100 - ((days_since_report - 90) / 90) * 50
        elif days_since_report <= 365:
            timeliness_score = 50 - ((days_since_report - 180) / 185) * 50
        else:
            timeliness_score = 0

        # Step 3: Calculate Completeness Score (40%)
        # Expected items: BS ~30, CIS ~20, CF ~10 = ~60 items
        if item_count >= 50:
            completeness_score = 100
        elif item_count >= 30:
            completeness_score = 70 + (item_count - 30) * 1.5
        elif item_count >= 10:
            completeness_score = 30 + (item_count - 10) * 2
        else:
            completeness_score = item_count * 3

        # Base score
        score = timeliness_score * 0.60 + completeness_score * 0.40

        # Penalty for very stale data (> 6 months)
        if days_since_report > 180:
            score = score * 0.85  # 15% penalty
            logger.info(f"V29: STALE DATA ({days_since_report} days old, Score: {score:.1f})")
        elif days_since_report > 90:
            score = score * 0.95  # 5% penalty
            logger.info(f"V29: Slightly outdated ({days_since_report} days, Score: {score:.1f})")
        else:
            logger.info(f"V29: Fresh data ({days_since_report} days, Score: {score:.1f})")

        return min(100, max(0, score))

    # ========================================================================
    # V30. Governance Quality (지배구조 품질) - Phase 3.8
    # ========================================================================

    async def calculate_v30_governance_quality(self):
        """
        V30. Governance Quality Strategy (Phase 3.8 - Paradigm Shift)

        핵심 개념: 지배구조 품질 = 주주가치 실현 가능성
        - 외국인/기관 수급: 정보우위 투자자의 평가
        - 배당 + 자사주: 주주환원 의지
        - Value-Up 정신: 주주친화적 경영

        Score calculation:
        1. Institutional Quality (40%): 외국인 + 기관 보유비율 합산
        2. Shareholder Return Commitment (35%): 배당성향 + 자사주 비율
        3. Ownership Stability (25%): 최대주주 지분율 (너무 높으면 감점)

        Data: kr_individual_investor_daily_trading, kr_stock_dividends, kr_treasury_stocks
        """
        # Combined query: 4개 테이블을 서브쿼리로 병합 (연결 수 4→1 최적화)
        combined_query = """
        SELECT
            -- 1. Foreign ownership (kr_foreign_ownership)
            (SELECT foreign_rate
             FROM kr_foreign_ownership
             WHERE symbol = $1
             ORDER BY date DESC
             LIMIT 1) as foreign_holding_rate,

            -- 2. Institutional buy ratio (kr_individual_investor_daily_trading)
            (SELECT AVG(inst_buy_ratio)
             FROM kr_individual_investor_daily_trading
             WHERE symbol = $1
               AND date >= COALESCE($2::date, CURRENT_DATE) - INTERVAL '30 days') as inst_rate,

            -- 3. Major shareholder ratio (kr_largest_shareholder)
            (SELECT trmend_posesn_stock_qota_rt
             FROM kr_largest_shareholder
             WHERE symbol = $1
             ORDER BY stlm_dt DESC
             LIMIT 1) as major_shareholder_ratio,

            -- 4. Payout ratio (kr_dividends)
            (SELECT thstrm
             FROM kr_dividends
             WHERE symbol = $1
               AND se = '(연결)현금배당성향(%)'
             ORDER BY stlm_dt DESC
             LIMIT 1) as payout_ratio,

            -- 5. Fallback foreign rate (kr_individual_investor_daily_trading)
            (SELECT AVG(foreign_buy_ratio)
             FROM kr_individual_investor_daily_trading
             WHERE symbol = $1
               AND date >= COALESCE($2::date, CURRENT_DATE) - INTERVAL '30 days') as alt_foreign_rate
        """

        # Execute single combined query (기존 4-5개 쿼리 → 1개로 병합)
        combined_result = await self.execute_query(combined_query, self.symbol, self.analysis_date)

        # Process results
        foreign_rate = 0.0
        inst_rate = 0.0
        major_shareholder_ratio = 0.0
        payout_ratio = 0.0

        if combined_result and len(combined_result) > 0:
            row = combined_result[0]

            if row.get('foreign_holding_rate'):
                foreign_rate = float(row['foreign_holding_rate'])

            if row.get('inst_rate'):
                inst_rate = float(row['inst_rate'])

            if row.get('major_shareholder_ratio'):
                major_shareholder_ratio = float(row['major_shareholder_ratio'])

            if row.get('payout_ratio'):
                payout_ratio = float(row['payout_ratio'])

            # Fallback: If no foreign ownership data, use alt_foreign_rate
            if foreign_rate == 0 and row.get('alt_foreign_rate'):
                foreign_rate = float(row['alt_foreign_rate'])

        # Step 4: Calculate component scores

        # Institutional Quality Score (40%)
        # Foreign + Inst combined: 0% = 0, 30% = 70, 50%+ = 100
        combined_inst = foreign_rate + inst_rate
        if combined_inst >= 50:
            inst_score = 100
        elif combined_inst <= 0:
            inst_score = 0
        else:
            inst_score = min(100, combined_inst * 2)

        # Shareholder Return Commitment Score (35%)
        # Payout ratio: 0% = 0, 30% = 70, 50%+ = 100 (但 >100% = 경고)
        if payout_ratio >= 100:
            payout_score = 60  # Unsustainable payout
        elif payout_ratio >= 50:
            payout_score = 100
        elif payout_ratio >= 30:
            payout_score = 70 + (payout_ratio - 30) * 1.5
        elif payout_ratio > 0:
            payout_score = payout_ratio * 2.33
        else:
            payout_score = 0

        # Ownership Stability Score (25%)
        # Major shareholder: 30-50% = best, <20% or >70% = concern
        if 30 <= major_shareholder_ratio <= 50:
            stability_score = 100  # Balanced control
        elif 20 <= major_shareholder_ratio < 30:
            stability_score = 80  # Slightly low
        elif 50 < major_shareholder_ratio <= 60:
            stability_score = 80  # Slightly high
        elif 10 <= major_shareholder_ratio < 20:
            stability_score = 60  # Low - takeover risk
        elif 60 < major_shareholder_ratio <= 70:
            stability_score = 60  # High - minority squeeze risk
        elif major_shareholder_ratio < 10:
            stability_score = 40  # Very low
        else:  # > 70%
            stability_score = 40  # Very high - poor governance

        # Base score
        score = inst_score * 0.40 + payout_score * 0.35 + stability_score * 0.25

        logger.info(f"V30: Governance (Foreign+Inst: {combined_inst:.1f}%, Payout: {payout_ratio:.1f}%, Major: {major_shareholder_ratio:.1f}%, Score: {score:.1f})")

        return min(100, max(0, score))


    # ========================================================================
    # Calculate All Value Factor Scores
    # ========================================================================

    async def calculate_all_strategies(self):
        """
        Calculate all value factor strategies (Phase 3.8 Upgrade)
        Returns: dict of {strategy_name: score}

        Active strategies (15 total - Phase 3.8):
        - V2 (Magic Formula), V3 (Net Cash Flow Yield), V4 (Sustainable Dividend)
        - V13 (Magic Formula), V14 (Dividend Growth)
        - V21~V26 (Korean-style Value strategies)
        - V27~V30 (NEW: Shareholder Value Paradigm - Phase 3.8)

        Deprecated (negative IC): V1, V5~V12, V15~V20
        """
        logger.info(f"Calculating all value factor strategies for {self.symbol}")

        # Phase 3.8: 15 active strategies (positive/neutral IC)
        # Deprecated (negative IC): V1, V5~V12, V15~V20
        strategies = {
            # Existing strategies with positive/neutral IC
            'V2_Magic_Formula': await self.calculate_v2(),
            'V3_Cash_Flow_Sustainability': await self.calculate_v3(),
            'V4_Sustainable_Dividend': await self.calculate_v4(),
            'V13_Magic_Formula_Enhanced': await self.calculate_v13(),
            'V14_Dividend_Growth': await self.calculate_v14(),
            # Korean-style value strategies (Phase 3.7)
            'V21_Korea_Adjusted_PBR': await self.calculate_v21_korea_adjusted_pbr(),
            'V22_Quality_Dividend': await self.calculate_v22_quality_dividend(),
            'V23_Asset_Growth_Value': await self.calculate_v23_asset_growth_value(),
            'V24_Operating_Leverage': await self.calculate_v24_operating_leverage(),
            'V25_Cash_Rich_Undervalued': await self.calculate_v25_cash_rich_undervalued(),
            'V26_Smart_Money_Value': await self.calculate_v26_smart_money_value(),
            # NEW: Shareholder Value Paradigm (Phase 3.8)
            'V27_Total_Shareholder_Return': await self.calculate_v27_total_shareholder_return(),
            'V28_Value_Trap_Filter': await self.calculate_v28_value_trap_filter(),
            'V29_Accounting_Freshness': await self.calculate_v29_accounting_freshness(),
            'V30_Governance_Quality': await self.calculate_v30_governance_quality(),
        }

        self.strategies_scores = strategies

        return strategies

    async def calculate_comprehensive_score(self):
        """
        Calculate comprehensive value factor score

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
            Weights: V1=0.3, V2=0.5, V3=1.2, ..., V16=0.1 (Sum=9.4)
            Scores: V1=85, V2=72, V3=65, ..., V16=68
            Weighted Average = (85×0.3 + 72×0.5 + ... + 68×0.1) / 9.4 = 72.2

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
        weights = VALUE_STRATEGY_WEIGHTS.get(market_state)

        if weights is None:
            logger.error(f"Invalid market state: {market_state}, using '기타'")
            weights = VALUE_STRATEGY_WEIGHTS['기타']
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

                # Extract strategy key (e.g., 'V1_Low_PER' -> 'V1')
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

            logger.info(f"Weighted score for {self.symbol}: {final_score:.2f} "
                       f"(market_state: {market_state}, weight_sum: {weight_sum:.2f})")

            return result
        else:
            logger.warning(f"No valid scores for weighted calculation: {self.symbol}")
            return None


async def main():
    """Main execution function"""
    print("\n" + "="*80)
    print("Korean Stock Value Factor System (Enhanced with Market State Weighting)")
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
        print("\nStep 2: Calculating value factor strategies...")
        from db_async import AsyncDatabaseManager
        db_manager = AsyncDatabaseManager()
        await db_manager.initialize()

        calculator = ValueFactorCalculator(symbol, db_manager, market_state=market_state)

        # Calculate all strategies
        strategies = await calculator.calculate_all_strategies()

        # Get weights for current market state
        weights = VALUE_STRATEGY_WEIGHTS.get(market_state, VALUE_STRATEGY_WEIGHTS['기타'])

        # Display results
        print("\n" + "="*80)
        print(f"Value Factor Strategies for {symbol}")
        print(f"Market State: {market_state}")
        print("="*80 + "\n")

        print(f"{'Strategy':<30s} {'Score':>8s} {'Weight':>8s} {'Weighted':>10s}")
        print("-"*80)

        for strategy_name, score in strategies.items():
            strategy_key = strategy_name.split('_')[0]
            weight = weights.get(strategy_key, 1.0)

            if score is not None:
                weighted = score * weight
                print(f"{strategy_name:<30s} {score:8.2f} {weight:8.2f} {weighted:10.2f}")
            else:
                print(f"{strategy_name:<30s} {'N/A':>8s} {weight:8.2f} {'N/A':>10s}")

        # Calculate scores
        comprehensive_score = await calculator.calculate_comprehensive_score()
        weighted_result = await calculator.calculate_weighted_score()

        # Display final scores
        print("\n" + "="*80)
        print("Final Scores")
        print("="*80)
        print(f"Simple Sum (unweighted):        {comprehensive_score:.2f} points")

        if weighted_result:
            print(f"\n[Market State-based Weighted Score]")
            print(f"Market State:                   {weighted_result['market_state']}")
            print(f"Weight Sum:                     {weighted_result['weight_sum']:.2f}")
            print(f"Valid Strategies:               {weighted_result['valid_strategies']}/16")
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
