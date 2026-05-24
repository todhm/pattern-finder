"""
Batch Weight Calculator for Korean Stock Quant Analysis
Optimized for processing 2800+ stocks with minimal database queries

Key optimization:
- Phase 1: Calculate common data once (economic cycle, market sentiment, sector cycle, scenario stats)
- Phase 2: Bulk fetch all stock-specific data (4-5 queries for all stocks)
- Phase 3: Calculate weights using memory operations only (no DB queries)

Performance: 36,400 queries -> ~16 queries (99.96% reduction)
Scenario stats: 2,763 queries -> 1 query (99.96% reduction)
"""

import os
import logging
import asyncio
import asyncpg
from dotenv import load_dotenv
from datetime import datetime
from decimal import Decimal
import json
import time
import psutil
import traceback

# Import from existing weight.py
import sys
sys.path.append(os.path.dirname(__file__))
from weight import WeightCalculator

# Import market classifier
try:
    from market_classifier import MarketClassifier
except ImportError:
    from kr.market_classifier import MarketClassifier

# Load environment variables
load_dotenv()

# ===========================
# Enhanced Logging Configuration
# ===========================

def setup_logging():
    r"""
    Setup enhanced logging with file and console handlers
    Creates detailed log files in C:\project\alpha\quant\log\
    """
    # Create log directory
    log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'log')
    os.makedirs(log_dir, exist_ok=True)

    # Generate log filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'batch_weight_{timestamp}.log')

    # Create logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)  # Capture all levels

    # Remove existing handlers
    logger.handlers.clear()

    # Console handler (INFO level)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)

    # File handler (DEBUG level - detailed)
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s'
    )
    file_handler.setFormatter(file_formatter)

    # Add handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    logger.info(f"Logging initialized: {log_file}")

    return logger, log_file

# Initialize logging
logger, CURRENT_LOG_FILE = setup_logging()


class AsyncBatchWeightCalculator:
    """Calculate weights for all stocks with optimized batch processing (Async)"""

    def __init__(self):
        self.pool = None
        self.common_data = {
            'economic_cycle': None,
            'market_sentiment': {},  # {exchange: data}
            'sector_cycle': {}       # {industry: data}
        }
        # Performance tracking
        self.performance_data = {
            'queries': [],  # List of query execution details
            'phases': {},   # Phase timings
            'memory_snapshots': [],  # Memory usage over time
            'start_time': None,
            'end_time': None
        }

    async def initialize(self, min_size=2, max_size=10):
        """Initialize async database connection pool"""
        try:
            database_url = os.getenv('DATABASE_URL')
            if not database_url:
                raise ValueError("DATABASE_URL not found in environment variables")

            # Convert to asyncpg format if needed
            db_url = database_url.replace("postgresql+asyncpg://", "postgresql://")

            self.pool = await asyncpg.create_pool(
                db_url,
                min_size=min_size,
                max_size=max_size,
                command_timeout=600,
                max_inactive_connection_lifetime=3600,
                server_settings={
                    'statement_timeout': '600000',
                    'tcp_keepalives_idle': '30',
                    'tcp_keepalives_interval': '10',
                    'tcp_keepalives_count': '5'
                }
            )

            logger.info(f"Database connection pool initialized: min={min_size}, max={max_size}")

        except Exception as e:
            logger.error(f"Failed to initialize database connection pool: {e}")
            raise

    async def execute_query(self, query, params=None, query_name="Unnamed Query"):
        """
        Execute SQL query and return results with performance tracking

        Args:
            query: SQL query string
            params: Query parameters (tuple or list)
            query_name: Descriptive name for logging

        Returns:
            List of dict rows
        """
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

        try:
            async with self.pool.acquire() as conn:
                if params:
                    result = await conn.fetch(query, *params)
                else:
                    result = await conn.fetch(query)

                # Convert asyncpg.Record to dict
                result = [dict(row) for row in result]

                # Record performance metrics
                end_time = time.time()
                end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                execution_time = end_time - start_time
                memory_delta = end_memory - start_memory
                row_count = len(result)

                # Store query performance data
                query_info = {
                    'name': query_name,
                    'execution_time': round(execution_time, 3),
                    'row_count': row_count,
                    'memory_start_mb': round(start_memory, 2),
                    'memory_end_mb': round(end_memory, 2),
                    'memory_delta_mb': round(memory_delta, 2),
                    'timestamp': datetime.now().isoformat(),
                    'query_preview': query[:200] + '...' if len(query) > 200 else query
                }
                self.performance_data['queries'].append(query_info)

                # Log query performance
                logger.debug(
                    f"Query: {query_name} | "
                    f"Time: {execution_time:.3f}s | "
                    f"Rows: {row_count:,} | "
                    f"Memory: {memory_delta:+.2f}MB"
                )

                return result

        except Exception as e:
            logger.error(f"Query execution failed: {query_name}")
            logger.error(f"Error: {e}")
            logger.error(f"Query preview: {query[:500]}")
            logger.debug(f"Full query: {query}")
            logger.debug(f"Parameters: {params}")
            raise

    # ========================================================================
    # PHASE 1: Calculate Common Data (Economic Cycle, Market Sentiment, Sector Cycle)
    # ========================================================================

    async def check_gdp_trend(self):
        """Check GDP growth trend for economic cycle analysis"""
        query = """
        WITH gdp_data AS (
            SELECT
                time_value,
                data_value as gdp_growth,
                LAG(data_value, 1) OVER (ORDER BY time_value) as prev_gdp,
                LAG(data_value, 2) OVER (ORDER BY time_value) as prev2_gdp
            FROM bok_economic_indicators
            WHERE stat_code = '200Y102'
                AND cycle = 'Q'
            ORDER BY time_value DESC
            LIMIT 3
        )
        SELECT gdp_growth, prev_gdp, prev2_gdp
        FROM gdp_data
        LIMIT 1
        """

        result = await self.execute_query(query)
        if not result or result[0]['gdp_growth'] is None:
            return 'NEUTRAL'

        current = float(result[0]['gdp_growth'])
        prev = float(result[0]['prev_gdp']) if result[0]['prev_gdp'] else None
        prev2 = float(result[0]['prev2_gdp']) if result[0]['prev2_gdp'] else None

        if prev is None or prev2 is None:
            return 'NEUTRAL'

        if current > prev and prev > prev2:
            return 'UP'
        elif current < prev and prev < prev2:
            return 'DOWN'
        else:
            return 'NEUTRAL'

    async def check_interest_rate_trend(self):
        """Check interest rate trend (3-month comparison)"""
        query = """
        SELECT
            data_value,
            time_value
        FROM bok_economic_indicators
        WHERE stat_code = '722Y001'
            AND cycle = 'D'
        ORDER BY time_value DESC
        LIMIT 90
        """

        result = await self.execute_query(query)
        if not result or len(result) < 60:
            return 'STABLE'

        current_rate = float(result[0]['data_value'])
        rate_3m_ago = float(result[60]['data_value'])

        rate_change = current_rate - rate_3m_ago

        if rate_change > 0.1:
            return 'TIGHTENING'
        elif rate_change < -0.1:
            return 'EASING'
        else:
            return 'STABLE'

    async def check_exchange_rate_trend(self):
        """Check KRW/USD exchange rate trend (1-month comparison)"""
        query = """
        SELECT
            data_value,
            time_value
        FROM exchange_rate
        WHERE stat_code = '731Y001'
            AND item_code1 = '0000001'
            AND cycle = 'D'
        ORDER BY time_value DESC
        LIMIT 30
        """

        result = await self.execute_query(query)
        if not result or len(result) < 20:
            return 'STABLE'

        current_rate = float(result[0]['data_value'])
        rate_1m_ago = float(result[20]['data_value'])

        rate_change_pct = (current_rate - rate_1m_ago) / rate_1m_ago * 100

        if rate_change_pct > 2:
            return 'WEAK'
        elif rate_change_pct < -2:
            return 'STRONG'
        else:
            return 'STABLE'

    async def check_cpi_trend(self):
        """Check CPI (Consumer Price Index) trend (3-month comparison)"""
        query = """
        SELECT
            data_value,
            time_value
        FROM bok_economic_indicators
        WHERE stat_code = '901Y009'
            AND cycle = 'M'
        ORDER BY time_value DESC
        LIMIT 4
        """

        result = await self.execute_query(query)
        if not result or len(result) < 4:
            return 'STABLE'

        current = float(result[0]['data_value'])
        prev1 = float(result[1]['data_value'])
        prev2 = float(result[2]['data_value'])
        prev3 = float(result[3]['data_value'])

        avg_change = ((current - prev1) + (prev1 - prev2) + (prev2 - prev3)) / 3

        if avg_change > 0.3:
            return 'INFLATION'
        elif avg_change < -0.1:
            return 'DEFLATION'
        else:
            return 'STABLE'

    async def check_sentiment_index(self):
        """Check economic sentiment index"""
        query = """
        SELECT
            data_value
        FROM bok_economic_indicators
        WHERE stat_code = '513Y001'
            AND item_name1 LIKE '%경제심리지수%'
            AND cycle = 'M'
        ORDER BY time_value DESC
        LIMIT 3
        """

        result = await self.execute_query(query)
        if not result or len(result) < 2:
            return 'NEUTRAL'

        avg_sentiment = sum(float(row['data_value']) for row in result) / len(result)

        if avg_sentiment > 102:
            return 'OPTIMISTIC'
        elif avg_sentiment < 98:
            return 'PESSIMISTIC'
        else:
            return 'NEUTRAL'

    def calculate_similarity_score(self, current_pattern, defined_pattern):
        """Calculate similarity score between two patterns using weighted matching"""
        WEIGHTS = {
            'gdp': 0.40,
            'rate': 0.20,
            'sentiment': 0.20,
            'exchange': 0.10,
            'cpi': 0.10
        }

        score = 0.0

        # GDP comparison (index 0)
        if current_pattern[0] == defined_pattern[0]:
            score += WEIGHTS['gdp']
        elif self._is_similar_direction(current_pattern[0], defined_pattern[0]):
            score += WEIGHTS['gdp'] * 0.5

        # Interest rate comparison (index 1)
        if current_pattern[1] == defined_pattern[1]:
            score += WEIGHTS['rate']
        elif self._is_similar_policy(current_pattern[1], defined_pattern[1]):
            score += WEIGHTS['rate'] * 0.5

        # Sentiment comparison (index 2)
        if current_pattern[2] == defined_pattern[2]:
            score += WEIGHTS['sentiment']
        elif current_pattern[2] == 'NEUTRAL' or defined_pattern[2] == 'NEUTRAL':
            score += WEIGHTS['sentiment'] * 0.3

        # Exchange rate comparison (index 3)
        if current_pattern[3] == defined_pattern[3]:
            score += WEIGHTS['exchange']
        elif current_pattern[3] == 'STABLE' or defined_pattern[3] == 'STABLE':
            score += WEIGHTS['exchange'] * 0.5

        # CPI comparison (index 4)
        if current_pattern[4] == defined_pattern[4]:
            score += WEIGHTS['cpi']
        elif current_pattern[4] == 'STABLE' or defined_pattern[4] == 'STABLE':
            score += WEIGHTS['cpi'] * 0.5

        return score

    def _is_similar_direction(self, val1, val2):
        """Check if two GDP trends are similar direction"""
        positive = ['UP']
        negative = ['DOWN']

        if val1 in positive and val2 in positive:
            return True
        if val1 in negative and val2 in negative:
            return True
        return False

    def _is_similar_policy(self, val1, val2):
        """Check if two interest rate policies are similar"""
        loose = ['EASING', 'STABLE']
        tight = ['TIGHTENING', 'STABLE']

        if val1 in loose and val2 in loose:
            return True
        if val1 in tight and val2 in tight:
            return True
        return False

    def match_cycle_matrix(self, gdp, rate, sentiment, exchange, cpi):
        """Match indicators to economic cycle using 32-pattern matrix with similarity scoring"""

        # 32-pattern cycle matrix (same as weight.py)
        CYCLE_MATRIX = {
            # === EXPANSION patterns (8) ===
            ('UP', 'TIGHTENING', 'OPTIMISTIC', 'STRONG', 'INFLATION'): 'EXPANSION',
            ('UP', 'TIGHTENING', 'OPTIMISTIC', 'STABLE', 'INFLATION'): 'EXPANSION',
            ('UP', 'STABLE', 'OPTIMISTIC', 'STABLE', 'INFLATION'): 'EXPANSION',
            ('UP', 'TIGHTENING', 'OPTIMISTIC', 'STABLE', 'STABLE'): 'EXPANSION',
            ('UP', 'STABLE', 'OPTIMISTIC', 'STABLE', 'STABLE'): 'EXPANSION',
            ('UP', 'STABLE', 'OPTIMISTIC', 'STRONG', 'STABLE'): 'EXPANSION',
            ('UP', 'TIGHTENING', 'NEUTRAL', 'STABLE', 'STABLE'): 'EXPANSION',
            ('UP', 'STABLE', 'NEUTRAL', 'STABLE', 'STABLE'): 'EXPANSION',

            # === RECOVERY patterns (8) ===
            ('UP', 'EASING', 'OPTIMISTIC', 'WEAK', 'STABLE'): 'RECOVERY',
            ('UP', 'EASING', 'OPTIMISTIC', 'STABLE', 'STABLE'): 'RECOVERY',
            ('UP', 'EASING', 'NEUTRAL', 'STABLE', 'STABLE'): 'RECOVERY',
            ('UP', 'EASING', 'NEUTRAL', 'WEAK', 'STABLE'): 'RECOVERY',
            ('NEUTRAL', 'EASING', 'OPTIMISTIC', 'WEAK', 'DEFLATION'): 'RECOVERY',
            ('NEUTRAL', 'EASING', 'OPTIMISTIC', 'STABLE', 'STABLE'): 'RECOVERY',
            ('NEUTRAL', 'EASING', 'NEUTRAL', 'STABLE', 'STABLE'): 'RECOVERY',
            ('DOWN', 'EASING', 'OPTIMISTIC', 'WEAK', 'STABLE'): 'RECOVERY',

            # === SLOWDOWN patterns (10) ===
            ('DOWN', 'TIGHTENING', 'PESSIMISTIC', 'WEAK', 'INFLATION'): 'SLOWDOWN',
            ('DOWN', 'TIGHTENING', 'NEUTRAL', 'WEAK', 'INFLATION'): 'SLOWDOWN',
            ('NEUTRAL', 'TIGHTENING', 'PESSIMISTIC', 'WEAK', 'INFLATION'): 'SLOWDOWN',
            ('DOWN', 'TIGHTENING', 'NEUTRAL', 'WEAK', 'STABLE'): 'SLOWDOWN',
            ('DOWN', 'TIGHTENING', 'PESSIMISTIC', 'STABLE', 'STABLE'): 'SLOWDOWN',
            ('NEUTRAL', 'TIGHTENING', 'PESSIMISTIC', 'STABLE', 'STABLE'): 'SLOWDOWN',
            ('DOWN', 'STABLE', 'PESSIMISTIC', 'WEAK', 'STABLE'): 'SLOWDOWN',
            ('DOWN', 'STABLE', 'PESSIMISTIC', 'STABLE', 'STABLE'): 'SLOWDOWN',
            ('DOWN', 'STABLE', 'NEUTRAL', 'WEAK', 'STABLE'): 'SLOWDOWN',
            ('NEUTRAL', 'STABLE', 'PESSIMISTIC', 'WEAK', 'STABLE'): 'SLOWDOWN',

            # === RECESSION patterns (4) ===
            ('DOWN', 'EASING', 'PESSIMISTIC', 'WEAK', 'DEFLATION'): 'RECESSION',
            ('DOWN', 'EASING', 'PESSIMISTIC', 'STABLE', 'DEFLATION'): 'RECESSION',
            ('DOWN', 'STABLE', 'PESSIMISTIC', 'WEAK', 'DEFLATION'): 'RECESSION',
            ('DOWN', 'STABLE', 'PESSIMISTIC', 'STABLE', 'DEFLATION'): 'RECESSION',

            # === NEUTRAL/transition patterns (2) ===
            ('NEUTRAL', 'STABLE', 'NEUTRAL', 'STABLE', 'STABLE'): 'NEUTRAL',
            ('DOWN', 'STABLE', 'NEUTRAL', 'STABLE', 'STABLE'): 'NEUTRAL',
        }

        current_pattern = (gdp, rate, sentiment, exchange, cpi)

        # Step 1: Try exact match
        if current_pattern in CYCLE_MATRIX:
            confidence = 'VERY_HIGH'
            similarity = 1.0
            logger.info(f"Exact match found for pattern {current_pattern}")
            return CYCLE_MATRIX[current_pattern], confidence, similarity

        # Step 2: Find most similar pattern using weighted scoring
        best_cycle = None
        best_score = 0.0
        best_pattern = None

        for pattern, cycle in CYCLE_MATRIX.items():
            score = self.calculate_similarity_score(current_pattern, pattern)
            if score > best_score:
                best_score = score
                best_cycle = cycle
                best_pattern = pattern

        # Step 3: Determine confidence level based on similarity score
        if best_score >= 0.8:
            confidence = 'HIGH'
        elif best_score >= 0.6:
            confidence = 'MEDIUM'
        elif best_score >= 0.4:
            confidence = 'LOW'
        else:
            confidence = 'VERY_LOW'

        logger.info(f"Best match: {best_pattern} -> {best_cycle} (similarity: {best_score:.2f}, confidence: {confidence})")

        # Step 4: If confidence is too low, use GDP-based fallback
        if best_score < 0.4:
            logger.warning(f"Low confidence ({best_score:.2f}), using GDP-based fallback")
            if gdp == 'UP':
                if rate == 'EASING' or sentiment == 'OPTIMISTIC':
                    return 'RECOVERY', confidence, best_score
                else:
                    return 'EXPANSION', confidence, best_score
            elif gdp == 'DOWN':
                if rate == 'TIGHTENING' or sentiment == 'PESSIMISTIC':
                    return 'SLOWDOWN', confidence, best_score
                else:
                    return 'RECESSION', confidence, best_score
            else:
                return 'NEUTRAL', confidence, best_score

        return best_cycle, confidence, best_score

    async def calculate_economic_cycle_data(self):
        """Phase 1-1: Calculate economic cycle once for all stocks (5 queries)"""
        logger.info("Phase 1-1: Calculating economic cycle data (common for all stocks)...")

        # Check each economic indicator
        gdp_trend = await self.check_gdp_trend()
        rate_trend = await self.check_interest_rate_trend()
        sentiment = await self.check_sentiment_index()
        exchange_trend = await self.check_exchange_rate_trend()
        cpi_trend = await self.check_cpi_trend()

        # Match patterns to cycle
        cycle_phase, confidence, similarity = self.match_cycle_matrix(
            gdp_trend, rate_trend, sentiment, exchange_trend, cpi_trend
        )

        logger.info(f"Economic cycle determined: {cycle_phase}")
        logger.info(f"Indicators: GDP={gdp_trend}, Rate={rate_trend}, Sentiment={sentiment}, "
                   f"Exchange={exchange_trend}, CPI={cpi_trend}")

        # Store in common data
        self.common_data['economic_cycle'] = {
            'cycle': cycle_phase,
            'gdp_trend': gdp_trend,
            'interest_rate_trend': rate_trend,
            'sentiment_index': sentiment,
            'exchange_rate_trend': exchange_trend,
            'cpi_trend': cpi_trend,
            'confidence': confidence,
            'similarity': similarity
        }

    async def calculate_market_sentiment_data(self):
        """Phase 1-2: Calculate market sentiment for KOSPI and KOSDAQ (2 queries)"""
        logger.info("Phase 1-2: Calculating market sentiment data (by exchange)...")

        for exchange in ['KOSPI', 'KOSDAQ']:
            query = """
            WITH market_data AS (
                SELECT
                    close,
                    LAG(close, 20) OVER (ORDER BY date) as close_20d_ago,
                    LAG(close, 60) OVER (ORDER BY date) as close_60d_ago
                FROM market_index
                WHERE exchange = $1
                ORDER BY date DESC
                LIMIT 1
            ),
            sentiment_indicators AS (
                SELECT
                    AVG(CASE WHEN net_buy_value > 0 THEN 1 ELSE 0 END) as foreign_buy_ratio,
                    AVG(ABS(net_buy_value)) as avg_trading_intensity
                FROM kr_investor_daily_trading
                WHERE investor_type = '외국인'
                    AND date >= CURRENT_DATE - INTERVAL '20 days'
            )
            SELECT
                m.*,
                s.*,
                CASE
                    WHEN (m.close - m.close_20d_ago) / m.close_20d_ago > 0.1 THEN 'OVERHEATED'
                    WHEN (m.close - m.close_60d_ago) / m.close_60d_ago < -0.2 THEN 'PANIC'
                    WHEN s.foreign_buy_ratio > 0.7 THEN 'GREED'
                    WHEN s.foreign_buy_ratio < 0.3 THEN 'FEAR'
                    ELSE 'NEUTRAL'
                END as market_sentiment
            FROM market_data m, sentiment_indicators s
            """

            result = await self.execute_query(query, (exchange,))

            if result and result[0]['market_sentiment']:
                sentiment = result[0]['market_sentiment']
                self.common_data['market_sentiment'][exchange] = sentiment
                logger.info(f"{exchange} market sentiment: {sentiment}")
            else:
                self.common_data['market_sentiment'][exchange] = 'NEUTRAL'
                logger.warning(f"No market sentiment data for {exchange}, using NEUTRAL")

    async def calculate_sector_cycle_data(self):
        """Phase 1-3: Calculate sector cycle for all industries at once (1 query)"""
        logger.info("Phase 1-3: Calculating sector cycle data for all industries...")

        # Modified query: removed WHERE symbol = %s condition to get all industries
        query = """
        WITH stock_returns AS (
            SELECT
                d.industry,
                i.symbol,
                i.date,
                i.close,
                LAG(i.close, 20) OVER (PARTITION BY i.symbol ORDER BY i.date) as close_20d_ago,
                i.trading_value
            FROM kr_stock_detail d
            JOIN kr_intraday_total i ON d.symbol = i.symbol
            WHERE i.date >= CURRENT_DATE - INTERVAL '40 days'
                AND d.industry IS NOT NULL
        ),
        sector_performance AS (
            SELECT
                industry,
                AVG((close - close_20d_ago) / NULLIF(close_20d_ago, 0)) as avg_return_20d,
                AVG(trading_value) as avg_trading_value,
                COUNT(DISTINCT symbol) as stock_count
            FROM stock_returns
            WHERE close_20d_ago IS NOT NULL
            GROUP BY industry
        )
        SELECT
            industry,
            avg_return_20d,
            avg_trading_value,
            stock_count,
            CASE
                WHEN avg_return_20d > 0.1 THEN 'HOT'
                WHEN avg_return_20d > 0.03 THEN 'GROWING'
                WHEN avg_return_20d > -0.03 THEN 'STABLE'
                WHEN avg_return_20d > -0.1 THEN 'DECLINING'
                ELSE 'COLD'
            END as sector_status
        FROM sector_performance
        """

        result = await self.execute_query(query)

        for row in result:
            industry = row['industry']
            sector_status = row['sector_status']
            self.common_data['sector_cycle'][industry] = sector_status

        logger.info(f"Sector cycle calculated for {len(self.common_data['sector_cycle'])} industries")

    async def precompute_scenario_stats(self):
        """
        Phase 1-4: Pre-compute scenario statistics for all (industry, score_bucket) combinations

        Optimization:
        - Before: 2,763 queries (1 per stock) with expensive LATERAL JOIN
        - After: 1 query for all combinations (~330 rows)
        - Expected improvement: 99.96% query reduction

        Key: (industry, score_bucket) where score_bucket = floor(final_score / 10) * 10
        Value: {bullish_count, sideways_count, bearish_count, total_count,
                bullish_avg, sideways_avg, bearish_avg,
                bullish_lower, bullish_upper, bearish_lower, bearish_upper,
                sideways_lower, sideways_upper}
        """
        logger.info("Phase 1-4: Pre-computing scenario statistics for all (industry, score_bucket) combinations...")

        query = """
        WITH similar_cases AS (
            SELECT
                d.industry,
                FLOOR(g.final_score / 10) * 10 as score_bucket,
                g.symbol,
                g.date as analysis_date,
                g.final_score,
                current_price.close as current_close,
                future_price.close as future_close,
                CASE
                    WHEN future_price.close IS NOT NULL AND current_price.close > 0
                    THEN (future_price.close - current_price.close) / current_price.close * 100
                    ELSE NULL
                END as return_3m
            FROM kr_stock_grade g
            JOIN kr_stock_detail d ON g.symbol = d.symbol
            JOIN kr_intraday_total current_price
                ON g.symbol = current_price.symbol AND g.date = current_price.date
            LEFT JOIN LATERAL (
                SELECT close
                FROM kr_intraday_total
                WHERE symbol = g.symbol
                    AND date >= g.date + INTERVAL '60 days'
                    AND date <= g.date + INTERVAL '66 days'
                ORDER BY date
                LIMIT 1
            ) future_price ON true
            WHERE g.date < CURRENT_DATE - INTERVAL '63 days'
                AND future_price.close IS NOT NULL
                AND d.industry IS NOT NULL
        )
        SELECT
            industry,
            score_bucket,
            COUNT(*) as total_count,
            COUNT(*) FILTER (WHERE return_3m > 10) as bullish_count,
            COUNT(*) FILTER (WHERE return_3m BETWEEN -10 AND 10) as sideways_count,
            COUNT(*) FILTER (WHERE return_3m < -10) as bearish_count,
            ROUND(AVG(return_3m) FILTER (WHERE return_3m > 10)::numeric, 1) as bullish_avg,
            ROUND(AVG(return_3m) FILTER (WHERE return_3m BETWEEN -10 AND 10)::numeric, 1) as sideways_avg,
            ROUND(AVG(return_3m) FILTER (WHERE return_3m < -10)::numeric, 1) as bearish_avg,
            ROUND(PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY return_3m)
                FILTER (WHERE return_3m > 10)::numeric, 1) as bullish_lower,
            ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY return_3m)
                FILTER (WHERE return_3m > 10)::numeric, 1) as bullish_upper,
            ROUND(PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY return_3m)
                FILTER (WHERE return_3m < -10)::numeric, 1) as bearish_lower,
            ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY return_3m)
                FILTER (WHERE return_3m < -10)::numeric, 1) as bearish_upper,
            ROUND(PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY return_3m)
                FILTER (WHERE return_3m BETWEEN -10 AND 10)::numeric, 1) as sideways_lower,
            ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY return_3m)
                FILTER (WHERE return_3m BETWEEN -10 AND 10)::numeric, 1) as sideways_upper
        FROM similar_cases
        GROUP BY industry, score_bucket
        HAVING COUNT(*) >= 5
        ORDER BY industry, score_bucket
        """

        result = await self.execute_query(query)

        # Store as nested dict: {industry: {score_bucket: stats}}
        scenario_stats = {}
        for row in result:
            industry = row['industry']
            score_bucket = int(row['score_bucket']) if row['score_bucket'] is not None else 50

            if industry not in scenario_stats:
                scenario_stats[industry] = {}

            scenario_stats[industry][score_bucket] = {
                'total_count': row['total_count'],
                'bullish_count': row['bullish_count'] or 0,
                'sideways_count': row['sideways_count'] or 0,
                'bearish_count': row['bearish_count'] or 0,
                'bullish_avg': float(row['bullish_avg']) if row['bullish_avg'] else None,
                'sideways_avg': float(row['sideways_avg']) if row['sideways_avg'] else None,
                'bearish_avg': float(row['bearish_avg']) if row['bearish_avg'] else None,
                'bullish_lower': float(row['bullish_lower']) if row['bullish_lower'] else 10,
                'bullish_upper': float(row['bullish_upper']) if row['bullish_upper'] else 25,
                'bearish_lower': float(row['bearish_lower']) if row['bearish_lower'] else -20,
                'bearish_upper': float(row['bearish_upper']) if row['bearish_upper'] else -10,
                'sideways_lower': float(row['sideways_lower']) if row['sideways_lower'] else -5,
                'sideways_upper': float(row['sideways_upper']) if row['sideways_upper'] else 5
            }

        self.common_data['scenario_stats'] = scenario_stats

        total_combinations = sum(len(buckets) for buckets in scenario_stats.values())
        logger.info(f"Scenario stats pre-computed: {len(scenario_stats)} industries, {total_combinations} combinations")

    def get_scenario_stats(self):
        """Get pre-computed scenario statistics (for use after process_all_stocks)"""
        return self.common_data.get('scenario_stats', {})

    # ========================================================================
    # PHASE 2: Bulk Fetch All Stock Data (4-5 queries for all stocks)
    # ========================================================================

    async def fetch_all_stock_data(self, symbols):
        """
        Phase 2: Fetch all stock data in minimal queries (OPTIMIZED)

        Optimization: Query 2, 3, 4 merged into 1 CTE query
        - Before: 4 separate queries (3x kr_intraday_total scans)
        - After: 2 queries (1x kr_intraday_total scan)
        - Expected improvement: 30-40% faster

        Returns: {symbol: {exchange, market_cap, industry, theme, ...}}
        """
        logger.info(f"Phase 2: Fetching data for {len(symbols)} stocks in bulk...")
        logger.debug(f"Number of symbols to fetch: {len(symbols)}")

        all_data = {}

        # Query 1: Basic info (exchange, industry, theme) from kr_stock_detail
        logger.info("Query 1/2: Fetching basic stock info...")
        query1 = """
        SELECT symbol, exchange, industry, theme
        FROM kr_stock_detail
        WHERE symbol = ANY($1)
        """
        result1 = await self.execute_query(query1, (symbols,), query_name="Query 1: Basic Info")

        for row in result1:
            symbol = row['symbol']
            all_data[symbol] = {
                'exchange': row['exchange'],
                'industry': row['industry'],
                'theme': row['theme']
            }

        # Query 2: OPTIMIZED - Combined query for market cap, liquidity, and volatility
        # This replaces the previous Query 2, 3, 4 with a single CTE query
        logger.info("Query 2/2: Fetching market data (optimized: combined latest/liquidity/volatility)...")
        query_combined = """
        WITH latest_data AS (
            SELECT DISTINCT ON (symbol)
                symbol,
                market_cap,
                close,
                date
            FROM kr_intraday_total
            WHERE symbol = ANY($1)
            ORDER BY symbol, date DESC
        ),
        liquidity_data AS (
            SELECT
                symbol,
                AVG(trading_value) as avg_trading_value
            FROM kr_intraday_total
            WHERE symbol = ANY($1)
              AND date >= CURRENT_DATE - INTERVAL '20 days'
            GROUP BY symbol
        ),
        volatility_data AS (
            SELECT
                symbol,
                STDDEV(log_return) * SQRT(252) as annual_volatility
            FROM (
                SELECT
                    symbol,
                    LN(close / NULLIF(LAG(close) OVER (PARTITION BY symbol ORDER BY date), 0)) as log_return
                FROM kr_intraday_total
                WHERE symbol = ANY($1)
                  AND date >= CURRENT_DATE - INTERVAL '60 days'
            ) daily_returns
            WHERE log_return IS NOT NULL
            GROUP BY symbol
        )
        SELECT
            l.symbol,
            l.market_cap,
            l.close,
            l.date,
            lq.avg_trading_value,
            v.annual_volatility
        FROM latest_data l
        LEFT JOIN liquidity_data lq ON l.symbol = lq.symbol
        LEFT JOIN volatility_data v ON l.symbol = v.symbol
        """

        result_combined = await self.execute_query(
            query_combined,
            (symbols,),
            query_name="Query 2: Combined Market Data (Latest + Liquidity + Volatility)"
        )

        for row in result_combined:
            symbol = row['symbol']
            if symbol in all_data:
                all_data[symbol].update({
                    'market_cap': float(row['market_cap']) if row['market_cap'] else None,
                    'close': float(row['close']) if row['close'] else None,
                    'latest_date': row['date'],
                    'avg_trading_value': float(row['avg_trading_value']) if row['avg_trading_value'] else None,
                    'annual_volatility': float(row['annual_volatility']) if row['annual_volatility'] else None
                })

        # Query 3: Supply/Demand data (Phase 3.10)
        logger.info("Query 3/3: Fetching supply/demand data (institutional + foreign flows)...")
        query_supply_demand = """
        SELECT
            symbol,
            SUM(inst_net_volume) as inst_net_30d,
            SUM(foreign_net_volume) as foreign_net_30d
        FROM kr_individual_investor_daily_trading
        WHERE symbol = ANY($1)
            AND date >= CURRENT_DATE - INTERVAL '30 days'
            AND date <= CURRENT_DATE
        GROUP BY symbol
        """

        result_supply = await self.execute_query(
            query_supply_demand,
            (symbols,),
            query_name="Query 3: Supply/Demand Data (30-day inst/foreign flows)"
        )

        for row in result_supply:
            symbol = row['symbol']
            if symbol in all_data:
                all_data[symbol].update({
                    'inst_net_30d': int(row['inst_net_30d']) if row['inst_net_30d'] else 0,
                    'foreign_net_30d': int(row['foreign_net_30d']) if row['foreign_net_30d'] else 0
                })

        logger.info(f"Bulk data fetch complete: {len(all_data)} stocks")
        logger.debug(f"Data completeness: {len(all_data)}/{len(symbols)} stocks have data")

        return all_data

    # ========================================================================
    # PHASE 3: Calculate Weights for Each Stock (No DB queries, memory only)
    # ========================================================================

    def get_market_type_weights(self, exchange):
        """Get market type weight adjustments"""
        market_weights = {
            'KOSPI': {
                'value': 0.8,  # Changed from 1.3 (Phase 3.7: KOSPI value weight reduction)
                'quality': 1.2,
                'momentum': 0.8,
                'growth': 0.7
            },
            'KOSDAQ': {
                'growth': 1.4,
                'momentum': 1.3,
                'value': 0.6,
                'quality': 0.7
            }
        }
        return market_weights.get(exchange, {'all': 1.0})

    def get_market_cap_weights(self, market_cap):
        """Get market cap weight adjustments"""
        if market_cap is None:
            return {'all': 1.0}

        if market_cap >= 10_000_000_000_000:  # 10 trillion won
            category = 'MEGA'
        elif market_cap >= 1_000_000_000_000:  # 1 trillion won
            category = 'LARGE'
        elif market_cap >= 200_000_000_000:  # 200 billion won
            category = 'MEDIUM'
        else:
            category = 'SMALL'

        cap_weights = {
            'MEGA': {
                'value': 1.2,
                'quality': 1.3,
                'momentum': 0.7,
                'growth': 0.8
            },
            'LARGE': {
                'value': 1.1,
                'quality': 1.2,
                'momentum': 0.9,
                'growth': 0.8
            },
            'MEDIUM': {
                'value': 0.9,
                'quality': 0.9,
                'momentum': 1.2,
                'growth': 1.0
            },
            'SMALL': {
                'value': 0.7,
                'quality': 0.6,
                'momentum': 1.5,
                'growth': 1.2
            }
        }
        return cap_weights.get(category, {'all': 1.0})

    def get_liquidity_weights(self, avg_trading_value):
        """Get liquidity weight adjustments"""
        if avg_trading_value is None:
            return {'all': 1.0}

        if avg_trading_value >= 10_000_000_000:  # 100억원
            liquidity_coef = 1.2
        elif avg_trading_value >= 1_000_000_000:  # 10억원
            liquidity_coef = 1.0
        else:
            liquidity_coef = 0.8

        return {'all_factors': liquidity_coef}

    def get_economic_cycle_weights(self, cycle_phase):
        """Get economic cycle weight adjustments (Phase 3.7: More aggressive weights)"""
        cycle_weights = {
            'EXPANSION': {
                'growth': 1.8,      # 1.3 -> 1.8 (Growth stocks thrive)
                'momentum': 1.5,    # 1.2 -> 1.5 (Trend following works)
                'value': 0.5,       # 0.8 -> 0.5 (Value underperforms)
                'quality': 0.6      # 0.7 -> 0.6
            },
            'SLOWDOWN': {
                'quality': 1.6,     # 1.3 -> 1.6 (Flight to quality)
                'value': 1.3,       # 1.1 -> 1.3
                'growth': 0.5,      # 0.7 -> 0.5 (Growth vulnerable)
                'momentum': 0.7     # 0.9 -> 0.7
            },
            'RECESSION': {
                'quality': 1.8,     # 1.4 -> 1.8 (Safety first)
                'value': 1.6,       # 1.3 -> 1.6 (Deep value opportunities)
                'growth': 0.3,      # 0.5 -> 0.3 (Avoid growth)
                'momentum': 0.5     # 0.8 -> 0.5 (Momentum fails)
            },
            'RECOVERY': {
                'value': 1.5,       # 1.2 -> 1.5 (Value rebounds first)
                'momentum': 1.4,    # 1.1 -> 1.4 (Catch the upturn)
                'growth': 1.0,      # 1.0 -> 1.0
                'quality': 0.6      # 0.7 -> 0.6
            },
            'NEUTRAL': {
                'all': 1.0
            }
        }
        return cycle_weights.get(cycle_phase, {'all': 1.0})

    def get_market_sentiment_weights(self, sentiment):
        """Get market sentiment weight adjustments (Phase 3.9: market state strategy)"""
        sentiment_weights = {
            'OVERHEATED': {  # Overheated: conservative
                'quality': 1.3,
                'value': 1.2,
                'growth': 0.8,
                'momentum': 0.7
            },
            'GREED': {  # Bull market: Growth emphasis
                'growth': 1.3,
                'momentum': 1.2,
                'value': 0.7,
                'quality': 0.8
            },
            'NEUTRAL': {  # Sideways: Value emphasis
                'value': 1.2,
                'quality': 1.1,
                'growth': 0.9,
                'momentum': 0.8
            },
            'FEAR': {  # Bear market: Quality emphasis
                'quality': 1.4,
                'value': 1.1,
                'growth': 0.7,
                'momentum': 0.8
            },
            'PANIC': {  # Panic: Quality max
                'quality': 1.5,
                'value': 1.2,
                'growth': 0.6,
                'momentum': 0.7
            }
        }
        return sentiment_weights.get(sentiment, {'all': 1.0})

    def get_sector_cycle_weights(self, sector_status):
        """Get sector cycle weight adjustments"""
        sector_weights = {
            'HOT': {
                'momentum': 1.4,
                'growth': 1.2,
                'value': 0.7,
                'quality': 0.7
            },
            'GROWING': {
                'momentum': 1.2,
                'growth': 1.1,
                'value': 0.9,
                'quality': 0.8
            },
            'STABLE': {
                'all': 1.0
            },
            'DECLINING': {
                'value': 1.2,
                'quality': 1.1,
                'momentum': 0.8,
                'growth': 0.9
            },
            'COLD': {
                'value': 1.4,
                'quality': 1.3,
                'momentum': 0.5,
                'growth': 0.8
            }
        }
        return sector_weights.get(sector_status, {'all': 1.0})

    def get_theme_weights(self, theme):
        """Get theme weight adjustments"""
        theme_weights_map = {
            'AI_BigData': {
                'value': 0.10,
                'quality': 0.15,
                'momentum': 0.35,
                'growth': 0.40
            },
            'Semiconductor': {
                'value': 0.20,
                'quality': 0.30,
                'momentum': 0.30,
                'growth': 0.20
            },
            'Battery': {
                'value': 0.15,
                'quality': 0.25,
                'momentum': 0.25,
                'growth': 0.35
            },
            'Bio_DrugRD': {
                'value': 0.05,
                'quality': 0.20,
                'momentum': 0.35,
                'growth': 0.40
            },
            'Pharma_CDMO': {
                'value': 0.25,
                'quality': 0.35,
                'momentum': 0.15,
                'growth': 0.25
            },
            'Advanced_Manufacturing': {
                'value': 0.20,
                'quality': 0.35,
                'momentum': 0.20,
                'growth': 0.25
            },
            'IT_Software': {
                'value': 0.15,
                'quality': 0.25,
                'momentum': 0.30,
                'growth': 0.30
            },
            'Electronics': {
                'value': 0.30,
                'quality': 0.35,
                'momentum': 0.15,
                'growth': 0.20
            },
            'Traditional_Manufacturing': {
                'value': 0.40,
                'quality': 0.30,
                'momentum': 0.15,
                'growth': 0.15
            },
            'Materials_Chemical': {
                'value': 0.30,
                'quality': 0.25,
                'momentum': 0.25,
                'growth': 0.20
            },
            'Consumer_Goods': {
                'value': 0.30,
                'quality': 0.35,
                'momentum': 0.15,
                'growth': 0.20
            },
            'Healthcare_Device': {
                'value': 0.20,
                'quality': 0.35,
                'momentum': 0.20,
                'growth': 0.25
            },
            'Energy': {
                'value': 0.35,
                'quality': 0.25,
                'momentum': 0.25,
                'growth': 0.15
            },
            'Telecom_Media': {
                'value': 0.35,
                'quality': 0.30,
                'momentum': 0.15,
                'growth': 0.20
            },
            'Finance': {
                'value': 0.35,
                'quality': 0.40,
                'momentum': 0.10,
                'growth': 0.15
            },
            'Shipbuilding': {
                'value': 0.35,
                'quality': 0.30,
                'momentum': 0.20,
                'growth': 0.15
            },
            'Others': {
                'value': 0.25,
                'quality': 0.25,
                'momentum': 0.25,
                'growth': 0.25
            }
        }
        return theme_weights_map.get(theme, {'all': 1.0})

    def get_volatility_weights(self, annual_volatility):
        """Get volatility weight adjustments"""
        if annual_volatility is None:
            return {'all': 1.0}

        if annual_volatility > 0.5:
            level = 'VERY_HIGH'
        elif annual_volatility > 0.35:
            level = 'HIGH'
        elif annual_volatility > 0.2:
            level = 'MEDIUM'
        else:
            level = 'LOW'

        volatility_weights = {
            'VERY_HIGH': {
                'momentum': 1.3,
                'value': 0.6,
                'quality': 1.4,
                'growth': 0.7
            },
            'HIGH': {
                'momentum': 1.1,
                'value': 0.8,
                'quality': 1.2,
                'growth': 0.9
            },
            'MEDIUM': {
                'all': 1.0
            },
            'LOW': {
                'value': 1.2,
                'quality': 0.9,
                'momentum': 0.8,
                'growth': 1.1
            }
        }
        return volatility_weights.get(level, {'all': 1.0})

    def get_supply_demand_weights(self, supply_demand):
        """Get supply/demand weight adjustments (Phase 3.10)"""
        supply_demand_weights = {
            'STRONG_BUY': {
                'momentum': 1.3,
                'growth': 1.2,
                'value': 0.8,
                'quality': 0.9
            },
            'INST_LED': {
                'quality': 1.2,
                'value': 1.1,
                'momentum': 0.9,
                'growth': 0.9
            },
            'FOREIGN_LED': {
                'momentum': 1.2,
                'growth': 1.1,
                'value': 0.9,
                'quality': 0.9
            },
            'STRONG_SELL': {
                'quality': 1.3,
                'value': 1.2,
                'momentum': 0.7,
                'growth': 0.8
            },
            'NEUTRAL': {
                'all': 1.0
            }
        }
        return supply_demand_weights.get(supply_demand, {'all': 1.0})

    def _classify_supply_demand(self, inst_net, foreign_net):
        """Classify supply/demand pattern from investor flows (Phase 3.10)"""
        if inst_net is None or foreign_net is None:
            return 'NEUTRAL'

        if inst_net > 0 and foreign_net > 0:
            return 'STRONG_BUY'
        elif inst_net > 0 and foreign_net <= 0:
            return 'INST_LED'
        elif inst_net <= 0 and foreign_net > 0:
            return 'FOREIGN_LED'
        elif inst_net < 0 and foreign_net < 0:
            return 'STRONG_SELL'
        else:
            return 'NEUTRAL'

    def _get_market_cap_category(self, market_cap) -> str:
        """Helper: Convert market cap to category"""
        if market_cap is None:
            return 'SMALL'

        if market_cap >= 10_000_000_000_000:
            return 'MEGA'
        elif market_cap >= 1_000_000_000_000:
            return 'LARGE'
        elif market_cap >= 200_000_000_000:
            return 'MEDIUM'
        else:
            return 'SMALL'

    def _get_liquidity_level(self, avg_trading_value) -> str:
        """Helper: Convert trading value to liquidity level"""
        if avg_trading_value is None:
            return 'LOW'

        if avg_trading_value >= 10_000_000_000:
            return 'HIGH'
        elif avg_trading_value >= 1_000_000_000:
            return 'MEDIUM'
        else:
            return 'LOW'

    def _get_volatility_level(self, annual_volatility) -> str:
        """Helper: Convert annual volatility to level"""
        if annual_volatility is None:
            return 'MEDIUM'

        if annual_volatility > 0.5:
            return 'VERY_HIGH'
        elif annual_volatility > 0.35:
            return 'HIGH'
        elif annual_volatility > 0.2:
            return 'MEDIUM'
        else:
            return 'LOW'

    def calculate_weights_for_stock(self, symbol, stock_data):
        """
        Phase 3: Calculate weights for single stock using pre-loaded data
        No database queries - pure memory operations
        """
        condition_weights = []

        # 1. Market Type
        exchange = stock_data.get('exchange')
        if exchange:
            condition_weights.append(self.get_market_type_weights(exchange))
        else:
            condition_weights.append({'all': 1.0})

        # 2. Market Cap
        market_cap = stock_data.get('market_cap')
        condition_weights.append(self.get_market_cap_weights(market_cap))

        # 3. Liquidity
        liquidity = stock_data.get('avg_trading_value')
        condition_weights.append(self.get_liquidity_weights(liquidity))

        # 4. Economic Cycle (common data)
        economic_cycle = self.common_data['economic_cycle']['cycle']
        condition_weights.append(self.get_economic_cycle_weights(economic_cycle))

        # 5. Market Sentiment (exchange-specific common data)
        if exchange and exchange in self.common_data['market_sentiment']:
            sentiment = self.common_data['market_sentiment'][exchange]
            condition_weights.append(self.get_market_sentiment_weights(sentiment))
        else:
            condition_weights.append({'all': 1.0})

        # 6. Sector Cycle (industry-specific common data)
        industry = stock_data.get('industry')
        if industry and industry in self.common_data['sector_cycle']:
            sector_cycle = self.common_data['sector_cycle'][industry]
            condition_weights.append(self.get_sector_cycle_weights(sector_cycle))
        else:
            condition_weights.append({'all': 1.0})

        # 7. Theme
        theme = stock_data.get('theme')
        condition_weights.append(self.get_theme_weights(theme))

        # 8. Volatility
        volatility = stock_data.get('annual_volatility')
        condition_weights.append(self.get_volatility_weights(volatility))

        # 9. Supply/Demand (Phase 3.10)
        inst_net = stock_data.get('inst_net_30d', 0)
        foreign_net = stock_data.get('foreign_net_30d', 0)
        supply_demand = self._classify_supply_demand(inst_net, foreign_net)
        condition_weights.append(self.get_supply_demand_weights(supply_demand))

        # Calculate final weights
        calculator = WeightCalculator()
        final_weights = calculator.calculate_final_weights(condition_weights)

        return final_weights

    # ========================================================================
    # MAIN BATCH PROCESSING
    # ========================================================================

    async def process_all_stocks(self, symbols):
        """
        Main entry point for batch processing with enhanced performance tracking

        Args:
            symbols: List of stock symbols (e.g., 2800 symbols)

        Returns:
            dict: {symbol: {weights, conditions, stock_data}}
        """
        start_time = datetime.now()
        self.performance_data['start_time'] = start_time.isoformat()

        logger.info(f"\n{'='*80}")
        logger.info(f"Starting batch processing for {len(symbols)} stocks...")
        logger.info(f"Start time: {start_time}")
        logger.info(f"{'='*80}\n")

        # Memory snapshot at start
        self._record_memory_snapshot("Start")

        # Phase 1: Calculate common data (4 methods, ~9 queries)
        logger.info(f"\n{'='*80}")
        logger.info("PHASE 1: Calculating common data for all stocks")
        logger.info(f"{'='*80}\n")

        phase1_start = datetime.now()
        await self.calculate_economic_cycle_data()      # 5 queries
        await self.calculate_market_sentiment_data()    # 2 queries (KOSPI, KOSDAQ)
        await self.calculate_sector_cycle_data()        # 1 query
        await self.precompute_scenario_stats()          # 1 query (Phase 1-4: scenario optimization)
        phase1_end = datetime.now()

        phase1_duration = (phase1_end - phase1_start).total_seconds()
        self.performance_data['phases']['phase1'] = phase1_duration
        logger.info(f"\nPhase 1 completed in {phase1_duration:.2f} seconds")
        self._record_memory_snapshot("After Phase 1")

        # Phase 2: Bulk fetch all stock data (2 queries - OPTIMIZED)
        logger.info(f"\n{'='*80}")
        logger.info("PHASE 2: Bulk fetching all stock data (OPTIMIZED)")
        logger.info(f"{'='*80}\n")

        phase2_start = datetime.now()
        all_stock_data = await self.fetch_all_stock_data(symbols)  # 2 queries (optimized from 4)
        phase2_end = datetime.now()

        phase2_duration = (phase2_end - phase2_start).total_seconds()
        self.performance_data['phases']['phase2'] = phase2_duration
        logger.info(f"\nPhase 2 completed in {phase2_duration:.2f} seconds")
        self._record_memory_snapshot("After Phase 2")

        # Phase 2.5: Classify all stocks into 19 market states (with caching)
        logger.info(f"\n{'='*80}")
        logger.info("PHASE 2.5: Classifying all stocks into market states")
        logger.info(f"{'='*80}\n")

        phase25_start = datetime.now()

        # Prepare conditions for all stocks
        all_conditions = {}
        for symbol, stock_data in all_stock_data.items():
            # Classify supply/demand (Phase 3.10)
            inst_net = stock_data.get('inst_net_30d', 0)
            foreign_net = stock_data.get('foreign_net_30d', 0)
            supply_demand = self._classify_supply_demand(inst_net, foreign_net)

            all_conditions[symbol] = {
                'exchange': stock_data.get('exchange'),
                'market_cap_category': self._get_market_cap_category(stock_data.get('market_cap')),
                'liquidity_level': self._get_liquidity_level(stock_data.get('avg_trading_value')),
                'economic_cycle': self.common_data['economic_cycle']['cycle'],
                'market_sentiment': self.common_data['market_sentiment'].get(stock_data.get('exchange'), 'NEUTRAL'),
                'sector_cycle': self.common_data['sector_cycle'].get(stock_data.get('industry'), 'STABLE'),
                'theme': stock_data.get('theme'),
                'volatility': self._get_volatility_level(stock_data.get('annual_volatility')),
                'supply_demand': supply_demand  # Phase 3.10
            }

        # Classify all stocks (with 24-hour caching)
        classifier = MarketClassifier()
        market_states = classifier.classify_batch(all_conditions)

        # Get and log classification statistics
        stats = classifier.get_classification_stats(market_states)
        logger.info(f"Classification completed: {stats['total_stocks']} stocks")
        logger.info(f"Coverage rate: {stats['coverage']['coverage_rate']:.2f}%")
        logger.info(f"Classified: {stats['coverage']['classified']}, Others: {stats['coverage']['others']}")

        phase25_end = datetime.now()
        phase25_duration = (phase25_end - phase25_start).total_seconds()
        self.performance_data['phases']['phase2_5'] = phase25_duration
        logger.info(f"\nPhase 2.5 completed in {phase25_duration:.2f} seconds")
        self._record_memory_snapshot("After Phase 2.5")

        # Phase 3: Calculate weights for each stock (no DB queries)
        logger.info(f"\n{'='*80}")
        logger.info("PHASE 3: Calculating weights for each stock (memory operations only)")
        logger.info(f"{'='*80}\n")

        phase3_start = datetime.now()
        results = {}
        processed_count = 0
        skipped_count = 0

        for symbol in symbols:
            if symbol not in all_stock_data:
                skipped_count += 1
                logger.warning(f"Skipping {symbol}: no data found")
                continue

            stock_data = all_stock_data[symbol]

            # Calculate weights (no DB queries)
            weights = self.calculate_weights_for_stock(symbol, stock_data)

            # Build result
            inst_net = stock_data.get('inst_net_30d', 0)
            foreign_net = stock_data.get('foreign_net_30d', 0)
            supply_demand = self._classify_supply_demand(inst_net, foreign_net)

            results[symbol] = {
                'weights': weights,
                'stock_data': stock_data,
                'conditions': {
                    'exchange': stock_data.get('exchange'),
                    'market_cap': stock_data.get('market_cap'),
                    'liquidity': stock_data.get('avg_trading_value'),
                    'economic_cycle': self.common_data['economic_cycle']['cycle'],
                    'market_sentiment': self.common_data['market_sentiment'].get(stock_data.get('exchange')),
                    'sector_cycle': self.common_data['sector_cycle'].get(stock_data.get('industry')),
                    'theme': stock_data.get('theme'),
                    'volatility': stock_data.get('annual_volatility'),
                    'supply_demand': supply_demand,  # Phase 3.10
                    'market_state': market_states.get(symbol, '기타')
                }
            }

            processed_count += 1

            # Progress logging
            if processed_count % 500 == 0:
                logger.info(f"Processed {processed_count}/{len(symbols)} stocks...")

        phase3_end = datetime.now()

        phase3_duration = (phase3_end - phase3_start).total_seconds()
        self.performance_data['phases']['phase3'] = phase3_duration
        logger.info(f"\nPhase 3 completed in {phase3_duration:.2f} seconds")
        self._record_memory_snapshot("After Phase 3")

        # Summary
        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()
        self.performance_data['end_time'] = end_time.isoformat()
        self.performance_data['total_time'] = total_time
        self.performance_data['total_stocks'] = len(symbols)
        self.performance_data['processed_stocks'] = processed_count
        self.performance_data['skipped_stocks'] = skipped_count

        logger.info(f"\n{'='*80}")
        logger.info("BATCH PROCESSING COMPLETED")
        logger.info(f"{'='*80}")
        logger.info(f"Total stocks requested: {len(symbols)}")
        logger.info(f"Successfully processed: {processed_count}")
        logger.info(f"Skipped (no data): {skipped_count}")
        logger.info(f"Total time: {total_time:.2f} seconds")
        logger.info(f"Average time per stock: {total_time/len(symbols):.4f} seconds")
        logger.info(f"{'='*80}\n")

        # Generate performance reports
        self._generate_performance_report()
        self._generate_diagnosis_report()

        return results

    # ========================================================================
    # PERFORMANCE TRACKING & DIAGNOSTICS
    # ========================================================================

    def _record_memory_snapshot(self, label):
        """Record memory usage at specific point"""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            snapshot = {
                'label': label,
                'timestamp': datetime.now().isoformat(),
                'rss_mb': round(memory_info.rss / 1024 / 1024, 2),
                'vms_mb': round(memory_info.vms / 1024 / 1024, 2),
                'percent': round(process.memory_percent(), 2)
            }
            self.performance_data['memory_snapshots'].append(snapshot)
            logger.debug(f"Memory snapshot ({label}): {snapshot['rss_mb']} MB RSS, {snapshot['percent']}%")
        except Exception as e:
            logger.warning(f"Failed to record memory snapshot: {e}")

    def _generate_performance_report(self):
        """Generate detailed performance report (JSON format)"""
        try:
            log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'log')
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            perf_file = os.path.join(log_dir, f'batch_weight_{timestamp}_performance.json')

            with open(perf_file, 'w', encoding='utf-8') as f:
                json.dump(self.performance_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Performance report saved: {perf_file}")

        except Exception as e:
            logger.error(f"Failed to generate performance report: {e}")

    def _generate_diagnosis_report(self):
        """Generate diagnosis report with bottleneck analysis"""
        try:
            log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'log')
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            diag_file = os.path.join(log_dir, f'batch_weight_{timestamp}_diagnosis.txt')

            with open(diag_file, 'w', encoding='utf-8') as f:
                f.write("="*80 + "\n")
                f.write("BATCH WEIGHT CALCULATOR - PERFORMANCE DIAGNOSIS REPORT\n")
                f.write("="*80 + "\n\n")

                # Summary
                f.write("### SUMMARY ###\n")
                f.write(f"Total time: {self.performance_data.get('total_time', 0):.2f}s\n")
                f.write(f"Total stocks: {self.performance_data.get('total_stocks', 0)}\n")
                f.write(f"Processed: {self.performance_data.get('processed_stocks', 0)}\n")
                f.write(f"Skipped: {self.performance_data.get('skipped_stocks', 0)}\n")
                f.write(f"Average per stock: {self.performance_data.get('total_time', 0) / max(1, self.performance_data.get('total_stocks', 1)):.4f}s\n\n")

                # Phase breakdown
                f.write("### PHASE BREAKDOWN ###\n")
                phases = self.performance_data.get('phases', {})
                total = sum(phases.values())
                for phase, duration in sorted(phases.items()):
                    pct = (duration / total * 100) if total > 0 else 0
                    f.write(f"{phase:12s}: {duration:7.2f}s ({pct:5.1f}%)\n")
                f.write(f"{'TOTAL':12s}: {total:7.2f}s (100.0%)\n\n")

                # Query analysis
                f.write("### QUERY ANALYSIS ###\n")
                queries = self.performance_data.get('queries', [])
                f.write(f"Total queries executed: {len(queries)}\n")
                total_query_time = sum(q.get('execution_time', 0) for q in queries)
                f.write(f"Total query time: {total_query_time:.2f}s\n\n")

                f.write("Top 5 slowest queries:\n")
                sorted_queries = sorted(queries, key=lambda x: x.get('execution_time', 0), reverse=True)
                for i, q in enumerate(sorted_queries[:5], 1):
                    f.write(f"  {i}. {q.get('name', 'Unnamed')}\n")
                    f.write(f"     Time: {q.get('execution_time', 0):.3f}s\n")
                    f.write(f"     Rows: {q.get('row_count', 0):,}\n")
                    f.write(f"     Memory: {q.get('memory_delta_mb', 0):+.2f}MB\n\n")

                # Memory analysis
                f.write("### MEMORY ANALYSIS ###\n")
                snapshots = self.performance_data.get('memory_snapshots', [])
                if snapshots:
                    f.write("Memory progression:\n")
                    for snap in snapshots:
                        f.write(f"  {snap.get('label', 'Unknown'):20s}: {snap.get('rss_mb', 0):8.2f} MB ({snap.get('percent', 0):5.2f}%)\n")

                    max_memory = max(s.get('rss_mb', 0) for s in snapshots)
                    min_memory = min(s.get('rss_mb', 0) for s in snapshots)
                    f.write(f"\n  Peak memory: {max_memory:.2f} MB\n")
                    f.write(f"  Memory growth: {max_memory - min_memory:.2f} MB\n\n")

                # Bottleneck diagnosis
                f.write("### BOTTLENECK DIAGNOSIS ###\n")
                self._diagnose_bottlenecks(f)

                # Recommendations
                f.write("\n### RECOMMENDATIONS ###\n")
                self._generate_recommendations(f)

            logger.info(f"Diagnosis report saved: {diag_file}")

        except Exception as e:
            logger.error(f"Failed to generate diagnosis report: {e}")
            logger.error(traceback.format_exc())

    def _diagnose_bottlenecks(self, file_handle):
        """Identify performance bottlenecks"""
        phases = self.performance_data.get('phases', {})
        total = sum(phases.values())

        if not phases:
            file_handle.write("No phase data available\n")
            return

        # Find slowest phase
        slowest_phase = max(phases.items(), key=lambda x: x[1])
        file_handle.write(f"Slowest phase: {slowest_phase[0]} ({slowest_phase[1]:.2f}s, {slowest_phase[1]/total*100:.1f}%)\n")

        # Check if Phase 2 is bottleneck
        if phases.get('phase2', 0) > total * 0.4:
            file_handle.write("WARNING: Phase 2 (data fetching) is a bottleneck (>40% of total time)\n")
            file_handle.write("  Possible causes:\n")
            file_handle.write("  - Missing database indexes (run create_indexes_optimization.sql)\n")
            file_handle.write("  - Slow network connection to database\n")
            file_handle.write("  - Large result set (consider batching)\n")

        # Check if Phase 3 is bottleneck
        if phases.get('phase3', 0) > total * 0.3:
            file_handle.write("WARNING: Phase 3 (calculation) is a bottleneck (>30% of total time)\n")
            file_handle.write("  Possible causes:\n")
            file_handle.write("  - CPU-intensive calculations\n")
            file_handle.write("  - Consider parallel processing (multiprocessing)\n")

        # Check query performance
        queries = self.performance_data.get('queries', [])
        slow_queries = [q for q in queries if q.get('execution_time', 0) > 5.0]
        if slow_queries:
            file_handle.write(f"\nWARNING: {len(slow_queries)} queries took >5 seconds\n")
            file_handle.write("  Run EXPLAIN ANALYZE to check query plans\n")

    def _generate_recommendations(self, file_handle):
        """Generate optimization recommendations"""
        phases = self.performance_data.get('phases', {})
        total = sum(phases.values())
        queries = self.performance_data.get('queries', [])

        recommendations = []

        # Check if indexes are needed
        total_query_time = sum(q.get('execution_time', 0) for q in queries)
        if total_query_time > total * 0.5:
            recommendations.append("1. Database indexes may help (run kr/create_indexes_optimization.sql)")

        # Check if parallel processing would help
        if phases.get('phase3', 0) > 10:
            recommendations.append("2. Consider parallel processing for Phase 3 (CPU calculations)")

        # Check memory usage
        snapshots = self.performance_data.get('memory_snapshots', [])
        if snapshots:
            max_memory = max(s.get('rss_mb', 0) for s in snapshots)
            if max_memory > 1000:  # >1GB
                recommendations.append(f"3. High memory usage detected ({max_memory:.0f}MB). Consider batching.")

        # Check if connection pooling would help
        if len(queries) > 20:
            recommendations.append("4. Consider connection pooling for better database performance")

        # Output recommendations
        if recommendations:
            for rec in recommendations:
                file_handle.write(f"{rec}\n")
        else:
            file_handle.write("No specific recommendations. Performance looks good!\n")

        file_handle.write("\nNext steps:\n")
        file_handle.write("- Review query execution plans with EXPLAIN ANALYZE\n")
        file_handle.write("- Monitor index usage after running create_indexes_optimization.sql\n")
        file_handle.write("- Consider implementing parallel processing if Phase 3 is slow\n")

    def check_query_performance(self, test_symbols=None):
        """
        Check query performance with EXPLAIN ANALYZE
        Useful for verifying index usage

        Args:
            test_symbols: List of test symbols (default: ['005930', '000660', '035420'])
        """
        if test_symbols is None:
            test_symbols = ['005930', '000660', '035420']  # Samsung, SK Hynix, NAVER

        logger.info("="*80)
        logger.info("QUERY PERFORMANCE ANALYSIS (EXPLAIN ANALYZE)")
        logger.info("="*80)

        # Test Query 1: Basic info
        logger.info("\n1. Testing basic info query (kr_stock_detail)...")
        query1 = """
        EXPLAIN ANALYZE
        SELECT symbol, exchange, industry, theme
        FROM kr_stock_detail
        WHERE symbol = ANY($1)
        """
        result1 = self.execute_query(query1, (test_symbols,), query_name="EXPLAIN: Basic Info")
        for row in result1:
            logger.info(f"  {row}")

        # Test Query 2: Combined market data
        logger.info("\n2. Testing combined market data query (kr_intraday_total)...")
        query2 = """
        EXPLAIN ANALYZE
        WITH latest_data AS (
            SELECT DISTINCT ON (symbol)
                symbol, market_cap, close, date
            FROM kr_intraday_total
            WHERE symbol = ANY($1)
            ORDER BY symbol, date DESC
        )
        SELECT * FROM latest_data
        """
        result2 = self.execute_query(query2, (test_symbols,), query_name="EXPLAIN: Latest Data")
        for row in result2:
            logger.info(f"  {row}")

        logger.info("\n" + "="*80)
        logger.info("Look for 'Index Scan' in the output above.")
        logger.info("If you see 'Seq Scan', indexes may not be created or not being used.")
        logger.info("Run create_indexes_optimization.sql to create recommended indexes.")
        logger.info("="*80)

    async def close(self):
        """Close database connection pool"""
        if self.pool:
            await self.pool.close()
            logger.info("Database connection pool closed")


async def get_all_korean_stock_symbols():
    """Get all Korean stock symbols from database"""
    try:
        database_url = os.getenv('DATABASE_URL')
        db_url = database_url.replace("postgresql+asyncpg://", "postgresql://")

        conn = await asyncpg.connect(db_url)

        try:
            symbols = await conn.fetch("SELECT DISTINCT symbol FROM kr_stock_detail ORDER BY symbol")
            symbols = [row['symbol'] for row in symbols]
            return symbols
        finally:
            await conn.close()

    except Exception as e:
        logger.error(f"Failed to get stock symbols: {e}")
        raise


async def main_batch():
    """Batch processing main function with enhanced logging"""
    print("\n" + "="*80)
    print("Korean Stock Batch Weight Calculator (OPTIMIZED)")
    print("Optimized for processing 2800+ stocks")
    print("="*80)
    print(f"Log file: {CURRENT_LOG_FILE}")
    print("="*80 + "\n")

    try:
        # Get all Korean stock symbols
        logger.info("Fetching all Korean stock symbols...")
        symbols = await get_all_korean_stock_symbols()
        logger.info(f"Found {len(symbols)} stocks to process")

        # Create calculator and process
        calculator = AsyncBatchWeightCalculator()
        await calculator.initialize()
        results = await calculator.process_all_stocks(symbols)

        # Display log file locations
        log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'log')
        print("\n" + "="*80)
        print("LOG FILES GENERATED")
        print("="*80)
        print(f"Main log: {CURRENT_LOG_FILE}")
        print(f"Performance data: {log_dir}\\batch_weight_*_performance.json")
        print(f"Diagnosis report: {log_dir}\\batch_weight_*_diagnosis.txt")
        print("="*80 + "\n")

        # Display sample results
        print("\n" + "="*80)
        print("SAMPLE RESULTS (first 5 stocks)")
        print("="*80 + "\n")

        for i, (symbol, data) in enumerate(list(results.items())[:5]):
            print(f"\n{symbol}:")
            print(f"  Market State: {data['conditions']['market_state']}")
            print(f"  Weights: Value={data['weights']['value']*100:.2f}%, "
                  f"Momentum={data['weights']['momentum']*100:.2f}%, "
                  f"Growth={data['weights']['growth']*100:.2f}%, "
                  f"Quality={data['weights']['quality']*100:.2f}%")
            print(f"  Economic Cycle: {data['conditions']['economic_cycle']}")
            print(f"  Market Sentiment: {data['conditions']['market_sentiment']}")
            print(f"  Sector Cycle: {data['conditions']['sector_cycle']}")

        await calculator.close()

        return results

    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main_batch())
