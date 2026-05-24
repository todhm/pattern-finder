"""
Condition Analyzer and Weight Calculator for Korean Stock Quant Analysis
Analyzes 8 conditions and calculates dynamic factor weights
"""

import os
import logging
import asyncio
from dotenv import load_dotenv
from decimal import Decimal

# Import market classifier
try:
    from market_classifier import MarketClassifier
except ImportError:
    from kr.market_classifier import MarketClassifier

# Load environment variables
load_dotenv()

# Logging configuration for debugging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ConditionAnalyzer:
    """Analyzes 8 market/stock conditions to determine dynamic weights"""

    def __init__(self, symbol, db_manager):
        self.symbol = symbol
        self.db_manager = db_manager
        self.conditions = {}
        self.condition_weights = []

    async def execute_query(self, query, *params):
        """Execute SQL query and return results"""
        try:
            result = await self.db_manager.execute_query(query, *params)
            return result
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            logger.error(f"Query: {query}")
            raise

    async def determine_market_type(self):
        """Condition 1: Determine KOSPI or KOSDAQ"""
        logger.info("Analyzing market type...")

        query = """
        SELECT exchange
        FROM kr_intraday_total
        WHERE symbol = $1
        ORDER BY date DESC
        LIMIT 1
        """

        result = await self.execute_query(query, self.symbol)

        if not result or not result[0]['exchange']:
            logger.warning(f"No exchange data found for {self.symbol}")
            return None

        exchange = result[0]['exchange']
        self.conditions['exchange'] = exchange

        # Weight adjustment coefficients
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

    async def determine_market_cap(self):
        """Condition 2: Determine market cap category"""
        logger.info("Analyzing market cap...")

        query = """
        SELECT market_cap
        FROM kr_intraday_total
        WHERE symbol = $1
        ORDER BY date DESC
        LIMIT 1
        """

        result = await self.execute_query(query, self.symbol)

        if not result or not result[0]['market_cap']:
            logger.warning(f"No market cap data found for {self.symbol}")
            return None

        market_cap = float(result[0]['market_cap'])
        self.conditions['market_cap'] = market_cap

        # Categorize based on trillion won
        if market_cap >= 10_000_000_000_000:  # 10 trillion won
            category = 'MEGA'
        elif market_cap >= 1_000_000_000_000:  # 1 trillion won
            category = 'LARGE'
        elif market_cap >= 200_000_000_000:  # 200 billion won
            category = 'MEDIUM'
        else:
            category = 'SMALL'

        self.conditions['market_cap_category'] = category

        # Weight adjustment coefficients
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

    async def determine_liquidity(self):
        """Condition 3: Determine liquidity level"""
        logger.info("Analyzing liquidity...")

        query = """
        SELECT AVG(trading_value) as avg_trading_value
        FROM kr_intraday_total
        WHERE symbol = $1
            AND date >= CURRENT_DATE - INTERVAL '20 days'
        """

        result = await self.execute_query(query, self.symbol)

        if not result or not result[0]['avg_trading_value']:
            logger.warning(f"No trading value data found for {self.symbol}")
            return None

        avg_trading = float(result[0]['avg_trading_value'])
        self.conditions['avg_trading_value'] = avg_trading

        # Categorize based on 100 million won (1억원 = 100,000,000)
        if avg_trading >= 10_000_000_000:  # 100억원
            level = 'HIGH'
            liquidity_coef = 1.2
        elif avg_trading >= 1_000_000_000:  # 10억원
            level = 'MEDIUM'
            liquidity_coef = 1.0
        else:
            level = 'LOW'
            liquidity_coef = 0.8

        self.conditions['liquidity_level'] = level

        return {'all_factors': liquidity_coef}

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

    async def calculate_similarity_score(self, current_pattern, defined_pattern):
        """
        Calculate similarity score between two patterns using weighted matching

        Weights:
        - GDP: 40% (most important)
        - Interest Rate: 20%
        - Sentiment: 20%
        - Exchange Rate: 10%
        - CPI: 10%

        Returns: float (0.0 ~ 1.0)
        """
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
            score += WEIGHTS['gdp'] * 0.5  # Partial credit for similar direction

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

    async def match_cycle_matrix(self, gdp, rate, sentiment, exchange, cpi):
        """Match indicators to economic cycle using 32-pattern matrix with similarity scoring"""

        # 32-pattern cycle matrix (expanded from 16)
        CYCLE_MATRIX = {
            # === EXPANSION patterns (8) ===
            # Strong expansion with overheating signs
            ('UP', 'TIGHTENING', 'OPTIMISTIC', 'STRONG', 'INFLATION'): 'EXPANSION',
            ('UP', 'TIGHTENING', 'OPTIMISTIC', 'STABLE', 'INFLATION'): 'EXPANSION',
            ('UP', 'STABLE', 'OPTIMISTIC', 'STABLE', 'INFLATION'): 'EXPANSION',
            ('UP', 'TIGHTENING', 'OPTIMISTIC', 'STABLE', 'STABLE'): 'EXPANSION',
            # Goldilocks pattern (ideal growth)
            ('UP', 'STABLE', 'OPTIMISTIC', 'STABLE', 'STABLE'): 'EXPANSION',
            ('UP', 'STABLE', 'OPTIMISTIC', 'STRONG', 'STABLE'): 'EXPANSION',
            # Mixed signals but still expanding
            ('UP', 'TIGHTENING', 'NEUTRAL', 'STABLE', 'STABLE'): 'EXPANSION',
            ('UP', 'STABLE', 'NEUTRAL', 'STABLE', 'STABLE'): 'EXPANSION',

            # === RECOVERY patterns (8) ===
            # Policy-driven recovery
            ('UP', 'EASING', 'OPTIMISTIC', 'WEAK', 'STABLE'): 'RECOVERY',
            ('UP', 'EASING', 'OPTIMISTIC', 'STABLE', 'STABLE'): 'RECOVERY',
            ('UP', 'EASING', 'NEUTRAL', 'STABLE', 'STABLE'): 'RECOVERY',
            ('UP', 'EASING', 'NEUTRAL', 'WEAK', 'STABLE'): 'RECOVERY',
            # Early recovery from bottom
            ('NEUTRAL', 'EASING', 'OPTIMISTIC', 'WEAK', 'DEFLATION'): 'RECOVERY',
            ('NEUTRAL', 'EASING', 'OPTIMISTIC', 'STABLE', 'STABLE'): 'RECOVERY',
            ('NEUTRAL', 'EASING', 'NEUTRAL', 'STABLE', 'STABLE'): 'RECOVERY',
            # V-shaped recovery
            ('DOWN', 'EASING', 'OPTIMISTIC', 'WEAK', 'STABLE'): 'RECOVERY',

            # === SLOWDOWN patterns (10) ===
            # Stagflation risk
            ('DOWN', 'TIGHTENING', 'PESSIMISTIC', 'WEAK', 'INFLATION'): 'SLOWDOWN',
            ('DOWN', 'TIGHTENING', 'NEUTRAL', 'WEAK', 'INFLATION'): 'SLOWDOWN',
            ('NEUTRAL', 'TIGHTENING', 'PESSIMISTIC', 'WEAK', 'INFLATION'): 'SLOWDOWN',
            # Policy tightening effects
            ('DOWN', 'TIGHTENING', 'NEUTRAL', 'WEAK', 'STABLE'): 'SLOWDOWN',
            ('DOWN', 'TIGHTENING', 'PESSIMISTIC', 'STABLE', 'STABLE'): 'SLOWDOWN',
            ('NEUTRAL', 'TIGHTENING', 'PESSIMISTIC', 'STABLE', 'STABLE'): 'SLOWDOWN',
            # Mixed signals slowdown
            ('DOWN', 'STABLE', 'PESSIMISTIC', 'WEAK', 'STABLE'): 'SLOWDOWN',
            ('DOWN', 'STABLE', 'PESSIMISTIC', 'STABLE', 'STABLE'): 'SLOWDOWN',
            ('DOWN', 'STABLE', 'NEUTRAL', 'WEAK', 'STABLE'): 'SLOWDOWN',
            ('NEUTRAL', 'STABLE', 'PESSIMISTIC', 'WEAK', 'STABLE'): 'SLOWDOWN',

            # === RECESSION patterns (4) ===
            # Deep recession
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
            self.conditions['cycle_confidence'] = 'VERY_HIGH'
            self.conditions['cycle_similarity'] = 1.0
            logger.info(f"Exact match found for pattern {current_pattern}")
            return CYCLE_MATRIX[current_pattern]

        # Step 2: Find most similar pattern using weighted scoring
        best_cycle = None
        best_score = 0.0
        best_pattern = None

        for pattern, cycle in CYCLE_MATRIX.items():
            score = await self.calculate_similarity_score(current_pattern, pattern)
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

        # Store similarity info
        self.conditions['cycle_confidence'] = confidence
        self.conditions['cycle_similarity'] = round(best_score, 2)
        self.conditions['matched_pattern'] = best_pattern

        logger.info(f"Best match: {best_pattern} -> {best_cycle} (similarity: {best_score:.2f}, confidence: {confidence})")

        # Step 4: If confidence is too low, use GDP-based fallback
        if best_score < 0.4:
            logger.warning(f"Low confidence ({best_score:.2f}), using GDP-based fallback")
            if gdp == 'UP':
                if rate == 'EASING' or sentiment == 'OPTIMISTIC':
                    return 'RECOVERY'
                else:
                    return 'EXPANSION'
            elif gdp == 'DOWN':
                if rate == 'TIGHTENING' or sentiment == 'PESSIMISTIC':
                    return 'SLOWDOWN'
                else:
                    return 'RECESSION'
            else:
                return 'NEUTRAL'

        return best_cycle

    async def determine_economic_cycle(self):
        """Condition 4: Determine economic cycle using multi-indicator matrix"""
        logger.info("Analyzing economic cycle with multi-indicator approach...")

        # Step 1: Check each economic indicator
        gdp_trend = await self.check_gdp_trend()
        rate_trend = await self.check_interest_rate_trend()
        sentiment = await self.check_sentiment_index()
        exchange_trend = await self.check_exchange_rate_trend()
        cpi_trend = await self.check_cpi_trend()

        # Step 2: Match patterns to cycle
        cycle_phase = await self.match_cycle_matrix(gdp_trend, rate_trend, sentiment, exchange_trend, cpi_trend)

        # Step 3: Store conditions for output
        self.conditions['economic_cycle'] = cycle_phase
        self.conditions['gdp_trend'] = gdp_trend
        self.conditions['interest_rate_trend'] = rate_trend
        self.conditions['sentiment_index'] = sentiment
        self.conditions['exchange_rate_trend'] = exchange_trend
        self.conditions['cpi_trend'] = cpi_trend

        logger.info(f"Indicators: GDP={gdp_trend}, Rate={rate_trend}, Sentiment={sentiment}, "
                   f"Exchange={exchange_trend}, CPI={cpi_trend} -> Cycle={cycle_phase}")

        # Economic cycle weight adjustments (Phase 3.7: More aggressive weights)
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

    async def determine_market_sentiment(self):
        """Condition 5: Determine market sentiment (Fear & Greed)"""
        logger.info("Analyzing market sentiment...")

        # First, get the exchange for this symbol
        exchange_query = """
        SELECT exchange FROM kr_stock_basic WHERE symbol = $1
        """
        exchange_result = await self.execute_query(exchange_query, self.symbol)

        if not exchange_result:
            logger.warning(f"No exchange info found for {self.symbol}")
            return {'all': 1.0}

        exchange = exchange_result[0]['exchange']

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

        result = await self.execute_query(query, exchange)

        if not result or not result[0]['market_sentiment']:
            logger.warning("No market sentiment data available")
            return {'all': 1.0}

        sentiment = result[0]['market_sentiment']
        self.conditions['market_sentiment'] = sentiment

        # Market sentiment weight adjustments (Phase 3.9: market state strategy)
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

    async def determine_sector_cycle(self):
        """Condition 6: Determine sector cycle"""
        logger.info("Analyzing sector cycle...")

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
            WHERE d.industry = (SELECT industry FROM kr_stock_detail WHERE symbol = $1)
                AND i.date >= CURRENT_DATE - INTERVAL '40 days'
        ),
        sector_performance AS (
            SELECT
                industry,
                AVG((close - close_20d_ago) / NULLIF(close_20d_ago, 0)) as avg_return_20d,
                AVG(trading_value) as avg_trading_value
            FROM stock_returns
            WHERE close_20d_ago IS NOT NULL
            GROUP BY industry
        )
        SELECT
            industry,
            avg_return_20d,
            CASE
                WHEN avg_return_20d > 0.1 THEN 'HOT'
                WHEN avg_return_20d > 0.03 THEN 'GROWING'
                WHEN avg_return_20d > -0.03 THEN 'STABLE'
                WHEN avg_return_20d > -0.1 THEN 'DECLINING'
                ELSE 'COLD'
            END as sector_status
        FROM sector_performance
        """

        result = await self.execute_query(query, self.symbol)

        if not result or not result[0]['sector_status']:
            logger.warning("No sector cycle data available")
            return {'all': 1.0}

        sector_status = result[0]['sector_status']
        self.conditions['sector_cycle'] = sector_status

        # Sector cycle weight adjustments
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

    async def determine_theme_classification(self):
        """Condition 7: Determine theme classification"""
        logger.info("Analyzing theme classification...")

        query = """
        SELECT theme, industry
        FROM kr_stock_detail
        WHERE symbol = $1
        """

        result = await self.execute_query(query, self.symbol)

        if not result:
            logger.warning(f"No theme data found for {self.symbol}")
            return {'all': 1.0}

        theme = result[0]['theme'] if result[0]['theme'] else ''
        industry = result[0]['industry'] if result[0]['industry'] else ''

        self.conditions['theme'] = theme
        self.conditions['industry'] = industry
        self.conditions['theme_type'] = theme

        # Theme-based weight mapping (16 themes)
        # Weights are absolute values that sum to 1.0 for each theme
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

    async def determine_volatility_level(self):
        """Condition 8: Determine volatility level (60-day)"""
        logger.info("Analyzing volatility level...")

        query = """
        WITH daily_returns AS (
            SELECT
                date,
                close,
                LN(close / NULLIF(LAG(close) OVER (ORDER BY date), 0)) as log_return
            FROM kr_intraday_total
            WHERE symbol = $1
                AND date >= CURRENT_DATE - INTERVAL '60 days'
            ORDER BY date
        )
        SELECT
            STDDEV(log_return) * SQRT(252) as annual_volatility,
            CASE
                WHEN STDDEV(log_return) * SQRT(252) > 0.5 THEN 'VERY_HIGH'
                WHEN STDDEV(log_return) * SQRT(252) > 0.35 THEN 'HIGH'
                WHEN STDDEV(log_return) * SQRT(252) > 0.2 THEN 'MEDIUM'
                ELSE 'LOW'
            END as volatility_level
        FROM daily_returns
        WHERE log_return IS NOT NULL
        """

        result = await self.execute_query(query, self.symbol)

        if not result or not result[0]['volatility_level']:
            logger.warning(f"No volatility data found for {self.symbol}")
            return {'all': 1.0}

        volatility_level = result[0]['volatility_level']
        annual_vol = float(result[0]['annual_volatility']) if result[0]['annual_volatility'] else 0

        self.conditions['volatility'] = volatility_level
        self.conditions['annual_volatility'] = annual_vol

        # Volatility weight adjustments
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

        return volatility_weights.get(volatility_level, {'all': 1.0})

    async def determine_supply_demand(self):
        """
        Determine supply/demand condition based on institutional and foreign investor flows
        (Phase 3.10: 수급 조건 추가)

        Returns:
            dict: Weight adjustments based on supply/demand pattern
        """
        query = """
        SELECT
            SUM(inst_net_volume) as inst_net_30d,
            SUM(foreign_net_volume) as foreign_net_30d,
            COUNT(*) as trading_days
        FROM kr_individual_investor_daily_trading
        WHERE symbol = $1
            AND date >= CURRENT_DATE - INTERVAL '30 days'
            AND date <= CURRENT_DATE
        """

        result = await self.execute_query(query, self.symbol)

        if not result or result[0]['trading_days'] is None or result[0]['trading_days'] < 10:
            logger.warning(f"Insufficient supply/demand data for {self.symbol}")
            self.conditions['supply_demand'] = 'NEUTRAL'
            return {'all': 1.0}

        inst_net = result[0]['inst_net_30d'] or 0
        foreign_net = result[0]['foreign_net_30d'] or 0

        # Classify supply/demand pattern
        if inst_net > 0 and foreign_net > 0:
            supply_demand = 'STRONG_BUY'
        elif inst_net > 0 and foreign_net <= 0:
            supply_demand = 'INST_LED'
        elif inst_net <= 0 and foreign_net > 0:
            supply_demand = 'FOREIGN_LED'
        elif inst_net < 0 and foreign_net < 0:
            supply_demand = 'STRONG_SELL'
        else:
            supply_demand = 'NEUTRAL'

        self.conditions['supply_demand'] = supply_demand
        self.conditions['inst_net_30d'] = inst_net
        self.conditions['foreign_net_30d'] = foreign_net

        logger.info(f"Supply/Demand: {supply_demand} (inst={inst_net:,}, foreign={foreign_net:,})")

        # Supply/demand weight adjustments
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

    async def analyze(self):
        """Execute all 9 condition analyses (Phase 3.10: 수급 조건 추가)"""
        logger.info(f"\n=== Starting analysis for symbol: {self.symbol} ===\n")

        try:
            # Execute all 9 condition determinations
            self.condition_weights.append(await self.determine_market_type())
            self.condition_weights.append(await self.determine_market_cap())
            self.condition_weights.append(await self.determine_liquidity())
            self.condition_weights.append(await self.determine_economic_cycle())
            self.condition_weights.append(await self.determine_market_sentiment())
            self.condition_weights.append(await self.determine_sector_cycle())
            self.condition_weights.append(await self.determine_theme_classification())
            self.condition_weights.append(await self.determine_volatility_level())
            self.condition_weights.append(await self.determine_supply_demand())  # Phase 3.10

            logger.info("All conditions analyzed successfully (9 conditions)")

            # Classify market state (19 classifications)
            classifier = MarketClassifier()
            market_state = classifier.classify(self.conditions)
            self.conditions['market_state'] = market_state
            logger.info(f"Market state classified: {market_state}")

            return self.conditions, self.condition_weights

        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            raise


class WeightCalculator:
    """Calculate final factor weights based on condition analysis"""

    def __init__(self):
        # Base weights (IC-based: Growth 0.495 >> others)
        self.base_weights = {
            'value': 0.15,      # IC 0.083
            'momentum': 0.20,   # IC 0.041
            'growth': 0.40,     # IC 0.495
            'quality': 0.25     # IC 0.004 (defensive)
        }

    def calculate_final_weights(self, condition_weights_list):
        """
        Calculate final weights by combining all condition adjustments

        Parameters:
        condition_weights_list: List of weight adjustment dictionaries from each condition

        Returns:
        dict: Final normalized weights for each factor
        """

        # Step 1: Accumulate adjustment coefficients
        cumulative_adjustments = {
            'value': 1.0,
            'momentum': 1.0,
            'growth': 1.0,
            'quality': 1.0
        }

        for condition_weights in condition_weights_list:
            if condition_weights:
                if 'all' in condition_weights:
                    # Apply to all factors equally
                    for factor in cumulative_adjustments:
                        cumulative_adjustments[factor] *= condition_weights['all']
                elif 'all_factors' in condition_weights:
                    # Liquidity-style adjustment
                    for factor in cumulative_adjustments:
                        cumulative_adjustments[factor] *= condition_weights['all_factors']
                else:
                    # Individual factor adjustments
                    for factor, adjustment in condition_weights.items():
                        if factor in cumulative_adjustments:
                            cumulative_adjustments[factor] *= adjustment

        # Step 2: Calculate adjusted weights
        adjusted_weights = {}
        for factor, base_weight in self.base_weights.items():
            adjusted_weights[factor] = base_weight * cumulative_adjustments[factor]

        # Step 3: Normalize (sum to 1)
        total = sum(adjusted_weights.values())
        final_weights = {
            factor: weight / total
            for factor, weight in adjusted_weights.items()
        }

        # Step 4: Apply extreme value limits (5% min, 50% max)
        for factor in final_weights:
            final_weights[factor] = max(0.05, min(0.50, final_weights[factor]))

        # Step 5: Re-normalize after applying limits
        total = sum(final_weights.values())
        final_weights = {
            factor: weight / total
            for factor, weight in final_weights.items()
        }

        return final_weights

    def get_neutralized_scores(self, market_state: str, scores: dict) -> dict:
        """
        Phase 6: market_state에 따라 예측력 없는 factor를 50점으로 중립화

        90일 IC 분석 결과, 특정 market_state에서 momentum과 quality의 예측력이
        없거나 역효과가 발생함. 해당 조건에서 50점(중립)으로 강제.

        Args:
            market_state: 현재 시장 상태 문자열
            scores: {'value': float, 'quality': float, 'momentum': float, 'growth': float}

        Returns:
            dict: 중립화 적용된 scores
        """
        if not market_state:
            return scores

        neutralized = scores.copy()

        # Momentum 중립화 조건 (IC 음수인 3개 상태)
        # - 모멘텀형, 모멘텀폭발: 이미 모멘텀으로 분류된 종목은 추가 효과 없음
        # - 역발상: 순방향 모멘텀이 역효과
        # - 확장+과열: 과열 시장에서 모멘텀 평균회귀
        momentum_neutralize = False

        if any(x in market_state for x in ['모멘텀형', '모멘텀폭발']):
            momentum_neutralize = True

        if '역발상' in market_state:
            momentum_neutralize = True

        if '확장' in market_state and '과열' in market_state:
            momentum_neutralize = True

        # Quality 중립화 조건 (IC 음수인 8개 상태)
        # - 침체/공포: 시장 전체 하락에서 quality 무의미
        # - 역발상: 역발상 전략에서 quality 역효과
        # - 탐욕: 투기 시장에서 quality 무시됨
        # - 과열+모멘텀, 테마+모멘텀폭발: 급등장에서 quality 무관
        quality_neutralize = False

        if any(x in market_state for x in ['침체', '공포']):
            quality_neutralize = True

        if '역발상' in market_state:
            quality_neutralize = True

        if '탐욕' in market_state:
            quality_neutralize = True

        if '과열' in market_state and '모멘텀' in market_state:
            quality_neutralize = True

        if '테마' in market_state and '모멘텀폭발' in market_state:
            quality_neutralize = True

        # 50점 중립화 적용
        if momentum_neutralize:
            neutralized['momentum'] = 50.0

        if quality_neutralize:
            neutralized['quality'] = 50.0

        return neutralized


def format_large_number(num):
    """Format large numbers in Korean units (trillion, billion)"""
    if num >= 1_000_000_000_000:
        return f"{num / 1_000_000_000_000:.1f}조원"
    elif num >= 100_000_000:
        return f"{num / 100_000_000:.0f}억원"
    else:
        return f"{num:,.0f}원"


async def main():
    """Main execution function"""
    print("\n" + "="*60)
    print("한국 주식 조건값 분석 및 가중치 계산 시스템")
    print("="*60 + "\n")

    # Get symbol input
    symbol = input("종목코드를 입력하세요: ").strip()

    if not symbol:
        print("종목코드가 입력되지 않았습니다.")
        return

    try:
        # Initialize database connection
        from db_async import AsyncDatabaseManager
        db_manager = AsyncDatabaseManager()
        await db_manager.initialize()

        # Analyze conditions
        analyzer = ConditionAnalyzer(symbol, db_manager)
        conditions, condition_weights = await analyzer.analyze()

        # Calculate final weights
        calculator = WeightCalculator()
        final_weights = calculator.calculate_final_weights(condition_weights)

        # Display results
        print("\n" + "="*60)
        print(f"종목: {symbol} 분석 결과")
        print("="*60 + "\n")

        print("[조건값 분석 결과]")
        print("-"*60)
        print(f"1. 거래소: {conditions.get('exchange', 'N/A')}")

        if 'market_cap' in conditions:
            market_cap_str = format_large_number(conditions['market_cap'])
            print(f"2. 시가총액: {conditions.get('market_cap_category', 'N/A')} ({market_cap_str})")
        else:
            print(f"2. 시가총액: N/A")

        if 'avg_trading_value' in conditions:
            trading_str = format_large_number(conditions['avg_trading_value'])
            print(f"3. 유동성: {conditions.get('liquidity_level', 'N/A')} (일평균 {trading_str})")
        else:
            print(f"3. 유동성: N/A")

        # Economic cycle with detailed info
        cycle_text = f"4. 경제사이클: {conditions.get('economic_cycle', 'N/A')}"
        if 'cycle_confidence' in conditions:
            confidence = conditions.get('cycle_confidence', 'N/A')
            similarity = conditions.get('cycle_similarity', 'N/A')
            cycle_text += f" (신뢰도: {confidence}, 유사도: {similarity})"
        print(cycle_text)

        # Show detailed indicators if available
        if 'gdp_trend' in conditions:
            print(f"   지표: GDP={conditions.get('gdp_trend', 'N/A')}, "
                  f"금리={conditions.get('interest_rate_trend', 'N/A')}, "
                  f"심리={conditions.get('sentiment_index', 'N/A')}, "
                  f"환율={conditions.get('exchange_rate_trend', 'N/A')}, "
                  f"물가={conditions.get('cpi_trend', 'N/A')}")

        print(f"5. 시장국면: {conditions.get('market_sentiment', 'N/A')}")
        print(f"6. 섹터사이클: {conditions.get('sector_cycle', 'N/A')}")
        print(f"7. 테마: {conditions.get('theme_type', 'N/A')}")

        if conditions.get('industry'):
            print(f"   산업: {conditions['industry']}")

        if 'annual_volatility' in conditions:
            vol_pct = conditions['annual_volatility'] * 100
            print(f"8. 변동성: {conditions.get('volatility', 'N/A')} (연환산 {vol_pct:.1f}%)")
        else:
            print(f"8. 변동성: N/A")

        print("\n" + "-"*60)
        print(f"[시장 상태 분류]")
        print(f"{conditions.get('market_state', 'N/A')}")

        print("\n" + "="*60)
        print("[최종 가중치]")
        print("-"*60)
        print(f"Value:    {final_weights['value']*100:6.2f}%")
        print(f"Momentum: {final_weights['momentum']*100:6.2f}%")
        print(f"Growth:   {final_weights['growth']*100:6.2f}%")
        print(f"Quality:  {final_weights['quality']*100:6.2f}%")
        print("-"*60)
        print(f"Total:    {sum(final_weights.values())*100:6.2f}%")
        print("="*60 + "\n")

        await db_manager.close()

    except Exception as e:
        logger.error(f"Error occurred: {e}")
        print(f"\n오류 발생: {e}")
        print("종목코드를 확인하거나 데이터베이스 연결을 확인해주세요.")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
