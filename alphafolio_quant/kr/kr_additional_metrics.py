"""
Additional Quantitative Metrics Calculator
추가 정량 지표 계산기

Calculates 13 metrics for comprehensive stock analysis:

Fundamental & Risk Metrics (9):
1. Analysis Confidence Score
2. Volatility (Annualized)
3. Expected Price Range (68% CI)
4. Downside Risk (VaR 95%)
5. Institutional/Foreign Flow
6. Factor Momentum
7. Industry Ranking
8. Maximum Drawdown (1Y)
9. Beta (Estimated)

Technical Indicators (4):
10. Support/Resistance Levels (2 levels each)
11. SuperTrend Indicator
12. Relative Strength (RS)
13. RS Rank

All results designed for kr_stock_grade table storage.
"""

import os
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

# Phase 3.9: Scenario Probability Calibration
try:
    from kr_probability_calibrator import ScenarioCalibrator
except ImportError:
    from kr.kr_probability_calibrator import ScenarioCalibrator

# Phase 3.10: Data Prefetcher for query optimization
try:
    from kr_data_prefetcher import PrefetchedDataCalculator
except ImportError:
    from kr.kr_data_prefetcher import PrefetchedDataCalculator

# Logger setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AdditionalMetricsCalculator:
    """
    Calculate additional quantitative metrics for stock analysis
    Designed for kr_stock_grade table storage
    """

    def __init__(self, symbol: str, db_manager, factor_scores: Dict = None, analysis_date=None, prefetched_data: Dict = None):
        """
        Initialize calculator

        Args:
            symbol: Stock symbol
            db_manager: AsyncDatabaseManager instance
            factor_scores: Dict containing factor analysis results from kr_main
            analysis_date: Optional analysis date (defaults to current date)
            prefetched_data: Optional prefetched data from KRDataPrefetcher (query optimization)
        """
        self.symbol = symbol
        self.db_manager = db_manager
        self.factor_scores = factor_scores or {}
        self.analysis_date = analysis_date
        # Phase 3.9: Scenario Calibrator (shared instance)
        self.calibrator = ScenarioCalibrator(db_manager)
        # Phase 3.10: Prefetched data calculator (83% query reduction)
        self.prefetched_data = prefetched_data
        self.prefetch_calc = PrefetchedDataCalculator(prefetched_data) if prefetched_data else None
        logger.info(f"Additional metrics calculator initialized for {symbol}")

    async def execute_query(self, query: str, *params):
        """Execute SQL query and return results"""
        try:
            return await self.db_manager.execute_query(query, *params)
        except Exception as e:
            logger.error(f"Query execution error: {e}")
            return None

    async def calculate_confidence_score(self) -> float:
        """
        1. Calculate analysis confidence score (0-100)
        Based on data completeness and valid strategy count

        Storage: confidence_score FLOAT
        """
        try:
            total_strategies = 0
            valid_strategies = 0

            # Count valid strategies from each factor
            for factor_name in ['value', 'quality', 'momentum', 'growth']:
                factor_data = self.factor_scores.get(factor_name, {})
                strategies = factor_data.get('strategies', {})

                for score in strategies.values():
                    total_strategies += 1
                    if score is not None:
                        valid_strategies += 1

            if total_strategies == 0:
                return 0.0

            # Data completeness ratio
            completeness = (valid_strategies / total_strategies) * 100

            # Additional confidence factors
            # Check if all 4 factors have weighted results
            weighted_results_count = sum(
                1 for factor in ['value', 'quality', 'momentum', 'growth']
                if self.factor_scores.get(factor, {}).get('weighted_result') is not None
            )
            factor_confidence = (weighted_results_count / 4) * 100

            # Combined confidence score
            confidence = (completeness * 0.7) + (factor_confidence * 0.3)

            logger.info(f"Confidence score: {confidence:.1f}% ({valid_strategies}/{total_strategies} strategies)")
            return round(confidence, 1)

        except Exception as e:
            logger.error(f"Confidence score calculation failed: {e}")
            return 0.0

    async def calculate_volatility_annual(self) -> Optional[float]:
        """
        2. Calculate annualized volatility (%)
        Based on 90-day daily returns

        Storage: volatility_annual FLOAT
        """
        try:
            # Phase 3.10: Use prefetched data if available
            if self.prefetch_calc:
                result = self.prefetch_calc.calculate_volatility_annual()
                if result is not None:
                    logger.info(f"Annual volatility: {result:.2f}% (prefetched)")
                    return result

            # Fallback to DB query
            query = """
            WITH daily_returns AS (
                SELECT
                    date,
                    close,
                    LAG(close) OVER (ORDER BY date) as prev_close,
                    ((close - LAG(close) OVER (ORDER BY date))::NUMERIC /
                     NULLIF(LAG(close) OVER (ORDER BY date), 0) * 100) as daily_return
                FROM kr_intraday_total
                WHERE symbol = $1
                    AND date >= COALESCE($2::date, CURRENT_DATE) - INTERVAL '90 days'
                    AND date <= COALESCE($2::date, CURRENT_DATE)
                    AND close IS NOT NULL
                ORDER BY date
            )
            SELECT
                STDDEV(daily_return) as std_dev,
                COUNT(*) as days_count
            FROM daily_returns
            WHERE daily_return IS NOT NULL
            """

            result = await self.execute_query(query, self.symbol, self.analysis_date)

            if not result or not result[0]['std_dev']:
                return None

            daily_std = float(result[0]['std_dev'])
            days_count = int(result[0]['days_count'])

            # Annualize: daily_std * sqrt(252)
            annual_volatility = daily_std * np.sqrt(252)

            logger.info(f"Annual volatility: {annual_volatility:.2f}% (based on {days_count} days)")
            return round(annual_volatility, 2)

        except Exception as e:
            logger.error(f"Volatility calculation failed: {e}")
            return None

    async def calculate_expected_range(self, volatility: float = None) -> dict:
        """
        3. Calculate expected price range (68% confidence interval)
        Current price ± 1 standard deviation (3 months and 1 year)

        Storage: expected_range_3m_min, expected_range_3m_max, expected_range_1y_min, expected_range_1y_max

        Returns:
            dict with keys: range_3m_min, range_3m_max, range_1y_min, range_1y_max
        """
        try:
            # Get current price
            query = """
            SELECT close as current_price
            FROM kr_intraday_total
            WHERE symbol = $1
                AND ($2::date IS NULL OR date = $2)
            ORDER BY date DESC
            LIMIT 1
            """

            result = await self.execute_query(query, self.symbol, self.analysis_date)

            if not result or not result[0]['current_price']:
                return None

            current_price = float(result[0]['current_price'])

            # Get or calculate volatility
            if volatility is None:
                volatility = await self.calculate_volatility_annual()
                if volatility is None:
                    return None

            # Convert annual volatility to 3-month volatility
            # 3 months ≈ 63 trading days
            volatility_3m = volatility / np.sqrt(252) * np.sqrt(63)

            # 1 year = 252 trading days (use annual volatility as-is)
            volatility_1y = volatility

            # Calculate 3-month range (current price ± volatility%)
            range_3m_min = current_price * (1 - volatility_3m / 100)
            range_3m_max = current_price * (1 + volatility_3m / 100)

            # Calculate 1-year range (current price ± volatility%)
            range_1y_min = current_price * (1 - volatility_1y / 100)
            range_1y_max = current_price * (1 + volatility_1y / 100)

            logger.info(f"Expected range (3M, 68% CI): {range_3m_min:.0f} - {range_3m_max:.0f} (current: {current_price:.0f})")
            logger.info(f"Expected range (1Y, 68% CI): {range_1y_min:.0f} - {range_1y_max:.0f} (current: {current_price:.0f})")

            return {
                'range_3m_min': round(range_3m_min, 2),
                'range_3m_max': round(range_3m_max, 2),
                'range_1y_min': round(range_1y_min, 2),
                'range_1y_max': round(range_1y_max, 2)
            }

        except Exception as e:
            logger.error(f"Expected range calculation failed: {e}")
            return None

    async def calculate_var_95(self) -> Optional[float]:
        """
        4. Calculate Value at Risk (VaR 95%)
        5th percentile of 90-day return distribution

        Storage: var_95 FLOAT (negative value, %)
        """
        try:
            # Phase 3.10: Use prefetched data if available
            if self.prefetch_calc:
                result = self.prefetch_calc.calculate_var_95()
                if result is not None:
                    logger.info(f"VaR 95%: -{result:.2f}% (prefetched)")
                    return -result  # Return as negative value

            # Fallback to DB query
            query = """
            WITH daily_returns AS (
                SELECT
                    ((close - LAG(close) OVER (ORDER BY date))::NUMERIC /
                     NULLIF(LAG(close) OVER (ORDER BY date), 0) * 100) as daily_return
                FROM kr_intraday_total
                WHERE symbol = $1
                    AND date >= COALESCE($2::date, CURRENT_DATE) - INTERVAL '90 days'
                    AND date <= COALESCE($2::date, CURRENT_DATE)
                    AND close IS NOT NULL
                ORDER BY date
            )
            SELECT
                PERCENTILE_CONT(0.05) WITHIN GROUP (ORDER BY daily_return) as var_95
            FROM daily_returns
            WHERE daily_return IS NOT NULL
            """

            result = await self.execute_query(query, self.symbol, self.analysis_date)

            if not result or result[0]['var_95'] is None:
                return None

            var_95 = float(result[0]['var_95'])

            logger.info(f"VaR 95%: {var_95:.2f}% (95% chance loss won't exceed this)")
            return round(var_95, 2)

        except Exception as e:
            logger.error(f"VaR calculation failed: {e}")
            return None

    async def calculate_cvar_95(self) -> Optional[float]:
        """
        4-1. Calculate Conditional Value at Risk (CVaR 95%) = Expected Shortfall
        Average of returns below VaR 95% threshold

        CVaR is more conservative than VaR as it considers the tail risk
        - VaR 95%: "95% of the time, loss won't exceed X%"
        - CVaR 95%: "When loss exceeds VaR, average loss is Y%"

        Storage: cvar_95 FLOAT (negative value, %)
        """
        try:
            # Phase 3.10: Use prefetched data if available
            if self.prefetch_calc:
                result = self.prefetch_calc.calculate_cvar_95()
                if result is not None:
                    logger.info(f"CVaR 95%: -{result:.2f}% (prefetched)")
                    return -result  # Return as negative value

            # Fallback to DB query
            query = """
            WITH daily_returns AS (
                SELECT
                    ((close - LAG(close) OVER (ORDER BY date))::NUMERIC /
                     NULLIF(LAG(close) OVER (ORDER BY date), 0) * 100) as daily_return
                FROM kr_intraday_total
                WHERE symbol = $1
                    AND date >= COALESCE($2::date, CURRENT_DATE) - INTERVAL '90 days'
                    AND date <= COALESCE($2::date, CURRENT_DATE)
                    AND close IS NOT NULL
                ORDER BY date
            ),
            var_threshold AS (
                SELECT PERCENTILE_CONT(0.05) WITHIN GROUP (ORDER BY daily_return) as var_95
                FROM daily_returns
                WHERE daily_return IS NOT NULL
            )
            SELECT
                AVG(dr.daily_return) as cvar_95,
                COUNT(*) as tail_count
            FROM daily_returns dr, var_threshold vt
            WHERE dr.daily_return IS NOT NULL
                AND dr.daily_return <= vt.var_95
            """

            result = await self.execute_query(query, self.symbol, self.analysis_date)

            if not result or result[0]['cvar_95'] is None:
                return None

            cvar_95 = float(result[0]['cvar_95'])
            tail_count = int(result[0]['tail_count']) if result[0]['tail_count'] else 0

            logger.info(f"CVaR 95%: {cvar_95:.2f}% (avg loss in worst {tail_count} days)")
            return round(cvar_95, 2)

        except Exception as e:
            logger.error(f"CVaR calculation failed: {e}")
            return None

    # ========================================================================
    # Phase 1 VaR Improvements: Hurst, EWMA, Period VaR
    # ========================================================================

    async def calculate_hurst_exponent(self) -> Optional[float]:
        """
        Calculate Hurst Exponent using R/S (Rescaled Range) Analysis.

        Storage: hurst_exponent FLOAT (0.1 ~ 0.9)
        - H > 0.5: Trending (momentum persists, risk grows faster than sqrt(T))
        - H = 0.5: Random Walk (risk grows as sqrt(T))
        - H < 0.5: Mean Reverting (risk grows slower than sqrt(T))
        """
        try:
            # Phase 3.10: Use prefetched data if available
            if self.prefetch_calc:
                result = self.prefetch_calc.calculate_hurst_exponent()
                if result is not None:
                    logger.info(f"Hurst Exponent: {result:.4f} (prefetched)")
                    return result

            # Fallback to DB query
            query = """
            WITH daily_returns AS (
                SELECT
                    ((close - LAG(close) OVER (ORDER BY date))::NUMERIC /
                     NULLIF(LAG(close) OVER (ORDER BY date), 0) * 100) as daily_return
                FROM kr_intraday_total
                WHERE symbol = $1
                    AND date >= COALESCE($2::date, CURRENT_DATE) - INTERVAL '252 days'
                    AND date <= COALESCE($2::date, CURRENT_DATE)
                    AND close IS NOT NULL
                ORDER BY date
            )
            SELECT ARRAY_AGG(daily_return ORDER BY daily_return) as returns
            FROM daily_returns
            WHERE daily_return IS NOT NULL
            """

            result = await self.execute_query(query, self.symbol, self.analysis_date)

            if not result or not result[0]['returns']:
                return 0.5  # Default

            returns = np.array([float(r) for r in result[0]['returns'] if r is not None])

            if len(returns) < 100:
                return 0.5  # Insufficient data

            # R/S Analysis
            N = len(returns)
            min_chunk = 8

            if N < min_chunk * 4:
                return 0.5

            chunk_sizes = []
            rs_values = []

            for chunk_size in range(min_chunk, N // 4):
                rs_list = []

                for start in range(0, N - chunk_size + 1, chunk_size):
                    chunk = returns[start:start + chunk_size]
                    mean_adj = chunk - np.mean(chunk)
                    cumsum = np.cumsum(mean_adj)
                    R = np.max(cumsum) - np.min(cumsum)
                    S = np.std(chunk, ddof=1)

                    if S > 0:
                        rs_list.append(R / S)

                if rs_list:
                    chunk_sizes.append(chunk_size)
                    rs_values.append(np.mean(rs_list))

            if len(chunk_sizes) < 3:
                return 0.5

            log_n = np.log(chunk_sizes)
            log_rs = np.log(rs_values)
            slope, _ = np.polyfit(log_n, log_rs, 1)

            H = max(0.1, min(0.9, slope))

            logger.info(f"Hurst Exponent: {H:.4f}")
            return round(H, 4)

        except Exception as e:
            logger.error(f"Hurst Exponent calculation failed: {e}")
            return 0.5

    async def calculate_var_95_ewma(self, lambda_param: float = 0.94) -> Optional[float]:
        """
        Calculate EWMA (Exponentially Weighted Moving Average) VaR 95%.
        EWMA gives more weight to recent volatility, capturing volatility clustering.

        Storage: var_95_ewma FLOAT (negative value, %)

        Args:
            lambda_param: Decay factor (default 0.94, RiskMetrics standard)
        """
        try:
            # Phase 3.10: Use prefetched data if available
            if self.prefetch_calc:
                result = self.prefetch_calc.calculate_var_95_ewma(lambda_param)
                if result is not None:
                    logger.info(f"VaR 95% (EWMA): -{result:.2f}% (prefetched)")
                    return -result  # Return as negative value

            # Fallback to DB query
            query = """
            WITH daily_returns AS (
                SELECT
                    ((close - LAG(close) OVER (ORDER BY date))::NUMERIC /
                     NULLIF(LAG(close) OVER (ORDER BY date), 0)) as daily_return
                FROM kr_intraday_total
                WHERE symbol = $1
                    AND date >= COALESCE($2::date, CURRENT_DATE) - INTERVAL '90 days'
                    AND date <= COALESCE($2::date, CURRENT_DATE)
                    AND close IS NOT NULL
                ORDER BY date
            )
            SELECT ARRAY_AGG(daily_return ORDER BY daily_return) as returns
            FROM daily_returns
            WHERE daily_return IS NOT NULL
            """

            result = await self.execute_query(query, self.symbol, self.analysis_date)

            if not result or not result[0]['returns']:
                return None

            returns = np.array([float(r) for r in result[0]['returns'] if r is not None])

            if len(returns) < 20:
                return None

            # EWMA Variance calculation
            ewma_var = returns[0] ** 2
            for r in returns[1:]:
                ewma_var = lambda_param * ewma_var + (1 - lambda_param) * (r ** 2)

            ewma_vol = np.sqrt(ewma_var) * 100  # Convert to percentage

            # VaR 95% (assuming normal distribution)
            var_95_ewma = -1.645 * ewma_vol

            logger.info(f"VaR 95% (EWMA): {var_95_ewma:.2f}%")
            return round(var_95_ewma, 2)

        except Exception as e:
            logger.error(f"EWMA VaR calculation failed: {e}")
            return None

    async def calculate_var_99(self) -> Optional[float]:
        """
        Calculate Value at Risk (99%) from daily returns.
        1st percentile of return distribution.

        Storage: var_99 FLOAT (negative value, %)
        """
        try:
            # Phase 3.10: Use prefetched data if available
            if self.prefetch_calc:
                result = self.prefetch_calc.calculate_var_99()
                if result is not None:
                    logger.info(f"VaR 99%: -{result:.2f}% (prefetched)")
                    return -result  # Return as negative value

            # Fallback to DB query
            query = """
            WITH daily_returns AS (
                SELECT
                    ((close - LAG(close) OVER (ORDER BY date))::NUMERIC /
                     NULLIF(LAG(close) OVER (ORDER BY date), 0) * 100) as daily_return
                FROM kr_intraday_total
                WHERE symbol = $1
                    AND date >= COALESCE($2::date, CURRENT_DATE) - INTERVAL '90 days'
                    AND date <= COALESCE($2::date, CURRENT_DATE)
                    AND close IS NOT NULL
                ORDER BY date
            )
            SELECT
                PERCENTILE_CONT(0.01) WITHIN GROUP (ORDER BY daily_return) as var_99
            FROM daily_returns
            WHERE daily_return IS NOT NULL
            """

            result = await self.execute_query(query, self.symbol, self.analysis_date)

            if not result or result[0]['var_99'] is None:
                return None

            var_99 = float(result[0]['var_99'])

            logger.info(f"VaR 99%: {var_99:.2f}%")
            return round(var_99, 2)

        except Exception as e:
            logger.error(f"VaR 99% calculation failed: {e}")
            return None

    async def calculate_period_var(self) -> Dict[str, Optional[float]]:
        """
        Calculate period-specific VaR using Hurst Exponent scaling.
        VaR_T = VaR_1d * T^H (Hurst-based scaling instead of sqrt(T))

        Storage: var_95_5d, var_95_20d, var_95_60d, var_99_60d FLOAT (negative values, %)
        """
        result = {
            'var_95_5d': None,
            'var_95_20d': None,
            'var_95_60d': None,
            'var_99_60d': None
        }

        try:
            # Phase 3.10: Use prefetched data if available
            if self.prefetch_calc:
                prefetch_result = self.prefetch_calc.calculate_period_var()
                if prefetch_result:
                    # Convert to negative values for storage
                    result['var_95_5d'] = -prefetch_result['var_95_5d'] if prefetch_result.get('var_95_5d') else None
                    result['var_95_20d'] = -prefetch_result['var_95_20d'] if prefetch_result.get('var_95_20d') else None
                    result['var_95_60d'] = -prefetch_result['var_95_60d'] if prefetch_result.get('var_95_60d') else None
                    result['var_99_60d'] = -prefetch_result['var_99_60d'] if prefetch_result.get('var_99_60d') else None

                    logger.info(f"Period VaR (prefetched): 5d={result['var_95_5d']}, 20d={result['var_95_20d']}, 60d={result['var_95_60d']}")
                    return result

            # Fallback: Calculate using individual methods
            var_95_1d = await self.calculate_var_95()
            var_99_1d = await self.calculate_var_99()
            hurst = await self.calculate_hurst_exponent()

            if var_95_1d is None:
                return result

            if hurst is None:
                hurst = 0.5

            # VaR is already negative from calculate_var_95(), use absolute value for scaling
            var_95_abs = abs(var_95_1d)

            # Scale VaR to different periods: VaR_T = VaR_1d * T^H
            result['var_95_5d'] = round(-var_95_abs * (5 ** hurst), 2)
            result['var_95_20d'] = round(-var_95_abs * (20 ** hurst), 2)
            result['var_95_60d'] = round(-var_95_abs * (60 ** hurst), 2)

            if var_99_1d is not None:
                var_99_abs = abs(var_99_1d)
                result['var_99_60d'] = round(-var_99_abs * (60 ** hurst), 2)

            logger.info(f"Period VaR: 5d={result['var_95_5d']}, 20d={result['var_95_20d']}, 60d={result['var_95_60d']}, H={hurst:.4f}")
            return result

        except Exception as e:
            logger.error(f"Period VaR calculation failed: {e}")
            return result

    # ========================================================================
    # Phase 2: Volatility Sizing Metrics
    # ========================================================================

    async def calculate_inv_vol_weight(self) -> Optional[float]:
        """
        Calculate Inverse Volatility Weight.
        Used for volatility-based position sizing.
        Higher weight = lower volatility = larger position allowed.

        Formula: 1 / volatility_annual

        Storage: inv_vol_weight FLOAT
        """
        try:
            # Use prefetched data if available
            if self.prefetch_calc:
                result = self.prefetch_calc.calculate_inv_vol_weight()
                if result is not None:
                    logger.info(f"Inverse Vol Weight: {result:.6f} (prefetched)")
                    return result

            # Fallback: calculate from volatility
            vol = await self.calculate_volatility_annual()
            if vol is None or vol <= 0:
                return None

            inv_vol = 1.0 / vol
            logger.info(f"Inverse Vol Weight: {inv_vol:.6f}")
            return round(inv_vol, 6)

        except Exception as e:
            logger.error(f"Inverse Vol Weight calculation failed: {e}")
            return None

    async def calculate_downside_vol(self) -> Optional[float]:
        """
        Calculate Downside Volatility (Semideviation).
        Only considers negative returns, better captures downside risk.

        Formula: std(negative returns only) * sqrt(252)

        Storage: downside_vol FLOAT (%)
        """
        try:
            # Use prefetched data if available
            if self.prefetch_calc:
                result = self.prefetch_calc.calculate_downside_vol()
                if result is not None:
                    logger.info(f"Downside Volatility: {result:.2f}% (prefetched)")
                    return result

            # Fallback to DB query
            query = """
            WITH daily_returns AS (
                SELECT
                    ((close - LAG(close) OVER (ORDER BY date))::NUMERIC /
                     NULLIF(LAG(close) OVER (ORDER BY date), 0) * 100) as daily_return
                FROM kr_intraday_total
                WHERE symbol = $1
                    AND date >= COALESCE($2::date, CURRENT_DATE) - INTERVAL '90 days'
                    AND date <= COALESCE($2::date, CURRENT_DATE)
                    AND close IS NOT NULL
                ORDER BY date
            )
            SELECT ARRAY_AGG(daily_return) as returns
            FROM daily_returns
            WHERE daily_return IS NOT NULL AND daily_return < 0
            """

            result = await self.execute_query(query, self.symbol, self.analysis_date)

            if not result or not result[0]['returns']:
                return None

            negative_returns = [float(r) for r in result[0]['returns'] if r is not None]

            if len(negative_returns) < 5:
                return None

            downside_std = float(np.std(negative_returns))
            downside_vol = downside_std * np.sqrt(252)

            logger.info(f"Downside Volatility: {downside_vol:.2f}%")
            return round(downside_vol, 2)

        except Exception as e:
            logger.error(f"Downside Volatility calculation failed: {e}")
            return None

    async def calculate_atr_metrics(self) -> Dict[str, Optional[float]]:
        """
        Calculate 20-day ATR and ATR percentage.

        ATR (Average True Range) measures volatility considering gaps.
        True Range = max(high-low, |high-prev_close|, |low-prev_close|)

        Storage: atr_20d FLOAT, atr_pct_20d FLOAT (%)
        """
        result = {
            'atr_20d': None,
            'atr_pct_20d': None
        }

        try:
            # Use prefetched data if available
            if self.prefetch_calc:
                atr = self.prefetch_calc.calculate_atr_20d()
                atr_pct = self.prefetch_calc.calculate_atr_pct_20d()
                if atr is not None:
                    result['atr_20d'] = atr
                    result['atr_pct_20d'] = atr_pct
                    logger.info(f"ATR 20d: {atr:.2f}, ATR%: {atr_pct:.2f}% (prefetched)")
                    return result

            # Fallback to DB query
            query = """
            WITH daily_data AS (
                SELECT
                    date,
                    high,
                    low,
                    close,
                    LAG(close) OVER (ORDER BY date) as prev_close
                FROM kr_intraday_total
                WHERE symbol = $1
                    AND date <= COALESCE($2::date, CURRENT_DATE)
                ORDER BY date DESC
                LIMIT 21
            ),
            true_range AS (
                SELECT
                    date,
                    GREATEST(
                        high - low,
                        ABS(high - COALESCE(prev_close, high)),
                        ABS(low - COALESCE(prev_close, low))
                    ) as tr,
                    close
                FROM daily_data
                WHERE prev_close IS NOT NULL
            )
            SELECT
                AVG(tr) as atr,
                (SELECT close FROM true_range ORDER BY date DESC LIMIT 1) as current_close
            FROM true_range
            """

            query_result = await self.execute_query(query, self.symbol, self.analysis_date)

            if not query_result or not query_result[0]['atr'] or not query_result[0]['current_close']:
                return result

            atr = float(query_result[0]['atr'])
            close = float(query_result[0]['current_close'])
            atr_pct = (atr / close) * 100

            result['atr_20d'] = round(atr, 2)
            result['atr_pct_20d'] = round(atr_pct, 2)

            logger.info(f"ATR 20d: {atr:.2f}, ATR%: {atr_pct:.2f}%")
            return result

        except Exception as e:
            logger.error(f"ATR metrics calculation failed: {e}")
            return result

    # ========================================================================
    # Phase 3: CVaR + Risk Budgeting Metrics
    # ========================================================================

    async def calculate_cvar_99(self) -> Optional[float]:
        """
        Calculate CVaR 99% (Expected Shortfall at 99%).
        Average loss in the worst 1% of cases.

        Storage: cvar_99 FLOAT (positive value, %)
        """
        try:
            # Use prefetched data if available
            if self.prefetch_calc:
                result = self.prefetch_calc.calculate_cvar_99()
                if result is not None:
                    logger.info(f"CVaR 99%: {result:.2f}% (prefetched)")
                    return result

            # Fallback to DB query
            query = """
            WITH daily_returns AS (
                SELECT
                    ((close - LAG(close) OVER (ORDER BY date))::NUMERIC /
                     NULLIF(LAG(close) OVER (ORDER BY date), 0) * 100) as daily_return
                FROM kr_intraday_total
                WHERE symbol = $1
                    AND date >= COALESCE($2::date, CURRENT_DATE) - INTERVAL '252 days'
                    AND date <= COALESCE($2::date, CURRENT_DATE)
                ORDER BY date
            ),
            var_threshold AS (
                SELECT PERCENTILE_CONT(0.01) WITHIN GROUP (ORDER BY daily_return) as var_99
                FROM daily_returns
                WHERE daily_return IS NOT NULL
            )
            SELECT
                AVG(dr.daily_return) as cvar_99,
                COUNT(*) as tail_count
            FROM daily_returns dr, var_threshold vt
            WHERE dr.daily_return IS NOT NULL
                AND dr.daily_return <= vt.var_99
            """

            result = await self.execute_query(query, self.symbol, self.analysis_date)

            if not result or result[0]['cvar_99'] is None:
                return None

            cvar_99 = abs(float(result[0]['cvar_99']))
            logger.info(f"CVaR 99%: {cvar_99:.2f}%")
            return round(cvar_99, 2)

        except Exception as e:
            logger.error(f"CVaR 99% calculation failed: {e}")
            return None

    async def calculate_corr_kospi(self, kospi_data: dict = None) -> Optional[float]:
        """
        Calculate correlation with KOSPI index (60 days).

        Storage: corr_kospi FLOAT (-1 ~ 1)

        Args:
            kospi_data: Precomputed KOSPI data {'dates': [...], 'returns': [...]}
                        If None, will query from DB
        """
        try:
            # Get stock returns
            if self.prefetch_calc:
                stock_returns = self.prefetch_calc.get_returns(60)
                if len(stock_returns) < 30:
                    return None
            else:
                # Fallback to DB query
                query = """
                SELECT
                    date,
                    ((close - LAG(close) OVER (ORDER BY date))::NUMERIC /
                     NULLIF(LAG(close) OVER (ORDER BY date), 0) * 100) as daily_return
                FROM kr_intraday_total
                WHERE symbol = $1
                    AND date >= COALESCE($2::date, CURRENT_DATE) - INTERVAL '90 days'
                    AND date <= COALESCE($2::date, CURRENT_DATE)
                ORDER BY date DESC
                """
                result = await self.execute_query(query, self.symbol, self.analysis_date)
                if not result or len(result) < 30:
                    return None
                stock_returns = [float(r['daily_return']) for r in result if r['daily_return'] is not None]

            # Get KOSPI returns
            if kospi_data and 'returns' in kospi_data and len(kospi_data['returns']) >= 30:
                kospi_returns = kospi_data['returns'][:len(stock_returns)]
            else:
                # Fallback: query KOSPI data
                kospi_query = """
                SELECT
                    ((close - LAG(close) OVER (ORDER BY date))::NUMERIC /
                     NULLIF(LAG(close) OVER (ORDER BY date), 0) * 100) as daily_return
                FROM market_index
                WHERE exchange = 'KOSPI'
                    AND date >= COALESCE($1::date, CURRENT_DATE) - INTERVAL '90 days'
                    AND date <= COALESCE($1::date, CURRENT_DATE)
                ORDER BY date DESC
                """
                kospi_result = await self.execute_query(kospi_query, self.analysis_date)
                if not kospi_result or len(kospi_result) < 30:
                    return None
                kospi_returns = [float(r['daily_return']) for r in kospi_result if r['daily_return'] is not None]

            # Align lengths
            min_len = min(len(stock_returns), len(kospi_returns))
            if min_len < 30:
                return None

            stock_returns = stock_returns[:min_len]
            kospi_returns = kospi_returns[:min_len]

            # Calculate correlation
            corr = float(np.corrcoef(stock_returns, kospi_returns)[0, 1])

            if np.isnan(corr):
                return None

            logger.info(f"KOSPI Correlation: {corr:.3f}")
            return round(corr, 3)

        except Exception as e:
            logger.error(f"KOSPI correlation calculation failed: {e}")
            return None

    async def calculate_tail_beta(self, kospi_data: dict = None) -> Optional[float]:
        """
        Calculate Tail Beta (sensitivity during market downturns).
        Beta calculated only when KOSPI return is in bottom 10%.

        Storage: tail_beta FLOAT

        Args:
            kospi_data: Precomputed KOSPI data {'dates': [...], 'returns': [...]}
        """
        try:
            # Get stock returns
            if self.prefetch_calc:
                stock_returns = self.prefetch_calc.get_returns(252)
                if len(stock_returns) < 60:
                    return None
            else:
                query = """
                SELECT
                    ((close - LAG(close) OVER (ORDER BY date))::NUMERIC /
                     NULLIF(LAG(close) OVER (ORDER BY date), 0) * 100) as daily_return
                FROM kr_intraday_total
                WHERE symbol = $1
                    AND date >= COALESCE($2::date, CURRENT_DATE) - INTERVAL '252 days'
                    AND date <= COALESCE($2::date, CURRENT_DATE)
                ORDER BY date DESC
                """
                result = await self.execute_query(query, self.symbol, self.analysis_date)
                if not result or len(result) < 60:
                    return None
                stock_returns = [float(r['daily_return']) for r in result if r['daily_return'] is not None]

            # Get KOSPI returns
            if kospi_data and 'returns' in kospi_data:
                kospi_returns = kospi_data['returns']
            else:
                kospi_query = """
                SELECT
                    ((close - LAG(close) OVER (ORDER BY date))::NUMERIC /
                     NULLIF(LAG(close) OVER (ORDER BY date), 0) * 100) as daily_return
                FROM market_index
                WHERE exchange = 'KOSPI'
                    AND date >= COALESCE($1::date, CURRENT_DATE) - INTERVAL '252 days'
                    AND date <= COALESCE($1::date, CURRENT_DATE)
                ORDER BY date DESC
                """
                kospi_result = await self.execute_query(kospi_query, self.analysis_date)
                if not kospi_result:
                    return None
                kospi_returns = [float(r['daily_return']) for r in kospi_result if r['daily_return'] is not None]

            # Align lengths
            min_len = min(len(stock_returns), len(kospi_returns))
            if min_len < 60:
                return None

            stock_returns = np.array(stock_returns[:min_len])
            kospi_returns = np.array(kospi_returns[:min_len])

            # Find tail threshold (bottom 10%)
            tail_threshold = np.percentile(kospi_returns, 10)

            # Filter to tail days only
            tail_mask = kospi_returns <= tail_threshold
            tail_stock = stock_returns[tail_mask]
            tail_kospi = kospi_returns[tail_mask]

            if len(tail_stock) < 5:
                return None

            # Calculate tail beta using regression
            # Beta = Cov(stock, kospi) / Var(kospi)
            cov = np.cov(tail_stock, tail_kospi)[0, 1]
            var_kospi = np.var(tail_kospi)

            if var_kospi == 0:
                return None

            tail_beta = cov / var_kospi

            logger.info(f"Tail Beta: {tail_beta:.3f} (from {len(tail_stock)} down days)")
            return round(tail_beta, 3)

        except Exception as e:
            logger.error(f"Tail Beta calculation failed: {e}")
            return None

    async def calculate_drawdown_duration_avg(self) -> Optional[float]:
        """
        Calculate average drawdown duration in days.
        Measures recovery time after drawdowns.

        Storage: drawdown_duration_avg FLOAT (days)
        """
        try:
            # Use prefetched data if available
            if self.prefetch_calc:
                result = self.prefetch_calc.calculate_drawdown_duration_avg()
                if result is not None:
                    logger.info(f"Avg Drawdown Duration: {result:.1f} days (prefetched)")
                    return result

            # Fallback to DB query
            query = """
            SELECT date, close
            FROM kr_intraday_total
            WHERE symbol = $1
                AND date >= COALESCE($2::date, CURRENT_DATE) - INTERVAL '252 days'
                AND date <= COALESCE($2::date, CURRENT_DATE)
            ORDER BY date ASC
            """
            result = await self.execute_query(query, self.symbol, self.analysis_date)

            if not result or len(result) < 60:
                return None

            closes = [float(r['close']) for r in result if r['close'] is not None]

            if len(closes) < 60:
                return None

            # Track drawdown durations
            durations = []
            running_max = closes[0]
            current_dd_start = None

            for i, close in enumerate(closes):
                if close >= running_max:
                    if current_dd_start is not None:
                        duration = i - current_dd_start
                        if duration > 0:
                            durations.append(duration)
                        current_dd_start = None
                    running_max = close
                else:
                    if current_dd_start is None:
                        current_dd_start = i

            if not durations:
                return 0.0

            avg_duration = float(np.mean(durations))
            logger.info(f"Avg Drawdown Duration: {avg_duration:.1f} days")
            return round(avg_duration, 1)

        except Exception as e:
            logger.error(f"Drawdown duration calculation failed: {e}")
            return None

    # ========================================================================
    # 멀티 에이전트 AI용 신규 지표 (Phase Agent)
    # ========================================================================

    async def calculate_atr_stop_take_profit(self) -> dict:
        """
        ATR 기반 손절/익절 계산 (Phase Agent)

        공식:
            ATR% = 14일 ATR / 현재가 × 100
            stop_loss = -2 × ATR%
            take_profit = 3 × ATR% (기본 RR 1.5)

        Returns:
            dict: {
                'atr_pct': ATR % (14일),
                'stop_loss_pct': 손절 기준 % (음수),
                'take_profit_pct': 익절 기준 % (양수),
                'risk_reward_ratio': 리스크리워드 비율
            }
        """
        try:
            query = """
            WITH daily_data AS (
                SELECT
                    date,
                    high,
                    low,
                    close,
                    LAG(close) OVER (ORDER BY date) as prev_close
                FROM kr_intraday_total
                WHERE symbol = $1
                    AND date <= COALESCE($2::date, CURRENT_DATE)
                ORDER BY date DESC
                LIMIT 15
            ),
            true_range AS (
                SELECT
                    date,
                    GREATEST(
                        high - low,
                        ABS(high - COALESCE(prev_close, high)),
                        ABS(low - COALESCE(prev_close, low))
                    ) as tr,
                    close
                FROM daily_data
                WHERE prev_close IS NOT NULL
            )
            SELECT
                AVG(tr) as atr,
                (SELECT close FROM true_range ORDER BY date DESC LIMIT 1) as current_close
            FROM true_range
            """

            result = await self.execute_query(query, self.symbol, self.analysis_date)

            if not result or not result[0]['atr'] or not result[0]['current_close']:
                return None

            atr = float(result[0]['atr'])
            close = float(result[0]['current_close'])
            atr_pct = (atr / close) * 100

            # 손절: 2 ATR, 익절: 3 ATR (RR 1.5)
            stop_loss_pct = -2 * atr_pct
            take_profit_pct = 3 * atr_pct
            risk_reward = abs(take_profit_pct / stop_loss_pct) if stop_loss_pct != 0 else 0

            logger.info(f"ATR Stop/Take: ATR%={atr_pct:.2f}%, SL={stop_loss_pct:.2f}%, TP={take_profit_pct:.2f}%, RR={risk_reward:.2f}")

            return {
                'atr_pct': round(atr_pct, 2),
                'stop_loss_pct': round(stop_loss_pct, 2),
                'take_profit_pct': round(take_profit_pct, 2),
                'risk_reward_ratio': round(risk_reward, 2)
            }

        except Exception as e:
            logger.error(f"ATR stop/take profit calculation failed: {e}")
            return None

    async def calculate_entry_timing_score(self, final_score: float, momentum_score: float) -> dict:
        """
        진입 타이밍 점수 계산 (Phase Agent)

        Phase 3.12: Changed from 2-week to 90-day trend for mid-term horizon alignment

        공식:
            entry_timing = final_score_level × 0.4
                         + final_score_trend × 0.3
                         + price_position × 0.2
                         + momentum_confirm × 0.1

        Args:
            final_score: 현재 final_score
            momentum_score: 현재 momentum_score

        Returns:
            dict: {
                'entry_timing_score': 진입 타이밍 점수 (0-100),
                'score_trend_90d': 90일간 final_score 변화,
                'price_position_52w': 52주 고저 대비 위치 (0-100%)
            }
        """
        try:
            # 1. 90일 전 final_score 조회 (Phase 3.12: 2주 → 90일)
            score_query = """
            SELECT final_score
            FROM kr_stock_grade
            WHERE symbol = $1
                AND date <= COALESCE($2::date, CURRENT_DATE) - INTERVAL '90 days'
            ORDER BY date DESC
            LIMIT 1
            """
            score_result = await self.execute_query(score_query, self.symbol, self.analysis_date)
            score_90d_ago = float(score_result[0]['final_score']) if score_result and score_result[0]['final_score'] else final_score

            # 2. 52주 고저 대비 위치
            position_query = """
            SELECT
                close as current_close,
                MIN(close) OVER () as low_52w,
                MAX(close) OVER () as high_52w
            FROM kr_intraday_total
            WHERE symbol = $1
                AND date >= COALESCE($2::date, CURRENT_DATE) - INTERVAL '252 days'
                AND date <= COALESCE($2::date, CURRENT_DATE)
                AND close IS NOT NULL
            ORDER BY date DESC
            LIMIT 1
            """
            position_result = await self.execute_query(position_query, self.symbol, self.analysis_date)

            if not position_result:
                return None

            current = float(position_result[0]['current_close'])
            low_52w = float(position_result[0]['low_52w'])
            high_52w = float(position_result[0]['high_52w'])

            # 52주 고저 대비 위치 (0-100%)
            price_range = high_52w - low_52w
            price_position = ((current - low_52w) / price_range * 100) if price_range > 0 else 50

            # 점수 계산 (Phase 3.12: 90일 기준)
            score_level = final_score  # 0-100
            score_trend = final_score - score_90d_ago  # -30 ~ +30 예상 (90일 변화폭 증가)
            trend_normalized = min(100, max(0, 50 + score_trend * 1.5))  # 0-100 정규화 (계수 조정)
            position_score = 100 - price_position  # 저점일수록 높은 점수
            momentum_confirm = 70 if momentum_score > 50 else 30

            # 가중 평균
            entry_timing = (
                score_level * 0.4 +
                trend_normalized * 0.3 +
                position_score * 0.2 +
                momentum_confirm * 0.1
            )

            logger.info(f"Entry Timing 90d: {entry_timing:.1f} (level={score_level:.1f}, trend_90d={score_trend:+.1f}, pos={price_position:.1f}%)")

            return {
                'entry_timing_score': round(entry_timing, 1),
                'score_trend_90d': round(score_trend, 2),
                'price_position_52w': round(price_position, 2)
            }

        except Exception as e:
            logger.error(f"Entry timing score calculation failed: {e}")
            return None

    async def calculate_position_size(self, volatility_annual: float = None) -> float:
        """
        변동성 기반 포지션 사이징 (Phase Agent)

        공식:
            base_allocation = 5%
            volatility_factor = 시장평균변동성 / 종목변동성
            position_size = base_allocation × volatility_factor

        제한: 최소 1%, 최대 10%

        Args:
            volatility_annual: 연환산 변동성 (없으면 계산)

        Returns:
            float: 권장 포트폴리오 비중 %
        """
        try:
            # 변동성 가져오기
            if volatility_annual is None:
                volatility_annual = await self.calculate_volatility_annual()

            if volatility_annual is None or volatility_annual <= 0:
                return 3.0  # 기본값

            # 시장 평균 변동성 (KOSPI 기준 약 20%)
            market_vol = 20.0

            # 변동성 역가중
            volatility_factor = market_vol / volatility_annual

            # 기본 배분 5%
            base_allocation = 5.0
            position_size = base_allocation * volatility_factor

            # 범위 제한 (1% ~ 10%)
            position_size = min(10.0, max(1.0, position_size))

            logger.info(f"Position Size: {position_size:.2f}% (vol={volatility_annual:.1f}%, factor={volatility_factor:.2f})")

            return round(position_size, 2)

        except Exception as e:
            logger.error(f"Position size calculation failed: {e}")
            return 3.0  # 기본값

    async def calculate_consecutive_trading_days(self) -> dict:
        """
        Calculate consecutive net buying/selling days for foreign and institutional investors.
        Also retrieves foreign ownership rate and recent block trade info.

        Returns:
            dict: {
                'foreign_consecutive': int (positive=net buy, negative=net sell),
                'inst_consecutive': int (positive=net buy, negative=net sell),
                'foreign_ownership_rate': float (current foreign ownership %),
                'foreign_ownership_change_30d': float (30-day change in ownership rate),
                'recent_block_trade': bool (block trade > 3% in last 5 days)
            }
        """
        try:
            result = {
                'foreign_consecutive': 0,
                'inst_consecutive': 0,
                'foreign_ownership_rate': 0.0,
                'foreign_ownership_change_30d': 0.0,
                'recent_block_trade': False
            }

            # 1. Calculate consecutive trading days (last 30 days)
            consecutive_query = """
                SELECT date, foreign_net_volume, inst_net_volume
                FROM kr_individual_investor_daily_trading
                WHERE symbol = $1
                ORDER BY date DESC
                LIMIT 30
            """
            rows = await self.db_manager.execute_query(consecutive_query, self.symbol)

            if rows:
                # Calculate foreign consecutive days
                foreign_consecutive = 0
                for row in rows:
                    foreign_net = row['foreign_net_volume'] or 0
                    if foreign_consecutive == 0:
                        # First day determines direction
                        if foreign_net > 0:
                            foreign_consecutive = 1
                        elif foreign_net < 0:
                            foreign_consecutive = -1
                    elif foreign_consecutive > 0 and foreign_net > 0:
                        foreign_consecutive += 1
                    elif foreign_consecutive < 0 and foreign_net < 0:
                        foreign_consecutive -= 1
                    else:
                        break
                result['foreign_consecutive'] = foreign_consecutive

                # Calculate institutional consecutive days
                inst_consecutive = 0
                for row in rows:
                    inst_net = row['inst_net_volume'] or 0
                    if inst_consecutive == 0:
                        if inst_net > 0:
                            inst_consecutive = 1
                        elif inst_net < 0:
                            inst_consecutive = -1
                    elif inst_consecutive > 0 and inst_net > 0:
                        inst_consecutive += 1
                    elif inst_consecutive < 0 and inst_net < 0:
                        inst_consecutive -= 1
                    else:
                        break
                result['inst_consecutive'] = inst_consecutive

            # 2. Get foreign ownership rate and 30-day change
            ownership_query = """
                WITH recent AS (
                    SELECT foreign_rate
                    FROM kr_foreign_ownership
                    WHERE symbol = $1
                    ORDER BY date DESC
                    LIMIT 1
                ),
                past AS (
                    SELECT foreign_rate
                    FROM kr_foreign_ownership
                    WHERE symbol = $1
                    AND date <= CURRENT_DATE - INTERVAL '30 days'
                    ORDER BY date DESC
                    LIMIT 1
                )
                SELECT
                    (SELECT foreign_rate FROM recent) as current_rate,
                    (SELECT foreign_rate FROM past) as past_rate
            """
            ownership_result = await self.db_manager.execute_query(ownership_query, self.symbol)
            if ownership_result and ownership_result[0]:
                current_rate = float(ownership_result[0]['current_rate'] or 0)
                past_rate = float(ownership_result[0]['past_rate'] or current_rate)
                result['foreign_ownership_rate'] = current_rate
                result['foreign_ownership_change_30d'] = current_rate - past_rate

            # 3. Check for recent block trades (last 5 days, > 3%)
            block_query = """
                SELECT COUNT(*) as block_count
                FROM kr_blocktrades
                WHERE symbol = $1
                AND date >= CURRENT_DATE - INTERVAL '5 days'
                AND block_volume_rate >= 3.0
            """
            block_result = await self.db_manager.execute_query(block_query, self.symbol)
            if block_result and block_result[0]:
                result['recent_block_trade'] = int(block_result[0]['block_count'] or 0) > 0

            logger.info(f"Consecutive trading days: foreign={result['foreign_consecutive']}, "
                       f"inst={result['inst_consecutive']}, ownership={result['foreign_ownership_rate']:.2f}%")

            return result

        except Exception as e:
            logger.error(f"Consecutive trading days calculation failed: {e}")
            return {
                'foreign_consecutive': 0,
                'inst_consecutive': 0,
                'foreign_ownership_rate': 0.0,
                'foreign_ownership_change_30d': 0.0,
                'recent_block_trade': False
            }

    async def calculate_technical_trigger_conditions(self) -> dict:
        """
        Calculate technical indicator conditions for trigger generation.

        Returns:
            dict: {
                'rsi_value': float,
                'rsi_oversold': bool (RSI < 30),
                'rsi_overbought': bool (RSI > 70),
                'macd_bullish': bool (MACD > Signal),
                'macd_cross_up': bool (recent golden cross),
                'bb_lower_touch': bool (price near lower band),
                'bb_upper_touch': bool (price near upper band),
                'near_52w_high': bool (within 5% of 52-week high),
                'near_52w_low': bool (within 5% of 52-week low),
                'volume_surge': bool (volume > 200% of 20-day avg),
                'volume_ratio': float (current vs 20-day avg)
            }
        """
        try:
            result = {
                'rsi_value': 50.0,
                'rsi_oversold': False,
                'rsi_overbought': False,
                'macd_bullish': False,
                'macd_cross_up': False,
                'bb_lower_touch': False,
                'bb_upper_touch': False,
                'near_52w_high': False,
                'near_52w_low': False,
                'volume_surge': False,
                'volume_ratio': 1.0
            }

            # 1. Get latest technical indicators
            indicator_query = """
                SELECT rsi, macd, macd_signal,
                       real_upper_band, real_middle_band, real_lower_band
                FROM kr_indicators
                WHERE symbol = $1
                ORDER BY date DESC
                LIMIT 2
            """
            indicators = await self.db_manager.execute_query(indicator_query, self.symbol)

            if indicators and len(indicators) >= 1:
                latest = indicators[0]
                rsi = float(latest['rsi'] or 50)
                macd = float(latest['macd'] or 0)
                macd_signal = float(latest['macd_signal'] or 0)
                bb_upper = float(latest['real_upper_band'] or 0)
                bb_middle = float(latest['real_middle_band'] or 0)
                bb_lower = float(latest['real_lower_band'] or 0)

                result['rsi_value'] = rsi
                result['rsi_oversold'] = rsi < 30
                result['rsi_overbought'] = rsi > 70
                result['macd_bullish'] = macd > macd_signal

                # Check for MACD golden cross (today bullish, yesterday bearish)
                if len(indicators) >= 2:
                    prev = indicators[1]
                    prev_macd = float(prev['macd'] or 0)
                    prev_signal = float(prev['macd_signal'] or 0)
                    if macd > macd_signal and prev_macd <= prev_signal:
                        result['macd_cross_up'] = True

            # 2. Get current price and check Bollinger Band position
            price_query = """
                SELECT close
                FROM kr_intraday_total
                WHERE symbol = $1
                ORDER BY date DESC
                LIMIT 1
            """
            price_result = await self.db_manager.execute_query(price_query, self.symbol)
            if price_result and indicators:
                current_price = float(price_result[0]['close'] or 0)
                if bb_lower > 0 and bb_upper > 0:
                    # Within 2% of lower band
                    if current_price <= bb_lower * 1.02:
                        result['bb_lower_touch'] = True
                    # Within 2% of upper band
                    if current_price >= bb_upper * 0.98:
                        result['bb_upper_touch'] = True

            # 3. Check 52-week high/low proximity
            high_low_query = """
                SELECT MAX(high) as high_52w, MIN(low) as low_52w
                FROM kr_intraday_total
                WHERE symbol = $1
                AND date >= CURRENT_DATE - INTERVAL '252 days'
            """
            hl_result = await self.db_manager.execute_query(high_low_query, self.symbol)
            if hl_result and hl_result[0] and price_result:
                high_52w = float(hl_result[0]['high_52w'] or 0)
                low_52w = float(hl_result[0]['low_52w'] or 0)
                current_price = float(price_result[0]['close'] or 0)

                if high_52w > 0:
                    # Within 5% of 52-week high
                    result['near_52w_high'] = current_price >= high_52w * 0.95
                if low_52w > 0:
                    # Within 5% of 52-week low
                    result['near_52w_low'] = current_price <= low_52w * 1.05

            # 4. Check volume surge (vs 20-day average)
            volume_query = """
                WITH recent_vol AS (
                    SELECT volume
                    FROM kr_intraday_total
                    WHERE symbol = $1
                    ORDER BY date DESC
                    LIMIT 1
                ),
                avg_vol AS (
                    SELECT AVG(volume) as avg_volume
                    FROM kr_intraday_total
                    WHERE symbol = $1
                    AND date >= CURRENT_DATE - INTERVAL '20 days'
                )
                SELECT
                    (SELECT volume FROM recent_vol) as current_volume,
                    (SELECT avg_volume FROM avg_vol) as avg_volume
            """
            vol_result = await self.db_manager.execute_query(volume_query, self.symbol)
            if vol_result and vol_result[0]:
                current_vol = float(vol_result[0]['current_volume'] or 0)
                avg_vol = float(vol_result[0]['avg_volume'] or 1)
                if avg_vol > 0:
                    volume_ratio = current_vol / avg_vol
                    result['volume_ratio'] = round(volume_ratio, 2)
                    result['volume_surge'] = volume_ratio >= 2.0

            logger.info(f"Technical conditions: RSI={result['rsi_value']:.1f}, "
                       f"MACD_bullish={result['macd_bullish']}, volume_ratio={result['volume_ratio']:.2f}")

            return result

        except Exception as e:
            logger.error(f"Technical trigger conditions calculation failed: {e}")
            return {
                'rsi_value': 50.0,
                'rsi_oversold': False,
                'rsi_overbought': False,
                'macd_bullish': False,
                'macd_cross_up': False,
                'bb_lower_touch': False,
                'bb_upper_touch': False,
                'near_52w_high': False,
                'near_52w_low': False,
                'volume_surge': False,
                'volume_ratio': 1.0
            }

    async def generate_action_triggers(
        self,
        final_score: float,
        sector_percentile: float,
        stop_loss_pct: float,
        take_profit_pct: float,
        consecutive_data: dict,
        technical_conditions: dict,
        market_sentiment: str,
        entry_timing_score: float,
        volatility_annual: float,
        max_drawdown_1y: float,
        momentum_score: float,
        value_score: float,
        quality_score: float
    ) -> dict:
        """
        Dynamic Action Trigger Generation (Phase 3.13)

        Generates context-aware triggers based on:
        - Supply/Demand: Foreign/Institutional consecutive trading days
        - Technical: RSI, MACD, Bollinger Bands, 52-week high/low, volume
        - Quant Model: final_score, sector rank, entry timing, factor scores
        - Risk: Stop loss, take profit, volatility, max drawdown
        - Macro: Market sentiment (PANIC/FEAR/NEUTRAL/GREED/OVERHEATED)

        Args:
            final_score: Current final_score (0-100)
            sector_percentile: Sector percentile rank (0-100, lower is better)
            stop_loss_pct: Stop loss percentage (negative)
            take_profit_pct: Take profit percentage (positive)
            consecutive_data: Dict with foreign/inst consecutive days
            technical_conditions: Dict with RSI, MACD, BB, volume conditions
            market_sentiment: Market regime (PANIC/FEAR/NEUTRAL/GREED/OVERHEATED)
            entry_timing_score: Entry timing score (0-100)
            volatility_annual: Annualized volatility (%)
            max_drawdown_1y: 1-year max drawdown (negative %)
            momentum_score: Momentum factor score (0-100)
            value_score: Value factor score (0-100)
            quality_score: Quality factor score (0-100)

        Returns:
            dict: {
                'buy_triggers': List of buy conditions,
                'sell_triggers': List of sell conditions,
                'hold_triggers': List of hold conditions
            }
        """
        try:
            buy_triggers = []
            sell_triggers = []
            hold_triggers = []

            # Extract consecutive data
            foreign_consec = consecutive_data.get('foreign_consecutive', 0)
            inst_consec = consecutive_data.get('inst_consecutive', 0)
            foreign_rate = consecutive_data.get('foreign_ownership_rate', 0)
            foreign_change = consecutive_data.get('foreign_ownership_change_30d', 0)
            recent_block = consecutive_data.get('recent_block_trade', False)

            # Extract technical conditions
            rsi_value = technical_conditions.get('rsi_value', 50)
            rsi_oversold = technical_conditions.get('rsi_oversold', False)
            rsi_overbought = technical_conditions.get('rsi_overbought', False)
            macd_bullish = technical_conditions.get('macd_bullish', False)
            macd_cross_up = technical_conditions.get('macd_cross_up', False)
            bb_lower = technical_conditions.get('bb_lower_touch', False)
            bb_upper = technical_conditions.get('bb_upper_touch', False)
            near_52w_high = technical_conditions.get('near_52w_high', False)
            near_52w_low = technical_conditions.get('near_52w_low', False)
            volume_surge = technical_conditions.get('volume_surge', False)
            volume_ratio = technical_conditions.get('volume_ratio', 1.0)

            # ================================================================
            # BUY TRIGGERS (context-aware)
            # ================================================================

            # 1. Supply/Demand Triggers
            if foreign_consec > 0:
                # Already in net buying streak - trigger for continuation
                buy_triggers.append(f"외국인 {foreign_consec + 5}일 연속 순매수 유지 시")
            elif foreign_consec < 0:
                # In net selling streak - trigger for reversal
                buy_triggers.append("외국인 5일 연속 순매수 전환 시")
            else:
                buy_triggers.append("외국인 순매수 시작 시")

            if inst_consec > 0:
                buy_triggers.append(f"기관 {inst_consec + 5}일 연속 순매수 유지 시")
            elif inst_consec <= 0:
                buy_triggers.append("기관 5일 연속 순매수 전환 시")

            # Foreign ownership threshold
            if foreign_rate < 30:
                buy_triggers.append(f"외국인 지분율 {foreign_rate + 2:.1f}% 돌파 시")

            # 2. Technical Triggers (conditional)
            if rsi_oversold:
                buy_triggers.append(f"RSI {rsi_value:.0f} -> 35 이상 반등 시")
            elif rsi_value < 40:
                buy_triggers.append("RSI 40 이상 회복 시")

            if not macd_bullish:
                buy_triggers.append("MACD 골든크로스 발생 시")

            if bb_lower:
                buy_triggers.append("볼린저밴드 중심선 회복 시")

            if near_52w_low:
                buy_triggers.append("52주 신저가 대비 10% 반등 시")
            elif not near_52w_high:
                buy_triggers.append("52주 신고가 돌파 시")

            if not volume_surge:
                buy_triggers.append("거래량 20일 평균 200% 돌파 시")

            # 3. Quant Model Triggers
            buy_triggers.append(f"final_score {final_score + 5:.0f}점 이상 상승 시")

            if sector_percentile > 10:
                buy_triggers.append(f"섹터 순위 상위 {max(5, sector_percentile - 10):.0f}% 진입 시")

            if entry_timing_score < 80:
                buy_triggers.append(f"Entry Timing {80}점 이상 도달 시")

            # 4. Market Sentiment Adjustments
            if market_sentiment in ['PANIC', 'FEAR']:
                buy_triggers.append("시장 심리 NEUTRAL 이상 회복 시")

            # ================================================================
            # SELL TRIGGERS (context-aware)
            # ================================================================

            # 1. Supply/Demand Triggers
            if foreign_consec > 0:
                sell_triggers.append("외국인 5일 연속 순매도 전환 시")
            elif foreign_consec < 0:
                sell_triggers.append(f"외국인 {abs(foreign_consec) + 5}일 연속 순매도 지속 시")

            if inst_consec > 0:
                sell_triggers.append("기관 5일 연속 순매도 전환 시")
            elif inst_consec < 0:
                sell_triggers.append(f"기관 {abs(inst_consec) + 5}일 연속 순매도 지속 시")

            # 2. Technical Triggers
            if rsi_overbought:
                sell_triggers.append(f"RSI {rsi_value:.0f} -> 65 이하 하락 시")
            elif rsi_value > 60:
                sell_triggers.append("RSI 70 이상 과매수 진입 시")

            if macd_bullish:
                sell_triggers.append("MACD 데드크로스 발생 시")

            if bb_upper:
                sell_triggers.append("볼린저밴드 상단 이탈 후 하락 시")

            if near_52w_high:
                sell_triggers.append("52주 신고가 대비 5% 하락 시")

            # 3. Risk Management Triggers
            sell_triggers.append(f"손절선 {stop_loss_pct:.1f}% 도달 시")
            sell_triggers.append(f"익절선 +{take_profit_pct:.1f}% 도달 시")

            # 4. Quant Model Triggers
            sell_triggers.append(f"final_score {max(0, final_score - 15):.0f}점 이하 하락 시")

            if volatility_annual > 40:
                sell_triggers.append(f"변동성 {volatility_annual + 10:.0f}% 이상 급등 시")

            if max_drawdown_1y and max_drawdown_1y < -25:
                sell_triggers.append(f"MDD {max_drawdown_1y - 5:.0f}% 이하 확대 시")

            # 5. Market Sentiment Adjustments
            if market_sentiment in ['GREED', 'OVERHEATED']:
                sell_triggers.append("시장 심리 OVERHEATED 진입 시 일부 차익실현")

            # ================================================================
            # HOLD TRIGGERS (stability conditions)
            # ================================================================

            # 1. Supply/Demand Stability
            if foreign_consec > 0:
                hold_triggers.append(f"외국인 순매수 기조 유지 시 (현재 {foreign_consec}일)")
            if inst_consec > 0:
                hold_triggers.append(f"기관 순매수 기조 유지 시 (현재 {inst_consec}일)")

            # 2. Score Stability
            hold_triggers.append(f"final_score {max(0, final_score - 5):.0f}~{min(100, final_score + 5):.0f}점 유지 시")

            # 3. Technical Stability
            if 40 <= rsi_value <= 60:
                hold_triggers.append(f"RSI 40~60 중립구간 유지 시 (현재 {rsi_value:.0f})")

            if macd_bullish:
                hold_triggers.append("MACD 상승 추세 유지 시")

            # 4. Volatility Stability
            if volatility_annual <= 35:
                hold_triggers.append(f"변동성 35% 이내 유지 시 (현재 {volatility_annual:.1f}%)")
            else:
                hold_triggers.append("변동성 안정화 시")

            # 5. Sector Position
            hold_triggers.append(f"섹터 순위 상위 {sector_percentile + 10:.0f}% 이내 유지 시")

            # 6. Market Stability
            if market_sentiment == 'NEUTRAL':
                hold_triggers.append("시장 심리 NEUTRAL 유지 시")

            logger.info(f"Generated dynamic triggers: buy={len(buy_triggers)}, "
                       f"sell={len(sell_triggers)}, hold={len(hold_triggers)}")

            return {
                'buy_triggers': buy_triggers,
                'sell_triggers': sell_triggers,
                'hold_triggers': hold_triggers
            }

        except Exception as e:
            logger.error(f"Trigger generation failed: {e}")
            return {
                'buy_triggers': ["데이터 부족으로 트리거 생성 불가"],
                'sell_triggers': ["데이터 부족으로 트리거 생성 불가"],
                'hold_triggers': ["데이터 부족으로 트리거 생성 불가"]
            }

    # ========================================================================
    # Macro Probability Mapping (based on market_sentiment)
    # OVERHEATED, GREED, NEUTRAL, FEAR, PANIC
    # ========================================================================
    MACRO_PROBABILITY = {
        'OVERHEATED': {'bullish': 0.30, 'sideways': 0.40, 'bearish': 0.30},  # High risk of correction
        'GREED': {'bullish': 0.50, 'sideways': 0.35, 'bearish': 0.15},       # Bull market
        'NEUTRAL': {'bullish': 0.35, 'sideways': 0.40, 'bearish': 0.25},     # Base rate
        'FEAR': {'bullish': 0.25, 'sideways': 0.35, 'bearish': 0.40},        # Cautious
        'PANIC': {'bullish': 0.20, 'sideways': 0.30, 'bearish': 0.50}        # High downside risk
    }

    async def calculate_scenario_probability(
        self,
        final_score: float,
        industry: str,
        market_sentiment: str = None,
        precomputed_scenario_stats: dict = None
    ) -> dict:
        """
        HYBRID 시나리오 확률 계산 (Phase 3.10 + Phase 3.12)

        Phase 3.12: Changed from 60-66 days to 90-96 days backtest horizon

        Macro (Top-Down) + Backtest (Bottom-Up) 결합:
        - Step 1: Macro 확률 가져오기 (market_sentiment 기반)
        - Step 2: Backtest 확률 가져오기 (90일 후 수익률 기반)
        - Step 3: Confidence 기반 가중치 계산
          - w_backtest = min(0.7, sample_count / 50)
          - w_macro = 1 - w_backtest
        - Step 4: 기존 Calibrator 적용

        Optimization (2,763 queries -> 0):
        - precomputed_scenario_stats가 있으면 사용 (배치 처리 시)
        - 없으면 기존 DB 쿼리로 fallback (단일 종목 분석 시)

        Args:
            final_score: 현재 final_score
            industry: 업종
            market_sentiment: 시장 심리 (OVERHEATED/GREED/NEUTRAL/FEAR/PANIC)
            precomputed_scenario_stats: 사전 계산된 시나리오 통계 {industry: {score_bucket: stats}}

        Returns:
            dict: 시나리오 확률 및 90일 후 예상 수익률
        """
        try:
            # ================================================================
            # Step 1: Get Macro-based probability (Top-Down)
            # ================================================================
            if market_sentiment and market_sentiment in self.MACRO_PROBABILITY:
                macro_prob = self.MACRO_PROBABILITY[market_sentiment]
            else:
                # Fallback to NEUTRAL
                macro_prob = self.MACRO_PROBABILITY['NEUTRAL']
                market_sentiment = 'NEUTRAL'

            macro_bull = macro_prob['bullish']
            macro_bear = macro_prob['bearish']
            macro_sideways = macro_prob['sideways']

            # ================================================================
            # Step 2: Get Backtest-based probability (Bottom-Up)
            # Use precomputed stats if available, otherwise query DB
            # ================================================================
            backtest_stats = None
            score_bucket = int(final_score // 10) * 10  # 10점 단위 버킷

            # Try precomputed stats first
            if precomputed_scenario_stats and industry in precomputed_scenario_stats:
                industry_stats = precomputed_scenario_stats[industry]
                # Try exact bucket, then adjacent buckets
                if score_bucket in industry_stats:
                    backtest_stats = industry_stats[score_bucket]
                elif score_bucket - 10 in industry_stats:
                    backtest_stats = industry_stats[score_bucket - 10]
                elif score_bucket + 10 in industry_stats:
                    backtest_stats = industry_stats[score_bucket + 10]

            # Fallback to DB query if no precomputed stats
            # Phase 3.12: Changed from 60-66 days to 90-96 days for mid-term horizon alignment
            if backtest_stats is None:
                query = """
                WITH similar_cases AS (
                    SELECT
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
                            AND date >= g.date + INTERVAL '90 days'
                            AND date <= g.date + INTERVAL '96 days'
                        ORDER BY date
                        LIMIT 1
                    ) future_price ON true
                    WHERE g.final_score BETWEEN $1 - 5 AND $1 + 5
                        AND d.industry = $2
                        AND g.date < CURRENT_DATE - INTERVAL '93 days'
                        AND future_price.close IS NOT NULL
                )
                SELECT
                    COUNT(*) as total_count,
                    COUNT(*) FILTER (WHERE return_3m > 10) as bullish_count,
                    COUNT(*) FILTER (WHERE return_3m BETWEEN -10 AND 10) as sideways_count,
                    COUNT(*) FILTER (WHERE return_3m < -10) as bearish_count,
                    ROUND(AVG(return_3m)::numeric, 2) as avg_return,
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
                """

                result = await self.execute_query(query, final_score, industry)

                if result and result[0]['total_count'] and result[0]['total_count'] >= 10:
                    backtest_stats = {
                        'total_count': result[0]['total_count'],
                        'bullish_count': result[0]['bullish_count'] or 0,
                        'sideways_count': result[0]['sideways_count'] or 0,
                        'bearish_count': result[0]['bearish_count'] or 0,
                        'avg_return': float(result[0]['avg_return']) if result[0]['avg_return'] else None,
                        'bullish_lower': result[0]['bullish_lower'] or 10,
                        'bullish_upper': result[0]['bullish_upper'] or 25,
                        'bearish_lower': result[0]['bearish_lower'] or -20,
                        'bearish_upper': result[0]['bearish_upper'] or -10,
                        'sideways_lower': result[0]['sideways_lower'] or -5,
                        'sideways_upper': result[0]['sideways_upper'] or 5
                    }

            # Extract backtest probabilities
            if backtest_stats and backtest_stats.get('total_count', 0) >= 5:
                total = backtest_stats['total_count']
                bullish = backtest_stats['bullish_count']
                sideways = backtest_stats['sideways_count']
                bearish = backtest_stats['bearish_count']

                backtest_bull = bullish / total
                backtest_bear = bearish / total
                backtest_sideways = sideways / total
                sample_count = total

                # Return ranges from backtest
                bullish_lower = backtest_stats.get('bullish_lower', 10)
                bullish_upper = backtest_stats.get('bullish_upper', 25)
                bearish_lower = backtest_stats.get('bearish_lower', -20)
                bearish_upper = backtest_stats.get('bearish_upper', -10)
                sideways_lower = backtest_stats.get('sideways_lower', -5)
                sideways_upper = backtest_stats.get('sideways_upper', 5)

                # Average return for bearish cap (Phase 3.9.1)
                backtest_avg_return = backtest_stats.get('avg_return', None)
            else:
                # No backtest data available
                backtest_bull = 0.33
                backtest_bear = 0.33
                backtest_sideways = 0.34
                sample_count = 0
                bullish_lower, bullish_upper = 10, 25
                bearish_lower, bearish_upper = -20, -10
                sideways_lower, sideways_upper = -5, 5
                backtest_avg_return = None
                logger.warning(f"Insufficient backtest samples (industry={industry})")

            # ================================================================
            # Step 3: Calculate confidence-based weights
            # More samples = more trust in backtest (max 70%)
            # ================================================================
            if sample_count > 0:
                w_backtest = min(0.70, sample_count / 50)
            else:
                w_backtest = 0.0  # No backtest data, use macro only
            w_macro = 1.0 - w_backtest

            # ================================================================
            # Step 4: Combine probabilities with weighted average
            # ================================================================
            hybrid_bull = w_macro * macro_bull + w_backtest * backtest_bull
            hybrid_bear = w_macro * macro_bear + w_backtest * backtest_bear
            hybrid_sideways = 1.0 - hybrid_bull - hybrid_bear

            # Convert to percentage (0-100)
            raw_bullish = round(hybrid_bull * 100)
            raw_bearish = round(hybrid_bear * 100)
            raw_sideways = 100 - raw_bullish - raw_bearish

            # ================================================================
            # Step 5: Apply existing calibration (theme, regime adjustments)
            # ================================================================
            theme_query = """
            SELECT theme FROM kr_stock_detail WHERE symbol = $1
            """
            theme_result = await self.execute_query(theme_query, self.symbol)
            theme = theme_result[0]['theme'] if theme_result and theme_result[0]['theme'] else 'Others'

            # Apply calibration (Phase 3.9.1: added backtest_avg_return for bearish cap)
            calibrated = self.calibrator.calibrate_probabilities(
                raw_bullish=raw_bullish,
                raw_bearish=raw_bearish,
                raw_sideways=raw_sideways,
                final_score=final_score,
                theme=theme,
                backtest_avg_return=backtest_avg_return
            )

            bullish_prob = calibrated['bullish']
            sideways_prob = calibrated['sideways']
            bearish_prob = calibrated['bearish']

            logger.info(f"Hybrid prob - macro_w={w_macro:.2f}, backtest_w={w_backtest:.2f}, "
                       f"samples={sample_count}, sentiment={market_sentiment}, precomputed={'Yes' if precomputed_scenario_stats else 'No'}")
            logger.info(f"Scenario hybrid: bullish={raw_bullish}%, sideways={raw_sideways}%, bearish={raw_bearish}%")
            logger.info(f"Scenario calibrated: bullish={bullish_prob}%, sideways={sideways_prob}%, bearish={bearish_prob}% [theme={theme}]")

            # Build scenario return strings (Phase 3.9.1: use theme-specific P25~P75 ranges)
            # Priority: Theme-specific ranges > Backtest ranges > Default ranges
            theme_bull_range = self.calibrator.get_return_range(theme, 'bullish')
            theme_side_range = self.calibrator.get_return_range(theme, 'sideways')
            theme_bear_range = self.calibrator.get_return_range(theme, 'bearish')

            # Use theme-specific ranges if available, otherwise fall back to backtest data
            final_bull_lower = theme_bull_range[0] if theme else bullish_lower
            final_bull_upper = theme_bull_range[1] if theme else bullish_upper
            final_side_lower = theme_side_range[0] if theme else sideways_lower
            final_side_upper = theme_side_range[1] if theme else sideways_upper
            final_bear_lower = theme_bear_range[0] if theme else bearish_lower
            final_bear_upper = theme_bear_range[1] if theme else bearish_upper

            scenario_bullish_return = f"+{final_bull_lower:.0f}~+{final_bull_upper:.0f}%"
            scenario_sideways_return = f"{final_side_lower:+.0f}~{final_side_upper:+.0f}%"
            scenario_bearish_return = f"{final_bear_upper:.0f}~{final_bear_lower:.0f}%"

            return {
                'scenario_bullish_prob': bullish_prob,
                'scenario_sideways_prob': sideways_prob,
                'scenario_bearish_prob': bearish_prob,
                'scenario_bullish_return': scenario_bullish_return,
                'scenario_sideways_return': scenario_sideways_return,
                'scenario_bearish_return': scenario_bearish_return,
                'scenario_sample_count': sample_count,
                'macro_weight': round(w_macro, 2),
                'backtest_weight': round(w_backtest, 2),
                'market_sentiment': market_sentiment
            }

        except Exception as e:
            logger.error(f"Scenario probability calculation failed: {e}")
            return {
                'scenario_bullish_prob': 33,
                'scenario_sideways_prob': 34,
                'scenario_bearish_prob': 33,
                'scenario_bullish_return': "+10~+20%",
                'scenario_sideways_return': "-5~+5%",
                'scenario_bearish_return': "-10~-15%",
                'scenario_sample_count': 0,
                'macro_weight': None,
                'backtest_weight': None,
                'market_sentiment': None
            }

    async def calculate_investor_flow(self) -> Tuple[Optional[int], Optional[int]]:
        """
        5. Calculate institutional and foreign investor flow (30 days)
        Net buying volume in shares

        Storage: inst_net_30d BIGINT, foreign_net_30d BIGINT
        """
        try:
            # Phase 3.10: Use prefetched data if available
            if self.prefetch_calc:
                inst_net, foreign_net = self.prefetch_calc.calculate_investor_flow_30d()
                if inst_net is not None or foreign_net is not None:
                    logger.info(f"30-day flow (prefetched) - Institution: {inst_net:,}, Foreign: {foreign_net:,}")
                    return inst_net, foreign_net

            # Fallback to DB query
            query = """
            SELECT
                SUM(inst_net_volume) as inst_net_30d,
                SUM(foreign_net_volume) as foreign_net_30d
            FROM kr_individual_investor_daily_trading
            WHERE symbol = $1
                AND date >= COALESCE($2::date, CURRENT_DATE) - INTERVAL '30 days'
                AND date <= COALESCE($2::date, CURRENT_DATE)
            """

            result = await self.execute_query(query, self.symbol, self.analysis_date)

            if not result:
                return None, None

            inst_net = result[0]['inst_net_30d']
            foreign_net = result[0]['foreign_net_30d']

            inst_net = int(inst_net) if inst_net else 0
            foreign_net = int(foreign_net) if foreign_net else 0

            logger.info(f"30-day flow - Institution: {inst_net:,} shares, Foreign: {foreign_net:,} shares")
            return inst_net, foreign_net

        except Exception as e:
            logger.error(f"Investor flow calculation failed: {e}")
            return None, None

    async def calculate_factor_momentum(self) -> Dict[str, Optional[float]]:
        """
        6. Calculate factor momentum (trend)
        Based on factor score time series if available, otherwise use current vs benchmark

        Storage: value_momentum, quality_momentum, momentum_momentum, growth_momentum FLOAT

        Returns simple indicator: positive (growing), negative (declining), 0 (neutral)
        """
        try:
            # For now, use simplified approach based on current factor scores vs 50 (neutral)
            # In full implementation, would compare current vs historical factor scores

            factor_momentum = {}

            for factor_name in ['value', 'quality', 'momentum', 'growth']:
                factor_data = self.factor_scores.get(factor_name, {})
                weighted_result = factor_data.get('weighted_result')

                if weighted_result:
                    score = weighted_result['weighted_score']
                    # Momentum indicator: score > 50 = positive, < 50 = negative
                    momentum = score - 50
                    factor_momentum[f'{factor_name}_momentum'] = round(momentum, 2)
                else:
                    factor_momentum[f'{factor_name}_momentum'] = None

            logger.info(f"Factor momentum: {factor_momentum}")
            return factor_momentum

        except Exception as e:
            logger.error(f"Factor momentum calculation failed: {e}")
            return {
                'value_momentum': None,
                'quality_momentum': None,
                'momentum_momentum': None,
                'growth_momentum': None
            }

    async def calculate_industry_ranking(self) -> Tuple[Optional[int], Optional[float]]:
        """
        7. Calculate industry ranking
        Rank within same industry based on final score (if available) or market cap

        Storage: industry_rank INTEGER, industry_percentile FLOAT
        """
        try:
            # Get stock's industry
            query_industry = """
            SELECT industry
            FROM kr_stock_detail
            WHERE symbol = $1
            """

            result = await self.execute_query(query_industry, self.symbol)

            if not result or not result[0]['industry']:
                return None, None

            industry = result[0]['industry']

            # Rank by market cap within industry (since we don't have historical scores yet)
            query_rank = """
            WITH industry_stocks AS (
                SELECT
                    sd.symbol,
                    it.market_cap,
                    ROW_NUMBER() OVER (ORDER BY it.market_cap DESC NULLS LAST) as rank,
                    COUNT(*) OVER () as total_count
                FROM kr_stock_detail sd
                JOIN kr_intraday_total it ON sd.symbol = it.symbol
                WHERE sd.industry = $1
                    AND it.date = COALESCE($3::date, (SELECT MAX(date) FROM kr_intraday_total))
                    AND it.market_cap IS NOT NULL
            )
            SELECT
                rank,
                total_count,
                ((total_count - rank + 1)::NUMERIC / total_count * 100) as percentile
            FROM industry_stocks
            WHERE symbol = $2
            """

            result = await self.db_manager.execute_query(query_rank, industry, self.symbol, self.analysis_date)

            if not result:
                return None, None

            rank = int(result[0]['rank'])
            total = int(result[0]['total_count'])
            percentile = float(result[0]['percentile'])

            logger.info(f"Industry ranking: {rank}/{total} (top {100-percentile:.1f}%)")
            return rank, round(percentile, 1)

        except Exception as e:
            logger.error(f"Industry ranking calculation failed: {e}")
            return None, None

    async def calculate_max_drawdown_1y(self) -> Optional[float]:
        """
        8. Calculate maximum drawdown over past 1 year

        Storage: max_drawdown_1y FLOAT (negative value, %)
        """
        try:
            # Phase 3.10: Use prefetched data if available
            if self.prefetch_calc:
                result = self.prefetch_calc.calculate_max_drawdown_1y()
                if result is not None:
                    logger.info(f"Max drawdown (1Y): {result:.2f}% (prefetched)")
                    return result

            # Fallback to DB query
            query = """
            WITH daily_prices AS (
                SELECT
                    date,
                    close,
                    MAX(close) OVER (ORDER BY date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) as running_max
                FROM kr_intraday_total
                WHERE symbol = $1
                    AND date >= COALESCE($2::date, CURRENT_DATE) - INTERVAL '252 days'
                    AND date <= COALESCE($2::date, CURRENT_DATE)
                    AND close IS NOT NULL
                ORDER BY date
            )
            SELECT
                MIN(((close - running_max)::NUMERIC / NULLIF(running_max, 0) * 100)) as max_drawdown
            FROM daily_prices
            """

            result = await self.execute_query(query, self.symbol, self.analysis_date)

            if not result or result[0]['max_drawdown'] is None:
                return None

            mdd = float(result[0]['max_drawdown'])

            logger.info(f"Max drawdown (1Y): {mdd:.2f}%")
            return round(mdd, 2)

        except Exception as e:
            logger.error(f"Max drawdown calculation failed: {e}")
            return None

    async def calculate_beta(self) -> Optional[float]:
        """
        9. Calculate beta vs market index (KOSPI or KOSDAQ)

        Storage: beta FLOAT
        """
        try:
            # Get stock's exchange
            query_exchange = """
            SELECT exchange
            FROM kr_stock_detail
            WHERE symbol = $1
            """

            result = await self.execute_query(query_exchange, self.symbol)

            if not result or not result[0]['exchange']:
                return None

            exchange = result[0]['exchange']

            # Use appropriate index
            # For now, estimate beta using volatility ratio as proxy
            # In full implementation, would use regression against index returns

            # Get stock volatility
            query_stock_vol = """
            WITH daily_returns AS (
                SELECT
                    ((close - LAG(close) OVER (ORDER BY date))::NUMERIC /
                     NULLIF(LAG(close) OVER (ORDER BY date), 0)) as daily_return
                FROM kr_intraday_total
                WHERE symbol = $1
                    AND date >= COALESCE($2::date, CURRENT_DATE) - INTERVAL '90 days'
                    AND date <= COALESCE($2::date, CURRENT_DATE)
                    AND close IS NOT NULL
                ORDER BY date
            )
            SELECT STDDEV(daily_return) as stock_std
            FROM daily_returns
            WHERE daily_return IS NOT NULL
            """

            result = await self.execute_query(query_stock_vol, self.symbol, self.analysis_date)

            if not result or not result[0]['stock_std']:
                return None

            stock_std = float(result[0]['stock_std'])

            # Estimate market std (simplified: KOSPI ~1.5%, KOSDAQ ~2.0% daily)
            market_std = 0.015 if exchange == 'KOSPI' else 0.020

            # Beta estimate = stock_std / market_std
            beta = stock_std / market_std

            logger.info(f"Beta (estimated): {beta:.2f} vs {exchange}")
            return round(beta, 2)

        except Exception as e:
            logger.error(f"Beta calculation failed: {e}")
            return None

    async def calculate_support_resistance(self) -> Dict[str, Optional[float]]:
        """
        10. Calculate support and resistance levels (2 levels each)
        Using Pivot Points + Recent highs/lows method

        Storage: support_1, support_2, resistance_1, resistance_2 DECIMAL(21,2)

        Returns:
            dict with keys: support_1, support_2, resistance_1, resistance_2
        """
        try:
            # Phase 3.10: Use prefetched data if available
            if self.prefetch_calc:
                result = self.prefetch_calc.calculate_support_resistance()
                if result and any(v is not None for v in result.values()):
                    logger.info(f"Support/Resistance (prefetched): S1={result['support_1']}, R1={result['resistance_1']}")
                    return result

            # Fallback to DB query
            # Get recent price data (20 days for short-term, 60 days for mid-term levels)
            query = """
            WITH recent_prices AS (
                SELECT
                    high,
                    low,
                    close
                FROM kr_intraday_total
                WHERE symbol = $1
                    AND date >= COALESCE($2::date, CURRENT_DATE) - INTERVAL '60 days'
                    AND date <= COALESCE($2::date, CURRENT_DATE)
                    AND high IS NOT NULL
                    AND low IS NOT NULL
                    AND close IS NOT NULL
                ORDER BY date DESC
                LIMIT 60
            ),
            latest_price AS (
                SELECT high, low, close
                FROM kr_intraday_total
                WHERE symbol = $1
                    AND ($2::date IS NULL OR date = $2)
                ORDER BY date DESC
                LIMIT 1
            ),
            price_stats AS (
                SELECT
                    MAX(high) as high_60d,
                    MIN(low) as low_60d,
                    (SELECT MAX(high) FROM (SELECT high FROM recent_prices LIMIT 20) t) as high_20d,
                    (SELECT MIN(low) FROM (SELECT low FROM recent_prices LIMIT 20) t) as low_20d,
                    (SELECT high FROM latest_price) as latest_high,
                    (SELECT low FROM latest_price) as latest_low,
                    (SELECT close FROM latest_price) as latest_close
                FROM recent_prices
            )
            SELECT
                latest_high,
                latest_low,
                latest_close,
                high_20d,
                low_20d,
                high_60d,
                low_60d
            FROM price_stats
            """

            result = await self.execute_query(query, self.symbol, self.analysis_date)

            if not result or not result[0]['latest_close']:
                return {
                    'support_1': None,
                    'support_2': None,
                    'resistance_1': None,
                    'resistance_2': None
                }

            data = result[0]
            high = float(data['latest_high'])
            low = float(data['latest_low'])
            close = float(data['latest_close'])
            high_20d = float(data['high_20d'])
            low_20d = float(data['low_20d'])
            high_60d = float(data['high_60d'])
            low_60d = float(data['low_60d'])

            # Calculate Pivot Point
            pivot = (high + low + close) / 3

            # Calculate support and resistance using Pivot Points
            resistance_pivot_1 = (2 * pivot) - low
            resistance_pivot_2 = pivot + (high - low)
            support_pivot_1 = (2 * pivot) - high
            support_pivot_2 = pivot - (high - low)

            # Combine with recent highs/lows for more realistic levels
            # Resistance: use max of pivot calculation and recent highs
            resistance_1 = round(max(resistance_pivot_1, high_20d), 2)
            resistance_2 = round(max(resistance_pivot_2, high_60d), 2)

            # Support: use min of pivot calculation and recent lows
            support_1 = round(min(support_pivot_1, low_20d), 2)
            support_2 = round(min(support_pivot_2, low_60d), 2)

            logger.info(f"Support levels: S1={support_1}, S2={support_2}")
            logger.info(f"Resistance levels: R1={resistance_1}, R2={resistance_2}")

            return {
                'support_1': support_1,
                'support_2': support_2,
                'resistance_1': resistance_1,
                'resistance_2': resistance_2
            }

        except Exception as e:
            logger.error(f"Support/Resistance calculation failed: {e}")
            return {
                'support_1': None,
                'support_2': None,
                'resistance_1': None,
                'resistance_2': None
            }

    async def calculate_supertrend(self, period: int = 90, multiplier: float = 2.0) -> Dict[str, Optional[any]]:
        """
        11. Calculate SuperTrend indicator
        Uses ATR (Average True Range) for trend detection

        Storage: supertrend_value DECIMAL(21,2), trend TEXT, signal TEXT

        Phase 3.12: Changed from 10-day to 90-day for mid-term horizon alignment
        Multiplier reduced from 3.0 to 2.0 for long-term trend sensitivity

        Args:
            period: ATR calculation period (default: 90)
            multiplier: ATR multiplier (default: 2.0)

        Returns:
            dict with keys: supertrend_value, trend, signal
        """
        try:
            # Phase 3.10: Use prefetched data if available
            if self.prefetch_calc:
                result = self.prefetch_calc.calculate_supertrend(period, multiplier)
                if result and result.get('supertrend_value') is not None:
                    logger.info(f"SuperTrend (prefetched): {result['supertrend_value']:.2f}, trend={result['trend']}")
                    return result

            # Fallback to DB query
            # Get recent price data for ATR calculation
            query = """
            WITH price_data AS (
                SELECT
                    date,
                    high,
                    low,
                    close,
                    LAG(close) OVER (ORDER BY date) as prev_close
                FROM kr_intraday_total
                WHERE symbol = $1
                    AND date >= COALESCE($2::date, CURRENT_DATE) - INTERVAL '120 days'
                    AND date <= COALESCE($2::date, CURRENT_DATE)
                    AND high IS NOT NULL
                    AND low IS NOT NULL
                    AND close IS NOT NULL
                ORDER BY date
            ),
            true_range AS (
                SELECT
                    date,
                    high,
                    low,
                    close,
                    GREATEST(
                        high - low,
                        ABS(high - prev_close),
                        ABS(low - prev_close)
                    ) as tr
                FROM price_data
                WHERE prev_close IS NOT NULL
            ),
            atr_calc AS (
                SELECT
                    date,
                    high,
                    low,
                    close,
                    tr,
                    AVG(tr) OVER (
                        ORDER BY date
                        ROWS BETWEEN $3 - 1 PRECEDING AND CURRENT ROW
                    ) as atr
                FROM true_range
            )
            SELECT
                high,
                low,
                close,
                atr
            FROM atr_calc
            ORDER BY date DESC
            LIMIT 1
            """

            result = await self.execute_query(query, self.symbol, self.analysis_date, period)

            if not result or not result[0]['atr']:
                return {
                    'supertrend_value': None,
                    'trend': None,
                    'signal': None
                }

            data = result[0]
            high = float(data['high'])
            low = float(data['low'])
            close = float(data['close'])
            atr = float(data['atr'])

            # Calculate basic bands
            hl_avg = (high + low) / 2
            basic_upper_band = hl_avg + (multiplier * atr)
            basic_lower_band = hl_avg - (multiplier * atr)

            # Determine trend
            # If close > upper band: uptrend, use lower band as SuperTrend
            # If close < lower band: downtrend, use upper band as SuperTrend
            if close > basic_upper_band:
                supertrend_value = basic_lower_band
                trend = '상승'
                signal = '매수' if close > supertrend_value else '보유'
            elif close < basic_lower_band:
                supertrend_value = basic_upper_band
                trend = '하락'
                signal = '매도' if close < supertrend_value else '보유'
            else:
                # Neutral zone
                supertrend_value = hl_avg
                trend = '중립'
                signal = '보유'

            logger.info(f"SuperTrend: {supertrend_value:.2f}, Trend: {trend}, Signal: {signal}")

            return {
                'supertrend_value': round(supertrend_value, 2),
                'trend': trend,
                'signal': signal
            }

        except Exception as e:
            logger.error(f"SuperTrend calculation failed: {e}")
            return {
                'supertrend_value': None,
                'trend': None,
                'signal': None
            }

    async def calculate_relative_strength(self, period: int = 90) -> Dict[str, Optional[any]]:
        """
        12-13. Calculate Relative Strength (RS) vs market index
        Measures stock performance relative to its market (KOSPI/KOSDAQ)

        Storage: rs_value DECIMAL(21,2), rs_rank TEXT

        Phase 3.12: Changed from 20-day to 90-day for mid-term horizon alignment

        Args:
            period: Comparison period in days (default: 90)

        Returns:
            dict with keys: rs_value, rs_rank
        """
        try:
            # Phase 3.10: Use prefetched data if available (for RS value calculation)
            if self.prefetch_calc:
                result = self.prefetch_calc.calculate_relative_strength()
                if result and result.get('rs_value') is not None:
                    logger.info(f"Relative Strength (prefetched): RS={result['rs_value']:.2f}")
                    return result

            # Fallback to DB query
            # Get stock's exchange first
            query_exchange = """
            SELECT exchange
            FROM kr_stock_detail
            WHERE symbol = $1
            """

            result = await self.execute_query(query_exchange, self.symbol)

            if not result or not result[0]['exchange']:
                return {'rs_value': None, 'rs_rank': None}

            exchange = result[0]['exchange']

            # Determine market index symbol based on exchange
            # KOSPI: ^KS11, KOSDAQ: ^KQ11 (Korean market index symbols)
            # For simplicity, we'll calculate RS using market cap weighted average
            # In production, you would use actual index data

            query = """
            WITH target_date AS (
                SELECT COALESCE($2::date, CURRENT_DATE) as analysis_date
            ),
            stock_current AS (
                SELECT close as current_price
                FROM kr_intraday_total
                WHERE symbol = $1
                    AND date = (SELECT analysis_date FROM target_date)
                LIMIT 1
            ),
            stock_past AS (
                SELECT close as past_price
                FROM kr_intraday_total
                WHERE symbol = $1
                    AND date <= (SELECT analysis_date FROM target_date) - MAKE_INTERVAL(days => $3)
                ORDER BY date DESC
                LIMIT 1
            ),
            market_stocks AS (
                SELECT
                    sd.symbol,
                    current_prices.close as current_close,
                    (
                        SELECT close
                        FROM kr_intraday_total
                        WHERE symbol = sd.symbol
                            AND date <= (SELECT analysis_date FROM target_date) - MAKE_INTERVAL(days => $3)
                        ORDER BY date DESC
                        LIMIT 1
                    ) as past_close
                FROM kr_stock_detail sd
                JOIN kr_intraday_total current_prices ON sd.symbol = current_prices.symbol
                WHERE sd.exchange = $4
                    AND current_prices.date = (SELECT analysis_date FROM target_date)
                    AND current_prices.close IS NOT NULL
                    AND current_prices.market_cap IS NOT NULL
            ),
            market_avg AS (
                SELECT
                    AVG(
                        ((current_close - past_close)::NUMERIC / NULLIF(past_close, 0) * 100)
                    ) as market_return
                FROM market_stocks
                WHERE past_close IS NOT NULL
            )
            SELECT
                ((sc.current_price - sp.past_price)::NUMERIC / NULLIF(sp.past_price, 0) * 100) as stock_return,
                ma.market_return
            FROM stock_current sc, stock_past sp, market_avg ma
            WHERE sp.past_price IS NOT NULL
            """

            result = await self.execute_query(query, self.symbol, self.analysis_date, period, exchange)

            if not result or result[0]['stock_return'] is None:
                return {'rs_value': None, 'rs_rank': None}

            stock_return = float(result[0]['stock_return'])
            market_return = float(result[0]['market_return']) if result[0]['market_return'] else 0.0

            # Calculate RS: difference between stock and market returns
            rs_value = stock_return - market_return

            # Determine RS rank
            if rs_value > 5.0:
                rs_rank = '매우강함'
            elif rs_value > 2.0:
                rs_rank = '강함'
            elif rs_value > -2.0:
                rs_rank = '중립'
            elif rs_value > -5.0:
                rs_rank = '약함'
            else:
                rs_rank = '매우약함'

            logger.info(f"RS ({period}d): {rs_value:.2f}% (Rank: {rs_rank})")
            logger.info(f"  Stock Return: {stock_return:.2f}%, Market Return: {market_return:.2f}%")

            return {
                'rs_value': round(rs_value, 2),
                'rs_rank': rs_rank
            }

        except Exception as e:
            logger.error(f"Relative Strength calculation failed: {e}")
            return {'rs_value': None, 'rs_rank': None}

    async def calculate_smart_money_signal_score(self) -> Tuple[Optional[int], Optional[str], Optional[int]]:
        """
        Calculate Smart Money Signal Score (0-100)
        Based on institutional and foreign investor flow patterns

        Returns validated pattern from 2,757 stock analysis:
        - Strong Inflow (>100%): +3.39% avg return -> 70-100 score
        - Accelerating Inflow: +2.51% avg return -> boosted score

        Storage: smart_money_signal_score INTEGER, smart_money_signal_text TEXT, smart_money_confidence INTEGER

        Returns:
            Tuple of (score, signal_text, confidence)
        """
        try:
            # Get current 30-day flow
            inst_net, foreign_net = await self.calculate_investor_flow()

            if inst_net is None and foreign_net is None:
                return 50, "No institutional data available", 0

            # Get 30-day flow from 1 month ago for change calculation
            query = """
            SELECT
                SUM(inst_net_volume) as inst_net_prev,
                SUM(foreign_net_volume) as foreign_net_prev
            FROM kr_individual_investor_daily_trading
            WHERE symbol = $1
                AND date >= COALESCE($2::date, CURRENT_DATE) - INTERVAL '60 days'
                AND date <= COALESCE($2::date, CURRENT_DATE) - INTERVAL '30 days'
            """

            result = await self.execute_query(query, self.symbol, self.analysis_date)

            if not result:
                inst_net_prev = 0
                foreign_net_prev = 0
            else:
                inst_net_prev = int(result[0]['inst_net_prev']) if result[0]['inst_net_prev'] else 0
                foreign_net_prev = int(result[0]['foreign_net_prev']) if result[0]['foreign_net_prev'] else 0

            # Calculate change percentage
            inst_change_pct = 0
            foreign_change_pct = 0

            if inst_net_prev != 0:
                inst_change_pct = ((inst_net - inst_net_prev) / abs(inst_net_prev)) * 100

            if foreign_net_prev != 0:
                foreign_change_pct = ((foreign_net - foreign_net_prev) / abs(foreign_net_prev)) * 100

            # Score calculation based on validated patterns
            score = 50  # Neutral baseline
            signal_text = ""
            confidence = 50

            # Explosive inflow (>500%)
            if inst_change_pct > 500 or foreign_change_pct > 500:
                score = 100
                signal_text = "EXPLOSIVE INSTITUTIONAL BUYING - Very strong bullish signal"
                confidence = 95

            # Very strong inflow (>200%)
            elif inst_change_pct > 200 or foreign_change_pct > 200:
                score = 85
                signal_text = "VERY STRONG INSTITUTIONAL BUYING - Strong bullish signal"
                confidence = 90

            # Strong inflow (>100%) - Validated: +3.39% avg return
            elif inst_change_pct > 100 or foreign_change_pct > 100:
                score = 70
                signal_text = "STRONG INSTITUTIONAL BUYING - Bullish signal (+3.39% historical avg)"
                confidence = 85

            # Moderate inflow (both >50%)
            elif inst_change_pct > 50 and foreign_change_pct > 50:
                score = 60
                signal_text = "MODERATE INSTITUTIONAL BUYING - Positive signal"
                confidence = 75

            # Mild inflow (both >0%)
            elif inst_change_pct > 0 and foreign_change_pct > 0:
                score = 55
                signal_text = "MILD INSTITUTIONAL BUYING - Slightly positive"
                confidence = 60

            # Mixed signals
            elif (inst_change_pct > 0 and foreign_change_pct < 0) or (inst_change_pct < 0 and foreign_change_pct > 0):
                score = 45
                signal_text = "MIXED INSTITUTIONAL SIGNALS - Uncertain"
                confidence = 40

            # Moderate outflow
            elif inst_change_pct < 0 and foreign_change_pct < 0:
                score = 30
                signal_text = "INSTITUTIONAL SELLING - Negative signal"
                confidence = 70

            # Strong outflow (<-50%) - Validated: -3.08% avg return
            elif inst_change_pct < -50 and foreign_change_pct < -50:
                score = 10
                signal_text = "STRONG INSTITUTIONAL SELLING - Bearish signal (-3.08% historical avg)"
                confidence = 85

            logger.info(f"Smart Money Score: {score}/100")
            logger.info(f"  Institutional change: {inst_change_pct:+.1f}%")
            logger.info(f"  Foreign change: {foreign_change_pct:+.1f}%")
            logger.info(f"  Signal: {signal_text}")

            return score, signal_text, confidence

        except Exception as e:
            logger.error(f"Smart money signal calculation failed: {e}")
            return 50, "Calculation error", 0

    async def calculate_volatility_context_score(self, volatility: Optional[float], smart_money_score: int) -> Tuple[Optional[int], Optional[str]]:
        """
        Calculate Volatility Context Score (0-100)
        Validates if volatility is opportunity or risk based on smart money flow

        Validated pattern from 888 stocks:
        - High Vol (>40%) + Smart Money Inflow: +5.89% avg return -> 85-90 score
        - High Vol (>40%) + Smart Money Outflow: -3.71% avg return -> 20-25 score
        - Difference: +9.60%p

        Storage: volatility_context_score INTEGER, volatility_context_text TEXT

        Returns:
            Tuple of (score, context_text)
        """
        try:
            if volatility is None:
                return 50, "Volatility data unavailable"

            # Classify volatility level
            if volatility > 70:
                vol_category = 'Very High'
            elif volatility > 50:
                vol_category = 'High'
            elif volatility > 30:
                vol_category = 'Medium'
            else:
                vol_category = 'Low'

            # Volatility context depends on smart money
            score = 50
            context_text = ""

            if vol_category == 'Very High':
                if smart_money_score >= 70:  # Strong inflow
                    score = 85
                    context_text = "Very high volatility with strong institutional buying - HIGH OPPORTUNITY"
                elif smart_money_score >= 50:  # Mild inflow
                    score = 60
                    context_text = "Very high volatility with mild institutional support - MODERATE RISK"
                else:  # Outflow or mixed
                    score = 20
                    context_text = "Very high volatility without institutional support - HIGH RISK"

            elif vol_category == 'High':
                if smart_money_score >= 70:  # Strong inflow
                    score = 90  # Best combination: +5.89% avg from 328 stocks
                    context_text = "High volatility driven by institutional accumulation - EXCELLENT OPPORTUNITY (+5.89% historical avg)"
                elif smart_money_score >= 50:
                    score = 65
                    context_text = "High volatility with institutional support - GOOD OPPORTUNITY"
                else:
                    score = 25  # Dangerous: -3.71% avg
                    context_text = "High volatility with fund exodus - STRONG RISK (-3.71% historical avg)"

            elif vol_category == 'Medium':
                if smart_money_score >= 70:
                    score = 75
                    context_text = "Medium volatility with strong buying - GOOD"
                elif smart_money_score >= 50:
                    score = 60
                    context_text = "Medium volatility - BALANCED"
                else:
                    score = 40
                    context_text = "Medium volatility with selling - CAUTION"

            else:  # Low volatility
                if smart_money_score >= 70:
                    score = 65
                    context_text = "Low volatility with institutional buying - STABLE ACCUMULATION"
                else:
                    score = 50
                    context_text = "Low volatility - NEUTRAL"

            logger.info(f"Volatility Context Score: {score}/100")
            logger.info(f"  Volatility: {volatility:.1f}% ({vol_category})")
            logger.info(f"  Context: {context_text}")

            return score, context_text

        except Exception as e:
            logger.error(f"Volatility context calculation failed: {e}")
            return 50, "Calculation error"

    async def calculate_factor_combination_bonus(self, value_score: float, quality_score: float,
                                                  momentum_score: float, growth_score: float,
                                                  smart_money_score: int) -> float:
        """
        Calculate Factor Combination Bonus (Phase 2)
        Rewards validated winning combinations from 2,757 stock analysis

        Validated patterns:
        - Low Value + High Growth + SM Inflow: +11.31% (358 stocks) -> +30 bonus
        - Med Value + High Growth + SM Inflow: +9.78% (233 stocks) -> +25 bonus
        - High Quality + High/Med Momentum + SM Inflow: +4.4~4.8% -> +15 bonus
        - Low Growth: Negative returns -> -20 penalty

        Returns normalized score (0-100 scale):
        - Raw bonus range: -20 to +30 (total 50 points)
        - Normalized range: 0 to 100 (100-point scale)
        - Neutral point: 0 -> 40 points
        - Formula: (raw_bonus + 20) / 50 * 100

        Storage: factor_combination_bonus DECIMAL(5,1)

        Returns:
            float: Normalized bonus score (0-100)
                - 0: Worst combination (raw -20)
                - 40: Neutral (raw 0)
                - 100: Best combination (raw +30)
        """
        try:
            raw_bonus = 0

            # Define thresholds based on 33rd, 67th percentile from analysis
            # Value: Low < 41.3 < Medium < 59.3 < High
            # Quality: Low < 64.8 < Medium < 77.6 < High
            # Momentum: Low < 32.7 < Medium < 42.0 < High
            # Growth: Low < 29.7 < Medium < 41.3 < High

            value_high = value_score > 59.3
            value_med = 41.3 < value_score <= 59.3
            value_low = value_score <= 41.3

            quality_high = quality_score > 77.6
            quality_med = 64.8 < quality_score <= 77.6

            momentum_high = momentum_score > 42.0
            momentum_med = 32.7 < momentum_score <= 42.0

            growth_high = growth_score > 41.3
            growth_med = 29.7 < growth_score <= 41.3
            growth_low = growth_score <= 29.7

            strong_sm = smart_money_score >= 70

            # Pattern 1: Growth Stock at Good Price + Smart Money
            # Low Value + High Growth + SM Inflow = +11.31% (358 stocks, 65.6% win rate)
            if value_low and growth_high and strong_sm:
                raw_bonus += 30
                logger.info(f"  [Bonus +30] Growth stock at good price + Smart Money (+11.31% pattern)")

            # Pattern 2: Medium Value + High Growth + SM Inflow = +9.78% (233 stocks, 63.9% win rate)
            elif value_med and growth_high and strong_sm:
                raw_bonus += 25
                logger.info(f"  [Bonus +25] Growth stock at fair price + Smart Money (+9.78% pattern)")

            # Pattern 3: Blue Chip Uptrend
            # High Quality + High/Med Momentum + SM Inflow = +4.4~4.8%
            elif quality_high and (momentum_high or momentum_med) and strong_sm:
                raw_bonus += 15
                logger.info(f"  [Bonus +15] Blue chip uptrend + Smart Money (+4.4~4.8% pattern)")

            # Pattern 4: High Growth + Inflow (without value consideration)
            elif growth_high and strong_sm:
                raw_bonus += 10
                logger.info(f"  [Bonus +10] High growth + Smart Money inflow")

            # Negative Pattern: Low Growth
            # Growth Low: Most had negative returns
            if growth_low:
                raw_bonus -= 20
                logger.info(f"  [Penalty -20] Low growth stock - historically negative returns")

            # Normalize to 0-100 scale
            # Raw range: -20 to +30 (50 points total)
            # Target range: 0 to 100 (100 points)
            normalized_bonus = (raw_bonus + 20) / 50 * 100

            logger.info(f"Factor Combination Bonus: raw={raw_bonus:+d}, normalized={normalized_bonus:.1f}")
            return round(normalized_bonus, 1)

        except Exception as e:
            logger.error(f"Factor combination bonus calculation failed: {e}")
            return 40.0  # Return neutral score (0 raw bonus -> 40 normalized)

    async def calculate_smart_money_momentum_score(self) -> int:
        """
        Calculate Smart Money Momentum Score (Phase 2)
        Based on institutional/foreign flow persistence and acceleration

        Validated patterns from 2,757 stocks:
        - Accelerating Inflow (consistent acceleration): +2.51% (2,485 stocks) -> 80 score
        - Recent Jump Only (sudden surge): -8.08% (18 stocks) -> 20 score (RISK!)
        - Persistent Inflow: Stable support -> 60 score

        Storage: smart_money_momentum_score INTEGER

        Returns:
            Momentum score (0-100)
        """
        try:
            # Get 3 data points to check momentum/acceleration
            # We need data from ~2 months ago, ~1 month ago, and current

            # Current 30-day (most recent)
            query_current = """
            SELECT
                SUM(inst_net_volume) as inst_net,
                SUM(foreign_net_volume) as foreign_net
            FROM kr_individual_investor_daily_trading
            WHERE symbol = $1
                AND date >= COALESCE($2::date, CURRENT_DATE) - INTERVAL '30 days'
                AND date <= COALESCE($2::date, CURRENT_DATE)
            """

            # 30-day from 1 month ago
            query_1m_ago = """
            SELECT
                SUM(inst_net_volume) as inst_net,
                SUM(foreign_net_volume) as foreign_net
            FROM kr_individual_investor_daily_trading
            WHERE symbol = $1
                AND date >= COALESCE($2::date, CURRENT_DATE) - INTERVAL '60 days'
                AND date <= COALESCE($2::date, CURRENT_DATE) - INTERVAL '30 days'
            """

            # 30-day from 2 months ago
            query_2m_ago = """
            SELECT
                SUM(inst_net_volume) as inst_net,
                SUM(foreign_net_volume) as foreign_net
            FROM kr_individual_investor_daily_trading
            WHERE symbol = $1
                AND date >= COALESCE($2::date, CURRENT_DATE) - INTERVAL '90 days'
                AND date <= COALESCE($2::date, CURRENT_DATE) - INTERVAL '60 days'
            """

            result_current = await self.execute_query(query_current, self.symbol, self.analysis_date)
            result_1m = await self.execute_query(query_1m_ago, self.symbol, self.analysis_date)
            result_2m = await self.execute_query(query_2m_ago, self.symbol, self.analysis_date)

            if not result_current or not result_1m or not result_2m:
                return 50  # Neutral if data incomplete

            inst_current = int(result_current[0]['inst_net']) if result_current[0]['inst_net'] else 0
            foreign_current = int(result_current[0]['foreign_net']) if result_current[0]['foreign_net'] else 0

            inst_1m = int(result_1m[0]['inst_net']) if result_1m[0]['inst_net'] else 0
            foreign_1m = int(result_1m[0]['foreign_net']) if result_1m[0]['foreign_net'] else 0

            inst_2m = int(result_2m[0]['inst_net']) if result_2m[0]['inst_net'] else 0
            foreign_2m = int(result_2m[0]['foreign_net']) if result_2m[0]['foreign_net'] else 0

            # Check patterns
            inst_accelerating = inst_current > inst_1m and inst_1m > inst_2m
            foreign_accelerating = foreign_current > foreign_1m and foreign_1m > foreign_2m

            inst_persistent = inst_current > 0 and inst_1m > 0 and inst_2m > 0
            foreign_persistent = foreign_current > 0 and foreign_1m > 0 and foreign_2m > 0

            # Recent jump: current >> 1m ago, but 1m ago <= 2m ago
            inst_jump = (inst_current > inst_1m * 2) and (inst_1m <= inst_2m)
            foreign_jump = (foreign_current > foreign_1m * 2) and (foreign_1m <= foreign_2m)

            score = 50  # Neutral baseline

            # Accelerating Inflow - Validated: +2.51% (2,485 stocks)
            if inst_accelerating or foreign_accelerating:
                score = 80
                pattern = "ACCELERATING INFLOW - Consistent buying pressure (+2.51% historical avg)"

            # Recent Jump Only - Validated: -8.08% (18 stocks) - HIGH RISK!
            elif inst_jump or foreign_jump:
                score = 20
                pattern = "SUDDEN JUMP - Likely too late, high risk (-8.08% historical avg)"

            # Persistent Inflow
            elif inst_persistent or foreign_persistent:
                score = 60
                pattern = "PERSISTENT INFLOW - Ongoing support, stable"

            # Declining or No Pattern
            else:
                score = 40
                pattern = "NO CLEAR MOMENTUM - Uncertain"

            logger.info(f"Smart Money Momentum Score: {score}/100")
            logger.info(f"  Pattern: {pattern}")
            logger.info(f"  Inst Flow: 2m={inst_2m:,}, 1m={inst_1m:,}, Current={inst_current:,}")
            logger.info(f"  Foreign Flow: 2m={foreign_2m:,}, 1m={foreign_1m:,}, Current={foreign_current:,}")

            return score

        except Exception as e:
            logger.error(f"Smart money momentum calculation failed: {e}")
            return 50

    async def calculate_score_momentum_signal(self) -> Tuple[int, str, Optional[float], Optional[float]]:
        """
        Calculate score momentum signal based on historical final_score changes

        Pattern validation (Sep-Oct 2025 data):
        - Rising Fast (>+10): +48.97% avg return (92 stocks) -> 100 points
        - Rising (+5 to +10): +9.17% avg return (226 stocks) -> 70 points
        - Stable (-5 to +5): +1.26% avg return (1,942 stocks) -> 50 points
        - Falling (-10 to -5): -2.12% avg return (233 stocks) -> 30 points
        - Falling Fast (<-10): -2.35% avg return (264 stocks) -> 10 points

        Returns:
            (score, signal_text, change_1m, change_2m)
        """
        try:
            # Get current score (from factor_scores context, not yet saved to DB)
            # Note: This will be the calculated score, not from DB
            # For first-time analysis, we won't have historical data
            current_score = None
            if hasattr(self, 'current_final_score'):
                current_score = self.current_final_score
            else:
                # Calculate on the fly from factor scores (handle None weighted_result)
                value_result = self.factor_scores.get('value', {}).get('weighted_result') or {}
                quality_result = self.factor_scores.get('quality', {}).get('weighted_result') or {}
                momentum_result = self.factor_scores.get('momentum', {}).get('weighted_result') or {}
                growth_result = self.factor_scores.get('growth', {}).get('weighted_result') or {}

                value_score = value_result.get('weighted_score', 0)
                quality_score = quality_result.get('weighted_score', 0)
                momentum_score = momentum_result.get('weighted_score', 0)
                growth_score = growth_result.get('weighted_score', 0)

                # Use Phase 2 weighting for current score estimate
                base_factor_score = (
                    value_score * 0.15 +
                    quality_score * 0.20 +
                    momentum_score * 0.15 +
                    growth_score * 0.50
                )
                current_score = base_factor_score * 0.30 + 50 * 0.70  # Conservative estimate

            # Get historical scores from database
            query_1m = """
                SELECT final_score
                FROM kr_stock_grade
                WHERE symbol = $1
                  AND date >= CURRENT_DATE - INTERVAL '40 days'
                  AND date <= CURRENT_DATE - INTERVAL '20 days'
                  AND final_score IS NOT NULL
                ORDER BY date DESC
                LIMIT 1
            """

            query_2m = """
                SELECT final_score
                FROM kr_stock_grade
                WHERE symbol = $1
                  AND date >= CURRENT_DATE - INTERVAL '70 days'
                  AND date <= CURRENT_DATE - INTERVAL '50 days'
                  AND final_score IS NOT NULL
                ORDER BY date DESC
                LIMIT 1
            """

            score_1m_ago = None
            score_2m_ago = None

            # Fetch 1 month ago score
            rows_1m = await self.db_manager.execute_query(query_1m, self.symbol)
            if rows_1m and len(rows_1m) > 0:
                score_1m_ago = float(rows_1m[0]['final_score'])

            # Fetch 2 months ago score
            rows_2m = await self.db_manager.execute_query(query_2m, self.symbol)
            if rows_2m and len(rows_2m) > 0:
                score_2m_ago = float(rows_2m[0]['final_score'])

            # Calculate changes
            change_1m = None
            change_2m = None

            if score_1m_ago is not None and current_score is not None:
                change_1m = current_score - score_1m_ago

            if score_2m_ago is not None and score_1m_ago is not None:
                change_2m = score_1m_ago - score_2m_ago

            # Determine score and signal based on patterns
            if change_1m is None:
                # No historical data - return neutral
                score = 50
                signal_text = "No historical data - Score momentum neutral"
                logger.info(f"Score Momentum: {score}/100 (No historical data)")

            elif change_1m > 10:
                # Check if accelerating
                if change_2m is not None and change_2m > 5:
                    score = 100
                    signal_text = f"ACCELERATING UPWARD: Score rising fast for 2 months (+{change_1m:.1f} recent, +{change_2m:.1f} prior) - STRONG BUY (+48.97% historical avg)"
                else:
                    score = 85
                    signal_text = f"RISING FAST: Score jumped +{change_1m:.1f} points - Potential breakout (+48.97% historical avg)"
                logger.info(f"Score Momentum: {score}/100 - RISING FAST")
                logger.info(f"  Change: 1m=+{change_1m:.1f}, 2m={'+' + str(change_2m) if change_2m else 'N/A'}")

            elif change_1m > 5:
                score = 70
                signal_text = f"RISING STEADILY: Score up +{change_1m:.1f} points - Positive momentum (+9.17% historical avg)"
                logger.info(f"Score Momentum: {score}/100 - RISING")
                logger.info(f"  Change: 1m=+{change_1m:.1f}")

            elif change_1m >= -5:
                score = 50
                signal_text = f"STABLE: Score change {change_1m:+.1f} points - Neutral (+1.26% historical avg)"
                logger.info(f"Score Momentum: {score}/100 - STABLE")

            elif change_1m >= -10:
                score = 30
                signal_text = f"DECLINING: Score down {change_1m:.1f} points - Caution (-2.12% historical avg)"
                logger.info(f"Score Momentum: {score}/100 - DECLINING")
                logger.info(f"  Change: 1m={change_1m:.1f}")

            else:  # change_1m < -10
                score = 10
                signal_text = f"FALLING FAST: Score dropped {change_1m:.1f} points - WARNING (-2.35% historical avg)"
                logger.info(f"Score Momentum: {score}/100 - FALLING FAST")
                logger.info(f"  Change: 1m={change_1m:.1f}, 2m={change_2m if change_2m else 'N/A'}")

            return score, signal_text, change_1m, change_2m

        except Exception as e:
            logger.error(f"Score momentum calculation failed: {e}")
            return 50, "Score momentum calculation error", None, None

    async def calculate_sector_rotation_signal(self) -> Tuple[int, Optional[float], Optional[int], Optional[float]]:
        """
        Calculate sector rotation signal based on recent sector performance

        OPTIMIZED VERSION (Phase 3.4 - Materialized View Integration)
        - Uses mv_sector_daily_performance for pre-calculated sector aggregates
        - Memory reduction: 5GB -> 10MB (500x improvement)
        - Speed improvement: 5-10s -> 0.01s (1000x faster)
        - Data refreshed automatically by kr_main.py before each date analysis

        Analysis from Sep-Oct 2025 data:
        - Top sector (금융서비스): +24.10% return, 82.5% win rate
        - Bottom sector (화학섬유소재): -3.32% return, 28.9% win rate
        - Gap: 27.42% alpha potential from sector selection

        Scoring:
        - Top 3 sectors: 90-100 points (+20 bonus)
        - Top 20% sectors: 70-89 points (+10 bonus)
        - Middle sectors: 40-69 points (neutral)
        - Bottom 20% sectors: 10-39 points (-15 penalty)
        - Bottom 3 sectors: 0-9 points (-20 penalty)

        Returns:
            (sector_score, sector_momentum, sector_rank, sector_percentile)
        """
        try:
            # Step 1: Get current stock's theme from kr_stock_detail
            query_theme = """
                SELECT theme
                FROM kr_stock_detail
                WHERE symbol = $1
            """

            theme_result = await self.db_manager.execute_query(query_theme, self.symbol)
            if not theme_result or len(theme_result) == 0:
                logger.info("Sector Rotation: Symbol not found in kr_stock_detail, returning neutral")
                return 50, None, None, None

            current_sector = theme_result[0]['theme']
            if not current_sector:
                logger.info("Sector Rotation: theme is NULL, returning neutral")
                return 50, None, None, None

            # Step 2: Get sector performance from Materialized View
            # Uses pre-calculated sector aggregates (refreshed by kr_main.py)
            # BEFORE: Complex JOIN query with 2,748 stocks × 30 days (5GB memory)
            # AFTER: Simple MV lookup for single row (10MB memory)
            query_mv = """
                SELECT
                    avg_return_30d as sector_momentum,
                    sector_rank,
                    stock_count,
                    (SELECT COUNT(DISTINCT sector_code)
                     FROM mv_sector_daily_performance
                     WHERE date = $2) as total_sectors
                FROM mv_sector_daily_performance
                WHERE date = $2 AND sector_code = $1
            """

            mv_result = await self.db_manager.execute_query(query_mv, current_sector, self.analysis_date)

            if not mv_result or len(mv_result) == 0:
                logger.info(f"Sector Rotation: No MV data for sector '{current_sector}' on {self.analysis_date}, returning neutral")
                return 50, None, None, None

            row = mv_result[0]
            current_sector_rank = row['sector_rank']
            current_sector_momentum = float(row['sector_momentum']) if row['sector_momentum'] else 0.0
            total_sectors = row['total_sectors']

            if not current_sector_rank or not total_sectors:
                logger.info("Sector Rotation: Invalid MV data, returning neutral")
                return 50, None, None, None

            # Step 3: Calculate percentile and score
            # percentile = "상위 X%" (e.g., rank 8 out of 17 = top 47.06%)
            percentile = (current_sector_rank / total_sectors) * 100

            # Score based on rank
            score = 50  # Default neutral

            if current_sector_rank <= 3:
                # Top 3 sectors
                score = 90 + (3 - current_sector_rank) * 3  # 90, 93, 96
                logger.info(f"Sector Rotation: {score}/100 - TOP 3 SECTOR")
                logger.info(f"  Sector: {current_sector} (Rank {current_sector_rank}/{total_sectors})")
                logger.info(f"  Momentum: {current_sector_momentum:+.2f}%")
                logger.info(f"  Percentile: {percentile:.1f}% (Top sector bonus: +20)")

            elif current_sector_rank <= max(3, int(total_sectors * 0.2)):
                # Top 20%
                top_20_pct_rank = int(total_sectors * 0.2)
                score = 70 + int((top_20_pct_rank - current_sector_rank) / top_20_pct_rank * 19)
                logger.info(f"Sector Rotation: {score}/100 - TOP 20%")
                logger.info(f"  Sector: {current_sector} (Rank {current_sector_rank}/{total_sectors})")
                logger.info(f"  Momentum: {current_sector_momentum:+.2f}%")
                logger.info(f"  Percentile: {percentile:.1f}% (Strong sector bonus: +10)")

            elif current_sector_rank >= total_sectors - 2:
                # Bottom 3 sectors
                score = 10 - (total_sectors - current_sector_rank) * 3  # 10, 7, 4
                score = max(0, score)
                logger.info(f"Sector Rotation: {score}/100 - BOTTOM 3 SECTOR (WARNING)")
                logger.info(f"  Sector: {current_sector} (Rank {current_sector_rank}/{total_sectors})")
                logger.info(f"  Momentum: {current_sector_momentum:+.2f}%")
                logger.info(f"  Percentile: {percentile:.1f}% (Weak sector penalty: -20)")

            elif current_sector_rank >= max(total_sectors - 2, int(total_sectors * 0.8)):
                # Bottom 20%
                bottom_20_pct_rank = int(total_sectors * 0.8)
                score = 10 + int((current_sector_rank - bottom_20_pct_rank) / (total_sectors - bottom_20_pct_rank) * 29)
                logger.info(f"Sector Rotation: {score}/100 - BOTTOM 20%")
                logger.info(f"  Sector: {current_sector} (Rank {current_sector_rank}/{total_sectors})")
                logger.info(f"  Momentum: {current_sector_momentum:+.2f}%")
                logger.info(f"  Percentile: {percentile:.1f}% (Weak sector penalty: -15)")

            else:
                # Middle sectors (20%-80%)
                # Linear interpolation from 40 to 69
                mid_start = max(3, int(total_sectors * 0.2))
                mid_end = max(total_sectors - 2, int(total_sectors * 0.8))
                if mid_end > mid_start:
                    score = 40 + int((mid_end - current_sector_rank) / (mid_end - mid_start) * 29)
                else:
                    score = 50
                logger.info(f"Sector Rotation: {score}/100 - MIDDLE")
                logger.info(f"  Sector: {current_sector} (Rank {current_sector_rank}/{total_sectors})")
                logger.info(f"  Momentum: {current_sector_momentum:+.2f}%")
                logger.info(f"  Percentile: {percentile:.1f}%")

            return score, current_sector_momentum, current_sector_rank, percentile

        except Exception as e:
            logger.error(f"Sector rotation calculation failed: {e}")
            return 50, None, None, None

    async def calculate_risk_adjusted_returns(self) -> Dict:
        """
        Risk-Adjusted Return metrics calculation (Phase 3.9)

        Returns:
            Dict: sharpe_ratio, sortino_ratio, calmar_ratio

        Note:
            - Based on 60-day holding period (highest IC according to phase3_9 analysis)
            - Risk-free rate: 3.5% annual
            - Sharpe Ratio: (Return - Rf) / Volatility
            - Sortino Ratio: (Return - Rf) / Downside Deviation
            - Calmar Ratio: Return / Max Drawdown
        """
        try:
            # Phase 3.10: Use prefetched data if available (eliminates heavy window function queries)
            if self.prefetch_calc:
                result = self.prefetch_calc.calculate_risk_adjusted_returns()
                if result and any(v is not None for v in result.values()):
                    sharpe_str = f"{result['sharpe_ratio']:.3f}" if result['sharpe_ratio'] is not None else "N/A"
                    sortino_str = f"{result['sortino_ratio']:.3f}" if result['sortino_ratio'] is not None else "N/A"
                    calmar_str = f"{result['calmar_ratio']:.3f}" if result['calmar_ratio'] is not None else "N/A"
                    logger.info(f"[{self.symbol}] Risk-Adjusted Returns (prefetched): Sharpe={sharpe_str}, Sortino={sortino_str}, Calmar={calmar_str}")
                    return result

            # Fallback to DB query
            # Risk-free rate (3.5% annual)
            rf_annual = 3.5
            holding_days = 60
            rf_60d = rf_annual * (holding_days / 252)

            # 1. Calculate 60-day return and volatility
            return_query = """
            WITH price_data AS (
                SELECT
                    date,
                    close,
                    FIRST_VALUE(close) OVER (ORDER BY date) as start_price,
                    LAST_VALUE(close) OVER (ORDER BY date ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) as end_price,
                    (close / LAG(close, 1) OVER (ORDER BY date) - 1) * 100 as daily_return
                FROM kr_intraday_total
                WHERE symbol = $1
                    AND date <= COALESCE($2::date, CURRENT_DATE)
                    AND date >= COALESCE($2::date, CURRENT_DATE) - INTERVAL '60 days'
                ORDER BY date
            )
            SELECT
                (MAX(end_price) - MAX(start_price)) / MAX(start_price) * 100 as return_60d,
                STDDEV(daily_return) as daily_std,
                COUNT(*) as trading_days
            FROM price_data
            WHERE daily_return IS NOT NULL
            """

            result = await self.execute_query(return_query, self.symbol, self.analysis_date)

            if not result or len(result) == 0:
                logger.warning(f"[{self.symbol}] No return data for risk-adjusted metrics")
                return {
                    'sharpe_ratio': None,
                    'sortino_ratio': None,
                    'calmar_ratio': None
                }

            row = result[0]
            return_60d = float(row['return_60d']) if row['return_60d'] else 0
            daily_std = float(row['daily_std']) if row['daily_std'] else 0
            trading_days = int(row['trading_days']) if row['trading_days'] else 0

            # 60-day volatility (daily std -> 60-day volatility)
            volatility_60d = daily_std * np.sqrt(trading_days) if trading_days > 0 else 0

            # 2. Sharpe Ratio
            if volatility_60d > 0:
                sharpe_ratio = (return_60d - rf_60d) / volatility_60d
            else:
                sharpe_ratio = 0

            # 3. Sortino Ratio (downside deviation only)
            downside_query = """
            WITH daily_returns AS (
                SELECT
                    (close / LAG(close, 1) OVER (ORDER BY date) - 1) * 100 as daily_return
                FROM kr_intraday_total
                WHERE symbol = $1
                    AND date <= COALESCE($2::date, CURRENT_DATE)
                    AND date >= COALESCE($2::date, CURRENT_DATE) - INTERVAL '60 days'
                ORDER BY date
            )
            SELECT
                STDDEV(daily_return) as downside_deviation,
                COUNT(*) as negative_days
            FROM daily_returns
            WHERE daily_return < 0 AND daily_return IS NOT NULL
            """

            downside_result = await self.execute_query(downside_query, self.symbol, self.analysis_date)

            if downside_result and len(downside_result) > 0 and downside_result[0]['downside_deviation']:
                downside_deviation = float(downside_result[0]['downside_deviation'])
                negative_days = int(downside_result[0]['negative_days'])
                downside_deviation_60d = downside_deviation * np.sqrt(negative_days) if negative_days > 0 else 0

                if downside_deviation_60d > 0:
                    sortino_ratio = (return_60d - rf_60d) / downside_deviation_60d
                else:
                    sortino_ratio = 0
            else:
                sortino_ratio = None

            # 4. Calmar Ratio (reuse MDD)
            max_drawdown = await self.calculate_max_drawdown_1y()

            if max_drawdown and max_drawdown != 0:
                # MDD is negative, use absolute value
                calmar_ratio = return_60d / abs(max_drawdown)
            else:
                calmar_ratio = None

            sharpe_str = f"{sharpe_ratio:.3f}" if sharpe_ratio is not None else "N/A"
            sortino_str = f"{sortino_ratio:.3f}" if sortino_ratio is not None else "N/A"
            calmar_str = f"{calmar_ratio:.3f}" if calmar_ratio is not None else "N/A"
            logger.info(f"[{self.symbol}] Risk-Adjusted Returns: Sharpe={sharpe_str}, Sortino={sortino_str}, Calmar={calmar_str}")

            return {
                'sharpe_ratio': round(sharpe_ratio, 4) if sharpe_ratio is not None else None,
                'sortino_ratio': round(sortino_ratio, 4) if sortino_ratio is not None else None,
                'calmar_ratio': round(calmar_ratio, 4) if calmar_ratio is not None else None
            }

        except Exception as e:
            logger.error(f"[{self.symbol}] Risk-adjusted returns calculation failed: {e}")
            return {
                'sharpe_ratio': None,
                'sortino_ratio': None,
                'calmar_ratio': None
            }

    async def calculate_all_metrics(self, kospi_data: dict = None) -> Dict:
        """
        Calculate all metrics (fundamental + technical + signals + Phase 3)

        Args:
            kospi_data: Precomputed KOSPI data for corr_kospi and tail_beta
                        {'dates': [...], 'returns': [...]}

        Returns:
            Dict with all metrics ready for kr_stock_grade table insertion
        """
        logger.info(f"Calculating all metrics for {self.symbol}...")

        # Calculate fundamental & risk metrics (9 + cvar_95)
        confidence = await self.calculate_confidence_score()
        volatility = await self.calculate_volatility_annual()
        expected_ranges = await self.calculate_expected_range(volatility)
        var_95 = await self.calculate_var_95()
        cvar_95 = await self.calculate_cvar_95()

        # Phase 1 VaR improvements (7 new metrics)
        hurst_exponent = await self.calculate_hurst_exponent()
        var_95_ewma = await self.calculate_var_95_ewma()
        var_99 = await self.calculate_var_99()
        period_var = await self.calculate_period_var()

        # Phase 2 Volatility Sizing (4 metrics, vol_percentile is batch calculated later)
        inv_vol_weight = await self.calculate_inv_vol_weight()
        downside_vol = await self.calculate_downside_vol()
        atr_metrics = await self.calculate_atr_metrics()

        # Phase 3 CVaR + Risk Budgeting (4 metrics, corr_sector_avg is batch calculated)
        cvar_99 = await self.calculate_cvar_99()
        corr_kospi = await self.calculate_corr_kospi(kospi_data)
        tail_beta = await self.calculate_tail_beta(kospi_data)
        drawdown_duration_avg = await self.calculate_drawdown_duration_avg()

        inst_net, foreign_net = await self.calculate_investor_flow()
        factor_momentum = await self.calculate_factor_momentum()
        industry_rank, industry_percentile = await self.calculate_industry_ranking()
        max_drawdown = await self.calculate_max_drawdown_1y()
        beta = await self.calculate_beta()

        # Calculate technical indicators (4)
        support_resistance = await self.calculate_support_resistance()
        supertrend = await self.calculate_supertrend()
        relative_strength = await self.calculate_relative_strength()

        # Calculate Phase 1 signal scores (2)
        smart_money_score, smart_money_text, smart_money_conf = await self.calculate_smart_money_signal_score()
        volatility_context_score, volatility_context_text = await self.calculate_volatility_context_score(volatility, smart_money_score)

        # Calculate Phase 2 signal scores (2)
        # Safely extract scores (handle None weighted_result from failed factor calculations)
        value_result = self.factor_scores.get('value', {}).get('weighted_result') or {}
        quality_result = self.factor_scores.get('quality', {}).get('weighted_result') or {}
        momentum_result = self.factor_scores.get('momentum', {}).get('weighted_result') or {}
        growth_result = self.factor_scores.get('growth', {}).get('weighted_result') or {}

        value_score = value_result.get('weighted_score', 0)
        quality_score = quality_result.get('weighted_score', 0)
        momentum_score = momentum_result.get('weighted_score', 0)
        growth_score = growth_result.get('weighted_score', 0)

        factor_combo_bonus = await self.calculate_factor_combination_bonus(
            value_score, quality_score, momentum_score, growth_score, smart_money_score
        )
        smart_money_momentum = await self.calculate_smart_money_momentum_score()

        # Calculate Phase 3.1 signal (1)
        score_momentum_signal, score_momentum_text, change_1m, change_2m = await self.calculate_score_momentum_signal()

        # Calculate Phase 3.2 signal (1)
        sector_rotation_score, sector_momentum, sector_rank, sector_percentile = await self.calculate_sector_rotation_signal()

        # Compile results
        metrics = {
            # Fundamental & Risk Metrics (9 + cvar_95)
            'confidence_score': confidence,
            'volatility_annual': volatility,
            'expected_range_3m_min': expected_ranges.get('range_3m_min') if expected_ranges else None,
            'expected_range_3m_max': expected_ranges.get('range_3m_max') if expected_ranges else None,
            'expected_range_1y_min': expected_ranges.get('range_1y_min') if expected_ranges else None,
            'expected_range_1y_max': expected_ranges.get('range_1y_max') if expected_ranges else None,
            'var_95': var_95,
            'cvar_95': cvar_95,
            # Phase 1 VaR improvements
            'hurst_exponent': hurst_exponent,
            'var_95_ewma': var_95_ewma,
            'var_95_5d': period_var.get('var_95_5d'),
            'var_95_20d': period_var.get('var_95_20d'),
            'var_95_60d': period_var.get('var_95_60d'),
            'var_99': var_99,
            'var_99_60d': period_var.get('var_99_60d'),
            # Phase 2 Volatility Sizing (vol_percentile is batch calculated in kr_main.py)
            'inv_vol_weight': inv_vol_weight,
            'downside_vol': downside_vol,
            'vol_percentile': None,  # Calculated in batch after all stocks analyzed
            'atr_20d': atr_metrics.get('atr_20d'),
            'atr_pct_20d': atr_metrics.get('atr_pct_20d'),
            # Phase 3 CVaR + Risk Budgeting (corr_sector_avg is batch calculated in kr_main.py)
            'cvar_99': cvar_99,
            'corr_kospi': corr_kospi,
            'corr_sector_avg': None,  # Calculated in batch after all stocks analyzed
            'tail_beta': tail_beta,
            'drawdown_duration_avg': drawdown_duration_avg,
            'inst_net_30d': inst_net,
            'foreign_net_30d': foreign_net,
            'value_momentum': factor_momentum.get('value_momentum'),
            'quality_momentum': factor_momentum.get('quality_momentum'),
            'momentum_momentum': factor_momentum.get('momentum_momentum'),
            'growth_momentum': factor_momentum.get('growth_momentum'),
            'industry_rank': industry_rank,
            'industry_percentile': industry_percentile,
            'max_drawdown_1y': max_drawdown,
            'beta': beta,

            # Technical Indicators (4)
            'support_1': support_resistance.get('support_1'),
            'support_2': support_resistance.get('support_2'),
            'resistance_1': support_resistance.get('resistance_1'),
            'resistance_2': support_resistance.get('resistance_2'),
            'supertrend_value': supertrend.get('supertrend_value'),
            'trend': supertrend.get('trend'),
            'signal': supertrend.get('signal'),
            'rs_value': relative_strength.get('rs_value'),
            'rs_rank': relative_strength.get('rs_rank'),

            # Phase 1 Signal Scores (2)
            'smart_money_signal_score': smart_money_score,
            'smart_money_signal_text': smart_money_text,
            'smart_money_confidence': smart_money_conf,
            'volatility_context_score': volatility_context_score,
            'volatility_context_text': volatility_context_text,

            # Phase 2 Signal Scores (2)
            'factor_combination_bonus': factor_combo_bonus,
            'smart_money_momentum_score': smart_money_momentum,

            # Phase 3.1 Signal (1)
            'score_momentum_signal': score_momentum_signal,
            'score_momentum_text': score_momentum_text,
            'score_change_1m': change_1m,
            'score_change_2m': change_2m,

            # Phase 3.2 Signal (1)
            'sector_rotation_score': sector_rotation_score,
            'sector_momentum': sector_momentum,
            'sector_rank': sector_rank,
            'sector_percentile': sector_percentile
        }

        # Phase 3.9: Risk-Adjusted Returns (3)
        risk_adjusted = await self.calculate_risk_adjusted_returns()
        metrics['sharpe_ratio'] = risk_adjusted.get('sharpe_ratio')
        metrics['sortino_ratio'] = risk_adjusted.get('sortino_ratio')
        metrics['calmar_ratio'] = risk_adjusted.get('calmar_ratio')

        logger.info("All 29 metrics calculated successfully")
        logger.info(f"  - Fundamental/Risk: 9 metrics")
        logger.info(f"  - Phase 1 VaR Improvements: 7 metrics (Hurst, EWMA, Period VaR)")
        logger.info(f"  - Technical: 4 indicators (S/R, SuperTrend, RS)")
        logger.info(f"  - Phase 1 Signals: 2 scores (Smart Money, Volatility Context)")
        logger.info(f"  - Phase 2 Signals: 2 scores (Factor Combo, SM Momentum)")
        logger.info(f"  - Phase 3.1 Signal: 1 score (Score Momentum)")
        logger.info(f"  - Phase 3.2 Signal: 1 score (Sector Rotation)")
        logger.info(f"  - Phase 3.9 Risk-Adjusted: 3 ratios (Sharpe, Sortino, Calmar)")
        return metrics

    def calculate_conviction_score(
        self,
        value_score: float,
        quality_score: float,
        momentum_score: float,
        growth_score: float
    ) -> float:
        """
        Calculate conviction score based on 4 factor score agreement

        Conviction Score measures how much the 4 factors agree with each other.
        Lower standard deviation = higher conviction (factors pointing same direction)

        Args:
            value_score: Value factor score (0-100)
            quality_score: Quality factor score (0-100)
            momentum_score: Momentum factor score (0-100)
            growth_score: Growth factor score (0-100)

        Returns:
            conviction_score: 0-100 (higher = more factor agreement)

        Logic:
            - Base conviction from std: lower std = higher conviction
            - std=0 -> 100 points, std=25 -> 0 points
            - Direction bonus: extreme average (high or low) gets bonus
        """
        import numpy as np

        scores = [
            float(value_score or 50),
            float(quality_score or 50),
            float(momentum_score or 50),
            float(growth_score or 50)
        ]

        mean_score = np.mean(scores)
        std_score = np.std(scores)

        # Base conviction from std: lower std = higher conviction
        # std=0 -> 100, std=25 -> 0
        base_conviction = max(0, 100 - std_score * 4)

        # Direction bonus: extreme average (high or low) gets bonus
        # avg=50 -> 0 bonus, avg=75 or avg=25 -> 12.5 bonus
        direction_bonus = abs(mean_score - 50) * 0.5

        # Final conviction score
        conviction = min(100, base_conviction * 0.8 + direction_bonus)

        logger.debug(
            f"Conviction Score: {conviction:.1f} "
            f"(mean={mean_score:.1f}, std={std_score:.1f}, base={base_conviction:.1f}, bonus={direction_bonus:.1f})"
        )

        return round(conviction, 2)

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")


if __name__ == '__main__':
    """Test additional metrics calculation"""
    from dotenv import load_dotenv
    load_dotenv()

    symbol = '005930'

    # Mock factor scores for testing
    mock_factor_scores = {
        'value': {'weighted_result': {'weighted_score': 45.0}},
        'quality': {'weighted_result': {'weighted_score': 75.0}},
        'momentum': {'weighted_result': {'weighted_score': 60.0}},
        'growth': {'weighted_result': {'weighted_score': 55.0}}
    }

    calc = AdditionalMetricsCalculator(symbol, mock_factor_scores)
    metrics = calc.calculate_all_metrics()

    print("\n" + "="*80)
    print(f"Additional Metrics for {symbol}")
    print("="*80)

    for key, value in metrics.items():
        if value is not None:
            if isinstance(value, float):
                print(f"{key:<25}: {value:.2f}")
            else:
                print(f"{key:<25}: {value:,}" if isinstance(value, int) else f"{key:<25}: {value}")
        else:
            print(f"{key:<25}: N/A")

    calc.close()
