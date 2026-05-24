"""
Korean Stock Data Prefetcher Module

Centralized data prefetching for stock analysis.
Eliminates redundant DB queries by loading all needed data upfront.

Query Optimization:
- kr_intraday_total: 15+ queries -> 1 query (260 days OHLCV)
- kr_stock_detail: 5+ queries -> 1 query (theme, sector, etc.)
- kr_individual_investor_daily_trading: 4+ queries -> 1 query (90 days)
- kr_foreign_ownership: 2+ queries -> 1 query
- kr_stock_grade: 3+ queries -> 1 query (60 days historical)

Total reduction: ~30 queries/stock -> ~5 queries/stock (83% reduction)

File: kr/kr_data_prefetcher.py
Date: 2025-12-26
"""

import asyncio
import logging
from datetime import date
from typing import Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class KRDataPrefetcher:
    """
    Centralized data prefetching for Korean stock analysis.
    Eliminates redundant DB queries by loading all needed data upfront.
    """

    def __init__(self, db):
        """
        Initialize prefetcher with database manager.

        Args:
            db: AsyncDatabaseManager instance
        """
        self.db = db

    async def prefetch_all(self, symbol: str, analysis_date: date) -> Dict:
        """
        Prefetch all common data in parallel.

        Args:
            symbol: Stock symbol (e.g., '005930')
            analysis_date: Analysis date

        Returns:
            {
                'stock_detail': {...},       # kr_stock_detail row
                'price_data': {...},         # 260 days OHLCV (structured)
                'investor_trading': {...},   # 90 days investor flow
                'foreign_ownership': {...},  # foreign ownership data
                'historical_grades': [...]   # 60 days past grades
            }
        """
        tasks = [
            self._prefetch_stock_detail(symbol),
            self._prefetch_daily_prices(symbol, analysis_date),
            self._prefetch_investor_trading(symbol, analysis_date),
            self._prefetch_foreign_ownership(symbol, analysis_date),
            self._prefetch_historical_grades(symbol, analysis_date)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        return {
            'stock_detail': results[0] if not isinstance(results[0], Exception) else None,
            'price_data': results[1] if not isinstance(results[1], Exception) else None,
            'investor_trading': results[2] if not isinstance(results[2], Exception) else None,
            'foreign_ownership': results[3] if not isinstance(results[3], Exception) else None,
            'historical_grades': results[4] if not isinstance(results[4], Exception) else None
        }

    async def _prefetch_stock_detail(self, symbol: str) -> Optional[Dict]:
        """
        Prefetch stock detail data.

        Columns include:
        - Core: symbol, stock_name, theme, exchange
        - Classification: industry, industry_code
        """
        query = """
        SELECT
            symbol, stock_name, theme, exchange,
            industry, industry_code, fiscal_month
        FROM kr_stock_detail
        WHERE symbol = $1
        """

        try:
            result = await self.db.execute_query(query, symbol)
            if result:
                return dict(result[0])
            return None
        except Exception as e:
            logger.warning(f"{symbol}: Failed to prefetch stock_detail - {e}")
            return None

    async def _prefetch_daily_prices(self, symbol: str, analysis_date: date) -> Optional[Dict]:
        """
        Prefetch 260 days of OHLCV price data from kr_intraday_total.

        Returns structured data for easy access:
        {
            'dates': [date1, date2, ...],      # newest first
            'opens': [open1, open2, ...],
            'highs': [high1, high2, ...],
            'lows': [low1, low2, ...],
            'closes': [close1, close2, ...],
            'volumes': [vol1, vol2, ...],
            'raw': [{'date': date, 'open': ..., ...}, ...]  # original rows
        }
        """
        query = """
        SELECT date, open, high, low, close, volume
        FROM kr_intraday_total
        WHERE symbol = $1 AND date <= $2
        ORDER BY date DESC
        LIMIT 260
        """

        try:
            result = await self.db.execute_query(query, symbol, analysis_date)
            if not result:
                return None

            # Structure the data for easy access
            price_data = {
                'dates': [],
                'opens': [],
                'highs': [],
                'lows': [],
                'closes': [],
                'volumes': [],
                'raw': []
            }

            for row in result:
                price_data['dates'].append(row['date'])
                price_data['opens'].append(float(row['open']) if row['open'] else None)
                price_data['highs'].append(float(row['high']) if row['high'] else None)
                price_data['lows'].append(float(row['low']) if row['low'] else None)
                price_data['closes'].append(float(row['close']) if row['close'] else None)
                price_data['volumes'].append(int(row['volume']) if row['volume'] else None)
                price_data['raw'].append(dict(row))

            return price_data

        except Exception as e:
            logger.warning(f"{symbol}: Failed to prefetch daily prices - {e}")
            return None

    async def _prefetch_investor_trading(self, symbol: str, analysis_date: date) -> Optional[Dict]:
        """
        Prefetch 90 days of investor trading data.

        Returns:
        {
            'dates': [date1, date2, ...],
            'inst_net': [inst_net1, inst_net2, ...],
            'foreign_net': [foreign_net1, foreign_net2, ...],
            'raw': [{'date': ..., 'inst_net_volume': ..., ...}, ...]
        }
        """
        query = """
        SELECT date, inst_net_volume, foreign_net_volume,
               retail_net_volume
        FROM kr_individual_investor_daily_trading
        WHERE symbol = $1
          AND date <= $2
          AND date >= $2 - INTERVAL '90 days'
        ORDER BY date DESC
        """

        try:
            result = await self.db.execute_query(query, symbol, analysis_date)
            if not result:
                return None

            trading_data = {
                'dates': [],
                'inst_net': [],
                'foreign_net': [],
                'retail_net': [],
                'raw': []
            }

            for row in result:
                trading_data['dates'].append(row['date'])
                trading_data['inst_net'].append(int(row['inst_net_volume']) if row['inst_net_volume'] else 0)
                trading_data['foreign_net'].append(int(row['foreign_net_volume']) if row['foreign_net_volume'] else 0)
                trading_data['retail_net'].append(int(row['retail_net_volume']) if row['retail_net_volume'] else 0)
                trading_data['raw'].append(dict(row))

            return trading_data

        except Exception as e:
            logger.warning(f"{symbol}: Failed to prefetch investor trading - {e}")
            return None

    async def _prefetch_foreign_ownership(self, symbol: str, analysis_date: date) -> Optional[Dict]:
        """
        Prefetch foreign ownership data (60 days).

        Returns:
        {
            'dates': [date1, date2, ...],
            'ownership_pct': [pct1, pct2, ...],
            'latest': {...}  # most recent row
        }
        """
        query = """
        SELECT date, foreign_rate, foreign_rate_limit,
               foreign_ownership, foreign_limit
        FROM kr_foreign_ownership
        WHERE symbol = $1
          AND date <= $2
          AND date >= $2 - INTERVAL '60 days'
        ORDER BY date DESC
        """

        try:
            result = await self.db.execute_query(query, symbol, analysis_date)
            if not result:
                return None

            ownership_data = {
                'dates': [],
                'ownership_pct': [],
                'raw': [],
                'latest': dict(result[0]) if result else None
            }

            for row in result:
                ownership_data['dates'].append(row['date'])
                ownership_data['ownership_pct'].append(
                    float(row['foreign_rate']) if row['foreign_rate'] else None
                )
                ownership_data['raw'].append(dict(row))

            return ownership_data

        except Exception as e:
            logger.warning(f"{symbol}: Failed to prefetch foreign ownership - {e}")
            return None

    async def _prefetch_historical_grades(self, symbol: str, analysis_date: date) -> Optional[List[Dict]]:
        """
        Prefetch historical stock grades (60 days) for factor momentum calculation.

        Returns list of past grade records with:
        - date, final_score, value_score, quality_score, momentum_score, growth_score
        """
        query = """
        SELECT date, final_score,
               value_score, quality_score, momentum_score, growth_score,
               final_grade
        FROM kr_stock_grade
        WHERE symbol = $1
          AND date < $2
          AND date >= $2 - INTERVAL '60 days'
        ORDER BY date DESC
        """

        try:
            result = await self.db.execute_query(query, symbol, analysis_date)
            if not result:
                return []
            return [dict(row) for row in result]

        except Exception as e:
            logger.warning(f"{symbol}: Failed to prefetch historical grades - {e}")
            return []


class PrefetchedDataCalculator:
    """
    Calculator class that uses prefetched data instead of DB queries.
    Provides all calculation methods needed by kr_additional_metrics.py.
    """

    def __init__(self, prefetched: Dict):
        """
        Initialize with prefetched data.

        Args:
            prefetched: Result from KRDataPrefetcher.prefetch_all()
        """
        self.prefetched = prefetched
        self.price_data = prefetched.get('price_data')
        self.investor_trading = prefetched.get('investor_trading')
        self.foreign_ownership = prefetched.get('foreign_ownership')
        self.historical_grades = prefetched.get('historical_grades')
        self.stock_detail = prefetched.get('stock_detail')

    def get_closes(self, days: int = 60) -> List[float]:
        """
        Get close prices for specified days.

        Args:
            days: Number of days (default 60)

        Returns:
            List of close prices (newest first)
        """
        if not self.price_data or 'closes' not in self.price_data:
            return []
        closes = self.price_data['closes'][:days]
        return [c for c in closes if c is not None]

    def get_returns(self, days: int = 60) -> List[float]:
        """
        Calculate daily returns from close prices.

        Args:
            days: Number of days for return calculation

        Returns:
            List of daily returns (percentage, newest first)
        """
        if not self.price_data or 'closes' not in self.price_data:
            return []

        closes = self.price_data['closes'][:days + 1]
        closes = [c for c in closes if c is not None]

        if len(closes) < 2:
            return []

        # Return = (today - yesterday) / yesterday * 100
        # Note: closes are newest first, so closes[i] is newer than closes[i+1]
        return [(closes[i] - closes[i + 1]) / closes[i + 1] * 100
                for i in range(len(closes) - 1)]

    def calculate_volatility_annual(self) -> Optional[float]:
        """
        Calculate annualized volatility from 60-day daily returns.
        Replaces kr_additional_metrics.calculate_volatility_annual()

        Returns:
            Annualized volatility (%) or None
        """
        returns = self.get_returns(60)
        if len(returns) < 10:
            return None

        daily_std = float(np.std(returns))
        annual_volatility = daily_std * np.sqrt(252)

        return round(annual_volatility, 2)

    def calculate_var_95(self) -> Optional[float]:
        """
        Calculate Value at Risk (95%) from daily returns.
        Replaces kr_additional_metrics.calculate_var_95()

        Returns:
            VaR 95% (positive value) or None
        """
        returns = self.get_returns(60)
        if len(returns) < 20:
            return None

        var_95 = float(np.percentile(returns, 5))
        return round(abs(var_95), 2)

    def calculate_cvar_95(self) -> Optional[float]:
        """
        Calculate Conditional VaR (Expected Shortfall) at 95%.
        Replaces kr_additional_metrics.calculate_cvar_95()

        Returns:
            CVaR 95% (positive value) or None
        """
        returns = self.get_returns(60)
        if len(returns) < 20:
            return None

        var_95 = float(np.percentile(returns, 5))
        cvar_returns = [r for r in returns if r <= var_95]
        cvar_95 = float(np.mean(cvar_returns)) if cvar_returns else var_95

        return round(abs(cvar_95), 2)

    def calculate_hurst_exponent(self) -> Optional[float]:
        """
        Calculate Hurst Exponent using R/S (Rescaled Range) Analysis.
        H > 0.5: Trending (momentum persists)
        H = 0.5: Random Walk
        H < 0.5: Mean Reverting

        Returns:
            Hurst exponent (0.1 ~ 0.9) or 0.5 as default
        """
        returns = self.get_returns(252)  # 1 year of returns
        if len(returns) < 100:
            return 0.5  # Default for insufficient data

        returns_arr = np.array(returns[::-1])  # Reverse to chronological order
        N = len(returns_arr)
        min_chunk = 8

        if N < min_chunk * 4:
            return 0.5

        chunk_sizes = []
        rs_values = []

        for chunk_size in range(min_chunk, N // 4):
            rs_list = []

            for start in range(0, N - chunk_size + 1, chunk_size):
                chunk = returns_arr[start:start + chunk_size]

                # Mean adjusted series
                mean_adj = chunk - np.mean(chunk)

                # Cumulative deviation
                cumsum = np.cumsum(mean_adj)

                # Range (R)
                R = np.max(cumsum) - np.min(cumsum)

                # Standard Deviation (S)
                S = np.std(chunk, ddof=1)

                if S > 0:
                    rs_list.append(R / S)

            if rs_list:
                chunk_sizes.append(chunk_size)
                rs_values.append(np.mean(rs_list))

        if len(chunk_sizes) < 3:
            return 0.5

        # Log-log regression: log(R/S) = H * log(n) + c
        log_n = np.log(chunk_sizes)
        log_rs = np.log(rs_values)
        slope, _ = np.polyfit(log_n, log_rs, 1)

        # Constrain H to valid range
        H = max(0.1, min(0.9, slope))

        return round(H, 4)

    def calculate_var_95_ewma(self, lambda_param: float = 0.94) -> Optional[float]:
        """
        Calculate EWMA (Exponentially Weighted Moving Average) VaR 95%.
        EWMA gives more weight to recent volatility, capturing volatility clustering.

        Args:
            lambda_param: Decay factor (default 0.94, RiskMetrics standard)

        Returns:
            EWMA VaR 95% (positive value, %) or None
        """
        returns = self.get_returns(90)
        if len(returns) < 20:
            return None

        # Reverse to chronological order (oldest first)
        returns_arr = np.array(returns[::-1]) / 100  # Convert to decimal

        # Calculate EWMA variance
        ewma_var = returns_arr[0] ** 2
        for r in returns_arr[1:]:
            ewma_var = lambda_param * ewma_var + (1 - lambda_param) * (r ** 2)

        ewma_vol = np.sqrt(ewma_var) * 100  # Convert back to percentage

        # VaR 95% assuming normal distribution: -1.645 * volatility
        var_95_ewma = 1.645 * ewma_vol

        return round(var_95_ewma, 2)

    def calculate_var_99(self) -> Optional[float]:
        """
        Calculate Value at Risk (99%) from daily returns.
        1st percentile of return distribution.

        Returns:
            VaR 99% (positive value, %) or None
        """
        returns = self.get_returns(90)
        if len(returns) < 20:
            return None

        var_99 = float(np.percentile(returns, 1))
        return round(abs(var_99), 2)

    def calculate_period_var(self) -> Dict[str, Optional[float]]:
        """
        Calculate period-specific VaR using Hurst Exponent scaling.
        VaR_T = VaR_1d * T^H (Hurst-based scaling instead of sqrt(T))

        Returns:
            {
                'var_95_5d': float,
                'var_95_20d': float,
                'var_95_60d': float,
                'var_99_60d': float
            }
        """
        result = {
            'var_95_5d': None,
            'var_95_20d': None,
            'var_95_60d': None,
            'var_99_60d': None
        }

        # Get 1-day VaR values
        var_95_1d = self.calculate_var_95()
        var_99_1d = self.calculate_var_99()

        if var_95_1d is None:
            return result

        # Get Hurst exponent for scaling
        hurst = self.calculate_hurst_exponent()
        if hurst is None:
            hurst = 0.5  # Default to sqrt(T) scaling

        # Scale VaR to different periods: VaR_T = VaR_1d * T^H
        result['var_95_5d'] = round(var_95_1d * (5 ** hurst), 2)
        result['var_95_20d'] = round(var_95_1d * (20 ** hurst), 2)
        result['var_95_60d'] = round(var_95_1d * (60 ** hurst), 2)

        if var_99_1d is not None:
            result['var_99_60d'] = round(var_99_1d * (60 ** hurst), 2)

        return result

    # ===== Phase 2: Volatility Sizing Metrics =====

    def calculate_inv_vol_weight(self) -> Optional[float]:
        """
        Calculate Inverse Volatility Weight.
        Used for volatility-based position sizing.
        Higher weight = lower volatility = larger position allowed.

        Formula: 1 / volatility_annual

        Returns:
            Inverse volatility weight or None
        """
        vol = self.calculate_volatility_annual()
        if vol is None or vol <= 0:
            return None

        inv_vol = 1.0 / vol
        return round(inv_vol, 6)

    def calculate_downside_vol(self) -> Optional[float]:
        """
        Calculate Downside Volatility (Semideviation).
        Only considers negative returns, better captures downside risk.

        Formula: std(negative returns only) * sqrt(252)

        Returns:
            Annualized downside volatility (%) or None
        """
        returns = self.get_returns(90)
        if len(returns) < 20:
            return None

        # Filter negative returns only
        negative_returns = [r for r in returns if r < 0]

        if len(negative_returns) < 5:
            return None

        downside_std = float(np.std(negative_returns))
        downside_vol = downside_std * np.sqrt(252)

        return round(downside_vol, 2)

    def calculate_atr_20d(self) -> Optional[float]:
        """
        Calculate 20-day Average True Range (ATR).
        ATR measures volatility considering gaps.

        True Range = max(high-low, |high-prev_close|, |low-prev_close|)

        Returns:
            20-day ATR (absolute value) or None
        """
        if not self.price_data:
            return None

        highs = self.price_data.get('highs', [])[:21]
        lows = self.price_data.get('lows', [])[:21]
        closes = self.price_data.get('closes', [])[:21]

        if len(highs) < 21 or len(lows) < 21 or len(closes) < 21:
            return None

        # Filter None values
        valid_data = []
        for i in range(20):
            if all(x is not None for x in [highs[i], lows[i], closes[i], closes[i+1]]):
                valid_data.append({
                    'high': float(highs[i]),
                    'low': float(lows[i]),
                    'close': float(closes[i]),
                    'prev_close': float(closes[i+1])
                })

        if len(valid_data) < 10:
            return None

        # Calculate True Range for each day
        true_ranges = []
        for d in valid_data:
            tr = max(
                d['high'] - d['low'],
                abs(d['high'] - d['prev_close']),
                abs(d['low'] - d['prev_close'])
            )
            true_ranges.append(tr)

        atr = float(np.mean(true_ranges))
        return round(atr, 2)

    def calculate_atr_pct_20d(self) -> Optional[float]:
        """
        Calculate 20-day ATR as percentage of current price.
        Normalized ATR for cross-stock comparison.

        Formula: (ATR / current_price) * 100

        Returns:
            ATR percentage (%) or None
        """
        atr = self.calculate_atr_20d()
        if atr is None:
            return None

        closes = self.get_closes(1)
        if not closes or closes[0] is None or closes[0] <= 0:
            return None

        current_price = float(closes[0])
        atr_pct = (atr / current_price) * 100

        return round(atr_pct, 2)

    # ===== Phase 3: CVaR + Risk Budgeting Metrics =====

    def calculate_cvar_99(self) -> Optional[float]:
        """
        Calculate CVaR 99% (Expected Shortfall at 99%).
        Average loss in the worst 1% of cases.
        Basel III recommended metric for extreme risk.

        Returns:
            CVaR 99% (positive value, %) or None
        """
        returns = self.get_returns(252)  # 1 year of returns
        if len(returns) < 60:
            return None

        # VaR 99% threshold (1st percentile)
        var_99 = float(np.percentile(returns, 1))

        # CVaR = average of returns below VaR 99%
        tail_returns = [r for r in returns if r <= var_99]

        if not tail_returns:
            return round(abs(var_99), 2)

        cvar_99 = float(np.mean(tail_returns))
        return round(abs(cvar_99), 2)

    def calculate_drawdown_duration_avg(self) -> Optional[float]:
        """
        Calculate average drawdown duration in days.
        Measures how long it takes to recover from drawdowns.

        Returns:
            Average drawdown duration (days) or None
        """
        if not self.price_data or 'closes' not in self.price_data:
            return None

        closes = self.price_data['closes'][:252]
        closes = [c for c in closes if c is not None]

        if len(closes) < 60:
            return None

        # Reverse to chronological order (oldest first)
        closes = closes[::-1]

        # Track drawdown durations
        durations = []
        running_max = closes[0]
        current_dd_start = None

        for i, close in enumerate(closes):
            if close >= running_max:
                # New high or recovered
                if current_dd_start is not None:
                    # Record duration of completed drawdown
                    duration = i - current_dd_start
                    if duration > 0:
                        durations.append(duration)
                    current_dd_start = None
                running_max = close
            else:
                # In drawdown
                if current_dd_start is None:
                    current_dd_start = i

        if not durations:
            return 0.0  # No drawdowns

        avg_duration = float(np.mean(durations))
        return round(avg_duration, 1)

    def calculate_max_drawdown_1y(self) -> Optional[float]:
        """
        Calculate maximum drawdown over 1 year (252 days).
        Replaces kr_additional_metrics.calculate_max_drawdown_1y()

        Returns:
            Maximum drawdown (negative %) or None
        """
        if not self.price_data or 'closes' not in self.price_data:
            return None

        closes = self.price_data['closes'][:252]
        closes = [c for c in closes if c is not None]

        if len(closes) < 20:
            return None

        # Reverse to chronological order (oldest first)
        closes = closes[::-1]

        # Calculate running maximum and drawdowns
        running_max = closes[0]
        max_drawdown = 0

        for close in closes:
            if close > running_max:
                running_max = close
            drawdown = (close - running_max) / running_max * 100
            if drawdown < max_drawdown:
                max_drawdown = drawdown

        return round(max_drawdown, 2)

    def calculate_risk_adjusted_returns(self, rf_annual: float = 3.5) -> Dict:
        """
        Calculate Sharpe, Sortino, and Calmar ratios.
        Replaces kr_additional_metrics.calculate_risk_adjusted_returns()

        Args:
            rf_annual: Risk-free rate (annual %, default 3.5%)

        Returns:
            {'sharpe_ratio': float, 'sortino_ratio': float, 'calmar_ratio': float}
        """
        returns = self.get_returns(60)
        if len(returns) < 20:
            return {'sharpe_ratio': None, 'sortino_ratio': None, 'calmar_ratio': None}

        # 60-day risk-free rate
        rf_60d = rf_annual * (60 / 252)

        # Total 60-day return
        closes = self.get_closes(61)
        if len(closes) < 2:
            return {'sharpe_ratio': None, 'sortino_ratio': None, 'calmar_ratio': None}

        return_60d = (closes[0] - closes[-1]) / closes[-1] * 100

        # Daily std -> 60-day volatility
        daily_std = float(np.std(returns))
        trading_days = len(returns)
        volatility_60d = daily_std * np.sqrt(trading_days) if trading_days > 0 else 0

        # Sharpe Ratio
        if volatility_60d > 0:
            sharpe_ratio = (return_60d - rf_60d) / volatility_60d
        else:
            sharpe_ratio = 0

        # Sortino Ratio (downside deviation only)
        negative_returns = [r for r in returns if r < 0]
        if negative_returns:
            downside_std = float(np.std(negative_returns))
            negative_days = len(negative_returns)
            downside_deviation_60d = downside_std * np.sqrt(negative_days) if negative_days > 0 else 0

            if downside_deviation_60d > 0:
                sortino_ratio = (return_60d - rf_60d) / downside_deviation_60d
            else:
                sortino_ratio = 0
        else:
            sortino_ratio = None

        # Calmar Ratio
        max_drawdown = self.calculate_max_drawdown_1y()
        if max_drawdown and max_drawdown != 0:
            calmar_ratio = return_60d / abs(max_drawdown)
        else:
            calmar_ratio = None

        return {
            'sharpe_ratio': round(sharpe_ratio, 4) if sharpe_ratio is not None else None,
            'sortino_ratio': round(sortino_ratio, 4) if sortino_ratio is not None else None,
            'calmar_ratio': round(calmar_ratio, 4) if calmar_ratio is not None else None
        }

    def calculate_support_resistance(self) -> Dict[str, Optional[float]]:
        """
        Calculate support and resistance levels from price data.
        Replaces kr_additional_metrics.calculate_support_resistance()

        Returns:
            {'support_1': float, 'support_2': float,
             'resistance_1': float, 'resistance_2': float}
        """
        if not self.price_data:
            return {'support_1': None, 'support_2': None,
                    'resistance_1': None, 'resistance_2': None}

        # Get 60 days of price data
        highs = [h for h in self.price_data['highs'][:60] if h is not None]
        lows = [l for l in self.price_data['lows'][:60] if l is not None]
        closes = [c for c in self.price_data['closes'][:60] if c is not None]

        if not highs or not lows or not closes:
            return {'support_1': None, 'support_2': None,
                    'resistance_1': None, 'resistance_2': None}

        # Latest price data for pivot point calculation (most recent day)
        latest_high = highs[0]
        latest_low = lows[0]
        latest_close = closes[0]

        # Pivot Point calculation
        pivot = (latest_high + latest_low + latest_close) / 3

        # Standard Pivot levels
        r1 = 2 * pivot - latest_low
        s1 = 2 * pivot - latest_high
        r2 = pivot + (latest_high - latest_low)
        s2 = pivot - (latest_high - latest_low)

        # Adjust with 60-day highs/lows for more accurate levels
        high_60d = max(highs)
        low_60d = min(lows)

        # Blend pivot levels with 60-day extremes
        resistance_1 = round((r1 + high_60d) / 2, 2)
        resistance_2 = round(max(r2, high_60d), 2)
        support_1 = round((s1 + low_60d) / 2, 2)
        support_2 = round(min(s2, low_60d), 2)

        return {
            'support_1': support_1,
            'support_2': support_2,
            'resistance_1': resistance_1,
            'resistance_2': resistance_2
        }

    def calculate_investor_flow_30d(self) -> Tuple[Optional[int], Optional[int]]:
        """
        Calculate 30-day institutional and foreign net volume.
        Replaces kr_additional_metrics.calculate_investor_flow()

        Returns:
            (inst_net_30d, foreign_net_30d) tuple
        """
        if not self.investor_trading:
            return (None, None)

        inst_net = self.investor_trading.get('inst_net', [])[:30]
        foreign_net = self.investor_trading.get('foreign_net', [])[:30]

        inst_total = sum(inst_net) if inst_net else None
        foreign_total = sum(foreign_net) if foreign_net else None

        return (inst_total, foreign_total)

    def calculate_smart_money_flow(self) -> Dict:
        """
        Calculate smart money signal components.
        Used by calculate_smart_money_signal_score()

        Returns:
            {
                'inst_net_30d': int,
                'foreign_net_30d': int,
                'inst_net_prev_30d': int,
                'foreign_net_prev_30d': int,
                'flow_acceleration': float
            }
        """
        if not self.investor_trading:
            return {}

        inst_net = self.investor_trading.get('inst_net', [])
        foreign_net = self.investor_trading.get('foreign_net', [])

        # Current 30 days
        inst_current = sum(inst_net[:30]) if len(inst_net) >= 30 else sum(inst_net)
        foreign_current = sum(foreign_net[:30]) if len(foreign_net) >= 30 else sum(foreign_net)

        # Previous 30 days (30-60 days ago)
        inst_prev = sum(inst_net[30:60]) if len(inst_net) >= 60 else 0
        foreign_prev = sum(foreign_net[30:60]) if len(foreign_net) >= 60 else 0

        # Flow acceleration
        current_total = inst_current + foreign_current
        prev_total = inst_prev + foreign_prev

        if prev_total != 0:
            flow_acceleration = (current_total - prev_total) / abs(prev_total) * 100
        else:
            flow_acceleration = 0 if current_total == 0 else 100

        return {
            'inst_net_30d': inst_current,
            'foreign_net_30d': foreign_current,
            'inst_net_prev_30d': inst_prev,
            'foreign_net_prev_30d': foreign_prev,
            'flow_acceleration': round(flow_acceleration, 2)
        }

    def calculate_factor_momentum(self) -> Dict[str, Optional[float]]:
        """
        Calculate factor momentum from historical grades.
        Replaces kr_additional_metrics.calculate_factor_momentum()

        Returns:
            {
                'value_momentum': float,
                'quality_momentum': float,
                'momentum_momentum': float,
                'growth_momentum': float
            }
        """
        result = {
            'value_momentum': None,
            'quality_momentum': None,
            'momentum_momentum': None,
            'growth_momentum': None
        }

        if not self.historical_grades or len(self.historical_grades) < 2:
            return result

        grades = self.historical_grades

        # Find 30-day old record for comparison
        current = grades[0] if grades else None
        past_30d = None

        for g in grades:
            # Find record approximately 30 days ago
            if len([x for x in grades if x]) >= 20:
                idx = min(20, len(grades) - 1)
                past_30d = grades[idx]
                break

        if not current or not past_30d:
            return result

        # Calculate momentum for each factor
        factors = ['value_score', 'quality_score', 'momentum_score', 'growth_score']
        momentum_keys = ['value_momentum', 'quality_momentum', 'momentum_momentum', 'growth_momentum']

        for factor, momentum_key in zip(factors, momentum_keys):
            current_score = current.get(factor)
            past_score = past_30d.get(factor)

            if current_score is not None and past_score is not None:
                momentum = float(current_score) - float(past_score)
                result[momentum_key] = round(momentum, 2)

        return result

    def calculate_supertrend(self, period: int = 90, multiplier: float = 2.0) -> Dict:
        """
        Calculate SuperTrend indicator from price data.
        Replaces kr_additional_metrics.calculate_supertrend()

        Args:
            period: ATR period (default 90)
            multiplier: ATR multiplier (default 2.0)

        Returns:
            {'supertrend_value': float, 'trend': str, 'signal': str}
        """
        if not self.price_data:
            return {'supertrend_value': None, 'trend': None, 'signal': None}

        highs = [h for h in self.price_data['highs'][:period] if h is not None]
        lows = [l for l in self.price_data['lows'][:period] if l is not None]
        closes = [c for c in self.price_data['closes'][:period] if c is not None]

        if len(highs) < period or len(lows) < period or len(closes) < period:
            return {'supertrend_value': None, 'trend': None, 'signal': None}

        # Calculate True Range
        tr_list = []
        for i in range(len(closes) - 1):
            high_low = highs[i] - lows[i]
            high_close = abs(highs[i] - closes[i + 1])
            low_close = abs(lows[i] - closes[i + 1])
            tr = max(high_low, high_close, low_close)
            tr_list.append(tr)

        if not tr_list:
            return {'supertrend_value': None, 'trend': None, 'signal': None}

        # ATR (simple average for simplicity)
        atr = np.mean(tr_list[:14]) if len(tr_list) >= 14 else np.mean(tr_list)

        # SuperTrend calculation
        latest_close = closes[0]
        latest_high = highs[0]
        latest_low = lows[0]
        hl2 = (latest_high + latest_low) / 2

        upper_band = hl2 + multiplier * atr
        lower_band = hl2 - multiplier * atr

        # Determine trend
        if latest_close > upper_band:
            trend = 'UP'
            supertrend_value = lower_band
            signal = 'BUY'
        elif latest_close < lower_band:
            trend = 'DOWN'
            supertrend_value = upper_band
            signal = 'SELL'
        else:
            trend = 'NEUTRAL'
            supertrend_value = (upper_band + lower_band) / 2
            signal = 'HOLD'

        return {
            'supertrend_value': round(supertrend_value, 2),
            'trend': trend,
            'signal': signal
        }

    def calculate_relative_strength(self) -> Dict:
        """
        Calculate IBD-style Relative Strength value.
        RS = 3M return * 40% + 6M return * 20% + 9M return * 20% + 12M return * 20%

        Returns:
            {'rs_value': float, 'rs_rank': int (placeholder)}
        """
        if not self.price_data or 'closes' not in self.price_data:
            return {'rs_value': None, 'rs_rank': None}

        closes = self.price_data['closes']
        closes = [c for c in closes if c is not None]

        if len(closes) < 63:  # Need at least 3 months
            return {'rs_value': None, 'rs_rank': None}

        def safe_return(days):
            if len(closes) > days:
                return (closes[0] - closes[days]) / closes[days] * 100
            return 0

        ret_3m = safe_return(63)
        ret_6m = safe_return(126) if len(closes) > 126 else ret_3m
        ret_9m = safe_return(189) if len(closes) > 189 else ret_6m
        ret_12m = safe_return(252) if len(closes) > 252 else ret_9m

        rs_value = ret_3m * 0.4 + ret_6m * 0.2 + ret_9m * 0.2 + ret_12m * 0.2

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

        return {
            'rs_value': round(rs_value, 2),
            'rs_rank': rs_rank
        }

    def get_theme(self) -> Optional[str]:
        """Get stock theme from prefetched stock_detail."""
        if self.stock_detail:
            return self.stock_detail.get('theme')
        return None

    def get_latest_price(self) -> Optional[float]:
        """Get latest close price."""
        if self.price_data and self.price_data.get('closes'):
            return self.price_data['closes'][0]
        return None

    def get_price_change_pct(self, days: int = 1) -> Optional[float]:
        """
        Get price change percentage over N days.

        Args:
            days: Number of days (default 1 for daily change)

        Returns:
            Price change percentage or None
        """
        if not self.price_data or 'closes' not in self.price_data:
            return None

        closes = self.price_data['closes']
        if len(closes) <= days:
            return None

        current = closes[0]
        past = closes[days]

        if current and past and past != 0:
            return round((current - past) / past * 100, 2)
        return None
