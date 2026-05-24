"""
US Agent Metrics - Multi-Agent AI Support Module

This module provides additional metrics for multi-agent AI systems:
1. ATR-based stop loss / take profit
2. Entry timing score
3. Position size calculation
4. IV percentile (US-specific)
5. Insider signal (US-specific)
6. Action triggers generation
7. Scenario probability analysis

Usage:
    from us_agent_metrics import USAgentMetrics

    agent_metrics = USAgentMetrics(db_manager)
    result = await agent_metrics.calculate_all(symbol, analysis_date)

File: us/us_agent_metrics.py
"""

import asyncio
import logging
from datetime import date, timedelta
from typing import Dict, List, Optional, Any
from decimal import Decimal

from us_db_async import AsyncDatabaseManager

logger = logging.getLogger(__name__)

# ========================================================================
# Phase 3.1 ATR Multiplier Configuration (Backtest Results)
# ========================================================================
# Backtest: 2024-10 ~ 2025-11 (14 months, 10000 samples)
# Optimization target: Max avg PnL with acceptable trigger rate
#
# | ATR Mult | Trigger | Avg PnL | Win Rate | P/L Ratio |
# |----------|---------|---------|----------|-----------|
# | 1.0x     | 70.7%   | +2.12%  | 15.7%    | 6.87      |
# | 2.0x     | 45.8%   | +6.93%  | 24.2%    | 4.65      |
# | 4.0x     | 17.7%   | +13.34% | 28.7%    | 3.77      |
# ========================================================================

ATR_MULTIPLIER_DEFAULT = 4.0

ATR_MULTIPLIER_BY_SECTOR = {
    'ENERGY': 1.5,
    'CONSUMER DEFENSIVE': 1.5,
    'BASIC MATERIALS': 3.5,
    'REAL ESTATE': 3.5,
    'TECHNOLOGY': 3.5,
    'FINANCIAL SERVICES': 3.5,
    'INDUSTRIALS': 3.5,
    'HEALTHCARE': 3.5,
    'UTILITIES': 3.5,
    'CONSUMER CYCLICAL': 3.5,
    'COMMUNICATION SERVICES': 3.5
}

# ========================================================================
# Phase 3.1 Scenario Calibration Coefficients (Regression Results)
# ========================================================================
# Analysis: Predicted vs Actual scenario occurrence rate
# Formula: calibrated_prob = slope * predicted_prob + intercept
#
# | Scenario | Predicted Avg | Actual Rate | Bias      |
# |----------|---------------|-------------|-----------|
# | bull     | 29.1%         | 24.7%       | Overest.  |
# | base     | 40.1%         | 47.0%       | Underest. |
# | bear     | 30.7%         | 28.3%       | Slight    |
#
# Calibration formulas (linear regression):
# - bull: calibrated = 0.124 * predicted + 0.233 (R²=0.113)
# - bear: calibrated = 0.612 * predicted + 0.095 (R²=0.956)
# - base: derived from (1 - bull - bear) to ensure sum = 100%
# ========================================================================

SCENARIO_CALIBRATION = {
    'bull': {'slope': 0.124, 'intercept': 0.233},
    'bear': {'slope': 0.612, 'intercept': 0.095}
}


class USAgentMetrics:
    """
    US Agent Metrics Calculator

    Provides additional metrics for multi-agent AI investment decisions:
    - Risk management (ATR-based stop/take profit, position sizing)
    - Entry timing analysis
    - US-specific indicators (IV percentile, insider signals)
    - Action triggers and scenario analysis
    """

    def __init__(self, db: AsyncDatabaseManager):
        """
        Initialize USAgentMetrics

        Args:
            db: AsyncDatabaseManager instance
        """
        self.db = db

    # ========================================================================
    # 1. ATR-based Stop Loss / Take Profit
    # ========================================================================

    async def calculate_atr_stop_take_profit(
        self, symbol: str, analysis_date: date, sector: Optional[str] = None
    ) -> Dict[str, Optional[float]]:
        """
        Calculate ATR-based stop loss and take profit levels

        Uses us_atr table (14-day ATR) and us_daily table for current price.

        Phase 3.1 Update:
            - ATR multiplier now sector-specific based on backtest results
            - Default: 4.0x ATR (optimal for avg PnL)
            - ENERGY/CONSUMER DEFENSIVE: 1.5x ATR (higher volatility)
            - Other sectors: 3.5x ATR

        Logic:
            - atr_pct = (ATR / close) * 100
            - atr_multiplier = sector-specific or default (4.0x)
            - stop_loss_pct = -atr_multiplier * atr_pct
            - take_profit_pct = atr_multiplier * 1.5 * atr_pct (R:R = 1.5)
            - risk_reward_ratio = 1.5

        Args:
            symbol: Stock symbol
            analysis_date: Analysis date
            sector: Stock sector (for sector-specific ATR multiplier)

        Returns:
            {
                'atr_pct': float or None,
                'stop_loss_pct': float or None,
                'take_profit_pct': float or None,
                'risk_reward_ratio': float or None,
                'atr_multiplier': float or None
            }
        """
        query = """
        SELECT a.atr, d.close
        FROM us_atr a
        JOIN us_daily d ON a.symbol = d.symbol AND a.date = d.date
        WHERE a.symbol = $1 AND a.date <= $2
        ORDER BY a.date DESC
        LIMIT 1
        """

        try:
            result = await self.db.execute_query(query, symbol, analysis_date)

            if not result or not result[0].get('atr') or not result[0].get('close'):
                return {
                    'atr_pct': None,
                    'stop_loss_pct': None,
                    'take_profit_pct': None,
                    'risk_reward_ratio': None,
                    'atr_multiplier': None
                }

            atr = float(result[0]['atr'])
            close = float(result[0]['close'])

            if close <= 0:
                return {
                    'atr_pct': None,
                    'stop_loss_pct': None,
                    'take_profit_pct': None,
                    'risk_reward_ratio': None,
                    'atr_multiplier': None
                }

            # Phase 3.1: Sector-specific ATR multiplier
            atr_multiplier = ATR_MULTIPLIER_BY_SECTOR.get(sector, ATR_MULTIPLIER_DEFAULT)

            atr_pct = (atr / close) * 100
            stop_loss_pct = -atr_multiplier * atr_pct
            take_profit_pct = atr_multiplier * 1.5 * atr_pct  # R:R = 1.5
            risk_reward_ratio = 1.5

            return {
                'atr_pct': round(atr_pct, 2),
                'stop_loss_pct': round(stop_loss_pct, 2),
                'take_profit_pct': round(take_profit_pct, 2),
                'risk_reward_ratio': round(risk_reward_ratio, 2),
                'atr_multiplier': atr_multiplier
            }

        except Exception as e:
            logger.error(f"{symbol}: ATR calculation failed - {e}")
            return {
                'atr_pct': None,
                'stop_loss_pct': None,
                'take_profit_pct': None,
                'risk_reward_ratio': None,
                'atr_multiplier': None
            }

    # ========================================================================
    # 2. Entry Timing Score
    # ========================================================================

    async def calculate_entry_timing_score(
        self, symbol: str, analysis_date: date, current_score: float
    ) -> Dict[str, Optional[float]]:
        """
        Calculate entry timing score

        Components:
            1. score_trend_2w: Score change over 2 weeks (from us_stock_grade)
            2. price_position_52w: Current price position in 52-week range (0-100)
            3. entry_timing_score: Composite score (0-100)

        Logic:
            - Score trend positive + price not at 52w high = Good entry
            - Score trend negative + price at 52w high = Bad entry

        Args:
            symbol: Stock symbol
            analysis_date: Analysis date
            current_score: Current final_score from us_stock_grade

        Returns:
            {
                'entry_timing_score': float or None,
                'score_trend_2w': float or None,
                'price_position_52w': float or None
            }
        """
        # Get score from 2 weeks ago
        score_query = """
        SELECT final_score
        FROM us_stock_grade
        WHERE symbol = $1 AND date <= $2::date - INTERVAL '14 days'
        ORDER BY date DESC
        LIMIT 1
        """

        # Get 52-week high/low and current price
        price_query = """
        WITH price_range AS (
            SELECT
                MAX(high) as high_52w,
                MIN(low) as low_52w
            FROM us_daily
            WHERE symbol = $1
              AND date > $2::date - INTERVAL '252 days'
              AND date <= $2
        ),
        current_price AS (
            SELECT close
            FROM us_daily
            WHERE symbol = $1 AND date <= $2
            ORDER BY date DESC
            LIMIT 1
        )
        SELECT
            pr.high_52w,
            pr.low_52w,
            cp.close as current_price
        FROM price_range pr, current_price cp
        """

        try:
            score_result, price_result = await asyncio.gather(
                self.db.execute_query(score_query, symbol, analysis_date),
                self.db.execute_query(price_query, symbol, analysis_date)
            )

            # Score trend calculation
            score_trend_2w = None
            if score_result and score_result[0].get('final_score') is not None:
                old_score = float(score_result[0]['final_score'])
                score_trend_2w = current_score - old_score

            # Price position calculation (0 = 52w low, 100 = 52w high)
            price_position_52w = None
            if price_result and price_result[0].get('high_52w'):
                r = price_result[0]
                high_52w = float(r['high_52w'])
                low_52w = float(r['low_52w'])
                current_price = float(r['current_price'])

                if high_52w > low_52w:
                    price_position_52w = ((current_price - low_52w) / (high_52w - low_52w)) * 100

            # Entry timing score calculation
            entry_timing_score = None
            if score_trend_2w is not None and price_position_52w is not None:
                # Positive score trend is good (0-50 points)
                trend_component = min(50, max(0, (score_trend_2w + 10) * 2.5))

                # Price not at extreme high is good (0-50 points)
                # Best entry: price at 30-70% of 52w range
                if price_position_52w <= 30:
                    position_component = 50  # Near 52w low - good entry
                elif price_position_52w <= 70:
                    position_component = 40  # Middle range - decent entry
                elif price_position_52w <= 90:
                    position_component = 25  # Near high - cautious
                else:
                    position_component = 10  # At 52w high - risky entry

                entry_timing_score = trend_component + position_component

            return {
                'entry_timing_score': round(entry_timing_score, 1) if entry_timing_score else None,
                'score_trend_2w': round(score_trend_2w, 2) if score_trend_2w else None,
                'price_position_52w': round(price_position_52w, 2) if price_position_52w else None
            }

        except Exception as e:
            logger.error(f"{symbol}: Entry timing calculation failed - {e}")
            return {
                'entry_timing_score': None,
                'score_trend_2w': None,
                'price_position_52w': None
            }

    # ========================================================================
    # 3. Position Size Calculation
    # ========================================================================

    async def calculate_position_size(
        self,
        symbol: str,
        analysis_date: date,
        stop_loss_pct: Optional[float],
        var_95: Optional[float],
        max_risk_per_trade: float = 2.0
    ) -> Optional[float]:
        """
        Calculate recommended position size as percentage of portfolio

        Logic:
            - Based on risk per trade (default 2% of portfolio)
            - Position size = max_risk_per_trade / abs(stop_loss_pct)
            - Adjusted by VaR if available
            - Capped at 10% maximum

        Args:
            symbol: Stock symbol
            analysis_date: Analysis date
            stop_loss_pct: Stop loss percentage (negative)
            var_95: Value at Risk 95% (positive)
            max_risk_per_trade: Maximum risk per trade as % of portfolio (default 2%)

        Returns:
            position_size_pct: Recommended position size (0-10%)
        """
        try:
            if stop_loss_pct is None or stop_loss_pct >= 0:
                return None

            # Base position size from stop loss
            base_position = max_risk_per_trade / abs(stop_loss_pct) * 100

            # Adjust by VaR if available (more volatile = smaller position)
            if var_95 and var_95 > 0:
                var_adjustment = min(1.0, 2.0 / var_95)  # Reduce if VaR > 2%
                base_position *= var_adjustment

            # Cap at 10% maximum position size
            position_size_pct = min(10.0, max(1.0, base_position))

            return round(position_size_pct, 1)

        except Exception as e:
            logger.error(f"{symbol}: Position size calculation failed - {e}")
            return None

    # ========================================================================
    # 4. Multi-Tier Stop Loss (Phase 3.3 Enhancement)
    # ========================================================================

    async def calculate_trailing_stop(
        self,
        symbol: str,
        analysis_date: date,
        entry_price: float,
        max_price_since_entry: float,
        current_price: float
    ) -> Dict[str, Any]:
        """
        Calculate Trailing Stop Loss

        Protects profits after 10%+ gain by selling on 5% drop from peak.

        Logic:
            - Activate when current_return > 10%
            - Trigger when price drops 5% from peak since entry

        Args:
            symbol: Stock symbol
            analysis_date: Analysis date
            entry_price: Entry price when position opened
            max_price_since_entry: Maximum price reached since entry
            current_price: Current price

        Returns:
            {
                'trailing_stop_triggered': bool,
                'current_return': float (decimal, e.g., 0.15 = 15%),
                'return_from_peak': float (decimal),
                'max_price_since_entry': float
            }
        """
        try:
            if entry_price <= 0 or max_price_since_entry <= 0 or current_price <= 0:
                return {
                    'trailing_stop_triggered': False,
                    'current_return': None,
                    'return_from_peak': None,
                    'max_price_since_entry': None
                }

            # Calculate returns
            current_return = (current_price - entry_price) / entry_price
            return_from_peak = (current_price - max_price_since_entry) / max_price_since_entry

            # Trigger conditions
            profit_threshold_met = current_return > 0.10  # 10% profit
            peak_drop_threshold_met = return_from_peak <= -0.05  # 5% drop from peak

            triggered = profit_threshold_met and peak_drop_threshold_met

            return {
                'trailing_stop_triggered': triggered,
                'current_return': round(current_return, 4),
                'return_from_peak': round(return_from_peak, 4),
                'max_price_since_entry': max_price_since_entry
            }

        except Exception as e:
            logger.error(f"{symbol}: Trailing stop calculation failed - {e}")
            return {
                'trailing_stop_triggered': False,
                'current_return': None,
                'return_from_peak': None,
                'max_price_since_entry': None
            }

    async def calculate_time_based_stop(
        self,
        symbol: str,
        analysis_date: date,
        entry_date: date,
        entry_price: float,
        current_price: float
    ) -> Dict[str, Any]:
        """
        Calculate Time-Based Stop Loss

        Exits underperforming positions after holding period.

        Logic:
            - After 60 days of holding
            - If position is down 5% or more
            - Exit to free capital for better opportunities

        Args:
            symbol: Stock symbol
            analysis_date: Analysis date
            entry_date: Date when position was opened
            entry_price: Entry price
            current_price: Current price

        Returns:
            {
                'time_stop_triggered': bool,
                'holding_days': int,
                'current_return': float (decimal)
            }
        """
        try:
            if entry_price <= 0 or current_price <= 0:
                return {
                    'time_stop_triggered': False,
                    'holding_days': None,
                    'current_return': None
                }

            # Calculate holding period
            holding_days = (analysis_date - entry_date).days

            # Calculate return
            current_return = (current_price - entry_price) / entry_price

            # Trigger conditions
            holding_threshold_met = holding_days >= 60  # 60 days
            loss_threshold_met = current_return <= -0.05  # -5% loss

            triggered = holding_threshold_met and loss_threshold_met

            return {
                'time_stop_triggered': triggered,
                'holding_days': holding_days,
                'current_return': round(current_return, 4)
            }

        except Exception as e:
            logger.error(f"{symbol}: Time-based stop calculation failed - {e}")
            return {
                'time_stop_triggered': False,
                'holding_days': None,
                'current_return': None
            }

    async def _get_score_rank(
        self,
        symbol: str,
        analysis_date: date,
        current_score: float
    ) -> Optional[float]:
        """
        Get score percentile rank among all stocks

        Args:
            symbol: Stock symbol
            analysis_date: Analysis date
            current_score: Current final_score

        Returns:
            Percentile rank (0-1), e.g., 0.85 = top 15%
        """
        query = """
        WITH score_ranks AS (
            SELECT
                symbol,
                final_score,
                PERCENT_RANK() OVER (ORDER BY final_score) as rank_pct
            FROM us_stock_grade
            WHERE date <= $1
            ORDER BY date DESC
            LIMIT 5000
        )
        SELECT rank_pct
        FROM score_ranks
        WHERE symbol = $2
        """

        try:
            result = await self.db.execute_query(query, analysis_date, symbol)

            if result and result[0].get('rank_pct') is not None:
                return float(result[0]['rank_pct'])

            return None

        except Exception as e:
            logger.error(f"{symbol}: Score rank calculation failed - {e}")
            return None

    async def calculate_score_degradation_stop(
        self,
        symbol: str,
        analysis_date: date,
        initial_score: float,
        initial_score_rank: float,
        current_score: float
    ) -> Dict[str, Any]:
        """
        Calculate Score Degradation Stop Loss

        Exits positions when factor scores deteriorate significantly.

        Logic:
            - Initial position was top 20% (rank > 0.80)
            - Current position dropped to bottom 50% (rank < 0.50)
            - Indicates fundamental deterioration

        Args:
            symbol: Stock symbol
            analysis_date: Analysis date
            initial_score: Final score when position opened
            initial_score_rank: Percentile rank at entry (0-1)
            current_score: Current final score

        Returns:
            {
                'score_degradation_stop_triggered': bool,
                'initial_score': float,
                'current_score': float,
                'initial_score_rank': float,
                'current_score_rank': float,
                'score_change': float
            }
        """
        try:
            # Get current score rank
            current_score_rank = await self._get_score_rank(symbol, analysis_date, current_score)

            if current_score_rank is None:
                return {
                    'score_degradation_stop_triggered': False,
                    'initial_score': initial_score,
                    'current_score': current_score,
                    'initial_score_rank': initial_score_rank,
                    'current_score_rank': None,
                    'score_change': current_score - initial_score
                }

            # Trigger conditions
            was_top_20_pct = initial_score_rank > 0.80  # Was in top 20%
            now_bottom_50_pct = current_score_rank < 0.50  # Now in bottom 50%

            triggered = was_top_20_pct and now_bottom_50_pct

            return {
                'score_degradation_stop_triggered': triggered,
                'initial_score': initial_score,
                'current_score': current_score,
                'initial_score_rank': round(initial_score_rank, 4),
                'current_score_rank': round(current_score_rank, 4),
                'score_change': round(current_score - initial_score, 2)
            }

        except Exception as e:
            logger.error(f"{symbol}: Score degradation stop calculation failed - {e}")
            return {
                'score_degradation_stop_triggered': False,
                'initial_score': initial_score,
                'current_score': current_score,
                'initial_score_rank': initial_score_rank,
                'current_score_rank': None,
                'score_change': None
            }

    def _determine_stop_type(
        self,
        atr_triggered: bool,
        trailing_triggered: bool,
        time_triggered: bool,
        score_triggered: bool
    ) -> str:
        """
        Determine which stop loss was triggered (priority order)

        Priority:
            1. Trailing Stop (protect profits)
            2. ATR Stop (volatility-based)
            3. Score Degradation (fundamental deterioration)
            4. Time-Based Stop (capital efficiency)

        Returns:
            'Trailing', 'ATR', 'Score_Degradation', 'Time_Based', or 'None'
        """
        if trailing_triggered:
            return 'Trailing'
        elif atr_triggered:
            return 'ATR'
        elif score_triggered:
            return 'Score_Degradation'
        elif time_triggered:
            return 'Time_Based'
        else:
            return 'None'

    async def calculate_multi_tier_stop_loss(
        self,
        symbol: str,
        analysis_date: date,
        sector: str,
        current_price: float,
        current_score: float,
        # Optional: Only provided for held positions
        entry_date: Optional[date] = None,
        entry_price: Optional[float] = None,
        max_price_since_entry: Optional[float] = None,
        initial_score: Optional[float] = None,
        initial_score_rank: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Calculate Multi-Tier Stop Loss (4 layers)

        Phase 3.3 Enhancement: Comprehensive risk management system

        Tiers:
            1. ATR Stop: Always calculated (volatility-based)
            2. Trailing Stop: For profitable positions (10%+ gain)
            3. Time-Based Stop: For long-held losers (60d + -5%)
            4. Score Degradation: For fundamental deterioration (top 20% -> bottom 50%)

        Args:
            symbol: Stock symbol
            analysis_date: Analysis date
            sector: Stock sector (for ATR multiplier)
            current_price: Current price
            current_score: Current final_score
            entry_date: Optional - Date position opened
            entry_price: Optional - Entry price
            max_price_since_entry: Optional - Maximum price since entry
            initial_score: Optional - Score at entry
            initial_score_rank: Optional - Score rank at entry (0-1)

        Returns:
            {
                'any_stop_triggered': bool,
                'stop_type': str ('Trailing', 'ATR', 'Score_Degradation', 'Time_Based', 'None'),
                'atr_stop': Dict,
                'trailing_stop': Dict or None,
                'time_stop': Dict or None,
                'score_stop': Dict or None,
                'recommendation': str
            }
        """
        try:
            # Tier 1: ATR Stop (always calculated)
            atr_stop = await self.calculate_atr_stop_take_profit(symbol, analysis_date, sector)
            atr_triggered = False

            # Check if ATR stop would be triggered
            # Note: Actual trigger requires comparison with entry price
            # For now, we mark as available but not triggered without entry data
            if entry_price and atr_stop.get('stop_loss_pct'):
                stop_loss_price = entry_price * (1 + atr_stop['stop_loss_pct'] / 100)
                atr_triggered = current_price <= stop_loss_price

            # Tier 2-4: Only if position data provided
            trailing_stop = None
            trailing_triggered = False

            time_stop = None
            time_triggered = False

            score_stop = None
            score_triggered = False

            if entry_price and max_price_since_entry:
                trailing_stop = await self.calculate_trailing_stop(
                    symbol, analysis_date, entry_price, max_price_since_entry, current_price
                )
                trailing_triggered = trailing_stop.get('trailing_stop_triggered', False)

            if entry_date and entry_price:
                time_stop = await self.calculate_time_based_stop(
                    symbol, analysis_date, entry_date, entry_price, current_price
                )
                time_triggered = time_stop.get('time_stop_triggered', False)

            if initial_score is not None and initial_score_rank is not None:
                score_stop = await self.calculate_score_degradation_stop(
                    symbol, analysis_date, initial_score, initial_score_rank, current_score
                )
                score_triggered = score_stop.get('score_degradation_stop_triggered', False)

            # Determine overall status
            any_stop_triggered = any([
                atr_triggered,
                trailing_triggered,
                time_triggered,
                score_triggered
            ])

            stop_type = self._determine_stop_type(
                atr_triggered, trailing_triggered, time_triggered, score_triggered
            )

            # Generate recommendation
            if any_stop_triggered:
                recommendation = f"SELL: {stop_type} stop triggered"
            else:
                recommendation = "HOLD: No stop loss triggered"

            return {
                'any_stop_triggered': any_stop_triggered,
                'stop_type': stop_type,
                'atr_stop': atr_stop,
                'trailing_stop': trailing_stop,
                'time_stop': time_stop,
                'score_stop': score_stop,
                'recommendation': recommendation
            }

        except Exception as e:
            logger.error(f"{symbol}: Multi-tier stop loss calculation failed - {e}")
            return {
                'any_stop_triggered': False,
                'stop_type': 'Error',
                'atr_stop': None,
                'trailing_stop': None,
                'time_stop': None,
                'score_stop': None,
                'recommendation': 'ERROR: Calculation failed'
            }

    # ========================================================================
    # 5. IV Percentile (US-specific)
    # ========================================================================

    async def calculate_iv_percentile(
        self, symbol: str, analysis_date: date
    ) -> Optional[int]:
        """
        Calculate IV Percentile - where current IV ranks in 1-year range

        Uses us_option_daily_summary table.
        Stored in us_stock_grade (requires 1-year data aggregation).

        Logic:
            - Get 252 trading days of IV data
            - Calculate percentile rank of current IV
            - 0 = lowest IV in year, 100 = highest IV in year

        Interpretation:
            - < 20: Low IV (cheap options, good for buying)
            - 20-80: Normal IV
            - > 80: High IV (expensive options, good for selling)

        Args:
            symbol: Stock symbol
            analysis_date: Analysis date

        Returns:
            iv_percentile: 0-100 or None if insufficient data
        """
        query = """
        SELECT avg_implied_volatility
        FROM us_option_daily_summary
        WHERE symbol = $1
          AND date >= $2::date - INTERVAL '252 days'
          AND date <= $2
          AND avg_implied_volatility IS NOT NULL
        ORDER BY date DESC
        """

        try:
            result = await self.db.execute_query(query, symbol, analysis_date)

            if not result or len(result) < 20:
                return None

            current_iv = float(result[0]['avg_implied_volatility'])
            iv_list = [float(r['avg_implied_volatility']) for r in result]

            # Calculate percentile
            count_below = sum(1 for iv in iv_list if iv <= current_iv)
            iv_percentile = (count_below / len(iv_list)) * 100

            return round(iv_percentile)

        except Exception as e:
            logger.error(f"{symbol}: IV percentile calculation failed - {e}")
            return None

    # ========================================================================
    # 6. Insider Signal (US-specific)
    # ========================================================================

    async def calculate_insider_signal(
        self, symbol: str, analysis_date: date
    ) -> Optional[str]:
        """
        Calculate insider trading signal from last 30 days

        Uses us_insider_transactions table.
        Stored in us_stock_grade (requires 30-day aggregation + judgment).

        Logic:
            - Aggregate buy/sell shares over 30 days
            - CEO/CFO trades weighted 2x
            - Net shares > 10,000: STRONG_BUY
            - Net shares > 0: BUY
            - Net shares < -10,000: STRONG_SELL
            - Net shares < 0: SELL
            - Otherwise: NEUTRAL

        Args:
            symbol: Stock symbol
            analysis_date: Analysis date

        Returns:
            insider_signal: 'STRONG_BUY', 'BUY', 'NEUTRAL', 'SELL', 'STRONG_SELL'
        """
        query = """
        SELECT
            acquisition_or_disposal,
            shares,
            executive_title
        FROM us_insider_transactions
        WHERE symbol = $1
          AND date >= $2::date - INTERVAL '30 days'
          AND date <= $2
        """

        try:
            result = await self.db.execute_query(query, symbol, analysis_date)

            if not result:
                return '중립'

            buy_shares = 0
            sell_shares = 0

            for r in result:
                shares = float(r['shares']) if r['shares'] else 0
                titles = r['executive_title'] or []

                # CEO/CFO trades weighted 2x
                weight = 2 if any('CEO' in str(t) or 'CFO' in str(t) for t in titles) else 1

                if r['acquisition_or_disposal'] == 'A':
                    buy_shares += shares * weight
                else:
                    sell_shares += shares * weight

            net_shares = buy_shares - sell_shares

            if net_shares > 10000:
                return '강력 매수'
            elif net_shares > 0:
                return '매수'
            elif net_shares < -10000:
                return '강력 매도'
            elif net_shares < 0:
                return '매도'
            else:
                return '중립'

        except Exception as e:
            logger.error(f"{symbol}: Insider signal calculation failed - {e}")
            return '중립'

    # ========================================================================
    # 7. Action Triggers Generation
    # ========================================================================

    def generate_triggers(
        self,
        final_score: float,
        sector_percentile: Optional[float],
        stop_loss_pct: Optional[float],
        take_profit_pct: Optional[float],
        insider_signal: Optional[str]
    ) -> Dict[str, List[str]]:
        """
        Generate action triggers for buy/sell/hold decisions

        Args:
            final_score: Current final score
            sector_percentile: Sector ranking percentile
            stop_loss_pct: Stop loss percentage (negative)
            take_profit_pct: Take profit percentage (positive)
            insider_signal: Current insider signal

        Returns:
            {
                'buy_triggers': List[str],
                'sell_triggers': List[str],
                'hold_triggers': List[str]
            }
        """
        buy_triggers = [
            f"최종점수 {final_score + 5:.0f}점 이상 (현재: {final_score:.0f}점)"
        ]

        if sector_percentile:
            buy_triggers.append(
                f"섹터 순위 상위 {max(5, sector_percentile - 10):.0f}%"
            )

        # US-specific buy triggers
        buy_triggers.append("풋/콜 비율 < 0.7")

        if insider_signal == '중립':
            buy_triggers.append("내부자 순매수 (CEO/CFO 매입)")

        buy_triggers.append("EPS 추정치 +3% 상향")

        # Sell triggers
        sell_triggers = [
            f"최종점수 {final_score - 15:.0f}점 미만 (현재: {final_score:.0f}점)"
        ]

        if stop_loss_pct:
            sell_triggers.append(f"손절가 도달 {stop_loss_pct:.1f}%")

        if take_profit_pct:
            sell_triggers.append(f"익절가 도달 +{take_profit_pct:.1f}%")

        # US-specific sell triggers
        sell_triggers.append("풋/콜 비율 > 1.2")

        if insider_signal in ['중립', '매수', '강력 매수']:
            sell_triggers.append("내부자 순매도 (CEO/CFO 매각)")

        # Hold triggers
        hold_triggers = [
            f"최종점수 {final_score - 5:.0f}~{final_score + 5:.0f}점 유지",
            "1주 내 실적 발표 (변동성 주의)"
        ]

        return {
            'buy_triggers': buy_triggers,
            'sell_triggers': sell_triggers,
            'hold_triggers': hold_triggers
        }

    # ========================================================================
    # 8. Scenario Probability (Placeholder - High Complexity)
    # ========================================================================

    async def calculate_scenario_probability(
        self,
        symbol: str,
        analysis_date: date,
        final_score: float,
        sector: str,
        regime_data: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Calculate scenario probabilities using HYBRID approach.
        Combines Macro (top-down) and Backtest (bottom-up) with confidence weighting.

        Phase 3.7 Update (Hybrid Probability):
            - Step 1: Get Macro-based probability (top-down)
            - Step 2: Get Backtest-based probability (bottom-up)
            - Step 3: Combine with confidence-based weighting
              - w_backtest = min(0.7, sample_count / 50)
              - w_macro = 1 - w_backtest
            - Step 4: Apply score adjustment

        Benefits:
            - Graceful degradation: Falls back to macro if no backtest samples
            - Data-driven: More samples = more trust in backtest
            - Market context: Always considers macro environment

        Args:
            symbol: Stock symbol
            analysis_date: Analysis date
            final_score: Current final score
            sector: Stock sector
            regime_data: Market regime data with scenario_probability (Optional)

        Returns:
            {
                'scenario_bullish_prob': int (0-100),
                'scenario_sideways_prob': int (0-100),
                'scenario_bearish_prob': int (0-100),
                'scenario_bullish_return': str (measured percentile),
                'scenario_sideways_return': str (measured percentile),
                'scenario_bearish_return': str (measured percentile),
                'scenario_sample_count': int,
                'macro_environment': str,
                'macro_weight': float,
                'backtest_weight': float
            }
        """
        try:
            macro_env = None

            # ================================================================
            # Step 1: Get Macro-based probability (Top-Down)
            # ================================================================
            if regime_data and 'scenario_probability' in regime_data:
                macro_prob = regime_data['scenario_probability']
                macro_bull = macro_prob.get('강세', 0.33)
                macro_bear = macro_prob.get('약세', 0.33)
                macro_sideways = 1.0 - macro_bull - macro_bear
                macro_env = regime_data.get('macro_environment', 'UNKNOWN')
                logger.debug(f"{symbol}: Macro Environment {macro_env} "
                            f"(bull={macro_bull:.0%}, bear={macro_bear:.0%})")
            else:
                # Fallback: Legacy score-based logic as macro proxy
                if final_score >= 75:
                    macro_bull, macro_bear = 0.55, 0.15
                elif final_score >= 60:
                    macro_bull, macro_bear = 0.40, 0.20
                elif final_score >= 45:
                    macro_bull, macro_bear = 0.25, 0.30
                else:
                    macro_bull, macro_bear = 0.15, 0.50
                macro_sideways = 1.0 - macro_bull - macro_bear
                macro_env = 'LEGACY'

            # ================================================================
            # Step 2: Get Backtest-based probability (Bottom-Up)
            # ================================================================
            backtest_data = await self._calculate_backtest_probability(final_score, sector)
            backtest_bull = backtest_data['bullish_prob']
            backtest_bear = backtest_data['bearish_prob']
            backtest_sideways = backtest_data['sideways_prob']
            sample_count = backtest_data['sample_count']

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

            # Step 5: Apply final_score adjustment (±10% range)
            score_adj = (final_score - 50) / 500
            adjusted_bull = hybrid_bull + score_adj
            adjusted_bear = hybrid_bear - score_adj

            # Step 6: Clamp to valid range [0.05, 0.85]
            adjusted_bull = max(0.05, min(0.85, adjusted_bull))
            adjusted_bear = max(0.05, min(0.85, adjusted_bear))

            # Step 7: Derive sideways to ensure sum = 1.0
            adjusted_sideways = 1.0 - adjusted_bull - adjusted_bear

            if adjusted_sideways < 0.10:
                excess = 0.10 - adjusted_sideways
                bull_ratio = adjusted_bull / (adjusted_bull + adjusted_bear)
                adjusted_bull -= excess * bull_ratio
                adjusted_bear -= excess * (1 - bull_ratio)
                adjusted_sideways = 0.10

            # Convert to integer percentages
            bullish_prob = round(adjusted_bull * 100)
            bearish_prob = round(adjusted_bear * 100)
            sideways_prob = 100 - bullish_prob - bearish_prob

            if sideways_prob < 0:
                sideways_prob = 0
                total = bullish_prob + bearish_prob
                if total > 0:
                    bullish_prob = round(bullish_prob / total * 100)
                    bearish_prob = 100 - bullish_prob

            # Step 8: Calculate expected returns from historical backtest
            return_data = await self._calculate_scenario_returns(final_score, sector)

            logger.debug(f"{symbol}: Hybrid prob - macro_w={w_macro:.2f}, backtest_w={w_backtest:.2f}, "
                        f"samples={sample_count}, final=bull:{bullish_prob}%/bear:{bearish_prob}%")

            return {
                'scenario_bullish_prob': bullish_prob,
                'scenario_sideways_prob': sideways_prob,
                'scenario_bearish_prob': bearish_prob,
                'scenario_bullish_return': return_data['bullish_return'],
                'scenario_sideways_return': return_data['sideways_return'],
                'scenario_bearish_return': return_data['bearish_return'],
                'scenario_sample_count': sample_count,
                'macro_environment': macro_env,
                'macro_weight': round(w_macro, 2),
                'backtest_weight': round(w_backtest, 2)
            }

        except Exception as e:
            logger.error(f"{symbol}: Scenario calculation failed - {e}")
            return {
                'scenario_bullish_prob': None,
                'scenario_sideways_prob': None,
                'scenario_bearish_prob': None,
                'scenario_bullish_return': None,
                'scenario_sideways_return': None,
                'scenario_bearish_return': None,
                'scenario_sample_count': None,
                'macro_environment': None,
                'macro_weight': None,
                'backtest_weight': None
            }

    async def _calculate_backtest_probability(
        self,
        final_score: float,
        sector: str
    ) -> Dict[str, Any]:
        """
        Calculate scenario probabilities from historical backtest data.

        Query similar cases (same sector, score ±5) and calculate:
        - Scenario occurrence rates (bullish/sideways/bearish)

        Scenario Definition:
        - Bullish: return_60d > +5%
        - Sideways: -5% <= return_60d <= +5%
        - Bearish: return_60d < -5%

        Returns:
            {
                'bullish_prob': float (0-1),
                'sideways_prob': float (0-1),
                'bearish_prob': float (0-1),
                'sample_count': int
            }
        """
        default_result = {
            'bullish_prob': 0.33,
            'sideways_prob': 0.34,
            'bearish_prob': 0.33,
            'sample_count': 0
        }

        try:
            query = """
            WITH similar_cases AS (
                SELECT
                    g.symbol,
                    g.date as analysis_date,
                    g.final_score,
                    d_start.close as start_close,
                    d_end.close as end_close,
                    CASE
                        WHEN d_end.close IS NOT NULL AND d_start.close > 0
                        THEN (d_end.close - d_start.close) / d_start.close * 100
                        ELSE NULL
                    END as return_60d
                FROM us_stock_grade g
                JOIN us_stock_basic b ON g.symbol = b.symbol
                JOIN us_daily d_start ON g.symbol = d_start.symbol AND g.date = d_start.date
                LEFT JOIN LATERAL (
                    SELECT close
                    FROM us_daily
                    WHERE symbol = g.symbol
                        AND date >= g.date + INTERVAL '55 days'
                        AND date <= g.date + INTERVAL '65 days'
                    ORDER BY date
                    LIMIT 1
                ) d_end ON true
                WHERE g.final_score BETWEEN $1 - 5 AND $1 + 5
                    AND b.sector = $2
                    AND g.date < CURRENT_DATE - INTERVAL '65 days'
                    AND d_end.close IS NOT NULL
            )
            SELECT
                COUNT(*) as total_count,
                COUNT(*) FILTER (WHERE return_60d > 5) as bullish_count,
                COUNT(*) FILTER (WHERE return_60d BETWEEN -5 AND 5) as sideways_count,
                COUNT(*) FILTER (WHERE return_60d < -5) as bearish_count
            FROM similar_cases
            """

            result = await self.db.execute_query(query, final_score, sector)

            if not result or not result[0]['total_count'] or result[0]['total_count'] < 10:
                logger.debug(f"Insufficient backtest samples (sector={sector}, score={final_score})")
                return default_result

            r = result[0]
            total = r['total_count']
            bullish = r['bullish_count'] or 0
            sideways = r['sideways_count'] or 0
            bearish = r['bearish_count'] or 0

            return {
                'bullish_prob': bullish / total,
                'sideways_prob': sideways / total,
                'bearish_prob': bearish / total,
                'sample_count': total
            }

        except Exception as e:
            logger.warning(f"Failed to calculate backtest probability: {e}")
            return default_result

    async def _calculate_scenario_returns(
        self,
        final_score: float,
        sector: str
    ) -> Dict[str, Any]:
        """
        Calculate expected returns from historical backtest data.

        Query similar cases (same sector, score ±5) and calculate:
        - 60-day forward return
        - Percentile 25-75 for each scenario

        Scenario Definition:
        - Bullish: return_60d > +5%
        - Sideways: -5% <= return_60d <= +5%
        - Bearish: return_60d < -5%

        Returns:
            {
                'bullish_return': str (e.g., '+12~+25%'),
                'sideways_return': str (e.g., '-3~+4%'),
                'bearish_return': str (e.g., '-18~-8%'),
                'sample_count': int
            }
        """
        default_returns = {
            'bullish_return': '+10~20%',
            'sideways_return': '-5~+5%',
            'bearish_return': '-15~-8%',
            'sample_count': 0
        }

        try:
            query = """
            WITH similar_cases AS (
                SELECT
                    g.symbol,
                    g.date as analysis_date,
                    g.final_score,
                    d_start.close as start_close,
                    d_end.close as end_close,
                    CASE
                        WHEN d_end.close IS NOT NULL AND d_start.close > 0
                        THEN (d_end.close - d_start.close) / d_start.close * 100
                        ELSE NULL
                    END as return_60d
                FROM us_stock_grade g
                JOIN us_stock_basic b ON g.symbol = b.symbol
                JOIN us_daily d_start ON g.symbol = d_start.symbol AND g.date = d_start.date
                LEFT JOIN LATERAL (
                    SELECT close
                    FROM us_daily
                    WHERE symbol = g.symbol
                        AND date >= g.date + INTERVAL '55 days'
                        AND date <= g.date + INTERVAL '65 days'
                    ORDER BY date
                    LIMIT 1
                ) d_end ON true
                WHERE g.final_score BETWEEN $1 - 5 AND $1 + 5
                    AND b.sector = $2
                    AND g.date < CURRENT_DATE - INTERVAL '65 days'
                    AND d_end.close IS NOT NULL
            )
            SELECT
                COUNT(*) as total_count,
                -- Bullish (>5%)
                ROUND(PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY return_60d)
                    FILTER (WHERE return_60d > 5)::numeric, 1) as bullish_p25,
                ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY return_60d)
                    FILTER (WHERE return_60d > 5)::numeric, 1) as bullish_p75,
                -- Sideways (-5% ~ +5%)
                ROUND(PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY return_60d)
                    FILTER (WHERE return_60d BETWEEN -5 AND 5)::numeric, 1) as sideways_p25,
                ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY return_60d)
                    FILTER (WHERE return_60d BETWEEN -5 AND 5)::numeric, 1) as sideways_p75,
                -- Bearish (<-5%)
                ROUND(PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY return_60d)
                    FILTER (WHERE return_60d < -5)::numeric, 1) as bearish_p25,
                ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY return_60d)
                    FILTER (WHERE return_60d < -5)::numeric, 1) as bearish_p75
            FROM similar_cases
            """

            result = await self.db.execute_query(query, final_score, sector)

            if not result or not result[0]['total_count'] or result[0]['total_count'] < 10:
                logger.debug(f"Insufficient samples for scenario returns (sector={sector}, score={final_score})")
                return default_returns

            r = result[0]
            sample_count = r['total_count']

            # Format return strings
            bullish_p25 = r['bullish_p25'] if r['bullish_p25'] is not None else 10
            bullish_p75 = r['bullish_p75'] if r['bullish_p75'] is not None else 20
            sideways_p25 = r['sideways_p25'] if r['sideways_p25'] is not None else -3
            sideways_p75 = r['sideways_p75'] if r['sideways_p75'] is not None else 3
            bearish_p25 = r['bearish_p25'] if r['bearish_p25'] is not None else -18
            bearish_p75 = r['bearish_p75'] if r['bearish_p75'] is not None else -8

            def format_return(p25, p75):
                p25_str = f"+{p25:.0f}" if p25 > 0 else f"{p25:.0f}"
                p75_str = f"+{p75:.0f}" if p75 > 0 else f"{p75:.0f}"
                return f"{p25_str}~{p75_str}%"

            return {
                'bullish_return': format_return(bullish_p25, bullish_p75),
                'sideways_return': format_return(sideways_p25, sideways_p75),
                'bearish_return': format_return(bearish_p25, bearish_p75),
                'sample_count': sample_count
            }

        except Exception as e:
            logger.warning(f"Failed to calculate scenario returns: {e}")
            return default_returns

    # ========================================================================
    # Main Entry Point: Calculate All Metrics
    # ========================================================================

    async def calculate_all(
        self,
        symbol: str,
        analysis_date: date,
        current_score: float,
        sector: str,
        var_95: Optional[float] = None,
        sector_percentile: Optional[float] = None,
        regime_data: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Calculate all agent metrics for a stock

        Args:
            symbol: Stock symbol
            analysis_date: Analysis date
            current_score: Current final_score
            sector: Stock sector
            var_95: VaR 95% value (optional, for position sizing)
            sector_percentile: Sector percentile rank (optional, for triggers)
            regime_data: Market regime data with scenario_probability (optional, Phase 3.5)

        Returns:
            Dictionary containing all calculated metrics
        """
        # Parallel execution of independent calculations
        # Phase 3.1: Pass sector for sector-specific ATR multiplier
        atr_task = self.calculate_atr_stop_take_profit(symbol, analysis_date, sector)
        timing_task = self.calculate_entry_timing_score(symbol, analysis_date, current_score)
        iv_task = self.calculate_iv_percentile(symbol, analysis_date)
        insider_task = self.calculate_insider_signal(symbol, analysis_date)
        # Phase 3.5: Pass regime_data for Macro-based scenario probability
        scenario_task = self.calculate_scenario_probability(symbol, analysis_date, current_score, sector, regime_data)

        atr_result, timing_result, iv_percentile, insider_signal, scenario_result = await asyncio.gather(
            atr_task, timing_task, iv_task, insider_task, scenario_task
        )

        # Position size (depends on ATR result)
        position_size_pct = await self.calculate_position_size(
            symbol,
            analysis_date,
            atr_result.get('stop_loss_pct'),
            var_95
        )

        # Generate triggers (synchronous)
        triggers = self.generate_triggers(
            final_score=current_score,
            sector_percentile=sector_percentile,
            stop_loss_pct=atr_result.get('stop_loss_pct'),
            take_profit_pct=atr_result.get('take_profit_pct'),
            insider_signal=insider_signal
        )

        return {
            # ATR-based risk management
            'atr_pct': atr_result.get('atr_pct'),
            'stop_loss_pct': atr_result.get('stop_loss_pct'),
            'take_profit_pct': atr_result.get('take_profit_pct'),
            'risk_reward_ratio': atr_result.get('risk_reward_ratio'),

            # Entry timing
            'entry_timing_score': timing_result.get('entry_timing_score'),
            'score_trend_2w': timing_result.get('score_trend_2w'),
            'price_position_52w': timing_result.get('price_position_52w'),

            # Position sizing
            'position_size_pct': position_size_pct,

            # US-specific (stored in us_stock_grade)
            'iv_percentile': iv_percentile,
            'insider_signal': insider_signal,

            # Triggers
            'buy_triggers': triggers['buy_triggers'],
            'sell_triggers': triggers['sell_triggers'],
            'hold_triggers': triggers['hold_triggers'],

            # Scenario analysis (Phase 3.5: Macro-based)
            'scenario_bullish_prob': scenario_result.get('scenario_bullish_prob'),
            'scenario_sideways_prob': scenario_result.get('scenario_sideways_prob'),
            'scenario_bearish_prob': scenario_result.get('scenario_bearish_prob'),
            'scenario_bullish_return': scenario_result.get('scenario_bullish_return'),
            'scenario_sideways_return': scenario_result.get('scenario_sideways_return'),
            'scenario_bearish_return': scenario_result.get('scenario_bearish_return'),
            'scenario_sample_count': scenario_result.get('scenario_sample_count'),
            'macro_environment': scenario_result.get('macro_environment')
        }
