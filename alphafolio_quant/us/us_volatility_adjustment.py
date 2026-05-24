"""
US Volatility Adjustment Module v1.0
====================================
Phase 3.4.2: Shared Volatility Engine for All Factors

Provides IV Percentile and HV-based score adjustments with
factor-specific sensitivity levels.

Factor Sensitivity:
- Momentum: 1.0 (100% effect - most sensitive to volatility)
- Growth: 0.7 (70% effect - high sensitivity)
- Value: 0.3 (30% effect - low sensitivity)
- Quality: 0.1 (10% effect - minimal sensitivity)

Logic:
- IV Percentile >= 90: Base modifier = -10
- IV Percentile >= 80: Base modifier = -7
- IV Percentile >= 70: Base modifier = -4
- IV Percentile <= 10: Base modifier = +5
- IV Percentile <= 20: Base modifier = +3
- IV Percentile <= 30: Base modifier = +1
- HV 20d >= 60%: Additional -3
- HV 20d >= 45%: Additional -1

Final modifier = base_modifier * sensitivity

Usage:
    from us_volatility_adjustment import USVolatilityAdjustment

    vol_adj = USVolatilityAdjustment(db_manager)
    await vol_adj.load_data(symbol, analysis_date, price_data)
    adjustment = vol_adj.get_adjustment('momentum')  # or 'growth', 'value', 'quality'

Author: Claude Code
Date: 2025-12-07
"""

import math
import logging
from datetime import date
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

FACTOR_SENSITIVITY = {
    'momentum': 1.0,   # Full effect - momentum is most sensitive to volatility
    'growth': 0.7,     # High sensitivity - growth stocks volatile
    'value': 0.3,      # Low sensitivity - value investing less timing-dependent
    'quality': 0.1     # Minimal sensitivity - quality is fundamental-driven
}

# IV Percentile thresholds and base modifiers
IV_THRESHOLDS = {
    'extreme_high': {'min': 90, 'modifier': -10, 'signal': 'EXTREME_HIGH'},
    'high': {'min': 80, 'modifier': -7, 'signal': 'HIGH'},
    'elevated': {'min': 70, 'modifier': -4, 'signal': 'ELEVATED'},
    'very_low': {'max': 10, 'modifier': +5, 'signal': 'VERY_LOW'},
    'low': {'max': 20, 'modifier': +3, 'signal': 'LOW'},
    'below_avg': {'max': 30, 'modifier': +1, 'signal': 'BELOW_AVG'}
}

# HV 20d thresholds
HV_THRESHOLDS = {
    'spike': {'min': 60, 'modifier': -3, 'signal': 'SPIKE'},
    'elevated': {'min': 45, 'modifier': -1, 'signal': 'ELEVATED'},
    'low': {'max': 25, 'modifier': 0, 'signal': 'LOW'},
    'very_low': {'max': 15, 'modifier': 0, 'signal': 'VERY_LOW'}
}


# ============================================================================
# Main Class
# ============================================================================

class USVolatilityAdjustment:
    """
    Shared Volatility Adjustment Engine

    Loads IV Percentile and HV data once per symbol,
    provides factor-specific adjustments based on sensitivity.
    """

    def __init__(self, db_manager):
        """
        Initialize with database manager

        Args:
            db_manager: Database manager with execute_query method
        """
        self.db = db_manager
        self.symbol: str = None
        self.analysis_date: date = None

        # Volatility metrics
        self.iv_percentile: Optional[float] = None
        self.hv_20d: Optional[float] = None
        self.current_iv: Optional[float] = None

        # Data loaded flag
        self._data_loaded: bool = False

    def _to_float(self, value: Any) -> float:
        """Safely convert value to float"""
        if value is None:
            return 0.0
        try:
            return float(value)
        except (ValueError, TypeError):
            return 0.0

    async def load_data(self, symbol: str, analysis_date: date,
                        price_data: Optional[List[Dict]] = None) -> bool:
        """
        Load IV Percentile and HV 20d data for a symbol

        Args:
            symbol: Stock symbol
            analysis_date: Analysis date
            price_data: Optional pre-loaded price data for HV calculation
                        (must have 'close' field, sorted desc by date)

        Returns:
            True if data loaded successfully
        """
        self.symbol = symbol
        self.analysis_date = analysis_date
        self.iv_percentile = None
        self.hv_20d = None
        self.current_iv = None
        self._data_loaded = False

        # 1. Load IV Percentile from us_option_daily_summary
        await self._load_iv_percentile()

        # 2. Calculate HV 20d from price data
        if price_data:
            self._calculate_hv_20d(price_data)

        self._data_loaded = True
        return True

    async def _load_iv_percentile(self):
        """
        Load IV Percentile from us_option_daily_summary

        Calculates percentile of current IV within 252-day history
        """
        query = """
        SELECT avg_implied_volatility * 100 as iv
        FROM us_option_daily_summary
        WHERE symbol = $1
          AND date <= $2
          AND avg_implied_volatility IS NOT NULL
        ORDER BY date DESC
        LIMIT 252
        """

        try:
            result = await self.db.execute_query(query, self.symbol, self.analysis_date)

            if result and len(result) >= 20:
                iv_values = [self._to_float(r['iv']) for r in result if r['iv']]

                if iv_values:
                    self.current_iv = iv_values[0]
                    # Percentile: ratio of values below current IV
                    below_count = sum(1 for iv in iv_values if iv < self.current_iv)
                    self.iv_percentile = round((below_count / len(iv_values)) * 100, 1)

                    logger.debug(f"{self.symbol}: IV Percentile = {self.iv_percentile}% "
                                f"(current IV = {self.current_iv:.1f}%)")

        except Exception as e:
            logger.warning(f"{self.symbol}: IV percentile query failed - {e}")

    def _calculate_hv_20d(self, price_data: List[Dict]):
        """
        Calculate 20-day Historical Volatility from price data

        Args:
            price_data: List of price records with 'close' field,
                       sorted descending by date (newest first)
        """
        if len(price_data) < 21:
            return

        try:
            prices = [p['close'] for p in price_data[:21] if p.get('close')]

            if len(prices) >= 21:
                # Calculate log returns
                returns = []
                for i in range(len(prices) - 1):
                    if prices[i] and prices[i+1] and prices[i+1] > 0:
                        ret = math.log(prices[i] / prices[i+1])
                        returns.append(ret)

                if len(returns) >= 15:
                    # Annualized volatility
                    mean_ret = sum(returns) / len(returns)
                    variance = sum((r - mean_ret) ** 2 for r in returns) / len(returns)
                    self.hv_20d = round(math.sqrt(variance) * math.sqrt(252) * 100, 1)

                    logger.debug(f"{self.symbol}: HV 20d = {self.hv_20d}%")

        except Exception as e:
            logger.warning(f"{self.symbol}: HV calculation failed - {e}")

    def get_adjustment(self, factor_type: str) -> Dict:
        """
        Get volatility adjustment for a specific factor type

        Args:
            factor_type: One of 'momentum', 'growth', 'value', 'quality'

        Returns:
            {
                'modifier': float (adjusted by sensitivity),
                'base_modifier': float (before sensitivity),
                'sensitivity': float,
                'reason': str,
                'iv_signal': str,
                'hv_signal': str,
                'iv_percentile': float or None,
                'hv_20d': float or None
            }
        """
        if not self._data_loaded:
            logger.warning(f"Volatility data not loaded. Call load_data() first.")
            return self._empty_result(factor_type)

        sensitivity = FACTOR_SENSITIVITY.get(factor_type.lower(), 0.5)

        base_modifier = 0.0
        reasons = []
        iv_signal = 'NEUTRAL'
        hv_signal = 'NEUTRAL'

        # 1. IV Percentile-based adjustment
        if self.iv_percentile is not None:
            iv_mod, iv_sig, iv_reason = self._calc_iv_modifier()
            base_modifier += iv_mod
            iv_signal = iv_sig
            if iv_reason:
                reasons.append(iv_reason)

        # 2. HV 20d-based adjustment
        if self.hv_20d is not None:
            hv_mod, hv_sig, hv_reason = self._calc_hv_modifier()
            base_modifier += hv_mod
            hv_signal = hv_sig
            if hv_reason:
                reasons.append(hv_reason)

        # Apply sensitivity
        adjusted_modifier = round(base_modifier * sensitivity, 1)

        # Clamp to reasonable range
        max_negative = -13 * sensitivity
        max_positive = 5 * sensitivity
        adjusted_modifier = max(max_negative, min(max_positive, adjusted_modifier))

        reason_str = ', '.join(reasons) if reasons else 'NEUTRAL'

        return {
            'modifier': adjusted_modifier,
            'base_modifier': base_modifier,
            'sensitivity': sensitivity,
            'reason': reason_str,
            'iv_signal': iv_signal,
            'hv_signal': hv_signal,
            'iv_percentile': self.iv_percentile,
            'hv_20d': self.hv_20d,
            'current_iv': self.current_iv
        }

    def _calc_iv_modifier(self) -> tuple:
        """Calculate IV-based modifier"""
        if self.iv_percentile is None:
            return 0.0, 'NEUTRAL', None

        pct = self.iv_percentile

        # High IV = negative modifier (reduce score)
        if pct >= 90:
            return -10, 'EXTREME_HIGH', f'IV_P90+({pct:.0f}%)'
        elif pct >= 80:
            return -7, 'HIGH', f'IV_P80+({pct:.0f}%)'
        elif pct >= 70:
            return -4, 'ELEVATED', f'IV_P70+({pct:.0f}%)'
        # Low IV = positive modifier (increase score)
        elif pct <= 10:
            return +5, 'VERY_LOW', f'IV_P10-({pct:.0f}%)'
        elif pct <= 20:
            return +3, 'LOW', f'IV_P20-({pct:.0f}%)'
        elif pct <= 30:
            return +1, 'BELOW_AVG', f'IV_P30-({pct:.0f}%)'

        return 0.0, 'NEUTRAL', None

    def _calc_hv_modifier(self) -> tuple:
        """Calculate HV-based modifier"""
        if self.hv_20d is None:
            return 0.0, 'NEUTRAL', None

        hv = self.hv_20d

        # High HV = additional negative modifier
        if hv >= 60:
            return -3, 'SPIKE', f'HV_SPIKE({hv:.0f}%)'
        elif hv >= 45:
            return -1, 'ELEVATED', f'HV_HIGH({hv:.0f}%)'
        # Low HV = no additional bonus (IV already covers this)
        elif hv <= 15:
            return 0, 'VERY_LOW', None
        elif hv <= 25:
            return 0, 'LOW', None

        return 0.0, 'NEUTRAL', None

    def _empty_result(self, factor_type: str) -> Dict:
        """Return empty result when data not loaded"""
        sensitivity = FACTOR_SENSITIVITY.get(factor_type.lower(), 0.5)
        return {
            'modifier': 0.0,
            'base_modifier': 0.0,
            'sensitivity': sensitivity,
            'reason': 'NO_DATA',
            'iv_signal': 'UNKNOWN',
            'hv_signal': 'UNKNOWN',
            'iv_percentile': None,
            'hv_20d': None,
            'current_iv': None
        }

    def get_all_adjustments(self) -> Dict[str, Dict]:
        """
        Get adjustments for all factor types at once

        Returns:
            {
                'momentum': {...},
                'growth': {...},
                'value': {...},
                'quality': {...}
            }
        """
        return {
            factor: self.get_adjustment(factor)
            for factor in FACTOR_SENSITIVITY.keys()
        }

    @property
    def is_high_volatility(self) -> bool:
        """Check if current volatility is elevated"""
        if self.iv_percentile is not None and self.iv_percentile >= 70:
            return True
        if self.hv_20d is not None and self.hv_20d >= 45:
            return True
        return False

    @property
    def is_low_volatility(self) -> bool:
        """Check if current volatility is low"""
        if self.iv_percentile is not None and self.iv_percentile <= 30:
            return True
        if self.hv_20d is not None and self.hv_20d <= 25:
            return True
        return False

    def get_volatility_summary(self) -> Dict:
        """
        Get summary of volatility state

        Returns:
            {
                'iv_percentile': float or None,
                'hv_20d': float or None,
                'current_iv': float or None,
                'volatility_state': 'HIGH' | 'LOW' | 'NEUTRAL',
                'description': str
            }
        """
        if self.is_high_volatility:
            state = 'HIGH'
            desc = 'Elevated volatility - reduce position sizing'
        elif self.is_low_volatility:
            state = 'LOW'
            desc = 'Low volatility - favorable for momentum strategies'
        else:
            state = 'NEUTRAL'
            desc = 'Normal volatility levels'

        return {
            'iv_percentile': self.iv_percentile,
            'hv_20d': self.hv_20d,
            'current_iv': self.current_iv,
            'volatility_state': state,
            'description': desc
        }


# ============================================================================
# Convenience Function
# ============================================================================

async def get_volatility_adjustment(
    symbol: str,
    analysis_date: date,
    factor_type: str,
    db_manager,
    price_data: Optional[List[Dict]] = None
) -> Dict:
    """
    Convenience function for one-off volatility adjustment

    Args:
        symbol: Stock symbol
        analysis_date: Analysis date
        factor_type: 'momentum', 'growth', 'value', or 'quality'
        db_manager: Database manager
        price_data: Optional price data for HV calculation

    Returns:
        Adjustment dict with modifier and signals
    """
    vol_adj = USVolatilityAdjustment(db_manager)
    await vol_adj.load_data(symbol, analysis_date, price_data)
    return vol_adj.get_adjustment(factor_type)
