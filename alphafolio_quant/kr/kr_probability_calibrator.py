"""
Scenario Probability Calibrator for Korean Stock Analysis (Phase 3.9)

Purpose: Fix systematic underprediction of Bullish/Bearish scenarios
- Bullish underprediction: +5.5~13.8%p error correction
- Bearish severe underprediction: +20.1%p error correction

Based on phase3_9 analysis results (2025-08-04 ~ 2025-09-25)

File: kr/kr_probability_calibrator.py
"""

import logging
from typing import Dict, Optional
from datetime import date

logger = logging.getLogger(__name__)


class ScenarioCalibrator:
    """
    Calibrate scenario probabilities based on empirical analysis

    Methodology:
    1. Apply empirical calibration map (from phase3_9 analysis)
    2. Adjust by market regime (GREED/NEUTRAL/FEAR/PANIC)
    3. Add theme risk premium (high-risk themes)
    4. Apply bearish probability floor (score-based)
    5. Normalize to 100%
    """

    # Empirical calibration mapping table (from phase3_9 CSV analysis)
    CALIBRATION_MAP = {
        'bullish': {
            (0, 20): 1.55,    # 10% predicted -> 15.5% actual
            (20, 40): 1.32,   # 30% predicted -> 39.6% actual
            (40, 60): 1.28,   # 50% predicted -> 63.8% actual
            (60, 80): 1.23,   # 70% predicted -> 85.9% actual
            (80, 100): 1.15
        },
        'bearish': {
            (0, 20): 1.50,    # Reduced from 3.01 (Phase 3.9.1 fix for over-prediction)
            (20, 40): 1.20,   # Reduced from 1.48 (Phase 3.9.1 fix)
            (40, 60): 1.39,   # 50% predicted -> 69.7% actual
            (60, 80): 1.26,   # 70% predicted -> 88.5% actual
            (80, 100): 1.20
        }
    }

    # Theme-specific calibration map (Phase 3.9.1 - based on empirical analysis)
    # Each theme has: bullish multiplier, bearish multiplier, bearish floor
    THEME_CALIBRATION_MAP = {
        'Semiconductor': {'bullish': 1.20, 'bearish': 0.63, 'floor': 8},
        'AI_BigData': {'bullish': 0.72, 'bearish': 1.24, 'floor': 15},
        'Finance': {'bullish': 0.55, 'bearish': 0.45, 'floor': 5},
        'Consumer_Goods': {'bullish': 0.42, 'bearish': 0.69, 'floor': 10},
        'Electronics': {'bullish': 0.75, 'bearish': 0.78, 'floor': 10},
        'Traditional_Manufacturing': {'bullish': 0.53, 'bearish': 0.68, 'floor': 10},
        'Advanced_Manufacturing': {'bullish': 0.93, 'bearish': 0.62, 'floor': 8},
        'Shipbuilding': {'bullish': 0.97, 'bearish': 0.77, 'floor': 8},
        'Battery': {'bullish': 0.97, 'bearish': 0.57, 'floor': 8},
        'Bio_DrugRD': {'bullish': 0.93, 'bearish': 0.73, 'floor': 10},
        'Pharma_CDMO': {'bullish': 0.70, 'bearish': 0.68, 'floor': 10},
        'Energy': {'bullish': 0.80, 'bearish': 0.46, 'floor': 5},
        'Materials_Chemical': {'bullish': 0.62, 'bearish': 0.67, 'floor': 10},
        'IT_Software': {'bullish': 0.59, 'bearish': 0.68, 'floor': 12},
        'Telecom_Media': {'bullish': 0.39, 'bearish': 0.81, 'floor': 15},
        'Healthcare_Device': {'bullish': 0.62, 'bearish': 0.85, 'floor': 12},
        'Others': {'bullish': 0.38, 'bearish': 0.70, 'floor': 10}
    }

    # Theme-specific return ranges (P25~P75 from empirical analysis)
    # Format: 'theme': {'bullish': (p25, p75), 'sideways': (p25, p75), 'bearish': (p25, p75)}
    THEME_RETURN_RANGES = {
        'Semiconductor': {
            'bullish': (22.4, 59.1),
            'sideways': (-5.3, 3.6),
            'bearish': (-21.4, -13.3)
        },
        'AI_BigData': {
            'bullish': (13.9, 37.2),
            'sideways': (-6.0, 4.0),
            'bearish': (-23.7, -13.9)
        },
        'Finance': {
            'bullish': (13.7, 27.8),
            'sideways': (-2.1, 1.7),
            'bearish': (-17.9, -11.8)
        },
        'Consumer_Goods': {
            'bullish': (13.5, 28.4),
            'sideways': (-6.0, 0.7),
            'bearish': (-21.7, -12.4)
        },
        'Electronics': {
            'bullish': (15.2, 38.6),
            'sideways': (-5.8, 2.2),
            'bearish': (-22.3, -12.7)
        },
        'Traditional_Manufacturing': {
            'bullish': (14.6, 40.0),
            'sideways': (-5.7, 1.8),
            'bearish': (-19.9, -12.1)
        },
        'Advanced_Manufacturing': {
            'bullish': (16.5, 46.9),
            'sideways': (-5.2, 3.3),
            'bearish': (-21.1, -12.5)
        },
        'Shipbuilding': {
            'bullish': (17.4, 46.4),
            'sideways': (0.0, 5.1),
            'bearish': (-24.3, -13.7)
        },
        'Battery': {
            'bullish': (16.7, 42.8),
            'sideways': (-5.0, 3.9),
            'bearish': (-21.7, -13.1)
        },
        'Bio_DrugRD': {
            'bullish': (20.3, 52.0),
            'sideways': (-4.5, 3.8),
            'bearish': (-29.2, -12.6)
        },
        'Pharma_CDMO': {
            'bullish': (16.4, 55.7),
            'sideways': (-5.9, 1.1),
            'bearish': (-22.4, -12.3)
        },
        'Energy': {
            'bullish': (18.8, 42.2),
            'sideways': (-4.3, 2.0),
            'bearish': (-20.4, -12.4)
        },
        'Materials_Chemical': {
            'bullish': (14.4, 37.2),
            'sideways': (-6.1, 1.4),
            'bearish': (-17.7, -11.6)
        },
        'IT_Software': {
            'bullish': (13.9, 25.8),
            'sideways': (-6.0, 1.0),
            'bearish': (-21.7, -12.5)
        },
        'Telecom_Media': {
            'bullish': (12.5, 25.3),
            'sideways': (-6.6, 0.8),
            'bearish': (-20.9, -12.7)
        },
        'Healthcare_Device': {
            'bullish': (16.4, 50.6),
            'sideways': (-5.9, 1.4),
            'bearish': (-22.3, -12.3)
        },
        'Others': {
            'bullish': (13.7, 33.7),
            'sideways': (-5.7, 1.0),
            'bearish': (-18.3, -11.9)
        }
    }

    # Default return ranges (used when theme not found)
    DEFAULT_RETURN_RANGES = {
        'bullish': (15.0, 35.0),
        'sideways': (-5.0, 2.0),
        'bearish': (-20.0, -12.0)
    }

    # Market regime adjustment coefficients
    REGIME_ADJUSTMENT = {
        'OVERHEATED': {'bullish_boost': 1.2, 'bearish_reduce': 0.85},
        'GREED': {'bullish_boost': 1.3, 'bearish_reduce': 0.8},
        'NEUTRAL': {'bullish_boost': 1.0, 'bearish_reduce': 1.0},
        'FEAR': {'bullish_boost': 0.7, 'bearish_reduce': 1.2},
        'PANIC': {'bullish_boost': 0.5, 'bearish_reduce': 1.4}
    }

    # Bearish probability floor (score-based) - used as fallback when theme not in THEME_CALIBRATION_MAP
    BEARISH_FLOOR = {
        (0, 20): 25,     # Low score: minimum 25% bearish probability
        (20, 35): 20,
        (35, 45): 15,
        (45, 60): 12,
        (60, 100): 8     # High score: minimum 8% bearish probability
    }

    def __init__(self, db_manager=None):
        """
        Initialize scenario calibrator

        Args:
            db_manager: Database manager instance (optional, for market regime lookup)
        """
        self.db_manager = db_manager

        # Phase 5: Rolling Calibrator integration
        self.rolling_calibrator = None
        self._init_rolling_calibrator()

    def _init_rolling_calibrator(self):
        """Initialize and load cached rolling calibrator if available"""
        try:
            from kr_rolling_calibrator import RollingCalibrator
            if self.db_manager:
                self.rolling_calibrator = RollingCalibrator(self.db_manager)
                # Try to load cached calibrators
                if self.rolling_calibrator.load_cached_calibrators():
                    logger.info("Rolling calibrator loaded from cache")
                else:
                    logger.info("No cached rolling calibrator available, using static calibration")
        except ImportError:
            logger.debug("Rolling calibrator module not available")
            self.rolling_calibrator = None
        except Exception as e:
            logger.warning(f"Failed to initialize rolling calibrator: {e}")
            self.rolling_calibrator = None

    def calibrate_probabilities(
        self,
        raw_bullish: int,
        raw_bearish: int,
        raw_sideways: int,
        final_score: float,
        theme: str,
        backtest_avg_return: Optional[float] = None
    ) -> Dict[str, int]:
        """
        Calibrate raw probabilities to corrected probabilities

        Args:
            raw_bullish: Raw bullish probability (0-100)
            raw_bearish: Raw bearish probability (0-100)
            raw_sideways: Raw sideways probability (0-100)
            final_score: Stock final score (0-100)
            theme: Stock theme classification
            backtest_avg_return: Average return from backtest data (for bearish cap)

        Returns:
            Dict with calibrated probabilities: {'bullish': int, 'sideways': int, 'bearish': int}
        """
        try:
            # Phase 5: Try rolling calibration first (if available and trained)
            if self.rolling_calibrator and self.rolling_calibrator.calibrators['bullish'].is_fitted:
                rolling_result = self.rolling_calibrator.calibrate_probabilities(
                    raw_bullish, raw_bearish, raw_sideways
                )
                logger.debug(f"Rolling calibration applied: raw=({raw_bullish}, {raw_sideways}, {raw_bearish}) -> "
                           f"cal=({rolling_result['bullish']}, {rolling_result['sideways']}, {rolling_result['bearish']})")
                return rolling_result

            # Fallback: Static calibration (Phase 3.9.1)
            # Step 1: Apply theme-specific calibration
            # Uses THEME_CALIBRATION_MAP if theme exists, otherwise falls back to CALIBRATION_MAP
            cal_bullish = self._apply_calibration_map('bullish', raw_bullish, theme)
            cal_bearish = self._apply_calibration_map('bearish', raw_bearish, theme)

            # Step 2: Get market regime and apply adjustment
            regime = self._get_market_regime()
            cal_bullish, cal_bearish = self._apply_regime_adjustment(
                cal_bullish, cal_bearish, regime
            )

            # Step 3: Apply theme-specific bearish floor (Phase 3.9.1)
            cal_bearish = self._apply_bearish_floor(cal_bearish, final_score, theme)

            # Step 4: Apply bearish cap based on backtest avg return (Phase 3.9.1)
            cal_bearish = self._apply_bearish_cap(cal_bearish, backtest_avg_return)

            # Step 5: Normalize to 100%
            calibrated = self._normalize(cal_bullish, raw_sideways, cal_bearish)

            logger.debug(f"Calibration: raw=({raw_bullish}, {raw_sideways}, {raw_bearish}) -> "
                        f"cal=({calibrated['bullish']}, {calibrated['sideways']}, {calibrated['bearish']}) "
                        f"[regime={regime}, theme={theme}]")

            return calibrated

        except Exception as e:
            logger.error(f"Calibration failed: {e}, returning raw probabilities")
            return {
                'bullish': raw_bullish,
                'sideways': raw_sideways,
                'bearish': raw_bearish
            }

    def _apply_calibration_map(self, scenario: str, raw_prob: int, theme: str = None) -> float:
        """
        Apply theme-specific or empirical calibration multiplier (Phase 3.9.1)

        Args:
            scenario: 'bullish' or 'bearish'
            raw_prob: Raw probability (0-100)
            theme: Stock theme classification (optional)

        Returns:
            Calibrated probability (float)
        """
        # Priority 1: Use theme-specific multiplier if available
        if theme and theme in self.THEME_CALIBRATION_MAP:
            multiplier = self.THEME_CALIBRATION_MAP[theme].get(scenario, 1.0)
            return raw_prob * multiplier

        # Priority 2: Fall back to range-based CALIBRATION_MAP
        if scenario not in self.CALIBRATION_MAP:
            return float(raw_prob)

        mapping = self.CALIBRATION_MAP[scenario]

        for (low, high), multiplier in mapping.items():
            if low <= raw_prob < high:
                return raw_prob * multiplier

        # Default: no adjustment
        return float(raw_prob)

    def _get_market_regime(self) -> str:
        """
        Get current market regime from market sentiment analysis

        Returns:
            Regime: 'OVERHEATED', 'GREED', 'NEUTRAL', 'FEAR', 'PANIC'

        Note:
            Uses KOSPI sentiment as proxy
            Fallback to 'NEUTRAL' if data unavailable
        """
        if not self.db_manager:
            return 'NEUTRAL'

        try:
            # Query market_index for KOSPI recent performance
            # Classify into regime based on change_rate
            query = """
            SELECT change_rate
            FROM market_index
            WHERE exchange = 'KOSPI'
            ORDER BY date DESC
            LIMIT 5
            """

            result = self.db_manager.execute_query_sync(query) if hasattr(self.db_manager, 'execute_query_sync') else None

            if not result or len(result) < 5:
                return 'NEUTRAL'

            # Calculate 5-day average change
            avg_change = sum(float(r['change_rate']) for r in result) / len(result)

            # Classify regime
            if avg_change > 2.0:
                return 'OVERHEATED'
            elif avg_change > 0.5:
                return 'GREED'
            elif avg_change > -0.5:
                return 'NEUTRAL'
            elif avg_change > -2.0:
                return 'FEAR'
            else:
                return 'PANIC'

        except Exception as e:
            logger.warning(f"Failed to get market regime: {e}, using NEUTRAL")
            return 'NEUTRAL'

    def _apply_regime_adjustment(
        self,
        bullish: float,
        bearish: float,
        regime: str
    ) -> tuple:
        """
        Apply market regime adjustment

        Args:
            bullish: Calibrated bullish probability
            bearish: Calibrated bearish probability
            regime: Market regime

        Returns:
            Tuple of (adjusted_bullish, adjusted_bearish)
        """
        adjustment = self.REGIME_ADJUSTMENT.get(regime, {'bullish_boost': 1.0, 'bearish_reduce': 1.0})

        adj_bullish = bullish * adjustment['bullish_boost']
        adj_bearish = bearish * adjustment['bearish_reduce']

        return adj_bullish, adj_bearish

    def _apply_bearish_floor(self, bearish: float, final_score: float, theme: str = None) -> float:
        """
        Apply minimum bearish probability floor (Phase 3.9.1: theme-specific floor)

        Args:
            bearish: Bearish probability
            final_score: Stock final score
            theme: Stock theme classification (optional)

        Returns:
            Bearish probability with floor applied
        """
        # Priority 1: Use theme-specific floor if available
        theme_floor = 0
        if theme and theme in self.THEME_CALIBRATION_MAP:
            theme_floor = self.THEME_CALIBRATION_MAP[theme].get('floor', 0)

        # Priority 2: Get score-based floor
        score_floor = 8  # Default
        for (low, high), floor in self.BEARISH_FLOOR.items():
            if low <= final_score < high:
                score_floor = floor
                break

        # Apply the higher of theme floor and score floor
        final_floor = max(theme_floor, score_floor)
        return max(bearish, final_floor)

    def _apply_bearish_cap(self, bearish: float, avg_return: Optional[float]) -> float:
        """
        Apply bearish probability cap based on backtest average return (Phase 3.9.1)

        If historical average return is positive, cap the bearish probability
        to prevent over-prediction of bearish scenarios for fundamentally good stocks.

        Args:
            bearish: Bearish probability after calibration
            avg_return: Average return from backtest data (%)

        Returns:
            Bearish probability with cap applied
        """
        if avg_return is None:
            return bearish

        if avg_return > 5.0:
            # Strong positive return history -> max 20% bearish
            capped = min(bearish, 20)
            if capped < bearish:
                logger.debug(f"Bearish capped: {bearish:.1f}% -> {capped}% (avg_return={avg_return:.1f}%)")
            return capped
        elif avg_return > 0:
            # Positive return history -> max 25% bearish
            capped = min(bearish, 25)
            if capped < bearish:
                logger.debug(f"Bearish capped: {bearish:.1f}% -> {capped}% (avg_return={avg_return:.1f}%)")
            return capped

        return bearish

    def _normalize(self, bullish: float, sideways: float, bearish: float) -> Dict[str, int]:
        """
        Normalize probabilities to sum to 100%

        Args:
            bullish: Bullish probability
            sideways: Sideways probability (raw)
            bearish: Bearish probability

        Returns:
            Dict with normalized integer probabilities
        """
        total = bullish + sideways + bearish

        if total == 0:
            return {'bullish': 33, 'sideways': 34, 'bearish': 33}

        # Normalize
        norm_bullish = round(bullish / total * 100)
        norm_bearish = round(bearish / total * 100)
        norm_sideways = 100 - norm_bullish - norm_bearish

        # Ensure sideways is non-negative
        if norm_sideways < 0:
            norm_sideways = 0
            norm_bullish = round(bullish / (bullish + bearish) * 100)
            norm_bearish = 100 - norm_bullish

        return {
            'bullish': norm_bullish,
            'sideways': norm_sideways,
            'bearish': norm_bearish
        }

    def get_return_range(self, theme: str, outcome: str) -> tuple:
        """
        Get theme-specific return range (P25~P75) for a given outcome (Phase 3.9.1)

        Args:
            theme: Stock theme classification
            outcome: 'bullish', 'sideways', or 'bearish'

        Returns:
            Tuple of (lower_bound, upper_bound) representing P25~P75 range
        """
        if theme and theme in self.THEME_RETURN_RANGES:
            ranges = self.THEME_RETURN_RANGES[theme]
            if outcome in ranges:
                return ranges[outcome]

        # Fallback to default ranges
        return self.DEFAULT_RETURN_RANGES.get(outcome, (10, 25))

    async def daily_update_rolling_calibration(self) -> Dict:
        """
        Daily update for rolling calibration (Phase 5)

        Should be called once per day in batch job (e.g., kr_main.py option 5)
        Trains new calibrators using rolling window and saves if valid

        Returns:
            dict: {
                'is_valid': bool,
                'improvements': {'bullish': float, 'bearish': float},
                'message': str
            }
        """
        if not self.rolling_calibrator:
            return {
                'is_valid': False,
                'improvements': {},
                'message': 'Rolling calibrator not initialized'
            }

        try:
            result = await self.rolling_calibrator.daily_calibration_update()

            if result['is_valid']:
                return {
                    'is_valid': True,
                    'improvements': result['improvements'],
                    'message': f"Rolling calibration updated: bullish {result['improvements'].get('bullish', 0):+.1f}%, bearish {result['improvements'].get('bearish', 0):+.1f}%"
                }
            else:
                return {
                    'is_valid': False,
                    'improvements': result.get('improvements', {}),
                    'message': 'Rolling calibration not applied (improvement below 5% threshold)'
                }

        except Exception as e:
            logger.error(f"Daily rolling calibration update failed: {e}")
            return {
                'is_valid': False,
                'improvements': {},
                'message': f'Update failed: {e}'
            }


# Validation test
if __name__ == '__main__':
    print("Scenario Calibrator Validation Test (Phase 3.9.1 - Theme-based Calibration)")
    print("=" * 80)

    calibrator = ScenarioCalibrator()

    # Test case 1: Semiconductor (Bullish 1.20x, Bearish 0.63x, floor 8)
    print("\nTest 1: Semiconductor - Bullish should INCREASE (1.20x)")
    result = calibrator.calibrate_probabilities(
        raw_bullish=50,
        raw_bearish=10,
        raw_sideways=40,
        final_score=70,
        theme='Semiconductor'
    )
    print(f"  Raw: Bullish=50%, Sideways=40%, Bearish=10%")
    print(f"  Calibrated: Bullish={result['bullish']}%, Sideways={result['sideways']}%, Bearish={result['bearish']}%")
    print(f"  Expected: Bullish ~60% (50*1.20), Bearish ~6% (10*0.63), floor 8%")

    # Test case 2: Finance (Bullish 0.55x, Bearish 0.45x, floor 5)
    print("\nTest 2: Finance - Both should DECREASE significantly")
    result = calibrator.calibrate_probabilities(
        raw_bullish=30,
        raw_bearish=30,
        raw_sideways=40,
        final_score=50,
        theme='Finance'
    )
    print(f"  Raw: Bullish=30%, Sideways=40%, Bearish=30%")
    print(f"  Calibrated: Bullish={result['bullish']}%, Sideways={result['sideways']}%, Bearish={result['bearish']}%")
    print(f"  Expected: Bullish ~17% (30*0.55), Bearish ~14% (30*0.45)")

    # Test case 3: AI_BigData (Bullish 0.72x, Bearish 1.24x - only theme with bearish INCREASE)
    print("\nTest 3: AI_BigData - Bearish should INCREASE (1.24x)")
    result = calibrator.calibrate_probabilities(
        raw_bullish=30,
        raw_bearish=30,
        raw_sideways=40,
        final_score=50,
        theme='AI_BigData'
    )
    print(f"  Raw: Bullish=30%, Sideways=40%, Bearish=30%")
    print(f"  Calibrated: Bullish={result['bullish']}%, Sideways={result['sideways']}%, Bearish={result['bearish']}%")
    print(f"  Expected: Bullish ~22% (30*0.72), Bearish ~37% (30*1.24)")

    # Test case 4: Electronics with positive avg_return (bearish cap should apply)
    print("\nTest 4: Electronics with avg_return=10% - Bearish cap should apply")
    result = calibrator.calibrate_probabilities(
        raw_bullish=30,
        raw_bearish=40,
        raw_sideways=30,
        final_score=68,
        theme='Electronics',
        backtest_avg_return=10.0
    )
    print(f"  Raw: Bullish=30%, Sideways=30%, Bearish=40%")
    print(f"  Calibrated: Bullish={result['bullish']}%, Sideways={result['sideways']}%, Bearish={result['bearish']}%")
    print(f"  Expected: Bearish capped at 20% (avg_return > 5%)")

    # Test case 5: Unknown theme (should use fallback CALIBRATION_MAP)
    print("\nTest 5: Unknown theme - Should use fallback calibration")
    result = calibrator.calibrate_probabilities(
        raw_bullish=10,
        raw_bearish=10,
        raw_sideways=80,
        final_score=30,
        theme='UnknownTheme'
    )
    print(f"  Raw: Bullish=10%, Sideways=80%, Bearish=10%")
    print(f"  Calibrated: Bullish={result['bullish']}%, Sideways={result['sideways']}%, Bearish={result['bearish']}%")
    print(f"  Expected: Uses fallback CALIBRATION_MAP (Bullish 1.55x, Bearish 1.50x)")

    print("\n" + "=" * 80)
    print("Theme-specific calibration multipliers:")
    for theme, config in calibrator.THEME_CALIBRATION_MAP.items():
        print(f"  {theme}: bullish={config['bullish']:.2f}x, bearish={config['bearish']:.2f}x, floor={config['floor']}")

    # Test return ranges (Phase 3.9.1)
    print("\n" + "=" * 80)
    print("Theme-specific return ranges (P25~P75):")
    test_themes = ['Semiconductor', 'Finance', 'AI_BigData', 'Electronics', 'UnknownTheme']
    for theme in test_themes:
        bull_range = calibrator.get_return_range(theme, 'bullish')
        side_range = calibrator.get_return_range(theme, 'sideways')
        bear_range = calibrator.get_return_range(theme, 'bearish')
        print(f"  {theme}:")
        print(f"    Bullish: +{bull_range[0]:.1f}% ~ +{bull_range[1]:.1f}%")
        print(f"    Sideways: {side_range[0]:+.1f}% ~ {side_range[1]:+.1f}%")
        print(f"    Bearish: {bear_range[0]:.1f}% ~ {bear_range[1]:.1f}%")
    print("=" * 80)
