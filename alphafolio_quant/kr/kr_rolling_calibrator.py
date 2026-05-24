"""
Rolling Window Out-of-Sample Calibrator for Korean Stock Analysis (Phase 5)

Purpose: Improve scenario probability calibration using rolling window approach
- Train on past data (9 months)
- Validate on recent data (2 months)
- Gap period (30 days) to prevent data leakage
- Apply only when validation shows 5%+ improvement

Based on '지표 개선 방안.md' Section 9

File: kr/kr_rolling_calibrator.py
"""

import logging
import numpy as np
from datetime import date, timedelta
from dateutil.relativedelta import relativedelta
from typing import Dict, List, Optional, Tuple
import pickle
import os

logger = logging.getLogger(__name__)


class PlattScaling:
    """
    Platt Scaling: Sigmoid function for probability calibration

    P_calibrated = 1 / (1 + exp(A * P_raw + B))

    Only 2 parameters (A, B) - low overfitting risk
    More stable out-of-sample performance than Isotonic Regression
    """

    def __init__(self):
        self.A = None
        self.B = None
        self.is_fitted = False

    def fit(self, probs: np.ndarray, labels: np.ndarray) -> 'PlattScaling':
        """
        Fit parameters A, B using Maximum Likelihood

        Args:
            probs: Raw predicted probabilities (0-1)
            labels: Actual outcomes (0 or 1)

        Returns:
            self
        """
        from scipy.optimize import minimize

        # Convert to numpy arrays
        probs = np.array(probs, dtype=float)
        labels = np.array(labels, dtype=float)

        # Filter out invalid values
        valid_mask = (probs >= 0) & (probs <= 1) & np.isfinite(probs) & np.isfinite(labels)
        probs = probs[valid_mask]
        labels = labels[valid_mask]

        if len(probs) < 10:
            logger.warning("Insufficient data for Platt Scaling fitting")
            self.A = 0
            self.B = 0
            self.is_fitted = False
            return self

        def neg_log_likelihood(params):
            A, B = params
            # Sigmoid transformation
            calibrated = 1 / (1 + np.exp(A * probs + B))
            # Clip to avoid log(0)
            calibrated = np.clip(calibrated, 1e-10, 1 - 1e-10)
            # Negative log likelihood
            ll = labels * np.log(calibrated) + (1 - labels) * np.log(1 - calibrated)
            return -np.sum(ll)

        # Initial guess
        initial_params = [0, 0]

        try:
            result = minimize(neg_log_likelihood, initial_params, method='BFGS')
            self.A, self.B = result.x
            self.is_fitted = True
            logger.info(f"PlattScaling fitted: A={self.A:.4f}, B={self.B:.4f}")
        except Exception as e:
            logger.error(f"PlattScaling fitting failed: {e}")
            self.A = 0
            self.B = 0
            self.is_fitted = False

        return self

    def transform(self, probs: np.ndarray) -> np.ndarray:
        """
        Apply calibration to raw probabilities

        Args:
            probs: Raw probabilities (0-1 or 0-100)

        Returns:
            Calibrated probabilities (same scale as input)
        """
        if not self.is_fitted or self.A is None:
            return probs

        # Detect if input is percentage (0-100) or probability (0-1)
        probs = np.array(probs, dtype=float)
        is_percentage = np.max(probs) > 1

        if is_percentage:
            probs_01 = probs / 100
        else:
            probs_01 = probs

        # Apply sigmoid transformation
        calibrated = 1 / (1 + np.exp(self.A * probs_01 + self.B))

        if is_percentage:
            return calibrated * 100
        return calibrated

    def save(self, filepath: str):
        """Save calibrator to file"""
        with open(filepath, 'wb') as f:
            pickle.dump({'A': self.A, 'B': self.B, 'is_fitted': self.is_fitted}, f)

    def load(self, filepath: str) -> bool:
        """Load calibrator from file"""
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                self.A = data['A']
                self.B = data['B']
                self.is_fitted = data['is_fitted']
            return True
        except Exception as e:
            logger.error(f"Failed to load PlattScaling: {e}")
            return False


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """
    Calculate Brier Score (lower is better)

    Brier Score = mean((predicted_prob - actual_outcome)^2)
    Range: 0 (perfect) to 1 (worst)

    Args:
        probs: Predicted probabilities (0-1)
        labels: Actual outcomes (0 or 1)

    Returns:
        Brier score
    """
    probs = np.array(probs, dtype=float)
    labels = np.array(labels, dtype=float)

    # Normalize to 0-1 if percentage
    if np.max(probs) > 1:
        probs = probs / 100

    return np.mean((probs - labels) ** 2)


class RollingCalibrator:
    """
    Rolling Window Out-of-Sample Calibrator

    Key principle: NEVER use future data

    Timeline:
    ----[Training]----[Validation]----[Gap]----[Apply]
        t-365        t-90           t-30        t(today)

    Example:
    - Training: 2025-01-15 ~ 2025-10-15 (9 months)
    - Validation: 2025-10-15 ~ 2025-12-15 (2 months)
    - Gap: 2025-12-15 ~ 2026-01-15 (1 month, prevents data leakage)
    - Apply: 2026-01-15 (today)
    """

    # Scenario outcome definitions (based on 90-day return)
    OUTCOME_THRESHOLDS = {
        'bullish': 10.0,    # return >= 10% -> bullish
        'bearish': -10.0    # return <= -10% -> bearish
        # sideways: -10% < return < 10%
    }

    def __init__(self, db_manager):
        """
        Initialize RollingCalibrator

        Args:
            db_manager: AsyncDatabaseManager instance
        """
        self.db_manager = db_manager
        self.calibrators = {
            'bullish': PlattScaling(),
            'bearish': PlattScaling()
            # sideways is derived (100 - bullish - bearish)
        }
        self.last_update = None
        self.validation_scores = {}

    async def _fetch_calibration_data(
        self,
        start_date: date,
        end_date: date
    ) -> Dict[str, List]:
        """
        Fetch historical predictions and actual outcomes

        Args:
            start_date: Start date for data fetch
            end_date: End date for data fetch

        Returns:
            dict: {
                'bullish_pred': [...],
                'bearish_pred': [...],
                'bullish_actual': [...],  # 1 if actual return >= 10%, else 0
                'bearish_actual': [...]   # 1 if actual return <= -10%, else 0
            }
        """
        query = """
        WITH predictions AS (
            SELECT
                g.symbol,
                g.date as pred_date,
                g.scenario_bullish_prob,
                g.scenario_bearish_prob,
                g.scenario_sideways_prob
            FROM kr_stock_grade g
            WHERE g.date >= $1 AND g.date <= $2
                AND g.scenario_bullish_prob IS NOT NULL
        ),
        outcomes AS (
            SELECT
                p.symbol,
                p.pred_date,
                p.scenario_bullish_prob,
                p.scenario_bearish_prob,
                -- Calculate 90-day return
                (
                    SELECT ((future.close - current.close) / NULLIF(current.close, 0) * 100)
                    FROM kr_intraday_total current
                    JOIN kr_intraday_total future ON future.symbol = current.symbol
                        AND future.date = (
                            SELECT MIN(f.date)
                            FROM kr_intraday_total f
                            WHERE f.symbol = current.symbol
                                AND f.date >= p.pred_date + INTERVAL '85 days'
                                AND f.date <= p.pred_date + INTERVAL '95 days'
                        )
                    WHERE current.symbol = p.symbol
                        AND current.date = p.pred_date
                ) as return_90d
            FROM predictions p
        )
        SELECT
            scenario_bullish_prob,
            scenario_bearish_prob,
            return_90d,
            CASE WHEN return_90d >= $3 THEN 1 ELSE 0 END as bullish_outcome,
            CASE WHEN return_90d <= $4 THEN 1 ELSE 0 END as bearish_outcome
        FROM outcomes
        WHERE return_90d IS NOT NULL
        """

        try:
            rows = await self.db_manager.execute_query(
                query,
                start_date,
                end_date,
                self.OUTCOME_THRESHOLDS['bullish'],
                self.OUTCOME_THRESHOLDS['bearish']
            )

            if not rows:
                return {
                    'bullish_pred': [],
                    'bearish_pred': [],
                    'bullish_actual': [],
                    'bearish_actual': []
                }

            result = {
                'bullish_pred': [float(r['scenario_bullish_prob']) / 100 for r in rows],
                'bearish_pred': [float(r['scenario_bearish_prob']) / 100 for r in rows],
                'bullish_actual': [int(r['bullish_outcome']) for r in rows],
                'bearish_actual': [int(r['bearish_outcome']) for r in rows]
            }

            logger.info(f"Fetched {len(rows)} calibration samples from {start_date} to {end_date}")
            return result

        except Exception as e:
            logger.error(f"Failed to fetch calibration data: {e}")
            return {
                'bullish_pred': [],
                'bearish_pred': [],
                'bullish_actual': [],
                'bearish_actual': []
            }

    async def rolling_calibration(
        self,
        today: date,
        train_months: int = 9,
        valid_months: int = 2,
        gap_days: int = 30
    ) -> Dict:
        """
        Rolling Window Out-of-Sample Calibration

        Args:
            today: Application date (today)
            train_months: Training period (months)
            valid_months: Validation period (months)
            gap_days: Gap period (days) - prevents data leakage

        Returns:
            dict: {
                'calibrators': {'bullish': PlattScaling, 'bearish': PlattScaling},
                'valid_scores': {'bullish': float, 'bearish': float},
                'improvements': {'bullish': float, 'bearish': float},
                'is_valid': bool
            }
        """
        # 1. Calculate time periods (NEVER use future data)
        valid_end = today - timedelta(days=gap_days)
        valid_start = valid_end - relativedelta(months=valid_months)
        train_end = valid_start - timedelta(days=1)
        train_start = train_end - relativedelta(months=train_months)

        logger.info(f"Rolling Calibration periods:")
        logger.info(f"  Train: {train_start} ~ {train_end}")
        logger.info(f"  Valid: {valid_start} ~ {valid_end}")
        logger.info(f"  Gap: {valid_end} ~ {today}")
        logger.info(f"  Apply: {today}")

        # 2. Fetch training data
        train_data = await self._fetch_calibration_data(train_start, train_end)

        if len(train_data['bullish_pred']) < 100:
            logger.warning(f"Insufficient training data: {len(train_data['bullish_pred'])} samples")
            return {
                'calibrators': None,
                'valid_scores': {},
                'improvements': {},
                'is_valid': False
            }

        # 3. Train calibrators
        new_calibrators = {
            'bullish': PlattScaling(),
            'bearish': PlattScaling()
        }

        new_calibrators['bullish'].fit(
            np.array(train_data['bullish_pred']),
            np.array(train_data['bullish_actual'])
        )
        new_calibrators['bearish'].fit(
            np.array(train_data['bearish_pred']),
            np.array(train_data['bearish_actual'])
        )

        # 4. Fetch validation data
        valid_data = await self._fetch_calibration_data(valid_start, valid_end)

        if len(valid_data['bullish_pred']) < 50:
            logger.warning(f"Insufficient validation data: {len(valid_data['bullish_pred'])} samples")
            return {
                'calibrators': None,
                'valid_scores': {},
                'improvements': {},
                'is_valid': False
            }

        # 5. Calculate validation scores (Brier Score)
        valid_scores = {}
        improvements = {}

        for scenario in ['bullish', 'bearish']:
            pred_key = f'{scenario}_pred'
            actual_key = f'{scenario}_actual'

            raw_probs = np.array(valid_data[pred_key])
            actual = np.array(valid_data[actual_key])

            # Before calibration
            brier_before = brier_score(raw_probs, actual)

            # After calibration
            calibrated_probs = new_calibrators[scenario].transform(raw_probs)
            brier_after = brier_score(calibrated_probs, actual)

            valid_scores[scenario] = brier_after
            improvement = (brier_before - brier_after) / brier_before * 100 if brier_before > 0 else 0
            improvements[scenario] = improvement

            logger.info(f"{scenario.capitalize()} Brier Score: {brier_before:.4f} -> {brier_after:.4f} ({improvement:+.1f}%)")

        # 6. Determine if calibration should be applied (5% improvement threshold)
        # Apply if EITHER bullish or bearish shows improvement
        is_valid = any(imp >= 5.0 for imp in improvements.values())

        if is_valid:
            logger.info("Rolling calibration VALID - will be applied")
            self.calibrators = new_calibrators
            self.validation_scores = valid_scores
        else:
            logger.info("Rolling calibration NOT valid - improvement below threshold")

        self.last_update = today

        return {
            'calibrators': new_calibrators if is_valid else None,
            'valid_scores': valid_scores,
            'improvements': improvements,
            'is_valid': is_valid
        }

    def calibrate_probabilities(
        self,
        bullish_prob: float,
        bearish_prob: float,
        sideways_prob: float
    ) -> Dict[str, int]:
        """
        Apply rolling calibration to scenario probabilities

        Args:
            bullish_prob: Raw bullish probability (0-100)
            bearish_prob: Raw bearish probability (0-100)
            sideways_prob: Raw sideways probability (0-100)

        Returns:
            dict: {'bullish': int, 'sideways': int, 'bearish': int}
        """
        if not self.calibrators['bullish'].is_fitted:
            # No calibration available, return raw probabilities
            return {
                'bullish': int(bullish_prob),
                'sideways': int(sideways_prob),
                'bearish': int(bearish_prob)
            }

        # Apply calibration
        cal_bullish = self.calibrators['bullish'].transform(np.array([bullish_prob]))[0]
        cal_bearish = self.calibrators['bearish'].transform(np.array([bearish_prob]))[0]

        # Normalize to 100%
        total = cal_bullish + sideways_prob + cal_bearish
        if total <= 0:
            return {'bullish': 33, 'sideways': 34, 'bearish': 33}

        norm_bullish = round(cal_bullish / total * 100)
        norm_bearish = round(cal_bearish / total * 100)
        norm_sideways = 100 - norm_bullish - norm_bearish

        # Ensure non-negative
        if norm_sideways < 0:
            norm_sideways = 0
            norm_bullish = round(cal_bullish / (cal_bullish + cal_bearish) * 100)
            norm_bearish = 100 - norm_bullish

        return {
            'bullish': norm_bullish,
            'sideways': norm_sideways,
            'bearish': norm_bearish
        }

    async def daily_calibration_update(self) -> Dict:
        """
        Daily update: Train new calibrator using rolling window

        Should be called once per day (e.g., in batch job)

        Returns:
            dict: Calibration result with validation scores
        """
        today = date.today()

        logger.info(f"Daily calibration update for {today}")

        result = await self.rolling_calibration(today)

        if result['is_valid']:
            # Save calibrators for persistence
            save_dir = os.path.join(os.path.dirname(__file__), 'calibrator_cache')
            os.makedirs(save_dir, exist_ok=True)

            self.calibrators['bullish'].save(os.path.join(save_dir, 'bullish_calibrator.pkl'))
            self.calibrators['bearish'].save(os.path.join(save_dir, 'bearish_calibrator.pkl'))

            logger.info(f"Calibrators saved to {save_dir}")

        return result

    def load_cached_calibrators(self) -> bool:
        """
        Load cached calibrators from disk

        Returns:
            bool: True if loaded successfully
        """
        save_dir = os.path.join(os.path.dirname(__file__), 'calibrator_cache')

        bullish_path = os.path.join(save_dir, 'bullish_calibrator.pkl')
        bearish_path = os.path.join(save_dir, 'bearish_calibrator.pkl')

        if os.path.exists(bullish_path) and os.path.exists(bearish_path):
            bullish_loaded = self.calibrators['bullish'].load(bullish_path)
            bearish_loaded = self.calibrators['bearish'].load(bearish_path)

            if bullish_loaded and bearish_loaded:
                logger.info("Cached calibrators loaded successfully")
                return True

        logger.info("No cached calibrators found")
        return False


# Validation test
if __name__ == '__main__':
    import asyncio

    print("=" * 80)
    print("Rolling Calibrator Validation Test (Phase 5)")
    print("=" * 80)

    # Test PlattScaling
    print("\n[Test 1] PlattScaling basic functionality")
    ps = PlattScaling()

    # Simulated data: model under-predicts bullish
    np.random.seed(42)
    raw_probs = np.random.uniform(0.2, 0.6, 100)  # Predicted 20-60%
    # Actual outcomes have higher rate than predicted
    actual_outcomes = (np.random.random(100) < raw_probs * 1.3).astype(int)

    ps.fit(raw_probs, actual_outcomes)
    print(f"  Fitted parameters: A={ps.A:.4f}, B={ps.B:.4f}")

    calibrated = ps.transform(raw_probs)
    print(f"  Raw probs mean: {np.mean(raw_probs):.3f}")
    print(f"  Calibrated probs mean: {np.mean(calibrated):.3f}")
    print(f"  Actual outcome rate: {np.mean(actual_outcomes):.3f}")

    # Brier score comparison
    brier_before = brier_score(raw_probs, actual_outcomes)
    brier_after = brier_score(calibrated, actual_outcomes)
    improvement = (brier_before - brier_after) / brier_before * 100
    print(f"  Brier Score: {brier_before:.4f} -> {brier_after:.4f} ({improvement:+.1f}%)")

    # Test percentage input
    print("\n[Test 2] PlattScaling with percentage input (0-100)")
    raw_pct = np.array([30, 50, 70, 90])
    calibrated_pct = ps.transform(raw_pct)
    print(f"  Raw: {raw_pct}")
    print(f"  Calibrated: {calibrated_pct.round(1)}")

    print("\n" + "=" * 80)
    print("PlattScaling tests completed")
    print("=" * 80)
