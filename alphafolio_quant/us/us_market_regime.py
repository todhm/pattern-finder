"""
US Market Regime Detector v4.0 (HMM Only)

시장 레짐 감지 및 정적 팩터 가중치 결정
+ HMM 기반 레짐 감지 (3-State)
+ 4-Dimension Regime Composite Score
+ 레짐별 정적 가중치 (Factor Timing 최소화)

Regimes (3-Regime Classification - HMM):
- BULL: 강세장 (Growth/Momentum 강조)
- NEUTRAL: 중립장 (균형)
- BEAR: 약세장 (Quality/Value 방어)

4-Dimension Composite Score (0-100 each):
- Liquidity Score: Fed RRP, SPX Volume, Credit Spread
- Macro Score: Fed Rate, CPI, Yield Curve, MOVE Index, DXY
- Volatility Score: VIX, IV Percentile, HV 20d
- Risk Appetite Score: Put/Call Ratio, IV Skew, News Sentiment

Data Sources:
- VIX Proxy: us_option_daily_summary (SPY avg_implied_volatility)
- SPY/QQQ: us_daily_etf
- Macro: us_fed_funds_rate, us_cpi, us_unemployment_rate, us_treasury_yield
- Extended: us_move_index, us_dollar_index, us_credit_spread, us_fed_rrp, us_vix

References:
- arXiv:2010.08601 - IC Volatility (std 3-11x mean)
- arXiv:2410.14841v1 - Factor Timing "Siren Song"
- MDPI JRFM 2020 - HMM Regime Detection

File: us/us_market_regime.py
"""

import logging
from typing import Dict, Optional, List, Tuple
from datetime import date, timedelta
from decimal import Decimal
import numpy as np

# HMM 라이브러리 (필수)
try:
    from hmmlearn.hmm import GaussianHMM
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    raise ImportError("hmmlearn package required. Install: pip install hmmlearn")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ========================================================================
# Macro Environment & Regime Definitions
# ========================================================================

# ========================================================================
# scenario_prob: 2024-10-01 ~ 2025-09-10 실측 데이터 기반 (SPY 60일 수익률)
# - Bullish: >+5%, Sideways: -5%~+5%, Bearish: <-5%
# - Base Rate (무조건적): Bullish=50%, Sideways=35%, Bearish=16%
# - STAGFLATION, HARD_LANDING: 실측값 사용
# - SOFT_LANDING, REFLATION, DEFLATION: 샘플 부족으로 Base Rate 기반 추정
# ========================================================================
MACRO_ENVIRONMENT = {
    'SOFT_LANDING': {
        'description': '연착륙 (GDP+, PMI>50, 인플레 하락)',
        'scenario_prob': {'강세': 0.55, '횡보': 0.32, '약세': 0.13},  # Base Rate보다 약간 긍정적 (추정)
        'factor_bias': {'growth': 1.2, 'momentum': 1.1, 'quality': 0.9, 'value': 0.8}
    },
    'HARD_LANDING': {
        'description': '경착륙 (GDP-, PMI<50, 인플레 하락)',
        'scenario_prob': {'강세': 0.57, '횡보': 0.30, '약세': 0.13},  # 실측: N=61일
        'factor_bias': {'growth': 0.7, 'momentum': 0.6, 'quality': 1.3, 'value': 1.0}
    },
    'REFLATION': {
        'description': '리플레이션 (GDP+, PMI>50, 인플레 상승)',
        'scenario_prob': {'강세': 0.52, '횡보': 0.33, '약세': 0.15},  # Base Rate 수준 (추정)
        'factor_bias': {'growth': 1.0, 'momentum': 1.2, 'quality': 0.9, 'value': 1.1}
    },
    'STAGFLATION': {
        'description': '스태그플레이션 (GDP-, PMI<50, 인플레 상승)',
        'scenario_prob': {'강세': 0.47, '횡보': 0.37, '약세': 0.16},  # 실측: N=175일
        'factor_bias': {'growth': 0.5, 'momentum': 0.5, 'quality': 1.4, 'value': 1.2}
    },
    'DEFLATION': {
        'description': '디플레이션 (GDP-, PMI<50, 인플레 급락)',
        'scenario_prob': {'강세': 0.45, '횡보': 0.35, '약세': 0.20},  # Base Rate보다 약간 부정적 (추정)
        'factor_bias': {'growth': 0.8, 'momentum': 0.7, 'quality': 1.2, 'value': 1.1}
    }
}


# ========================================================================
# HMM Regime Weights (3-State: BULL/NEUTRAL/BEAR)
# ========================================================================
# 학술 연구 기반 정적 가중치
# - Factor Timing 최소화 (IC 변동성 3-11x mean)
# - 레짐 전환 시에만 리밸런싱 (분기별 권고)
# - Transaction Cost 고려

REGIME_WEIGHTS = {
    'BULL': {
        'description': '강세장 (HMM: Low Vol + Positive Return)',
        'growth': 0.35,      # Growth 강조
        'momentum': 0.25,    # Momentum 활용
        'quality': 0.20,     # Quality 기본
        'value': 0.20        # Value 축소
    },
    'NEUTRAL': {
        'description': '중립장 (HMM: Mixed Signals)',
        'growth': 0.28,      # 약간 Growth bias
        'quality': 0.27,     # Quality 균형
        'value': 0.25,       # Value 균형
        'momentum': 0.20     # Momentum 축소
    },
    'BEAR': {
        'description': '약세장 (HMM: High Vol + Negative Return)',
        'quality': 0.40,     # Quality 방어
        'value': 0.30,       # Value 방어
        'growth': 0.20,      # Growth 축소
        'momentum': 0.10     # Momentum 최소화
    }
}


# ========================================================================
# HMM Regime Detector Class
# ========================================================================

class HMMRegimeDetector:
    """
    Hidden Markov Model 기반 시장 레짐 감지 (v2.0)

    학술 연구 근거:
    - MDPI JRFM 2020: HMM Regime Detection
    - 3-State 모델: BULL/NEUTRAL/BEAR

    Features:
    - SPY 일간 수익률
    - 20일 실현 변동성

    Usage:
        detector = HMMRegimeDetector(n_regimes=3)
        await detector.fit(db_manager, lookback_days=504)
        result = await detector.predict_current_regime(db_manager)
    """

    def __init__(self, n_regimes: int = 3):
        """
        Args:
            n_regimes: 레짐 수 (기본 3: BULL/NEUTRAL/BEAR)
        """
        self.n_regimes = n_regimes
        self.model = None
        self.is_fitted = False
        self.regime_labels = {}
        self.regime_stats = {}

        if not HMM_AVAILABLE:
            logger.warning("hmmlearn not installed. HMM regime detection unavailable.")

    @staticmethod
    def _calculate_rolling_volatility(returns: np.ndarray, window: int = 20) -> np.ndarray:
        """
        Rolling 변동성 계산 (numpy 벡터화 - 효율성 최적화)

        Args:
            returns: 일간 수익률 배열
            window: Rolling window 크기 (기본 20일)

        Returns:
            연환산 변동성 배열
        """
        n = len(returns)
        volatilities = np.full(n, 0.15)  # 기본값

        if n < window:
            # 데이터 부족 시 전체 std 사용
            if n > 1:
                volatilities[:] = np.std(returns) * np.sqrt(252)
            return volatilities

        # Strided trick for efficient rolling window
        # https://numpy.org/doc/stable/reference/generated/numpy.lib.stride_tricks.sliding_window_view.html
        try:
            from numpy.lib.stride_tricks import sliding_window_view
            rolling_windows = sliding_window_view(returns, window)
            rolling_std = np.std(rolling_windows, axis=1) * np.sqrt(252)

            # 앞부분 (window 미만)은 expanding std 사용
            for i in range(1, window):
                volatilities[i] = np.std(returns[:i+1]) * np.sqrt(252)

            # window 이후는 rolling std
            volatilities[window-1:] = rolling_std

        except ImportError:
            # numpy 버전이 낮을 경우 fallback
            for i in range(n):
                if i < window - 1:
                    volatilities[i] = np.std(returns[:i+1]) * np.sqrt(252) if i > 0 else 0.15
                else:
                    volatilities[i] = np.std(returns[i-window+1:i+1]) * np.sqrt(252)

        return volatilities

    async def fit(self, db_manager, lookback_days: int = 504) -> 'HMMRegimeDetector':
        """
        HMM 모델 학습 (2년 데이터)

        Args:
            db_manager: AsyncDatabaseManager instance
            lookback_days: 학습 데이터 기간 (기본 504일 = 2년)

        Returns:
            self (fitted model)
        """
        if not HMM_AVAILABLE:
            raise ImportError("hmmlearn package required. Install: pip install hmmlearn")

        query = """
        WITH daily_features AS (
            SELECT
                date,
                close,
                LAG(close) OVER (ORDER BY date) as prev_close
            FROM us_daily_etf
            WHERE symbol = 'SPY'
            ORDER BY date DESC
            LIMIT $1
        )
        SELECT
            date,
            CASE WHEN prev_close > 0
                 THEN (close - prev_close) / prev_close
                 ELSE 0 END as daily_return,
            close
        FROM daily_features
        WHERE prev_close IS NOT NULL
        ORDER BY date
        """

        result = await db_manager.execute_query(query, lookback_days + 1)

        if not result or len(result) < 100:
            raise ValueError(f"Insufficient data for HMM training: {len(result) if result else 0} records")

        # Feature 계산
        returns = np.array([float(r['daily_return']) for r in result])

        # 20일 실현 변동성 계산 (numpy 벡터화 - 효율성 최적화)
        volatilities = self._calculate_rolling_volatility(returns, window=20)

        # Feature matrix (수익률, 변동성)
        features = np.column_stack([returns, volatilities])

        # NaN/Inf 제거
        valid_mask = np.isfinite(features).all(axis=1)
        features = features[valid_mask]

        if len(features) < 100:
            raise ValueError(f"Insufficient valid data after filtering: {len(features)} records")

        # HMM 모델 초기화 및 학습
        self.model = GaussianHMM(
            n_components=self.n_regimes,
            covariance_type="full",
            n_iter=1000,
            random_state=42
        )

        self.model.fit(features)
        self.is_fitted = True

        # 레짐 특성 분석 및 라벨링
        self._analyze_regimes(features)

        logger.info(f"HMM model fitted with {len(features)} samples, {self.n_regimes} regimes")
        for regime_id, label in self.regime_labels.items():
            stats = self.regime_stats[regime_id]
            logger.info(f"  {label}: mean_ret={stats['mean_return']*100:.2f}%, "
                       f"mean_vol={stats['mean_vol']*100:.1f}%, count={stats['count']}")

        return self

    def _analyze_regimes(self, features: np.ndarray):
        """
        각 레짐의 특성 분석 및 라벨링

        수익률 기준으로 BULL/NEUTRAL/BEAR 분류
        """
        states = self.model.predict(features)

        # 각 상태의 평균 수익률/변동성 계산
        regime_stats = {}
        for i in range(self.n_regimes):
            mask = states == i
            if mask.sum() > 0:
                regime_stats[i] = {
                    'mean_return': features[mask, 0].mean(),
                    'mean_vol': features[mask, 1].mean(),
                    'count': int(mask.sum())
                }
            else:
                regime_stats[i] = {
                    'mean_return': 0,
                    'mean_vol': 0.15,
                    'count': 0
                }

        # 수익률 기준 정렬하여 라벨링
        sorted_regimes = sorted(
            regime_stats.items(),
            key=lambda x: x[1]['mean_return'],
            reverse=True
        )

        # 3-State 라벨링
        if self.n_regimes == 3:
            self.regime_labels = {
                sorted_regimes[0][0]: 'BULL',
                sorted_regimes[1][0]: 'NEUTRAL',
                sorted_regimes[2][0]: 'BEAR'
            }
        else:
            # n_regimes > 3인 경우 숫자 라벨
            self.regime_labels = {
                sorted_regimes[i][0]: f'REGIME_{i}'
                for i in range(self.n_regimes)
            }

        self.regime_stats = regime_stats

    async def predict_current_regime(
        self,
        db_manager,
        analysis_date: Optional[date] = None
    ) -> Dict:
        """
        현재 레짐 예측

        Args:
            db_manager: AsyncDatabaseManager instance
            analysis_date: 분석 날짜 (None이면 최신)

        Returns:
            {
                'regime': str (BULL/NEUTRAL/BEAR),
                'regime_id': int,
                'transition_probs': Dict[str, float],
                'regime_stats': Dict,
                'weights': Dict[str, float]
            }
        """
        if not self.is_fitted:
            await self.fit(db_manager)

        # 최근 30일 데이터로 현재 상태 예측
        query = """
        WITH daily_features AS (
            SELECT
                date,
                close,
                LAG(close) OVER (ORDER BY date) as prev_close
            FROM us_daily_etf
            WHERE symbol = 'SPY'
              AND ($1::DATE IS NULL OR date <= $1::DATE)
            ORDER BY date DESC
            LIMIT 31
        )
        SELECT
            date,
            CASE WHEN prev_close > 0
                 THEN (close - prev_close) / prev_close
                 ELSE 0 END as daily_return,
            close
        FROM daily_features
        WHERE prev_close IS NOT NULL
        ORDER BY date
        """

        result = await db_manager.execute_query(query, analysis_date)

        if not result or len(result) < 20:
            logger.warning("Insufficient recent data for HMM prediction, using NEUTRAL")
            return self._default_response()

        # Feature 계산 (numpy 벡터화 - 효율성 최적화)
        returns = np.array([float(r['daily_return']) for r in result])
        volatilities = self._calculate_rolling_volatility(returns, window=20)
        features = np.column_stack([returns, volatilities])

        # NaN/Inf 처리
        valid_mask = np.isfinite(features).all(axis=1)
        if not valid_mask.any():
            return self._default_response()

        features = features[valid_mask]

        # 현재 상태 예측
        try:
            current_state = self.model.predict(features)[-1]
        except Exception as e:
            logger.error(f"HMM prediction failed: {e}")
            return self._default_response()

        # 상태 전이 확률
        transition_probs = self.model.transmat_[current_state]

        # 현재 레짐 라벨
        current_regime = self.regime_labels.get(current_state, 'NEUTRAL')

        # v2.0 정적 가중치
        weights = REGIME_WEIGHTS.get(current_regime, REGIME_WEIGHTS['NEUTRAL']).copy()
        # description 제거
        weights = {k: v for k, v in weights.items() if k != 'description'}

        return {
            'regime': current_regime,
            'regime_id': int(current_state),
            'transition_probs': {
                self.regime_labels.get(i, f'STATE_{i}'): float(transition_probs[i])
                for i in range(self.n_regimes)
            },
            'regime_stats': self.regime_stats.get(current_state, {}),
            'weights': weights,
            'method': 'HMM'
        }

    def _default_response(self) -> Dict:
        """기본 응답 (NEUTRAL)"""
        weights = REGIME_WEIGHTS['NEUTRAL'].copy()
        weights = {k: v for k, v in weights.items() if k != 'description'}
        return {
            'regime': 'NEUTRAL',
            'regime_id': -1,
            'transition_probs': {},
            'regime_stats': {},
            'weights': weights,
            'method': 'HMM_FALLBACK'
        }


class USMarketRegimeDetector:
    """시장 레짐 감지 및 정적 가중치 결정 (v4.0: HMM Only)"""

    def __init__(
        self,
        db_manager,
        analysis_date: Optional[date] = None
    ):
        """
        Args:
            db_manager: AsyncDatabaseManager instance
            analysis_date: 분석 날짜 (None이면 최신)
        """
        self.db = db_manager
        self.analysis_date = analysis_date
        self.indicators = {}

        # HMM 레짐 감지기 (lazy initialization)
        self._hmm_detector: Optional[HMMRegimeDetector] = None

    async def _get_hmm_detector(self) -> HMMRegimeDetector:
        """HMM 감지기 lazy initialization"""
        if self._hmm_detector is None:
            self._hmm_detector = HMMRegimeDetector(n_regimes=3)
            await self._hmm_detector.fit(self.db)
        return self._hmm_detector

    async def detect_regime(self) -> Dict:
        """
        시장 레짐 감지 및 결과 반환 (v4.0: HMM Only)

        Returns:
            {
                'regime': str (BULL/NEUTRAL/BEAR),
                'weights': {'value': float, 'quality': float, ...},
                'indicators': {...},
                'description': str,
                'composite_scores': {...},
                'method': str ('HMM')
            }
        """
        return await self._detect_regime_hmm()

    async def _detect_regime_hmm(self) -> Dict:
        """
        HMM 기반 레짐 감지 (v2.0)

        - 3-State HMM (BULL/NEUTRAL/BEAR)
        - 정적 가중치 (Factor Timing 최소화)
        - Transaction Cost 고려
        """
        try:
            # HMM 감지기 초기화 및 예측
            hmm_detector = await self._get_hmm_detector()
            hmm_result = await hmm_detector.predict_current_regime(
                self.db,
                self.analysis_date
            )

            regime = hmm_result['regime']
            weights = hmm_result['weights']

            # 기존 지표도 수집 (모니터링/로깅용)
            vix_data = await self._get_vix_proxy()
            market_data = await self._get_market_indicators()

            self.indicators = {
                **vix_data,
                **market_data,
                'hmm_regime_stats': hmm_result.get('regime_stats', {}),
                'hmm_transition_probs': hmm_result.get('transition_probs', {})
            }

            # Composite Score (간소화)
            composite_scores = {
                'liquidity_score': 50.0,
                'macro_score': 50.0,
                'volatility_score': 50.0,
                'risk_appetite_score': 50.0,
                'composite_score': 50.0
            }

            # 가능하면 계산
            try:
                macro_data = await self._get_macro_indicators()
                yield_data = await self._get_yield_curve()
                extended_data = await self._get_extended_indicators()
                gdp_pmi_data = await self._get_gdp_pmi_data()

                self.indicators.update({
                    **macro_data,
                    **yield_data,
                    **extended_data,
                    **gdp_pmi_data
                })

                composite_scores = self._calculate_composite_scores()
            except Exception:
                pass  # 실패해도 HMM 결과 사용

            # Macro Environment (HMM 레짐 기반 매핑)
            macro_env_map = {
                'BULL': 'SOFT_LANDING',
                'NEUTRAL': 'REFLATION',
                'BEAR': 'HARD_LANDING'
            }
            macro_env = macro_env_map.get(regime, 'SOFT_LANDING')

            # Scenario Probability (HMM 전이 확률 기반)
            transition_probs = hmm_result.get('transition_probs', {})
            scenario_prob = {
                '강세': transition_probs.get('BULL', 0.33),
                '횡보': transition_probs.get('NEUTRAL', 0.34),
                '약세': transition_probs.get('BEAR', 0.33)
            }

            logger.info(f"Market Regime: {regime} ({REGIME_WEIGHTS[regime]['description']}) [HMM]")
            logger.info(f"HMM Stats: {hmm_result.get('regime_stats', {})}")
            logger.info(f"Transition Probs: BULL={scenario_prob['강세']:.0%}, "
                       f"NEUTRAL={scenario_prob['횡보']:.0%}, BEAR={scenario_prob['약세']:.0%}")
            logger.info(f"Weights (Static v2.0): V={weights['value']:.0%}, Q={weights['quality']:.0%}, "
                       f"M={weights['momentum']:.0%}, G={weights['growth']:.0%}")

            return {
                'regime': regime,
                'weights': weights,
                'indicators': self.indicators,
                'description': REGIME_WEIGHTS[regime]['description'],
                'composite_scores': composite_scores,
                'macro_environment': macro_env,
                'macro_env_description': MACRO_ENVIRONMENT[macro_env]['description'],
                'scenario_probability': scenario_prob,
                'method': 'HMM',
                'hmm_regime_id': hmm_result.get('regime_id', -1),
                'hmm_transition_probs': transition_probs
            }

        except Exception as e:
            logger.error(f"HMM regime detection failed: {e}")
            # Fallback to NEUTRAL
            fallback_weights = REGIME_WEIGHTS['NEUTRAL'].copy()
            fallback_weights = {k: v for k, v in fallback_weights.items() if k != 'description'}
            return {
                'regime': 'NEUTRAL',
                'weights': fallback_weights,
                'indicators': {},
                'description': 'Fallback to NEUTRAL due to error',
                'composite_scores': {
                    'liquidity_score': 50.0,
                    'macro_score': 50.0,
                    'volatility_score': 50.0,
                    'risk_appetite_score': 50.0,
                    'composite_score': 50.0
                },
                'macro_environment': 'SOFT_LANDING',
                'macro_env_description': 'Fallback',
                'scenario_probability': {'강세': 0.33, '횡보': 0.34, '약세': 0.33},
                'method': 'HMM_FALLBACK'
            }

    async def _get_vix_proxy(self) -> Dict:
        """SPY 옵션 IV로 VIX 대용 지표 조회"""

        query = """
        SELECT
            date,
            avg_implied_volatility * 100 as vix_proxy,
            avg_call_iv * 100 as call_iv,
            avg_put_iv * 100 as put_iv,
            CASE WHEN total_call_volume > 0
                 THEN total_put_volume::float / total_call_volume
                 ELSE 1.0 END as put_call_ratio
        FROM us_option_daily_summary
        WHERE symbol = 'SPY'
            AND ($1::DATE IS NULL OR date <= $1::DATE)
        ORDER BY date DESC
        LIMIT 1
        """

        try:
            result = await self.db.execute_query(query, self.analysis_date)

            if result and result[0]['vix_proxy']:
                row = result[0]
                vix = float(row['vix_proxy']) if row['vix_proxy'] else 20.0
                call_iv = float(row['call_iv']) if row['call_iv'] else vix
                put_iv = float(row['put_iv']) if row['put_iv'] else vix
                pcr = float(row['put_call_ratio']) if row['put_call_ratio'] else 1.0

                return {
                    'vix_proxy': vix,
                    'put_call_ratio': pcr,
                    'iv_skew': put_iv - call_iv,
                    'vix_date': row['date']
                }
        except Exception as e:
            logger.warning(f"VIX proxy query failed: {e}")

        # Fallback
        return {
            'vix_proxy': 20.0,
            'put_call_ratio': 1.0,
            'iv_skew': 0.0,
            'vix_date': None
        }

    async def _get_market_indicators(self) -> Dict:
        """SPY, QQQ 시장 지표 조회"""

        # SPY 수익률 및 MA200
        spy_query = """
        WITH spy_prices AS (
            SELECT
                date,
                close,
                LAG(close, 21) OVER (ORDER BY date) as close_1m,
                LAG(close, 63) OVER (ORDER BY date) as close_3m,
                AVG(close) OVER (ORDER BY date ROWS BETWEEN 199 PRECEDING AND CURRENT ROW) as ma200
            FROM us_daily_etf
            WHERE symbol = 'SPY'
            ORDER BY date DESC
        )
        SELECT
            date,
            close,
            close_1m,
            close_3m,
            ma200,
            CASE WHEN close_1m > 0 THEN (close - close_1m) / close_1m * 100 ELSE 0 END as return_1m,
            CASE WHEN close_3m > 0 THEN (close - close_3m) / close_3m * 100 ELSE 0 END as return_3m,
            CASE WHEN ma200 > 0 THEN (close - ma200) / ma200 * 100 ELSE 0 END as ma200_distance
        FROM spy_prices
        WHERE ($1::DATE IS NULL OR date <= $1::DATE)
        ORDER BY date DESC
        LIMIT 1
        """

        # QQQ 수익률 (NASDAQ 대용)
        qqq_query = """
        WITH qqq_prices AS (
            SELECT
                date,
                close,
                LAG(close, 63) OVER (ORDER BY date) as close_3m
            FROM us_daily_etf
            WHERE symbol = 'QQQ'
            ORDER BY date DESC
        )
        SELECT
            CASE WHEN close_3m > 0 THEN (close - close_3m) / close_3m * 100 ELSE 0 END as qqq_return_3m
        FROM qqq_prices
        WHERE ($1::DATE IS NULL OR date <= $1::DATE)
        ORDER BY date DESC
        LIMIT 1
        """

        result = {
            'spy_return_1m': 0.0,
            'spy_return_3m': 0.0,
            'spy_ma200_distance': 0.0,
            'nasdaq_vs_spy_3m': 0.0
        }

        try:
            spy_result = await self.db.execute_query(spy_query, self.analysis_date)
            if spy_result and spy_result[0]:
                row = spy_result[0]
                result['spy_return_1m'] = float(row['return_1m'] or 0)
                result['spy_return_3m'] = float(row['return_3m'] or 0)
                result['spy_ma200_distance'] = float(row['ma200_distance'] or 0)

            qqq_result = await self.db.execute_query(qqq_query, self.analysis_date)
            if qqq_result and qqq_result[0]:
                qqq_return = float(qqq_result[0]['qqq_return_3m'] or 0)
                result['nasdaq_vs_spy_3m'] = qqq_return - result['spy_return_3m']

        except Exception as e:
            logger.warning(f"Market indicators query failed: {e}")

        return result

    async def _get_macro_indicators(self) -> Dict:
        """매크로 지표 조회"""

        result = {
            'fed_rate': 5.0,
            'fed_rate_change_6m': 0.0,
            'cpi_yoy': 3.0,
            'unemployment_rate': 4.0
        }

        try:
            # Fed Rate
            fed_query = """
            SELECT
                (SELECT value FROM us_fed_funds_rate
                 WHERE ($1::DATE IS NULL OR date <= $1::DATE)
                 ORDER BY date DESC LIMIT 1) as current_rate,
                (SELECT value FROM us_fed_funds_rate
                 WHERE date <= COALESCE($1::DATE, CURRENT_DATE) - INTERVAL '6 months'
                 ORDER BY date DESC LIMIT 1) as rate_6m_ago
            """
            fed_result = await self.db.execute_query(fed_query, self.analysis_date)
            if fed_result and fed_result[0]:
                current = float(fed_result[0]['current_rate'] or 5.0)
                ago = float(fed_result[0]['rate_6m_ago'] or current)
                result['fed_rate'] = current
                result['fed_rate_change_6m'] = current - ago

            # CPI YoY
            cpi_query = """
            SELECT
                (SELECT value FROM us_cpi
                 WHERE ($1::DATE IS NULL OR date <= $1::DATE)
                 ORDER BY date DESC LIMIT 1) as current_cpi,
                (SELECT value FROM us_cpi
                 WHERE date <= COALESCE($1::DATE, CURRENT_DATE) - INTERVAL '1 year'
                 ORDER BY date DESC LIMIT 1) as cpi_1y_ago
            """
            cpi_result = await self.db.execute_query(cpi_query, self.analysis_date)
            if cpi_result and cpi_result[0]:
                current = float(cpi_result[0]['current_cpi'] or 300)
                ago = float(cpi_result[0]['cpi_1y_ago'] or 290)
                if ago > 0:
                    result['cpi_yoy'] = (current - ago) / ago * 100

            # Unemployment
            unemp_query = """
            SELECT value FROM us_unemployment_rate
            WHERE ($1::DATE IS NULL OR date <= $1::DATE)
            ORDER BY date DESC LIMIT 1
            """
            unemp_result = await self.db.execute_query(unemp_query, self.analysis_date)
            if unemp_result and unemp_result[0]:
                result['unemployment_rate'] = float(unemp_result[0]['value'] or 4.0)

        except Exception as e:
            logger.warning(f"Macro indicators query failed: {e}")

        return result

    async def _get_yield_curve(self) -> Dict:
        """국채 수익률 곡선 조회 (Phase 3.4 Macro Engine)"""

        result = {
            'yield_10y': 4.2,
            'yield_2y': 3.8,
            'yield_spread': 0.4,
            'yield_curve_signal': 'NORMAL'
        }

        try:
            query = """
            SELECT
                t10.value as yield_10y,
                t2.value as yield_2y,
                (t10.value - t2.value) as spread
            FROM us_treasury_yield t10
            JOIN us_treasury_yield t2
                ON t10.date = t2.date
            WHERE t10.interval = '10year'
              AND t2.interval = '2year'
              AND ($1::DATE IS NULL OR t10.date <= $1::DATE)
            ORDER BY t10.date DESC
            LIMIT 1
            """

            yield_result = await self.db.execute_query(query, self.analysis_date)

            if yield_result and yield_result[0]:
                row = yield_result[0]
                y10 = float(row['yield_10y'] or 4.2)
                y2 = float(row['yield_2y'] or 3.8)
                spread = float(row['spread'] or 0.4)

                result['yield_10y'] = y10
                result['yield_2y'] = y2
                result['yield_spread'] = spread

                # Yield Curve Signal 분류
                if spread < 0:
                    result['yield_curve_signal'] = 'INVERTED'  # 경기침체 신호
                elif spread > 1.0:
                    result['yield_curve_signal'] = 'STEEPENING'  # 경기확장 신호
                else:
                    result['yield_curve_signal'] = 'NORMAL'

        except Exception as e:
            logger.warning(f"Yield curve query failed: {e}")

        return result

    async def _get_extended_indicators(self) -> Dict:
        """
        Phase 3.4.2: 확장 지표 조회
        - MOVE Index, DXY, Credit Spread, Fed RRP, VIX, HV 20d
        """
        result = {
            'move_index': None,
            'dxy': None,
            'credit_spread': None,
            'fed_rrp': None,
            'vix_actual': None,
            'hv_20d': None,
            'iv_percentile': None
        }

        try:
            # MOVE Index (채권 변동성)
            move_query = """
            SELECT value FROM us_move_index
            WHERE ($1::DATE IS NULL OR date <= $1::DATE)
            ORDER BY date DESC LIMIT 1
            """
            move_result = await self.db.execute_query(move_query, self.analysis_date)
            if move_result and move_result[0].get('value'):
                result['move_index'] = float(move_result[0]['value'])

            # DXY (달러 인덱스)
            dxy_query = """
            SELECT value FROM us_dollar_index
            WHERE ($1::DATE IS NULL OR date <= $1::DATE)
            ORDER BY date DESC LIMIT 1
            """
            dxy_result = await self.db.execute_query(dxy_query, self.analysis_date)
            if dxy_result and dxy_result[0].get('value'):
                result['dxy'] = float(dxy_result[0]['value'])

            # Credit Spread (신용 스프레드)
            credit_query = """
            SELECT value FROM us_credit_spread
            WHERE ($1::DATE IS NULL OR date <= $1::DATE)
            ORDER BY date DESC LIMIT 1
            """
            credit_result = await self.db.execute_query(credit_query, self.analysis_date)
            if credit_result and credit_result[0].get('value'):
                result['credit_spread'] = float(credit_result[0]['value'])

            # Fed RRP (역레포 - 유동성 지표)
            rrp_query = """
            SELECT value FROM us_fed_rrp
            WHERE ($1::DATE IS NULL OR date <= $1::DATE)
            ORDER BY date DESC LIMIT 1
            """
            rrp_result = await self.db.execute_query(rrp_query, self.analysis_date)
            if rrp_result and rrp_result[0].get('value'):
                result['fed_rrp'] = float(rrp_result[0]['value'])

            # VIX (실제 VIX)
            vix_query = """
            SELECT value FROM us_vix
            WHERE ($1::DATE IS NULL OR date <= $1::DATE)
            ORDER BY date DESC LIMIT 1
            """
            vix_result = await self.db.execute_query(vix_query, self.analysis_date)
            if vix_result and vix_result[0].get('value'):
                result['vix_actual'] = float(vix_result[0]['value'])

            # HV 20d (SPY Historical Volatility)
            hv_query = """
            WITH daily_returns AS (
                SELECT
                    date,
                    (close - LAG(close) OVER (ORDER BY date)) / LAG(close) OVER (ORDER BY date) as daily_return
                FROM us_daily_etf
                WHERE symbol = 'SPY'
                  AND ($1::DATE IS NULL OR date <= $1::DATE)
                ORDER BY date DESC
                LIMIT 21
            )
            SELECT STDDEV(daily_return) * SQRT(252) * 100 as hv_20d
            FROM daily_returns
            WHERE daily_return IS NOT NULL
            """
            hv_result = await self.db.execute_query(hv_query, self.analysis_date)
            if hv_result and hv_result[0].get('hv_20d'):
                result['hv_20d'] = float(hv_result[0]['hv_20d'])

            # IV Percentile (SPY IV의 1년 백분위)
            iv_pct_query = """
            WITH iv_history AS (
                SELECT avg_implied_volatility * 100 as iv
                FROM us_option_daily_summary
                WHERE symbol = 'SPY'
                  AND ($1::DATE IS NULL OR date <= $1::DATE)
                ORDER BY date DESC
                LIMIT 252
            ),
            current_iv AS (
                SELECT iv FROM iv_history LIMIT 1
            )
            SELECT
                (SELECT iv FROM current_iv) as current_iv,
                PERCENT_RANK() OVER (ORDER BY iv) * 100 as iv_percentile
            FROM iv_history
            ORDER BY iv DESC
            LIMIT 1
            """
            iv_pct_result = await self.db.execute_query(iv_pct_query, self.analysis_date)
            if iv_pct_result and iv_pct_result[0].get('iv_percentile'):
                result['iv_percentile'] = float(iv_pct_result[0]['iv_percentile'])

        except Exception as e:
            logger.warning(f"Extended indicators query failed: {e}")

        return result

    async def _get_gdp_pmi_data(self) -> Dict:
        """
        GDP 및 PMI 데이터 조회 (5-Macro Environment 분류용)

        Returns:
            {
                'gdp_growth': float (최근 분기 GDP 성장률 %),
                'gdp_trend': str ('POSITIVE' | 'NEGATIVE' | 'NEUTRAL'),
                'pmi_value': float (최근 PMI 지수),
                'pmi_signal': str ('EXPANSION' | 'CONTRACTION'),
                'cpi_trend': str ('RISING' | 'FALLING' | 'STABLE')
            }
        """
        result = {
            'gdp_growth': None,
            'gdp_trend': 'NEUTRAL',
            'pmi_value': None,
            'pmi_signal': 'NEUTRAL',
            'cpi_trend': 'STABLE'
        }

        try:
            # GDP 성장률 조회 (최근 2개 분기)
            gdp_query = """
            SELECT date, value
            FROM us_gdp
            WHERE ($1::DATE IS NULL OR date <= $1::DATE)
            ORDER BY date DESC
            LIMIT 2
            """
            gdp_result = await self.db.execute_query(gdp_query, self.analysis_date)

            if gdp_result and len(gdp_result) >= 1:
                current_gdp = float(gdp_result[0]['value'])
                result['gdp_growth'] = current_gdp

                if current_gdp > 1.0:
                    result['gdp_trend'] = 'POSITIVE'
                elif current_gdp < 0:
                    result['gdp_trend'] = 'NEGATIVE'
                else:
                    result['gdp_trend'] = 'NEUTRAL'

            # PMI 조회 (최근 값)
            pmi_query = """
            SELECT date, value
            FROM us_pmi
            WHERE ($1::DATE IS NULL OR date <= $1::DATE)
            ORDER BY date DESC
            LIMIT 1
            """
            pmi_result = await self.db.execute_query(pmi_query, self.analysis_date)

            if pmi_result and pmi_result[0].get('value'):
                pmi_value = float(pmi_result[0]['value'])
                result['pmi_value'] = pmi_value
                # IPMAN은 2017=100 기준, 100 이상이면 확장
                result['pmi_signal'] = 'EXPANSION' if pmi_value >= 100 else 'CONTRACTION'

            # CPI Trend 조회 (3개월 전 대비)
            cpi_query = """
            SELECT
                (SELECT value FROM us_cpi
                 WHERE ($1::DATE IS NULL OR date <= $1::DATE)
                 ORDER BY date DESC LIMIT 1) as current_cpi,
                (SELECT value FROM us_cpi
                 WHERE date <= COALESCE($1::DATE, CURRENT_DATE) - INTERVAL '3 months'
                 ORDER BY date DESC LIMIT 1) as cpi_3m_ago
            """
            cpi_result = await self.db.execute_query(cpi_query, self.analysis_date)

            if cpi_result and cpi_result[0]:
                current = cpi_result[0].get('current_cpi')
                ago = cpi_result[0].get('cpi_3m_ago')
                if current and ago:
                    current = float(current)
                    ago = float(ago)
                    cpi_change = (current - ago) / ago * 100 if ago > 0 else 0

                    if cpi_change > 0.5:
                        result['cpi_trend'] = 'RISING'
                    elif cpi_change < -0.3:
                        result['cpi_trend'] = 'FALLING'
                    else:
                        result['cpi_trend'] = 'STABLE'

        except Exception as e:
            logger.warning(f"GDP/PMI data query failed: {e}")

        return result

    def detect_macro_environment(self) -> str:
        """
        5-Macro Environment 분류

        GDP/PMI/CPI 조합으로 환경 결정:
        - SOFT_LANDING: GDP+, PMI 확장, 인플레 하락
        - HARD_LANDING: GDP-, PMI 수축, 인플레 하락
        - REFLATION: GDP+, PMI 확장, 인플레 상승
        - STAGFLATION: GDP-, PMI 수축, 인플레 상승
        - DEFLATION: GDP-, PMI 수축, 인플레 급락

        Returns:
            str: Macro environment name
        """
        gdp_trend = self.indicators.get('gdp_trend', 'NEUTRAL')
        pmi_signal = self.indicators.get('pmi_signal', 'NEUTRAL')
        cpi_trend = self.indicators.get('cpi_trend', 'STABLE')

        # 분류 로직
        gdp_positive = gdp_trend == 'POSITIVE'
        gdp_negative = gdp_trend == 'NEGATIVE'
        pmi_expansion = pmi_signal == 'EXPANSION'
        cpi_rising = cpi_trend == 'RISING'
        cpi_falling = cpi_trend == 'FALLING'

        # 1. STAGFLATION (최악 - 우선 체크)
        if gdp_negative and not pmi_expansion and cpi_rising:
            return 'STAGFLATION'

        # 2. HARD_LANDING
        if gdp_negative and not pmi_expansion and cpi_falling:
            return 'HARD_LANDING'

        # 3. DEFLATION (급격한 물가 하락 + 경기 침체)
        if gdp_negative and not pmi_expansion and cpi_trend == 'STABLE':
            # CPI가 안정이지만 GDP/PMI 모두 부정적
            return 'DEFLATION'

        # 4. SOFT_LANDING (이상적)
        if gdp_positive and pmi_expansion and not cpi_rising:
            return 'SOFT_LANDING'

        # 5. REFLATION
        if gdp_positive and pmi_expansion and cpi_rising:
            return 'REFLATION'

        # 6. 혼합 상황 - 가장 가까운 환경 선택
        if gdp_positive:
            return 'SOFT_LANDING' if cpi_falling else 'REFLATION'
        else:
            return 'HARD_LANDING' if cpi_falling else 'STAGFLATION'

    def get_scenario_probability(self, macro_env: str = None) -> Dict[str, float]:
        """
        Macro Environment → 3-Scenario 확률 변환

        Args:
            macro_env: Macro environment (None이면 현재 감지된 환경 사용)

        Returns:
            {'강세': float, '횡보': float, '약세': float}
        """
        if macro_env is None:
            macro_env = self.detect_macro_environment()

        if macro_env in MACRO_ENVIRONMENT:
            return MACRO_ENVIRONMENT[macro_env]['scenario_prob'].copy()

        # Fallback
        return {'강세': 0.33, '횡보': 0.34, '약세': 0.33}

    def _calculate_composite_scores(self) -> Dict:
        """
        Phase 3.4.2: 4-Dimension Composite Score 계산

        각 Dimension 0-100 점수:
        - 높을수록 해당 차원이 "긍정적" (투자에 유리)
        - Liquidity 높음 = 유동성 풍부
        - Macro 높음 = 거시환경 양호
        - Volatility 높음 = 변동성 낮음 (안정)
        - Risk Appetite 높음 = 위험 선호

        Returns:
            {
                'liquidity_score': float (0-100),
                'macro_score': float (0-100),
                'volatility_score': float (0-100),
                'risk_appetite_score': float (0-100),
                'composite_score': float (0-100)
            }
        """
        # 1. Liquidity Score (유동성)
        liquidity_score = self._calc_liquidity_score()

        # 2. Macro Score (거시경제)
        macro_score = self._calc_macro_score()

        # 3. Volatility Score (변동성 - 역수 개념)
        volatility_score = self._calc_volatility_score()

        # 4. Risk Appetite Score (위험 선호)
        risk_appetite_score = self._calc_risk_appetite_score()

        # Composite (가중 평균)
        # L:20%, M:30%, V:25%, R:25%
        composite = (
            liquidity_score * 0.20 +
            macro_score * 0.30 +
            volatility_score * 0.25 +
            risk_appetite_score * 0.25
        )

        return {
            'liquidity_score': round(liquidity_score, 1),
            'macro_score': round(macro_score, 1),
            'volatility_score': round(volatility_score, 1),
            'risk_appetite_score': round(risk_appetite_score, 1),
            'composite_score': round(composite, 1)
        }

    def _calc_liquidity_score(self) -> float:
        """
        Liquidity Score 계산 (0-100)
        높을수록 유동성 풍부

        지표:
        - Fed RRP: 낮을수록 좋음 (시장에 유동성 공급)
        - Credit Spread: 낮을수록 좋음 (신용 여건 양호)
        - SPY Volume: 높을수록 좋음 (유동성 풍부)
        """
        score = 50.0  # 기본값

        # Fed RRP (역상관 - 낮을수록 좋음)
        fed_rrp = self.indicators.get('fed_rrp')
        if fed_rrp is not None:
            # 0B = 100점, 2000B = 0점 (선형)
            rrp_score = max(0, min(100, 100 - (fed_rrp / 20)))  # 2000B 기준
            score = score * 0.5 + rrp_score * 0.5

        # Credit Spread (역상관 - 낮을수록 좋음)
        credit_spread = self.indicators.get('credit_spread')
        if credit_spread is not None:
            # 1% = 100점, 5% = 0점
            cs_score = max(0, min(100, (5 - credit_spread) / 4 * 100))
            score = score * 0.6 + cs_score * 0.4

        return max(0, min(100, score))

    def _calc_macro_score(self) -> float:
        """
        Macro Score 계산 (0-100)
        높을수록 거시환경 양호

        지표:
        - Fed Rate Change: 인하 시 좋음
        - CPI YoY: 낮을수록 좋음 (2% 목표)
        - Yield Curve: Steepening 좋음
        - MOVE Index: 낮을수록 좋음 (채권 안정)
        - DXY: 적정 수준 선호 (극단 회피)
        """
        components = []

        # Fed Rate Change (인하 = 좋음)
        fed_change = self.indicators.get('fed_rate_change_6m', 0)
        # -0.5 = 100점, +0.5 = 0점
        fed_score = max(0, min(100, 50 - fed_change * 100))
        components.append(fed_score)

        # CPI YoY (2% 목표, 벗어날수록 감점)
        cpi = self.indicators.get('cpi_yoy', 3)
        # 2% = 100점, 6% = 0점
        cpi_score = max(0, min(100, 100 - abs(cpi - 2) * 25))
        components.append(cpi_score)

        # Yield Curve Signal
        yield_signal = self.indicators.get('yield_curve_signal', 'NORMAL')
        if yield_signal == 'STEEPENING':
            yield_score = 80
        elif yield_signal == 'NORMAL':
            yield_score = 50
        else:  # INVERTED
            yield_score = 20
        components.append(yield_score)

        # MOVE Index (채권 변동성 - 낮을수록 좋음)
        move = self.indicators.get('move_index')
        if move is not None:
            # 80 = 100점, 150 = 0점
            move_score = max(0, min(100, (150 - move) / 70 * 100))
            components.append(move_score)

        # DXY (극단 회피 - 100-105 적정)
        dxy = self.indicators.get('dxy')
        if dxy is not None:
            # 100-105 = 80점, 90 or 115 = 40점
            if 100 <= dxy <= 105:
                dxy_score = 80
            elif 95 <= dxy < 100 or 105 < dxy <= 110:
                dxy_score = 60
            else:
                dxy_score = 40
            components.append(dxy_score)

        return sum(components) / len(components) if components else 50.0

    def _calc_volatility_score(self) -> float:
        """
        Volatility Score 계산 (0-100)
        높을수록 변동성 낮음 (안정적)

        지표:
        - VIX: 낮을수록 좋음
        - IV Percentile: 낮을수록 좋음
        - HV 20d: 낮을수록 좋음
        """
        components = []

        # VIX (낮을수록 좋음)
        vix = self.indicators.get('vix_actual') or self.indicators.get('vix_proxy', 20)
        # 12 = 100점, 40 = 0점
        vix_score = max(0, min(100, (40 - vix) / 28 * 100))
        components.append(vix_score)

        # IV Percentile (낮을수록 좋음)
        iv_pct = self.indicators.get('iv_percentile')
        if iv_pct is not None:
            # 0% = 100점, 100% = 0점
            iv_score = 100 - iv_pct
            components.append(iv_score)

        # HV 20d (낮을수록 좋음)
        hv = self.indicators.get('hv_20d')
        if hv is not None:
            # 10 = 100점, 40 = 0점
            hv_score = max(0, min(100, (40 - hv) / 30 * 100))
            components.append(hv_score)

        return sum(components) / len(components) if components else 50.0

    def _calc_risk_appetite_score(self) -> float:
        """
        Risk Appetite Score 계산 (0-100)
        높을수록 위험 선호 (낙관적)

        지표:
        - Put/Call Ratio: 낮을수록 좋음 (콜 선호)
        - IV Skew: 낮을수록 좋음 (풋 프리미엄 낮음)
        - NASDAQ vs SPY: 높을수록 좋음 (성장주 선호)
        """
        components = []

        # Put/Call Ratio (낮을수록 낙관적)
        pcr = self.indicators.get('put_call_ratio', 1.0)
        # 0.5 = 100점, 1.5 = 0점
        pcr_score = max(0, min(100, (1.5 - pcr) / 1.0 * 100))
        components.append(pcr_score)

        # IV Skew (Put IV - Call IV, 낮을수록 좋음)
        iv_skew = self.indicators.get('iv_skew', 0)
        # -5 = 100점, +10 = 0점
        skew_score = max(0, min(100, (10 - iv_skew) / 15 * 100))
        components.append(skew_score)

        # NASDAQ vs SPY 3M (높을수록 성장 선호)
        nasdaq_vs_spy = self.indicators.get('nasdaq_vs_spy_3m', 0)
        # +5% = 80점, 0% = 50점, -5% = 20점
        nq_score = max(0, min(100, 50 + nasdaq_vs_spy * 6))
        components.append(nq_score)

        return sum(components) / len(components) if components else 50.0

    async def save_to_db(self, regime_data: Dict) -> bool:
        """레짐 결과를 DB에 저장"""

        try:
            query = """
            INSERT INTO us_market_regime (
                date, regime,
                vix_proxy, put_call_ratio, iv_skew,
                spy_return_1m, spy_return_3m, spy_ma200_distance, nasdaq_vs_spy_3m,
                fed_rate, fed_rate_change_6m, cpi_yoy, unemployment_rate,
                weight_value, weight_quality, weight_momentum, weight_growth
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17
            )
            ON CONFLICT (date) DO UPDATE SET
                regime = EXCLUDED.regime,
                vix_proxy = EXCLUDED.vix_proxy,
                put_call_ratio = EXCLUDED.put_call_ratio,
                iv_skew = EXCLUDED.iv_skew,
                spy_return_1m = EXCLUDED.spy_return_1m,
                spy_return_3m = EXCLUDED.spy_return_3m,
                spy_ma200_distance = EXCLUDED.spy_ma200_distance,
                nasdaq_vs_spy_3m = EXCLUDED.nasdaq_vs_spy_3m,
                fed_rate = EXCLUDED.fed_rate,
                fed_rate_change_6m = EXCLUDED.fed_rate_change_6m,
                cpi_yoy = EXCLUDED.cpi_yoy,
                unemployment_rate = EXCLUDED.unemployment_rate,
                weight_value = EXCLUDED.weight_value,
                weight_quality = EXCLUDED.weight_quality,
                weight_momentum = EXCLUDED.weight_momentum,
                weight_growth = EXCLUDED.weight_growth
            """

            ind = regime_data['indicators']
            weights = regime_data['weights']
            target_date = self.analysis_date or date.today()

            await self.db.execute(
                query,
                target_date,
                regime_data['regime'],
                ind.get('vix_proxy'),
                ind.get('put_call_ratio'),
                ind.get('iv_skew'),
                ind.get('spy_return_1m'),
                ind.get('spy_return_3m'),
                ind.get('spy_ma200_distance'),
                ind.get('nasdaq_vs_spy_3m'),
                ind.get('fed_rate'),
                ind.get('fed_rate_change_6m'),
                ind.get('cpi_yoy'),
                ind.get('unemployment_rate'),
                weights['value'],
                weights['quality'],
                weights['momentum'],
                weights['growth']
            )

            logger.info(f"Regime saved to DB: {target_date} -> {regime_data['regime']}")
            return True

        except Exception as e:
            logger.error(f"Failed to save regime to DB: {e}")
            return False


async def get_current_regime(db_manager, analysis_date: Optional[date] = None) -> Dict:
    """현재 시장 레짐 조회 (편의 함수)"""

    detector = USMarketRegimeDetector(db_manager, analysis_date)
    return await detector.detect_regime()


# ========================================================================
# Test
# ========================================================================

if __name__ == '__main__':
    import asyncio
    from us_db_async import AsyncDatabaseManager

    async def test():
        db = AsyncDatabaseManager()
        await db.initialize()

        detector = USMarketRegimeDetector(db)
        result = await detector.detect_regime()

        print("\n" + "="*60)
        print("Market Regime Detection Result")
        print("="*60)
        print(f"Regime: {result['regime']}")
        print(f"Description: {result['description']}")
        print(f"\nWeights:")
        for k, v in result['weights'].items():
            print(f"  {k}: {v:.0%}")
        print(f"\nIndicators:")
        for k, v in result['indicators'].items():
            if isinstance(v, float):
                print(f"  {k}: {v:.2f}")
            else:
                print(f"  {k}: {v}")

        await db.close()

    asyncio.run(test())
