"""
US Quant System v2.0 - Main Integration Module

핵심 변경:
1. 시장 레짐 기반 동적 가중치
2. 섹터별 기준값 적용
3. 절대 점수 기반 등급 (섹터 상대 순위 아님)
4. 4개 Factor v2.0 통합

Usage:
    python us_main.py

File: us/us_main.py
"""

import asyncio
import logging
import json
import os
import time
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple
from decimal import Decimal
import numpy as np

# Local imports
from us_db_async import AsyncDatabaseManager
from us_market_regime import USMarketRegimeDetector
from us_sector_benchmarks import USSectorBenchmarks
from us_value_factor import calculate_value_score
from us_quality_factor import calculate_quality_score
from us_momentum_factor import calculate_momentum_score
from us_growth_factor import calculate_growth_score
from us_factor_interactions import USFactorInteractions, apply_pattern_adjustment
from us_agent_metrics import USAgentMetrics

# Phase 3.1 imports
from us_outlier_risk import USOutlierRisk
from us_exchange_optimizer import USExchangeOptimizer
from us_sector_dynamic_weights import USSectorDynamicWeights
from us_healthcare_optimizer import USHealthcareOptimizer
from us_nasdaq_optimizer import USNASDAQOptimizer
from us_event_engine import USEventEngine  # Phase 3.4 Event Engine
from us_volatility_adjustment import USVolatilityAdjustment  # Phase 3.4.2
from us_data_prefetcher import USDataPrefetcher  # Phase 3.4.3 Query Optimization
from us_alternative_matcher import USAlternativeStockMatcher  # Alternative stock matching

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ========================================================================
# Grade Thresholds (Absolute Score Based)
# ========================================================================

# 절대 점수 기준 등급 (섹터 상대 순위 아님!)
# "아프리카 상위 15% 부자 vs 미국 상위 15% 부자" 문제 해결
# 7단계 한글 등급 체계 (stock_agents/us와 통일)
GRADE_THRESHOLDS = {
    '강력 매수': 85,  # 85점 이상 = 최우수
    '매수':     75,  # 75-85점 = 우수
    '매수 고려': 65,  # 65-75점 = 양호
    '중립':     55,  # 55-65점 = 보통 이상
    '매도 고려': 45,  # 45-55점 = 보통
    '매도':     35,  # 35-45점 = 주의
    '강력 매도': 0    # 35점 미만 = 위험
}


class USQuantSystemV2:
    """US Quant System v3.0 (HMM Regime Detection Support)"""

    # 클래스 레벨 HMM 캐시 (모든 인스턴스 공유)
    _hmm_detector_cache = None
    _hmm_cache_initialized = False

    def __init__(self, db_manager: AsyncDatabaseManager,
                 with_event_modifier: bool = True,
                 improved_factors: bool = False):
        """
        Args:
            db_manager: AsyncDatabaseManager instance
            with_event_modifier: apply event_engine total_modifier (option/GEX/
                earnings/insider/news) to the score. Pass-A sets False (ranks
                symbols before options exist); Pass-B sets True (folds in the
                collected option signals).
            improved_factors: reco-only factor improvement (IC 분석 검증).
                현재 적용: value 게이팅 — value_score 는 소형/비유동(거래가능)
                에서만 예측력(IC +0.04)이고 대형/유동에선 역전(IC -0.03). 그래서
                일평균 거래대금이 [floor, ceiling] 밖이면 value 가중치를 quality
                로 옮긴다(value 중립화). backtest 경로는 False 로 두어 baseline
                불변. OOS(전체기간) 재검증 전까지 reco 한정.
        """
        self.db = db_manager
        self.with_event_modifier = with_event_modifier
        self.improved_factors = improved_factors
        # value 게이팅 유동성 경계 (일평균 거래대금, USD). env 로 조정 가능.
        # ※ 단일 레짐(COVID) 분석 기반 — OOS 재검증 후 재튜닝 필요.
        self._imp_liq_floor = float(os.getenv("IMP_LIQ_FLOOR_USD", "5000000"))    # 거래가능 하한
        self._imp_liq_ceil = float(os.getenv("IMP_LIQ_CEIL_USD", "500000000"))   # 대형/유동 상한
        self.sector_benchmarks = USSectorBenchmarks(db_manager)

        # Phase 3.1 modules
        self.outlier_risk = USOutlierRisk()
        self.exchange_optimizer = USExchangeOptimizer()
        self.sector_weights = USSectorDynamicWeights()
        self.healthcare_optimizer = USHealthcareOptimizer()
        self.nasdaq_optimizer = USNASDAQOptimizer()
        self.event_engine = USEventEngine(db_manager)  # Phase 3.4 Event Engine

        # HMM 전용 모드 (v3.0: 규칙 기반 제거)
        self.use_hmm = True
        self.use_static_weights = True

        # Cache
        self.current_regime = None
        self.regime_weights = None
        self.regime_method = None  # 'HMM' or 'RULE_BASED'
        self.sector_cache = {}

    async def _init_hmm_detector(self):
        """
        HMM Detector 초기화 (시스템 레벨 캐싱)
        - 한 번만 학습하고 재사용
        - 모든 날짜에 동일한 학습된 모델 사용
        """
        if not USQuantSystemV2._hmm_cache_initialized:
            from us_market_regime import HMMRegimeDetector
            if True:  # HMM 전용 모드
                logger.info("Initializing HMM Detector (one-time training)...")
                USQuantSystemV2._hmm_detector_cache = HMMRegimeDetector(n_regimes=3)
                await USQuantSystemV2._hmm_detector_cache.fit(self.db, lookback_days=504)
                USQuantSystemV2._hmm_cache_initialized = True
                logger.info("HMM Detector ready (cached for all dates)")

    async def run(self, analysis_date: Optional[date] = None,
                  symbols: Optional[List[str]] = None) -> Dict:
        """
        전체 분석 실행

        Args:
            analysis_date: 분석 날짜 (기본: 오늘)
            symbols: 분석할 종목 리스트 (기본: 전체)

        Returns:
            {
                'date': date,
                'regime': str,
                'total_processed': int,
                'grade_distribution': Dict,
                'top_stocks': List
            }
        """
        analysis_date = analysis_date or date.today()
        logger.info(f"Starting US Quant System v2.0 - Date: {analysis_date}")

        # 1. 시장 레짐 감지
        logger.info("Step 1: Detecting market regime...")
        await self._detect_regime(analysis_date)
        logger.info(f"  Regime: {self.current_regime}")
        logger.info(f"  Weights: {self.regime_weights}")

        # 2. 분석 대상 종목 로드 — analysis_date 기준 PIT universe
        if symbols is None:
            symbols = await self._load_analysis_symbols(analysis_date)
        logger.info(f"Step 2: Loaded {len(symbols)} symbols for analysis (PIT @ {analysis_date})")

        # 3. 각 종목 분석
        logger.info("Step 3: Analyzing stocks...")
        results = []
        processed = 0
        errors = 0

        for symbol in symbols:
            try:
                result = await self._analyze_stock(symbol, analysis_date)
                if result:
                    results.append(result)
                    processed += 1

                    if processed % 100 == 0:
                        logger.info(f"  Processed {processed}/{len(symbols)} stocks")

            except Exception as e:
                logger.error(f"  {symbol}: Analysis failed - {e}")
                errors += 1

        logger.info(f"  Total: {processed} processed, {errors} errors")

        # 4. 결과 저장
        logger.info("Step 4: Saving results...")
        await self._save_results(results, analysis_date)

        # 5. 통계 생성
        grade_dist = self._calculate_grade_distribution(results)
        top_stocks = sorted(results, key=lambda x: x['total_score'], reverse=True)[:20]

        logger.info("="*60)
        logger.info("US Quant System v2.0 Complete")
        logger.info(f"  Date: {analysis_date}")
        logger.info(f"  Regime: {self.current_regime}")
        logger.info(f"  Total Processed: {processed}")
        logger.info(f"  Grade Distribution: {grade_dist}")
        logger.info("="*60)

        return {
            'date': analysis_date,
            'regime': self.current_regime,
            'total_processed': processed,
            'grade_distribution': grade_dist,
            'top_stocks': [{'symbol': s['symbol'], 'score': s['total_score'], 'grade': s['grade']}
                           for s in top_stocks]
        }

    async def _detect_regime(self, analysis_date: date):
        """
        시장 레짐 감지 및 가중치 설정 (v3.0: HMM 전용)

        - HMM 기반 3-State 레짐 감지 (BULL/NEUTRAL/BEAR)
        - 정적 가중치 사용 (Factor Timing 최소화)

        효율성 최적화:
        - HMM 모델은 시스템 레벨에서 한 번만 학습
        - 여러 날짜 분석 시 동일한 학습된 모델 재사용
        """
        # HMM 캐시 미초기화 시 초기화
        if not USQuantSystemV2._hmm_cache_initialized:
            await self._init_hmm_detector()

        # HMM 모드: 캐시된 HMM Detector 사용
        regime_result = await self._detect_regime_with_cached_hmm(analysis_date)

        self.current_regime = regime_result['regime']
        self.regime_weights = regime_result['weights']
        self.regime_data = regime_result
        self.macro_environment = regime_result.get('macro_environment', 'UNKNOWN')
        self.scenario_probability = regime_result.get('scenario_probability', {})
        self.regime_method = regime_result.get('method', 'UNKNOWN')

    async def _detect_regime_with_cached_hmm(self, analysis_date: date) -> Dict:
        """
        캐시된 HMM Detector를 사용한 레짐 감지 (효율성 최적화)
        - HMM 재학습 없이 예측만 수행
        - 기존 지표 조회는 별도 수행
        """
        from us_market_regime import REGIME_WEIGHTS, MACRO_ENVIRONMENT

        # 캐시된 HMM으로 레짐 예측
        hmm_result = await USQuantSystemV2._hmm_detector_cache.predict_current_regime(
            self.db, analysis_date
        )

        regime = hmm_result['regime']
        weights = hmm_result['weights']

        # Macro Environment 매핑
        macro_env_map = {
            'BULL': 'SOFT_LANDING',
            'NEUTRAL': 'REFLATION',
            'BEAR': 'HARD_LANDING'
        }
        macro_env = macro_env_map.get(regime, 'SOFT_LANDING')

        # Scenario Probability
        transition_probs = hmm_result.get('transition_probs', {})
        scenario_prob = {
            '강세': transition_probs.get('BULL', 0.33),
            '횡보': transition_probs.get('NEUTRAL', 0.34),
            '약세': transition_probs.get('BEAR', 0.33)
        }

        return {
            'regime': regime,
            'weights': weights,
            'indicators': {},
            'description': REGIME_WEIGHTS[regime]['description'],
            'composite_scores': {
                'liquidity_score': 50.0,
                'macro_score': 50.0,
                'volatility_score': 50.0,
                'risk_appetite_score': 50.0,
                'composite_score': 50.0
            },
            'macro_environment': macro_env,
            'macro_env_description': MACRO_ENVIRONMENT[macro_env]['description'],
            'scenario_probability': scenario_prob,
            'method': 'HMM_CACHED',
            'hmm_regime_id': hmm_result.get('regime_id', -1),
            'hmm_transition_probs': transition_probs
        }

    async def _load_analysis_symbols(self, analysis_date: Optional[date] = None) -> List[str]:
        """분석 대상 종목 로드 — Point-in-Time universe (survivorship-bias 제거).

        analysis_date 가 주어지면 us_listing_status 와 JOIN 해서:
          - ipo_date <= analysis_date (그 시점에 이미 상장)
          - delisting_date IS NULL OR delisting_date > analysis_date (그 시점에 아직 상장 중)
          - asset_type = 'Stock' (ETF/SPAC 등 제외)
        조건을 만족하는 종목만 반환. us_stock_basic 의 sector 정보는 LEFT JOIN
        하되 sector NULL 인 delisted 종목도 universe 에 포함 (단, 일부 sector
        기반 점수는 계산 안 됨).

        analysis_date=None 인 경우 (legacy) → 현재 sector 가 있는 모든 종목.

        Note: us_listing_status 가 비어있으면 (data 수집 전) 자동으로 legacy
        쿼리로 폴백 — orchestrator 첫 run 깨지지 않도록.
        """
        if analysis_date is None:
            query = """
            SELECT DISTINCT symbol FROM us_stock_basic
            WHERE sector IS NOT NULL AND sector != ''
            ORDER BY symbol
            """
            result = await self.db.execute_query(query)
            return [r['symbol'] for r in result] if result else []

        # us_listing_status 존재 + 데이터 유무 체크 (graceful fallback)
        has_ls = await self.db.execute_query("""
            SELECT EXISTS (SELECT 1 FROM information_schema.tables
                           WHERE table_name='us_listing_status') AS exists
        """)
        ls_exists = bool(has_ls and has_ls[0]['exists'])
        if ls_exists:
            row_count = await self.db.execute_query("SELECT COUNT(*) AS n FROM us_listing_status")
            ls_exists = ls_exists and (row_count and row_count[0]['n'] > 0)

        if not ls_exists:
            # 폴백: us_stock_basic 기반 (survivorship bias 있음 — 경고 로그)
            logger.warning(
                f"_load_analysis_symbols: us_listing_status 비어있음 → "
                f"survivorship-biased universe 사용 ({analysis_date})")
            query = """
            SELECT DISTINCT symbol FROM us_stock_basic
            WHERE sector IS NOT NULL AND sector != '' AND date <= $1
            ORDER BY symbol
            """
            result = await self.db.execute_query(query, analysis_date)
            return [r['symbol'] for r in result] if result else []

        # PIT universe — us_listing_status 기준
        pit_query = """
        SELECT DISTINCT ls.symbol
        FROM us_listing_status ls
        WHERE (ls.ipo_date IS NULL OR ls.ipo_date <= $1)
          AND (ls.delisting_date IS NULL OR ls.delisting_date > $1)
          AND (ls.asset_type IS NULL OR ls.asset_type = 'Stock')
        ORDER BY ls.symbol
        """
        result = await self.db.execute_query(pit_query, analysis_date)
        return [r['symbol'] for r in result] if result else []

    async def _get_sector_benchmarks(self, sector: str,
                                       analysis_date: Optional[date] = None) -> Dict:
        """섹터 벤치마크 로드 (point-in-time + 캐시).

        analysis_date 를 USSectorBenchmarks 로 전달해서 us_stock_basic 의 2.85M
        row 전체가 아닌 그 시점의 ~6k row 만 PERCENTILE_CONT 집계하도록 한다.
        캐시 key 는 (sector, analysis_date) 로 분리.
        """
        cache_key = (sector, analysis_date)
        if cache_key not in self.sector_cache:
            benchmarks = await self.sector_benchmarks.get_sector_benchmarks(
                sector, analysis_date=analysis_date)
            self.sector_cache[cache_key] = benchmarks
        return self.sector_cache[cache_key]

    async def _analyze_stock(self, symbol: str, analysis_date: date,
                              spy_returns: Optional[List[float]] = None) -> Optional[Dict]:
        """개별 종목 분석 (Phase 3.1 통합, Phase 3.4.3 쿼리 최적화, Phase 4 리스크 지표)"""

        # ============================================================
        # Phase 3.4.3: Prefetch all common data (5 parallel queries)
        # Reduces DB queries from ~35/stock to ~20/stock (43% reduction)
        # ============================================================
        prefetcher = USDataPrefetcher(self.db)
        prefetched = await prefetcher.prefetch_all(symbol, analysis_date)

        # Validate required data
        stock_basic = prefetched.get('stock_basic')
        price_data = prefetched.get('price_data')

        if not stock_basic:
            return None

        sector = stock_basic.get('sector')
        exchange = stock_basic.get('exchange', 'NYSE')
        if not sector:
            return None

        # Build stock_info from prefetched data
        stock_info = {
            'symbol': symbol,
            'stock_name': stock_basic.get('stock_name'),
            'sector': sector,
            'industry': stock_basic.get('industry'),
            'exchange': exchange,
            'market_cap': stock_basic.get('market_cap'),
            'beta': stock_basic.get('beta')
        }

        # 섹터 벤치마크 로드 (point-in-time)
        benchmarks = await self._get_sector_benchmarks(sector, analysis_date=analysis_date)

        # Phase 3.4.3: Create volatility engine from prefetched price data
        volatility_engine = self._create_volatility_engine_from_data(price_data)

        # ============================================================
        # Phase 3.4.3: Calculate factors with prefetched data
        # Factor modules will use prefetched_data instead of DB queries
        # ============================================================
        value_task = calculate_value_score(
            symbol, self.db, benchmarks, analysis_date,
            volatility_engine, prefetched_data=prefetched
        )
        quality_task = calculate_quality_score(
            symbol, self.db, benchmarks, analysis_date,
            volatility_engine, prefetched_data=prefetched
        )
        momentum_task = calculate_momentum_score(
            symbol, self.db, benchmarks, analysis_date, exchange,
            volatility_engine, prefetched_data=prefetched
        )
        growth_task = calculate_growth_score(
            symbol, self.db, benchmarks, analysis_date, exchange,
            volatility_engine, prefetched_data=prefetched
        )

        # Phase 3.4.3: Use _from_data methods for risk calculations
        var_cvar_result = self._calculate_var_cvar_from_data(price_data)
        rs_value = self._calculate_rs_value_from_data(price_data)
        ors_result = self._calculate_ors_from_data(price_data, stock_basic.get('market_cap'))
        risk_metrics = self._calculate_risk_metrics_from_data(price_data)

        # Phase 4: Calculate tail_beta and corr_spy using SPY returns
        tail_beta = None
        corr_spy = None
        if spy_returns:
            stock_returns = self._calculate_stock_returns_from_data(price_data)
            if stock_returns:
                tail_beta = self._calculate_tail_beta(stock_returns, spy_returns)
                corr_spy = self._calculate_corr_spy(stock_returns, spy_returns)

        # FAST_BACKFILL_MODE skips the pure-metadata calculators
        # (agent_metrics / factor_momentum / sector_metrics — they don't feed
        # final_score) for bulk historical backfill speed.
        _FAST = os.getenv("FAST_BACKFILL_MODE", "0").lower() in ("1", "true", "yes")

        # event_engine total_modifier carries the option signals
        # (options_modifier = PCR/IV skew, gex_modifier = gamma exposure from
        # us_option_daily_summary). Applied only when with_event_modifier is set
        # (Pass-B). Pass-A runs BEFORE options are collected and turns it off —
        # it only ranks symbols to pick which ones get options, so the event
        # modifier would be noise (stale/empty options) and pure cost.
        if self.with_event_modifier:
            event_task = self.event_engine.calculate_event_modifier(symbol, analysis_date)
            value_result, quality_result, momentum_result, growth_result, event_result = await asyncio.gather(
                value_task, quality_task, momentum_task, growth_task, event_task
            )
        else:
            event_result = {'total_modifier': 0, 'reason': 'events disabled (Pass-A)'}
            value_result, quality_result, momentum_result, growth_result = await asyncio.gather(
                value_task, quality_task, momentum_task, growth_task
            )

        # 개별 점수
        value_score = value_result['value_score']
        quality_score = quality_result['quality_score']
        momentum_score = momentum_result['momentum_score']
        growth_score = growth_result['growth_score']

        # Phase 3.1: ORS 결과
        outlier_risk_score = ors_result.get('ors', 0)
        risk_flag = ors_result.get('risk_flag', 'NORMAL')

        # Phase 3.3: Healthcare 최적화 (Growth-centric + Sub-sector differentiation)
        # Phase 3.2: Cash Runway 결과 초기화
        subsector = None
        cash_runway_result = None

        if sector == 'Healthcare':
            # Healthcare 서브섹터 분류
            subsector = await self.healthcare_optimizer.classify_healthcare_subsector(
                stock_info, symbol, self.db, analysis_date
            )

            # Healthcare 전용 가중치 적용 (0-1 범위 -> 0-100 스케일 변환)
            healthcare_weights = self.healthcare_optimizer.get_healthcare_weights(subsector)
            # healthcare_weights = {'growth': 0.60, 'quality': 0.25, ...} sum = 1.00
            adjusted_weights = {
                'growth': healthcare_weights['growth'] * 100,
                'momentum': healthcare_weights['momentum'] * 100,
                'quality': healthcare_weights['quality'] * 100,
                'value': healthcare_weights['value'] * 100
            }  # Sum = 100 (100-point scale maintained)

            # Validation: Weight sum must be 100
            weight_sum = sum(adjusted_weights.values())
            if abs(weight_sum - 100.0) > 0.01:
                logger.warning(
                    f"[Healthcare] Weight sum != 100: {weight_sum:.2f} for {symbol} "
                    f"(subsector: {subsector})"
                )

            # Phase 3.2: Cash Runway 평가 (Healthcare 섹터만)
            cash_runway_result = await self.healthcare_optimizer.evaluate_cash_runway_risk(
                symbol, self.db, analysis_date, subsector
            )

        # Phase 3.3: NASDAQ 최적화 (Momentum 축소 + Quality 강화 + 섹터별 차별화)
        elif exchange == 'NASDAQ':
            # NASDAQ 섹터별 가중치 적용
            # Technology/Healthcare는 Momentum 10%, 나머지는 기본 가중치
            nasdaq_weights = self.nasdaq_optimizer.get_nasdaq_sector_weights(sector)
            # nasdaq_weights = {'growth': 0.50, 'quality': 0.25, ...} sum = 1.00

            adjusted_weights = {
                'growth': nasdaq_weights['growth'] * 100,
                'momentum': nasdaq_weights['momentum'] * 100,
                'quality': nasdaq_weights['quality'] * 100,
                'value': nasdaq_weights['value'] * 100
            }  # Sum = 100 (100-point scale maintained)

            # Validation: Weight sum must be 100
            weight_sum = sum(adjusted_weights.values())
            if abs(weight_sum - 100.0) > 0.01:
                logger.warning(
                    f"[NASDAQ] Weight sum != 100: {weight_sum:.2f} for {symbol} "
                    f"(sector: {sector})"
                )

        else:
            # NYSE/기타 거래소: Phase 3.1 기존 로직
            base_weights = {
                'growth': self.regime_weights['growth'] * 100,
                'momentum': self.regime_weights['momentum'] * 100,
                'quality': self.regime_weights['quality'] * 100,
                'value': self.regime_weights['value'] * 100
            }
            adjusted_weights = self.sector_weights.get_combined_weights(base_weights, exchange, sector)

        # Base score (Phase 3.1: 조정된 가중치 사용)
        w_quality = adjusted_weights['quality']
        w_value = adjusted_weights['value']
        # improved_factors(reco): value 게이팅 — 대형/유동·거래불가 종목은 value
        # 기여를 quality 로 이전(value 중립화). IC 분석: value 는 소형/비유동에서만
        # 예측력(+), 대형/유동에서 역전(-). backtest(improved_factors=False)는 불변.
        if self.improved_factors and self._value_gated(price_data):
            w_quality = w_quality + w_value
            w_value = 0.0
            logger.debug(f"[{symbol}] improved: value gated → quality "
                         f"(w_quality={w_quality:.1f})")
        base_score = (
            growth_score * adjusted_weights['growth'] / 100 +
            momentum_score * adjusted_weights['momentum'] / 100 +
            quality_score * w_quality / 100 +
            value_score * w_value / 100
        )

        # Factor Interaction 계산 (Phase 3)
        interaction_calc = USFactorInteractions({
            'value_score': value_score,
            'quality_score': quality_score,
            'momentum_score': momentum_score,
            'growth_score': growth_score
        })
        interaction_result = interaction_calc.calculate()
        interaction_score = interaction_result['interaction_score']
        conviction_score = interaction_result['conviction_score']

        # Final score = base 70% + interaction 30%
        total_score = base_score * 0.70 + interaction_score * 0.30

        # Phase 3.2: Cash Runway 패널티 적용 (Healthcare 섹터만)
        cash_runway_penalty = 0.0
        if cash_runway_result and cash_runway_result.get('penalty', 0) > 0:
            cash_runway_penalty = cash_runway_result['penalty']
            total_score = max(0, total_score - cash_runway_penalty)  # 최소 0점
            logger.info(
                f"[{symbol}] Cash Runway penalty applied: -{cash_runway_penalty:.1f} "
                f"({cash_runway_result.get('reason', 'N/A')})"
            )

        # Phase 3.4: Event Modifier 적용
        event_modifier = event_result.get('total_modifier', 0)
        if event_modifier != 0:
            total_score = max(0, min(100, total_score + event_modifier))  # 0-100 클램핑
            logger.debug(
                f"[{symbol}] Event modifier applied: {event_modifier:+.1f} "
                f"({event_result.get('reason', 'N/A')})"
            )

        # Phase 3.5: Pattern-based score adjustment (V-Q+M-G+, V+Q+M+G+, V-Q-M-G-)
        market_cap = stock_basic.get('market_cap') or 0
        total_score, pattern_applied = apply_pattern_adjustment(
            total_score, value_score, quality_score, momentum_score, growth_score, market_cap
        )
        if pattern_applied:
            logger.info(f"[{symbol}] Pattern adjustment applied: {pattern_applied}")

        # 등급 결정 (절대 점수 기반, 7단계 한글)
        grade = self._determine_grade(total_score)
        opinion = grade  # grade 자체가 한글 투자의견

        # Phase 3.4.3: confidence_score (factor alignment - lower std = higher confidence)
        factor_scores = [value_score, quality_score, momentum_score, growth_score]
        factor_std = np.std(factor_scores)
        confidence_score = round(max(0, 100 - (factor_std / 35 * 100)), 1)

        # Phase 3.4.3: factor_combination_bonus
        factor_combination_bonus = self._calculate_factor_combination_bonus(
            value_score, quality_score, momentum_score, growth_score
        )

        # Agent Metrics 계산 (Phase 4) — skipped in FAST_BACKFILL_MODE
        if _FAST:
            agent_result = {}
            factor_momentum_result = {}
        else:
            agent_metrics = USAgentMetrics(self.db)
            agent_result = await agent_metrics.calculate_all(
                symbol=symbol,
                analysis_date=analysis_date,
                current_score=total_score,
                sector=sector,
                var_95=var_cvar_result.get('var_95'),
                sector_percentile=None
            )

            # Phase 3.4.3 Stage 3: Factor Momentum (4 columns)
            factor_momentum_result = await self._calculate_factor_momentum(
                symbol, analysis_date,
                {
                    'value': value_score,
                    'quality': quality_score,
                    'momentum': momentum_score,
                    'growth': growth_score
                }
            )

        # Phase 3.4.3 Stage 4: Sector Metrics (4 columns) — skipped in FAST mode
        sector_metrics_result = {} if _FAST else await self._calculate_sector_metrics(sector, analysis_date)

        # Phase 3.4.3 Stage 4: Text Generation (5 columns)
        text_metrics = {
            'volatility_annual': risk_metrics.get('volatility_annual'),
            'max_drawdown_1y': risk_metrics.get('max_drawdown_1y'),
            'beta': float(stock_info['beta']) if stock_info.get('beta') else None,
            'total_score': total_score,
            'score_trend_2w': agent_result.get('score_trend_2w'),
            'regime': self.current_regime
        }
        text_result = self._generate_analysis_texts(text_metrics)

        return {
            'symbol': symbol,
            'sector': sector,
            'exchange': exchange,  # Phase 3.1
            # Phase 3.4.3: stock_name, beta, industry, confidence_score, factor_combination_bonus
            'stock_name': stock_info.get('stock_name'),
            'beta': float(stock_info['beta']) if stock_info.get('beta') else None,
            'industry': stock_info.get('industry'),
            'confidence_score': confidence_score,
            'factor_combination_bonus': factor_combination_bonus,
            'total_score': round(total_score, 1),
            'base_score': round(base_score, 1),
            'interaction_score': round(interaction_score, 1),
            'conviction_score': round(conviction_score, 1),
            'grade': grade,
            'opinion': opinion,
            'value_score': value_score,
            'quality_score': quality_score,
            'momentum_score': momentum_score,
            'growth_score': growth_score,
            'value_detail': value_result.get('strategies', {}),
            'quality_detail': quality_result.get('strategies', {}),
            'momentum_detail': momentum_result.get('strategies', {}),
            'growth_detail': growth_result.get('strategies', {}),
            'regime': self.current_regime,
            'regime_weights': self.regime_weights,
            'var_95': var_cvar_result['var_95'],
            'cvar_95': var_cvar_result['cvar_95'],
            # Phase 4.1: Hurst Exponent and Period-based VaR
            'hurst_exponent': var_cvar_result.get('hurst_exponent'),
            'var_95_5d': var_cvar_result.get('var_95_5d'),
            'var_95_20d': var_cvar_result.get('var_95_20d'),
            'var_95_60d': var_cvar_result.get('var_95_60d'),
            'var_95_90d': var_cvar_result.get('var_95_90d'),
            'var_99': var_cvar_result.get('var_99'),
            'var_99_90d': var_cvar_result.get('var_99_90d'),
            'rs_value': rs_value,
            # Phase 3.1: ORS and Adjusted Weights
            'outlier_risk_score': outlier_risk_score,
            'risk_flag': risk_flag,
            'weight_growth': round(adjusted_weights['growth'], 1),
            'weight_momentum': round(adjusted_weights['momentum'], 1),
            'weight_quality': round(adjusted_weights['quality'], 1),
            'weight_value': round(adjusted_weights['value'], 1),
            # Agent Metrics (Phase 4)
            'atr_pct': agent_result.get('atr_pct'),
            'stop_loss_pct': agent_result.get('stop_loss_pct'),
            'take_profit_pct': agent_result.get('take_profit_pct'),
            'risk_reward_ratio': agent_result.get('risk_reward_ratio'),
            'entry_timing_score': agent_result.get('entry_timing_score'),
            'score_trend_2w': agent_result.get('score_trend_2w'),
            'price_position_52w': agent_result.get('price_position_52w'),
            'position_size_pct': agent_result.get('position_size_pct'),
            'iv_percentile': agent_result.get('iv_percentile'),
            'insider_signal': agent_result.get('insider_signal'),
            'buy_triggers': agent_result.get('buy_triggers'),
            'sell_triggers': agent_result.get('sell_triggers'),
            'hold_triggers': agent_result.get('hold_triggers'),
            'scenario_bullish_prob': agent_result.get('scenario_bullish_prob'),
            'scenario_sideways_prob': agent_result.get('scenario_sideways_prob'),
            'scenario_bearish_prob': agent_result.get('scenario_bearish_prob'),
            'scenario_bullish_return': agent_result.get('scenario_bullish_return'),
            'scenario_sideways_return': agent_result.get('scenario_sideways_return'),
            'scenario_bearish_return': agent_result.get('scenario_bearish_return'),
            'scenario_sample_count': agent_result.get('scenario_sample_count'),
            # Phase 3.2: Cash Runway (Healthcare/Biotech only)
            'healthcare_subsector': subsector,
            'cash_runway_months': cash_runway_result.get('runway_months') if cash_runway_result else None,
            'cash_runway_risk': cash_runway_result.get('risk_level') if cash_runway_result else None,
            'cash_runway_penalty': cash_runway_penalty,
            'is_cash_burning': cash_runway_result.get('is_burning') if cash_runway_result else None,
            # Phase 3.4: Event Engine
            'event_modifier': event_modifier,
            'earnings_modifier': event_result.get('earnings_modifier', 0),
            'options_modifier': event_result.get('options_modifier', 0),
            'insider_modifier': event_result.get('insider_modifier', 0),
            'event_reason': event_result.get('reason', 'N/A'),
            # Phase 3.4.3: Risk Metrics from prefetched price data
            'volatility_annual': risk_metrics.get('volatility_annual'),
            'max_drawdown_1y': risk_metrics.get('max_drawdown_1y'),
            'sharpe_ratio': risk_metrics.get('sharpe_ratio'),
            'sortino_ratio': risk_metrics.get('sortino_ratio'),
            'calmar_ratio': risk_metrics.get('calmar_ratio'),
            # Phase 4: Additional Risk Metrics (9 columns)
            'downside_vol': risk_metrics.get('downside_vol'),
            'tail_beta': tail_beta,
            'corr_spy': corr_spy,
            'cvar_99': risk_metrics.get('cvar_99'),
            'var_95_ewma': risk_metrics.get('var_95_ewma'),
            'inv_vol_weight': risk_metrics.get('inv_vol_weight'),
            'vol_percentile': None,  # Calculated after batch processing
            'corr_sector_avg': None,  # Calculated after batch processing
            'drawdown_duration_avg': risk_metrics.get('drawdown_duration_avg'),
            # Phase 3.4.3 Stage 3: Factor Momentum (4 columns)
            'value_momentum': factor_momentum_result.get('value_momentum'),
            'quality_momentum': factor_momentum_result.get('quality_momentum'),
            'momentum_momentum': factor_momentum_result.get('momentum_momentum'),
            'growth_momentum': factor_momentum_result.get('growth_momentum'),
            # Phase 3.4.3 Stage 4: Sector Metrics (4 columns)
            'sector_rotation_score': sector_metrics_result.get('sector_rotation_score'),
            'sector_momentum': sector_metrics_result.get('sector_momentum'),
            'sector_rank': sector_metrics_result.get('sector_rank'),
            'sector_percentile': sector_metrics_result.get('sector_percentile'),
            # Phase 3.4.3 Stage 4: Text Generation (5 columns)
            'risk_profile_text': text_result.get('risk_profile_text'),
            'risk_recommendation': text_result.get('risk_recommendation'),
            'time_series_text': text_result.get('time_series_text'),
            'signal_overall': text_result.get('signal_overall'),
            'market_state': text_result.get('market_state')
        }

    async def _get_stock_info(self, symbol: str, analysis_date: Optional[date] = None) -> Optional[Dict]:
        """종목 기본 정보 조회 (point-in-time, look-ahead bias 제거).

        analysis_date 가 주어지면 그 시점에 유효했던 row 만 반환.
        ``date <= analysis_date`` 가 source='api' (최신 snapshot) 를 자연스럽게
        제외하고 가장 최근 source='computed' 행을 선택.
        """
        if analysis_date is not None:
            query = """
            SELECT symbol, sector, market_cap, exchange, stock_name, beta, industry
            FROM us_stock_basic
            WHERE symbol = $1 AND date <= $2
            ORDER BY date DESC
            LIMIT 1
            """
            result = await self.db.execute_query(query, symbol, analysis_date)
        else:
            query = """
            SELECT symbol, sector, market_cap, exchange, stock_name, beta, industry
            FROM us_stock_basic
            WHERE symbol = $1
            ORDER BY date DESC
            LIMIT 1
            """
            result = await self.db.execute_query(query, symbol)
        return dict(result[0]) if result else None

    async def _create_volatility_engine(self, symbol: str, analysis_date: date) -> Optional[USVolatilityAdjustment]:
        """
        Phase 3.4.2: Create shared volatility engine for all factors

        Loads price data once and creates a USVolatilityAdjustment instance
        that can be shared across all factor calculations.
        """
        try:
            # Load price data for HV calculation
            query = """
            SELECT date, close
            FROM us_daily
            WHERE symbol = $1
              AND ($2::DATE IS NULL OR date <= $2::DATE)
            ORDER BY date DESC
            LIMIT 30
            """

            result = await self.db.execute_query(query, symbol, analysis_date)

            if not result:
                return None

            price_data = [
                {'date': r['date'], 'close': float(r['close']) if r['close'] else None}
                for r in result
            ]

            # Create and initialize volatility engine
            vol_engine = USVolatilityAdjustment(self.db)
            await vol_engine.load_data(symbol, analysis_date, price_data)

            return vol_engine

        except Exception as e:
            logger.warning(f"{symbol}: Failed to create volatility engine - {e}")
            return None

    def _determine_grade(self, score: float) -> str:
        """절대 점수 기반 등급 결정 (7단계 한글)"""

        if score >= GRADE_THRESHOLDS['강력 매수']:
            return '강력 매수'
        elif score >= GRADE_THRESHOLDS['매수']:
            return '매수'
        elif score >= GRADE_THRESHOLDS['매수 고려']:
            return '매수 고려'
        elif score >= GRADE_THRESHOLDS['중립']:
            return '중립'
        elif score >= GRADE_THRESHOLDS['매도 고려']:
            return '매도 고려'
        elif score >= GRADE_THRESHOLDS['매도']:
            return '매도'
        else:
            return '강력 매도'

    def _calculate_factor_combination_bonus(self, value: float, quality: float,
                                             momentum: float, growth: float) -> int:
        """
        Phase 3.4.3: Calculate factor combination bonus/penalty

        Proven factor patterns from academic research:
        - Quality Compounder: High Q + High G = sustained growth
        - Deep Value Turnaround: High V + High Q + Low M = contrarian opportunity
        - GARP: High G + Reasonable V = growth at reasonable price
        - Momentum Speculation: Low V + High M = speculative risk

        Returns:
            int: Bonus/penalty in range [-20, +30]
        """
        bonus = 0

        # Pattern 1: Quality Compounder (Q>=70, G>=70) -> +15
        if quality >= 70 and growth >= 70:
            bonus += 15

        # Pattern 2: Deep Value Turnaround (V>=75, Q>=60, M<40) -> +20
        if value >= 75 and quality >= 60 and momentum < 40:
            bonus += 20

        # Pattern 3: GARP - Growth at Reasonable Price (G>=70, V>=50) -> +10
        if growth >= 70 and value >= 50:
            bonus += 10

        # Pattern 4: Momentum Speculation Penalty (V<30, M>=80) -> -10
        if value < 30 and momentum >= 80:
            bonus -= 10

        # Pattern 5: All-around Strong (all >= 65) -> +15
        if all(s >= 65 for s in [value, quality, momentum, growth]):
            bonus += 15

        # Clamp to [-20, +30]
        return max(-20, min(30, bonus))

    async def _calculate_var_cvar(self, symbol: str, analysis_date: date) -> Dict:
        """
        Historical VaR 95% and CVaR 95% calculation
        - us_daily: date(DATE), symbol(VARCHAR), close(DECIMAL(21,2))
        - VaR 95%: 5th percentile of daily returns (past 252 days)
        - CVaR 95%: Average of returns below VaR 95%
        """
        query = """
        SELECT close
        FROM us_daily
        WHERE symbol = $1 AND date <= $2
        ORDER BY date DESC
        LIMIT 253
        """
        rows = await self.db.execute_query(query, symbol, analysis_date)

        if not rows or len(rows) < 30:
            return {'var_95': None, 'cvar_95': None}

        # close: DECIMAL(21,2) -> float
        prices = [float(r['close']) for r in rows][::-1]  # Oldest first
        returns = [(prices[i] - prices[i-1]) / prices[i-1] * 100
                   for i in range(1, len(prices))]

        var_95 = float(np.percentile(returns, 5))  # 5th percentile (worst 5%)
        cvar_returns = [r for r in returns if r <= var_95]
        cvar_95 = float(np.mean(cvar_returns)) if cvar_returns else var_95

        return {
            'var_95': round(abs(var_95), 2),   # DECIMAL(6,2) - positive value for loss
            'cvar_95': round(abs(cvar_95), 2)  # DECIMAL(6,2)
        }

    async def _calculate_rs_value(self, symbol: str, analysis_date: date) -> Optional[float]:
        """
        IBD-style Relative Strength Value
        - us_daily: date(DATE), symbol(VARCHAR), close(DECIMAL(21,2))
        - RS = 3M return * 40% + 6M return * 20% + 9M return * 20% + 12M return * 20%
        - Return: rs_value for DECIMAL(21,2)
        """
        query = """
        WITH price_data AS (
            SELECT
                date,
                close,
                LAG(close, 63) OVER (ORDER BY date) as price_3m,
                LAG(close, 126) OVER (ORDER BY date) as price_6m,
                LAG(close, 189) OVER (ORDER BY date) as price_9m,
                LAG(close, 252) OVER (ORDER BY date) as price_12m
            FROM us_daily
            WHERE symbol = $1 AND date <= $2
            ORDER BY date DESC
            LIMIT 1
        )
        SELECT close, price_3m, price_6m, price_9m, price_12m
        FROM price_data
        WHERE price_12m IS NOT NULL
        """
        rows = await self.db.execute_query(query, symbol, analysis_date)

        if not rows:
            return None

        r = rows[0]
        close = float(r['close'])

        ret_3m = (close / float(r['price_3m']) - 1) * 100 if r['price_3m'] else 0
        ret_6m = (close / float(r['price_6m']) - 1) * 100 if r['price_6m'] else 0
        ret_9m = (close / float(r['price_9m']) - 1) * 100 if r['price_9m'] else 0
        ret_12m = (close / float(r['price_12m']) - 1) * 100 if r['price_12m'] else 0

        rs_value = ret_3m * 0.4 + ret_6m * 0.2 + ret_9m * 0.2 + ret_12m * 0.2
        return round(rs_value, 2)  # DECIMAL(21,2)

    async def _calculate_ors(self, symbol: str, analysis_date: date) -> Dict:
        """
        Phase 3.1: Outlier Risk Score (ORS) 계산
        - price, avg_volume, volatility_252d, market_cap 기반
        - 점수 범위: 0-100
        """
        # us_stock_basic is now point-in-time — pick the most recent row
        # with date <= analysis_date (excludes future source='api' snapshots).
        query = """
        SELECT
            b.market_cap,
            d.close as price,
            d.volume
        FROM (
            SELECT market_cap FROM us_stock_basic
            WHERE symbol = $1 AND date <= $2
            ORDER BY date DESC LIMIT 1
        ) b
        LEFT JOIN us_daily d ON d.symbol = $1 AND d.date = $2
        """
        rows = await self.db.execute_query(query, symbol, analysis_date)

        if not rows or not rows[0].get('price'):
            return {'ors': 0, 'risk_flag': 'NORMAL'}

        r = rows[0]
        price = float(r['price']) if r['price'] else 0
        market_cap = float(r['market_cap']) if r['market_cap'] else 0

        # 평균 거래량 계산 (최근 20일)
        vol_query = """
        SELECT AVG(volume) as avg_volume
        FROM (
            SELECT volume
            FROM us_daily
            WHERE symbol = $1 AND date <= $2
            ORDER BY date DESC
            LIMIT 20
        ) sub
        """
        vol_rows = await self.db.execute_query(vol_query, symbol, analysis_date)
        avg_volume = float(vol_rows[0]['avg_volume']) if vol_rows and vol_rows[0].get('avg_volume') else 0

        # 변동성 계산 (연환산)
        vol_price_query = """
        SELECT close
        FROM us_daily
        WHERE symbol = $1 AND date <= $2
        ORDER BY date DESC
        LIMIT 253
        """
        vol_price_rows = await self.db.execute_query(vol_price_query, symbol, analysis_date)

        volatility = 0
        if vol_price_rows and len(vol_price_rows) > 20:
            prices = [float(r['close']) for r in vol_price_rows if r['close']]
            if len(prices) > 1:
                returns = [(prices[i] - prices[i+1]) / prices[i+1] for i in range(len(prices)-1)]
                import statistics
                volatility = statistics.stdev(returns) * (252 ** 0.5) * 100 if returns else 0

        # ORS 계산 (USOutlierRisk 모듈 사용)
        ors_result = self.outlier_risk.calculate_ors_single(
            price=price,
            avg_volume=avg_volume,
            volatility=volatility,
            market_cap=market_cap
        )

        return ors_result

    # ========================================================================
    # Phase 3.4.3: Prefetched Data Methods (Query Optimization)
    # ========================================================================

    def _calculate_var_cvar_from_data(self, price_data: Dict) -> Dict:
        """
        Calculate VaR 95% and CVaR 95% from prefetched price data.
        No DB query required.

        Args:
            price_data: Prefetched price data from USDataPrefetcher

        Returns:
            {'var_95': float, 'cvar_95': float}
        """
        if not price_data or 'closes' not in price_data:
            return {'var_95': None, 'cvar_95': None}

        closes = price_data['closes'][:253]  # Up to 252 trading days
        closes = [c for c in closes if c is not None]

        if len(closes) < 30:
            return {'var_95': None, 'cvar_95': None}

        # Calculate returns (oldest first for consistency)
        prices = closes[::-1]
        returns = [(prices[i] - prices[i - 1]) / prices[i - 1] * 100
                   for i in range(1, len(prices))]

        var_95 = float(np.percentile(returns, 5))  # 5th percentile (worst 5%)
        cvar_returns = [r for r in returns if r <= var_95]
        cvar_95 = float(np.mean(cvar_returns)) if cvar_returns else var_95

        # Phase 4.1: Hurst Exponent (reuse existing returns array)
        hurst = self._calculate_hurst_exponent(returns)

        # Phase 4.1: Period-based VaR with Hurst scaling
        var_1d = abs(var_95)
        var_95_5d = self._scale_var_by_hurst(var_1d, 5, hurst)
        var_95_20d = self._scale_var_by_hurst(var_1d, 20, hurst)
        var_95_60d = self._scale_var_by_hurst(var_1d, 60, hurst)
        var_95_90d = self._scale_var_by_hurst(var_1d, 90, hurst)

        # VaR 99% (reuse existing returns array)
        var_99 = float(np.percentile(returns, 1))  # 1st percentile
        var_99_90d = self._scale_var_by_hurst(abs(var_99), 90, hurst)

        return {
            'var_95': round(abs(var_95), 2),
            'cvar_95': round(abs(cvar_95), 2),
            'hurst_exponent': round(hurst, 4),
            'var_95_5d': round(var_95_5d, 2),
            'var_95_20d': round(var_95_20d, 2),
            'var_95_60d': round(var_95_60d, 2),
            'var_95_90d': round(var_95_90d, 2),
            'var_99': round(abs(var_99), 2),
            'var_99_90d': round(var_99_90d, 2)
        }

    def _calculate_hurst_exponent(self, returns: list, min_chunk: int = 8) -> float:
        """
        R/S (Rescaled Range) Analysis for Hurst Exponent calculation.
        No additional DB query - reuses existing returns array.

        Args:
            returns: Daily return list (already calculated)
            min_chunk: Minimum chunk size

        Returns:
            H: Hurst exponent (0 < H < 1)
            - H > 0.5: Trending (persistence)
            - H = 0.5: Random Walk
            - H < 0.5: Mean Reverting
        """
        N = len(returns)
        if N < min_chunk * 4:
            return 0.5  # Default when insufficient data

        returns_arr = np.array(returns)
        chunk_sizes = []
        rs_values = []

        for chunk_size in range(min_chunk, N // 4):
            rs_list = []

            for start in range(0, N - chunk_size + 1, chunk_size):
                chunk = returns_arr[start:start + chunk_size]

                # Mean adjustment
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

        # Log-log regression for H estimation
        log_n = np.log(chunk_sizes)
        log_rs = np.log(rs_values)

        slope, _ = np.polyfit(log_n, log_rs, 1)

        # Clamp H to valid range (0.1 ~ 0.9)
        H = max(0.1, min(0.9, slope))

        return H

    def _scale_var_by_hurst(self, var_1d: float, target_days: int, hurst: float) -> float:
        """
        Hurst exponent based VaR scaling.
        Pure calculation - no DB query.

        Args:
            var_1d: 1-day VaR (%, positive)
            target_days: Target holding period (5, 20, 60, 90 days)
            hurst: Hurst exponent

        Returns:
            var_Td: T-day VaR (%, positive)
        """
        scale_factor = target_days ** hurst
        var_Td = var_1d * scale_factor
        return var_Td

    def _value_gated(self, price_data) -> bool:
        """value 기여를 중립화할지 판정 (improved_factors 전용).

        일평균 거래대금(최근 20일)이 floor 미만(사실상 거래 불가) 또는 ceiling
        이상(대형/유동 — value 가 역전되는 구간)이면 True. True 면 _analyze_stock
        이 value 가중치를 quality 로 이전해 value 기여를 중립화한다. 데이터
        부족 시 게이팅하지 않음(False, 보수적)."""
        pdct = price_data or {}
        closes = pdct.get('closes') or []
        volumes = pdct.get('volumes') or []
        n = min(len(closes), len(volumes), 20)
        dvs = [closes[i] * volumes[i] for i in range(n)
               if closes[i] is not None and volumes[i] is not None]
        if len(dvs) < 5:
            return False
        adv = sum(dvs) / len(dvs)
        return adv < self._imp_liq_floor or adv >= self._imp_liq_ceil

    def _calculate_rs_value_from_data(self, price_data: Dict) -> Optional[float]:
        """
        Calculate IBD-style Relative Strength Value from prefetched price data.
        RS = 3M return * 40% + 6M return * 20% + 9M return * 20% + 12M return * 20%

        Args:
            price_data: Prefetched price data from USDataPrefetcher

        Returns:
            RS value or None
        """
        if not price_data or 'closes' not in price_data:
            return None

        closes = price_data['closes']
        if not closes or len(closes) < 253:
            return None

        current_price = closes[0]
        if current_price is None:
            return None

        def safe_return(days: int) -> float:
            if len(closes) > days and closes[days] is not None and closes[days] > 0:
                return (current_price / closes[days] - 1) * 100
            return 0

        ret_3m = safe_return(63)
        ret_6m = safe_return(126)
        ret_9m = safe_return(189)
        ret_12m = safe_return(252)

        rs_value = ret_3m * 0.4 + ret_6m * 0.2 + ret_9m * 0.2 + ret_12m * 0.2
        return round(rs_value, 2)

    def _calculate_ors_from_data(self, price_data: Dict, market_cap: float) -> Dict:
        """
        Calculate Outlier Risk Score from prefetched data.
        No DB query required.

        Args:
            price_data: Prefetched price data from USDataPrefetcher
            market_cap: Market cap from stock_basic

        Returns:
            {'ors': float, 'risk_flag': str}
        """
        if not price_data or 'closes' not in price_data:
            return {'ors': 0, 'risk_flag': 'NORMAL'}

        closes = price_data['closes']
        volumes = price_data.get('volumes', [])

        if not closes or closes[0] is None:
            return {'ors': 0, 'risk_flag': 'NORMAL'}

        price = closes[0]

        # Average volume (20 days)
        vol_20d = [v for v in volumes[:20] if v is not None]
        avg_volume = sum(vol_20d) / len(vol_20d) if vol_20d else 0

        # Volatility (annualized, 252 days)
        volatility = 0
        closes_clean = [c for c in closes[:253] if c is not None]
        if len(closes_clean) > 20:
            returns = [(closes_clean[i] - closes_clean[i + 1]) / closes_clean[i + 1]
                       for i in range(len(closes_clean) - 1)]
            if returns:
                import statistics
                volatility = statistics.stdev(returns) * (252 ** 0.5) * 100

        # Use outlier_risk module
        return self.outlier_risk.calculate_ors_single(
            price=price,
            avg_volume=avg_volume,
            volatility=volatility,
            market_cap=float(market_cap) if market_cap else 0
        )

    def _calculate_risk_metrics_from_data(self, price_data: Dict) -> Dict:
        """
        Calculate risk metrics from prefetched price data.
        Phase 3.4.3: volatility_annual, max_drawdown_1y, sharpe/sortino/calmar_ratio
        Phase 4: downside_vol, cvar_99, var_95_ewma, inv_vol_weight, drawdown_duration_avg

        Args:
            price_data: Prefetched price data from USDataPrefetcher

        Returns:
            {
                'volatility_annual': float,  # Annualized volatility (60-day)
                'max_drawdown_1y': float,    # Maximum drawdown (252-day)
                'sharpe_ratio': float,       # Risk-adjusted return
                'sortino_ratio': float,      # Downside risk-adjusted return
                'calmar_ratio': float,       # Return / Max Drawdown
                'downside_vol': float,       # Downside volatility (annualized)
                'cvar_99': float,            # CVaR 99% (Expected Shortfall)
                'var_95_ewma': float,        # EWMA VaR 95%
                'inv_vol_weight': float,     # Inverse volatility weight
                'drawdown_duration_avg': float  # Average drawdown duration (days)
            }
        """
        null_result = {
            'volatility_annual': None,
            'max_drawdown_1y': None,
            'sharpe_ratio': None,
            'sortino_ratio': None,
            'calmar_ratio': None,
            'downside_vol': None,
            'cvar_99': None,
            'var_95_ewma': None,
            'inv_vol_weight': None,
            'drawdown_duration_avg': None
        }

        if not price_data or 'closes' not in price_data:
            return null_result

        closes = price_data['closes']
        closes = [c for c in closes if c is not None]

        if len(closes) < 60:
            return null_result

        # Calculate returns (60 days for volatility, 252 for VaR)
        returns_60d = [(closes[i] - closes[i + 1]) / closes[i + 1]
                       for i in range(min(60, len(closes) - 1))]
        returns_252d = [(closes[i] - closes[i + 1]) / closes[i + 1]
                        for i in range(min(252, len(closes) - 1))]

        if not returns_60d:
            return null_result

        # 1. Volatility (annualized, 60-day)
        volatility_annual = np.std(returns_60d) * np.sqrt(252) * 100

        # 2. Max Drawdown (1 year) with duration tracking
        closes_1y = closes[:min(252, len(closes))][::-1]  # oldest first
        running_max = closes_1y[0]
        max_drawdown = 0
        drawdown_durations = []
        current_dd_start = None

        for i, price in enumerate(closes_1y):
            if price >= running_max:
                running_max = price
                if current_dd_start is not None:
                    drawdown_durations.append(i - current_dd_start)
                    current_dd_start = None
            else:
                if current_dd_start is None:
                    current_dd_start = i
                drawdown = (price - running_max) / running_max * 100
                max_drawdown = min(max_drawdown, drawdown)

        # 3. Risk-Free Rate (approximate 4.5% annual = 0.0178% daily)
        rf_daily = 0.045 / 252

        # 4. Sharpe Ratio (annualized)
        avg_return = np.mean(returns_60d)
        std_return = np.std(returns_60d)
        sharpe_ratio = None
        if std_return > 0:
            sharpe_ratio = (avg_return - rf_daily) / std_return * np.sqrt(252)

        # 5. Sortino Ratio (downside deviation)
        downside_returns = [r for r in returns_60d if r < 0]
        sortino_ratio = None
        downside_vol = None
        if downside_returns:
            downside_std = np.std(downside_returns)
            # Phase 4: Downside volatility (annualized)
            downside_vol = downside_std * np.sqrt(252) * 100
            if downside_std > 0:
                sortino_ratio = (avg_return - rf_daily) / downside_std * np.sqrt(252)

        # 6. Calmar Ratio (annual return / max drawdown)
        calmar_ratio = None
        annual_return = avg_return * 252 * 100
        if max_drawdown < 0:
            calmar_ratio = annual_return / abs(max_drawdown)

        # Phase 4: CVaR 99% (Expected Shortfall at 99%)
        cvar_99 = None
        if len(returns_252d) >= 100:
            sorted_returns = sorted(returns_252d)
            var_99_idx = int(len(sorted_returns) * 0.01)
            var_99_threshold = sorted_returns[var_99_idx] if var_99_idx > 0 else sorted_returns[0]
            tail_returns = [r for r in sorted_returns if r <= var_99_threshold]
            if tail_returns:
                cvar_99 = abs(np.mean(tail_returns) * 100)

        # Phase 4: EWMA VaR 95% (Exponentially Weighted Moving Average, lambda=0.94)
        var_95_ewma = None
        if len(returns_252d) >= 60:
            lambda_decay = 0.94
            weights = np.array([(1 - lambda_decay) * (lambda_decay ** i) for i in range(len(returns_252d))])
            weights = weights / weights.sum()
            ewma_var = np.sqrt(np.sum(weights * np.array(returns_252d) ** 2))
            var_95_ewma = abs(ewma_var * 1.645 * 100)  # 95% z-score = 1.645

        # Phase 4: Inverse Volatility Weight
        inv_vol_weight = None
        if volatility_annual and volatility_annual > 0:
            inv_vol_weight = 1.0 / volatility_annual

        # Phase 4: Average Drawdown Duration
        drawdown_duration_avg = None
        if drawdown_durations:
            drawdown_duration_avg = np.mean(drawdown_durations)

        return {
            'volatility_annual': round(volatility_annual, 2),
            'max_drawdown_1y': round(max_drawdown, 2),
            'sharpe_ratio': round(sharpe_ratio, 4) if sharpe_ratio is not None else None,
            'sortino_ratio': round(sortino_ratio, 4) if sortino_ratio is not None else None,
            'calmar_ratio': round(calmar_ratio, 4) if calmar_ratio is not None else None,
            'downside_vol': round(downside_vol, 2) if downside_vol is not None else None,
            'cvar_99': round(cvar_99, 2) if cvar_99 is not None else None,
            'var_95_ewma': round(var_95_ewma, 2) if var_95_ewma is not None else None,
            'inv_vol_weight': round(inv_vol_weight, 6) if inv_vol_weight is not None else None,
            'drawdown_duration_avg': round(drawdown_duration_avg, 2) if drawdown_duration_avg is not None else None
        }

    def _create_volatility_engine_from_data(self, price_data: Dict) -> Optional[USVolatilityAdjustment]:
        """
        Create shared volatility engine from prefetched price data.
        No DB query required.

        Args:
            price_data: Prefetched price data from USDataPrefetcher

        Returns:
            USVolatilityAdjustment instance or None
        """
        if not price_data or 'raw' not in price_data:
            return None

        try:
            # Extract price data for volatility engine (30 days)
            raw_data = price_data['raw'][:30]
            engine_data = [
                {'date': r['date'], 'close': float(r['close']) if r.get('close') else None}
                for r in raw_data
            ]

            vol_engine = USVolatilityAdjustment(self.db)
            # Note: load_data_sync is a synchronous method that accepts pre-loaded data
            vol_engine.price_data = engine_data
            vol_engine.is_loaded = True

            return vol_engine

        except Exception as e:
            logger.warning(f"Failed to create volatility engine from data: {e}")
            return None

    # ========================================================================
    # Phase 4: SPY Correlation & Tail Beta Calculations
    # ========================================================================

    async def _prefetch_spy_returns(self, analysis_date: date) -> Optional[List[float]]:
        """
        Prefetch SPY returns for tail_beta and corr_spy calculations.
        Called once per batch, reused for all stocks in the batch.

        Args:
            analysis_date: Analysis date

        Returns:
            List of SPY daily returns (252 days, newest first) or None
        """
        query = """
        SELECT date, close FROM us_daily_etf
        WHERE symbol = 'SPY' AND date <= $1
        ORDER BY date DESC
        LIMIT 260
        """
        try:
            result = await self.db.execute_query(query, analysis_date)
            if not result or len(result) < 60:
                return None

            closes = [float(r['close']) for r in result if r['close'] is not None]
            if len(closes) < 60:
                return None

            # Calculate returns
            returns = [(closes[i] - closes[i + 1]) / closes[i + 1]
                       for i in range(min(252, len(closes) - 1))]
            return returns

        except Exception as e:
            logger.warning(f"Failed to prefetch SPY returns: {e}")
            return None

    def _calculate_tail_beta(self, stock_returns: List[float], spy_returns: List[float]) -> Optional[float]:
        """
        Calculate tail beta (beta during market stress, SPY return < -2%).
        Measures stock sensitivity during market downturns.

        Args:
            stock_returns: Stock daily returns (newest first)
            spy_returns: SPY daily returns (newest first)

        Returns:
            Tail beta or None if insufficient data
        """
        if not stock_returns or not spy_returns:
            return None

        # Align lengths
        min_len = min(len(stock_returns), len(spy_returns))
        if min_len < 20:
            return None

        stock_rets = np.array(stock_returns[:min_len])
        spy_rets = np.array(spy_returns[:min_len])

        # Filter for tail events (SPY down more than 2%)
        tail_mask = spy_rets < -0.02
        tail_count = np.sum(tail_mask)

        if tail_count < 5:  # Need at least 5 tail events
            return None

        tail_stock = stock_rets[tail_mask]
        tail_spy = spy_rets[tail_mask]

        # Calculate tail beta using regression
        spy_var = np.var(tail_spy)
        if spy_var == 0:
            return None

        covariance = np.cov(tail_stock, tail_spy)[0, 1]
        tail_beta = covariance / spy_var

        return round(tail_beta, 4)

    def _calculate_corr_spy(self, stock_returns: List[float], spy_returns: List[float],
                            days: int = 60) -> Optional[float]:
        """
        Calculate correlation between stock and SPY.

        Args:
            stock_returns: Stock daily returns (newest first)
            spy_returns: SPY daily returns (newest first)
            days: Number of days for correlation calculation (default 60)

        Returns:
            Correlation coefficient (-1 to 1) or None
        """
        if not stock_returns or not spy_returns:
            return None

        # Use specified number of days
        min_len = min(len(stock_returns), len(spy_returns), days)
        if min_len < 20:
            return None

        stock_rets = np.array(stock_returns[:min_len])
        spy_rets = np.array(spy_returns[:min_len])

        # Calculate correlation
        corr_matrix = np.corrcoef(stock_rets, spy_rets)
        corr = corr_matrix[0, 1]

        if np.isnan(corr):
            return None

        return round(corr, 4)

    def _calculate_stock_returns_from_data(self, price_data: Dict, days: int = 252) -> Optional[List[float]]:
        """
        Calculate daily returns from prefetched price data.

        Args:
            price_data: Prefetched price data
            days: Number of days for returns calculation

        Returns:
            List of daily returns (newest first) or None
        """
        if not price_data or 'closes' not in price_data:
            return None

        closes = price_data['closes']
        closes = [c for c in closes if c is not None]

        if len(closes) < 20:
            return None

        returns = [(closes[i] - closes[i + 1]) / closes[i + 1]
                   for i in range(min(days, len(closes) - 1))]

        return returns if returns else None

    # ========================================================================
    # Phase 3.4.3 Stage 3: Factor Momentum & Sector Metrics
    # ========================================================================

    async def _calculate_factor_momentum(self, symbol: str, analysis_date: date,
                                          current_scores: Dict) -> Dict[str, Optional[float]]:
        """
        Calculate factor score momentum (change vs previous month).
        Phase 3.4.3 Stage 3: 4 columns

        Args:
            symbol: Stock symbol
            analysis_date: Current analysis date
            current_scores: Dict with 'value', 'quality', 'momentum', 'growth' keys

        Returns:
            {
                'value_momentum': float,      # DECIMAL(5,1)
                'quality_momentum': float,
                'momentum_momentum': float,
                'growth_momentum': float
            }
        """
        null_result = {
            'value_momentum': None,
            'quality_momentum': None,
            'momentum_momentum': None,
            'growth_momentum': None
        }

        try:
            # Get scores from ~30 days ago
            prev_date = analysis_date - timedelta(days=30)
            query = """
                SELECT value_score, quality_score, momentum_score, growth_score
                FROM us_stock_grade
                WHERE symbol = $1 AND date <= $2
                ORDER BY date DESC LIMIT 1
            """
            result = await self.db.execute_query(query, symbol, prev_date)

            if not result:
                return null_result

            prev_row = result[0]

            # Calculate momentum (current - previous)
            def calc_momentum(current: float, prev_val) -> Optional[float]:
                if prev_val is None:
                    return None
                return round(current - float(prev_val), 1)

            return {
                'value_momentum': calc_momentum(current_scores['value'], prev_row.get('value_score')),
                'quality_momentum': calc_momentum(current_scores['quality'], prev_row.get('quality_score')),
                'momentum_momentum': calc_momentum(current_scores['momentum'], prev_row.get('momentum_score')),
                'growth_momentum': calc_momentum(current_scores['growth'], prev_row.get('growth_score'))
            }

        except Exception as e:
            logger.debug(f"{symbol}: Factor momentum calculation failed - {e}")
            return null_result

    async def _calculate_sector_metrics(self, sector: str, analysis_date: date) -> Dict:
        """
        Calculate sector-level metrics from materialized view.
        Phase 3.4.3 Stage 4: 4 columns

        Args:
            sector: Sector name (e.g., 'Technology')
            analysis_date: Analysis date

        Returns:
            {
                'sector_rotation_score': int,  # 0-100
                'sector_momentum': float,      # DECIMAL(5,2)
                'sector_rank': int,
                'sector_percentile': float     # DECIMAL(5,2)
            }
        """
        null_result = {
            'sector_rotation_score': None,
            'sector_momentum': None,
            'sector_rank': None,
            'sector_percentile': None
        }

        if not sector:
            return null_result

        try:
            # Query mv_us_sector_daily_performance
            query = """
                SELECT
                    avg_return_30d as momentum,
                    sector_rank,
                    stock_count,
                    (SELECT COUNT(DISTINCT sector_code)
                     FROM mv_us_sector_daily_performance
                     WHERE date = $1) as total_sectors
                FROM mv_us_sector_daily_performance
                WHERE date = $1 AND sector_code = $2
            """
            result = await self.db.execute_query(query, analysis_date, sector)

            if not result:
                # Try previous trading days if no data for exact date
                for days_back in [1, 2, 3]:
                    fallback_date = analysis_date - timedelta(days=days_back)
                    result = await self.db.execute_query(query, fallback_date, sector)
                    if result:
                        break

            if not result:
                return null_result

            row = result[0]
            momentum = float(row['momentum']) if row.get('momentum') is not None else 0
            sector_rank = int(row['sector_rank']) if row.get('sector_rank') else None
            total_sectors = int(row['total_sectors']) if row.get('total_sectors') else 11

            # Calculate percentile (lower rank = better = lower percentile)
            sector_percentile = None
            if sector_rank and total_sectors > 0:
                sector_percentile = round((sector_rank / total_sectors) * 100, 2)

            # Calculate rotation score (0-100)
            # Higher momentum = higher score, lower rank = higher score
            rotation_score = None
            if sector_rank is not None:
                # Momentum contribution: +/-30 points based on momentum
                momentum_component = min(30, max(-30, momentum * 3))
                # Rank contribution: 70 points for rank 1, decreasing
                rank_component = max(0, 70 - (sector_rank - 1) * 7)
                rotation_score = int(min(100, max(0, 50 + momentum_component + (rank_component - 35))))

            return {
                'sector_rotation_score': rotation_score,
                'sector_momentum': round(momentum, 2),
                'sector_rank': sector_rank,
                'sector_percentile': sector_percentile
            }

        except Exception as e:
            logger.debug(f"Sector metrics calculation failed for {sector}: {e}")
            return null_result

    def _generate_analysis_texts(self, metrics: Dict) -> Dict[str, str]:
        """
        Generate human-readable analysis text.
        Phase 3.4.3 Stage 4: 5 columns

        Args:
            metrics: Dict containing all calculated metrics

        Returns:
            {
                'risk_profile_text': str,
                'risk_recommendation': str,
                'time_series_text': str,
                'signal_overall': str,
                'market_state': str
            }
        """
        # Extract metrics with defaults
        vol = metrics.get('volatility_annual') or 30
        mdd = metrics.get('max_drawdown_1y') or -15
        beta = metrics.get('beta') or 1.0
        final_score = metrics.get('total_score') or 50
        score_trend = metrics.get('score_trend_2w') or 0
        regime = metrics.get('regime') or 'NEUTRAL'
        vix_level = 20  # Default VIX estimate

        # 1. risk_profile_text
        if vol > 50 or mdd < -40 or (beta and beta > 1.8):
            risk_profile = "고위험 - 높은 변동성과 상당한 낙폭 가능성"
        elif vol > 35 or mdd < -25 or (beta and beta > 1.3):
            risk_profile = "중고위험 - 평균 이상의 시장 변동성 노출"
        elif vol > 25 or mdd < -15:
            risk_profile = "중위험 - 평균 수준의 시장 변동성 노출"
        else:
            risk_profile = "저위험 - 평균 이하 변동성 및 제한적 낙폭"

        # 2. risk_recommendation
        if vol > 45:
            risk_rec = "공격적 투자자에게 적합 (높은 위험 감내 및 장기 투자 가능자)"
        elif vol > 30:
            risk_rec = "균형형 투자자에게 적합 (성장 추구 및 적정 위험 수용자)"
        else:
            risk_rec = "보수적 투자자에게 적합 (원금 보존 우선 투자자)"

        # 3. time_series_text
        if score_trend and score_trend > 5:
            time_series = "개선 중 - 최근 2주간 점수 상승 추세"
        elif score_trend and score_trend < -5:
            time_series = "하락 중 - 최근 2주간 점수 하락 추세"
        else:
            time_series = "안정적 - 최근 2주간 점수 변동 미미"

        # 4. signal_overall
        if final_score >= 80:
            signal = "강력 매수 - 모든 팩터에서 탁월한 정렬"
        elif final_score >= 70:
            signal = "매수 - 우수한 펀더멘털과 유리한 위험-수익 구조"
        elif final_score >= 60:
            signal = "매수 고려 - 평균 이상 지표와 긍정적 촉매 존재"
        elif final_score >= 50:
            signal = "중립 - 균형 잡힌 위험-수익과 중립적 전망"
        elif final_score >= 40:
            signal = "비중 축소 - 평균 이하 펀더멘털, 포지션 조정 검토"
        elif final_score >= 30:
            signal = "매도 - 부진한 지표로 하방 위험 존재"
        else:
            signal = "강력 매도 - 심각한 펀더멘털 악화"

        # 5. market_state (regime + context)
        regime_map = {
            'BULL': '강세장',
            'NEUTRAL': '중립장',
            'BEAR': '약세장',
            'EXPANSION': '확장장',
            'CONTRACTION': '수축장'
        }
        market_state = regime_map.get(regime, regime)

        return {
            'risk_profile_text': risk_profile,
            'risk_recommendation': risk_rec,
            'time_series_text': time_series,
            'signal_overall': signal,
            'market_state': market_state
        }

    def _calculate_grade_distribution(self, results: List[Dict]) -> Dict:
        """등급 분포 계산"""

        distribution = {'강력 매수': 0, '매수': 0, '매수 고려': 0, '중립': 0, '매도 고려': 0, '매도': 0, '강력 매도': 0}

        for r in results:
            grade = r['grade']
            distribution[grade] = distribution.get(grade, 0) + 1

        return distribution

    async def _save_results(self, results: List[Dict], analysis_date: date):
        """결과 저장"""

        if not results:
            return

        # us_stock_grade 테이블에 저장 (UPSERT)
        for result in results:
            await self._save_stock_grade(result, analysis_date)

    async def _save_stock_grade(self, result: Dict, analysis_date: date):
        """개별 종목 등급 저장"""

        # detail을 JSON으로 변환
        def serialize_detail(detail: Dict) -> str:
            """detail dict를 JSON 문자열로 변환"""
            clean_detail = {}
            for key, value in detail.items():
                if value and isinstance(value, dict):
                    clean_value = {}
                    for k, v in value.items():
                        if v is not None:
                            if isinstance(v, (int, float)):
                                clean_value[k] = round(v, 4) if isinstance(v, float) else v
                            elif isinstance(v, dict):
                                clean_value[k] = {kk: round(vv, 4) if isinstance(vv, float) else vv
                                                   for kk, vv in v.items() if vv is not None}
                            else:
                                clean_value[k] = v
                    clean_detail[key] = clean_value
            return json.dumps(clean_detail)

        # Serialize triggers to JSON
        def serialize_triggers(triggers: list) -> str:
            """Convert triggers list to JSON string"""
            if not triggers:
                return json.dumps([])
            return json.dumps(triggers)

        query = """
        INSERT INTO us_stock_grade (
            date, symbol, stock_name, beta, confidence_score, factor_combination_bonus,
            final_score, final_grade,
            value_score, quality_score, momentum_score, growth_score,
            value_v2_detail, quality_v2_detail,
            momentum_v2_detail, growth_v2_detail,
            var_95, cvar_95,
            hurst_exponent, var_95_5d, var_95_20d, var_95_60d, var_95_90d, var_99, var_99_90d,
            rs_value,
            interaction_score, conviction_score,
            entry_timing_score, score_trend_2w, price_position_52w,
            atr_pct, stop_loss_pct, take_profit_pct, risk_reward_ratio, position_size_pct,
            scenario_bullish_prob, scenario_sideways_prob, scenario_bearish_prob,
            scenario_bullish_return, scenario_sideways_return, scenario_bearish_return,
            scenario_sample_count,
            buy_triggers, sell_triggers, hold_triggers,
            iv_percentile, insider_signal,
            outlier_risk_score, risk_flag,
            weight_growth, weight_momentum, weight_quality, weight_value,
            volatility_annual, max_drawdown_1y, sharpe_ratio, sortino_ratio, calmar_ratio,
            value_momentum, quality_momentum, momentum_momentum, growth_momentum,
            sector_rotation_score, sector_momentum, sector_rank, sector_percentile,
            risk_profile_text, risk_recommendation, time_series_text, signal_overall, market_state,
            downside_vol, tail_beta, corr_spy, cvar_99, var_95_ewma,
            inv_vol_weight, vol_percentile, corr_sector_avg, drawdown_duration_avg
        ) VALUES (
            $1, $2, $3, $4, $5, $6,
            $7, $8,
            $9, $10, $11, $12,
            $13::jsonb, $14::jsonb, $15::jsonb, $16::jsonb,
            $17, $18,
            $19, $20, $21, $22, $23, $24, $25,
            $26,
            $27, $28,
            $29, $30, $31,
            $32, $33, $34, $35, $36,
            $37, $38, $39,
            $40, $41, $42,
            $43,
            $44::jsonb, $45::jsonb, $46::jsonb,
            $47, $48,
            $49, $50,
            $51, $52, $53, $54,
            $55, $56, $57, $58, $59,
            $60, $61, $62, $63,
            $64, $65, $66, $67,
            $68, $69, $70, $71, $72,
            $73, $74, $75, $76, $77,
            $78, $79, $80, $81
        )
        ON CONFLICT (date, symbol) DO UPDATE SET
            stock_name = EXCLUDED.stock_name,
            beta = EXCLUDED.beta,
            confidence_score = EXCLUDED.confidence_score,
            factor_combination_bonus = EXCLUDED.factor_combination_bonus,
            final_score = EXCLUDED.final_score,
            final_grade = EXCLUDED.final_grade,
            value_score = EXCLUDED.value_score,
            quality_score = EXCLUDED.quality_score,
            momentum_score = EXCLUDED.momentum_score,
            growth_score = EXCLUDED.growth_score,
            value_v2_detail = EXCLUDED.value_v2_detail,
            quality_v2_detail = EXCLUDED.quality_v2_detail,
            momentum_v2_detail = EXCLUDED.momentum_v2_detail,
            growth_v2_detail = EXCLUDED.growth_v2_detail,
            var_95 = EXCLUDED.var_95,
            cvar_95 = EXCLUDED.cvar_95,
            hurst_exponent = EXCLUDED.hurst_exponent,
            var_95_5d = EXCLUDED.var_95_5d,
            var_95_20d = EXCLUDED.var_95_20d,
            var_95_60d = EXCLUDED.var_95_60d,
            var_95_90d = EXCLUDED.var_95_90d,
            var_99 = EXCLUDED.var_99,
            var_99_90d = EXCLUDED.var_99_90d,
            rs_value = EXCLUDED.rs_value,
            interaction_score = EXCLUDED.interaction_score,
            conviction_score = EXCLUDED.conviction_score,
            entry_timing_score = EXCLUDED.entry_timing_score,
            score_trend_2w = EXCLUDED.score_trend_2w,
            price_position_52w = EXCLUDED.price_position_52w,
            atr_pct = EXCLUDED.atr_pct,
            stop_loss_pct = EXCLUDED.stop_loss_pct,
            take_profit_pct = EXCLUDED.take_profit_pct,
            risk_reward_ratio = EXCLUDED.risk_reward_ratio,
            position_size_pct = EXCLUDED.position_size_pct,
            scenario_bullish_prob = EXCLUDED.scenario_bullish_prob,
            scenario_sideways_prob = EXCLUDED.scenario_sideways_prob,
            scenario_bearish_prob = EXCLUDED.scenario_bearish_prob,
            scenario_bullish_return = EXCLUDED.scenario_bullish_return,
            scenario_sideways_return = EXCLUDED.scenario_sideways_return,
            scenario_bearish_return = EXCLUDED.scenario_bearish_return,
            scenario_sample_count = EXCLUDED.scenario_sample_count,
            buy_triggers = EXCLUDED.buy_triggers,
            sell_triggers = EXCLUDED.sell_triggers,
            hold_triggers = EXCLUDED.hold_triggers,
            iv_percentile = EXCLUDED.iv_percentile,
            insider_signal = EXCLUDED.insider_signal,
            outlier_risk_score = EXCLUDED.outlier_risk_score,
            risk_flag = EXCLUDED.risk_flag,
            weight_growth = EXCLUDED.weight_growth,
            weight_momentum = EXCLUDED.weight_momentum,
            weight_quality = EXCLUDED.weight_quality,
            weight_value = EXCLUDED.weight_value,
            volatility_annual = EXCLUDED.volatility_annual,
            max_drawdown_1y = EXCLUDED.max_drawdown_1y,
            sharpe_ratio = EXCLUDED.sharpe_ratio,
            sortino_ratio = EXCLUDED.sortino_ratio,
            calmar_ratio = EXCLUDED.calmar_ratio,
            value_momentum = EXCLUDED.value_momentum,
            quality_momentum = EXCLUDED.quality_momentum,
            momentum_momentum = EXCLUDED.momentum_momentum,
            growth_momentum = EXCLUDED.growth_momentum,
            sector_rotation_score = EXCLUDED.sector_rotation_score,
            sector_momentum = EXCLUDED.sector_momentum,
            sector_rank = EXCLUDED.sector_rank,
            sector_percentile = EXCLUDED.sector_percentile,
            risk_profile_text = EXCLUDED.risk_profile_text,
            risk_recommendation = EXCLUDED.risk_recommendation,
            time_series_text = EXCLUDED.time_series_text,
            signal_overall = EXCLUDED.signal_overall,
            market_state = EXCLUDED.market_state,
            downside_vol = EXCLUDED.downside_vol,
            tail_beta = EXCLUDED.tail_beta,
            corr_spy = EXCLUDED.corr_spy,
            cvar_99 = EXCLUDED.cvar_99,
            var_95_ewma = EXCLUDED.var_95_ewma,
            inv_vol_weight = EXCLUDED.inv_vol_weight,
            vol_percentile = EXCLUDED.vol_percentile,
            corr_sector_avg = EXCLUDED.corr_sector_avg,
            drawdown_duration_avg = EXCLUDED.drawdown_duration_avg,
            updated_at = CURRENT_TIMESTAMP
        """

        try:
            await self.db.execute_query(
                query,
                analysis_date,                                      # $1
                result['symbol'],                                   # $2
                result.get('stock_name'),                           # $3 (Phase 3.4.3)
                result.get('beta'),                                 # $4 (Phase 3.4.3)
                result.get('confidence_score'),                     # $5 (Phase 3.4.3)
                result.get('factor_combination_bonus'),             # $6 (Phase 3.4.3)
                result['total_score'],                              # $7
                result['grade'],                                    # $8
                result['value_score'],                              # $9
                result['quality_score'],                            # $10
                result['momentum_score'],                           # $11
                result['growth_score'],                             # $12
                serialize_detail(result['value_detail']),           # $13
                serialize_detail(result['quality_detail']),         # $14
                serialize_detail(result['momentum_detail']),        # $15
                serialize_detail(result['growth_detail']),          # $16
                result.get('var_95'),                               # $17
                result.get('cvar_95'),                              # $18
                # Phase 4.1: Hurst Exponent and Period-based VaR
                result.get('hurst_exponent'),                       # $19
                result.get('var_95_5d'),                            # $20
                result.get('var_95_20d'),                           # $21
                result.get('var_95_60d'),                           # $22
                result.get('var_95_90d'),                           # $23
                result.get('var_99'),                               # $24
                result.get('var_99_90d'),                           # $25
                result.get('rs_value'),                             # $26
                result.get('interaction_score'),                    # $27
                result.get('conviction_score'),                     # $28
                result.get('entry_timing_score'),                   # $29
                result.get('score_trend_2w'),                       # $30
                result.get('price_position_52w'),                   # $31
                result.get('atr_pct'),                              # $32
                result.get('stop_loss_pct'),                        # $33
                result.get('take_profit_pct'),                      # $34
                result.get('risk_reward_ratio'),                    # $35
                result.get('position_size_pct'),                    # $36
                result.get('scenario_bullish_prob'),                # $37
                result.get('scenario_sideways_prob'),               # $38
                result.get('scenario_bearish_prob'),                # $39
                result.get('scenario_bullish_return'),              # $40
                result.get('scenario_sideways_return'),             # $41
                result.get('scenario_bearish_return'),              # $42
                result.get('scenario_sample_count'),                # $43
                serialize_triggers(result.get('buy_triggers')),     # $44
                serialize_triggers(result.get('sell_triggers')),    # $45
                serialize_triggers(result.get('hold_triggers')),    # $46
                result.get('iv_percentile'),                        # $47
                result.get('insider_signal'),                       # $48
                # Phase 3.1 fields
                result.get('outlier_risk_score'),                   # $49
                result.get('risk_flag'),                            # $50
                result.get('weight_growth'),                        # $51
                result.get('weight_momentum'),                      # $52
                result.get('weight_quality'),                       # $53
                result.get('weight_value'),                         # $54
                # Phase 3.4.3: Risk Metrics
                result.get('volatility_annual'),                    # $55
                result.get('max_drawdown_1y'),                      # $56
                result.get('sharpe_ratio'),                         # $57
                result.get('sortino_ratio'),                        # $58
                result.get('calmar_ratio'),                         # $59
                # Phase 3.4.3 Stage 3: Factor Momentum
                result.get('value_momentum'),                       # $60
                result.get('quality_momentum'),                     # $61
                result.get('momentum_momentum'),                    # $62
                result.get('growth_momentum'),                      # $63
                # Phase 3.4.3 Stage 4: Sector Metrics
                result.get('sector_rotation_score'),                # $64
                result.get('sector_momentum'),                      # $65
                result.get('sector_rank'),                          # $66
                result.get('sector_percentile'),                    # $67
                # Phase 3.4.3 Stage 4: Text Generation
                result.get('risk_profile_text'),                    # $68
                result.get('risk_recommendation'),                  # $69
                result.get('time_series_text'),                     # $70
                result.get('signal_overall'),                       # $71
                result.get('market_state'),                         # $72
                # Phase 4: Additional Risk Metrics (9 columns)
                result.get('downside_vol'),                         # $73
                result.get('tail_beta'),                            # $74
                result.get('corr_spy'),                             # $75
                result.get('cvar_99'),                              # $76
                result.get('var_95_ewma'),                          # $77
                result.get('inv_vol_weight'),                       # $78
                result.get('vol_percentile'),                       # $79
                result.get('corr_sector_avg'),                      # $80
                result.get('drawdown_duration_avg')                 # $81
            )
        except Exception as e:
            logger.error(f"{result['symbol']}: Failed to save - {e}")
            # Debug: Log DECIMAL(6,4) and other numeric column values
            logger.error(f"  DEBUG - corr_spy: {result.get('corr_spy')}, "
                        f"tail_beta: {result.get('tail_beta')}, "
                        f"downside_vol: {result.get('downside_vol')}, "
                        f"cvar_99: {result.get('cvar_99')}, "
                        f"var_95_ewma: {result.get('var_95_ewma')}, "
                        f"inv_vol_weight: {result.get('inv_vol_weight')}")


# ========================================================================
# Convenience Functions
# ========================================================================

async def analyze_single_stock_standalone(symbol: str, db_manager, analysis_date: Optional[date] = None,
                                          verbose: bool = True) -> Optional[Dict]:
    """단일 종목 분석 (standalone) - 디버깅 로그 포함"""

    analysis_date = analysis_date or date.today()

    if verbose:
        print(f"\n{'='*80}")
        print(f"미국 주식 v2.0 분석 시작: {symbol}")
        print(f"분석 날짜: {analysis_date}")
        print(f"{'='*80}\n")

    try:
        system = USQuantSystemV2(db_manager)

        # Step 1: 레짐 감지
        if verbose:
            print("Step 1: 시장 레짐 감지 중...")
        await system._detect_regime(analysis_date)
        if verbose:
            print(f"  레짐: {system.current_regime} / Macro: {getattr(system, 'macro_environment', 'N/A')}")
            print(f"  가중치: Value={system.regime_weights['value']:.0%}, "
                  f"Quality={system.regime_weights['quality']:.0%}, "
                  f"Momentum={system.regime_weights['momentum']:.0%}, "
                  f"Growth={system.regime_weights['growth']:.0%}\n")

        # Step 2: 종목 기본 정보 확인
        if verbose:
            print("Step 2: 종목 기본 정보 확인 중...")
        stock_info = await system._get_stock_info(symbol)
        if not stock_info:
            print(f"  [오류] {symbol}: 종목 기본 정보 없음 (us_stock_basic)")
            return None
        sector = stock_info.get('sector')
        if not sector:
            print(f"  [오류] {symbol}: 섹터 정보 없음")
            return None
        if verbose:
            print(f"  섹터: {sector}")
            print(f"  시가총액: ${stock_info.get('market_cap') or 0:,.0f}\n")

        # Step 3: 섹터 벤치마크 로드
        if verbose:
            print("Step 3: 섹터 벤치마크 로드 중...")
        benchmarks = await system._get_sector_benchmarks(sector)
        if verbose:
            print(f"  PE Median: {benchmarks.get('pe_median', 'N/A')}")
            print(f"  Forward PE Median: {benchmarks.get('forward_pe_median', 'N/A')}")
            print(f"  Gross Margin Median: {benchmarks.get('gross_margin_median', 'N/A')}\n")

        # Step 4: Factor 점수 계산
        if verbose:
            print("Step 4: Factor 점수 계산 중 (병렬 처리)...")

        value_task = calculate_value_score(symbol, db_manager, benchmarks, analysis_date)
        quality_task = calculate_quality_score(symbol, db_manager, benchmarks, analysis_date)
        # Phase 3.2: NASDAQ이면 전용 Momentum 가중치 사용
        exchange = stock_info.get('exchange', 'NYSE')
        momentum_task = calculate_momentum_score(symbol, db_manager, benchmarks, analysis_date, exchange)
        growth_task = calculate_growth_score(symbol, db_manager, benchmarks, analysis_date)

        results = await asyncio.gather(
            value_task, quality_task, momentum_task, growth_task,
            return_exceptions=True
        )

        # 결과 처리
        value_result = results[0] if not isinstance(results[0], Exception) else {'value_score': 50.0, 'strategies': {}}
        quality_result = results[1] if not isinstance(results[1], Exception) else {'quality_score': 50.0, 'strategies': {}}
        momentum_result = results[2] if not isinstance(results[2], Exception) else {'momentum_score': 50.0, 'strategies': {}}
        growth_result = results[3] if not isinstance(results[3], Exception) else {'growth_score': 50.0, 'strategies': {}}

        # 예외 로깅
        for i, (name, res) in enumerate([('Value', results[0]), ('Quality', results[1]),
                                          ('Momentum', results[2]), ('Growth', results[3])]):
            if isinstance(res, Exception):
                print(f"  [경고] {name} Factor 계산 실패: {res}")

        value_score = value_result['value_score']
        quality_score = quality_result['quality_score']
        momentum_score = momentum_result['momentum_score']
        growth_score = growth_result['growth_score']

        if verbose:
            print(f"  가치:    {value_score:.1f}")
            print(f"  품질:    {quality_score:.1f}")
            print(f"  모멘텀:  {momentum_score:.1f}")
            print(f"  성장:    {growth_score:.1f}\n")

        # Step 5: 최종 점수 계산 (Phase 3: Factor Interaction 적용)
        if verbose:
            print("Step 5: 최종 점수 계산 중...")

        # Base score (기존 가중합)
        base_score = (
            value_score * system.regime_weights['value'] +
            quality_score * system.regime_weights['quality'] +
            momentum_score * system.regime_weights['momentum'] +
            growth_score * system.regime_weights['growth']
        )

        # Factor Interaction 계산 (Phase 3)
        interaction_calc = USFactorInteractions({
            'value_score': value_score,
            'quality_score': quality_score,
            'momentum_score': momentum_score,
            'growth_score': growth_score
        })
        interaction_result = interaction_calc.calculate()
        interaction_score = interaction_result['interaction_score']
        conviction_score = interaction_result['conviction_score']

        # Final score = base 70% + interaction 30%
        total_score = base_score * 0.70 + interaction_score * 0.30

        if verbose:
            print(f"  Base Score 계산:")
            print(f"    {value_score:.1f} x {system.regime_weights['value']:.2f} = {value_score * system.regime_weights['value']:.2f}")
            print(f"    {quality_score:.1f} x {system.regime_weights['quality']:.2f} = {quality_score * system.regime_weights['quality']:.2f}")
            print(f"    {momentum_score:.1f} x {system.regime_weights['momentum']:.2f} = {momentum_score * system.regime_weights['momentum']:.2f}")
            print(f"    {growth_score:.1f} x {system.regime_weights['growth']:.2f} = {growth_score * system.regime_weights['growth']:.2f}")
            print(f"  Base Score: {base_score:.1f}")
            print(f"\n  Factor Interaction:")
            print(f"    I1 (Growth x Quality):   {interaction_result['i1_growth_quality']:.1f}")
            print(f"    I2 (Growth x Momentum):  {interaction_result['i2_growth_momentum']:.1f}")
            print(f"    I3 (Quality x Value):    {interaction_result['i3_quality_value']:.1f}")
            print(f"    I4 (Momentum x Quality): {interaction_result['i4_momentum_quality']:.1f}")
            print(f"    I5 (Conviction):         {conviction_score:.1f}")
            print(f"  Interaction Score: {interaction_score:.1f}")
            print(f"\n  Final Score = {base_score:.1f} x 0.70 + {interaction_score:.1f} x 0.30 = {total_score:.1f}\n")

        # Phase 3.5: Pattern-based score adjustment
        market_cap = stock_info.get('market_cap') or 0
        total_score, pattern_applied = apply_pattern_adjustment(
            total_score, value_score, quality_score, momentum_score, growth_score, market_cap
        )
        if pattern_applied and verbose:
            print(f"  Pattern adjustment applied: {pattern_applied}")
            print(f"  Adjusted Score: {total_score:.1f}\n")

        # Step 6: 등급 결정 (7단계 한글)
        grade = system._determine_grade(total_score)
        opinion = grade  # grade 자체가 한글 투자의견

        if verbose:
            print("Step 6: 등급 결정...")
            print(f"  등급: {grade}")
            print(f"  기준: 강력 매수(85+), 매수(75+), 매수 고려(65+), 중립(55+), 매도 고려(45+), 매도(35+), 강력 매도(<35)\n")

        # 결과 구성
        result = {
            'symbol': symbol,
            'sector': sector,
            'total_score': round(total_score, 1),
            'base_score': round(base_score, 1),
            'interaction_score': round(interaction_score, 1),
            'conviction_score': round(conviction_score, 1),
            'grade': grade,
            'opinion': opinion,
            'value_score': value_score,
            'quality_score': quality_score,
            'momentum_score': momentum_score,
            'growth_score': growth_score,
            'value_detail': value_result.get('strategies', {}),
            'quality_detail': quality_result.get('strategies', {}),
            'momentum_detail': momentum_result.get('strategies', {}),
            'growth_detail': growth_result.get('strategies', {}),
            'regime': system.current_regime,
            'regime_weights': system.regime_weights
        }

        # Step 7: DB 저장
        if verbose:
            print("Step 7: 데이터베이스 저장 중...")
        await system._save_stock_grade(result, analysis_date)
        if verbose:
            print(f"  저장 완료 (날짜: {analysis_date})\n")

        # 최종 요약
        if verbose:
            print("="*80)
            print("분석 요약")
            print("="*80)
            print(f"심볼: {symbol}")
            print(f"섹터: {sector}")
            print(f"시장 레짐: {system.current_regime}")
            print(f"\n팩터 점수:")
            print(f"  가치:       {value_score:>6.1f}")
            print(f"  품질:       {quality_score:>6.1f}")
            print(f"  모멘텀:     {momentum_score:>6.1f}")
            print(f"  성장:       {growth_score:>6.1f}")
            print(f"\nPhase 3 점수:")
            print(f"  Base Score:        {base_score:>6.1f}")
            print(f"  Interaction Score: {interaction_score:>6.1f}")
            print(f"  Conviction Score:  {conviction_score:>6.1f}")
            print(f"\n최종 결과:")
            print(f"  총점:    {total_score:>6.1f}")
            print(f"  등급:    {grade}")
            print("="*80 + "\n")

        # Alternative stock matching for sell-grade stocks (Option 1)
        if grade in ('매도 고려', '매도', '강력 매도'):
            if verbose:
                print("Alternative Stock 매칭 중...")
            matcher = USAlternativeStockMatcher(db_manager)
            match_success = await matcher.match_single_symbol(symbol, analysis_date)
            if verbose and match_success:
                # Query and display matched alternative
                alt_query = """
                    SELECT alt_symbol, alt_stock_name, alt_final_grade, alt_final_score,
                           alt_match_type, alt_reasons
                    FROM us_stock_grade
                    WHERE symbol = $1 AND date = $2
                """
                alt_result = await db_manager.execute_query(alt_query, symbol, analysis_date)
                if alt_result and alt_result[0]['alt_symbol']:
                    alt = alt_result[0]
                    print(f"  대체 종목: {alt['alt_symbol']} {alt['alt_stock_name']}")
                    print(f"  등급/점수: {alt['alt_final_grade']} ({alt['alt_final_score']}점)")
                    print(f"  매칭 유형: {alt['alt_match_type']}")
                    if alt['alt_reasons']:
                        reasons = alt['alt_reasons']
                        print(f"  추천 이유:")
                        for i, reason in enumerate(reasons, 1):
                            print(f"    {i}. {reason['text']}")
                    print()

        return result

    except Exception as e:
        logger.error(f"{symbol} 분석 실패: {e}", exc_info=True)
        if verbose:
            print(f"\n[오류] {symbol} 분석 실패")
            print(f"이유: {str(e)}\n")
        return None


async def analyze_all_stocks_specific_dates(
    db_manager,
    date_list: List[date],
    symbols_filter: Optional[List[str]] = None,
    with_event_modifier: bool = True,
    improved_factors: bool = False
) -> Dict:
    """
    전체 종목 특정 날짜 분석 - 디버깅 로그 포함 (v3.0: HMM 전용)

    Args:
        db_manager: AsyncDatabaseManager instance
        date_list: 분석 날짜 리스트
        symbols_filter: 분석 대상 종목 화이트리스트. 지정 시 이 종목들만
                        분석 (orchestrator EM8 pre-filter 와 연동).
    """
    print("\n" + "="*80)
    print("미국 주식 v3.0 - 전체 종목 특정 날짜 분석")
    print("="*80)
    print(f"분석 날짜: {len(date_list)}개")
    for idx, d in enumerate(date_list, 1):
        print(f"  {idx}. {d}")
    print(f"레짐 감지: HMM (3-State: BULL/NEUTRAL/BEAR)")
    print(f"가중치 모드: 정적 (분기별 리밸런싱 권고)")
    print(f"DB 커넥션 풀: min_size=10, max_size=30")
    print(f"배치 크기: 25개 종목 (동시 처리)")
    print("="*80 + "\n")

    # Suppress verbose logging
    original_levels = {}
    loggers_to_suppress = [
        'root', '__main__', 'us_main',
        'us_market_regime', 'us_sector_benchmarks',
        'us_value_factor', 'us_quality_factor',
        'us_momentum_factor', 'us_growth_factor'
    ]

    for logger_name in loggers_to_suppress:
        target_logger = logging.getLogger(logger_name)
        original_levels[logger_name] = target_logger.level
        target_logger.setLevel(logging.ERROR)

    total_start_time = time.time()
    date_results = {}
    total_success = 0
    total_fail = 0
    total_processed = 0
    all_failed_symbols = []  # 전체 실패 종목 추적

    system = USQuantSystemV2(db_manager, with_event_modifier=with_event_modifier,
                             improved_factors=improved_factors)

    # HMM 모델 초기화 (한 번만 학습)
    print("HMM 모델 초기화 중 (2년 데이터 학습, 1회만 실행)...")
    await system._init_hmm_detector()
    print("HMM 모델 준비 완료\n")

    # ------------------------------------------------------------------
    # Range-mode bulk prefetch (NEW): collapse ~570k per-(symbol,date) DB
    # queries to ~5 bulk fetches, and ~14k sector PERCENTILE_CONT queries
    # to ~4 GROUP BY queries. Loads into class-level caches; subsequent
    # prefetch_all / get_sector_benchmarks calls hit memory.
    # ------------------------------------------------------------------
    if date_list:
        bulk_start = min(date_list)
        bulk_end = max(date_list)

        # Union of symbols across all dates — use symbols_filter when given,
        # else pull from us_daily for the range.
        if symbols_filter:
            bulk_symbols = sorted(set(symbols_filter))
        else:
            sym_rows = await db_manager.execute_query(
                """SELECT DISTINCT symbol FROM us_daily
                   WHERE date BETWEEN $1 AND $2""",
                bulk_start, bulk_end)
            bulk_symbols = sorted({r['symbol'] for r in sym_rows})

        if bulk_symbols:
            print(f"Bulk prefetch: {len(bulk_symbols)} symbols × "
                  f"{(bulk_end - bulk_start).days + 1}d ({bulk_start} ~ {bulk_end})")
            t0 = time.time()
            from us_data_prefetcher import USDataPrefetcher
            await USDataPrefetcher.bulk_prefetch_range(
                db_manager, bulk_symbols, bulk_start, bulk_end)
            print(f"  → prefetch_all data loaded ({time.time()-t0:.1f}s)")
            t0 = time.time()
            await USSectorBenchmarks.bulk_precompute_range(
                db_manager, bulk_start, bulk_end)
            print(f"  → sector benchmarks precomputed ({time.time()-t0:.1f}s)")

    for date_idx, analysis_date in enumerate(date_list, 1):
        date_start_time = time.time()

        print(f"[날짜 {date_idx}/{len(date_list)}] {analysis_date} - 분석 시작...")

        # Sector/Industry MV 데이터 생성 (해당 날짜 데이터 없으면 INSERT)
        try:
            await db_manager.refresh_sector_performance_for_date(analysis_date)
            await db_manager.refresh_industry_performance_for_date(analysis_date)
        except Exception as mv_err:
            logger.warning(f"MV refresh failed for {analysis_date}: {mv_err}")

        # 레짐 감지
        try:
            await system._detect_regime(analysis_date)
            method_str = f"[{system.regime_method}]" if system.regime_method else ""
            print(f"  레짐: {system.current_regime} {method_str} / Macro: {getattr(system, 'macro_environment', 'N/A')}")
            print(f"  가중치: V={system.regime_weights['value']:.0%}, "
                  f"Q={system.regime_weights['quality']:.0%}, "
                  f"M={system.regime_weights['momentum']:.0%}, "
                  f"G={system.regime_weights['growth']:.0%}")
        except Exception as e:
            print(f"  [오류] 레짐 감지 실패: {e}")
            print(f"  기본 레짐(NEUTRAL) 사용")

        # 해당 날짜에 데이터가 있는 종목 조회
        # symbols_filter 지정 시 그 화이트리스트로 제한 (EM8 pre-filter 연동)
        # us_stock_basic is point-in-time (one row per trading day per symbol).
        # Use ``b.date <= $1`` so the sector check sees the most recent row that
        # existed on the analysis date, not a future one.
        # PIT universe (survivorship-bias 제거):
        #   us_listing_status 의 (ipo_date, delisting_date) 기준으로 그 시점에
        #   상장 중이었던 종목만. 가격 데이터 (us_daily) 도 그 날 있어야 분석
        #   가능. us_listing_status 가 비어있으면 (마이그레이션/수집 전) 자동
        #   폴백.
        ls_check = await db_manager.execute_query("""
            SELECT EXISTS (SELECT 1 FROM information_schema.tables
                           WHERE table_name='us_listing_status') AS exists
        """)
        ls_ready = bool(ls_check and ls_check[0]['exists'])
        if ls_ready:
            row_check = await db_manager.execute_query(
                "SELECT COUNT(*) AS n FROM us_listing_status")
            ls_ready = ls_ready and (row_check and row_check[0]['n'] > 0)

        if ls_ready and symbols_filter:
            symbol_query = """
            SELECT DISTINCT d.symbol
            FROM us_daily d
            JOIN us_listing_status ls ON ls.symbol = d.symbol
            LEFT JOIN us_stock_basic b ON b.symbol = d.symbol AND b.date <= $1
            WHERE d.date = $1
              AND (ls.ipo_date IS NULL OR ls.ipo_date <= $1)
              AND (ls.delisting_date IS NULL OR ls.delisting_date > $1)
              AND (ls.asset_type IS NULL OR ls.asset_type = 'Stock')
              AND d.symbol = ANY($2::text[])
            ORDER BY d.symbol
            """
            symbol_result = await db_manager.execute_query(
                symbol_query, analysis_date, symbols_filter)
        elif ls_ready:
            symbol_query = """
            SELECT DISTINCT d.symbol
            FROM us_daily d
            JOIN us_listing_status ls ON ls.symbol = d.symbol
            WHERE d.date = $1
              AND (ls.ipo_date IS NULL OR ls.ipo_date <= $1)
              AND (ls.delisting_date IS NULL OR ls.delisting_date > $1)
              AND (ls.asset_type IS NULL OR ls.asset_type = 'Stock')
            ORDER BY d.symbol
            """
            symbol_result = await db_manager.execute_query(symbol_query, analysis_date)
        elif symbols_filter:
            # 폴백 (legacy, survivorship-biased)
            symbol_query = """
            SELECT DISTINCT b.symbol
            FROM us_stock_basic b
            INNER JOIN us_daily d ON b.symbol = d.symbol AND d.date = $1
            WHERE b.date <= $1
              AND b.sector IS NOT NULL AND b.sector != ''
              AND b.symbol = ANY($2::text[])
            ORDER BY b.symbol
            """
            symbol_result = await db_manager.execute_query(
                symbol_query, analysis_date, symbols_filter)
            logger.warning(f"PIT universe 불가 (us_listing_status 비어있음) → "
                           f"survivorship-biased universe 사용 @ {analysis_date}")
        else:
            symbol_query = """
            SELECT DISTINCT b.symbol
            FROM us_stock_basic b
            INNER JOIN us_daily d ON b.symbol = d.symbol AND d.date = $1
            WHERE b.date <= $1
              AND b.sector IS NOT NULL AND b.sector != ''
            ORDER BY b.symbol
            """
            symbol_result = await db_manager.execute_query(symbol_query, analysis_date)
            logger.warning(f"PIT universe 불가 → survivorship-biased universe @ {analysis_date}")
        symbols = [r['symbol'] for r in symbol_result] if symbol_result else []

        if not symbols:
            print(f"  [경고] {analysis_date}에 데이터가 있는 종목이 없습니다.\n")
            continue

        print(f"  -> {len(symbols):,}개 종목 발견")

        # 섹터 캐시 초기화
        system.sector_cache = {}

        # Phase 4: SPY 수익률 프리페치 (tail_beta, corr_spy 계산용)
        spy_returns = await system._prefetch_spy_returns(analysis_date)
        if spy_returns:
            print(f"  SPY 수익률 프리페치 완료: {len(spy_returns)}일")
        else:
            print(f"  [경고] SPY 수익률 프리페치 실패 - tail_beta, corr_spy 계산 불가")

        # 배치 처리
        # FAST_BACKFILL_MODE uses larger batches + matching pool size to reduce
        # per-batch overhead. The 25/30 default leaves pool contention as the
        # dominant bottleneck (25 syms × ~5 queries each through pool=30).
        BATCH_SIZE = 50 if os.getenv("FAST_BACKFILL_MODE", "0").lower() in ("1","true","yes") else 25
        batches = [symbols[i:i + BATCH_SIZE] for i in range(0, len(symbols), BATCH_SIZE)]

        date_success = 0
        date_fail = 0
        date_failed_symbols = []  # 날짜별 실패 종목

        for batch_idx, batch in enumerate(batches, 1):
            batch_start_time = time.time()
            tasks = []
            for symbol in batch:
                tasks.append(system._analyze_stock(symbol, analysis_date, spy_returns=spy_returns))

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # 결과 저장
            batch_success = 0
            batch_fail = 0
            batch_errors = []

            for idx, result in enumerate(results):
                symbol = batch[idx]
                if isinstance(result, Exception):
                    date_fail += 1
                    batch_fail += 1
                    date_failed_symbols.append(symbol)
                    batch_errors.append(f"{symbol}: {type(result).__name__}: {str(result)[:100]}")
                elif result:
                    try:
                        await system._save_stock_grade(result, analysis_date)
                        date_success += 1
                        batch_success += 1
                    except Exception as save_err:
                        date_fail += 1
                        batch_fail += 1
                        date_failed_symbols.append(symbol)
                        batch_errors.append(f"{symbol}: DB저장실패-{save_err}")
                else:
                    date_fail += 1
                    batch_fail += 1
                    date_failed_symbols.append(symbol)
                    batch_errors.append(f"{symbol}: 결과없음")

            # 진행 상황
            batch_time = time.time() - batch_start_time
            current_time = datetime.now().strftime("%H:%M:%S")
            print(f"  [{current_time}] 배치 {batch_idx}/{len(batches)} | "
                  f"성공: {batch_success} | 실패: {batch_fail} | "
                  f"시간: {batch_time:.1f}초")

            # 배치 내 에러 상세 (있는 경우만)
            if batch_errors and len(batch_errors) <= 5:
                for err in batch_errors:
                    print(f"    [실패] {err}")
            elif batch_errors:
                # 첫 번째 에러만 출력
                print(f"    [실패] {batch_errors[0]}")
                print(f"    [실패] ... 외 {len(batch_errors)-1}개 종목")

        date_time = time.time() - date_start_time
        success_rate = (date_success / len(symbols) * 100) if symbols else 0

        print(f"\n[날짜 {date_idx}/{len(date_list)}] {analysis_date} 완료")
        print(f"  종목: {len(symbols):,}개 | 성공: {date_success:,} ({success_rate:.1f}%) | "
              f"실패: {date_fail:,} | 시간: {date_time:.1f}초")

        # 실패 종목 요약 (10개 이하면 전체, 초과면 앞 10개)
        if date_failed_symbols:
            if len(date_failed_symbols) <= 10:
                print(f"  실패 종목: {', '.join(date_failed_symbols)}")
            else:
                print(f"  실패 종목 (일부): {', '.join(date_failed_symbols[:10])}... 외 {len(date_failed_symbols)-10}개")
            all_failed_symbols.extend(date_failed_symbols)

        # RS Rank 배치 계산 (해당 날짜의 모든 종목 기준 상대 순위)
        print(f"  RS Rank 계산 중...")
        rs_rank_query = """
        WITH rs_ranked AS (
            SELECT
                symbol,
                rs_value,
                NTILE(100) OVER (ORDER BY rs_value DESC) as percentile
            FROM us_stock_grade
            WHERE date = $1 AND rs_value IS NOT NULL
        )
        UPDATE us_stock_grade g
        SET rs_rank = CASE
            WHEN r.percentile <= 10 THEN '매우강함 (Top 10%)'
            WHEN r.percentile <= 20 THEN '강함 (Top 20%)'
            WHEN r.percentile <= 40 THEN '보통 (Top 40%)'
            WHEN r.percentile <= 60 THEN '약함 (Top 60%)'
            WHEN r.percentile <= 80 THEN '매우약함 (Top 80%)'
            ELSE '최하위 (Bottom 20%)'
        END
        FROM rs_ranked r
        WHERE g.symbol = r.symbol AND g.date = $1
        """
        try:
            await db_manager.execute_query(rs_rank_query, analysis_date)
            print(f"  RS Rank 계산 완료")
        except Exception as e:
            print(f"  [경고] RS Rank 계산 실패: {e}")

        # Phase 3.4.3 Stage 3: Industry Ranking 배치 계산
        print(f"  Industry Rank 계산 중...")
        industry_rank_query = """
        WITH basic AS (
            -- us_stock_basic 는 종목당 수백~수천 행(시점별 'computed' source)이라
            -- g.symbol=b.symbol 직접 조인은 grade 행을 ~1300배로 부풀려 RANK()/
            -- PERCENT_RANK() 가 (468종목 날짜에 industry_rank 최대 17만 같은) 완전히
            -- 왜곡된 값을 만든다. 종목당 최신 1행으로 dedupe 후 조인한다.
            SELECT DISTINCT ON (symbol) symbol, industry
            FROM us_stock_basic
            WHERE industry IS NOT NULL AND industry != ''
            ORDER BY symbol, date DESC
        ),
        industry_rankings AS (
            SELECT
                g.symbol,
                b.industry,
                RANK() OVER (PARTITION BY b.industry ORDER BY g.final_score DESC) as industry_rank,
                ROUND(
                    (PERCENT_RANK() OVER (PARTITION BY b.industry ORDER BY g.final_score DESC) * 100)::NUMERIC,
                    1
                ) as industry_percentile
            FROM us_stock_grade g
            JOIN basic b ON g.symbol = b.symbol
            WHERE g.date = $1
        )
        UPDATE us_stock_grade g
        SET
            industry_rank = r.industry_rank,
            industry_percentile = r.industry_percentile
        FROM industry_rankings r
        WHERE g.symbol = r.symbol AND g.date = $1
        """
        try:
            await db_manager.execute_query(industry_rank_query, analysis_date)
            print(f"  Industry Rank 계산 완료")
        except Exception as e:
            print(f"  [경고] Industry Rank 계산 실패: {e}")

        # Phase 4: vol_percentile 배치 계산 (전체 종목 기준 변동성 백분위)
        print(f"  Volatility Percentile 계산 중...")
        vol_percentile_query = """
        WITH vol_ranked AS (
            SELECT
                symbol,
                volatility_annual,
                NTILE(100) OVER (ORDER BY volatility_annual ASC) as vol_pct
            FROM us_stock_grade
            WHERE date = $1 AND volatility_annual IS NOT NULL
        )
        UPDATE us_stock_grade g
        SET vol_percentile = r.vol_pct
        FROM vol_ranked r
        WHERE g.symbol = r.symbol AND g.date = $1
        """
        try:
            await db_manager.execute_query(vol_percentile_query, analysis_date)
            print(f"  Volatility Percentile 계산 완료")
        except Exception as e:
            print(f"  [경고] Volatility Percentile 계산 실패: {e}")

        # Phase 4: corr_sector_avg 배치 계산 (섹터 내 평균 상관관계)
        print(f"  Sector Correlation Avg 계산 중...")
        corr_sector_avg_query = """
        WITH basic AS (
            -- 종목당 최신 1행으로 dedupe — industry_rank 와 동일 사유(직접 조인 시
            -- ~1300배 중복). AVG 자체는 중복에 불변이라 값은 같았지만, 쓸데없이
            -- 1300배 큰 조인을 피해 성능을 회복한다.
            SELECT DISTINCT ON (symbol) symbol, sector
            FROM us_stock_basic
            WHERE sector IS NOT NULL AND sector != ''
            ORDER BY symbol, date DESC
        ),
        sector_corr AS (
            SELECT
                g.symbol,
                b.sector,
                g.corr_spy,
                AVG(g.corr_spy) OVER (PARTITION BY b.sector) as sector_avg_corr
            FROM us_stock_grade g
            JOIN basic b ON g.symbol = b.symbol
            WHERE g.date = $1
              AND g.corr_spy IS NOT NULL
        )
        UPDATE us_stock_grade g
        SET corr_sector_avg = r.sector_avg_corr
        FROM sector_corr r
        WHERE g.symbol = r.symbol AND g.date = $1
        """
        try:
            await db_manager.execute_query(corr_sector_avg_query, analysis_date)
            print(f"  Sector Correlation Avg 계산 완료")
        except Exception as e:
            print(f"  [경고] Sector Correlation Avg 계산 실패: {e}")

        # Phase 4: Factor rankings batch calculation
        print(f"  Factor Rankings 계산 중...")
        factor_rankings_query = """
        WITH base AS (
            SELECT symbol
            FROM us_stock_grade
            WHERE date = $1
              AND final_grade != '평가 불가'
        ),
        value_ranked AS (
            SELECT
                symbol,
                RANK() OVER (ORDER BY value_score DESC) as rank_num,
                COUNT(*) OVER (PARTITION BY value_score) as tie_count,
                ROUND((PERCENT_RANK() OVER (ORDER BY value_score DESC) * 100)::NUMERIC, 1) as percentile
            FROM us_stock_grade
            WHERE date = $1
              AND final_grade != '평가 불가'
              AND value_score IS NOT NULL
        ),
        quality_ranked AS (
            SELECT
                symbol,
                RANK() OVER (ORDER BY quality_score DESC) as rank_num,
                COUNT(*) OVER (PARTITION BY quality_score) as tie_count,
                ROUND((PERCENT_RANK() OVER (ORDER BY quality_score DESC) * 100)::NUMERIC, 1) as percentile
            FROM us_stock_grade
            WHERE date = $1
              AND final_grade != '평가 불가'
              AND quality_score IS NOT NULL
        ),
        momentum_ranked AS (
            SELECT
                symbol,
                RANK() OVER (ORDER BY momentum_score DESC) as rank_num,
                COUNT(*) OVER (PARTITION BY momentum_score) as tie_count,
                ROUND((PERCENT_RANK() OVER (ORDER BY momentum_score DESC) * 100)::NUMERIC, 1) as percentile
            FROM us_stock_grade
            WHERE date = $1
              AND final_grade != '평가 불가'
              AND momentum_score IS NOT NULL
        ),
        growth_ranked AS (
            SELECT
                symbol,
                RANK() OVER (ORDER BY growth_score DESC) as rank_num,
                COUNT(*) OVER (PARTITION BY growth_score) as tie_count,
                ROUND((PERCENT_RANK() OVER (ORDER BY growth_score DESC) * 100)::NUMERIC, 1) as percentile
            FROM us_stock_grade
            WHERE date = $1
              AND final_grade != '평가 불가'
              AND growth_score IS NOT NULL
        ),
        combined AS (
            SELECT
                b.symbol,
                vr.rank_num as v_rank, vr.tie_count as v_tie, vr.percentile as v_pct,
                qr.rank_num as q_rank, qr.tie_count as q_tie, qr.percentile as q_pct,
                mr.rank_num as m_rank, mr.tie_count as m_tie, mr.percentile as m_pct,
                gr.rank_num as g_rank, gr.tie_count as g_tie, gr.percentile as g_pct
            FROM base b
            LEFT JOIN value_ranked vr ON b.symbol = vr.symbol
            LEFT JOIN quality_ranked qr ON b.symbol = qr.symbol
            LEFT JOIN momentum_ranked mr ON b.symbol = mr.symbol
            LEFT JOIN growth_ranked gr ON b.symbol = gr.symbol
        )
        UPDATE us_stock_grade g
        SET
            value_rank = CASE
                WHEN c.v_rank IS NULL THEN NULL
                WHEN c.v_tie > 1 THEN '공동 ' || c.v_rank || '위'
                ELSE c.v_rank || '위'
            END,
            value_percentile = c.v_pct,
            quality_rank = CASE
                WHEN c.q_rank IS NULL THEN NULL
                WHEN c.q_tie > 1 THEN '공동 ' || c.q_rank || '위'
                ELSE c.q_rank || '위'
            END,
            quality_percentile = c.q_pct,
            momentum_rank = CASE
                WHEN c.m_rank IS NULL THEN NULL
                WHEN c.m_tie > 1 THEN '공동 ' || c.m_rank || '위'
                ELSE c.m_rank || '위'
            END,
            momentum_percentile = c.m_pct,
            growth_rank = CASE
                WHEN c.g_rank IS NULL THEN NULL
                WHEN c.g_tie > 1 THEN '공동 ' || c.g_rank || '위'
                ELSE c.g_rank || '위'
            END,
            growth_percentile = c.g_pct
        FROM combined c
        WHERE g.symbol = c.symbol AND g.date = $1
        """
        try:
            await db_manager.execute_query(factor_rankings_query, analysis_date)
            print(f"  Factor Rankings 계산 완료")
        except Exception as e:
            print(f"  [경고] Factor Rankings 계산 실패: {e}")

        # Alternative stock matching for this date
        print(f"  Alternative Stock 매칭 중...")
        matcher = USAlternativeStockMatcher(db_manager)
        match_result = await matcher.match_for_date(analysis_date)
        print(f"  Alternative Stock 매칭 완료: {match_result['matched']}개 성공, {match_result['failed']}개 실패")
        print()

        date_results[str(analysis_date)] = {
            'success': date_success,
            'fail': date_fail,
            'total': len(symbols),
            'regime': system.current_regime,
            'failed_symbols': date_failed_symbols
        }

        total_success += date_success
        total_fail += date_fail
        total_processed += len(symbols)

    total_time = time.time() - total_start_time

    # 로그 레벨 복구
    for logger_name, level in original_levels.items():
        logging.getLogger(logger_name).setLevel(level)

    # 최종 요약
    print("\n" + "="*80)
    print("전체 종목 특정 날짜 분석 완료 (v2.0)")
    print("="*80)
    print(f"분석 날짜: {len(date_list)}개")
    print(f"총 분석 수: {total_processed:,}개")
    print(f"성공: {total_success:,}개 ({(total_success/total_processed*100) if total_processed > 0 else 0:.1f}%)")
    print(f"실패: {total_fail:,}개 ({(total_fail/total_processed*100) if total_processed > 0 else 0:.1f}%)")
    print(f"총 시간: {total_time/60:.1f}분 ({total_time:.0f}초)")
    if total_processed > 0:
        print(f"평균 속도: {total_processed/total_time:.1f} 종목/초")

    # 날짜별 요약
    print(f"\n날짜별 요약:")
    for d, r in date_results.items():
        rate = (r['success'] / r['total'] * 100) if r['total'] > 0 else 0
        print(f"  {d}: {r['success']:,}/{r['total']:,} ({rate:.1f}%) - {r['regime']}")

    # 전체 실패 종목 (중복 제거)
    if all_failed_symbols:
        unique_failed = list(set(all_failed_symbols))
        print(f"\n실패 종목 총 {len(unique_failed)}개 (고유)")
        if len(unique_failed) <= 20:
            print(f"  {', '.join(sorted(unique_failed))}")

    print("="*80 + "\n")

    return {
        'dates': len(date_list),
        'total_processed': total_processed,
        'total_success': total_success,
        'total_fail': total_fail,
        'total_time': total_time,
        'date_results': date_results,
        'all_failed_symbols': list(set(all_failed_symbols))
    }


def parse_dates(dates_input: str) -> List[date]:
    """날짜 문자열 파싱 (개별 날짜, 범위, 혼합 지원)"""

    segments = [seg.strip() for seg in dates_input.split(',')]
    date_list = []

    for segment in segments:
        if '~' in segment:
            # 날짜 범위 처리
            parts = segment.split('~')
            if len(parts) != 2:
                print(f"잘못된 범위 형식: {segment}")
                continue

            start_str = parts[0].strip()
            end_str = parts[1].strip()

            start_date = datetime.strptime(start_str, '%Y-%m-%d').date()
            end_date = datetime.strptime(end_str, '%Y-%m-%d').date()

            if start_date > end_date:
                print(f"시작 날짜가 종료 날짜보다 늦습니다: {segment}")
                continue

            current_date = start_date
            while current_date <= end_date:
                if current_date not in date_list:
                    date_list.append(current_date)
                current_date += timedelta(days=1)
        else:
            # 개별 날짜 처리
            parsed_date = datetime.strptime(segment, '%Y-%m-%d').date()
            if parsed_date not in date_list:
                date_list.append(parsed_date)

    date_list.sort()
    return date_list


# ========================================================================
# Main Entry Point
# ========================================================================

async def run_option1(target_date=None, symbols=None, with_event_modifier=True,
                      improved_factors=False):
    """
    Option 1: Analyze stocks for given target_date.

    Args:
        target_date: Analysis date (date object). If None, auto-detects from us_daily.
        with_event_modifier: pass-through to analyze_all_stocks_specific_dates;
            False for Pass-A (pre-options ranking), True for Pass-B.
        symbols: Optional whitelist of symbols to analyze (e.g., EM8 pre-filter
                 top-N). If None, full us_stock_basic universe is analyzed.

    Returns:
        dict: Analysis result summary
    """
    from datetime import date as date_type, timedelta

    # Initialize DB pool (max_size=30, Railway Pro optimization)
    db_manager = AsyncDatabaseManager()
    await db_manager.initialize(min_size=10, max_size=50)

    try:
        # Auto-detect latest data date if not specified
        if target_date is None:
            today = date_type.today()
            check_date = today - timedelta(days=1)
            max_attempts = 7
            found_date = None

            for attempt in range(max_attempts):
                result = await db_manager.execute_query(
                    "SELECT COUNT(*) as cnt FROM us_daily WHERE date = $1",
                    check_date
                )
                count = result[0]['cnt'] if result else 0
                if count > 0:
                    found_date = check_date
                    logger.info(f"Data found: {found_date} ({count:,} records)")
                    break
                check_date -= timedelta(days=1)

            if not found_date:
                raise ValueError(f"No data found in us_daily within last {max_attempts} days")

            analysis_date = found_date
        else:
            analysis_date = target_date

        logger.info(f"run_option1 started: {analysis_date}")

        # Run full analysis (with optional symbols whitelist from EM8 pre-filter)
        await analyze_all_stocks_specific_dates(
            db_manager, [analysis_date], symbols_filter=symbols,
            with_event_modifier=with_event_modifier,
            improved_factors=improved_factors)

        # US Prediction Collector
        import sys
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if root_dir not in sys.path:
            sys.path.insert(0, root_dir)

        from us_prediction_collector import (
            get_connection,
            batch_record_grades,
            batch_update_returns,
            update_stats
        )

        pred_conn = await get_connection()
        try:
            grades_count = await batch_record_grades(pred_conn, analysis_date, analysis_date)
            returns_count = await batch_update_returns(pred_conn)
            stats_count = await update_stats(pred_conn)
        finally:
            await pred_conn.close()

        result = {
            "date": str(analysis_date),
            "grades_recorded": grades_count,
            "returns_updated": returns_count,
            "stats_updated": stats_count
        }

        logger.info(f"run_option1 completed: {result}")
        return result

    finally:
        await db_manager.close()


async def main():
    """메인 실행"""

    print("\n" + "="*80)
    print("미국 주식 퀀트 분석 시스템 v2.0")
    print("="*80 + "\n")

    # DB 초기화
    db_manager = AsyncDatabaseManager()
    await db_manager.initialize()

    try:
        while True:
            print("분석 옵션 선택:")
            print("  1. 전체 종목 최신 날짜 분석 (자동 감지)")
            print("  2. 전체 종목 분석 (특정 날짜)")
            print()

            choice = input("선택 (1-2): ").strip()

            if choice == '1':
                # Option 1: Analyze all stocks for latest date (auto-detect)
                print("\n전체 종목 최신 날짜 분석")
                print("-" * 50)
                print("us_daily 테이블에서 가장 최신 데이터 날짜를 자동 감지합니다.")

                print("\n커넥션 풀 최적화 중...")
                await db_manager.close()

                result = await run_option1()

                print("\n" + "="*60)
                print("US Prediction Collector 결과")
                print("="*60)
                print(f"  - 분석 날짜: {result['date']}")
                print(f"  - 등급 기록: {result['grades_recorded']:,}건")
                print(f"  - 수익률 갱신: {result['returns_updated']:,}건")
                print(f"  - 통계 갱신: {result['stats_updated']:,}건")
                print("="*60)
                print("\n프로그램 종료합니다.")
                break

            elif choice == '2':
                # 옵션 2: 전체 종목 분석 (특정 날짜)
                print("\n전체 종목 특정 날짜 분석 (v3.0)")
                print("-" * 50)
                print("\n[HMM 레짐 감지 모드]")
                print("  - 3-State Hidden Markov Model (BULL/NEUTRAL/BEAR)")
                print("  - 정적 가중치 (분기별 리밸런싱 권고)")

                print("\n" + "-" * 50)
                print("분석할 날짜 입력")
                print("사용법:")
                print("  - 개별 날짜: 2025-08-01, 2025-08-05")
                print("  - 날짜 범위: 2025-08-01~2025-08-05 (1일부터 5일까지)")
                print("  - 혼합: 2025-08-01, 2025-08-05~2025-08-08, 2025-09-04")
                dates_input = input("\n날짜 입력: ").strip()

                if not dates_input:
                    print("날짜가 입력되지 않았습니다.\n")
                    continue

                try:
                    date_list = parse_dates(dates_input)

                    if not date_list:
                        print("유효한 날짜가 없습니다.\n")
                        continue

                    print(f"\n입력된 날짜: {len(date_list)}개")
                    for idx, d in enumerate(date_list, 1):
                        print(f"  {idx}. {d}")
                    print(f"레짐 모드: HMM (3-State)")

                    confirm = input("\n분석을 시작하시겠습니까? (y/n): ").strip().lower()
                    if confirm == 'y':
                        # 커넥션 풀 재초기화
                        print("\n커넥션 풀 최적화 중...")
                        await db_manager.close()
                        db_manager = AsyncDatabaseManager()
                        await db_manager.initialize(min_size=10, max_size=50)

                        await analyze_all_stocks_specific_dates(
                            db_manager,
                            date_list
                        )
                        print("\n프로그램 종료합니다.")
                        break
                    else:
                        print("취소되었습니다.\n")

                except ValueError as e:
                    print(f"잘못된 날짜 형식입니다. YYYY-MM-DD 형식을 사용하세요. (오류: {e})\n")
                except Exception as e:
                    print(f"오류 발생: {e}\n")

            else:
                print("잘못된 선택입니다. 다시 시도하세요.\n")

    except KeyboardInterrupt:
        print("\n\n사용자에 의해 중단되었습니다.")

    except Exception as e:
        logger.error(f"오류 발생: {e}")
        print(f"\n오류 발생: {e}")

    finally:
        await db_manager.close()
        print("\n데이터베이스 연결을 종료했습니다.")


if __name__ == '__main__':
    asyncio.run(main())
