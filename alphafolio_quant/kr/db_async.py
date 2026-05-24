"""
Async Database Connection Pool Manager for Korean Stock Analysis

This module provides:
1. Async database connection pool (asyncpg)
2. kr_stock_grade table save functions (UPSERT)
3. Utility functions for async DB operations

Based on us_stock_grade.py patterns
"""

import os
import asyncio
import asyncpg
from typing import Dict, List, Optional, Any
from datetime import datetime, date
from dotenv import load_dotenv
import logging

# Load environment
load_dotenv()

# Logger
logger = logging.getLogger(__name__)

# ===========================
# Database Connection Pool Manager
# ===========================

class AsyncDatabaseManager:
    """
    Async database connection pool manager

    Features:
    - Connection pooling with asyncpg
    - Automatic retry logic
    - Connection health check
    """

    def __init__(self):
        self.connection_pool = None

    async def initialize(self, min_size=10, max_size=45):
        """
        Initialize connection pool (Railway Pro 최적화 - KR+US 동시 실행 지원)

        Args:
            min_size: Minimum pool size (default 10)
            max_size: Maximum pool size (default 45 for concurrent kr+us execution)
        """
        try:
            database_url = os.getenv('DATABASE_URL')
            if not database_url:
                raise ValueError("DATABASE_URL not found in environment variables")

            # Convert to asyncpg format if needed
            db_url = database_url.replace("postgresql+asyncpg://", "postgresql://")

            self.connection_pool = await asyncpg.create_pool(
                db_url,
                min_size=min_size,
                max_size=max_size,
                command_timeout=600,
                timeout=45,  # Connection acquisition timeout (increased for stability)
                max_inactive_connection_lifetime=1800,  # 30 min (match US settings, prevent premature connection close)
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

    async def execute_query(self, query: str, *params) -> List[Dict]:
        """
        Execute query and return results

        Args:
            query: SQL query string
            params: Query parameters

        Returns:
            List of dict results
        """
        max_retries = 3
        for attempt in range(max_retries + 1):
            try:
                async with self.connection_pool.acquire() as conn:
                    # Connection health check
                    try:
                        await conn.fetchval('SELECT 1')
                    except Exception as health_error:
                        # 건강 검사 실패 시 연결을 풀에서 강제 제거
                        logger.warning(f"Connection health check failed (attempt {attempt + 1}): {str(health_error)[:50]}")
                        conn.terminate()  # 끊긴 연결을 풀에서 제거
                        if attempt < max_retries:
                            wait_time = 2 ** attempt
                            await asyncio.sleep(wait_time)
                            continue
                        raise health_error

                    rows = await conn.fetch(query, *params)
                    return [dict(row) for row in rows]

            except Exception as e:
                error_msg = str(e).lower()

                # Retryable errors
                retryable_errors = [
                    "too many clients",
                    "connection was closed",
                    "connection does not exist",
                    "semaphore",  # Windows semaphore timeout
                    "getaddrinfo failed"
                ]

                is_retryable = any(err in error_msg for err in retryable_errors)

                if is_retryable and attempt < max_retries:
                    wait_time = 2 ** (attempt + 1)
                    logger.warning(f"DB retry {attempt + 1}/{max_retries}: {str(e)[:50]}... waiting {wait_time}s")
                    await asyncio.sleep(wait_time)
                    continue

                # Final attempt failed - log error details
                if attempt == max_retries:
                    logger.error(f"Query execution failed: {e}")
                    logger.error(f"Query: {query}")
                    raise

                # Non-retryable errors on first attempts
                wait_time = 2 ** attempt
                await asyncio.sleep(wait_time)

    async def execute(self, query: str, *params) -> str:
        """
        Execute query without returning results (INSERT, UPDATE, DELETE)

        Args:
            query: SQL query string
            params: Query parameters

        Returns:
            Status string
        """
        max_retries = 3
        for attempt in range(max_retries + 1):
            try:
                async with self.connection_pool.acquire() as conn:
                    # Connection health check
                    try:
                        await conn.fetchval('SELECT 1')
                    except Exception as health_error:
                        logger.warning(f"Execute health check failed (attempt {attempt + 1}): {str(health_error)[:50]}")
                        conn.terminate()
                        if attempt < max_retries:
                            wait_time = 2 ** attempt
                            await asyncio.sleep(wait_time)
                            continue
                        raise health_error

                    return await conn.execute(query, *params)
            except Exception as e:
                error_msg = str(e).lower()
                retryable_errors = ["connection was closed", "connection does not exist", "semaphore"]
                is_retryable = any(err in error_msg for err in retryable_errors)

                if is_retryable and attempt < max_retries:
                    wait_time = 2 ** (attempt + 1)
                    logger.warning(f"DB execute retry {attempt + 1}/{max_retries}: {str(e)[:50]}...")
                    await asyncio.sleep(wait_time)
                    continue
                raise

    async def executemany(self, query: str, params_list: List[tuple]) -> None:
        """
        Execute batch insert/update

        Args:
            query: SQL query string
            params_list: List of parameter tuples
        """
        max_retries = 3
        for attempt in range(max_retries + 1):
            try:
                async with self.connection_pool.acquire() as conn:
                    # Connection health check
                    try:
                        await conn.fetchval('SELECT 1')
                    except Exception as health_error:
                        logger.warning(f"Executemany health check failed (attempt {attempt + 1}): {str(health_error)[:50]}")
                        conn.terminate()
                        if attempt < max_retries:
                            wait_time = 2 ** attempt
                            await asyncio.sleep(wait_time)
                            continue
                        raise health_error

                    await conn.executemany(query, params_list)
                    return
            except Exception as e:
                error_msg = str(e).lower()
                retryable_errors = ["connection was closed", "connection does not exist", "semaphore"]
                is_retryable = any(err in error_msg for err in retryable_errors)

                if is_retryable and attempt < max_retries:
                    wait_time = 2 ** (attempt + 1)
                    logger.warning(f"DB executemany retry {attempt + 1}/{max_retries}: {str(e)[:50]}...")
                    await asyncio.sleep(wait_time)
                    continue
                raise

    async def close(self):
        """Close connection pool"""
        if self.connection_pool:
            await self.connection_pool.close()
            logger.info("Database connection pool closed")

    async def refresh_sector_performance_for_date(self, target_date: date) -> bool:
        """
        특정 날짜의 섹터 성과 데이터를 Materialized View에 갱신

        목적: calculate_sector_rotation_signal() 메모리 오버플로우 방지
        - 복잡한 JOIN (2,748 종목 × 30일) -> 사전 계산된 집계 데이터 조회
        - 메모리 사용량: 5GB -> 10MB (500배 감소)
        - 실행 속도: 5-10초 -> 0.01초 (1000배 개선)

        Args:
            target_date: 갱신할 날짜 (YYYY-MM-DD)

        Returns:
            bool: 성공 시 True, 실패 시 False

        동작:
            1. mv_sector_daily_performance에 해당 날짜 데이터 존재 확인
            2. 존재하면 SKIP (중복 방지)
            3. 존재하지 않으면 섹터별 집계 계산 후 INSERT

        사용 위치:
            - kr_main.py::analyze_all_stocks_specific_dates()
            - 각 날짜 분석 시작 전 자동 호출
        """
        try:
            # Step 1: 해당 날짜의 데이터가 이미 존재하는지 확인
            check_query = """
            SELECT COUNT(*) as cnt
            FROM mv_sector_daily_performance
            WHERE date = $1
            """

            result = await self.execute_query(check_query, target_date)

            if result and result[0]['cnt'] > 0:
                logger.debug(f"[MV Refresh] {target_date} 섹터 데이터 이미 존재 (SKIP)")
                return True

            # Step 2: 데이터가 없으면 계산 후 INSERT
            logger.info(f"[MV Refresh] {target_date} 섹터 데이터 생성 시작...")

            refresh_query = """
            INSERT INTO mv_sector_daily_performance (date, sector_code, avg_return_30d, stock_count, sector_rank)
            SELECT
                $1::date as date,
                s.theme as sector_code,
                AVG(
                    (d.close - d_30d.close)::NUMERIC / NULLIF(d_30d.close, 0) * 100
                ) as avg_return_30d,
                COUNT(DISTINCT s.symbol) as stock_count,
                ROW_NUMBER() OVER (
                    ORDER BY AVG(
                        (d.close - d_30d.close)::NUMERIC / NULLIF(d_30d.close, 0) * 100
                    ) DESC NULLS LAST
                ) as sector_rank
            FROM kr_stock_detail s
            INNER JOIN kr_intraday_total d
                ON s.symbol = d.symbol
                AND d.date = $1
            LEFT JOIN kr_intraday_total d_30d
                ON s.symbol = d_30d.symbol
                AND d_30d.date = (
                    SELECT MAX(date)
                    FROM kr_intraday_total
                    WHERE symbol = s.symbol
                    AND date <= $1 - INTERVAL '30 days'
                    AND date >= $1 - INTERVAL '40 days'
                )
            WHERE
                d.close IS NOT NULL
                AND d_30d.close IS NOT NULL
                AND d_30d.date IS NOT NULL
                AND s.theme IS NOT NULL
                AND s.theme != ''
            GROUP BY s.theme
            ON CONFLICT (date, sector_code) DO NOTHING
            """

            await self.execute(refresh_query, target_date)

            # Step 3: 삽입된 섹터 개수 확인
            count_query = """
            SELECT COUNT(*) as cnt
            FROM mv_sector_daily_performance
            WHERE date = $1
            """

            count_result = await self.execute_query(count_query, target_date)
            sector_count = count_result[0]['cnt'] if count_result else 0

            logger.info(f"[MV Refresh] {target_date} 섹터 데이터 생성 완료 ({sector_count}개 섹터)")
            return True

        except Exception as e:
            logger.error(f"[MV Refresh] {target_date} 섹터 데이터 갱신 실패: {e}")
            return False

# ===========================
# kr_stock_grade Table Save Functions
# ===========================

async def save_trading_halted_to_kr_stock_grade(
    db_manager: AsyncDatabaseManager,
    symbol: str,
    analysis_date: date,
    stock_name: str = None
) -> bool:
    """
    Save trading halted stock with minimal columns (only stock_name, symbol, date, final_grade)

    Trading halted stocks are identified by: open = 0 or null AND close > 0
    These stocks skip full quant analysis and only store basic info.

    Args:
        db_manager: AsyncDatabaseManager instance
        symbol: Stock symbol
        analysis_date: Analysis date
        stock_name: Stock name (optional)

    Returns:
        bool: True if successful
    """
    try:
        query = """
        INSERT INTO kr_stock_grade (symbol, date, stock_name, final_grade)
        VALUES ($1, $2, $3, $4)
        ON CONFLICT (symbol, date)
        DO UPDATE SET
            stock_name = EXCLUDED.stock_name,
            final_grade = EXCLUDED.final_grade
        """
        await db_manager.execute_query(
            query, symbol, analysis_date, stock_name, '거래 정지'
        )
        return True
    except Exception as e:
        logger.error(f"Failed to save trading halted stock {symbol}: {e}")
        return False


async def save_to_kr_stock_grade(
    db_manager: AsyncDatabaseManager,
    symbol: str,
    analysis_date: date,
    data: Dict[str, Any]
) -> bool:
    """
    Save single stock analysis to kr_stock_grade table (UPSERT)

    Args:
        db_manager: AsyncDatabaseManager instance
        symbol: Stock symbol
        analysis_date: Analysis date
        data: Analysis result dictionary

    Returns:
        bool: True if successful

    Data dict structure:
        {
            'final_grade': str,
            'final_score': float,
            'value_score': float,
            'quality_score': float,
            'momentum_score': float,
            'growth_score': float,
            'confidence_score': float,
            'expected_range_3m_min': int,
            'expected_range_3m_max': int,
            'expected_range_1y_min': int,
            'expected_range_1y_max': int,
            'var_95': float,
            'inst_net_30d': int,
            'foreign_net_30d': int,
            'value_momentum': float,
            'quality_momentum': float,
            'momentum_momentum': float,
            'growth_momentum': float,
            'industry_rank': int,
            'industry_percentile': float,
            'beta': float,
            'volatility_annual': float,
            'max_drawdown_1y': float,
            'risk_profile_text': str,
            'risk_recommendation': str,
            'time_series_text': str,
            'signal_overall': str,
            'market_state': str
        }
    """
    try:
        # Validate DECIMAL(5,1) fields before saving
        score_fields = {
            'final_score': data.get('final_score'),
            'value_score': data.get('value_score'),
            'quality_score': data.get('quality_score'),
            'momentum_score': data.get('momentum_score'),
            'growth_score': data.get('growth_score'),
            'confidence_score': data.get('confidence_score'),
            'industry_percentile': data.get('industry_percentile')
        }

        # Check for overflow risk (DECIMAL(5,1) max: 9999.9)
        overflow_detected = False
        for field_name, value in score_fields.items():
            if value is not None and abs(value) >= 10000:
                logger.error(f"[{symbol}] OVERFLOW: {field_name} = {value:.2f} (limit: 9999.9)")
                overflow_detected = True

        if overflow_detected:
            logger.error(f"[{symbol}] All score values: {score_fields}")

        # Validate DECIMAL(5,2) risk metrics and clip to safe range
        # DECIMAL(5,2) limit: -999.99 ~ 999.99
        risk_metrics = {
            'volatility_annual': data.get('volatility_annual'),
            'beta': data.get('beta'),
            'max_drawdown_1y': data.get('max_drawdown_1y'),
            'var_95': data.get('var_95'),
            'cvar_95': data.get('cvar_95')
        }

        for field_name, value in risk_metrics.items():
            if value is not None:
                if abs(value) >= 1000:
                    original_value = value
                    clipped_value = 999.99 if value > 0 else -999.99
                    data[field_name] = clipped_value
                    logger.warning(
                        f"[{symbol}] CLIPPED {field_name}: {original_value:.2f} -> {clipped_value:.2f} "
                        f"(DECIMAL(5,2) limit)"
                    )

        # Validate DECIMAL(8,4) ratio metrics and clip to safe range
        # DECIMAL(8,4) limit: -9999.9999 ~ 9999.9999
        ratio_metrics = {
            'sharpe_ratio': data.get('sharpe_ratio'),
            'sortino_ratio': data.get('sortino_ratio'),
            'calmar_ratio': data.get('calmar_ratio'),
        }

        for field_name, value in ratio_metrics.items():
            if value is not None:
                if abs(value) >= 10000:
                    original_value = value
                    clipped_value = 9999.9999 if value > 0 else -9999.9999
                    data[field_name] = clipped_value
                    logger.warning(
                        f"[{symbol}] CLIPPED {field_name}: {original_value:.4f} -> {clipped_value:.4f} "
                        f"(DECIMAL(8,4) limit)"
                    )

        # Phase Agent: 컬럼 구조 변경 (2025-11-27)
        # 제거: base_*_score, expected_range_*, support_*, resistance_*, supertrend_*, trend, signal
        # 제거: smart_money_*, volatility_context_*, score_momentum_*, score_change_*
        # 추가: cvar_95, entry_timing_*, atr_*, stop_loss_*, take_profit_*, risk_reward_*
        # 추가: position_size_*, scenario_*, buy/sell/hold_triggers
        query = """
        INSERT INTO kr_stock_grade (
            stock_name, symbol, date, final_grade, final_score,
            value_score, quality_score, momentum_score, growth_score,
            confidence_score,
            var_95, cvar_95,
            hurst_exponent, var_95_ewma, var_95_5d, var_95_20d, var_95_60d, var_99, var_99_60d,
            inv_vol_weight, downside_vol, vol_percentile, atr_20d, atr_pct_20d,
            volatility_annual, max_drawdown_1y, beta,
            inst_net_30d, foreign_net_30d,
            value_momentum, quality_momentum, momentum_momentum, growth_momentum,
            industry_rank, industry_percentile,
            rs_value, rs_rank,
            factor_combination_bonus,
            sector_rotation_score, sector_momentum, sector_rank, sector_percentile,
            entry_timing_score, score_trend_2w, price_position_52w,
            atr_pct, stop_loss_pct, take_profit_pct, risk_reward_ratio, position_size_pct,
            scenario_bullish_prob, scenario_sideways_prob, scenario_bearish_prob,
            scenario_bullish_return, scenario_sideways_return, scenario_bearish_return, scenario_sample_count,
            buy_triggers, sell_triggers, hold_triggers,
            risk_profile_text, risk_recommendation,
            time_series_text, signal_overall,
            market_state, created_at,
            value_v2_detail, quality_v2_detail, momentum_v2_detail, growth_v2_detail,
            sharpe_ratio, sortino_ratio, calmar_ratio,
            conviction_score, outlier_risk_score, risk_flag,
            cvar_99, corr_kospi, corr_sector_avg, tail_beta, drawdown_duration_avg
        ) VALUES (
            $1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
            $11, $12, $13, $14, $15, $16, $17, $18, $19, $20,
            $21, $22, $23, $24, $25, $26, $27, $28, $29, $30,
            $31, $32, $33, $34, $35, $36, $37, $38, $39, $40,
            $41, $42, $43, $44, $45, $46, $47, $48, $49, $50,
            $51, $52, $53, $54, $55, $56, $57, $58, $59, $60,
            $61, $62, $63, $64, $65, $66, $67, $68, $69, $70,
            $71, $72, $73, $74, $75, $76, $77, $78, $79, $80, $81
        )
        ON CONFLICT (symbol, date)
        DO UPDATE SET
            stock_name = EXCLUDED.stock_name,
            final_grade = EXCLUDED.final_grade,
            final_score = EXCLUDED.final_score,
            value_score = EXCLUDED.value_score,
            quality_score = EXCLUDED.quality_score,
            momentum_score = EXCLUDED.momentum_score,
            growth_score = EXCLUDED.growth_score,
            confidence_score = EXCLUDED.confidence_score,
            var_95 = EXCLUDED.var_95,
            cvar_95 = EXCLUDED.cvar_95,
            hurst_exponent = EXCLUDED.hurst_exponent,
            var_95_ewma = EXCLUDED.var_95_ewma,
            var_95_5d = EXCLUDED.var_95_5d,
            var_95_20d = EXCLUDED.var_95_20d,
            var_95_60d = EXCLUDED.var_95_60d,
            var_99 = EXCLUDED.var_99,
            var_99_60d = EXCLUDED.var_99_60d,
            inv_vol_weight = EXCLUDED.inv_vol_weight,
            downside_vol = EXCLUDED.downside_vol,
            vol_percentile = EXCLUDED.vol_percentile,
            atr_20d = EXCLUDED.atr_20d,
            atr_pct_20d = EXCLUDED.atr_pct_20d,
            volatility_annual = EXCLUDED.volatility_annual,
            max_drawdown_1y = EXCLUDED.max_drawdown_1y,
            beta = EXCLUDED.beta,
            inst_net_30d = EXCLUDED.inst_net_30d,
            foreign_net_30d = EXCLUDED.foreign_net_30d,
            value_momentum = EXCLUDED.value_momentum,
            quality_momentum = EXCLUDED.quality_momentum,
            momentum_momentum = EXCLUDED.momentum_momentum,
            growth_momentum = EXCLUDED.growth_momentum,
            industry_rank = EXCLUDED.industry_rank,
            industry_percentile = EXCLUDED.industry_percentile,
            rs_value = EXCLUDED.rs_value,
            rs_rank = EXCLUDED.rs_rank,
            factor_combination_bonus = EXCLUDED.factor_combination_bonus,
            sector_rotation_score = EXCLUDED.sector_rotation_score,
            sector_momentum = EXCLUDED.sector_momentum,
            sector_rank = EXCLUDED.sector_rank,
            sector_percentile = EXCLUDED.sector_percentile,
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
            risk_profile_text = EXCLUDED.risk_profile_text,
            risk_recommendation = EXCLUDED.risk_recommendation,
            time_series_text = EXCLUDED.time_series_text,
            signal_overall = EXCLUDED.signal_overall,
            market_state = EXCLUDED.market_state,
            value_v2_detail = EXCLUDED.value_v2_detail,
            quality_v2_detail = EXCLUDED.quality_v2_detail,
            momentum_v2_detail = EXCLUDED.momentum_v2_detail,
            growth_v2_detail = EXCLUDED.growth_v2_detail,
            sharpe_ratio = EXCLUDED.sharpe_ratio,
            sortino_ratio = EXCLUDED.sortino_ratio,
            calmar_ratio = EXCLUDED.calmar_ratio,
            conviction_score = EXCLUDED.conviction_score,
            outlier_risk_score = EXCLUDED.outlier_risk_score,
            risk_flag = EXCLUDED.risk_flag,
            cvar_99 = EXCLUDED.cvar_99,
            corr_kospi = EXCLUDED.corr_kospi,
            corr_sector_avg = EXCLUDED.corr_sector_avg,
            tail_beta = EXCLUDED.tail_beta,
            drawdown_duration_avg = EXCLUDED.drawdown_duration_avg
        """

        await db_manager.execute(
            query,
            data.get('stock_name'),
            symbol,
            analysis_date,
            data.get('final_grade'),
            data.get('final_score'),
            data.get('value_score'),
            data.get('quality_score'),
            data.get('momentum_score'),
            data.get('growth_score'),
            data.get('confidence_score'),
            data.get('var_95'),
            data.get('cvar_95'),
            data.get('hurst_exponent'),
            data.get('var_95_ewma'),
            data.get('var_95_5d'),
            data.get('var_95_20d'),
            data.get('var_95_60d'),
            data.get('var_99'),
            data.get('var_99_60d'),
            data.get('inv_vol_weight'),
            data.get('downside_vol'),
            data.get('vol_percentile'),
            data.get('atr_20d'),
            data.get('atr_pct_20d'),
            data.get('volatility_annual'),
            data.get('max_drawdown_1y'),
            data.get('beta'),
            data.get('inst_net_30d'),
            data.get('foreign_net_30d'),
            data.get('value_momentum'),
            data.get('quality_momentum'),
            data.get('momentum_momentum'),
            data.get('growth_momentum'),
            data.get('industry_rank'),
            data.get('industry_percentile'),
            data.get('rs_value'),
            data.get('rs_rank'),
            data.get('factor_combination_bonus'),
            data.get('sector_rotation_score'),
            data.get('sector_momentum'),
            data.get('sector_rank'),
            data.get('sector_percentile'),
            data.get('entry_timing_score'),
            data.get('score_trend_2w'),
            data.get('price_position_52w'),
            data.get('atr_pct'),
            data.get('stop_loss_pct'),
            data.get('take_profit_pct'),
            data.get('risk_reward_ratio'),
            data.get('position_size_pct'),
            data.get('scenario_bullish_prob'),
            data.get('scenario_sideways_prob'),
            data.get('scenario_bearish_prob'),
            data.get('scenario_bullish_return'),
            data.get('scenario_sideways_return'),
            data.get('scenario_bearish_return'),
            data.get('scenario_sample_count'),
            data.get('buy_triggers'),
            data.get('sell_triggers'),
            data.get('hold_triggers'),
            data.get('risk_profile_text'),
            data.get('risk_recommendation'),
            data.get('time_series_text'),
            data.get('signal_overall'),
            data.get('market_state'),
            datetime.now(),
            data.get('value_v2_detail'),
            data.get('quality_v2_detail'),
            data.get('momentum_v2_detail'),
            data.get('growth_v2_detail'),
            data.get('sharpe_ratio'),
            data.get('sortino_ratio'),
            data.get('calmar_ratio'),
            data.get('conviction_score'),
            data.get('outlier_risk_score'),
            data.get('risk_flag'),
            data.get('cvar_99'),
            data.get('corr_kospi'),
            data.get('corr_sector_avg'),
            data.get('tail_beta'),
            data.get('drawdown_duration_avg')
        )

        logger.debug(f"Saved {symbol} to kr_stock_grade (date: {analysis_date})")
        return True

    except Exception as e:
        logger.error(f"Failed to save {symbol} to kr_stock_grade: {e}")
        return False


async def save_to_kr_stock_grade_light(
    db_manager: AsyncDatabaseManager,
    symbol: str,
    analysis_date: date,
    data: Dict[str, Any]
) -> bool:
    """
    Save light analysis to kr_stock_grade (updates only specified fields, preserves others)

    Args:
        db_manager: AsyncDatabaseManager instance
        symbol: Stock symbol
        analysis_date: Analysis date
        data: Analysis result dictionary (light version)

    Returns:
        bool: True if successful
    """
    try:
        # Light mode: only update core fields, preserve existing data for other columns
        query = """
        INSERT INTO kr_stock_grade (
            stock_name, symbol, date, final_grade, final_score,
            value_score, quality_score, momentum_score, growth_score,
            confidence_score, market_state,
            rs_value, sector_rotation_score, factor_combination_bonus, entry_timing_score,
            created_at
        ) VALUES (
            $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16
        )
        ON CONFLICT (symbol, date)
        DO UPDATE SET
            stock_name = EXCLUDED.stock_name,
            final_grade = EXCLUDED.final_grade,
            final_score = EXCLUDED.final_score,
            value_score = EXCLUDED.value_score,
            quality_score = EXCLUDED.quality_score,
            momentum_score = EXCLUDED.momentum_score,
            growth_score = EXCLUDED.growth_score,
            confidence_score = EXCLUDED.confidence_score,
            market_state = EXCLUDED.market_state,
            rs_value = EXCLUDED.rs_value,
            sector_rotation_score = EXCLUDED.sector_rotation_score,
            factor_combination_bonus = EXCLUDED.factor_combination_bonus,
            entry_timing_score = EXCLUDED.entry_timing_score
        """

        await db_manager.execute(
            query,
            data.get('stock_name'),
            symbol,
            analysis_date,
            data.get('final_grade'),
            data.get('final_score'),
            data.get('value_score'),
            data.get('quality_score'),
            data.get('momentum_score'),
            data.get('growth_score'),
            data.get('confidence_score'),
            data.get('market_state'),
            data.get('rs_value'),
            data.get('sector_rotation_score'),
            data.get('factor_combination_bonus'),
            data.get('entry_timing_score'),
            datetime.now()
        )

        logger.debug(f"Saved {symbol} to kr_stock_grade [LIGHT] (date: {analysis_date})")
        return True

    except Exception as e:
        logger.error(f"Failed to save {symbol} to kr_stock_grade [LIGHT]: {e}")
        return False


async def batch_save_to_kr_stock_grade(
    db_manager: AsyncDatabaseManager,
    results: List[Dict[str, Any]]
) -> int:
    """
    Batch save multiple stock analysis results to kr_stock_grade

    Args:
        db_manager: AsyncDatabaseManager instance
        results: List of analysis results
                 Each result: {'symbol': str, 'date': date, 'data': dict}

    Returns:
        int: Number of successfully saved records
    """
    try:
        query = """
        INSERT INTO kr_stock_grade (
            symbol, date, final_grade, final_score,
            value_score, quality_score, momentum_score, growth_score,
            confidence_score,
            expected_range_3m_min, expected_range_3m_max,
            expected_range_1y_min, expected_range_1y_max,
            var_95, inst_net_30d, foreign_net_30d,
            value_momentum, quality_momentum, momentum_momentum, growth_momentum,
            industry_rank, industry_percentile,
            beta, volatility_annual, max_drawdown_1y,
            risk_profile_text, risk_recommendation,
            time_series_text, signal_overall,
            support_1, support_2, resistance_1, resistance_2,
            supertrend_value, trend, signal, rs_value, rs_rank,
            market_state, created_at
        ) VALUES (
            $1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
            $11, $12, $13, $14, $15, $16, $17, $18, $19, $20,
            $21, $22, $23, $24, $25, $26, $27, $28, $29, $30,
            $31, $32, $33, $34, $35, $36, $37, $38, $39, $40
        )
        ON CONFLICT (symbol, date)
        DO UPDATE SET
            final_grade = EXCLUDED.final_grade,
            final_score = EXCLUDED.final_score,
            value_score = EXCLUDED.value_score,
            quality_score = EXCLUDED.quality_score,
            momentum_score = EXCLUDED.momentum_score,
            growth_score = EXCLUDED.growth_score,
            confidence_score = EXCLUDED.confidence_score,
            expected_range_3m_min = EXCLUDED.expected_range_3m_min,
            expected_range_3m_max = EXCLUDED.expected_range_3m_max,
            expected_range_1y_min = EXCLUDED.expected_range_1y_min,
            expected_range_1y_max = EXCLUDED.expected_range_1y_max,
            var_95 = EXCLUDED.var_95,
            inst_net_30d = EXCLUDED.inst_net_30d,
            foreign_net_30d = EXCLUDED.foreign_net_30d,
            value_momentum = EXCLUDED.value_momentum,
            quality_momentum = EXCLUDED.quality_momentum,
            momentum_momentum = EXCLUDED.momentum_momentum,
            growth_momentum = EXCLUDED.growth_momentum,
            industry_rank = EXCLUDED.industry_rank,
            industry_percentile = EXCLUDED.industry_percentile,
            beta = EXCLUDED.beta,
            volatility_annual = EXCLUDED.volatility_annual,
            max_drawdown_1y = EXCLUDED.max_drawdown_1y,
            risk_profile_text = EXCLUDED.risk_profile_text,
            risk_recommendation = EXCLUDED.risk_recommendation,
            time_series_text = EXCLUDED.time_series_text,
            signal_overall = EXCLUDED.signal_overall,
            support_1 = EXCLUDED.support_1,
            support_2 = EXCLUDED.support_2,
            resistance_1 = EXCLUDED.resistance_1,
            resistance_2 = EXCLUDED.resistance_2,
            supertrend_value = EXCLUDED.supertrend_value,
            trend = EXCLUDED.trend,
            signal = EXCLUDED.signal,
            rs_value = EXCLUDED.rs_value,
            rs_rank = EXCLUDED.rs_rank,
            market_state = EXCLUDED.market_state
        """

        # Prepare batch parameters
        params_list = []
        for result in results:
            symbol = result['symbol']
            analysis_date = result['date']
            data = result['data']

            params_list.append((
                symbol,
                analysis_date,
                data.get('final_grade'),
                data.get('final_score'),
                data.get('value_score'),
                data.get('quality_score'),
                data.get('momentum_score'),
                data.get('growth_score'),
                data.get('confidence_score'),
                data.get('expected_range_3m_min'),
                data.get('expected_range_3m_max'),
                data.get('expected_range_1y_min'),
                data.get('expected_range_1y_max'),
                data.get('var_95'),
                data.get('inst_net_30d'),
                data.get('foreign_net_30d'),
                data.get('value_momentum'),
                data.get('quality_momentum'),
                data.get('momentum_momentum'),
                data.get('growth_momentum'),
                data.get('industry_rank'),
                data.get('industry_percentile'),
                data.get('beta'),
                data.get('volatility_annual'),
                data.get('max_drawdown_1y'),
                data.get('risk_profile_text'),
                data.get('risk_recommendation'),
                data.get('time_series_text'),
                data.get('signal_overall'),
                data.get('support_1'),
                data.get('support_2'),
                data.get('resistance_1'),
                data.get('resistance_2'),
                data.get('supertrend_value'),
                data.get('trend'),
                data.get('signal'),
                data.get('rs_value'),
                data.get('rs_rank'),
                data.get('market_state'),
                datetime.now()
            ))

        # Execute batch insert
        await db_manager.executemany(query, params_list)

        logger.info(f"Batch saved {len(results)} records to kr_stock_grade")
        return len(results)

    except Exception as e:
        logger.error(f"Failed to batch save to kr_stock_grade: {e}")
        return 0


# ===========================
# Utility Functions
# ===========================

def extract_grade_data_from_analyzer(analyzer) -> Dict[str, Any]:
    """
    Extract kr_stock_grade data from StockAnalyzer instance

    Args:
        analyzer: StockAnalyzer instance (after analyze() called)

    Returns:
        dict: Data for kr_stock_grade table
    """
    data = {
        'final_grade': analyzer.final_grade,
        'final_score': analyzer.final_score,
        'market_state': analyzer.market_state
    }

    # Factor scores
    if analyzer.weighted_factor_scores:
        data['value_score'] = analyzer.weighted_factor_scores.get('value', {}).get('weighted_contribution')
        data['quality_score'] = analyzer.weighted_factor_scores.get('quality', {}).get('weighted_contribution')
        data['momentum_score'] = analyzer.weighted_factor_scores.get('momentum', {}).get('weighted_contribution')
        data['growth_score'] = analyzer.weighted_factor_scores.get('growth', {}).get('weighted_contribution')

    # Additional metrics
    if analyzer.additional_metrics:
        m = analyzer.additional_metrics
        data['confidence_score'] = m.get('confidence_score')
        data['expected_range_3m_min'] = m.get('expected_range_3m_min')
        data['expected_range_3m_max'] = m.get('expected_range_3m_max')
        data['expected_range_1y_min'] = m.get('expected_range_1y_min')
        data['expected_range_1y_max'] = m.get('expected_range_1y_max')
        data['var_95'] = m.get('var_95')
        data['inst_net_30d'] = m.get('inst_net_30d')
        data['foreign_net_30d'] = m.get('foreign_net_30d')
        data['value_momentum'] = m.get('value_momentum')
        data['quality_momentum'] = m.get('quality_momentum')
        data['momentum_momentum'] = m.get('momentum_momentum')
        data['growth_momentum'] = m.get('growth_momentum')
        data['industry_rank'] = m.get('industry_rank')
        data['industry_percentile'] = m.get('industry_percentile')
        data['beta'] = m.get('beta')
        data['volatility_annual'] = m.get('volatility_annual')
        data['max_drawdown_1y'] = m.get('max_drawdown_1y')

    # Interpretations
    if analyzer.interpretations:
        interp = analyzer.interpretations

        risk_profile = interp.get('risk_profile', {})
        data['risk_profile_text'] = risk_profile.get('interpretation')
        data['risk_recommendation'] = risk_profile.get('recommendation')

        time_series = interp.get('time_series_trend', {})
        data['time_series_text'] = time_series.get('interpretation')

        signals = interp.get('signal_lights', {})
        data['signal_overall'] = signals.get('overall_assessment')

    return data
