# -*- coding: utf-8 -*-
"""
KR Stock Prediction History & Stats Collector

한국 종목의 예측 이력 수집 및 적중률 통계 갱신
- kr_stock_prediction_history: 일별 등급 이력 기록
- kr_stock_prediction_stats: 종목별 적중률 집계

효율화 적용:
- 데이터 일괄 조회 (1 SELECT로 모든 데이터 가져오기)
- 배치 INSERT (unnest 사용)
- 배치 UPDATE (unnest 사용)
- 메모리 내 계산 후 일괄 저장

Created: 2026-01-21
"""

import os
import asyncio
import asyncpg
import logging
from datetime import date, timedelta
from typing import List, Dict, Tuple, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Database Connection
# =============================================================================

async def get_connection() -> asyncpg.Connection:
    """Get database connection"""
    database_url = os.getenv('DATABASE_URL')
    if not database_url:
        raise ValueError("DATABASE_URL not found in environment variables")

    db_url = database_url.replace("postgresql+asyncpg://", "postgresql://")
    return await asyncpg.connect(db_url, timeout=120)


# =============================================================================
# Batch Record Grades (효율화: 1 SELECT + 1 INSERT)
# =============================================================================

async def batch_record_grades(conn: asyncpg.Connection, start_date: date, end_date: date) -> int:
    """
    기간 내 모든 등급을 한 번에 prediction_history에 기록

    효율화:
    - 1번의 SELECT로 모든 등급 데이터 조회
    - 1번의 INSERT로 모든 데이터 저장 (ON CONFLICT 처리)

    Args:
        conn: Database connection
        start_date: 시작일
        end_date: 종료일

    Returns:
        기록된 레코드 수
    """
    logger.info(f"Recording grades from {start_date} to {end_date}")

    # 1 SELECT: 기간 내 모든 등급 조회
    query = """
    INSERT INTO kr_stock_prediction_history (symbol, grade_date, predicted_grade, grade_direction)
    SELECT
        symbol,
        date as grade_date,
        final_grade as predicted_grade,
        CASE
            WHEN final_grade IN ('강력 매수', '매수', '매수 고려') THEN 'BUY'
            WHEN final_grade IN ('매도 고려', '매도', '강력 매도') THEN 'SELL'
        END as grade_direction
    FROM kr_stock_grade
    WHERE date BETWEEN $1 AND $2
      AND final_grade != '중립'
      AND final_grade IN ('강력 매수', '매수', '매수 고려', '매도 고려', '매도', '강력 매도')
    ON CONFLICT (symbol, grade_date) DO UPDATE SET
        predicted_grade = EXCLUDED.predicted_grade,
        grade_direction = EXCLUDED.grade_direction,
        updated_at = NOW()
    """

    result = await conn.execute(query, start_date, end_date)
    count = int(result.split()[-1]) if result else 0
    logger.info(f"Recorded grades: {count} records")
    return count


# =============================================================================
# Batch Update Returns (효율화: 1 SELECT + 1 UPDATE with unnest)
# =============================================================================

async def batch_update_returns(conn: asyncpg.Connection) -> int:
    """
    90일 이상 지난 모든 등급의 실제 수익률을 한 번에 갱신

    효율화:
    - 1번의 SELECT로 갱신 필요한 모든 데이터 조회
    - 1번의 UPDATE (unnest)로 모든 수익률 갱신

    Returns:
        갱신된 레코드 수
    """
    cutoff_date = date.today() - timedelta(days=90)
    logger.info(f"Updating returns for grades before {cutoff_date}")

    # Step 1: 갱신 필요한 모든 데이터 한 번에 조회 (1 SELECT)
    query = """
    SELECT
        h.symbol,
        h.grade_date,
        p1.close as start_price,
        p2.close as end_price
    FROM kr_stock_prediction_history h
    JOIN kr_intraday_total p1 ON h.symbol = p1.symbol AND h.grade_date = p1.date
    JOIN kr_intraday_total p2 ON h.symbol = p2.symbol AND p2.date = h.grade_date + INTERVAL '90 days'
    WHERE h.grade_date <= $1
      AND h.actual_return_90d IS NULL
      AND p1.close > 0
      AND p2.close IS NOT NULL
    """

    rows = await conn.fetch(query, cutoff_date)

    if not rows:
        logger.info("No records to update")
        return 0

    # Step 2: 메모리에서 수익률 계산
    symbols = []
    grade_dates = []
    returns = []

    for row in rows:
        start_price = float(row['start_price'])
        end_price = float(row['end_price'])
        if start_price > 0:
            actual_return = (end_price - start_price) / start_price
            symbols.append(row['symbol'])
            grade_dates.append(row['grade_date'])
            returns.append(round(actual_return, 6))

    if not symbols:
        logger.info("No valid returns to update")
        return 0

    # Step 3: 1번의 UPDATE (unnest) - is_success는 트리거에서 자동 계산
    update_query = """
    UPDATE kr_stock_prediction_history AS h
    SET actual_return_90d = v.ret,
        updated_at = NOW()
    FROM (
        SELECT
            unnest($1::text[]) AS symbol,
            unnest($2::date[]) AS grade_date,
            unnest($3::float[]) AS ret
    ) AS v
    WHERE h.symbol = v.symbol AND h.grade_date = v.grade_date
    """

    await conn.execute(update_query, symbols, grade_dates, returns)
    logger.info(f"Updated returns: {len(symbols)} records")
    return len(symbols)


# =============================================================================
# Update Stats (효율화: 1 INSERT/UPDATE)
# =============================================================================

async def update_stats(conn: asyncpg.Connection) -> int:
    """
    prediction_stats 테이블 전체 갱신

    효율화:
    - 1번의 집계 쿼리로 모든 통계 계산
    - UPSERT로 한 번에 저장

    Returns:
        갱신된 레코드 수
    """
    logger.info("Updating stats table")

    query = """
    INSERT INTO kr_stock_prediction_stats (
        symbol, total_signals, total_successes, hit_rate,
        buy_signals, buy_successes, buy_hit_rate, buy_avg_return_90d,
        sell_signals, sell_successes, sell_hit_rate, sell_avg_return_90d,
        latest_grade, latest_grade_date, updated_at
    )
    SELECT
        h.symbol,
        COUNT(*) as total_signals,
        SUM(CASE WHEN is_success THEN 1 ELSE 0 END) as total_successes,
        AVG(CASE WHEN is_success THEN 1.0 ELSE 0.0 END) as hit_rate,
        SUM(CASE WHEN grade_direction = 'BUY' THEN 1 ELSE 0 END) as buy_signals,
        SUM(CASE WHEN grade_direction = 'BUY' AND is_success THEN 1 ELSE 0 END) as buy_successes,
        AVG(CASE WHEN grade_direction = 'BUY' AND is_success IS NOT NULL THEN
            CASE WHEN is_success THEN 1.0 ELSE 0.0 END END) as buy_hit_rate,
        AVG(CASE WHEN grade_direction = 'BUY' THEN actual_return_90d END) as buy_avg_return_90d,
        SUM(CASE WHEN grade_direction = 'SELL' THEN 1 ELSE 0 END) as sell_signals,
        SUM(CASE WHEN grade_direction = 'SELL' AND is_success THEN 1 ELSE 0 END) as sell_successes,
        AVG(CASE WHEN grade_direction = 'SELL' AND is_success IS NOT NULL THEN
            CASE WHEN is_success THEN 1.0 ELSE 0.0 END END) as sell_hit_rate,
        AVG(CASE WHEN grade_direction = 'SELL' THEN actual_return_90d END) as sell_avg_return_90d,
        (SELECT predicted_grade FROM kr_stock_prediction_history
         WHERE symbol = h.symbol ORDER BY grade_date DESC LIMIT 1) as latest_grade,
        (SELECT grade_date FROM kr_stock_prediction_history
         WHERE symbol = h.symbol ORDER BY grade_date DESC LIMIT 1) as latest_grade_date,
        NOW() as updated_at
    FROM kr_stock_prediction_history h
    WHERE actual_return_90d IS NOT NULL
    GROUP BY h.symbol
    ON CONFLICT (symbol) DO UPDATE SET
        total_signals = EXCLUDED.total_signals,
        total_successes = EXCLUDED.total_successes,
        hit_rate = EXCLUDED.hit_rate,
        buy_signals = EXCLUDED.buy_signals,
        buy_successes = EXCLUDED.buy_successes,
        buy_hit_rate = EXCLUDED.buy_hit_rate,
        buy_avg_return_90d = EXCLUDED.buy_avg_return_90d,
        sell_signals = EXCLUDED.sell_signals,
        sell_successes = EXCLUDED.sell_successes,
        sell_hit_rate = EXCLUDED.sell_hit_rate,
        sell_avg_return_90d = EXCLUDED.sell_avg_return_90d,
        latest_grade = EXCLUDED.latest_grade,
        latest_grade_date = EXCLUDED.latest_grade_date,
        updated_at = NOW()
    """

    result = await conn.execute(query)
    count = int(result.split()[-1]) if result else 0
    logger.info(f"Updated stats: {count} records")
    return count


# =============================================================================
# Backfill Historical Data (효율화된 버전)
# =============================================================================

async def backfill_history(conn: asyncpg.Connection) -> Dict[str, int]:
    """
    과거 데이터 일괄 수집 (효율화)

    1. kr_stock_grade에서 가장 오래된 날짜부터 시작
    2. 90일 이후 데이터가 없으면 수익률 계산 불가 → 완료 처리

    Returns:
        {'grades_recorded': int, 'returns_updated': int, 'stats_updated': int}
    """
    # Step 1: 분석 데이터의 날짜 범위 확인 (1 SELECT)
    date_range = await conn.fetchrow("""
        SELECT MIN(date) as min_date, MAX(date) as max_date
        FROM kr_stock_grade
        WHERE final_grade IS NOT NULL
    """)

    if not date_range or not date_range['min_date']:
        logger.warning("No grade data found in kr_stock_grade")
        return {'grades_recorded': 0, 'returns_updated': 0, 'stats_updated': 0}

    start_date = date_range['min_date']
    end_date = date_range['max_date']
    logger.info(f"Grade data range: {start_date} ~ {end_date}")

    # Step 2: 가격 데이터 범위 확인 (수익률 계산 가능 여부)
    price_range = await conn.fetchrow("""
        SELECT MIN(date) as min_date, MAX(date) as max_date
        FROM kr_intraday_total
        WHERE close IS NOT NULL
    """)

    if price_range and price_range['max_date']:
        # 90일 후 가격 데이터가 있는 등급만 수익률 계산 가능
        latest_calculable = price_range['max_date'] - timedelta(days=90)
        logger.info(f"Price data available until: {price_range['max_date']}")
        logger.info(f"Returns calculable for grades until: {latest_calculable}")

    # Step 3: 등급 이력 일괄 기록 (1 INSERT)
    grades_count = await batch_record_grades(conn, start_date, end_date)

    # Step 4: 수익률 일괄 갱신 (1 SELECT + 1 UPDATE)
    returns_count = await batch_update_returns(conn)

    # Step 5: 통계 갱신 (1 INSERT/UPDATE)
    stats_count = await update_stats(conn)

    return {
        'grades_recorded': grades_count,
        'returns_updated': returns_count,
        'stats_updated': stats_count
    }


# =============================================================================
# Full Rebuild (Delete All + Recalculate)
# =============================================================================

async def full_rebuild(conn: asyncpg.Connection) -> Dict[str, int]:
    """
    기존 데이터 삭제 후 전체 재계산

    1. prediction_history 테이블 전체 삭제
    2. prediction_stats 테이블 전체 삭제
    3. 전체 기간 등급 기록
    4. 전체 수익률 갱신
    5. 통계 재생성

    Returns:
        {'grades_recorded': int, 'returns_updated': int, 'stats_updated': int}
    """
    logger.info("Starting full rebuild - deleting all existing data")

    # Step 1: 기존 데이터 삭제
    delete_history = await conn.execute("DELETE FROM kr_stock_prediction_history")
    delete_stats = await conn.execute("DELETE FROM kr_stock_prediction_stats")
    logger.info(f"Deleted history: {delete_history}, stats: {delete_stats}")

    # Step 2: 분석 데이터의 날짜 범위 확인
    date_range = await conn.fetchrow("""
        SELECT MIN(date) as min_date, MAX(date) as max_date
        FROM kr_stock_grade
        WHERE final_grade IS NOT NULL
    """)

    if not date_range or not date_range['min_date']:
        logger.warning("No grade data found in kr_stock_grade")
        return {'grades_recorded': 0, 'returns_updated': 0, 'stats_updated': 0}

    start_date = date_range['min_date']
    end_date = date_range['max_date']
    logger.info(f"Rebuilding for date range: {start_date} ~ {end_date}")

    # Step 3: 등급 이력 기록
    grades_count = await batch_record_grades(conn, start_date, end_date)

    # Step 4: 수익률 갱신
    returns_count = await batch_update_returns(conn)

    # Step 5: 통계 갱신
    stats_count = await update_stats(conn)

    return {
        'grades_recorded': grades_count,
        'returns_updated': returns_count,
        'stats_updated': stats_count
    }


# =============================================================================
# Collect Today's Data
# =============================================================================

async def collect_today(conn: asyncpg.Connection) -> Dict[str, int]:
    """오늘 데이터 수집"""
    today = date.today()
    logger.info(f"Collecting today's data: {today}")

    # 오늘 등급만 기록
    grades_count = await batch_record_grades(conn, today, today)

    # 90일 전 등급의 수익률 갱신 (전체 대상)
    returns_count = await batch_update_returns(conn)

    # 통계 갱신
    stats_count = await update_stats(conn)

    return {
        'grades_recorded': grades_count,
        'returns_updated': returns_count,
        'stats_updated': stats_count
    }


# =============================================================================
# Main
# =============================================================================

async def main():
    print("=" * 60)
    print("KR Stock Prediction Collector")
    print("=" * 60)
    print()
    print("실행할 작업을 선택하세요:")
    print("  1. 과거 데이터 일괄 수집 (Backfill)")
    print("  2. 오늘 데이터 수집")
    print("  3. 전체 재계산 (기존 데이터 삭제 후 재생성)")
    print()

    choice = input("선택 (1, 2 또는 3): ").strip()

    if choice not in ['1', '2', '3']:
        print("잘못된 선택입니다. 1, 2 또는 3을 입력하세요.")
        return

    conn = await get_connection()

    try:
        if choice == '1':
            result = await backfill_history(conn)
            print()
            print("=" * 60)
            print("Backfill 완료")
            print(f"  - 등급 기록: {result['grades_recorded']:,}건")
            print(f"  - 수익률 갱신: {result['returns_updated']:,}건")
            print(f"  - 통계 갱신: {result['stats_updated']:,}건")
            print("=" * 60)

        elif choice == '2':
            result = await collect_today(conn)
            print()
            print("=" * 60)
            print("오늘 데이터 수집 완료")
            print(f"  - 등급 기록: {result['grades_recorded']:,}건")
            print(f"  - 수익률 갱신: {result['returns_updated']:,}건")
            print(f"  - 통계 갱신: {result['stats_updated']:,}건")
            print("=" * 60)

        elif choice == '3':
            confirm = input("기존 데이터가 모두 삭제됩니다. 계속하시겠습니까? (y/n): ").strip().lower()
            if confirm != 'y':
                print("취소되었습니다.")
                return
            result = await full_rebuild(conn)
            print()
            print("=" * 60)
            print("전체 재계산 완료")
            print(f"  - 등급 기록: {result['grades_recorded']:,}건")
            print(f"  - 수익률 갱신: {result['returns_updated']:,}건")
            print(f"  - 통계 갱신: {result['stats_updated']:,}건")
            print("=" * 60)

        logger.info("KR Prediction Collector completed successfully")

    except Exception as e:
        logger.error(f"Error: {e}")
        raise
    finally:
        await conn.close()


if __name__ == '__main__':
    asyncio.run(main())
