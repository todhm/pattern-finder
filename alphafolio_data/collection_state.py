"""Per-(collection, symbol, date) state helpers.

Replaces the legacy binary "has symbol ever been seen" logic with
date-granular tracking. Backed by the `collection_state` table created
in Alembic 0006.

Typical usage from a collector:

    pool = await asyncpg.create_pool(DATABASE_URL, ...)
    missing = await get_missing_dates(pool, "us_daily_etf", "SPY",
                                       start, end)
    if not missing:
        return  # nothing to fetch
    # ... fetch + INSERT ...
    await mark_collected(pool, "us_daily_etf", "SPY", dates_inserted)
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import List, Sequence, Optional

import asyncpg


def _business_days(start: date, end: date) -> List[date]:
    out = []
    d = start
    while d <= end:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


async def get_collected_dates(pool: asyncpg.Pool, collection_name: str,
                              symbol: str, start: date, end: date) -> set:
    """Return set of dates already recorded for (collection, symbol) in range."""
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """SELECT date FROM collection_state
               WHERE collection_name=$1 AND symbol=$2
                 AND date BETWEEN $3 AND $4""",
            collection_name, symbol, start, end,
        )
    return {r["date"] for r in rows}


async def get_missing_dates(pool: asyncpg.Pool, collection_name: str,
                            symbol: str, start: date, end: date,
                            business_days_only: bool = True) -> List[date]:
    """Return business days (or all days) in [start, end] NOT yet collected."""
    existing = await get_collected_dates(pool, collection_name, symbol, start, end)
    days = _business_days(start, end) if business_days_only else [
        start + timedelta(days=i) for i in range((end - start).days + 1)
    ]
    return [d for d in days if d not in existing]


async def mark_collected(pool: asyncpg.Pool, collection_name: str,
                         symbol: str, dates: Sequence[date],
                         status: str = "success") -> int:
    """Mark (collection, symbol, dates...) as collected with given status.

    status:
      - 'success' (default) — data fetched & stored
      - 'no_data' — API explicitly said no data for this symbol
      - 'failed'  — other error (retry next time)
    """
    if not dates:
        return 0
    async with pool.acquire() as conn:
        await conn.executemany(
            """INSERT INTO collection_state (collection_name, symbol, date, status)
               VALUES ($1, $2, $3, $4)
               ON CONFLICT (collection_name, symbol, date) DO UPDATE SET
                   status = EXCLUDED.status,
                   collected_at = NOW()""",
            [(collection_name, symbol, d, status) for d in dates],
        )
    return len(dates)


async def mark_no_data(pool: asyncpg.Pool, collection_name: str,
                       symbol: str, date_val: date) -> None:
    """Record that the API explicitly returned no data for (symbol, date).

    Future runs will skip this symbol for this date to avoid wasting API calls.
    """
    await mark_collected(pool, collection_name, symbol, [date_val], status="no_data")


async def get_no_data_symbols(pool: asyncpg.Pool, collection_name: str,
                              recent_days: int = 7,
                              exclude_today: bool = True) -> set:
    """Symbols marked 'no_data' within the last `recent_days` days. These
    should be skipped on the next fetch to avoid wasting API calls.

    exclude_today=True (default): 오늘 mark 된 종목은 결과에서 제외 → 같은 날
    여러 번 시도할 수 있도록 (당일 데이터는 시간에 따라 바뀔 수 있으므로 매번
    새로 시도하라는 정책).
    """
    today_clause = "AND date < CURRENT_DATE" if exclude_today else ""
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            f"""SELECT DISTINCT symbol FROM collection_state
                WHERE collection_name = $1 AND status = 'no_data'
                  AND date >= CURRENT_DATE - $2::int
                  {today_clause}""",
            collection_name, recent_days,
        )
    return {r["symbol"] for r in rows}


async def mark_no_data_via_conn(conn, collection_name: str, symbol: str,
                                date_val: Optional[date] = None) -> None:
    """pool 없이 단일 connection 으로 no_data mark (collector 내부에서 사용).

    date_val=None 이면 CURRENT_DATE. ON CONFLICT 로 같은 날 중복 호출 안전.
    """
    if date_val is None:
        await conn.execute(
            """INSERT INTO collection_state (collection_name, symbol, date, status)
               VALUES ($1, $2, CURRENT_DATE, 'no_data')
               ON CONFLICT (collection_name, symbol, date) DO UPDATE
                 SET status='no_data', collected_at=NOW()""",
            collection_name, symbol)
    else:
        await conn.execute(
            """INSERT INTO collection_state (collection_name, symbol, date, status)
               VALUES ($1, $2, $3, 'no_data')
               ON CONFLICT (collection_name, symbol, date) DO UPDATE
                 SET status='no_data', collected_at=NOW()""",
            collection_name, symbol, date_val)


async def get_no_data_symbols_via_conn(conn, collection_name: str,
                                       recent_days: int = 7,
                                       exclude_today: bool = True) -> set:
    """pool 없이 단일 connection 으로 조회 (collector 내부 get_active_symbols 등)."""
    today_clause = "AND date < CURRENT_DATE" if exclude_today else ""
    rows = await conn.fetch(
        f"""SELECT DISTINCT symbol FROM collection_state
            WHERE collection_name = $1 AND status = 'no_data'
              AND date >= CURRENT_DATE - $2::int
              {today_clause}""",
        collection_name, recent_days)
    return {r["symbol"] for r in rows}


async def bulk_mark_collected(pool: asyncpg.Pool, collection_name: str,
                              triples: Sequence[tuple]) -> int:
    """Bulk mark from (symbol, date) tuples."""
    if not triples:
        return 0
    async with pool.acquire() as conn:
        await conn.executemany(
            """INSERT INTO collection_state (collection_name, symbol, date)
               VALUES ($1, $2, $3)
               ON CONFLICT DO NOTHING""",
            [(collection_name, sym, d) for (sym, d) in triples],
        )
    return len(triples)


async def get_symbols_with_missing(pool: asyncpg.Pool, collection_name: str,
                                   symbols: Sequence[str],
                                   start: date, end: date) -> List[str]:
    """Filter `symbols` down to those that have at least one missing
    business day in [start, end]. Useful to skip API calls for already-
    fully-collected symbols."""
    if not symbols:
        return []
    expected_count = len(_business_days(start, end))
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """SELECT symbol, COUNT(*) AS cnt
               FROM collection_state
               WHERE collection_name=$1 AND symbol = ANY($2::text[])
                 AND date BETWEEN $3 AND $4
               GROUP BY symbol""",
            collection_name, list(symbols), start, end,
        )
    have_full = {r["symbol"] for r in rows if r["cnt"] >= expected_count}
    return [s for s in symbols if s not in have_full]


async def coverage_summary(pool: asyncpg.Pool, collection_name: str,
                           start: date, end: date) -> dict:
    """Diagnostic: how many distinct (symbol, date) pairs are covered?"""
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """SELECT COUNT(*) AS pairs,
                      COUNT(DISTINCT symbol) AS symbols,
                      COUNT(DISTINCT date)   AS dates
               FROM collection_state
               WHERE collection_name=$1
                 AND date BETWEEN $2 AND $3""",
            collection_name, start, end,
        )
    return dict(row)
