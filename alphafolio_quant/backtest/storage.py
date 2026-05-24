"""DB schema + CRUD for backtest runs.

Tables (auto-created in alphafolio DB on first use):
- backtest_runs:        run metadata + final metrics (JSONB)
- backtest_nav_history: daily NAV time series per run
- backtest_trades:      individual buy/sell records per run
"""
import os
import json
import logging
from typing import Optional, List, Dict

import asyncpg

logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL")

_pool: Optional[asyncpg.Pool] = None

SCHEMA_DDL = """
CREATE TABLE IF NOT EXISTS backtest_runs (
    run_id TEXT PRIMARY KEY,
    country TEXT NOT NULL,
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    initial_cash NUMERIC NOT NULL,
    top_n INT NOT NULL,
    rebal_freq_days INT NOT NULL,
    params JSONB NOT NULL,
    status TEXT NOT NULL,
    metrics JSONB,
    error TEXT,
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS backtest_nav_history (
    run_id TEXT NOT NULL REFERENCES backtest_runs(run_id) ON DELETE CASCADE,
    date DATE NOT NULL,
    nav NUMERIC NOT NULL,
    cash NUMERIC NOT NULL,
    holdings_value NUMERIC NOT NULL,
    holdings_count INT NOT NULL,
    benchmark_value NUMERIC,
    PRIMARY KEY (run_id, date)
);

CREATE TABLE IF NOT EXISTS backtest_trades (
    id BIGSERIAL PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES backtest_runs(run_id) ON DELETE CASCADE,
    date DATE NOT NULL,
    symbol TEXT NOT NULL,
    action TEXT NOT NULL,
    shares NUMERIC NOT NULL,
    price NUMERIC NOT NULL,
    amount NUMERIC NOT NULL,
    commission NUMERIC NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_backtest_trades_run_date
    ON backtest_trades(run_id, date);
"""


async def get_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        if not DATABASE_URL:
            raise RuntimeError("DATABASE_URL not configured")
        _pool = await asyncpg.create_pool(DATABASE_URL, min_size=2, max_size=10)
        async with _pool.acquire() as conn:
            await conn.execute(SCHEMA_DDL)
        logger.info("Backtest tables ensured (backtest_runs / nav_history / trades)")
    return _pool


async def create_run(run_id: str, params: dict):
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO backtest_runs
                (run_id, country, start_date, end_date, initial_cash,
                 top_n, rebal_freq_days, params, status)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, 'running')
            """,
            run_id,
            params["country"],
            params["start_date"],
            params["end_date"],
            params["initial_cash"],
            params["top_n"],
            params["rebal_freq_days"],
            json.dumps(params, default=str),
        )


async def update_run_complete(run_id: str, metrics: dict):
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE backtest_runs
            SET status='completed', metrics=$1::jsonb, completed_at=NOW()
            WHERE run_id=$2
            """,
            json.dumps(metrics, default=str),
            run_id,
        )


async def update_run_failed(run_id: str, error: str):
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE backtest_runs
            SET status='failed', error=$1, completed_at=NOW()
            WHERE run_id=$2
            """,
            error,
            run_id,
        )


async def save_nav(run_id: str, nav_rows: List[Dict]):
    if not nav_rows:
        return
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.executemany(
            """
            INSERT INTO backtest_nav_history
                (run_id, date, nav, cash, holdings_value, holdings_count, benchmark_value)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            ON CONFLICT (run_id, date) DO NOTHING
            """,
            [
                (
                    run_id,
                    r["date"],
                    r["nav"],
                    r["cash"],
                    r["holdings_value"],
                    r["holdings_count"],
                    r.get("benchmark_value"),
                )
                for r in nav_rows
            ],
        )


async def save_trades(run_id: str, trades: List[Dict]):
    if not trades:
        return
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.executemany(
            """
            INSERT INTO backtest_trades
                (run_id, date, symbol, action, shares, price, amount, commission)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            """,
            [
                (
                    run_id,
                    t["date"],
                    t["symbol"],
                    t["action"],
                    t["shares"],
                    t["price"],
                    t["amount"],
                    t["commission"],
                )
                for t in trades
            ],
        )


async def get_run(run_id: str) -> Optional[dict]:
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM backtest_runs WHERE run_id=$1", run_id
        )
        return dict(row) if row else None


async def list_runs(limit: int = 50) -> List[dict]:
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT run_id, country, start_date, end_date, status,
                   metrics, started_at, completed_at
            FROM backtest_runs
            ORDER BY started_at DESC
            LIMIT $1
            """,
            limit,
        )
        return [dict(r) for r in rows]


async def get_nav_history(run_id: str) -> List[dict]:
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT date, nav, cash, holdings_value, holdings_count, benchmark_value
            FROM backtest_nav_history
            WHERE run_id=$1
            ORDER BY date
            """,
            run_id,
        )
        return [dict(r) for r in rows]


async def get_trades(run_id: str) -> List[dict]:
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT date, symbol, action, shares, price, amount, commission
            FROM backtest_trades
            WHERE run_id=$1
            ORDER BY date, symbol
            """,
            run_id,
        )
        return [dict(r) for r in rows]


async def delete_run(run_id: str) -> bool:
    pool = await get_pool()
    async with pool.acquire() as conn:
        res = await conn.execute("DELETE FROM backtest_runs WHERE run_id=$1", run_id)
        return res.endswith("1")
