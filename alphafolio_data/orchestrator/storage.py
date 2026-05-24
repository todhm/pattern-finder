"""Postgres persistence for orchestrator runs/tasks/logs."""
from __future__ import annotations

import json
import logging
import os
from typing import List, Optional

import asyncpg

logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL")
_pool: Optional[asyncpg.Pool] = None

SCHEMA_DDL = """
CREATE TABLE IF NOT EXISTS orch_runs (
    run_id     TEXT PRIMARY KEY,
    country    TEXT NOT NULL,
    start_date DATE NOT NULL,
    end_date   DATE NOT NULL,
    params     JSONB NOT NULL,
    status     TEXT NOT NULL,
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    error TEXT
);
CREATE TABLE IF NOT EXISTS orch_tasks (
    run_id  TEXT NOT NULL REFERENCES orch_runs(run_id) ON DELETE CASCADE,
    task_id TEXT NOT NULL,
    name    TEXT NOT NULL,
    status  TEXT NOT NULL,
    seq     INT NOT NULL,
    depends_on TEXT[],
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    duration_seconds DOUBLE PRECISION,
    output  JSONB,
    error   TEXT,
    PRIMARY KEY (run_id, task_id)
);
CREATE INDEX IF NOT EXISTS idx_orch_tasks_run_seq ON orch_tasks(run_id, seq);
CREATE TABLE IF NOT EXISTS orch_logs (
    id      BIGSERIAL PRIMARY KEY,
    run_id  TEXT NOT NULL,
    task_id TEXT,
    ts      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    level   TEXT NOT NULL,
    message TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_orch_logs_run_ts ON orch_logs(run_id, id);
"""


async def get_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        if not DATABASE_URL:
            raise RuntimeError("DATABASE_URL not set")
        _pool = await asyncpg.create_pool(DATABASE_URL, min_size=2, max_size=10)
        async with _pool.acquire() as conn:
            await conn.execute(SCHEMA_DDL)
        logger.info("[orchestrator] tables ensured")
    return _pool


async def create_run(run_id, country, start_date, end_date, params: dict):
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """INSERT INTO orch_runs (run_id,country,start_date,end_date,params,status)
               VALUES ($1,$2,$3,$4,$5,'pending')""",
            run_id, country, start_date, end_date, json.dumps(params, default=str))


async def update_run(run_id, status, error=None, completed=False):
    pool = await get_pool()
    async with pool.acquire() as conn:
        if completed:
            await conn.execute(
                "UPDATE orch_runs SET status=$1, error=$2, completed_at=NOW() WHERE run_id=$3",
                status, error, run_id)
        else:
            await conn.execute(
                "UPDATE orch_runs SET status=$1, error=$2 WHERE run_id=$3",
                status, error, run_id)


async def get_run(run_id):
    pool = await get_pool()
    async with pool.acquire() as conn:
        r = await conn.fetchrow("SELECT * FROM orch_runs WHERE run_id=$1", run_id)
        return dict(r) if r else None


async def list_runs(limit=50):
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """SELECT run_id,country,start_date,end_date,status,started_at,completed_at,error
               FROM orch_runs ORDER BY started_at DESC LIMIT $1""", limit)
        return [dict(r) for r in rows]


async def init_tasks(run_id, task_defs: List[dict]):
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.executemany(
            """INSERT INTO orch_tasks (run_id,task_id,name,status,seq,depends_on)
               VALUES ($1,$2,$3,'pending',$4,$5)
               ON CONFLICT (run_id,task_id) DO NOTHING""",
            [(run_id, t["id"], t["name"], i, t.get("depends_on", []))
             for i, t in enumerate(task_defs)])


async def update_task(run_id, task_id, status, output=None, error=None,
                      mark_start=False, mark_end=False):
    pool = await get_pool()
    async with pool.acquire() as conn:
        sets, vals = ["status=$1"], [status]
        i = 2
        if mark_start:
            sets.append("started_at=NOW()")
        if mark_end:
            sets.append("completed_at=NOW()")
            sets.append("duration_seconds=EXTRACT(EPOCH FROM (NOW()-COALESCE(started_at,NOW())))")
        if output is not None:
            sets.append(f"output=${i}::jsonb"); vals.append(json.dumps(output, default=str)); i += 1
        if error is not None:
            sets.append(f"error=${i}"); vals.append(error); i += 1
        vals.extend([run_id, task_id])
        await conn.execute(
            f"UPDATE orch_tasks SET {','.join(sets)} WHERE run_id=${i} AND task_id=${i+1}", *vals)


async def list_tasks(run_id):
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """SELECT task_id,name,status,seq,depends_on,started_at,completed_at,
                      duration_seconds,output,error
               FROM orch_tasks WHERE run_id=$1 ORDER BY seq""", run_id)
        return [dict(r) for r in rows]


async def reset_task(run_id, task_id):
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """UPDATE orch_tasks SET status='pending', started_at=NULL,
                      completed_at=NULL, duration_seconds=NULL, error=NULL
               WHERE run_id=$1 AND task_id=$2""", run_id, task_id)


async def log(run_id, level, message, task_id=None):
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO orch_logs (run_id,task_id,level,message) VALUES ($1,$2,$3,$4)",
            run_id, task_id, level, message[:8000])


async def tail_logs(run_id, after_id=0, limit=300):
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """SELECT id,task_id,ts,level,message FROM orch_logs
               WHERE run_id=$1 AND id>$2 ORDER BY id LIMIT $3""",
            run_id, after_id, limit)
        return [dict(r) for r in rows]
