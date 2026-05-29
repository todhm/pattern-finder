"""Standalone options lookback backfill (summary-only).

Collects us_option_daily_summary for the lookback window BEFORE the backtest
start so that 252-day IV percentile / volatility calculations have continuous
history from day 1 of the backtest range.

- Symbol set: union of per-date top-N grades over the MAIN backtest range
  (2025-05-20 ~ 2026-05-20) — the same symbols that will be analyzed — held
  fixed across every lookback date (union mode).
- Storage: summary_only (no raw us_option rows).
- Resumable: skips dates already present in us_option_daily_summary.
- Safe to run in parallel with grades_pass_b: only appends older dates that
  Pass-B does not read for the current range.

Run (detached) inside the alphafolio_data container:
    python -m us.backfill_options_lookback 2024-05-20 2025-05-19
"""
import asyncio
import logging
import os
import sys
from datetime import date, datetime

import asyncpg

from us.us_option import USOptionCollector

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("backfill_options_lookback")

DATABASE_URL = os.getenv("DATABASE_URL")
# Union symbol set is derived from grades over the main backtest range.
MAIN_START = os.getenv("OPT_BACKFILL_MAIN_START", "2025-05-20")
MAIN_END = os.getenv("OPT_BACKFILL_MAIN_END", "2026-05-20")
TOP_N = os.getenv("OPT_BACKFILL_TOP_N", "200")


async def _trading_days(pool, start: date, end: date):
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT DISTINCT date FROM us_daily "
            "WHERE date BETWEEN $1 AND $2 ORDER BY date",
            start, end)
        done = await conn.fetch(
            "SELECT DISTINCT date FROM us_option_daily_summary "
            "WHERE date BETWEEN $1 AND $2",
            start, end)
    done_set = {r["date"] for r in done}
    return [r["date"] for r in rows if r["date"] not in done_set]


async def main(start: date, end: date):
    api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    if not api_key:
        raise RuntimeError("ALPHAVANTAGE_API_KEY not set")

    pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=2)
    try:
        dates = await _trading_days(pool, start, end)
    finally:
        await pool.close()

    logger.info(f"Lookback backfill: {len(dates)} dates to process "
                f"({start} ~ {end}), union symbols from {MAIN_START}~{MAIN_END}")

    # Union mode: get_active_symbols() computes the union top-N over the MAIN
    # range regardless of the per-date target. Set once for the whole loop.
    os.environ["US_OPTION_DYNAMIC_TOP_N"] = TOP_N
    os.environ["US_OPTION_DYNAMIC_START_DATE"] = MAIN_START
    os.environ["US_OPTION_DYNAMIC_END_DATE"] = MAIN_END

    ok = fail = 0
    for i, d in enumerate(dates, 1):
        os.environ["US_OPTION_TARGET_GRADE_DATE"] = d.isoformat()
        try:
            col = USOptionCollector(api_key, DATABASE_URL, 0.3,
                                    target_date=d, summary_only=True)
            await col.init_pool()
            try:
                await col.run_collection_summary_only()
            finally:
                await col.close_pool()
            ok += 1
            logger.info(f"[{i}/{len(dates)}] {d} done")
        except Exception as e:
            fail += 1
            logger.error(f"[{i}/{len(dates)}] {d} failed: {e}")

    logger.info(f"Lookback backfill complete: ok={ok}, fail={fail}")


if __name__ == "__main__":
    s = datetime.strptime(sys.argv[1], "%Y-%m-%d").date()
    e = datetime.strptime(sys.argv[2], "%Y-%m-%d").date()
    asyncio.run(main(s, e))
