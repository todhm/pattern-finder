"""Task implementations + DAG topology.

Tasks call:
- Local alphafolio_data endpoints (via localhost) for data collection
- alphafolio_quant for grade generation (POST /backtest/generate-grades) and
  backtest (POST /backtest/run)

Each task is an async function taking a Ctx and returning a small dict that
gets stored in orch_tasks.output. Logs go to orch_logs via ctx.log().
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Callable, Dict, List, Optional

import asyncpg
import httpx

logger = logging.getLogger(__name__)

SELF_URL  = os.getenv("SELF_URL",  "http://localhost:8000")
QUANT_URL = os.getenv("QUANT_SERVICE_URL", "http://alphafolio_quant:8000")
API_KEY   = os.getenv("API_SECRET_KEY", "")
DATABASE_URL = os.getenv("DATABASE_URL")


@dataclass
class Ctx:
    run_id: str
    country: str
    start_date: date
    end_date: date
    params: dict
    storage: "object"
    _task_id: Optional[str] = None

    async def log(self, level: str, msg: str):
        await self.storage.log(self.run_id, level, msg, self._task_id)


import asyncio as _asyncio
from contextlib import asynccontextmanager


# =============================================================================
# Backtest lookback buffers — used by every data-collection task to extend
# ``ctx.start_date`` backward so analyses at the start of the requested range
# have enough historical context (no NaN, no fallback responses).
#
# Source-by-source requirement (longest lookback used by alphafolio_quant):
#   - us_daily         : 260 trading days (VAR/CVAR, volatility, momentum,
#                         52-week high/low, 200-day MA)
#   - us_daily_etf SPY : 504 trading days = 2 years (HMM regime fit at startup)
#   - us_weekly        : 200 weeks ≈ 4 years (calculator weekly indicators) —
#                         us_weekly already collects from 2020-01-02, no
#                         per-run extension needed
#   - us_income / balance / cash_flow : 12 quarters ≈ 3 years (TTM, growth,
#                         analyst metrics). Enforced via the AV API call
#                         (TIME_SERIES_LIMIT or equivalent), not via
#                         ctx.start_date.
#
# We translate trading-day requirements to calendar-day buffers with a
# generous margin (52 weekends + holidays + skew).
# =============================================================================
DAILY_LOOKBACK_CALENDAR_DAYS = 400   # legacy LOOKBACK_BUFFER_DAYS, kept for compat
ETF_LOOKBACK_CALENDAR_DAYS = 800     # 504 trading days × ~365/252 + safety
FINANCIALS_LOOKBACK_QUARTERS = 12    # latest 12 quarters of fundamentals


class _OrchLogHandler(logging.Handler):
    """Python logging → orch_logs DB bridge.

    Captures log records from named loggers (us.alphavantage, us.us_etf, etc.)
    while a task runs and writes them to orch_logs as the task's logs.
    Uses an asyncio.Queue to keep `logging.emit()` non-blocking (logging is
    typically synchronous; DB write is awaited by a separate consumer task).
    """

    LEVEL_MAP = {
        "DEBUG": "info",
        "INFO": "info",
        "WARNING": "warn",
        "ERROR": "error",
        "CRITICAL": "error",
    }

    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self.queue: _asyncio.Queue = _asyncio.Queue(maxsize=2000)
        self.consumer: Optional[_asyncio.Task] = None

    def emit(self, record):
        try:
            msg = f"{record.name} - {record.getMessage()}"
            level = self.LEVEL_MAP.get(record.levelname, "info")
            # Truncate to keep DB rows reasonable
            self.queue.put_nowait((level, msg[:500]))
        except Exception:
            pass  # Never raise from a logging handler

    async def _consume(self):
        try:
            while True:
                level, msg = await self.queue.get()
                try:
                    await self.ctx.log(level, msg)
                except Exception:
                    pass
        except _asyncio.CancelledError:
            # Flush any remaining queued messages before stopping
            while not self.queue.empty():
                try:
                    level, msg = self.queue.get_nowait()
                    await self.ctx.log(level, msg)
                except Exception:
                    break

    def start(self):
        self.consumer = _asyncio.create_task(self._consume())

    async def stop(self):
        if self.consumer:
            self.consumer.cancel()
            try:
                await self.consumer
            except _asyncio.CancelledError:
                pass


@asynccontextmanager
async def capture_logs(ctx: "Ctx",
                       loggers: tuple = ("us", "kr", "index", "utils", "main"),
                       level: int = logging.WARNING):
    """Capture WARN+ logs from given Python loggers into orch_logs while
    the task runs. After exit, handler is removed and no further records
    leak in.

    `loggers` are matched as prefixes (e.g. 'us' matches 'us.alphavantage',
    'us.us_etf', etc.). Default captures all collector + main logs.
    """
    handler = _OrchLogHandler(ctx)
    handler.setLevel(level)
    targets = [logging.getLogger(n) for n in loggers]
    for lg in targets:
        lg.addHandler(handler)
    handler.start()
    try:
        yield
    finally:
        for lg in targets:
            lg.removeHandler(handler)
        await handler.stop()


@asynccontextmanager
async def monitor_progress(ctx: "Ctx", sql: str, label: str, interval: int = 30):
    """Background coroutine: polls a row-count SQL every `interval` seconds
    and pushes the current count to orch_logs as INFO. Cancelled when the
    enclosing 'async with' block exits. Use this to surface progress for
    long-running tasks that block in a single API call.

    Example:
        async with monitor_progress(ctx, "SELECT COUNT(*) FROM us_daily",
                                    "us_daily rows", interval=30):
            await long_running_thing()
    """
    pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=1)

    async def _poll():
        try:
            while True:
                await _asyncio.sleep(interval)
                try:
                    async with pool.acquire() as conn:
                        cnt = await conn.fetchval(sql)
                    await ctx.log("info", f"{label}: {cnt:,}")
                except Exception as e:
                    await ctx.log("warn", f"monitor {label}: {e}")
        except _asyncio.CancelledError:
            pass

    task = _asyncio.create_task(_poll())
    try:
        yield
    finally:
        task.cancel()
        try:
            await task
        except _asyncio.CancelledError:
            pass
        await pool.close()


# ------------------------------------------------------------ HTTP helpers
async def _self_post(ctx: Ctx, path: str, params: Optional[dict] = None) -> dict:
    url = f"{SELF_URL}/{path.lstrip('/')}"
    await ctx.log("run", f"→ POST {path}")
    async with httpx.AsyncClient(timeout=7200) as client:
        r = await client.post(url, params=params or {}, headers={"X-API-KEY": API_KEY})
        try:
            body = r.json()
        except Exception:
            body = {"raw": r.text[:500]}
        if r.status_code >= 400:
            raise RuntimeError(f"{path} → {r.status_code}: {json.dumps(body)[:300]}")
        await ctx.log("ok", f"← {path}: {json.dumps(body)[:200]}")
        return body


async def _quant_post(ctx: Ctx, path: str, body: dict) -> dict:
    url = f"{QUANT_URL}/{path.lstrip('/')}"
    await ctx.log("run", f"→ QUANT {path} {json.dumps(body, default=str)[:200]}")
    async with httpx.AsyncClient(timeout=86400) as client:
        r = await client.post(url, json=body)
        try:
            payload = r.json()
        except Exception:
            payload = {"raw": r.text[:500]}
        if r.status_code >= 400:
            raise RuntimeError(f"quant/{path} → {r.status_code}: {json.dumps(payload)[:300]}")
        await ctx.log("ok", f"← quant/{path}: {json.dumps(payload)[:200]}")
        return payload


# ================================================================= TASKS

async def task_partitions(ctx: Ctx) -> dict:
    return await _self_post(ctx, "admin/create-partitions")


async def task_stock_listing(ctx: Ctx) -> dict:
    if ctx.country != "US":
        return {"skipped": "KR run"}
    return await _self_post(ctx, "collect/us/stock-listing")


async def task_finnhub_symbol(ctx: Ctx) -> dict:
    if ctx.country != "US":
        return {"skipped": "KR run"}
    return await _self_post(ctx, "collect/us/finnhub-symbol")


async def task_stock_basic(ctx: Ctx) -> dict:
    if ctx.country != "US":
        return {"skipped": "KR run"}
    async with capture_logs(ctx), monitor_progress(
        ctx,
        "SELECT COUNT(*) FROM us_stock_basic",
        "us_stock_basic rows", interval=30):
        return await _self_post(ctx, "collect/us/stock-basic")


async def task_us_daily(ctx: Ctx) -> dict:
    """Incremental OHLCV pipeline — 종목별 missing range만 정확히 fetch.

    동작:
    1. 종목별 us_daily의 MIN(date)/MAX(date) 조회
    2. 필요 범위 = [ctx.start_date - 260일, ctx.end_date]
       (260일 = EM8 240일 lookback + 버퍼)
    3. 종목 분류:
       - "covered"   : 필요 범위 내 모든 일자 us_daily에 존재 → SKIP (API 콜 0)
       - "forward"   : 최근 일자만 누락 → outputsize=compact (100일치, 빠름)
       - "backward"  : 과거 일자 누락 (EM8 lookback 등) → outputsize=full (20년치)
       - "new"       : us_daily에 0행 → outputsize=full
    4. DailyCollector(target_symbols=needs_action, outputsize_per_symbol=map)
       → 누락 일자만 INSERT (ON CONFLICT 로 중복 방지)

    매일 cron 호출 시 (ctx.start_date=ctx.end_date=오늘):
      거의 모든 종목 "forward" 또는 "covered" → 빠른 incremental 동작
    """
    if ctx.country != "US":
        return {"skipped": "KR run"}

    api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    if not api_key:
        raise RuntimeError("ALPHAVANTAGE_API_KEY not set")

    # EM8 = 240 trading days lookback. 240 × (365/252) ≈ 348 calendar days.
    # 400 = 348 + safety margin (holidays, missing days).
    LOOKBACK_BUFFER_DAYS = DAILY_LOOKBACK_CALENDAR_DAYS
    extended_start = ctx.start_date - timedelta(days=LOOKBACK_BUFFER_DAYS)

    # 종목별 us_daily 범위 + 활성 심볼 목록 한 번에 조회
    pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=3)
    try:
        async with pool.acquire() as conn:
            range_rows = await conn.fetch(
                "SELECT symbol, MIN(date) AS mind, MAX(date) AS maxd "
                "FROM us_daily GROUP BY symbol")
            symbol_range = {r["symbol"]: (r["mind"], r["maxd"]) for r in range_rows}
            active_rows = await conn.fetch(
                "SELECT symbol FROM us_stock_basic WHERE is_active = true")
            active_symbols = [r["symbol"] for r in active_rows]
    finally:
        await pool.close()

    # 분류
    needed_start, needed_end = extended_start, ctx.end_date
    outputsize_map: Dict[str, str] = {}
    target_symbols: List[str] = []
    counts = {"covered": 0, "forward": 0, "backward": 0, "new": 0}

    for sym in active_symbols:
        if sym not in symbol_range:
            counts["new"] += 1
            outputsize_map[sym] = "full"
            target_symbols.append(sym)
            continue
        mind, maxd = symbol_range[sym]
        if mind <= needed_start and maxd >= needed_end:
            counts["covered"] += 1
            continue
        if mind > needed_start:
            # 과거 데이터 부족 → full (20년치, AV가 한 번에 줌)
            counts["backward"] += 1
            outputsize_map[sym] = "full"
            target_symbols.append(sym)
        else:
            # 최근 데이터만 부족 → compact (100일치 응답이면 충분)
            counts["forward"] += 1
            outputsize_map[sym] = "compact"
            target_symbols.append(sym)

    await ctx.log("info",
                  f"종목 분류 (필요범위 {needed_start}~{needed_end}): "
                  f"covered={counts['covered']}, "
                  f"forward={counts['forward']}, "
                  f"backward={counts['backward']}, "
                  f"new={counts['new']} "
                  f"→ {len(target_symbols)}개만 fetch")

    if not target_symbols:
        return {"status": "all_covered", **counts, "rows_in_range": 0}

    from us.alphavantage import DailyCollector
    collector = DailyCollector(
        api_key=api_key,
        database_url=DATABASE_URL,
        call_interval=0.2,
        start_date=extended_start,
        end_date=ctx.end_date,
        outputsize="full",
        outputsize_per_symbol=outputsize_map,
    )
    collector.target_symbols = target_symbols
    await ctx.log("run",
                  f"DailyCollector incremental, range {extended_start}~{ctx.end_date}")
    await collector.init_pool()
    try:
        async with capture_logs(ctx), monitor_progress(
            ctx,
            f"SELECT COUNT(*) FROM us_daily WHERE date BETWEEN '{ctx.start_date}' AND '{ctx.end_date}'",
            "us_daily rows (in range)", interval=30):
            await collector.run_collection_optimized()
    finally:
        await collector.close_pool()

    # Quick sanity count
    pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=2)
    try:
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """SELECT COUNT(*) AS rows, COUNT(DISTINCT symbol) AS syms,
                          MIN(date) AS earliest, MAX(date) AS latest
                   FROM us_daily WHERE date BETWEEN $1 AND $2""",
                ctx.start_date, ctx.end_date)
    finally:
        await pool.close()

    return {
        "rows": row["rows"],
        "symbols": row["syms"],
        "earliest": str(row["earliest"]),
        "latest": str(row["latest"]),
    }


async def task_financials(ctx: Ctx) -> dict:
    """Income / Balance / CashFlow 3종 병렬 백필.

    /collect/us/financials-core 엔드포인트는 스케줄 체크로 둘째/넷째주 일요일에만
    실행되어 silent skip되는 문제 → 직접 collector 호출로 우회.

    각 collector는 AV 100 calls/min (0.6초 간격) 으로 동작. 3개 동시 실행 →
    총 약 30분 ~ 수시간 (재무제표는 분기별 모든 history 한 콜에 반환).
    """
    if ctx.country != "US":
        return {"skipped": "KR run"}

    api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    if not api_key:
        raise RuntimeError("ALPHAVANTAGE_API_KEY not set")

    from datetime import datetime, time as time_obj
    from us.finance_data import (
        IncomeStatementCollector,
        BalanceSheetCollector,
        CashFlowCollector,
    )

    # Deadline = 12 hours from now (재무제표는 느림)
    deadline = datetime.now() + timedelta(hours=12)
    # 0.4s × 3 collectors = 450 calls/min total. Empirical: 0.3s (600/min)
    # triggered ~80 rate-limit warnings/min from AV ("Information" responses)
    # — 600/min is the docs cap but the burst limiter is tighter. 0.4s
    # leaves headroom so the useful-response ratio stays high.
    call_interval = 0.4   # 150 calls/min × 3 parallel = 450 total

    # Skip symbols already collected within 6 months (DB-based dedup, 기존 로직)
    skip_income, skip_balance, skip_cashflow = set(), set(), set()
    pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=3)
    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch("""
                WITH inc_skip AS (
                  SELECT DISTINCT symbol FROM us_income_statement
                  WHERE fiscal_date_ending >= CURRENT_DATE - INTERVAL '6 months'
                ),
                bal_skip AS (
                  SELECT DISTINCT symbol FROM us_balance_sheet
                  WHERE fiscal_date_ending >= CURRENT_DATE - INTERVAL '6 months'
                ),
                cf_skip AS (
                  SELECT DISTINCT symbol FROM us_cash_flow
                  WHERE fiscal_date_ending >= CURRENT_DATE - INTERVAL '6 months'
                )
                SELECT 'income' AS t, symbol FROM inc_skip
                UNION ALL SELECT 'balance', symbol FROM bal_skip
                UNION ALL SELECT 'cashflow', symbol FROM cf_skip
            """)
        skip_income   = {r["symbol"] for r in rows if r["t"] == "income"}
        skip_balance  = {r["symbol"] for r in rows if r["t"] == "balance"}
        skip_cashflow = {r["symbol"] for r in rows if r["t"] == "cashflow"}
    finally:
        await pool.close()
    await ctx.log("info",
                  f"skip 분석: income={len(skip_income)}, "
                  f"balance={len(skip_balance)}, cashflow={len(skip_cashflow)}")

    async with capture_logs(ctx), monitor_progress(
        ctx,
        "SELECT (SELECT COUNT(*) FROM us_income_statement) || ' inc, ' || "
        "       (SELECT COUNT(*) FROM us_balance_sheet) || ' bal, ' || "
        "       (SELECT COUNT(*) FROM us_cash_flow) || ' cf'",
        "financials rows", interval=60):
        income_collector = IncomeStatementCollector(
            api_key, DATABASE_URL, call_interval=call_interval,
            deadline=deadline, skip_symbols=skip_income)
        balance_collector = BalanceSheetCollector(
            api_key, DATABASE_URL, call_interval=call_interval,
            deadline=deadline, skip_symbols=skip_balance)
        cashflow_collector = CashFlowCollector(
            api_key, DATABASE_URL, call_interval=call_interval,
            deadline=deadline, skip_symbols=skip_cashflow)

        await ctx.log("run",
                      f"3 collectors parallel, deadline={deadline.isoformat()}")
        results = await asyncio.gather(
            income_collector.run_collection_optimized(),
            balance_collector.run_collection_optimized(),
            cashflow_collector.run_collection_optimized(),
            return_exceptions=True,
        )

    # 결과 + sanity check
    pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=2)
    try:
        async with pool.acquire() as conn:
            inc_cnt = await conn.fetchval("SELECT COUNT(*) FROM us_income_statement")
            bal_cnt = await conn.fetchval("SELECT COUNT(*) FROM us_balance_sheet")
            cf_cnt  = await conn.fetchval("SELECT COUNT(*) FROM us_cash_flow")
    finally:
        await pool.close()

    errs = [str(r) for r in results if isinstance(r, Exception)]
    if (inc_cnt + bal_cnt + cf_cnt) == 0:
        raise RuntimeError(
            f"All 3 financials tables still empty. errors={errs[:2]}")

    return {
        "income_statement_rows": inc_cnt,
        "balance_sheet_rows":    bal_cnt,
        "cash_flow_rows":        cf_cnt,
        "errors":                errs[:5],
    }


async def task_us_etf(ctx: Ctx) -> dict:
    """ETF backfill via direct Alpha Vantage call + collection_state.

    Bypasses USETFDataCollector (which has broken silent-skip logic).
    Strategy per ETF symbol:
      1. Query collection_state to find missing dates in [start, end]
      2. If 0 missing → skip (no API call)
      3. Else: AV TIME_SERIES_DAILY outputsize=full (one call, 20yr response)
      4. Filter response to missing dates only → INSERT us_daily_etf
      5. Mark each inserted date in collection_state
    """
    if ctx.country != "US":
        return {"skipped": "KR run"}

    import aiohttp
    from us.us_etf import SECTOR_ETFS
    import collection_state as cstate

    api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    if not api_key:
        raise RuntimeError("ALPHAVANTAGE_API_KEY not set")

    etf_symbols = sorted(set(SECTOR_ETFS.values()))

    # ETFs (especially SPY) drive HMM regime detection which requires a 504
    # trading-day fit window plus a 31-day predict window. Without backward
    # extension the regime detector falls back to NEUTRAL for the first ~30
    # dates of any backtest and the HMM training collapses into degenerate
    # clusters (BULL/NEUTRAL/BEAR with near-identical stats). Pull
    # ETF_LOOKBACK_CALENDAR_DAYS (~800d = 2 + 1 years) before ctx.start_date.
    etf_start = ctx.start_date - timedelta(days=ETF_LOOKBACK_CALENDAR_DAYS)

    await ctx.log("info",
                  f"Backfilling {len(etf_symbols)} ETFs from {etf_start} "
                  f"(={ctx.start_date} - {ETF_LOOKBACK_CALENDAR_DAYS}d lookback) "
                  f"to {ctx.end_date}")

    pool = await asyncpg.create_pool(DATABASE_URL, min_size=2, max_size=8)
    inserted_total = 0
    skipped_total = 0
    failed: list = []

    async with capture_logs(ctx), monitor_progress(
        ctx,
        f"SELECT COUNT(*) FROM us_daily_etf WHERE date BETWEEN '{etf_start}' AND '{ctx.end_date}'",
        "us_daily_etf rows (in range)", interval=30,
    ):
        async with aiohttp.ClientSession() as session:
            for sym in etf_symbols:
                # 1) Compute missing dates
                missing = await cstate.get_missing_dates(
                    pool, "us_daily_etf", sym, etf_start, ctx.end_date)
                if not missing:
                    skipped_total += 1
                    continue

                # 2) Fetch one full-history AV call
                params = {
                    "function": "TIME_SERIES_DAILY",
                    "symbol": sym,
                    "outputsize": "full",
                    "apikey": api_key,
                    "datatype": "json",
                }
                try:
                    async with session.get(
                        "https://www.alphavantage.co/query",
                        params=params, timeout=aiohttp.ClientTimeout(total=60),
                    ) as resp:
                        data = await resp.json()
                except Exception as e:
                    await ctx.log("error", f"{sym} fetch error: {e}")
                    failed.append({"symbol": sym, "error": str(e)[:200]})
                    await asyncio.sleep(0.2)
                    continue

                series = data.get("Time Series (Daily)")
                if not series:
                    msg = data.get("Note") or data.get("Error Message") or "no series"
                    await ctx.log("warn", f"{sym} no data: {msg[:120]}")
                    failed.append({"symbol": sym, "error": str(msg)[:200]})
                    await asyncio.sleep(0.2)
                    continue

                # 3) Filter to missing dates + INSERT
                missing_set = {d.isoformat() for d in missing}
                rows = []
                parse_errors = 0
                for date_str, ohlcv in series.items():
                    if date_str not in missing_set:
                        continue
                    try:
                        rows.append((
                            sym,
                            date.fromisoformat(date_str),
                            float(ohlcv.get("1. open")    or 0),
                            float(ohlcv.get("2. high")    or 0),
                            float(ohlcv.get("3. low")     or 0),
                            float(ohlcv.get("4. close")   or 0),
                            int(float(ohlcv.get("5. volume") or 0)),
                        ))
                    except Exception as parse_err:
                        parse_errors += 1
                        if parse_errors <= 2:
                            await ctx.log("warn",
                                          f"{sym} parse err {date_str}: {parse_err}")

                if rows:
                    async with pool.acquire() as conn:
                        await conn.executemany(
                            """INSERT INTO us_daily_etf
                               (symbol, date, open, high, low, close, volume)
                               VALUES ($1, $2, $3, $4, $5, $6, $7)
                               ON CONFLICT (symbol, date) DO UPDATE SET
                                   open=EXCLUDED.open, high=EXCLUDED.high,
                                   low=EXCLUDED.low, close=EXCLUDED.close,
                                   volume=EXCLUDED.volume""",
                            rows,
                        )
                    # 4) Mark collected dates
                    await cstate.mark_collected(
                        pool, "us_daily_etf", sym, [r[1] for r in rows])
                    inserted_total += len(rows)
                    await ctx.log("ok",
                                  f"{sym}: +{len(rows)} rows (had {len(missing)} missing)")

                await asyncio.sleep(0.2)  # AV rate limit safety

        # fail 종목 → no_data mark (today exclude 라 내일 자동 재시도)
        if failed:
            async with pool.acquire() as conn_mark:
                await conn_mark.executemany(
                    """INSERT INTO collection_state
                       (collection_name, symbol, date, status)
                       VALUES ('us_daily_etf', $1, CURRENT_DATE, 'no_data')
                       ON CONFLICT (collection_name, symbol, date) DO UPDATE
                         SET status='no_data', collected_at=NOW()""",
                    [(f["symbol"],) for f in failed])
            await ctx.log("info",
                          f"no_data marked for {len(failed)} failed ETF symbols")

    await pool.close()

    if inserted_total == 0 and not failed and skipped_total == len(etf_symbols):
        await ctx.log("info", "All ETFs already fully collected — nothing to do")

    return {
        "etf_count":   len(etf_symbols),
        "inserted":    inserted_total,
        "skipped_all_done": skipped_total,
        "failed_count": len(failed),
        "first_failures": failed[:5],
    }


async def task_us_weekly(ctx: Ctx) -> dict:
    """US Weekly OHLCV 백필. us_calculator 가 이 데이터를 필수로 요구하기 때문.

    WeeklyCollector 가 TIME_SERIES_WEEKLY API 호출 → 종목당 1콜로 전체 weekly
    히스토리 (수년치). 4,592 종목 × 0.2초 ≈ 15~20분.
    """
    if ctx.country != "US":
        return {"skipped": "KR run"}

    api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    if not api_key:
        raise RuntimeError("ALPHAVANTAGE_API_KEY not set")

    from us.alphavantage import WeeklyCollector

    async with capture_logs(ctx), monitor_progress(
        ctx,
        "SELECT COUNT(*) FROM us_weekly",
        "us_weekly rows", interval=30):
        col = WeeklyCollector(api_key, DATABASE_URL, max_concurrent=20)
        await col.run_collection()

    pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=2)
    try:
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """SELECT COUNT(*) AS rows, COUNT(DISTINCT symbol) AS syms,
                          COUNT(DISTINCT date) AS weeks,
                          MIN(date) AS earliest, MAX(date) AS latest
                   FROM us_weekly""")
    finally:
        await pool.close()

    if row["rows"] == 0:
        raise RuntimeError(
            "us_weekly empty after WeeklyCollector ran — check AV key/quota")

    return dict(row)


async def task_us_calculator(ctx: Ctx) -> dict:
    """Compute 14 technical indicators per date in ctx range.

    run_calculator() processes ONE date at a time (it queries us_daily
    WHERE date = $1). To get 1년치 indicators we loop through every
    trading day in [extended_start, end_date] and call run_calculator
    once per date.

    API 콜 0 — 순수 DB 계산. 일자당 약 5~10초.
    """
    if ctx.country != "US":
        return {"skipped": "KR run"}

    # Same extended range as us_daily so indicators match (EM8 lookback safe)
    # EM8 = 240 trading days lookback. 240 × (365/252) ≈ 348 calendar days.
    # 400 = 348 + safety margin (holidays, missing days).
    LOOKBACK_BUFFER_DAYS = DAILY_LOOKBACK_CALENDAR_DAYS
    extended_start = ctx.start_date - timedelta(days=LOOKBACK_BUFFER_DAYS)

    # Get trading days actually in us_daily within range
    pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=3)
    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT DISTINCT date FROM us_daily
                   WHERE date BETWEEN $1 AND $2 ORDER BY date""",
                extended_start, ctx.end_date)
            trading_days = [r["date"] for r in rows]

            # Skip dates that already have indicators populated
            done_rows = await conn.fetch(
                """SELECT DISTINCT date FROM us_indicators
                   WHERE date BETWEEN $1 AND $2""",
                extended_start, ctx.end_date)
            done_dates = {r["date"] for r in done_rows}
    finally:
        await pool.close()

    pending_dates = [d for d in trading_days if d not in done_dates]
    await ctx.log("info",
                  f"us_calculator: {len(trading_days)} trading days in range, "
                  f"{len(done_dates)} already done, "
                  f"{len(pending_dates)} to process")

    if not pending_dates:
        return {"status": "all_done", "dates": len(trading_days)}

    from us.us_calculator import USTechnicalIndicatorCalculator
    async with capture_logs(ctx), monitor_progress(
        ctx,
        "SELECT COUNT(*) FROM us_indicators",
        "us_indicators rows", interval=60):
        calc = USTechnicalIndicatorCalculator(database_url=DATABASE_URL,
                                              max_concurrent_batches=40)
        # Single batched call: compute full indicator series once per symbol
        # for every pending date. ~200x less compute than the per-date loop
        # (pandas was internally producing N rolling values per indicator
        # call but the per-date path discarded all but iloc[-1]).
        result = await calc.run_calculator_range(pending_dates)

    return {"status": "ok",
            "processed_dates": len(pending_dates),
            "rows_upserted": result.get("rows_upserted", 0)}


async def task_macros(ctx: Ctx) -> dict:
    if ctx.country != "US":
        return {"skipped": "KR run"}
    out = {}
    for ep in ["collect/us/fed-funds-rate", "collect/us/treasury-yield",
               "collect/us/cpi", "collect/us/unemployment-rate"]:
        try:
            out[ep] = await _self_post(ctx, ep)
        except Exception as e:
            await ctx.log("warn", f"{ep} failed ({e}) — continuing")
            out[ep] = {"error": str(e)[:200]}
    return out


async def task_kr_daily(ctx: Ctx) -> dict:
    if ctx.country != "KR":
        return {"skipped": "US run"}
    return await _self_post(ctx, "collect/kr/daily-complete")


async def task_kr_dart(ctx: Ctx) -> dict:
    if ctx.country != "KR":
        return {"skipped": "US run"}
    out = {}
    for ep in ["collect/kr/dart/company-info",
               "collect/kr/dart/financial-position",
               "collect/kr/dart/dividends"]:
        try:
            out[ep] = await _self_post(ctx, ep)
        except Exception as e:
            await ctx.log("warn", f"{ep} failed ({e})")
            out[ep] = {"error": str(e)[:200]}
    return out


# ------------------------------------- Sector / industry MV bulk populate
async def task_us_mv_sector_refresh(ctx: Ctx) -> dict:
    """Bulk-populate ``mv_us_sector_daily_performance`` for the full date range.

    The legacy code in alphafolio_quant calls ``refresh_sector_performance_for_date``
    per-date inside grades_pass_a, which is slow and was silently failing
    because the table didn't exist (migration 0015 creates it). Computing the
    whole range in one SQL is much faster — sector_count × dates is tiny
    relative to one day's grade work, and skipping the per-date refresh
    saves a roundtrip per grades_pass_a date.

    Formula matches the legacy per-date refresh exactly:
        avg_return_30d = AVG((close_d - close_{d-30}) / close_{d-30} * 100)
                         over symbols in sector, where close_{d-30} is the
                         most recent us_daily row with d-40 <= date <= d-30.
        sector_rank    = ROW_NUMBER per date over sectors by avg_return_30d DESC.
    """
    if ctx.country != "US":
        return {"skipped": "KR run"}

    pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=3)
    try:
        async with capture_logs(ctx):
            await ctx.log("info",
                          f"mv_us_sector_refresh: {ctx.start_date} ~ {ctx.end_date}")
            async with pool.acquire() as conn:
                # 1) Drop existing rows in range to avoid stale sector ranks.
                await conn.execute("""
                    DELETE FROM mv_us_sector_daily_performance
                    WHERE date BETWEEN $1 AND $2
                """, ctx.start_date, ctx.end_date)

                # 2) Bulk INSERT.
                # us_stock_basic is point-in-time per (symbol, date) so we
                # use the row with date <= analysis date (latest sector
                # classification known on that day).
                result = await conn.execute("""
                    INSERT INTO mv_us_sector_daily_performance
                        (date, sector_code, avg_return_30d, stock_count, sector_rank)
                    WITH dates AS (
                        SELECT DISTINCT date FROM us_daily
                        WHERE date BETWEEN $1 AND $2
                    ),
                    sym_sector AS (
                        SELECT DISTINCT ON (symbol, dates.date)
                               symbol, dates.date AS asof, sector
                        FROM us_stock_basic b, dates
                        WHERE b.date <= dates.date
                          AND b.sector IS NOT NULL AND b.sector != ''
                        ORDER BY symbol, dates.date, b.date DESC
                    ),
                    daily_close AS (
                        SELECT cur.symbol, cur.date, cur.close AS close_now,
                               LAG(cur.close, 30) OVER (
                                 PARTITION BY cur.symbol ORDER BY cur.date) AS close_30d_ago
                        FROM us_daily cur
                        WHERE cur.date BETWEEN $1 - INTERVAL '60 days' AND $2
                    ),
                    rets AS (
                        SELECT dc.date,
                               ss.sector,
                               ((dc.close_now - dc.close_30d_ago) / NULLIF(dc.close_30d_ago, 0)) * 100 AS ret_30d,
                               dc.symbol
                        FROM daily_close dc
                        JOIN sym_sector ss
                          ON ss.symbol = dc.symbol AND ss.asof = dc.date
                        WHERE dc.date BETWEEN $1 AND $2
                          AND dc.close_now IS NOT NULL
                          AND dc.close_30d_ago IS NOT NULL
                    ),
                    agg AS (
                        SELECT date, sector,
                               AVG(ret_30d) AS avg_ret,
                               COUNT(DISTINCT symbol) AS cnt
                        FROM rets
                        GROUP BY date, sector
                    )
                    SELECT date, sector,
                           avg_ret, cnt,
                           ROW_NUMBER() OVER (PARTITION BY date ORDER BY avg_ret DESC NULLS LAST)
                    FROM agg
                    ON CONFLICT (date, sector_code) DO NOTHING
                """, ctx.start_date, ctx.end_date)

                row_count = await conn.fetchval(
                    "SELECT COUNT(*) FROM mv_us_sector_daily_performance "
                    "WHERE date BETWEEN $1 AND $2",
                    ctx.start_date, ctx.end_date)
                date_count = await conn.fetchval(
                    "SELECT COUNT(DISTINCT date) FROM mv_us_sector_daily_performance "
                    "WHERE date BETWEEN $1 AND $2",
                    ctx.start_date, ctx.end_date)
            await ctx.log("info",
                          f"mv_us_sector_refresh: {row_count} rows across "
                          f"{date_count} dates")
            return {"rows": row_count, "dates": date_count}
    finally:
        await pool.close()


# ------------------------------------------------------- EM8 사전 필터 (NEW)
async def task_em8_pre_filter(ctx: Ctx) -> dict:
    """매 거래일별로 EM8(252일 가격 모멘텀)을 SQL 일괄 계산하여
    상위 N개 종목을 daily_top_symbols 테이블에 저장.

    이 결과를 grades_pass_a 에서 사용하면 4592종목 → 500종목으로
    분석 대상이 줄어 약 9~10배 단축.

    EM8 공식과 SQL 결과가 수학적으로 동일함은 검증 완료
    (Python per-stock vs SQL bulk, diff < 1e-15).
    """
    if ctx.country != "US":
        return {"skipped": "KR run"}

    top_n = int(ctx.params.get("prefilter_top_n", 500))

    pool = await asyncpg.create_pool(DATABASE_URL, min_size=2, max_size=5)
    dates_processed = []
    failed_dates = []

    async with capture_logs(ctx):
        await ctx.log("info",
                      f"EM8 pre-filter starting: top-{top_n}, "
                      f"{ctx.start_date}~{ctx.end_date}")
        d = ctx.start_date
        while d <= ctx.end_date:
            if d.weekday() >= 5:
                d += timedelta(days=1)
                continue
            try:
                async with pool.acquire() as conn:
                    rows = await conn.fetch("""
                        WITH ranked AS (
                          SELECT symbol, close,
                                 ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY date DESC) AS rn
                          FROM us_daily WHERE date <= $1
                        ),
                        pts AS (
                          SELECT symbol,
                            MAX(CASE WHEN rn = 21  THEN close END) AS p_skip,
                            MAX(CASE WHEN rn = 63  THEN close END) AS p_63,
                            MAX(CASE WHEN rn = 126 THEN close END) AS p_126,
                            MAX(CASE WHEN rn = 189 THEN close END) AS p_189,
                            MAX(CASE WHEN rn = 240 THEN close END) AS p_240
                          FROM ranked GROUP BY symbol
                        ),
                        em8 AS (
                          SELECT symbol,
                            ((p_skip - p_63 ) / NULLIF(p_63,  0)) * 100 * 0.4 +
                            ((p_skip - p_126) / NULLIF(p_126, 0)) * 100 * 0.2 +
                            ((p_skip - p_189) / NULLIF(p_189, 0)) * 100 * 0.2 +
                            ((p_skip - p_240) / NULLIF(p_240, 0)) * 100 * 0.2 AS rs_value
                          FROM pts
                          WHERE p_skip IS NOT NULL AND p_63 IS NOT NULL
                            AND p_126 IS NOT NULL AND p_189 IS NOT NULL
                            AND p_240 IS NOT NULL
                        ),
                        ranked_em8 AS (
                          SELECT symbol, rs_value,
                                 ROW_NUMBER() OVER (ORDER BY rs_value DESC NULLS LAST) AS rank
                          FROM em8
                        )
                        SELECT symbol, rs_value, rank FROM ranked_em8
                        WHERE rank <= $2 ORDER BY rank
                    """, d, top_n)

                    if rows:
                        await conn.executemany("""
                            INSERT INTO daily_top_symbols
                                (date, symbol, rank, em8_score, computed_at)
                            VALUES ($1, $2, $3, $4, NOW())
                            ON CONFLICT (date, symbol) DO UPDATE SET
                                rank = EXCLUDED.rank,
                                em8_score = EXCLUDED.em8_score,
                                computed_at = NOW()
                        """, [(d, r["symbol"], r["rank"], float(r["rs_value"]))
                              for r in rows])
                        dates_processed.append(str(d))
                        if len(dates_processed) % 10 == 0:
                            await ctx.log("info",
                                          f"em8 pre-filter: {len(dates_processed)} dates done, "
                                          f"latest [{d}] top5: "
                                          f"{', '.join(r['symbol'] for r in rows[:5])}")
                    else:
                        failed_dates.append(str(d))
                        await ctx.log("warn",
                                      f"em8 {d}: 0 symbols (insufficient history)")
            except Exception as e:
                failed_dates.append(str(d))
                await ctx.log("error", f"em8 {d}: {e}")
            d += timedelta(days=1)
    await pool.close()

    return {
        "top_n":            top_n,
        "dates_processed":  len(dates_processed),
        "failed_dates":     len(failed_dates),
        "sample_failures":  failed_dates[:5],
    }


# ------------------------------------------------------- Pass-A grades (via quant)
async def task_grades_pass_a(ctx: Ctx) -> dict:
    """Generate grades. quant 가 daily_top_symbols 를 자동으로 참조하므로
    EM8 사전 필터를 통과한 종목들만 분석함 (use_prefilter=True 신호).

    Pass-A runs BEFORE options are collected — it only ranks symbols to decide
    which get options backfilled. So with_event_modifier=False: skip the
    event_engine (option/GEX/earnings) modifier, which would be stale noise
    here and pure cost. Pass-B re-grades the same universe with events on."""
    return await _quant_post(ctx, "backtest/generate-grades", {
        "country":             ctx.country,
        "start_date":          str(ctx.start_date),
        "end_date":            str(ctx.end_date),
        "skip_existing":       True,
        "use_prefilter":       True,
        "prefilter_top_n":     ctx.params.get("prefilter_top_n", 500),
        "with_event_modifier": False,
    })


# ------------------------------------------ Top-N symbol selection (DB-side)
async def task_select_top_n(ctx: Ctx) -> dict:
    if ctx.country != "US":
        return {"skipped": "options pass only for US"}
    top_n = int(ctx.params.get("option_top_n", 50))
    pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=3)
    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT DISTINCT symbol FROM (
                  SELECT date, symbol, final_score,
                         ROW_NUMBER() OVER (PARTITION BY date ORDER BY final_score DESC NULLS LAST) AS rn
                  FROM us_stock_grade
                  WHERE date BETWEEN $1 AND $2
                    AND final_grade IN ('STRONG_BUY','BUY','NEUTRAL')
                    AND final_score IS NOT NULL
                ) t WHERE rn <= $3
            """, ctx.start_date, ctx.end_date, top_n)
            symbols = [r["symbol"] for r in rows]
    finally:
        await pool.close()
    await ctx.log("info", f"top-{top_n}/date → {len(symbols)} unique symbols")
    return {"top_n_per_date": top_n, "unique_symbols": len(symbols),
            "sample": symbols[:25]}


# --------------------- Pass-B options backfill (only the top-N symbols, looped)
async def task_options_top_n(ctx: Ctx) -> dict:
    """Options backfill for top-N symbols per date.

    Idempotent / resumable: dates that already have rows in
    ``us_option_daily_summary`` are skipped — useful when the task got killed
    mid-run (e.g., previous DiskFullError after 198/252 dates). The us_option
    raw partition is dropped automatically inside the collector once the
    summary is computed, so skipping by summary presence is the right signal.
    """
    if ctx.country != "US":
        return {"skipped": "options only for US"}

    # Build skip set. The valid-date source is ``us_stock_grade``, NOT
    # ``us_daily``: a single stray us_daily row on a market holiday
    # (e.g., Presidents Day 2026-02-16) would otherwise fool the skip and
    # the collector would fall back to a 540-symbol whitelist returning 0
    # contracts each (15 min wasted per holiday).
    # Using us_stock_grade is also semantically correct — the collector
    # needs top-N grades to pick symbols, so any date without grades is
    # unusable regardless of trading-day status.
    pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=2)
    try:
        async with pool.acquire() as conn:
            done_rows = await conn.fetch(
                """SELECT DISTINCT date FROM us_option_daily_summary
                   WHERE date BETWEEN $1 AND $2""",
                ctx.start_date, ctx.end_date)
            already_done = {r["date"] for r in done_rows}
            grade_rows = await conn.fetch(
                """SELECT DISTINCT date FROM us_stock_grade
                   WHERE date BETWEEN $1 AND $2""",
                ctx.start_date, ctx.end_date)
            graded_days = {r["date"] for r in grade_rows}
    finally:
        await pool.close()
    await ctx.log("info",
                  f"options_top_n: {len(already_done)} already summarized, "
                  f"{len(graded_days)} dates have grades (=trading days w/ top-N)")

    d = ctx.start_date
    ok, fail, skipped, holidays = 0, 0, 0, 0
    while d <= ctx.end_date:
        if d.weekday() >= 5:
            d += timedelta(days=1); continue
        if d not in graded_days:
            # No grades for this date — either market holiday or grades_pass_a
            # didn't cover it. Either way, options collection is unusable.
            holidays += 1
            d += timedelta(days=1); continue
        if d in already_done:
            skipped += 1
            d += timedelta(days=1); continue
        try:
            # Per-date mode (no start_date/end_date → collector picks that date's
            # top-N by grade). The score-affecting option signals (event_engine
            # options_modifier + gex_modifier, growth NQ4) read only the latest
            # option row ≤ analysis_date, so per-date coverage is sufficient for
            # backtest results. Union (continuous 252-day history) is only needed
            # for agent_metrics iv_percentile, which doesn't feed final_score.
            await _self_post(ctx, "collect/us/options-top-n",
                             params={"target_date": d.isoformat(),
                                     "top_n":       ctx.params.get("option_top_n", 50)})
            ok += 1
        except Exception as e:
            await ctx.log("error", f"options {d}: {e}")
            fail += 1
        d += timedelta(days=1)
    return {"success_days": ok, "failed_days": fail,
            "skipped_days": skipped, "holiday_days": holidays}


# -------------------------------------------- Pass-B grades w/ options (top-N)
async def task_grades_pass_b(ctx: Ctx) -> dict:
    if ctx.country != "US":
        return {"skipped": "Pass-B only for US"}
    # generate-grades is idempotent via skip_existing=False so it overwrites.
    # use_prefilter=True restricts re-grading to the EM8 top-N per date (same
    # universe as Pass-A) — without it the endpoint defaults to the FULL
    # ~5,500-symbol universe (10x slower: ~19h vs ~1.5h). Options were only
    # collected for the top-N union, so re-grading the full universe wastes
    # time on ~5,000 symbols that have no options and thus produce identical
    # grades to Pass-A.
    return await _quant_post(ctx, "backtest/generate-grades", {
        "country": ctx.country,
        "start_date": str(ctx.start_date),
        "end_date":   str(ctx.end_date),
        "skip_existing": False,
        "use_prefilter":   True,
        "prefilter_top_n": ctx.params.get("prefilter_top_n", 500),
        # Pass-B folds in option signals (options_modifier + gex_modifier) now
        # that options are collected — this is the entire point of the 2nd pass.
        "with_event_modifier": True,
    })


async def task_backtest(ctx: Ctx) -> dict:
    return await _quant_post(ctx, "backtest/run", {
        "country": ctx.country,
        "start_date": str(ctx.start_date),
        "end_date":   str(ctx.end_date),
        "initial_cash":     ctx.params.get("initial_cash", 10_000_000),
        "top_n":            ctx.params.get("backtest_top_n", 10),
        "rebal_freq_days":  ctx.params.get("rebal_freq_days", 5),
    })


async def task_earnings_history(ctx: Ctx) -> dict:
    """AV EARNINGS endpoint 호출 → us_earnings_history 적재.

    각 종목당 1 call → quarterlyEarnings 의 reportedDate (실제 공시일자) 캡쳐.
    financials 3종의 available_at 정밀화에 사용 (income/balance/cashflow 는
    모두 같은 10-Q/10-K 에 묶여 같은 reported_date 공유).

    rate-limit retry decorator (15s 지수 backoff, max 5분 cap) 로 안전.
    """
    if ctx.country != "US":
        return {"skipped": "KR run"}
    api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    if not api_key:
        raise RuntimeError("ALPHAVANTAGE_API_KEY not set")

    from us.alphavantage import EarningsHistoryCollector
    async with capture_logs(ctx), monitor_progress(
        ctx,
        "SELECT COUNT(*) FROM us_earnings_history",
        "us_earnings_history rows", interval=30):
        col = EarningsHistoryCollector(api_key, DATABASE_URL, max_concurrent=3)
        result = await col.run_collection()
    return result


async def task_us_stock_basic_compute(ctx: Ctx) -> dict:
    """us_stock_basic 의 source='computed' row 적재 — 시점별 정밀 재계산.

    pandas 벡터화 + as-of merge 로 look-ahead-bias 안전:
      - us_daily 의 각 (symbol, date) 마다 fiscal_date_ending + 45 일 reporting
        lag 시점 까지의 us_income_statement / us_balance_sheet 만 사용
      - 분기별 trailing-4Q rolling 으로 TTM 지표 (revenue, net_income, ebitda,
        operating_income, gross_profit) 산출
      - YoY growth: 현재 분기 / 4분기 전 분기

    채우는 컬럼 (17 개 시점별 + 메타):
      market_cap, per, pricetobookratio, eps, dilutedepsttm, bookvalue,
      sharesoutstanding, revenuettm, grossprofitttm, profitmargin,
      operatingmarginttm, returnonequityttm, returnonassetsttm,
      pricetosalesratiottm, evtorevenue, evtoebitda,
      quarterlyearningsgrowthyoy, quarterlyrevenuegrowthyoy,
      week52high, week52low, day50movingaverage, day200movingaverage,
      + 메타 (stock_name, exchange, currency, sector, industry, beta, is_active)

    bulk INSERT 전략: pandas → asyncpg copy_records_to_table 로 임시 테이블 적재
    → 본 테이블 INSERT … ON CONFLICT (symbol, date, source) DO UPDATE.
    """
    if ctx.country != "US":
        return {"skipped": "KR run"}

    import pandas as pd
    import numpy as np

    LOOKBACK_BUFFER_DAYS = DAILY_LOOKBACK_CALENDAR_DAYS
    start = ctx.start_date - timedelta(days=LOOKBACK_BUFFER_DAYS)
    end = ctx.end_date

    async with capture_logs(ctx), monitor_progress(
        ctx,
        "SELECT COUNT(*) FROM us_stock_basic WHERE source='computed'",
        "us_stock_basic computed rows", interval=60):

        pool = await asyncpg.create_pool(DATABASE_URL, min_size=2, max_size=5)
        try:
            # ---------- 1) 모든 소스 데이터 한 번에 로드 ----------
            async with pool.acquire() as conn:
                daily_rows = await conn.fetch(
                    "SELECT symbol, date, close FROM us_daily "
                    "WHERE date BETWEEN $1 AND $2", start, end)
                inc_rows = await conn.fetch(
                    """SELECT symbol, fiscal_date_ending, available_at,
                              net_income, total_revenue, operating_income,
                              gross_profit, ebitda
                       FROM us_income_statement
                       WHERE fiscal_date_ending >= $1::date - INTERVAL '5 years'""",
                    start)
                bal_rows = await conn.fetch(
                    """SELECT symbol, fiscal_date_ending, available_at,
                              total_assets, total_shareholder_equity,
                              common_stock_shares_outstanding,
                              short_long_term_debt_total,
                              cash_and_short_term_investments
                       FROM us_balance_sheet
                       WHERE fiscal_date_ending >= $1::date - INTERVAL '5 years'""",
                    start)
                api_rows = await conn.fetch(
                    """SELECT DISTINCT ON (symbol) symbol, stock_name, exchange,
                              currency, sector, industry, beta, is_active,
                              sharesoutstanding AS api_shares
                       FROM us_stock_basic WHERE source='api'
                       ORDER BY symbol, date DESC""")

            logger.info(f"loaded: daily={len(daily_rows)} inc={len(inc_rows)} "
                        f"bal={len(bal_rows)} api_meta={len(api_rows)}")

            if not daily_rows:
                return {"computed_rows": 0, "reason": "no us_daily rows in range"}

            daily_df = pd.DataFrame(daily_rows, columns=['symbol', 'date', 'close'])
            daily_df['date'] = pd.to_datetime(daily_df['date'])
            daily_df['close'] = daily_df['close'].astype('float64')

            api_df = pd.DataFrame(api_rows,
                columns=['symbol', 'stock_name', 'exchange', 'currency',
                         'sector', 'industry', 'beta', 'is_active', 'api_shares'])

            # ---------- 2) financials → 분기별 TTM + YoY ----------
            if inc_rows or bal_rows:
                inc_df = pd.DataFrame(inc_rows, columns=[
                    'symbol', 'fiscal_date_ending', 'available_at_inc',
                    'net_income', 'total_revenue', 'operating_income',
                    'gross_profit', 'ebitda'])
                bal_df = pd.DataFrame(bal_rows, columns=[
                    'symbol', 'fiscal_date_ending', 'available_at_bal',
                    'total_assets', 'total_shareholder_equity',
                    'common_stock_shares_outstanding',
                    'short_long_term_debt_total',
                    'cash_and_short_term_investments'])
                fin_df = pd.merge(inc_df, bal_df,
                                  on=['symbol', 'fiscal_date_ending'], how='outer')
            else:
                # financials 아직 적재 전 — 메타만 채운 row 적재
                fin_df = pd.DataFrame(columns=['symbol', 'fiscal_date_ending'])

            if not fin_df.empty:
                fin_df['fiscal_date_ending'] = pd.to_datetime(fin_df['fiscal_date_ending'])
                fin_df = fin_df.sort_values(['symbol', 'fiscal_date_ending']).reset_index(drop=True)

                # 모든 numeric 컬럼 float64
                num_cols = ['net_income', 'total_revenue', 'operating_income',
                            'gross_profit', 'ebitda',
                            'total_assets', 'total_shareholder_equity',
                            'common_stock_shares_outstanding',
                            'short_long_term_debt_total',
                            'cash_and_short_term_investments']
                for c in num_cols:
                    fin_df[c] = pd.to_numeric(fin_df[c], errors='coerce')

                grp = fin_df.groupby('symbol', sort=False)
                # TTM = trailing 4Q sum
                for c in ['net_income', 'total_revenue', 'operating_income',
                          'gross_profit', 'ebitda']:
                    fin_df[f'{c}_ttm'] = grp[c].transform(
                        lambda s: s.rolling(4, min_periods=1).sum())
                # 분모용 2Q avg
                for c in ['total_assets', 'total_shareholder_equity']:
                    fin_df[f'{c}_avg'] = grp[c].transform(
                        lambda s: s.rolling(2, min_periods=1).mean())
                # YoY = 4Q ago
                fin_df['net_income_4q_ago'] = grp['net_income'].shift(4)
                fin_df['revenue_4q_ago'] = grp['total_revenue'].shift(4)

                # available_at: us_income_statement/us_balance_sheet 의 정밀 값 사용
                # (이미 financials INSERT 시 us_earnings_history.reported_date 로 채워짐).
                # 두 컬럼 중 우선 income, 없으면 balance, 둘 다 NULL 이면 +45일 fallback.
                fin_df['available_at'] = pd.to_datetime(
                    fin_df.get('available_at_inc')).fillna(
                    pd.to_datetime(fin_df.get('available_at_bal'))).fillna(
                    fin_df['fiscal_date_ending'] + pd.Timedelta(days=45))
                fin_df = fin_df.dropna(subset=['available_at'])
                # merge_asof (pandas 3.x) requires the on-key globally monotonic
                # even when `by` is used — single-key sort, not hierarchical.
                fin_df = fin_df.sort_values('available_at', kind='mergesort').reset_index(drop=True)

                # ---------- 3) as-of merge ----------
                daily_df = daily_df.sort_values('date', kind='mergesort').reset_index(drop=True)
                merged = pd.merge_asof(
                    daily_df, fin_df,
                    left_on='date', right_on='available_at',
                    by='symbol', direction='backward')
            else:
                merged = daily_df.copy()

            # ---------- 4) us_daily rolling — 52w high/low, 50/200 MA ----------
            daily_df_sorted = daily_df.sort_values(['symbol', 'date']).reset_index(drop=True)
            grpd = daily_df_sorted.groupby('symbol', sort=False)['close']
            daily_df_sorted['week52high'] = grpd.transform(
                lambda s: s.rolling(252, min_periods=1).max())
            daily_df_sorted['week52low'] = grpd.transform(
                lambda s: s.rolling(252, min_periods=1).min())
            daily_df_sorted['day50movingaverage'] = grpd.transform(
                lambda s: s.rolling(50, min_periods=1).mean())
            daily_df_sorted['day200movingaverage'] = grpd.transform(
                lambda s: s.rolling(200, min_periods=1).mean())

            merged = pd.merge(merged,
                daily_df_sorted[['symbol', 'date', 'week52high', 'week52low',
                                 'day50movingaverage', 'day200movingaverage']],
                on=['symbol', 'date'], how='left')

            # 메타 join
            merged = merged.merge(api_df, on='symbol', how='left')

            # ---------- 5) 시점별 17 컬럼 벡터 계산 ----------
            # shares 출처: balance_sheet 의 시점별 값 우선, 없으면 today api 값 fallback
            so = merged.get('common_stock_shares_outstanding')
            if so is None:
                so = pd.Series([np.nan] * len(merged))
            so = so.fillna(merged['api_shares']).astype('float64')
            close = merged['close']

            def _safe_div(num, den):
                d = den.where(den > 0)
                return num / d

            merged['sharesoutstanding'] = so
            merged['market_cap'] = (close * so).where(so > 0)
            # eps_ttm = net_income_ttm / shares
            ni_ttm = merged.get('net_income_ttm', pd.Series([np.nan] * len(merged)))
            merged['eps'] = _safe_div(ni_ttm, so)
            merged['dilutedepsttm'] = merged['eps']
            merged['per'] = _safe_div(close, merged['eps'])
            # bookvalue per share
            equity = merged.get('total_shareholder_equity',
                                pd.Series([np.nan] * len(merged)))
            merged['bookvalue'] = _safe_div(equity, so)
            merged['pricetobookratio'] = _safe_div(close, merged['bookvalue'])
            # TTM aggregates
            rev_ttm = merged.get('total_revenue_ttm', pd.Series([np.nan] * len(merged)))
            merged['revenuettm'] = rev_ttm
            merged['grossprofitttm'] = merged.get('gross_profit_ttm')
            merged['profitmargin'] = _safe_div(ni_ttm, rev_ttm)
            merged['operatingmarginttm'] = _safe_div(
                merged.get('operating_income_ttm',
                           pd.Series([np.nan] * len(merged))), rev_ttm)
            merged['returnonequityttm'] = _safe_div(ni_ttm,
                merged.get('total_shareholder_equity_avg',
                           pd.Series([np.nan] * len(merged))))
            merged['returnonassetsttm'] = _safe_div(ni_ttm,
                merged.get('total_assets_avg',
                           pd.Series([np.nan] * len(merged))))
            merged['pricetosalesratiottm'] = _safe_div(merged['market_cap'], rev_ttm)
            # Enterprise Value
            debt = merged.get('short_long_term_debt_total',
                              pd.Series([np.nan] * len(merged))).fillna(0)
            cash = merged.get('cash_and_short_term_investments',
                              pd.Series([np.nan] * len(merged))).fillna(0)
            ev = merged['market_cap'].fillna(0) + debt - cash
            merged['evtorevenue'] = _safe_div(ev, rev_ttm)
            merged['evtoebitda'] = _safe_div(ev,
                merged.get('ebitda_ttm', pd.Series([np.nan] * len(merged))))
            # YoY
            ni = merged.get('net_income', pd.Series([np.nan] * len(merged)))
            ni_yoy_base = merged.get('net_income_4q_ago',
                                     pd.Series([np.nan] * len(merged)))
            rev = merged.get('total_revenue', pd.Series([np.nan] * len(merged)))
            rev_yoy_base = merged.get('revenue_4q_ago',
                                      pd.Series([np.nan] * len(merged)))
            merged['quarterlyearningsgrowthyoy'] = _safe_div(ni, ni_yoy_base.abs()) - 1
            merged['quarterlyrevenuegrowthyoy'] = _safe_div(rev, rev_yoy_base.abs()) - 1

            # market_cap → BIGINT 변환
            merged['market_cap'] = merged['market_cap'].astype('Float64').round().astype('Int64')

            # ---------- 6) bulk INSERT via temp table + ON CONFLICT ----------
            now_ts = datetime.now(timezone.utc)
            insert_cols = [
                'symbol', 'date', 'source',
                'market_cap', 'per', 'pricetobookratio',
                'eps', 'dilutedepsttm', 'bookvalue', 'sharesoutstanding',
                'revenuettm', 'grossprofitttm',
                'profitmargin', 'operatingmarginttm',
                'returnonequityttm', 'returnonassetsttm',
                'pricetosalesratiottm', 'evtorevenue', 'evtoebitda',
                'quarterlyearningsgrowthyoy', 'quarterlyrevenuegrowthyoy',
                'week52high', 'week52low',
                'day50movingaverage', 'day200movingaverage',
                'stock_name', 'exchange', 'currency', 'sector', 'industry',
                'beta', 'is_active', 'created_at', 'updated_at',
            ]

            out = pd.DataFrame()
            out['symbol'] = merged['symbol']
            out['date'] = merged['date'].dt.date
            out['source'] = 'computed'
            for c in ['market_cap', 'per', 'pricetobookratio',
                      'eps', 'dilutedepsttm', 'bookvalue', 'sharesoutstanding',
                      'revenuettm', 'grossprofitttm',
                      'profitmargin', 'operatingmarginttm',
                      'returnonequityttm', 'returnonassetsttm',
                      'pricetosalesratiottm', 'evtorevenue', 'evtoebitda',
                      'quarterlyearningsgrowthyoy', 'quarterlyrevenuegrowthyoy',
                      'week52high', 'week52low',
                      'day50movingaverage', 'day200movingaverage']:
                out[c] = merged[c] if c in merged.columns else None
            for c in ['stock_name', 'exchange', 'currency', 'sector', 'industry',
                      'beta', 'is_active']:
                out[c] = merged[c]
            out['created_at'] = now_ts
            out['updated_at'] = now_ts

            # NaN / NaT → None (asyncpg)
            out = out.astype(object).where(pd.notna(out), None)

            # BIGINT 컬럼: pandas Int64 NA → None, 그 외 int
            def _to_bigint(v):
                if v is None or (isinstance(v, float) and (v != v)):
                    return None
                try:
                    return int(v)
                except (TypeError, ValueError):
                    return None
            # Series.apply with mixed int+None coerces back to float64 (None→NaN).
            # Build with explicit dtype=object to preserve None for asyncpg.
            for _bc in ('market_cap', 'revenuettm', 'grossprofitttm', 'sharesoutstanding'):
                out[_bc] = pd.Series(
                    [_to_bigint(v) for v in out[_bc]],
                    index=out.index, dtype=object)

            records = list(out[insert_cols].itertuples(index=False, name=None))
            total_records = len(records)
            logger.info(f"prepared {total_records} computed rows for bulk INSERT")

            # ---------- 7) temp table + MERGE ----------
            CHUNK = 50000
            inserted = 0
            async with pool.acquire() as conn:
                async with conn.transaction():
                    # Must be inside the txn — otherwise asyncpg autocommits the
                    # CREATE and `ON COMMIT DROP` fires immediately, dropping the
                    # table before the COPY runs.
                    await conn.execute("""
                        CREATE TEMP TABLE _us_stock_basic_compute (LIKE us_stock_basic
                        INCLUDING DEFAULTS) ON COMMIT DROP""")
                    for i in range(0, total_records, CHUNK):
                        chunk = records[i:i + CHUNK]
                        await conn.copy_records_to_table(
                            '_us_stock_basic_compute',
                            records=chunk, columns=insert_cols)
                        inserted += len(chunk)
                        logger.info(f"  staged {inserted}/{total_records}")

                    update_cols = [c for c in insert_cols
                                   if c not in ('symbol', 'date', 'source', 'created_at')]
                    set_clause = ', '.join(f"{c} = EXCLUDED.{c}" for c in update_cols)
                    await conn.execute(f"""
                        INSERT INTO us_stock_basic ({', '.join(insert_cols)})
                        SELECT {', '.join(insert_cols)} FROM _us_stock_basic_compute
                        ON CONFLICT (symbol, date, source) DO UPDATE SET {set_clause}
                    """)

                computed_count = await conn.fetchval(
                    """SELECT COUNT(*) FROM us_stock_basic
                       WHERE source='computed' AND date BETWEEN $1 AND $2""",
                    start, end)
        finally:
            await pool.close()

    return {"computed_rows": computed_count,
            "date_range": f"{start.isoformat()}~{end.isoformat()}",
            "staged": inserted}


# ============================================================ DAG registry
TASK_REGISTRY: dict[str, Callable] = {
    "partitions":    task_partitions,
    "stock_listing": task_stock_listing,
    "finnhub_symbol": task_finnhub_symbol,
    "stock_basic":   task_stock_basic,
    "stock_basic_compute": task_us_stock_basic_compute,
    "earnings_history": task_earnings_history,
    "us_daily":      task_us_daily,
    "us_etf":        task_us_etf,
    "us_weekly":     task_us_weekly,
    "us_calculator": task_us_calculator,
    "financials":    task_financials,
    "macros":        task_macros,
    "kr_daily":      task_kr_daily,
    "kr_dart":       task_kr_dart,
    "em8_pre_filter": task_em8_pre_filter,
    "us_mv_sector_refresh": task_us_mv_sector_refresh,
    "grades_pass_a": task_grades_pass_a,
    "select_top_n":  task_select_top_n,
    "options_top_n": task_options_top_n,
    "grades_pass_b": task_grades_pass_b,
    "backtest":      task_backtest,
}


def build_dag(country: str) -> List[dict]:
    if country == "US":
        return [
            {"id": "partitions",     "name": "0. DB partitions",                  "depends_on": []},
            {"id": "stock_listing",  "name": "1. NASDAQ/NYSE listing",            "depends_on": ["partitions"]},
            {"id": "finnhub_symbol", "name": "2. Finnhub symbol master",          "depends_on": ["stock_listing"]},
            {"id": "stock_basic",    "name": "3. US Stock Basic (fundamentals)",  "depends_on": ["finnhub_symbol"]},
            {"id": "us_daily",       "name": "4. US Daily OHLCV (outputsize=full)","depends_on": ["stock_basic"]},
            {"id": "us_etf",         "name": "5. US ETF daily (SPY/QQQ — 시장레짐)", "depends_on": ["partitions"]},
            {"id": "us_weekly",      "name": "5b. US Weekly OHLCV (indicator 의존)", "depends_on": ["stock_basic"]},
            {"id": "us_calculator",  "name": "6. Technical indicators (14종, DB 계산)", "depends_on": ["us_daily","us_weekly"]},
            {"id": "earnings_history", "name": "7a. Earnings history (reportedDate)", "depends_on": ["stock_basic"]},
            {"id": "financials",     "name": "7. Income / Balance / CashFlow",    "depends_on": ["stock_basic","earnings_history"]},
            {"id": "stock_basic_compute", "name": "7b. Stock Basic 시점별 computed (pandas asof, 17 컬럼)", "depends_on": ["us_daily","stock_basic","financials"]},
            {"id": "macros",         "name": "8. Macros (Fed/Treasury/CPI/UE)",   "depends_on": ["partitions"]},
            {"id": "em8_pre_filter", "name": "9. EM8 pre-filter (top-500/date)",  "depends_on": ["us_daily","us_weekly","us_calculator"]},
            {"id": "us_mv_sector_refresh", "name": "9b. Sector MV (mv_us_sector_daily_performance)", "depends_on": ["us_daily","stock_basic"]},
            {"id": "grades_pass_a",  "name": "10. Pass-A grades (top-N only)",    "depends_on": ["us_calculator","us_etf","financials","macros","em8_pre_filter","stock_basic_compute","us_mv_sector_refresh"]},
            {"id": "select_top_n",   "name": "10. Pick top-N symbols",            "depends_on": ["grades_pass_a"]},
            {"id": "options_top_n",  "name": "11. Options backfill (top-N only)", "depends_on": ["select_top_n"]},
            {"id": "grades_pass_b",  "name": "12. Pass-B grades (with options)",  "depends_on": ["options_top_n"]},
            {"id": "backtest",       "name": "13. Run backtest",                  "depends_on": ["grades_pass_b"]},
        ]
    else:
        return [
            {"id": "partitions",     "name": "0. DB partitions",          "depends_on": []},
            {"id": "kr_daily",       "name": "1. KR daily pipeline",       "depends_on": ["partitions"]},
            {"id": "kr_dart",        "name": "2. KR DART filings",         "depends_on": ["partitions"]},
            {"id": "grades_pass_a",  "name": "3. Generate grades",         "depends_on": ["kr_daily","kr_dart"]},
            {"id": "backtest",       "name": "4. Run backtest",            "depends_on": ["grades_pass_a"]},
        ]
