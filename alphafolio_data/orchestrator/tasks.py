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

async def _fetch_catalysts_for_symbols(api_key: str, symbols: list,
                                       news_per_symbol: int = 5,
                                       insider_lookback_days: int = 90) -> dict:
    """buy_now 종목별 호재 — bullish 뉴스 + 내부자 매수.

    각 종목당 AV NEWS_SENTIMENT + INSIDER_TRANSACTIONS 1콜씩 (총 6콜 for top-3).
    bullish 뉴스 상위 N + 최근 N일 인사이더 매수 집계를 반환. us_news /
    us_insider_transactions 에 best-effort 저장(ON CONFLICT DO NOTHING).
    호출 실패는 무시 — reco 본체 출력에 영향 없음.
    """
    import aiohttp
    import asyncio as _asyncio
    from datetime import datetime as _dt, timedelta as _td_l
    base = "https://www.alphavantage.co/query"
    result: dict = {}
    pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=3)
    try:
        async with aiohttp.ClientSession() as session:
            for sym in symbols:
                rec = {"news": [], "insider_buys": None}

                # ----- News (AV NEWS_SENTIMENT) -----
                try:
                    async with session.get(base, params={
                        "function": "NEWS_SENTIMENT", "tickers": sym,
                        "limit": "50", "apikey": api_key,
                    }, timeout=aiohttp.ClientTimeout(total=20)) as r:
                        data = await r.json()
                    feed = data.get("feed", []) if isinstance(data, dict) else []
                    bullish, persist = [], []
                    for art in feed:
                        ts_list = art.get("ticker_sentiment", []) or []
                        ts = next((t for t in ts_list if t.get("ticker") == sym), None)
                        if not ts:
                            continue
                        label = ts.get("ticker_sentiment_label", "") or ""
                        try:    score = float(ts.get("ticker_sentiment_score") or 0)
                        except: score = 0.0
                        try:    rel = float(ts.get("relevance_score") or 0)
                        except: rel = 0.0
                        tp_raw = art.get("time_published", "") or ""
                        try:    tp = _dt.strptime(tp_raw[:15], "%Y%m%dT%H%M%S")
                        except: tp = None
                        is_bull = label in ("Bullish", "Somewhat-Bullish") or score >= 0.15
                        if is_bull:
                            bullish.append({
                                "title": (art.get("title") or "")[:140],
                                "url": art.get("url"),
                                "published": tp.strftime("%Y-%m-%d") if tp else None,
                                "source": art.get("source") or "",
                                "sentiment": label or "—",
                                "score": round(score, 2),
                            })
                        if art.get("url"):
                            persist.append((
                                art.get("title"), art.get("url"), tp,
                                art.get("summary"), art.get("source"),
                                art.get("source_domain"),
                                json.dumps(art.get("topics") or []),
                                float(art.get("overall_sentiment_score") or 0) or None,
                                art.get("overall_sentiment_label"),
                                sym, rel, score, label,
                            ))
                    bullish.sort(key=lambda x: (x["published"] or "", x["score"]), reverse=True)
                    rec["news"] = bullish[:news_per_symbol]
                    if persist:
                        try:
                            async with pool.acquire() as conn:
                                await conn.executemany(
                                    """INSERT INTO us_news
                                       (title,url,time_published,summary,source,source_domain,
                                        topics,overall_sentiment_score,overall_sentiment_label,
                                        ticker,relevance_score_t,ticker_sentiment_score,
                                        ticker_sentiment_label,created_at)
                                       VALUES ($1,$2,$3,$4,$5,$6,$7::jsonb,$8,$9,$10,$11,$12,$13,now())
                                       ON CONFLICT (url, ticker) DO NOTHING""",
                                    persist)
                        except Exception:
                            pass
                except Exception:
                    pass

                # ----- Insider transactions (AV INSIDER_TRANSACTIONS) -----
                try:
                    async with session.get(base, params={
                        "function": "INSIDER_TRANSACTIONS", "symbol": sym,
                        "apikey": api_key,
                    }, timeout=aiohttp.ClientTimeout(total=20)) as r:
                        data = await r.json()
                    txs = data.get("data", []) if isinstance(data, dict) else []
                    cutoff = _dt.now().date() - _td_l(days=insider_lookback_days)
                    buys, dollar, execs = 0, 0.0, set()
                    ins_persist = []
                    for t in txs:
                        ds = (t.get("transaction_date") or "").strip()
                        try:    tdate = _dt.strptime(ds, "%Y-%m-%d").date()
                        except: continue
                        # 모든 거래(매수+매도)를 적재 — event_engine 이 두 방향
                        # 모두 활용 (인사이더 매도는 -, 매수는 + signal).
                        name = (t.get("executive") or "").strip()
                        title = (t.get("executive_title") or "").strip()
                        if not name:
                            continue
                        sec = (t.get("security_type") or "")[:255]
                        acq = (t.get("acquisition_or_disposal") or "")[:255]
                        try:    shares = float(t.get("shares") or 0) or None
                        except: shares = None
                        try:    price = float(t.get("share_price") or 0) or None
                        except: price = None
                        ins_persist.append((tdate, sym, [name], [title],
                                            sec, acq, shares, price))
                        # 최근 N일 매수 집계(reco 화면용)
                        if tdate >= cutoff and acq.upper() == "A":
                            buys += 1
                            if shares and price:
                                dollar += shares * price
                            execs.add(name)
                    if ins_persist:
                        try:
                            async with pool.acquire() as conn:
                                await conn.executemany(
                                    """INSERT INTO us_insider_transactions
                                       (date, symbol, executive, executive_title,
                                        security_type, acquisition_or_disposal,
                                        shares, share_price, created_at)
                                       VALUES ($1,$2,$3,$4,$5,$6,$7,$8,now())
                                       ON CONFLICT (date, symbol, executive, executive_title)
                                       DO UPDATE SET
                                         security_type = EXCLUDED.security_type,
                                         acquisition_or_disposal = EXCLUDED.acquisition_or_disposal,
                                         shares = EXCLUDED.shares,
                                         share_price = EXCLUDED.share_price""",
                                    ins_persist)
                        except Exception:
                            pass
                    if buys > 0:
                        rec["insider_buys"] = {
                            "count": buys,
                            "dollar_value": int(dollar),
                            "distinct_executives": len(execs),
                            "lookback_days": insider_lookback_days,
                        }
                except Exception:
                    pass

                result[sym] = rec
                await _asyncio.sleep(0.2)
    finally:
        await pool.close()
    return result


async def _symbols_due_for_report(days_threshold: int) -> list:
    """Publication-aware skip 용 — '발표일(reported_date) 기준 새 분기보고서가
    출시됐을 가능성이 있는' 종목 리스트.

    각 종목의 us_earnings_history MAX(reported_date) 가 days_threshold 일 이상
    지났거나 reportedDate 가 한 번도 없는(=신규/누락) 종목 = 새 10-Q/10-K 가
    이미 나왔거나 곧 나올 시점 → 재수집 대상. 그 외(최근에 보고한 종목)는 다음
    분기까지 새 statement 가 안 나오므로 skip 안전.

    수집 시각(updated_at)이 아닌 **발표일** 기반이라, "n일 전에 받아왔으면
    날짜와 상관없이 안 받음" 식의 잘못된 collection-time skip 이 아니다.
    """
    pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=2)
    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                WITH last_rep AS (
                    SELECT symbol, MAX(reported_date) AS last_rep
                    FROM us_earnings_history GROUP BY symbol
                ),
                active AS (
                    SELECT DISTINCT symbol FROM us_stock_basic WHERE is_active = true
                )
                SELECT a.symbol
                FROM active a LEFT JOIN last_rep r USING (symbol)
                WHERE r.last_rep IS NULL
                   OR r.last_rep < CURRENT_DATE - make_interval(days => $1)
                """,
                days_threshold,
            )
    finally:
        await pool.close()
    return [r["symbol"] for r in rows]


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
    # Publication-aware skip — 발표일(reported_date) 기준 due 종목만 재수집.
    # 마지막 분기보고서가 75일 이상 지났거나 미보고 = 새 10-Q/10-K 가 출시됐을
    # 가능성 → 그 종목만 fetch. 최근 분기보고를 한 종목은 다음 분기까지 새
    # statement 가 안 나오므로 skip 안전. (collection-time blanket skip 이 아님)
    days = int(ctx.params.get("fundamentals_due_days", 75))
    targets = await _symbols_due_for_report(days)
    if not targets:
        await ctx.log("ok", f"stock_basic skip: 발표일 기준 due 종목 0개 "
                            f"(모든 활성 종목이 최근 {days}일 내 보고)")
        return {"status": "skipped_no_due", "due_count": 0,
                "threshold_days": days}
    await ctx.log("info",
                  f"stock_basic: due {len(targets)}종목만 재수집 "
                  f"(전체 활성 대비)")
    api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    if not api_key:
        raise RuntimeError("ALPHAVANTAGE_API_KEY not set")
    from us.alphavantage import AlphaVantageCollector
    async with capture_logs(ctx), monitor_progress(
        ctx,
        "SELECT COUNT(*) FROM us_stock_basic",
        "us_stock_basic rows", interval=30):
        collector = AlphaVantageCollector(api_key, DATABASE_URL)
        await collector.collect_stock_data(symbols=targets)
    return {"status": "completed", "due_count": len(targets),
            "threshold_days": days}


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
            # DISTINCT 필수 — us_stock_basic 에는 'computed' source 가 종목별로
            # 수백 행(시점별)이라 DISTINCT 없으면 ~6,200종목이 2.86M 항목으로
            # 부풀어, 종목 분류 카운트가 폭증하고 collector 가 같은 종목을 수십~
            # 수백 번 처리해 us_daily 가 정상보다 수십 배 느려진다.
            active_rows = await conn.fetch(
                "SELECT DISTINCT symbol FROM us_stock_basic WHERE is_active = true")
            active_symbols = [r["symbol"] for r in active_rows]
    finally:
        await pool.close()

    # 분류 — needed_end 가 주말이면 직전 평일로 클립. AV 는 휴장일에 새
    # 데이터를 만들지 않으므로 토/일을 needed_end 로 쓰면 6,200종목이 전부
    # "forward (recent 누락)" 으로 분류돼 의미 없는 compact fetch 6,200번이
    # 돌아간다(같은 금요일 데이터만 반복 수신, 45분+ 낭비). 토→금, 일→금.
    # US 공휴일(예: Presidents Day)은 별도 calendar 가 없으면 못 잡지만,
    # 가장 빈번한 케이스(주말)는 이걸로 해결된다.
    needed_end_eff = ctx.end_date
    while needed_end_eff.weekday() >= 5:    # Sat=5, Sun=6
        needed_end_eff -= timedelta(days=1)
    needed_start, needed_end = extended_start, needed_end_eff
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


async def task_news_history_backfill(ctx: Ctx) -> dict:
    """11b. 백테스트 윈도우 동안 select_top_n 의 후보 종목 news 분기-단위 backfill.

    reco DAG 의 ``news_insider_top_n`` 과 동일 위치 (select_top_n → grades_pass_b)
    에 끼는 11b 노드. 단 reco 가 "당일" 의 top-3 만 받는 반면 이 노드는 backtest
    윈도우 전체 (start_date~end_date) 의 일자별 top-N union 종목들에 대해 분기
    단위로 historical news 를 받는다. AV NEWS_SENTIMENT 가 2018-01 부터 안정.

    Universe: select_top_n 와 동일 — us_stock_grade 의 (date in window,
              rank<=option_top_n by final_score, grade in STRONG_BUY/BUY/NEUTRAL)
              union. grades_pass_a 결과를 그대로 활용 → reco 와 일관.
    호출 단위: (symbol, quarter). limit 1000 (분기당 뉴스가 보통 <100건).
    Skip:    collection_state 'us_news_history' (symbol + quarter_start_date)
             success 마킹 → 다음 run 자동 skip. 0 건도 success (영구).

    Params:
      option_top_n        (int,   default 50):  종목 universe 의 일자별 top-N
                                                (select_top_n 와 동일 값 권장)
      news_call_interval  (float, default 0.5): API 페이스 (초/콜)
      news_max_concurrent (int,   default 3):   동시 호출 수

    비용 (option_top_n=50, 7.5년 = 30 분기):
      top-N union ≈ 200-500 unique symbols × 30 q ≈ 6k-15k calls / 1-3h
    """
    if ctx.country != "US":
        return {"skipped": "KR run"}
    api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    if not api_key:
        raise RuntimeError("ALPHAVANTAGE_API_KEY not set")

    import aiohttp, json as _json
    from datetime import datetime as _dt

    top_n           = int(ctx.params.get("option_top_n", 50))
    call_interval   = float(ctx.params.get("news_call_interval", 0.5))
    max_concurrent  = int(ctx.params.get("news_max_concurrent", 3))

    # AV NEWS_SENTIMENT 는 2018-01 부터 안정 → start_date 클립
    av_news_min_date = _dt(2018, 1, 1).date()
    win_start = max(ctx.start_date, av_news_min_date)
    win_end   = ctx.end_date

    # 분기 chunks 생성
    def _quarter_chunks(s, e):
        q_start = s.replace(month=((s.month - 1) // 3) * 3 + 1, day=1)
        chunks = []
        cur = q_start
        while cur <= e:
            ny, nm = (cur.year + 1, 1) if cur.month >= 10 else (cur.year, cur.month + 3)
            nxt = cur.replace(year=ny, month=nm, day=1)
            chunks.append((max(cur, s), min(nxt - timedelta(days=1), e)))
            cur = nxt
        return chunks

    chunks = _quarter_chunks(win_start, win_end)
    await ctx.log("info",
                  f"news_history: window {win_start}~{win_end}, "
                  f"{len(chunks)} quarter-chunks")

    # Universe = select_top_n union (grades_pass_a 결과 활용)
    pool = await asyncpg.create_pool(DATABASE_URL, min_size=2, max_size=6)
    try:
        async with pool.acquire() as conn:
            uni_rows = await conn.fetch("""
                SELECT DISTINCT symbol FROM (
                  SELECT date, symbol, final_score,
                         ROW_NUMBER() OVER (PARTITION BY date
                                            ORDER BY final_score DESC NULLS LAST) AS rn
                  FROM us_stock_grade
                  WHERE date BETWEEN $1 AND $2
                    AND final_grade IN ('STRONG_BUY','BUY','NEUTRAL','강력 매수','매수','중립','매수 고려')
                    AND final_score IS NOT NULL
                ) t WHERE rn <= $3
                ORDER BY symbol
            """, ctx.start_date, ctx.end_date, top_n)
            universe = [r["symbol"] for r in uni_rows]

            done_rows = await conn.fetch("""
                SELECT symbol, date FROM collection_state
                WHERE collection_name='us_news_history' AND status='success'
            """)
            done_set = {(r["symbol"], r["date"]) for r in done_rows}
    finally:
        await pool.close()

    await ctx.log("info",
                  f"news_history: universe={len(universe)} (top-{top_n}/date union), "
                  f"already-done chunks={len(done_set)}")

    # Plan tasks: 각 (symbol, chunk) 가 한 unit
    todo = []
    for sym in universe:
        for (cs, ce) in chunks:
            if (sym, cs) in done_set:
                continue
            todo.append((sym, cs, ce))
    if not todo:
        await ctx.log("ok", "news_history: all chunks already collected")
        return {"universe": len(universe), "chunks": len(chunks), "todo": 0,
                "status": "clean"}

    await ctx.log("info",
                  f"news_history: {len(todo)} (symbol, quarter) units to fetch "
                  f"(~{len(todo)*call_interval/60:.0f} minutes at {call_interval}s/call)")

    sem = asyncio.Semaphore(max_concurrent)
    base_url = "https://www.alphavantage.co/query"
    saved_total = 0
    failures = 0

    async def _fetch_chunk(session, symbol, cs, ce, max_retries=4):
        params = {
            "function": "NEWS_SENTIMENT", "tickers": symbol,
            "time_from": cs.strftime("%Y%m%dT0000"),
            "time_to":   ce.strftime("%Y%m%dT2359"),
            "limit": "1000",
            "sort": "RELEVANCE",
            "apikey": api_key,
        }
        for attempt in range(max_retries):
            try:
                async with session.get(base_url, params=params,
                                       timeout=aiohttp.ClientTimeout(total=30)) as r:
                    if r.status != 200:
                        return None, f"HTTP {r.status}"
                    data = await r.json()
                if "Error Message" in data:
                    return None, data["Error Message"][:80]
                if "Note" in data or "Information" in data:
                    msg = (data.get("Note") or data.get("Information") or "")[:80]
                    if attempt < max_retries - 1:
                        await asyncio.sleep(5 * (2 ** attempt))
                        continue
                    return None, f"rate-limited: {msg}"
                return data.get("feed", []), None
            except asyncio.TimeoutError:
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 * (2 ** attempt))
                    continue
                return None, "timeout retries exhausted"
            except Exception as e:
                return None, str(e)[:80]
        return None, "unknown"

    upsert_sql = """
        INSERT INTO us_news (
            title, url, time_published, authors, summary, banner_image,
            source, category_within_source, source_domain, topics,
            overall_sentiment_score, overall_sentiment_label,
            ticker, relevance_score_t, ticker_sentiment_score,
            ticker_sentiment_label, created_at
        ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,NOW())
        ON CONFLICT (url, ticker) DO NOTHING
    """

    def _parse_tp(s):
        if not s:
            return None
        try:
            return _dt.strptime(s, "%Y%m%dT%H%M%S")
        except Exception:
            try:
                return _dt.strptime(s, "%Y%m%dT%H%M")
            except Exception:
                return None

    pool = await asyncpg.create_pool(DATABASE_URL, min_size=2, max_size=8)

    async def _process(session, symbol, cs, ce):
        nonlocal saved_total, failures
        async with sem:
            feed, err = await _fetch_chunk(session, symbol, cs, ce)
            await asyncio.sleep(call_interval)
            if feed is None:
                failures += 1
                return
            # 같은 url+ticker 가 한 feed 안에 중복일 수 있어 dedup
            seen = set()
            rows = []
            for item in feed:
                url = item.get("url")
                if not url:
                    continue
                tp = _parse_tp(item.get("time_published"))
                if tp is None:
                    continue
                # ticker_sentiment 안에서 우리 symbol 의 점수/관련도 추출
                ts_score, ts_label, rel = None, None, None
                for ts in item.get("ticker_sentiment", []) or []:
                    if ts.get("ticker") == symbol:
                        try: ts_score = float(ts.get("ticker_sentiment_score"))
                        except Exception: pass
                        ts_label = ts.get("ticker_sentiment_label")
                        try: rel = float(ts.get("relevance_score"))
                        except Exception: pass
                        break
                key = (url, symbol)
                if key in seen:
                    continue
                seen.add(key)
                try: o_score = float(item.get("overall_sentiment_score"))
                except Exception: o_score = None
                rows.append((
                    item.get("title"), url, tp,
                    ", ".join(item.get("authors", []) or [])[:1000],
                    item.get("summary"), item.get("banner_image"),
                    item.get("source"), item.get("category_within_source"),
                    item.get("source_domain"),
                    _json.dumps(item.get("topics", [])),
                    o_score, item.get("overall_sentiment_label"),
                    symbol, rel, ts_score, ts_label,
                ))
            inserted = 0
            try:
                async with pool.acquire() as conn:
                    async with conn.transaction():
                        if rows:
                            await conn.executemany(upsert_sql, rows)
                            inserted = len(rows)
                        # success 마킹 — 0건이라도 (그 분기에 그 종목 뉴스 없음)
                        await conn.execute("""
                            INSERT INTO collection_state
                              (collection_name, symbol, date, status)
                            VALUES ('us_news_history', $1, $2, 'success')
                            ON CONFLICT (collection_name, symbol, date) DO UPDATE
                              SET status='success', collected_at=NOW()
                        """, symbol, cs)
            except Exception as e:
                failures += 1
                await ctx.log("warn", f"news_history {symbol} {cs}: db err {str(e)[:80]}")
                return
            saved_total += inserted

    try:
        async with capture_logs(ctx), monitor_progress(
            ctx,
            "SELECT COUNT(*) FROM us_news",
            "us_news rows", interval=120):
            async with aiohttp.ClientSession(headers={"User-Agent": "Mozilla/5.0"}) as session:
                # ctx.log 진행 상황: 1000 units 마다
                tasks = []
                for i, (sym, cs, ce) in enumerate(todo):
                    tasks.append(asyncio.create_task(_process(session, sym, cs, ce)))
                    if (i + 1) % 500 == 0:
                        # backpressure: 500 in-flight 이상 쌓이지 않도록
                        done, pending = await asyncio.wait(
                            tasks, return_when=asyncio.ALL_COMPLETED, timeout=None)
                        tasks = list(pending)
                        await ctx.log("info",
                                      f"news_history progress: {i+1}/{len(todo)} units, "
                                      f"saved={saved_total}, fail={failures}")
                if tasks:
                    await asyncio.wait(tasks, return_when=asyncio.ALL_COMPLETED)
    finally:
        await pool.close()

    await ctx.log("ok",
                  f"news_history done: {len(todo)} units, saved={saved_total} rows, "
                  f"failures={failures}")
    return {"universe": len(universe), "chunks_per_symbol": len(chunks),
            "units_attempted": len(todo), "rows_saved": saved_total,
            "failures": failures, "status": "completed"}


async def task_listing_status(ctx: Ctx) -> dict:
    """AV LISTING_STATUS → us_listing_status (active + delisted).

    백테스트 시점별 PIT universe 구성용. 1회 호출에 13k active + 9k delisted
    (1997~) 가 CSV 로 옴. cheap (2 calls).
    """
    if ctx.country != "US":
        return {"skipped": "KR run"}
    api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    if not api_key:
        raise RuntimeError("ALPHAVANTAGE_API_KEY not set")
    from us.finance_data import ListingStatusCollector
    async with capture_logs(ctx):
        col = ListingStatusCollector(api_key, DATABASE_URL)
        result = await col.run_collection()
    await ctx.log("ok",
                  f"listing_status: active={result.get('active_rows')}, "
                  f"delisted={result.get('delisted_rows')}")
    return result


async def task_financials_verify(ctx: Ctx) -> dict:
    """task_financials 의 belt-and-suspenders 검증 + 누락 종목 재시도.

    이전 silent-drop 버그 (AV "Note"/"Information" rate-limit 응답을 그냥 drop
    했던 것) 가 finance_data.py 의 retry-with-backoff 로 고쳐졌지만, 대량
    parallel 호출 + burst-limit 조합에선 일부 종목이 그래도 누락될 수 있음.

    이 노드는 task_financials 직후에 돌면서:
      1) 12 개월 내 데이터 없는 active 종목 (≡ '누락') 식별
      2) collection_state 에 no_data 로 마킹된 종목은 제외 (AV 자체에 없음)
      3) 누락 종목만 좁은 universe 로 collector 재실행 — call_interval 0.7s
         (≈85 calls/min) 의 conservative pace, 3 collector 직렬 (parallel X)
         → burst-limit 회피
      4) 결과는 warning 로그만 남기고 raise 안 함 — 다음 run 이 자연 backfill

    사용자가 "다음 백테스트 때 자연 backfill" 하길 원했으므로 이 노드 자체가
    그 메커니즘. backtest DAG (kind=full or reco) 가 grades_pass_a 전에 이걸
    돌리도록 dependency 가 들어가 있음.
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

    # 누락 식별: active=true 인데 12 개월 내 fiscal row 가 없고, no_data 로
    # 마킹되지도 않은 종목들. (active=false 면 자연 폐기, no_data 면 AV 가
    # 실제로 안 주는 것이라 재시도 무의미.)
    GAP_SQL = """
        WITH active AS (SELECT DISTINCT symbol FROM us_stock_basic WHERE is_active=true),
        present AS (
          SELECT DISTINCT symbol FROM {tbl}
          WHERE fiscal_date_ending >= CURRENT_DATE - INTERVAL '12 months'
        ),
        nd AS (
          SELECT DISTINCT symbol FROM collection_state
          WHERE collection_name = $1 AND status='no_data'
        )
        SELECT a.symbol
        FROM active a
        LEFT JOIN present p USING(symbol)
        LEFT JOIN nd USING(symbol)
        WHERE p.symbol IS NULL AND nd.symbol IS NULL
        ORDER BY a.symbol
    """

    pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=3)
    try:
        async with pool.acquire() as conn:
            inc_gap = [r["symbol"] for r in await conn.fetch(
                GAP_SQL.format(tbl="us_income_statement"), "us_income_statement")]
            bal_gap = [r["symbol"] for r in await conn.fetch(
                GAP_SQL.format(tbl="us_balance_sheet"), "us_balance_sheet")]
            cf_gap  = [r["symbol"] for r in await conn.fetch(
                GAP_SQL.format(tbl="us_cash_flow"), "us_cash_flow")]
    finally:
        await pool.close()

    await ctx.log("info",
                  f"financials_verify gaps — income:{len(inc_gap)}, "
                  f"balance:{len(bal_gap)}, cashflow:{len(cf_gap)}")

    # 누락 0 → 깔끔, 그대로 종료
    if not inc_gap and not bal_gap and not cf_gap:
        return {"income_gap": 0, "balance_gap": 0, "cashflow_gap": 0,
                "status": "clean"}

    # 좁은 deadline (2 시간) — 누락이 적어 그 안에 다 들어와야 함
    deadline = datetime.now() + timedelta(hours=2)
    call_interval = 0.7   # ~85 calls/min, conservative

    # 핵심 트릭: collector 가 skip_symbols 가 set 일 때 그것만 제외하므로,
    # "처리해야 할 종목" 을 받는 메커니즘이 없다. 대신 'all active' 에서
    # 'gap' 만 남기려면 skip = active - gap 으로 invert.
    async def _run_one(CollectorCls, gap_list: list, label: str) -> dict:
        if not gap_list:
            return {"skipped": "no gap"}
        async with capture_logs(ctx):
            col = CollectorCls(api_key, DATABASE_URL,
                               call_interval=call_interval, deadline=deadline)
            # 직접 universe 주입 — get_existing_symbols 우회
            col.skip_symbols = None
            col._gap_only = set(gap_list)
            # monkey-patch: get_existing_symbols 가 gap 만 반환하도록
            async def _gap_universe():
                return list(gap_list)
            col.get_existing_symbols = _gap_universe
            await col.run_collection_optimized()
        return {"attempted": len(gap_list)}

    # 직렬 실행 (병렬 X) — burst-limit 회피
    inc_result = await _run_one(IncomeStatementCollector, inc_gap, "INCOME")
    bal_result = await _run_one(BalanceSheetCollector,    bal_gap, "BALANCE")
    cf_result  = await _run_one(CashFlowCollector,        cf_gap,  "CASHFLOW")

    # 사후 점검
    pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=2)
    try:
        async with pool.acquire() as conn:
            inc_after = [r["symbol"] for r in await conn.fetch(
                GAP_SQL.format(tbl="us_income_statement"), "us_income_statement")]
            bal_after = [r["symbol"] for r in await conn.fetch(
                GAP_SQL.format(tbl="us_balance_sheet"), "us_balance_sheet")]
            cf_after  = [r["symbol"] for r in await conn.fetch(
                GAP_SQL.format(tbl="us_cash_flow"), "us_cash_flow")]
    finally:
        await pool.close()

    await ctx.log("ok",
                  f"financials_verify after-pass gaps — "
                  f"income:{len(inc_after)} (was {len(inc_gap)}), "
                  f"balance:{len(bal_after)} (was {len(bal_gap)}), "
                  f"cashflow:{len(cf_after)} (was {len(cf_gap)}). "
                  f"잔여 누락은 다음 run 이 자연 backfill.")

    # available_at fallback 비율 로깅 — 백테스트 leak 위험 정량화
    # +45d fallback 이 많을수록 look-ahead 위험. 정상 reported_date 비율을
    # 분기 연도별로 집계해 보고.
    pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=2)
    try:
        async with pool.acquire() as conn:
            fallback_dist = await conn.fetch("""
                SELECT EXTRACT(YEAR FROM fiscal_date_ending)::int AS year,
                       COUNT(*) AS rows,
                       COUNT(*) FILTER (
                         WHERE available_at = (fiscal_date_ending + INTERVAL '45 days')::date
                       ) AS fallback_rows,
                       ROUND(100.0 * COUNT(*) FILTER (
                         WHERE available_at = (fiscal_date_ending + INTERVAL '45 days')::date
                       ) / NULLIF(COUNT(*),0), 1) AS pct_fallback
                FROM us_income_statement
                GROUP BY 1 ORDER BY 1 DESC LIMIT 15
            """)
    finally:
        await pool.close()

    fb_summary = [
        {"year": int(r["year"]), "rows": int(r["rows"]),
         "fallback_rows": int(r["fallback_rows"]),
         "pct_fallback": float(r["pct_fallback"]) if r["pct_fallback"] is not None else None}
        for r in fallback_dist
    ]
    high_fb = [y for y in fb_summary if y["pct_fallback"] and y["pct_fallback"] > 30]
    if high_fb:
        parts = [f"{y['year']}({y['pct_fallback']}%)" for y in high_fb[:6]]
        await ctx.log("warn",
                      "+45d fallback >30% (look-ahead 위험) 분기-연도: "
                      + ", ".join(parts)
                      + ". earnings_history backfill 로 reported_date 보강 필요.")
    else:
        await ctx.log("ok", "+45d fallback 비율 모든 연도 <30% — look-ahead 위험 낮음.")

    return {
        "income_gap_before": len(inc_gap),  "income_gap_after": len(inc_after),
        "balance_gap_before": len(bal_gap), "balance_gap_after": len(bal_after),
        "cashflow_gap_before": len(cf_gap), "cashflow_gap_after": len(cf_after),
        "fallback_by_year": fb_summary,
        "status": "completed",
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

    # ctx.start_date 가 collector 의 start_date 보다 더 과거이면 weekly
    # 데이터 누락 → us_calculator 가 weekly 지표 계산 못 함. extra buffer
    # 60일 (≈ EM8 lookback margin) 으로 더 일찍 잡음.
    weekly_start = ctx.start_date - timedelta(days=60)
    async with capture_logs(ctx), monitor_progress(
        ctx,
        "SELECT COUNT(*) FROM us_weekly",
        "us_weekly rows", interval=30):
        col = WeeklyCollector(api_key, DATABASE_URL, max_concurrent=20,
                              start_date=weekly_start)
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
    """FED Funds / Treasury / CPI / Unemployment 매크로 backfill.

    ctx.start_date 를 collector 의 start_date 로 주입 — 이전엔 base 가
    date(2025,1,1) 하드코드 + endpoint 가 그 default 만 사용해서 2019-2024
    macro 전부 결손이었음 (검증: us_fed_funds_rate / us_cpi / us_unemployment
    모두 16 rows / 2025-01-01 부터만). 이제 ctx.start_date - 30d 부터 적재.
    Endpoint 대신 collector 직접 호출 — start_date 주입 깔끔.
    """
    if ctx.country != "US":
        return {"skipped": "KR run"}
    api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    if not api_key:
        raise RuntimeError("ALPHAVANTAGE_API_KEY not set")
    from us.finance_data import (FederalFundsRateCollector, TreasuryYieldCollector,
                                  CPICollector, UnemploymentRateCollector)
    macro_start = ctx.start_date - timedelta(days=30)
    collectors = [
        ("us_fed_funds_rate",    FederalFundsRateCollector),
        ("us_treasury_yield",    TreasuryYieldCollector),
        ("us_cpi",               CPICollector),
        ("us_unemployment_rate", UnemploymentRateCollector),
    ]
    out = {}
    for name, Cls in collectors:
        try:
            async with capture_logs(ctx):
                col = Cls(api_key, DATABASE_URL, call_interval=0.2,
                          start_date=macro_start)
                await col.run_collection_optimized()
            out[name] = "ok"
        except Exception as e:
            await ctx.log("warn", f"{name} failed ({e}) — continuing")
            out[name] = {"error": str(e)[:200]}
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


# ---------------------------------------------- News + Insider 사전수집 (reco 전용)
async def task_news_insider_top_n(ctx: Ctx) -> dict:
    """Reco DAG 전용 — top-N 후보 종목들의 news / insider 데이터를 미리 적재.

    event_engine 이 grade 계산 시 us_news (7일 lookback) 와 us_insider_transactions
    (90일 lookback) 를 읽어 news_modifier / insider_modifier 를 산출하는데,
    테이블이 비어있으면 두 modifier 가 0 으로 무력화돼 호재가 grade 점수에
    반영되지 않는다. 이 task 가 select_top_n 직후, grades_pass_b 직전에
    돌아 두 테이블을 채워준다.

    full 백테스트 DAG 에는 들어가지 않음 (1년치 backfill 비용이 크고, 역사
    뉴스는 AV 가 회수 안 줌). reco DAG (kind='reco') 한정.
    """
    if ctx.country != "US":
        return {"skipped": "US only"}
    top_n = int(ctx.params.get("option_top_n", 50))
    pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=3)
    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT DISTINCT symbol FROM (
                  SELECT date, symbol, final_score,
                         ROW_NUMBER() OVER (PARTITION BY date
                                            ORDER BY final_score DESC NULLS LAST) AS rn
                  FROM us_stock_grade
                  WHERE date BETWEEN $1 AND $2
                    AND final_grade IN ('STRONG_BUY','BUY','NEUTRAL',
                                        '강력 매수','매수','매수 고려','중립')
                    AND final_score IS NOT NULL
                ) t WHERE rn <= $3
                """,
                ctx.start_date, ctx.end_date, top_n,
            )
            symbols = [r["symbol"] for r in rows]
    finally:
        await pool.close()

    if not symbols:
        return {"status": "no_top_n_symbols"}

    api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    if not api_key:
        raise RuntimeError("ALPHAVANTAGE_API_KEY not set")

    await ctx.log("info",
                  f"news/insider 사전수집: {len(symbols)}종목 × 2 endpoint")
    result = await _fetch_catalysts_for_symbols(
        api_key, symbols,
        news_per_symbol=10,
        insider_lookback_days=90,
    )
    bullish_news = sum(len(v.get("news") or []) for v in result.values())
    insider_active = sum(1 for v in result.values() if v.get("insider_buys"))
    await ctx.log("ok",
                  f"news/insider 적재 완료 → bullish news {bullish_news}건, "
                  f"인사이더 매수 활성 종목 {insider_active}/{len(symbols)}")
    return {
        "status": "completed",
        "symbols_processed": len(symbols),
        "bullish_news_total": bullish_news,
        "insider_active_symbols": insider_active,
    }


# ---------------------------------------------- 최고 signal 종목 추천 (DAG node)
async def task_top_signal_reco(ctx: Ctx) -> dict:
    """현재 시점 '최고 signal' 종목 추천.

    백테스트에서 가장 견고하게 우수했던 전략 — top-3 / STRONG_BUY(강력 매수) /
    20거래일 리밸 (총수익 +166.1%, Sharpe 1.87, MDD -25.3%, Calmar 6.56) —
    의 종목 선택 규칙을 그대로 적용한다: 최신 grade 일자에서 final_grade='강력
    매수'인 종목을 final_score 내림차순으로 정렬해 상위를 추천.

    output:
      - buy_now    : 실제 매수 대상(reco_top_n, 기본 3) — 전략이 보유하는 종목
      - watchlist  : 차순위 STRONG_BUY 후보(reco_watch_n, 기본 7)
    dashboard 의 task output(JSON)이 곧 추천 '페이지' 역할을 한다.
    """
    if ctx.country != "US":
        return {"skipped": "US only"}

    reco_n = int(ctx.params.get("reco_top_n", 3))
    watch_n = int(ctx.params.get("reco_watch_n", 7))

    pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=2)
    try:
        async with pool.acquire() as conn:
            as_of = await conn.fetchval("SELECT MAX(date) FROM us_stock_grade")
            if as_of is None:
                return {"error": "no grades in us_stock_grade"}
            rows = await conn.fetch(
                """
                SELECT g.symbol, g.final_score, g.final_grade, d.close
                FROM us_stock_grade g
                LEFT JOIN us_daily d ON d.symbol = g.symbol AND d.date = g.date
                WHERE g.date = $1
                  AND g.final_grade = '강력 매수'
                  AND g.final_score IS NOT NULL
                ORDER BY g.final_score DESC
                LIMIT $2
                """,
                as_of, reco_n + watch_n,
            )
            sb_count = await conn.fetchval(
                "SELECT COUNT(*) FROM us_stock_grade "
                "WHERE date = $1 AND final_grade = '강력 매수'",
                as_of,
            )
    finally:
        await pool.close()

    def _fmt(rank, r):
        return {
            "rank": rank,
            "symbol": r["symbol"],
            "final_score": round(float(r["final_score"]), 2),
            "final_grade": r["final_grade"],
            "close": round(float(r["close"]), 2) if r["close"] is not None else None,
        }

    picks = [_fmt(i + 1, r) for i, r in enumerate(rows)]

    # 실거래 일정. 백테스트 simulator 는 등급일(as_of) 종가에 체결하지만
    # 실제로는 종가 확정 후에야 등급을 알 수 있으므로 live 권장 매수는 다음
    # 거래일 시가. 리밸런싱 주기는 최우수 전략(top-3 STRONG_BUY / 20 거래일).
    def _next_bday(d, n=1):
        from datetime import timedelta as _td_local
        while n > 0:
            d = d + _td_local(days=1)
            if d.weekday() < 5:   # Mon-Fri
                n -= 1
        return d

    rebal_n = int(ctx.params.get("strategy_rebal_freq_days", 20))
    buy_date = _next_bday(as_of, 1)
    next_rebal = _next_bday(buy_date, rebal_n)

    # 호재 — buy_now 종목만 (top-3 × 2 API콜 = 6 호출, ~3초). best-effort.
    catalysts = {}
    buy_now_symbols = [p["symbol"] for p in picks[:reco_n]]
    av_key = os.getenv("ALPHAVANTAGE_API_KEY")
    if av_key and buy_now_symbols:
        try:
            catalysts = await _fetch_catalysts_for_symbols(av_key, buy_now_symbols)
            await ctx.log("info", "catalysts: " + ", ".join(
                f"{s}:📰{len(v.get('news') or [])}"
                + (f"/💰{v['insider_buys']['count']}건"
                   if v.get('insider_buys') else "")
                for s, v in catalysts.items()))
        except Exception as e:
            await ctx.log("warn", f"catalyst fetch failed (ignored): {e}")

    out = {
        "as_of_date": str(as_of),
        "strategy": ("top-3 / STRONG_BUY(강력 매수) / 20거래일 리밸 — "
                     "backtest +166.1% (Sharpe 1.87, MDD -25.3%, Calmar 6.56)"),
        "buy_now": picks[:reco_n],
        "watchlist": picks[reco_n:reco_n + watch_n],
        "strong_buy_universe_size": sb_count,
        "execution": {
            "buy_date": str(buy_date),
            "buy_timing": "open",
            "buy_note": f"실거래: {buy_date} 시가(OPEN)에 매수 권장. "
                        f"백테스트는 {as_of} 종가(CLOSE) 기준 — 종가 확정 후에야 "
                        f"등급을 알 수 있어 live 는 다음 거래일 OPEN 이 현실적.",
            "rebal_freq_days": rebal_n,
            "next_rebal_date": str(next_rebal),
            "next_rebal_note": f"{buy_date} 이후 {rebal_n} 거래일째인 "
                               f"{next_rebal} 에 reco DAG 재실행 → 그날 신규 "
                               f"top-3 로 교체(시가 매수/매도).",
        },
        "catalysts": catalysts,
    }
    await ctx.log("ok", f"[reco] {as_of} buy_now="
                  f"{[p['symbol'] for p in out['buy_now']]} "
                  f"buy={buy_date} next_rebal={next_rebal}")
    return out


async def task_earnings_history(ctx: Ctx) -> dict:
    """AV EARNINGS endpoint 호출 → us_earnings_history 적재.

    각 종목당 1 call → quarterlyEarnings 의 reportedDate (실제 공시일자) 캡쳐.
    financials 3종의 available_at 정밀화에 사용 (income/balance/cashflow 는
    모두 같은 10-Q/10-K 에 묶여 같은 reported_date 공유).

    rate-limit retry decorator (15s 지수 backoff, max 5분 cap) 로 안전.
    """
    if ctx.country != "US":
        return {"skipped": "KR run"}
    # Publication-aware: 새 분기보고서가 출시됐을 가능성이 있는 종목만 재수집
    # (stock_basic 과 동일 로직). reported_date 가 75일+ 지났거나 없는 종목만.
    days = int(ctx.params.get("fundamentals_due_days", 75))
    targets = await _symbols_due_for_report(days)
    if not targets:
        await ctx.log("ok", f"earnings_history skip: 발표일 기준 due 종목 0개")
        return {"status": "skipped_no_due", "due_count": 0,
                "threshold_days": days}
    await ctx.log("info",
                  f"earnings_history: due {len(targets)}종목만 재수집")

    api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    if not api_key:
        raise RuntimeError("ALPHAVANTAGE_API_KEY not set")

    from us.alphavantage import EarningsHistoryCollector
    async with capture_logs(ctx), monitor_progress(
        ctx,
        "SELECT COUNT(*) FROM us_earnings_history",
        "us_earnings_history rows", interval=30):
        col = EarningsHistoryCollector(api_key, DATABASE_URL,
                                       max_concurrent=3,
                                       target_symbols=targets)
        result = await col.run_collection()
    return result


async def _compute_stock_basic_window(
    pool, load_start, load_end, output_start, output_end,
    *, api_df=None,
) -> "tuple[int, int]":
    """Pure compute helper — 하나의 [load_start, load_end] 윈도우 처리.

    INSERT 대상은 [output_start, output_end] 만 (lookback row 는 drop).
    청크 외부 루프 / 단일 처리 둘 다 같은 함수 사용 → 결과 동등성 보장.

    Returns (staged, total_records_in_chunk).

    lookback 요구:
      - daily: output_start - 400 days  (252-day rolling 안전)
      - financials: load_start - 5 years (TTM 4Q + YoY 4Q-ago 안전)
      - api_df: 시점 무관 메타 (한 번 로드 후 청크 간 공유 가능)
    """
    import pandas as pd
    import numpy as np
    import gc

    async with pool.acquire() as conn:
        daily_rows = await conn.fetch(
            "SELECT symbol, date, close FROM us_daily "
            "WHERE date BETWEEN $1 AND $2", load_start, load_end)
        inc_rows = await conn.fetch(
            """SELECT symbol, fiscal_date_ending, available_at,
                      net_income, total_revenue, operating_income,
                      gross_profit, ebitda
               FROM us_income_statement
               WHERE fiscal_date_ending >= $1::date - INTERVAL '5 years'
                 AND fiscal_date_ending <= $2::date""",
            load_start, load_end)
        bal_rows = await conn.fetch(
            """SELECT symbol, fiscal_date_ending, available_at,
                      total_assets, total_shareholder_equity,
                      common_stock_shares_outstanding,
                      short_long_term_debt_total,
                      cash_and_short_term_investments
               FROM us_balance_sheet
               WHERE fiscal_date_ending >= $1::date - INTERVAL '5 years'
                 AND fiscal_date_ending <= $2::date""",
            load_start, load_end)
        if api_df is None:
            api_rows = await conn.fetch(
                """SELECT DISTINCT ON (symbol) symbol, stock_name, exchange,
                          currency, sector, industry, beta, is_active,
                          sharesoutstanding AS api_shares
                   FROM us_stock_basic WHERE source='api'
                   ORDER BY symbol, date DESC""")
            api_df = pd.DataFrame(api_rows,
                columns=['symbol', 'stock_name', 'exchange', 'currency',
                         'sector', 'industry', 'beta', 'is_active', 'api_shares'])

    logger.info(f"[compute_window {output_start}~{output_end}] "
                f"loaded daily={len(daily_rows)} inc={len(inc_rows)} bal={len(bal_rows)}")

    if not daily_rows:
        return 0, 0

    daily_df = pd.DataFrame(daily_rows, columns=['symbol', 'date', 'close'])
    daily_df['date'] = pd.to_datetime(daily_df['date'])
    daily_df['close'] = daily_df['close'].astype('float64')

    # ---------- financials → TTM + YoY ----------
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
        del inc_df, bal_df
    else:
        fin_df = pd.DataFrame(columns=['symbol', 'fiscal_date_ending'])

    if not fin_df.empty:
        fin_df['fiscal_date_ending'] = pd.to_datetime(fin_df['fiscal_date_ending'])
        fin_df = fin_df.sort_values(['symbol', 'fiscal_date_ending']).reset_index(drop=True)

        num_cols = ['net_income', 'total_revenue', 'operating_income',
                    'gross_profit', 'ebitda',
                    'total_assets', 'total_shareholder_equity',
                    'common_stock_shares_outstanding',
                    'short_long_term_debt_total',
                    'cash_and_short_term_investments']
        for c in num_cols:
            fin_df[c] = pd.to_numeric(fin_df[c], errors='coerce')

        grp = fin_df.groupby('symbol', sort=False)
        for c in ['net_income', 'total_revenue', 'operating_income',
                  'gross_profit', 'ebitda']:
            fin_df[f'{c}_ttm'] = grp[c].transform(
                lambda s: s.rolling(4, min_periods=1).sum())
        for c in ['total_assets', 'total_shareholder_equity']:
            fin_df[f'{c}_avg'] = grp[c].transform(
                lambda s: s.rolling(2, min_periods=1).mean())
        fin_df['net_income_4q_ago'] = grp['net_income'].shift(4)
        fin_df['revenue_4q_ago'] = grp['total_revenue'].shift(4)

        fin_df['available_at'] = pd.to_datetime(
            fin_df.get('available_at_inc')).fillna(
            pd.to_datetime(fin_df.get('available_at_bal'))).fillna(
            fin_df['fiscal_date_ending'] + pd.Timedelta(days=45))
        fin_df = fin_df.dropna(subset=['available_at'])
        fin_df = fin_df.sort_values('available_at', kind='mergesort').reset_index(drop=True)

        daily_df_for_merge = daily_df.sort_values('date', kind='mergesort').reset_index(drop=True)
        merged = pd.merge_asof(
            daily_df_for_merge, fin_df,
            left_on='date', right_on='available_at',
            by='symbol', direction='backward')
        del fin_df, daily_df_for_merge
    else:
        merged = daily_df.copy()

    # ---------- rolling (52w high/low, 50/200 MA) ----------
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
    del daily_df, daily_df_sorted

    merged = merged.merge(api_df, on='symbol', how='left')

    # ---------- 17 컬럼 벡터 계산 ----------
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
    ni_ttm = merged.get('net_income_ttm', pd.Series([np.nan] * len(merged)))
    merged['eps'] = _safe_div(ni_ttm, so)
    merged['dilutedepsttm'] = merged['eps']
    merged['per'] = _safe_div(close, merged['eps'])
    equity = merged.get('total_shareholder_equity',
                        pd.Series([np.nan] * len(merged)))
    merged['bookvalue'] = _safe_div(equity, so)
    merged['pricetobookratio'] = _safe_div(close, merged['bookvalue'])
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
    debt = merged.get('short_long_term_debt_total',
                      pd.Series([np.nan] * len(merged))).fillna(0)
    cash = merged.get('cash_and_short_term_investments',
                      pd.Series([np.nan] * len(merged))).fillna(0)
    ev = merged['market_cap'].fillna(0) + debt - cash
    merged['evtorevenue'] = _safe_div(ev, rev_ttm)
    merged['evtoebitda'] = _safe_div(ev,
        merged.get('ebitda_ttm', pd.Series([np.nan] * len(merged))))
    ni = merged.get('net_income', pd.Series([np.nan] * len(merged)))
    ni_yoy_base = merged.get('net_income_4q_ago',
                             pd.Series([np.nan] * len(merged)))
    rev = merged.get('total_revenue', pd.Series([np.nan] * len(merged)))
    rev_yoy_base = merged.get('revenue_4q_ago',
                              pd.Series([np.nan] * len(merged)))
    merged['quarterlyearningsgrowthyoy'] = _safe_div(ni, ni_yoy_base.abs()) - 1
    merged['quarterlyrevenuegrowthyoy'] = _safe_div(rev, rev_yoy_base.abs()) - 1

    merged['market_cap'] = merged['market_cap'].astype('Float64').round().astype('Int64')

    # ---------- 출력 윈도우 필터 (lookback row drop) ----------
    out_start_ts = pd.Timestamp(output_start)
    out_end_ts = pd.Timestamp(output_end)
    merged = merged[(merged['date'] >= out_start_ts) & (merged['date'] <= out_end_ts)]
    merged = merged.reset_index(drop=True)
    logger.info(f"[compute_window {output_start}~{output_end}] "
                f"after output filter: {len(merged)} rows")

    if merged.empty:
        return 0, 0

    # ---------- INSERT ----------
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

    out = out.astype(object).where(pd.notna(out), None)

    def _to_bigint(v):
        if v is None or (isinstance(v, float) and (v != v)):
            return None
        try:
            return int(v)
        except (TypeError, ValueError):
            return None
    for _bc in ('market_cap', 'revenuettm', 'grossprofitttm', 'sharesoutstanding'):
        out[_bc] = pd.Series(
            [_to_bigint(v) for v in out[_bc]],
            index=out.index, dtype=object)

    records = list(out[insert_cols].itertuples(index=False, name=None))
    total_records = len(records)
    del merged, out
    gc.collect()

    CHUNK = 50000
    inserted = 0
    async with pool.acquire() as conn:
        async with conn.transaction():
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
    del records
    gc.collect()
    return inserted, total_records


async def task_us_stock_basic_compute(ctx: Ctx) -> dict:
    """us_stock_basic source='computed' row 적재 — 시점별 정밀 재계산.

    pandas 벡터화 + as-of merge 로 look-ahead-bias 안전:
      - us_daily 각 (symbol, date) 마다 fiscal_date_ending + 45 일 reporting lag
        시점 까지의 us_income_statement / us_balance_sheet 만 사용
      - 분기별 trailing-4Q rolling 으로 TTM (revenue/net_income/ebitda/op/gross)
      - YoY = 현재 분기 / 4분기 전 분기

    채우는 컬럼 (17 시점별 + 메타):
      market_cap, per, pricetobookratio, eps, dilutedepsttm, bookvalue,
      sharesoutstanding, revenuettm, grossprofitttm, profitmargin,
      operatingmarginttm, returnonequityttm, returnonassetsttm,
      pricetosalesratiottm, evtorevenue, evtoebitda,
      quarterlyearningsgrowthyoy, quarterlyrevenuegrowthyoy,
      week52high, week52low, day50movingaverage, day200movingaverage,
      + 메타 (stock_name, exchange, currency, sector, industry, beta, is_active)

    Memory-safe 청크 처리:
      [ctx.start_date - LOOKBACK_BUFFER ~ ctx.end_date] 를 1년씩 분할.
      각 청크: load = [chunk_out_start - 400d, chunk_out_end], INSERT 는 [chunk_out_start,
      chunk_out_end] 만. lookback 400 days 면 252-day rolling 안전, financials
      는 load_start - 5 years 까지 가져와 TTM/YoY 안전. 청크 사이 gc.collect().
      → 7.5 년 단일 처리 시 9M 行 pandas → OOM 이었던 게 청크당 ~1M 行으로 안전.

    bulk INSERT 전략: pandas → asyncpg copy_records_to_table 로 임시 테이블 적재
    → 본 테이블 INSERT … ON CONFLICT (symbol, date, source) DO UPDATE.
    """
    if ctx.country != "US":
        return {"skipped": "KR run"}

    import pandas as pd
    from datetime import date as _date

    LOOKBACK_BUFFER_DAYS = DAILY_LOOKBACK_CALENDAR_DAYS
    chunk_days = int(ctx.params.get("stock_basic_compute_chunk_days", 365))
    # 800 days = 단일 vs 청크 동등성 보장. 실증: lookback 420d 시 3 rows
    # 차이 (ZWZZT-class sparse test ticker), 600d 시 14 rows (HAFN-class
    # sparse + 최근 IPO), 800d 시 107,680 rows 전체 동일 (PASS). 200거래일
    # rolling 의 worst-case sparse 종목 + 신규 IPO 직후 buffer 모두 cover.
    daily_lookback = max(800, LOOKBACK_BUFFER_DAYS)

    # 출력 윈도우 = [ctx.start_date - LOOKBACK_BUFFER, ctx.end_date]
    overall_out_start = ctx.start_date - timedelta(days=LOOKBACK_BUFFER_DAYS)
    overall_out_end = ctx.end_date

    # 청크 분할
    chunks = []
    cs = overall_out_start
    while cs <= overall_out_end:
        ce = min(cs + timedelta(days=chunk_days - 1), overall_out_end)
        chunks.append((cs, ce))
        cs = ce + timedelta(days=1)

    await ctx.log("info",
                  f"stock_basic_compute: window {overall_out_start}~{overall_out_end} "
                  f"→ {len(chunks)} chunks ({chunk_days}d each, daily lookback {daily_lookback}d)")

    async with capture_logs(ctx), monitor_progress(
        ctx,
        "SELECT COUNT(*) FROM us_stock_basic WHERE source='computed'",
        "us_stock_basic computed rows", interval=60):

        pool = await asyncpg.create_pool(DATABASE_URL, min_size=2, max_size=5)
        try:
            # api 메타는 한 번만 로드 → 청크 간 공유
            async with pool.acquire() as conn:
                api_rows = await conn.fetch(
                    """SELECT DISTINCT ON (symbol) symbol, stock_name, exchange,
                              currency, sector, industry, beta, is_active,
                              sharesoutstanding AS api_shares
                       FROM us_stock_basic WHERE source='api'
                       ORDER BY symbol, date DESC""")
            api_df = pd.DataFrame(api_rows,
                columns=['symbol', 'stock_name', 'exchange', 'currency',
                         'sector', 'industry', 'beta', 'is_active', 'api_shares'])
            await ctx.log("info", f"api_meta loaded: {len(api_df)} symbols")

            grand_inserted = 0
            grand_total = 0
            for i, (out_start, out_end) in enumerate(chunks, 1):
                load_start = out_start - timedelta(days=daily_lookback)
                await ctx.log("info",
                              f"chunk {i}/{len(chunks)} out=[{out_start}~{out_end}] "
                              f"load=[{load_start}~{out_end}]")
                inserted, total = await _compute_stock_basic_window(
                    pool, load_start, out_end, out_start, out_end, api_df=api_df)
                grand_inserted += inserted
                grand_total += total
                await ctx.log("info",
                              f"chunk {i}/{len(chunks)} done: {inserted} rows staged "
                              f"(cumulative {grand_inserted})")

            # 최종 카운트
            async with pool.acquire() as conn:
                computed_count = await conn.fetchval(
                    """SELECT COUNT(*) FROM us_stock_basic
                       WHERE source='computed' AND date BETWEEN $1 AND $2""",
                    overall_out_start, overall_out_end)
        finally:
            await pool.close()

    return {"computed_rows": computed_count,
            "date_range": f"{overall_out_start.isoformat()}~{overall_out_end.isoformat()}",
            "chunks": len(chunks),
            "staged": grand_inserted,
            "total_records_across_chunks": grand_total}


async def _LEGACY_task_us_stock_basic_compute_single_pass(ctx: Ctx) -> dict:
    """Kept temporarily for equality testing — DELETE after chunk validation."""
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
    "financials_verify": task_financials_verify,
    "listing_status": task_listing_status,
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
    "news_insider_top_n":   task_news_insider_top_n,
    "news_history_backfill": task_news_history_backfill,
    "top_signal_reco": task_top_signal_reco,
}


def build_dag(country: str, kind: str = "full") -> List[dict]:
    """DAG 정의.

    kind="full"  : 데이터 수집 → 등급 → 백테스트 (종단 = backtest).
    kind="reco"  : 동일한 당일 데이터/등급 체인을 그대로 타되, 종단 노드만
                   backtest → top_signal_reco 로 교체. 즉 backtester 와 같은
                   구조로 "당일 데이터 적재 → Pass-A → 옵션 → Pass-B" 를 줄줄이
                   엮은 뒤 마지막에 최신 등급으로 추천을 산출한다. 데이터 태스크는
                   모두 멱등/증분이라 이미 쌓인 날짜는 건너뛰어 빠르게 끝난다.
                   (짧은 최근 구간으로 실행 — main.py 가 reco 기본 윈도우를 좁힘)
    """
    if country == "US":
        dag = [
            {"id": "partitions",     "name": "0. DB partitions",                  "depends_on": []},
            {"id": "stock_listing",  "name": "1. NASDAQ/NYSE listing",            "depends_on": ["partitions"]},
            {"id": "listing_status", "name": "1b. AV listing_status (active+delisted, PIT universe)", "depends_on": ["partitions"]},
            {"id": "finnhub_symbol", "name": "2. Finnhub symbol master",          "depends_on": ["stock_listing"]},
            {"id": "stock_basic",    "name": "3. US Stock Basic (fundamentals)",  "depends_on": ["finnhub_symbol"]},
            {"id": "us_daily",       "name": "4. US Daily OHLCV (outputsize=full)","depends_on": ["stock_basic"]},
            {"id": "us_etf",         "name": "5. US ETF daily (SPY/QQQ — 시장레짐)", "depends_on": ["partitions"]},
            {"id": "us_weekly",      "name": "5b. US Weekly OHLCV (indicator 의존)", "depends_on": ["stock_basic"]},
            {"id": "us_calculator",  "name": "6. Technical indicators (14종, DB 계산)", "depends_on": ["us_daily","us_weekly"]},
            {"id": "earnings_history", "name": "7a. Earnings history (reportedDate)", "depends_on": ["stock_basic"]},
            {"id": "financials",     "name": "7. Income / Balance / CashFlow",    "depends_on": ["stock_basic","earnings_history"]},
            {"id": "financials_verify", "name": "7a-2. Financials gap-fill (silent-drop 보호)", "depends_on": ["financials"]},
            {"id": "stock_basic_compute", "name": "7b. Stock Basic 시점별 computed (pandas asof, 17 컬럼)", "depends_on": ["us_daily","stock_basic","financials_verify"]},
            {"id": "macros",         "name": "8. Macros (Fed/Treasury/CPI/UE)",   "depends_on": ["partitions"]},
            {"id": "em8_pre_filter", "name": "9. EM8 pre-filter (top-500/date)",  "depends_on": ["us_daily","us_weekly","us_calculator"]},
            {"id": "us_mv_sector_refresh", "name": "9b. Sector MV (mv_us_sector_daily_performance)", "depends_on": ["us_daily","stock_basic"]},
            {"id": "grades_pass_a",  "name": "10. Pass-A grades (top-N only)",    "depends_on": ["us_calculator","us_etf","financials_verify","macros","em8_pre_filter","stock_basic_compute","us_mv_sector_refresh","listing_status"]},
            {"id": "select_top_n",   "name": "10. Pick top-N symbols",            "depends_on": ["grades_pass_a"]},
            {"id": "options_top_n",  "name": "11. Options backfill (top-N only)", "depends_on": ["select_top_n"]},
            {"id": "news_history_backfill", "name": "11b. News history backfill (top-N union, 분기 단위)", "depends_on": ["select_top_n"]},
            {"id": "grades_pass_b",  "name": "12. Pass-B grades (with options)",  "depends_on": ["options_top_n","news_history_backfill"]},
            {"id": "backtest",       "name": "13. Run backtest",                  "depends_on": ["grades_pass_b"]},
        ]
        reco_dep = "grades_pass_b"
    else:
        dag = [
            {"id": "partitions",     "name": "0. DB partitions",          "depends_on": []},
            {"id": "kr_daily",       "name": "1. KR daily pipeline",       "depends_on": ["partitions"]},
            {"id": "kr_dart",        "name": "2. KR DART filings",         "depends_on": ["partitions"]},
            {"id": "grades_pass_a",  "name": "3. Generate grades",         "depends_on": ["kr_daily","kr_dart"]},
            {"id": "backtest",       "name": "4. Run backtest",            "depends_on": ["grades_pass_a"]},
        ]
        reco_dep = "grades_pass_a"

    if kind == "reco":
        # 종단 backtest → top_signal_reco 교체. 추가로 select_top_n 직후
        # news/insider 사전수집을 끼워, grades_pass_b 의 event_modifier 가
        # news_modifier + insider_modifier 를 실제로 점수에 반영하게 한다
        # (테이블이 비어있으면 두 modifier 가 0 으로 무력화됨). full 백테스트
        # 에는 추가하지 않음 — 역사 뉴스 backfill 비용/유효성 문제.
        dag = [t for t in dag if t["id"] != "backtest"]
        if country == "US":
            dag.append({
                "id": "news_insider_top_n",
                "name": "11b. News + Insider 사전수집 (top-N, reco 한정)",
                "depends_on": ["select_top_n"],
            })
            # grades_pass_b 가 news_insider 완료도 기다리도록 의존성 추가
            for t in dag:
                if t["id"] == "grades_pass_b":
                    deps = list(t.get("depends_on") or [])
                    if "news_insider_top_n" not in deps:
                        deps.append("news_insider_top_n")
                    t["depends_on"] = deps
                    break
        dag.append({
            "id": "top_signal_reco",
            "name": "14. 최고 signal 종목 추천 (top-3 STRONG_BUY / 20일 리밸 전략)",
            "depends_on": [reco_dep],
        })
    return dag
