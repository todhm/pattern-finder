"""
Analysis Router - Handles AI agent strategy generation requests
"""
import json
from datetime import datetime
from typing import Literal, Optional

from fastapi import APIRouter, Depends, BackgroundTasks, Header, HTTPException
from pydantic import BaseModel
import redis.asyncio as redis

from api.config import api_settings
from api.dependencies import get_task_redis, get_cache_redis
from api.utils.cache import calculate_ttl, get_cache_key_strategy


async def verify_api_key(x_api_key: str = Header(...)):
    """API 키 인증. API_SECRET_KEY 미설정 시 인증 건너뜀 (로컬 개발 호환)"""
    if not api_settings.API_SECRET_KEY:
        return
    if x_api_key != api_settings.API_SECRET_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


router = APIRouter(prefix="/api/analysis", tags=["analysis"])


# Request/Response Models
class GenerateRequest(BaseModel):
    symbol: str
    market: Literal["KR", "US"]


class GenerateResponse(BaseModel):
    status: Literal["started", "already_running", "completed", "error"]
    message: str
    started_at: Optional[str] = None
    started_by: Optional[str] = None


class StatusResponse(BaseModel):
    status: Literal["idle", "running", "completed", "error"]
    message: str
    started_at: Optional[str] = None


# Redis key patterns
def get_running_key(symbol: str) -> str:
    return f"agent:running:{symbol}"


def get_started_at_key(symbol: str) -> str:
    return f"agent:started_at:{symbol}"


def get_started_by_key(symbol: str) -> str:
    return f"agent:started_by:{symbol}"


async def run_agent_task(symbol: str, market: str, task_redis: redis.Redis):
    """
    Background task to run the agent.
    This is executed in background after the API returns.

    - target_date는 항상 None으로 전달
    - kr_main.py / us_main.py 내부에서 date.today() 사용
    - 완료 시 cache_redis에 strategy 캐시 저장
    """
    result = None
    try:
        if market == "KR":
            from kr.kr_main import run as run_kr
            result = await run_kr(symbol, None, save_to_db=True)
        else:
            from us.us_main import run as run_us
            result = await run_us(symbol, None, save_to_db=True)

        # Save strategy to cache if successful
        if result and not result.get("error"):
            try:
                cache_redis = await get_cache_redis()
                cache_key = get_cache_key_strategy(symbol)
                ttl = calculate_ttl(market)

                await cache_redis.set(
                    cache_key,
                    json.dumps(result, ensure_ascii=False),
                    ex=ttl
                )
                print(f"[INFO] Strategy cached for {symbol} with TTL={ttl}s")

                # Invalidate stock detail cache so next page load fetches fresh data from DB
                detail_cache_key = f"stock:detail:{symbol}"
                await cache_redis.delete(detail_cache_key)
                print(f"[INFO] Invalidated detail cache for {symbol}")
            except Exception as cache_err:
                print(f"[WARNING] Failed to cache strategy for {symbol}: {cache_err}")

    except Exception as e:
        print(f"[ERROR] Agent execution failed for {symbol}: {e}")
    finally:
        # Clean up Redis keys
        await task_redis.delete(
            get_running_key(symbol),
            get_started_at_key(symbol),
            get_started_by_key(symbol)
        )


# Endpoints
@router.post("/generate", response_model=GenerateResponse)
async def generate_strategy(
    request: GenerateRequest,
    background_tasks: BackgroundTasks,
    task_redis: redis.Redis = Depends(get_task_redis),
    _: None = Depends(verify_api_key),
):
    """
    Generate investment strategy for a stock.

    - symbol: 종목코드 (예: 005930, AAPL)
    - market: KR이면 kr_main.py 실행, US면 us_main.py 실행
    - 날짜는 항상 None으로 전달 (kr_main/us_main 내부에서 date.today() 사용)
    - Only the first request triggers the agent execution
    - Subsequent requests return "already_running" status
    - Uses Redis SETNX for atomic lock acquisition
    """
    symbol = request.symbol.upper()
    market = request.market

    running_key = get_running_key(symbol)
    started_at_key = get_started_at_key(symbol)
    started_by_key = get_started_by_key(symbol)

    # Try to acquire lock using SETNX
    is_new = await task_redis.setnx(running_key, "1")

    if is_new:
        # First request - start the agent
        started_at = datetime.now().isoformat()

        # Set additional info and TTL (safety measure)
        await task_redis.set(started_at_key, started_at, ex=api_settings.AGENT_LOCK_TTL)
        await task_redis.set(started_by_key, "api", ex=api_settings.AGENT_LOCK_TTL)
        await task_redis.expire(running_key, api_settings.AGENT_LOCK_TTL)

        # Run agent in background (target_date = None)
        background_tasks.add_task(run_agent_task, symbol, market, task_redis)

        return GenerateResponse(
            status="started",
            message=f"멀티 AI 에이전트가 {symbol} 투자 전략을 생성 중입니다. 2~3분 소요 예정입니다.",
            started_at=started_at
        )
    else:
        # Already running
        started_at = await task_redis.get(started_at_key)
        started_by = await task_redis.get(started_by_key)

        return GenerateResponse(
            status="already_running",
            message=f"현재 {symbol} 전략이 생성 중입니다. 잠시만 기다려 주세요.",
            started_at=started_at,
            started_by=started_by
        )


@router.get("/status/{symbol}", response_model=StatusResponse)
async def get_status(
    symbol: str,
    task_redis: redis.Redis = Depends(get_task_redis),
    _: None = Depends(verify_api_key),
):
    """
    Check the execution status of a strategy generation.
    """
    symbol = symbol.upper()
    running_key = get_running_key(symbol)
    started_at_key = get_started_at_key(symbol)

    is_running = await task_redis.exists(running_key)

    if is_running:
        started_at = await task_redis.get(started_at_key)
        return StatusResponse(
            status="running",
            message=f"{symbol} 전략 생성 중입니다.",
            started_at=started_at
        )
    else:
        return StatusResponse(
            status="idle",
            message="실행 중인 작업이 없습니다."
        )


@router.delete("/cancel/{symbol}")
async def cancel_generation(
    symbol: str,
    task_redis: redis.Redis = Depends(get_task_redis)
):
    """
    Cancel/clear a stuck execution (admin only).
    Note: This only clears the Redis lock, not the actual running process.
    """
    symbol = symbol.upper()

    await task_redis.delete(
        get_running_key(symbol),
        get_started_at_key(symbol),
        get_started_by_key(symbol)
    )

    return {"status": "cleared", "message": f"{symbol} 실행 상태가 초기화되었습니다."}
