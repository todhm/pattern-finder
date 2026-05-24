"""
Dependency injection for FastAPI
"""
import redis.asyncio as redis
from typing import Optional
from contextlib import asynccontextmanager

from api.config import api_settings, RedisDB


class RedisManager:
    """Redis connection manager for different databases"""

    _cache_redis: Optional[redis.Redis] = None
    _task_redis: Optional[redis.Redis] = None

    @classmethod
    async def get_cache_redis(cls) -> redis.Redis:
        """Get Redis connection for caching (Index 0)"""
        if cls._cache_redis is None:
            cls._cache_redis = redis.from_url(
                api_settings.REDIS_URL,
                db=RedisDB.CACHE,
                decode_responses=True
            )
        return cls._cache_redis

    @classmethod
    async def get_task_redis(cls) -> redis.Redis:
        """Get Redis connection for task state management (Index 1)"""
        if cls._task_redis is None:
            cls._task_redis = redis.from_url(
                api_settings.REDIS_URL,
                db=RedisDB.TASK,
                decode_responses=True
            )
        return cls._task_redis

    @classmethod
    async def close_all(cls):
        """Close all Redis connections"""
        if cls._cache_redis:
            await cls._cache_redis.close()
            cls._cache_redis = None
        if cls._task_redis:
            await cls._task_redis.close()
            cls._task_redis = None


# FastAPI dependency functions
async def get_task_redis() -> redis.Redis:
    """Dependency for task Redis"""
    return await RedisManager.get_task_redis()


async def get_cache_redis() -> redis.Redis:
    """Dependency for cache Redis"""
    return await RedisManager.get_cache_redis()
