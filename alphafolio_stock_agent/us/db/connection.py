"""
PostgreSQL connection pool management module
Async database connection management using asyncpg
"""
import asyncpg
from contextlib import asynccontextmanager
from typing import Optional
from us.config import settings


class Database:
    """Database connection pool management class"""
    pool: Optional[asyncpg.Pool] = None

    @classmethod
    async def connect(cls) -> None:
        """Create database connection pool"""
        if cls.pool is not None:
            return

        # asyncpg uses postgresql:// format (convert from postgresql+asyncpg://)
        url = settings.DATABASE_URL
        if "postgresql+asyncpg://" in url:
            url = url.replace("postgresql+asyncpg://", "postgresql://")

        cls.pool = await asyncpg.create_pool(
            url,
            min_size=1,
            max_size=5,
            command_timeout=60,
            ssl='prefer'
        )

    @classmethod
    async def disconnect(cls) -> None:
        """Close database connection pool"""
        if cls.pool:
            await cls.pool.close()
            cls.pool = None

    @classmethod
    @asynccontextmanager
    async def get_connection(cls):
        """Connection context manager

        Usage:
            async with Database.get_connection() as conn:
                row = await conn.fetchrow("SELECT * FROM table WHERE id = $1", id)
        """
        if cls.pool is None:
            await cls.connect()

        async with cls.pool.acquire() as conn:
            yield conn

    @classmethod
    async def execute(cls, query: str, *args) -> str:
        """Execute single query (INSERT, UPDATE, DELETE)"""
        async with cls.get_connection() as conn:
            return await conn.execute(query, *args)

    @classmethod
    async def fetchrow(cls, query: str, *args) -> Optional[dict]:
        """Fetch single row"""
        async with cls.get_connection() as conn:
            row = await conn.fetchrow(query, *args)
            return dict(row) if row else None

    @classmethod
    async def fetch(cls, query: str, *args) -> list[dict]:
        """Fetch multiple rows"""
        async with cls.get_connection() as conn:
            rows = await conn.fetch(query, *args)
            return [dict(row) for row in rows]
