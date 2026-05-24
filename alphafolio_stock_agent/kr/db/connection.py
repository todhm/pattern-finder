"""
PostgreSQL 연결 풀 관리 모듈
asyncpg를 사용한 비동기 데이터베이스 연결 관리
"""
import asyncpg
from contextlib import asynccontextmanager
from typing import Optional
from kr.config import settings


class Database:
    """데이터베이스 연결 풀 관리 클래스"""
    pool: Optional[asyncpg.Pool] = None

    @classmethod
    async def connect(cls) -> None:
        """데이터베이스 연결 풀 생성"""
        if cls.pool is not None:
            return

        # asyncpg는 postgresql:// 형식 사용 (postgresql+asyncpg:// 변환)
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
        """데이터베이스 연결 풀 종료"""
        if cls.pool:
            await cls.pool.close()
            cls.pool = None

    @classmethod
    @asynccontextmanager
    async def get_connection(cls):
        """연결 컨텍스트 매니저

        사용 예시:
            async with Database.get_connection() as conn:
                row = await conn.fetchrow("SELECT * FROM table WHERE id = $1", id)
        """
        if cls.pool is None:
            await cls.connect()

        async with cls.pool.acquire() as conn:
            yield conn

    @classmethod
    async def execute(cls, query: str, *args) -> str:
        """단일 쿼리 실행 (INSERT, UPDATE, DELETE)"""
        async with cls.get_connection() as conn:
            return await conn.execute(query, *args)

    @classmethod
    async def fetchrow(cls, query: str, *args) -> Optional[dict]:
        """단일 행 조회"""
        async with cls.get_connection() as conn:
            row = await conn.fetchrow(query, *args)
            return dict(row) if row else None

    @classmethod
    async def fetch(cls, query: str, *args) -> list[dict]:
        """다중 행 조회"""
        async with cls.get_connection() as conn:
            rows = await conn.fetch(query, *args)
            return [dict(row) for row in rows]
