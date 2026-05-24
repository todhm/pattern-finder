"""
API Configuration Settings
"""
from pydantic_settings import BaseSettings
from typing import Optional


class APISettings(BaseSettings):
    # Database
    DATABASE_URL: str

    # Redis
    REDIS_URL: str = "redis://localhost:6379"

    # OpenAI
    OPENAI_API_KEY: str
    OPENAI_MODEL: str = "gpt-4o-mini"

    # Search APIs (optional)
    SERPER_API_KEY: Optional[str] = None
    GOOGLE_API_KEY: Optional[str] = None
    GOOGLE_SEARCH_ENGINE_ID: Optional[str] = None
    NAVER_CLIENT_ID: Optional[str] = None
    NAVER_CLIENT_SECRET: Optional[str] = None

    # Authentication
    API_SECRET_KEY: Optional[str] = None

    # GCP (optional)
    GCP_SA_KEY: Optional[str] = None

    # Agent execution settings
    AGENT_EXECUTION_TTL: int = 300  # 5 minutes max execution time (safety)
    AGENT_LOCK_TTL: int = 300  # Lock expires after 5 minutes

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


api_settings = APISettings()


# Redis database indices
class RedisDB:
    CACHE = 0       # stock-detail caching
    TASK = 1        # agent execution state management
    STREAM = 2      # AI chat streaming (existing)
    TASK_RESULT = 3 # agent execution results
