"""Engine + session-factory singletons.

Kept module-level so the connection pool is reused across repository
instances and across Streamlit reruns. ``DATABASE_URL`` is provided
by ``docker-compose.yaml`` for the backtester service.
"""

from __future__ import annotations

import os
from functools import lru_cache

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker


def _dsn() -> str:
    return os.environ.get(
        "DATABASE_URL",
        "postgresql://backtester:backtester@db:5432/backtester",
    )


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """Return a process-wide SQLAlchemy engine. Lazy so module import
    doesn't require DB connectivity (tests / tooling)."""
    return create_engine(_dsn(), pool_pre_ping=True, future=True)


@lru_cache(maxsize=1)
def _session_factory() -> sessionmaker:
    return sessionmaker(bind=get_engine(), expire_on_commit=False, future=True)


def session_scope() -> Session:
    """Return a new session — caller is responsible for context
    management (commit/rollback/close). Use via ``with session_scope() as s:``.
    """
    return _session_factory()()
