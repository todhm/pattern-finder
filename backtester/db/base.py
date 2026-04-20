"""Shared SQLAlchemy declarative base.

Every ORM model in the codebase must inherit from ``Base`` defined
here so Alembic's autogenerate picks up the whole schema from a
single ``target_metadata`` import.
"""

from __future__ import annotations

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""
