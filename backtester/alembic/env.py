"""Alembic runtime environment.

Customized for this project:

* ``sqlalchemy.url`` is taken from ``DATABASE_URL`` (set by
  docker-compose) rather than the ``alembic.ini`` placeholder, so
  ``alembic upgrade head`` can run inside or outside the container
  with no config editing.
* ``target_metadata`` points at ``db.base.Base.metadata`` and imports
  every ORM module so ``alembic revision --autogenerate`` sees every
  table mapped to :class:`Base` in the codebase.
"""

from __future__ import annotations

import os
from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool

from alembic import context

# Import-side effects: register every ORM model on Base.metadata.
from db.base import Base
from signals.adapters import orm as _signals_orm  # noqa: F401

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Prefer ``DATABASE_URL`` from the environment; fall back to the ini
# value so local override still works.
_env_url = os.environ.get("DATABASE_URL")
if _env_url:
    config.set_main_option("sqlalchemy.url", _env_url)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection, target_metadata=target_metadata
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
