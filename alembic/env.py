"""Alembic environment.

DSN is resolved from `OPEN_RECOGNITION_DATABASE_URL` (same env var the
app uses), with the asyncpg-style scheme rewritten to psycopg3 for
the sync SQLAlchemy engine alembic needs.
"""

from __future__ import annotations

from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from infrastructure.persistence.db import dsn

config = context.config
if config.config_file_name:
    fileConfig(config.config_file_name)


def _sa_url() -> str:
    raw = dsn()
    if raw.startswith("postgresql://"):
        return "postgresql+psycopg://" + raw[len("postgresql://") :]
    return raw


def run_migrations_offline() -> None:
    context.configure(
        url=_sa_url(),
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    section = config.get_section(config.config_ini_section, {}) or {}
    section["sqlalchemy.url"] = _sa_url()
    connectable = engine_from_config(
        section, prefix="sqlalchemy.", poolclass=pool.NullPool
    )
    with connectable.connect() as conn:
        context.configure(connection=conn)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
