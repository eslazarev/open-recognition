"""asyncpg pool + pgvector codec registration + alembic entry point."""

from __future__ import annotations

import os
from pathlib import Path

import asyncpg
from pgvector.asyncpg import register_vector

DEFAULT_DSN = "postgresql://face_rekon:face_rekon@localhost:5432/face_rekon"

# Project root is three levels above this file:
#   src/infrastructure/persistence/db.py
_PROJECT_ROOT = Path(__file__).resolve().parents[3]


def dsn() -> str:
    return os.environ.get("FACE_REKON_DATABASE_URL", DEFAULT_DSN)


async def _init_connection(conn: asyncpg.Connection) -> None:
    await register_vector(conn)


def run_migrations() -> None:
    """Apply all alembic migrations up to head.

    Sync — alembic uses a sync SQLAlchemy engine. Wrap with
    `asyncio.to_thread()` when called from an async context.
    """
    from alembic.command import upgrade
    from alembic.config import Config

    ini = os.environ.get("FACE_REKON_ALEMBIC_INI") or str(
        _PROJECT_ROOT / "alembic.ini"
    )
    cfg = Config(ini)
    upgrade(cfg, "head")


async def create_pool(min_size: int = 1, max_size: int = 10) -> asyncpg.Pool:
    return await asyncpg.create_pool(
        dsn=dsn(),
        min_size=min_size,
        max_size=max_size,
        init=_init_connection,
    )
