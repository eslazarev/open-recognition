"""Integration-test fixtures: real Postgres+pgvector via testcontainers.

The container is spun up once per session and reused; individual tests
namespace their own collections (random suffix) and clean up after.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Iterator
from datetime import UTC, datetime
from uuid import uuid4

import pytest
import pytest_asyncio


def _docker_available() -> bool:
    import shutil

    if not shutil.which("docker"):
        return False
    import subprocess

    return subprocess.run(
        ["docker", "info"], capture_output=True, timeout=5
    ).returncode == 0


pytestmark = pytest.mark.skipif(
    not _docker_available(),
    reason="docker daemon not reachable",
)


@pytest.fixture(scope="session")
def postgres_dsn() -> Iterator[str]:
    """One pgvector container shared by the whole integration session."""
    from testcontainers.postgres import PostgresContainer

    container = PostgresContainer(
        image="pgvector/pgvector:pg16",
        username="face_rekon",
        password="face_rekon",
        dbname="face_rekon",
    )
    container.start()
    try:
        host = container.get_container_host_ip()
        port = container.get_exposed_port(5432)
        dsn = f"postgresql://face_rekon:face_rekon@{host}:{port}/face_rekon"

        # Point both the app code and any inline asyncpg connect calls
        # at the ephemeral container.
        previous = os.environ.get("FACE_REKON_DATABASE_URL")
        os.environ["FACE_REKON_DATABASE_URL"] = dsn
        try:
            # Apply alembic head against the fresh database.
            from infrastructure.persistence.db import run_migrations

            run_migrations()
            yield dsn
        finally:
            if previous is None:
                os.environ.pop("FACE_REKON_DATABASE_URL", None)
            else:
                os.environ["FACE_REKON_DATABASE_URL"] = previous
    finally:
        container.stop()


@pytest_asyncio.fixture
async def pool(postgres_dsn: str) -> AsyncIterator:
    from infrastructure.persistence.db import create_pool

    pool = await create_pool(min_size=2, max_size=4)
    try:
        yield pool
    finally:
        await pool.close()


@pytest_asyncio.fixture
async def collection_id(pool) -> AsyncIterator[str]:
    """Create a fresh collection for one test and drop it after."""
    from domain.collection import FACE_MODEL_VERSION, Collection
    from infrastructure.persistence.collection_repo import (
        PgCollectionRepository,
    )

    repo = PgCollectionRepository(pool=pool)
    cid = f"itest-{uuid4().hex[:8]}"
    await repo.create(
        Collection(
            collection_id=cid,
            face_model_version=FACE_MODEL_VERSION,
            created_at=datetime.now(UTC),
        )
    )
    try:
        yield cid
    finally:
        await repo.delete(cid)
