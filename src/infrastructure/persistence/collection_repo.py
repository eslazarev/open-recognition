"""Postgres-backed CollectionRepository."""

from __future__ import annotations

from dataclasses import dataclass

import asyncpg

from domain.collection import Collection


@dataclass(slots=True)
class PgCollectionRepository:
    pool: asyncpg.Pool

    async def create(self, collection: Collection) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO collection (collection_id, face_model_version, created_at)
                VALUES ($1, $2, $3)
                """,
                collection.collection_id,
                collection.face_model_version,
                collection.created_at,
            )

    async def delete(self, collection_id: str) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM collection WHERE collection_id = $1",
                collection_id,
            )

    async def get(self, collection_id: str) -> Collection | None:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT collection_id, face_model_version, created_at
                FROM collection WHERE collection_id = $1
                """,
                collection_id,
            )
        if row is None:
            return None
        return Collection(
            collection_id=row["collection_id"],
            face_model_version=row["face_model_version"],
            created_at=row["created_at"],
        )

    async def list(
        self, limit: int, offset: int
    ) -> tuple[list[Collection], int]:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT collection_id, face_model_version, created_at
                FROM collection
                ORDER BY collection_id
                LIMIT $1 OFFSET $2
                """,
                limit,
                offset,
            )
            total = await conn.fetchval("SELECT count(*) FROM collection")
        collections = [
            Collection(
                collection_id=r["collection_id"],
                face_model_version=r["face_model_version"],
                created_at=r["created_at"],
            )
            for r in rows
        ]
        return collections, int(total or 0)
