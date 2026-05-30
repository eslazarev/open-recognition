"""Postgres-backed FaceRepository.

This is the only place in the codebase that issues pgvector `<=>`
queries. Callers convert AWS percentages to cosine thresholds via
`domain.similarity` before invoking `search()`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from uuid import UUID

import asyncpg
import numpy as np

from application.ports import FaceMatch
from domain.embedding import Embedding
from domain.face import BoundingBox, Face
from domain.face_record import FaceRecord
from domain.similarity import similarity_pct


def _row_to_record(row: asyncpg.Record) -> FaceRecord:
    bbox_dict = (
        row["bbox"] if isinstance(row["bbox"], dict) else json.loads(row["bbox"])
    )
    bbox = BoundingBox(
        width=bbox_dict["Width"],
        height=bbox_dict["Height"],
        left=bbox_dict["Left"],
        top=bbox_dict["Top"],
    )
    face = Face(bbox=bbox, confidence=float(row["confidence"]))
    embedding = Embedding(vector=np.asarray(row["embedding"], dtype=np.float32))
    return FaceRecord(
        face_id=row["face_id"],
        collection_id=row["collection_id"],
        image_id=row["image_id"],
        face=face,
        embedding=embedding,
        external_image_id=row["external_image_id"],
    )


@dataclass(slots=True)
class PgFaceRepository:
    pool: asyncpg.Pool

    async def add_many(self, records: list[FaceRecord]) -> None:
        if not records:
            return
        rows = [
            (
                r.face_id,
                r.collection_id,
                r.external_image_id,
                r.image_id,
                json.dumps(
                    {
                        "Width": r.face.bbox.width,
                        "Height": r.face.bbox.height,
                        "Left": r.face.bbox.left,
                        "Top": r.face.bbox.top,
                    }
                ),
                r.face.confidence,
                r.embedding.vector,
            )
            for r in records
        ]
        async with self.pool.acquire() as conn:
            await conn.executemany(
                """
                INSERT INTO face (
                    face_id, collection_id, external_image_id, image_id,
                    bbox, confidence, embedding
                ) VALUES ($1, $2, $3, $4, $5::jsonb, $6, $7)
                """,
                rows,
            )

    async def delete_many(
        self, collection_id: str, face_ids: list[UUID]
    ) -> list[UUID]:
        if not face_ids:
            return []
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                DELETE FROM face
                WHERE collection_id = $1 AND face_id = ANY($2::uuid[])
                RETURNING face_id
                """,
                collection_id,
                face_ids,
            )
        return [r["face_id"] for r in rows]

    async def count(self, collection_id: str) -> int:
        async with self.pool.acquire() as conn:
            val = await conn.fetchval(
                "SELECT count(*) FROM face WHERE collection_id = $1",
                collection_id,
            )
        return int(val or 0)

    async def list(
        self, collection_id: str, limit: int, offset: int
    ) -> tuple[list[FaceRecord], int]:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT face_id, collection_id, external_image_id, image_id,
                       bbox, confidence, embedding
                FROM face
                WHERE collection_id = $1
                ORDER BY created_at, face_id
                LIMIT $2 OFFSET $3
                """,
                collection_id,
                limit,
                offset,
            )
            total = await conn.fetchval(
                "SELECT count(*) FROM face WHERE collection_id = $1",
                collection_id,
            )
        return [_row_to_record(r) for r in rows], int(total or 0)

    async def search(
        self,
        collection_id: str,
        query: Embedding,
        max_faces: int,
        cosine_threshold: float,
    ) -> list[FaceMatch]:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT face_id, collection_id, external_image_id, image_id,
                       bbox, confidence, embedding,
                       1 - (embedding <=> $1) AS cos_sim
                FROM face
                WHERE collection_id = $2
                  AND 1 - (embedding <=> $1) >= $3
                ORDER BY embedding <=> $1
                LIMIT $4
                """,
                query.vector,
                collection_id,
                cosine_threshold,
                max_faces,
            )
        return [
            FaceMatch(
                face_record=_row_to_record(r),
                similarity=similarity_pct(float(r["cos_sim"])),
            )
            for r in rows
        ]
