"""Repository integration tests against a real pgvector instance."""

from __future__ import annotations

from uuid import uuid4

import numpy as np
import pytest

from domain.embedding import EMBEDDING_DIM, Embedding
from domain.face import BoundingBox, Face
from domain.face_record import FaceRecord
from domain.similarity import cosine_threshold_from_pct
from infrastructure.persistence.collection_repo import (
    PgCollectionRepository,
)
from infrastructure.persistence.face_repo import PgFaceRepository


def _emb(seed: int) -> Embedding:
    rng = np.random.default_rng(seed)
    return Embedding.from_array(rng.standard_normal(EMBEDDING_DIM).astype(np.float32))


def _face_record(collection_id: str, embedding: Embedding, label: str) -> FaceRecord:
    bbox = BoundingBox(width=0.3, height=0.3, left=0.2, top=0.2)
    return FaceRecord(
        face_id=uuid4(),
        collection_id=collection_id,
        image_id=uuid4(),
        face=Face(bbox=bbox, confidence=95.0),
        embedding=embedding,
        external_image_id=label,
    )


async def test_collection_round_trip(pool, collection_id: str) -> None:
    repo = PgCollectionRepository(pool=pool)
    got = await repo.get(collection_id)
    assert got is not None
    assert got.collection_id == collection_id


async def test_face_insert_count_list_delete(pool, collection_id: str) -> None:
    frepo = PgFaceRepository(pool=pool)
    records = [
        _face_record(collection_id, _emb(1), "alice"),
        _face_record(collection_id, _emb(2), "bob"),
        _face_record(collection_id, _emb(3), "carol"),
    ]
    await frepo.add_many(records)
    assert await frepo.count(collection_id) == 3

    listed, total = await frepo.list(collection_id, limit=10, offset=0)
    assert total == 3
    assert {r.external_image_id for r in listed} == {"alice", "bob", "carol"}

    deleted = await frepo.delete_many(
        collection_id, [records[0].face_id, records[2].face_id]
    )
    assert set(deleted) == {records[0].face_id, records[2].face_id}
    assert await frepo.count(collection_id) == 1


async def test_hnsw_search_returns_closest_first(pool, collection_id: str) -> None:
    frepo = PgFaceRepository(pool=pool)
    alice_emb = _emb(7)
    bob_emb = _emb(8)
    records = [
        _face_record(collection_id, alice_emb, "alice"),
        _face_record(collection_id, bob_emb, "bob"),
    ]
    await frepo.add_many(records)

    matches = await frepo.search(
        collection_id,
        query=alice_emb,
        max_faces=5,
        cosine_threshold=cosine_threshold_from_pct(0.0),  # accept everything
    )
    assert matches[0].face_record.external_image_id == "alice"
    assert matches[0].similarity == pytest.approx(100.0, abs=1e-3)
    # Two random 128-d vectors are nearly orthogonal, so bob should
    # come back well below alice.
    assert matches[1].face_record.external_image_id == "bob"
    assert matches[1].similarity < matches[0].similarity


async def test_search_respects_cosine_threshold(pool, collection_id: str) -> None:
    frepo = PgFaceRepository(pool=pool)
    await frepo.add_many([_face_record(collection_id, _emb(1), "x")])
    # Threshold 99% maps to cos ≥ 0.42 — a totally unrelated query
    # should not pass it.
    matches = await frepo.search(
        collection_id,
        query=_emb(99),
        max_faces=5,
        cosine_threshold=cosine_threshold_from_pct(99.0),
    )
    assert matches == []


async def test_collection_delete_cascades_to_faces(pool) -> None:
    from datetime import UTC, datetime

    from domain.collection import FACE_MODEL_VERSION, Collection

    crepo = PgCollectionRepository(pool=pool)
    frepo = PgFaceRepository(pool=pool)
    cid = f"cascade-{uuid4().hex[:8]}"
    await crepo.create(
        Collection(
            collection_id=cid,
            face_model_version=FACE_MODEL_VERSION,
            created_at=datetime.now(UTC),
        )
    )
    await frepo.add_many(
        [_face_record(cid, _emb(i), f"f{i}") for i in range(5)]
    )
    assert await frepo.count(cid) == 5

    await crepo.delete(cid)
    # FK ON DELETE CASCADE → orphan faces are gone.
    assert await frepo.count(cid) == 0
    assert await crepo.get(cid) is None
