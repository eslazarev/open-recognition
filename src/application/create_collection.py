"""CreateCollection use case."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from application.ports import CollectionRepository
from domain.collection import (
    FACE_MODEL_VERSION,
    Collection,
    validate_collection_id,
)
from domain.errors import ResourceAlreadyExistsError


@dataclass(frozen=True, slots=True)
class CreateCollectionResult:
    collection: Collection


async def create_collection(
    collection_id: str, repo: CollectionRepository
) -> CreateCollectionResult:
    cid = validate_collection_id(collection_id)
    if await repo.get(cid) is not None:
        raise ResourceAlreadyExistsError(f"Collection {cid!r} already exists")
    collection = Collection(
        collection_id=cid,
        face_model_version=FACE_MODEL_VERSION,
        created_at=datetime.now(UTC),
    )
    await repo.create(collection)
    return CreateCollectionResult(collection=collection)
