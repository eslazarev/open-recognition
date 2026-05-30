"""DeleteFaces use case."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from application.ports import CollectionRepository, FaceRepository
from domain.collection import validate_collection_id
from domain.errors import ResourceNotFoundError


@dataclass(frozen=True, slots=True)
class DeleteFacesResult:
    deleted: list[UUID]


async def delete_faces(
    collection_id: str,
    face_ids: list[UUID],
    collection_repo: CollectionRepository,
    face_repo: FaceRepository,
) -> DeleteFacesResult:
    cid = validate_collection_id(collection_id)
    if await collection_repo.get(cid) is None:
        raise ResourceNotFoundError(f"Collection {cid!r} not found")
    deleted = await face_repo.delete_many(cid, face_ids)
    return DeleteFacesResult(deleted=deleted)
