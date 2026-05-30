"""ListFaces use case."""

from __future__ import annotations

from dataclasses import dataclass

from application.ports import CollectionRepository, FaceRepository
from domain.collection import validate_collection_id
from domain.errors import ResourceNotFoundError
from domain.face_record import FaceRecord


@dataclass(frozen=True, slots=True)
class ListFacesResult:
    records: list[FaceRecord]
    next_offset: int | None


async def list_faces(
    collection_id: str,
    max_results: int,
    offset: int,
    collection_repo: CollectionRepository,
    face_repo: FaceRepository,
) -> ListFacesResult:
    cid = validate_collection_id(collection_id)
    if await collection_repo.get(cid) is None:
        raise ResourceNotFoundError(f"Collection {cid!r} not found")
    limit = max(1, min(max_results, 4096))
    records, total = await face_repo.list(cid, limit=limit, offset=offset)
    consumed = offset + len(records)
    next_offset = consumed if consumed < total else None
    return ListFacesResult(records=records, next_offset=next_offset)
