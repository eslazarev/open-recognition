"""DescribeCollection use case."""

from __future__ import annotations

from dataclasses import dataclass

from application.ports import CollectionRepository, FaceRepository
from domain.collection import Collection, validate_collection_id
from domain.errors import ResourceNotFoundError


@dataclass(frozen=True, slots=True)
class DescribeCollectionResult:
    collection: Collection
    face_count: int


async def describe_collection(
    collection_id: str,
    collection_repo: CollectionRepository,
    face_repo: FaceRepository,
) -> DescribeCollectionResult:
    cid = validate_collection_id(collection_id)
    collection = await collection_repo.get(cid)
    if collection is None:
        raise ResourceNotFoundError(f"Collection {cid!r} not found")
    return DescribeCollectionResult(
        collection=collection,
        face_count=await face_repo.count(cid),
    )
