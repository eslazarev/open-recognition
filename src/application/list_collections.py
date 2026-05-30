"""ListCollections use case (pagination via opaque NextToken/offset)."""

from __future__ import annotations

from dataclasses import dataclass

from application.ports import CollectionRepository
from domain.collection import Collection

DEFAULT_MAX_RESULTS = 100


@dataclass(frozen=True, slots=True)
class ListCollectionsResult:
    collections: list[Collection]
    next_offset: int | None


async def list_collections(
    repo: CollectionRepository, max_results: int, offset: int
) -> ListCollectionsResult:
    limit = max(1, min(max_results, 1000))
    collections, total = await repo.list(limit=limit, offset=offset)
    consumed = offset + len(collections)
    next_offset = consumed if consumed < total else None
    return ListCollectionsResult(collections=collections, next_offset=next_offset)
