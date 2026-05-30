"""DeleteCollection use case."""

from __future__ import annotations

from application.ports import CollectionRepository
from domain.collection import validate_collection_id
from domain.errors import ResourceNotFoundError


async def delete_collection(collection_id: str, repo: CollectionRepository) -> None:
    cid = validate_collection_id(collection_id)
    if await repo.get(cid) is None:
        raise ResourceNotFoundError(f"Collection {cid!r} not found")
    await repo.delete(cid)
