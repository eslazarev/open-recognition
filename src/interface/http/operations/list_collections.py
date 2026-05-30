from __future__ import annotations

from typing import Any

from fastapi import Request

from application.list_collections import list_collections
from interface.http.operations._common import (
    decode_next_token,
    encode_next_token,
)
from interface.http.schemas import ListCollectionsRequest


async def handle(request: Request, payload: dict[str, Any]) -> dict[str, Any]:
    req = ListCollectionsRequest.model_validate(payload)
    offset = decode_next_token(req.next_token)
    result = await list_collections(
        repo=request.app.state.collection_repo,
        max_results=req.max_results,
        offset=offset,
    )
    return {
        "CollectionIds": [c.collection_id for c in result.collections],
        "FaceModelVersions": [c.face_model_version for c in result.collections],
        "NextToken": encode_next_token(result.next_offset),
    }
