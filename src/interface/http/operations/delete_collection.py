from __future__ import annotations

from typing import Any

from fastapi import Request

from application.delete_collection import delete_collection
from interface.http.schemas import DeleteCollectionRequest


async def handle(request: Request, payload: dict[str, Any]) -> dict[str, Any]:
    req = DeleteCollectionRequest.model_validate(payload)
    await delete_collection(req.collection_id, repo=request.app.state.collection_repo)
    return {"StatusCode": 200}
