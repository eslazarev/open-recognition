from __future__ import annotations

from typing import Any

from fastapi import Request

from application.create_collection import create_collection
from interface.http.schemas import CreateCollectionRequest


async def handle(request: Request, payload: dict[str, Any]) -> dict[str, Any]:
    req = CreateCollectionRequest.model_validate(payload)
    result = await create_collection(
        req.collection_id, repo=request.app.state.collection_repo
    )
    return {
        "CollectionArn": result.collection.arn(),
        "FaceModelVersion": result.collection.face_model_version,
        "StatusCode": 200,
    }
