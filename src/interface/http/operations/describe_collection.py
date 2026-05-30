from __future__ import annotations

from typing import Any

from fastapi import Request

from application.describe_collection import describe_collection
from interface.http.schemas import DescribeCollectionRequest


async def handle(request: Request, payload: dict[str, Any]) -> dict[str, Any]:
    req = DescribeCollectionRequest.model_validate(payload)
    result = await describe_collection(
        req.collection_id,
        collection_repo=request.app.state.collection_repo,
        face_repo=request.app.state.face_repo,
    )
    return {
        "FaceCount": result.face_count,
        "FaceModelVersion": result.collection.face_model_version,
        "CollectionARN": result.collection.arn(),
        "CreationTimestamp": result.collection.created_at.timestamp(),
    }
