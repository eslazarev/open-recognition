from __future__ import annotations

from typing import Any

from fastapi import Request

from application.list_faces import list_faces
from domain.collection import FACE_MODEL_VERSION
from interface.http.operations._common import (
    bbox_dict,
    decode_next_token,
    encode_next_token,
)
from interface.http.schemas import ListFacesRequest


async def handle(request: Request, payload: dict[str, Any]) -> dict[str, Any]:
    req = ListFacesRequest.model_validate(payload)
    offset = decode_next_token(req.next_token)
    result = await list_faces(
        collection_id=req.collection_id,
        max_results=req.max_results,
        offset=offset,
        collection_repo=request.app.state.collection_repo,
        face_repo=request.app.state.face_repo,
    )
    return {
        "Faces": [
            {
                "FaceId": str(r.face_id),
                "BoundingBox": bbox_dict(r.face.bbox),
                "ImageId": str(r.image_id),
                "ExternalImageId": r.external_image_id,
                "Confidence": r.face.confidence,
            }
            for r in result.records
        ],
        "FaceModelVersion": FACE_MODEL_VERSION,
        "NextToken": encode_next_token(result.next_offset),
    }
