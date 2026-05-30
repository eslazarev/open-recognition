from __future__ import annotations

from typing import Any

from fastapi import Request

from application.index_faces import index_faces
from domain.collection import FACE_MODEL_VERSION
from interface.http.operations._common import (
    bbox_dict,
    decode_aws_image,
    landmarks_list,
)
from interface.http.schemas import IndexFacesRequest


async def handle(request: Request, payload: dict[str, Any]) -> dict[str, Any]:
    req = IndexFacesRequest.model_validate(payload)
    image = decode_aws_image(req.image)
    result = await index_faces(
        collection_id=req.collection_id,
        image=image,
        external_image_id=req.external_image_id,
        max_faces=req.max_faces,
        detector=request.app.state.detector,
        recognizer=request.app.state.recognizer,
        collection_repo=request.app.state.collection_repo,
        face_repo=request.app.state.face_repo,
        quality_filter=req.quality_filter,
    )

    return {
        "FaceRecords": [
            {
                "Face": {
                    "FaceId": str(r.face_id),
                    "BoundingBox": bbox_dict(r.face.bbox),
                    "ImageId": str(r.image_id),
                    "ExternalImageId": r.external_image_id,
                    "Confidence": r.face.confidence,
                },
                "FaceDetail": {
                    "BoundingBox": bbox_dict(r.face.bbox),
                    "Confidence": r.face.confidence,
                    "Landmarks": landmarks_list(r.face),
                },
            }
            for r in result.records
        ],
        "UnindexedFaces": [
            {
                "Reasons": u.reasons,
                "FaceDetail": {
                    "BoundingBox": bbox_dict(u.face.bbox),
                    "Confidence": u.face.confidence,
                    "Landmarks": landmarks_list(u.face),
                },
            }
            for u in result.unindexed
        ],
        "FaceModelVersion": FACE_MODEL_VERSION,
    }
