from __future__ import annotations

from typing import Any

from fastapi import Request

from application.search_faces_by_image import search_faces_by_image
from domain.collection import FACE_MODEL_VERSION
from interface.http.operations._common import bbox_dict, decode_aws_image
from interface.http.schemas import SearchFacesByImageRequest


async def handle(request: Request, payload: dict[str, Any]) -> dict[str, Any]:
    req = SearchFacesByImageRequest.model_validate(payload)
    image = decode_aws_image(req.image)
    result = await search_faces_by_image(
        collection_id=req.collection_id,
        image=image,
        max_faces=req.max_faces,
        face_match_threshold=req.face_match_threshold,
        detector=request.app.state.detector,
        recognizer=request.app.state.recognizer,
        collection_repo=request.app.state.collection_repo,
        face_repo=request.app.state.face_repo,
        quality_filter=req.quality_filter,
    )
    return {
        "SearchedFaceBoundingBox": bbox_dict(result.searched_face.bbox),
        "SearchedFaceConfidence": result.searched_face.confidence,
        "FaceMatches": [
            {
                "Similarity": m.similarity,
                "Face": {
                    "FaceId": str(m.face_record.face_id),
                    "BoundingBox": bbox_dict(m.face_record.face.bbox),
                    "ImageId": str(m.face_record.image_id),
                    "ExternalImageId": m.face_record.external_image_id,
                    "Confidence": m.face_record.face.confidence,
                },
            }
            for m in result.matches
        ],
        "FaceModelVersion": FACE_MODEL_VERSION,
    }
