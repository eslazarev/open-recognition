from __future__ import annotations

import asyncio
from typing import Any

from fastapi import Request

from application.compare_faces import compare_faces
from interface.http.operations._common import (
    bbox_dict,
    decode_aws_image,
    landmarks_list,
)
from interface.http.schemas import CompareFacesRequest


async def handle(request: Request, payload: dict[str, Any]) -> dict[str, Any]:
    req = CompareFacesRequest.model_validate(payload)
    source = decode_aws_image(req.source_image)
    target = decode_aws_image(req.target_image)
    # compare_faces is fully sync (detect + embed N times); push it to
    # the threadpool. Inside, the cv2 pool gives true parallelism across
    # concurrent requests; cv2 releases the GIL during inference.
    result = await asyncio.to_thread(
        compare_faces,
        source,
        target,
        req.similarity_threshold,
        request.app.state.detector,
        request.app.state.recognizer,
    )
    return {
        "SourceImageFace": {
            "BoundingBox": bbox_dict(result.source_face.bbox),
            "Confidence": result.source_face.confidence,
        },
        "FaceMatches": [
            {
                "Similarity": m.similarity,
                "Face": {
                    "BoundingBox": bbox_dict(m.face.bbox),
                    "Confidence": m.face.confidence,
                    "Landmarks": landmarks_list(m.face),
                },
            }
            for m in result.face_matches
        ],
        "UnmatchedFaces": [
            {
                "BoundingBox": bbox_dict(f.bbox),
                "Confidence": f.confidence,
                "Landmarks": landmarks_list(f),
            }
            for f in result.unmatched_faces
        ],
    }
