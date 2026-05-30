from __future__ import annotations

import asyncio
from typing import Any

from fastapi import Request

from application.detect_faces import detect_faces
from interface.http.operations._common import (
    bbox_dict,
    decode_aws_image,
    landmarks_list,
)
from interface.http.schemas import DetectFacesRequest


async def handle(request: Request, payload: dict[str, Any]) -> dict[str, Any]:
    req = DetectFacesRequest.model_validate(payload)
    image = decode_aws_image(req.image)
    # The CV inference is the only work in this use case; hand it off
    # so the event loop stays responsive to other requests.
    result = await asyncio.to_thread(
        detect_faces, image, request.app.state.detector
    )
    return {
        "FaceDetails": [
            {
                "BoundingBox": bbox_dict(f.bbox),
                "Confidence": f.confidence,
                "Landmarks": landmarks_list(f),
            }
            for f in result.faces
        ]
    }
