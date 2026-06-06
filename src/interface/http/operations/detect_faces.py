from __future__ import annotations

import asyncio
from typing import Any

from fastapi import Request

from application.detect_faces import detect_faces
from domain.face_attributes import FaceAttributes
from interface.http.attributes import requested_attributes
from interface.http.operations._common import (
    bbox_dict,
    decode_aws_image,
    landmarks_list,
)
from interface.http.schemas import DetectFacesRequest


def _attr_dict(a: FaceAttributes) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if a.pose is not None:
        out["Pose"] = {"Roll": a.pose.roll, "Yaw": a.pose.yaw, "Pitch": a.pose.pitch}
    if a.quality is not None:
        out["Quality"] = {"Brightness": a.quality.brightness, "Sharpness": a.quality.sharpness}
    if a.emotions:
        out["Emotions"] = [{"Type": e.type, "Confidence": e.confidence} for e in a.emotions]
    if a.smile is not None:
        out["Smile"] = {"Value": a.smile.value, "Confidence": a.smile.confidence}
    if a.eyes_open is not None:
        out["EyesOpen"] = {"Value": a.eyes_open.value, "Confidence": a.eyes_open.confidence}
    if a.mouth_open is not None:
        out["MouthOpen"] = {"Value": a.mouth_open.value, "Confidence": a.mouth_open.confidence}
    return out


async def handle(request: Request, payload: dict[str, Any]) -> dict[str, Any]:
    req = DetectFacesRequest.model_validate(payload)
    image = decode_aws_image(req.image)
    requested = requested_attributes(req.attributes)
    analyzer = getattr(request.app.state, "analyzer", None)
    result = await asyncio.to_thread(
        detect_faces, image, request.app.state.detector, analyzer, requested
    )
    details = []
    for face, attrs in zip(result.faces, result.attributes, strict=False):
        lms = (
            [{"Type": lm.type, "X": lm.x, "Y": lm.y} for lm in attrs.landmarks]
            if attrs.landmarks else landmarks_list(face)
        )
        details.append(
            {
                "BoundingBox": bbox_dict(face.bbox),
                "Confidence": face.confidence,
                "Landmarks": lms,
                **_attr_dict(attrs),
            }
        )
    return {"FaceDetails": details}
