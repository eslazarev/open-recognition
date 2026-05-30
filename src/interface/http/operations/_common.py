"""Helpers shared across operation handlers."""

from __future__ import annotations

import base64
from typing import Any

import numpy as np
from numpy.typing import NDArray

from domain.errors import InvalidImageFormatError, InvalidS3ObjectError
from domain.face import BoundingBox as DomBox
from domain.face import Face as DomFace
from infrastructure.cv.image_decoder import decode_image


def decode_aws_image(image: dict[str, Any] | Any) -> NDArray[np.uint8]:
    """Pull Image.Bytes (base64 string) out of the request and decode."""
    payload = image.model_dump(by_alias=True) if hasattr(image, "model_dump") else image
    b64 = payload.get("Bytes")
    if not b64:
        if payload.get("S3Object"):
            raise InvalidS3ObjectError("S3Object is not supported by this server")
        raise InvalidImageFormatError("Image.Bytes is required")
    if isinstance(b64, str):
        try:
            raw = base64.b64decode(b64, validate=True)
        except (ValueError, base64.binascii.Error) as exc:
            raise InvalidImageFormatError("Image.Bytes is not valid base64") from exc
    elif isinstance(b64, (bytes, bytearray)):
        raw = bytes(b64)
    else:
        raise InvalidImageFormatError("Image.Bytes must be a base64 string")
    return decode_image(raw)


def bbox_dict(bbox: DomBox) -> dict[str, float]:
    return {
        "Width": bbox.width,
        "Height": bbox.height,
        "Left": bbox.left,
        "Top": bbox.top,
    }


def landmarks_list(face: DomFace) -> list[dict[str, Any]]:
    return [{"Type": lm.type, "X": lm.x, "Y": lm.y} for lm in face.landmarks]


def encode_next_token(offset: int | None) -> str | None:
    if offset is None:
        return None
    return base64.urlsafe_b64encode(str(offset).encode()).decode()


def decode_next_token(token: str | None) -> int:
    if not token:
        return 0
    try:
        return int(base64.urlsafe_b64decode(token.encode()).decode())
    except (ValueError, base64.binascii.Error) as exc:
        from domain.errors import InvalidParameterValueError

        raise InvalidParameterValueError(f"Invalid NextToken: {token!r}") from exc
