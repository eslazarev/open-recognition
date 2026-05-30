"""Collection aggregate root."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime

from domain.errors import InvalidParameterValueError

# AWS Rekognition collection-id rules: [a-zA-Z0-9_.\-]+, max 255 chars.
_COLLECTION_ID_RE = re.compile(r"^[a-zA-Z0-9_.\-]{1,255}$")

# Single, immutable model version string baked into responses.
FACE_MODEL_VERSION = "sface-2021dec-1"


@dataclass(frozen=True, slots=True)
class Collection:
    collection_id: str
    face_model_version: str
    created_at: datetime

    def arn(self) -> str:
        return f"arn:open-recognition:rekognition:::collection/{self.collection_id}"


def validate_collection_id(value: str) -> str:
    if not _COLLECTION_ID_RE.fullmatch(value):
        raise InvalidParameterValueError(
            f"Invalid CollectionId: {value!r}. "
            "Must match [a-zA-Z0-9_.-]{1,255}."
        )
    return value
