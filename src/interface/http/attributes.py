"""Map the AWS DetectFaces `Attributes` parameter to internal analyzer keys.

We only back a subset with permissive models (pose, quality, emotions, smile,
landmarks, eyes_open, mouth_open).
Unsupported AWS names (GENDER, AGE_RANGE, BEARD, …) are silently ignored.
BoundingBox / Confidence / Landmarks are part of DEFAULT and are always
included unless an explicit non-DEFAULT attribute set is requested.
"""

from __future__ import annotations

_DEFAULT = frozenset({"pose", "quality", "landmarks"})
_ALL = frozenset({"pose", "quality", "landmarks", "emotions", "smile", "eyes_open", "mouth_open"})
_NAME_MAP = {
    "POSE": "pose",
    "QUALITY": "quality",
    "EMOTIONS": "emotions",
    "SMILE": "smile",
    "EYES_OPEN": "eyes_open",
    "MOUTH_OPEN": "mouth_open",
}


def requested_attributes(attributes: list[str] | None) -> set[str]:
    if not attributes:
        return set(_DEFAULT)
    req: set[str] = set()
    for raw in attributes:
        name = raw.strip().upper()
        if name == "DEFAULT":
            req |= _DEFAULT
        elif name == "ALL":
            req |= _ALL
        elif name in _NAME_MAP:
            req.add(_NAME_MAP[name])
        # unsupported names are ignored
    return req or set(_DEFAULT)
