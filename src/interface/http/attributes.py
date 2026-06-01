"""Map the AWS DetectFaces `Attributes` parameter to internal analyzer keys.

We only back a subset with permissive models (pose, quality, emotions, smile).
Unsupported AWS names (GENDER, AGE_RANGE, BEARD, …) are silently ignored.
BoundingBox / Confidence / Landmarks are always returned by the handler and
are not represented here.
"""

from __future__ import annotations

_DEFAULT = frozenset({"pose", "quality"})
_ALL = frozenset({"pose", "quality", "emotions", "smile"})
_NAME_MAP = {
    "POSE": "pose",
    "QUALITY": "quality",
    "EMOTIONS": "emotions",
    "SMILE": "smile",
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
