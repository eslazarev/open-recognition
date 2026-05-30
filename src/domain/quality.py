"""QualityFilter policy for face acceptance.

Pure domain logic — given a detected `Face` (bbox, confidence,
landmarks) and a `QualityFilter` level, decide whether the face is
good enough to index/search, and if not, which AWS-style `Reasons`
to attach.

We only use signals that come for free out of YuNet:
- `confidence` (YuNet score, [0, 100])
- bbox area as a fraction of the image
- approximate roll angle from the eye landmarks

Brightness and sharpness checks would need crop-level analysis and
are intentionally not implemented here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

from domain.face import Face


class QualityFilter(StrEnum):
    NONE = "NONE"
    AUTO = "AUTO"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


@dataclass(frozen=True, slots=True)
class QualityPolicy:
    min_confidence: float       # YuNet score, [0, 100]
    min_relative_area: float    # bbox.width * bbox.height, [0, 1]
    max_roll_degrees: float     # absolute tilt of the eye line


_PRESETS: dict[QualityFilter, QualityPolicy] = {
    QualityFilter.NONE:   QualityPolicy(0.0,  0.0,    180.0),
    QualityFilter.AUTO:   QualityPolicy(60.0, 0.001,  45.0),
    QualityFilter.LOW:    QualityPolicy(70.0, 0.005,  40.0),
    QualityFilter.MEDIUM: QualityPolicy(85.0, 0.01,   30.0),
    QualityFilter.HIGH:   QualityPolicy(95.0, 0.02,   20.0),
}


def policy_for(qf: QualityFilter) -> QualityPolicy:
    return _PRESETS[qf]


def assess_face(face: Face, qf: QualityFilter) -> list[str]:
    """Return AWS-style rejection reasons; empty list means face passes."""
    policy = _PRESETS[qf]
    reasons: list[str] = []
    if face.confidence < policy.min_confidence:
        reasons.append("LOW_CONFIDENCE")
    area = face.bbox.width * face.bbox.height
    if area < policy.min_relative_area:
        reasons.append("SMALL_BOUNDING_BOX")
    if _eye_roll_degrees(face) > policy.max_roll_degrees:
        reasons.append("EXTREME_POSE")
    return reasons


def _eye_roll_degrees(face: Face) -> float:
    """Absolute roll angle between the two eye landmarks, in degrees.

    Returns 0.0 if both eye landmarks are not present — we can't
    measure pose without them, so we don't penalise.
    """
    eyes = {lm.type: lm for lm in face.landmarks if lm.type in ("eyeLeft", "eyeRight")}
    if "eyeLeft" not in eyes or "eyeRight" not in eyes:
        return 0.0
    dx = eyes["eyeRight"].x - eyes["eyeLeft"].x
    dy = eyes["eyeRight"].y - eyes["eyeLeft"].y
    return abs(math.degrees(math.atan2(dy, dx)))
