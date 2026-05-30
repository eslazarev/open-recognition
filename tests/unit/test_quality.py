import pytest

from domain.face import BoundingBox, Face, Landmark
from domain.quality import QualityFilter, assess_face


def _face(
    *,
    confidence: float = 99.0,
    side: float = 0.4,
    eye_tilt_deg: float = 0.0,
) -> Face:
    """Build a Face with a centred bbox and synthetic eye landmarks."""
    import math

    bbox = BoundingBox(width=side, height=side, left=0.1, top=0.1)
    # Anchor eyes around (0.4, 0.3), with a configurable rotation.
    cx, cy, sep = 0.4, 0.3, 0.1
    dx = sep * math.cos(math.radians(eye_tilt_deg))
    dy = sep * math.sin(math.radians(eye_tilt_deg))
    landmarks = (
        Landmark(type="eyeLeft", x=cx - dx / 2, y=cy - dy / 2),
        Landmark(type="eyeRight", x=cx + dx / 2, y=cy + dy / 2),
        Landmark(type="nose", x=cx, y=cy + 0.05),
        Landmark(type="mouthLeft", x=cx - 0.04, y=cy + 0.1),
        Landmark(type="mouthRight", x=cx + 0.04, y=cy + 0.1),
    )
    return Face(bbox=bbox, confidence=confidence, landmarks=landmarks)


def test_none_filter_accepts_anything() -> None:
    f = _face(confidence=1.0, side=0.001, eye_tilt_deg=90.0)
    assert assess_face(f, QualityFilter.NONE) == []


def test_auto_filter_accepts_normal_face() -> None:
    f = _face(confidence=85.0, side=0.3, eye_tilt_deg=3.0)
    assert assess_face(f, QualityFilter.AUTO) == []


@pytest.mark.parametrize(
    "qf,expected",
    [
        (QualityFilter.AUTO, "LOW_CONFIDENCE"),
        (QualityFilter.LOW, "LOW_CONFIDENCE"),
        (QualityFilter.MEDIUM, "LOW_CONFIDENCE"),
        (QualityFilter.HIGH, "LOW_CONFIDENCE"),
    ],
)
def test_low_confidence_rejected_at_every_level(qf: QualityFilter, expected: str) -> None:
    f = _face(confidence=10.0)
    assert expected in assess_face(f, qf)


def test_small_bbox_rejected_at_medium_and_higher() -> None:
    # area = 0.0837 — passes AUTO(0.001), LOW(0.005); fails MEDIUM(0.01)?
    # Pick area that lands inside [LOW, MEDIUM): 0.007.
    import math

    side = math.sqrt(0.007)
    f = _face(side=side)
    assert "SMALL_BOUNDING_BOX" not in assess_face(f, QualityFilter.AUTO)
    assert "SMALL_BOUNDING_BOX" not in assess_face(f, QualityFilter.LOW)
    assert "SMALL_BOUNDING_BOX" in assess_face(f, QualityFilter.MEDIUM)
    assert "SMALL_BOUNDING_BOX" in assess_face(f, QualityFilter.HIGH)


def test_tilted_face_rejected_at_medium_and_higher() -> None:
    # tilt that sits between LOW(40°) and MEDIUM(30°) presets.
    f = _face(eye_tilt_deg=33.0)
    assert "EXTREME_POSE" not in assess_face(f, QualityFilter.AUTO)
    assert "EXTREME_POSE" not in assess_face(f, QualityFilter.LOW)
    assert "EXTREME_POSE" in assess_face(f, QualityFilter.MEDIUM)
    assert "EXTREME_POSE" in assess_face(f, QualityFilter.HIGH)


def test_multiple_reasons_can_be_combined() -> None:
    f = _face(confidence=50.0, side=0.05, eye_tilt_deg=40.0)
    reasons = assess_face(f, QualityFilter.HIGH)
    assert set(reasons) == {"LOW_CONFIDENCE", "SMALL_BOUNDING_BOX", "EXTREME_POSE"}


def test_missing_eye_landmarks_does_not_trigger_pose() -> None:
    bbox = BoundingBox(width=0.3, height=0.3, left=0.1, top=0.1)
    f = Face(bbox=bbox, confidence=99.0, landmarks=())
    assert "EXTREME_POSE" not in assess_face(f, QualityFilter.HIGH)
