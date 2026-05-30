import pytest

from domain.face import BoundingBox, Face, Landmark


def test_bbox_accepts_relative_coords() -> None:
    b = BoundingBox(width=0.4, height=0.6, left=0.1, top=0.2)
    assert b.width == 0.4


def test_bbox_rejects_out_of_range_dimensions() -> None:
    with pytest.raises(ValueError):
        BoundingBox(width=1.2, height=0.5, left=0.0, top=0.0)


def test_face_rejects_out_of_range_confidence() -> None:
    bbox = BoundingBox(width=0.1, height=0.1, left=0.0, top=0.0)
    with pytest.raises(ValueError):
        Face(bbox=bbox, confidence=101.0)


def test_face_default_landmarks_empty() -> None:
    bbox = BoundingBox(width=0.1, height=0.1, left=0.0, top=0.0)
    f = Face(bbox=bbox, confidence=99.0)
    assert f.landmarks == ()


def test_landmark_stores_xy() -> None:
    lm = Landmark(type="eyeLeft", x=0.1, y=0.2)
    assert (lm.x, lm.y) == (0.1, 0.2)
