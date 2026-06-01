import numpy as np

from application.ports import DetectedFace
from domain.face import BoundingBox, Face, Landmark
from infrastructure.cv.composite_analyzer import CompositeFaceAnalyzer

_LMS = (
    Landmark("eyeLeft", 0.40, 0.40), Landmark("eyeRight", 0.60, 0.40),
    Landmark("nose", 0.50, 0.52), Landmark("mouthLeft", 0.42, 0.65),
    Landmark("mouthRight", 0.58, 0.65),
)
_FACE = Face(bbox=BoundingBox(0.25, 0.25, 0.5, 0.5), confidence=99.0, landmarks=_LMS)
_DET = DetectedFace(face=_FACE, raw_row=np.zeros(15, dtype=np.float32))


class _FakeFER:
    def predict(self, crop):
        return [0.01, 0.01, 0.01, 0.90, 0.05, 0.01, 0.01]  # happy dominant


class _FakeAligner:
    def align(self, image, detected):
        return np.full((112, 112, 3), 128, dtype=np.uint8)


def _analyzer():
    return CompositeFaceAnalyzer(fer=_FakeFER(), aligner=_FakeAligner())


def test_default_returns_pose_and_quality_only():
    img = np.full((200, 200, 3), 120, dtype=np.uint8)
    fa = _analyzer().analyze(img, _DET, {"pose", "quality"})
    assert fa.pose is not None and fa.quality is not None
    assert fa.emotions == () and fa.smile is None


def test_emotions_and_smile():
    img = np.full((200, 200, 3), 120, dtype=np.uint8)
    fa = _analyzer().analyze(img, _DET, {"emotions", "smile"})
    assert fa.emotions[0].type == "HAPPY"
    assert all(0 <= e.confidence <= 100 for e in fa.emotions)
    assert fa.smile is not None and fa.smile.value is True
    assert fa.pose is None and fa.quality is None
