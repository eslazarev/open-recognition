import numpy as np

from application.ports import DetectedFace
from domain.face import BoundingBox, Face, Landmark
from infrastructure.cv.composite_analyzer import CompositeFaceAnalyzer, _confidence


def test_confidence_logistic():
    # 50% exactly at the boundary
    assert _confidence(0.10, 0.10, 45.0) == 50.0
    # clearly open -> high confidence
    assert _confidence(0.30, 0.10, 45.0) > 95.0
    # clearly closed -> high confidence (in the 'closed' decision)
    assert _confidence(0.00, 0.10, 45.0) > 95.0
    # monotonic away from threshold, always >= 50
    for v in (0.0, 0.05, 0.1, 0.2, 0.4):
        assert _confidence(v, 0.10, 45.0) >= 50.0

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


class _FakeMesh:
    def landmarks(self, image, detected):
        pts = np.full((478, 2), 0.5)
        for i in (160, 158, 385, 387): pts[i] = (0.5, 0.4)
        for i in (153, 144, 373, 380): pts[i] = (0.5, 0.6)
        pts[33] = (0.45, 0.5); pts[133] = (0.55, 0.5)
        pts[362] = (0.45, 0.5); pts[263] = (0.55, 0.5)
        pts[468] = (0.45, 0.4); pts[473] = (0.55, 0.4)   # pupils (inter-pupil dist)
        pts[13] = (0.5, 0.42); pts[14] = (0.5, 0.58)     # wide lip gap -> open
        return pts


def _analyzer():
    return CompositeFaceAnalyzer(fer=_FakeFER(), aligner=_FakeAligner(), mesh=_FakeMesh())


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


def test_landmarks_and_eyes_mouth():
    img = np.full((200, 200, 3), 120, dtype=np.uint8)
    fa = _analyzer().analyze(img, _DET,
                             {"landmarks", "eyes_open", "mouth_open"})
    assert {lm.type for lm in fa.landmarks} >= {"leftPupil", "chinBottom", "nose"}
    assert fa.eyes_open is not None and fa.eyes_open.value is True
    assert fa.mouth_open is not None


class _WideSmileMesh:
    """Wide mouth (corners 0.30 apart) but a modest lip gap (0.03) — a
    teeth-showing smile. inter-pupil 0.10, so gap/inter-pupil = 0.15 -> open.
    The old gap/mouth-width metric scored this 0.10 and read it closed.
    """

    def landmarks(self, image, detected):
        pts = np.full((478, 2), 0.5)
        pts[468] = (0.45, 0.40); pts[473] = (0.55, 0.40)  # inter-pupil 0.10
        pts[78] = (0.35, 0.50); pts[308] = (0.65, 0.50)   # WIDE mouth
        pts[13] = (0.50, 0.485); pts[14] = (0.50, 0.515)  # modest gap 0.03
        return pts


def test_wide_smile_mouth_is_detected_open():
    # square image: gap/inter-pupil = 0.03/0.10 = 0.15 >= 0.10 threshold
    analyzer = CompositeFaceAnalyzer(fer=_FakeFER(), aligner=_FakeAligner(), mesh=_WideSmileMesh())
    fa = analyzer.analyze(np.full((300, 300, 3), 120, dtype=np.uint8), _DET, {"mouth_open"})
    assert fa.mouth_open is not None and fa.mouth_open.value is True
