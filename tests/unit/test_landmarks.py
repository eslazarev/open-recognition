import numpy as np

from domain.landmarks import (
    AWS_LANDMARK_INDEX, named_landmarks, eye_aspect_ratio, mouth_open_ratio,
)


def _mesh():
    rng = np.random.default_rng(0)
    return rng.random((478, 2))


def test_named_landmarks_cover_aws_types():
    lms = named_landmarks(_mesh())
    types = {lm.type for lm in lms}
    for must in ("eyeLeft", "eyeRight", "nose", "mouthLeft", "mouthRight",
                 "leftPupil", "rightPupil", "chinBottom"):
        assert must in types
    assert len(lms) == len(AWS_LANDMARK_INDEX)
    for lm in lms:
        assert 0.0 <= lm.x <= 1.0 and 0.0 <= lm.y <= 1.0


def test_ear_open_vs_closed():
    pts = np.full((478, 2), 0.5)
    for i in (160, 158):
        pts[i] = (0.50, 0.40)
    for i in (153, 144):
        pts[i] = (0.50, 0.60)
    pts[33] = (0.45, 0.50); pts[133] = (0.55, 0.50)
    for i in (385, 387):
        pts[i] = (0.50, 0.40)
    for i in (373, 380):
        pts[i] = (0.50, 0.60)
    pts[362] = (0.45, 0.50); pts[263] = (0.55, 0.50)
    open_ear = eye_aspect_ratio(pts, 1000, 1000)
    for i in (160, 158, 153, 144, 385, 387, 373, 380):
        pts[i] = (0.50, 0.50)
    closed_ear = eye_aspect_ratio(pts, 1000, 1000)
    assert open_ear > closed_ear
    assert closed_ear < 0.1


def test_mouth_open_ratio_uses_inter_pupil_and_separates():
    pts = np.full((478, 2), 0.5)
    pts[468] = (0.45, 0.40); pts[473] = (0.55, 0.40)   # pupils -> inter-pupil dist
    pts[13] = (0.50, 0.45); pts[14] = (0.50, 0.55)     # lips apart -> open
    open_ratio = mouth_open_ratio(pts, 1000, 1000)
    pts[13] = (0.50, 0.50); pts[14] = (0.50, 0.50)     # lips together -> closed
    closed_ratio = mouth_open_ratio(pts, 1000, 1000)
    assert open_ratio > closed_ratio
    assert closed_ratio < 0.02


def test_wide_smile_not_penalized_by_mouth_width():
    # Wide mouth corners must NOT lower the open ratio (the old bug): the ratio
    # depends on inter-pupil distance, not mouth width.
    narrow = np.full((478, 2), 0.5)
    narrow[468] = (0.45, 0.40); narrow[473] = (0.55, 0.40)
    narrow[78] = (0.47, 0.50); narrow[308] = (0.53, 0.50)   # narrow mouth
    narrow[13] = (0.50, 0.47); narrow[14] = (0.50, 0.53)
    wide = narrow.copy()
    wide[78] = (0.35, 0.50); wide[308] = (0.65, 0.50)        # same lips, wide mouth
    assert abs(mouth_open_ratio(narrow, 1000, 1000) - mouth_open_ratio(wide, 1000, 1000)) < 1e-9
