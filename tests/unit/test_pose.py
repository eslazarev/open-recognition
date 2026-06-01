from domain.face import Landmark
from infrastructure.cv.pose import estimate_pose

_FRONTAL = (
    Landmark("eyeLeft",   0.40, 0.40),
    Landmark("eyeRight",  0.60, 0.40),
    Landmark("nose",      0.50, 0.52),
    Landmark("mouthLeft", 0.42, 0.65),
    Landmark("mouthRight",0.58, 0.65),
)


def test_frontal_face_has_small_angles():
    p = estimate_pose(_FRONTAL, 1000, 1000)
    assert abs(p.roll) < 12 and abs(p.yaw) < 12 and abs(p.pitch) < 20


def test_roll_changes_when_eyes_tilted():
    tilted = (
        Landmark("eyeLeft",   0.40, 0.45),
        Landmark("eyeRight",  0.60, 0.35),
        Landmark("nose",      0.50, 0.52),
        Landmark("mouthLeft", 0.42, 0.68),
        Landmark("mouthRight",0.58, 0.62),
    )
    p = estimate_pose(tilted, 1000, 1000)
    assert abs(p.roll) > 5
