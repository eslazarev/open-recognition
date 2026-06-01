from domain.face_attributes import (
    BinaryAttr,
    Emotion,
    FaceAttributes,
    ImageQuality,
    Pose,
)


def test_value_objects_construct():
    p = Pose(roll=1.0, yaw=2.0, pitch=3.0)
    q = ImageQuality(brightness=50.0, sharpness=20.0)
    e = Emotion(type="HAPPY", confidence=99.0)
    s = BinaryAttr(value=True, confidence=99.0)
    assert (p.roll, p.yaw, p.pitch) == (1.0, 2.0, 3.0)
    assert (q.brightness, q.sharpness) == (50.0, 20.0)
    assert (e.type, e.confidence) == ("HAPPY", 99.0)
    assert (s.value, s.confidence) == (True, 99.0)


def test_face_attributes_all_optional():
    fa = FaceAttributes()
    assert fa.pose is None and fa.quality is None
    assert fa.emotions == () and fa.smile is None
