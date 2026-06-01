from interface.http.schemas import (
    BinaryAttrShape,
    BoundingBox,
    EmotionShape,
    FaceDetail,
    ImageQualityShape,
    PoseShape,
)


def test_facedetail_emits_attributes_pascalcase():
    fd = FaceDetail(
        bounding_box=BoundingBox(width=0.1, height=0.1, left=0.1, top=0.1),
        confidence=99.0,
        pose=PoseShape(roll=1.0, yaw=2.0, pitch=3.0),
        quality=ImageQualityShape(brightness=50.0, sharpness=20.0),
        emotions=[EmotionShape(type="HAPPY", confidence=99.0)],
        smile=BinaryAttrShape(value=True, confidence=99.0),
    )
    d = fd.dump()
    assert d["Pose"] == {"Roll": 1.0, "Yaw": 2.0, "Pitch": 3.0}
    assert d["Quality"] == {"Brightness": 50.0, "Sharpness": 20.0}
    assert d["Emotions"][0] == {"Type": "HAPPY", "Confidence": 99.0}
    assert d["Smile"] == {"Value": True, "Confidence": 99.0}


def test_facedetail_omits_unset_attributes():
    fd = FaceDetail(
        bounding_box=BoundingBox(width=0.1, height=0.1, left=0.1, top=0.1),
        confidence=99.0,
    )
    d = fd.dump()
    assert "Pose" not in d and "Emotions" not in d and "AgeRange" not in d
