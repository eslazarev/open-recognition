from interface.http.attributes import requested_attributes


def test_default_is_pose_quality():
    assert requested_attributes(None) == {"pose", "quality"}
    assert requested_attributes([]) == {"pose", "quality"}
    assert requested_attributes(["DEFAULT"]) == {"pose", "quality"}


def test_all_adds_emotions_smile():
    assert requested_attributes(["ALL"]) == {"pose", "quality", "emotions", "smile"}


def test_explicit_subset():
    assert requested_attributes(["EMOTIONS"]) == {"emotions"}
    assert requested_attributes(["SMILE", "POSE"]) == {"smile", "pose"}


def test_unsupported_names_ignored_fall_back_to_default():
    assert requested_attributes(["GENDER", "AGE_RANGE"]) == {"pose", "quality"}
