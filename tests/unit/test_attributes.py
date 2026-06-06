from interface.http.attributes import requested_attributes


def test_default_includes_landmarks():
    assert requested_attributes(None) == {"pose", "quality", "landmarks"}
    assert requested_attributes([]) == {"pose", "quality", "landmarks"}
    assert requested_attributes(["DEFAULT"]) == {"pose", "quality", "landmarks"}


def test_all_set():
    assert requested_attributes(["ALL"]) == {
        "pose", "quality", "landmarks", "emotions", "smile", "eyes_open", "mouth_open",
    }


def test_explicit_subset():
    assert requested_attributes(["EMOTIONS"]) == {"emotions"}
    assert requested_attributes(["SMILE", "POSE"]) == {"smile", "pose"}


def test_explicit_eyes_open():
    assert requested_attributes(["EYES_OPEN"]) == {"eyes_open"}


def test_unsupported_names_ignored_fall_back_to_default():
    assert requested_attributes(["GENDER", "AGE_RANGE"]) == {"pose", "quality", "landmarks"}
