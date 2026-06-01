import numpy as np
import pytest

from infrastructure.cv.fer_recognizer import EXPRESSIONS, FERRecognizer


@pytest.fixture(scope="module")
def fer():
    try:
        return FERRecognizer(pool_size=1)
    except Exception as exc:
        pytest.skip(f"FER model unavailable: {exc}")


def test_predict_returns_valid_distribution(fer):
    aligned = np.full((112, 112, 3), 128, dtype=np.uint8)
    probs = fer.predict(aligned)
    assert len(probs) == len(EXPRESSIONS) == 7
    assert abs(sum(probs) - 1.0) < 1e-3
    assert all(0.0 <= p <= 1.0 for p in probs)
