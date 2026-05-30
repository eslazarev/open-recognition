from datetime import UTC, datetime

import pytest

from domain.collection import (
    FACE_MODEL_VERSION,
    Collection,
    validate_collection_id,
)
from domain.errors import InvalidParameterValueError


def test_collection_arn_uses_collection_id() -> None:
    c = Collection(
        collection_id="people",
        face_model_version=FACE_MODEL_VERSION,
        created_at=datetime.now(UTC),
    )
    assert c.arn().endswith("/people")


@pytest.mark.parametrize("good", ["people", "team_2024", "a-b.c", "X" * 255])
def test_validate_collection_id_accepts_legal(good: str) -> None:
    assert validate_collection_id(good) == good


@pytest.mark.parametrize("bad", ["", "with space", "слово", "X" * 256, "p/q"])
def test_validate_collection_id_rejects_illegal(bad: str) -> None:
    with pytest.raises(InvalidParameterValueError):
        validate_collection_id(bad)
