"""HTTP-layer integration: FastAPI TestClient + real Postgres + real CV.

Unlike tests/e2e/, this never opens a TCP port — TestClient drives
the ASGI app in-process. Useful for exercising wire dispatch,
Pydantic validation, and error envelopes without paying uvicorn
startup cost per test.
"""

from __future__ import annotations

import base64
from collections.abc import Iterator
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

FIXTURE = Path(__file__).parent.parent / "fixtures" / "face_a.jpg"
HEADERS = {"Content-Type": "application/x-amz-json-1.1"}


def _amz(operation: str) -> dict[str, str]:
    return {**HEADERS, "X-Amz-Target": f"RekognitionService.{operation}"}


@pytest.fixture(scope="session")
def http_client(postgres_dsn: str) -> Iterator[TestClient]:
    # Importing inside the fixture ensures the app picks up the
    # OPEN_RECOGNITION_DATABASE_URL set by postgres_dsn (session-scoped).
    from interface.http.app import app

    with TestClient(app) as client:
        yield client


@pytest.fixture(scope="session")
def b64_image() -> str:
    return base64.b64encode(FIXTURE.read_bytes()).decode()


def test_unknown_operation_returns_404_with_aws_envelope(http_client: TestClient) -> None:
    r = http_client.post(
        "/", headers=_amz("DoesNotExist"), content="{}",
    )
    assert r.status_code == 404
    body = r.json()
    assert body["__type"] == "UnknownOperationException"
    assert r.headers["x-amzn-errortype"] == "UnknownOperationException"


def test_invalid_collection_id_returns_400(http_client: TestClient) -> None:
    r = http_client.post(
        "/",
        headers=_amz("CreateCollection"),
        json={"CollectionId": "has space"},
    )
    assert r.status_code == 400
    assert r.json()["__type"] == "InvalidParameterException"


def test_describe_missing_collection_404_shape(http_client: TestClient) -> None:
    r = http_client.post(
        "/",
        headers=_amz("DescribeCollection"),
        json={"CollectionId": "no-such-collection-xyz"},
    )
    # AWS uses ResourceNotFoundException, HTTP 400.
    assert r.status_code == 400
    assert r.json()["__type"] == "ResourceNotFoundException"


def test_invalid_quality_filter_value_returns_400(
    http_client: TestClient, b64_image: str
) -> None:
    # First create a real collection so the request reaches Pydantic
    # validation (otherwise it would 400 on the missing collection).
    cid = f"it-{uuid4().hex[:8]}"
    http_client.post("/", headers=_amz("CreateCollection"), json={"CollectionId": cid})
    try:
        r = http_client.post(
            "/",
            headers=_amz("IndexFaces"),
            json={
                "CollectionId": cid,
                "Image": {"Bytes": b64_image},
                "QualityFilter": "EXTREME",
            },
        )
        assert r.status_code == 400
        assert r.json()["__type"] == "InvalidParameterException"
    finally:
        http_client.post(
            "/", headers=_amz("DeleteCollection"), json={"CollectionId": cid}
        )


def test_full_lifecycle_through_test_client(
    http_client: TestClient, b64_image: str
) -> None:
    cid = f"it-{uuid4().hex[:8]}"
    try:
        # Create
        r = http_client.post(
            "/", headers=_amz("CreateCollection"), json={"CollectionId": cid}
        )
        assert r.status_code == 200
        assert "CollectionArn" in r.json()

        # Index lena
        r = http_client.post(
            "/",
            headers=_amz("IndexFaces"),
            json={
                "CollectionId": cid,
                "Image": {"Bytes": b64_image},
                "ExternalImageId": "lena",
                "MaxFaces": 1,
            },
        )
        assert r.status_code == 200
        body = r.json()
        assert len(body["FaceRecords"]) == 1
        face_id = body["FaceRecords"][0]["Face"]["FaceId"]

        # Search and find herself with near-100% similarity
        r = http_client.post(
            "/",
            headers=_amz("SearchFacesByImage"),
            json={
                "CollectionId": cid,
                "Image": {"Bytes": b64_image},
                "FaceMatchThreshold": 80.0,
            },
        )
        assert r.status_code == 200
        matches = r.json()["FaceMatches"]
        assert any(m["Face"]["FaceId"] == face_id and m["Similarity"] > 99
                   for m in matches)
    finally:
        http_client.post(
            "/", headers=_amz("DeleteCollection"), json={"CollectionId": cid}
        )
