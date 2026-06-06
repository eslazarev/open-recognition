"""End-to-end test driving the service through the real boto3 client.

The server must be running on http://127.0.0.1:8080 and Postgres must
be reachable. Both are normally started via:

    docker compose up -d
    uv run uvicorn interface.http.app:app --host 127.0.0.1 --port 8080
"""

from __future__ import annotations

import uuid
from pathlib import Path

import boto3
import pytest

ENDPOINT = "http://127.0.0.1:8080"
FIXTURE = Path(__file__).parent.parent / "fixtures" / "face_a.jpg"


@pytest.fixture(scope="module")
def client():
    return boto3.client(
        "rekognition",
        endpoint_url=ENDPOINT,
        region_name="us-east-1",
        aws_access_key_id="test",
        aws_secret_access_key="test",
    )


@pytest.fixture(scope="module")
def image_bytes() -> bytes:
    return FIXTURE.read_bytes()


@pytest.fixture
def collection_id(client):
    cid = f"e2e-{uuid.uuid4().hex[:8]}"
    client.create_collection(CollectionId=cid)
    yield cid
    try:
        client.delete_collection(CollectionId=cid)
    except client.exceptions.ResourceNotFoundException:
        pass


def test_detect_faces_returns_face_details(client, image_bytes):
    resp = client.detect_faces(Image={"Bytes": image_bytes})
    assert len(resp["FaceDetails"]) >= 1
    fd = resp["FaceDetails"][0]
    assert 0.0 < fd["BoundingBox"]["Width"] < 1.0
    assert fd["Confidence"] > 50.0
    landmark_types = {lm["Type"] for lm in fd["Landmarks"]}
    assert {"eyeLeft", "eyeRight", "nose"}.issubset(landmark_types)


def test_compare_faces_self_is_high_similarity(client, image_bytes):
    resp = client.compare_faces(
        SourceImage={"Bytes": image_bytes},
        TargetImage={"Bytes": image_bytes},
        SimilarityThreshold=80.0,
    )
    assert len(resp["FaceMatches"]) == 1
    assert resp["FaceMatches"][0]["Similarity"] > 99.0
    assert resp["UnmatchedFaces"] == []


def test_collection_lifecycle(client, collection_id):
    desc = client.describe_collection(CollectionId=collection_id)
    assert desc["FaceCount"] == 0
    assert desc["FaceModelVersion"] == "sface-2021dec-1"
    assert collection_id in desc["CollectionARN"]

    listing = client.list_collections(MaxResults=1000)
    assert collection_id in listing["CollectionIds"]


def test_index_then_search_finds_same_face(client, collection_id, image_bytes):
    idx = client.index_faces(
        CollectionId=collection_id,
        Image={"Bytes": image_bytes},
        ExternalImageId="lena",
        MaxFaces=1,
    )
    assert len(idx["FaceRecords"]) == 1
    face_id = idx["FaceRecords"][0]["Face"]["FaceId"]

    desc = client.describe_collection(CollectionId=collection_id)
    assert desc["FaceCount"] == 1

    listed = client.list_faces(CollectionId=collection_id)
    assert any(f["FaceId"] == face_id for f in listed["Faces"])

    search = client.search_faces_by_image(
        CollectionId=collection_id,
        Image={"Bytes": image_bytes},
        FaceMatchThreshold=80.0,
        MaxFaces=5,
    )
    assert any(
        m["Face"]["FaceId"] == face_id and m["Similarity"] > 99.0
        for m in search["FaceMatches"]
    )

    deleted = client.delete_faces(
        CollectionId=collection_id, FaceIds=[face_id]
    )
    assert deleted["DeletedFaces"] == [face_id]
    assert client.describe_collection(CollectionId=collection_id)["FaceCount"] == 0


def test_search_in_empty_collection_returns_no_matches(
    client, collection_id, image_bytes
):
    resp = client.search_faces_by_image(
        CollectionId=collection_id,
        Image={"Bytes": image_bytes},
        FaceMatchThreshold=80.0,
    )
    assert resp["FaceMatches"] == []


def test_describe_nonexistent_collection_raises(client):
    with pytest.raises(client.exceptions.ResourceNotFoundException):
        client.describe_collection(CollectionId="does-not-exist-xyz")


def test_unknown_operation_returns_aws_error(client, image_bytes):
    with pytest.raises(client.exceptions.InvalidParameterException):
        client.compare_faces(
            SourceImage={"Bytes": image_bytes},
            TargetImage={"Bytes": image_bytes},
            SimilarityThreshold=150.0,
        )


def test_quality_filter_medium_accepts_lena(
    client, collection_id, image_bytes
):
    # Lena (YuNet confidence ≈ 91) — passes MEDIUM (≥85) bar.
    resp = client.index_faces(
        CollectionId=collection_id,
        Image={"Bytes": image_bytes},
        ExternalImageId="lena",
        MaxFaces=1,
        QualityFilter="MEDIUM",
    )
    assert len(resp["FaceRecords"]) == 1
    assert resp["UnindexedFaces"] == []


def test_quality_filter_high_rejects_lena_as_low_confidence(
    client, collection_id, image_bytes
):
    # Same face fails HIGH (≥95 confidence) — proves the filter actually fires.
    resp = client.index_faces(
        CollectionId=collection_id,
        Image={"Bytes": image_bytes},
        ExternalImageId="lena",
        MaxFaces=1,
        QualityFilter="HIGH",
    )
    assert resp["FaceRecords"] == []
    assert len(resp["UnindexedFaces"]) == 1
    assert "LOW_CONFIDENCE" in resp["UnindexedFaces"][0]["Reasons"]


def test_invalid_quality_filter_value_is_invalid_parameter(
    client, collection_id, image_bytes
):
    with pytest.raises(client.exceptions.ClientError) as exc_info:
        client.index_faces(
            CollectionId=collection_id,
            Image={"Bytes": image_bytes},
            ExternalImageId="lena",
            MaxFaces=1,
            QualityFilter="EXTREME",  # not in enum — boto3 may not type-check it
        )
    # boto3 may reject it client-side, or the server returns InvalidParameter.
    err = exc_info.value
    if hasattr(err, "response"):
        assert err.response["Error"]["Code"] in (
            "InvalidParameterException",
            "ValidationException",
        )


def test_detect_faces_all_attributes(client, image_bytes):
    r = client.detect_faces(Image={"Bytes": image_bytes}, Attributes=["ALL"])
    fd = r["FaceDetails"][0]
    assert "Pose" in fd and {"Roll", "Yaw", "Pitch"} <= set(fd["Pose"])
    assert "Quality" in fd and {"Brightness", "Sharpness"} <= set(fd["Quality"])
    assert "Emotions" in fd and fd["Emotions"][0]["Type"]
    assert "Smile" in fd


def test_detect_faces_named_landmarks_and_eyes_mouth(client, image_bytes):
    r = client.detect_faces(Image={"Bytes": image_bytes}, Attributes=["ALL"])
    fd = r["FaceDetails"][0]
    types = {lm["Type"] for lm in fd["Landmarks"]}
    assert {"leftPupil", "rightPupil", "chinBottom", "nose"} <= types
    assert len(fd["Landmarks"]) > 5
    assert "EyesOpen" in fd and "Value" in fd["EyesOpen"]
    assert "MouthOpen" in fd and "Value" in fd["MouthOpen"]
