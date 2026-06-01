"""Pydantic models for the AWS Rekognition wire shapes (PascalCase)."""

from __future__ import annotations

from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field

from domain.quality import QualityFilter


class AwsModel(BaseModel):
    """Base model: accept PascalCase aliases on input, emit them on output."""

    model_config = ConfigDict(
        populate_by_name=True,
        extra="ignore",
        alias_generator=lambda name: "".join(p[:1].upper() + p[1:] for p in name.split("_")),
    )

    def dump(self) -> dict[str, Any]:
        return self.model_dump(by_alias=True, exclude_none=True, mode="json")


# ---------- shared shapes ----------

class S3Object(AwsModel):
    bucket: str | None = None
    name: str | None = None
    version: str | None = None


class Image(AwsModel):
    bytes_: Annotated[str | None, Field(alias="Bytes")] = None
    s3_object: S3Object | None = None


class BoundingBox(AwsModel):
    width: float
    height: float
    left: float
    top: float


class Landmark(AwsModel):
    type: str = Field(alias="Type")
    x: float = Field(alias="X")
    y: float = Field(alias="Y")


class PoseShape(AwsModel):
    roll: float = Field(alias="Roll")
    yaw: float = Field(alias="Yaw")
    pitch: float = Field(alias="Pitch")


class ImageQualityShape(AwsModel):
    brightness: float
    sharpness: float


class EmotionShape(AwsModel):
    type: str = Field(alias="Type")
    confidence: float


class BinaryAttrShape(AwsModel):
    value: bool = Field(alias="Value")
    confidence: float


class AgeRangeShape(AwsModel):
    low: int = Field(alias="Low")
    high: int = Field(alias="High")


class GenderShape(AwsModel):
    value: str = Field(alias="Value")
    confidence: float


class FaceDetail(AwsModel):
    bounding_box: BoundingBox
    confidence: float
    landmarks: list[Landmark] = []
    # populated when requested:
    pose: PoseShape | None = None
    quality: ImageQualityShape | None = None
    emotions: list[EmotionShape] | None = None
    smile: BinaryAttrShape | None = None
    # schema-present for AWS shape parity, never populated (no permissive model):
    age_range: AgeRangeShape | None = None
    gender: GenderShape | None = None
    eyeglasses: BinaryAttrShape | None = None
    sunglasses: BinaryAttrShape | None = None
    beard: BinaryAttrShape | None = None
    mustache: BinaryAttrShape | None = None
    eyes_open: BinaryAttrShape | None = None
    mouth_open: BinaryAttrShape | None = None


class FaceShape(AwsModel):
    """The lightweight Face shape AWS returns in collection ops."""

    face_id: str
    bounding_box: BoundingBox
    image_id: str
    external_image_id: str | None = None
    confidence: float


# ---------- DetectFaces ----------

class DetectFacesRequest(AwsModel):
    image: Image
    attributes: list[str] | None = None


class DetectFacesResponse(AwsModel):
    face_details: list[FaceDetail]


# ---------- CompareFaces ----------

class CompareFacesRequest(AwsModel):
    source_image: Image
    target_image: Image
    similarity_threshold: float = 80.0


class ComparedSourceImageFace(AwsModel):
    bounding_box: BoundingBox
    confidence: float


class ComparedFace(AwsModel):
    bounding_box: BoundingBox
    confidence: float
    landmarks: list[Landmark] = []


class CompareFacesMatch(AwsModel):
    similarity: float
    face: ComparedFace


class CompareFacesResponse(AwsModel):
    source_image_face: ComparedSourceImageFace
    face_matches: list[CompareFacesMatch] = []
    unmatched_faces: list[ComparedFace] = []


# ---------- CreateCollection ----------

class CreateCollectionRequest(AwsModel):
    collection_id: str


class CreateCollectionResponse(AwsModel):
    collection_arn: str
    face_model_version: str
    status_code: int = 200


# ---------- DeleteCollection ----------

class DeleteCollectionRequest(AwsModel):
    collection_id: str


class DeleteCollectionResponse(AwsModel):
    status_code: int = 200


# ---------- DescribeCollection ----------

class DescribeCollectionRequest(AwsModel):
    collection_id: str


class DescribeCollectionResponse(AwsModel):
    face_count: int
    face_model_version: str
    collection_arn: str
    creation_timestamp: float


# ---------- ListCollections ----------

class ListCollectionsRequest(AwsModel):
    max_results: int = 100
    next_token: str | None = None


class ListCollectionsResponse(AwsModel):
    collection_ids: list[str]
    face_model_versions: list[str]
    next_token: str | None = None


# ---------- IndexFaces ----------

class IndexFacesRequest(AwsModel):
    collection_id: str
    image: Image
    external_image_id: str | None = None
    max_faces: int = 1
    quality_filter: QualityFilter = QualityFilter.AUTO
    detection_attributes: list[str] | None = None


class FaceRecordShape(AwsModel):
    face: FaceShape
    face_detail: FaceDetail


class UnindexedFaceShape(AwsModel):
    reasons: list[str]
    face_detail: FaceDetail


class IndexFacesResponse(AwsModel):
    face_records: list[FaceRecordShape] = []
    unindexed_faces: list[UnindexedFaceShape] = []
    face_model_version: str


# ---------- ListFaces ----------

class ListFacesRequest(AwsModel):
    collection_id: str
    max_results: int = 1000
    next_token: str | None = None


class ListFacesResponse(AwsModel):
    faces: list[FaceShape]
    face_model_version: str
    next_token: str | None = None


# ---------- DeleteFaces ----------

class DeleteFacesRequest(AwsModel):
    collection_id: str
    face_ids: list[str]


class DeleteFacesResponse(AwsModel):
    deleted_faces: list[str]


# ---------- SearchFacesByImage ----------

class SearchFacesByImageRequest(AwsModel):
    collection_id: str
    image: Image
    max_faces: int = 5
    face_match_threshold: float = 80.0
    quality_filter: QualityFilter = QualityFilter.AUTO


class FaceMatchShape(AwsModel):
    similarity: float
    face: FaceShape


class SearchFacesByImageResponse(AwsModel):
    searched_face_bounding_box: BoundingBox
    searched_face_confidence: float
    face_matches: list[FaceMatchShape] = []
    face_model_version: str
