# Operations

All ten operations are exposed under `RekognitionService.<Name>` and follow the
AWS wire shape exactly — request fields, response keys, error codes. For the
full request/response schemas, use the interactive [API reference](api.md).

| Operation | Stateful? | What it does |
|---|---|---|
| `DetectFaces` | no | YuNet → `FaceDetails[]` with `BoundingBox`, `Confidence`, `Landmarks` |
| `CompareFaces` | no | Detect+embed both images, return matches above `SimilarityThreshold` |
| `CreateCollection` | yes | New row in `collection`, returns `CollectionArn` + `FaceModelVersion` |
| `DescribeCollection` | yes | Aggregates: `FaceCount`, `FaceModelVersion`, `CreationTimestamp` |
| `ListCollections` | yes | Paginated via `NextToken` (opaque base64 offset) |
| `DeleteCollection` | yes | Drops the collection and cascades to its faces |
| `IndexFaces` | yes | Detect → quality filter → embed → insert; returns `FaceRecords[]` and `UnindexedFaces[]` |
| `ListFaces` | yes | Paginated face list, no embeddings in response |
| `DeleteFaces` | yes | Deletes by `FaceIds[]`, returns the ones actually removed |
| `SearchFacesByImage` | yes | Detect+embed query → HNSW cosine top-k filtered by threshold |

## Face attributes

`DetectFaces` honours the `Attributes` parameter:

- **`DEFAULT`** → `BoundingBox`, `Confidence`, `Landmarks`, `Pose`, `Quality`
- **`ALL`** → also adds `Emotions`, `Smile`, `EyesOpen`, `MouthOpen`

What each is computed from:

- **Pose** — Roll/Yaw/Pitch via `solvePnP` (approximate).
- **Quality** — Brightness/Sharpness, heuristic 0–100.
- **Landmarks** — the AWS-named set (~30 types) via MediaPipe Face Mesh
  (`onnxruntime`), validated against real AWS to within ~0.5% for
  eyes/pupils/mouth. If the mesh model is absent, `Landmarks` falls back to
  YuNet's 5 points.
- **Emotions / Smile** — FER MobileFaceNet.
- **EyesOpen / MouthOpen** — eye/mouth aspect ratio.

`IndexFaces` accepts but ignores `DetectionAttributes`; its `FaceDetail` carries
only box/confidence/landmarks.

!!! warning "Not populated"
    `AgeRange`, `Gender`, `Eyeglasses`, `Sunglasses`, `Beard`, `Mustache` are
    **not** filled in — there is no permissively-licensed free model for them.
    The keys exist in the response shape for AWS parity but are omitted from
    responses.

## QualityFilter

`IndexFaces` and `SearchFacesByImage` accept AWS's `QualityFilter` enum to
reject low-quality detections before they're indexed or used for search:

| Filter | `confidence ≥` | `bbox area ≥` | `eye-line roll ≤` |
|---|---|---|---|
| `NONE`   | 0  | 0      | 180° |
| `AUTO`   | 60 | 0.001  | 45°  |
| `LOW`    | 70 | 0.005  | 40°  |
| `MEDIUM` | 85 | 0.01   | 30°  |
| `HIGH`   | 95 | 0.02   | 20°  |

Rejection reasons match the AWS strings: `LOW_CONFIDENCE`, `SMALL_BOUNDING_BOX`,
`EXTREME_POSE`. Brightness/sharpness are not used as filter signals.

## Image source

Only inline `Image.Bytes` (base64) is supported, capped at **5 MB** to match the
AWS Rekognition limit. The `S3Object` source returns `InvalidS3ObjectException`.

## Errors

Errors are HTTP 4xx/5xx with an AWS-shaped body and matching header, so boto3
raises the correct exception class:

```json
{"__type": "InvalidParameterException", "Message": "..."}
```

```
x-amzn-errortype: InvalidParameterException
```

## Not implemented

`DetectLabels`, `DetectText`, `DetectModerationLabels`, `RecognizeCelebrities`,
all **video** operations (`StartFaceDetection`, `StartFaceSearch`, …), SigV4
authentication, and per-call billing/quotas are intentionally out of scope. See
[Security](security.md) for the authentication caveat.
