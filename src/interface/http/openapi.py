"""OpenAPI 3.1 document for the Rekognition Faces API.

The wire protocol is AWS JSON-1.1: a single `POST /` dispatched by the
`X-Amz-Target` header. That shape doesn't describe itself well in OpenAPI
(ten operations, one path), so this builds a navigable document with one
path per action, each pointing at the Pydantic request/response schemas in
`schemas.py`. The server also accepts `POST /<Action>` as an alias, so the
Swagger "Try it out" button works against a live instance.

`build_openapi()` is pure — no running server needed — so it backs both the
live `/openapi.json` and the checked-in `docs/openapi.json`.
"""

from __future__ import annotations

from typing import Any

from pydantic.json_schema import models_json_schema

from interface.http import schemas as S

API_VERSION = "0.1.0"

# action, request model, response model, summary
OPERATIONS: list[tuple[str, type, type, str]] = [
    ("DetectFaces", S.DetectFacesRequest, S.DetectFacesResponse,
     "Detect faces in an image — bounding boxes, confidence, and landmarks."),
    ("CompareFaces", S.CompareFacesRequest, S.CompareFacesResponse,
     "Compare the largest face in the source image against the target image."),
    ("CreateCollection", S.CreateCollectionRequest, S.CreateCollectionResponse,
     "Create a face collection."),
    ("DeleteCollection", S.DeleteCollectionRequest, S.DeleteCollectionResponse,
     "Delete a collection and cascade to its faces."),
    ("DescribeCollection", S.DescribeCollectionRequest, S.DescribeCollectionResponse,
     "Return face count, model version, and creation time for a collection."),
    ("ListCollections", S.ListCollectionsRequest, S.ListCollectionsResponse,
     "List collection IDs, paginated via NextToken."),
    ("IndexFaces", S.IndexFacesRequest, S.IndexFacesResponse,
     "Detect, quality-filter, embed, and store faces from an image."),
    ("ListFaces", S.ListFacesRequest, S.ListFacesResponse,
     "List faces in a collection, paginated via NextToken."),
    ("DeleteFaces", S.DeleteFacesRequest, S.DeleteFacesResponse,
     "Delete faces from a collection by FaceId."),
    ("SearchFacesByImage", S.SearchFacesByImageRequest, S.SearchFacesByImageResponse,
     "Embed the largest face in the image and return the nearest matches."),
]

_INFO_DESCRIPTION = """\
Self-hosted, **boto3-compatible** drop-in for the AWS Rekognition Faces API.

The native protocol is **AWS JSON-1.1**: every call is `POST /` with an
`X-Amz-Target: RekognitionService.<Action>` header and a JSON body. Point
`boto3.client("rekognition", endpoint_url=...)` at the server and your
existing code works unchanged.

For exploration this document also exposes each action as `POST /<Action>`
(e.g. `POST /DetectFaces`), which the server accepts as an alias — that is
what the **Try it out** button uses. Both forms hit the same handlers.

Errors follow AWS: HTTP 4xx/5xx with `{"__type": "...", "Message": "..."}`
and an `x-amzn-errortype` header.
"""

_ERROR_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "__type": {"type": "string", "examples": ["InvalidParameterException"]},
        "Message": {"type": "string"},
    },
    "required": ["__type"],
}


def build_openapi() -> dict[str, Any]:
    models = [(req, "validation") for _, req, _, _ in OPERATIONS]
    models += [(resp, "serialization") for _, _, resp, _ in OPERATIONS]

    key_map, defs = models_json_schema(
        models, by_alias=True, ref_template="#/components/schemas/{model}"
    )
    component_schemas: dict[str, Any] = dict(defs.get("$defs", {}))
    component_schemas["Error"] = _ERROR_SCHEMA

    def ref(model: type, mode: str) -> dict[str, str]:
        return {"$ref": key_map[(model, mode)]["$ref"]}

    paths: dict[str, Any] = {}
    for action, req, resp, summary in OPERATIONS:
        paths[f"/{action}"] = {
            "post": {
                "operationId": action,
                "tags": ["Faces"],
                "summary": summary,
                "description": (
                    f"AWS JSON-1.1 action **{action}**. Canonical call: "
                    f"`POST /` with header "
                    f"`X-Amz-Target: RekognitionService.{action}`."
                ),
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/x-amz-json-1.1": {"schema": ref(req, "validation")},
                        "application/json": {"schema": ref(req, "validation")},
                    },
                },
                "responses": {
                    "200": {
                        "description": "Success",
                        "content": {
                            "application/x-amz-json-1.1": {"schema": ref(resp, "serialization")},
                            "application/json": {"schema": ref(resp, "serialization")},
                        },
                    },
                    "400": {
                        "description": "AWS error (InvalidParameterException, "
                                       "ResourceNotFoundException, …)",
                        "content": {
                            "application/x-amz-json-1.1": {
                                "schema": {"$ref": "#/components/schemas/Error"}
                            }
                        },
                    },
                },
            }
        }

    return {
        "openapi": "3.1.0",
        "info": {
            "title": "open-recognition",
            "version": API_VERSION,
            "description": _INFO_DESCRIPTION,
            "license": {"name": "MIT"},
        },
        "tags": [{"name": "Faces", "description": "AWS Rekognition Faces API operations."}],
        "paths": paths,
        "components": {"schemas": component_schemas},
    }
