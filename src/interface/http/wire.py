"""AWS JSON-1.1 wire layer: single POST / endpoint dispatched by X-Amz-Target."""

from __future__ import annotations

import json
import logging
from typing import Any, Awaitable, Callable

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from domain.errors import (
    DomainError,
    InvalidParameterValueError,
    UnknownOperationError,
)
from interface.http.operations import (
    compare_faces,
    create_collection,
    delete_collection,
    delete_faces,
    describe_collection,
    detect_faces,
    index_faces,
    list_collections,
    list_faces,
    search_faces_by_image,
)

log = logging.getLogger(__name__)

Handler = Callable[[Request, dict[str, Any]], Awaitable[dict[str, Any]]]

DISPATCH: dict[str, Handler] = {
    "DetectFaces": detect_faces.handle,
    "CompareFaces": compare_faces.handle,
    "CreateCollection": create_collection.handle,
    "DeleteCollection": delete_collection.handle,
    "DescribeCollection": describe_collection.handle,
    "ListCollections": list_collections.handle,
    "IndexFaces": index_faces.handle,
    "ListFaces": list_faces.handle,
    "DeleteFaces": delete_faces.handle,
    "SearchFacesByImage": search_faces_by_image.handle,
}

router = APIRouter()


def _aws_error(exc: DomainError) -> JSONResponse:
    body = {"__type": exc.aws_code, "Message": str(exc)}
    return JSONResponse(
        status_code=exc.http_status,
        content=body,
        headers={"x-amzn-errortype": exc.aws_code},
    )


async def _dispatch(request: Request, op: str, label: str) -> JSONResponse:
    handler = DISPATCH.get(op)
    if handler is None:
        return _aws_error(UnknownOperationError(f"Unknown operation: {label!r}"))

    raw = await request.body()
    try:
        payload: dict[str, Any] = json.loads(raw) if raw else {}
    except json.JSONDecodeError as exc:
        return _aws_error(UnknownOperationError(f"Malformed JSON body: {exc}"))

    try:
        result = await handler(request, payload)
    except DomainError as exc:
        return _aws_error(exc)
    except ValidationError as exc:
        # Bad shape / unknown enum / wrong type → AWS InvalidParameterException.
        return _aws_error(InvalidParameterValueError(str(exc)))
    except Exception:  # noqa: BLE001
        log.exception("Unhandled error in %s", op)
        return _aws_error(UnknownOperationError("Internal server error"))

    return JSONResponse(content=result)


@router.post("/")
async def rekognition_entrypoint(request: Request) -> JSONResponse:
    """Canonical AWS JSON-1.1 entrypoint — operation from the X-Amz-Target header."""
    target = request.headers.get("x-amz-target", "")
    op = target.partition(".")[2]
    return await _dispatch(request, op, target)


@router.post("/{action}")
async def rekognition_by_path(request: Request, action: str) -> JSONResponse:
    """Convenience alias `POST /<Action>` (used by Swagger "Try it out").

    The X-Amz-Target header, if present, still wins so boto3-style calls are
    unaffected; otherwise the operation comes from the path.
    """
    target = request.headers.get("x-amz-target", "")
    op = target.partition(".")[2] if target else action.rpartition(".")[2] or action
    return await _dispatch(request, op, target or action)
