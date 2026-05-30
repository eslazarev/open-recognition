"""HTMX + Jinja2 frontend — the "Faces Playground" at /ui.

A thin adapter, not a second implementation: each route turns a posted form
(and any uploaded image → base64) into the same payload dict the AWS wire
layer builds, calls the **existing** operation handler, and renders the
returned AWS-shape dict as an HTML fragment. Business logic stays in
`application/` (reached via `interface/http/operations/`); the JSON wire layer
is untouched.
"""

from __future__ import annotations

import base64
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from domain.errors import DomainError
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

_HERE = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(_HERE / "templates"))

router = APIRouter()

Handler = Callable[[Request, dict[str, Any]], Awaitable[dict[str, Any]]]


async def _b64(file: UploadFile) -> tuple[str, str]:
    """Return (base64, data-uri) for an uploaded image."""
    raw = await file.read()
    b64 = base64.b64encode(raw).decode()
    uri = f"data:{file.content_type or 'image/jpeg'};base64,{b64}"
    return b64, uri


async def _render(
    request: Request,
    handler: Handler,
    payload: dict[str, Any],
    template: str,
    extra: dict[str, Any] | None = None,
) -> HTMLResponse:
    try:
        result = await handler(request, payload)
    except DomainError as exc:
        return templates.TemplateResponse(
            request,
            "fragments/error.html",
            {"type": exc.aws_code, "message": str(exc)},
        )
    ctx: dict[str, Any] = {"r": result}
    if extra:
        ctx.update(extra)
    return templates.TemplateResponse(request, template, ctx)


@router.get("/ui", response_class=HTMLResponse)
async def ui_index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "index.html", {})


# ---------- stateless image ops ----------

@router.post("/ui/detect", response_class=HTMLResponse)
async def ui_detect(request: Request, image: UploadFile = File(...)) -> HTMLResponse:
    b64, uri = await _b64(image)
    return await _render(
        request, detect_faces.handle,
        {"Image": {"Bytes": b64}},
        "fragments/detect.html", {"img": uri},
    )


@router.post("/ui/compare", response_class=HTMLResponse)
async def ui_compare(
    request: Request,
    source: UploadFile = File(...),
    target: UploadFile = File(...),
    threshold: float = Form(0.0),
) -> HTMLResponse:
    sb, su = await _b64(source)
    tb, tu = await _b64(target)
    return await _render(
        request, compare_faces.handle,
        {"SourceImage": {"Bytes": sb}, "TargetImage": {"Bytes": tb},
         "SimilarityThreshold": threshold},
        "fragments/compare.html", {"source_img": su, "target_img": tu},
    )


# ---------- collections ----------

@router.get("/ui/collections", response_class=HTMLResponse)
async def ui_collections(request: Request) -> HTMLResponse:
    return await _render(request, list_collections.handle, {}, "fragments/collections.html")


@router.get("/ui/collections/options", response_class=HTMLResponse)
async def ui_collection_options(request: Request) -> HTMLResponse:
    """<option> list for the collection-picker dropdowns."""
    return await _render(request, list_collections.handle, {}, "fragments/options.html")


@router.post("/ui/collections/create", response_class=HTMLResponse)
async def ui_collection_create(
    request: Request, collection_id: str = Form(...)
) -> HTMLResponse:
    try:
        await create_collection.handle(request, {"CollectionId": collection_id})
    except DomainError as exc:
        return templates.TemplateResponse(
            request, "fragments/error.html",
            {"type": exc.aws_code, "message": str(exc)},
        )
    resp = await _render(request, list_collections.handle, {}, "fragments/collections.html")
    resp.headers["HX-Trigger"] = "collections-changed"  # refresh the dropdowns
    return resp


@router.post("/ui/collections/delete", response_class=HTMLResponse)
async def ui_collection_delete(
    request: Request, collection_id: str = Form(...)
) -> HTMLResponse:
    try:
        await delete_collection.handle(request, {"CollectionId": collection_id})
    except DomainError as exc:
        return templates.TemplateResponse(
            request, "fragments/error.html",
            {"type": exc.aws_code, "message": str(exc)},
        )
    resp = await _render(request, list_collections.handle, {}, "fragments/collections.html")
    resp.headers["HX-Trigger"] = "collections-changed"  # refresh the dropdowns
    return resp


@router.post("/ui/collections/describe", response_class=HTMLResponse)
async def ui_collection_describe(
    request: Request, collection_id: str = Form(...)
) -> HTMLResponse:
    return await _render(
        request, describe_collection.handle,
        {"CollectionId": collection_id},
        "fragments/describe.html", {"collection_id": collection_id},
    )


# ---------- faces in a collection ----------

@router.post("/ui/index", response_class=HTMLResponse)
async def ui_index_faces(
    request: Request,
    collection_id: str = Form(...),
    image: UploadFile = File(...),
    external_image_id: str = Form(""),
    max_faces: int = Form(1),
    quality_filter: str = Form("AUTO"),
) -> HTMLResponse:
    b64, uri = await _b64(image)
    return await _render(
        request, index_faces.handle,
        {"CollectionId": collection_id, "Image": {"Bytes": b64},
         "ExternalImageId": external_image_id or None,
         "MaxFaces": max_faces, "QualityFilter": quality_filter},
        "fragments/index_faces.html", {"img": uri},
    )


@router.post("/ui/faces", response_class=HTMLResponse)
async def ui_faces(request: Request, collection_id: str = Form(...)) -> HTMLResponse:
    return await _render(
        request, list_faces.handle,
        {"CollectionId": collection_id, "MaxResults": 500},
        "fragments/faces.html", {"collection_id": collection_id},
    )


@router.post("/ui/faces/delete", response_class=HTMLResponse)
async def ui_faces_delete(
    request: Request, collection_id: str = Form(...), face_id: str = Form(...)
) -> HTMLResponse:
    try:
        await delete_faces.handle(
            request, {"CollectionId": collection_id, "FaceIds": [face_id]}
        )
    except DomainError as exc:
        return templates.TemplateResponse(
            request, "fragments/error.html",
            {"type": exc.aws_code, "message": str(exc)},
        )
    return await _render(
        request, list_faces.handle,
        {"CollectionId": collection_id, "MaxResults": 500},
        "fragments/faces.html", {"collection_id": collection_id},
    )


# ---------- search ----------

@router.post("/ui/search", response_class=HTMLResponse)
async def ui_search(
    request: Request,
    collection_id: str = Form(...),
    image: UploadFile = File(...),
    threshold: float = Form(80.0),
    max_faces: int = Form(5),
) -> HTMLResponse:
    b64, uri = await _b64(image)
    return await _render(
        request, search_faces_by_image.handle,
        {"CollectionId": collection_id, "Image": {"Bytes": b64},
         "FaceMatchThreshold": threshold, "MaxFaces": max_faces},
        "fragments/search.html", {"img": uri},
    )
