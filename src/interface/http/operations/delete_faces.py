from __future__ import annotations

from typing import Any
from uuid import UUID

from fastapi import Request

from application.delete_faces import delete_faces
from domain.errors import InvalidParameterValueError
from interface.http.schemas import DeleteFacesRequest


async def handle(request: Request, payload: dict[str, Any]) -> dict[str, Any]:
    req = DeleteFacesRequest.model_validate(payload)
    try:
        ids = [UUID(s) for s in req.face_ids]
    except ValueError as exc:
        raise InvalidParameterValueError(f"Invalid FaceId in request: {exc}") from exc
    result = await delete_faces(
        collection_id=req.collection_id,
        face_ids=ids,
        collection_repo=request.app.state.collection_repo,
        face_repo=request.app.state.face_repo,
    )
    return {"DeletedFaces": [str(d) for d in result.deleted]}
