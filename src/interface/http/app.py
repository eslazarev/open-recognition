"""FastAPI application factory + lifespan-managed resources."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI

from infrastructure.cv.sface_recognizer import SFaceRecognizer
from infrastructure.cv.yunet_detector import YuNetDetector
from infrastructure.persistence.collection_repo import PgCollectionRepository
from infrastructure.persistence.db import create_pool, run_migrations
from infrastructure.persistence.face_repo import PgFaceRepository
from interface.http.openapi import build_openapi
from interface.http.wire import router


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    await asyncio.to_thread(run_migrations)
    pool = await create_pool()
    detector, recognizer = await asyncio.gather(
        asyncio.to_thread(YuNetDetector),
        asyncio.to_thread(SFaceRecognizer),
    )
    app.state.pool = pool
    app.state.detector = detector
    app.state.recognizer = recognizer
    app.state.collection_repo = PgCollectionRepository(pool=pool)
    app.state.face_repo = PgFaceRepository(pool=pool)
    try:
        yield
    finally:
        await pool.close()


def create_app() -> FastAPI:
    app = FastAPI(
        title="open-recognition",
        description="Self-hosted, boto3-compatible AWS Rekognition Faces API.",
        lifespan=lifespan,
    )
    app.include_router(router)
    # Serve our hand-built spec (the AWS JSON-1.1 protocol doesn't describe
    # itself via FastAPI's route introspection) at /openapi.json and /docs.
    app.openapi = lambda: build_openapi()  # type: ignore[method-assign]
    return app


app = create_app()
