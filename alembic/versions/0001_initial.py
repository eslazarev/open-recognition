"""initial schema: pgvector extension, collection + face tables, HNSW index

Revision ID: 0001
Revises:
Create Date: 2026-05-29
"""
from __future__ import annotations

from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS collection (
            collection_id      TEXT        PRIMARY KEY,
            face_model_version TEXT        NOT NULL,
            created_at         TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS face (
            face_id            UUID        PRIMARY KEY,
            collection_id      TEXT        NOT NULL
                                           REFERENCES collection(collection_id) ON DELETE CASCADE,
            external_image_id  TEXT,
            image_id           UUID        NOT NULL,
            bbox               JSONB       NOT NULL,
            confidence         REAL        NOT NULL,
            embedding          vector(128) NOT NULL,
            created_at         TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS face_collection_idx ON face(collection_id)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS face_embedding_hnsw "
        "ON face USING hnsw (embedding vector_cosine_ops)"
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS face")
    op.execute("DROP TABLE IF EXISTS collection")
    # `vector` extension is intentionally left in place; other schemas may use it.
