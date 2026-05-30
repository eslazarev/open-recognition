"""LFW-driven stress test for the open-recognition stack.

Phases:
  1. Walk LFW images, embed up to N via the real YuNet+SFace pipeline.
  2. Bulk-insert into pgvector (batched) and measure ingest throughput.
  3. Report DB sizes (table + HNSW index) straight from `pg_total_relation_size`.
  4. Run sequential search latency benchmark, report p50/p95/p99/max.
  5. Run concurrent search stress with W workers × Q queries per worker.

Run with:
  uv run python scripts/stress_test.py \\
      --lfw /tmp/lfw/lfw_funneled --ingest 3000 --queries 200 --workers 10
"""

from __future__ import annotations

import argparse
import asyncio
import random
import statistics
import time
from pathlib import Path
from uuid import uuid4

from application.ports import DetectedFace
from domain.collection import FACE_MODEL_VERSION, Collection
from domain.embedding import Embedding
from domain.face_record import FaceRecord
from domain.similarity import cosine_threshold_from_pct
from infrastructure.cv.image_decoder import decode_image
from infrastructure.cv.sface_recognizer import SFaceRecognizer
from infrastructure.cv.yunet_detector import YuNetDetector
from infrastructure.persistence.collection_repo import PgCollectionRepository
from infrastructure.persistence.db import create_pool, run_migrations
from infrastructure.persistence.face_repo import PgFaceRepository


def percentile(data: list[float], p: float) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    k = (len(s) - 1) * p / 100
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


def fmt_ms(values: list[float]) -> str:
    return (
        f"n={len(values)}  "
        f"p50={percentile(values, 50):.2f}  "
        f"p95={percentile(values, 95):.2f}  "
        f"p99={percentile(values, 99):.2f}  "
        f"max={max(values):.2f} ms"
    )


def embed_image(
    path: Path,
    detector: YuNetDetector,
    recognizer: SFaceRecognizer,
) -> tuple[DetectedFace, Embedding] | None:
    try:
        blob = path.read_bytes()
        img = decode_image(blob)
    except Exception:
        return None
    faces = detector.detect(img)
    if not faces:
        return None
    largest = max(faces, key=lambda d: d.face.bbox.width * d.face.bbox.height)
    return largest, recognizer.embed(img, largest)


async def main(args: argparse.Namespace) -> None:
    lfw = Path(args.lfw)
    paths = sorted(lfw.rglob("*.jpg"))
    print(f"discovered {len(paths)} LFW jpgs")

    random.Random(42).shuffle(paths)
    ingest_paths = paths[: args.ingest]
    query_paths = paths[args.ingest : args.ingest + args.queries]

    await asyncio.to_thread(run_migrations)
    pool = await create_pool(min_size=args.workers + 4, max_size=args.workers + 4)
    # Pre-warm: drain and re-release every conn so pgvector codec is ready.
    async with pool.acquire() as c:
        await c.fetchval("SELECT 1")
    crepo = PgCollectionRepository(pool=pool)
    frepo = PgFaceRepository(pool=pool)

    collection_id = f"lfw-{uuid4().hex[:8]}"
    from datetime import UTC, datetime
    await crepo.create(
        Collection(
            collection_id=collection_id,
            face_model_version=FACE_MODEL_VERSION,
            created_at=datetime.now(UTC),
        )
    )
    print(f"created collection {collection_id}")

    print(f"\n== Phase 1: loading CV models ==")
    t0 = time.perf_counter()
    detector = YuNetDetector()
    recognizer = SFaceRecognizer()
    print(f"  models ready in {time.perf_counter() - t0:.2f}s")

    print(f"\n== Phase 2: embed + ingest {args.ingest} faces (batched) ==")
    t_embed_total = 0.0
    t_db_total = 0.0
    skipped = 0
    inserted = 0
    batch: list[FaceRecord] = []
    BATCH = 200
    t_start = time.perf_counter()
    for i, path in enumerate(ingest_paths, 1):
        te = time.perf_counter()
        result = embed_image(path, detector, recognizer)
        t_embed_total += time.perf_counter() - te
        if result is None:
            skipped += 1
            continue
        detected, embedding = result
        batch.append(
            FaceRecord(
                face_id=uuid4(),
                collection_id=collection_id,
                image_id=uuid4(),
                face=detected.face,
                embedding=embedding,
                external_image_id=path.stem,
            )
        )
        if len(batch) >= BATCH:
            td = time.perf_counter()
            await frepo.add_many(batch)
            t_db_total += time.perf_counter() - td
            inserted += len(batch)
            batch = []
            if i % 500 == 0:
                rate = inserted / (time.perf_counter() - t_start)
                print(f"  {i:>5d}/{args.ingest} processed | inserted={inserted} "
                      f"skipped={skipped} | overall {rate:.1f} faces/s")

    if batch:
        td = time.perf_counter()
        await frepo.add_many(batch)
        t_db_total += time.perf_counter() - td
        inserted += len(batch)

    wall = time.perf_counter() - t_start
    print(
        f"  done: inserted={inserted}, skipped={skipped}, wall={wall:.1f}s\n"
        f"  embed time: {t_embed_total:.1f}s ({t_embed_total/max(inserted,1)*1000:.1f} ms/face)\n"
        f"  db insert time: {t_db_total:.2f}s ({t_db_total/max(inserted,1)*1000:.2f} ms/face)\n"
        f"  overall rate: {inserted/wall:.1f} faces/sec"
    )

    print(f"\n== Phase 3: DB sizes ==")
    async with pool.acquire() as conn:
        count = await conn.fetchval(
            "SELECT count(*) FROM face WHERE collection_id = $1", collection_id
        )
        table_size = await conn.fetchval(
            "SELECT pg_size_pretty(pg_total_relation_size('face'))"
        )
        index_sizes = await conn.fetch(
            """
            SELECT indexrelname, pg_size_pretty(pg_relation_size(indexrelid)) AS size
            FROM pg_stat_user_indexes WHERE relname = 'face'
            """
        )
    print(f"  faces in collection: {count}")
    print(f"  face table total size: {table_size}")
    for row in index_sizes:
        print(f"  index {row['indexrelname']}: {row['size']}")

    print(f"\n== Phase 4: sequential search latency ({len(query_paths)} queries) ==")
    embed_lat: list[float] = []
    search_lat: list[float] = []
    cos_threshold = cosine_threshold_from_pct(80.0)
    hits = 0
    for path in query_paths:
        te = time.perf_counter()
        result = embed_image(path, detector, recognizer)
        embed_lat.append((time.perf_counter() - te) * 1000)
        if result is None:
            continue
        _, query = result
        ts = time.perf_counter()
        matches = await frepo.search(
            collection_id,
            query=query,
            max_faces=5,
            cosine_threshold=cos_threshold,
        )
        search_lat.append((time.perf_counter() - ts) * 1000)
        if matches:
            hits += 1
    print(f"  embed (YuNet+SFace):    {fmt_ms(embed_lat)}")
    print(f"  pgvector HNSW search:   {fmt_ms(search_lat)}")
    print(f"  queries that hit ≥1 match @ threshold 80: {hits}/{len(search_lat)}")

    print(f"\n== Phase 5: concurrent search stress ({args.workers} workers × {args.qper} queries) ==")
    # Pre-embed query vectors once; we measure DB only here.
    queries = []
    for path in query_paths[: args.qper]:
        r = embed_image(path, detector, recognizer)
        if r is not None:
            queries.append(r[1])
    if len(queries) < 10:
        print("  not enough query vectors, skipping")
    else:
        async def worker(worker_id: int) -> list[float]:
            latencies = []
            for q in random.Random(worker_id).sample(queries, k=args.qper):
                ts = time.perf_counter()
                await frepo.search(
                    collection_id, query=q, max_faces=5,
                    cosine_threshold=cos_threshold,
                )
                latencies.append((time.perf_counter() - ts) * 1000)
            return latencies

        t0 = time.perf_counter()
        results = await asyncio.gather(*(worker(i) for i in range(args.workers)))
        wall = time.perf_counter() - t0
        flat = [v for sub in results for v in sub]
        total_q = len(flat)
        print(f"  total queries: {total_q} in {wall:.2f}s "
              f"({total_q/wall:.0f} q/s aggregate)")
        print(f"  latency:                {fmt_ms(flat)}")

    print(f"\n== Cleanup ==")
    await crepo.delete(collection_id)
    print(f"  dropped collection {collection_id}")
    await pool.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lfw", required=True)
    parser.add_argument("--ingest", type=int, default=3000)
    parser.add_argument("--queries", type=int, default=200)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--qper", type=int, default=20)
    asyncio.run(main(parser.parse_args()))
