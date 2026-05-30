"""Benchmark concurrent CV throughput vs pool size.

For each pool size N we instantiate fresh YuNetDetector / SFaceRecognizer
pools, run K concurrent asyncio workers each doing detect+embed on a
shared image set, and report aggregate throughput. This shows whether
the queue-based CV pool actually unlocks parallelism (it does only if
cv2 releases the GIL during inference — which it does for the C++
backends).

Usage:
  uv run python scripts/cv_bench.py --lfw /tmp/lfw/lfw_funneled --images 200
"""

from __future__ import annotations

import argparse
import asyncio
import random
import statistics
import time
from pathlib import Path

from infrastructure.cv.image_decoder import decode_image
from infrastructure.cv.sface_recognizer import SFaceRecognizer
from infrastructure.cv.yunet_detector import YuNetDetector


async def cv_op(blob: bytes, detector: YuNetDetector, recognizer: SFaceRecognizer) -> None:
    img = decode_image(blob)
    detected = await asyncio.to_thread(detector.detect, img)
    if not detected:
        return
    largest = max(detected, key=lambda d: d.face.bbox.width * d.face.bbox.height)
    await asyncio.to_thread(recognizer.embed, img, largest)


async def run_scenario(
    *,
    pool_size: int,
    workers: int,
    per_worker: int,
    image_blobs: list[bytes],
) -> tuple[float, float]:
    detector = YuNetDetector(pool_size=pool_size)
    recognizer = SFaceRecognizer(pool_size=pool_size)

    # Warm-up: one op per pooled instance, so the next timed run is fair.
    warm = await asyncio.gather(
        *(cv_op(random.choice(image_blobs), detector, recognizer)
          for _ in range(pool_size))
    )
    del warm

    t0 = time.perf_counter()
    tasks = []
    for w in range(workers):
        for _ in range(per_worker):
            tasks.append(cv_op(random.choice(image_blobs), detector, recognizer))
    await asyncio.gather(*tasks)
    wall = time.perf_counter() - t0
    return wall, len(tasks) / wall


async def main(args: argparse.Namespace) -> None:
    paths = sorted(Path(args.lfw).rglob("*.jpg"))
    random.Random(0).shuffle(paths)
    sample = paths[: args.images]
    blobs = [p.read_bytes() for p in sample]
    print(f"loaded {len(blobs)} sample images")

    print(f"\n{'pool':>6} {'workers':>8} {'ops':>6} {'wall(s)':>10} {'fps':>10}")
    print("-" * 50)
    for pool_size in args.pool_sizes:
        for workers in args.workers:
            wall, fps = await run_scenario(
                pool_size=pool_size,
                workers=workers,
                per_worker=args.per_worker,
                image_blobs=blobs,
            )
            print(
                f"{pool_size:>6} {workers:>8} "
                f"{workers * args.per_worker:>6} {wall:>10.2f} {fps:>10.1f}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lfw", required=True)
    parser.add_argument("--images", type=int, default=200,
                        help="how many distinct LFW images to cycle through")
    parser.add_argument("--pool-sizes", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 4, 8, 16])
    parser.add_argument("--per-worker", type=int, default=20,
                        help="ops per worker per scenario")
    asyncio.run(main(parser.parse_args()))
