# Performance

Measured on an Apple Silicon Mac, single uvicorn process, Postgres in a local
Docker container. **Your numbers will vary.** The benchmark scripts
(`scripts/cv_bench.py` and `scripts/stress_test.py`) are in the repo — run them
on your own hardware.

## CV throughput

Concurrent `detect+embed` on 200 LFW images, varying CV pool size and worker
count:

| pool | workers | aggregate fps | vs baseline |
|---|---|---|---|
| 1 | 1  | 77  | 1.0× (baseline, single instance) |
| 1 | 16 | 71  | 0.9× (queues on the lock) |
| 4 | 4  | 194 | 2.5× |
| 4 | 8  | 206 | 2.7× |
| 8 | 4  | 292 | **3.8×** |
| 8 | 16 | 255 | 3.3× (queue overhead exceeds parallelism) |

cv2 releases the GIL during inference, so a pool of distinct instances unlocks
real parallelism. A single locked instance hard-caps you at one core's worth of
throughput regardless of worker count. Set the pool size with
[`OPEN_RECOGNITION_CV_POOL_SIZE`](configuration.md).

## Search latency

Against a collection of 5 000 faces (from LFW), with the HNSW cosine index:

| | p50 | p95 | p99 | max |
|---|---|---|---|---|
| Embed query (YuNet+SFace, CPU) | 15.3 ms | 16.7 | 17.2 | 39.9 |
| pgvector HNSW search | 3.85 ms | 7.0 | 9.3 | 10.1 |

Concurrent stress: 20 workers × 50 search queries = 1 000 queries against
pre-embedded vectors finished in 0.17 s — **5 809 queries/sec aggregate**.

**CV inference is the bottleneck; the DB isn't.**

## Storage cost

At 5 000 indexed faces:

| | size | per face |
|---|---|---|
| `face` table (data + TOAST) | 8.4 MB | 1.7 KB |
| `face_embedding_hnsw` index | 4.0 MB | 0.8 KB |
| `face_pkey` | 280 KB | trivial |
| `face_collection_idx` | 64 KB | trivial |
| **Total** | **~12.7 MB** | **~2.5 KB** |

Extrapolation: 1 million faces ≈ 2.5 GB.
