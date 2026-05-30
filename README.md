# open-recognition

**A self-hosted, boto3-compatible drop-in for the AWS Rekognition Faces API.**

You point `boto3.client("rekognition")` at this server with one argument and
keep your existing code. No vendor lock-in, no per-image billing, no images
leaving your network.

<p align="center">
  <img src="docs/img/recognition.png" alt="Each person enrolled with one photo and recognised from a different, unseen photo — every similarity score is live output from the server's own API." width="640">
</p>

<p align="center"><sub>
  Enrol one photo per person, search with a <em>different</em> one — every
  score above is live API output. Faces are freely-licensed
  (<a href="docs/img/faces/CREDITS.md">credits</a>); regenerate with
  <code>scripts/make_readme_figures.py</code>.
</sub></p>

Get it running — Docker is the only prerequisite. One command brings up
Postgres **and** the server:

```bash
git clone https://github.com/eslazarev/open-recognition.git
cd open-recognition

docker compose up -d        # postgres + the API server, listening on :8080
```

The image bundles the models and runs migrations on startup; see
[Quick start](#quick-start) for what happens on first boot and how to run the
server straight from `uv` instead. With it listening on `:8080`, your existing
code only changes by one argument:

```python
client = boto3.client(
    "rekognition",
    endpoint_url="http://localhost:8080",   # ← only line that changes
    region_name="us-east-1",
    aws_access_key_id="x", aws_secret_access_key="x",  # ignored by the server
)
client.create_collection(CollectionId="team")
client.index_faces(CollectionId="team", Image={"Bytes": jpeg}, ExternalImageId="alice")
client.search_faces_by_image(CollectionId="team", Image={"Bytes": jpeg})
```

No SDK? It's just AWS JSON-1.1 over HTTP — `POST /` with an `X-Amz-Target`
header and a JSON body. The image goes in as base64 under `Image.Bytes`:

```bash
IMG=$(base64 -i alice.jpg)          # macOS; on Linux: base64 -w0 alice.jpg

curl -s http://localhost:8080/ \
  -H 'X-Amz-Target: RekognitionService.DetectFaces' \
  -H 'Content-Type: application/x-amz-json-1.1' \
  -d "{\"Image\": {\"Bytes\": \"$IMG\"}}"
# {"FaceDetails":[{"BoundingBox":{...},"Confidence":99.5,"Landmarks":[...]}]}
```

Prefer clicking to curling? A **Faces Playground** lives at **`/ui`** — upload
an image and see detected faces with boxes and landmarks, create collections,
index, search, and compare, all in the browser (HTMX, no build step). The raw
API reference is the Swagger UI at **`/docs`**, with the spec at `/openapi.json`
— also checked in at [`docs/openapi.json`](docs/openapi.json). For exploration
the server accepts `POST /<Action>` (e.g. `POST /DetectFaces`) as an alias,
which is what Swagger's "Try it out" uses; boto3 keeps using the canonical
`POST /` + header.

Behind the scenes it's [YuNet] for detection, [SFace] for 128-d embeddings,
and pgvector's HNSW index for sub-millisecond search. Embeddings, not images,
are stored.

[YuNet]: https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet
[SFace]: https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface

---

## Table of contents

- [What you get](#what-you-get)
- [Quick start](#quick-start)
- [Try it on real faces](#try-it-on-real-faces)
- [Web playground](#web-playground)
- [Supported operations](#supported-operations)
- [How a request actually flows](#how-a-request-actually-flows)
- [Architecture](#architecture)
  - [Layered structure (DDD)](#layered-structure-ddd)
  - [The AWS wire protocol](#the-aws-wire-protocol)
  - [Why this stack](#why-this-stack)
- [Configuration](#configuration)
- [QualityFilter](#qualityfilter)
- [Performance](#performance)
  - [CV throughput](#cv-throughput)
  - [Search latency](#search-latency)
  - [Storage cost](#storage-cost)
- [What's not implemented](#whats-not-implemented)
- [Development](#development)
  - [Running tests](#running-tests)
  - [Stress benchmark](#stress-benchmark)
  - [Database migrations](#database-migrations)
- [Security notes](#security-notes)
- [Models and credits](#models-and-credits)
- [License](#license)

---

## What you get

| | open-recognition | AWS Rekognition Faces API |
|---|---|---|
| **SDK** | `boto3.client("rekognition", endpoint_url=…)` | `boto3.client("rekognition")` |
| **Wire protocol** | AWS JSON-1.1 (identical) | AWS JSON-1.1 |
| **Operations** | 10 Faces API operations | All Faces + Labels + Moderation + … |
| **Detector** | YuNet (`face_detection_yunet_2023mar.onnx`, 232 KB) | proprietary |
| **Recognizer** | SFace (`face_recognition_sface_2021dec.onnx`, 38 MB, 128-d) | proprietary |
| **Vector store** | Postgres + pgvector HNSW (`vector_cosine_ops`) | proprietary |
| **Throughput (one process)** | ~290 detect+embed/sec at pool=8, 5 800 search/sec | unbounded, you pay per call |
| **Search latency p95** | 4 ms (5 000 faces, local PG) | tens of ms over network |
| **Cost per million faces** | ~$0 (electricity) + 2.5 GB disk | $1 per 1 000 IndexFaces (~$1 000) + storage |
| **Where your photos go** | your Postgres | AWS |
| **What it can't do** | quality attributes (age, emotion, pose angles), celebrity, moderation, video | all of those |

If you only use the Faces API and want to stop paying per image, this is a
straight swap. If you need DetectLabels, age estimation, or video — keep using
real Rekognition.

---

## Quick start

You need Docker. `docker compose up` builds the server image and starts it
alongside Postgres:

```bash
git clone https://github.com/eslazarev/open-recognition.git
cd open-recognition

docker compose up -d        # postgres + the API server on :8080
```

Two services come up: `postgres` (`pgvector/pgvector:pg16`) and `app` (the
server, built from the `Dockerfile`). The image bundles YuNet (~232 KB) and
SFace (~38 MB), verified against the SHA256 values pinned in `model_loader.py`.
Migrations (`alembic upgrade head`) run automatically in the FastAPI lifespan,
so the schema is ready on first request. Tail the logs with
`docker compose logs -f app`; stop everything with `docker compose down`.

**Running the server from source instead.** For development — live reload, no
rebuild — run Postgres in Docker but the server from `uv`:

```bash
docker compose up -d postgres                       # just the database
uv sync --extra dev                                 # Python 3.12 + deps
uv run uvicorn interface.http.app:app --port 8080
```

Run from source and the models download into `models/` on first request (the
Docker image bakes them in instead). Either way migrations run on startup.

Then, in another terminal:

```python
import base64, boto3

client = boto3.client(
    "rekognition",
    endpoint_url="http://localhost:8080",
    region_name="us-east-1",
    aws_access_key_id="x",
    aws_secret_access_key="x",
)

with open("alice.jpg", "rb") as f:
    photo = f.read()

client.create_collection(CollectionId="team")
client.index_faces(
    CollectionId="team",
    Image={"Bytes": photo},
    ExternalImageId="alice",
    MaxFaces=1,
)

result = client.search_faces_by_image(
    CollectionId="team",
    Image={"Bytes": photo},
    FaceMatchThreshold=80.0,
)
print(result["FaceMatches"][0]["Similarity"])   # → 99.9999...
```

The same `boto3` code points at real AWS by removing `endpoint_url`.

---

## Try it on real faces

`scripts/demo_real_faces.py` drives the whole stack through the boto3 SDK on
real photographs. It enrols **one** photo of each person, then searches with a
**different, previously-unseen** photo of the same people — the honest test of
recognition, not a self-compare — and throws in a stranger who was never
enrolled to check for false positives.

<p align="center">
  <img src="docs/img/detection.png" alt="DetectFaces output drawn on a real portrait: YuNet bounding box, confidence, and five facial landmarks." width="360">
</p>

Detection comes first: YuNet returns the bounding box, a confidence score, and
five landmarks (eyes, nose, mouth) that SFace uses to align the crop before
embedding. The box and dots above are drawn straight from a `DetectFaces`
response.

It uses [LFW] (Labeled Faces in the Wild), a public set of ~13 000 labelled
face photos. Grab it once (~243 MB, no auth):

```bash
mkdir -p /tmp/lfw && cd /tmp/lfw
curl -sLO https://ndownloader.figshare.com/files/5976015
mv 5976015 lfw.tgz && tar xzf lfw.tgz
```

Then, with the server running ([Quick start](#quick-start)):

```bash
cd ~/repos/open-recognition
uv run python scripts/demo_real_faces.py --lfw /tmp/lfw/lfw_funneled
```

Output on the bundled defaults (five world leaders + one stranger):

```
=== INDEXING (1 photo per person) ===
  George_W_Bush          indexed  conf= 94.1  George_W_Bush_0001.jpg
  Colin_Powell           indexed  conf= 92.1  Colin_Powell_0001.jpg
  Tony_Blair             indexed  conf= 94.1  Tony_Blair_0001.jpg
  Donald_Rumsfeld        indexed  conf= 93.5  Donald_Rumsfeld_0001.jpg
  Gerhard_Schroeder      indexed  conf= 94.3  Gerhard_Schroeder_0001.jpg

Collection now holds 5 faces (model sface-2021dec-1)

=== SEARCH (different, previously-unseen photo of each person) ===
  [match] George_W_Bush          -> George_W_Bush          sim=100.0
  [match] Colin_Powell           -> Colin_Powell           sim=100.0
  [match] Tony_Blair             -> Tony_Blair             sim=100.0
  [match] Donald_Rumsfeld        -> Donald_Rumsfeld        sim=100.0
  [match] Gerhard_Schroeder      -> Gerhard_Schroeder      sim=100.0

=== STRANGER (not enrolled — should NOT match) ===
  [ok] Hugo_Chavez -> no match (correct: not enrolled)

=== COMPARE FACES ===
  George_W_Bush vs George_W_Bush (diff photos):  100.0%  (expect high)
  George_W_Bush vs Colin_Powell (diff people):    0.0%  (expect low)

=== RESULT: 5/5 people recognised from unseen photos ===
```

Each person is recognised from a photo the server never indexed, the stranger
is correctly rejected, and a same-person CompareFaces saturates near 100% while
two different people sit near 0% — the similarity curve collapses genuine
matches to the top of the scale and pushes non-matches to the bottom. The
script exits non-zero if any enrolled person fails to match, so it doubles as a
smoke test.

### Telling people apart

The flip side of recognition is **not** matching the wrong person.
`CompareFaces` between two different faces scores far below the `80`
threshold, so they're never confused:

```python
def compare(a, b):
    r = client.compare_faces(
        SourceImage={"Bytes": open(a, "rb").read()},
        TargetImage={"Bytes": open(b, "rb").read()},
        SimilarityThreshold=0.0,          # 0 → always return the raw score
    )
    m = r["FaceMatches"]
    return m[0]["Similarity"] if m else 0.0

compare("biden.jpg", "merkel.jpg")     # → 0.3   (different people)
compare("merkel.jpg", "trudeau.jpg")   # → 0.4
compare("biden.jpg", "trudeau.jpg")    # → 26.5  (closest pair — still rejected)
compare("biden.jpg", "biden2.jpg")     # → 100.0 (same person, different photo)
```

<p align="center">
  <img src="docs/img/comparison.png" alt="CompareFaces on three pairs of different people — all score well below the 80% match threshold." width="600">
</p>

<p align="center"><sub>
  Different people, scored live by <code>CompareFaces</code>. Even the closest
  pair stays well under the 80% threshold, so no false match. Faces are
  freely-licensed (<a href="docs/img/faces/CREDITS.md">credits</a>).
</sub></p>

Useful flags:

```bash
# Point at a server on a different port
uv run python scripts/demo_real_faces.py --lfw /tmp/lfw/lfw_funneled \
    --endpoint http://127.0.0.1:8090

# Pick your own people (any LFW folder names with >=2 photos) and stranger
uv run python scripts/demo_real_faces.py --lfw /tmp/lfw/lfw_funneled \
    --people Serena_Williams Vladimir_Putin Jennifer_Aniston \
    --stranger Roh_Moo-hyun --threshold 90
```

`--people` takes any LFW folder names (run `ls /tmp/lfw/lfw_funneled` to see
them; pick ones with at least two photos), and `--threshold` is the AWS-style
`FaceMatchThreshold`.

[LFW]: https://vis-www.cs.umass.edu/lfw/

---

## Web playground

The server also ships a browser UI at **`/ui`** — a self-contained HTMX
playground (no build step, no JS framework) covering every operation. Upload an
image and see detected faces with boxes and landmarks, manage collections,
index, search, and compare, without writing a line of code. It posts to the
same handlers as the API (via the `POST /<Action>` aliases); the raw reference
stays at `/docs` (Swagger).

<table>
<tr>
<td width="50%"><img src="docs/img/ui/detect.png" alt="Detect tab"><br><sub><b>Detect</b> — box, confidence, and 5 landmarks drawn on the upload</sub></td>
<td width="50%"><img src="docs/img/ui/compare.png" alt="Compare tab"><br><sub><b>Compare</b> — similarity verdict between two faces</sub></td>
</tr>
<tr>
<td><img src="docs/img/ui/search.png" alt="Search tab"><br><sub><b>Search</b> — query image against an enrolled collection</sub></td>
<td><img src="docs/img/ui/faces.png" alt="Faces tab"><br><sub><b>Faces</b> — what's enrolled in a collection</sub></td>
</tr>
<tr>
<td><img src="docs/img/ui/index.png" alt="Index tab"><br><sub><b>Index</b> — enrol a face under a label</sub></td>
<td><img src="docs/img/ui/collections.png" alt="Collections tab"><br><sub><b>Collections</b> — create, describe, delete</sub></td>
</tr>
</table>

---

## Supported operations

All ten operations exposed under `RekognitionService.<Name>` follow the AWS
wire shape exactly — request fields, response keys, error codes.

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

Two parameters that AWS supports and we accept-but-ignore: `DetectionAttributes`
(we don't compute age/gender/emotion) and the `S3Object` image source (we only
read `Image.Bytes` and return `InvalidS3ObjectException` for `S3Object`).

---

## How a request actually flows

A `search_faces_by_image` call hits eight files and crosses three layers.
This is the shape of every request:

```
boto3.client.search_faces_by_image(...)
   │  serialises to JSON-1.1, POST / with X-Amz-Target
   ▼
interface/http/wire.py        ← dispatch table on X-Amz-Target
   ▼
interface/http/operations/    ← thin handler, validates with Pydantic,
  search_faces_by_image.py      pulls deps from app.state
   ▼
application/                  ← use case: pure business logic
  search_faces_by_image.py      domain validation, calls ports
   ▼  ▼  ▼
domain/  infrastructure/cv/   infrastructure/persistence/
similarity.py  yunet+sface     face_repo.py (the only <=> in SQL)
```

Each layer's job:

- **`interface/http`** — knows AWS JSON-1.1. Doesn't know about embeddings or SQL.
- **`application`** — knows the use case. Doesn't know about HTTP or which cv2 class detects faces.
- **`domain`** — pure Python. No I/O, no FastAPI, no cv2, no asyncpg.
- **`infrastructure`** — adapters: cv2, asyncpg, pgvector. Implement the protocols defined in `application/ports.py`.

You can replace pgvector with Qdrant by rewriting `infrastructure/persistence/`
and not touching anything else. We've designed it so you can. You probably
shouldn't unless you have a reason.

---

## Architecture

### Layered structure (DDD)

```
src/
├── domain/                       pure, zero I/O
│   ├── face.py                   Face, BoundingBox, Landmark
│   ├── embedding.py              Embedding (128-d, L2-normalised)
│   ├── face_record.py            FaceRecord aggregate
│   ├── collection.py             Collection aggregate, validate_collection_id()
│   ├── similarity.py             cosine ↔ AWS-style percentage
│   ├── quality.py                QualityFilter, assess_face()
│   └── errors.py                 DomainError → AWS error code mapping
├── application/                  use cases (one per operation)
│   ├── ports.py                  FaceDetector, FaceRecognizer, repositories (Protocol)
│   ├── detect_faces.py
│   ├── compare_faces.py
│   ├── index_faces.py
│   ├── search_faces_by_image.py
│   └── … (5 more, mostly DB-only)
├── infrastructure/
│   ├── cv/
│   │   ├── model_loader.py       checksum-verified download from opencv_zoo
│   │   ├── yunet_detector.py     queue.SimpleQueue of cv2.FaceDetectorYN
│   │   ├── sface_recognizer.py   queue.SimpleQueue of cv2.FaceRecognizerSF
│   │   └── image_decoder.py      base64 + PIL → np.ndarray BGR
│   └── persistence/
│       ├── db.py                 asyncpg pool, run_migrations()
│       ├── collection_repo.py
│       └── face_repo.py          only place with pgvector <=> in SQL
└── interface/http/
    ├── app.py                    FastAPI factory + lifespan
    ├── wire.py                   POST / → X-Amz-Target dispatch, AWS errors
    ├── schemas.py                Pydantic models with PascalCase aliases
    └── operations/               one file per X-Amz-Target action
```

There's one rule we hold to: **each layer can only import from the layer
below it**. The compiler doesn't enforce it but `grep` does — `domain/`
contains zero references to `infrastructure` or `interface`.

### The AWS wire protocol

Rekognition is JSON-1.1, single endpoint, dispatched by header:

```
POST / HTTP/1.1
X-Amz-Target: RekognitionService.DetectFaces
Content-Type: application/x-amz-json-1.1
Authorization: AWS4-HMAC-SHA256 …          ← we ignore this
Content-Length: …

{"Image": {"Bytes": "<base64>"}, "Attributes": ["DEFAULT"]}
```

Successful response is HTTP 200 + JSON body. Errors are HTTP 4xx/5xx with:

```json
{"__type": "InvalidParameterException", "Message": "..."}
```

plus the header `x-amzn-errortype: InvalidParameterException`. boto3 parses
both and raises the right exception class.

`wire.py` is the entire dispatch — a `dict[str, Handler]` keyed by the part
after the dot in `X-Amz-Target`. Adding a new operation is: write the use
case, write the handler, add one line to the dispatch table. We skip SigV4
verification entirely — `Authorization` headers can be anything; boto3
generates them automatically.

### Why this stack

- **YuNet (232 KB)** — accurate small-face detector that runs on CPU in ~5 ms per call. It returns confidence and 5 landmarks, which is what SFace needs for alignment.
- **SFace (38 MB)** — 128-d embedding network, also CPU-only, ~10 ms per face. Documented native threshold cos ≥ 0.363 — we use cosine, mapped to an AWS-style percentage.
- **pgvector + HNSW** — single binary you already know how to back up, index that scales O(log N), `vector_cosine_ops` is built for L2-normalised vectors. We L2-normalise on insert so the cosine operator works directly.
- **FastAPI** — ASGI, good Pydantic integration, doesn't get in the way. We use it as a thin shell over the dispatcher; no Depends, no router magic.
- **asyncpg** — fastest Python Postgres driver. Pairs with `pgvector.asyncpg.register_vector` for native `vector` codec.
- **uv** — fast lockfile-driven Python; one tool for venv + install + run.
- **alembic** — boring schema migrations. The `vector` extension is created in the first migration alongside the tables.

---

## Configuration

Everything is environment variables. There's no config file.

| Variable | Default | What it does |
|---|---|---|
| `OPEN_RECOGNITION_DATABASE_URL` | `postgresql://open_recognition:open_recognition@localhost:5432/open_recognition` | Postgres DSN. asyncpg-style; alembic's env.py rewrites it to `postgresql+psycopg://` internally. |
| `OPEN_RECOGNITION_CV_POOL_SIZE` | `min(4, cpu_count())` | Number of cv2 detector/recognizer instances in each pool. Higher = more parallel inference, ~15 MB extra RAM per slot. |
| `OPEN_RECOGNITION_MODELS_DIR` | `./models` | Where ONNX files live. Auto-created. |
| `OPEN_RECOGNITION_ALEMBIC_INI` | `<project_root>/alembic.ini` | Override only if running alembic from a non-standard location. |

The compose `app` service sets `OPEN_RECOGNITION_DATABASE_URL` to reach the
`postgres` service over the compose network; everything else uses the
defaults above. To run against a managed Postgres, point
`OPEN_RECOGNITION_DATABASE_URL` at it (in the compose file, or in your shell when
running the server from `uv`).

---

## QualityFilter

AWS lets you reject low-quality detections before they're indexed or used
for search. We honour the same parameter with the same enum:

| Filter | `confidence ≥` | `bbox area ≥` | `eye-line roll ≤` |
|---|---|---|---|
| `NONE`   | 0  | 0      | 180° |
| `AUTO`   | 60 | 0.001  | 45°  |
| `LOW`    | 70 | 0.005  | 40°  |
| `MEDIUM` | 85 | 0.01   | 30°  |
| `HIGH`   | 95 | 0.02   | 20°  |

Rejection reasons match AWS strings: `LOW_CONFIDENCE`,
`SMALL_BOUNDING_BOX`, `EXTREME_POSE`. The Lena reference image
(YuNet confidence ≈ 91) passes `MEDIUM` and lands in `UnindexedFaces` with
`LOW_CONFIDENCE` under `HIGH` — that's how you tell the filter is actually
firing.

We don't compute brightness or sharpness — those would need crop-level
pixel analysis. If you hit real recall noise where the existing signals
aren't enough, add them in `domain/quality.py`. The presets are tuned for
YuNet's confidence distribution; if you swap detectors, retune.

---

## Performance

Measured on an Apple Silicon Mac, single uvicorn process, Postgres in a
local Docker container. Your numbers will vary. The benchmark scripts
(`scripts/cv_bench.py` and `scripts/stress_test.py`) are in the repo —
run them on your own hardware.

### CV throughput

Concurrent `detect+embed` on 200 LFW images, varying CV pool size and
worker count:

| pool | workers | aggregate fps | vs baseline |
|---|---|---|---|
| 1 | 1  | 77  | 1.0× (baseline, single instance) |
| 1 | 16 | 71  | 0.9× (queues on the lock) |
| 4 | 4  | 194 | 2.5× |
| 4 | 8  | 206 | 2.7× |
| 8 | 4  | 292 | **3.8×** |
| 8 | 16 | 255 | 3.3× (queue overhead exceeds parallelism) |

The takeaway: cv2 releases the GIL during inference, so a pool of distinct
instances unlocks real parallelism. `threading.Lock` around a single
instance hard-caps you at one core's worth of throughput regardless of
worker count.

### Search latency

Against a collection of 5 000 faces (also from LFW), with HNSW cosine
index:

| | p50 | p95 | p99 | max |
|---|---|---|---|---|
| Embed query (YuNet+SFace, CPU) | 15.3 ms | 16.7 | 17.2 | 39.9 |
| pgvector HNSW search | 3.85 ms | 7.0 | 9.3 | 10.1 |

Concurrent stress: 20 workers × 50 search queries = 1 000 queries against
pre-embedded vectors, finished in 0.17 s — **5 809 queries/sec aggregate**.

CV inference is the bottleneck; the DB isn't.

### Storage cost

At 5 000 indexed faces:

| | size | per face |
|---|---|---|
| `face` table (data + TOAST) | 8.4 MB | 1.7 KB |
| `face_embedding_hnsw` index | 4.0 MB | 0.8 KB |
| `face_pkey` | 280 KB | trivial |
| `face_collection_idx` | 64 KB | trivial |
| **Total** | **~12.7 MB** | **~2.5 KB** |

Extrapolation: 1 million faces ≈ 2.5 GB. HNSW build time at that scale
isn't measured yet.

---

## What's not implemented

Plenty of AWS Rekognition is intentionally out of scope. Don't try to use
us as a full replacement.

- **Face attributes**: no `AgeRange`, `Gender`, `Emotions`, `Eyeglasses`, `Beard`, `Pose`, `Quality.Brightness/Sharpness`. We return `BoundingBox`, `Confidence`, `Landmarks`. `Attributes=["ALL"]` is silently downgraded to default.
- **DetectLabels, DetectText, DetectModerationLabels, RecognizeCelebrities** — these aren't faces, different models.
- **Video** — `StartFaceDetection`, `StartFaceSearch`, etc. Use the image API on extracted frames.
- **S3Object image source** — only `Image.Bytes` is supported. `S3Object` returns `InvalidS3ObjectException`.
- **SigV4 authentication** — the `Authorization` header is ignored. Don't expose this server to the public internet without putting it behind a real authenticator.
- **Multi-tenant isolation** — collection IDs are a flat namespace. If you need per-tenant separation, run separate Postgres schemas or separate instances.
- **Per-call billing / quotas** — there isn't any. Be careful what you point at it.

---

## Development

Python 3.12, managed by `uv`. For the dev loop, run Postgres in Docker but
the server from `uv` so reloads don't need an image rebuild:

```bash
uv sync --extra dev                                # installs everything
docker compose up -d postgres                       # just the database
uv run uvicorn interface.http.app:app   # runs the server
```

### Running tests

There are three layers:

```bash
uv run pytest tests/unit            # 47 tests, ~0.1 s, pure Python + fakes
uv run pytest tests/integration     # 10 tests, ~3 s, real Postgres via testcontainers
uv run pytest tests/e2e             # 9 tests, ~1 s, requires a running uvicorn on :8080
uv run pytest                       # all three, ~5 s
```

The integration tests spin up a fresh `pgvector/pgvector:pg16` container,
apply alembic head, and tear it all down at session end. They auto-skip if
Docker isn't reachable.

The e2e tests use the real `boto3` SDK, pointed at `http://127.0.0.1:8080`.
Start the server first.

### Stress benchmark

End-to-end ingest + search benchmark with real LFW faces:

```bash
# Get LFW (~243 MB, public mirror, no auth needed)
mkdir -p /tmp/lfw && cd /tmp/lfw
curl -sLO https://ndownloader.figshare.com/files/5976015
mv 5976015 lfw.tgz && tar xzf lfw.tgz

cd ~/repos/open-recognition
uv run python scripts/stress_test.py \
    --lfw /tmp/lfw/lfw_funneled \
    --ingest 5000 --queries 300 --workers 20 --qper 50
```

Outputs ingest throughput, DB table+index sizes, sequential search
latency percentiles, and concurrent search throughput. See
[Performance](#performance) for example output.

### Database migrations

Schema lives in `alembic/versions/`. The server runs `alembic upgrade head`
in its lifespan, so during normal use you don't need to touch it. For
manual operations:

```bash
uv run alembic current               # what's applied
uv run alembic history               # all revisions
uv run alembic upgrade head          # apply pending
uv run alembic revision -m "add X"   # new migration template
```

Migrations use `op.execute(...)` with raw SQL — we don't reflect SQLAlchemy
models, because pgvector types and HNSW indexes are easier to express
directly.

---

## Security notes

A few things to know before you let this anywhere near production traffic:

- **No authentication.** SigV4 headers are ignored. Don't put this behind a
  public load balancer; put it behind a real authenticator (a reverse proxy
  with JWT, a service mesh, mTLS, your usual choice).
- **ONNX checksums are pinned.** Both YuNet and SFace are verified against
  hardcoded SHA256 hashes in `model_loader.py` on every load. A mismatch
  deletes the file, downloads once more, and refuses to start if it still
  doesn't match. Upstream `opencv_zoo` would have to be compromised in two
  ways simultaneously to slip a bad model past us, and the maintainer
  would notice the next `git pull`.
- **Image bytes are decoded with Pillow.** Pillow has had its share of
  CVEs; keep your dependencies updated. We cap inline payloads at 5 MB
  (`MAX_BYTES` in `image_decoder.py`) — matches the AWS Rekognition limit.
- **Embeddings are not images.** What's stored in Postgres is a 128-d
  float vector. Without the SFace model you can't reconstruct the face,
  but face embeddings are still personal data under GDPR. Treat the
  database accordingly: backups, retention, deletion-on-request.

---

## Models and credits

This repository includes pre-downloaded model weights in `models/`. Both
are redistributed under permissive licences with attribution preserved:

| Model | File | License | Source |
|---|---|---|---|
| **YuNet** | `face_detection_yunet_2023mar.onnx` | MIT | Wu et al., [opencv_zoo/face_detection_yunet] |
| **SFace** | `face_recognition_sface_2021dec.onnx` | Apache 2.0 | Zhong et al. (NJU), [opencv_zoo/face_recognition_sface] |

[opencv_zoo/face_detection_yunet]: https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet
[opencv_zoo/face_recognition_sface]: https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface

If you re-publish this repo or a fork, keep the model attribution in
this section. If you'd rather not vendor the binaries, delete `models/`
and on first request `model_loader.py` will fetch them from
`opencv_zoo` and verify the pinned SHA256.

---

## License

Source: MIT (see `LICENSE`).
Bundled model weights: see the [Models and credits](#models-and-credits) table.
