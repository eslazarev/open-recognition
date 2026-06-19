# Installation

The only hard prerequisite is **Docker** (for the all-in-one Compose path) or a
**Kubernetes cluster + Helm** (for the chart). To run the server from source you
also need **Python 3.12** and [`uv`](https://docs.astral.sh/uv/).

## Docker Compose (recommended)

One command brings up Postgres **and** the server:

```bash
git clone https://github.com/eslazarev/open-recognition.git
cd open-recognition

docker compose up -d        # postgres + the API server, listening on :8080
```

Two services come up: `postgres` (`pgvector/pgvector:pg16`) and `app` (the
server, built from the `Dockerfile`). The image bundles YuNet (~232 KB) and
SFace (~38 MB), verified against the SHA256 values pinned in `model_loader.py`.
Migrations (`alembic upgrade head`) run automatically in the FastAPI lifespan,
so the schema is ready on first request.

```bash
docker compose logs -f app   # tail the server
docker compose down          # stop everything
```

## Helm (Kubernetes)

The chart is published as an OCI artifact on Docker Hub. It deploys the API as a
`Deployment` + `Service` and, by default, a bundled **pgvector** `StatefulSet`:

```bash
helm install rekog oci://registry-1.docker.io/eslazarev/open-recognition --version 0.1.0
```

See [Deployment](deployment.md) for values, external-database setup, and Ingress.

## From source (development)

For the dev loop — live reload, no image rebuild — run Postgres in Docker but
the server from `uv`:

```bash
docker compose up -d postgres                       # just the database
uv sync --extra dev                                 # Python 3.12 + deps
uv run uvicorn interface.http.app:app --port 8080
```

Run from source and the models download into `models/` on first request (the
Docker image bakes them in instead). Either way migrations run on startup.

## Verify it's up

```bash
curl -s http://localhost:8080/openapi.json -o /dev/null -w '%{http_code}\n'   # 200
```

Then open the [Faces Playground](http://localhost:8080/ui) or the
[Swagger UI](http://localhost:8080/docs) in a browser.
