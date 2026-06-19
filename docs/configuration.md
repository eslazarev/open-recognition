# Configuration

Everything is environment variables. There's no config file.

| Variable | Default | What it does |
|---|---|---|
| `OPEN_RECOGNITION_DATABASE_URL` | `postgresql://open_recognition:open_recognition@localhost:5432/open_recognition` | Postgres DSN. asyncpg-style; alembic's `env.py` rewrites it to `postgresql+psycopg://` internally. |
| `OPEN_RECOGNITION_CV_POOL_SIZE` | `min(4, cpu_count())` | Number of cv2 detector/recognizer instances in each pool. Higher = more parallel inference, ~15 MB extra RAM per slot. |
| `OPEN_RECOGNITION_MODELS_DIR` | `./models` | Where ONNX files live. Auto-created. |
| `OPEN_RECOGNITION_ALEMBIC_INI` | `<project_root>/alembic.ini` | Override only if running alembic from a non-standard location. |

## Database URL

The DSN is the one setting you'll almost always change. Examples:

```bash
# Local Postgres
export OPEN_RECOGNITION_DATABASE_URL="postgresql://user:pass@localhost:5432/open_recognition"

# Managed Postgres (must have the pgvector extension available)
export OPEN_RECOGNITION_DATABASE_URL="postgresql://user:pass@db.example.com:5432/mydb"
```

Under Docker Compose this is set on the `app` service to reach the `postgres`
service. Under Helm it is derived automatically for the bundled database, or
taken from `externalDatabase` — see [Deployment](deployment.md).

The app creates the `vector` extension itself on first migration
(`CREATE EXTENSION IF NOT EXISTS vector`), so the database role needs privilege
to create extensions, or the extension must be pre-created by an admin.

## Tuning inference parallelism

`OPEN_RECOGNITION_CV_POOL_SIZE` controls how many cv2 model instances sit in the
detector/recognizer pools. cv2 releases the GIL during inference, so a pool of
distinct instances unlocks real CPU parallelism — see [Performance](performance.md)
for measured throughput at different pool sizes. Each slot costs ~15 MB RAM.
