# Runtime image for the open-recognition server.
# Base deps only — the `dev` extra (pytest, boto3, ruff, mypy) is for the
# test suite and the demo script, not for running the server.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

# opencv-contrib-python needs libGL and glib at runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Resolve dependencies first so they cache across source changes.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

# Then the project itself (source, models, alembic migrations).
COPY . .
RUN uv sync --frozen --no-dev

EXPOSE 8080

CMD ["uv", "run", "--no-dev", "uvicorn", \
     "interface.http.app:app", \
     "--host", "0.0.0.0", "--port", "8080"]
