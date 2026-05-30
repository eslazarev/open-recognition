"""Write the OpenAPI spec to docs/openapi.json.

Pure — no running server or database needed. Keep the checked-in file in sync
with the schemas:

  uv run python scripts/dump_openapi.py
"""

from __future__ import annotations

import json
from pathlib import Path

from interface.http.openapi import build_openapi

OUT = Path(__file__).resolve().parents[1] / "docs" / "openapi.json"


def main() -> None:
    spec = build_openapi()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(spec, indent=2) + "\n")
    print(f"wrote {OUT}  ({len(spec['paths'])} operations, "
          f"{len(spec['components']['schemas'])} schemas)")


if __name__ == "__main__":
    main()
