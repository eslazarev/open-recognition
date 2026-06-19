# API reference

The interactive reference below is rendered directly from the server's shipped
OpenAPI spec ([`docs/openapi.json`](openapi.json)) — the same spec the running
server serves at `/openapi.json` and renders as Swagger UI at `/docs`. It is the
single source of truth for request fields, response keys, and error shapes.

!!! note "Two ways to call every operation"
    boto3 and other AWS SDKs use the canonical `POST /` with an
    `X-Amz-Target: RekognitionService.<Action>` header. For exploration the
    server also exposes `POST /<Action>` aliases (e.g. `POST /DetectFaces`),
    which is what the "Try it out" buttons below use. Both reach the same
    handlers. See [Operations](operations.md) for the high-level list and
    [Quickstart](quickstart.md) for runnable examples.

<swagger-ui src="openapi.json"/>
