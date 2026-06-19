# Deployment

## Docker Compose

The bundled `docker-compose.yml` is the simplest production-ish setup: the
server plus a `pgvector/pgvector:pg16` database with a persistent volume.

```bash
docker compose up -d
```

The `app` service sets `OPEN_RECOGNITION_DATABASE_URL` to reach the `postgres`
service over the Compose network. To use a managed Postgres instead, point that
variable at it (see [Configuration](configuration.md)).

## Helm (Kubernetes)

The chart lives in [`charts/open-recognition`](https://github.com/eslazarev/open-recognition/tree/main/charts/open-recognition)
and is published as an OCI artifact:

```bash
helm install rekog oci://registry-1.docker.io/eslazarev/open-recognition --version 0.1.0
```

It deploys the API as a `Deployment` + `Service` (optional `Ingress`) and, by
default, a bundled **pgvector** `StatefulSet` with a PVC.

### Database options

=== "Bundled pgvector (default)"

    ```yaml
    postgresql:
      enabled: true
      auth:
        username: open_recognition
        password: change-me        # set a real password
        database: open_recognition
      persistence:
        size: 20Gi
        storageClass: ""           # "" = cluster default
    ```

=== "External database (inline DSN)"

    ```yaml
    postgresql:
      enabled: false
    externalDatabase:
      url: "postgresql://user:pass@db.example.com:5432/open_recognition"
    ```

=== "External database (existing Secret)"

    ```yaml
    postgresql:
      enabled: false
    externalDatabase:
      existingSecret: my-db-secret
      existingSecretKey: OPEN_RECOGNITION_DATABASE_URL
    ```

The external database must have the `pgvector` extension available — the app
runs `CREATE EXTENSION IF NOT EXISTS vector` on startup.

### Expose via Ingress

```yaml
ingress:
  enabled: true
  className: nginx
  hosts:
    - host: rekognition.example.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: rekognition-tls
      hosts:
        - rekognition.example.com
```

### Key values

| Key | Default | Description |
|---|---|---|
| `image.repository` | `eslazarev/open-recognition` | Application container image |
| `image.tag` | `""` (uses `Chart.appVersion`) | Image tag |
| `replicaCount` | `1` | App replicas |
| `postgresql.enabled` | `true` | Deploy a bundled pgvector StatefulSet |
| `postgresql.persistence.size` | `8Gi` | PVC size |
| `externalDatabase.url` | `""` | DSN when `postgresql.enabled=false` |
| `service.type` | `ClusterIP` | Service type |
| `ingress.enabled` | `false` | Create an Ingress |
| `resources` | 250m/512Mi → 2/2Gi | App pod resources (ONNX + OpenCV need headroom) |

See the chart `values.yaml` for the full list.

!!! note "Probes and startup"
    Startup runs DB migrations and loads four ONNX models, so the startup probe
    is deliberately generous. Liveness/readiness probe `/openapi.json`, which
    does not touch the database.
