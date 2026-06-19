# open-recognition

Self-hosted, boto3-compatible **AWS Rekognition Faces API**. Detect, index, compare and search faces using YuNet (detection) + SFace (128-d embeddings), backed by **PostgreSQL/pgvector** for similarity search, with a built-in HTMX playground UI. It speaks the AWS Rekognition JSON-1.1 protocol, so an unmodified `boto3` client works against it by swapping `endpoint_url`.

This chart deploys the server as a Kubernetes `Deployment` + `Service` (with optional `Ingress`). It can bring up a bundled **pgvector** `StatefulSet` out of the box, or connect to an external managed Postgres.

- Source: https://github.com/eslazarev/open-recognition
- Image: `docker.io/eslazarev/open-recognition`
- License: MIT

## TL;DR

```bash
helm install rekog ./charts/open-recognition
kubectl port-forward svc/rekog-open-recognition 8080:80
# open http://localhost:8080/ui
```

## Prerequisites

- Kubernetes 1.21+
- A default StorageClass (for the bundled Postgres PVC), or set `postgresql.persistence.enabled=false` for ephemeral demo storage
- If using an external database: a PostgreSQL instance **with the pgvector extension available** (e.g. Neon, Supabase, or `pgvector/pgvector`). The app runs `CREATE EXTENSION IF NOT EXISTS vector` on startup.

## Install

From the published chart repository (GitHub Pages):

```bash
helm repo add open-recognition https://eslazarev.github.io/open-recognition
helm repo update
helm install rekog open-recognition/open-recognition --version 0.1.0
```

Or from a cloned repo:

```bash
git clone https://github.com/eslazarev/open-recognition
cd open-recognition
helm install rekog ./charts/open-recognition
```

## Uninstall

```bash
helm uninstall rekog
# the bundled Postgres PVC is retained by design; delete it explicitly if desired:
kubectl delete pvc -l app.kubernetes.io/instance=rekog,app.kubernetes.io/component=postgresql
```

## Database options

### Bundled pgvector (default)

`postgresql.enabled=true` deploys a single-replica `pgvector/pgvector:pg16` StatefulSet with a PVC, and the app DSN is wired automatically. Good for demos and small self-hosted setups.

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

### External database (managed Postgres)

Disable the bundled DB and point at your own. Either pass the DSN inline (rendered into a chart-managed Secret):

```yaml
postgresql:
  enabled: false
externalDatabase:
  url: "postgresql://user:pass@db.example.com:5432/open_recognition"
```

…or reference a pre-created Secret (DSN never touches values):

```yaml
postgresql:
  enabled: false
externalDatabase:
  existingSecret: my-db-secret
  existingSecretKey: OPEN_RECOGNITION_DATABASE_URL
```

## Expose it

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

## Values

| Key | Type | Default | Description |
|---|---|---|---|
| `image.repository` | string | `eslazarev/open-recognition` | Container image repository |
| `image.tag` | string | `""` (uses `Chart.appVersion`) | Image tag |
| `image.pullPolicy` | string | `IfNotPresent` | Pull policy |
| `replicaCount` | int | `1` | Number of app replicas |
| `command` | list | venv uvicorn | Container command override (runs the synced venv binary so the rootfs can stay read-only) |
| `containerPort` | int | `8080` | Port the app listens on |
| `service.type` | string | `ClusterIP` | Service type |
| `service.port` | int | `80` | Service port |
| `ingress.enabled` | bool | `false` | Create an Ingress |
| `ingress.className` | string | `""` | IngressClass name |
| `ingress.hosts` | list | `open-recognition.local` | Ingress hosts/paths |
| `ingress.tls` | list | `[]` | Ingress TLS config |
| `postgresql.enabled` | bool | `true` | Deploy a bundled pgvector StatefulSet |
| `postgresql.image.repository` | string | `pgvector/pgvector` | Postgres image (must ship pgvector) |
| `postgresql.image.tag` | string | `pg16` | Postgres image tag |
| `postgresql.auth.username` | string | `open_recognition` | DB username |
| `postgresql.auth.password` | string | `open_recognition` | DB password — **change in production** |
| `postgresql.auth.database` | string | `open_recognition` | DB name |
| `postgresql.persistence.enabled` | bool | `true` | Use a PVC for DB data |
| `postgresql.persistence.size` | string | `8Gi` | PVC size |
| `postgresql.persistence.storageClass` | string | `""` | StorageClass (`""` = cluster default) |
| `externalDatabase.url` | string | `""` | DSN when `postgresql.enabled=false` |
| `externalDatabase.existingSecret` | string | `""` | Pre-created Secret holding the DSN |
| `externalDatabase.existingSecretKey` | string | `OPEN_RECOGNITION_DATABASE_URL` | Key within that Secret |
| `probePath` | string | `/openapi.json` | HTTP path for startup/liveness/readiness probes |
| `resources` | object | 250m/512Mi → 2/2Gi | App pod resources (ONNX + OpenCV need headroom) |
| `podSecurityContext` / `securityContext` | object | hardened defaults | Run as non-root, read-only rootfs, drop all caps |
| `serviceAccount.create` | bool | `true` | Create a ServiceAccount |

See `values.yaml` for the full list and defaults.

## Using it with boto3

```python
import boto3

rekog = boto3.client(
    "rekognition",
    endpoint_url="http://localhost:8080",  # or your Ingress URL
    region_name="us-east-1",
    aws_access_key_id="x", aws_secret_access_key="x",
)
rekog.create_collection(CollectionId="people")
rekog.index_faces(CollectionId="people", Image={"Bytes": open("face.jpg", "rb").read()})
```

The HTMX playground is at `/ui`, OpenAPI docs at `/docs`.
