# Quickstart

With the server [running](installation.md) on `:8080`, here are three ways to
talk to it. All three hit the same handlers.

## With boto3

The only change from real AWS is `endpoint_url`. Credentials are required by the
SDK but ignored by the server.

```python
import boto3

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

The same code points at real AWS by removing `endpoint_url`.

## With raw HTTP

It's just AWS JSON-1.1 over HTTP — `POST /` with an `X-Amz-Target` header and a
JSON body. The image goes in as base64 under `Image.Bytes`:

```bash
IMG=$(base64 -i alice.jpg)          # macOS; on Linux: base64 -w0 alice.jpg

curl -s http://localhost:8080/ \
  -H 'X-Amz-Target: RekognitionService.DetectFaces' \
  -H 'Content-Type: application/x-amz-json-1.1' \
  -d "{\"Image\": {\"Bytes\": \"$IMG\"}}"
# {"FaceDetails":[{"BoundingBox":{...},"Confidence":99.5,"Landmarks":[...]}]}
```

For exploration the server also accepts `POST /<Action>` (e.g.
`POST /DetectFaces`) as an alias — this is what Swagger's "Try it out" uses.
boto3 keeps using the canonical `POST /` + header.

## In the browser

A **Faces Playground** lives at [`/ui`](http://localhost:8080/ui) — a
self-contained HTMX page (no build step). Upload an image and see detected faces
with boxes and landmarks, create collections, index, search, and compare,
without writing a line of code.

The raw API reference is the Swagger UI at [`/docs`](http://localhost:8080/docs),
with the spec at `/openapi.json` (also rendered in [API reference](api.md)).

## Telling people apart

`CompareFaces` between two different faces scores far below the `80` threshold,
so they're never confused:

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
compare("biden.jpg", "biden2.jpg")     # → 100.0 (same person, different photo)
```

For a full end-to-end demo on real photographs (enrol one photo, search with a
different unseen one, reject a stranger), run `scripts/demo_real_faces.py` —
see the repository README.
