# Security notes

A few things to know before you let this anywhere near production traffic.

!!! danger "No authentication"
    SigV4 headers are **ignored** — any `Authorization` value is accepted. Do
    not put this behind a public load balancer. Put it behind a real
    authenticator: a reverse proxy with JWT, a service mesh, mTLS — your usual
    choice.

- **ONNX checksums are pinned.** YuNet and SFace are verified against hardcoded
  SHA256 hashes in `model_loader.py` on every load. A mismatch deletes the file,
  downloads once more, and refuses to start if it still doesn't match.

- **Image bytes are decoded with Pillow.** Pillow has had its share of CVEs;
  keep dependencies updated. Inline payloads are capped at **5 MB**
  (`MAX_BYTES` in `image_decoder.py`), matching the AWS Rekognition limit.

- **Embeddings are not images.** What's stored in Postgres is a 128-d float
  vector; without the SFace model you can't reconstruct the face. But face
  embeddings are still **personal data under GDPR** — treat the database
  accordingly: backups, retention, deletion-on-request.

- **No multi-tenant isolation.** Collection IDs are a flat namespace. For
  per-tenant separation, run separate Postgres schemas or separate instances.

- **No per-call billing or quotas.** There isn't any rate limiting. Be careful
  what you point at it.

To report a vulnerability, see the repository's security policy.
