"""Similarity scoring and AWS-compatible percentage conversions.

Single source of truth for every place that converts between cosine
similarity (range [-1, 1]) and the AWS-Rekognition-style percentage
(range [0, 100]). All handlers and the face repository import from
here — never compute these mappings inline.

# Calibration

The cosine → percentage mapping is a logistic fitted against 200
LFW pairs scored on real AWS Rekognition CompareFaces (Mar 2026).
Fit script: `scripts/fit_calibration.py`. Raw data: `calibration.csv`.

Logistic form:
    pct(cos) = 100 / (1 + exp(-k * (cos - c)))

With our k and c, the implied cosine cutoffs for common AWS thresholds:

    AWS Similarity ≥  80%  ↔  cosine ≥ 0.353
    AWS Similarity ≥  90%  ↔  cosine ≥ 0.370
    AWS Similarity ≥  95%  ↔  cosine ≥ 0.385
    AWS Similarity ≥  99%  ↔  cosine ≥ 0.419

RMSE on calibration set: 10.2 percentage points (vs 39.5 for the
previous linear mapping). The residual is dominated by the narrow
AWS transition zone around cos ≈ 0.33; outside that zone the fit
matches AWS to <1 percentage point.
"""

from __future__ import annotations

import math

import numpy as np

from domain.embedding import Embedding

# Fitted constants from scripts/fit_calibration.py against 200 LFW pairs.
AWS_CALIBRATION_K = 49.0
AWS_CALIBRATION_C = 0.325


def cosine(a: Embedding, b: Embedding) -> float:
    """Cosine similarity of two unit-norm embeddings, in [-1, 1]."""
    return float(np.dot(a.vector, b.vector))


def similarity_pct(cos: float) -> float:
    """Map cosine [-1, 1] → AWS-style percentage [0, 100].

    Uses the fitted logistic so a `SimilarityThreshold=80` against
    this service has roughly the same precision/recall as the same
    threshold against AWS Rekognition.
    """
    z = AWS_CALIBRATION_K * (cos - AWS_CALIBRATION_C)
    # Clip the exponent to avoid math.exp overflow at extreme cosines.
    if z > 50:
        return 100.0
    if z < -50:
        return 0.0
    return 100.0 / (1.0 + math.exp(-z))


def cosine_threshold_from_pct(pct: float) -> float:
    """Inverse of similarity_pct: AWS percentage → cosine threshold.

    Boundary semantics:
    - pct == 0   → -1.0  ("accept everything")
    - pct == 100 → +1.0  ("require identical")
    - else: logistic inverse.
    """
    if not 0.0 <= pct <= 100.0:
        raise ValueError(f"Percentage must be in [0, 100], got {pct}")
    if pct <= 0.0:
        return -1.0
    if pct >= 100.0:
        return 1.0
    return AWS_CALIBRATION_C - math.log((100.0 - pct) / pct) / AWS_CALIBRATION_K
