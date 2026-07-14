#!/usr/bin/env python3
"""Generate deterministic synthetic embedding exports for the vecdrift demo.

Three exports of the same 48-document anchor corpus are written to the
directory given as argv[1] (default: the current directory):

* ``model_v1.jsonl``   — the "old" model: 48 anchors, dim 8, four topic
  clusters (documents within a cluster are close, clusters are far apart).
* ``model_v2.jsonl``   — a well-behaved upgrade: the same geometry rotated
  into dim 8 by exact Givens rotations, uniformly scaled, plus tiny noise.
  Cosine geometry is rotation- and scale-invariant, so vecdrift says OK.
* ``model_v3.jsonl``   — a drifting upgrade to dim 12: same construction,
  but 10 of the 48 documents are re-embedded with the *wrong* cluster
  direction mixed in. Their neighborhoods break; vecdrift flags it.

Everything is seeded (``random.Random(7)``), so the files — and every
number in the README quickstart — are reproducible bit-for-bit.
"""

from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

CLUSTERS = 4
PER_CLUSTER = 12
N_DOCS = CLUSTERS * PER_CLUSTER  # 48
DIM_V1 = 8
DIM_V3 = 12
DRIFTED = [f"doc-{i:02d}" for i in (3, 7, 11, 15, 19, 23, 27, 31, 35, 39)]


def _unit(vector):
    scale = math.sqrt(sum(component * component for component in vector))
    return [component / scale for component in vector]


def _cluster_axes(dim: int, rng: random.Random):
    """Nearly-orthogonal cluster center directions via Gram-Schmidt."""
    axes = []
    while len(axes) < CLUSTERS:
        candidate = [rng.gauss(0.0, 1.0) for _ in range(dim)]
        for axis in axes:
            proj = sum(a * b for a, b in zip(candidate, axis))
            candidate = [c - proj * a for c, a in zip(candidate, axis)]
        if math.sqrt(sum(c * c for c in candidate)) > 1e-6:
            axes.append(_unit(candidate))
    return axes


def make_v1():
    """The base corpus: cluster axis + per-document jitter, dim 8."""
    rng = random.Random(7)
    axes = _cluster_axes(DIM_V1, rng)
    docs = []
    for i in range(N_DOCS):
        axis = axes[i // PER_CLUSTER]
        jitter = [rng.gauss(0.0, 0.25) for _ in range(DIM_V1)]
        docs.append((f"doc-{i:02d}", [a + j for a, j in zip(axis, jitter)]))
    return docs


def _givens(vectors, dim_out, rng: random.Random, rotations: int):
    """Pad to dim_out and apply exact Givens rotations (orthogonal map)."""
    rotated = [vector + [0.0] * (dim_out - len(vector)) for vector in vectors]
    for _ in range(rotations):
        p, q = rng.sample(range(dim_out), 2)
        theta = rng.uniform(0.3, 2.8)
        c, s = math.cos(theta), math.sin(theta)
        for vector in rotated:
            vp, vq = vector[p], vector[q]
            vector[p] = c * vp - s * vq
            vector[q] = s * vp + c * vq
    return rotated


def make_v2(v1_docs):
    """Compatible upgrade: rotate + scale 1.7x + noise sigma=0.002."""
    rng = random.Random(11)
    vectors = _givens([vector for _, vector in v1_docs], DIM_V1, rng, rotations=24)
    docs = []
    for (doc_id, _), vector in zip(v1_docs, vectors):
        noisy = [1.7 * component + rng.gauss(0.0, 0.002) for component in vector]
        docs.append((doc_id, noisy))
    return docs


def make_v3(v1_docs):
    """Drifting upgrade to dim 12: 10 documents lose their cluster identity."""
    rng = random.Random(13)
    vectors = _givens([vector for _, vector in v1_docs], DIM_V3, rng, rotations=36)
    wrong_axis = _unit([rng.gauss(0.0, 1.0) for _ in range(DIM_V3)])
    docs = []
    for (doc_id, _), vector in zip(v1_docs, vectors):
        if doc_id in DRIFTED:
            # Mostly the wrong direction, a little of the original: the
            # document "moved" in embedding space, as bad tokenization or a
            # changed pooling strategy would make it move.
            vector = [0.25 * component + 1.5 * w for component, w in zip(vector, wrong_axis)]
        noisy = [component + rng.gauss(0.0, 0.002) for component in vector]
        docs.append((doc_id, noisy))
    return docs


def write_jsonl(docs, path: Path):
    lines = [
        json.dumps({"id": doc_id, "vector": [round(c, 6) for c in vector]})
        for doc_id, vector in docs
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    out_dir.mkdir(parents=True, exist_ok=True)
    v1 = make_v1()
    write_jsonl(v1, out_dir / "model_v1.jsonl")
    write_jsonl(make_v2(v1), out_dir / "model_v2.jsonl")
    write_jsonl(make_v3(v1), out_dir / "model_v3.jsonl")
    print(f"wrote model_v1.jsonl, model_v2.jsonl, model_v3.jsonl to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
