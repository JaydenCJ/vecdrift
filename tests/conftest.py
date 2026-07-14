"""Shared test fixtures and deterministic vector factories.

Everything here is seeded or hand-constructed: no wall clock, no network,
no RNG without a fixed seed. The rotation helpers build *exact* orthogonal
maps (Givens rotations), which is what makes "geometry is invariant under
rotation" testable to tight tolerances.
"""

from __future__ import annotations

import math
import random
from typing import List, Tuple

import pytest

from vecdrift.vectors import VectorSet


def make_set(pairs: List[Tuple[str, List[float]]], source: str = "test") -> VectorSet:
    """Build a VectorSet from (id, vector) pairs without touching disk."""
    return VectorSet(
        ids=[anchor_id for anchor_id, _ in pairs],
        vectors=[[float(c) for c in vector] for _, vector in pairs],
        source=source,
    )


def clustered_vectors(
    n_clusters: int = 3,
    per_cluster: int = 6,
    dim: int = 6,
    seed: int = 42,
    jitter: float = 0.2,
) -> VectorSet:
    """A corpus with clear cluster structure (so neighborhoods are stable)."""
    rng = random.Random(seed)
    axes = []
    while len(axes) < n_clusters:
        candidate = [rng.gauss(0.0, 1.0) for _ in range(dim)]
        for axis in axes:
            proj = sum(a * b for a, b in zip(candidate, axis))
            candidate = [c - proj * a for c, a in zip(candidate, axis)]
        scale = math.sqrt(sum(c * c for c in candidate))
        if scale > 1e-6:
            axes.append([c / scale for c in candidate])
    pairs = []
    for i in range(n_clusters * per_cluster):
        axis = axes[i // per_cluster]
        vector = [a + rng.gauss(0.0, jitter) for a in axis]
        pairs.append((f"doc-{i:02d}", vector))
    return make_set(pairs, source="clustered")


def rotate(vector_set: VectorSet, seed: int = 5, rotations: int = 12,
           scale: float = 1.0, pad_to: int = 0) -> VectorSet:
    """Apply an exact orthogonal map (Givens rotations) + uniform scale.

    Cosine geometry is invariant under this, so the rotated set must compare
    as zero drift — the key property the whole tool rests on.
    """
    rng = random.Random(seed)
    dim = max(vector_set.dim, pad_to)
    vectors = [list(v) + [0.0] * (dim - len(v)) for v in vector_set.vectors]
    for _ in range(rotations):
        p, q = rng.sample(range(dim), 2)
        theta = rng.uniform(0.2, 3.0)
        c, s = math.cos(theta), math.sin(theta)
        for vector in vectors:
            vp, vq = vector[p], vector[q]
            vector[p] = c * vp - s * vq
            vector[q] = s * vp + c * vq
    return VectorSet(
        ids=list(vector_set.ids),
        vectors=[[scale * c for c in vector] for vector in vectors],
        source="rotated",
    )


@pytest.fixture()
def corpus() -> VectorSet:
    return clustered_vectors()
