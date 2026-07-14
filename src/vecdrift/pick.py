"""Anchor selection: pick a diverse, fixed subset from a large export.

You do not want to snapshot a million vectors — you want a few dozen to a
few hundred anchors that *span* the corpus, so that a geometry change
anywhere shows up in some anchor pair. ``pick_anchors`` uses farthest-point
sampling on cosine distance: start from the vector farthest from the corpus
centroid, then repeatedly add the vector farthest (in cosine distance) from
everything already selected. The procedure is fully deterministic — no RNG,
ties broken by anchor id — so two people picking from the same export get
the same anchor set.
"""

from __future__ import annotations

from typing import List

from .linalg import cosine, norm
from .vectors import VectorSet

__all__ = ["pick_anchors"]


def _centroid(vectors: List[List[float]]) -> List[float]:
    dim = len(vectors[0])
    total = [0.0] * dim
    for vector in vectors:
        for i, component in enumerate(vector):
            total[i] += component
    return [component / len(vectors) for component in total]


def pick_anchors(vector_set: VectorSet, count: int) -> VectorSet:
    """Select ``count`` diverse anchors via farthest-point sampling.

    Returns the selection as a new :class:`VectorSet` in the *original file
    order* (not selection order), so repeated picks diff cleanly. If
    ``count`` >= the export size, the whole export is returned.
    """
    if count < 1:
        raise ValueError("count must be >= 1")
    n = len(vector_set)
    if count >= n:
        return vector_set.subset(list(vector_set.ids))

    centroid = _centroid(vector_set.vectors)
    if norm(centroid) == 0.0:
        # Degenerate but possible (vectors cancel out): fall back to the
        # smallest id, which is still deterministic and order-independent.
        seed = min(range(n), key=lambda i: vector_set.ids[i])
    else:
        # Seed = the vector farthest (cosine distance) from the centroid;
        # ties resolved by smallest id so the pick is order-independent.
        gaps = [1.0 - cosine(vector, centroid) for vector in vector_set.vectors]
        best_gap = max(gaps)
        tied = [i for i in range(n) if abs(gaps[i] - best_gap) < 1e-12]
        seed = min(tied, key=lambda i: vector_set.ids[i])

    selected = [seed]
    # min cosine-distance from each candidate to the selected set
    min_dist = [
        1.0 - cosine(vector_set.vectors[i], vector_set.vectors[seed]) for i in range(n)
    ]
    min_dist[seed] = -1.0  # never re-pick

    while len(selected) < count:
        best = max(min_dist)
        tied = [i for i in range(n) if abs(min_dist[i] - best) < 1e-12 and min_dist[i] >= 0.0]
        nxt = min(tied, key=lambda i: vector_set.ids[i])
        selected.append(nxt)
        min_dist[nxt] = -1.0
        for i in range(n):
            if min_dist[i] < 0.0:
                continue
            dist = 1.0 - cosine(vector_set.vectors[i], vector_set.vectors[nxt])
            if dist < min_dist[i]:
                min_dist[i] = dist

    chosen = set(selected)
    keep = [anchor_id for i, anchor_id in enumerate(vector_set.ids) if i in chosen]
    return vector_set.subset(keep)
