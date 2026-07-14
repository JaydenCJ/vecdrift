"""Neighborhood metrics between two anchor geometries.

These are the metrics that map most directly onto vector-search behavior:
``overlap@k`` is a proxy for recall@k (how many of yesterday's top-k
neighbors are still in today's top-k), and rank shift measures how far the
survivors moved. Both operate on the deterministic neighbor orderings
produced by :meth:`vecdrift.geometry.Geometry.neighbors`.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

__all__ = ["effective_k", "overlap_at_k", "rank_shift"]


def effective_k(k: int, n_anchors: int) -> int:
    """Clamp ``k`` to the number of available neighbors (n - 1)."""
    if k < 1:
        raise ValueError("k must be >= 1")
    return min(k, max(n_anchors - 1, 0))


def overlap_at_k(base_order: Sequence[str], cand_order: Sequence[str], k: int) -> float:
    """Fraction of the baseline top-k neighbors still in the candidate top-k.

    Both orderings must rank the same id universe (vecdrift compares
    matched anchors only, so that holds by construction). Returns a value
    in [0, 1]; 1.0 means the neighborhood is intact.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    k = min(k, len(base_order), len(cand_order))
    if k == 0:
        return 1.0
    base_top = set(base_order[:k])
    cand_top = set(cand_order[:k])
    return len(base_top & cand_top) / k


def rank_shift(base_order: Sequence[str], cand_order: Sequence[str], k: int) -> float:
    """Mean absolute rank displacement of the baseline top-k neighbors.

    For each of the baseline's top-k neighbors, find its rank in the
    candidate ordering and average ``|rank_new - rank_old|``. A shift of
    0.0 means every close neighbor kept its exact position; large values
    mean re-ranking that users will see as changed search results.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    k = min(k, len(base_order))
    if k == 0:
        return 0.0
    cand_rank: Dict[str, int] = {anchor_id: r for r, anchor_id in enumerate(cand_order)}
    shifts: List[int] = []
    for base_rank, anchor_id in enumerate(base_order[:k]):
        if anchor_id not in cand_rank:
            raise ValueError(f"id {anchor_id!r} missing from candidate ordering")
        shifts.append(abs(cand_rank[anchor_id] - base_rank))
    return sum(shifts) / len(shifts)
