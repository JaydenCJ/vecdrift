"""Pure-Python vector math and rank statistics.

Everything in this module is a small, deterministic function over plain
``list[float]`` values. ``math.fsum`` is used for every accumulation so
results do not depend on summation order or platform-specific FPU quirks.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence

__all__ = [
    "dot",
    "norm",
    "cosine",
    "mean",
    "pstdev",
    "pearson",
    "rankdata",
    "spearman",
]


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    """Dot product of two equal-length vectors."""
    if len(a) != len(b):
        raise ValueError(f"dimension mismatch: {len(a)} vs {len(b)}")
    return math.fsum(x * y for x, y in zip(a, b))


def norm(a: Sequence[float]) -> float:
    """Euclidean (L2) norm."""
    return math.sqrt(math.fsum(x * x for x in a))


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity, clamped to [-1.0, 1.0] against rounding overshoot."""
    na = norm(a)
    nb = norm(b)
    if na == 0.0 or nb == 0.0:
        raise ValueError("cosine similarity is undefined for a zero vector")
    value = dot(a, b) / (na * nb)
    return max(-1.0, min(1.0, value))


def mean(values: Sequence[float]) -> float:
    """Arithmetic mean; raises on an empty sequence."""
    if not values:
        raise ValueError("mean of an empty sequence")
    return math.fsum(values) / len(values)


def pstdev(values: Sequence[float]) -> float:
    """Population standard deviation (denominator N, not N-1)."""
    m = mean(values)
    return math.sqrt(math.fsum((v - m) ** 2 for v in values) / len(values))


def pearson(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    """Pearson correlation coefficient.

    Returns ``None`` when either side has (numerically) zero variance —
    correlation is undefined there, and callers such as the verdict logic
    treat ``None`` as "no signal" rather than as agreement or disagreement.
    """
    if len(xs) != len(ys):
        raise ValueError(f"length mismatch: {len(xs)} vs {len(ys)}")
    if len(xs) < 2:
        return None
    mx = mean(xs)
    my = mean(ys)
    cov = math.fsum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = math.fsum((x - mx) ** 2 for x in xs)
    vy = math.fsum((y - my) ** 2 for y in ys)
    if vx <= 1e-24 or vy <= 1e-24:
        return None
    value = cov / math.sqrt(vx * vy)
    return max(-1.0, min(1.0, value))


def rankdata(values: Sequence[float]) -> List[float]:
    """Ranks (1-based) with ties assigned the average of their positions.

    This is the standard "average" tie method, so ``spearman`` below matches
    what scipy would report without vecdrift depending on scipy.
    """
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for pos in range(i, j + 1):
            ranks[order[pos]] = avg_rank
        i = j + 1
    return ranks


def spearman(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    """Spearman rank correlation: Pearson over tie-averaged ranks."""
    if len(xs) != len(ys):
        raise ValueError(f"length mismatch: {len(xs)} vs {len(ys)}")
    if len(xs) < 2:
        return None
    return pearson(rankdata(xs), rankdata(ys))
