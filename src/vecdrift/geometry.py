"""Anchor-set geometry: condensed pairwise similarity and neighbor orders.

The core idea of vecdrift lives here. Two embedding models can never be
compared vector-by-vector (different dimensions, arbitrary rotations), but
the *relative* geometry of a fixed anchor set — who is close to whom — is
exactly what vector search consumes. A :class:`Geometry` is that relative
structure: the upper triangle of the anchor-pair cosine matrix, stored as a
flat "condensed" list, plus deterministic per-anchor neighbor orderings.
"""

from __future__ import annotations

from typing import Iterator, List, Sequence, Tuple

from .linalg import cosine
from .vectors import VectorSet

__all__ = ["Geometry", "condensed_length", "condensed_index"]


def condensed_length(n: int) -> int:
    """Number of unordered pairs among ``n`` anchors."""
    return n * (n - 1) // 2


def condensed_index(i: int, j: int, n: int) -> int:
    """Index of pair (i, j), i < j, in a condensed upper-triangle list."""
    if i == j:
        raise ValueError("no self-pairs in a condensed matrix")
    if i > j:
        i, j = j, i
    if j >= n or i < 0:
        raise IndexError(f"pair ({i}, {j}) out of range for n={n}")
    return i * n - i * (i + 1) // 2 + (j - i - 1)


class Geometry:
    """Pairwise cosine structure of an ordered anchor set."""

    def __init__(self, ids: Sequence[str], sims: Sequence[float]) -> None:
        expected = condensed_length(len(ids))
        if len(sims) != expected:
            raise ValueError(
                f"{len(ids)} anchors need {expected} pair similarities, got {len(sims)}"
            )
        self.ids: List[str] = list(ids)
        self.sims: List[float] = list(sims)
        self._n = len(self.ids)
        self._pos = {anchor_id: i for i, anchor_id in enumerate(self.ids)}

    def __len__(self) -> int:
        return self._n

    @classmethod
    def from_vectors(cls, vector_set: VectorSet) -> "Geometry":
        """Compute the condensed cosine matrix of a vector export."""
        n = len(vector_set)
        sims: List[float] = []
        for i in range(n):
            vi = vector_set.vectors[i]
            for j in range(i + 1, n):
                sims.append(cosine(vi, vector_set.vectors[j]))
        return cls(vector_set.ids, sims)

    def sim(self, i: int, j: int) -> float:
        """Cosine similarity between anchors at positions ``i`` and ``j``."""
        return self.sims[condensed_index(i, j, self._n)]

    def sim_by_id(self, id_a: str, id_b: str) -> float:
        return self.sim(self._pos[id_a], self._pos[id_b])

    def pairs(self) -> Iterator[Tuple[int, int, float]]:
        """Yield every (i, j, similarity) with i < j, in condensed order."""
        idx = 0
        for i in range(self._n):
            for j in range(i + 1, self._n):
                yield i, j, self.sims[idx]
                idx += 1

    def neighbors(self, i: int) -> List[int]:
        """All other anchor positions ordered by similarity to anchor ``i``.

        Ties are broken by ascending anchor id so the ordering is fully
        deterministic regardless of input order — essential for stable
        overlap@k numbers across runs and machines.
        """
        others = [j for j in range(self._n) if j != i]
        others.sort(key=lambda j: (-self.sim(i, j), self.ids[j]))
        return others

    def neighbor_ids(self, anchor_id: str) -> List[str]:
        return [self.ids[j] for j in self.neighbors(self._pos[anchor_id])]

    def subset(self, keep_ids: Sequence[str]) -> "Geometry":
        """Restrict to ``keep_ids`` (in the order given) without recomputing.

        Used when a baseline and a candidate only partially overlap: the
        stored condensed matrix is sliced instead of needing raw vectors.
        """
        positions = []
        for anchor_id in keep_ids:
            if anchor_id not in self._pos:
                raise KeyError(anchor_id)
            positions.append(self._pos[anchor_id])
        sims: List[float] = []
        m = len(positions)
        for a in range(m):
            for b in range(a + 1, m):
                sims.append(self.sim(positions[a], positions[b]))
        return Geometry(list(keep_ids), sims)
