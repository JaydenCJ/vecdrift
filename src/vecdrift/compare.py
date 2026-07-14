"""Comparison engine: align two anchor geometries and measure the drift.

``compare`` is the heart of vecdrift. It aligns two baselines by anchor id
(either side may be a saved baseline or an in-memory snapshot of raw
vectors), restricts both geometries to the shared ids, and produces a
:class:`DriftReport` with global metrics, per-anchor drift, norm statistics,
and a graded verdict.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .baseline import Baseline
from .errors import PairingError
from .geometry import Geometry
from .linalg import mean, pearson, pstdev, spearman
from .metrics import effective_k, overlap_at_k, rank_shift
from .verdict import Thresholds, Verdict, evaluate

__all__ = ["AnchorDrift", "NormStats", "DriftReport", "compare", "MIN_MATCHED"]

MIN_MATCHED = 3  # fewer shared anchors than this gives no usable geometry


@dataclass(frozen=True)
class AnchorDrift:
    """Drift measured around a single anchor."""

    id: str
    overlap: float       # overlap@k of its neighborhood
    rank_shift: float    # mean displacement of its baseline top-k neighbors
    mean_delta: float    # mean |delta cosine| over its pairs
    max_delta: float     # worst single pair involving this anchor


@dataclass(frozen=True)
class NormStats:
    """Distribution of vector norms on one side of the comparison."""

    mean: float
    std: float
    min: float
    max: float

    @classmethod
    def from_norms(cls, norms: List[float]) -> "NormStats":
        return cls(mean=mean(norms), std=pstdev(norms), min=min(norms), max=max(norms))


@dataclass
class DriftReport:
    """Everything ``vecdrift compare`` knows about one comparison."""

    baseline_label: str
    candidate_label: str
    baseline_count: int
    candidate_count: int
    baseline_dim: int
    candidate_dim: int
    matched: int
    missing_ids: List[str]          # in baseline, absent from candidate
    extra_ids: List[str]            # in candidate, absent from baseline
    k: int                          # requested k
    k_effective: int                # k actually used (clamped to matched - 1)
    pearson: Optional[float]
    spearman: Optional[float]
    mean_delta: float
    max_delta: float
    max_delta_pair: Tuple[str, str]
    mean_overlap: float
    min_overlap: float
    min_overlap_id: str
    mean_rank_shift: float
    baseline_norms: NormStats
    candidate_norms: NormStats
    anchors: List[AnchorDrift] = field(default_factory=list)
    verdict: Verdict = Verdict.OK
    reasons: List[str] = field(default_factory=list)

    def worst_anchors(self, limit: int = 5) -> List[AnchorDrift]:
        """Anchors ranked by damage: lowest overlap first, then largest delta."""
        ranked = sorted(
            self.anchors, key=lambda a: (a.overlap, -a.mean_delta, a.id)
        )
        return ranked[:limit]

    @property
    def norms_comparable(self) -> bool:
        """Norm distributions only mean anything within one embedding space."""
        return self.baseline_dim == self.candidate_dim


def _align(base: Baseline, cand: Baseline) -> Tuple[List[str], List[str], List[str]]:
    cand_ids = set(cand.ids)
    base_ids = set(base.ids)
    matched = [anchor_id for anchor_id in base.ids if anchor_id in cand_ids]
    missing = [anchor_id for anchor_id in base.ids if anchor_id not in cand_ids]
    extra = [anchor_id for anchor_id in cand.ids if anchor_id not in base_ids]
    return matched, missing, extra


def compare(
    base: Baseline,
    cand: Baseline,
    k: int = 10,
    thresholds: Thresholds = Thresholds(),
) -> DriftReport:
    """Measure geometry drift from ``base`` to ``cand``.

    Raises :class:`PairingError` when fewer than :data:`MIN_MATCHED` anchor
    ids are shared — with no common anchors there is no geometry to compare,
    and failing loudly beats a meaningless verdict.
    """
    matched, missing, extra = _align(base, cand)
    if len(matched) < MIN_MATCHED:
        n = len(matched)
        raise PairingError(
            f"only {n} anchor id{'' if n == 1 else 's'} shared between the two sets "
            f"(need at least {MIN_MATCHED}); are these exports of the same anchor corpus?"
        )

    base_pos = {anchor_id: i for i, anchor_id in enumerate(base.ids)}
    cand_pos = {anchor_id: i for i, anchor_id in enumerate(cand.ids)}
    base_geo = base.geometry().subset(matched)
    cand_geo = cand.geometry().subset(matched)
    m = len(matched)
    k_eff = effective_k(k, m)

    # --- pairwise similarity structure -----------------------------------
    deltas = [abs(a - b) for a, b in zip(base_geo.sims, cand_geo.sims)]
    mean_delta = mean(deltas)
    # First index of the largest delta => deterministic max pair on ties.
    max_idx = max(range(len(deltas)), key=lambda i: (deltas[i], -i))
    max_pair = ("", "")
    per_anchor_deltas: List[List[float]] = [[] for _ in range(m)]
    for idx, (i, j, _) in enumerate(base_geo.pairs()):
        per_anchor_deltas[i].append(deltas[idx])
        per_anchor_deltas[j].append(deltas[idx])
        if idx == max_idx:
            max_pair = (matched[i], matched[j])

    corr_pearson = pearson(base_geo.sims, cand_geo.sims)
    corr_spearman = spearman(base_geo.sims, cand_geo.sims)

    # --- neighborhood structure -------------------------------------------
    anchors: List[AnchorDrift] = []
    for i, anchor_id in enumerate(matched):
        base_order = [matched[j] for j in base_geo.neighbors(i)]
        cand_order = [matched[j] for j in cand_geo.neighbors(i)]
        anchors.append(
            AnchorDrift(
                id=anchor_id,
                overlap=overlap_at_k(base_order, cand_order, k_eff),
                rank_shift=rank_shift(base_order, cand_order, k_eff),
                mean_delta=mean(per_anchor_deltas[i]),
                max_delta=max(per_anchor_deltas[i]),
            )
        )

    mean_overlap = mean([a.overlap for a in anchors])
    worst = min(anchors, key=lambda a: (a.overlap, a.id))
    mean_shift = mean([a.rank_shift for a in anchors])

    verdict, reasons = evaluate(mean_overlap, corr_pearson, mean_delta, thresholds)

    return DriftReport(
        baseline_label=base.label or base.source or "baseline",
        candidate_label=cand.label or cand.source or "candidate",
        baseline_count=len(base),
        candidate_count=len(cand),
        baseline_dim=base.dim,
        candidate_dim=cand.dim,
        matched=m,
        missing_ids=missing,
        extra_ids=extra,
        k=k,
        k_effective=k_eff,
        pearson=corr_pearson,
        spearman=corr_spearman,
        mean_delta=mean_delta,
        max_delta=deltas[max_idx],
        max_delta_pair=max_pair,
        mean_overlap=mean_overlap,
        min_overlap=worst.overlap,
        min_overlap_id=worst.id,
        mean_rank_shift=mean_shift,
        baseline_norms=NormStats.from_norms(
            [base.norms[base_pos[a]] for a in matched]
        ),
        candidate_norms=NormStats.from_norms(
            [cand.norms[cand_pos[a]] for a in matched]
        ),
        anchors=anchors,
        verdict=verdict,
        reasons=reasons,
    )
