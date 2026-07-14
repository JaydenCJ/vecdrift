"""Threshold policy: turn drift metrics into a re-embed-or-not verdict.

The three-tier verdict (``OK`` / ``WARN`` / ``RE-EMBED``) is the whole point
of vecdrift — a number like "pearson 0.983" is not actionable in CI, but
"WARN: spot-check before shipping" is. Thresholds are explicit, documented
defaults that every team can override per corpus.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import List, Optional, Tuple

__all__ = ["Verdict", "Thresholds", "evaluate"]


class Verdict(enum.Enum):
    """Severity-ordered comparison outcome."""

    OK = "OK"
    WARN = "WARN"
    REEMBED = "RE-EMBED"

    @property
    def severity(self) -> int:
        return {"OK": 0, "WARN": 1, "RE-EMBED": 2}[self.value]


@dataclass(frozen=True)
class Thresholds:
    """Two-tier metric gates.

    A comparison is ``OK`` when every ``ok_*`` gate passes, ``WARN`` when
    every ``warn_*`` gate passes, otherwise ``RE-EMBED``. The defaults are
    tuned for retrieval corpora: an intact top-10 neighborhood (overlap
    >= 0.95) with near-perfect pairwise-similarity correlation is safe;
    overlap below 0.80 means users are already seeing different results.
    """

    ok_overlap: float = 0.95
    ok_correlation: float = 0.995
    ok_delta: float = 0.02
    warn_overlap: float = 0.80
    warn_correlation: float = 0.97
    warn_delta: float = 0.05

    def __post_init__(self) -> None:
        for name in ("ok_overlap", "ok_correlation", "warn_overlap", "warn_correlation"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")
        for name in ("ok_delta", "warn_delta"):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be >= 0")
        if self.warn_overlap > self.ok_overlap:
            raise ValueError("warn_overlap must not exceed ok_overlap")
        if self.warn_correlation > self.ok_correlation:
            raise ValueError("warn_correlation must not exceed ok_correlation")
        if self.warn_delta < self.ok_delta:
            raise ValueError("warn_delta must not be below ok_delta")


def _tier_failures(
    tier: str,
    min_overlap: float,
    min_correlation: float,
    max_delta: float,
    mean_overlap: float,
    correlation: Optional[float],
    mean_delta: float,
) -> List[str]:
    failures: List[str] = []
    if mean_overlap < min_overlap:
        failures.append(
            f"mean neighborhood overlap {mean_overlap:.3f} < {tier} threshold {min_overlap:.3f}"
        )
    # A None correlation (zero-variance geometry, e.g. an orthonormal anchor
    # set) carries no signal, so it never fails a gate by itself; overlap and
    # delta still guard those cases.
    if correlation is not None and correlation < min_correlation:
        failures.append(
            f"pairwise similarity correlation {correlation:.4f} < {tier} threshold {min_correlation:.4f}"
        )
    if mean_delta > max_delta:
        failures.append(
            f"mean |delta similarity| {mean_delta:.4f} > {tier} threshold {max_delta:.4f}"
        )
    return failures


def evaluate(
    mean_overlap: float,
    correlation: Optional[float],
    mean_delta: float,
    thresholds: Thresholds = Thresholds(),
) -> Tuple[Verdict, List[str]]:
    """Grade drift metrics against a threshold policy.

    Returns the verdict plus human-readable reasons: for ``WARN`` the list
    explains which OK gates failed; for ``RE-EMBED`` it explains which WARN
    gates failed (the ones that make the drift unshippable).
    """
    ok_failures = _tier_failures(
        "ok", thresholds.ok_overlap, thresholds.ok_correlation, thresholds.ok_delta,
        mean_overlap, correlation, mean_delta,
    )
    if not ok_failures:
        return Verdict.OK, []
    warn_failures = _tier_failures(
        "warn", thresholds.warn_overlap, thresholds.warn_correlation, thresholds.warn_delta,
        mean_overlap, correlation, mean_delta,
    )
    if not warn_failures:
        return Verdict.WARN, ok_failures
    return Verdict.REEMBED, warn_failures
