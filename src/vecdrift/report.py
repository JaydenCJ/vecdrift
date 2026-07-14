"""Rendering a DriftReport as terminal text or machine-readable JSON.

The text renderer is what humans read in CI logs; the JSON renderer is what
dashboards and scripts consume (``vecdrift compare --json``). Both are pure
functions of the report, with stable field order, so output diffs cleanly.
"""

from __future__ import annotations

import json
from typing import Optional

from .compare import DriftReport

__all__ = ["render_text", "render_json", "report_to_dict"]


def _fmt_corr(value: Optional[float]) -> str:
    return f"{value:.4f}" if value is not None else "n/a (zero-variance geometry)"


def render_text(report: DriftReport, worst: int = 5) -> str:
    """Human-readable comparison summary, one metric per line."""
    lines = []
    lines.append(
        f"vecdrift: {report.baseline_label} ({report.baseline_count} anchors, dim "
        f"{report.baseline_dim}) vs {report.candidate_label} "
        f"({report.candidate_count} anchors, dim {report.candidate_dim})"
    )
    lines.append(
        f"matched anchors : {report.matched}"
        f" ({len(report.missing_ids)} missing from candidate,"
        f" {len(report.extra_ids)} extra)"
    )
    lines.append("")
    lines.append("pairwise geometry")
    lines.append(f"  similarity correlation (pearson)  : {_fmt_corr(report.pearson)}")
    lines.append(f"  similarity correlation (spearman) : {_fmt_corr(report.spearman)}")
    lines.append(f"  mean |delta similarity|           : {report.mean_delta:.4f}")
    lines.append(
        f"  max  |delta similarity|           : {report.max_delta:.4f}"
        f"  ({report.max_delta_pair[0]} vs {report.max_delta_pair[1]})"
    )
    lines.append("")
    lines.append(f"neighborhoods (k={report.k_effective})")
    lines.append(f"  mean overlap@{report.k_effective:<3d} : {report.mean_overlap:.3f}")
    lines.append(
        f"  min  overlap@{report.k_effective:<3d} : {report.min_overlap:.3f}"
        f"  ({report.min_overlap_id})"
    )
    lines.append(f"  mean rank shift  : {report.mean_rank_shift:.2f}")
    if report.norms_comparable:
        lines.append("")
        lines.append("vector norms (same dim, comparable)")
        lines.append(
            f"  baseline  mean {report.baseline_norms.mean:.4f}"
            f"  std {report.baseline_norms.std:.4f}"
        )
        lines.append(
            f"  candidate mean {report.candidate_norms.mean:.4f}"
            f"  std {report.candidate_norms.std:.4f}"
        )
    if worst > 0 and report.verdict.value != "OK":
        lines.append("")
        lines.append("worst anchors")
        for anchor in report.worst_anchors(worst):
            lines.append(
                f"  {anchor.id:<20s} overlap {anchor.overlap:.2f}"
                f"  rank shift {anchor.rank_shift:.1f}"
                f"  mean |dsim| {anchor.mean_delta:.4f}"
            )
    lines.append("")
    lines.append(f"verdict: {report.verdict.value}")
    for reason in report.reasons:
        lines.append(f"  - {reason}")
    if report.verdict.value == "WARN":
        lines.append("  spot-check top queries against the new index before shipping.")
    elif report.verdict.value == "RE-EMBED":
        lines.append("  re-embed the corpus before switching models; recall will change.")
    return "\n".join(lines)


def report_to_dict(report: DriftReport, worst: int = 5) -> dict:
    """Plain-dict form of the report (everything JSON-serializable)."""
    return {
        "baseline": {
            "label": report.baseline_label,
            "count": report.baseline_count,
            "dim": report.baseline_dim,
            "norms": vars(report.baseline_norms).copy(),
        },
        "candidate": {
            "label": report.candidate_label,
            "count": report.candidate_count,
            "dim": report.candidate_dim,
            "norms": vars(report.candidate_norms).copy(),
        },
        "alignment": {
            "matched": report.matched,
            "missing_ids": list(report.missing_ids),
            "extra_ids": list(report.extra_ids),
        },
        "metrics": {
            "k": report.k,
            "k_effective": report.k_effective,
            "pearson": report.pearson,
            "spearman": report.spearman,
            "mean_delta_similarity": report.mean_delta,
            "max_delta_similarity": report.max_delta,
            "max_delta_pair": list(report.max_delta_pair),
            "mean_overlap": report.mean_overlap,
            "min_overlap": report.min_overlap,
            "min_overlap_id": report.min_overlap_id,
            "mean_rank_shift": report.mean_rank_shift,
        },
        "worst_anchors": [
            {
                "id": anchor.id,
                "overlap": anchor.overlap,
                "rank_shift": anchor.rank_shift,
                "mean_delta": anchor.mean_delta,
                "max_delta": anchor.max_delta,
            }
            for anchor in report.worst_anchors(worst)
        ],
        "verdict": report.verdict.value,
        "reasons": list(report.reasons),
    }


def render_json(report: DriftReport, worst: int = 5) -> str:
    """The dict form serialized with sorted keys (stable for diffing)."""
    return json.dumps(report_to_dict(report, worst), sort_keys=True, indent=2)
