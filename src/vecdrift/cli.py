"""Command-line interface for vecdrift.

Subcommands:

* ``snapshot`` — freeze the geometry of a vector export into a baseline file
* ``compare``  — grade a candidate export against a baseline (or another
  export) and exit non-zero on drift, per ``--fail-on``
* ``inspect``  — sanity-check a vector export (count, dim, norms, near-dupes)
* ``pick``     — select a diverse anchor subset from a large export

Exit codes: 0 = success / verdict passed, 1 = drift at or above the
``--fail-on`` level, 2 = usage or input error.
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from . import __version__
from .baseline import load_reference, snapshot
from .compare import compare
from .errors import VecdriftError
from .geometry import Geometry
from .linalg import mean, norm, pstdev
from .pick import pick_anchors
from .report import render_json, render_text
from .vectors import load_vectors, write_jsonl
from .verdict import Thresholds

__all__ = ["main", "build_parser"]

_FAIL_LEVELS = {"never": 3, "re-embed": 2, "warn": 1}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vecdrift",
        description=(
            "Detect embedding-space drift across model versions with "
            "anchor-pair distance checks — fully offline, over exported vectors."
        ),
    )
    parser.add_argument(
        "--version", action="version", version=f"vecdrift {__version__}"
    )
    sub = parser.add_subparsers(dest="command", metavar="command")

    p_snap = sub.add_parser(
        "snapshot", help="freeze a vector export's geometry into a baseline file"
    )
    p_snap.add_argument("vectors", help="vector export (.jsonl / .json / .csv)")
    p_snap.add_argument(
        "-o", "--output", required=True, help="baseline file to write (.json)"
    )
    p_snap.add_argument(
        "--label", default="", help="human-readable label stored in the baseline"
    )

    p_cmp = sub.add_parser(
        "compare", help="grade a candidate export against a baseline"
    )
    p_cmp.add_argument(
        "baseline", help="baseline .json from `vecdrift snapshot`, or a raw export"
    )
    p_cmp.add_argument("candidate", help="candidate vector export (or baseline)")
    p_cmp.add_argument(
        "-k", type=int, default=10, help="neighborhood size for overlap@k (default 10)"
    )
    p_cmp.add_argument("--json", action="store_true", help="emit a JSON report")
    p_cmp.add_argument(
        "--fail-on",
        choices=("never", "warn", "re-embed"),  # in severity order
        default="re-embed",
        help="verdict level that makes the exit code 1 (default: re-embed)",
    )
    p_cmp.add_argument(
        "--worst", type=int, default=5, help="how many worst anchors to list (default 5)"
    )
    gates = p_cmp.add_argument_group(
        "verdict gates", "inclusive thresholds; see docs/metrics.md for the defaults' rationale"
    )
    for flag, default, text in (
        ("--ok-overlap", 0.95, "minimum mean overlap@k for OK"),
        ("--ok-correlation", 0.995, "minimum pairwise-similarity pearson for OK"),
        ("--ok-delta", 0.02, "maximum mean |delta similarity| for OK"),
        ("--warn-overlap", 0.80, "minimum mean overlap@k for WARN; below => RE-EMBED"),
        ("--warn-correlation", 0.97, "minimum pearson for WARN; below => RE-EMBED"),
        ("--warn-delta", 0.05, "maximum mean |delta similarity| for WARN; above => RE-EMBED"),
    ):
        gates.add_argument(
            flag, type=float, default=default, metavar="X",
            help=f"{text} (default {default})",
        )

    p_ins = sub.add_parser("inspect", help="sanity-check a vector export")
    p_ins.add_argument("vectors", help="vector export (.jsonl / .json / .csv)")
    p_ins.add_argument(
        "--dupes", type=int, default=3,
        help="how many highest-similarity pairs to list (default 3, 0 to skip)",
    )

    p_pick = sub.add_parser(
        "pick", help="select a diverse anchor subset (farthest-point sampling)"
    )
    p_pick.add_argument("vectors", help="vector export to pick from")
    p_pick.add_argument("-n", "--count", type=int, required=True, help="anchors to keep")
    p_pick.add_argument("-o", "--output", required=True, help="output .jsonl path")

    return parser


def _cmd_snapshot(args: argparse.Namespace) -> int:
    vector_set = load_vectors(args.vectors)
    base = snapshot(vector_set, label=args.label or args.vectors)
    base.save(args.output)
    print(
        f"baseline written: {args.output} "
        f"({len(base)} anchor{'' if len(base) == 1 else 's'}, dim {base.dim}, "
        f"{len(base.sims)} pair similarit{'y' if len(base.sims) == 1 else 'ies'})"
    )
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    thresholds = Thresholds(
        ok_overlap=args.ok_overlap,
        ok_correlation=args.ok_correlation,
        ok_delta=args.ok_delta,
        warn_overlap=args.warn_overlap,
        warn_correlation=args.warn_correlation,
        warn_delta=args.warn_delta,
    )
    if args.k < 1:
        raise VecdriftError("-k must be >= 1")
    base = load_reference(args.baseline)
    cand = load_reference(args.candidate)
    report = compare(base, cand, k=args.k, thresholds=thresholds)
    if args.json:
        print(render_json(report, worst=args.worst))
    else:
        print(render_text(report, worst=args.worst))
    if report.verdict.severity >= _FAIL_LEVELS[args.fail_on]:
        return 1
    return 0


def _cmd_inspect(args: argparse.Namespace) -> int:
    vector_set = load_vectors(args.vectors)
    norms = [norm(vector) for vector in vector_set.vectors]
    print(f"{args.vectors}")
    print(f"  anchors : {len(vector_set)}")
    print(f"  dim     : {vector_set.dim}")
    print(
        f"  norms   : mean {mean(norms):.4f}  std {pstdev(norms):.4f}"
        f"  min {min(norms):.4f}  max {max(norms):.4f}"
    )
    if args.dupes > 0 and len(vector_set) >= 2:
        geometry = Geometry.from_vectors(vector_set)
        top = sorted(
            geometry.pairs(),
            key=lambda pair: (-pair[2], geometry.ids[pair[0]], geometry.ids[pair[1]]),
        )[: args.dupes]
        print("  closest pairs (near-duplicate check):")
        for i, j, sim in top:
            print(f"    {geometry.ids[i]} vs {geometry.ids[j]}  cosine {sim:.4f}")
    return 0


def _cmd_pick(args: argparse.Namespace) -> int:
    vector_set = load_vectors(args.vectors)
    if args.count < 1:
        raise VecdriftError("--count must be >= 1")
    picked = pick_anchors(vector_set, args.count)
    write_jsonl(picked, args.output)
    print(
        f"picked {len(picked)} of {len(vector_set)} anchors -> {args.output} "
        f"(farthest-point sampling, deterministic)"
    )
    return 0


_COMMANDS = {
    "snapshot": _cmd_snapshot,
    "compare": _cmd_compare,
    "inspect": _cmd_inspect,
    "pick": _cmd_pick,
}


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 2
    try:
        return _COMMANDS[args.command](args)
    except VecdriftError as exc:
        print(f"vecdrift: error: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"vecdrift: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
