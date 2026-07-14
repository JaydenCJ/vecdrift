"""vecdrift — detect embedding-space drift across model versions.

vecdrift compares the *relative geometry* of a fixed anchor set under two
embedding models (or two versions of one model). Because it works on
pairwise cosine structure rather than raw coordinates, it handles different
dimensionalities and arbitrary rotations — and it runs fully offline on
exported vectors, with zero runtime dependencies.

Typical library use::

    from vecdrift import load_vectors, snapshot, compare

    base = snapshot(load_vectors("v1_export.jsonl"), label="model-v1")
    base.save("baseline.json")
    # ... later, after re-embedding the same anchors with the new model:
    report = compare(base, snapshot(load_vectors("v2_export.jsonl")))
    print(report.verdict.value)   # "OK" | "WARN" | "RE-EMBED"
"""

from .baseline import Baseline, load_baseline, load_reference, snapshot
from .compare import AnchorDrift, DriftReport, NormStats, compare
from .errors import BaselineError, InputError, PairingError, VecdriftError
from .geometry import Geometry
from .pick import pick_anchors
from .report import render_json, render_text, report_to_dict
from .vectors import VectorSet, load_vectors, write_jsonl
from .verdict import Thresholds, Verdict, evaluate

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "AnchorDrift",
    "Baseline",
    "BaselineError",
    "DriftReport",
    "Geometry",
    "InputError",
    "NormStats",
    "PairingError",
    "Thresholds",
    "VecdriftError",
    "VectorSet",
    "Verdict",
    "compare",
    "evaluate",
    "load_baseline",
    "load_reference",
    "load_vectors",
    "pick_anchors",
    "render_json",
    "render_text",
    "report_to_dict",
    "snapshot",
    "write_jsonl",
]
