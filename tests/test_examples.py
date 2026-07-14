"""The shipped example must actually demonstrate the tool's claims."""

import runpy
import sys
from pathlib import Path

import pytest

from vecdrift import compare, load_vectors, snapshot
from vecdrift.verdict import Verdict

SCRIPT = Path(__file__).resolve().parent.parent / "examples" / "generate_exports.py"


def run_generator(out_dir: Path) -> None:
    """Execute the example script exactly as `python generate_exports.py`."""
    argv = sys.argv
    sys.argv = [str(SCRIPT), str(out_dir)]
    try:
        with pytest.raises(SystemExit) as excinfo:
            runpy.run_path(str(SCRIPT), run_name="__main__")
        assert excinfo.value.code == 0
    finally:
        sys.argv = argv


@pytest.fixture(scope="module")
def example_dir(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("exports")
    run_generator(out)
    return out


def test_generator_writes_three_deterministic_exports(tmp_path, example_dir):
    run_generator(tmp_path)
    for name in ("model_v1.jsonl", "model_v2.jsonl", "model_v3.jsonl"):
        assert (example_dir / name).exists()
        assert (tmp_path / name).read_bytes() == (example_dir / name).read_bytes()


def test_example_corpus_shape(example_dir):
    v1 = load_vectors(example_dir / "model_v1.jsonl")
    v3 = load_vectors(example_dir / "model_v3.jsonl")
    assert len(v1) == 48 and v1.dim == 8
    assert len(v3) == 48 and v3.dim == 12
    assert v1.ids == v3.ids


def test_example_v2_upgrade_is_ok(example_dir):
    base = snapshot(load_vectors(example_dir / "model_v1.jsonl"), "v1")
    cand = snapshot(load_vectors(example_dir / "model_v2.jsonl"), "v2")
    report = compare(base, cand)
    assert report.verdict is Verdict.OK
    assert report.mean_overlap == 1.0


def test_example_v3_upgrade_demands_reembed(example_dir):
    base = snapshot(load_vectors(example_dir / "model_v1.jsonl"), "v1")
    cand = snapshot(load_vectors(example_dir / "model_v3.jsonl"), "v3")
    report = compare(base, cand)
    assert report.verdict is Verdict.REEMBED
    # The generator drifted exactly these documents; the worst anchors
    # reported must come from that set — vecdrift names the right culprits.
    drifted = {f"doc-{i:02d}" for i in (3, 7, 11, 15, 19, 23, 27, 31, 35, 39)}
    assert {anchor.id for anchor in report.worst_anchors(5)} <= drifted
