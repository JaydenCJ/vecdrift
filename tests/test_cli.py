"""End-to-end tests for the CLI entry point (in-process, no subprocess)."""

import json

import pytest

from vecdrift import __version__
from vecdrift.cli import main

from conftest import clustered_vectors, make_set, rotate


@pytest.fixture()
def exports(tmp_path):
    """Write a baseline export, a clean upgrade, and a damaged upgrade."""
    from vecdrift.vectors import write_jsonl

    corpus = clustered_vectors()
    good = rotate(corpus, seed=8, rotations=16, scale=1.3)
    vectors = [list(v) for v in corpus.vectors]
    for i in range(7):
        vectors[i] = [(-1.0) ** (i + j) * (j + 1.0) for j in range(corpus.dim)]
    bad = make_set(list(zip(corpus.ids, vectors)))

    paths = {}
    for name, vs in (("v1", corpus), ("good", good), ("bad", bad)):
        path = tmp_path / f"{name}.jsonl"
        write_jsonl(vs, path)
        paths[name] = str(path)
    return paths


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert excinfo.value.code == 0
    assert capsys.readouterr().out.strip() == f"vecdrift {__version__}"


def test_no_command_prints_help_and_exits_2(capsys):
    assert main([]) == 2
    assert "snapshot" in capsys.readouterr().out


def test_compare_help_documents_every_verdict_gate_flag(capsys):
    # The README's "Verdict gates" table promises these flags; keep --help
    # honest so users never have to read the source to discover a threshold.
    with pytest.raises(SystemExit) as excinfo:
        main(["compare", "--help"])
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    for flag in ("--ok-overlap", "--ok-correlation", "--ok-delta",
                 "--warn-overlap", "--warn-correlation", "--warn-delta"):
        assert flag in out
    assert "{never,warn,re-embed}" in out  # --fail-on choices, severity order


def test_snapshot_writes_a_labelled_baseline(tmp_path, exports, capsys):
    out = tmp_path / "baseline.json"
    rc = main(["snapshot", exports["v1"], "-o", str(out), "--label", "model-v1"])
    assert rc == 0
    assert "18 anchors" in capsys.readouterr().out
    data = json.loads(out.read_text())
    assert data["format"] == "vecdrift-baseline"
    assert data["label"] == "model-v1"
    assert data["count"] == 18


def test_compare_ok_exits_zero(tmp_path, exports, capsys):
    base = tmp_path / "baseline.json"
    main(["snapshot", exports["v1"], "-o", str(base)])
    capsys.readouterr()
    rc = main(["compare", str(base), exports["good"]])
    out = capsys.readouterr().out
    assert rc == 0
    assert "verdict: OK" in out


def test_compare_drift_exits_one_and_lists_worst_anchors(tmp_path, exports, capsys):
    base = tmp_path / "baseline.json"
    main(["snapshot", exports["v1"], "-o", str(base)])
    rc = main(["compare", str(base), exports["bad"]])
    out = capsys.readouterr().out
    assert rc == 1
    assert "verdict: RE-EMBED" in out
    assert "worst anchors" in out


def test_compare_fail_on_gates_the_exit_code(tmp_path, exports, capsys):
    base = tmp_path / "baseline.json"
    main(["snapshot", exports["v1"], "-o", str(base)])
    capsys.readouterr()
    # --fail-on never: even RE-EMBED reports, but the exit code stays 0.
    assert main(["compare", str(base), exports["bad"], "--fail-on", "never"]) == 0
    capsys.readouterr()
    # Relax the warn gates so the damaged export grades WARN, then fail on it.
    args = [
        "compare", str(base), exports["bad"],
        "--warn-overlap", "0.1", "--warn-correlation", "0.0", "--warn-delta", "10",
    ]
    assert main(args) == 0  # default --fail-on re-embed lets WARN pass
    assert "verdict: WARN" in capsys.readouterr().out
    assert main(args + ["--fail-on", "warn"]) == 1


def test_compare_json_output_parses(tmp_path, exports, capsys):
    base = tmp_path / "baseline.json"
    main(["snapshot", exports["v1"], "-o", str(base)])
    capsys.readouterr()
    rc = main(["compare", str(base), exports["bad"], "--json"])
    data = json.loads(capsys.readouterr().out)
    assert rc == 1
    assert data["verdict"] == "RE-EMBED"
    assert data["alignment"]["matched"] == 18


def test_compare_two_raw_exports_without_a_baseline(exports, capsys):
    # Both sides can be raw exports: snapshotting happens in memory.
    rc = main(["compare", exports["v1"], exports["good"]])
    assert rc == 0
    assert "verdict: OK" in capsys.readouterr().out


def test_compare_error_paths_exit_two(tmp_path, exports, capsys):
    assert main(["compare", str(tmp_path / "no.json"), exports["v1"]]) == 2
    assert "vecdrift: error:" in capsys.readouterr().err
    assert main(["compare", exports["v1"], exports["good"], "-k", "0"]) == 2
    assert "-k must be >= 1" in capsys.readouterr().err
    rc = main([
        "compare", exports["v1"], exports["good"],
        "--ok-overlap", "0.5", "--warn-overlap", "0.9",
    ])
    assert rc == 2
    assert "warn_overlap" in capsys.readouterr().err


def test_inspect_prints_stats_and_optionally_near_duplicates(exports, capsys):
    assert main(["inspect", exports["v1"]]) == 0
    out = capsys.readouterr().out
    assert "anchors : 18" in out
    assert "dim     : 6" in out
    assert "closest pairs" in out
    assert main(["inspect", exports["v1"], "--dupes", "0"]) == 0
    assert "closest pairs" not in capsys.readouterr().out


def test_pick_writes_requested_count_deterministically(tmp_path, exports, capsys):
    p1, p2 = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    assert main(["pick", exports["v1"], "-n", "6", "-o", str(p1)]) == 0
    assert "picked 6 of 18" in capsys.readouterr().out
    assert len(p1.read_text().strip().splitlines()) == 6
    main(["pick", exports["v1"], "-n", "6", "-o", str(p2)])
    assert p1.read_bytes() == p2.read_bytes()


def test_pick_count_below_one_exits_two(tmp_path, exports, capsys):
    rc = main(["pick", exports["v1"], "-n", "0", "-o", str(tmp_path / "x.jsonl")])
    assert rc == 2
    assert "--count" in capsys.readouterr().err
