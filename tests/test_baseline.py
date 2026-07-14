"""Tests for baseline snapshotting, serialization, and detection."""

import json

import pytest

from vecdrift.baseline import (
    FORMAT_VERSION,
    load_baseline,
    load_reference,
    snapshot,
)
from vecdrift.errors import BaselineError, InputError
from vecdrift.geometry import Geometry, condensed_length

from conftest import make_set


def small_set():
    return make_set([("a", [3, 4]), ("b", [1, 0]), ("c", [0, 2])])


def broken_copy(tmp_path, name, mutate):
    """Save a valid baseline dict with one field corrupted."""
    data = snapshot(small_set()).to_dict()
    mutate(data)
    path = tmp_path / name
    path.write_text(json.dumps(data))
    return path


def test_snapshot_captures_ids_dim_norms_and_rounded_sims():
    base = snapshot(small_set(), label="v1")
    assert base.ids == ["a", "b", "c"]
    assert base.dim == 2
    assert base.norms == [5.0, 1.0, 2.0]
    assert len(base.sims) == condensed_length(3)
    assert base.label == "v1"
    # cos of [1,1] vs [1,0] = 1/sqrt(2) = 0.70710678... -> stored as 0.707107
    rounded = snapshot(make_set([("a", [1, 1]), ("b", [1, 0]), ("c", [0, 1])]))
    assert rounded.sims[0] == 0.707107


def test_save_load_round_trip_is_stable(tmp_path):
    base = snapshot(small_set(), label="round-trip")
    p1, p2 = tmp_path / "a.json", tmp_path / "b.json"
    base.save(p1)
    base.save(p2)
    loaded = load_baseline(p1)
    assert (loaded.ids, loaded.dim, loaded.norms, loaded.sims, loaded.label) == (
        base.ids, base.dim, base.norms, base.sims, "round-trip",
    )
    # sort_keys + trailing newline: saved twice, byte-identical (diffable).
    assert p1.read_bytes() == p2.read_bytes()
    assert p1.read_text().endswith("\n")


def test_load_rejects_non_baseline_json_and_invalid_json(tmp_path):
    other = tmp_path / "other.json"
    other.write_text(json.dumps({"hello": "world"}))
    with pytest.raises(BaselineError, match="not a vecdrift-baseline"):
        load_baseline(other)
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    with pytest.raises(BaselineError, match="invalid JSON"):
        load_baseline(bad)


def test_load_rejects_future_format_version(tmp_path):
    path = broken_copy(
        tmp_path, "future.json", lambda d: d.update(version=FORMAT_VERSION + 1)
    )
    with pytest.raises(BaselineError, match="not supported"):
        load_baseline(path)


def test_load_rejects_structurally_broken_baselines(tmp_path):
    cases = [
        ("missing.json", lambda d: d.pop("norms"), "'norms'"),
        ("dupes.json", lambda d: d.update(ids=["a", "a", "c"]), "duplicate ids"),
        ("short.json", lambda d: d.update(pair_sims=d["pair_sims"][:-1]), "pair_sims"),
    ]
    for name, mutate, message in cases:
        path = broken_copy(tmp_path, name, mutate)
        with pytest.raises(BaselineError, match=message):
            load_baseline(path)


def test_load_reference_detects_a_saved_baseline(tmp_path):
    path = tmp_path / "baseline.json"
    snapshot(small_set(), label="stored").save(path)
    assert load_reference(path).label == "stored"


def test_load_reference_snapshots_raw_exports(tmp_path):
    jsonl = tmp_path / "export.jsonl"
    jsonl.write_text('{"id": "a", "vector": [1, 0]}\n{"id": "b", "vector": [0, 1]}\n')
    ref = load_reference(jsonl)
    assert ref.ids == ["a", "b"]
    assert ref.label == "export"  # falls back to the file stem
    # A .json file WITHOUT the baseline marker is treated as a vector export.
    raw = tmp_path / "raw.json"
    raw.write_text(json.dumps({"a": [1, 0], "b": [0, 1]}))
    assert sorted(load_reference(raw).ids) == ["a", "b"]


def test_load_reference_missing_file_raises_input_error(tmp_path):
    with pytest.raises(InputError):
        load_reference(tmp_path / "missing.json")


def test_geometry_reconstructs_from_a_baseline_to_rounding_precision():
    vs = small_set()
    stored = snapshot(vs).geometry()
    direct = Geometry.from_vectors(vs)
    for a, b in zip(stored.sims, direct.sims):
        assert a == pytest.approx(b, abs=1e-6)
