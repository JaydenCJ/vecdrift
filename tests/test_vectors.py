"""Tests for loading, validating, and writing vector exports."""

import json

import pytest

from vecdrift.errors import InputError
from vecdrift.vectors import load_vectors, write_jsonl

from conftest import make_set


# --- JSONL --------------------------------------------------------------


def test_load_jsonl_basic_and_integer_id_coercion(tmp_path):
    path = tmp_path / "export.jsonl"
    path.write_text(
        '{"id": "a", "vector": [1.0, 0.0]}\n{"id": 7, "vector": [0.0, 1.0]}\n'
    )
    vs = load_vectors(path)
    assert vs.ids == ["a", "7"]
    assert vs.dim == 2
    assert vs.vector("7") == [0.0, 1.0]


def test_load_jsonl_ignores_extra_keys_and_blank_lines(tmp_path):
    # Real datastore dumps carry text/tags/metadata; they must load as-is.
    path = tmp_path / "export.jsonl"
    path.write_text(
        '{"id": "a", "vector": [1, 2], "text": "hello", "tags": ["x"]}\n'
        "\n"
        '{"id": "b", "vector": [3, 4]}\n'
    )
    vs = load_vectors(path)
    assert len(vs) == 2
    assert vs.vector("a") == [1.0, 2.0]


def test_load_jsonl_reports_line_numbers_and_missing_fields(tmp_path):
    bad_json = tmp_path / "bad.jsonl"
    bad_json.write_text('{"id": "a", "vector": [1, 0]}\n{not json}\n')
    with pytest.raises(InputError, match=r":2"):
        load_vectors(bad_json)
    misnamed = tmp_path / "misnamed.jsonl"
    misnamed.write_text('{"id": "a", "embedding": [1, 0]}\n')
    with pytest.raises(InputError, match='missing "vector"'):
        load_vectors(misnamed)


# --- JSON ---------------------------------------------------------------


def test_load_json_accepts_all_three_shapes(tmp_path):
    list_form = tmp_path / "list.json"
    list_form.write_text(
        json.dumps([{"id": "a", "vector": [1, 0]}, {"id": "b", "vector": [0, 1]}])
    )
    keyed_form = tmp_path / "keyed.json"
    keyed_form.write_text(json.dumps({"vectors": [{"id": "a", "vector": [1, 2]}]}))
    mapping_form = tmp_path / "map.json"
    mapping_form.write_text(json.dumps({"a": [1, 0], "b": [0, 1]}))
    assert load_vectors(list_form).ids == ["a", "b"]
    assert load_vectors(keyed_form).vector("a") == [1.0, 2.0]
    assert sorted(load_vectors(mapping_form).ids) == ["a", "b"]
    scalar = tmp_path / "scalar.json"
    scalar.write_text("42")
    with pytest.raises(InputError, match="top level"):
        load_vectors(scalar)


# --- CSV ----------------------------------------------------------------


def test_load_csv_basic(tmp_path):
    path = tmp_path / "export.csv"
    path.write_text("id,v0,v1,v2\na,1,0,0\nb,0,1,0\n")
    vs = load_vectors(path)
    assert vs.ids == ["a", "b"]
    assert vs.dim == 3


def test_load_csv_rejects_bad_header_ragged_and_non_numeric_rows(tmp_path):
    unnamed = tmp_path / "unnamed.csv"
    unnamed.write_text("name,v0\na,1\n")
    with pytest.raises(InputError, match='must be named "id"'):
        load_vectors(unnamed)
    ragged = tmp_path / "ragged.csv"
    ragged.write_text("id,v0,v1\na,1,2\nb,3\n")
    with pytest.raises(InputError, match="expected 3 columns"):
        load_vectors(ragged)
    words = tmp_path / "words.csv"
    words.write_text("id,v0,v1\na,1,oops\n")
    with pytest.raises(InputError, match="not a number"):
        load_vectors(words)


# --- shared validation ----------------------------------------------------


def test_duplicate_ids_and_ragged_dimensions_are_rejected(tmp_path):
    dupes = tmp_path / "dupes.jsonl"
    dupes.write_text('{"id": "a", "vector": [1, 0]}\n{"id": "a", "vector": [0, 1]}\n')
    with pytest.raises(InputError, match="duplicate id"):
        load_vectors(dupes)
    ragged = tmp_path / "ragged.jsonl"
    ragged.write_text('{"id": "a", "vector": [1, 0]}\n{"id": "b", "vector": [1, 0, 0]}\n')
    with pytest.raises(InputError, match="dimension 3 differs"):
        load_vectors(ragged)


def test_non_numeric_components_are_rejected(tmp_path):
    # Python's json module happily parses NaN/Infinity; we must not — and
    # true/false would silently coerce to 1.0/0.0, so they are refused too.
    nan = tmp_path / "nan.jsonl"
    nan.write_text('{"id": "a", "vector": [1.0, NaN]}\n')
    with pytest.raises(InputError, match="non-finite"):
        load_vectors(nan)
    boolean = tmp_path / "bool.jsonl"
    boolean.write_text('{"id": "a", "vector": [true, 1.0]}\n')
    with pytest.raises(InputError, match="not a number"):
        load_vectors(boolean)


def test_zero_vectors_and_empty_ids_are_rejected(tmp_path):
    zero = tmp_path / "zero.jsonl"
    zero.write_text('{"id": "a", "vector": [0.0, 0.0]}\n')
    with pytest.raises(InputError, match="zero vector"):
        load_vectors(zero)
    anon = tmp_path / "anon.jsonl"
    anon.write_text('{"id": "", "vector": [1, 0]}\n')
    with pytest.raises(InputError, match="empty id"):
        load_vectors(anon)


def test_empty_file_missing_file_and_bad_extension_are_rejected(tmp_path):
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    with pytest.raises(InputError, match="no vectors"):
        load_vectors(empty)
    with pytest.raises(InputError):
        load_vectors(tmp_path / "nope.jsonl")
    parquet = tmp_path / "export.parquet"
    parquet.write_text("whatever")
    with pytest.raises(InputError, match="unsupported extension"):
        load_vectors(parquet)


# --- VectorSet + round-trip -----------------------------------------------


def test_vector_set_subset_and_membership():
    vs = make_set([("a", [1, 0]), ("b", [0, 1]), ("c", [1, 1])])
    sub = vs.subset(["c", "a"])
    assert sub.ids == ["c", "a"]  # requested order is preserved
    assert sub.vector("c") == [1.0, 1.0]
    assert "a" in vs and "z" not in vs
    assert len(vs) == 3
    with pytest.raises(KeyError):
        vs.subset(["zz"])


def test_write_jsonl_round_trips(tmp_path):
    vs = make_set([("a", [1.5, -0.25]), ("b", [0.0, 2.0])])
    path = tmp_path / "out.jsonl"
    write_jsonl(vs, path)
    loaded = load_vectors(path)
    assert loaded.ids == vs.ids
    assert loaded.vectors == vs.vectors
