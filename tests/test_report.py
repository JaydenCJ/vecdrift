"""Tests for text and JSON report rendering."""

import json

import pytest

from vecdrift.baseline import snapshot
from vecdrift.compare import compare
from vecdrift.report import render_json, render_text, report_to_dict

from conftest import clustered_vectors, make_set


@pytest.fixture()
def ok_report():
    corpus = clustered_vectors()
    return compare(snapshot(corpus, "v1"), snapshot(corpus, "v2"))


@pytest.fixture()
def drift_report():
    corpus = clustered_vectors()
    vectors = [list(v) for v in corpus.vectors]
    for i in range(6):
        vectors[i] = [(-1.0) ** (i + j) * (j + 1.0) for j in range(corpus.dim)]
    damaged = make_set(list(zip(corpus.ids, vectors)))
    return compare(snapshot(corpus, "v1"), snapshot(damaged, "v2"))


def test_text_report_contains_headline_metrics(ok_report):
    text = render_text(ok_report)
    assert "verdict: OK" in text
    assert "mean overlap@10" in text
    assert "pearson" in text and "spearman" in text
    assert "v1" in text and "v2" in text


def test_text_report_reserves_worst_anchor_list_for_drift(ok_report, drift_report):
    assert "worst anchors" not in render_text(ok_report)
    text = render_text(drift_report)
    assert "worst anchors" in text
    assert "doc-00" in text
    assert "threshold" in text  # the reasons explain which gate failed


def test_text_report_marks_undefined_correlation(ok_report):
    ok_report.pearson = None
    assert "n/a (zero-variance geometry)" in render_text(ok_report)


def test_json_report_is_valid_and_sorted(ok_report):
    payload = render_json(ok_report)
    data = json.loads(payload)
    assert data["verdict"] == "OK"
    # sort_keys means re-serializing round-trips byte-for-byte: diffable.
    assert json.dumps(data, sort_keys=True, indent=2) == payload


def test_json_report_structure_and_worst_limit(drift_report):
    data = report_to_dict(drift_report)
    assert set(data) == {
        "baseline", "candidate", "alignment", "metrics", "worst_anchors",
        "verdict", "reasons",
    }
    assert data["metrics"]["k"] == 10
    assert data["alignment"]["matched"] == 18
    assert len(data["worst_anchors"]) == 5
    assert data["verdict"] in ("WARN", "RE-EMBED")
    assert data["reasons"]
    assert len(report_to_dict(drift_report, worst=2)["worst_anchors"]) == 2


def test_norm_block_only_rendered_for_same_dim(ok_report):
    assert "vector norms (same dim, comparable)" in render_text(ok_report)
    ok_report.candidate_dim = ok_report.baseline_dim + 2
    assert "vector norms" not in render_text(ok_report)
