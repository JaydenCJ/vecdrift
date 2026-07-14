"""Tests for the comparison engine — the invariants the tool rests on."""

import pytest

from vecdrift.baseline import snapshot
from vecdrift.compare import MIN_MATCHED, compare
from vecdrift.errors import PairingError
from vecdrift.verdict import Verdict

from conftest import clustered_vectors, make_set, rotate


def test_identical_exports_show_zero_drift(corpus):
    report = compare(snapshot(corpus, "v1"), snapshot(corpus, "v1-again"))
    assert report.verdict is Verdict.OK
    assert report.mean_delta == pytest.approx(0.0, abs=1e-9)
    assert report.mean_overlap == 1.0
    assert report.mean_rank_shift == 0.0
    assert report.pearson == pytest.approx(1.0)
    assert (report.baseline_label, report.candidate_label) == ("v1", "v1-again")


def test_rotation_and_scale_are_invisible(corpus):
    # THE core claim: cosine geometry survives any orthogonal map + uniform
    # scale, so a well-behaved model upgrade must compare as OK.
    upgraded = rotate(corpus, seed=9, rotations=20, scale=3.5)
    report = compare(snapshot(corpus), snapshot(upgraded))
    assert report.verdict is Verdict.OK
    assert report.pearson == pytest.approx(1.0, abs=1e-6)
    assert report.mean_overlap == 1.0


def test_dimension_change_alone_is_invisible(corpus):
    # Padding to a higher dim + rotating is still orthogonal: no drift —
    # this is what lets vecdrift compare a 384-dim model to a 1536-dim one.
    upgraded = rotate(corpus, seed=3, rotations=30, pad_to=11)
    report = compare(snapshot(corpus), snapshot(upgraded))
    assert report.baseline_dim == 6
    assert report.candidate_dim == 11
    assert report.verdict is Verdict.OK
    assert report.norms_comparable is False  # norms mean nothing across dims


def test_perturbing_one_anchor_names_that_anchor(corpus):
    vectors = [list(v) for v in corpus.vectors]
    victim = corpus.ids.index("doc-05")
    # Point doc-05 somewhere unrelated: its neighborhood must collapse.
    vectors[victim] = [(-1.0) ** i * 2.0 for i in range(corpus.dim)]
    damaged = make_set(list(zip(corpus.ids, vectors)))
    report = compare(snapshot(corpus), snapshot(damaged))
    assert report.min_overlap_id == "doc-05"
    assert report.worst_anchors(1)[0].id == "doc-05"
    assert report.min_overlap < report.mean_overlap


def test_verdict_degrades_monotonically_with_damage(corpus):
    def damage(count):
        vectors = [list(v) for v in corpus.vectors]
        for i in range(count):
            vectors[i] = [(-1.0) ** (i + j) * (j + 1.0) for j in range(corpus.dim)]
        return make_set(list(zip(corpus.ids, vectors)))

    severities = [
        compare(snapshot(corpus), snapshot(damage(count))).verdict.severity
        for count in (0, 2, 12)
    ]
    assert severities[0] == 0  # untouched -> OK
    assert severities == sorted(severities)
    assert severities[-1] == Verdict.REEMBED.severity


def test_alignment_reports_missing_and_extra_ids(corpus):
    cand = corpus.subset([i for i in corpus.ids if i not in ("doc-00", "doc-01")])
    extra = make_set(
        list(zip(cand.ids, cand.vectors)) + [("doc-99", [1.0] * corpus.dim)]
    )
    report = compare(snapshot(corpus), snapshot(extra))
    assert report.missing_ids == ["doc-00", "doc-01"]
    assert report.extra_ids == ["doc-99"]
    assert report.matched == len(corpus) - 2


def test_metrics_use_only_matched_anchors(corpus):
    # Dropping anchors must not by itself create drift on the survivors.
    cand = corpus.subset(corpus.ids[4:])
    report = compare(snapshot(corpus), snapshot(cand))
    assert report.verdict is Verdict.OK
    assert report.mean_delta == pytest.approx(0.0, abs=1e-6)


def test_too_few_shared_ids_raises_pairing_error():
    base = make_set([("a", [1, 0]), ("b", [0, 1]), ("c", [1, 1])])
    cand = make_set([("a", [1, 0]), ("x", [0, 1]), ("y", [1, 1])])
    with pytest.raises(PairingError, match="only 1 anchor id"):
        compare(snapshot(base), snapshot(cand))
    assert MIN_MATCHED == 3


def test_k_is_clamped_to_matched_minus_one():
    vs = make_set([("a", [1, 0]), ("b", [0.9, 0.1]), ("c", [0, 1]), ("d", [0.1, 0.9])])
    report = compare(snapshot(vs), snapshot(vs), k=50)
    assert report.k == 50
    assert report.k_effective == 3


def test_max_delta_pair_names_the_right_pair():
    base = make_set([("a", [1, 0]), ("b", [0, 1]), ("c", [1, 1])])
    # Move only c: pairs (a,c) and (b,c) change, (a,b) does not.
    cand = make_set([("a", [1, 0]), ("b", [0, 1]), ("c", [1, -1])])
    report = compare(snapshot(base), snapshot(cand))
    assert "c" in report.max_delta_pair
    assert report.max_delta > 0.5


def test_norm_stats_reflect_uniform_scaling(corpus):
    scaled = rotate(corpus, rotations=0, scale=2.0)
    report = compare(snapshot(corpus), snapshot(scaled))
    assert report.norms_comparable is True
    assert report.candidate_norms.mean == pytest.approx(
        2.0 * report.baseline_norms.mean, rel=1e-4
    )


def test_worst_anchors_sorted_by_overlap_then_delta(corpus):
    vectors = [list(v) for v in corpus.vectors]
    vectors[0] = [(-1.0) ** i * 3.0 for i in range(corpus.dim)]
    vectors[1] = [(-1.0) ** (i + 1) * 3.0 for i in range(corpus.dim)]
    report = compare(snapshot(corpus), snapshot(make_set(list(zip(corpus.ids, vectors)))))
    worst = report.worst_anchors(4)
    overlaps = [anchor.overlap for anchor in worst]
    assert overlaps == sorted(overlaps)
    assert {"doc-00", "doc-01"} <= {anchor.id for anchor in worst}


def test_compare_is_deterministic_across_runs():
    a = clustered_vectors(seed=1)
    b = clustered_vectors(seed=2)
    r1 = compare(snapshot(a), snapshot(b))
    r2 = compare(snapshot(a), snapshot(b))
    assert r1.mean_overlap == r2.mean_overlap
    assert r1.mean_delta == r2.mean_delta
    assert [x.id for x in r1.worst_anchors()] == [x.id for x in r2.worst_anchors()]
