"""Tests for the condensed pairwise-similarity structure."""

import pytest

from vecdrift.geometry import Geometry, condensed_index, condensed_length

from conftest import make_set


def square_set():
    """Four unit vectors at 0/90/180/270 degrees: hand-checkable geometry."""
    return make_set(
        [("e", [1, 0]), ("n", [0, 1]), ("w", [-1, 0]), ("s", [0, -1])]
    )


def test_condensed_indexing_enumerates_the_upper_triangle():
    assert condensed_length(1) == 0
    assert condensed_length(4) == 6
    assert condensed_length(48) == 1128
    n = 5
    seen = [condensed_index(i, j, n) for i in range(n) for j in range(i + 1, n)]
    assert seen == list(range(condensed_length(n)))
    assert condensed_index(1, 3, 5) == condensed_index(3, 1, 5)
    with pytest.raises(ValueError):
        condensed_index(2, 2, 5)  # no self-pairs
    with pytest.raises(IndexError):
        condensed_index(0, 5, 5)  # out of range


def test_from_vectors_computes_hand_checked_cosines():
    geo = Geometry.from_vectors(square_set())
    assert geo.sim_by_id("e", "n") == pytest.approx(0.0)
    assert geo.sim_by_id("e", "w") == pytest.approx(-1.0)
    assert geo.sim_by_id("n", "s") == pytest.approx(-1.0)
    assert geo.sim_by_id("e", "s") == pytest.approx(0.0)


def test_sim_accessor_matches_pairs_iteration_in_both_argument_orders():
    geo = Geometry.from_vectors(square_set())
    assert len(geo) == 4
    for i, j, sim in geo.pairs():
        assert geo.sim(i, j) == sim
        assert geo.sim(j, i) == sim  # order of arguments must not matter


def test_constructor_rejects_wrong_sims_length():
    with pytest.raises(ValueError, match="pair similarities"):
        Geometry(["a", "b", "c"], [0.5])


def test_neighbors_are_ordered_by_similarity():
    vs = make_set(
        [("a", [1, 0]), ("close", [0.9, 0.1]), ("mid", [0.5, 0.5]), ("far", [-1, 0])]
    )
    geo = Geometry.from_vectors(vs)
    assert geo.neighbor_ids("a") == ["close", "mid", "far"]


def test_neighbor_ties_break_by_ascending_id():
    # n and s are both exactly orthogonal to e; w is opposite. The tie
    # between n and s must resolve alphabetically, deterministically.
    geo = Geometry.from_vectors(square_set())
    assert geo.neighbor_ids("e") == ["n", "s", "w"]


def test_subset_slices_without_recomputation(corpus):
    geo = Geometry.from_vectors(corpus)
    keep = ["doc-02", "doc-07", "doc-11", "doc-16"]
    sliced = geo.subset(keep)
    recomputed = Geometry.from_vectors(corpus.subset(keep))
    assert sliced.ids == recomputed.ids
    for a, b in zip(sliced.sims, recomputed.sims):
        assert a == pytest.approx(b, abs=1e-12)
    with pytest.raises(KeyError):
        geo.subset(["doc-02", "nope"])
