"""Tests for the neighborhood metrics (overlap@k, rank shift)."""

import pytest

from vecdrift.metrics import effective_k, overlap_at_k, rank_shift


def test_effective_k_clamps_to_available_neighbors_and_rejects_zero():
    assert effective_k(10, 5) == 4
    assert effective_k(3, 100) == 3
    with pytest.raises(ValueError, match="k must be"):
        effective_k(0, 10)


def test_overlap_is_a_set_metric_over_the_top_k_window():
    order = ["a", "b", "c", "d"]
    assert overlap_at_k(order, order, 3) == 1.0
    # Reshuffling inside the window is invisible: overlap ignores order.
    assert overlap_at_k(order, ["c", "a", "b", "d"], 3) == 1.0


def test_overlap_measures_partial_and_total_neighborhood_loss():
    assert overlap_at_k(["a", "b", "x", "y"], ["x", "y", "a", "b"], 2) == 0.0
    # top-3 of base = {a,b,c}; of cand = {a,x,y} -> 1 shared of 3.
    assert overlap_at_k(
        ["a", "b", "c", "x", "y"], ["a", "x", "y", "b", "c"], 3
    ) == pytest.approx(1 / 3)


def test_overlap_clamps_k_and_rejects_k_below_one():
    assert overlap_at_k(["a", "b"], ["b", "a"], 10) == 1.0
    with pytest.raises(ValueError):
        overlap_at_k(["a"], ["a"], 0)


def test_rank_shift_measures_displacement_of_close_neighbors():
    order = ["a", "b", "c", "d"]
    assert rank_shift(order, order, 4) == 0.0
    # a<->b swapped: each moved by exactly 1 position; c unmoved.
    assert rank_shift(["a", "b", "c"], ["b", "a", "c"], 3) == pytest.approx(2 / 3)
    # base top-1 "a" fell to position 3 in the candidate: shift of 2.
    assert rank_shift(["a", "b", "c"], ["b", "c", "a"], 1) == pytest.approx(2.0)


def test_rank_shift_clamps_k_and_requires_ids_in_candidate():
    assert rank_shift(["a", "b"], ["a", "b"], 99) == 0.0
    with pytest.raises(ValueError, match="missing from candidate"):
        rank_shift(["a", "b"], ["b", "z"], 2)
