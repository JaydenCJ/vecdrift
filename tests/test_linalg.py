"""Unit tests for the pure vector-math and rank-statistics primitives."""

import math

import pytest

from vecdrift.linalg import (
    cosine,
    dot,
    mean,
    norm,
    pearson,
    pstdev,
    rankdata,
    spearman,
)


def test_dot_and_norm_on_known_values():
    assert dot([1.0, 0.0], [0.0, 1.0]) == 0.0
    assert dot([1.0, 2.0], [3.0, 4.0]) == pytest.approx(11.0)
    assert norm([3.0, 4.0]) == pytest.approx(5.0)
    with pytest.raises(ValueError, match="dimension mismatch"):
        dot([1.0], [1.0, 2.0])


def test_cosine_on_parallel_orthogonal_and_opposite_vectors():
    assert cosine([2.0, 2.0], [5.0, 5.0]) == pytest.approx(1.0)
    assert cosine([1.0, 0.0], [0.0, 3.0]) == pytest.approx(0.0)
    assert cosine([1.0, 0.0], [-3.0, 0.0]) == pytest.approx(-1.0)


def test_cosine_is_scale_invariant():
    a = [0.3, -1.2, 0.7]
    b = [1.1, 0.4, -0.2]
    assert cosine(a, b) == pytest.approx(cosine([10 * x for x in a], b))


def test_cosine_rejects_zero_vectors_and_clamps_overshoot():
    # A zero vector has no direction; failing loudly beats returning garbage.
    with pytest.raises(ValueError, match="zero vector"):
        cosine([0.0, 0.0], [1.0, 0.0])
    # Nearly-identical vectors can produce 1.0000000000000002 in float math.
    a = [0.1] * 50
    assert cosine(a, a) <= 1.0


def test_mean_and_pstdev_on_known_values():
    values = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
    assert mean(values) == pytest.approx(5.0)
    assert pstdev(values) == pytest.approx(2.0)
    with pytest.raises(ValueError, match="empty"):
        mean([])


def test_pearson_detects_perfect_positive_and_negative_correlation():
    assert pearson([1.0, 2.0, 3.0], [10.0, 20.0, 30.0]) == pytest.approx(1.0)
    assert pearson([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]) == pytest.approx(-1.0)
    # Tiny values must not overshoot 1.0 through float rounding.
    xs = [1e-9 * i for i in range(10)]
    result = pearson(xs, xs)
    assert result is not None and -1.0 <= result <= 1.0


def test_pearson_edge_cases():
    # Zero variance or a single pair: correlation carries no information,
    # and None (not 0.0 or 1.0) is the honest answer downstream code checks.
    assert pearson([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]) is None
    assert pearson([1.0], [2.0]) is None
    with pytest.raises(ValueError, match="length mismatch"):
        pearson([1.0, 2.0], [1.0])


def test_rankdata_orders_and_averages_ties():
    assert rankdata([30.0, 10.0, 20.0]) == [3.0, 1.0, 2.0]
    # The two 20s occupy positions 2 and 3 -> both get rank 2.5.
    assert rankdata([10.0, 20.0, 20.0, 40.0]) == [1.0, 2.5, 2.5, 4.0]
    assert rankdata([7.0, 7.0, 7.0]) == [2.0, 2.0, 2.0]


def test_spearman_is_rank_based():
    xs = [1.0, 2.0, 3.0, 4.0]
    ys = [math.exp(x) for x in xs]  # nonlinear but monotonic -> exactly 1
    assert spearman(xs, ys) == pytest.approx(1.0)
    assert spearman([1.0, 2.0, 3.0], [9.0, 5.0, 1.0]) == pytest.approx(-1.0)
    assert spearman([5.0, 5.0, 5.0], [1.0, 2.0, 3.0]) is None
