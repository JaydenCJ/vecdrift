"""Tests for deterministic farthest-point anchor selection."""

import pytest

from vecdrift.pick import pick_anchors

from conftest import clustered_vectors, make_set


def test_pick_count_handling(corpus):
    assert len(pick_anchors(corpus, 5)) == 5
    assert pick_anchors(corpus, 999).ids == corpus.ids  # count > size -> all
    with pytest.raises(ValueError, match="count must be"):
        pick_anchors(corpus, 0)


def test_pick_preserves_original_file_order_and_vectors(corpus):
    picked = pick_anchors(corpus, 7)
    positions = [corpus.ids.index(anchor_id) for anchor_id in picked.ids]
    assert positions == sorted(positions)
    for anchor_id in picked.ids:
        assert picked.vector(anchor_id) == corpus.vector(anchor_id)


def test_pick_covers_every_cluster():
    # 3 well-separated clusters; picking 3 anchors must take one from each —
    # that is the whole point of farthest-point sampling for drift coverage.
    corpus = clustered_vectors(n_clusters=3, per_cluster=6, jitter=0.05)
    picked = pick_anchors(corpus, 3)
    clusters = {corpus.ids.index(anchor_id) // 6 for anchor_id in picked.ids}
    assert clusters == {0, 1, 2}


def test_pick_is_input_order_independent():
    corpus = clustered_vectors(seed=77)
    reversed_pairs = list(zip(corpus.ids, corpus.vectors))[::-1]
    shuffled = make_set(reversed_pairs)
    a = pick_anchors(corpus, 6)
    b = pick_anchors(shuffled, 6)
    assert sorted(a.ids) == sorted(b.ids)


def test_pick_handles_cancelling_vectors_deterministically():
    # Vectors sum to zero -> centroid is the zero vector; the seed anchor
    # falls back to the smallest id instead of crashing on cosine(0).
    vs = make_set(
        [("b", [1.0, 0.0]), ("a", [-1.0, 0.0]), ("c", [0.0, 1.0]), ("d", [0.0, -1.0])]
    )
    picked = pick_anchors(vs, 2)
    assert "a" in picked.ids
