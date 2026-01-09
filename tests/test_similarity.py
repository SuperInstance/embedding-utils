"""Tests for similarity module."""

import pytest
import numpy as np

from embedding_utils import (
    cosine_similarity,
    euclidean_distance,
    dot_product,
    manhattan_distance,
    jaccard_similarity,
    pairwise_similarity,
    find_similar,
    find_top_k,
)


class TestCosineSimilarity:
    """Test cosine similarity function."""

    def test_identical_vectors(self):
        """Test identical vectors have similarity 1."""
        result = cosine_similarity([1, 2, 3], [1, 2, 3])
        assert result == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        """Test orthogonal vectors have similarity 0."""
        result = cosine_similarity([1, 0, 0], [0, 1, 0])
        assert result == pytest.approx(0.0)

    def test_opposite_vectors(self):
        """Test opposite vectors have similarity -1."""
        result = cosine_similarity([1, 0, 0], [-1, 0, 0])
        assert result == pytest.approx(-1.0)

    def test_numpy_arrays(self):
        """Test with numpy arrays."""
        result = cosine_similarity(np.array([1, 2, 3]), np.array([1, 2, 3]))
        assert result == pytest.approx(1.0)

    def test_zero_vector(self):
        """Test with zero vector."""
        result = cosine_similarity([0, 0, 0], [1, 2, 3])
        assert result == 0.0


class TestEuclideanDistance:
    """Test Euclidean distance function."""

    def test_identical_vectors(self):
        """Test identical vectors have distance 0."""
        result = euclidean_distance([1, 2, 3], [1, 2, 3])
        assert result == pytest.approx(0.0)

    def test_unit_distance(self):
        """Test unit distance."""
        result = euclidean_distance([0, 0], [1, 0])
        assert result == pytest.approx(1.0)

    def test_diagonal_distance(self):
        """Test diagonal distance."""
        result = euclidean_distance([0, 0], [1, 1])
        assert result == pytest.approx(1.4142135623730951)


class TestDotProduct:
    """Test dot product function."""

    def test_simple_dot_product(self):
        """Test simple dot product."""
        result = dot_product([1, 2, 3], [4, 5, 6])
        assert result == pytest.approx(32.0)

    def test_zero_dot_product(self):
        """Test zero dot product."""
        result = dot_product([1, 0, 0], [0, 1, 0])
        assert result == pytest.approx(0.0)


class TestManhattanDistance:
    """Test Manhattan distance function."""

    def test_simple_manhattan(self):
        """Test simple Manhattan distance."""
        result = manhattan_distance([0, 0], [3, 4])
        assert result == pytest.approx(7.0)

    def test_identical_vectors(self):
        """Test identical vectors."""
        result = manhattan_distance([1, 2], [1, 2])
        assert result == pytest.approx(0.0)


class TestJaccardSimilarity:
    """Test Jaccard similarity function."""

    def test_identical_vectors(self):
        """Test identical vectors."""
        result = jaccard_similarity([1, 2, 3], [1, 2, 3])
        assert result == pytest.approx(1.0)

    def test_disjoint_vectors(self):
        """Test disjoint vectors."""
        result = jaccard_similarity([1, 0, 0], [0, 1, 0])
        assert result == pytest.approx(0.0)

    def test_overlapping_vectors(self):
        """Test overlapping vectors."""
        result = jaccard_similarity([1, 2], [2, 3])
        # min([1,2], [2,3]) = [1, 2], sum = 3
        # max([1,2], [2,3]) = [2, 3], sum = 5
        # 3/5 = 0.6
        assert result == pytest.approx(0.6)


class TestPairwiseSimilarity:
    """Test pairwise similarity function."""

    def test_pairwise_cosine(self):
        """Test pairwise cosine similarity."""
        vectors = [[1, 0, 0], [0, 1, 0], [1, 1, 0]]
        result = pairwise_similarity(vectors, metric="cosine")

        assert result.shape == (3, 3)
        assert result[0, 0] == pytest.approx(1.0)  # Identical to self
        assert result[0, 1] == pytest.approx(0.0)  # Orthogonal
        assert result[0, 2] == pytest.approx(0.70710678)  # 45 degrees

    def test_pairwise_euclidean(self):
        """Test pairwise Euclidean distance."""
        vectors = [[1, 0, 0], [0, 1, 0]]
        result = pairwise_similarity(vectors, metric="euclidean")

        assert result[0, 0] == pytest.approx(0.0)  # Distance to self
        assert result[0, 1] == pytest.approx(1.41421356)

    def test_symmetric(self):
        """Test that matrix is symmetric."""
        vectors = [[1, 2], [3, 4], [5, 6]]
        result = pairwise_similarity(vectors, metric="cosine")

        np.testing.assert_array_almost_equal(result, result.T)


class TestFindSimilar:
    """Test find_similar function."""

    def test_find_above_threshold(self):
        """Test finding vectors above threshold."""
        query = [1, 0, 0]
        candidates = [
            [1, 0, 0],      # similarity 1.0
            [0, 1, 0],      # similarity 0.0
            [0.9, 0.1, 0],  # similarity ~0.99
        ]
        result = find_similar(query, candidates, threshold=0.8)

        assert len(result) == 2
        assert result[0][0] == 0  # First candidate
        assert result[1][0] == 2  # Third candidate

    def test_empty_results(self):
        """Test with no matches above threshold."""
        query = [1, 0, 0]
        candidates = [[0, 1, 0], [0, 0, 1]]
        result = find_similar(query, candidates, threshold=0.9)

        assert len(result) == 0

    def test_sorted_by_score(self):
        """Test results are sorted by score descending."""
        query = [1, 0, 0]
        candidates = [[0.5, 0.5, 0], [1, 0, 0], [0.8, 0.2, 0]]
        result = find_similar(query, candidates, threshold=0.5)

        scores = [r[1] for r in result]
        assert scores == sorted(scores, reverse=True)


class TestFindTopK:
    """Test find_top_k function."""

    def test_find_top_2(self):
        """Test finding top 2 similar."""
        query = [1, 0, 0]
        candidates = [
            [1, 0, 0],      # 1.0
            [0, 1, 0],      # 0.0
            [0.9, 0.1, 0],  # ~0.99
            [0.5, 0.5, 0],  # ~0.707
        ]
        result = find_top_k(query, candidates, k=2)

        assert len(result) == 2
        assert result[0][0] == 0  # Most similar
        assert result[1][0] == 2  # Second most similar

    def test_k_larger_than_candidates(self):
        """Test when k is larger than candidate count."""
        query = [1, 0]
        candidates = [[1, 0], [0, 1]]
        result = find_top_k(query, candidates, k=10)

        assert len(result) == 2

    def test_with_euclidean_metric(self):
        """Test with Euclidean metric."""
        query = [0, 0]
        candidates = [[1, 0], [2, 0], [3, 0]]
        result = find_top_k(query, candidates, k=2, metric="euclidean")

        # Negated distance: closest = highest score
        assert result[0][0] == 0  # [1, 0] is closest
        assert result[1][0] == 1  # [2, 0] is second closest
