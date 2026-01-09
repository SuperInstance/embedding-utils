"""Tests for vectors module."""

import pytest
import numpy as np

from embedding_utils import (
    normalize_vector,
    normalize_vectors,
    mean_vector,
    weighted_mean_vector,
    vector_magnitude,
    vector_sum,
    vector_difference,
    vector_divide,
    scale_vector,
    clip_vector,
    concatenate_vectors,
)


class TestNormalizeVector:
    """Test normalize_vector function."""

    def test_normalize_simple(self):
        """Test normalizing a simple vector."""
        result = normalize_vector([3, 4])
        expected = np.array([0.6, 0.8], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)

    def test_magnitude_is_one(self):
        """Test normalized vector has magnitude 1."""
        result = normalize_vector([1, 2, 3])
        magnitude = np.linalg.norm(result)
        assert magnitude == pytest.approx(1.0)

    def test_zero_vector(self):
        """Test zero vector stays zero."""
        result = normalize_vector([0, 0, 0])
        np.testing.assert_array_equal(result, np.array([0, 0, 0], dtype=np.float32))

    def test_already_normalized(self):
        """Test already normalized vector."""
        result = normalize_vector([1, 0, 0])
        np.testing.assert_array_almost_equal(result, np.array([1, 0, 0], dtype=np.float32))


class TestNormalizeVectors:
    """Test normalize_vectors function."""

    def test_normalize_multiple(self):
        """Test normalizing multiple vectors."""
        result = normalize_vectors([[3, 4], [6, 8]])
        assert len(result) == 2

        # Both should have same direction
        np.testing.assert_array_almost_equal(result[0], [0.6, 0.8])
        np.testing.assert_array_almost_equal(result[1], [0.6, 0.8])

    def test_empty_list(self):
        """Test empty list."""
        result = normalize_vectors([])
        assert result == []


class TestVectorMagnitude:
    """Test vector_magnitude function."""

    def test_simple_magnitude(self):
        """Test simple magnitude calculation."""
        result = vector_magnitude([3, 4])
        assert result == pytest.approx(5.0)

    def test_zero_magnitude(self):
        """Test zero vector magnitude."""
        result = vector_magnitude([0, 0, 0])
        assert result == pytest.approx(0.0)

    def test_unit_vector(self):
        """Test unit vector magnitude."""
        result = vector_magnitude([1, 0, 0])
        assert result == pytest.approx(1.0)


class TestMeanVector:
    """Test mean_vector function."""

    def test_simple_mean(self):
        """Test simple mean calculation."""
        result = mean_vector([[1, 2], [3, 4], [5, 6]])
        np.testing.assert_array_almost_equal(result, np.array([3., 4.], dtype=np.float32))

    def test_single_vector(self):
        """Test mean of single vector."""
        result = mean_vector([[1, 2, 3]])
        np.testing.assert_array_almost_equal(result, np.array([1., 2., 3.], dtype=np.float32))

    def test_empty_list_raises(self):
        """Test empty list raises error."""
        with pytest.raises(ValueError):
            mean_vector([])

    def test_weighted_mean(self):
        """Test weighted mean."""
        result = mean_vector([[1, 2], [3, 4]], weights=[0.5, 0.5])
        np.testing.assert_array_almost_equal(result, np.array([2., 3.], dtype=np.float32))

    def test_weights_mismatch_raises(self):
        """Test weights length mismatch raises error."""
        with pytest.raises(ValueError):
            mean_vector([[1, 2], [3, 4]], weights=[0.5])


class TestWeightedMeanVector:
    """Test weighted_mean_vector function."""

    def test_simple_weighted_mean(self):
        """Test simple weighted mean."""
        result = weighted_mean_vector([[1, 2], [3, 4]], [0.75, 0.25])
        # 0.75 * [1, 2] + 0.25 * [3, 4] = [0.75 + 0.75, 1.5 + 1] = [1.5, 2.5]
        np.testing.assert_array_almost_equal(result, np.array([1.5, 2.5], dtype=np.float32))

    def test_unnormalized_weights(self):
        """Test weights don't need to sum to 1."""
        result = weighted_mean_vector([[1, 2], [3, 4]], [3, 1])
        # [3*1 + 1*3] / 4 = 1.5, [3*2 + 1*4] / 4 = 2.5
        np.testing.assert_array_almost_equal(result, np.array([1.5, 2.5], dtype=np.float32))


class TestVectorSum:
    """Test vector_sum function."""

    def test_simple_sum(self):
        """Test simple sum."""
        result = vector_sum([[1, 2], [3, 4]])
        np.testing.assert_array_almost_equal(result, np.array([4., 6.], dtype=np.float32))

    def test_empty_list_raises(self):
        """Test empty list raises error."""
        with pytest.raises(ValueError):
            vector_sum([])

    def test_multiple_vectors(self):
        """Test sum of multiple vectors."""
        result = vector_sum([[1], [2], [3], [4]])
        np.testing.assert_array_almost_equal(result, np.array([10.], dtype=np.float32))


class TestVectorDifference:
    """Test vector_difference function."""

    def test_simple_difference(self):
        """Test simple difference."""
        result = vector_difference([5, 5], [2, 3])
        np.testing.assert_array_almost_equal(result, np.array([3., 2.], dtype=np.float32))

    def test_negative_result(self):
        """Test negative result."""
        result = vector_difference([2, 3], [5, 5])
        np.testing.assert_array_almost_equal(result, np.array([-3., -2.], dtype=np.float32))


class TestVectorDivide:
    """Test vector_divide function."""

    def test_simple_divide(self):
        """Test simple division."""
        result = vector_divide([4, 6], 2)
        np.testing.assert_array_almost_equal(result, np.array([2., 3.], dtype=np.float32))

    def test_divide_by_fraction(self):
        """Test dividing by fraction."""
        result = vector_divide([1, 2], 0.5)
        np.testing.assert_array_almost_equal(result, np.array([2., 4.], dtype=np.float32))


class TestScaleVector:
    """Test scale_vector function."""

    def test_scale_up(self):
        """Test scaling up."""
        result = scale_vector([1, 2], 3)
        np.testing.assert_array_almost_equal(result, np.array([3., 6.], dtype=np.float32))

    def test_scale_down(self):
        """Test scaling down."""
        result = scale_vector([4, 6], 0.5)
        np.testing.assert_array_almost_equal(result, np.array([2., 3.], dtype=np.float32))

    def test_scale_by_zero(self):
        """Test scaling by zero."""
        result = scale_vector([1, 2, 3], 0)
        np.testing.assert_array_almost_equal(result, np.array([0., 0., 0.], dtype=np.float32))


class TestClipVector:
    """Test clip_vector function."""

    def test_clip_above(self):
        """Test clipping values above max."""
        result = clip_vector([1, 2, 3], 0, 2)
        np.testing.assert_array_almost_equal(result, np.array([1., 2., 2.], dtype=np.float32))

    def test_clip_below(self):
        """Test clipping values below min."""
        result = clip_vector([-1, 0, 1], 0, 2)
        np.testing.assert_array_almost_equal(result, np.array([0., 0., 1.], dtype=np.float32))

    def test_clip_both(self):
        """Test clipping both sides."""
        result = clip_vector([-1, 0, 1, 2, 3], 0, 2)
        np.testing.assert_array_almost_equal(result, np.array([0., 0., 1., 2., 2.], dtype=np.float32))


class TestConcatenateVectors:
    """Test concatenate_vectors function."""

    def test_simple_concat(self):
        """Test simple concatenation."""
        result = concatenate_vectors([[1, 2], [3, 4]])
        np.testing.assert_array_almost_equal(result, np.array([1., 2., 3., 4.], dtype=np.float32))

    def test_different_sizes(self):
        """Test concatenating different sized vectors."""
        result = concatenate_vectors([[1], [2, 3], [4, 5, 6]])
        np.testing.assert_array_almost_equal(result, np.array([1., 2., 3., 4., 5., 6.], dtype=np.float32))

    def test_empty_list(self):
        """Test empty list."""
        result = concatenate_vectors([])
        np.testing.assert_array_equal(result, np.array([], dtype=np.float32))

    def test_single_vector(self):
        """Test single vector."""
        result = concatenate_vectors([[1, 2, 3]])
        np.testing.assert_array_almost_equal(result, np.array([1., 2., 3.], dtype=np.float32))
