"""
Vector operations for embeddings.
"""

from typing import List, Union, Optional
import numpy as np


Vector = Union[List[float], np.ndarray]


def _to_numpy(v: Vector) -> np.ndarray:
    """Convert vector to numpy array."""
    if isinstance(v, list):
        return np.array(v, dtype=np.float32)
    return v


def normalize_vector(v: Vector) -> np.ndarray:
    """
    Normalize a vector to unit length (L2 normalization).

    Args:
        v: Vector to normalize

    Returns:
        Normalized vector

    Example:
        >>> normalize_vector([3, 4])
        array([0.6, 0.8], dtype=float32)
    """
    v = _to_numpy(v)
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def normalize_vectors(vectors: List[Vector]) -> List[np.ndarray]:
    """
    Normalize multiple vectors to unit length.

    Args:
        vectors: List of vectors to normalize

    Returns:
        List of normalized vectors

    Example:
        >>> normalize_vectors([[3, 4], [6, 8]])
        [array([0.6, 0.8]), array([0.6, 0.8])]
    """
    return [normalize_vector(v) for v in vectors]


def vector_magnitude(v: Vector) -> float:
    """
    Calculate the magnitude (L2 norm) of a vector.

    Args:
        v: Vector

    Returns:
        Magnitude (length) of the vector

    Example:
        >>> vector_magnitude([3, 4])
        5.0
    """
    v = _to_numpy(v)
    return float(np.linalg.norm(v))


def mean_vector(vectors: List[Vector], weights: Optional[List[float]] = None) -> np.ndarray:
    """
    Calculate the mean of multiple vectors.

    Args:
        vectors: List of vectors
        weights: Optional weights for weighted mean

    Returns:
        Mean vector

    Example:
        >>> mean_vector([[1, 2], [3, 4], [5, 6]])
        array([3., 4.], dtype=float32)
    """
    if not vectors:
        raise ValueError("Cannot compute mean of empty list")

    vectors_array = np.array([_to_numpy(v) for v in vectors], dtype=np.float32)

    if weights is not None:
        if len(weights) != len(vectors):
            raise ValueError("Weights length must match vectors length")
        weights = np.array(weights, dtype=np.float32)
        weights = weights / weights.sum()  # Normalize weights
        return np.average(vectors_array, axis=0, weights=weights)

    return np.mean(vectors_array, axis=0)


def weighted_mean_vector(vectors: List[Vector], weights: List[float]) -> np.ndarray:
    """
    Calculate weighted mean of vectors.

    Args:
        vectors: List of vectors
        weights: Weight for each vector (doesn't need to sum to 1)

    Returns:
        Weighted mean vector

    Example:
        >>> weighted_mean_vector([[1, 2], [3, 4]], [0.7, 0.3])
        array([1.6, 2.6], dtype=float32)
    """
    return mean_vector(vectors, weights)


def vector_sum(vectors: List[Vector]) -> np.ndarray:
    """
    Calculate the sum of multiple vectors.

    Args:
        vectors: List of vectors

    Returns:
        Sum vector

    Example:
        >>> vector_sum([[1, 2], [3, 4]])
        array([4., 6.], dtype=float32)
    """
    if not vectors:
        raise ValueError("Cannot compute sum of empty list")

    vectors_array = np.array([_to_numpy(v) for v in vectors], dtype=np.float32)
    return np.sum(vectors_array, axis=0)


def vector_difference(a: Vector, b: Vector) -> np.ndarray:
    """
    Calculate the difference between two vectors (a - b).

    Args:
        a: First vector
        b: Second vector

    Returns:
        Difference vector

    Example:
        >>> vector_difference([5, 5], [2, 3])
        array([3., 2.], dtype=float32)
    """
    a = _to_numpy(a)
    b = _to_numpy(b)
    return a - b


def vector_divide(v: Vector, scalar: float) -> np.ndarray:
    """
    Divide a vector by a scalar.

    Args:
        v: Vector to divide
        scalar: Scalar to divide by

    Returns:
        Divided vector

    Example:
        >>> vector_divide([4, 6], 2)
        array([2., 3.], dtype=float32)
    """
    v = _to_numpy(v)
    return v / scalar


def scale_vector(v: Vector, factor: float) -> np.ndarray:
    """
    Scale a vector by a factor.

    Args:
        v: Vector to scale
        factor: Scaling factor

    Returns:
        Scaled vector

    Example:
        >>> scale_vector([1, 2], 3)
        array([3., 6.], dtype=float32)
    """
    v = _to_numpy(v)
    return v * factor


def clip_vector(v: Vector, min_val: float, max_val: float) -> np.ndarray:
    """
    Clip vector values to a range.

    Args:
        v: Vector to clip
        min_val: Minimum value
        max_val: Maximum value

    Returns:
        Clipped vector

    Example:
        >>> clip_vector([-1, 0, 1, 2, 3], 0, 2)
        array([0., 0., 1., 2., 2.], dtype=float32)
    """
    v = _to_numpy(v)
    return np.clip(v, min_val, max_val)


def concatenate_vectors(vectors: List[Vector]) -> np.ndarray:
    """
    Concatenate multiple vectors into one.

    Args:
        vectors: List of vectors to concatenate

    Returns:
        Concatenated vector

    Example:
        >>> concatenate_vectors([[1, 2], [3, 4, 5]])
        array([1., 2., 3., 4., 5.], dtype=float32)
    """
    if not vectors:
        return np.array([], dtype=np.float32)

    return np.concatenate([_to_numpy(v).flatten() for v in vectors])
