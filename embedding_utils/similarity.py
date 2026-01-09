"""
Similarity metrics for embedding vectors.
"""

import math
from typing import List, Tuple, Optional, Union, Callable
import numpy as np


Vector = Union[List[float], np.ndarray]


def _to_numpy(v: Vector) -> np.ndarray:
    """Convert vector to numpy array."""
    if isinstance(v, list):
        return np.array(v, dtype=np.float32)
    return v


def cosine_similarity(a: Vector, b: Vector) -> float:
    """
    Calculate cosine similarity between two vectors.

    Args:
        a: First vector
        b: Second vector

    Returns:
        Similarity score between -1 and 1 (1 = identical)

    Example:
        >>> cosine_similarity([1, 0, 0], [1, 0, 0])
        1.0
        >>> cosine_similarity([1, 0, 0], [0, 1, 0])
        0.0
    """
    a = _to_numpy(a)
    b = _to_numpy(b)

    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(dot_product / (norm_a * norm_b))


def euclidean_distance(a: Vector, b: Vector) -> float:
    """
    Calculate Euclidean distance between two vectors.

    Args:
        a: First vector
        b: Second vector

    Returns:
        Distance score (0 = identical)

    Example:
        >>> euclidean_distance([1, 0, 0], [1, 0, 0])
        0.0
        >>> euclidean_distance([1, 0, 0], [0, 1, 0])
        1.4142135623730951
    """
    a = _to_numpy(a)
    b = _to_numpy(b)
    return float(np.linalg.norm(a - b))


def dot_product(a: Vector, b: Vector) -> float:
    """
    Calculate dot product of two vectors.

    Args:
        a: First vector
        b: Second vector

    Returns:
        Dot product score

    Example:
        >>> dot_product([1, 2, 3], [4, 5, 6])
        32.0
    """
    a = _to_numpy(a)
    b = _to_numpy(b)
    return float(np.dot(a, b))


def manhattan_distance(a: Vector, b: Vector) -> float:
    """
    Calculate Manhattan (L1) distance between two vectors.

    Args:
        a: First vector
        b: Second vector

    Returns:
        Distance score (0 = identical)
    """
    a = _to_numpy(a)
    b = _to_numpy(b)
    return float(np.sum(np.abs(a - b)))


def jaccard_similarity(a: Vector, b: Vector) -> float:
    """
    Calculate Jaccard similarity between two binary vectors.

    For non-binary vectors, uses a continuous approximation.

    Args:
        a: First vector
        b: Second vector

    Returns:
        Similarity score between 0 and 1 (1 = identical)
    """
    a = _to_numpy(a)
    b = _to_numpy(b)

    # For continuous vectors, use intersection over minimum
    intersection = np.sum(np.minimum(a, b))
    union = np.sum(np.maximum(a, b))

    if union == 0:
        return 0.0

    return float(intersection / union)


def pairwise_similarity(
    vectors: List[Vector],
    metric: str = "cosine"
) -> np.ndarray:
    """
    Calculate pairwise similarity matrix for a list of vectors.

    Args:
        vectors: List of vectors
        metric: Similarity metric ("cosine", "euclidean", "dot", "manhattan", "jaccard")

    Returns:
        NxN similarity matrix

    Example:
        >>> vectors = [[1, 0, 0], [0, 1, 0], [1, 1, 0]]
        >>> pairwise_similarity(vectors, metric="cosine")
        array([[1. , 0. , 0.707...],
               [0. , 1. , 0.707...],
               [0.707..., 0.707..., 1. ]])
    """
    n = len(vectors)
    matrix = np.zeros((n, n), dtype=np.float32)

    metric_func = {
        "cosine": cosine_similarity,
        "euclidean": euclidean_distance,
        "dot": dot_product,
        "manhattan": manhattan_distance,
        "jaccard": jaccard_similarity,
    }.get(metric, cosine_similarity)

    for i in range(n):
        for j in range(i, n):
            score = metric_func(vectors[i], vectors[j])
            matrix[i, j] = score
            matrix[j, i] = score

    return matrix


def find_similar(
    query: Vector,
    candidates: List[Vector],
    threshold: float = 0.7,
    metric: str = "cosine"
) -> List[Tuple[int, float]]:
    """
    Find all candidates above a similarity threshold.

    Args:
        query: Query vector
        candidates: List of candidate vectors
        threshold: Minimum similarity score (0-1 for cosine)
        metric: Similarity metric to use

    Returns:
        List of (index, score) tuples for candidates above threshold

    Example:
        >>> query = [1, 0, 0]
        >>> candidates = [[1, 0, 0], [0, 1, 0], [0.9, 0.1, 0]]
        >>> find_similar(query, candidates, threshold=0.8)
        [(0, 1.0), (2, 0.99...)]
    """
    metric_func = {
        "cosine": cosine_similarity,
        "euclidean": lambda a, b: -euclidean_distance(a, b),  # Negate for similarity
        "dot": dot_product,
        "manhattan": lambda a, b: -manhattan_distance(a, b),
        "jaccard": jaccard_similarity,
    }.get(metric, cosine_similarity)

    results = []
    for i, candidate in enumerate(candidates):
        score = metric_func(query, candidate)
        if score >= threshold:
            results.append((i, score))

    # Sort by score descending
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def find_top_k(
    query: Vector,
    candidates: List[Vector],
    k: int = 5,
    metric: str = "cosine"
) -> List[Tuple[int, float]]:
    """
    Find top K most similar candidates.

    Args:
        query: Query vector
        candidates: List of candidate vectors
        k: Number of results to return
        metric: Similarity metric to use

    Returns:
        List of (index, score) tuples for top K candidates

    Example:
        >>> query = [1, 0, 0]
        >>> candidates = [[1, 0, 0], [0, 1, 0], [0.9, 0.1, 0]]
        >>> find_top_k(query, candidates, k=2)
        [(0, 1.0), (2, 0.99...)]
    """
    k = min(k, len(candidates))

    metric_func = {
        "cosine": cosine_similarity,
        "euclidean": lambda a, b: -euclidean_distance(a, b),
        "dot": dot_product,
        "manhattan": lambda a, b: -manhattan_distance(a, b),
        "jaccard": jaccard_similarity,
    }.get(metric, cosine_similarity)

    # Calculate all scores
    scores = [(i, metric_func(query, candidate))
              for i, candidate in enumerate(candidates)]

    # Sort and return top K
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:k]
