"""
Embedding Utils - Utilities for working with text embeddings.

Features:
- Similarity metrics (cosine, euclidean, dot product)
- Vector operations (normalization, batching)
- Batch embedding processing
- Embedding caching and storage
- Dimensionality reduction utilities
"""

from .similarity import (
    cosine_similarity,
    euclidean_distance,
    dot_product,
    manhattan_distance,
    jaccard_similarity,
    pairwise_similarity,
    find_similar,
    find_top_k,
)
from .vectors import (
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
from .embeddings import (
    EmbeddingCache,
    EmbeddingBatcher,
    EmbeddingSearch,
    EmbeddingIndex,
)

__version__ = "1.0.0"
__all__ = [
    # Similarity
    "cosine_similarity",
    "euclidean_distance",
    "dot_product",
    "manhattan_distance",
    "jaccard_similarity",
    "pairwise_similarity",
    "find_similar",
    "find_top_k",
    # Vectors
    "normalize_vector",
    "normalize_vectors",
    "mean_vector",
    "weighted_mean_vector",
    "vector_magnitude",
    "vector_sum",
    "vector_difference",
    "vector_divide",
    "scale_vector",
    "clip_vector",
    "concatenate_vectors",
    # Embeddings
    "EmbeddingCache",
    "EmbeddingBatcher",
    "EmbeddingSearch",
    "EmbeddingIndex",
]
