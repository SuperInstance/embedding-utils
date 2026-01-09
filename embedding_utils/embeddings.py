"""
High-level embedding utilities.
"""

import pickle
import hashlib
from typing import List, Dict, Optional, Any, Callable, Union, Tuple
from pathlib import Path
import numpy as np

from .similarity import cosine_similarity, find_top_k
from .vectors import normalize_vector


Vector = Union[List[float], np.ndarray]


class EmbeddingCache:
    """
    Cache for text embeddings to avoid recomputing.

    Example:
        cache = EmbeddingCache()
        cache.set("hello", [0.1, 0.2, ...])
        embedding = cache.get("hello")
    """

    def __init__(self, max_size: int = 10000):
        """
        Initialize the cache.

        Args:
            max_size: Maximum number of cached embeddings
        """
        self.max_size = max_size
        self._cache: Dict[str, np.ndarray] = {}
        self._access_order: List[str] = []

    def _make_key(self, text: str, model: Optional[str] = None) -> str:
        """Create a cache key from text and model."""
        if model:
            return f"{model}:{text}"
        return text

    def get(self, text: str, model: Optional[str] = None) -> Optional[np.ndarray]:
        """
        Get cached embedding for text.

        Args:
            text: Text to look up
            model: Optional model name for namespacing

        Returns:
            Cached embedding or None
        """
        key = self._make_key(text, model)
        if key in self._cache:
            # Update access order
            self._access_order.remove(key)
            self._access_order.append(key)
            return self._cache[key].copy()
        return None

    def set(self, text: str, embedding: Vector, model: Optional[str] = None) -> None:
        """
        Cache an embedding for text.

        Args:
            text: Text to cache
            embedding: Embedding vector
            model: Optional model name for namespacing
        """
        key = self._make_key(text, model)

        # Enforce max size
        if len(self._cache) >= self.max_size and key not in self._cache:
            oldest = self._access_order.pop(0)
            del self._cache[oldest]

        self._cache[key] = np.array(embedding, dtype=np.float32)

        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)

    def clear(self) -> None:
        """Clear all cached embeddings."""
        self._cache.clear()
        self._access_order.clear()

    def size(self) -> int:
        """Get number of cached embeddings."""
        return len(self._cache)

    def save(self, path: Union[str, Path]) -> None:
        """
        Save cache to disk.

        Args:
            path: Path to save cache
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'wb') as f:
            pickle.dump({
                'cache': self._cache,
                'order': self._access_order,
                'max_size': self.max_size
            }, f)

    def load(self, path: Union[str, Path]) -> None:
        """
        Load cache from disk.

        Args:
            path: Path to load cache from
        """
        path = Path(path)
        if not path.exists():
            return

        with open(path, 'rb') as f:
            data = pickle.load(f)
            self._cache = data['cache']
            self._access_order = data['order']
            self.max_size = data.get('max_size', self.max_size)


class EmbeddingBatcher:
    """
    Batch text for efficient embedding generation.

    Example:
        batcher = EmbeddingBatcher(batch_size=32, max_tokens=8000)
        batches = batcher.create_batches(texts)
    """

    def __init__(
        self,
        batch_size: int = 32,
        max_tokens: int = 8000,
        tokenize_fn: Optional[Callable[[str], int]] = None
    ):
        """
        Initialize the batcher.

        Args:
            batch_size: Maximum texts per batch
            max_tokens: Maximum tokens per batch (approximate)
            tokenize_fn: Optional function to count tokens
        """
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        self.tokenize_fn = tokenize_fn or (lambda x: len(x.split()))

    def create_batches(self, texts: List[str]) -> List[List[str]]:
        """
        Split texts into batches for processing.

        Args:
            texts: List of texts to batch

        Returns:
            List of text batches
        """
        if not texts:
            return []

        batches = []
        current_batch = []
        current_tokens = 0

        for text in texts:
            text_tokens = self.tokenize_fn(text)

            # Check if text alone exceeds limit
            if text_tokens > self.max_tokens:
                # Truncate or skip
                text_tokens = self.max_tokens

            # Check if we need a new batch
            if (len(current_batch) >= self.batch_size or
                current_tokens + text_tokens > self.max_tokens):

                if current_batch:
                    batches.append(current_batch)
                current_batch = []
                current_tokens = 0

            current_batch.append(text)
            current_tokens += text_tokens

        if current_batch:
            batches.append(current_batch)

        return batches

    def batch_embeddings(
        self,
        texts: List[str],
        embed_fn: Callable[[List[str]], List[np.ndarray]]
    ) -> List[np.ndarray]:
        """
        Generate embeddings for texts in batches.

        Args:
            texts: List of texts to embed
            embed_fn: Function that takes a list of texts and returns embeddings

        Returns:
            List of embeddings
        """
        batches = self.create_batches(texts)
        all_embeddings = []

        for batch in batches:
            embeddings = embed_fn(batch)
            all_embeddings.extend(embeddings)

        return all_embeddings


class EmbeddingIndex:
    """
    Simple in-memory index for similarity search.

    Example:
        index = EmbeddingIndex()
        index.add("doc1", [0.1, 0.2, ...])
        index.add("doc2", [0.3, 0.4, ...])
        results = index.search([0.15, 0.25], k=2)
    """

    def __init__(self, normalize: bool = True):
        """
        Initialize the index.

        Args:
            normalize: Whether to normalize vectors for cosine similarity
        """
        self.normalize = normalize
        self._ids: List[str] = []
        self._vectors: List[np.ndarray] = []

    def add(self, doc_id: str, vector: Vector) -> None:
        """
        Add a document to the index.

        Args:
            doc_id: Document identifier
            vector: Embedding vector
        """
        vec = np.array(vector, dtype=np.float32)
        if self.normalize:
            vec = normalize_vector(vec)

        self._ids.append(doc_id)
        self._vectors.append(vec)

    def add_batch(self, doc_ids: List[str], vectors: List[Vector]) -> None:
        """
        Add multiple documents to the index.

        Args:
            doc_ids: List of document identifiers
            vectors: List of embedding vectors
        """
        for doc_id, vector in zip(doc_ids, vectors):
            self.add(doc_id, vector)

    def search(
        self,
        query: Vector,
        k: int = 10,
        min_score: float = 0.0
    ) -> List[Tuple[str, float]]:
        """
        Search for similar documents.

        Args:
            query: Query embedding
            k: Number of results to return
            min_score: Minimum similarity score

        Returns:
            List of (doc_id, score) tuples
        """
        if not self._vectors:
            return []

        query_vec = np.array(query, dtype=np.float32)
        if self.normalize:
            query_vec = normalize_vector(query_vec)

        # Calculate similarities
        scores = [
            float(cosine_similarity(query_vec, vec))
            for vec in self._vectors
        ]

        # Sort by score
        indexed_scores = list(zip(self._ids, scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        # Filter by min_score and take top k
        results = [
            (doc_id, score)
            for doc_id, score in indexed_scores
            if score >= min_score
        ][:k]

        return results

    def get_vector(self, doc_id: str) -> Optional[np.ndarray]:
        """Get the vector for a document ID."""
        try:
            idx = self._ids.index(doc_id)
            return self._vectors[idx].copy()
        except ValueError:
            return None

    def size(self) -> int:
        """Get number of documents in index."""
        return len(self._ids)

    def clear(self) -> None:
        """Clear all documents from index."""
        self._ids.clear()
        self._vectors.clear()


class EmbeddingSearch:
    """
    High-level search interface with optional filtering.

    Example:
        search = EmbeddingSearch()
        search.add("doc1", [0.1, 0.2, ...], metadata={"category": "tech"})
        search.add("doc2", [0.3, 0.4, ...], metadata={"category": "news"})
        results = search.search([0.15, 0.25], filter=lambda x: x["category"] == "tech")
    """

    def __init__(self):
        """Initialize the search index."""
        self.index = EmbeddingIndex()
        self._metadata: Dict[str, Dict[str, Any]] = {}

    def add(
        self,
        doc_id: str,
        vector: Vector,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add a document to the search index.

        Args:
            doc_id: Document identifier
            vector: Embedding vector
            metadata: Optional metadata for filtering
        """
        self.index.add(doc_id, vector)
        if metadata:
            self._metadata[doc_id] = metadata

    def search(
        self,
        query: Vector,
        k: int = 10,
        filter_fn: Optional[Callable[[Dict[str, Any]], bool]] = None
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Search with optional metadata filtering.

        Args:
            query: Query embedding
            k: Number of results to return
            filter_fn: Optional function to filter results by metadata

        Returns:
            List of (doc_id, score, metadata) tuples
        """
        if filter_fn is None:
            # No filtering, use simple search
            results = self.index.search(query, k=k)
            return [
                (doc_id, score, self._metadata.get(doc_id, {}))
                for doc_id, score in results
            ]

        # With filtering - need to check metadata
        query_vec = np.array(query, dtype=np.float32)
        query_vec = normalize_vector(query_vec)

        scored_results = []
        for doc_id, vec in zip(self.index._ids, self.index._vectors):
            metadata = self._metadata.get(doc_id, {})
            if not filter_fn(metadata):
                continue

            score = float(cosine_similarity(query_vec, vec))
            scored_results.append((doc_id, score, metadata))

        # Sort and return top k
        scored_results.sort(key=lambda x: x[1], reverse=True)
        return scored_results[:k]

    def delete(self, doc_id: str) -> bool:
        """Remove a document from the index."""
        try:
            idx = self.index._ids.index(doc_id)
            self.index._ids.pop(idx)
            self.index._vectors.pop(idx)
            self._metadata.pop(doc_id, None)
            return True
        except ValueError:
            return False

    def clear(self) -> None:
        """Clear all documents."""
        self.index.clear()
        self._metadata.clear()
