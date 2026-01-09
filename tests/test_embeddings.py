"""Tests for embeddings module."""

import pytest
import tempfile
import pickle
from pathlib import Path

from embedding_utils import (
    EmbeddingCache,
    EmbeddingBatcher,
    EmbeddingIndex,
    EmbeddingSearch,
)


class TestEmbeddingCache:
    """Test EmbeddingCache class."""

    def test_set_and_get(self):
        """Test setting and getting embeddings."""
        cache = EmbeddingCache()
        cache.set("hello", [0.1, 0.2, 0.3])
        result = cache.get("hello")

        assert result is not None
        assert list(result) == [0.1, 0.2, 0.3]

    def test_get_missing(self):
        """Test getting missing key returns None."""
        cache = EmbeddingCache()
        result = cache.get("missing")
        assert result is None

    def test_with_model_namespacing(self):
        """Test model namespacing."""
        cache = EmbeddingCache()
        cache.set("hello", [0.1, 0.2], model="gpt-3")
        cache.set("hello", [0.3, 0.4], model="claude")

        result1 = cache.get("hello", model="gpt-3")
        result2 = cache.get("hello", model="claude")

        assert list(result1) == [0.1, 0.2]
        assert list(result2) == [0.3, 0.4]

    def test_max_size_eviction(self):
        """Test LRU eviction when max size is reached."""
        cache = EmbeddingCache(max_size=2)

        cache.set("a", [1, 1])
        cache.set("b", [2, 2])
        cache.set("c", [3, 3])  # Should evict "a"

        assert cache.get("a") is None
        assert cache.get("b") is not None
        assert cache.get("c") is not None

    def test_access_updates_order(self):
        """Test that accessing updates LRU order."""
        cache = EmbeddingCache(max_size=2)

        cache.set("a", [1, 1])
        cache.set("b", [2, 2])
        cache.get("a")  # Access "a" to make it recently used
        cache.set("c", [3, 3])  # Should evict "b"

        assert cache.get("a") is not None
        assert cache.get("b") is None
        assert cache.get("c") is not None

    def test_clear(self):
        """Test clearing cache."""
        cache = EmbeddingCache()
        cache.set("a", [1, 1])
        cache.set("b", [2, 2])
        cache.clear()

        assert cache.size() == 0
        assert cache.get("a") is None

    def test_size(self):
        """Test size method."""
        cache = EmbeddingCache()
        assert cache.size() == 0

        cache.set("a", [1, 1])
        assert cache.size() == 1

        cache.set("b", [2, 2])
        assert cache.size() == 2

    def test_save_and_load(self):
        """Test saving and loading cache."""
        cache = EmbeddingCache()
        cache.set("hello", [0.1, 0.2, 0.3])

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as f:
            path = f.name

        try:
            cache.save(path)

            new_cache = EmbeddingCache()
            new_cache.load(path)

            result = new_cache.get("hello")
            assert list(result) == [0.1, 0.2, 0.3]
        finally:
            Path(path).unlink(missing_ok=True)


class TestEmbeddingBatcher:
    """Test EmbeddingBatcher class."""

    def test_create_batches(self):
        """Test creating batches."""
        batcher = EmbeddingBatcher(batch_size=2)
        texts = ["a", "b", "c", "d", "e"]
        batches = batcher.create_batches(texts)

        assert len(batches) == 3
        assert len(batches[0]) == 2
        assert len(batches[1]) == 2
        assert len(batches[2]) == 1

    def test_empty_list(self):
        """Test empty list returns no batches."""
        batcher = EmbeddingBatcher()
        batches = batcher.create_batches([])
        assert batches == []

    def test_max_tokens_limit(self):
        """Test batch splitting by token limit."""
        batcher = EmbeddingBatcher(
            batch_size=10,
            max_tokens=5,
            tokenize_fn=lambda x: len(x.split())
        )
        texts = ["a b c", "d e", "f g h"]
        batches = batcher.create_batches(texts)

        # First batch: "a b c" (3 tokens)
        # Can't fit "d e" (2 tokens) because 3+2=5, but next check would be for "f g h"
        # Actually, let me trace through:
        # - "a b c": 3 tokens, batch = ["a b c"], current = 3
        # - "d e": 2 tokens, 3+2=5 <= 5, batch = ["a b c", "d e"], current = 5
        # - "f g h": 3 tokens, 5+3=8 > 5, start new batch
        # Result: [["a b c", "d e"], ["f g h"]]
        assert len(batches) == 2

    def test_batch_embeddings(self):
        """Test batch_embeddings method."""
        batcher = EmbeddingBatcher(batch_size=2)

        def embed_fn(texts):
            return [[len(t)] for t in texts]

        texts = ["a", "bb", "ccc"]
        results = batcher.batch_embeddings(texts, embed_fn)

        assert results == [[1], [2], [3]]


class TestEmbeddingIndex:
    """Test EmbeddingIndex class."""

    def test_add_and_search(self):
        """Test adding and searching."""
        index = EmbeddingIndex()

        index.add("doc1", [1, 0, 0])
        index.add("doc2", [0, 1, 0])
        index.add("doc3", [0.9, 0.1, 0])

        results = index.search([1, 0, 0], k=2)

        assert len(results) == 2
        assert results[0][0] == "doc1"  # Most similar
        assert results[0][1] == pytest.approx(1.0)

    def test_add_batch(self):
        """Test adding multiple documents."""
        index = EmbeddingIndex()
        index.add_batch(
            ["doc1", "doc2", "doc3"],
            [[1, 0], [0, 1], [1, 1]]
        )

        assert index.size() == 3

    def test_search_empty_index(self):
        """Test searching empty index."""
        index = EmbeddingIndex()
        results = index.search([1, 0, 0])
        assert results == []

    def test_search_with_min_score(self):
        """Test search with minimum score filter."""
        index = EmbeddingIndex()

        index.add("doc1", [1, 0, 0])
        index.add("doc2", [0.5, 0.5, 0])
        index.add("doc3", [0, 1, 0])

        results = index.search([1, 0, 0], k=10, min_score=0.8)

        assert len(results) == 1
        assert results[0][0] == "doc1"

    def test_get_vector(self):
        """Test getting vector by doc_id."""
        index = EmbeddingIndex()
        index.add("doc1", [0.1, 0.2, 0.3])

        vec = index.get_vector("doc1")
        # Note: vectors are normalized by default
        assert vec is not None
        # Check magnitude is ~1 (normalized)
        import numpy as np
        assert float(np.linalg.norm(vec)) == pytest.approx(1.0)

    def test_get_vector_missing(self):
        """Test getting missing doc_id."""
        index = EmbeddingIndex()
        assert index.get_vector("missing") is None

    def test_clear(self):
        """Test clearing index."""
        index = EmbeddingIndex()
        index.add("doc1", [1, 0])
        index.clear()

        assert index.size() == 0

    def test_normalization_disabled(self):
        """Test with normalization disabled."""
        index = EmbeddingIndex(normalize=False)
        index.add("doc1", [2, 0])
        index.add("doc2", [1, 0])

        # Without normalization, [2,0] and [1,0] won't have cosine similarity of 1
        # But the vectors are still stored as-is
        assert index.size() == 2


class TestEmbeddingSearch:
    """Test EmbeddingSearch class."""

    def test_add_and_search(self):
        """Test adding and searching."""
        search = EmbeddingSearch()

        search.add("doc1", [1, 0, 0], metadata={"category": "tech"})
        search.add("doc2", [0, 1, 0], metadata={"category": "news"})

        results = search.search([1, 0, 0], k=2)

        assert len(results) == 2
        assert results[0][0] == "doc1"

    def test_search_with_filter(self):
        """Test searching with metadata filter."""
        search = EmbeddingSearch()

        search.add("doc1", [1, 0, 0], metadata={"category": "tech"})
        search.add("doc2", [0.9, 0.1, 0], metadata={"category": "tech"})
        search.add("doc3", [0, 1, 0], metadata={"category": "news"})

        results = search.search(
            [1, 0, 0],
            k=10,
            filter_fn=lambda m: m.get("category") == "tech"
        )

        assert len(results) == 2
        assert all(r[2].get("category") == "tech" for r in results)

    def test_search_without_filter(self):
        """Test searching without filter returns all metadata."""
        search = EmbeddingSearch()

        search.add("doc1", [1, 0], metadata={"cat": "a"})
        search.add("doc2", [0, 1], metadata={"cat": "b"})

        results = search.search([1, 0], k=10)

        assert len(results) == 2
        assert results[0][2] == {"cat": "a"}

    def test_delete(self):
        """Test deleting a document."""
        search = EmbeddingSearch()

        search.add("doc1", [1, 0])
        search.add("doc2", [0, 1])

        assert search.delete("doc1") is True
        assert search.index.size() == 1
        assert search.delete("missing") is False

    def test_clear(self):
        """Test clearing search index."""
        search = EmbeddingSearch()

        search.add("doc1", [1, 0], metadata={"x": 1})
        search.clear()

        assert search.index.size() == 0
        assert len(search._metadata) == 0
