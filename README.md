# Embedding Utils

A Python library for working with text embeddings, similarity search, and vector operations.

## Features

- **Similarity Metrics**: Cosine similarity, Euclidean distance, dot product, Manhattan distance, Jaccard similarity
- **Vector Operations**: Normalization, mean, weighted mean, concatenation, scaling
- **Embedding Cache**: In-memory cache with LRU eviction and disk persistence
- **Batch Processing**: Efficient batch embedding generation
- **Similarity Search**: In-memory index with optional metadata filtering

## Installation

```bash
pip install embedding-utils
```

## Quick Start

```python
from embedding_utils import cosine_similarity, find_top_k

# Calculate similarity between two vectors
similarity = cosine_similarity([1, 0, 0], [1, 0, 0])
print(similarity)  # 1.0

# Find most similar vectors
query = [1, 0, 0]
candidates = [[1, 0, 0], [0, 1, 0], [0.9, 0.1, 0]]
results = find_top_k(query, candidates, k=2)
print(results)  # [(0, 1.0), (2, 0.99...)]
```

## Similarity Metrics

### Cosine Similarity

```python
from embedding_utils import cosine_similarity

# Returns value between -1 and 1 (1 = identical direction)
similarity = cosine_similarity([1, 2, 3], [4, 5, 6])
```

### Euclidean Distance

```python
from embedding_utils import euclidean_distance

# Returns distance (0 = identical)
distance = euclidean_distance([1, 0, 0], [0, 1, 0])
```

### Pairwise Similarity Matrix

```python
from embedding_utils import pairwise_similarity

vectors = [[1, 0, 0], [0, 1, 0], [1, 1, 0]]
matrix = pairwise_similarity(vectors, metric="cosine")
print(matrix)
# [[1.   0.   0.707]
#  [0.   1.   0.707]
#  [0.707 0.707 1.  ]]
```

## Vector Operations

### Normalization

```python
from embedding_utils import normalize_vector, normalize_vectors

# Single vector
normalized = normalize_vector([3, 4])  # [0.6, 0.8]

# Multiple vectors
normalized_list = normalize_vectors([[3, 4], [6, 8]])
```

### Mean Vector

```python
from embedding_utils import mean_vector, weighted_mean_vector

# Simple mean
mean = mean_vector([[1, 2], [3, 4], [5, 6]])  # [3, 4]

# Weighted mean
weighted = weighted_mean_vector([[1, 2], [3, 4]], [0.7, 0.3])
```

### Other Operations

```python
from embedding_utils import (
    vector_sum,
    vector_difference,
    scale_vector,
    concatenate_vectors,
)

sum_vec = vector_sum([[1, 2], [3, 4]])  # [4, 6]
diff = vector_difference([5, 5], [2, 3])  # [3, 2]
scaled = scale_vector([1, 2], 3)  # [3, 6]
concat = concatenate_vectors([[1, 2], [3, 4, 5]])  # [1, 2, 3, 4, 5]
```

## Embedding Cache

Cache embeddings to avoid recomputation:

```python
from embedding_utils import EmbeddingCache

cache = EmbeddingCache(max_size=10000)

# Set and get
cache.set("hello world", [0.1, 0.2, 0.3])
embedding = cache.get("hello world")

# With model namespacing
cache.set("hello", [0.1, 0.2], model="gpt-3")
embedding = cache.get("hello", model="gpt-3")

# Save to disk
cache.save("/path/to/cache.pkl")
cache.load("/path/to/cache.pkl")
```

## Batch Processing

Process texts in batches for efficiency:

```python
from embedding_utils import EmbeddingBatcher

batcher = EmbeddingBatcher(batch_size=32, max_tokens=8000)

texts = ["text 1", "text 2", ...] * 1000

# Get batches
batches = batcher.create_batches(texts)

# Or generate embeddings directly
def embed_fn(texts):
    # Your embedding function here
    return [model.encode(t) for t in texts]

embeddings = batcher.batch_embeddings(texts, embed_fn)
```

## Similarity Search

In-memory index for fast similarity search:

```python
from embedding_utils import EmbeddingIndex

# Create index
index = EmbeddingIndex()

# Add documents
index.add("doc1", [0.1, 0.2, 0.3])
index.add("doc2", [0.4, 0.5, 0.6])
index.add_batch(["doc3", "doc4"], [[0.7, 0.8, 0.9], [0.2, 0.3, 0.4]])

# Search
results = index.search([0.15, 0.25, 0.35], k=2)
# [("doc1", 0.98), ("doc4", 0.85)]
```

### Search with Metadata Filtering

```python
from embedding_utils import EmbeddingSearch

search = EmbeddingSearch()

# Add documents with metadata
search.add("doc1", [0.1, 0.2, 0.3], metadata={"category": "tech", "date": "2024-01-01"})
search.add("doc2", [0.4, 0.5, 0.6], metadata={"category": "news", "date": "2024-01-02"})

# Search with filter
results = search.search(
    [0.15, 0.25, 0.35],
    k=5,
    filter_fn=lambda m: m.get("category") == "tech"
)
```

## API Reference

### Similarity Functions

| Function | Description |
|----------|-------------|
| `cosine_similarity(a, b)` | Cosine similarity (-1 to 1) |
| `euclidean_distance(a, b)` | Euclidean distance |
| `dot_product(a, b)` | Dot product |
| `manhattan_distance(a, b)` | L1 distance |
| `jaccard_similarity(a, b)` | Jaccard similarity |
| `pairwise_similarity(vectors, metric)` | Pairwise similarity matrix |
| `find_similar(query, candidates, threshold)` | Find candidates above threshold |
| `find_top_k(query, candidates, k)` | Find top K candidates |

### Vector Functions

| Function | Description |
|----------|-------------|
| `normalize_vector(v)` | L2 normalize to unit length |
| `normalize_vectors(vectors)` | Normalize multiple vectors |
| `mean_vector(vectors, weights)` | Mean of vectors |
| `weighted_mean_vector(vectors, weights)` | Weighted mean |
| `vector_sum(vectors)` | Sum of vectors |
| `vector_difference(a, b)` | Difference (a - b) |
| `scale_vector(v, factor)` | Scale by factor |
| `concatenate_vectors(vectors)` | Concatenate vectors |

## License

MIT License - see LICENSE file for details.
