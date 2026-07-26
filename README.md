# Atlas

An in-memory vector database implemented in C++.

The project was built to explore the mechanics of similarity search rather than to reproduce the feature set of existing systems such as Milvus or Qdrant. It implements multiple search strategies behind a common HTTP interface so that different indexing approaches can be compared under the same workload.

The current implementation supports exact search using a linear scan together with tree-based indexing, multiple distance metrics, and a JSON REST API.

---

## Motivation

Vector databases are usually presented as complete products. This project focuses on the underlying algorithms instead.

The primary objectives were:

* implement vector storage without external database libraries
* compare exact search strategies
* separate storage, indexing, and API concerns
* expose the system through a simple HTTP interface

The emphasis is on understanding implementation trade-offs rather than maximizing performance.

---

## System Overview

<img width="898" height="729" alt="Screenshot 2026-07-26 175829" src="https://github.com/user-attachments/assets/c80ee20c-bc28-435f-be95-e740ce877c3a" />

The HTTP layer is responsible only for request parsing and response serialization. Search algorithms operate independently from the transport layer, allowing implementations to be evaluated without modifying the client.

---

## Search Algorithms

### Brute Force

Computes the distance between the query vector and every stored vector.

Properties

* exact nearest-neighbor search
* linear time complexity
* baseline for correctness and benchmarking

### KD-Tree

A spatial index that recursively partitions vectors by dimension.

Properties

* reduced search space for lower-dimensional data
* exact search
* performance depends on dimensionality and data distribution

### HNSW

An experimental implementation of Hierarchical Navigable Small World graphs.

The implementation is included for experimentation and is not yet intended to be feature-complete.

---

## Distance Metrics

Three similarity metrics are currently available.

| Metric             | Typical Use          |
| ------------------ | -------------------- |
| Cosine Distance    | Embedding similarity |
| Euclidean Distance | Geometric distance   |
| Manhattan Distance | Grid-based distance  |

The search layer is independent of the distance metric, allowing the same API to evaluate different similarity functions.

---

## API

### Insert

```http
POST /insert
```

```json
{
    "id": 1,
    "label": "example",
    "values": [0.12, 0.51, 0.87]
}
```

### Search

```http
POST /search
```

```json
{
    "query": [0.10, 0.49, 0.90],
    "k": 5,
    "metric": "cosine"
}
```

### Response

```json
[
    {
        "id": 1,
        "label": "example",
        "distance": 0.013
    }
]
```

---

## Repository Layout

```
.
├── main.cpp          Application and search implementations
├── httplib.h         HTTP server library
├── json.hpp          JSON serialization
├── server_test.cpp   API tests
└── index.html        Simple client
```

---

## Design Decisions

The implementation intentionally keeps all vectors in memory.

This simplifies ownership and avoids persistence concerns while making search behavior easier to reason about during development.

Search algorithms are implemented independently rather than tightly coupled to storage. This allows different indexing strategies to be substituted without changing the external API.

Distance computation is also isolated from indexing logic, making it possible to evaluate different similarity metrics without modifying the search implementations.

---

## Current Limitations

* No persistent storage
* No vector deletion in indexed structures
* Single-process architecture
* No concurrent query execution
* Limited benchmarking infrastructure


---

## Future Work

Potential extensions include:

* Approximate nearest-neighbor benchmarks
* Persistent storage
* SIMD acceleration
* Multi-threaded query execution
* Additional index structures (IVF, PQ, DiskANN)
* Performance profiling on larger datasets

---

## References

* "Nearest Neighbor Search in High-Dimensional Spaces" = https://link.springer.com/chapter/10.1007/978-3-642-22993-0_1
* KD-Tree  = https://www.cs.cmu.edu/~ckingsf/bioinfo-lectures/kdtrees.pdf
* HNSW: *Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs* = https://ieeexplore.ieee.org/document/8594636
