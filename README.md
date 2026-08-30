# Semantic Search & Response Platform

[![CI](https://github.com/brohum10/semantic-search-platform/actions/workflows/ci.yml/badge.svg)](https://github.com/brohum10/semantic-search-platform/actions/workflows/ci.yml)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.x-000000.svg)](https://flask.palletsprojects.com/)

A local-first retrieval service that persists messages in SQLite, indexes normalized vectors with FAISS, reranks results with recency, and exposes search plus source-backed extractive responses through Flask.

## What it demonstrates

- Batch and single-message ingestion with input and metadata validation
- Deterministic vector embeddings with a replaceable model boundary
- FAISS inner-product search with a NumPy fallback
- Hybrid candidate reranking using semantic similarity, recency, and lexical overlap
- SQLite persistence in WAL mode with parameterized queries
- Search and extractive-response endpoints that always return their sources
- Automated tests, coverage enforcement, Docker, CI, and a deterministic benchmark

## Quick start

Requirements: Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
semantic-search
```

Index a message:

```bash
curl -X POST http://localhost:8080/v1/messages \
  -H 'Content-Type: application/json' \
  -d '{"content":"Retries use capped exponential backoff.","metadata":{"source":"runbook"}}'
```

Search:

```bash
curl 'http://localhost:8080/v1/search?q=how+do+retries+work&limit=5'
```

Create a source-backed extractive response:

```bash
curl -X POST http://localhost:8080/v1/respond \
  -H 'Content-Type: application/json' \
  -d '{"query":"How should failed requests be retried?","limit":3}'
```

Docker is also supported:

```bash
docker compose up --build
```

## Architecture

```text
Flask API -> validation -> embedding -> FAISS/NumPy candidate search
                      |-> SQLite persistence
                      `-> similarity + recency reranking -> results / extractive response
```

See [docs/architecture.md](docs/architecture.md) for the component diagram, ranking formula, design choices, and production extensions.

## Tests

```bash
pytest --cov=semantic_search --cov-report=term-missing
```

The suite covers deterministic embeddings, nearest-neighbor ordering, SQLite round trips, hybrid ranking, content validation, extractive responses, and the public API. Coverage must remain at or above 85%.

## Reproducible benchmark

```bash
semantic-search-benchmark \
  --messages 100000 \
  --queries 500 \
  --output benchmarks/latest.json
```

The benchmark generates a labeled synthetic corpus, indexes it through the same service path used by the API, and reports indexing time, Recall@10, MRR, p50 latency, and p95 latency. `benchmarks/latest.json` contains the latest measured run; it is not a hand-edited performance claim.

Latest local run on an arm64 Mac (August 30, 2026):

| Messages | Queries | Indexing | Recall@10 | MRR | p50 | p95 |
|---:|---:|---:|---:|---:|---:|---:|
| 100,000 | 500 | 3.786 s | 0.9800 | 0.9473 | 4.793 ms | 5.196 ms |

The corpus is deterministic and synthetic, so these figures validate implementation performance and metric calculation rather than claiming production relevance quality on a real user dataset.

## API

| Method | Route | Purpose |
|---|---|---|
| `POST` | `/v1/messages` | Validate and index one message |
| `POST` | `/v1/messages/bulk` | Index a batch in one SQLite transaction |
| `GET` | `/v1/search?q=...&limit=...` | Retrieve and rerank relevant messages |
| `POST` | `/v1/respond` | Compose an extractive response with sources |
| `GET` | `/v1/health` | Report status, message count, and vector backend |

## Design note

The default embedder uses deterministic feature hashing so the project works without external model downloads or API keys. The interface is intentionally small: a neural sentence-embedding adapter can replace it without changing persistence, FAISS search, ranking, tests, or the HTTP contract.

## License

MIT
