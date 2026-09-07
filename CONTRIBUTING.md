# Contributing to Semantic Search Platform

Contributions should keep retrieval behavior explainable, deterministic, and reproducible without requiring an external service.

## Local checks

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
ruff format --check .
ruff check .
pytest --cov=semantic_search --cov-report=term-missing
python -m build
```

## Expectations

- Preserve the exact local fallback when optional FAISS acceleration is unavailable.
- Include ranking explanations for new scoring signals.
- Add retrieval-quality cases for ranking or query-processing changes.
- Keep request, batch, result, metadata, and persistence bounds explicit.
- Update the OpenAPI and architecture documents when contracts or scoring semantics change.

Pull requests should include correctness tests, quality metrics where relevant, and benchmark evidence for performance claims.
