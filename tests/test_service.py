from datetime import UTC, datetime, timedelta

import pytest

from semantic_search.service import SearchService


def test_search_combines_similarity_and_recency(tmp_path) -> None:
    now = datetime(2026, 8, 30, tzinfo=UTC)
    service = SearchService(tmp_path / "messages.db", prefer_faiss=False)
    service.add_messages(
        [
            {
                "content": "database indexing performance",
                "created_at": (now - timedelta(days=180)).isoformat(),
            },
            {
                "content": "database indexing performance",
                "created_at": (now - timedelta(hours=1)).isoformat(),
            },
        ]
    )

    results = service.search("database indexing performance", limit=2, now=now)

    assert results[0].message.created_at > results[1].message.created_at
    assert results[0].similarity == pytest.approx(results[1].similarity)


def test_response_is_extractive_and_returns_sources(tmp_path) -> None:
    service = SearchService(tmp_path / "messages.db", prefer_faiss=False)
    service.add_messages([{"content": "Retry failed requests with capped exponential backoff."}])

    response = service.respond("How should failed requests be retried?")

    assert "capped exponential backoff" in response["answer"]
    assert response["mode"] == "extractive"
    assert len(response["sources"]) == 1


def test_empty_response_and_validation_paths(tmp_path) -> None:
    service = SearchService(tmp_path / "messages.db", prefer_faiss=False)

    assert service.respond("missing information")["sources"] == []
    with pytest.raises(ValueError, match="at least one"):
        service.add_messages([])
    with pytest.raises(ValueError, match="metadata"):
        service.add_messages([{"content": "valid", "metadata": []}])
    with pytest.raises(ValueError, match="limit"):
        service.search("valid query", limit=51)


def test_accepts_naive_iso_timestamp_as_utc(tmp_path) -> None:
    service = SearchService(tmp_path / "messages.db", prefer_faiss=False)
    message = service.add_messages(
        [{"content": "timestamp test", "created_at": "2026-08-30T12:00:00"}]
    )[0]

    assert message.created_at.tzinfo is UTC


def test_rejects_invalid_weight_configuration(tmp_path) -> None:
    with pytest.raises(ValueError, match="weights"):
        SearchService(tmp_path / "messages.db", similarity_weight=0.7, recency_weight=0.2)

    with pytest.raises(ValueError, match="half-life"):
        SearchService(tmp_path / "messages.db", recency_half_life_days=0)


@pytest.mark.parametrize("content", ["", "   ", "bad\x00content"])
def test_rejects_invalid_content(tmp_path, content: str) -> None:
    service = SearchService(tmp_path / "messages.db", prefer_faiss=False)

    with pytest.raises(ValueError):
        service.add_messages([{"content": content}])


def test_hybrid_retrieval_explains_scores_and_prioritizes_exact_identifiers(tmp_path) -> None:
    service = SearchService(tmp_path / "messages.db", dimension=64, prefer_faiss=False)
    service.add_messages(
        [
            {"content": "General request retry documentation"},
            {"content": "Incident RQ-4812 requires idempotent retry handling"},
        ]
    )

    results = service.search("RQ-4812 retry", limit=2)

    assert results[0].message.content.startswith("Incident RQ-4812")
    assert results[0].matched_terms == ("4812", "retry", "rq")
    assert results[0].explanation["bm25_score"] > 0
    assert sum(
        results[0].explanation[key]
        for key in ("semantic_contribution", "lexical_contribution", "recency_contribution")
    ) == pytest.approx(results[0].score)


def test_search_filters_by_nested_metadata_and_time_window(tmp_path) -> None:
    service = SearchService(tmp_path / "messages.db", prefer_faiss=False)
    service.add_messages(
        [
            {
                "content": "platform deployment process",
                "created_at": "2026-02-01T00:00:00Z",
                "metadata": {"team": {"name": "platform"}, "environment": "production"},
            },
            {
                "content": "platform deployment checklist",
                "created_at": "2025-02-01T00:00:00Z",
                "metadata": {"team": {"name": "platform"}, "environment": "staging"},
            },
        ]
    )

    results = service.search(
        "platform deployment",
        metadata={"team.name": "platform", "environment": "production"},
        created_after="2026-01-01T00:00:00Z",
        created_before="2026-03-01T00:00:00Z",
    )

    assert len(results) == 1
    assert results[0].message.metadata["environment"] == "production"
    assert service.search("platform", metadata={"missing.path": True}) == []


def test_search_rejects_invalid_filters_and_time_ranges(tmp_path) -> None:
    service = SearchService(tmp_path / "messages.db", prefer_faiss=False)

    with pytest.raises(ValueError, match="metadata filter"):
        service.search("query", metadata=[])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="created_after"):
        service.search("query", created_after="not-a-date")
    with pytest.raises(ValueError, match="later"):
        service.search(
            "query",
            created_after="2026-02-01T00:00:00Z",
            created_before="2026-01-01T00:00:00Z",
        )


def test_message_lifecycle_stats_and_restart_rebuild_indexes(tmp_path) -> None:
    database = tmp_path / "messages.db"
    service = SearchService(database, prefer_faiss=False)
    created = service.add_messages(
        [{"content": "durable semantic index", "metadata": {"kind": "note"}}]
    )[0]

    assert service.get_message(created.id) == created
    assert service.stats()["lexical_vocabulary"] == 3
    service.close()

    restored = SearchService(database, prefer_faiss=False)
    assert restored.search("durable index")[0].message.id == created.id
    restored.delete_message(created.id)
    assert restored.health()["messages"] == 0
    with pytest.raises(KeyError, match="not found"):
        restored.get_message(created.id)
    with pytest.raises(KeyError, match="not found"):
        restored.delete_message(created.id)


def test_batch_and_metadata_validation(tmp_path) -> None:
    service = SearchService(tmp_path / "messages.db", prefer_faiss=False)

    with pytest.raises(ValueError, match="at most"):
        service.add_messages([{"content": "valid"}] * 1_001)
    with pytest.raises(ValueError, match="object"):
        service.add_messages(["invalid"])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="serializable"):
        service.add_messages([{"content": "valid", "metadata": {"bad": object()}}])
    with pytest.raises(ValueError, match="too large"):
        service.add_messages([{"content": "valid", "metadata": {"large": "x" * 20_001}}])


def test_response_uses_query_focused_excerpt_and_confidence(tmp_path) -> None:
    service = SearchService(tmp_path / "messages.db", prefer_faiss=False)
    content = (
        "prefix " * 100
        + "The circuit breaker prevents retry storms during outages. "
        + "suffix " * 100
    )
    service.add_messages([{"content": content}])

    response = service.respond("How does the circuit breaker prevent retry storms?")

    assert "circuit breaker" in response["answer"]
    assert response["answer"].startswith("Based on")
    assert response["confidence"] in {"low", "medium", "high"}
