from datetime import UTC, datetime, timedelta

import pytest

from semantic_search.service import SearchService


def test_search_combines_similarity_and_recency(tmp_path) -> None:
    now = datetime(2026, 8, 30, tzinfo=UTC)
    service = SearchService(tmp_path / "messages.db", prefer_faiss=False)
    service.add_messages([
        {"content": "database indexing performance", "created_at": (now - timedelta(days=180)).isoformat()},
        {"content": "database indexing performance", "created_at": (now - timedelta(hours=1)).isoformat()},
    ])

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
    message = service.add_messages([{"content": "timestamp test", "created_at": "2026-08-30T12:00:00"}])[0]

    assert message.created_at.tzinfo is UTC


def test_rejects_invalid_weight_configuration(tmp_path) -> None:
    with pytest.raises(ValueError, match="weights"):
        SearchService(tmp_path / "messages.db", similarity_weight=0.7, recency_weight=0.2)


@pytest.mark.parametrize("content", ["", "   ", "bad\x00content"])
def test_rejects_invalid_content(tmp_path, content: str) -> None:
    service = SearchService(tmp_path / "messages.db", prefer_faiss=False)

    with pytest.raises(ValueError):
        service.add_messages([{"content": content}])
