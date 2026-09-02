def test_api_indexes_searches_and_reports_health(tmp_path) -> None:
    from semantic_search.api import create_app

    app = create_app(tmp_path / "api.db", prefer_faiss=False)
    client = app.test_client()

    created = client.post(
        "/v1/messages", json={"content": "FAISS vector search with content validation"}
    )
    searched = client.get("/v1/search?q=vector+search&limit=5")
    health = client.get("/v1/health")

    assert created.status_code == 201
    assert searched.status_code == 200
    assert searched.json["results"][0]["content"].startswith("FAISS")
    assert health.json == {"messages": 1, "status": "ok", "vector_backend": "numpy"}


def test_api_returns_validation_errors(tmp_path) -> None:
    from semantic_search.api import create_app

    client = create_app(tmp_path / "api.db", prefer_faiss=False).test_client()

    response = client.get("/v1/search?q=&limit=10")

    assert response.status_code == 400
    assert response.json["error"]["code"] == "invalid_request"
    assert "query" in response.json["error"]["message"]
    assert response.headers["X-Request-ID"]


def test_bulk_and_response_endpoints(tmp_path) -> None:
    from semantic_search.api import create_app

    client = create_app(tmp_path / "api.db", prefer_faiss=False).test_client()
    bulk = client.post(
        "/v1/messages/bulk",
        json={
            "messages": [
                {"content": "Use idempotency keys when retrying writes."},
                {"content": "Use a bounded queue for backpressure."},
            ]
        },
    )
    response = client.post(
        "/v1/respond", json={"query": "How should writes be retried?", "limit": 2}
    )
    invalid = client.post("/v1/messages/bulk", json={"messages": "not-an-array"})

    assert bulk.status_code == 201
    assert bulk.json["created"] == 2
    assert response.status_code == 200
    assert response.json["sources"]
    assert invalid.status_code == 400


def test_filtered_search_message_lifecycle_and_stats(tmp_path) -> None:
    from semantic_search.api import create_app

    client = create_app(tmp_path / "api.db", prefer_faiss=False).test_client()
    first = client.post(
        "/v1/messages",
        json={
            "content": "PostgreSQL indexes accelerate account queries",
            "created_at": "2026-01-02T00:00:00Z",
            "metadata": {"team": {"name": "platform"}},
        },
    ).json
    client.post(
        "/v1/messages",
        json={"content": "PostgreSQL backup policy", "metadata": {"team": {"name": "security"}}},
    )

    searched = client.post(
        "/v1/search",
        json={
            "query": "PostgreSQL indexes",
            "filters": {
                "metadata": {"team.name": "platform"},
                "created_after": "2026-01-01T00:00:00Z",
            },
        },
    )
    fetched = client.get(f"/v1/messages/{first['id']}")
    stats = client.get("/v1/stats")
    deleted = client.delete(f"/v1/messages/{first['id']}")
    missing = client.get(f"/v1/messages/{first['id']}")

    assert searched.status_code == 200
    assert searched.json["count"] == 1
    assert searched.json["results"][0]["matched_terms"] == ["indexes", "postgresql"]
    assert fetched.json["id"] == first["id"]
    assert stats.json["messages"] == 2
    assert stats.json["lexical_vocabulary"] > 0
    assert deleted.status_code == 204
    assert missing.status_code == 404
    assert missing.json["error"]["code"] == "not_found"


def test_api_rejects_bad_filters_limits_and_oversized_requests(tmp_path, monkeypatch) -> None:
    from semantic_search.api import create_app

    monkeypatch.setenv("SEARCH_MAX_REQUEST_BYTES", "80")
    client = create_app(tmp_path / "api.db", prefer_faiss=False).test_client()

    bad_filters = client.post("/v1/search", json={"query": "valid", "filters": []})
    bad_limit = client.get("/v1/search?q=valid&limit=many")
    oversized = client.post("/v1/messages", json={"content": "x" * 200})

    assert bad_filters.status_code == 400
    assert bad_limit.status_code == 400
    assert oversized.status_code == 413
