def test_api_indexes_searches_and_reports_health(tmp_path) -> None:
    from semantic_search.api import create_app

    app = create_app(tmp_path / "api.db", prefer_faiss=False)
    client = app.test_client()

    created = client.post("/v1/messages", json={"content": "FAISS vector search with content validation"})
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
    assert "query" in response.json["error"]


def test_bulk_and_response_endpoints(tmp_path) -> None:
    from semantic_search.api import create_app

    client = create_app(tmp_path / "api.db", prefer_faiss=False).test_client()
    bulk = client.post("/v1/messages/bulk", json={"messages": [
        {"content": "Use idempotency keys when retrying writes."},
        {"content": "Use a bounded queue for backpressure."},
    ]})
    response = client.post("/v1/respond", json={"query": "How should writes be retried?", "limit": 2})
    invalid = client.post("/v1/messages/bulk", json={"messages": "not-an-array"})

    assert bulk.status_code == 201
    assert bulk.json["created"] == 2
    assert response.status_code == 200
    assert response.json["sources"]
    assert invalid.status_code == 400
