from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from uuid import uuid4

from flask import Flask, g, jsonify, request
from werkzeug.exceptions import RequestEntityTooLarge

from .service import SearchService


def create_app(database_path: str | Path | None = None, *, prefer_faiss: bool = True) -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = int(os.environ.get("SEARCH_MAX_REQUEST_BYTES", "1048576"))
    resolved_path = database_path or os.environ.get("SEARCH_DATABASE", "data/messages.db")
    if resolved_path != ":memory:":
        Path(resolved_path).parent.mkdir(parents=True, exist_ok=True)
    service = SearchService(resolved_path, prefer_faiss=prefer_faiss)
    app.extensions["search_service"] = service

    @app.before_request
    def assign_request_id() -> None:
        g.request_id = request.headers.get("X-Request-ID", str(uuid4()))[:128]

    @app.after_request
    def add_response_headers(response: Any) -> Any:
        response.headers["X-Request-ID"] = g.request_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    @app.get("/v1/health")
    def health() -> tuple[Any, int]:
        return jsonify(service.health()), 200

    @app.get("/v1/stats")
    def stats() -> tuple[Any, int]:
        return jsonify(service.stats()), 200

    @app.post("/v1/messages")
    def add_message() -> tuple[Any, int]:
        payload = request.get_json(silent=True) or {}
        messages = service.add_messages([payload])
        return jsonify(_message_dict(messages[0])), 201

    @app.post("/v1/messages/bulk")
    def add_messages() -> tuple[Any, int]:
        payload = request.get_json(silent=True) or {}
        messages = payload.get("messages")
        if not isinstance(messages, list):
            raise ValueError("messages must be an array")
        created = service.add_messages(messages)
        return jsonify(
            {"created": len(created), "messages": [_message_dict(item) for item in created]}
        ), 201

    @app.get("/v1/messages/<int:message_id>")
    def get_message(message_id: int) -> tuple[Any, int]:
        return jsonify(_message_dict(service.get_message(message_id))), 200

    @app.delete("/v1/messages/<int:message_id>")
    def delete_message(message_id: int) -> tuple[str, int]:
        service.delete_message(message_id)
        return "", 204

    @app.get("/v1/search")
    def search() -> tuple[Any, int]:
        query = request.args.get("q", "")
        limit = _parse_limit(request.args.get("limit", "10"))
        results = service.search(query, limit=limit)
        return jsonify(_search_response(query, results)), 200

    @app.post("/v1/search")
    def search_with_filters() -> tuple[Any, int]:
        payload = request.get_json(silent=True) or {}
        query = payload.get("query", "")
        limit = _parse_limit(payload.get("limit", 10))
        filters = payload.get("filters", {})
        if not isinstance(filters, dict):
            raise ValueError("filters must be an object")
        results = service.search(
            query,
            limit=limit,
            metadata=filters.get("metadata"),
            created_after=filters.get("created_after"),
            created_before=filters.get("created_before"),
        )
        return jsonify(_search_response(query, results)), 200

    @app.post("/v1/respond")
    def respond() -> tuple[Any, int]:
        payload = request.get_json(silent=True) or {}
        query = payload.get("query", "")
        limit = _parse_limit(payload.get("limit", 3))
        return jsonify(service.respond(query, limit=limit)), 200

    @app.errorhandler(ValueError)
    def invalid_request(error: ValueError) -> tuple[Any, int]:
        return jsonify({"error": {"code": "invalid_request", "message": str(error)}}), 400

    @app.errorhandler(KeyError)
    def not_found(error: KeyError) -> tuple[Any, int]:
        return jsonify({"error": {"code": "not_found", "message": str(error.args[0])}}), 404

    @app.errorhandler(RequestEntityTooLarge)
    def request_too_large(error: RequestEntityTooLarge) -> tuple[Any, int]:
        return jsonify({"error": {"code": "request_too_large", "message": error.description}}), 413

    return app


def _message_dict(message: Any) -> dict[str, Any]:
    return {
        "id": message.id,
        "content": message.content,
        "created_at": message.created_at.isoformat(),
        "metadata": message.metadata,
    }


def _search_response(query: str, results: list[Any]) -> dict[str, Any]:
    return {
        "query": query,
        "count": len(results),
        "results": [result.to_dict() for result in results],
    }


def _parse_limit(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as error:
        raise ValueError("limit must be an integer") from error


def main() -> None:
    app = create_app()
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", "8080")))


if __name__ == "__main__":
    main()
