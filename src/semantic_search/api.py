from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request

from .service import SearchService


def create_app(database_path: str | Path | None = None, *, prefer_faiss: bool = True) -> Flask:
    app = Flask(__name__)
    resolved_path = database_path or os.environ.get("SEARCH_DATABASE", "data/messages.db")
    if resolved_path != ":memory:":
        Path(resolved_path).parent.mkdir(parents=True, exist_ok=True)
    service = SearchService(resolved_path, prefer_faiss=prefer_faiss)
    app.extensions["search_service"] = service

    @app.get("/v1/health")
    def health() -> tuple[Any, int]:
        return jsonify(service.health()), 200

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
        return jsonify({"created": len(created), "messages": [_message_dict(item) for item in created]}), 201

    @app.get("/v1/search")
    def search() -> tuple[Any, int]:
        query = request.args.get("q", "")
        limit = _parse_limit(request.args.get("limit", "10"))
        results = service.search(query, limit=limit)
        return jsonify({"query": query, "results": [result.to_dict() for result in results]}), 200

    @app.post("/v1/respond")
    def respond() -> tuple[Any, int]:
        payload = request.get_json(silent=True) or {}
        query = payload.get("query", "")
        limit = _parse_limit(payload.get("limit", 3))
        return jsonify(service.respond(query, limit=limit)), 200

    @app.errorhandler(ValueError)
    def invalid_request(error: ValueError) -> tuple[Any, int]:
        return jsonify({"error": str(error)}), 400

    return app


def _message_dict(message: Any) -> dict[str, Any]:
    return {
        "id": message.id,
        "content": message.content,
        "created_at": message.created_at.isoformat(),
        "metadata": message.metadata,
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
