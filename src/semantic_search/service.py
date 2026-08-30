from __future__ import annotations

import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .embeddings import TOKEN_PATTERN, HashingEmbedder
from .models import Message, SearchResult
from .store import MessageStore
from .vector_index import VectorIndex


class SearchService:
    def __init__(
        self,
        database_path: str | Path,
        *,
        dimension: int = 128,
        similarity_weight: float = 0.85,
        recency_weight: float = 0.10,
        lexical_weight: float = 0.05,
        recency_half_life_days: float = 30.0,
        prefer_faiss: bool = True,
    ) -> None:
        if not math.isclose(similarity_weight + recency_weight + lexical_weight, 1.0):
            raise ValueError("ranking weights must sum to 1")
        self.store = MessageStore(database_path)
        self.embedder = HashingEmbedder(dimension)
        self.index = VectorIndex(dimension, prefer_faiss=prefer_faiss)
        self.similarity_weight = similarity_weight
        self.recency_weight = recency_weight
        self.lexical_weight = lexical_weight
        self.recency_half_life_days = recency_half_life_days
        existing = self.store.list_all()
        self.index.add([message.id for message in existing], self.embedder.embed_many(m.content for m in existing))

    @property
    def backend(self) -> str:
        return self.index.backend

    def add_messages(self, payloads: list[dict[str, Any]]) -> list[Message]:
        if not payloads:
            raise ValueError("at least one message is required")
        prepared: list[tuple[str, datetime, dict[str, Any]]] = []
        for payload in payloads:
            content = self._validate_text(payload.get("content"), field="content", max_length=10_000)
            created_at = self._parse_datetime(payload.get("created_at"))
            metadata = payload.get("metadata", {})
            if not isinstance(metadata, dict):
                raise ValueError("metadata must be an object")
            if len(str(metadata)) > 20_000:
                raise ValueError("metadata is too large")
            prepared.append((content, created_at, metadata))
        messages = self.store.add_many(prepared)
        self.index.add([message.id for message in messages], self.embedder.embed_many(m.content for m in messages))
        return messages

    def search(self, query: str, *, limit: int = 10, now: datetime | None = None) -> list[SearchResult]:
        query = self._validate_text(query, field="query", max_length=1_000)
        if limit < 1 or limit > 50:
            raise ValueError("limit must be between 1 and 50")
        candidates = self.index.search(self.embedder.embed(query), min(max(limit * 5, 50), len(self.index)))
        messages = self.store.get_many([neighbor.message_id for neighbor in candidates])
        reference_time = (now or datetime.now(UTC)).astimezone(UTC)
        query_terms = set(TOKEN_PATTERN.findall(query.lower()))
        results: list[SearchResult] = []
        for neighbor in candidates:
            message = messages[neighbor.message_id]
            age_days = max(0.0, (reference_time - message.created_at).total_seconds() / 86_400)
            recency = math.exp(-math.log(2) * age_days / self.recency_half_life_days)
            similarity = (neighbor.similarity + 1.0) / 2.0
            content_terms = set(TOKEN_PATTERN.findall(message.content.lower()))
            lexical = len(query_terms & content_terms) / max(1, len(query_terms))
            score = (
                self.similarity_weight * similarity
                + self.recency_weight * recency
                + self.lexical_weight * lexical
            )
            results.append(SearchResult(message, similarity, recency, lexical, score))
        results.sort(key=lambda result: (result.score, result.message.id), reverse=True)
        return results[:limit]

    def respond(self, query: str, *, limit: int = 3) -> dict[str, Any]:
        results = self.search(query, limit=limit)
        if not results:
            return {"answer": "No relevant indexed context was found.", "sources": []}
        excerpts = [self._excerpt(result.message.content, 280) for result in results]
        return {
            "answer": "Based on the indexed context: " + " ".join(excerpts),
            "sources": [result.to_dict() for result in results],
            "mode": "extractive",
        }

    def health(self) -> dict[str, Any]:
        return {"status": "ok", "messages": self.store.count(), "vector_backend": self.backend}

    @staticmethod
    def _parse_datetime(value: Any) -> datetime:
        if value is None:
            return datetime.now(UTC)
        if isinstance(value, datetime):
            parsed = value
        elif isinstance(value, str):
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        else:
            raise ValueError("created_at must be an ISO-8601 timestamp")
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return parsed.astimezone(UTC)

    @staticmethod
    def _validate_text(value: Any, *, field: str, max_length: int) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field} must be a non-empty string")
        normalized = " ".join(value.split())
        if len(normalized) > max_length:
            raise ValueError(f"{field} exceeds {max_length} characters")
        if any(ord(character) < 32 and character not in "\t\n\r" for character in value):
            raise ValueError(f"{field} contains unsupported control characters")
        return normalized

    @staticmethod
    def _excerpt(content: str, limit: int) -> str:
        return content if len(content) <= limit else content[: limit - 3].rstrip() + "..."
