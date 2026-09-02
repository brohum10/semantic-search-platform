from __future__ import annotations

import json
import math
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .embeddings import TOKEN_PATTERN, HashingEmbedder
from .lexical_index import Bm25Index
from .models import Message, SearchResult
from .store import MessageStore
from .vector_index import VectorIndex


class SearchService:
    def __init__(
        self,
        database_path: str | Path,
        *,
        dimension: int = 128,
        similarity_weight: float = 0.70,
        recency_weight: float = 0.10,
        lexical_weight: float = 0.20,
        recency_half_life_days: float = 30.0,
        prefer_faiss: bool = True,
    ) -> None:
        if not math.isclose(similarity_weight + recency_weight + lexical_weight, 1.0):
            raise ValueError("ranking weights must sum to 1")
        if recency_half_life_days <= 0:
            raise ValueError("recency half-life must be positive")
        self.store = MessageStore(database_path)
        self.embedder = HashingEmbedder(dimension)
        self.index = VectorIndex(dimension, prefer_faiss=prefer_faiss)
        self.lexical_index = Bm25Index()
        self.similarity_weight = similarity_weight
        self.recency_weight = recency_weight
        self.lexical_weight = lexical_weight
        self.recency_half_life_days = recency_half_life_days
        self._lock = threading.RLock()
        existing = self.store.list_all()
        self.index.add(
            [message.id for message in existing],
            self.embedder.embed_many(m.content for m in existing),
        )
        self.lexical_index.add(
            [message.id for message in existing], (message.content for message in existing)
        )

    @property
    def backend(self) -> str:
        return self.index.backend

    def add_messages(self, payloads: list[dict[str, Any]]) -> list[Message]:
        if not payloads:
            raise ValueError("at least one message is required")
        if len(payloads) > 1_000:
            raise ValueError("a batch may contain at most 1000 messages")
        prepared: list[tuple[str, datetime, dict[str, Any]]] = []
        for payload in payloads:
            if not isinstance(payload, dict):
                raise ValueError("each message must be an object")
            content = self._validate_text(
                payload.get("content"), field="content", max_length=10_000
            )
            created_at = self._parse_datetime(payload.get("created_at"))
            metadata = payload.get("metadata", {})
            if not isinstance(metadata, dict):
                raise ValueError("metadata must be an object")
            try:
                encoded_metadata = json.dumps(metadata, sort_keys=True, separators=(",", ":"))
            except (TypeError, ValueError) as error:
                raise ValueError("metadata must be JSON serializable") from error
            if len(encoded_metadata.encode("utf-8")) > 20_000:
                raise ValueError("metadata is too large")
            prepared.append((content, created_at, metadata))
        with self._lock:
            messages = self.store.add_many(prepared)
            identifiers = [message.id for message in messages]
            self.index.add(identifiers, self.embedder.embed_many(m.content for m in messages))
            self.lexical_index.add(identifiers, (message.content for message in messages))
        return messages

    def get_message(self, message_id: int) -> Message:
        message = self.store.get(message_id)
        if message is None:
            raise KeyError(f"message {message_id} was not found")
        return message

    def delete_message(self, message_id: int) -> None:
        with self._lock:
            if not self.store.delete(message_id):
                raise KeyError(f"message {message_id} was not found")
            self.index.remove(message_id)
            self.lexical_index.remove(message_id)

    def search(
        self,
        query: str,
        *,
        limit: int = 10,
        now: datetime | None = None,
        metadata: dict[str, Any] | None = None,
        created_after: datetime | str | None = None,
        created_before: datetime | str | None = None,
    ) -> list[SearchResult]:
        query = self._validate_text(query, field="query", max_length=1_000)
        if limit < 1 or limit > 50:
            raise ValueError("limit must be between 1 and 50")
        if metadata is not None and not isinstance(metadata, dict):
            raise ValueError("metadata filter must be an object")
        after = self._parse_optional_datetime(created_after, field="created_after")
        before = self._parse_optional_datetime(created_before, field="created_before")
        if after and before and after > before:
            raise ValueError("created_after must not be later than created_before")

        query_vector = self.embedder.embed(query)
        has_filters = bool(metadata) or after is not None or before is not None
        candidate_limit = (
            len(self.index) if has_filters else min(max(limit * 10, 100), len(self.index))
        )
        with self._lock:
            vector_candidates = self.index.search(query_vector, candidate_limit)
            lexical_candidates = self.lexical_index.search(query, candidate_limit)
            candidate_ids = {
                candidate.message_id for candidate in [*vector_candidates, *lexical_candidates]
            }
            messages = self.store.get_many(list(candidate_ids))
            similarities = self.index.similarities(query_vector, candidate_ids)

        reference_time = self._parse_datetime(now) if now is not None else datetime.now(UTC)
        query_terms = set(TOKEN_PATTERN.findall(query.lower()))
        lexical_scores = {neighbor.message_id: neighbor.score for neighbor in lexical_candidates}
        max_lexical = max(lexical_scores.values(), default=0.0)
        results: list[SearchResult] = []
        for message_id in candidate_ids:
            message = messages.get(message_id)
            if message is None or not self._matches_filters(message, metadata or {}, after, before):
                continue
            age_days = max(0.0, (reference_time - message.created_at).total_seconds() / 86_400)
            recency = math.exp(-math.log(2) * age_days / self.recency_half_life_days)
            raw_similarity = similarities.get(message_id, 0.0)
            similarity = max(0.0, min(1.0, (raw_similarity + 1.0) / 2.0))
            content_terms = set(TOKEN_PATTERN.findall(message.content.lower()))
            matched_terms = tuple(sorted(query_terms & content_terms))
            lexical_raw = lexical_scores.get(message_id, 0.0)
            lexical = lexical_raw / max_lexical if max_lexical else 0.0
            score = (
                self.similarity_weight * similarity
                + self.recency_weight * recency
                + self.lexical_weight * lexical
            )
            results.append(
                SearchResult(
                    message,
                    similarity,
                    recency,
                    lexical,
                    score,
                    matched_terms,
                    {
                        "raw_cosine_similarity": raw_similarity,
                        "bm25_score": lexical_raw,
                        "semantic_contribution": self.similarity_weight * similarity,
                        "recency_contribution": self.recency_weight * recency,
                        "lexical_contribution": self.lexical_weight * lexical,
                    },
                )
            )
        results.sort(key=lambda result: (result.score, result.message.id), reverse=True)
        return results[:limit]

    def respond(self, query: str, *, limit: int = 3) -> dict[str, Any]:
        results = self.search(query, limit=limit)
        if not results:
            return {"answer": "No relevant indexed context was found.", "sources": []}
        query_terms = set(TOKEN_PATTERN.findall(query.lower()))
        excerpts = [self._excerpt(result.message.content, 280, query_terms) for result in results]
        return {
            "answer": "Based on the indexed context: " + " ".join(excerpts),
            "sources": [result.to_dict() for result in results],
            "mode": "extractive",
            "confidence": self._confidence(results[0].score),
        }

    def health(self) -> dict[str, Any]:
        return {"status": "ok", "messages": self.store.count(), "vector_backend": self.backend}

    def stats(self) -> dict[str, Any]:
        return {
            "messages": self.store.count(),
            "vector_backend": self.backend,
            "embedding_dimension": self.embedder.dimension,
            "lexical_vocabulary": self.lexical_index.vocabulary_size,
            "ranking_weights": {
                "semantic": self.similarity_weight,
                "lexical": self.lexical_weight,
                "recency": self.recency_weight,
            },
        }

    def close(self) -> None:
        self.store.close()

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

    @classmethod
    def _parse_optional_datetime(cls, value: Any, *, field: str) -> datetime | None:
        if value is None:
            return None
        try:
            return cls._parse_datetime(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{field} must be an ISO-8601 timestamp") from error

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
    def _matches_filters(
        message: Message,
        metadata: dict[str, Any],
        created_after: datetime | None,
        created_before: datetime | None,
    ) -> bool:
        if created_after and message.created_at < created_after:
            return False
        if created_before and message.created_at > created_before:
            return False
        for path, expected in metadata.items():
            value: Any = message.metadata
            for part in path.split("."):
                if not isinstance(value, dict) or part not in value:
                    return False
                value = value[part]
            if value != expected:
                return False
        return True

    @staticmethod
    def _confidence(score: float) -> str:
        if score >= 0.75:
            return "high"
        if score >= 0.55:
            return "medium"
        return "low"

    @staticmethod
    def _excerpt(content: str, limit: int, query_terms: set[str]) -> str:
        if len(content) <= limit:
            return content
        lowered = content.lower()
        positions = [lowered.find(term) for term in query_terms if lowered.find(term) >= 0]
        center = min(positions, default=0)
        start = max(0, center - limit // 3)
        end = min(len(content), start + limit)
        start = max(0, end - limit)
        excerpt = content[start:end].strip()
        return ("..." if start else "") + excerpt + ("..." if end < len(content) else "")
