from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass(frozen=True, slots=True)
class Message:
    id: int
    content: str
    created_at: datetime
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class SearchResult:
    message: Message
    similarity: float
    recency: float
    lexical: float
    score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.message.id,
            "content": self.message.content,
            "created_at": self.message.created_at.isoformat(),
            "metadata": self.message.metadata,
            "similarity": round(self.similarity, 6),
            "recency": round(self.recency, 6),
            "lexical": round(self.lexical, 6),
            "score": round(self.score, 6),
        }
