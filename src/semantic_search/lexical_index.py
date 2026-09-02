from __future__ import annotations

import math
import threading
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

from .embeddings import TOKEN_PATTERN


@dataclass(frozen=True, slots=True)
class LexicalNeighbor:
    message_id: int
    score: float
    matched_terms: tuple[str, ...]


class Bm25Index:
    """Small in-memory BM25 index used alongside dense retrieval.

    The implementation favors inspectability over cleverness: postings contain
    term frequencies, while document lengths and corpus statistics are updated
    incrementally as messages are ingested.
    """

    def __init__(
        self,
        *,
        k1: float = 1.5,
        b: float = 0.75,
        max_document_frequency_ratio: float = 0.8,
    ) -> None:
        if k1 <= 0 or not 0 <= b <= 1:
            raise ValueError("BM25 requires k1 > 0 and b between 0 and 1")
        if not 0 < max_document_frequency_ratio <= 1:
            raise ValueError("max document-frequency ratio must be between 0 and 1")
        self.k1 = k1
        self.b = b
        self.max_document_frequency_ratio = max_document_frequency_ratio
        self._postings: dict[str, dict[int, int]] = defaultdict(dict)
        self._document_lengths: dict[int, int] = {}
        self._total_tokens = 0
        self._lock = threading.RLock()

    def __len__(self) -> int:
        with self._lock:
            return len(self._document_lengths)

    @property
    def vocabulary_size(self) -> int:
        with self._lock:
            return len(self._postings)

    def add(self, message_ids: Iterable[int], texts: Iterable[str]) -> None:
        pairs = list(zip(message_ids, texts, strict=True))
        with self._lock:
            for message_id, text in pairs:
                if message_id in self._document_lengths:
                    raise ValueError(f"message {message_id} is already indexed")
                frequencies = Counter(TOKEN_PATTERN.findall(text.lower()))
                length = sum(frequencies.values())
                self._document_lengths[message_id] = length
                self._total_tokens += length
                for term, frequency in frequencies.items():
                    self._postings[term][message_id] = frequency

    def remove(self, message_id: int) -> bool:
        with self._lock:
            length = self._document_lengths.pop(message_id, None)
            if length is None:
                return False
            self._total_tokens -= length
            empty_terms: list[str] = []
            for term, posting in self._postings.items():
                posting.pop(message_id, None)
                if not posting:
                    empty_terms.append(term)
            for term in empty_terms:
                del self._postings[term]
            return True

    def search(self, query: str, limit: int) -> list[LexicalNeighbor]:
        terms = tuple(dict.fromkeys(TOKEN_PATTERN.findall(query.lower())))
        if limit < 1 or not terms:
            return []
        with self._lock:
            document_count = len(self._document_lengths)
            if document_count == 0:
                return []
            average_length = self._total_tokens / document_count or 1.0
            scores: dict[int, float] = defaultdict(float)
            matches: dict[int, set[str]] = defaultdict(set)
            available = [(term, self._postings[term]) for term in terms if term in self._postings]
            selected = [
                (term, posting)
                for term, posting in available
                if len(posting) / document_count <= self.max_document_frequency_ratio
            ]
            if not selected and available:
                smallest_posting = min(len(posting) for _, posting in available)
                selected = [
                    (term, posting)
                    for term, posting in available
                    if len(posting) == smallest_posting
                ]
            for term, posting in selected:
                document_frequency = len(posting)
                inverse_frequency = math.log(
                    1 + (document_count - document_frequency + 0.5) / (document_frequency + 0.5)
                )
                for message_id, frequency in posting.items():
                    length = self._document_lengths[message_id]
                    denominator = frequency + self.k1 * (
                        1 - self.b + self.b * length / average_length
                    )
                    scores[message_id] += (
                        inverse_frequency * frequency * (self.k1 + 1) / denominator
                    )
                    matches[message_id].add(term)
            ordered = sorted(
                scores, key=lambda message_id: (scores[message_id], message_id), reverse=True
            )
            return [
                LexicalNeighbor(message_id, scores[message_id], tuple(sorted(matches[message_id])))
                for message_id in ordered[:limit]
            ]
