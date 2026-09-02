from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class Neighbor:
    message_id: int
    similarity: float


class VectorIndex:
    """Cosine-similarity index backed by FAISS when available, NumPy otherwise."""

    def __init__(self, dimension: int, *, prefer_faiss: bool = True) -> None:
        self.dimension = dimension
        self._ids = np.empty(0, dtype=np.int64)
        self._matrix = np.empty((0, dimension), dtype=np.float32)
        self._faiss = None
        self._index = None
        self._lock = threading.RLock()
        if prefer_faiss:
            try:
                import faiss  # type: ignore

                self._faiss = faiss
                self._index = faiss.IndexFlatIP(dimension)
            except ImportError:
                pass

    @property
    def backend(self) -> str:
        return "faiss" if self._index is not None else "numpy"

    def __len__(self) -> int:
        with self._lock:
            return int(self._ids.size)

    def add(self, message_ids: list[int], vectors: NDArray[np.float32]) -> None:
        if not message_ids:
            return
        matrix = np.ascontiguousarray(vectors, dtype=np.float32)
        if matrix.shape != (len(message_ids), self.dimension):
            raise ValueError("vector batch has the wrong shape")
        with self._lock:
            if len(set(message_ids)) != len(message_ids) or any(
                message_id in self._ids for message_id in message_ids
            ):
                raise ValueError("message IDs must be unique")
            self._ids = np.concatenate((self._ids, np.asarray(message_ids, dtype=np.int64)))
            self._matrix = np.vstack((self._matrix, matrix))
            if self._index is not None:
                self._index.add(matrix)

    def remove(self, message_id: int) -> bool:
        with self._lock:
            matches = np.flatnonzero(self._ids == message_id)
            if matches.size == 0:
                return False
            keep = self._ids != message_id
            self._ids = self._ids[keep]
            self._matrix = self._matrix[keep]
            if self._index is not None:
                self._index.reset()
                if self._matrix.size:
                    self._index.add(np.ascontiguousarray(self._matrix, dtype=np.float32))
            return True

    def search(self, query: NDArray[np.float32], limit: int) -> list[Neighbor]:
        with self._lock:
            if len(self) == 0 or limit < 1:
                return []
            limit = min(limit, len(self))
            vector = np.ascontiguousarray(query.reshape(1, self.dimension), dtype=np.float32)
            if self._index is not None:
                scores, positions = self._index.search(vector, limit)
                return [
                    Neighbor(int(self._ids[position]), float(score))
                    for score, position in zip(scores[0], positions[0], strict=True)
                    if position >= 0
                ]
            scores = self._matrix @ vector[0]
            positions = np.argpartition(scores, -limit)[-limit:]
            positions = positions[np.argsort(scores[positions])[::-1]]
            return [
                Neighbor(int(self._ids[position]), float(scores[position]))
                for position in positions
            ]

    def similarities(self, query: NDArray[np.float32], message_ids: set[int]) -> dict[int, float]:
        """Calculate exact cosine scores for an arbitrary candidate set."""
        if not message_ids:
            return {}
        with self._lock:
            positions = [
                position for position, value in enumerate(self._ids) if int(value) in message_ids
            ]
            scores = self._matrix[positions] @ query
            return {
                int(self._ids[position]): float(score)
                for position, score in zip(positions, scores, strict=True)
            }
