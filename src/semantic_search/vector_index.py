from __future__ import annotations

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
        return int(self._ids.size)

    def add(self, message_ids: list[int], vectors: NDArray[np.float32]) -> None:
        if not message_ids:
            return
        matrix = np.ascontiguousarray(vectors, dtype=np.float32)
        if matrix.shape != (len(message_ids), self.dimension):
            raise ValueError("vector batch has the wrong shape")
        self._ids = np.concatenate((self._ids, np.asarray(message_ids, dtype=np.int64)))
        if self._index is not None:
            self._index.add(matrix)
        else:
            self._matrix = np.vstack((self._matrix, matrix))

    def search(self, query: NDArray[np.float32], limit: int) -> list[Neighbor]:
        if len(self) == 0 or limit < 1:
            return []
        limit = min(limit, len(self))
        vector = np.ascontiguousarray(query.reshape(1, self.dimension), dtype=np.float32)
        if self._index is not None:
            scores, positions = self._index.search(vector, limit)
            return [
                Neighbor(int(self._ids[position]), float(score))
                for score, position in zip(scores[0], positions[0])
                if position >= 0
            ]
        scores = self._matrix @ vector[0]
        positions = np.argpartition(scores, -limit)[-limit:]
        positions = positions[np.argsort(scores[positions])[::-1]]
        return [Neighbor(int(self._ids[position]), float(scores[position])) for position in positions]
