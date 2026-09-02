from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from collections.abc import Iterable

import numpy as np
from numpy.typing import NDArray

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


class HashingEmbedder:
    """Dependency-light, deterministic text vectors suitable for local demos.

    Production deployments can replace this class with a hosted or local neural
    embedding adapter without changing storage, ranking, or API code.
    """

    def __init__(self, dimension: int = 128) -> None:
        if dimension < 32:
            raise ValueError("dimension must be at least 32")
        self.dimension = dimension

    def embed(self, text: str) -> NDArray[np.float32]:
        tokens = TOKEN_PATTERN.findall(text.lower())
        features = tokens + [
            f"{left}_{right}" for left, right in zip(tokens, tokens[1:], strict=False)
        ]
        counts = Counter(features)
        vector = np.zeros(self.dimension, dtype=np.float32)
        for feature, count in counts.items():
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=16).digest()
            index = int.from_bytes(digest[:8], "little") % self.dimension
            sign = 1.0 if digest[8] & 1 else -1.0
            vector[index] += sign * (1.0 + math.log(count))
        norm = float(np.linalg.norm(vector))
        if norm > 0:
            vector /= norm
        return vector

    def embed_many(self, texts: Iterable[str]) -> NDArray[np.float32]:
        vectors = [self.embed(text) for text in texts]
        if not vectors:
            return np.empty((0, self.dimension), dtype=np.float32)
        return np.ascontiguousarray(np.vstack(vectors), dtype=np.float32)
