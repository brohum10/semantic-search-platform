import numpy as np
import pytest

from semantic_search.vector_index import VectorIndex


def test_numpy_backend_returns_nearest_vectors() -> None:
    index = VectorIndex(3, prefer_faiss=False)
    index.add(
        [10, 20, 30],
        np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32),
    )

    neighbors = index.search(np.asarray([0.9, 0.1, 0], dtype=np.float32), 2)

    assert [neighbor.message_id for neighbor in neighbors] == [10, 20]


def test_faiss_backend_and_validation() -> None:
    index = VectorIndex(2, prefer_faiss=True)
    index.add([1], np.asarray([[1, 0]], dtype=np.float32))

    assert index.backend in {"faiss", "numpy"}
    assert index.search(np.asarray([1, 0], dtype=np.float32), 1)[0].message_id == 1
    assert index.search(np.asarray([1, 0], dtype=np.float32), 0) == []
    with np.testing.assert_raises(ValueError):
        index.add([2], np.asarray([[1, 0, 0]], dtype=np.float32))


def test_index_scores_arbitrary_candidates_and_supports_deletion() -> None:
    index = VectorIndex(2, prefer_faiss=False)
    index.add([10, 20], np.asarray([[1, 0], [0, 1]], dtype=np.float32))

    scores = index.similarities(np.asarray([0.8, 0.2], dtype=np.float32), {10, 20, 404})

    assert scores == {10: pytest.approx(0.8), 20: pytest.approx(0.2)}
    assert index.remove(404) is False
    assert index.remove(10) is True
    assert [item.message_id for item in index.search(np.asarray([1, 0], dtype=np.float32), 5)] == [
        20
    ]
    assert index.similarities(np.asarray([1, 0], dtype=np.float32), set()) == {}


def test_index_rejects_duplicate_ids() -> None:
    index = VectorIndex(2, prefer_faiss=False)
    index.add([1], np.asarray([[1, 0]], dtype=np.float32))
    with np.testing.assert_raises(ValueError):
        index.add([1], np.asarray([[0, 1]], dtype=np.float32))
