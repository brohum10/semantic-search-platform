import numpy as np

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
