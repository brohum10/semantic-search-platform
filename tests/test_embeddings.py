import numpy as np

from semantic_search.embeddings import HashingEmbedder


def test_embedding_is_deterministic_and_normalized() -> None:
    embedder = HashingEmbedder(64)
    first = embedder.embed("reliable distributed systems")
    second = embedder.embed("reliable distributed systems")

    np.testing.assert_array_equal(first, second)
    assert np.isclose(np.linalg.norm(first), 1.0)


def test_similar_text_has_higher_similarity() -> None:
    embedder = HashingEmbedder(128)
    query = embedder.embed("database indexing performance")
    related = embedder.embed("database indexing and query performance")
    unrelated = embedder.embed("mobile color palette typography")

    assert float(query @ related) > float(query @ unrelated)


def test_empty_batch_and_invalid_dimension() -> None:
    embedder = HashingEmbedder(32)

    assert embedder.embed_many([]).shape == (0, 32)
    with np.testing.assert_raises(ValueError):
        HashingEmbedder(16)
