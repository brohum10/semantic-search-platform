import pytest

from semantic_search.lexical_index import Bm25Index


def test_bm25_ranks_rare_repeated_terms_and_reports_matches() -> None:
    index = Bm25Index()
    index.add(
        [1, 2, 3],
        [
            "database storage basics",
            "database database database indexing",
            "network protocol basics",
        ],
    )

    results = index.search("database indexing", 3)

    assert [result.message_id for result in results] == [2, 1]
    assert results[0].matched_terms == ("database", "indexing")
    assert results[0].score > results[1].score


def test_bm25_remove_and_empty_paths() -> None:
    index = Bm25Index()
    index.add([1], ["one unique token"])

    assert len(index) == 1
    assert index.vocabulary_size == 3
    assert index.search("", 5) == []
    assert index.search("unknown", 5) == []
    assert index.search("unique", 0) == []
    assert index.remove(404) is False
    assert index.remove(1) is True
    assert len(index) == 0
    assert index.vocabulary_size == 0


def test_bm25_validates_configuration_and_duplicate_documents() -> None:
    with pytest.raises(ValueError, match="k1"):
        Bm25Index(k1=0)
    with pytest.raises(ValueError, match="between"):
        Bm25Index(b=2)
    with pytest.raises(ValueError, match="frequency"):
        Bm25Index(max_document_frequency_ratio=0)

    index = Bm25Index()
    index.add([1], ["content"])
    with pytest.raises(ValueError, match="already"):
        index.add([1], ["replacement"])
    with pytest.raises(ValueError):
        index.add([2, 3], ["only one text"])


def test_bm25_skips_corpus_wide_terms_when_selective_terms_exist() -> None:
    index = Bm25Index(max_document_frequency_ratio=0.5)
    index.add(
        [1, 2, 3],
        ["common alpha", "common beta", "common gamma"],
    )

    results = index.search("common beta", 3)

    assert [result.message_id for result in results] == [2]
