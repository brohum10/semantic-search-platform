from datetime import UTC, datetime

from semantic_search.store import MessageStore


def test_store_round_trips_messages(tmp_path) -> None:
    store = MessageStore(tmp_path / "messages.db")
    created = store.add_many(
        [("hello search", datetime(2026, 1, 1, tzinfo=UTC), {"source": "test"})]
    )

    restored = store.list_all()

    assert created == restored
    assert store.count() == 1
    assert store.get_many([]) == {}


def test_store_get_delete_and_close(tmp_path) -> None:
    store = MessageStore(tmp_path / "messages.db")
    created = store.add_many([("delete me", datetime(2026, 1, 1, tzinfo=UTC), {})])[0]

    assert store.get(created.id) == created
    assert store.get(404) is None
    assert store.delete(404) is False
    assert store.delete(created.id) is True
    assert store.count() == 0
    store.close()
