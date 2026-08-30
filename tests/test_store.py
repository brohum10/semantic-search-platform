from datetime import UTC, datetime

from semantic_search.store import MessageStore


def test_store_round_trips_messages(tmp_path) -> None:
    store = MessageStore(tmp_path / "messages.db")
    created = store.add_many([("hello search", datetime(2026, 1, 1, tzinfo=UTC), {"source": "test"})])

    restored = store.list_all()

    assert created == restored
    assert store.count() == 1
    assert store.get_many([]) == {}
