from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .models import Message


class MessageStore:
    def __init__(self, database_path: str | Path) -> None:
        self._connection = sqlite3.connect(str(database_path), check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._connection:
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                )
            """)

    def add_many(self, items: Iterable[tuple[str, datetime, dict[str, Any]]]) -> list[Message]:
        messages: list[Message] = []
        with self._lock, self._connection:
            for content, created_at, metadata in items:
                cursor = self._connection.execute(
                    "INSERT INTO messages(content, created_at, metadata_json) VALUES (?, ?, ?)",
                    (
                        content,
                        created_at.astimezone(UTC).isoformat(),
                        json.dumps(metadata, sort_keys=True),
                    ),
                )
                messages.append(
                    Message(int(cursor.lastrowid), content, created_at.astimezone(UTC), metadata)
                )
        return messages

    def list_all(self) -> list[Message]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT id, content, created_at, metadata_json FROM messages ORDER BY id"
            ).fetchall()
        return [self._from_row(row) for row in rows]

    def get_many(self, message_ids: list[int]) -> dict[int, Message]:
        if not message_ids:
            return {}
        placeholders = ",".join("?" for _ in message_ids)
        with self._lock:
            rows = self._connection.execute(
                "SELECT id, content, created_at, metadata_json "
                f"FROM messages WHERE id IN ({placeholders})",
                message_ids,
            ).fetchall()
        return {int(row["id"]): self._from_row(row) for row in rows}

    def get(self, message_id: int) -> Message | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT id, content, created_at, metadata_json FROM messages WHERE id = ?",
                (message_id,),
            ).fetchone()
        return self._from_row(row) if row is not None else None

    def delete(self, message_id: int) -> bool:
        with self._lock, self._connection:
            cursor = self._connection.execute("DELETE FROM messages WHERE id = ?", (message_id,))
        return cursor.rowcount == 1

    def count(self) -> int:
        with self._lock:
            row = self._connection.execute("SELECT COUNT(*) AS total FROM messages").fetchone()
        return int(row["total"])

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    @staticmethod
    def _from_row(row: sqlite3.Row) -> Message:
        return Message(
            int(row["id"]),
            str(row["content"]),
            datetime.fromisoformat(str(row["created_at"])),
            json.loads(str(row["metadata_json"])),
        )
