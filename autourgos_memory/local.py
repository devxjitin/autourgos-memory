"""
local.py — Disk-backed memory: JSON file and SQLite.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from errno import EEXIST
from typing import Any, Generator, List, Optional
from uuid import uuid4

from autourgos_core import open_sqlite, row_cap_evict

from .base import BaseMemory, MemoryMessage, format_conversation_banner

logger = logging.getLogger(__name__)


# ── LocalShortTermMemory ───────────────────────────────────────────────────────

class LocalShortTermMemory(BaseMemory):
    """Disk-backed short-term memory stored as a JSON file.

    Thread-safe via a file-level lock. Uses atomic write (tmp → replace) to
    prevent corruption. Safe for multiple processes reading the same file.

    Parameters
    ----------
    file_path : str
        Path to the JSON memory file. Created automatically if missing.
    max_messages : int
        Rolling cap — oldest messages are pruned after each write.
    name : str
        Human-readable identifier.
    lock_timeout_seconds : float
        Seconds to wait when acquiring the file lock before raising TimeoutError.
    """

    def __init__(
        self,
        file_path: str = "./data/local_memory.json",
        max_messages: int = 20,
        name: str = "local",
        create_if_missing: bool = True,
        lock_timeout_seconds: float = 10.0,
    ) -> None:
        if not isinstance(file_path, str) or not file_path.strip():
            raise ValueError("file_path must be a non-empty string")
        if not isinstance(max_messages, int) or max_messages < 1:
            raise ValueError("max_messages must be an integer >= 1")
        self.file_path = file_path
        self.max_messages = max_messages
        self.name = name
        self.lock_timeout_seconds = lock_timeout_seconds
        self._thread_lock = threading.RLock()
        self._lock_path = f"{os.path.abspath(self.file_path)}.lock"
        folder = os.path.dirname(os.path.abspath(self.file_path))
        if folder:
            os.makedirs(folder, exist_ok=True)
        if create_if_missing:
            with self._file_lock():
                if not os.path.exists(self.file_path):
                    self._atomic_write(self.file_path, [])

    @contextmanager
    def _file_lock(self) -> Generator[None, None, None]:
        deadline = time.time() + self.lock_timeout_seconds
        fd = None
        while True:
            with self._thread_lock:
                try:
                    fd = os.open(self._lock_path, os.O_CREAT | os.O_EXCL | os.O_RDWR)
                    break
                except OSError as exc:
                    if exc.errno != EEXIST:
                        raise
                    self._clear_stale_lock()
            if time.time() >= deadline:
                raise TimeoutError(f"Timed out acquiring memory lock for {self.file_path}")
            time.sleep(0.01)
        try:
            yield
        finally:
            os.close(fd)
            try:
                os.remove(self._lock_path)
            except FileNotFoundError:
                pass

    def _clear_stale_lock(self) -> None:
        try:
            stat = os.stat(self._lock_path)
        except FileNotFoundError:
            return
        if (time.time() - stat.st_mtime) <= self.lock_timeout_seconds:
            return
        try:
            os.remove(self._lock_path)
        except FileNotFoundError:
            pass

    def _atomic_write(self, path: str, payload: list) -> None:
        tmp = f"{path}.{uuid4().hex}.tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        os.replace(tmp, path)

    def _load(self) -> List[MemoryMessage]:
        with self._file_lock():
            if not os.path.exists(self.file_path):
                return []
            with open(self.file_path, "r", encoding="utf-8-sig") as fh:
                raw = fh.read().strip()
            if not raw:
                return []
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                # Same recovery as add_message(): a corrupted file is treated
                # as an empty history rather than propagating a crash to
                # every read path (get_messages/format_for_llm). Previously
                # this branch raised while add_message() silently repaired
                # the same corruption -- inconsistent behavior for the two
                # halves of the same file.
                logger.warning(
                    "memory file %s contained invalid JSON; treating as an empty list",
                    self.file_path,
                    exc_info=True,
                )
                return []
        if not isinstance(payload, list):
            raise ValueError("Memory file must contain a JSON array")
        return [MemoryMessage.from_dict(item) for item in payload]

    def add_message(self, role: str, content: str, timestamp: Optional[datetime] = None) -> MemoryMessage:
        msg = MemoryMessage(role=role, content=content, timestamp=timestamp or datetime.now(timezone.utc))
        with self._file_lock():
            try:
                with open(self.file_path, "r", encoding="utf-8-sig") as fh:
                    raw = fh.read().strip()
                payload = json.loads(raw) if raw else []
            except FileNotFoundError:
                payload = []
            except json.JSONDecodeError:
                logger.warning(
                    "memory file %s contained invalid JSON; starting from an empty list",
                    self.file_path,
                    exc_info=True,
                )
                payload = []
            messages = [MemoryMessage.from_dict(item) for item in payload]
            messages.append(msg)
            if len(messages) > self.max_messages:
                messages = messages[-self.max_messages:]
            self._atomic_write(self.file_path, [m.to_dict() for m in messages])
        return msg

    def get_messages(self) -> List[MemoryMessage]:
        return self._load()

    def clear(self) -> None:
        with self._file_lock():
            self._atomic_write(self.file_path, [])

    def format_for_llm(self, query: Optional[str] = None) -> str:
        return format_conversation_banner(self._load())


# ── SQLiteMemory ───────────────────────────────────────────────────────────────

class SQLiteMemory(BaseMemory):
    """Persistent, thread-safe memory backed by SQLite (WAL mode).

    Compared to :class:`LocalShortTermMemory` (JSON file), SQLite is safer
    under concurrent writes — no external lock file needed — and large
    histories are cheap to query.

    Parameters
    ----------
    db_path : str
        Path to the ``.db`` file. Use ``":memory:"`` for an ephemeral in-process DB.
    max_messages : int or None
        Rolling cap. ``None`` = unlimited.
    name : str
        Human-readable identifier.
    """

    def __init__(
        self,
        db_path: str = "./data/autourgos_memory.db",
        max_messages: Optional[int] = 500,
        name: str = "sqlite",
    ) -> None:
        self.db_path = db_path
        self.max_messages = max_messages
        self.name = name
        self._lock = threading.RLock()
        self._conn = open_sqlite(db_path)
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS messages (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                role      TEXT    NOT NULL,
                content   TEXT    NOT NULL,
                timestamp TEXT    NOT NULL
            )"""
        )
        self._conn.commit()

    def add_message(self, role: str, content: str, timestamp: Optional[datetime] = None) -> MemoryMessage:
        ts = timestamp or datetime.now(timezone.utc)
        msg = MemoryMessage(role=role, content=content, timestamp=ts)
        with self._lock:
            self._conn.execute(
                "INSERT INTO messages (role, content, timestamp) VALUES (?, ?, ?)",
                (role, content, ts.isoformat()),
            )
            if self.max_messages is not None:
                row_cap_evict(self._conn, "messages", "id", self.max_messages)
            self._conn.commit()
        return msg

    def get_messages(self, limit: Optional[int] = None) -> List[MemoryMessage]:
        with self._lock:
            if limit is not None:
                rows = self._conn.execute(
                    "SELECT role, content, timestamp FROM messages ORDER BY id DESC LIMIT ?", (limit,)
                ).fetchall()
                rows = list(reversed(rows))
            else:
                rows = self._conn.execute(
                    "SELECT role, content, timestamp FROM messages ORDER BY id ASC"
                ).fetchall()
        result = []
        for role, content, ts_str in rows:
            dt = datetime.fromisoformat(ts_str)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            result.append(MemoryMessage(role=role, content=content, timestamp=dt))
        return result

    def format_for_llm(self, query: Optional[str] = None) -> str:
        return format_conversation_banner(self.get_messages())

    def clear(self) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM messages")
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> "SQLiteMemory":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"SQLiteMemory(db_path={self.db_path!r}, name={self.name!r})"
