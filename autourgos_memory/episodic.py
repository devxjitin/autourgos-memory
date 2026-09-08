"""
episodic.py — Structured task/outcome log for autonomous agents.

Unlike the freeform conversational recall the rest of the memory family
provides, EpisodicMemory records *what was tried and what happened*: a
task description, the actions taken, and an explicit outcome. Retrieval
surfaces past episodes relevant to a new task -- e.g. "last time you tried
this, it failed because X" -- so an agent can avoid repeating a known-bad
approach.

Persisted to SQLite (survives process restarts); relevance scoring reuses
this package's own tested TF-IDF KeywordRetriever (see
:mod:`autourgos_memory.semantic`) rather than duplicating that algorithm.
"""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, List, Optional

from autourgos_core import open_sqlite, row_cap_evict

from .semantic import KeywordRetriever
from .base import BaseRetriever, Document

_VALID_OUTCOMES = {"success", "fail", "partial"}


class EpisodicMemoryError(Exception):
    """Raised for misuse of EpisodicMemory (invalid outcome, etc.)."""


@dataclass(frozen=True)
class Episode:
    """A single recorded task attempt."""
    task: str
    outcome: str
    actions_taken: List[str] = field(default_factory=list)
    notes: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        if not self.task or not self.task.strip():
            raise EpisodicMemoryError("task must be a non-empty string.")
        if self.outcome not in _VALID_OUTCOMES:
            raise EpisodicMemoryError(
                f"outcome must be one of {sorted(_VALID_OUTCOMES)}, got {self.outcome!r}."
            )

    def to_document(self) -> Document:
        """Render this episode as a Document for retrieval/display."""
        actions = "; ".join(self.actions_taken) if self.actions_taken else "(none recorded)"
        content = (
            f"Task: {self.task}\n"
            f"Outcome: {self.outcome}\n"
            f"Actions taken: {actions}"
        )
        if self.notes:
            content += f"\nNotes: {self.notes}"
        return Document(
            content=content,
            metadata={
                "task": self.task,
                "outcome": self.outcome,
                "actions_taken": self.actions_taken,
                "notes": self.notes,
                "timestamp": self.timestamp.astimezone(timezone.utc).isoformat(),
            },
        )


class EpisodicMemory(BaseRetriever):
    """SQLite-persisted, TF-IDF-retrieved log of task attempts and outcomes.

    Parameters
    ----------
    db_path : str
        SQLite file path. ``":memory:"`` (default) keeps everything in RAM
        and discards it when the process exits; pass a real file path for
        recall across restarts.
    max_episodes : int, optional
        Oldest episodes are dropped once this count is exceeded. ``None``
        (default) keeps everything.
    """

    def __init__(self, db_path: str = ":memory:", max_episodes: Optional[int] = None) -> None:
        if max_episodes is not None and (not isinstance(max_episodes, int) or max_episodes < 1):
            raise ValueError("max_episodes must be an integer >= 1 or None")

        self.max_episodes = max_episodes
        self._lock = threading.RLock()
        self._retriever = KeywordRetriever(max_documents=max_episodes)

        self._conn = open_sqlite(db_path)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS episodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task TEXT NOT NULL,
                outcome TEXT NOT NULL,
                actions_taken TEXT NOT NULL,
                notes TEXT NOT NULL,
                timestamp TEXT NOT NULL
            )
            """
        )
        self._conn.commit()
        self._rehydrate()

    def _rehydrate(self) -> None:
        """Reload persisted episodes into the in-process TF-IDF index."""
        rows = self._conn.execute(
            "SELECT task, outcome, actions_taken, notes, timestamp FROM episodes ORDER BY id ASC"
        ).fetchall()
        for task, outcome, actions_json, notes, ts in rows:
            episode = Episode(
                task=task,
                outcome=outcome,
                actions_taken=json.loads(actions_json),
                notes=notes,
                timestamp=datetime.fromisoformat(ts),
            )
            self._retriever.add_document(episode.to_document())

    def remember(
        self,
        task: str,
        outcome: str,
        actions_taken: Optional[List[str]] = None,
        notes: str = "",
    ) -> Episode:
        """Record a task attempt. outcome must be 'success', 'fail', or 'partial'."""
        episode = Episode(
            task=task,
            outcome=outcome,
            actions_taken=list(actions_taken or []),
            notes=notes,
        )
        with self._lock:
            self._conn.execute(
                "INSERT INTO episodes (task, outcome, actions_taken, notes, timestamp) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    episode.task,
                    episode.outcome,
                    json.dumps(episode.actions_taken),
                    episode.notes,
                    episode.timestamp.astimezone(timezone.utc).isoformat(),
                ),
            )
            self._conn.commit()
            if self.max_episodes is not None:
                row_cap_evict(self._conn, "episodes", "id", self.max_episodes)
                self._conn.commit()
            self._retriever.add_document(episode.to_document())
        return episode

    def retrieve(self, query: str, top_k: int = 5) -> List[Document]:
        with self._lock:
            return self._retriever.retrieve(query, top_k=top_k)

    def clear(self) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM episodes")
            self._conn.commit()
            self._retriever.clear()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> "EpisodicMemory":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()
