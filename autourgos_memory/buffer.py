"""
buffer.py — In-memory short-term buffers.
"""
from __future__ import annotations
import sys
import threading
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

from .base import BaseMemory, MemoryMessage, ROLE_TO_OPENAI, format_conversation_banner


class RuntimeShortTermMemory(BaseMemory):
    """In-memory ring buffer bounded by message count.

    Parameters
    ----------
    max_messages : int
        Maximum messages kept. Oldest are dropped when exceeded. Default 20.
    name : str
        Human-readable identifier.
    """

    def __init__(self, max_messages: int = 20, name: str = "runtime") -> None:
        if not isinstance(max_messages, int) or max_messages < 1:
            raise ValueError("max_messages must be an integer >= 1")
        self.max_messages = max_messages
        self.name = name
        self._messages: List[MemoryMessage] = []

    def add_message(self, role: str, content: str, timestamp: Optional[datetime] = None) -> MemoryMessage:
        msg = MemoryMessage(role=role, content=content, timestamp=timestamp or datetime.now(timezone.utc))
        self._messages.append(msg)
        if len(self._messages) > self.max_messages:
            self._messages = self._messages[-self.max_messages:]
        return msg

    def get_messages(self) -> List[Dict[str, str]]:
        return [{"role": ROLE_TO_OPENAI.get(m.role, m.role), "content": m.content} for m in self._messages]

    def clear(self) -> None:
        self._messages = []

    def format_for_llm(self, query: Optional[str] = None) -> str:
        return format_conversation_banner(self._messages, include_timestamps=False)


class ConversationBufferMemory(RuntimeShortTermMemory):
    """Unbounded in-memory conversation buffer (no truncation).

    Use this when you want to keep every message in RAM for the session.
    For long conversations, prefer :class:`RuntimeShortTermMemory` with a cap,
    or :class:`~autourgos_memory.summary.SummaryBufferedMemory` for LLM compression.
    """

    def __init__(self, name: str = "conversation") -> None:
        super().__init__(max_messages=sys.maxsize, name=name)


class ExpiringBufferMemory(BaseMemory):
    """In-memory buffer where each message carries a time-to-live.

    Meant for a long-running (possibly background) agent's temporary,
    run-scoped facts -- things worth remembering for the next few minutes
    or hours of a task, but that should not silently persist into a later,
    unrelated run the way an unbounded buffer would. Expired messages are
    purged lazily (on the next add/read call, no background thread) and
    never appear in ``get_messages()``/``format_for_llm()``.

    Parameters
    ----------
    default_ttl_seconds : float, optional
        Applied to a message when ``add_message()``/``add_user_message()``
        etc. don't pass their own ``ttl_seconds``. ``None`` (default) means
        messages never expire unless a per-call ``ttl_seconds`` is given.
    max_messages : int, optional
        Ring-buffer cap on *live* (non-expired) messages. ``None`` (default)
        is unbounded.
    name : str
        Human-readable identifier.
    """

    def __init__(
        self,
        default_ttl_seconds: Optional[float] = None,
        max_messages: Optional[int] = None,
        name: str = "expiring",
    ) -> None:
        if default_ttl_seconds is not None and default_ttl_seconds <= 0:
            raise ValueError("default_ttl_seconds must be > 0 or None")
        if max_messages is not None and (not isinstance(max_messages, int) or max_messages < 1):
            raise ValueError("max_messages must be an integer >= 1 or None")
        self.default_ttl_seconds = default_ttl_seconds
        self.max_messages = max_messages
        self.name = name
        self._entries: List[Tuple[MemoryMessage, Optional[datetime]]] = []
        self._lock = threading.RLock()

    def _purge_expired(self) -> None:
        now = datetime.now(timezone.utc)
        self._entries = [
            (msg, expires_at) for msg, expires_at in self._entries
            if expires_at is None or expires_at > now
        ]

    def add_message(
        self,
        role: str,
        content: str,
        timestamp: Optional[datetime] = None,
        ttl_seconds: Optional[float] = None,
    ) -> MemoryMessage:
        with self._lock:
            ts = timestamp or datetime.now(timezone.utc)
            msg = MemoryMessage(role=role, content=content, timestamp=ts)
            ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl_seconds
            expires_at = ts + timedelta(seconds=ttl) if ttl is not None else None
            self._purge_expired()
            self._entries.append((msg, expires_at))
            if self.max_messages is not None and len(self._entries) > self.max_messages:
                self._entries = self._entries[-self.max_messages:]
            return msg

    def add_user_message(self, content: str, ttl_seconds: Optional[float] = None) -> MemoryMessage:
        return self.add_message("user", content, ttl_seconds=ttl_seconds)

    def add_agent_message(self, content: str, ttl_seconds: Optional[float] = None) -> MemoryMessage:
        return self.add_message("agent", content, ttl_seconds=ttl_seconds)

    def add_system_message(self, content: str, ttl_seconds: Optional[float] = None) -> MemoryMessage:
        return self.add_message("system", content, ttl_seconds=ttl_seconds)

    def add_tool_message(self, tool_name: str, result: str, ttl_seconds: Optional[float] = None) -> MemoryMessage:
        return self.add_message("tool", f"[{tool_name} returned]: {result}", ttl_seconds=ttl_seconds)

    def get_messages(self) -> List[Dict[str, str]]:
        with self._lock:
            self._purge_expired()
            return [{"role": ROLE_TO_OPENAI.get(m.role, m.role), "content": m.content} for m, _ in self._entries]

    def clear(self) -> None:
        with self._lock:
            self._entries = []

    def format_for_llm(self, query: Optional[str] = None) -> str:
        with self._lock:
            self._purge_expired()
            return format_conversation_banner([m for m, _ in self._entries], include_timestamps=False)
