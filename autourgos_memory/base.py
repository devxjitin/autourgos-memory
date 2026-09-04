"""
base.py — Core memory interfaces for Autourgos.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import warnings

_ALLOWED_ROLES = {"user", "agent", "system", "tool"}

ROLE_TO_OPENAI = {"user": "user", "agent": "assistant", "system": "system", "tool": "tool"}


@dataclass(frozen=True)
class MemoryMessage:
    """A single message in memory."""
    role: str
    content: str
    timestamp: datetime

    def __post_init__(self) -> None:
        if self.role not in _ALLOWED_ROLES:
            raise ValueError(f"Invalid role '{self.role}'. Allowed: {sorted(_ALLOWED_ROLES)}")
        if not isinstance(self.content, str):
            raise ValueError("content must be a string")
        if not isinstance(self.timestamp, datetime):
            raise ValueError("timestamp must be a datetime")

    def to_dict(self) -> Dict[str, str]:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.astimezone(timezone.utc).isoformat(),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "MemoryMessage":
        ts = payload.get("timestamp")
        if not isinstance(ts, str):
            raise ValueError("Invalid message payload: timestamp must be a string")
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return cls(
            role=str(payload.get("role", "")),
            content=str(payload.get("content", "")),
            timestamp=dt,
        )


class BaseMemory(ABC):
    """Abstract interface for agent memory."""

    @abstractmethod
    def add_message(self, role: str, content: str, timestamp: Optional[datetime] = None) -> MemoryMessage: ...

    def add_user_message(self, content: str) -> MemoryMessage:
        return self.add_message("user", content)

    def add_ai_message(self, content: str) -> MemoryMessage:
        warnings.warn("add_ai_message() is deprecated; use add_agent_message().", DeprecationWarning, stacklevel=2)
        return self.add_agent_message(content)

    def add_agent_message(self, content: str) -> MemoryMessage:
        if type(self).add_ai_message is not BaseMemory.add_ai_message:
            return self.add_ai_message(content)
        return self.add_message("agent", content)

    def add_system_message(self, content: str) -> MemoryMessage:
        return self.add_message("system", content)

    def add_tool_message(self, tool_name: str, result: str) -> MemoryMessage:
        return self.add_message("tool", f"[{tool_name} returned]: {result}")

    def get_context(self, query: Optional[str] = None) -> str:
        warnings.warn("get_context() is deprecated; use format_for_llm().", DeprecationWarning, stacklevel=2)
        return self.format_for_llm(query)

    def format_for_llm(self, query: Optional[str] = None) -> str:
        if type(self).get_context is not BaseMemory.get_context:
            return self.get_context(query)
        raise NotImplementedError("Subclasses must implement format_for_llm")

    @abstractmethod
    def clear(self) -> None: ...


def format_conversation_banner(messages: List[MemoryMessage], *, include_timestamps: bool = True) -> str:
    """Render messages as the standard "Previous Conversation Context" banner."""
    if not messages:
        return ""
    if include_timestamps:
        lines = "\n".join(f"[{m.timestamp.isoformat()}] {m.role}: {m.content}" for m in messages)
    else:
        lines = "\n".join(f"{m.role}: {m.content}" for m in messages)
    return f"\n--- Previous Conversation Context ---\n{lines}\n--------------------------------------\n"


@dataclass
class Document:
    """A retrieved document chunk."""
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    score: float = 0.0
    source: str = ""

    def __str__(self) -> str:
        src = f" (source: {self.source})" if self.source else ""
        return f"{self.content}{src}"


class BaseRetriever(ABC):
    """Abstract retriever interface."""

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> List[Document]: ...

    async def aretrieve(self, query: str, top_k: int = 5) -> List[Document]:
        import asyncio
        return await asyncio.to_thread(self.retrieve, query, top_k)


class RetrievalAugmentedMemory(BaseMemory):
    """Dual-store: sliding short-term buffer + retriever-backed long-term recall.

    Shared shape behind ``KeywordMemory`` (TF-IDF) and ``VectorMemory``
    (embeddings) -- every message goes to ``short_term`` (recent turns,
    always included) and is indexed into ``retriever`` (older, relevant
    turns, surfaced only when a query is given). Subclasses own
    constructing their own ``retriever``/``short_term`` and call
    ``super().__init__(short_term=..., retriever=..., top_k=...)``.
    """

    def __init__(self, *, short_term: BaseMemory, retriever: BaseRetriever, top_k: int = 3) -> None:
        self.short_term = short_term
        self.retriever = retriever
        self.top_k = top_k

    def _index(self, content: str, role: str, ts: datetime) -> None:
        self.retriever.add_document(Document(
            content=content,
            metadata={"role": role, "timestamp": ts.astimezone(timezone.utc).isoformat()},
        ))

    def add_message(self, role: str, content: str, timestamp: Optional[datetime] = None) -> MemoryMessage:
        msg = self.short_term.add_message(role, content, timestamp)
        self._index(content, role, msg.timestamp)
        return msg

    def add_tool_message(self, tool_name: str, result: str) -> MemoryMessage:
        msg = self.short_term.add_tool_message(tool_name, result)
        self._index(msg.content, "tool", msg.timestamp)
        return msg

    def format_for_llm(self, query: Optional[str] = None) -> str:
        st_context = self.short_term.format_for_llm()
        if not query:
            return st_context
        recent: set = set()
        get_msgs = getattr(self.short_term, "get_messages", None)
        if callable(get_msgs):
            recent = {
                m.content if hasattr(m, "content") else m.get("content", "")
                for m in get_msgs()
            }
        relevant = [d for d in self.retriever.retrieve(query, top_k=self.top_k) if d.content not in recent]
        if not relevant:
            return st_context
        past = "\n--- Relevant Past Context ---\n"
        for doc in relevant:
            prefix = f"[{doc.metadata['role']}]: " if "role" in doc.metadata else ""
            past += f"{prefix}{doc.content}\n"
        past += "-----------------------------\n\n"
        return past + st_context

    def clear(self) -> None:
        self.short_term.clear()
        self.retriever.clear()
