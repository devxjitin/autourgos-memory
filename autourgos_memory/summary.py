"""
summary.py — LLM-compressed rolling summary memory.
"""
from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from typing import Any, List, Optional

from .base import BaseMemory, MemoryMessage

logger = logging.getLogger(__name__)


class SummaryBufferedMemory(BaseMemory):
    """Memory that compresses older history into a rolling LLM summary.

    Keeps the last ``max_messages`` messages in full. When the buffer
    overflows, excess messages are fed to the LLM for summarization and
    merged into ``moving_summary``. If no LLM is provided, raw
    concatenation is used as a fallback.

    Parameters
    ----------
    llm : any, optional
        LLM with ``.invoke(prompt: str)`` method. If omitted, history is
        concatenated verbatim (no compression).
    max_messages : int
        Number of recent messages to keep in full. Default 10.
    moving_summary : str
        Seed summary to start with (optional).
    max_summary_chars : int, optional
        Cap on ``moving_summary``'s length for the raw-concatenation
        fallback path (used when no ``llm`` is given, and whenever LLM
        compression fails). Without a real LLM call to actually compress
        anything, that fallback previously appended full raw message text
        to ``moving_summary`` forever with no limit -- oldest content is
        now trimmed from the front (kept: the most recent overflow) once
        this cap is exceeded. ``None`` (default) disables the cap, matching
        prior behavior. Does not affect the successful-LLM-compression
        path, which already produces a bounded, model-written summary.
    """

    def __init__(
        self,
        llm: Optional[Any] = None,
        max_messages: int = 10,
        moving_summary: str = "",
        max_summary_chars: Optional[int] = None,
    ) -> None:
        if not isinstance(max_messages, int) or max_messages < 1:
            raise ValueError("max_messages must be an integer >= 1")
        if max_summary_chars is not None and (not isinstance(max_summary_chars, int) or max_summary_chars < 1):
            raise ValueError("max_summary_chars must be an integer >= 1 or None")
        self.llm = llm
        self.max_messages = max_messages
        self.moving_summary = moving_summary
        self.max_summary_chars = max_summary_chars
        self._messages: List[MemoryMessage] = []
        self._lock = threading.RLock()

    def _bound_summary(self, summary: str) -> str:
        """Trim *summary* to max_summary_chars, keeping the tail (most
        recent content) -- only used for the raw-concatenation fallback,
        never for a successfully LLM-compressed summary."""
        if self.max_summary_chars is None or len(summary) <= self.max_summary_chars:
            return summary
        marker = "[...earlier summary trimmed...]\n"
        keep = max(self.max_summary_chars - len(marker), 0)
        return marker + summary[-keep:] if keep else marker.strip()

    def _fallback_concatenation(self, to_summarize: List[MemoryMessage]) -> str:
        lines = [f"[{m.role}]: {m.content}" for m in to_summarize]
        prefix = f"{self.moving_summary}\n" if self.moving_summary else ""
        return self._bound_summary(prefix + "\n".join(lines))

    def _update_summary(self, to_summarize: List[MemoryMessage]) -> None:
        if not to_summarize:
            return
        if not self.llm:
            self.moving_summary = self._fallback_concatenation(to_summarize)
            return
        prompt = (
            "You are a conversation summarizer.\n"
            "Compress the conversation below and integrate it with the existing summary.\n\n"
            f"Existing Summary:\n{self.moving_summary or 'No existing summary.'}\n\n"
            "New messages to summarize:\n"
        )
        for m in to_summarize:
            prompt += f"[{m.role}]: {m.content}\n"
        prompt += "\nProvide a concise, updated summary of the entire conversation so far."
        try:
            response = self.llm.invoke(prompt)
            if isinstance(response, dict):
                summary = response.get("response", response.get("content", ""))
            else:
                summary = response
            if summary is None:
                # str(None) == "None" -- without this check, an LLM
                # returning None (e.g. a misbehaving wrapper, or a mocked
                # LLM in tests) permanently baked the literal string "None"
                # into the summary instead of falling back like any other
                # unusable response.
                raise ValueError("llm.invoke() returned None")
            stripped = str(summary).strip()
            if not stripped:
                raise ValueError("llm.invoke() returned an empty summary")
            self.moving_summary = stripped
        except Exception as exc:
            logger.warning("Summary compression failed: %s. Falling back to raw concatenation.", exc)
            self.moving_summary = self._fallback_concatenation(to_summarize)

    def add_message(self, role: str, content: str, timestamp: Optional[datetime] = None) -> MemoryMessage:
        with self._lock:
            msg = MemoryMessage(role=role, content=content, timestamp=timestamp or datetime.now(timezone.utc))
            self._messages.append(msg)
            if len(self._messages) > self.max_messages:
                overflow = self._messages[:-self.max_messages]
                self._messages = self._messages[-self.max_messages:]
                self._update_summary(overflow)
            return msg

    def get_messages(self) -> List[MemoryMessage]:
        with self._lock:
            return list(self._messages)

    def format_for_llm(self, query: Optional[str] = None) -> str:
        with self._lock:
            parts = []
            if self.moving_summary:
                parts.append(
                    "--- Summary of Past Conversation ---\n"
                    f"{self.moving_summary}\n"
                    "------------------------------------"
                )
            if self._messages:
                recent = "\n".join(f"[{m.timestamp.isoformat()}] {m.role}: {m.content}" for m in self._messages)
                parts.append(
                    "--- Recent Conversation Context ---\n"
                    f"{recent}\n"
                    "-----------------------------------"
                )
            return "\n\n".join(parts)

    def clear(self) -> None:
        with self._lock:
            self._messages.clear()
            self.moving_summary = ""
