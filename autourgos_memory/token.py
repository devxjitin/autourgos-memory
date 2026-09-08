"""
token.py — Token-bounded short-term memory.
"""
from __future__ import annotations

import logging
import threading
from collections import deque
from datetime import datetime, timezone
from typing import Callable, Deque, List, Optional

from .base import BaseMemory, MemoryMessage, format_conversation_banner

logger = logging.getLogger(__name__)


def _default_token_estimator(text: Optional[str]) -> int:
    """Estimate tokens. Uses tiktoken if installed, otherwise a heuristic."""
    if not text:
        return 0
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text, disallowed_special=()))
    except (ImportError, ModuleNotFoundError):
        pass
    except Exception:
        logger.warning(
            "tiktoken is installed but token estimation failed; "
            "falling back to heuristic estimator.",
            exc_info=True,
        )
    # Unicode-aware heuristic
    total = 0.0
    for ch in text:
        o = ord(ch)
        if (0x4E00 <= o <= 0x9FFF or 0x3040 <= o <= 0x309F or
                0x30A0 <= o <= 0x30FF or 0x1100 <= o <= 0x11FF or
                0xAC00 <= o <= 0xD7AF):
            total += 1.5
        elif ch in ('{', '}', '[', ']', '"', "'", ':', ',', ';', '=', '+', '-', '*', '/', '<', '>', '&', '|', '^', '%'):
            total += 0.5
        else:
            total += 0.25
    return max(1, int(total))


class TokenBufferedMemory(BaseMemory):
    """Short-term memory bounded by maximum token count.

    Oldest messages are evicted from the front whenever the total token
    count exceeds ``max_tokens``. Optionally uses ``tiktoken`` for accurate
    counts; falls back to a character-based heuristic if not installed.

    Parameters
    ----------
    max_tokens : int
        Token budget. Default 2000.
    token_estimator : callable, optional
        Custom ``(text: str) -> int`` estimator. Defaults to tiktoken / heuristic.
    oversized_message_policy : str
        What to do with a single new message whose own token cost already
        exceeds ``max_tokens`` (evicting every other message still wouldn't
        bring the buffer under budget). One of:

        * ``"keep"`` (default) -- store it anyway and log a warning. Matches
          prior behavior: never silently lose the newest message, but the
          token cap is knowingly violated for it.
        * ``"truncate"`` -- cut the message content down to fit the budget
          (appending a ``[TRUNCATED]`` marker) before storing it, so the cap
          is actually respected.
        * ``"drop"`` -- discard the message entirely and log a warning,
          leaving the rest of the buffer untouched.
    """

    _OVERSIZED_POLICIES = ("keep", "truncate", "drop")

    def __init__(
        self,
        max_tokens: int = 2000,
        token_estimator: Optional[Callable[[str], int]] = None,
        oversized_message_policy: str = "keep",
    ) -> None:
        if not isinstance(max_tokens, int) or max_tokens < 1:
            raise ValueError("max_tokens must be an integer >= 1")
        if oversized_message_policy not in self._OVERSIZED_POLICIES:
            raise ValueError(
                f"oversized_message_policy must be one of {self._OVERSIZED_POLICIES}, "
                f"got {oversized_message_policy!r}"
            )
        self.max_tokens                = max_tokens
        self.token_estimator           = token_estimator or _default_token_estimator
        self.oversized_message_policy  = oversized_message_policy
        self._messages: Deque[MemoryMessage] = deque()
        self._total_tokens = 0
        self._lock = threading.RLock()

    def _msg_tokens(self, msg: MemoryMessage) -> int:
        return self.token_estimator(msg.role) + self.token_estimator(msg.content)

    def _truncate_to_budget(self, role: str, content: str) -> str:
        """Return the largest prefix of *content* whose (role + prefix +
        truncation marker) token estimate fits within max_tokens.

        Best-effort: token_estimator isn't guaranteed to be exactly additive
        over concatenation (true for tiktoken/heuristic in general), but
        budgeting the marker's own cost up front keeps the common case
        within max_tokens instead of re-triggering the oversized-message
        path for the truncated result itself.
        """
        role_tokens   = self.token_estimator(role)
        marker        = " [TRUNCATED]"
        marker_tokens = self.token_estimator(marker)
        budget = self.max_tokens - role_tokens - marker_tokens
        if budget <= 0:
            return marker.strip()

        low, high, best = 0, len(content), 0
        while low <= high:
            mid = (low + high) // 2
            if self.token_estimator(content[:mid]) <= budget:
                best = mid
                low = mid + 1
            else:
                high = mid - 1

        if best >= len(content):
            return content
        return content[:best] + marker

    def _enforce_limit(self) -> None:
        while self._messages and self._total_tokens > self.max_tokens:
            if len(self._messages) == 1:
                only_msg = self._messages[0]
                if self._msg_tokens(only_msg) > self.max_tokens:
                    logger.warning(
                        "message exceeds max_tokens budget by itself; keeping anyway "
                        "(role=%r, tokens=%d, max_tokens=%d)",
                        only_msg.role, self._msg_tokens(only_msg), self.max_tokens,
                    )
                    break
            removed = self._messages.popleft()
            self._total_tokens -= self._msg_tokens(removed)

    def add_message(self, role: str, content: str, timestamp: Optional[datetime] = None) -> MemoryMessage:
        with self._lock:
            own_tokens = self.token_estimator(role) + self.token_estimator(content)

            if own_tokens > self.max_tokens and self.oversized_message_policy != "keep":
                if self.oversized_message_policy == "drop":
                    logger.warning(
                        "dropping oversized message that alone exceeds max_tokens "
                        "(role=%r, tokens=%d, max_tokens=%d)",
                        role, own_tokens, self.max_tokens,
                    )
                    return MemoryMessage(role=role, content=content, timestamp=timestamp or datetime.now(timezone.utc))
                # "truncate"
                truncated_content = self._truncate_to_budget(role, content)
                logger.info(
                    "truncating oversized message to fit max_tokens "
                    "(role=%r, original_tokens=%d, max_tokens=%d)",
                    role, own_tokens, self.max_tokens,
                )
                content = truncated_content

            msg = MemoryMessage(role=role, content=content, timestamp=timestamp or datetime.now(timezone.utc))
            self._messages.append(msg)
            self._total_tokens += self._msg_tokens(msg)
            self._enforce_limit()
            return msg

    def get_messages(self) -> List[MemoryMessage]:
        with self._lock:
            return list(self._messages)

    @property
    def total_tokens(self) -> int:
        with self._lock:
            return self._total_tokens

    def format_for_llm(self, query: Optional[str] = None) -> str:
        with self._lock:
            return format_conversation_banner(list(self._messages))

    def clear(self) -> None:
        with self._lock:
            self._messages.clear()
            self._total_tokens = 0
