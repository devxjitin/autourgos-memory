"""
autourgos-memory — Unified memory package for Autourgos agents.

Base interfaces (BaseMemory, MemoryMessage, Document, BaseRetriever) plus
every concrete memory implementation in one package, selectable by direct
import::

    from autourgos_memory import RuntimeShortTermMemory
    memory = RuntimeShortTermMemory(max_messages=20)

VectorMemory/VectorRetriever require the ``autourgos-memory[vector]`` extra
(numpy); if numpy isn't installed those names are simply absent from this
module's namespace.
"""
import logging

from .base import (
    ROLE_TO_OPENAI,
    BaseMemory,
    BaseRetriever,
    Document,
    MemoryMessage,
    RetrievalAugmentedMemory,
    format_conversation_banner,
)
from .buffer import RuntimeShortTermMemory, ConversationBufferMemory, ExpiringBufferMemory
from .local import LocalShortTermMemory, SQLiteMemory
from .semantic import (
    tokenize,
    KeywordRetriever,
    KeywordMemory,
    SimpleSemanticRetriever,
    HierarchicalSemanticMemory,
)
from .summary import SummaryBufferedMemory
from .token import TokenBufferedMemory
from .episodic import Episode, EpisodicMemory, EpisodicMemoryError

logger = logging.getLogger(__name__)

__all__ = [
    "BaseMemory", "BaseRetriever", "Document", "MemoryMessage",
    "RetrievalAugmentedMemory", "format_conversation_banner", "ROLE_TO_OPENAI",
    "RuntimeShortTermMemory", "ConversationBufferMemory", "ExpiringBufferMemory",
    "LocalShortTermMemory", "SQLiteMemory",
    "tokenize", "KeywordRetriever", "KeywordMemory", "SimpleSemanticRetriever", "HierarchicalSemanticMemory",
    "SummaryBufferedMemory",
    "TokenBufferedMemory",
    "Episode", "EpisodicMemory", "EpisodicMemoryError",
]

# VectorMemory/VectorRetriever need numpy (the `[vector]` extra); keep the
# rest of the package importable when it isn't installed.
try:
    from .vector import VectorMemory, VectorMemoryError, VectorRetriever
    __all__ += ["VectorMemory", "VectorRetriever", "VectorMemoryError"]
except ImportError:
    pass

from autourgos_core import package_version

__version__ = package_version("autourgos-memory", fallback="2.1.0", logger=logger)
