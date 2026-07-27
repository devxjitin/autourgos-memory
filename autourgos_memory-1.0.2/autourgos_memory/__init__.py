"""
autourgos-memory — Base memory interfaces for Autourgos.

Install the full suite::

    pip install autourgos-memory autourgos-buffer-memory autourgos-local-memory
    pip install autourgos-semantic-memory autourgos-summary-memory autourgos-token-memory

Quick imports::

    from autourgos_memory import BaseMemory, MemoryMessage, Document, BaseRetriever
"""
import logging

from .base import BaseMemory, BaseRetriever, Document, MemoryMessage

logger = logging.getLogger(__name__)

# soft re-exports of concrete implementations
try:
    from autourgos_buffer_memory import RuntimeShortTermMemory, ConversationBufferMemory
except ImportError:
    pass
try:
    from autourgos_local_memory import LocalShortTermMemory, SQLiteMemory
except ImportError:
    pass
try:
    from autourgos_semantic_memory import KeywordRetriever, KeywordMemory, SimpleSemanticRetriever, HierarchicalSemanticMemory
except ImportError:
    pass
try:
    from autourgos_summary_memory import SummaryBufferedMemory
except ImportError:
    pass
try:
    from autourgos_token_memory import TokenBufferedMemory
except ImportError:
    pass

try:
    from importlib.metadata import version as _v
    __version__ = _v("autourgos-memory")
except Exception:
    logger.debug("could not resolve installed version for autourgos-memory", exc_info=True)
    __version__ = "1.0.2"

__all__ = [
    "BaseMemory", "BaseRetriever", "Document", "MemoryMessage",
    "RuntimeShortTermMemory", "ConversationBufferMemory",
    "LocalShortTermMemory", "SQLiteMemory",
    "KeywordRetriever", "KeywordMemory", "SimpleSemanticRetriever", "HierarchicalSemanticMemory",
    "SummaryBufferedMemory",
    "TokenBufferedMemory",
]
