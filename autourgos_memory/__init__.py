"""
autourgos-memory — Base memory interfaces for Autourgos.

Install the full suite::

    pip install autourgos-memory autourgos-buffer-memory autourgos-local-memory
    pip install autourgos-semantic-memory autourgos-summary-memory autourgos-token-memory
    pip install autourgos-vector-memory autourgos-episodic-memory

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
    from autourgos_vector_memory import VectorMemory, VectorRetriever
except ImportError:
    pass
try:
    from autourgos_episodic_memory import Episode, EpisodicMemory
except ImportError:
    pass

from autourgos_core import package_version

__version__ = package_version("autourgos-memory", fallback="1.1.0", logger=logger)

__all__ = [
    "BaseMemory", "BaseRetriever", "Document", "MemoryMessage",
    "RuntimeShortTermMemory", "ConversationBufferMemory",
    "LocalShortTermMemory", "SQLiteMemory",
    "KeywordRetriever", "KeywordMemory", "SimpleSemanticRetriever", "HierarchicalSemanticMemory",
    "SummaryBufferedMemory",
    "TokenBufferedMemory",
    "VectorMemory", "VectorRetriever",
    "Episode", "EpisodicMemory",
]
