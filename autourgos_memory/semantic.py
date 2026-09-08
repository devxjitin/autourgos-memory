"""
semantic.py — TF-IDF keyword retrieval memory.

The short-term buffer used by ``KeywordMemory`` is provided by this
package's own ``RuntimeShortTermMemory`` (see :mod:`autourgos_memory.buffer`)
rather than a private duplicate of the same ring-buffer algorithm.
"""
from __future__ import annotations

import math
import re
import sys
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from .buffer import RuntimeShortTermMemory
from .base import BaseMemory, BaseRetriever, Document, MemoryMessage, RetrievalAugmentedMemory


def tokenize(text: str) -> List[str]:
    if not text:
        return []
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


# ── KeywordRetriever ───────────────────────────────────────────────────────────

class KeywordRetriever(BaseRetriever):
    """Zero-dependency TF-IDF cosine-similarity retriever."""

    def __init__(self, max_documents: Optional[int] = None) -> None:
        if max_documents is not None and (not isinstance(max_documents, int) or max_documents < 1):
            raise ValueError("max_documents must be an integer >= 1 or None")
        self.max_documents = max_documents
        self.documents: List[Document] = []
        self._doc_tfs: List[Dict[str, float]] = []
        self._df: Dict[str, int] = {}
        # Cache of IDF-weighted document vectors + norms, keyed to the
        # corpus state they were computed against. Invalidated (rebuilt)
        # whenever the corpus changes (add_document/add_documents/clear),
        # so repeated retrieve() calls between corpus changes are fast.
        self._doc_vecs_cache: List[Dict[str, float]] = []
        self._doc_norms_cache: List[float] = []
        self._cache_dirty: bool = True

    def _idf(self, term: str, N: int) -> float:
        return math.log((1 + N) / (1 + self._df.get(term, 0))) + 1.0

    def _tfidf_vector(self, tf: Dict[str, float], N: int) -> Dict[str, float]:
        return {t: w * self._idf(t, N) for t, w in tf.items()}

    @staticmethod
    def _norm(vec: Dict[str, float]) -> float:
        s = sum(v * v for v in vec.values())
        return math.sqrt(s) if s else 0.0

    @staticmethod
    def _cosine(q_vec: Dict[str, float], q_norm: float, d_vec: Dict[str, float], d_norm: float) -> float:
        if not q_norm or not d_norm:
            return 0.0
        dot = sum(q_vec[t] * d_vec[t] for t in q_vec if t in d_vec)
        return dot / (q_norm * d_norm)

    def add_document(self, doc: Document) -> None:
        tokens = tokenize(doc.content)
        raw: Dict[str, int] = {}
        for t in tokens:
            raw[t] = raw.get(t, 0) + 1
        n = len(tokens) or 1
        tf = {t: c / n for t, c in raw.items()}
        for t in tf:
            self._df[t] = self._df.get(t, 0) + 1
        self.documents.append(doc)
        self._doc_tfs.append(tf)
        self._cache_dirty = True
        self._enforce_max_documents()

    def _enforce_max_documents(self) -> None:
        """FIFO-evict the oldest document(s) once the corpus exceeds max_documents."""
        if self.max_documents is None:
            return
        while len(self.documents) > self.max_documents:
            self.documents.pop(0)
            evicted_tf = self._doc_tfs.pop(0)
            for t in evicted_tf:
                if t in self._df:
                    self._df[t] -= 1
                    if self._df[t] <= 0:
                        del self._df[t]
            self._cache_dirty = True

    def add_documents(self, docs: List[Document]) -> None:
        for doc in docs:
            self.add_document(doc)

    def clear(self) -> None:
        self.documents.clear()
        self._doc_tfs.clear()
        self._df.clear()
        self._doc_vecs_cache.clear()
        self._doc_norms_cache.clear()
        self._cache_dirty = True

    def _rebuild_doc_cache(self) -> None:
        """Recompute cached IDF-weighted document vectors/norms.

        Term frequencies (``self._doc_tfs``) never change once a document is
        added, but the IDF weighting depends on the whole corpus (N and
        document frequencies), which does change whenever a document is
        added. So this cache must be rebuilt whenever the corpus changes,
        not on every ``retrieve()`` call.
        """
        N = len(self.documents)
        self._doc_vecs_cache = [self._tfidf_vector(tf, N) for tf in self._doc_tfs]
        self._doc_norms_cache = [self._norm(vec) for vec in self._doc_vecs_cache]
        self._cache_dirty = False

    def retrieve(self, query: str, top_k: int = 5) -> List[Document]:
        if not self.documents or not query.strip():
            return []
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        N = len(self.documents)
        q_raw: Dict[str, int] = {}
        for t in query_tokens:
            q_raw[t] = q_raw.get(t, 0) + 1
        n_q = len(query_tokens)
        q_tf = {t: c / n_q for t, c in q_raw.items()}
        q_vec = self._tfidf_vector(q_tf, N)
        q_norm = self._norm(q_vec)
        if self._cache_dirty:
            self._rebuild_doc_cache()
        scored: List[Tuple[Document, float]] = []
        for doc, d_vec, d_norm in zip(self.documents, self._doc_vecs_cache, self._doc_norms_cache):
            score = self._cosine(q_vec, q_norm, d_vec, d_norm)
            if score > 0.0:
                scored.append((doc, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [Document(content=d.content, metadata=d.metadata, score=s, source=d.source) for d, s in scored[:top_k]]


# ── KeywordMemory ──────────────────────────────────────────────────────────────

class KeywordMemory(RetrievalAugmentedMemory):
    """Dual-store: sliding short-term buffer + TF-IDF keyword retrieval."""

    def __init__(
        self,
        short_term: Optional[BaseMemory] = None,
        retriever: Optional[BaseRetriever] = None,
        top_k: int = 3,
        max_documents: Optional[int] = None,
    ) -> None:
        super().__init__(
            short_term=short_term or RuntimeShortTermMemory(max_messages=10, name="keyword"),
            retriever=retriever or KeywordRetriever(max_documents=max_documents),
            top_k=top_k,
        )


# back-compat aliases
SimpleSemanticRetriever    = KeywordRetriever
HierarchicalSemanticMemory = KeywordMemory
