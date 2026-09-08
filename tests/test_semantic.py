"""Tests for KeywordRetriever (TF-IDF cosine-similarity retrieval)."""
from autourgos_memory.semantic import KeywordRetriever
from autourgos_memory.base import Document


def _docs():
    return [
        Document(content="the cat sat on the mat"),
        Document(content="dogs are loyal animals"),
        Document(content="the cat chased the dog"),
        Document(content="cats and dogs can be friends"),
    ]


def test_add_document_and_retrieve_normal():
    r = KeywordRetriever()
    r.add_documents(_docs())
    results = r.retrieve("cat dog", top_k=4)
    assert len(results) > 0
    # Best-matching document should mention both cat and dog.
    assert "cat" in results[0].content and "dog" in results[0].content


def test_retrieve_top_k_limit():
    r = KeywordRetriever()
    r.add_documents(_docs())
    results = r.retrieve("cat dog", top_k=1)
    assert len(results) == 1


def test_retrieve_results_ranked_descending_by_score():
    r = KeywordRetriever()
    r.add_documents(_docs())
    results = r.retrieve("cat dog", top_k=4)
    scores = [d.score for d in results]
    assert scores == sorted(scores, reverse=True)


def test_repeated_retrieve_calls_are_stable_and_use_cache():
    r = KeywordRetriever()
    r.add_documents(_docs())
    first = [(d.content, d.score) for d in r.retrieve("cat dog", top_k=4)]
    second = [(d.content, d.score) for d in r.retrieve("cat dog", top_k=4)]
    assert first == second


def test_cache_invalidated_after_new_document_added():
    r = KeywordRetriever()
    r.add_documents(_docs())
    before = [(d.content, d.score) for d in r.retrieve("cat", top_k=10)]
    r.add_document(Document(content="another cat story here"))
    after = [(d.content, d.score) for d in r.retrieve("cat", top_k=10)]
    # New document should now appear, and scores should reflect the
    # updated corpus (IDF changes with corpus size).
    assert any("another cat story" in c for c, _ in after)
    assert before != after

def test_max_documents_none_is_unbounded_default():
    r = KeywordRetriever()
    r.add_documents(_docs())
    assert len(r.documents) == len(_docs())


def test_max_documents_evicts_oldest_fifo():
    r = KeywordRetriever(max_documents=2)
    docs = _docs()
    for d in docs:
        r.add_document(d)
    assert len(r.documents) == 2
    # Oldest two should have been evicted; only the last two remain, in order.
    assert [d.content for d in r.documents] == [docs[2].content, docs[3].content]


def test_max_documents_retrieve_still_works_after_eviction():
    r = KeywordRetriever(max_documents=2)
    for d in _docs():
        r.add_document(d)
    results = r.retrieve("cat dog", top_k=2)
    assert len(results) > 0
