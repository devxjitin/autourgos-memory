from datetime import datetime, timezone

import pytest

from autourgos_memory import (
    BaseMemory,
    BaseRetriever,
    Document,
    MemoryMessage,
    RetrievalAugmentedMemory,
    ROLE_TO_OPENAI,
    format_conversation_banner,
)


class DummyMemory(BaseMemory):
    """Minimal concrete BaseMemory implementing only add_message/clear."""

    def __init__(self) -> None:
        self.messages = []

    def add_message(self, role, content, timestamp=None):
        msg = MemoryMessage(role=role, content=content, timestamp=timestamp or datetime.now(timezone.utc))
        self.messages.append(msg)
        return msg

    def get_messages(self):
        return list(self.messages)

    def format_for_llm(self, query=None):
        return format_conversation_banner(self.messages)

    def clear(self):
        self.messages = []


def test_add_user_message_defaults_to_add_message():
    mem = DummyMemory()
    msg = mem.add_user_message("hello")
    assert msg.role == "user"
    assert msg.content == "hello"
    assert mem.messages == [msg]


def test_add_agent_message_defaults_to_add_message():
    mem = DummyMemory()
    msg = mem.add_agent_message("hi there")
    assert msg.role == "agent"


def test_add_system_message_defaults_to_add_message():
    mem = DummyMemory()
    msg = mem.add_system_message("system note")
    assert msg.role == "system"
    assert msg.content == "system note"


def test_add_tool_message_formats_result():
    mem = DummyMemory()
    msg = mem.add_tool_message("calculator", "42")
    assert msg.role == "tool"
    assert msg.content == "[calculator returned]: 42"


def test_add_agent_message_legacy_ai_dispatch_preserved():
    class LegacyMemory(DummyMemory):
        def add_ai_message(self, content):
            msg = MemoryMessage(role="agent", content=f"legacy:{content}", timestamp=datetime.now(timezone.utc))
            self.messages.append(msg)
            return msg

    mem = LegacyMemory()
    msg = mem.add_agent_message("y")
    assert msg.content == "legacy:y"


def test_add_ai_message_warns_on_default_memory():
    mem = DummyMemory()
    with pytest.warns(DeprecationWarning):
        msg = mem.add_ai_message("z")
    assert msg.role == "agent"
    assert msg.content == "z"


def test_cannot_instantiate_without_add_message():
    class Incomplete(BaseMemory):
        def clear(self):
            pass

    with pytest.raises(TypeError):
        Incomplete()


def test_format_conversation_banner_empty():
    assert format_conversation_banner([]) == ""


def test_format_conversation_banner_with_timestamps():
    ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
    msgs = [MemoryMessage(role="user", content="hi", timestamp=ts)]
    result = format_conversation_banner(msgs, include_timestamps=True)
    assert f"[{ts.isoformat()}] user: hi" in result
    assert result.startswith("\n--- Previous Conversation Context ---\n")
    assert result.endswith("--------------------------------------\n")


def test_format_conversation_banner_without_timestamps():
    ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
    msgs = [MemoryMessage(role="user", content="hi", timestamp=ts)]
    result = format_conversation_banner(msgs, include_timestamps=False)
    assert "user: hi" in result
    assert "2026-01-01" not in result


def test_role_to_openai_mapping():
    assert ROLE_TO_OPENAI == {"user": "user", "agent": "assistant", "system": "system", "tool": "tool"}


class StubRetriever(BaseRetriever):
    def __init__(self):
        self.docs = []

    def add_document(self, doc):
        self.docs.append(doc)

    def retrieve(self, query, top_k=5):
        return [d for d in self.docs if query in d.content][:top_k]

    def clear(self):
        self.docs = []


class StubRAM(RetrievalAugmentedMemory):
    def __init__(self):
        super().__init__(short_term=DummyMemory(), retriever=StubRetriever(), top_k=3)


def test_retrieval_augmented_memory_indexes_on_add():
    mem = StubRAM()
    mem.add_user_message("the sky is blue")
    assert len(mem.retriever.docs) == 1
    assert mem.retriever.docs[0].content == "the sky is blue"
    assert mem.retriever.docs[0].metadata["role"] == "user"


def test_retrieval_augmented_memory_format_for_llm_no_query():
    mem = StubRAM()
    mem.add_user_message("hello")
    result = mem.format_for_llm()
    assert "hello" in result
    assert "Relevant Past Context" not in result


def test_retrieval_augmented_memory_format_for_llm_with_query_surfaces_relevant():
    mem = StubRAM()
    mem.retriever.add_document(Document(content="banana smoothie recipe", metadata={"role": "user"}))
    mem.add_user_message("what's new")
    result = mem.format_for_llm(query="banana")
    assert "Relevant Past Context" in result
    assert "banana smoothie recipe" in result


def test_retrieval_augmented_memory_clear():
    mem = StubRAM()
    mem.add_user_message("hi")
    mem.clear()
    assert mem.short_term.get_messages() == []
    assert mem.retriever.docs == []


def test_retrieval_augmented_memory_add_tool_message():
    mem = StubRAM()
    msg = mem.add_tool_message("calc", "4")
    assert msg.content == "[calc returned]: 4"
    assert mem.retriever.docs[0].metadata["role"] == "tool"
