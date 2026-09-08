"""Smoke tests for RuntimeShortTermMemory, ConversationBufferMemory, ExpiringBufferMemory."""
import time

import pytest

from autourgos_memory import RuntimeShortTermMemory, ConversationBufferMemory, ExpiringBufferMemory


def test_add_and_get_messages_normal():
    mem = RuntimeShortTermMemory(max_messages=10)
    mem.add_user_message("hello")
    mem.add_agent_message("hi there")
    msgs = mem.get_messages()
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert [m["content"] for m in msgs] == ["hello", "hi there"]


def test_ring_buffer_evicts_oldest():
    mem = RuntimeShortTermMemory(max_messages=2)
    mem.add_user_message("a")
    mem.add_user_message("b")
    mem.add_user_message("c")
    msgs = mem.get_messages()
    assert [m["content"] for m in msgs] == ["b", "c"]


def test_clear_empties_buffer():
    mem = RuntimeShortTermMemory(max_messages=5)
    mem.add_user_message("x")
    mem.clear()
    assert mem.get_messages() == []
    assert mem.format_for_llm() == ""


def test_conversation_buffer_memory_is_unbounded():
    mem = ConversationBufferMemory()
    for i in range(50):
        mem.add_user_message(str(i))
    assert len(mem.get_messages()) == 50


# ── ExpiringBufferMemory ───────────────────────────────────────────────────

def test_expiring_buffer_no_ttl_never_expires():
    mem = ExpiringBufferMemory()
    mem.add_user_message("hello")
    assert [m["content"] for m in mem.get_messages()] == ["hello"]


def test_expiring_buffer_message_expires_after_ttl():
    mem = ExpiringBufferMemory()
    mem.add_user_message("temporary fact", ttl_seconds=0.05)
    assert len(mem.get_messages()) == 1
    time.sleep(0.1)
    assert mem.get_messages() == []
    assert mem.format_for_llm() == ""


def test_expiring_buffer_default_ttl_applies_to_all_messages():
    mem = ExpiringBufferMemory(default_ttl_seconds=0.05)
    mem.add_user_message("a")
    mem.add_agent_message("b")
    time.sleep(0.1)
    assert mem.get_messages() == []


def test_expiring_buffer_per_message_ttl_overrides_default():
    mem = ExpiringBufferMemory(default_ttl_seconds=100)
    mem.add_user_message("short-lived", ttl_seconds=0.05)
    mem.add_user_message("long-lived")
    time.sleep(0.1)
    contents = [m["content"] for m in mem.get_messages()]
    assert contents == ["long-lived"]


def test_expiring_buffer_max_messages_evicts_oldest_live_messages():
    mem = ExpiringBufferMemory(max_messages=2)
    mem.add_user_message("a")
    mem.add_user_message("b")
    mem.add_user_message("c")
    contents = [m["content"] for m in mem.get_messages()]
    assert contents == ["b", "c"]


def test_expiring_buffer_clear_empties_buffer():
    mem = ExpiringBufferMemory()
    mem.add_user_message("x")
    mem.clear()
    assert mem.get_messages() == []


def test_expiring_buffer_rejects_invalid_ttl_and_max_messages():
    with pytest.raises(ValueError):
        ExpiringBufferMemory(default_ttl_seconds=0)
    with pytest.raises(ValueError):
        ExpiringBufferMemory(default_ttl_seconds=-1)
    with pytest.raises(ValueError):
        ExpiringBufferMemory(max_messages=0)
