"""Smoke tests for SummaryBufferedMemory."""
import pytest

from autourgos_memory import SummaryBufferedMemory


class _NoneLLM:
    """Simulates a misbehaving/mocked LLM wrapper that returns None."""

    def invoke(self, prompt):
        return None


class _EmptyLLM:
    def invoke(self, prompt):
        return "   "


class _AlwaysFailsLLM:
    def invoke(self, prompt):
        raise RuntimeError("provider unavailable")


def test_add_and_get_messages_normal():
    mem = SummaryBufferedMemory(max_messages=10)
    mem.add_user_message("hello")
    mem.add_agent_message("hi there")
    msgs = mem.get_messages()
    assert [m.content for m in msgs] == ["hello", "hi there"]


def test_add_system_message():
    mem = SummaryBufferedMemory(max_messages=10)
    msg = mem.add_system_message("policy note")
    assert msg.role == "system"
    assert msg.content == "policy note"
    assert [m.role for m in mem.get_messages()] == ["system"]


def test_overflow_triggers_summary_without_llm():
    mem = SummaryBufferedMemory(max_messages=2)
    mem.add_user_message("a")
    mem.add_user_message("b")
    mem.add_user_message("c")
    msgs = mem.get_messages()
    assert [m.content for m in msgs] == ["b", "c"]
    assert "a" in mem.moving_summary


def test_llm_returning_none_falls_back_instead_of_baking_in_the_string_none():
    """str(None) == 'None' -- an LLM returning None used to permanently
    bake the literal string "None" into moving_summary instead of falling
    back to raw concatenation like any other unusable/failed response."""
    mem = SummaryBufferedMemory(max_messages=2, llm=_NoneLLM())
    mem.add_user_message("a")
    mem.add_user_message("b")
    mem.add_user_message("c")

    assert mem.moving_summary != "None"
    assert "a" in mem.moving_summary


def test_llm_returning_empty_string_falls_back_instead_of_wiping_summary():
    mem = SummaryBufferedMemory(max_messages=2, llm=_EmptyLLM())
    mem.add_user_message("a")
    mem.add_user_message("b")
    mem.add_user_message("c")

    assert mem.moving_summary.strip() != ""
    assert "a" in mem.moving_summary


def test_clear_resets_messages_and_summary():
    mem = SummaryBufferedMemory(max_messages=1)
    mem.add_user_message("a")
    mem.add_user_message("b")
    mem.clear()
    assert mem.get_messages() == []
    assert mem.moving_summary == ""


def test_format_for_llm_includes_summary_and_recent():
    mem = SummaryBufferedMemory(max_messages=1)
    mem.add_user_message("a")
    mem.add_user_message("b")
    text = mem.format_for_llm()
    assert "Summary of Past Conversation" in text
    assert "Recent Conversation Context" in text


def test_invalid_max_summary_chars_raises():
    with pytest.raises(ValueError):
        SummaryBufferedMemory(max_messages=1, max_summary_chars=0)


def test_max_summary_chars_bounds_no_llm_fallback_growth():
    """
    Regression for Finding #7: without an LLM, every overflow appended full
    raw message text to moving_summary with no limit -- it grew without
    bound across a long-running conversation. max_summary_chars must keep
    it capped, keeping the most recent overflow content (trimmed from the
    front) rather than growing forever.
    """
    mem = SummaryBufferedMemory(max_messages=1, max_summary_chars=200)
    for i in range(50):
        mem.add_user_message(f"message number {i} with some extra padding text")
    assert len(mem.moving_summary) <= 200
    # Most recent overflow content must survive the trim, not be the first
    # thing discarded.
    assert "message number 48" in mem.moving_summary


def test_max_summary_chars_bounds_llm_failure_fallback_growth():
    """
    Regression for Finding #7's "if compression fails, do not append
    unlimited raw history" guidance: repeated compression failures must not
    grow moving_summary without bound either.
    """
    mem = SummaryBufferedMemory(max_messages=1, llm=_AlwaysFailsLLM(), max_summary_chars=200)
    for i in range(50):
        mem.add_user_message(f"message number {i} with some extra padding text")
    assert len(mem.moving_summary) <= 200
    assert "message number 48" in mem.moving_summary


def test_max_summary_chars_none_preserves_unbounded_default():
    mem = SummaryBufferedMemory(max_messages=1)
    for i in range(50):
        mem.add_user_message(f"message number {i} with some extra padding text")
    assert len(mem.moving_summary) > 200
