"""Smoke tests for LocalShortTermMemory and SQLiteMemory."""
import os

import pytest

from autourgos_memory import LocalShortTermMemory, SQLiteMemory


def test_local_short_term_memory_add_and_get(tmp_path):
    file_path = os.path.join(tmp_path, "mem.json")
    mem = LocalShortTermMemory(file_path=file_path, max_messages=10)
    mem.add_user_message("hello")
    mem.add_agent_message("hi there")
    msgs = mem.get_messages()
    assert [m.content for m in msgs] == ["hello", "hi there"]


def test_local_short_term_memory_rolling_cap(tmp_path):
    file_path = os.path.join(tmp_path, "mem.json")
    mem = LocalShortTermMemory(file_path=file_path, max_messages=2)
    mem.add_user_message("a")
    mem.add_user_message("b")
    mem.add_user_message("c")
    msgs = mem.get_messages()
    assert [m.content for m in msgs] == ["b", "c"]


def test_local_short_term_memory_clear(tmp_path):
    file_path = os.path.join(tmp_path, "mem.json")
    mem = LocalShortTermMemory(file_path=file_path)
    mem.add_user_message("x")
    mem.clear()
    assert mem.get_messages() == []


def test_local_short_term_memory_get_messages_on_corrupted_file(tmp_path):
    """get_messages()/format_for_llm() (via _load()) used to raise
    json.JSONDecodeError uncaught on a corrupted file, while add_message()
    silently recovered by resetting to an empty list -- inconsistent
    corruption handling for two halves of the same file. Reads must recover
    the same way writes do."""
    file_path = os.path.join(tmp_path, "mem.json")
    mem = LocalShortTermMemory(file_path=file_path, max_messages=10)
    mem.add_user_message("hello")

    with open(file_path, "w", encoding="utf-8") as fh:
        fh.write("{not valid json,,,")

    assert mem.get_messages() == []
    assert mem.format_for_llm() == ""


def test_local_short_term_memory_add_message_recovers_from_corrupted_file(tmp_path):
    file_path = os.path.join(tmp_path, "mem.json")
    mem = LocalShortTermMemory(file_path=file_path, max_messages=10)
    mem.add_user_message("hello")

    with open(file_path, "w", encoding="utf-8") as fh:
        fh.write("{not valid json,,,")

    mem.add_user_message("recovered")
    assert [m.content for m in mem.get_messages()] == ["recovered"]


def test_sqlite_memory_add_get_clear():
    mem = SQLiteMemory(db_path=":memory:", max_messages=None)
    mem.add_user_message("hello")
    mem.add_agent_message("hi there")
    msgs = mem.get_messages()
    assert [m.content for m in msgs] == ["hello", "hi there"]
    mem.clear()
    assert mem.get_messages() == []
    mem.close()


def test_sqlite_memory_add_system_message():
    mem = SQLiteMemory(db_path=":memory:", max_messages=None)
    msg = mem.add_system_message("policy note")
    assert msg.role == "system"
    assert msg.content == "policy note"
    assert [m.role for m in mem.get_messages()] == ["system"]
    mem.close()


def test_sqlite_memory_supports_context_manager_and_closes_connection():
    import sqlite3
    with SQLiteMemory(db_path=":memory:", max_messages=None) as mem:
        mem.add_user_message("hello")
        assert [m.content for m in mem.get_messages()] == ["hello"]
    with pytest.raises(sqlite3.ProgrammingError):
        mem.get_messages()


def test_sqlite_memory_rolling_cap_keeps_newest():
    """Regression: eviction query migrated to autourgos_core.row_cap_evict --
    must still keep exactly the newest max_messages rows."""
    mem = SQLiteMemory(db_path=":memory:", max_messages=3)
    for i in range(6):
        mem.add_user_message(f"msg{i}")
    msgs = mem.get_messages()
    assert [m.content for m in msgs] == ["msg3", "msg4", "msg5"]
    mem.close()


def test_sqlite_memory_db_path_with_missing_parent_dir_is_created(tmp_path):
    db_path = str(tmp_path / "nested" / "does" / "not" / "exist" / "mem.db")
    mem = SQLiteMemory(db_path=db_path)
    mem.add_user_message("hello")
    mem.close()
    assert os.path.exists(db_path)
