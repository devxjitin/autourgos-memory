"""Tests for EpisodicMemory."""
import os

import pytest

from autourgos_memory import Episode, EpisodicMemory, EpisodicMemoryError


def test_remember_returns_episode_with_fields():
    memory = EpisodicMemory()
    ep = memory.remember(
        task="Deploy the app",
        outcome="fail",
        actions_taken=["ran deploy.sh"],
        notes="Permission error.",
    )
    assert isinstance(ep, Episode)
    assert ep.task == "Deploy the app"
    assert ep.outcome == "fail"
    assert ep.actions_taken == ["ran deploy.sh"]
    assert ep.notes == "Permission error."


def test_invalid_outcome_rejected():
    memory = EpisodicMemory()
    with pytest.raises(EpisodicMemoryError):
        memory.remember(task="do something", outcome="maybe")


def test_empty_task_rejected():
    memory = EpisodicMemory()
    with pytest.raises(EpisodicMemoryError):
        memory.remember(task="   ", outcome="success")


def test_retrieve_ranks_by_task_relevance():
    memory = EpisodicMemory()
    memory.remember(task="Deploy the app to production", outcome="fail",
                     actions_taken=["hit S3 permission error"])
    memory.remember(task="Fix the failing test suite", outcome="success",
                     actions_taken=["regenerated stale fixture"])

    results = memory.retrieve("deploying to production", top_k=1)
    assert len(results) == 1
    assert results[0].metadata["task"] == "Deploy the app to production"
    assert results[0].metadata["outcome"] == "fail"


def test_retrieve_empty_store_returns_empty_list():
    memory = EpisodicMemory()
    assert memory.retrieve("anything", top_k=5) == []


def test_max_episodes_evicts_oldest():
    memory = EpisodicMemory(max_episodes=2)
    memory.remember(task="first task", outcome="success")
    memory.remember(task="second task", outcome="success")
    memory.remember(task="third task", outcome="success")

    results = memory.retrieve("task", top_k=10)
    tasks = {d.metadata["task"] for d in results}
    assert "first task" not in tasks
    assert tasks == {"second task", "third task"}


def test_persistence_across_reopen(tmp_path):
    db_path = str(tmp_path / "episodes.db")
    memory = EpisodicMemory(db_path=db_path)
    memory.remember(task="Deploy the app to production", outcome="fail",
                     actions_taken=["hit S3 permission error"])
    memory.close()

    memory2 = EpisodicMemory(db_path=db_path)
    results = memory2.retrieve("deploying to production", top_k=1)
    assert len(results) == 1
    assert results[0].metadata["task"] == "Deploy the app to production"
    assert results[0].metadata["outcome"] == "fail"


def test_clear_empties_store():
    memory = EpisodicMemory()
    memory.remember(task="Deploy the app", outcome="success")
    memory.clear()
    assert memory.retrieve("Deploy the app", top_k=5) == []


def test_max_episodes_rejects_invalid_values():
    with pytest.raises(ValueError):
        EpisodicMemory(max_episodes=0)
    with pytest.raises(ValueError):
        EpisodicMemory(max_episodes=-1)


def test_supports_context_manager_and_closes_connection():
    import sqlite3
    with EpisodicMemory() as memory:
        memory.remember(task="Deploy the app", outcome="success")
        assert memory.retrieve("Deploy the app", top_k=5)
    with pytest.raises(sqlite3.ProgrammingError):
        memory.remember(task="after close", outcome="success")


def test_db_path_with_missing_parent_dir_is_created_automatically(tmp_path):
    """Regression: a fresh db_path whose directory doesn't exist yet used
    to raise sqlite3.OperationalError -- open_sqlite() now creates it."""
    db_path = str(tmp_path / "nested" / "does" / "not" / "exist" / "episodes.db")
    memory = EpisodicMemory(db_path=db_path)
    memory.remember(task="test", outcome="success")
    memory.close()
    assert os.path.exists(db_path)
