"""Tests for reindex_state module."""

import pytest

from jcodemunch_mcp.reindex_state import (
    _get_state, _freshness_mode, _repo_states,
    mark_reindex_start, mark_reindex_done, mark_reindex_failed,
    get_reindex_status, is_any_reindex_in_progress,
    set_freshness_mode, get_freshness_mode, await_freshness_if_strict,
    wait_for_fresh_result,
)


@pytest.fixture(autouse=True)
def reset_state():
    """Reset module-level state before each test."""
    _repo_states.clear()
    _freshness_mode.clear()
    yield
    _repo_states.clear()
    _freshness_mode.clear()


class TestRepoStateCreation:
    def test_get_state_creates_new_state(self):
        state = _get_state("test/repo")
        assert state is not None

    def test_get_state_returns_same_instance(self):
        state1 = _get_state("test/repo")
        state2 = _get_state("test/repo")
        assert state1 is state2

    def test_get_state_different_repos_are_independent(self):
        state1 = _get_state("repo/a")
        state2 = _get_state("repo/b")
        assert state1 is not state2


class TestMarkReindexStart:
    def test_mark_reindex_start(self):
        mark_reindex_start("test/repo")
        status = get_reindex_status("test/repo")
        assert status["reindexing"] is True
        assert status["reindex_finished"] is False
        assert status["reindex_error"] is None
        assert status["last_reindex_start"] > 0


class TestMarkReindexDone:
    def test_mark_reindex_done(self):
        mark_reindex_start("test/repo")
        mark_reindex_done("test/repo", {"symbol_count": 42})
        status = get_reindex_status("test/repo")
        assert status["reindexing"] is False
        assert status["reindex_finished"] is True
        assert status["reindex_error"] is None
        assert status["last_reindex_done"] > 0

    def test_mark_reindex_done_clears_error(self):
        mark_reindex_start("test/repo")
        mark_reindex_failed("test/repo", "some error")
        mark_reindex_done("test/repo")
        status = get_reindex_status("test/repo")
        assert status["reindex_error"] is None


class TestMarkReindexFailed:
    def test_mark_reindex_failed(self):
        mark_reindex_start("test/repo")
        mark_reindex_failed("test/repo", "parse error")
        status = get_reindex_status("test/repo")
        assert status["reindexing"] is False
        assert status["reindex_finished"] is True
        assert status["reindex_error"] == "parse error"
