"""Tests for JCODEMUNCH_PATH_MAP env var parsing and path remapping."""

import logging
import pytest

from jcodemunch_mcp.path_map import parse_path_map, ENV_VAR


def test_parse_unset(monkeypatch):
    monkeypatch.delenv(ENV_VAR, raising=False)
    assert parse_path_map() == []


def test_parse_whitespace_only(monkeypatch):
    monkeypatch.setenv(ENV_VAR, "   ")
    assert parse_path_map() == []


def test_parse_single_pair(monkeypatch):
    monkeypatch.setenv(ENV_VAR, "/home/user=/mnt/user")
    assert parse_path_map() == [("/home/user", "/mnt/user")]


def test_parse_multiple_pairs(monkeypatch):
    monkeypatch.setenv(ENV_VAR, "/a=/b,/c=/d")
    assert parse_path_map() == [("/a", "/b"), ("/c", "/d")]


def test_parse_equals_in_path(monkeypatch):
    """Last = is the separator; earlier = chars belong to the original path."""
    monkeypatch.setenv(ENV_VAR, "/home/user/a=b=/new/path")
    assert parse_path_map() == [("/home/user/a=b", "/new/path")]


def test_parse_malformed_no_equals_skipped(monkeypatch, caplog):
    monkeypatch.setenv(ENV_VAR, "/valid=/ok,noequalssign,/also=/fine")
    with caplog.at_level(logging.WARNING):
        result = parse_path_map()
    assert result == [("/valid", "/ok"), ("/also", "/fine")]
    assert any("noequalssign" in r.message for r in caplog.records)


def test_parse_empty_orig_skipped(monkeypatch, caplog):
    monkeypatch.setenv(ENV_VAR, "=/new/path")
    with caplog.at_level(logging.WARNING):
        result = parse_path_map()
    assert result == []
    assert len(caplog.records) >= 1


def test_parse_empty_new_skipped(monkeypatch, caplog):
    monkeypatch.setenv(ENV_VAR, "/old/path=")
    with caplog.at_level(logging.WARNING):
        result = parse_path_map()
    assert result == []
    assert len(caplog.records) >= 1


def test_parse_whitespace_stripped(monkeypatch):
    """Leading/trailing whitespace in tokens is stripped."""
    monkeypatch.setenv(ENV_VAR, " /home/user = /mnt/user ")
    assert parse_path_map() == [("/home/user", "/mnt/user")]
