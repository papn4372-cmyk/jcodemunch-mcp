"""Tests for JSON→SQLite migration."""

import json
from pathlib import Path

from jcodemunch_mcp.storage.index_store import IndexStore
from jcodemunch_mcp.storage.sqlite_store import SQLiteIndexStore
from jcodemunch_mcp.parser.symbols import Symbol


def _make_symbol(name: str) -> Symbol:
    return Symbol(
        id=f"main.py::{name}#function", file="main.py", name=name,
        qualified_name=name, kind="function", language="python",
        signature=f"def {name}()", line=1, end_line=3,
        byte_offset=0, byte_length=20,
    )


def test_migrate_json_to_sqlite(tmp_path):
    """Auto-migration from JSON produces identical CodeIndex."""
    # Create a JSON index using the original IndexStore
    json_store = IndexStore(base_path=str(tmp_path))
    sym = _make_symbol("greet")
    json_store.save_index(
        owner="local", name="test-abc123",
        source_files=["main.py"], symbols=[sym],
        raw_files={"main.py": "def greet(): pass"},
        file_hashes={"main.py": "hash1"},
        git_head="abc",
        file_summaries={"main.py": "Greeting module"},
        source_root="/tmp/proj",
        display_name="test",
        file_mtimes={"main.py": 1234567890000000000},
    )

    # Load via JSON to get expected values
    json_index = json_store.load_index("local", "test-abc123")
    assert json_index is not None

    # Now migrate
    sqlite_store = SQLiteIndexStore(base_path=str(tmp_path))
    json_path = tmp_path / "local-test-abc123.json"
    migrated = sqlite_store.migrate_from_json(json_path, "local", "test-abc123")

    assert migrated is not None
    assert migrated.repo == json_index.repo
    assert len(migrated.symbols) == len(json_index.symbols)
    assert migrated.file_hashes == json_index.file_hashes
    assert migrated.git_head == json_index.git_head
    assert migrated.display_name == json_index.display_name

    # Verify .json.migrated exists
    assert (tmp_path / "local-test-abc123.json.migrated").exists()
    assert not (tmp_path / "local-test-abc123.json").exists()
