---
phase: 1
plan: 2026-03-23-config-jsonc
subsystem: config
tags: [python, tdd, jsonc, mcp, config, cli]
dependency-graph:
  requires: []
  provides: [centralized-jsonc-config]
  affects: [server, security, cli]
tech-stack:
  added: []
  patterns: [tdd, config-layering, env-var-deprecation]
---

# Phase 1 Plan: Centralized JSONC Config Implementation Summary

## One-Liner

Implemented two-layer JSONC config system (`~/.code-index/config.jsonc` + per-project `.jcodemunch.jsonc`) with TDD, replacing ~23 deprecated `JCODEMUNCH_*` env vars with a centralized config module.

## Overview

Replaced scattered `JCODEMUNCH_*` env var reads across server.py, security.py with a single `config.py` module supporting:
- **JSONC parsing** with `//` and `/* */` comment stripping
- **Two-layer config**: global `config.jsonc` + per-project `.jcodemunch.jsonc`
- **Env var fallback**: deprecated env vars still work with one-time warnings (removed in v2.0)
- **Token-control levers**: `disabled_tools`, `languages`, `meta_fields`, `descriptions`
- **SQL gating**: `search_columns` auto-disabled when SQL not in languages

## Tasks Completed

| Task | Name | Commit |
|------|------|--------|
| 1.1 | JSONC line comment stripping | b597e00 |
| 1.2 | JSONC block comment stripping | 0f2003c |
| 1.3 | Config defaults and type definitions | 86d1cb1 |
| 1.4 | Config loading - missing file | 08910cc |
| 1.5 | Config loading - valid JSONC | 92d6f9e |
| 1.6 | Config type validation | 9675d23 |
| 1.7 | Project config loading with merge | 6d5f216 |
| 1.8 | Tool and language checking functions | 5fbc5de |
| 1.9 | Template generation | f1cc354 |
| 1.10 | get_descriptions() | e2571ed |
| 1.11 | Env var fallback with warnings | d8d7f01 |
| 2.1 | Dynamic tool filtering | e31ebea |
| 2.1b | Dynamic language enum | c502bc0 |
| 2.2 | Remove suppress_meta, meta_fields config | 547bc3b |
| 2.3/2.5 | Descriptions merge with _shared | 49ac94a |
| 3.1 | Security module config integration | 2530e48 |
| 4.1 | config --init CLI command | 8c5a7ca |
| 6.1 | SQL gating auto-disables search_columns | 8788509 |
| fix | test_tools.py env var tests | 5733602 |

## Key Files Created/Modified

### Created
- `src/jcodemunch_mcp/config.py` — Centralized config module (~370 lines)
  - `_strip_jsonc()` — JSONC comment stripper
  - `DEFAULTS`, `CONFIG_TYPES`, `ENV_VAR_MAPPING` constants
  - `load_config()`, `load_project_config()`, `get()`, `is_tool_disabled()`, `is_language_enabled()`, `get_descriptions()`, `generate_template()`
  - `_validate_type()`, `_parse_env_value()`, `_apply_env_var_fallback()`
- `tests/test_config.py` — 23 tests for config module

### Modified
- `src/jcodemunch_mcp/server.py` — 12 key changes:
  - Import config module
  - Add `_build_language_enum()` helper
  - Add `_apply_description_overrides()` helper
  - Dynamic tool filtering (disabled_tools)
  - Dynamic language enum (languages)
  - Remove suppress_meta param, replace with meta_fields config
  - SQL gating (auto-disable search_columns when SQL absent)
  - Config init CLI support
- `src/jcodemunch_mcp/security.py` — 3 env var reads replaced with config.get():
  - `get_max_index_files()`
  - `get_max_folder_files()`
  - `get_extra_ignore_patterns()`
- `tests/test_security.py` — Updated 10 env-var tests to config-based
- `tests/test_tools.py` — Updated 4 env-var tests to config-based
- `tests/test_cli.py` — Added 2 tests for `config --init`
- `tests/test_server.py` — Added 11 new tests for server integration

## Decisions Made

1. **JSONC template uses active values** — Template generates with ALL values active (no commented-out lines) to avoid trailing comma complexity in the JSONC parser. Users can edit to comment out unwanted values.

2. **JSONC parser handles trailing commas** — Block comment `/* comment */` strips preceding `,` to avoid JSON parse errors from trailing commas left by commented-out lines.

3. **Descriptions merge: tool-specific > _shared > original** — Tool-specific param descriptions override `_shared`, which overrides original. `_tool` replaces entire tool description.

4. **SQL gating runs after disabled_tools filter** — SQL gating happens AFTER user-specified `disabled_tools`, so explicit `search_columns` in `disabled_tools` still works when SQL is in languages.

5. **Env var fallback uses one-time warning** — `_DEPRECATED_ENV_VARS_LOGGED` set tracks which vars have been warned to avoid spam.

6. **Config loaded once at startup** — Global `config.jsonc` loaded once, per-project `.jcodemunch.jsonc` cached by repo path.

7. **Per-call max_files param still works** — `get_max_index_files(max_files=5)` and `get_max_folder_files(max_files=5)` take precedence over config for explicit overrides.

## Deviations from Plan

### Auto-fixed Issues

**[Rule 1 - Bug] JSONC parser missing trailing comma handling**
- Found during: Task 1.9 (Template Generation)
- Issue: When block comment `/* comment */` appears on its own line between JSON key-value pairs, the preceding `,` is left behind and breaks JSON parsing.
- Fix: Enhanced `_strip_jsonc()` to detect `,` immediately before `/*` and strip both.
- Files modified: `src/jcodemunch_mcp/config.py`
- Commit: f1cc354

**[Rule 2 - Missing] Descriptions merge only runs when tool-specific entry exists**
- Found during: Phase 2.3 (descriptions merge)
- Issue: `_apply_description_overrides` skipped tools without explicit entry in descriptions config, preventing `_shared` from applying.
- Fix: Changed `tool_desc = descriptions.get(tool.name)` to `descriptions.get(tool.name, {})` and removed the `if not tool_desc: continue` guard.
- Files modified: `src/jcodemunch_mcp/server.py`
- Commit: a13f98d

**[Rule 3 - Blocking] Type safety in get_max_index_files/get_max_folder_files**
- Found during: Phase 3 (security integration)
- Issue: `max(value, 0)` crashes when `config.get()` returns a string (invalid type).
- Fix: Added `isinstance(value, int) and value > 0` guard before returning.
- Files modified: `src/jcodemunch_mcp/security.py`
- Commit: 2530e48

## Test Results

| Test Suite | Passed | Failed | Skipped |
|-----------|--------|--------|---------|
| test_config.py | 23 | 0 | 0 |
| test_server.py | 38 | 0 | 0 |
| test_security.py | 105 | 0 | 4 |
| test_cli.py | 4 | 0 | 0 |
| test_tools.py | ~50 | 0 | 0 |
| **Total** | **997** | **0** | **4** |

> Note: 1 pre-existing failure in `test_storage.py::test_index_integrity_checksum` (unrelated to this work, fails on baseline).

## Commits (21 total on feat/centralized-config-jsonc)

```
8788509 test+feat: Phase 6 - SQL gating auto-disables search_columns
5733602 fix: Update test_tools.py env var tests to use config
8c5a7ca test+feat: Phase 4 - config --init CLI command
2530e48 test+feat: Phase 3 - Security module config integration
49ac94a test+feat: Complete Phase 2 - Server Integration
a13f98d test+feat: Descriptions config merge in list_tools
547bc3b test+feat: Remove suppress_meta param, add meta_fields config
c502bc0 test+feat: Dynamic language enum in search_symbols schema
e31ebea test+feat: Dynamic tool filtering in list_tools
d8d7f01 test+feat: Env var fallback with deprecation warnings
e2571ed test+feat: get_descriptions() function
f1cc354 test+feat: Config template generation, improved JSONC parser
5fbc5de test+feat: Tool and language checking functions
6d5f216 test+feat: Project config loading with merge
9675d23 test+feat: Config type validation with warnings
92d6f9e test+feat: Config loading with valid JSONC
08910cc test+feat: Config loading with missing file handling
86d1cb1 test+feat: Config defaults and type definitions
0f2003c test+feat: JSONC block comment stripping
b597e00 test+feat: JSONC line comment stripping with TDD
```

## Duration

Started: 2026-03-23 (session)
Completed: 2026-03-23
