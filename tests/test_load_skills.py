"""Contract tests for ``_load_skills`` — the skill-source router/dispatcher.

This is the routing function in the CLI: it selects between the built-in pool and a
user-supplied JSON file. Contract (from cli.py):
  - path is None        -> return SAMPLE_SKILLS verbatim (same object, not a copy)
  - path is a JSON list -> one Skill per element
  - each element must have a "name" key (KeyError otherwise)
  - "description" and "body" are optional, defaulting to ""
  - the file must be valid JSON (JSONDecodeError on garbage)
  - the top-level must be a list (iterating a dict raises TypeError)
"""

import json
from pathlib import Path

import pytest

from skillrouter_test.cli import _load_skills
from skillrouter_test.models import Skill
from skillrouter_test.sample_data import SAMPLE_SKILLS


@pytest.fixture
def skills_file(tmp_path: Path) -> Path:
    def _write(payload: object) -> Path:
        p = tmp_path / "skills.json"
        p.write_text(json.dumps(payload))
        return p

    return _write  # type: ignore[return-value]


# --- None path: default/fallback branch -----------------------------------

def test_none_returns_sample_skills_object_identity() -> None:
    # The fallback must hand back the *same* list object, not a defensive copy —
    # a copy would silently break any caller mutating the shared pool.
    assert _load_skills(None) is SAMPLE_SKILLS


def test_none_returns_full_built_in_pool() -> None:
    assert len(_load_skills(None)) == len(SAMPLE_SKILLS)
    assert [s.name for s in _load_skills(None)] == [s.name for s in SAMPLE_SKILLS]


def test_none_is_deterministic_across_calls() -> None:
    # Idempotency: calling twice yields identical results.
    assert _load_skills(None) == _load_skills(None)


# --- JSON list path --------------------------------------------------------

def test_json_list_parses_one_skill_per_element(skills_file) -> None:
    path = skills_file([{"name": "alpha"}, {"name": "beta"}])
    result = _load_skills(path)
    assert [s.name for s in result] == ["alpha", "beta"]
    assert all(isinstance(s, Skill) for s in result)


def test_json_optional_fields_default_to_empty(skills_file) -> None:
    path = skills_file([{"name": "x"}])
    [skill] = _load_skills(path)
    assert skill.description == ""
    assert skill.body == ""


def test_json_all_fields_populated(skills_file) -> None:
    path = skills_file([{"name": "n", "description": "d", "body": "b"}])
    [skill] = _load_skills(path)
    assert skill == Skill("n", "d", "b")


def test_json_extra_keys_are_ignored(skills_file) -> None:
    # Forward-compat: unknown keys must not break parsing.
    path = skills_file([{"name": "n", "version": 2, "tags": []}])
    [skill] = _load_skills(path)
    assert skill == Skill("n", "", "")


def test_empty_json_list_yields_empty_pool(skills_file) -> None:
    # Boundary: zero skills.
    assert _load_skills(skills_file([])) == []


# --- error paths -----------------------------------------------------------

def test_missing_name_key_raises_key_error(skills_file) -> None:
    path = skills_file([{"description": "d"}])
    with pytest.raises(KeyError, match="name"):
        _load_skills(path)


def test_malformed_json_raises_decode_error(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("{not valid json")
    with pytest.raises(json.JSONDecodeError):
        _load_skills(path)


def test_top_level_dict_raises_type_error(skills_file) -> None:
    # Iterating a dict yields its keys (str); indexing str by str -> TypeError.
    path = skills_file({"name": "x"})
    with pytest.raises(TypeError):
        _load_skills(path)
