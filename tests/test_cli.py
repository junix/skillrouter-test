"""Contract tests for the torch-free CLI surface.

The weight-lifecycle subcommands (``available``, ``doctor``, ``describe``,
``download``) are deliberately importable WITHOUT torch/transformers model
loading — they only consult :mod:`skillrouter_test.weights`. That makes them
unit-testable in isolation via ``typer.testing.CliRunner``.

Contract (from cli.py):
  - ``available``       : read-only probe; missing weights are NOT fatal (exit 0);
                          ``--json`` emits ``{"ok": bool, "models": [...]}``.
  - ``doctor``          : always exit 0 if the binary is healthy; ``--json`` emits
                          ``{"binary_ok": true, "backend": {"transformers": bool}, ...}``.
  - ``describe``        : ``--json`` emits name/source/capabilities/models.
  - ``download``        : idempotent; ``--dry-run`` never touches the network;
                          unknown ``--model`` -> exit code 2 (EXIT_USAGE).
  - ``_source_path``    : walks up from cli.py to the dir containing pyproject.toml
                          and rewrites the home prefix to ``~``.
  - ``_table``          : 3-column display; score rendered with ``{:.4f}``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from rich.console import Console
from typer.testing import CliRunner

from skillrouter_test import weights as W
from skillrouter_test.cli import _source_path, _table, app

runner = CliRunner()


@pytest.fixture(autouse=True)
def _isolated_weight_env(monkeypatch, tmp_path):
    """Point the weight cache at an empty tmp dir and clear repo overrides so
    every CLI probe is deterministic (both checkpoints reported missing)."""
    monkeypatch.setenv("SKILLROUTER_MODEL_DIR", str(tmp_path))
    monkeypatch.delenv("SR_EMB_PATH", raising=False)
    monkeypatch.delenv("SR_RANK_PATH", raising=False)
    yield


# --- _source_path ------------------------------------------------------------

def test_source_path_returns_string_with_pyproject_root():
    src = _source_path()
    assert isinstance(src, str)
    assert src  # non-empty
    # The returned root (with ~ expanded back) must contain pyproject.toml.
    root = Path(src.replace("~", os.path.expanduser("~"), 1))
    assert (root / "pyproject.toml").is_file()


def test_source_path_rewrites_home_to_tilde():
    src = _source_path()
    home = os.path.expanduser("~")
    if src.startswith(home):
        assert src.startswith("~")
    # And never returns a path that still carries the literal home prefix.
    assert not src.startswith(home + "/")


def test_source_path_locates_skillrouter_root():
    # The root is the skillrouter repo dir itself.
    src = _source_path()
    assert src.rstrip("/").endswith("skillrouter")


# --- _table: rendering contract ---------------------------------------------

def test_table_has_three_columns_and_title():
    table = _table("my-title", [(1, "alpha", 0.5)])
    assert table.title == "my-title"
    assert len(table.columns) == 3


def test_table_row_count_matches_input():
    rows = [(1, "a", 0.1), (2, "b", 0.2), (3, "c", 0.3)]
    table = _table("t", rows)
    assert table.row_count == 3


def test_table_renders_score_with_four_decimals():
    # Render to a capturing console and verify the score formatting contract.
    console = Console(record=True, width=120, force_terminal=False, color_system=None)
    console.print(_table("t", [(1, "alpha", 0.5)]))
    out = console.export_text()
    assert "alpha" in out
    assert "0.5000" in out


def test_table_renders_rank_and_all_names():
    console = Console(record=True, width=120, force_terminal=False, color_system=None)
    console.print(_table("t", [(1, "alpha", 0.12), (2, "beta", -3.0)]))
    out = console.export_text()
    assert "1" in out and "2" in out
    assert "alpha" in out and "beta" in out
    assert "0.1200" in out
    assert "-3.0000" in out


def test_table_empty_rows_renders_title_only():
    table = _table("empty-title", [])
    assert table.row_count == 0
    console = Console(record=True, width=120, force_terminal=False, color_system=None)
    console.print(table)
    assert "empty-title" in console.export_text()


# --- available: read-only, missing-not-fatal --------------------------------

def test_available_json_exits_zero_with_payload_shape():
    result = runner.invoke(app, ["available", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert set(payload) >= {"ok", "models"}
    assert payload["ok"] is False  # empty cache -> not ready


def test_available_json_lists_exactly_two_managed_models():
    payload = json.loads(runner.invoke(app, ["available", "--json"]).stdout)
    names = [m["name"] for m in payload["models"]]
    assert names == ["skillrouter-embedding", "skillrouter-reranker"]


def test_available_json_missing_model_has_fix_and_requires():
    payload = json.loads(runner.invoke(app, ["available", "--json"]).stdout)
    emb = next(m for m in payload["models"] if m["name"] == "skillrouter-embedding")
    assert emb["ready"] is False
    assert emb["fix"] == "skillrouter download --model skillrouter-embedding"
    assert emb["requires"]
    assert emb["expected_path"]


def test_available_table_form_mentions_both_models():
    result = runner.invoke(app, ["available"])
    assert result.exit_code == 0
    assert "skillrouter-embedding" in result.stdout
    assert "skillrouter-reranker" in result.stdout


def test_available_is_missing_not_fatal_when_cache_empty():
    # Even with nothing cached, exit code stays 0 (reported, not fatal).
    assert runner.invoke(app, ["available"]).exit_code == 0


def test_available_reflects_cache_populated(monkeypatch, tmp_path):
    # Populate a fake embedding snapshot -> available reports ok for that model.
    monkeypatch.setenv("SKILLROUTER_MODEL_DIR", str(tmp_path))
    repo = "pipizhao/SkillRouter-Embedding-0.6B"
    snap = tmp_path / ("models--" + repo.replace("/", "--")) / "snapshots" / "r"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")
    (snap / "model.safetensors").write_bytes(b"\0")
    monkeypatch.setattr(W, "_module_available", lambda name: True)
    payload = json.loads(runner.invoke(app, ["available", "--json"]).stdout)
    emb = next(m for m in payload["models"] if m["name"] == "skillrouter-embedding")
    rank = next(m for m in payload["models"] if m["name"] == "skillrouter-reranker")
    assert emb["ready"] is True and rank["ready"] is False
    assert payload["ok"] is False  # still False because reranker is missing


# --- doctor -----------------------------------------------------------------

def test_doctor_json_binary_ok_and_backend_reported():
    result = runner.invoke(app, ["doctor", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["binary_ok"] is True
    assert payload["backend"] == {"transformers": True}


def test_doctor_json_includes_models_and_ok_field():
    payload = json.loads(runner.invoke(app, ["doctor", "--json"]).stdout)
    assert "ok" in payload
    assert len(payload["models"]) == 2


def test_doctor_table_form_reports_backend_line():
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "transformers=yes" in result.stdout
    assert "skillrouter-embedding" in result.stdout


def test_doctor_reports_missing_backend_when_unavailable(monkeypatch, tmp_path):
    monkeypatch.setenv("SKILLROUTER_MODEL_DIR", str(tmp_path))
    monkeypatch.setattr(W, "_module_available", lambda name: False)
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "transformers=no" in result.stdout


# --- describe ---------------------------------------------------------------

def test_describe_json_contract():
    result = runner.invoke(app, ["describe", "--json"])
    assert result.exit_code == 0
    info = json.loads(result.stdout)
    assert info["name"] == "skillrouter"
    assert info["capabilities"] == ["retrieve", "rerank", "route"]
    assert "source" in info and info["source"]
    assert len(info["models"]) == 2


def test_describe_json_models_have_kind_local_weights():
    info = json.loads(runner.invoke(app, ["describe", "--json"]).stdout)
    for m in info["models"]:
        assert set(m) == {"id", "repo", "kind"}
        assert m["kind"] == "local-weights"


def test_describe_json_model_ids_and_repos():
    info = json.loads(runner.invoke(app, ["describe", "--json"]).stdout)
    by_id = {m["id"]: m["repo"] for m in info["models"]}
    assert by_id == {
        "skillrouter-embedding": "pipizhao/SkillRouter-Embedding-0.6B",
        "skillrouter-reranker": "pipizhao/SkillRouter-Reranker-0.6B",
    }


def test_describe_source_path_resolves_to_repo():
    info = json.loads(runner.invoke(app, ["describe", "--json"]).stdout)
    root = Path(info["source"].replace("~", os.path.expanduser("~"), 1))
    assert (root / "pyproject.toml").is_file()


# --- download ---------------------------------------------------------------

def test_download_dry_run_json_never_touches_network(monkeypatch):
    # Dry-run on an empty cache reports would_download for both, exit 0.
    import huggingface_hub

    def boom(*a, **k):
        raise AssertionError("dry-run must not call snapshot_download")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", boom)
    result = runner.invoke(app, ["download", "--dry-run", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert len(payload["results"]) == 2
    for r in payload["results"]:
        assert r["from_cache"] is False
        assert r["would_download"] is True


def test_download_dry_run_table_form_exits_zero():
    result = runner.invoke(app, ["download", "--dry-run"])
    assert result.exit_code == 0
    assert "would download" in result.stdout


def test_download_unknown_model_exits_usage_code(monkeypatch):
    monkeypatch.setattr(W, "_module_available", lambda name: True)
    result = runner.invoke(app, ["download", "--model", "bogus", "--json"])
    assert result.exit_code == W.EXIT_USAGE
    assert "unknown model: bogus" in result.stdout


def test_download_known_single_model(monkeypatch, tmp_path):
    # Selecting only the embedding model produces exactly one result.
    monkeypatch.setenv("SKILLROUTER_MODEL_DIR", str(tmp_path))
    monkeypatch.setattr(W, "_module_available", lambda name: True)
    result = runner.invoke(app, ["download", "--model", "skillrouter-embedding", "--dry-run", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    [r] = payload["results"]
    assert r["model"] == "skillrouter-embedding"


def test_download_cached_model_exits_zero_without_network(monkeypatch, tmp_path):
    monkeypatch.setenv("SKILLROUTER_MODEL_DIR", str(tmp_path))
    monkeypatch.setattr(W, "_module_available", lambda name: True)
    repo = "pipizhao/SkillRouter-Embedding-0.6B"
    snap = tmp_path / ("models--" + repo.replace("/", "--")) / "snapshots" / "r"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")
    (snap / "model.safetensors").write_bytes(b"\0" * 32)

    import huggingface_hub

    def boom(*a, **k):
        raise AssertionError("cached model must not re-download without --force")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", boom)
    result = runner.invoke(app, ["download", "--model", "skillrouter-embedding", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    [r] = payload["results"]
    assert r["from_cache"] is True


# --- command help / registration --------------------------------------------

def test_all_lifecycle_commands_are_registered():
    from skillrouter_test.cli import app as _app

    names = {cmd.name or cmd.callback.__name__ for cmd in _app.registered_commands}
    for expected in ("available", "doctor", "describe", "download"):
        assert expected in names, f"missing command: {expected}"


def test_describe_help_lists_capabilities():
    result = runner.invoke(app, ["describe", "--help"])
    assert result.exit_code == 0
    assert "capabilities" in result.stdout.lower() or "self-describe" in result.stdout.lower()
