from __future__ import annotations

from pathlib import Path

import pytest

from skillrouter_test import weights as W


def test_probe_missing(monkeypatch, tmp_path):
    """An empty cache reports not-ready (stat-only; no network, no model load)."""
    monkeypatch.setenv("SKILLROUTER_MODEL_DIR", str(tmp_path))
    monkeypatch.setattr(W, "_module_available", lambda name: True)
    st = W.probe_resource({"id": "skillrouter-embedding", "repo": "pipizhao/SkillRouter-Embedding-0.6B"})
    assert st.ready is False
    assert st.expected_path  # always tells the caller where it looked


def test_probe_ready_with_fake_snapshot(monkeypatch, tmp_path):
    """config.json + a weight file in the HF cache layout -> ready, found_path set."""
    monkeypatch.setenv("SKILLROUTER_MODEL_DIR", str(tmp_path))
    monkeypatch.setattr(W, "_module_available", lambda name: True)
    repo = "pipizhao/SkillRouter-Embedding-0.6B"
    snap = tmp_path / ("models--" + repo.replace("/", "--")) / "snapshots" / "abc"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")
    (snap / "model.safetensors").write_bytes(b"\0" * 16)
    st = W.probe_resource({"id": "skillrouter-embedding", "repo": repo})
    assert st.ready is True
    assert st.found_path and st.found_path.endswith("model.safetensors")


def test_probe_sharded_snapshot(monkeypatch, tmp_path):
    """Sharded checkpoints expose model.safetensors.index.json + part files."""
    monkeypatch.setenv("SKILLROUTER_MODEL_DIR", str(tmp_path))
    monkeypatch.setattr(W, "_module_available", lambda name: True)
    repo = "pipizhao/SkillRouter-Reranker-0.6B"
    snap = tmp_path / ("models--" + repo.replace("/", "--")) / "snapshots" / "rev"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")
    (snap / "model.safetensors.index.json").write_text("{}")
    (snap / "model-00001-of-00002.safetensors").write_bytes(b"\0")
    st = W.probe_resource({"id": "skillrouter-reranker", "repo": repo})
    assert st.ready is True


def test_download_dry_run_does_not_touch_network(monkeypatch, tmp_path):
    monkeypatch.setenv("SKILLROUTER_MODEL_DIR", str(tmp_path))
    monkeypatch.setattr(W, "_module_available", lambda name: True)
    res = {"id": "skillrouter-embedding", "repo": "pipizhao/SkillRouter-Embedding-0.6B"}
    called = {"snapshot": False}

    def boom(*a, **k):
        called["snapshot"] = True
        raise AssertionError("dry-run must not call snapshot_download")

    monkeypatch.setattr("huggingface_hub.snapshot_download", boom, raising=False)
    from_cache, info, code = W.download_resource(res, dry_run=True)
    assert code == W.EXIT_OK
    assert from_cache is False
    assert info["would_download"] is True
    assert called["snapshot"] is False


def test_download_is_idempotent_when_cached(monkeypatch, tmp_path):
    monkeypatch.setenv("SKILLROUTER_MODEL_DIR", str(tmp_path))
    monkeypatch.setattr(W, "_module_available", lambda name: True)
    repo = "pipizhao/SkillRouter-Embedding-0.6B"
    snap = tmp_path / ("models--" + repo.replace("/", "--")) / "snapshots" / "r"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")
    (snap / "model.safetensors").write_bytes(b"\0" * 32)
    res = {"id": "skillrouter-embedding", "repo": repo}

    def boom(*a, **k):
        raise AssertionError("cached model must not re-download")

    monkeypatch.setattr("huggingface_hub.snapshot_download", boom, raising=False)
    from_cache, info, code = W.download_resource(res)
    assert code == W.EXIT_OK and from_cache is True and info["bytes"] >= 32


def test_local_path_override_is_used_as_is(monkeypatch, tmp_path):
    d = tmp_path / "local-ckpt"
    d.mkdir()
    (d / "config.json").write_text("{}")
    (d / "model.safetensors").write_bytes(b"\0")
    monkeypatch.setenv("SR_EMB_PATH", str(d))
    res = W.managed_resources()[0]
    assert W.probe_resource(res).ready is True
    from_cache, info, code = W.download_resource(res)
    assert code == W.EXIT_OK and from_cache is True


def test_managed_resources_honors_env():
    import os

    base = {"SR_EMB_PATH": "", "SR_RANK_PATH": ""}
    saved = {k: os.environ.get(k) for k in base}
    try:
        for k in base:
            os.environ.pop(k, None)
        assert W.managed_resources()[0]["repo"] == "pipizhao/SkillRouter-Embedding-0.6B"
        os.environ["SR_EMB_PATH"] = "org/custom-emb"
        assert W.managed_resources()[0]["repo"] == "org/custom-emb"
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# --- managed_resources: id/repo shape + per-resource env independence --------

def test_managed_resources_has_two_entries_with_expected_ids():
    res = W.managed_resources()
    assert [r["id"] for r in res] == ["skillrouter-embedding", "skillrouter-reranker"]


def test_managed_resources_default_repos_when_env_unset(monkeypatch):
    monkeypatch.delenv("SR_EMB_PATH", raising=False)
    monkeypatch.delenv("SR_RANK_PATH", raising=False)
    res = W.managed_resources()
    assert res[0]["repo"] == "pipizhao/SkillRouter-Embedding-0.6B"
    assert res[1]["repo"] == "pipizhao/SkillRouter-Reranker-0.6B"


def test_managed_resources_env_overrides_are_independent(monkeypatch):
    monkeypatch.setenv("SR_RANK_PATH", "org/custom-rank")
    monkeypatch.delenv("SR_EMB_PATH", raising=False)
    res = W.managed_resources()
    assert res[0]["repo"] == "pipizhao/SkillRouter-Embedding-0.6B"
    assert res[1]["repo"] == "org/custom-rank"


# --- _module_available -------------------------------------------------------

def test_module_available_returns_true_for_installed():
    # transformers is a hard dependency of this project.
    assert W._module_available("transformers") is True


def test_module_available_returns_false_for_missing():
    assert W._module_available("definitely_not_a_real_pkg_zzz123") is False


def test_module_available_fallback_on_import_error(monkeypatch):
    # If find_spec raises ImportError, the function falls back to a sys.modules
    # membership check — a module already imported but unresolvable via find_spec.
    import importlib.util
    import sys

    def boom(_name):
        raise ImportError("simulated parent-missing error")

    monkeypatch.setattr(importlib.util, "find_spec", boom)
    monkeypatch.setitem(sys.modules, "fake_already_imported", object())
    assert W._module_available("fake_already_imported") is True


def test_module_available_fallback_on_value_error(monkeypatch):
    import importlib.util
    import sys

    def boom(_name):
        raise ValueError("simulated")

    monkeypatch.setattr(importlib.util, "find_spec", boom)
    monkeypatch.setitem(sys.modules, "fake_val_imported", object())
    assert W._module_available("fake_val_imported") is True


def test_module_available_fallback_absent_returns_false(monkeypatch):
    # find_spec raises AND the name is not in sys.modules -> False.
    import importlib.util

    def boom(_name):
        raise ImportError("simulated")

    monkeypatch.setattr(importlib.util, "find_spec", boom)
    assert W._module_available("not_in_sys_modules_either") is False


# --- hf_hub_cache_dir: three-way env resolution -----------------------------

def test_cache_dir_uses_skillrouter_model_dir_override(monkeypatch, tmp_path):
    monkeypatch.setenv("SKILLROUTER_MODEL_DIR", str(tmp_path))
    monkeypatch.setenv("HF_HOME", "/should/be/ignored")
    # Override is used as-is — no "/hub" suffix appended.
    assert W.hf_hub_cache_dir() == tmp_path


def test_cache_dir_override_expands_user(monkeypatch):
    monkeypatch.setenv("SKILLROUTER_MODEL_DIR", "~/my-cache")
    monkeypatch.delenv("HF_HOME", raising=False)
    expected = Path("~/my-cache").expanduser()
    assert W.hf_hub_cache_dir() == expected


def test_cache_dir_uses_hf_home_with_hub_suffix(monkeypatch, tmp_path):
    monkeypatch.delenv("SKILLROUTER_MODEL_DIR", raising=False)
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    assert W.hf_hub_cache_dir() == tmp_path / "hub"


def test_cache_dir_default_when_no_env(monkeypatch):
    monkeypatch.delenv("SKILLROUTER_MODEL_DIR", raising=False)
    monkeypatch.delenv("HF_HOME", raising=False)
    assert W.hf_hub_cache_dir() == Path.home() / ".cache" / "huggingface" / "hub"


def test_cache_dir_model_dir_beats_hf_home(monkeypatch, tmp_path):
    # MODEL_DIR_ENV is checked first and wins over HF_HOME.
    override = tmp_path / "winner"
    monkeypatch.setenv("SKILLROUTER_MODEL_DIR", str(override))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "loser"))
    assert W.hf_hub_cache_dir() == override


# --- _is_local_path ----------------------------------------------------------

def test_is_local_path_false_for_plain_repo_id():
    # No "/" -> cannot be a path.
    assert W._is_local_path("pipizhao") is False


def test_is_local_path_false_for_nonexistent_dir_path():
    assert W._is_local_path("/definitely/does/not/exist/abc") is False


def test_is_local_path_true_for_existing_dir(tmp_path):
    assert W._is_local_path(str(tmp_path)) is True


def test_is_local_path_false_for_existing_file_not_dir(tmp_path):
    f = tmp_path / "file.json"
    f.write_text("{}")
    # Existing file (not a dir) -> False.
    assert W._is_local_path(str(f)) is False


# --- _snapshot_dir_for -------------------------------------------------------

def test_snapshot_dir_for_local_path_returns_path_itself(tmp_path):
    assert W._snapshot_dir_for(str(tmp_path)) == tmp_path


def test_snapshot_dir_for_missing_snapshots_base_returns_base(monkeypatch, tmp_path):
    monkeypatch.setenv("SKILLROUTER_MODEL_DIR", str(tmp_path))
    repo = "pipizhao/SkillRouter-Embedding-0.6B"
    expected = tmp_path / ("models--" + repo.replace("/", "--")) / "snapshots"
    # No snapshots dir created -> returns the (non-existent) base path.
    assert W._snapshot_dir_for(repo) == expected


def test_snapshot_dir_for_empty_snapshots_dir_returns_base(monkeypatch, tmp_path):
    monkeypatch.setenv("SKILLROUTER_MODEL_DIR", str(tmp_path))
    repo = "pipizhao/SkillRouter-Embedding-0.6B"
    base = tmp_path / ("models--" + repo.replace("/", "--")) / "snapshots"
    base.mkdir(parents=True)  # empty
    assert W._snapshot_dir_for(repo) == base


def test_snapshot_dir_for_single_snapshot(monkeypatch, tmp_path):
    monkeypatch.setenv("SKILLROUTER_MODEL_DIR", str(tmp_path))
    repo = "pipizhao/SkillRouter-Embedding-0.6B"
    snap = tmp_path / ("models--" + repo.replace("/", "--")) / "snapshots" / "rev1"
    snap.mkdir(parents=True)
    assert W._snapshot_dir_for(repo) == snap


def test_snapshot_dir_for_picks_most_recent_by_mtime(monkeypatch, tmp_path):
    monkeypatch.setenv("SKILLROUTER_MODEL_DIR", str(tmp_path))
    repo = "pipizhao/SkillRouter-Embedding-0.6B"
    base = tmp_path / ("models--" + repo.replace("/", "--")) / "snapshots"
    old = base / "old"
    new = base / "new"
    old.mkdir(parents=True)
    new.mkdir()
    # Force new to be strictly newer than old.
    import os
    import time

    old_mtime = old.stat().st_mtime
    os.utime(old, (old_mtime, old_mtime))
    time.sleep(0.05)
    now = time.time()
    os.utime(new, (now, now))
    assert W._snapshot_dir_for(repo) == new


# --- _weight_marker: file-type detection + precedence -----------------------

def _snap(tmp_path, name="pipizhao/SkillRouter-Embedding-0.6B"):
    return tmp_path / ("models--" + name.replace("/", "--")) / "snapshots" / "r"


def test_weight_marker_single_safetensors(tmp_path):
    snap = _snap(tmp_path)
    snap.mkdir(parents=True)
    (snap / "model.safetensors").write_bytes(b"\0")
    assert W._weight_marker(snap) == "model.safetensors"


def test_weight_marker_sharded_index(tmp_path):
    snap = _snap(tmp_path)
    snap.mkdir(parents=True)
    (snap / "model.safetensors.index.json").write_text("{}")
    assert W._weight_marker(snap) == "model.safetensors.index.json"


def test_weight_marker_legacy_pytorch_bin(tmp_path):
    snap = _snap(tmp_path)
    snap.mkdir(parents=True)
    (snap / "pytorch_model.bin").write_bytes(b"\0")
    assert W._weight_marker(snap) == "pytorch_model.bin"


def test_weight_marker_glob_fallback_for_arbitrary_safetensors(tmp_path):
    snap = _snap(tmp_path)
    snap.mkdir(parents=True)
    (snap / "model-00001-of-00002.safetensors").write_bytes(b"\0")
    assert W._weight_marker(snap) == "model-00001-of-00002.safetensors"


def test_weight_marker_none_when_no_weights(tmp_path):
    snap = _snap(tmp_path)
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")
    (snap / "README.md").write_text("nope")
    assert W._weight_marker(snap) is None


def test_weight_marker_none_for_empty_dir(tmp_path):
    snap = _snap(tmp_path)
    snap.mkdir(parents=True)
    assert W._weight_marker(snap) is None


def test_weight_marker_safetensors_beats_index_json(tmp_path):
    # Tuple order: model.safetensors is checked before model.safetensors.index.json.
    snap = _snap(tmp_path)
    snap.mkdir(parents=True)
    (snap / "model.safetensors").write_bytes(b"\0")
    (snap / "model.safetensors.index.json").write_text("{}")
    assert W._weight_marker(snap) == "model.safetensors"


def test_weight_marker_named_file_beats_glob(tmp_path):
    # The explicit "model.safetensors" check wins over the *.safetensors glob.
    snap = _snap(tmp_path)
    snap.mkdir(parents=True)
    (snap / "model.safetensors").write_bytes(b"\0")
    (snap / "model-00001-of-00002.safetensors").write_bytes(b"\0")
    assert W._weight_marker(snap) == "model.safetensors"


# --- _snapshot_size ----------------------------------------------------------

def test_snapshot_size_zero_for_non_dir(tmp_path):
    f = tmp_path / "notadir"
    f.write_text("x")
    assert W._snapshot_size(f) == 0


def test_snapshot_size_zero_for_empty_dir(tmp_path):
    assert W._snapshot_size(tmp_path) == 0


def test_snapshot_size_sums_files(tmp_path):
    (tmp_path / "a.bin").write_bytes(b"\0" * 100)
    (tmp_path / "b.bin").write_bytes(b"\0" * 50)
    assert W._snapshot_size(tmp_path) == 150


def test_snapshot_size_walks_subdirs_via_rglob(tmp_path):
    sub = tmp_path / "sub"
    sub.mkdir()
    (tmp_path / "top.bin").write_bytes(b"\0" * 10)
    (sub / "deep.bin").write_bytes(b"\0" * 40)
    assert W._snapshot_size(tmp_path) == 50


# --- WeightStatus serialization ---------------------------------------------

def test_weight_status_to_ready_dict_shape():
    st = W.WeightStatus(name="n", repo="r", found_path="/p/model.safetensors")
    st.ready = True
    d = st.to_ready_dict()
    assert d == {"name": "n", "kind": "local-weights", "repo": "r", "ready": True, "found_path": "/p/model.safetensors"}


def test_weight_status_to_ready_dict_found_path_none_passes_through():
    st = W.WeightStatus(name="n", repo="r")
    st.ready = True
    assert st.to_ready_dict()["found_path"] is None


def test_weight_status_to_missing_dict_with_backend():
    st = W.WeightStatus(name="skillrouter-embedding", repo="org/r", expected_path="/p")
    d = st.to_missing_dict()
    assert d["ready"] is False
    assert d["requires"] == "huggingface repo org/r"  # backend_installed default True
    assert d["expected_path"] == "/p"
    assert d["fix"] == "skillrouter download --model skillrouter-embedding"
    assert d["name"] == "skillrouter-embedding" and d["repo"] == "org/r"


def test_weight_status_to_missing_dict_without_backend():
    st = W.WeightStatus(name="skillrouter-embedding", repo="org/r", expected_path="/p", backend_installed=False)
    d = st.to_missing_dict()
    # When the backend itself is missing, the requires string prepends the backend hint.
    assert d["requires"] == "transformers installed + huggingface repo org/r"


# --- probe_resource: backend-missing short-circuit + malformed snapshot ------

def test_probe_resource_backend_missing_short_circuits_before_stat(monkeypatch, tmp_path):
    # Even with a complete snapshot on disk, no transformers -> not ready,
    # backend_installed=False, and the snapshot contents are never inspected.
    monkeypatch.setenv("SKILLROUTER_MODEL_DIR", str(tmp_path))
    monkeypatch.setattr(W, "_module_available", lambda name: False)
    repo = "pipizhao/SkillRouter-Embedding-0.6B"
    snap = tmp_path / ("models--" + repo.replace("/", "--")) / "snapshots" / "r"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")
    (snap / "model.safetensors").write_bytes(b"\0")
    st = W.probe_resource({"id": "skillrouter-embedding", "repo": repo})
    assert st.ready is False
    assert st.backend_installed is False
    assert st.found_path is None


def test_probe_resource_missing_config_json_not_ready(monkeypatch, tmp_path):
    monkeypatch.setenv("SKILLROUTER_MODEL_DIR", str(tmp_path))
    monkeypatch.setattr(W, "_module_available", lambda name: True)
    repo = "pipizhao/SkillRouter-Embedding-0.6B"
    snap = tmp_path / ("models--" + repo.replace("/", "--")) / "snapshots" / "r"
    snap.mkdir(parents=True)
    (snap / "model.safetensors").write_bytes(b"\0")  # weight present, no config.json
    st = W.probe_resource({"id": "skillrouter-embedding", "repo": repo})
    assert st.ready is False
    assert st.found_path is None


def test_probe_resource_config_but_no_weight_not_ready(monkeypatch, tmp_path):
    monkeypatch.setenv("SKILLROUTER_MODEL_DIR", str(tmp_path))
    monkeypatch.setattr(W, "_module_available", lambda name: True)
    repo = "pipizhao/SkillRouter-Embedding-0.6B"
    snap = tmp_path / ("models--" + repo.replace("/", "--")) / "snapshots" / "r"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")  # no weight file
    st = W.probe_resource({"id": "skillrouter-embedding", "repo": repo})
    assert st.ready is False


def test_probe_resource_legacy_pytorch_bin_is_ready(monkeypatch, tmp_path):
    monkeypatch.setenv("SKILLROUTER_MODEL_DIR", str(tmp_path))
    monkeypatch.setattr(W, "_module_available", lambda name: True)
    repo = "pipizhao/SkillRouter-Reranker-0.6B"
    snap = tmp_path / ("models--" + repo.replace("/", "--")) / "snapshots" / "r"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")
    (snap / "pytorch_model.bin").write_bytes(b"\0" * 8)
    st = W.probe_resource({"id": "skillrouter-reranker", "repo": repo})
    assert st.ready is True
    assert st.found_path and st.found_path.endswith("pytorch_model.bin")


def test_probe_resource_carries_repo_and_expected_path(monkeypatch, tmp_path):
    monkeypatch.setenv("SKILLROUTER_MODEL_DIR", str(tmp_path))
    monkeypatch.setattr(W, "_module_available", lambda name: True)
    repo = "org/custom"
    st = W.probe_resource({"id": "x", "repo": repo})
    assert st.repo == repo
    assert st.name == "x"
    assert st.expected_path  # always populated


# --- probe_all ---------------------------------------------------------------

def test_probe_all_returns_one_status_per_managed_resource(monkeypatch, tmp_path):
    monkeypatch.setenv("SKILLROUTER_MODEL_DIR", str(tmp_path))
    monkeypatch.delenv("SR_EMB_PATH", raising=False)
    monkeypatch.delenv("SR_RANK_PATH", raising=False)
    statuses = W.probe_all()
    assert len(statuses) == 2
    assert {s.name for s in statuses} == {"skillrouter-embedding", "skillrouter-reranker"}
    assert all(isinstance(s, W.WeightStatus) for s in statuses)
    # Empty cache -> both not ready.
    assert all(s.ready is False for s in statuses)


# --- classify_download_error: full branch matrix ----------------------------

def test_classify_timeout_by_exception_name():
    assert W.classify_download_error(TimeoutError("x")) == W.EXIT_TIMEOUT


def test_classify_timeout_by_message_timeout():
    assert W.classify_download_error(RuntimeError("operation timeout")) == W.EXIT_TIMEOUT


def test_classify_timeout_by_message_timed_out():
    assert W.classify_download_error(RuntimeError("connection timed out")) == W.EXIT_TIMEOUT


def test_classify_http_error_by_name():
    class HTTPError(Exception):
        pass

    assert W.classify_download_error(HTTPError("500")) == W.EXIT_NETWORK


def test_classify_request_exception_by_name():
    class RequestException(Exception):
        pass

    assert W.classify_download_error(RequestException("boom")) == W.EXIT_NETWORK


def test_classify_builtin_connection_error_by_name():
    # The builtin ConnectionError's type name lowercased is "connectionerror".
    assert W.classify_download_error(ConnectionError("refused")) == W.EXIT_NETWORK


@pytest.mark.parametrize(
    "msg",
    [
        "the network is unreachable",
        "ssl certificate verify failed",
        "failed to resolve hostname",
        "name or service not known",
        "connection reset by peer",
        "httperror while fetching",  # msg-substring match (name-independent)
    ],
)
def test_classify_network_by_message_hints(msg):
    assert W.classify_download_error(RuntimeError(msg)) == W.EXIT_NETWORK


def test_classify_timeout_beats_network_when_both_present():
    # A network-flavored exception whose message also mentions timeout must
    # resolve to TIMEOUT (timeout branch is checked first).
    class HTTPError(Exception):
        pass

    assert W.classify_download_error(HTTPError("gateway timeout")) == W.EXIT_TIMEOUT


def test_classify_falls_through_to_validation():
    assert W.classify_download_error(ValueError("bad value")) == W.EXIT_VALIDATION
    assert W.classify_download_error(RuntimeError("disk full")) == W.EXIT_VALIDATION
    assert W.classify_download_error(PermissionError("denied")) == W.EXIT_VALIDATION


def test_classify_accepts_base_exception_subtype():
    # The signature is BaseException; KeyboardInterrupt-like classes are in scope.
    class MyKeyboard(BaseException):
        pass

    # No timeout/network hints -> validation fallback.
    assert W.classify_download_error(MyKeyboard("stop")) == W.EXIT_VALIDATION


# --- download_resource: error + force + dry-run branches --------------------

def test_download_dry_run_when_not_cached(monkeypatch, tmp_path):
    monkeypatch.setenv("SKILLROUTER_MODEL_DIR", str(tmp_path))
    monkeypatch.setattr(W, "_module_available", lambda name: True)
    res = {"id": "skillrouter-embedding", "repo": "pipizhao/SkillRouter-Embedding-0.6B"}

    def boom(*a, **k):
        raise AssertionError("dry-run must not call snapshot_download")

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", boom)
    from_cache, info, code = W.download_resource(res, dry_run=True)
    assert code == W.EXIT_OK
    assert from_cache is False
    assert info["would_download"] is True
    assert info["from_cache"] is False
    assert "dest" in info


def test_download_missing_hub_dependency(monkeypatch, tmp_path):
    # When huggingface_hub is not importable, download returns a structured
    # MissingDependency error with EXIT_VALIDATION — no network attempt.
    monkeypatch.setenv("SKILLROUTER_MODEL_DIR", str(tmp_path))

    def fake_avail(name):
        return name != "huggingface_hub"

    monkeypatch.setattr(W, "_module_available", fake_avail)
    res = {"id": "skillrouter-embedding", "repo": "pipizhao/SkillRouter-Embedding-0.6B"}
    from_cache, info, code = W.download_resource(res)
    assert code == W.EXIT_VALIDATION
    assert from_cache is False
    assert info["error_type"] == "MissingDependency"
    assert "huggingface" in info["error"].lower()


def test_download_classifies_timeout_from_snapshot(monkeypatch, tmp_path):
    monkeypatch.setenv("SKILLROUTER_MODEL_DIR", str(tmp_path))
    monkeypatch.setattr(W, "_module_available", lambda name: True)
    import huggingface_hub

    def boom(repo_id, force_download):
        raise TimeoutError("read timed out")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", boom)
    res = {"id": "skillrouter-embedding", "repo": "pipizhao/SkillRouter-Embedding-0.6B"}
    from_cache, info, code = W.download_resource(res)
    assert code == W.EXIT_TIMEOUT
    assert from_cache is False
    assert info["error_type"] == "TimeoutError"
    assert "timed out" in info["error"].lower()


def test_download_classifies_network_error_from_snapshot(monkeypatch, tmp_path):
    monkeypatch.setenv("SKILLROUTER_MODEL_DIR", str(tmp_path))
    monkeypatch.setattr(W, "_module_available", lambda name: True)
    import huggingface_hub

    def boom(repo_id, force_download):
        raise ConnectionError("network unreachable")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", boom)
    res = {"id": "skillrouter-reranker", "repo": "pipizhao/SkillRouter-Reranker-0.6B"}
    _from_cache, info, code = W.download_resource(res)
    assert code == W.EXIT_NETWORK


def test_download_classifies_validation_for_generic_error(monkeypatch, tmp_path):
    monkeypatch.setenv("SKILLROUTER_MODEL_DIR", str(tmp_path))
    monkeypatch.setattr(W, "_module_available", lambda name: True)
    import huggingface_hub

    def boom(repo_id, force_download):
        raise PermissionError("disk full")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", boom)
    res = {"id": "skillrouter-embedding", "repo": "pipizhao/SkillRouter-Embedding-0.6B"}
    _from_cache, info, code = W.download_resource(res)
    assert code == W.EXIT_VALIDATION


def test_download_force_bypasses_cache_and_re_fetches(monkeypatch, tmp_path):
    # A ready snapshot + force=True MUST call snapshot_download(force_download=True),
    # ignoring the cache short-circuit.
    monkeypatch.setenv("SKILLROUTER_MODEL_DIR", str(tmp_path))
    monkeypatch.setattr(W, "_module_available", lambda name: True)
    repo = "pipizhao/SkillRouter-Embedding-0.6B"
    snap = tmp_path / ("models--" + repo.replace("/", "--")) / "snapshots" / "r"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")
    (snap / "model.safetensors").write_bytes(b"\0" * 32)
    res = {"id": "skillrouter-embedding", "repo": repo}

    calls = {"n": 0, "force": None}

    def fake_snapshot(repo_id, force_download):
        calls["n"] += 1
        calls["force"] = force_download
        return str(snap)

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot)
    from_cache, info, code = W.download_resource(res, force=True)
    assert code == W.EXIT_OK
    assert calls["n"] == 1
    assert calls["force"] is True
    assert from_cache is False  # the explicit download branch always reports False


def test_download_local_path_override_never_touches_network(monkeypatch, tmp_path):
    d = tmp_path / "local-ckpt"
    d.mkdir()
    (d / "config.json").write_text("{}")
    (d / "model.safetensors").write_bytes(b"\0")
    monkeypatch.setenv("SR_EMB_PATH", str(d))

    import huggingface_hub

    def boom(*a, **k):
        raise AssertionError("local override must not call snapshot_download")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", boom)
    res = {"id": "skillrouter-embedding", "repo": str(d)}
    from_cache, info, code = W.download_resource(res, force=True, dry_run=True)
    assert code == W.EXIT_OK
    assert from_cache is True
    assert info["from_cache"] is True
    assert info["dest"] == str(d)
