from __future__ import annotations

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
