"""Read-only readiness probe + idempotent download for the two SkillRouter
checkpoints. Both are transformers models fetched on first use via
``AutoModel.from_pretrained`` (see :mod:`skillrouter_test.models`); this module
mirrors that path so a model fetched by ``skillrouter download`` is the one the
pipeline loads.

Contract (skill design-cli §十九):

* :func:`probe_resource` / :func:`probe_all` are READ-ONLY and fast — only
  ``Path``/``stat`` calls plus an ``importlib`` probe for ``transformers``.
  They never import ``torch``, never load a model, never touch the network,
  never write. This backs ``skillrouter available``.
* :func:`download_resource` is idempotent + ``--force`` + ``--dry-run``;
  :func:`classify_download_error` maps failures to ADR-657 exit codes.

Managed resources (two, honoring the ``SR_EMB_PATH`` / ``SR_RANK_PATH`` overrides
used by :mod:`models` — an override may be a local dir or an HF repo id):

* ``skillrouter-embedding`` — ``pipizhao/SkillRouter-Embedding-0.6B``
* ``skillrouter-reranker``  — ``pipizhao/SkillRouter-Reranker-0.6B``
"""

from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass
from pathlib import Path

# Exit codes — ADR-657.
EXIT_OK = 0
EXIT_USAGE = 2
EXIT_VALIDATION = 3  # disk / permission / checksum
EXIT_NETWORK = 4
EXIT_TIMEOUT = 124

#: Environment override for the HF cache root (skill design-cli §6 convention).
MODEL_DIR_ENV = "SKILLROUTER_MODEL_DIR"

_EMB_REPO_DEFAULT = "pipizhao/SkillRouter-Embedding-0.6B"
_RANK_REPO_DEFAULT = "pipizhao/SkillRouter-Reranker-0.6B"

#: ``config.json`` must be present for a snapshot to count as complete.
PROBE_FILES = ("config.json",)


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        import sys

        return name in sys.modules


def managed_resources() -> list[dict]:
    """The two SkillRouter checkpoints, honoring SR_EMB_PATH/SR_RANK_PATH."""
    return [
        {"id": "skillrouter-embedding", "repo": os.environ.get("SR_EMB_PATH", _EMB_REPO_DEFAULT)},
        {"id": "skillrouter-reranker", "repo": os.environ.get("SR_RANK_PATH", _RANK_REPO_DEFAULT)},
    ]


def hf_hub_cache_dir() -> Path:
    """Root of the HF hub cache. SKILLROUTER_MODEL_DIR → HF_HOME/hub → platform default."""
    override = os.environ.get(MODEL_DIR_ENV)
    if override:
        return Path(override).expanduser()
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home).expanduser() / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def _is_local_path(repo: str) -> bool:
    """An override that points at an existing local dir is used as-is by transformers."""
    return "/" in repo and Path(repo).expanduser().is_dir()


def _snapshot_dir_for(repo: str) -> Path:
    """On-disk snapshot dir for ``repo`` (HF cache layout) or the local dir itself."""
    if _is_local_path(repo):
        return Path(repo).expanduser()
    repo_dir_name = "models--" + repo.replace("/", "--")
    base = hf_hub_cache_dir() / repo_dir_name / "snapshots"
    if not base.is_dir():
        return base
    snaps = [p for p in base.iterdir() if p.is_dir()]
    if not snaps:
        return base
    return max(snaps, key=lambda p: p.stat().st_mtime)


def _weight_marker(snap: Path) -> str | None:
    """A transformers snapshot is complete once config.json + a weight file exist.
    The 0.6B checkpoints may be a single safetensors, sharded (.index.json + parts),
    or legacy pytorch_model.bin."""
    for m in ("model.safetensors", "model.safetensors.index.json", "pytorch_model.bin"):
        if (snap / m).is_file():
            return m
    for f in snap.glob("*.safetensors"):
        if f.is_file():
            return f.name
    return None


@dataclass
class WeightStatus:
    """Result of probing one local-weight resource."""

    name: str
    repo: str
    kind: str = "local-weights"
    ready: bool = False
    found_path: str | None = None
    expected_path: str = ""
    backend_installed: bool = True

    def to_ready_dict(self) -> dict:
        return {"name": self.name, "kind": self.kind, "repo": self.repo, "ready": True, "found_path": self.found_path}

    def to_missing_dict(self) -> dict:
        return {
            "name": self.name,
            "kind": self.kind,
            "repo": self.repo,
            "ready": False,
            "requires": (
                f"transformers installed + huggingface repo {self.repo}"
                if not self.backend_installed
                else f"huggingface repo {self.repo}"
            ),
            "expected_path": self.expected_path,
            "fix": f"skillrouter download --model {self.name}",
        }


def probe_resource(res: dict) -> WeightStatus:
    """Read-only probe of one checkpoint. ``importlib`` + ``stat`` only."""
    repo = res["repo"]
    backend = _module_available("transformers")
    snap = _snapshot_dir_for(repo)
    expected = str(snap)
    st = WeightStatus(name=res["id"], repo=repo, expected_path=expected, backend_installed=backend)
    if not backend:
        return st
    try:
        if not snap.is_dir() or not all((snap / f).is_file() for f in PROBE_FILES):
            return st
        marker = _weight_marker(snap)
        if marker is None:
            return st
        st.ready = True
        st.found_path = str(snap / marker)
        return st
    except OSError:
        return st


def probe_all() -> list[WeightStatus]:
    return [probe_resource(r) for r in managed_resources()]


def classify_download_error(exc: BaseException) -> int:
    """Map a download exception to an ADR-657 exit code."""
    name = type(exc).__name__.lower()
    msg = str(exc).lower()
    if "timeout" in name or "timeout" in msg or "timed out" in msg:
        return EXIT_TIMEOUT
    network_hints = ("connection", "unreachable", "network", "httperror", "ssl", "resolve", "name or service")
    if any(h in name for h in ("httperror", "requestexception", "connectionerror")) or any(h in msg for h in network_hints):
        return EXIT_NETWORK
    return EXIT_VALIDATION


def _snapshot_size(snap: Path) -> int:
    if not snap.is_dir():
        return 0
    total = 0
    try:
        for e in snap.rglob("*"):
            if e.is_file():
                total += e.stat().st_size
    except OSError:
        pass
    return total


def download_resource(res: dict, *, force: bool = False, dry_run: bool = False) -> tuple[bool, dict, int]:
    """Idempotently ensure one checkpoint is cached. Returns ``(from_cache, info, exit_code)``."""
    repo = res["repo"]
    model_id = res["id"]
    if _is_local_path(repo):
        return True, {"model": model_id, "repo": repo, "dest": repo, "from_cache": True, "note": "local path override; nothing to download"}, EXIT_OK

    status = probe_resource(res)
    expected = status.expected_path or str(_snapshot_dir_for(repo))
    if status.ready and not force:
        return True, {"model": model_id, "repo": repo, "dest": status.found_path, "bytes": _snapshot_size(_snapshot_dir_for(repo)), "from_cache": True}, EXIT_OK

    if dry_run:
        return False, {"model": model_id, "repo": repo, "dest": expected, "from_cache": False, "would_download": True}, EXIT_OK

    if not _module_available("huggingface_hub"):
        return False, {"model": model_id, "error": "huggingface-hub is required to download weights", "error_type": "MissingDependency"}, EXIT_VALIDATION

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        return False, {"model": model_id, "error": str(exc), "error_type": type(exc).__name__}, EXIT_VALIDATION

    try:
        snapshot_download(repo_id=repo, force_download=force)
    except BaseException as exc:  # noqa: BLE001 — classify for the caller
        return False, {"model": model_id, "error": str(exc), "error_type": type(exc).__name__}, classify_download_error(exc)

    refreshed = probe_resource(res)
    return False, {"model": model_id, "repo": repo, "dest": refreshed.found_path or expected, "bytes": _snapshot_size(_snapshot_dir_for(repo)), "from_cache": False}, EXIT_OK
