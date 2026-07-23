"""Contract tests for device/dtype resolution helpers.

Contract (from models.py):

``pick_device(requested)``
  - requested is a concrete device (not "auto") -> returned verbatim, no probe
  - requested is None  -> auto-select: cuda > mps > cpu
  - requested == "auto" -> auto-select: cuda > mps > cpu

``_dtype_for(device)``
  - "cuda" -> torch.bfloat16
  - anything else (cpu, mps, custom) -> torch.float32
"""

import pytest
import torch

from skillrouter_test import models


# --- _dtype_for ------------------------------------------------------------

@pytest.mark.parametrize(
    ("device", "expected"),
    [("cuda", torch.bfloat16), ("cpu", torch.float32), ("mps", torch.float32)],
)
def test_dtype_for_returns_expected(device: str, expected: torch.dtype) -> None:
    assert models._dtype_for(device) is expected


def test_dtype_for_defaults_to_float32_for_unknown_device() -> None:
    # Custom/unknown device strings fall through to the float32 branch.
    assert models._dtype_for("tpu") is torch.float32


# --- pick_device: explicit passthrough (no probing) ------------------------

@pytest.mark.parametrize("requested", ["cuda", "cpu", "mps", "cuda:1", "tpu"])
def test_explicit_device_returned_verbatim(requested: str, monkeypatch: pytest.MonkeyPatch) -> None:
    # Any concrete device is returned untouched, even if it would fail availability.
    sabotage = lambda *a, **k: pytest.fail("availability must NOT be probed for an explicit device")
    monkeypatch.setattr(models.torch.cuda, "is_available", sabotage)
    monkeypatch.setattr(models.torch.backends.mps, "is_available", sabotage)
    assert models.pick_device(requested) == requested


def test_explicit_device_does_not_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    # "auto" is the ONLY string that triggers probing; verify that an explicit
    # device short-circuits before touching the availability checks.
    calls = {"cuda": 0, "mps": 0}

    def boom_cuda():
        calls["cuda"] += 1
        raise RuntimeError("should not probe")

    monkeypatch.setattr(models.torch.cuda, "is_available", boom_cuda)
    monkeypatch.setattr(models.torch.backends.mps, "is_available", lambda: False)
    assert models.pick_device("cpu") == "cpu"
    assert calls == {"cuda": 0, "mps": 0}


# --- pick_device: auto-selection ladder cuda > mps > cpu -------------------

def test_auto_prefers_cuda_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(models.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(models.torch.backends.mps, "is_available", lambda: True)  # cuda wins
    assert models.pick_device("auto") == "cuda"
    assert models.pick_device(None) == "cuda"


def test_auto_falls_back_to_mps_when_no_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(models.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(models.torch.backends.mps, "is_available", lambda: True)
    assert models.pick_device("auto") == "mps"
    assert models.pick_device(None) == "mps"


def test_auto_falls_back_to_cpu_when_nothing_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(models.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(models.torch.backends.mps, "is_available", lambda: False)
    assert models.pick_device("auto") == "cpu"
    assert models.pick_device(None) == "cpu"


def test_none_and_auto_select_identically(monkeypatch: pytest.MonkeyPatch) -> None:
    # The two entry points for "auto" must agree under every availability config.
    monkeypatch.setattr(models.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(models.torch.backends.mps, "is_available", lambda: True)
    assert models.pick_device(None) == models.pick_device("auto")


# --- short-circuit precedence: explicit path beats availability -----------

def test_explicit_cpu_ignores_cuda_availability(monkeypatch: pytest.MonkeyPatch) -> None:
    # Even with cuda present, an explicit "cpu" must win — proves the ladder is
    # only consulted on the auto/None branch.
    monkeypatch.setattr(models.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(models.torch.backends.mps, "is_available", lambda: True)
    assert models.pick_device("cpu") == "cpu"
