"""Every test starts from the package defaults, whatever CLM_* settings the shell has
(the scripts in "Claude Code/scripts" export some of them)."""
import os

import pytest

CLM_SETTINGS = ("CLM_ENCODER", "CLM_MODELS_DIR", "CLM_RUNS_DIR", "CLM_CKPT", "CLM_CKPT_DIR", "CLM_EMB_URL",
                "CLM_EMB_MODEL", "CLM_EMB_MAX_TOKENS", "CLM_DEVICE", "CLM_ACTION_CACHE", "CLM_API_KEY")


@pytest.fixture(autouse=True)
def _default_settings(monkeypatch):
    for k in CLM_SETTINGS:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")      # tests never reach the network
    yield


def pytest_configure(config):
    os.environ.pop("CLM_ENCODER", None)
