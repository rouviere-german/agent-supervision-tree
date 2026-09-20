"""Shared fixtures. Every test runs offline: non-loopback sockets are blocked."""

from __future__ import annotations

import logging
import os
import socket
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import structlog
from hypothesis import HealthCheck, settings

# The profile sets how deep every property and stateful test searches: a test that pins its own
# max_examples or stateful_step_count would ignore it, so CI would search no deeper than a dev
# run. Expensive campaigns scale their budget with ``helpers.scaled_settings`` instead.
settings.register_profile("dev", max_examples=60, stateful_step_count=15, deadline=None)
settings.register_profile(
    "ci",
    max_examples=200,
    stateful_step_count=40,
    deadline=None,
    derandomize=True,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "dev"))

REPO_ROOT = Path(__file__).resolve().parents[1]
_LOOPBACK = frozenset({"127.0.0.1", "::1", "localhost"})

# Make ``import helpers`` work from every test module, wherever it lives.
if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))


@pytest.fixture(autouse=True)
def _offline_and_clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    for name in list(os.environ):
        if name.startswith("APP_"):
            monkeypatch.delenv(name, raising=False)

    real_connect = socket.socket.connect

    def guarded_connect(self: socket.socket, address: Any) -> None:
        host = address[0] if isinstance(address, tuple) else str(address)
        if host not in _LOOPBACK:
            raise RuntimeError(f"tests must not use the network (attempted {host})")
        real_connect(self, address)

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)


@pytest.fixture
def restore_logging() -> Iterator[None]:
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    yield
    structlog.reset_defaults()
    root.handlers = handlers
    root.setLevel(level)
    logging.getLogger("uvicorn.access").disabled = False


@pytest.fixture
def repo_root() -> Path:
    return REPO_ROOT
