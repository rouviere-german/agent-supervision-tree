"""Shared test helpers: a controllable clock, tiny actors and waiting utilities.

``tests/conftest.py`` puts this directory on ``sys.path``, so every test module can
``import helpers`` regardless of where it lives.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterable
from typing import Any

from agent_supervision_tree.runtime.actor import Actor, ActorContext, ExitReason
from agent_supervision_tree.runtime.errors import ActorExit, MessageFailed
from agent_supervision_tree.runtime.messages import Envelope
from agent_supervision_tree.runtime.supervisor import ActorSpec, ChildSpec, SupervisorSpec
from agent_supervision_tree.runtime.system import ActorSystem

__all__ = [
    "BaseCrashingActor",
    "BlockingActor",
    "CountingActor",
    "CrashingActor",
    "FakeClock",
    "HardCrash",
    "SlowStartActor",
    "actor_child",
    "build_system",
    "counter",
    "envelope",
    "wait_for",
]


class FakeClock:
    """A monotonic clock the test advances by hand (restart intensity windows are exact)."""

    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> float:
        self.now += seconds
        return self.now


class CountingActor(Actor):
    """Counts what it receives; optionally fails or exits on specific message kinds."""

    def __init__(
        self,
        *,
        log: list[tuple[str, str]] | None = None,
        name: str = "counting",
        accepts: Iterable[str] | None = None,
    ) -> None:
        self.received: list[Envelope] = []
        self.starts = 0
        self.stops: list[ExitReason] = []
        self.log = log if log is not None else []
        self.name = name
        if accepts is not None:
            self.accepts = frozenset(accepts)

    async def on_start(self, ctx: ActorContext) -> None:
        self.starts += 1
        self.log.append((self.name, "start"))

    async def receive(self, envelope: Envelope, ctx: ActorContext) -> None:
        self.received.append(envelope)
        payload = envelope.payload if isinstance(envelope.payload, dict) else {}
        action = payload.get("action")
        if action == "fail":
            raise MessageFailed(str(payload.get("reason", "failed")))
        if action == "fail_final":
            raise MessageFailed(str(payload.get("reason", "failed")), retryable=False)
        if action == "crash":
            raise RuntimeError("actor crashed on purpose")
        if action == "exit":
            raise ActorExit
        if action == "quarantine":
            ctx.quarantine(envelope, "by_request")

    async def on_stop(self, reason: ExitReason, ctx: ActorContext) -> None:
        self.stops.append(reason)
        self.log.append((self.name, f"stop:{reason.kind}"))


class CrashingActor(Actor):
    """Crashes ``crashes`` times on start, then behaves; used for restart-intensity tests."""

    def __init__(self, counter: list[int], crashes: int = 10**9) -> None:
        self.counter = counter
        self.crashes = crashes

    async def on_start(self, ctx: ActorContext) -> None:
        self.counter.append(1)
        if len(self.counter) <= self.crashes:
            raise RuntimeError("start failed")

    async def receive(self, envelope: Envelope, ctx: ActorContext) -> None:
        return None


class SlowStartActor(Actor):
    """Takes ``delay`` seconds to start; used for start-timeout and ordering tests."""

    def __init__(self, delay: float, log: list[str] | None = None, name: str = "slow") -> None:
        self.delay = delay
        self.log = log if log is not None else []
        self.name = name

    async def on_start(self, ctx: ActorContext) -> None:
        await asyncio.sleep(self.delay)
        self.log.append(f"{self.name}:started")

    async def receive(self, envelope: Envelope, ctx: ActorContext) -> None:
        return None


class HardCrash(BaseException):
    """A ``BaseException`` a behavior can raise: ``except Exception`` does not catch it."""


class BaseCrashingActor(Actor):
    """Raises a ``BaseException`` from ``receive`` while the message is still in flight.

    Third-party tool handlers and SDKs raise ``SystemExit`` (``sys.exit()``) and other
    ``BaseException`` subclasses; the runner must contain them like any other failure.
    """

    def __init__(self, error: BaseException) -> None:
        self.error = error
        self.handled: list[str] = []
        self.stops: list[ExitReason] = []

    async def receive(self, envelope: Envelope, ctx: ActorContext) -> None:
        self.handled.append(envelope.id)
        raise self.error

    async def on_stop(self, reason: ExitReason, ctx: ActorContext) -> None:
        self.stops.append(reason)


class BlockingActor(Actor):
    """Blocks in ``receive`` until the test releases it (in-flight redelivery tests)."""

    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.handled: list[str] = []

    async def receive(self, envelope: Envelope, ctx: ActorContext) -> None:
        self.entered.set()
        await self.release.wait()
        self.handled.append(envelope.id)


def actor_child(
    child_id: str,
    factory: Callable[[], Actor],
    *,
    restart: str = "permanent",
    capacity: int = 8,
    max_attempts: int = 3,
    shutdown_timeout: float = 2.0,
    start_timeout: float = 2.0,
    stop_timeout: float = 2.0,
) -> ChildSpec:
    return ChildSpec(
        id=child_id,
        spec=ActorSpec(
            factory=factory,
            mailbox_capacity=capacity,
            max_attempts=max_attempts,
            start_timeout=start_timeout,
            stop_timeout=stop_timeout,
        ),
        restart=restart,  # type: ignore[arg-type]
        shutdown_timeout=shutdown_timeout,
    )


def build_system(
    children: Iterable[ChildSpec],
    *,
    strategy: str = "one_for_one",
    max_restarts: int = 3,
    max_seconds: float = 5.0,
    **kwargs: Any,
) -> ActorSystem:
    spec = SupervisorSpec(
        strategy=strategy,  # type: ignore[arg-type]
        children=tuple(children),
        max_restarts=max_restarts,
        max_seconds=max_seconds,
    )
    return ActorSystem(spec, **kwargs)


def envelope(kind: str = "test.message", **payload: Any) -> Envelope:
    return Envelope.create(kind, payload)


def counter(system: ActorSystem, name: str, **labels: str) -> float:
    """A metric sample by its public registry name (no private prometheus_client access)."""
    value = system.metrics.registry.get_sample_value(name, labels or None)
    return 0.0 if value is None else float(value)


async def wait_for(
    predicate: Callable[[], bool], *, timeout: float = 2.0, interval: float = 0.001
) -> bool:
    """Poll ``predicate`` until it holds or ``timeout`` (wall clock) elapses."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if predicate():
            return True
        await asyncio.sleep(interval)
    return predicate()
