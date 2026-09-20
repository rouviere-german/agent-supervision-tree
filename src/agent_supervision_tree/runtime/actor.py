"""Actors, their context, and the runner that gives one actor incarnation its lifecycle.

An :class:`Actor` is a behavior (``receive``) plus lifecycle hooks (``on_start``, ``on_stop``).
An :class:`ActorRunner` owns one *incarnation*: it constructs the actor, runs the receive loop
over the slot's mailbox, tracks in-flight messages, and turns whatever ends the loop into an
:class:`ExitReason` that the supervisor decides on. The mailbox is not owned by the runner, so
a crash cannot take unacknowledged messages with it: on exit the runner puts them back for the
next incarnation, counting an attempt when the exit was abnormal, and quarantines a message
whose attempts are used up.

Tasks an actor spawns through :meth:`ActorContext.spawn` are structured: they are cancelled
when the actor exits, and a spawned task that fails crashes the actor (the way a linked
process would in Erlang), so no failure is silently lost in a background task.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Callable, Coroutine, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol

from opentelemetry.trace import Tracer
from pydantic import JsonValue

from agent_supervision_tree.core.logging import get_logger
from agent_supervision_tree.runtime.errors import ActorExit, MessageFailed, QuarantineFullError
from agent_supervision_tree.runtime.mailbox import Mailbox
from agent_supervision_tree.runtime.messages import Envelope

if TYPE_CHECKING:
    from agent_supervision_tree.runtime.audit import AuditChain
    from agent_supervision_tree.runtime.metrics import RuntimeMetrics
    from agent_supervision_tree.runtime.quarantine import Quarantine

__all__ = [
    "Actor",
    "ActorContext",
    "ActorRef",
    "ActorRunner",
    "ExitKind",
    "ExitReason",
    "MonitorCallback",
    "SystemAPI",
]

ExitKind = Literal["normal", "shutdown", "error", "killed"]
MonitorCallback = Callable[[str, int, "ExitReason"], None]
_log = get_logger("agent_supervision_tree.actor")


@dataclass(frozen=True, slots=True)
class ExitReason:
    """Why an incarnation ended. ``abnormal`` exits count as failures for supervision."""

    kind: ExitKind
    detail: str = ""
    error: BaseException | None = field(default=None, compare=False)

    @classmethod
    def normal(cls) -> ExitReason:
        return cls("normal")

    @classmethod
    def shutdown(cls, detail: str = "") -> ExitReason:
        return cls("shutdown", detail)

    @classmethod
    def failed(cls, error: BaseException, detail: str = "") -> ExitReason:
        return cls("error", detail or type(error).__name__, error)

    @classmethod
    def killed(cls, detail: str) -> ExitReason:
        return cls("killed", detail)

    @property
    def abnormal(self) -> bool:
        return self.kind in ("error", "killed")

    def label(self) -> str:
        """A low-cardinality label for metrics and audit records."""
        return self.kind if not self.detail else f"{self.kind}:{self.detail}"[:64]


class SystemAPI(Protocol):
    """What actors may ask of the runtime (implemented by ``ActorSystem``)."""

    metrics: RuntimeMetrics
    audit: AuditChain
    quarantine: Quarantine
    tracer: Tracer

    def clock(self) -> float: ...

    def mailbox(self, path: str) -> Mailbox: ...

    def on_actor_exit(self, path: str, incarnation: int, reason: ExitReason) -> None: ...


class ActorRef:
    """A handle on a slot's mailbox; valid across restarts of the actor behind it."""

    __slots__ = ("_system", "path")

    def __init__(self, path: str, system: SystemAPI) -> None:
        self.path = path
        self._system = system

    @property
    def mailbox(self) -> Mailbox:
        return self._system.mailbox(self.path)

    def try_send(self, envelope: Envelope) -> bool:
        return self.mailbox.try_send(envelope)

    async def send(self, envelope: Envelope, *, timeout: float | None = None) -> None:
        await self.mailbox.send(envelope, timeout=timeout)


class ActorContext:
    """Everything a behavior needs: identity, clock, messaging, quarantine, structured tasks."""

    def __init__(self, runner: ActorRunner, system: SystemAPI) -> None:
        self._runner = runner
        self._system = system
        self.log = get_logger("agent_supervision_tree.actor", actor=runner.path)

    @property
    def path(self) -> str:
        return self._runner.path

    @property
    def incarnation(self) -> int:
        return self._runner.incarnation

    @property
    def system(self) -> SystemAPI:
        return self._system

    def clock(self) -> float:
        return self._system.clock()

    def ref(self, path: str) -> ActorRef:
        return ActorRef(path, self._system)

    async def send(self, path: str, envelope: Envelope, *, timeout: float | None = None) -> None:
        await self._system.mailbox(path).send(envelope, timeout=timeout)

    def try_send(self, path: str, envelope: Envelope) -> bool:
        return self._system.mailbox(path).try_send(envelope)

    def spawn(self, coro: Coroutine[Any, Any, object], *, name: str = "") -> asyncio.Task[object]:
        """Run ``coro`` as a task tied to this incarnation (cancelled on exit, failure crashes)."""
        return self._runner.spawn(coro, name=name)

    def quarantine(self, envelope: Envelope, reason: str) -> None:
        """Record a message as undeliverable. The runtime frees its credit afterwards."""
        self._runner.quarantine(envelope, reason)

    def audit(self, action: str, data: Mapping[str, JsonValue] | None = None) -> None:
        self._system.audit.append(self.path, action, data)


class Actor(ABC):
    """Base class for behaviors. Subclasses set ``accepts`` to the message kinds they handle."""

    accepts: frozenset[str] | None = None

    async def on_start(self, ctx: ActorContext) -> None:
        """Called once per incarnation before the first message. Raising fails the start."""
        return None

    @abstractmethod
    async def receive(self, envelope: Envelope, ctx: ActorContext) -> None:
        """Handle one message. Raise :class:`MessageFailed` to reject only this message."""

    async def on_stop(self, reason: ExitReason, ctx: ActorContext) -> None:
        """Called once per incarnation after the loop ends, whatever the reason."""
        return None


class ActorRunner:
    """One incarnation of the actor in a slot. Created by the supervisor for every (re)start."""

    def __init__(
        self,
        *,
        path: str,
        factory: Callable[[], Actor],
        mailbox: Mailbox,
        system: SystemAPI,
        incarnation: int,
        max_attempts: int = 3,
        start_timeout: float = 5.0,
        stop_timeout: float = 5.0,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        self.path = path
        self.incarnation = incarnation
        self.started: asyncio.Event = asyncio.Event()
        self.inflight: dict[str, Envelope] = {}
        self.processed_total = 0
        self.exit_reason: ExitReason | None = None
        self._factory = factory
        self._mailbox = mailbox
        self._system = system
        self._max_attempts = max_attempts
        self._start_timeout = start_timeout
        self._stop_timeout = stop_timeout
        self._task: asyncio.Task[ExitReason] | None = None
        self._spawned: set[asyncio.Task[object]] = set()
        self._exit_override: ExitReason | None = None
        self._exiting = False
        self._monitors: list[MonitorCallback] = []

    # --- control -------------------------------------------------------------------------

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done() and not self._exiting

    def monitor(self, callback: MonitorCallback) -> None:
        """Register a callback invoked once with (path, incarnation, reason) when this exits."""
        self._monitors.append(callback)

    def stop(self) -> None:
        """Ask the incarnation to exit with reason ``shutdown``."""
        self._cancel(None)

    def kill(self, reason: ExitReason) -> None:
        """Force an exit with an explicit reason (fault injection, links)."""
        self._cancel(reason)

    def _cancel(self, reason: ExitReason | None) -> None:
        if self._task is None or self._task.done() or self._exiting:
            return
        if reason is not None and self._exit_override is None:
            self._exit_override = reason
        self._task.cancel()

    def spawn(self, coro: Coroutine[Any, Any, object], *, name: str = "") -> asyncio.Task[object]:
        task = asyncio.create_task(coro, name=name or f"{self.path}#{self.incarnation}")
        self._spawned.add(task)
        task.add_done_callback(self._spawned_done)
        return task

    def _spawned_done(self, task: asyncio.Task[object]) -> None:
        self._spawned.discard(task)
        if task.cancelled() or self._exiting:
            return
        error = task.exception()
        if error is not None:
            self._cancel(ExitReason.failed(error, detail="spawned_task_failed"))

    # --- the loop ------------------------------------------------------------------------

    async def run(self) -> ExitReason:
        """Run the incarnation to completion and return why it ended.

        Only ``KeyboardInterrupt`` leaves this method, and only after the exit path has run: an
        operator's interrupt must still reach the process. Every other failure -- including
        ``SystemExit`` from a third-party tool handler that calls ``sys.exit()`` -- becomes an
        :class:`ExitReason` the supervisor decides on, because a failure that skipped the exit
        path would strand the in-flight message and its mailbox credit.
        """
        self._task = asyncio.current_task()
        actor: Actor | None = None
        ctx = ActorContext(self, self._system)
        reason = ExitReason.shutdown("never_started")
        try:
            actor = self._factory()
            async with asyncio.timeout(self._start_timeout):
                await actor.on_start(ctx)
            self.started.set()
            await self._loop(actor, ctx)
            reason = ExitReason.normal()
        except ActorExit:
            reason = ExitReason.normal()
        except asyncio.CancelledError:
            if self._task is not None:
                self._task.uncancel()
            reason = self._exit_override or ExitReason.shutdown()
        except Exception as exc:  # noqa: BLE001 - every failure becomes an exit reason
            reason = self._exit_override or ExitReason.failed(exc)
        except BaseException as exc:  # SystemExit and friends are contained, not propagated
            reason = self._exit_override or ExitReason.failed(exc)
            if isinstance(exc, KeyboardInterrupt):
                raise  # after the exit path below: an operator's interrupt still reaches the loop
        finally:
            reason = await self._exit(actor, ctx, reason)
        return reason

    async def _exit(self, actor: Actor | None, ctx: ActorContext, reason: ExitReason) -> ExitReason:
        """Release the incarnation's resources and announce the exit. Runs on every exit path."""
        self._exiting = True
        self.started.set()
        await self._cancel_spawned()
        self._release_inflight(reason)
        if actor is not None:
            await self._call_on_stop(actor, reason, ctx)
        self.exit_reason = reason
        for callback in self._monitors:
            callback(self.path, self.incarnation, reason)
        self._system.on_actor_exit(self.path, self.incarnation, reason)
        return reason

    async def _loop(self, actor: Actor, ctx: ActorContext) -> None:
        while True:
            envelope = await self._mailbox.receive()
            self.inflight[envelope.id] = envelope
            await self._handle(actor, ctx, envelope)

    async def _handle(self, actor: Actor, ctx: ActorContext, envelope: Envelope) -> None:
        if actor.accepts is not None and envelope.kind not in actor.accepts:
            self._finish(envelope)
            self.quarantine(envelope, "unexpected_kind")
            return
        if envelope.expired(self._system.clock()):
            self._finish(envelope)
            self.quarantine(envelope, "deadline_exceeded")
            return
        attributes: dict[str, str | int] = {
            "actor.path": self.path,
            "actor.incarnation": self.incarnation,
            "message.id": envelope.id,
            "message.kind": envelope.kind,
            "message.attempt": envelope.attempt,
        }
        with self._system.tracer.start_as_current_span(
            "actor.receive", context=envelope.otel_context(), attributes=attributes
        ):
            try:
                await actor.receive(envelope, ctx)
            except MessageFailed as failed:
                self._nack(envelope, failed.reason, retryable=failed.retryable)
                return
        self._finish(envelope)
        self.processed_total += 1
        self._system.metrics.messages_processed_total.labels(self.path).inc()

    def _finish(self, envelope: Envelope) -> None:
        self.inflight.pop(envelope.id, None)
        self._mailbox.ack()

    def quarantine(self, envelope: Envelope, reason: str) -> None:
        try:
            self._system.quarantine.add(envelope, reason=reason, actor=self.path)
        except QuarantineFullError:
            # Counted as lost: the delivery bench asserts this counter stays at zero.
            self._system.metrics.quarantine_overflow_total.inc()
            _log.error("quarantine_overflow", actor=self.path, message_id=envelope.id)
            return
        self._system.metrics.quarantine_total.labels(reason).inc()
        self._system.audit.append(
            self.path,
            "message.quarantined",
            {"message_id": envelope.id, "reason": reason, "attempt": envelope.attempt},
        )

    def _nack(self, envelope: Envelope, reason: str, *, retryable: bool = True) -> None:
        """The behavior rejected the message: redeliver later or quarantine after K attempts."""
        self.inflight.pop(envelope.id, None)
        self._system.metrics.message_failures_total.labels(self.path, reason).inc()
        if not retryable or envelope.attempt >= self._max_attempts:
            self.quarantine(envelope, reason)
            self._mailbox.release_inflight()
            return
        self._mailbox.requeue(envelope.next_attempt())
        self._system.metrics.redeliveries_total.labels(self.path).inc()

    def _release_inflight(self, reason: ExitReason) -> None:
        """Hand unacknowledged messages back to the mailbox for the next incarnation."""
        pending = list(self.inflight.values())
        self.inflight.clear()
        # Front of the queue, in original order: the last one pushed ends up first.
        for envelope in reversed(pending):
            if not reason.abnormal:
                self._mailbox.push_front(envelope)
                continue
            if envelope.attempt >= self._max_attempts:
                self.quarantine(envelope, f"actor_{reason.kind}")
                self._mailbox.release_inflight()
                continue
            self._mailbox.push_front(envelope.next_attempt())
            self._system.metrics.redeliveries_total.labels(self.path).inc()

    async def _cancel_spawned(self) -> None:
        tasks = list(self._spawned)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _call_on_stop(self, actor: Actor, reason: ExitReason, ctx: ActorContext) -> None:
        try:
            async with asyncio.timeout(self._stop_timeout):
                await actor.on_stop(reason, ctx)
        except asyncio.CancelledError:
            if self._task is not None:
                self._task.uncancel()
        except BaseException as exc:  # a failing hook must not mask the exit reason
            _log.warning("on_stop_failed", actor=self.path, error_type=type(exc).__name__)
            if isinstance(exc, KeyboardInterrupt):
                raise
