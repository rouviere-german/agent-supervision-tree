"""Supervisors: child specs, restart strategies, restart intensity and escalation.

A supervisor owns an ordered list of child *slots*. Each slot has a spec (an actor factory or a
nested supervisor spec), a restart type and, for actors, a mailbox that outlives incarnations.
Child exits arrive on an internal queue and are handled one at a time, like the ``EXIT``
messages of an OTP supervisor, so decisions never interleave.

The rules mirror the OTP supervisor behaviour (see :mod:`agent_supervision_tree.runtime.reference`
for the pure model these are tested against), with one documented difference: a supervisor
that exceeds its restart intensity exits with kind ``error`` (detail ``max_restart_intensity``)
rather than OTP's literal ``shutdown``, so the parent treats the escalation as a failure and
restarts a transient sub-supervisor as well as a permanent one.

Graceful shutdown (:meth:`Supervisor.shutdown`) closes each actor mailbox to new sends, waits
for it to drain within the deadline, then stops the incarnation, in reverse start order and
recursively for nested supervisors.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from typing import Any

from pydantic import JsonValue

from agent_supervision_tree.core.logging import get_logger
from agent_supervision_tree.runtime.actor import (
    Actor,
    ActorRunner,
    ExitReason,
    MonitorCallback,
    SystemAPI,
)
from agent_supervision_tree.runtime.errors import ActorNotRunningError
from agent_supervision_tree.runtime.reference import Decision, ExitClass, RestartType, Strategy

__all__ = ["ActorSpec", "ChildSpec", "Supervisor", "SupervisorSpec", "TreeNode"]

_CHILD_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_log = get_logger("agent_supervision_tree.supervisor")
TreeNode = dict[str, Any]


@dataclass(frozen=True, slots=True)
class ActorSpec:
    """How to start an actor child. The mailbox is created once and kept across restarts."""

    factory: Callable[[], Actor]
    mailbox_capacity: int = 64
    max_attempts: int = 3
    start_timeout: float = 5.0
    stop_timeout: float = 5.0


@dataclass(frozen=True, slots=True)
class SupervisorSpec:
    strategy: Strategy
    children: tuple[ChildSpec, ...]
    max_restarts: int = 3
    max_seconds: float = 5.0

    def __post_init__(self) -> None:
        if self.max_restarts < 0 or self.max_seconds <= 0:
            raise ValueError("max_restarts must be >= 0 and max_seconds > 0")
        ids = [child.id for child in self.children]
        if len(set(ids)) != len(ids):
            raise ValueError("child ids must be unique")


@dataclass(frozen=True, slots=True)
class ChildSpec:
    id: str
    spec: ActorSpec | SupervisorSpec
    restart: RestartType = "permanent"
    shutdown_timeout: float = 5.0

    def __post_init__(self) -> None:
        if not _CHILD_ID.fullmatch(self.id):
            raise ValueError(f"invalid child id {self.id!r}")


class _Escalate(Exception):  # noqa: N818 - internal control flow
    """Restart intensity exceeded: terminate everything and exit."""


@dataclass
class _Slot:
    spec: ChildSpec
    path: str
    incarnation: int = 0
    runnable: ActorRunner | Supervisor | None = None
    task: asyncio.Task[ExitReason] | None = None
    expected_exit: bool = False
    running: bool = False
    exits: list[ExitReason] = field(default_factory=list)


class Supervisor:
    def __init__(
        self,
        spec: SupervisorSpec,
        *,
        path: str,
        system: SystemAPI,
        incarnation: int = 1,
    ) -> None:
        self.spec = spec
        self.path = path
        self.incarnation = incarnation
        self.started: asyncio.Event = asyncio.Event()
        self.decisions: list[Decision] = []
        self.decision_events: asyncio.Queue[Decision] = asyncio.Queue()
        self.exit_reason: ExitReason | None = None
        self.restart_stamps: list[float] = []
        self._system = system
        self._slots: dict[str, _Slot] = {
            child.id: _Slot(child, f"{path}/{child.id}") for child in spec.children
        }
        self._order: list[str] = [child.id for child in spec.children]
        self._events: asyncio.Queue[tuple[str, int, ExitReason]] = asyncio.Queue()
        self._task: asyncio.Task[ExitReason] | None = None
        self._exit_override: ExitReason | None = None
        self._stopping = False
        self._exiting = False
        self._monitors: list[MonitorCallback] = []

    # --- control -------------------------------------------------------------------------

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done() and not self._exiting

    @property
    def alive(self) -> bool:
        return self.running and not self._stopping

    def monitor(self, callback: MonitorCallback) -> None:
        self._monitors.append(callback)

    def stop(self) -> None:
        self._cancel(None)

    def kill(self, reason: ExitReason) -> None:
        self._cancel(reason)

    def _cancel(self, reason: ExitReason | None) -> None:
        if self._task is None or self._task.done() or self._exiting:
            return
        if reason is not None and self._exit_override is None:
            self._exit_override = reason
        self._task.cancel()

    # --- lookups -------------------------------------------------------------------------

    def child(self, child_id: str) -> ActorRunner | Supervisor:
        slot = self._slots.get(child_id)
        if slot is None or slot.runnable is None or not slot.running:
            raise ActorNotRunningError(f"{self.path}/{child_id} is not running")
        return slot.runnable

    def resolve(self, relative_path: str) -> ActorRunner | Supervisor:
        """Find the running incarnation at ``child/grandchild/...`` below this supervisor."""
        head, _, rest = relative_path.partition("/")
        found = self.child(head)
        if not rest:
            return found
        if isinstance(found, Supervisor):
            return found.resolve(rest)
        raise ActorNotRunningError(f"{found.path} is an actor and has no child {rest!r}")

    def running_ids(self) -> tuple[str, ...]:
        return tuple(cid for cid in self._order if self._slots[cid].running)

    def starts(self) -> dict[str, int]:
        return {cid: self._slots[cid].incarnation for cid in self._order}

    def restarts_in_window(self) -> int:
        now = self._system.clock()
        return sum(1 for stamp in self.restart_stamps if now - stamp <= self.spec.max_seconds)

    def tree(self) -> TreeNode:
        children: list[TreeNode] = []
        for cid in self._order:
            slot = self._slots[cid]
            runnable = slot.runnable
            if isinstance(runnable, Supervisor) and slot.running:
                node = runnable.tree()
            else:
                node = self._actor_node(slot)
            node["restart"] = slot.spec.restart
            children.append(node)
        return {
            "path": self.path,
            "kind": "supervisor",
            "running": self.running,
            "incarnation": self.incarnation,
            "strategy": self.spec.strategy,
            "max_restarts": self.spec.max_restarts,
            "max_seconds": self.spec.max_seconds,
            "restarts_in_window": self.restarts_in_window(),
            "children": children,
        }

    def _actor_node(self, slot: _Slot) -> TreeNode:
        node: TreeNode = {
            "path": slot.path,
            "kind": "supervisor" if isinstance(slot.spec.spec, SupervisorSpec) else "actor",
            "running": slot.running,
            "incarnation": slot.incarnation,
            "exits": [reason.label() for reason in slot.exits[-5:]],
        }
        if isinstance(slot.spec.spec, ActorSpec):
            mailbox = self._system.mailbox(slot.path)
            node["mailbox"] = {
                "depth": mailbox.depth,
                "inflight": mailbox.inflight,
                "credits": mailbox.credits,
                "capacity": mailbox.capacity,
                "closed": mailbox.closed,
            }
            runner = slot.runnable
            if isinstance(runner, ActorRunner):
                node["processed"] = runner.processed_total
        return node

    # --- lifecycle -----------------------------------------------------------------------

    async def run(self) -> ExitReason:
        self._task = asyncio.current_task()
        reason: ExitReason
        try:
            for cid in self._order:
                await self._start_child(self._slots[cid])
            self.started.set()
            while True:
                event = await self._events.get()
                await self._handle_exit(*event)
        except _Escalate:
            reason = ExitReason("error", "max_restart_intensity")
        except asyncio.CancelledError:
            if self._task is not None:
                self._task.uncancel()
            reason = self._exit_override or ExitReason.shutdown()
        except Exception as exc:  # noqa: BLE001 - a supervisor bug is still an exit reason
            reason = ExitReason.failed(exc)
        self._exiting = True
        self.started.set()
        await self._terminate(self._running_slots_reversed())
        self.exit_reason = reason
        for callback in self._monitors:
            callback(self.path, self.incarnation, reason)
        self._system.on_actor_exit(self.path, self.incarnation, reason)
        return reason

    async def shutdown(self, deadline: float) -> None:
        """Drain and stop children in reverse start order, then exit with ``shutdown``."""
        self._stopping = True
        for slot in self._running_slots_reversed():
            runnable = slot.runnable
            if isinstance(runnable, Supervisor):
                await runnable.shutdown(deadline)
            elif isinstance(runnable, ActorRunner):
                mailbox = self._system.mailbox(slot.path)
                mailbox.close()
                try:
                    async with asyncio.timeout_at(deadline):
                        await mailbox.wait_idle()
                except TimeoutError:
                    _log.warning("drain_timeout", actor=slot.path, depth=mailbox.depth)
            await self._terminate([slot])
        self.stop()

    def _running_slots_reversed(self) -> list[_Slot]:
        return [self._slots[cid] for cid in reversed(self._order) if self._slots[cid].running]

    async def _start_child(self, slot: _Slot) -> None:
        slot.incarnation += 1
        child = slot.spec
        runnable: ActorRunner | Supervisor
        if isinstance(child.spec, ActorSpec):
            spec = child.spec
            runnable = ActorRunner(
                path=slot.path,
                factory=spec.factory,
                mailbox=self._system.mailbox(slot.path),
                system=self._system,
                incarnation=slot.incarnation,
                max_attempts=spec.max_attempts,
                start_timeout=spec.start_timeout,
                stop_timeout=spec.stop_timeout,
            )
        else:
            runnable = Supervisor(
                child.spec, path=slot.path, system=self._system, incarnation=slot.incarnation
            )
        slot.runnable = runnable
        slot.expected_exit = False
        slot.running = True
        slot.task = asyncio.create_task(runnable.run(), name=slot.path)
        slot.task.add_done_callback(partial(self._child_done, slot, slot.incarnation))
        await runnable.started.wait()

    def _child_done(self, slot: _Slot, incarnation: int, task: asyncio.Task[ExitReason]) -> None:
        if task.cancelled():
            reason = ExitReason.shutdown("cancelled_before_start")
        elif task.exception() is not None:
            reason = ExitReason.failed(task.exception() or RuntimeError("unknown"))
        else:
            reason = task.result()
        slot.exits.append(reason)
        self._events.put_nowait((slot.spec.id, incarnation, reason))

    async def _terminate(self, slots: list[_Slot]) -> None:
        for slot in slots:
            runnable, task = slot.runnable, slot.task
            if runnable is None or task is None or task.done():
                slot.running = False
                continue
            slot.expected_exit = True
            runnable.stop()
            done, _ = await asyncio.wait({task}, timeout=slot.spec.shutdown_timeout)
            if not done:
                runnable.kill(ExitReason.killed("shutdown_timeout"))
                await asyncio.wait({task})
            slot.running = False

    # --- decisions -----------------------------------------------------------------------

    async def _handle_exit(self, child_id: str, incarnation: int, reason: ExitReason) -> None:
        slot = self._slots.get(child_id)
        if (
            slot is None
            or incarnation != slot.incarnation
            or slot.expected_exit
            or self._stopping
            or not slot.running
        ):
            return
        slot.running = False
        slot.task = None
        exit_class = _classify(reason)
        self._system.metrics.child_exits_total.labels(slot.path, reason.label()).inc()
        if not _should_restart(slot.spec.restart, exit_class):
            removed: tuple[str, ...] = ()
            if slot.spec.restart == "temporary":
                self._remove(slot)
                removed = (child_id,)
            self._record(Decision(child_id, exit_class, removed=removed))
            return
        now = self._system.clock()
        if not self._add_restart(now):
            terminated = tuple(s.spec.id for s in self._running_slots_reversed())
            await self._terminate(self._running_slots_reversed())
            self._system.metrics.escalations_total.labels(self.path).inc()
            self._system.audit.append(
                self.path,
                "supervisor.escalated",
                {
                    "child": child_id,
                    "reason": reason.label(),
                    "restarts_in_window": len(self.restart_stamps),
                    "max_restarts": self.spec.max_restarts,
                },
            )
            self._record(Decision(child_id, exit_class, terminated=terminated, escalated=True))
            raise _Escalate
        await self._restart(slot, exit_class, reason)

    def _add_restart(self, now: float) -> bool:
        self.restart_stamps.append(now)
        self.restart_stamps = [t for t in self.restart_stamps if now - t <= self.spec.max_seconds]
        return len(self.restart_stamps) <= self.spec.max_restarts

    def _affected(self, slot: _Slot) -> list[_Slot]:
        if self.spec.strategy == "one_for_one":
            return [slot]
        if self.spec.strategy == "one_for_all":
            return [self._slots[cid] for cid in self._order]
        index = self._order.index(slot.spec.id)
        return [self._slots[cid] for cid in self._order[index:]]

    async def _restart(self, slot: _Slot, exit_class: ExitClass, reason: ExitReason) -> None:
        began = self._system.clock()
        affected = self._affected(slot)
        to_terminate = [s for s in reversed(affected) if s.running]
        terminated = tuple(s.spec.id for s in to_terminate)
        await self._terminate(to_terminate)
        removed = tuple(s.spec.id for s in affected if s.spec.restart == "temporary")
        for gone in [s for s in affected if s.spec.restart == "temporary"]:
            self._remove(gone)
        restarted: list[str] = []
        for member in affected:
            if member.spec.id in removed:
                continue
            await self._start_child(member)
            restarted.append(member.spec.id)
            self._system.metrics.restarts_total.labels(member.path, reason.label()).inc()
        latency = self._system.clock() - began
        self._system.metrics.restart_latency_seconds.observe(latency)
        restarted_json: list[JsonValue] = list(restarted)
        terminated_json: list[JsonValue] = list(terminated)
        self._system.audit.append(
            self.path,
            "actor.restarted",
            {
                "child": slot.spec.id,
                "reason": reason.label(),
                "strategy": self.spec.strategy,
                "restarted": restarted_json,
                "terminated": terminated_json,
                "latency_ms": round(latency * 1000, 3),
            },
        )
        self._record(
            Decision(
                slot.spec.id,
                exit_class,
                restarted=tuple(restarted),
                terminated=terminated,
                removed=removed,
            )
        )

    def _remove(self, slot: _Slot) -> None:
        self._slots.pop(slot.spec.id, None)
        self._order.remove(slot.spec.id)

    def _record(self, decision: Decision) -> None:
        self.decisions.append(decision)
        self.decision_events.put_nowait(decision)


def _classify(reason: ExitReason) -> ExitClass:
    if reason.abnormal:
        return "abnormal"
    return "normal" if reason.kind == "normal" else "shutdown"


def _should_restart(restart: RestartType, exit_class: ExitClass) -> bool:
    if restart == "permanent":
        return True
    if restart == "temporary":
        return False
    return exit_class == "abnormal"
