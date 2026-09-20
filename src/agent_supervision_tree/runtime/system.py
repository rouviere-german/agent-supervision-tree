"""The actor system: owns the root supervisor, the mailboxes, links, monitors and shutdown.

Mailboxes are registered by path when the system is built from the root spec and created
lazily, so a mailbox exists before its actor starts and survives every restart. The tree is
static: every actor path is known up front, which is what makes introspection, ledger nodes
and least-privilege tool allowlists checkable at build time.

Stopping is graceful and bounded: :meth:`ActorSystem.stop` asks the root to drain and stop its
children in reverse start order within the deadline, then awaits the root task. When the root
exits on its own (restart intensity exceeded at the top of the tree) the system is stopped and
``exit_code`` is 70 (``EX_SOFTWARE``), so a process wrapper can restart the whole thing.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.trace import Tracer
from prometheus_client import CollectorRegistry

from agent_supervision_tree.core.logging import get_logger
from agent_supervision_tree.core.tracing import build_tracer_provider
from agent_supervision_tree.runtime.actor import ActorRunner, ExitReason, MonitorCallback
from agent_supervision_tree.runtime.audit import AuditChain
from agent_supervision_tree.runtime.errors import ActorNotRunningError, ActorSystemStoppedError
from agent_supervision_tree.runtime.mailbox import Mailbox
from agent_supervision_tree.runtime.metrics import RuntimeMetrics, create_runtime_metrics
from agent_supervision_tree.runtime.quarantine import Quarantine
from agent_supervision_tree.runtime.supervisor import (
    ActorSpec,
    Supervisor,
    SupervisorSpec,
    TreeNode,
)

__all__ = ["EXIT_SOFTWARE", "ActorSystem"]

EXIT_SOFTWARE = 70
_log = get_logger("agent_supervision_tree.system")


class ActorSystem:
    def __init__(
        self,
        root_spec: SupervisorSpec,
        *,
        root_id: str = "root",
        registry: CollectorRegistry | None = None,
        audit_path: Path | None = None,
        quarantine_max_records: int = 10_000,
        max_message_bytes: int = 65_536,
        tracer_provider: TracerProvider | None = None,
        clock: Callable[[], float] | None = None,
        lag_interval: float = 0.25,
    ) -> None:
        self.root_spec = root_spec
        self.root_path = root_id
        self.metrics: RuntimeMetrics = create_runtime_metrics(registry)
        self.audit: AuditChain = AuditChain(audit_path)
        self.quarantine: Quarantine = Quarantine(quarantine_max_records)
        provider = tracer_provider or build_tracer_provider("agent-supervision-tree")
        self.tracer: Tracer = provider.get_tracer("agent_supervision_tree.runtime")
        self.max_message_bytes = max_message_bytes
        self._clock = clock
        self._lag_interval = lag_interval
        self._capacities: dict[str, int] = {}
        self._mailboxes: dict[str, Mailbox] = {}
        self._links: dict[str, set[str]] = defaultdict(set)
        self._monitors: dict[str, list[MonitorCallback]] = defaultdict(list)
        self.exits: list[tuple[str, int, ExitReason]] = []
        self.root: Supervisor | None = None
        self._root_task: asyncio.Task[ExitReason] | None = None
        self._lag_task: asyncio.Task[None] | None = None
        self._baseline_tasks = 0
        self.loop_lag_seconds = 0.0
        self.exit_code: int | None = None
        self._register(root_id, root_spec)

    def _register(self, path: str, spec: SupervisorSpec) -> None:
        for child in spec.children:
            child_path = f"{path}/{child.id}"
            if isinstance(child.spec, ActorSpec):
                self._capacities[child_path] = child.spec.mailbox_capacity
            else:
                self._register(child_path, child.spec)

    # --- SystemAPI -----------------------------------------------------------------------

    def clock(self) -> float:
        if self._clock is not None:
            return self._clock()
        return asyncio.get_running_loop().time()

    def mailbox(self, path: str) -> Mailbox:
        mailbox = self._mailboxes.get(path)
        if mailbox is None:
            capacity = self._capacities.get(path)
            if capacity is None:
                raise ActorNotRunningError(f"{path} is not an actor of this system")
            mailbox = Mailbox(path, capacity, max_message_bytes=self.max_message_bytes)
            self._mailboxes[path] = mailbox
        return mailbox

    def mailboxes(self) -> dict[str, Mailbox]:
        return dict(self._mailboxes)

    def actor_paths(self) -> list[str]:
        return list(self._capacities)

    def on_actor_exit(self, path: str, incarnation: int, reason: ExitReason) -> None:
        self.exits.append((path, incarnation, reason))
        for callback in list(self._monitors.get(path, ())):
            callback(path, incarnation, reason)
        if reason.abnormal:
            for linked in list(self._links.get(path, ())):
                self._kill_quietly(linked, ExitReason.killed(f"linked_exit:{path}"))

    def _kill_quietly(self, path: str, reason: ExitReason) -> None:
        try:
            self.resolve(path).kill(reason)
        except ActorNotRunningError:
            return

    # --- links and monitors --------------------------------------------------------------

    def link(self, first: str, second: str) -> None:
        """Bidirectional: an abnormal exit of either kills the other, which is then restarted."""
        self._links[first].add(second)
        self._links[second].add(first)

    def monitor(self, path: str, callback: MonitorCallback) -> None:
        """Be told of every exit at ``path``, across incarnations."""
        self._monitors[path].append(callback)

    # --- lifecycle -----------------------------------------------------------------------

    @property
    def running(self) -> bool:
        return self._root_task is not None and not self._root_task.done()

    async def start(self) -> None:
        if self.root is not None:
            raise RuntimeError("the system was already started")
        self._baseline_tasks = len(asyncio.all_tasks())
        self.root = Supervisor(self.root_spec, path=self.root_path, system=self)
        self._root_task = asyncio.create_task(self.root.run(), name=self.root_path)
        self._root_task.add_done_callback(self._root_done)
        self._lag_task = asyncio.create_task(self._measure_lag(), name="loop-lag")
        await self.root.started.wait()
        self.audit.append("system", "system.started", {"root": self.root_path})

    def _root_done(self, task: asyncio.Task[ExitReason]) -> None:
        # Nothing is supervised any more: the lag sampler must not outlive the tree, so the
        # process' task count returns to its baseline even when the root died on its own.
        if self._lag_task is not None:
            self._lag_task.cancel()
        reason = task.result() if not task.cancelled() and task.exception() is None else None
        if reason is not None and reason.kind == "shutdown":
            self.exit_code = 0
        else:
            self.exit_code = EXIT_SOFTWARE
            _log.error("root_supervisor_failed", reason=reason.label() if reason else "unknown")
        self.audit.append(
            "system", "system.stopped", {"exit_code": self.exit_code, "reason": str(reason)}
        )
        self.audit.flush()

    async def stop(self, timeout: float = 10.0) -> int:
        """Drain and stop the tree within ``timeout`` seconds; returns the exit code."""
        if self.root is None or self._root_task is None:
            raise ActorSystemStoppedError("the system is not running")
        if not self._root_task.done():
            await self.root.shutdown(self.clock() + timeout)
            await self._root_task
        if self._lag_task is not None:
            self._lag_task.cancel()
            await asyncio.gather(self._lag_task, return_exceptions=True)
            self._lag_task = None
        self.audit.close()
        return self.exit_code if self.exit_code is not None else EXIT_SOFTWARE

    async def wait_stopped(self) -> int:
        """Wait for the root to exit on its own (escalation) and return the exit code."""
        if self._root_task is None:
            raise ActorSystemStoppedError("the system is not running")
        await self._root_task
        return await self.stop()

    async def _measure_lag(self) -> None:
        while True:
            expected = self.clock() + self._lag_interval
            await asyncio.sleep(self._lag_interval)
            self.loop_lag_seconds = max(0.0, self.clock() - expected)
            self.metrics.event_loop_lag_seconds.set(self.loop_lag_seconds)

    # --- introspection and fault injection -----------------------------------------------

    def resolve(self, path: str) -> ActorRunner | Supervisor:
        if self.root is None or not self.root.running:
            raise ActorNotRunningError("the root supervisor is not running")
        if path == self.root_path:
            return self.root
        prefix = f"{self.root_path}/"
        if not path.startswith(prefix):
            raise ActorNotRunningError(f"{path} is outside {self.root_path}")
        return self.root.resolve(path[len(prefix) :])

    def kill(self, path: str, reason: ExitReason) -> None:
        """Fault injection: end the incarnation at ``path`` with ``reason``."""
        self.resolve(path).kill(reason)
        self.audit.append("system", "actor.killed", {"path": path, "reason": reason.label()})

    def task_count(self) -> int:
        return len(asyncio.all_tasks())

    @property
    def baseline_tasks(self) -> int:
        return self._baseline_tasks

    def refresh_gauges(self) -> None:
        for path, mailbox in self._mailboxes.items():
            self.metrics.mailbox_depth.labels(path).set(mailbox.depth)
            self.metrics.mailbox_credits.labels(path).set(mailbox.credits)
        self.metrics.quarantine_size.set(len(self.quarantine))
        self.metrics.tasks_active.set(self.task_count())

    def tree(self) -> TreeNode:
        self.refresh_gauges()
        node: TreeNode = self.root.tree() if self.root is not None else {"path": self.root_path}
        node["quarantine"] = {"size": len(self.quarantine), "reasons": self.quarantine.reasons()}
        node["loop_lag_ms"] = round(self.loop_lag_seconds * 1000, 3)
        node["tasks"] = self.task_count()
        node["exit_code"] = self.exit_code
        return node
