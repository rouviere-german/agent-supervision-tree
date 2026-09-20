"""A pure reference model of OTP supervisor semantics.

This module has no asyncio, no tasks and no mailboxes: it is a state machine over child specs
that answers one question, *given this exit, what does a supervisor do?* The operational
:class:`~agent_supervision_tree.runtime.supervisor.Supervisor` does not import it. Instead the
model-based test in ``tests/unit/test_supervisor_reference.py`` drives both with the same
random exit sequences and checks that the observable outcome (which children are alive, how
many times each was started, whether the supervisor escalated) is identical.

The rules follow the Erlang/OTP ``supervisor`` behaviour (OTP design principles, "Supervisor
Behaviour"):

- ``permanent`` children are always restarted; ``transient`` only after an abnormal exit;
  ``temporary`` never, and their spec is deleted when they exit.
- ``one_for_one`` restarts only the exited child. ``one_for_all`` terminates every other
  running child (in reverse start order) and restarts all remaining specs in start order.
  ``rest_for_one`` does the same for the specs after the exited child in start order.
- Restart intensity: every restart decision is stamped; if more than ``max_restarts`` stamps
  fall within the last ``max_seconds``, the supervisor terminates all children and exits with
  ``shutdown``, which its own parent treats as an abnormal exit (escalation).
- A running ``temporary`` sibling caught in a one_for_all or rest_for_one restart is
  terminated and, since its spec is deleted, not restarted.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

__all__ = [
    "Decision",
    "ExitClass",
    "ReferenceChild",
    "ReferenceSupervisor",
    "RestartType",
    "Strategy",
    "should_restart",
]

RestartType = Literal["permanent", "transient", "temporary"]
Strategy = Literal["one_for_one", "one_for_all", "rest_for_one"]
ExitClass = Literal["normal", "shutdown", "abnormal"]


def should_restart(restart: RestartType, exit_class: ExitClass) -> bool:
    """OTP ``do_restart/4``: permanent always, transient on abnormal exits, temporary never."""
    if restart == "permanent":
        return True
    if restart == "temporary":
        return False
    return exit_class == "abnormal"


@dataclass
class ReferenceChild:
    id: str
    restart: RestartType
    running: bool = True
    starts: int = 1


@dataclass(frozen=True)
class Decision:
    """What the supervisor did in response to one child exit."""

    child: str
    exit_class: ExitClass
    restarted: tuple[str, ...] = ()
    terminated: tuple[str, ...] = ()
    removed: tuple[str, ...] = ()
    escalated: bool = False
    ignored: bool = False


@dataclass
class ReferenceSupervisor:
    strategy: Strategy
    max_restarts: int
    max_seconds: float
    children: list[ReferenceChild]
    alive: bool = True
    restart_stamps: list[float] = field(default_factory=list)
    decisions: list[Decision] = field(default_factory=list)

    @classmethod
    def build(
        cls,
        strategy: Strategy,
        max_restarts: int,
        max_seconds: float,
        specs: Sequence[tuple[str, RestartType]],
    ) -> ReferenceSupervisor:
        return cls(
            strategy, max_restarts, max_seconds, [ReferenceChild(cid, r) for cid, r in specs]
        )

    # --- observations --------------------------------------------------------------------

    def running_ids(self) -> tuple[str, ...]:
        return tuple(c.id for c in self.children if c.running)

    def starts(self) -> dict[str, int]:
        return {c.id: c.starts for c in self.children}

    def _find(self, child_id: str) -> ReferenceChild | None:
        return next((c for c in self.children if c.id == child_id), None)

    # --- the model -----------------------------------------------------------------------

    def child_exit(self, child_id: str, exit_class: ExitClass, now: float) -> Decision:
        decision = self._decide(child_id, exit_class, now)
        self.decisions.append(decision)
        return decision

    def _decide(self, child_id: str, exit_class: ExitClass, now: float) -> Decision:
        child = self._find(child_id)
        if not self.alive or child is None or not child.running:
            return Decision(child_id, exit_class, ignored=True)
        child.running = False
        if not should_restart(child.restart, exit_class):
            removed: tuple[str, ...] = ()
            if child.restart == "temporary":
                self.children.remove(child)
                removed = (child.id,)
            return Decision(child_id, exit_class, removed=removed)
        if not self._add_restart(now):
            return self._escalate(child_id, exit_class)
        return self._restart(child, exit_class)

    def _add_restart(self, now: float) -> bool:
        """OTP ``add_restart/1``: stamp, drop stamps outside the window, compare the count."""
        self.restart_stamps.append(now)
        self.restart_stamps = [t for t in self.restart_stamps if now - t <= self.max_seconds]
        return len(self.restart_stamps) <= self.max_restarts

    def _escalate(self, child_id: str, exit_class: ExitClass) -> Decision:
        terminated = tuple(c.id for c in reversed(self.children) if c.running)
        for child in self.children:
            child.running = False
        self.alive = False
        return Decision(child_id, exit_class, terminated=terminated, escalated=True)

    def _restart(self, child: ReferenceChild, exit_class: ExitClass) -> Decision:
        index = self.children.index(child)
        if self.strategy == "one_for_one":
            affected = [child]
        elif self.strategy == "one_for_all":
            affected = list(self.children)
        else:
            affected = self.children[index:]
        terminated = tuple(c.id for c in reversed(affected) if c.running)
        removed = tuple(c.id for c in affected if c.running and c.restart == "temporary")
        for stopped in affected:
            if stopped.running and stopped.restart == "temporary":
                self.children.remove(stopped)
            stopped.running = False
        restarted = tuple(c.id for c in affected if c.id not in removed)
        for started in affected:
            if started.id in restarted:
                started.running = True
                started.starts += 1
        return Decision(
            child.id, exit_class, restarted=restarted, terminated=terminated, removed=removed
        )
