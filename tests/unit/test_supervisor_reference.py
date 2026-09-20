"""Model-based test: the running supervisor must agree with the pure OTP reference model.

The reference model in :mod:`agent_supervision_tree.runtime.reference` has no asyncio and no
mailboxes; it is a small state machine over child specs. The stateful test below drives *both*
with the same randomly generated sequence of child exits and clock advances, and after every
step the running tree and the model must report the same running children, the same number of
starts, the same escalation state and the same decision. Anything the operational supervisor
does that OTP would not do shows up here as a diff.
"""

from __future__ import annotations

import asyncio

import pytest
from hypothesis import HealthCheck
from hypothesis import strategies as st
from hypothesis.stateful import (
    RuleBasedStateMachine,
    initialize,
    invariant,
    precondition,
    rule,
    run_state_machine_as_test,
)

from agent_supervision_tree.runtime.actor import ExitReason
from agent_supervision_tree.runtime.reference import (
    Decision,
    ExitClass,
    ReferenceSupervisor,
    RestartType,
    Strategy,
    should_restart,
)
from agent_supervision_tree.runtime.supervisor import Supervisor
from agent_supervision_tree.runtime.system import ActorSystem
from helpers import (
    CountingActor,
    FakeClock,
    actor_child,
    build_system,
    scaled_settings,
    wait_for,
)

pytestmark = pytest.mark.unit

STRATEGIES: tuple[Strategy, ...] = ("one_for_one", "one_for_all", "rest_for_one")
RESTARTS: tuple[RestartType, ...] = ("permanent", "transient", "temporary")
EXIT_CLASSES: tuple[ExitClass, ...] = ("abnormal", "normal", "shutdown")
WINDOW = 10.0
_EXIT_REASONS: dict[ExitClass, ExitReason] = {
    "abnormal": ExitReason.killed("chaos"),
    "normal": ExitReason.normal(),
    "shutdown": ExitReason.shutdown("asked"),
}


# --- the reference model on its own ----------------------------------------------------------


def test_restart_types_follow_the_otp_rules() -> None:
    assert should_restart("permanent", "normal")
    assert should_restart("permanent", "abnormal")
    assert should_restart("transient", "abnormal")
    assert not should_restart("transient", "normal")
    assert not should_restart("transient", "shutdown")
    assert not should_restart("temporary", "abnormal")


def test_the_model_ignores_exits_of_children_that_are_not_running() -> None:
    model = ReferenceSupervisor.build("one_for_one", 5, WINDOW, [("a", "permanent")])
    model.child_exit("a", "abnormal", 0.0)
    model.children[0].running = False
    ignored = model.child_exit("a", "abnormal", 0.0)
    assert ignored.ignored
    assert model.child_exit("missing", "abnormal", 0.0).ignored


def test_the_model_escalates_and_stays_dead() -> None:
    model = ReferenceSupervisor.build(
        "one_for_one", 1, WINDOW, [("a", "permanent"), ("b", "permanent")]
    )
    model.child_exit("a", "abnormal", 0.0)
    escalation = model.child_exit("a", "abnormal", 1.0)
    assert escalation.escalated
    assert escalation.terminated == ("b",)
    assert model.running_ids() == ()
    assert not model.alive
    assert model.child_exit("b", "abnormal", 2.0).ignored


def test_the_model_forgets_restarts_outside_the_window() -> None:
    model = ReferenceSupervisor.build("one_for_one", 1, WINDOW, [("a", "permanent")])
    for index in range(4):
        decision = model.child_exit("a", "abnormal", index * (WINDOW + 1))
        assert not decision.escalated
    assert model.starts() == {"a": 5}


def test_decisions_are_comparable_values() -> None:
    assert Decision("a", "abnormal", restarted=("a",)) == Decision(
        "a", "abnormal", restarted=("a",)
    )


# --- the running supervisor against the model ------------------------------------------------


class SupervisionMachine(RuleBasedStateMachine):
    """Drive a live tree and the reference model with the same exits; compare after each step."""

    def __init__(self) -> None:
        super().__init__()
        self.loop = asyncio.new_event_loop()
        self.clock = FakeClock(0.0)
        self.system: ActorSystem | None = None
        self.model: ReferenceSupervisor | None = None

    @initialize(
        strategy=st.sampled_from(STRATEGIES),
        restarts=st.lists(st.sampled_from(RESTARTS), min_size=1, max_size=4),
        max_restarts=st.integers(min_value=0, max_value=3),
    )
    def build(self, strategy: Strategy, restarts: list[RestartType], max_restarts: int) -> None:
        specs = [(f"c{index}", restart) for index, restart in enumerate(restarts)]
        children = [
            actor_child(child_id, CountingActor, restart=restart) for child_id, restart in specs
        ]
        self.system = build_system(
            children,
            strategy=strategy,
            max_restarts=max_restarts,
            max_seconds=WINDOW,
            clock=self.clock,
        )
        self.loop.run_until_complete(self.system.start())
        self.model = ReferenceSupervisor.build(strategy, max_restarts, WINDOW, specs)

    # --- access -----------------------------------------------------------------------

    @property
    def supervisor(self) -> Supervisor:
        assert self.system is not None
        assert self.system.root is not None
        return self.system.root

    @property
    def reference(self) -> ReferenceSupervisor:
        assert self.model is not None
        return self.model

    def _ready(self) -> bool:
        return (
            self.system is not None
            and self.supervisor.running
            and bool(self.supervisor.running_ids())
        )

    # --- rules ------------------------------------------------------------------------

    @precondition(lambda self: self._ready())
    @rule(pick=st.integers(min_value=0, max_value=9), exit_class=st.sampled_from(EXIT_CLASSES))
    def child_exits(self, pick: int, exit_class: ExitClass) -> None:
        running = self.supervisor.running_ids()
        child_id = running[pick % len(running)]
        expected = len(self.supervisor.decisions) + 1
        self.supervisor.child(child_id).kill(_EXIT_REASONS[exit_class])
        model_decision = self.reference.child_exit(child_id, exit_class, self.clock.now)
        settled = self.loop.run_until_complete(
            _wait(lambda: len(self.supervisor.decisions) >= expected or not self.supervisor.running)
        )
        assert settled, "the supervisor never acted on the child exit"
        assert self.supervisor.decisions[-1] == model_decision

    @rule(seconds=st.floats(min_value=0.0, max_value=2 * WINDOW, allow_nan=False))
    def time_passes(self, seconds: float) -> None:
        """Always available, so a dead tree can still make progress (and stay dead)."""
        self.clock.advance(seconds)

    # --- invariants -------------------------------------------------------------------

    @invariant()
    def the_tree_matches_the_model(self) -> None:
        if self.system is None or self.model is None:
            return
        assert self.supervisor.running_ids() == self.reference.running_ids()
        assert self.supervisor.starts() == self.reference.starts()
        assert self.supervisor.running == self.reference.alive
        acted_on = [item for item in self.reference.decisions if not item.ignored]
        assert len(self.supervisor.decisions) == len(acted_on)

    def teardown(self) -> None:
        if self.system is not None and self.system.root is not None:
            self.loop.run_until_complete(self.system.stop(timeout=2))
        self.loop.close()


async def _wait(predicate: object) -> bool:
    return await wait_for(predicate, timeout=2.0)  # type: ignore[arg-type]


# Depth comes from the hypothesis profile (dev: 60 examples x 15 steps, ci: 200 x 40), so CI
# searches deeper than a dev run instead of both running one hardcoded budget. Measured on this
# machine: 250 examples x 40 steps takes 4.5 s, so the flagship claim can afford the full budget.
MACHINE_SETTINGS = scaled_settings(suppress_health_check=[HealthCheck.too_slow])


def test_the_live_tree_matches_the_reference_model() -> None:
    """Random exit sequences must produce identical decisions in the tree and in the model."""
    run_state_machine_as_test(SupervisionMachine, settings=MACHINE_SETTINGS)
