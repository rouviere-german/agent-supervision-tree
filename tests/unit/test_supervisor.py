"""Supervisors: the three strategies, restart types, intensity, escalation and shutdown."""

from __future__ import annotations

import asyncio

import pytest

from agent_supervision_tree.runtime.actor import ExitReason
from agent_supervision_tree.runtime.errors import ActorNotRunningError
from agent_supervision_tree.runtime.messages import Envelope
from agent_supervision_tree.runtime.supervisor import (
    ActorSpec,
    ChildSpec,
    Supervisor,
    SupervisorSpec,
)
from agent_supervision_tree.runtime.system import EXIT_SOFTWARE, ActorSystem
from helpers import CountingActor, FakeClock, SlowStartActor, actor_child, build_system, wait_for

pytestmark = pytest.mark.unit


def three_children(log: list[tuple[str, str]], **restarts: str) -> list[ChildSpec]:
    return [
        actor_child(
            name,
            lambda name=name: CountingActor(log=log, name=name),  # type: ignore[misc]
            restart=restarts.get(name, "permanent"),
        )
        for name in ("a", "b", "c")
    ]


async def crash(system: ActorSystem, path: str) -> None:
    """Kill the incarnation at ``path`` without waiting for what the supervisor decides."""
    system.kill(path, ExitReason.killed("test"))


def live_incarnation(system: ActorSystem, path: str) -> int:
    try:
        runnable = system.resolve(path)
    except ActorNotRunningError:
        return -1
    return runnable.incarnation if runnable.running else -1


async def crash_and_decide(system: ActorSystem, path: str) -> bool:
    """Kill ``path`` and wait until the supervisor has finished acting on the exit."""
    before = len(root_of(system).decisions)
    system.kill(path, ExitReason.killed("test"))
    return await wait_for(lambda: len(root_of(system).decisions) > before)


async def crash_and_restart(system: ActorSystem, path: str) -> bool:
    """Kill ``path`` and wait until a *new* incarnation is running (the usual crash loop)."""
    before = live_incarnation(system, path)
    system.kill(path, ExitReason.killed("test"))
    return await wait_for(lambda: live_incarnation(system, path) > before)


def root_of(system: ActorSystem) -> Supervisor:
    assert system.root is not None
    return system.root


def decisions(system: ActorSystem) -> list[str]:
    return [decision.child for decision in root_of(system).decisions]


# --- specs ---------------------------------------------------------------------------------


def test_child_ids_and_intensity_are_validated() -> None:
    spec = ActorSpec(factory=CountingActor)
    with pytest.raises(ValueError, match="invalid child id"):
        ChildSpec(id="Bad Id", spec=spec)
    with pytest.raises(ValueError, match="child ids must be unique"):
        SupervisorSpec(
            strategy="one_for_one",
            children=(ChildSpec(id="a", spec=spec), ChildSpec(id="a", spec=spec)),
        )
    with pytest.raises(ValueError, match="max_restarts"):
        SupervisorSpec(strategy="one_for_one", children=(), max_restarts=-1)


# --- strategies ----------------------------------------------------------------------------


async def test_one_for_one_restarts_only_the_child_that_died() -> None:
    log: list[tuple[str, str]] = []
    system = build_system(three_children(log), strategy="one_for_one", max_restarts=5)
    await system.start()
    try:
        assert await crash_and_restart(system, "root/b")
        assert root_of(system).starts() == {"a": 1, "b": 2, "c": 1}
        assert root_of(system).running_ids() == ("a", "b", "c")
        assert log[-2:] == [("b", "stop:killed"), ("b", "start")]
    finally:
        await system.stop(timeout=2)


async def test_one_for_all_terminates_in_reverse_order_and_restarts_in_start_order() -> None:
    log: list[tuple[str, str]] = []
    system = build_system(three_children(log), strategy="one_for_all", max_restarts=5)
    await system.start()
    try:
        del log[:]
        assert await crash_and_decide(system, "root/b")
        assert root_of(system).starts() == {"a": 2, "b": 2, "c": 2}
        # b died on its own; c and a are terminated in reverse start order, then all restart.
        assert log == [
            ("b", "stop:killed"),
            ("c", "stop:shutdown"),
            ("a", "stop:shutdown"),
            ("a", "start"),
            ("b", "start"),
            ("c", "start"),
        ]
        decision = root_of(system).decisions[-1]
        assert decision.terminated == ("c", "a")
        assert decision.restarted == ("a", "b", "c")
    finally:
        await system.stop(timeout=2)


async def test_rest_for_one_restarts_only_the_children_started_after_it() -> None:
    log: list[tuple[str, str]] = []
    system = build_system(three_children(log), strategy="rest_for_one", max_restarts=5)
    await system.start()
    try:
        del log[:]
        assert await crash_and_decide(system, "root/b")
        assert root_of(system).starts() == {"a": 1, "b": 2, "c": 2}
        assert log == [
            ("b", "stop:killed"),
            ("c", "stop:shutdown"),
            ("b", "start"),
            ("c", "start"),
        ]
    finally:
        await system.stop(timeout=2)


# --- restart types -------------------------------------------------------------------------


async def test_a_transient_child_is_restarted_only_after_an_abnormal_exit() -> None:
    log: list[tuple[str, str]] = []
    system = build_system(three_children(log, b="transient"), max_restarts=5)
    await system.start()
    try:
        assert system.mailbox("root/b").try_send(
            Envelope.create("test.message", {"action": "exit"})
        )
        assert await wait_for(lambda: "b" not in root_of(system).running_ids())
        assert root_of(system).starts() == {"a": 1, "b": 1, "c": 1}
        assert await crash_and_restart(system, "root/c")
    finally:
        await system.stop(timeout=2)


async def test_a_temporary_child_is_never_restarted_and_its_spec_is_dropped() -> None:
    log: list[tuple[str, str]] = []
    system = build_system(three_children(log, b="temporary"), max_restarts=5)
    await system.start()
    try:
        assert await crash_and_decide(system, "root/b")
        assert root_of(system).running_ids() == ("a", "c")
        assert "b" not in root_of(system).starts()
        assert root_of(system).decisions[-1].removed == ("b",)
        with pytest.raises(ActorNotRunningError):
            root_of(system).child("b")
    finally:
        await system.stop(timeout=2)


async def test_a_temporary_sibling_caught_in_a_one_for_all_restart_is_not_restarted() -> None:
    log: list[tuple[str, str]] = []
    system = build_system(
        three_children(log, c="temporary"), strategy="one_for_all", max_restarts=5
    )
    await system.start()
    try:
        assert await crash_and_decide(system, "root/a")
        assert root_of(system).running_ids() == ("a", "b")
        decision = root_of(system).decisions[-1]
        assert decision.removed == ("c",)
        assert decision.restarted == ("a", "b")
    finally:
        await system.stop(timeout=2)


# --- intensity and escalation ---------------------------------------------------------------


async def test_restarts_outside_the_window_do_not_count_towards_the_intensity() -> None:
    clock = FakeClock(1_000.0)
    system = build_system(
        [actor_child("a", CountingActor)], max_restarts=1, max_seconds=10.0, clock=clock
    )
    await system.start()
    try:
        for _ in range(3):
            assert await crash_and_restart(system, "root/a")
            clock.advance(11.0)  # consecutive restarts fall in different windows
        assert root_of(system).starts()["a"] == 4
        assert root_of(system).restarts_in_window() == 0
    finally:
        await system.stop(timeout=2)


async def test_exceeding_the_intensity_terminates_the_children_and_escalates() -> None:
    clock = FakeClock(0.0)
    log: list[tuple[str, str]] = []
    system = build_system(three_children(log), max_restarts=2, max_seconds=60.0, clock=clock)
    await system.start()
    try:
        assert await crash_and_restart(system, "root/a")
        assert await crash_and_restart(system, "root/a")
        await crash(system, "root/a")  # the third restart exceeds max_restarts=2
        assert await wait_for(lambda: system.exit_code is not None)
        decision = root_of(system).decisions[-1]
        assert decision.escalated
        assert decision.terminated == ("c", "b")  # reverse start order, a is already down
        assert system.exit_code == EXIT_SOFTWARE
        exit_reason = root_of(system).exit_reason
        assert exit_reason is not None
        assert exit_reason.label() == "error:max_restart_intensity"
        assert system.audit.count("supervisor.escalated") == 1
    finally:
        await system.stop(timeout=2)


async def test_a_sub_supervisor_that_escalates_is_itself_restarted_by_its_parent() -> None:
    clock = FakeClock(0.0)
    inner = SupervisorSpec(
        strategy="one_for_one",
        children=(actor_child("leaf", CountingActor),),
        max_restarts=1,
        max_seconds=60.0,
    )
    system = build_system(
        [ChildSpec(id="pool", spec=inner), actor_child("other", CountingActor)],
        max_restarts=5,
        max_seconds=60.0,
        clock=clock,
    )
    await system.start()
    try:
        assert await crash_and_restart(system, "root/pool/leaf")
        await crash(system, "root/pool/leaf")  # the pool allows only one restart
        # The pool exceeded its own intensity, exited, and the root restarted the whole pool.
        assert await wait_for(lambda: root_of(system).starts()["pool"] == 2)
        assert root_of(system).running_ids() == ("pool", "other")
        assert system.resolve("root/pool/leaf") is not None
        assert system.exit_code is None  # the root itself is fine
    finally:
        await system.stop(timeout=2)


async def test_a_failing_start_is_retried_until_the_intensity_is_exceeded() -> None:
    from helpers import CrashingActor

    starts: list[int] = []
    clock = FakeClock(0.0)
    system = build_system(
        [actor_child("bad", lambda: CrashingActor(starts))],
        max_restarts=3,
        max_seconds=60.0,
        clock=clock,
    )
    await system.start()
    try:
        assert await wait_for(lambda: system.exit_code == EXIT_SOFTWARE, timeout=3)
        assert len(starts) == 4  # the initial start plus max_restarts=3 retries
    finally:
        await system.stop(timeout=2)


# --- introspection and shutdown ---------------------------------------------------------------


async def test_the_tree_snapshot_reports_paths_mailboxes_and_restarts() -> None:
    log: list[tuple[str, str]] = []
    inner = SupervisorSpec(
        strategy="one_for_one", children=(actor_child("leaf", CountingActor),), max_restarts=3
    )
    system = build_system([ChildSpec(id="pool", spec=inner), *three_children(log)], max_restarts=5)
    await system.start()
    try:
        assert await crash_and_restart(system, "root/a")
        tree = system.tree()
        assert tree["path"] == "root"
        assert tree["strategy"] == "one_for_one"
        assert tree["restarts_in_window"] == 1
        pool = tree["children"][0]
        assert pool["kind"] == "supervisor"
        assert pool["children"][0]["path"] == "root/pool/leaf"
        child_a = next(node for node in tree["children"] if node["path"] == "root/a")
        assert child_a["incarnation"] == 2
        assert child_a["exits"] == ["killed:test"]
        assert child_a["mailbox"]["capacity"] == 8
        assert tree["tasks"] >= 1
        assert tree["quarantine"] == {"size": 0, "reasons": {}}
    finally:
        await system.stop(timeout=2)


async def test_shutdown_drains_mailboxes_and_stops_in_reverse_start_order() -> None:
    log: list[tuple[str, str]] = []
    system = build_system(three_children(log), max_restarts=5)
    await system.start()
    for name in ("a", "b", "c"):
        for index in range(3):
            assert system.mailbox(f"root/{name}").try_send(
                Envelope.create("test.message", {"n": index})
            )
    del log[:]
    assert await system.stop(timeout=5) == 0
    assert [entry for entry in log if entry[1].startswith("stop")] == [
        ("c", "stop:shutdown"),
        ("b", "stop:shutdown"),
        ("a", "stop:shutdown"),
    ]
    # Everything queued was processed before the actor was stopped.
    for name in ("a", "b", "c"):
        assert system.mailbox(f"root/{name}").idle


async def test_shutdown_kills_a_child_that_outlasts_the_deadline() -> None:
    from helpers import BlockingActor

    actors: list[BlockingActor] = []

    def factory() -> BlockingActor:
        actor = BlockingActor()
        actors.append(actor)
        return actor

    system = build_system([actor_child("blocker", factory, shutdown_timeout=0.05)])
    await system.start()
    assert system.mailbox("root/blocker").try_send(Envelope.create("test.message", {}))
    await asyncio.wait_for(actors[0].entered.wait(), timeout=2)
    assert await system.stop(timeout=0.2) == 0
    assert not system.running


async def test_starting_twice_is_refused_and_stopping_when_stopped_raises() -> None:
    system = build_system([actor_child("a", CountingActor)])
    await system.start()
    with pytest.raises(RuntimeError, match="already started"):
        await system.start()
    await system.stop(timeout=2)
    idle = build_system([actor_child("a", CountingActor)])
    with pytest.raises(Exception, match="not running"):
        await idle.stop()


async def test_children_start_in_order_and_the_system_waits_for_each() -> None:
    log: list[str] = []
    system = build_system(
        [
            actor_child("first", lambda: SlowStartActor(0.02, log, "first")),
            actor_child("second", lambda: SlowStartActor(0.0, log, "second")),
        ]
    )
    await system.start()
    try:
        assert log == ["first:started", "second:started"]
    finally:
        await system.stop(timeout=2)


async def test_links_kill_the_linked_actor_which_its_supervisor_restarts() -> None:
    log: list[tuple[str, str]] = []
    system = build_system(three_children(log), max_restarts=5)
    await system.start()
    try:
        system.link("root/a", "root/b")
        await crash(system, "root/a")
        assert await wait_for(lambda: root_of(system).starts() == {"a": 2, "b": 2, "c": 1})
        assert await wait_for(lambda: root_of(system).running_ids() == ("a", "b", "c"))
        reasons = [reason.detail for path, _, reason in system.exits if path == "root/b"]
        assert reasons == ["linked_exit:root/a"]
    finally:
        await system.stop(timeout=2)


async def test_monitors_are_notified_of_every_exit() -> None:
    seen: list[tuple[str, int, str]] = []
    system = build_system([actor_child("a", CountingActor)], max_restarts=5)
    await system.start()
    system.monitor("root/a", lambda path, inc, reason: seen.append((path, inc, reason.label())))
    try:
        assert await crash_and_restart(system, "root/a")
        assert seen == [("root/a", 1, "killed:test")]
        assert await crash_and_restart(system, "root/a")
        assert seen[1][1] == 2
    finally:
        await system.stop(timeout=2)
