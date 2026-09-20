"""Actors: lifecycle, structured spawned tasks, message failures and in-flight redelivery."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from functools import partial

import pytest

import helpers
from agent_supervision_tree.runtime.actor import ActorRef, ExitReason
from agent_supervision_tree.runtime.errors import (
    ActorNotRunningError,
    MessageFailed,
    QuarantineFullError,
)
from agent_supervision_tree.runtime.messages import Envelope
from agent_supervision_tree.runtime.system import EXIT_SOFTWARE, ActorSystem
from helpers import BlockingActor, CountingActor, actor_child, build_system, counter, wait_for

pytestmark = pytest.mark.unit


@pytest.fixture
async def system() -> AsyncIterator[ActorSystem]:
    actors: dict[str, CountingActor] = {}

    def factory() -> CountingActor:
        actor = CountingActor(name="worker")
        actors["latest"] = actor
        return actor

    built = build_system([actor_child("worker", factory, capacity=4)])
    built.state = actors  # type: ignore[attr-defined]
    await built.start()
    try:
        yield built
    finally:
        if built.running:
            await built.stop(timeout=2)


def latest(system: ActorSystem) -> CountingActor:
    return system.state["latest"]  # type: ignore[attr-defined,no-any-return]


def send(system: ActorSystem, **payload: object) -> Envelope:
    envelope = Envelope.create("test.message", payload)  # type: ignore[arg-type]
    assert system.mailbox("root/worker").try_send(envelope)
    return envelope


async def test_lifecycle_hooks_run_once_per_incarnation(system: ActorSystem) -> None:
    actor = latest(system)
    assert actor.starts == 1
    send(system, action="noop")
    assert await wait_for(lambda: len(actor.received) == 1)
    await system.stop(timeout=2)
    assert [reason.kind for reason in actor.stops] == ["shutdown"]


async def test_exit_reason_labels_are_low_cardinality() -> None:
    assert ExitReason.normal().label() == "normal"
    assert ExitReason.shutdown("drained").label() == "shutdown:drained"
    assert ExitReason.failed(RuntimeError("boom")).label() == "error:RuntimeError"
    assert ExitReason.killed("fault_injection").abnormal
    assert not ExitReason.shutdown().abnormal


async def test_a_behavior_exception_ends_the_incarnation_and_a_new_one_starts(
    system: ActorSystem,
) -> None:
    first = latest(system)
    send(system, action="crash")
    assert await wait_for(lambda: latest(system) is not first)
    assert latest(system).starts == 1
    assert first.stops[0].kind == "error"
    assert first.stops[0].detail == "RuntimeError"
    assert await wait_for(
        lambda: (
            counter(system, "ast_restarts_total", actor="root/worker", reason="error:RuntimeError")
            == 1
        )
    )


async def test_actor_exit_is_a_normal_exit(system: ActorSystem) -> None:
    first = latest(system)
    send(system, action="exit")
    assert await wait_for(lambda: first.stops != [])
    assert first.stops[0].kind == "normal"


async def test_a_message_of_an_unexpected_kind_is_quarantined_not_executed() -> None:
    def factory() -> CountingActor:
        return CountingActor(name="picky", accepts=["known.kind"])

    system = build_system([actor_child("picky", factory)])
    await system.start()
    try:
        envelope = Envelope.create("other.kind", {"a": 1})
        assert system.mailbox("root/picky").try_send(envelope)
        assert await wait_for(lambda: envelope.id in system.quarantine)
        record = system.quarantine.get(envelope.id)
        assert record is not None
        assert record.reason == "unexpected_kind"
    finally:
        await system.stop(timeout=2)


async def test_an_expired_message_is_quarantined_before_the_behavior_sees_it() -> None:
    clock = helpers.FakeClock(100.0)
    system = build_system([actor_child("worker", CountingActor)], clock=clock)
    await system.start()
    try:
        envelope = Envelope(kind="test.message", payload={}, deadline=50.0)
        assert system.mailbox("root/worker").try_send(envelope)
        assert await wait_for(lambda: envelope.id in system.quarantine)
        record = system.quarantine.get(envelope.id)
        assert record is not None
        assert record.reason == "deadline_exceeded"
    finally:
        await system.stop(timeout=2)


async def test_a_retryable_failure_is_redelivered_then_quarantined(system: ActorSystem) -> None:
    envelope = send(system, action="fail", reason="malformed_output")
    assert await wait_for(lambda: envelope.id in system.quarantine)
    actor = latest(system)
    assert [item.attempt for item in actor.received] == [1, 2, 3]
    record = system.quarantine.get(envelope.id)
    assert record is not None
    assert (record.reason, record.attempts) == ("malformed_output", 3)
    assert actor.stops == []  # the actor kept running the whole time
    assert system.mailbox("root/worker").credits == 4


async def test_a_non_retryable_failure_is_quarantined_at_once(system: ActorSystem) -> None:
    envelope = send(system, action="fail_final", reason="refused")
    assert await wait_for(lambda: envelope.id in system.quarantine)
    assert [item.attempt for item in latest(system).received] == [1]
    record = system.quarantine.get(envelope.id)
    assert record is not None
    assert record.reason == "refused"


async def test_a_behavior_can_quarantine_a_message_itself(system: ActorSystem) -> None:
    envelope = send(system, action="quarantine")
    assert await wait_for(lambda: envelope.id in system.quarantine)
    record = system.quarantine.get(envelope.id)
    assert record is not None
    assert record.reason == "by_request"


async def test_in_flight_work_is_redelivered_to_the_next_incarnation() -> None:
    actors: list[BlockingActor] = []

    def factory() -> BlockingActor:
        actor = BlockingActor()
        actors.append(actor)
        return actor

    system = build_system([actor_child("blocker", factory)])
    await system.start()
    try:
        envelope = Envelope.create("test.message", {"n": 1})
        assert system.mailbox("root/blocker").try_send(envelope)
        await asyncio.wait_for(actors[0].entered.wait(), timeout=2)
        system.kill("root/blocker", ExitReason.killed("chaos"))
        assert await wait_for(lambda: len(actors) == 2)
        actors[1].release.set()
        assert await wait_for(lambda: actors[1].handled == [envelope.id])
        # Redelivery counts as a second attempt and nothing was lost.
        assert envelope.id not in system.quarantine
        assert system.mailbox("root/blocker").idle
    finally:
        await system.stop(timeout=2)


async def test_in_flight_work_is_quarantined_once_the_attempts_are_used_up() -> None:
    actors: list[BlockingActor] = []

    def factory() -> BlockingActor:
        actor = BlockingActor()
        actors.append(actor)
        return actor

    system = build_system(
        [actor_child("blocker", factory, max_attempts=2)], max_restarts=10, max_seconds=60
    )
    await system.start()
    try:
        envelope = Envelope.create("test.message", {"n": 1})
        assert system.mailbox("root/blocker").try_send(envelope)
        for incarnation in range(1, 3):
            await asyncio.wait_for(actors[incarnation - 1].entered.wait(), timeout=2)
            system.kill("root/blocker", ExitReason.killed("chaos"))
            assert await wait_for(lambda n=incarnation: len(actors) == n + 1)  # type: ignore[misc]
        assert await wait_for(lambda: envelope.id in system.quarantine)
        record = system.quarantine.get(envelope.id)
        assert record is not None
        assert record.reason == "actor_killed"
    finally:
        await system.stop(timeout=2)


async def test_a_spawned_task_that_fails_crashes_its_actor() -> None:
    from agent_supervision_tree.runtime.actor import Actor, ActorContext

    class Spawner(Actor):
        async def receive(self, envelope: Envelope, ctx: ActorContext) -> None:
            ctx.spawn(self._boom(), name="boom")

        async def _boom(self) -> None:
            await asyncio.sleep(0)
            raise RuntimeError("background failure")

    exits: list[ExitReason] = []
    system = build_system([actor_child("spawner", Spawner)])
    await system.start()
    system.monitor("root/spawner", lambda path, incarnation, reason: exits.append(reason))
    try:
        assert system.mailbox("root/spawner").try_send(Envelope.create("test.message", {}))
        assert await wait_for(lambda: bool(exits))
        assert exits[0].kind == "error"
        assert exits[0].detail == "spawned_task_failed"
    finally:
        await system.stop(timeout=2)


async def test_spawned_tasks_are_cancelled_when_the_actor_exits() -> None:
    from agent_supervision_tree.runtime.actor import Actor, ActorContext

    running: list[asyncio.Task[object]] = []

    class Spawner(Actor):
        async def on_start(self, ctx: ActorContext) -> None:
            running.append(ctx.spawn(asyncio.sleep(60), name="sleeper"))

        async def receive(self, envelope: Envelope, ctx: ActorContext) -> None:
            return None

    system = build_system([actor_child("spawner", Spawner)])
    await system.start()
    await system.stop(timeout=2)
    assert await wait_for(running[0].done)
    assert running[0].cancelled()


async def test_a_failing_on_stop_hook_does_not_mask_the_exit_reason() -> None:
    from agent_supervision_tree.runtime.actor import Actor, ActorContext

    class BadStop(Actor):
        async def receive(self, envelope: Envelope, ctx: ActorContext) -> None:
            raise RuntimeError("primary failure")

        async def on_stop(self, reason: ExitReason, ctx: ActorContext) -> None:
            raise ValueError("secondary failure")

    exits: list[ExitReason] = []
    system = build_system([actor_child("bad", BadStop)])
    await system.start()
    system.monitor("root/bad", lambda path, incarnation, reason: exits.append(reason))
    try:
        assert system.mailbox("root/bad").try_send(Envelope.create("test.message", {}))
        assert await wait_for(lambda: bool(exits))
        assert exits[0].detail == "RuntimeError"
    finally:
        await system.stop(timeout=2)


async def test_a_start_that_takes_too_long_is_a_failure() -> None:
    from helpers import SlowStartActor

    system = build_system(
        [actor_child("slow", partial(SlowStartActor, 5.0), start_timeout=0.05)],
        max_restarts=0,
        max_seconds=60,
    )
    await system.start()
    assert await wait_for(lambda: any(path == "root/slow" for path, _, _ in system.exits))
    reason = next(item for path, _, item in system.exits if path == "root/slow")
    assert reason.kind == "error"
    assert reason.detail == "TimeoutError"
    # max_restarts=0, so the root escalates at once and the process is told to restart.
    assert await wait_for(lambda: system.exit_code == EXIT_SOFTWARE)
    assert await system.stop(timeout=1) == EXIT_SOFTWARE


async def test_quarantine_overflow_is_counted_rather_than_hidden() -> None:
    system = build_system([actor_child("worker", CountingActor)], quarantine_max_records=1)
    await system.start()
    try:
        for index in range(3):
            assert system.mailbox("root/worker").try_send(
                Envelope.create("test.message", {"action": "fail_final", "n": index})
            )
        assert await wait_for(lambda: counter(system, "ast_quarantine_overflow_total") >= 2)
        assert len(system.quarantine) == 1
        assert system.quarantine.overflow_total >= 2
    finally:
        await system.stop(timeout=2)


async def test_actor_refs_follow_restarts(system: ActorSystem) -> None:
    ref = ActorRef("root/worker", system)
    assert ref.try_send(Envelope.create("test.message", {"action": "crash"}))
    assert await wait_for(lambda: latest(system).starts == 1 and len(system.exits) >= 1)
    # The same ref still resolves after the restart, because it names the slot, not the actor.
    await ref.send(Envelope.create("test.message", {"action": "noop"}), timeout=1)
    assert await wait_for(lambda: len(latest(system).received) == 1)


async def test_unknown_paths_are_refused(system: ActorSystem) -> None:
    with pytest.raises(ActorNotRunningError):
        system.mailbox("root/nope")
    with pytest.raises(ActorNotRunningError):
        system.resolve("elsewhere/worker")
    with pytest.raises(ActorNotRunningError):
        system.resolve("root/worker/deeper")


async def test_message_failed_carries_a_reason_and_a_retry_flag() -> None:
    failure = MessageFailed("schema_violation", "missing vendor", retryable=False)
    assert str(failure) == "schema_violation: missing vendor"
    assert failure.reason == "schema_violation"
    assert not failure.retryable


async def test_quarantine_full_error_is_raised_by_the_quarantine_itself() -> None:
    from agent_supervision_tree.runtime.quarantine import Quarantine

    quarantine = Quarantine(max_records=1)
    quarantine.add(Envelope.create("a.kind", {"n": 1}), reason="poison", actor="root/w")
    with pytest.raises(QuarantineFullError):
        quarantine.add(Envelope.create("a.kind", {"n": 2}), reason="poison", actor="root/w")


async def test_each_receive_opens_a_span_in_the_message_trace_context() -> None:
    """The OTel context travels in the envelope, so a stage's work joins the ingress trace."""
    from opentelemetry import trace as otel_trace
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from agent_supervision_tree.core.tracing import build_tracer_provider

    exporter = InMemorySpanExporter()
    provider = build_tracer_provider("test", extra_exporter=exporter)
    system = build_system([actor_child("worker", CountingActor)], tracer_provider=provider)
    await system.start()
    try:
        tracer = provider.get_tracer("test")
        with tracer.start_as_current_span("ingest") as ingest:
            trace_id = ingest.get_span_context().trace_id
            envelope = Envelope.create("test.message", {"n": 1})
        assert system.mailbox("root/worker").try_send(envelope)
        assert await wait_for(
            lambda: any(span.name == "actor.receive" for span in exporter.get_finished_spans())
        )
    finally:
        await system.stop(timeout=2)
    span = next(item for item in exporter.get_finished_spans() if item.name == "actor.receive")
    assert span.context is not None
    assert span.context.trace_id == trace_id
    attributes = dict(span.attributes or {})
    assert attributes["actor.path"] == "root/worker"
    assert attributes["message.id"] == envelope.id
    assert attributes["message.kind"] == "test.message"
    assert otel_trace.get_current_span().get_span_context().trace_id != trace_id
