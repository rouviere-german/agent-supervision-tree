"""The workload's actors: least privilege, the approval binding, dedupe and fault injection."""

from __future__ import annotations

import asyncio

import pytest

from agent_supervision_tree.agentic.errors import ToolArgumentError, ToolNotAllowedError
from agent_supervision_tree.agentic.providers.base import ToolUseBlock
from agent_supervision_tree.agentic.tools import Capability, ToolRegistry
from agent_supervision_tree.runtime.actor import Actor, ActorContext, ExitReason
from agent_supervision_tree.runtime.errors import ActorNotRunningError
from agent_supervision_tree.runtime.messages import Envelope
from agent_supervision_tree.runtime.system import ActorSystem
from agent_supervision_tree.workload.actors import (
    APPROVAL_KIND,
    SINK_KIND,
    VALIDATE_KIND,
    ApprovalActor,
    ApprovalDecision,
    ApprovalRequest,
    FaultInjector,
    InMemoryWebhook,
    PoolRouter,
    SinkState,
    Validator,
    WebhookSink,
    build_webhook_registry,
    default_approval_policy,
)
from agent_supervision_tree.workload.fake_llm import FaultingFakeProvider, FaultProfile
from agent_supervision_tree.workload.schemas import DocumentMessage, Extraction
from helpers import actor_child, build_system, wait_for

pytestmark = pytest.mark.unit

WEBHOOK_URL = "https://hooks.example.internal/invoices"
APPROVAL_PATH = "root/approval"
SINK_PATH = "root/sink"


def extraction(total: int = 300, document_id: str = "doc-abcdef") -> Extraction:
    return Extraction.model_validate(
        {
            "document_id": document_id,
            "vendor": "Nordwind Logistik GmbH",
            "currency": "EUR",
            "total_cents": total,
            "line_items": [{"description": "pallet transport", "amount_cents": total}],
            "due_date": "2026-11-02",
        }
    )


# --- the sink's tool ------------------------------------------------------------------------


def test_the_sink_has_exactly_one_tool_with_one_capability() -> None:
    registry = build_webhook_registry(WEBHOOK_URL, InMemoryWebhook())
    assert registry.names == ("post_webhook",)
    tool = registry.get("post_webhook")
    assert tool.capabilities == frozenset({Capability.NETWORK_EGRESS})
    assert tool.needs_approval


def test_the_webhook_url_is_fixed_by_configuration() -> None:
    webhook = InMemoryWebhook()
    registry = build_webhook_registry(WEBHOOK_URL, webhook)
    call = registry.validate_call(
        ToolUseBlock(
            id="1",
            name="post_webhook",
            arguments={"url": "https://evil.example/collect", "payload": {"a": 1}},
        )
    )
    with pytest.raises(ToolNotAllowedError, match="fixed by configuration"):
        call.run(authorized_hash=call.call_hash)
    assert webhook.posts == []


def test_arguments_cannot_change_after_approval() -> None:
    registry = build_webhook_registry(WEBHOOK_URL, InMemoryWebhook())
    call = registry.validate_call(
        ToolUseBlock(id="1", name="post_webhook", arguments={"url": WEBHOOK_URL, "payload": {}})
    )
    with pytest.raises(ToolArgumentError, match="changed after authorization"):
        call.run(authorized_hash="0" * 64)


def test_an_extractor_registry_grants_nothing() -> None:
    registry = ToolRegistry(())
    assert registry.specs() == ()
    with pytest.raises(ToolNotAllowedError):
        registry.get("post_webhook")


# --- the approval policy --------------------------------------------------------------------


def test_the_default_policy_denies_by_vendor_and_by_amount() -> None:
    registry = build_webhook_registry(WEBHOOK_URL, InMemoryWebhook())
    policy = default_approval_policy(max_total_cents=1_000)

    def decide(payload: dict[str, object]) -> tuple[bool, str]:
        call = registry.validate_call(
            ToolUseBlock(
                id="1", name="post_webhook", arguments={"url": WEBHOOK_URL, "payload": payload}
            )
        )
        return policy(call)

    assert decide(extraction(500).model_dump(mode="json")) == (True, "policy")
    assert decide(extraction(5_000).model_dump(mode="json")) == (False, "total_above_limit")
    attacker = {**extraction(500).model_dump(mode="json"), "vendor": "attacker corp"}
    assert decide(attacker) == (False, "vendor_denied")
    assert decide({"no": "payload"}) == (False, "total_above_limit")


# --- the sink under supervision ---------------------------------------------------------------


class RogueApprover(Actor):
    """Answers approvals with a decision for a *different* call, to test the binding."""

    accepts = frozenset({APPROVAL_KIND})

    def __init__(self, *, forge_hash: bool = True) -> None:
        self.forge_hash = forge_hash

    async def receive(self, envelope: Envelope, ctx: ActorContext) -> None:
        request = envelope.payload
        assert isinstance(request, ApprovalRequest)
        call_hash = "f" * 64 if self.forge_hash else request.call.call_hash
        message_id = request.message_id if self.forge_hash else "another-message"
        request.reply.set_result(
            ApprovalDecision(message_id, call_hash, True, "approver:rogue", "forged")
        )


async def start_sink(
    *, approver: Actor | None = None, max_total_cents: int = 50_000_000
) -> tuple[ActorSystem, SinkState, InMemoryWebhook]:
    state = SinkState()
    webhook = InMemoryWebhook()

    def sink_factory() -> WebhookSink:
        return WebhookSink(
            state=state,
            tools=build_webhook_registry(WEBHOOK_URL, webhook),
            approval_path=APPROVAL_PATH,
            webhook_url=WEBHOOK_URL,
        )

    def approval_factory() -> Actor:
        return approver or ApprovalActor(default_approval_policy(max_total_cents=max_total_cents))

    system = build_system(
        [
            actor_child("approval", approval_factory, capacity=8, max_attempts=1),
            actor_child("sink", sink_factory, capacity=8),
        ],
        max_restarts=5,
    )
    await system.start()
    return system, state, webhook


def sink_message(payload: Extraction, message_id: str | None = None) -> Envelope:
    return Envelope.create(
        SINK_KIND, payload.model_dump(mode="json"), ttl=2.0, message_id=message_id
    )


async def test_an_approved_extraction_is_posted_once() -> None:
    system, state, webhook = await start_sink()
    try:
        message = sink_message(extraction())
        assert system.mailbox(SINK_PATH).try_send(message)
        assert await wait_for(lambda: state.posted == 1)
        assert webhook.posts == [(WEBHOOK_URL, extraction().model_dump(mode="json"))]
        assert state.outcome(message.id) == "posted"
        assert system.audit.count("approval.decided") == 1
    finally:
        await system.stop(timeout=2)


async def test_a_redelivered_message_is_deduplicated_by_id() -> None:
    system, state, webhook = await start_sink()
    try:
        message = sink_message(extraction())
        assert system.mailbox(SINK_PATH).try_send(message)
        assert await wait_for(lambda: state.posted == 1)
        assert system.mailbox(SINK_PATH).try_send(sink_message(extraction(), message.id))
        assert await wait_for(lambda: state.duplicates == 1)
        assert len(webhook.posts) == 1
    finally:
        await system.stop(timeout=2)


async def test_a_denied_extraction_is_never_posted() -> None:
    system, state, webhook = await start_sink(max_total_cents=100)
    try:
        message = sink_message(extraction(total=5_000))
        assert system.mailbox(SINK_PATH).try_send(message)
        assert await wait_for(lambda: state.denied == 1)
        assert webhook.posts == []
        assert state.outcome(message.id) == "denied"
    finally:
        await system.stop(timeout=2)


async def test_an_approval_for_a_different_call_is_refused() -> None:
    system, state, webhook = await start_sink(approver=RogueApprover())
    try:
        message = sink_message(extraction())
        assert system.mailbox(SINK_PATH).try_send(message)
        assert await wait_for(lambda: message.id in system.quarantine)
        record = system.quarantine.get(message.id)
        assert record is not None
        assert record.reason == "approval_mismatch"
        assert webhook.posts == []
        assert state.posted == 0
    finally:
        await system.stop(timeout=2)


async def test_an_approval_for_a_different_message_is_refused() -> None:
    system, _, webhook = await start_sink(approver=RogueApprover(forge_hash=False))
    try:
        message = sink_message(extraction())
        assert system.mailbox(SINK_PATH).try_send(message)
        assert await wait_for(lambda: message.id in system.quarantine)
        assert webhook.posts == []
    finally:
        await system.stop(timeout=2)


async def test_a_payload_that_is_not_an_extraction_never_reaches_the_tool() -> None:
    system, _, webhook = await start_sink()
    try:
        message = Envelope.create(SINK_KIND, {"webhook_url": "https://evil.example"}, ttl=2.0)
        assert system.mailbox(SINK_PATH).try_send(message)
        assert await wait_for(lambda: message.id in system.quarantine)
        record = system.quarantine.get(message.id)
        assert record is not None
        assert record.reason == "schema_violation"
        assert webhook.posts == []
    finally:
        await system.stop(timeout=2)


async def test_a_failing_webhook_is_retried_and_then_quarantined() -> None:
    system, state, webhook = await start_sink()
    webhook.fail_next = 10
    try:
        message = sink_message(extraction())
        assert system.mailbox(SINK_PATH).try_send(message)
        assert await wait_for(lambda: message.id in system.quarantine)
        record = system.quarantine.get(message.id)
        assert record is not None
        assert (record.reason, record.attempts) == ("webhook_failed", 3)
        assert state.posted == 0
    finally:
        await system.stop(timeout=2)


async def test_an_approval_that_never_answers_expires_with_the_message() -> None:
    class SilentApprover(Actor):
        accepts = frozenset({APPROVAL_KIND})

        async def receive(self, envelope: Envelope, ctx: ActorContext) -> None:
            await asyncio.sleep(10)

    system, _, webhook = await start_sink(approver=SilentApprover())
    try:
        message = Envelope.create(SINK_KIND, extraction().model_dump(mode="json"), ttl=0.05)
        assert system.mailbox(SINK_PATH).try_send(message)
        assert await wait_for(lambda: message.id in system.quarantine, timeout=3)
        record = system.quarantine.get(message.id)
        assert record is not None
        assert record.reason == "deadline_exceeded"
        assert webhook.posts == []
    finally:
        await system.stop(timeout=2)


# --- the validator --------------------------------------------------------------------------


async def test_the_validator_forwards_only_consistent_extractions() -> None:
    system = build_system(
        [
            actor_child("sink", _Collector, capacity=8),
            actor_child(
                "validator", lambda: Validator("root/sink", max_total_cents=10_000), capacity=8
            ),
        ],
        max_restarts=5,
    )
    await system.start()
    try:
        good = Envelope.create(VALIDATE_KIND, extraction(300).model_dump(mode="json"), ttl=2)
        assert system.mailbox("root/validator").try_send(good)
        assert await wait_for(lambda: good.id in system.quarantine or _Collector.seen > 0)
        inconsistent = extraction(300).model_dump(mode="json")
        inconsistent["total_cents"] = 999
        bad = Envelope.create(VALIDATE_KIND, inconsistent, ttl=2)
        assert system.mailbox("root/validator").try_send(bad)
        assert await wait_for(lambda: bad.id in system.quarantine)
        assert system.quarantine.get(bad.id).reason == "validation_failed"  # type: ignore[union-attr]
        too_big = Envelope.create(VALIDATE_KIND, extraction(50_000).model_dump(mode="json"), ttl=2)
        assert system.mailbox("root/validator").try_send(too_big)
        assert await wait_for(lambda: too_big.id in system.quarantine)
        assert _Collector.seen == 1
    finally:
        _Collector.seen = 0
        await system.stop(timeout=2)


class _Collector(Actor):
    seen = 0

    async def receive(self, envelope: Envelope, ctx: ActorContext) -> None:
        type(self).seen += 1


# --- routing, sink state and fault injection ---------------------------------------------------


def test_sink_state_is_bounded_and_counts_outcomes() -> None:
    with pytest.raises(ValueError, match="capacity"):
        SinkState(0)
    state = SinkState(capacity=2)
    state.record("a", "posted")
    state.record("b", "denied")
    state.record("c", "posted")
    assert len(state) == 2
    assert "a" not in state  # the oldest entry was evicted
    assert (state.posted, state.denied) == (2, 1)
    assert state.outcome("c") == "posted"


def test_a_pool_router_needs_at_least_one_worker() -> None:
    with pytest.raises(ValueError, match="at least one worker"):
        PoolRouter([])


async def test_fault_commands_crash_actors_hang_calls_and_inject_poison() -> None:
    killed: list[tuple[str, str]] = []
    ingested: list[DocumentMessage] = []
    provider = FaultingFakeProvider(FaultProfile(latency_ms=0.0), seed=1)

    def kill(path: str, reason: ExitReason) -> None:
        if path == "root/missing":
            raise ActorNotRunningError(path)
        killed.append((path, reason.label()))

    def ingest(document: DocumentMessage) -> bool:
        ingested.append(document)
        return True

    system = build_system(
        [
            actor_child(
                "faults",
                lambda: FaultInjector(kill=kill, ingest=ingest, provider=provider),
                capacity=8,
            )
        ],
        max_restarts=5,
    )
    await system.start()
    try:
        for command in (
            {"op": "crash", "path": "root/sink"},
            {"op": "poison", "count": 3},
            {"op": "hang", "count": 2},
            {"op": "fault_rate", "rate": 0.5},
        ):
            assert system.mailbox("root/faults").try_send(
                Envelope.create("fault.command", command, ttl=2)  # type: ignore[arg-type]
            )
        assert await wait_for(lambda: system.audit.count("fault.injected") == 4)
        assert killed == [("root/sink", "killed:fault_injection")]
        assert len(ingested) == 3
        assert all("POISON" in document.text for document in ingested)
        assert provider.profile.fault_rate == pytest.approx(0.5)

        bad = Envelope.create("fault.command", {"op": "crash"}, ttl=2)
        assert system.mailbox("root/faults").try_send(bad)
        assert await wait_for(lambda: bad.id in system.quarantine)
        unknown = Envelope.create("fault.command", {"op": "crash", "path": "root/missing"}, ttl=2)
        assert system.mailbox("root/faults").try_send(unknown)
        assert await wait_for(lambda: unknown.id in system.quarantine)
        assert system.quarantine.get(unknown.id).reason == "unknown_actor"  # type: ignore[union-attr]
    finally:
        await system.stop(timeout=2)
