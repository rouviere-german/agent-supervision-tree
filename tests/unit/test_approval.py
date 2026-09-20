from __future__ import annotations

import itertools
import json
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import BaseModel, ConfigDict, Field, JsonValue

from agent_supervision_tree.agentic.approval import (
    ApprovalGate,
    ApprovalStatus,
    ApprovalStore,
    tool_call_hash,
)
from agent_supervision_tree.agentic.errors import (
    ApprovalError,
    ApprovalRequiredError,
    ToolArgumentError,
)
from agent_supervision_tree.agentic.providers.base import ToolUseBlock
from agent_supervision_tree.agentic.tools import (
    Capability,
    ToolDefinition,
    ToolRegistry,
    ValidatedCall,
)
from agent_supervision_tree.core.canonical import canonical_json

pytestmark = pytest.mark.unit


class WriteArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str
    content: str = ""


class ReadArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str


class MailArgs(BaseModel):
    """An argument model whose canonical dump hides a field the handler still receives."""

    model_config = ConfigDict(extra="forbid")
    to: str
    bcc: str = Field(default="", exclude=True)


class Clock:
    def __init__(self) -> None:
        self.now = 1_000.0

    def __call__(self) -> float:
        return self.now


def make_registry() -> ToolRegistry:
    registry = ToolRegistry({Capability.READ_DATA, Capability.WRITE_DATA})
    registry.register(
        ToolDefinition(
            name="write_file",
            description="Write a file.",
            args_model=WriteArgs,
            capabilities=frozenset({Capability.WRITE_DATA}),
            handler=lambda args: "ok",
        )
    )
    registry.register(
        ToolDefinition(
            name="read_file",
            description="Read a file.",
            args_model=ReadArgs,
            capabilities=frozenset({Capability.READ_DATA}),
            handler=lambda args: "data",
        )
    )
    return registry


def validated(registry: ToolRegistry, name: str, **arguments: JsonValue) -> ValidatedCall:
    return registry.validate_call(ToolUseBlock(id="c1", name=name, arguments=arguments))


def make_store(clock: Clock | None = None, path: Path | None = None) -> ApprovalStore:
    counter = itertools.count()
    return ApprovalStore(
        path, ttl_seconds=60.0, clock=clock or Clock(), id_factory=lambda: f"req-{next(counter)}"
    )


def test_call_hash_binds_the_tool_and_every_argument() -> None:
    base = tool_call_hash("write_file", {"path": "a.txt", "content": "x"})
    assert base == tool_call_hash("write_file", {"content": "x", "path": "a.txt"})
    assert base != tool_call_hash("write_file", {"path": "a.txt", "content": "y"})
    assert base != tool_call_hash("delete_file", {"path": "a.txt", "content": "x"})


_args = st.dictionaries(st.text(max_size=5), st.integers() | st.text(max_size=5), max_size=4)


@given(_args, _args)
def test_hash_equality_coincides_with_canonical_equality(
    a: dict[str, int | str], b: dict[str, int | str]
) -> None:
    same_hash = tool_call_hash("t", a) == tool_call_hash("t", b)
    assert same_hash == (canonical_json(a) == canonical_json(b))


def test_side_effects_block_until_a_different_principal_approves() -> None:
    registry = make_registry()
    store = make_store()
    gate = ApprovalGate(store, principal="agent:1")
    call = validated(registry, "write_file", path="a.txt", content="x")

    with pytest.raises(ApprovalRequiredError) as first:
        gate.authorize(call)
    with pytest.raises(ApprovalRequiredError) as second:
        gate.authorize(call)
    assert first.value.request_id == second.value.request_id
    assert len(store.pending()) == 1

    with pytest.raises(ApprovalError, match="separation of duties"):
        store.decide(first.value.request_id, approver="agent:1", approve=True)
    decided = store.decide(first.value.request_id, approver="human:alice", approve=True)
    assert decided.status is ApprovalStatus.APPROVED

    assert gate.authorize(call) == first.value.call_hash
    assert store.get(first.value.request_id).status is ApprovalStatus.CONSUMED
    with pytest.raises(ApprovalRequiredError):
        gate.authorize(call)


def test_an_approval_does_not_transfer_to_changed_arguments() -> None:
    registry = make_registry()
    store = make_store()
    gate = ApprovalGate(store, principal="agent:1")
    approved_call = validated(registry, "write_file", path="a.txt")
    with pytest.raises(ApprovalRequiredError) as pending:
        gate.authorize(approved_call)
    store.decide(pending.value.request_id, approver="human:alice", approve=True)

    with pytest.raises(ApprovalRequiredError):
        gate.authorize(validated(registry, "write_file", path="b.txt"))
    assert store.get(pending.value.request_id).status is ApprovalStatus.APPROVED


def test_denied_requests_stay_blocked_and_cannot_be_redecided() -> None:
    registry = make_registry()
    store = make_store()
    gate = ApprovalGate(store, principal="agent:1")
    call = validated(registry, "write_file", path="a.txt")
    with pytest.raises(ApprovalRequiredError) as pending:
        gate.authorize(call)
    store.decide(pending.value.request_id, approver="human:bob", approve=False, reason="no")
    with pytest.raises(ApprovalError, match="not pending"):
        store.decide(pending.value.request_id, approver="human:bob", approve=True)
    with pytest.raises(ApprovalRequiredError):
        gate.authorize(call)


def test_approvals_expire() -> None:
    clock = Clock()
    registry = make_registry()
    store = make_store(clock)
    gate = ApprovalGate(store, principal="agent:1")
    call = validated(registry, "write_file", path="a.txt")
    with pytest.raises(ApprovalRequiredError) as pending:
        gate.authorize(call)
    clock.now += 120
    with pytest.raises(ApprovalError, match="expired"):
        store.decide(pending.value.request_id, approver="human:alice", approve=True)
    assert store.pending() == []


def test_approved_but_expired_calls_cannot_be_consumed() -> None:
    clock = Clock()
    registry = make_registry()
    store = make_store(clock)
    call = validated(registry, "write_file", path="a.txt")
    record = store.request(call, requested_by="agent:1")
    store.decide(record.id, approver="human:alice", approve=True)
    clock.now += 120
    assert store.consume(record.call_hash, principal="agent:1") is None


def test_read_only_tools_need_no_approval() -> None:
    store = make_store()
    gate = ApprovalGate(store, principal="agent:1")
    call = validated(make_registry(), "read_file", path="a.txt")
    assert gate.authorize(call) == tool_call_hash("read_file", {"path": "a.txt"})
    assert store.pending() == []


def test_store_state_persists_across_instances(tmp_path: Path) -> None:
    path = tmp_path / "approvals.json"
    call = validated(make_registry(), "write_file", path="a.txt")
    record = make_store(path=path).request(call, requested_by="agent:1")

    reopened = make_store(path=path)
    assert [r.id for r in reopened.pending()] == [record.id]
    reopened.decide(record.id, approver="human:alice", approve=True)
    assert make_store(path=path).get(record.id).status is ApprovalStatus.APPROVED


def test_unknown_request_ids_are_rejected() -> None:
    with pytest.raises(ApprovalError, match="unknown"):
        make_store().decide("missing", approver="human:alice", approve=True)


def test_an_approval_is_bound_to_the_principal_that_requested_it() -> None:
    registry = make_registry()
    store = make_store()
    call = validated(registry, "write_file", path="a.txt")
    with pytest.raises(ApprovalRequiredError) as pending:
        ApprovalGate(store, principal="agent:1").authorize(call)
    store.decide(pending.value.request_id, approver="human:bob", approve=True)

    # Neither the approver nor any other principal may run the call on agent:1's approval.
    with pytest.raises(ApprovalRequiredError):
        ApprovalGate(store, principal="human:bob").authorize(call)
    with pytest.raises(ApprovalRequiredError):
        ApprovalGate(store, principal="agent:2").authorize(call)
    assert store.get(pending.value.request_id).status is ApprovalStatus.APPROVED
    assert ApprovalGate(store, principal="agent:1").authorize(call) == pending.value.call_hash


def test_arguments_hidden_from_the_canonical_form_cannot_be_substituted_after_approval() -> None:
    delivered: list[str] = []

    def send_mail(args: MailArgs) -> JsonValue:
        delivered.append(args.bcc)
        return "sent"

    registry = ToolRegistry({Capability.SEND_MESSAGE})
    registry.register(
        ToolDefinition(
            name="send_mail",
            description="Send mail.",
            args_model=MailArgs,
            capabilities=frozenset({Capability.SEND_MESSAGE}),
            handler=send_mail,
        )
    )
    store = make_store()
    gate = ApprovalGate(store, principal="agent:1")
    with pytest.raises(ApprovalRequiredError) as pending:
        gate.authorize(validated(registry, "send_mail", to="boss@example.com"))
    store.decide(pending.value.request_id, approver="human:alice", approve=True)

    with pytest.raises(ToolArgumentError, match="canonical"):
        validated(registry, "send_mail", to="boss@example.com", bcc="attacker@evil.example")
    assert delivered == []
    assert store.get(pending.value.request_id).status is ApprovalStatus.APPROVED


def test_arguments_mutated_after_authorization_are_not_run() -> None:
    registry = make_registry()
    store = make_store()
    gate = ApprovalGate(store, principal="agent:1")
    call = validated(registry, "write_file", path="a.txt")
    with pytest.raises(ApprovalRequiredError) as pending:
        gate.authorize(call)
    store.decide(pending.value.request_id, approver="human:alice", approve=True)
    call_hash = gate.authorize(call)

    args = call.args
    assert isinstance(args, WriteArgs)
    args.path = "/etc/passwd"
    with pytest.raises(ToolArgumentError, match="changed"):
        call.run(authorized_hash=call_hash)


def test_a_persisted_record_whose_arguments_do_not_match_its_hash_is_refused(
    tmp_path: Path,
) -> None:
    path = tmp_path / "approvals.json"
    call = validated(make_registry(), "write_file", path="danger.txt")
    make_store(path=path).request(call, requested_by="agent:1")
    data = json.loads(path.read_text(encoding="utf-8"))
    data["records"][0]["args"]["path"] = "harmless.txt"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ApprovalError, match="failed validation"):
        make_store(path=path)


def test_instances_sharing_a_file_see_decisions_and_consume_an_approval_once(
    tmp_path: Path,
) -> None:
    path = tmp_path / "approvals.json"
    call = validated(make_registry(), "write_file", path="a.txt")
    gate = ApprovalGate(make_store(path=path), principal="agent:1")
    second_worker = ApprovalGate(make_store(path=path), principal="agent:1")
    reviewer = make_store(path=path)

    with pytest.raises(ApprovalRequiredError) as pending:
        gate.authorize(call)
    reviewer.decide(pending.value.request_id, approver="human:alice", approve=True)

    assert gate.authorize(call) == pending.value.call_hash
    with pytest.raises(ApprovalRequiredError):
        second_worker.authorize(call)
    assert reviewer.get(pending.value.request_id).status is ApprovalStatus.CONSUMED
