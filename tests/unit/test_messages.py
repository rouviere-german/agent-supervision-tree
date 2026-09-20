"""Envelopes: identity across stages, attempts, deadlines, size limits and trace context."""

from __future__ import annotations

import asyncio

import pytest
from hypothesis import given
from hypothesis import strategies as st
from opentelemetry import trace as otel_trace
from opentelemetry.sdk.trace import TracerProvider

from agent_supervision_tree.runtime.errors import MessageTooLargeError
from agent_supervision_tree.runtime.messages import Envelope, json_size, now

pytestmark = pytest.mark.unit


def test_kind_and_attempt_are_validated() -> None:
    with pytest.raises(ValueError, match="invalid message kind"):
        Envelope(kind="Not A Kind", payload={})
    with pytest.raises(ValueError, match="attempt starts at 1"):
        Envelope(kind="a.kind", payload={}, attempt=0)


def test_create_assigns_an_id_a_deadline_and_a_size() -> None:
    envelope = Envelope.create("doc.ingest", {"id": "doc-abcdef", "text": "hello"}, ttl=5)
    assert len(envelope.id) == 32
    assert envelope.attempt == 1
    assert envelope.size_bytes == json_size({"id": "doc-abcdef", "text": "hello"})
    remaining = envelope.remaining()
    assert remaining is not None
    assert 0 < remaining <= 5
    assert not envelope.expired()


def test_create_enforces_the_size_limit_before_anything_is_queued() -> None:
    with pytest.raises(MessageTooLargeError) as info:
        Envelope.create("doc.ingest", {"text": "x" * 500}, max_bytes=64)
    assert info.value.size > info.value.limit == 64


def test_derive_keeps_identity_and_deadline_but_restarts_attempts() -> None:
    first = Envelope.create("doc.ingest", {"id": "doc-abcdef", "text": "t"}, ttl=5)
    retried = first.next_attempt().next_attempt()
    assert retried.attempt == 3
    assert retried.id == first.id

    forwarded = retried.derive("doc.extract", {"vendor": "acme"})
    assert forwarded.id == first.id
    assert forwarded.deadline == first.deadline
    assert forwarded.created_at == first.created_at
    assert forwarded.attempt == 1
    assert forwarded.kind == "doc.extract"


def test_expiry_is_measured_against_the_given_clock() -> None:
    envelope = Envelope(kind="a.kind", payload={}, deadline=100.0)
    assert envelope.remaining(at=90.0) == 10.0
    assert not envelope.expired(at=90.0)
    assert envelope.expired(at=100.0)
    assert envelope.remaining(at=150.0) == -50.0


def test_messages_without_a_deadline_never_expire() -> None:
    envelope = Envelope(kind="a.kind", payload={})
    assert envelope.remaining() is None
    assert not envelope.expired()


def test_non_json_payloads_report_size_zero_and_are_not_size_limited() -> None:
    assert json_size(object()) == 0
    control = Envelope(kind="llm.call", payload=object())
    assert control.size_bytes == 0


def test_now_falls_back_to_a_monotonic_clock_outside_an_event_loop() -> None:
    outside = now()
    assert outside > 0
    inside = asyncio.run(_loop_time())
    assert isinstance(inside, float)


async def _loop_time() -> float:
    return now()


def test_trace_context_is_captured_and_re_extracted() -> None:
    provider = TracerProvider()
    tracer = provider.get_tracer("test")
    with tracer.start_as_current_span("ingest"):
        envelope = Envelope.create("doc.ingest", {"id": "doc-abcdef", "text": "t"})
    assert envelope.traceparent is not None
    context = envelope.otel_context()
    span_context = otel_trace.get_current_span(context).get_span_context()
    assert format(span_context.trace_id, "032x") in envelope.traceparent


def test_an_envelope_without_trace_context_yields_an_empty_context() -> None:
    envelope = Envelope(kind="a.kind", payload={}, traceparent=None)
    assert not otel_trace.get_current_span(envelope.otel_context()).get_span_context().is_valid


@given(
    payload=st.dictionaries(
        st.text(min_size=1, max_size=8), st.integers() | st.text(max_size=16), max_size=5
    ),
    stages=st.lists(st.sampled_from(["a.one", "b.two", "c.three"]), min_size=1, max_size=6),
)
def test_message_identity_survives_any_chain_of_stages_and_attempts(
    payload: dict[str, object], stages: list[str]
) -> None:
    """Whatever route a message takes, its id, deadline and creation time never change."""
    original = Envelope.create("doc.ingest", payload, ttl=30)  # type: ignore[arg-type]
    current = original
    for stage in stages:
        current = current.next_attempt().derive(stage, payload)
    assert current.id == original.id
    assert current.deadline == original.deadline
    assert current.created_at == original.created_at
    assert current.attempt == 1
