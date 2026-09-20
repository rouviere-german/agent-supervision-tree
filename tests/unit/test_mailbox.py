"""Bounded mailboxes: the credit invariant, backpressure, redelivery and draining."""

from __future__ import annotations

import asyncio

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.stateful import (
    RuleBasedStateMachine,
    initialize,
    invariant,
    precondition,
    rule,
    run_state_machine_as_test,
)

from agent_supervision_tree.runtime.errors import MailboxClosedError, MessageTooLargeError
from agent_supervision_tree.runtime.mailbox import Mailbox
from agent_supervision_tree.runtime.messages import Envelope

pytestmark = pytest.mark.unit


def message(index: int = 0) -> Envelope:
    return Envelope.create("test.message", {"index": index})


def test_capacity_must_be_positive() -> None:
    with pytest.raises(ValueError, match="capacity"):
        Mailbox("m", 0)


async def test_credits_are_released_on_ack_not_on_receive() -> None:
    mailbox = Mailbox("m", 2)
    assert mailbox.try_send(message(1))
    assert mailbox.credits == 1
    received = await mailbox.receive()
    # Dequeued but not finished: the credit is still held, so the bound covers in-flight work.
    assert (mailbox.depth, mailbox.inflight, mailbox.credits) == (0, 1, 1)
    mailbox.ack()
    assert (mailbox.inflight, mailbox.credits) == (0, 2)
    assert received.payload == {"index": 1}


async def test_try_send_sheds_when_no_credit_is_left() -> None:
    mailbox = Mailbox("m", 1)
    assert mailbox.try_send(message(1))
    assert not mailbox.try_send(message(2))
    assert mailbox.rejected_total == 1
    assert mailbox.sent_total == 1


async def test_send_waits_for_a_credit_and_then_proceeds() -> None:
    mailbox = Mailbox("m", 1)
    assert mailbox.try_send(message(1))
    waiting = asyncio.create_task(mailbox.send(message(2)))
    await asyncio.sleep(0)
    assert not waiting.done()
    await mailbox.receive()
    mailbox.ack()
    await asyncio.wait_for(waiting, timeout=1)
    assert mailbox.depth == 1


async def test_send_times_out_without_consuming_a_credit() -> None:
    mailbox = Mailbox("m", 1)
    assert mailbox.try_send(message(1))
    with pytest.raises(TimeoutError):
        await mailbox.send(message(2), timeout=0.01)
    assert mailbox.credits == 0
    assert mailbox.depth == 1


async def test_waiters_are_woken_one_per_freed_credit() -> None:
    mailbox = Mailbox("m", 1)
    assert mailbox.try_send(message(0))
    first = asyncio.create_task(mailbox.send(message(1)))
    second = asyncio.create_task(mailbox.send(message(2)))
    await asyncio.sleep(0)
    await mailbox.receive()
    mailbox.ack()
    await asyncio.wait_for(first, timeout=1)
    assert not second.done()
    await mailbox.receive()
    mailbox.ack()
    await asyncio.wait_for(second, timeout=1)
    assert mailbox.depth == 1


async def test_a_cancelled_waiter_passes_its_wakeup_on() -> None:
    mailbox = Mailbox("m", 1)
    assert mailbox.try_send(message(0))
    first = asyncio.create_task(mailbox.send(message(1)))
    second = asyncio.create_task(mailbox.send(message(2)))
    await asyncio.sleep(0)
    await mailbox.receive()
    first.cancel()  # woken by the ack below, then cancelled before it could resume
    mailbox.ack()
    with pytest.raises(asyncio.CancelledError):
        await first
    await asyncio.wait_for(second, timeout=1)
    assert mailbox.depth == 1


async def test_push_front_and_requeue_place_messages_correctly() -> None:
    mailbox = Mailbox("m", 4)
    for index in range(3):
        assert mailbox.try_send(message(index))
    first = await mailbox.receive()
    mailbox.push_front(first)
    assert [item.payload["index"] for item in mailbox] == [0, 1, 2]
    again = await mailbox.receive()
    mailbox.requeue(again)
    assert [item.payload["index"] for item in mailbox] == [1, 2, 0]


async def test_returning_a_message_that_is_not_in_flight_is_a_bug() -> None:
    mailbox = Mailbox("m", 2)
    with pytest.raises(RuntimeError, match="not in flight"):
        mailbox.push_front(message(1))
    with pytest.raises(RuntimeError, match="ack without"):
        mailbox.ack()


async def test_oversized_messages_are_rejected_before_they_are_queued() -> None:
    mailbox = Mailbox("m", 2, max_message_bytes=32)
    big = Envelope.create("test.message", {"text": "x" * 200})
    with pytest.raises(MessageTooLargeError) as info:
        mailbox.try_send(big)
    assert info.value.limit == 32
    assert mailbox.depth == 0
    assert mailbox.rejected_total == 1


async def test_close_refuses_new_sends_but_keeps_queued_messages() -> None:
    mailbox = Mailbox("m", 2)
    assert mailbox.try_send(message(1))
    mailbox.close()
    assert mailbox.closed
    with pytest.raises(MailboxClosedError):
        mailbox.try_send(message(2))
    with pytest.raises(MailboxClosedError):
        await mailbox.send(message(3))
    assert (await mailbox.receive()).payload == {"index": 1}


async def test_a_waiting_producer_is_released_when_the_mailbox_closes() -> None:
    mailbox = Mailbox("m", 1)
    assert mailbox.try_send(message(1))
    waiting = asyncio.create_task(mailbox.send(message(2)))
    await asyncio.sleep(0)
    mailbox.close()
    with pytest.raises(MailboxClosedError):
        await waiting


async def test_drain_frees_every_credit_and_returns_the_messages() -> None:
    mailbox = Mailbox("m", 4)
    for index in range(3):
        assert mailbox.try_send(message(index))
    drained = mailbox.drain()
    assert [item.payload["index"] for item in drained] == [0, 1, 2]
    assert mailbox.credits == 4
    assert mailbox.idle


async def test_wait_idle_returns_once_nothing_is_queued_or_in_flight() -> None:
    mailbox = Mailbox("m", 2)
    assert mailbox.try_send(message(1))
    waiter = asyncio.create_task(mailbox.wait_idle())
    await asyncio.sleep(0)
    assert not waiter.done()
    await mailbox.receive()
    await asyncio.sleep(0)
    assert not waiter.done()  # still in flight
    mailbox.ack()
    await asyncio.wait_for(waiter, timeout=1)


@settings(max_examples=50, deadline=None)
@given(
    capacity=st.integers(min_value=1, max_value=8),
    operations=st.lists(st.sampled_from(["send", "receive", "ack", "requeue"]), max_size=40),
)
def test_credit_conservation_holds_for_any_operation_sequence(
    capacity: int, operations: list[str]
) -> None:
    """queued + in_flight + credits == capacity, always, whatever the caller does."""

    async def scenario() -> None:
        mailbox = Mailbox("m", capacity)
        inflight: list[Envelope] = []
        for index, operation in enumerate(operations):
            if operation == "send":
                mailbox.try_send(message(index))
            elif operation == "receive" and mailbox.depth:
                inflight.append(await mailbox.receive())
            elif operation == "ack" and inflight:
                inflight.pop()
                mailbox.ack()
            elif operation == "requeue" and inflight:
                mailbox.requeue(inflight.pop())
            assert mailbox.depth + mailbox.inflight + mailbox.credits == capacity
            assert 0 <= mailbox.credits <= capacity
        assert mailbox.depth + mailbox.inflight + mailbox.credits == capacity

    asyncio.run(scenario())


class MailboxMachine(RuleBasedStateMachine):
    """Stateful model: a mailbox never exceeds its bound and never loses a message."""

    def __init__(self) -> None:
        super().__init__()
        self.mailbox = Mailbox("stateful", 3)
        self.loop = asyncio.new_event_loop()
        self.inflight: list[Envelope] = []
        self.sent = 0
        self.finished = 0
        self.counter = 0

    @initialize(capacity=st.integers(min_value=1, max_value=5))
    def start(self, capacity: int) -> None:
        self.mailbox = Mailbox("stateful", capacity)

    @rule()
    def send(self) -> None:
        self.counter += 1
        if self.mailbox.try_send(message(self.counter)):
            self.sent += 1

    @precondition(lambda self: self.mailbox.depth > 0)
    @rule()
    def receive(self) -> None:
        self.inflight.append(self.loop.run_until_complete(self.mailbox.receive()))

    @precondition(lambda self: bool(self.inflight))
    @rule()
    def ack(self) -> None:
        self.inflight.pop()
        self.mailbox.ack()
        self.finished += 1

    @precondition(lambda self: bool(self.inflight))
    @rule()
    def redeliver(self) -> None:
        self.mailbox.push_front(self.inflight.pop())

    @invariant()
    def bound_holds(self) -> None:
        mailbox = self.mailbox
        assert mailbox.depth + mailbox.inflight + mailbox.credits == mailbox.capacity
        assert mailbox.depth <= mailbox.capacity
        assert self.sent - self.finished == mailbox.depth + mailbox.inflight

    def teardown(self) -> None:
        self.loop.close()


def test_a_mailbox_never_exceeds_its_bound_or_loses_a_message() -> None:
    """Randomized send/receive/ack/redeliver sequences keep the credit invariant."""
    run_state_machine_as_test(MailboxMachine, settings=settings(max_examples=40, deadline=None))
