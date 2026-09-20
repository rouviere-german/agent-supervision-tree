"""Bounded mailboxes with credit-based backpressure.

A mailbox holds ``capacity`` credits. Sending consumes one; the credit is returned only when the
receiver *acknowledges* the message after processing it, not when it dequeues it. So at every
moment ``queued + in_flight + credits == capacity``: the bound covers work that has left the
queue but is not finished, which is what limits memory and outstanding provider calls.

Producers either wait for a credit (:meth:`Mailbox.send`) or are told there is none
(:meth:`Mailbox.try_send`), which is how the ingress turns overload into a 429 instead of an
unbounded queue. A mailbox belongs to the runtime, not to the actor instance consuming it, so
it survives the actor's restart; in-flight messages of a crashed actor are put back at the front
(:meth:`Mailbox.push_front`) for the replacement, and messages a behavior rejects are requeued at
the back (:meth:`Mailbox.requeue`) so one poison message does not starve the rest.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections import deque
from collections.abc import Iterator

from agent_supervision_tree.runtime.errors import MailboxClosedError, MessageTooLargeError
from agent_supervision_tree.runtime.messages import Envelope

__all__ = ["Mailbox"]


class Mailbox:
    def __init__(self, name: str, capacity: int, *, max_message_bytes: int | None = None) -> None:
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        self.name = name
        self.capacity = capacity
        self.max_message_bytes = max_message_bytes
        self._items: deque[Envelope] = deque()
        self._credits = capacity
        self._inflight = 0
        self._closed = False
        self._credit_waiters: deque[asyncio.Future[None]] = deque()
        self._item_waiters: deque[asyncio.Future[None]] = deque()
        self._drained_waiters: deque[asyncio.Future[None]] = deque()
        self.sent_total = 0
        self.rejected_total = 0

    # --- introspection -------------------------------------------------------------------

    @property
    def depth(self) -> int:
        return len(self._items)

    @property
    def inflight(self) -> int:
        return self._inflight

    @property
    def credits(self) -> int:
        return self._credits

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def idle(self) -> bool:
        """No queued and no in-flight messages."""
        return not self._items and self._inflight == 0

    def __iter__(self) -> Iterator[Envelope]:
        return iter(tuple(self._items))

    # --- producers -----------------------------------------------------------------------

    def _check(self, envelope: Envelope) -> None:
        if self._closed:
            raise MailboxClosedError(f"mailbox {self.name} is closed")
        if self.max_message_bytes is not None and envelope.size_bytes > self.max_message_bytes:
            self.rejected_total += 1
            raise MessageTooLargeError(envelope.size_bytes, self.max_message_bytes)

    def _enqueue(self, envelope: Envelope) -> None:
        self._credits -= 1
        self._items.append(envelope)
        self.sent_total += 1
        _wake_one(self._item_waiters)

    def try_send(self, envelope: Envelope) -> bool:
        """Send without waiting. False means no credit is available (shed the message)."""
        self._check(envelope)
        if self._credits == 0:
            self.rejected_total += 1
            return False
        self._enqueue(envelope)
        return True

    async def send(self, envelope: Envelope, *, timeout: float | None = None) -> None:
        """Send, waiting for a credit. Raises TimeoutError when ``timeout`` elapses first."""
        self._check(envelope)
        async with asyncio.timeout(timeout):
            while self._credits == 0:
                await _wait(self._credit_waiters)
                if self._closed:
                    raise MailboxClosedError(f"mailbox {self.name} is closed")
        self._enqueue(envelope)

    # --- consumers -----------------------------------------------------------------------

    async def receive(self) -> Envelope:
        """Dequeue the next message; it stays counted as in flight until ``ack``."""
        while not self._items:
            await _wait(self._item_waiters)
        envelope = self._items.popleft()
        self._inflight += 1
        return envelope

    def ack(self) -> None:
        """Finish one in-flight message and return its credit to producers."""
        if self._inflight == 0:
            raise RuntimeError("ack without an in-flight message")
        self._inflight -= 1
        self._credits += 1
        _wake_one(self._credit_waiters)
        self._notify_drained()

    def push_front(self, envelope: Envelope) -> None:
        """Return an in-flight message to the head of the queue (redelivery after a crash)."""
        self._return(envelope, front=True)

    def requeue(self, envelope: Envelope) -> None:
        """Return an in-flight message to the tail of the queue (retry after a nack)."""
        self._return(envelope, front=False)

    def _return(self, envelope: Envelope, *, front: bool) -> None:
        if self._inflight == 0:
            raise RuntimeError("cannot return a message that is not in flight")
        self._inflight -= 1
        if front:
            self._items.appendleft(envelope)
        else:
            self._items.append(envelope)
        _wake_one(self._item_waiters)

    def release_inflight(self) -> None:
        """Drop one in-flight message for good (quarantined or expired) and free its credit."""
        self.ack()

    # --- lifecycle -----------------------------------------------------------------------

    def close(self) -> None:
        """Refuse new sends. Queued messages can still be received and acknowledged."""
        self._closed = True
        _wake_all(self._credit_waiters)
        self._notify_drained()

    def drain(self) -> list[Envelope]:
        """Remove every queued message (used when a mailbox is abandoned) and free credits."""
        items = list(self._items)
        self._items.clear()
        self._credits += len(items)
        _wake_all(self._credit_waiters)
        self._notify_drained()
        return items

    async def wait_idle(self) -> None:
        """Wait until nothing is queued or in flight."""
        while not self.idle:
            await _wait(self._drained_waiters)

    def _notify_drained(self) -> None:
        if self.idle:
            _wake_all(self._drained_waiters)


async def _wait(waiters: deque[asyncio.Future[None]]) -> None:
    future: asyncio.Future[None] = asyncio.get_running_loop().create_future()
    waiters.append(future)
    try:
        await future
    except asyncio.CancelledError:
        if future.done() and not future.cancelled():
            # Woken, then cancelled before resuming: pass the wakeup on so it is not lost.
            _wake_one(waiters)
        raise
    finally:
        if not future.done():
            future.cancel()
        with contextlib.suppress(ValueError):  # already popped by a waker
            waiters.remove(future)


def _wake_one(waiters: deque[asyncio.Future[None]]) -> None:
    while waiters:
        future = waiters.popleft()
        if not future.done():
            future.set_result(None)
            return


def _wake_all(waiters: deque[asyncio.Future[None]]) -> None:
    while waiters:
        future = waiters.popleft()
        if not future.done():
            future.set_result(None)
