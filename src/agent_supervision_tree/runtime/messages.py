"""Envelopes: every message carries an id, an attempt count, a deadline and trace context.

The id is assigned once, at ingress, and kept by every stage that forwards the message, so a
sink can deduplicate redeliveries and the delivery invariant (*every accepted id ends exactly
once in a sink or in the quarantine*) can be checked per id. Deadlines are absolute values on
the event loop's monotonic clock; ``remaining`` is what a provider call may still spend.
"""

from __future__ import annotations

import asyncio
import re
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any

from opentelemetry import context as otel_context
from opentelemetry.propagate import extract, inject
from pydantic import JsonValue

from agent_supervision_tree.core.canonical import CanonicalJSONError, canonical_json
from agent_supervision_tree.runtime.errors import MessageTooLargeError

__all__ = ["KIND_PATTERN", "Envelope", "json_size", "now"]

KIND_PATTERN = re.compile(r"^[a-z][a-z0-9_.]{0,63}$")


def now() -> float:
    """The event loop's monotonic clock (falls back to ``time.monotonic`` outside a loop)."""
    try:
        return asyncio.get_running_loop().time()
    except RuntimeError:
        import time  # noqa: PLC0415 - only needed outside an event loop

        return time.monotonic()


def json_size(payload: object) -> int:
    """UTF-8 size of the canonical JSON of ``payload``; 0 for values outside the JSON model.

    Internal control messages (replies carrying futures, supervisor commands) are not JSON and
    never cross the ingress boundary, so they are not size-limited.
    """
    try:
        return len(canonical_json(payload).encode("utf-8"))
    except CanonicalJSONError:
        return 0


def _current_traceparent() -> str | None:
    carrier: dict[str, str] = {}
    inject(carrier)
    return carrier.get("traceparent")


@dataclass(frozen=True, slots=True)
class Envelope:
    """An immutable message. Attempts and stages produce copies, never mutations."""

    kind: str
    payload: Any
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    attempt: int = 1
    deadline: float | None = None
    traceparent: str | None = field(default_factory=_current_traceparent)
    created_at: float = field(default_factory=now)
    size_bytes: int = 0

    def __post_init__(self) -> None:
        if not KIND_PATTERN.fullmatch(self.kind):
            raise ValueError(f"invalid message kind {self.kind!r}")
        if self.attempt < 1:
            raise ValueError("attempt starts at 1")

    @classmethod
    def create(
        cls,
        kind: str,
        payload: Mapping[str, JsonValue],
        *,
        ttl: float | None = None,
        max_bytes: int | None = None,
        message_id: str | None = None,
    ) -> Envelope:
        """Build an ingress message with a JSON payload, enforcing the size limit up front."""
        size = json_size(dict(payload))
        if max_bytes is not None and size > max_bytes:
            raise MessageTooLargeError(size, max_bytes)
        deadline = now() + ttl if ttl is not None else None
        return cls(
            kind=kind,
            payload=dict(payload),
            id=message_id or uuid.uuid4().hex,
            deadline=deadline,
            size_bytes=size,
        )

    def derive(self, kind: str, payload: Any) -> Envelope:
        """A message for the next stage: same id, deadline and trace; attempt restarts at 1."""
        return Envelope(
            kind=kind,
            payload=payload,
            id=self.id,
            deadline=self.deadline,
            traceparent=self.traceparent,
            created_at=self.created_at,
            size_bytes=json_size(payload),
        )

    def next_attempt(self) -> Envelope:
        return replace(self, attempt=self.attempt + 1)

    def remaining(self, at: float | None = None) -> float | None:
        """Seconds left before the deadline (negative when expired), or None without one."""
        if self.deadline is None:
            return None
        return self.deadline - (now() if at is None else at)

    def expired(self, at: float | None = None) -> bool:
        remaining = self.remaining(at)
        return remaining is not None and remaining <= 0

    def otel_context(self) -> otel_context.Context:
        """The propagated trace context (an empty context when the message carries none)."""
        if self.traceparent is None:
            return otel_context.Context()
        return extract({"traceparent": self.traceparent})
