"""Exception hierarchy of the actor runtime.

Two families matter to actors:

- :class:`MessageFailed` is raised by a behavior to say *this message* could not be processed
  (malformed model output, a document that fails its schema). The runtime records the attempt
  and either redelivers the message or quarantines it; the actor keeps running.
- Any other exception raised by a behavior is an *actor* failure: the actor exits, its
  supervisor decides whether to restart it, and unacknowledged messages are redelivered to the
  replacement (see :mod:`agent_supervision_tree.runtime.actor`).
"""

from __future__ import annotations

__all__ = [
    "ActorExit",
    "ActorNotRunningError",
    "ActorSystemStoppedError",
    "DeadlineExceededError",
    "MailboxClosedError",
    "MessageFailed",
    "MessageTooLargeError",
    "ProviderCrashedError",
    "ProviderExhaustedError",
    "QuarantineFullError",
    "QuotaExceededError",
    "SupervisionError",
]


class SupervisionError(Exception):
    """Base class for every error raised by the runtime."""


class MailboxClosedError(SupervisionError):
    """The mailbox no longer accepts messages (the system is draining)."""


class MessageTooLargeError(SupervisionError):
    """A message payload exceeds the configured size limit."""

    def __init__(self, size: int, limit: int) -> None:
        super().__init__(f"message of {size} bytes exceeds the limit of {limit} bytes")
        self.size = size
        self.limit = limit


class MessageFailed(SupervisionError):  # noqa: N818 - deliberately not an *Error: it is a signal
    """Raised by a behavior when the current message cannot be processed.

    ``reason`` is a short, low-cardinality label recorded on redelivery and quarantine. A
    non-retryable failure (a deterministic refusal, an invalid document) is quarantined at once
    instead of using up the remaining attempts.
    """

    def __init__(self, reason: str, detail: str = "", *, retryable: bool = True) -> None:
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail
        self.retryable = retryable


class ActorExit(SupervisionError):  # noqa: N818 - a control-flow signal, not an error
    """Raised by a behavior to stop its actor with a normal exit."""


class ActorNotRunningError(SupervisionError):
    """A path names no running actor (never spawned, stopped, or between restarts)."""


class ActorSystemStoppedError(SupervisionError):
    """The system is not running."""


class DeadlineExceededError(SupervisionError):
    """A message or call ran past its deadline."""


class QuotaExceededError(SupervisionError):
    """A reservation would exceed the remainder of a node or one of its ancestors."""

    def __init__(self, path: str, dimension: str, requested: int, remaining: int) -> None:
        super().__init__(
            f"quota {dimension} exceeded at {path}: requested {requested}, remaining {remaining}"
        )
        self.path = path
        self.dimension = dimension
        self.requested = requested
        self.remaining = remaining


class ProviderCrashedError(SupervisionError):
    """The provider actor exited while a call was in flight; the caller may retry."""


class ProviderExhaustedError(SupervisionError):
    """Every model in the fallback chain failed and the policy does not allow deferring."""


class QuarantineFullError(SupervisionError):
    """The quarantine is at capacity; the record was dropped and counted."""
