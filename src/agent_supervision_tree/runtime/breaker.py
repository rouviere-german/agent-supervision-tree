"""Per-provider circuit breaker and bulkhead.

The breaker follows the closed / open / half-open state machine (Nygard, *Release It!*,
2007): ``failure_threshold`` consecutive failures open it; after ``reset_timeout`` seconds one
trial call is let through; a success closes it, a failure reopens it. Only provider-health
failures count (rate limits, unavailability, timeouts). Refusals and bad requests are the
caller's problem, not the provider's, and leave the breaker alone.

A half-open trial that never reports back would be worse than an open breaker, because nothing
else is admitted while it is outstanding. So a trial is handed back explicitly when the call
turns out not to run (:meth:`CircuitBreaker.release_trial`), and abandoned automatically after
``reset_timeout`` if its caller vanished without saying anything.

The bulkhead caps the number of concurrent provider calls with a semaphore, so a slow provider
can occupy at most ``size`` tasks rather than one per queued message.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Literal

__all__ = ["BreakerState", "Bulkhead", "CircuitBreaker"]

BreakerState = Literal["closed", "open", "half_open"]
_STATE_VALUE: dict[BreakerState, int] = {"closed": 0, "half_open": 1, "open": 2}


class CircuitBreaker:
    def __init__(
        self,
        *,
        failure_threshold: int = 5,
        reset_timeout: float = 5.0,
        clock: Callable[[], float],
    ) -> None:
        if failure_threshold < 1 or reset_timeout <= 0:
            raise ValueError("failure_threshold must be >= 1 and reset_timeout > 0")
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self._clock = clock
        self._state: BreakerState = "closed"
        self._consecutive_failures = 0
        self._opened_at = 0.0
        self._trial_in_flight = False
        self._trial_started_at = 0.0
        self.opened_total = 0
        self.rejected_total = 0
        self.abandoned_trials_total = 0

    @property
    def state(self) -> BreakerState:
        self._maybe_half_open()
        return self._state

    @property
    def state_value(self) -> int:
        """0 closed, 1 half-open, 2 open (for the ``breaker_state`` gauge)."""
        return _STATE_VALUE[self.state]

    @property
    def consecutive_failures(self) -> int:
        return self._consecutive_failures

    def _maybe_half_open(self) -> None:
        now = self._clock()
        if self._state == "open" and now - self._opened_at >= self.reset_timeout:
            self._state = "half_open"
            self._trial_in_flight = False
            return
        if (
            self._state == "half_open"
            and self._trial_in_flight
            and now - self._trial_started_at >= self.reset_timeout
        ):
            # The trial never reported back: its caller was cancelled, or the call never ran.
            # Staying half-open with a phantom trial would reject every caller forever, so the
            # trial is abandoned and the next caller gets it instead.
            self._trial_in_flight = False
            self.abandoned_trials_total += 1

    def allow(self) -> bool:
        """Whether a call may start now. Half-open admits exactly one trial at a time."""
        self._maybe_half_open()
        if self._state == "closed":
            return True
        if self._state == "half_open" and not self._trial_in_flight:
            self._trial_in_flight = True
            self._trial_started_at = self._clock()
            return True
        self.rejected_total += 1
        return False

    def record_success(self) -> None:
        self._consecutive_failures = 0
        self._trial_in_flight = False
        self._state = "closed"

    def record_failure(self) -> None:
        self._consecutive_failures += 1
        self._trial_in_flight = False
        if self._state == "half_open" or self._consecutive_failures >= self.failure_threshold:
            self._open()

    def _open(self) -> None:
        if self._state != "open":
            self.opened_total += 1
        self._state = "open"
        self._opened_at = self._clock()

    def release_trial(self) -> None:
        """Hand the half-open trial back because the call never ran (a refused reservation)."""
        self._trial_in_flight = False

    def reset(self) -> None:
        self._state = "closed"
        self._consecutive_failures = 0
        self._trial_in_flight = False


class Bulkhead:
    def __init__(self, size: int) -> None:
        if size < 1:
            raise ValueError("bulkhead size must be at least 1")
        self.size = size
        self._semaphore = asyncio.Semaphore(size)
        self._in_use = 0
        self.peak_in_use = 0

    @property
    def in_use(self) -> int:
        return self._in_use

    @property
    def available(self) -> int:
        return self.size - self._in_use

    async def acquire(self) -> None:
        await self._semaphore.acquire()
        self._in_use += 1
        self.peak_in_use = max(self.peak_in_use, self._in_use)

    def release(self) -> None:
        if self._in_use == 0:
            raise RuntimeError("bulkhead release without acquire")
        self._in_use -= 1
        self._semaphore.release()
