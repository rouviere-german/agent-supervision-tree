"""The provider circuit breaker and the bulkhead."""

from __future__ import annotations

import asyncio

import pytest

from agent_supervision_tree.runtime.breaker import Bulkhead, CircuitBreaker
from helpers import FakeClock

pytestmark = pytest.mark.unit


def make_breaker(clock: FakeClock, *, threshold: int = 3, reset: float = 5.0) -> CircuitBreaker:
    return CircuitBreaker(failure_threshold=threshold, reset_timeout=reset, clock=clock)


def state(breaker: CircuitBreaker) -> str:
    """The breaker state as a plain string (a Literal would be narrowed by the type checker)."""
    return str(breaker.state)


def test_arguments_are_validated() -> None:
    with pytest.raises(ValueError, match="failure_threshold"):
        CircuitBreaker(failure_threshold=0, reset_timeout=1, clock=FakeClock())
    with pytest.raises(ValueError, match="failure_threshold"):
        CircuitBreaker(failure_threshold=1, reset_timeout=0, clock=FakeClock())
    with pytest.raises(ValueError, match="bulkhead size"):
        Bulkhead(0)


def test_consecutive_failures_open_the_breaker() -> None:
    clock = FakeClock()
    breaker = make_breaker(clock)
    for _ in range(2):
        breaker.record_failure()
    assert state(breaker) == "closed"
    assert breaker.allow()
    breaker.record_failure()
    assert state(breaker) == "open"
    assert breaker.state_value == 2
    assert not breaker.allow()
    assert breaker.rejected_total == 1
    assert breaker.opened_total == 1


def test_a_success_resets_the_failure_streak() -> None:
    clock = FakeClock()
    breaker = make_breaker(clock)
    breaker.record_failure()
    breaker.record_failure()
    breaker.record_success()
    assert breaker.consecutive_failures == 0
    breaker.record_failure()
    assert state(breaker) == "closed"


def test_half_open_admits_exactly_one_trial_call() -> None:
    clock = FakeClock()
    breaker = make_breaker(clock, threshold=1, reset=5.0)
    breaker.record_failure()
    assert state(breaker) == "open"
    clock.advance(5.0)
    assert state(breaker) == "half_open"
    assert breaker.state_value == 1
    assert breaker.allow()  # the trial
    assert not breaker.allow()  # nobody else gets through
    breaker.record_success()
    assert state(breaker) == "closed"
    assert breaker.allow()


def test_a_failed_trial_reopens_the_breaker_immediately() -> None:
    clock = FakeClock()
    breaker = make_breaker(clock, threshold=5, reset=2.0)
    for _ in range(5):
        breaker.record_failure()
    clock.advance(2.0)
    assert breaker.allow()
    breaker.record_failure()  # one failure is enough while half-open
    assert state(breaker) == "open"
    assert breaker.opened_total == 2
    clock.advance(1.0)
    assert state(breaker) == "open"


def test_a_trial_that_never_reports_back_is_abandoned_instead_of_blocking_forever() -> None:
    """A phantom trial would reject every caller for good; the breaker must recover on its own."""
    clock = FakeClock()
    breaker = make_breaker(clock, threshold=1, reset=2.0)
    breaker.record_failure()
    clock.advance(2.0)
    assert breaker.allow()  # the trial starts...
    assert not breaker.allow()  # ... and holds the slot
    clock.advance(2.0)  # its caller vanished without reporting success or failure
    assert breaker.allow()
    assert breaker.abandoned_trials_total == 1


def test_a_trial_can_be_handed_back_when_the_call_never_ran() -> None:
    clock = FakeClock()
    breaker = make_breaker(clock, threshold=1, reset=2.0)
    breaker.record_failure()
    clock.advance(2.0)
    assert breaker.allow()
    breaker.release_trial()  # the budget refused the call, so nothing was learned
    assert breaker.allow()
    assert state(breaker) == "half_open"
    assert breaker.abandoned_trials_total == 0


def test_reset_closes_the_breaker_by_hand() -> None:
    clock = FakeClock()
    breaker = make_breaker(clock, threshold=1)
    breaker.record_failure()
    breaker.reset()
    assert state(breaker) == "closed"
    assert breaker.consecutive_failures == 0


async def test_the_bulkhead_caps_concurrency_and_tracks_its_peak() -> None:
    bulkhead = Bulkhead(2)
    await bulkhead.acquire()
    await bulkhead.acquire()
    assert (bulkhead.in_use, bulkhead.available) == (2, 0)
    waiting = asyncio.create_task(bulkhead.acquire())
    await asyncio.sleep(0)
    assert not waiting.done()
    bulkhead.release()
    await asyncio.wait_for(waiting, timeout=1)
    assert bulkhead.peak_in_use == 2
    bulkhead.release()
    bulkhead.release()
    with pytest.raises(RuntimeError, match="release without acquire"):
        bulkhead.release()
