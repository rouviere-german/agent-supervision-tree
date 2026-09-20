"""The provider actor: bulkhead, breaker, retries, deadlines, budgets and model fallback."""

from __future__ import annotations

import asyncio
from collections.abc import Callable

import pytest

from agent_supervision_tree.agentic.errors import (
    ProviderError,
    ProviderRateLimitedError,
    ProviderRefusalError,
    ProviderUnavailableError,
)
from agent_supervision_tree.agentic.providers.base import (
    ChatMessage,
    LLMRequest,
    LLMResponse,
    Usage,
)
from agent_supervision_tree.agentic.tokens import estimate_tokens, simulate_usage_tokens
from agent_supervision_tree.core.canonical import canonical_json
from agent_supervision_tree.runtime.actor import Actor, ActorContext, ExitReason
from agent_supervision_tree.runtime.breaker import CircuitBreaker
from agent_supervision_tree.runtime.errors import (
    DeadlineExceededError,
    ProviderCrashedError,
    ProviderExhaustedError,
)
from agent_supervision_tree.runtime.ledger import QuotaLedger
from agent_supervision_tree.runtime.messages import Envelope
from agent_supervision_tree.runtime.provider_actor import (
    AsyncLLMProvider,
    FallbackPolicy,
    ProviderActor,
    ProviderOutcome,
    RetryPolicy,
    ThreadedProvider,
    ask_provider,
)
from agent_supervision_tree.runtime.system import ActorSystem
from helpers import actor_child, build_system, wait_for

pytestmark = pytest.mark.unit

CALLER_PATH = "root/caller"
PROVIDER_PATH = "root/provider"
Behaviour = Callable[[LLMRequest], LLMResponse] | BaseException | float


def answer(model: str = "claude-sonnet-5", output: int = 10) -> Callable[[LLMRequest], LLMResponse]:
    def build(request: LLMRequest) -> LLMResponse:
        return LLMResponse(
            model=model,
            stop_reason="end_turn",
            text="{}",
            parsed={},
            usage=Usage(input_tokens=20, output_tokens=output),
        )

    return build


class ScriptedProvider:
    """Plays a script: a builder returns a response, an exception is raised, a float sleeps."""

    name = "scripted"

    def __init__(self, script: list[Behaviour]) -> None:
        self.script = list(script)
        self.requests: list[LLMRequest] = []

    async def acomplete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        step = self.script[min(len(self.requests) - 1, len(self.script) - 1)]
        if isinstance(step, BaseException):
            raise step
        if isinstance(step, float):
            await asyncio.sleep(step)
            raise ProviderUnavailableError("the hung call finally failed")
        return step(request)


class Caller(Actor):
    """Calls the provider actor and records the outcome or the exception it got back."""

    accepts = frozenset({"call.request"})

    def __init__(self, results: list[object], *, ttl: float | None = 1.0) -> None:
        self.results = results
        self.ttl = ttl

    async def receive(self, envelope: Envelope, ctx: ActorContext) -> None:
        payload = envelope.payload if isinstance(envelope.payload, dict) else {}
        request = LLMRequest(
            messages=(ChatMessage.user(str(payload.get("text", "hello"))),),
            role="extractor",
            max_tokens=64,
        )
        deadline = None if self.ttl is None else ctx.clock() + self.ttl
        try:
            self.results.append(
                await ask_provider(
                    ctx, PROVIDER_PATH, request, ledger_path=CALLER_PATH, deadline=deadline
                )
            )
        except (
            ProviderError,
            DeadlineExceededError,
            ProviderExhaustedError,
            ProviderCrashedError,
        ) as exc:
            self.results.append(exc)


def make_ledger(tokens: int = 10_000, usd: int = 10_000) -> QuotaLedger:
    ledger = QuotaLedger("root", tokens=tokens, usd_micros=usd)
    ledger.add_node(CALLER_PATH, tokens=tokens, usd_micros=usd)
    return ledger


async def start_system(
    provider: AsyncLLMProvider,
    ledger: QuotaLedger,
    *,
    results: list[object],
    retry: RetryPolicy | None = None,
    fallback: FallbackPolicy | None = None,
    call_timeout: float = 0.5,
    breaker_threshold: int = 3,
    ttl: float | None = 1.0,
    bulkhead: int = 4,
) -> ActorSystem:
    holder: dict[str, ProviderActor] = {}

    def provider_factory() -> ProviderActor:
        actor = ProviderActor(
            provider,
            ledger=ledger,
            breaker=CircuitBreaker(
                failure_threshold=breaker_threshold,
                reset_timeout=60.0,
                clock=lambda: asyncio.get_running_loop().time(),
            ),
            bulkhead_size=bulkhead,
            retry=retry or RetryPolicy(max_attempts=2, base_delay=0.0, max_delay=0.0),
            fallback=fallback or FallbackPolicy(),
            call_timeout=call_timeout,
            model_for_role={"extractor": "claude-sonnet-5", "fallback": "claude-haiku-4-5"},
        )
        holder["actor"] = actor
        return actor

    system = build_system(
        [
            actor_child("provider", provider_factory, capacity=8, max_attempts=1),
            actor_child("caller", lambda: Caller(results, ttl=ttl), capacity=8),
        ],
        max_restarts=5,
    )
    system.state = holder  # type: ignore[attr-defined]
    await system.start()
    return system


@pytest.fixture
def results() -> list[object]:
    return []


def call(system: ActorSystem, text: str = "hello") -> Envelope:
    envelope = Envelope.create("call.request", {"text": text})
    assert system.mailbox(CALLER_PATH).try_send(envelope)
    return envelope


async def test_a_successful_call_settles_the_reservation(results: list[object]) -> None:
    provider = ScriptedProvider([answer()])
    ledger = make_ledger()
    system = await start_system(provider, ledger, results=results)
    try:
        call(system)
        assert await wait_for(lambda: bool(results))
        outcome = results[0]
        assert isinstance(outcome, ProviderOutcome)
        assert (outcome.status, outcome.role, outcome.attempts, outcome.fallbacks) == (
            "ok",
            "extractor",
            1,
            0,
        )
        node = ledger.node(CALLER_PATH)
        assert node.spent_tokens == 30  # 20 input + 10 output, as reported by the provider
        assert node.reserved_tokens == 0
        assert ledger.node("root").spent_tokens == 30
        assert ledger.check_invariants() == []
    finally:
        await system.stop(timeout=2)


async def test_a_rate_limited_call_is_retried_and_then_succeeds(results: list[object]) -> None:
    provider = ScriptedProvider([ProviderRateLimitedError("429", retry_after=0.001), answer()])
    ledger = make_ledger()
    system = await start_system(provider, ledger, results=results)
    try:
        call(system)
        assert await wait_for(lambda: bool(results))
        outcome = results[0]
        assert isinstance(outcome, ProviderOutcome)
        assert outcome.status == "ok"
        assert outcome.attempts == 2
        assert len(provider.requests) == 2
        # The failed attempt released its hold; only the successful one was settled.
        assert ledger.node(CALLER_PATH).spent_tokens == 30
        assert ledger.node(CALLER_PATH).reserved_tokens == 0
    finally:
        await system.stop(timeout=2)


async def test_a_refusal_is_raised_to_the_caller_without_retrying(results: list[object]) -> None:
    refusal = ProviderRefusalError(category="policy", model="claude-sonnet-5")
    provider = ScriptedProvider([refusal, answer()])
    ledger = make_ledger()
    system = await start_system(provider, ledger, results=results)
    try:
        call(system)
        assert await wait_for(lambda: bool(results))
        assert isinstance(results[0], ProviderRefusalError)
        assert len(provider.requests) == 1  # no retry, no fallback: the answer will not change
        assert ledger.node(CALLER_PATH).reserved_tokens == 0
    finally:
        await system.stop(timeout=2)


async def test_a_billed_failure_is_charged_to_the_budget(results: list[object]) -> None:
    billed = ProviderError(
        "truncated",
        retryable=False,
        model="claude-sonnet-5",
        usage=Usage(input_tokens=7, output_tokens=3),
    )
    provider = ScriptedProvider([billed])
    ledger = make_ledger()
    system = await start_system(provider, ledger, results=results)
    try:
        call(system)
        assert await wait_for(lambda: bool(results))
        assert ledger.node(CALLER_PATH).spent_tokens == 10
        assert system.audit.count("budget.billed_failure") == 1
    finally:
        await system.stop(timeout=2)


async def test_repeated_failures_open_the_breaker_and_the_call_is_deferred(
    results: list[object],
) -> None:
    provider = ScriptedProvider([ProviderUnavailableError("529")])
    ledger = make_ledger()
    system = await start_system(
        provider,
        ledger,
        results=results,
        retry=RetryPolicy(max_attempts=2, base_delay=0.0, max_delay=0.0),
        fallback=FallbackPolicy(chain=("fallback",), allow_defer=True),
        breaker_threshold=2,
        ttl=2.0,
    )
    try:
        call(system)
        assert await wait_for(lambda: bool(results))
        outcome = results[0]
        assert isinstance(outcome, ProviderOutcome)
        assert outcome.status == "deferred"
        assert outcome.reason == "breaker_open"
        assert outcome.fallbacks == 1
        # Two attempts opened the breaker; the fallback role never reached the provider.
        assert len(provider.requests) == 2
        assert ledger.node(CALLER_PATH).reserved_tokens == 0
    finally:
        await system.stop(timeout=2)


async def test_without_defer_an_exhausted_chain_raises(results: list[object]) -> None:
    provider = ScriptedProvider([ProviderUnavailableError("529")])
    ledger = make_ledger()
    system = await start_system(
        provider,
        ledger,
        results=results,
        fallback=FallbackPolicy(chain=(), allow_defer=False),
        ttl=2.0,
    )
    try:
        call(system)
        assert await wait_for(lambda: bool(results))
        assert isinstance(results[0], ProviderExhaustedError)
    finally:
        await system.stop(timeout=2)


async def test_a_hung_call_is_killed_by_the_call_timeout(results: list[object]) -> None:
    provider = ScriptedProvider([30.0])
    ledger = make_ledger()
    system = await start_system(
        provider,
        ledger,
        results=results,
        call_timeout=0.02,
        retry=RetryPolicy(max_attempts=1, base_delay=0.0, max_delay=0.0),
        fallback=FallbackPolicy(chain=(), allow_defer=True),
        ttl=5.0,
    )
    try:
        call(system)
        assert await wait_for(lambda: bool(results), timeout=3)
        outcome = results[0]
        assert isinstance(outcome, ProviderOutcome)
        assert outcome.status == "deferred"
        assert outcome.reason == "timeout"
        assert ledger.node(CALLER_PATH).reserved_tokens == 0  # the hold was released
        assert ledger.node(CALLER_PATH).spent_tokens == 0
    finally:
        await system.stop(timeout=2)


async def test_a_message_deadline_beats_the_call_timeout(results: list[object]) -> None:
    provider = ScriptedProvider([30.0])
    ledger = make_ledger()
    system = await start_system(provider, ledger, results=results, call_timeout=30.0, ttl=0.05)
    try:
        call(system)
        assert await wait_for(lambda: bool(results), timeout=3)
        assert isinstance(results[0], DeadlineExceededError)
    finally:
        await system.stop(timeout=2)


async def test_a_call_with_no_time_left_never_reaches_the_provider(results: list[object]) -> None:
    provider = ScriptedProvider([answer()])
    ledger = make_ledger()
    system = await start_system(provider, ledger, results=results, ttl=-1.0)
    try:
        call(system)
        assert await wait_for(lambda: bool(results))
        assert isinstance(results[0], DeadlineExceededError)
        assert provider.requests == []
    finally:
        await system.stop(timeout=2)


async def test_an_exhausted_budget_refuses_the_call_and_is_audited(results: list[object]) -> None:
    provider = ScriptedProvider([answer()])
    ledger = make_ledger(tokens=10, usd=10)
    system = await start_system(
        provider,
        ledger,
        results=results,
        fallback=FallbackPolicy(chain=(), allow_defer=True),
        ttl=2.0,
    )
    try:
        call(system)
        assert await wait_for(lambda: bool(results))
        outcome = results[0]
        assert isinstance(outcome, ProviderOutcome)
        assert outcome.status == "deferred"
        assert outcome.reason.startswith("quota:")
        assert provider.requests == []
        assert system.audit.count("budget.refused") >= 1
    finally:
        await system.stop(timeout=2)


async def test_a_budget_refusal_does_not_consume_the_half_open_trial(
    results: list[object],
) -> None:
    """A refused reservation teaches nothing about the provider, so the trial must go back."""
    provider = ScriptedProvider([answer()])
    ledger = make_ledger(tokens=10, usd=10)
    system = await start_system(
        provider,
        ledger,
        results=results,
        fallback=FallbackPolicy(chain=(), allow_defer=True),
        breaker_threshold=1,
        ttl=2.0,
    )
    actor = system.state["actor"]  # type: ignore[attr-defined]
    breaker = actor.breaker
    breaker.record_failure()  # open it
    await asyncio.sleep(0.01)
    breaker.reset_timeout = 0.0  # half-open on the next look
    try:
        call(system)
        assert await wait_for(lambda: bool(results))
        outcome = results[0]
        assert isinstance(outcome, ProviderOutcome)
        assert outcome.reason.startswith("quota:")
        assert breaker.allow()  # the trial was handed back, not burned
        assert breaker.abandoned_trials_total == 0
    finally:
        await system.stop(timeout=2)


async def test_an_exhausted_root_budget_is_a_system_wide_cost_breaker(
    results: list[object],
) -> None:
    provider = ScriptedProvider([answer()])
    ledger = make_ledger(tokens=100, usd=100)
    ledger.settle(ledger.reserve("root", tokens=100, usd_micros=100), tokens=100, usd_micros=100)
    system = await start_system(provider, ledger, results=results, ttl=2.0)
    try:
        call(system)
        assert await wait_for(lambda: bool(results))
        outcome = results[0]
        assert isinstance(outcome, ProviderOutcome)
        assert (outcome.status, outcome.reason) == ("deferred", "cost_breaker_open")
        assert provider.requests == []
    finally:
        await system.stop(timeout=2)


async def test_a_crashing_provider_actor_fails_in_flight_calls(results: list[object]) -> None:
    provider = ScriptedProvider([30.0])
    ledger = make_ledger()
    system = await start_system(provider, ledger, results=results, call_timeout=30.0, ttl=None)
    try:
        call(system)
        assert await wait_for(lambda: bool(provider.requests))
        system.kill(PROVIDER_PATH, ExitReason.killed("chaos"))
        assert await wait_for(lambda: bool(results), timeout=3)
        assert isinstance(results[0], ProviderCrashedError)
    finally:
        await system.stop(timeout=2)


async def test_the_fallback_role_is_tried_with_its_own_model(results: list[object]) -> None:
    provider = ScriptedProvider(
        [
            ProviderUnavailableError("529"),
            ProviderUnavailableError("529"),
            answer("claude-haiku-4-5"),
        ]
    )
    ledger = make_ledger()
    system = await start_system(
        provider,
        ledger,
        results=results,
        retry=RetryPolicy(max_attempts=2, base_delay=0.0, max_delay=0.0),
        fallback=FallbackPolicy(chain=("fallback",), allow_defer=True),
        breaker_threshold=5,
        ttl=2.0,
    )
    try:
        call(system)
        assert await wait_for(lambda: bool(results), timeout=3)
        outcome = results[0]
        assert isinstance(outcome, ProviderOutcome)
        assert (outcome.status, outcome.role, outcome.fallbacks) == ("ok", "fallback", 1)
        assert outcome.model == "claude-haiku-4-5"
        assert [request.role for request in provider.requests] == [
            "extractor",
            "extractor",
            "fallback",
        ]
    finally:
        await system.stop(timeout=2)


async def test_the_bulkhead_limits_concurrent_calls(results: list[object]) -> None:
    started = asyncio.Event()
    concurrent = 0
    peak = 0

    class Watching(ScriptedProvider):
        async def acomplete(self, request: LLMRequest) -> LLMResponse:
            nonlocal concurrent, peak
            concurrent += 1
            peak = max(peak, concurrent)
            started.set()
            try:
                await asyncio.sleep(0.02)
                return answer()(request)
            finally:
                concurrent -= 1

    provider = Watching([answer()])
    ledger = make_ledger()
    system = await start_system(provider, ledger, results=results, bulkhead=2, ttl=5.0)
    try:
        for index in range(6):
            call(system, text=f"message {index}")
        assert await wait_for(lambda: len(results) == 6, timeout=5)
        assert peak <= 2
    finally:
        await system.stop(timeout=2)


def test_retry_policy_bounds_the_delay() -> None:
    import random

    policy = RetryPolicy(max_attempts=3, base_delay=0.1, max_delay=0.4)
    rng = random.Random(0)
    for attempt in range(1, 4):
        delay = policy.delay(attempt, rng, None)
        assert 0.0 <= delay <= 0.4
    assert policy.delay(1, rng, 2.0) >= 2.0  # retry-after is a floor
    with pytest.raises(ValueError, match="invalid retry policy"):
        RetryPolicy(max_attempts=0)
    with pytest.raises(ValueError, match="invalid retry policy"):
        RetryPolicy(base_delay=1.0, max_delay=0.1)


def test_provider_actor_arguments_are_validated() -> None:
    ledger = make_ledger()
    breaker = CircuitBreaker(failure_threshold=1, reset_timeout=1, clock=lambda: 0.0)
    with pytest.raises(ValueError, match="bulkhead_size"):
        ProviderActor(ScriptedProvider([]), ledger=ledger, breaker=breaker, bulkhead_size=0)
    with pytest.raises(ValueError, match="bulkhead_size"):
        ProviderActor(ScriptedProvider([]), ledger=ledger, breaker=breaker, call_timeout=0)


async def test_threaded_provider_adapts_a_synchronous_client() -> None:
    class SyncProvider:
        name = "sync"

        def complete(self, request: LLMRequest) -> LLMResponse:
            return answer()(request)

    adapted = ThreadedProvider(SyncProvider())
    assert adapted.name == "threaded:sync"
    response = await adapted.acomplete(
        LLMRequest(messages=(ChatMessage.user("hi"),), max_tokens=16)
    )
    assert response.usage.total_tokens == 30


class OvershootingProvider:
    """Reports more usage than the request allowed: an unreliable or hostile provider."""

    name = "overshooting"

    def __init__(self, output_tokens: int) -> None:
        self.output_tokens = output_tokens
        self.requests: list[LLMRequest] = []

    async def acomplete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        return LLMResponse(
            model="claude-sonnet-5",
            stop_reason="end_turn",
            text="{}",
            parsed={},
            usage=Usage(input_tokens=20, output_tokens=self.output_tokens),
        )


async def test_a_provider_that_bills_more_than_it_was_allowed_is_audited_and_cuts_off(
    results: list[object],
) -> None:
    """The overshoot path is live: spend above the hold is counted, audited and acted on."""
    provider = OvershootingProvider(output_tokens=1_000)
    ledger = make_ledger(tokens=500, usd=10_000)
    system = await start_system(
        provider,
        ledger,
        results=results,
        fallback=FallbackPolicy(chain=(), allow_defer=True),
        ttl=2.0,
    )
    try:
        call(system)
        assert await wait_for(lambda: bool(results))
        assert isinstance(results[0], ProviderOutcome)
        assert results[0].status == "ok"
        node = ledger.node(CALLER_PATH)
        assert node.spent_tokens == 1_020  # reported usage, not the hold: nothing is hidden
        assert node.overshoot_total == 1
        assert node.overshoot_tokens > 0
        assert system.audit.count("budget.overshoot") == 1
        # The ledger authorized only what fitted, so its own invariant still holds ...
        assert ledger.check_invariants() == []
        assert node.authorized_tokens <= node.limit_tokens
        assert ledger.overshoots() == ["root", CALLER_PATH]
        # ... and the overshoot immediately stops the next call: a cost breaker, not a surprise.
        call(system, "second")
        assert await wait_for(lambda: len(results) == 2)
        assert isinstance(results[1], ProviderOutcome)
        assert results[1].status == "deferred"
        assert results[1].reason == "cost_breaker_open"  # the root node is exhausted
        assert len(provider.requests) == 1
    finally:
        await system.stop(timeout=2)


async def test_the_reservation_is_larger_than_the_usage_a_realistic_provider_reports() -> None:
    """The reservation must not be the same expression as the reported usage."""
    request = LLMRequest(
        messages=(ChatMessage.user("invoice 4711, total 12.50 EUR, due 2026-01-31"),),
        role="extractor",
        max_tokens=64,
    )
    rendered = canonical_json(request.model_dump(mode="json"))
    assert estimate_tokens(rendered) > simulate_usage_tokens(rendered)
