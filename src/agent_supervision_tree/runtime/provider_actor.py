"""The provider actor: bulkhead, circuit breaker, jittered retries, deadline kills, fallback.

Workers do not call the model directly. They send an ``llm.call`` message carrying a reply
future (:func:`ask_provider`), and the provider actor runs the call under its policies:

1. A **bulkhead** semaphore caps concurrent calls; the actor's bounded mailbox caps waiting ones.
2. The system-wide **cost breaker** (root ledger node exhausted) answers with a deterministic
   *defer* instead of calling any model.
3. Per attempt: the **circuit breaker** must admit the call; an upper-bound **reservation** is
   taken on the caller's ledger node (and so on every ancestor); the call runs under
   ``asyncio.timeout`` bounded by both the call timeout and the message deadline, so a hung
   call is cancelled rather than waited for; the reservation is settled with the billed usage
   (also for billed failures) or released.
4. Retryable failures (429, 5xx, timeouts) back off with full jitter (Brooker, "Exponential
   Backoff And Jitter", AWS Architecture Blog, 2015), honouring ``retry-after`` as a floor.
5. When the attempts for a role are used up, the next role in the **fallback chain** is tried
   (``claude-sonnet-5`` then ``claude-haiku-4-5`` in the reference workload); when the chain is
   exhausted the policy either defers or raises :class:`ProviderExhaustedError`.

Non-retryable errors (refusals, invalid requests, bad credentials) are raised to the caller at
once: no amount of retrying or model switching fixes them. If the provider actor itself
crashes, every in-flight reply future fails with :class:`ProviderCrashedError`, so callers are
never left waiting on a dead actor.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

from agent_supervision_tree.agentic.errors import ProviderError, ProviderRateLimitedError
from agent_supervision_tree.agentic.pricing import cost_usd_micros
from agent_supervision_tree.agentic.providers.base import LLMProvider, LLMRequest, LLMResponse
from agent_supervision_tree.agentic.providers.fake import estimate_tokens
from agent_supervision_tree.core.canonical import canonical_json
from agent_supervision_tree.runtime.actor import Actor, ActorContext
from agent_supervision_tree.runtime.breaker import Bulkhead, CircuitBreaker
from agent_supervision_tree.runtime.errors import (
    DeadlineExceededError,
    ProviderCrashedError,
    ProviderExhaustedError,
    QuotaExceededError,
)
from agent_supervision_tree.runtime.ledger import QuotaLedger, Reservation
from agent_supervision_tree.runtime.messages import Envelope

__all__ = [
    "CALL_KIND",
    "AsyncLLMProvider",
    "FallbackPolicy",
    "LLMCall",
    "ProviderActor",
    "ProviderOutcome",
    "RetryPolicy",
    "ThreadedProvider",
    "ask_provider",
]

CALL_KIND = "llm.call"
_DEFAULT_MODEL = "claude-opus-5"


@runtime_checkable
class AsyncLLMProvider(Protocol):
    name: str

    async def acomplete(self, request: LLMRequest) -> LLMResponse: ...


class ThreadedProvider:
    """Adapt a synchronous :class:`LLMProvider` (the Anthropic SDK client) to the async protocol.

    A thread cannot be cancelled: on a deadline kill the caller stops waiting while the thread
    finishes on its own. The SDK's own timeout bounds how long that takes.
    """

    def __init__(self, inner: LLMProvider) -> None:
        self._inner = inner
        self.name = f"threaded:{inner.name}"

    async def acomplete(self, request: LLMRequest) -> LLMResponse:
        return await asyncio.to_thread(self._inner.complete, request)


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    max_attempts: int = 3
    base_delay: float = 0.05
    max_delay: float = 1.0

    def __post_init__(self) -> None:
        if self.max_attempts < 1 or self.base_delay < 0 or self.max_delay < self.base_delay:
            raise ValueError("invalid retry policy")

    def delay(self, attempt: int, rng: random.Random, retry_after: float | None) -> float:
        cap = min(self.max_delay, self.base_delay * (2 ** (attempt - 1)))
        jittered = rng.uniform(0.0, cap)
        return max(jittered, retry_after or 0.0)


@dataclass(frozen=True, slots=True)
class FallbackPolicy:
    """Roles to try after the request's own role, and whether deferring is allowed."""

    chain: tuple[str, ...] = ()
    allow_defer: bool = True


@dataclass(frozen=True, slots=True)
class ProviderOutcome:
    status: Literal["ok", "deferred"]
    role: str
    model: str
    attempts: int
    fallbacks: int
    response: LLMResponse | None = None
    reason: str = ""


@dataclass(eq=False)
class LLMCall:
    request: LLMRequest
    ledger_path: str
    deadline: float | None
    reply: asyncio.Future[ProviderOutcome] = field(repr=False)


async def ask_provider(
    ctx: ActorContext,
    provider_path: str,
    request: LLMRequest,
    *,
    ledger_path: str,
    deadline: float | None,
) -> ProviderOutcome:
    """Send a call to the provider actor and wait for its reply within the deadline."""
    loop = asyncio.get_running_loop()
    call = LLMCall(request, ledger_path, deadline, loop.create_future())
    envelope = Envelope(kind=CALL_KIND, payload=call, deadline=deadline)
    remaining = None if deadline is None else deadline - ctx.clock()
    if remaining is not None and remaining <= 0:
        raise DeadlineExceededError("no time left to call the provider")
    try:
        await ctx.send(provider_path, envelope, timeout=remaining)
        async with asyncio.timeout(remaining):
            return await call.reply
    except TimeoutError as exc:
        call.reply.cancel()
        raise DeadlineExceededError("provider call ran past the message deadline") from exc


class ProviderActor(Actor):
    accepts = frozenset({CALL_KIND})

    def __init__(
        self,
        provider: AsyncLLMProvider,
        *,
        ledger: QuotaLedger,
        breaker: CircuitBreaker,
        bulkhead_size: int = 4,
        retry: RetryPolicy | None = None,
        fallback: FallbackPolicy | None = None,
        call_timeout: float = 5.0,
        model_for_role: Mapping[str, str] | None = None,
        seed: int = 0,
    ) -> None:
        if bulkhead_size < 1 or call_timeout <= 0:
            raise ValueError("bulkhead_size must be >= 1 and call_timeout > 0")
        self.provider = provider
        self.ledger = ledger
        self.breaker = breaker
        self.retry = retry or RetryPolicy()
        self.fallback = fallback or FallbackPolicy()
        self.call_timeout = call_timeout
        self.model_for_role = dict(model_for_role or {})
        self._bulkhead_size = bulkhead_size
        self._rng = random.Random(seed)  # noqa: S311  # nosec B311 - jitter, not cryptography
        self._bulkhead: Bulkhead | None = None
        self.calls_total = 0

    @property
    def bulkhead(self) -> Bulkhead:
        if self._bulkhead is None:
            self._bulkhead = Bulkhead(self._bulkhead_size)
        return self._bulkhead

    async def on_start(self, ctx: ActorContext) -> None:
        self._bulkhead = Bulkhead(self._bulkhead_size)

    async def receive(self, envelope: Envelope, ctx: ActorContext) -> None:
        call = envelope.payload
        if not isinstance(call, LLMCall) or call.reply.done():
            return
        await self.bulkhead.acquire()
        ctx.spawn(self._execute(call, ctx), name=f"llm-call:{envelope.id[:8]}")

    async def _execute(self, call: LLMCall, ctx: ActorContext) -> None:
        task = asyncio.current_task()
        if task is not None:
            # A caller that gives up (deadline) cancels the reply, which cancels the call.
            call.reply.add_done_callback(lambda f: task.cancel() if f.cancelled() else None)
        try:
            outcome = await self._run(call, ctx)
        except asyncio.CancelledError:
            if not call.reply.done():
                call.reply.set_exception(ProviderCrashedError("provider actor stopped"))
            raise
        except (ProviderError, DeadlineExceededError, ProviderExhaustedError) as exc:
            if not call.reply.done():
                call.reply.set_exception(exc)
        else:
            if not call.reply.done():
                call.reply.set_result(outcome)
        finally:
            self.bulkhead.release()

    async def _run(self, call: LLMCall, ctx: ActorContext) -> ProviderOutcome:
        roles = (call.request.role, *self.fallback.chain)
        attempts = 0
        reason = "exhausted"
        for index, role in enumerate(roles):
            if index:
                ctx.system.metrics.provider_fallbacks_total.labels(roles[index - 1], role).inc()
            if self.ledger.exhausted(self.ledger.root):
                reason = "cost_breaker_open"
                break
            model = self.model_for_role.get(role, _DEFAULT_MODEL)
            response, used, reason = await self._attempt_role(call, role, model, ctx)
            attempts += used
            if response is not None:
                return ProviderOutcome("ok", role, model, attempts, index, response=response)
        if not self.fallback.allow_defer:
            raise ProviderExhaustedError(f"no model could serve the call ({reason})")
        ctx.system.metrics.provider_deferred_total.labels(reason.split(":")[0]).inc()
        return ProviderOutcome("deferred", roles[-1], "", attempts, len(roles) - 1, reason=reason)

    async def _attempt_role(
        self, call: LLMCall, role: str, model: str, ctx: ActorContext
    ) -> tuple[LLMResponse | None, int, str]:
        """Try one role up to ``max_attempts`` times. Returns (response, attempts, reason)."""
        request = call.request.model_copy(update={"role": role})
        reason = "exhausted"
        for attempt in range(1, self.retry.max_attempts + 1):
            remaining = self._remaining(call, ctx)
            if not self.breaker.allow():
                return None, attempt - 1, "breaker_open"
            reservation = self._reserve(call, request, model, ctx)
            if reservation is None:
                # The call never happened, so a half-open trial must go back to the breaker.
                self.breaker.release_trial()
                return None, attempt - 1, f"quota:{call.ledger_path}"
            self.calls_total += 1
            response, retry_after, reason = await self._call_once(
                request, model, reservation, remaining, ctx
            )
            if response is not None:
                return response, attempt, "ok"
            if attempt < self.retry.max_attempts:
                delay = self.retry.delay(attempt, self._rng, retry_after)
                await asyncio.sleep(min(delay, self._remaining(call, ctx)))
        return None, self.retry.max_attempts, reason

    def _remaining(self, call: LLMCall, ctx: ActorContext) -> float:
        if call.deadline is None:
            return self.call_timeout
        remaining = call.deadline - ctx.clock()
        if remaining <= 0:
            raise DeadlineExceededError("message deadline passed before the provider call")
        return min(remaining, self.call_timeout)

    def _reserve(
        self, call: LLMCall, request: LLMRequest, model: str, ctx: ActorContext
    ) -> Reservation | None:
        input_estimate = estimate_tokens(canonical_json(request.model_dump(mode="json")))
        tokens = input_estimate + request.max_tokens
        usd = cost_usd_micros(model, input_estimate, request.max_tokens)
        try:
            return self.ledger.reserve(call.ledger_path, tokens=tokens, usd_micros=usd)
        except QuotaExceededError as exc:
            ctx.system.metrics.budget_refusals_total.labels(exc.path).inc()
            ctx.audit(
                "budget.refused",
                {
                    "node": exc.path,
                    "dimension": exc.dimension,
                    "requested": exc.requested,
                    "remaining": exc.remaining,
                    "caller": call.ledger_path,
                },
            )
            return None

    async def _call_once(
        self,
        request: LLMRequest,
        model: str,
        reservation: Reservation,
        remaining: float,
        ctx: ActorContext,
    ) -> tuple[LLMResponse | None, float | None, str]:
        metrics = ctx.system.metrics
        started = ctx.clock()
        outcome = "ok"
        retry_after: float | None = None
        response: LLMResponse | None = None
        try:
            async with asyncio.timeout(remaining):
                response = await self.provider.acomplete(request)
        except TimeoutError:
            self.ledger.release(reservation)
            self.breaker.record_failure()
            outcome = "timeout"
        except ProviderError as exc:
            self._settle_failure(reservation, exc, ctx)
            if not exc.retryable:
                metrics.provider_calls_total.labels(model, type(exc).__name__).inc()
                raise
            self.breaker.record_failure()
            outcome = type(exc).__name__
            if isinstance(exc, ProviderRateLimitedError):
                retry_after = exc.retry_after
        else:
            usage = response.usage
            billed = cost_usd_micros(response.model, usage.input_tokens, usage.output_tokens)
            within = self.ledger.settle(reservation, tokens=usage.total_tokens, usd_micros=billed)
            if not within:
                ctx.audit(
                    "budget.overshoot",
                    {"node": reservation.path, "tokens": usage.total_tokens, "usd": billed},
                )
            self.breaker.record_success()
        metrics.provider_calls_total.labels(model, outcome).inc()
        metrics.provider_call_seconds.labels(model).observe(ctx.clock() - started)
        return response, retry_after, outcome

    def _settle_failure(
        self, reservation: Reservation, exc: ProviderError, ctx: ActorContext
    ) -> None:
        if exc.usage is None:
            self.ledger.release(reservation)
            return
        billed = cost_usd_micros(
            exc.model or _DEFAULT_MODEL, exc.usage.input_tokens, exc.usage.output_tokens
        )
        self.ledger.settle(reservation, tokens=exc.usage.total_tokens, usd_micros=billed)
        ctx.audit(
            "budget.billed_failure",
            {"node": reservation.path, "error": type(exc).__name__, "usd": billed},
        )
