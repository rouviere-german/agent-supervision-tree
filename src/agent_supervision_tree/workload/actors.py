"""The reference workload's actors.

Ingestor -> ExtractorPool (workers) -> Validator -> WebhookSink, with a ProviderActor the
workers call, an ApprovalActor that gates the sink's outbound effect, and a FaultInjector that
exists only when fault injection is enabled.

Security properties enforced here rather than by prompting:

- Extractor workers get an **empty tool registry**: the request carries no tools, and a
  ``tool_use`` in the answer is rejected as a message failure instead of executed.
- Document text reaches the model only inside an untrusted-data envelope, and the answer is
  validated against a **closed schema** (``extra="forbid"``, strict). Injected text cannot add a
  field, change the route or approve itself: the message is rejected, not rerouted.
- The sink's webhook call is a tool with ``NETWORK_EGRESS``; the **approval** it needs is bound
  to the canonical hash of the exact call (URL and payload), issued by a separate actor, and the
  call runs only with that hash. The URL is fixed by configuration and checked again by the
  handler.
- The sink **deduplicates by message id**, so redelivery after a crash never posts twice.
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError

from agent_supervision_tree.agentic.envelope import UNTRUSTED_DATA_POLICY, UntrustedData
from agent_supervision_tree.agentic.errors import (
    ProviderRefusalError,
    ProviderRequestError,
    ToolArgumentError,
    ToolNotAllowedError,
)
from agent_supervision_tree.agentic.providers.base import (
    ChatMessage,
    LLMRequest,
    TextBlock,
    ToolUseBlock,
)
from agent_supervision_tree.agentic.tools import (
    Capability,
    ToolDefinition,
    ToolRegistry,
    ValidatedCall,
)
from agent_supervision_tree.runtime.actor import Actor, ActorContext, ExitReason
from agent_supervision_tree.runtime.errors import (
    ActorNotRunningError,
    DeadlineExceededError,
    MailboxClosedError,
    MessageFailed,
    ProviderCrashedError,
    ProviderExhaustedError,
)
from agent_supervision_tree.runtime.mailbox import Mailbox
from agent_supervision_tree.runtime.messages import Envelope
from agent_supervision_tree.runtime.provider_actor import ProviderOutcome, ask_provider
from agent_supervision_tree.workload.documents import generate_documents
from agent_supervision_tree.workload.fake_llm import DOCUMENT_SOURCE, FaultingFakeProvider
from agent_supervision_tree.workload.schemas import (
    EXTRACTION_SCHEMA,
    DocumentMessage,
    FaultCommand,
    validate_extraction,
)

__all__ = [
    "APPROVAL_KIND",
    "EXTRACTOR_SYSTEM_PROMPT",
    "EXTRACT_KIND",
    "FAULT_KIND",
    "INGEST_KIND",
    "SINK_KIND",
    "VALIDATE_KIND",
    "ApprovalActor",
    "ApprovalDecision",
    "ApprovalPolicy",
    "ApprovalRequest",
    "ExtractorWorker",
    "FaultInjector",
    "InMemoryWebhook",
    "Ingestor",
    "PoolRouter",
    "SinkState",
    "Validator",
    "WebhookArgs",
    "WebhookSink",
    "WebhookTransport",
    "ask_approval",
    "build_extraction_request",
    "build_webhook_registry",
    "default_approval_policy",
    "forward",
]

INGEST_KIND = "document.ingest"
EXTRACT_KIND = "document.extract"
VALIDATE_KIND = "extraction.validate"
SINK_KIND = "extraction.sink"
APPROVAL_KIND = "approval.request"
FAULT_KIND = "fault.command"
EXTRACTOR_SYSTEM_PROMPT = (
    "You extract invoice fields from exactly one document and answer only with JSON that "
    "matches the output schema. You have no tools.\n\n" + UNTRUSTED_DATA_POLICY
)
SinkOutcome = Literal["posted", "denied"]


async def forward(ctx: ActorContext, path: str, envelope: Envelope) -> None:
    """Hand a message to the next stage, waiting for a credit only until its deadline.

    A stage that blocks forever on a full downstream mailbox would hold its own credit forever
    and stall the tree, so the wait is bounded by the message's own deadline and a message that
    cannot be handed over is failed (redelivered, then quarantined) instead.
    """
    remaining = envelope.remaining(ctx.clock())
    if remaining is not None and remaining <= 0:
        raise MessageFailed("deadline_exceeded", f"before {path}", retryable=False)
    try:
        await ctx.send(path, envelope, timeout=remaining)
    except TimeoutError as exc:
        raise MessageFailed("downstream_full", path) from exc
    except MailboxClosedError as exc:
        raise MessageFailed("downstream_closed", path, retryable=False) from exc


# --- requests ------------------------------------------------------------------------------


def build_extraction_request(document: DocumentMessage, tools: ToolRegistry) -> LLMRequest:
    """The extraction prompt: the document as delimited untrusted data, schema output only."""
    envelope = UntrustedData(source=DOCUMENT_SOURCE, content=document.text)
    prompt = f"Extract the invoice fields of document {document.id}.\n\n{envelope.render()}"
    return LLMRequest(
        system=EXTRACTOR_SYSTEM_PROMPT,
        messages=(ChatMessage(role="user", content=(TextBlock(text=prompt),)),),
        tools=tools.specs(),
        max_tokens=512,
        output_schema=EXTRACTION_SCHEMA,
        role="extractor",
    )


class PoolRouter:
    """Route to the worker with the most free credits; wait on it when every worker is full."""

    def __init__(self, paths: Sequence[str]) -> None:
        if not paths:
            raise ValueError("a pool needs at least one worker")
        self.paths = tuple(paths)

    def ranked(self, ctx: ActorContext) -> list[Mailbox]:
        mailboxes = [ctx.system.mailbox(path) for path in self.paths]
        return sorted(mailboxes, key=lambda mailbox: mailbox.credits, reverse=True)

    async def route(self, ctx: ActorContext, envelope: Envelope) -> None:
        ranked = self.ranked(ctx)
        try:
            for mailbox in ranked:
                if mailbox.try_send(envelope):
                    return
        except MailboxClosedError as exc:
            raise MessageFailed("downstream_closed", mailbox.name, retryable=False) from exc
        await forward(ctx, ranked[0].name, envelope)


# --- pipeline stages -----------------------------------------------------------------------


class Ingestor(Actor):
    accepts = frozenset({INGEST_KIND})

    def __init__(self, router: PoolRouter) -> None:
        self.router = router

    async def receive(self, envelope: Envelope, ctx: ActorContext) -> None:
        try:
            document = DocumentMessage.model_validate(envelope.payload)
        except ValidationError as exc:
            raise MessageFailed("invalid_document", retryable=False) from exc
        await self.router.route(ctx, envelope.derive(EXTRACT_KIND, document.model_dump()))


class ExtractorWorker(Actor):
    accepts = frozenset({EXTRACT_KIND})

    def __init__(
        self, *, provider_path: str, validator_path: str, ledger_path: str, tools: ToolRegistry
    ) -> None:
        self.provider_path = provider_path
        self.validator_path = validator_path
        self.ledger_path = ledger_path
        self.tools = tools

    async def receive(self, envelope: Envelope, ctx: ActorContext) -> None:
        try:
            document = DocumentMessage.model_validate(envelope.payload)
        except ValidationError as exc:
            raise MessageFailed("invalid_document", retryable=False) from exc
        request = build_extraction_request(document, self.tools)
        outcome = await self._call(ctx, request, envelope)
        if outcome.status == "deferred" or outcome.response is None:
            # A deferral is usually transient (an open breaker, a momentary quota refusal), so
            # the message is redelivered and only quarantined once its attempts are used up.
            raise MessageFailed("deferred", outcome.reason)
        response = outcome.response
        if response.tool_calls:
            ctx.audit(
                "tool.denied",
                {"tools": [call.name for call in response.tool_calls], "reason": "no_tools"},
            )
            raise MessageFailed("tool_call_not_allowed", retryable=False)
        if response.parsed is None:
            raise MessageFailed("malformed_output")
        try:
            extraction = validate_extraction(response.parsed)
        except ValidationError as exc:
            raise MessageFailed("schema_violation") from exc
        if extraction.document_id != document.id:
            raise MessageFailed("document_mismatch")
        payload = extraction.model_dump(mode="json")
        await forward(ctx, self.validator_path, envelope.derive(VALIDATE_KIND, payload))

    async def _call(
        self, ctx: ActorContext, request: LLMRequest, envelope: Envelope
    ) -> ProviderOutcome:
        try:
            return await ask_provider(
                ctx,
                self.provider_path,
                request,
                ledger_path=self.ledger_path,
                deadline=envelope.deadline,
            )
        except ProviderCrashedError as exc:
            raise MessageFailed("provider_crashed") from exc
        except ProviderExhaustedError as exc:
            raise MessageFailed("provider_exhausted") from exc
        except DeadlineExceededError as exc:
            raise MessageFailed("deadline_exceeded", retryable=False) from exc
        except ProviderRefusalError as exc:
            raise MessageFailed("refused", retryable=False) from exc
        except ProviderRequestError as exc:
            raise MessageFailed("bad_request", retryable=False) from exc


class Validator(Actor):
    accepts = frozenset({VALIDATE_KIND})

    def __init__(self, sink_path: str, *, max_total_cents: int = 100_000_000) -> None:
        self.sink_path = sink_path
        self.max_total_cents = max_total_cents

    async def receive(self, envelope: Envelope, ctx: ActorContext) -> None:
        try:
            extraction = validate_extraction(envelope.payload)
        except ValidationError as exc:
            raise MessageFailed("schema_violation", retryable=False) from exc
        if not extraction.consistent():
            raise MessageFailed("validation_failed", "total != sum(line items)", retryable=False)
        if extraction.total_cents > self.max_total_cents:
            raise MessageFailed("validation_failed", "total above limit", retryable=False)
        await forward(ctx, self.sink_path, envelope.derive(SINK_KIND, envelope.payload))


# --- approvals -----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ApprovalDecision:
    message_id: str
    call_hash: str
    approved: bool
    decided_by: str
    reason: str


@dataclass(eq=False)
class ApprovalRequest:
    message_id: str
    call: ValidatedCall
    reply: asyncio.Future[ApprovalDecision] = field(repr=False)


ApprovalPolicy = Callable[[ValidatedCall], tuple[bool, str]]


def default_approval_policy(
    denied_vendors: frozenset[str] = frozenset({"attacker corp"}),
    *,
    max_total_cents: int = 50_000_000,
) -> ApprovalPolicy:
    """Approve invoices unless the vendor is denied or the total is above the limit."""

    def policy(call: ValidatedCall) -> tuple[bool, str]:
        payload = call.canonical_args.get("payload")
        if not isinstance(payload, dict):
            return False, "payload_missing"
        vendor = str(payload.get("vendor", "")).lower()
        if vendor in denied_vendors:
            return False, "vendor_denied"
        total = payload.get("total_cents")
        if not isinstance(total, int) or total > max_total_cents:
            return False, "total_above_limit"
        return True, "policy"

    return policy


class ApprovalActor(Actor):
    """Decides approvals for a different principal (the sink), bound to the call hash."""

    accepts = frozenset({APPROVAL_KIND})

    def __init__(self, policy: ApprovalPolicy, *, principal: str = "approver:policy") -> None:
        self.policy = policy
        self.principal = principal

    async def receive(self, envelope: Envelope, ctx: ActorContext) -> None:
        request = envelope.payload
        if not isinstance(request, ApprovalRequest) or request.reply.done():
            return
        approved, reason = self.policy(request.call)
        decision = ApprovalDecision(
            request.message_id, request.call.call_hash, approved, self.principal, reason
        )
        ctx.system.metrics.approvals_total.labels("approved" if approved else "denied").inc()
        ctx.audit(
            "approval.decided",
            {
                "message_id": request.message_id,
                "tool": request.call.tool.name,
                "call_hash": decision.call_hash,
                "approved": approved,
                "reason": reason,
                "decided_by": self.principal,
            },
        )
        request.reply.set_result(decision)


async def ask_approval(
    ctx: ActorContext,
    approval_path: str,
    call: ValidatedCall,
    *,
    message_id: str,
    deadline: float | None,
) -> ApprovalDecision:
    loop = asyncio.get_running_loop()
    request = ApprovalRequest(message_id, call, loop.create_future())
    remaining = None if deadline is None else deadline - ctx.clock()
    if remaining is not None and remaining <= 0:
        raise DeadlineExceededError("no time left to ask for approval")
    try:
        await ctx.send(
            approval_path, Envelope(kind=APPROVAL_KIND, payload=request), timeout=remaining
        )
        async with asyncio.timeout(remaining):
            return await request.reply
    except TimeoutError as exc:
        request.reply.cancel()
        raise DeadlineExceededError("approval ran past the message deadline") from exc


# --- the sink and its tool -----------------------------------------------------------------


class WebhookArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: str = Field(pattern=r"^https://[A-Za-z0-9.-]+(/[A-Za-z0-9._/-]*)?$", max_length=512)
    payload: dict[str, JsonValue]


class WebhookTransport(Protocol):
    def post(self, url: str, payload: Mapping[str, JsonValue]) -> None: ...


class InMemoryWebhook:
    """Records posts instead of making them; can be told to fail the next N posts."""

    def __init__(self) -> None:
        self.posts: list[tuple[str, dict[str, JsonValue]]] = []
        self.fail_next = 0

    def post(self, url: str, payload: Mapping[str, JsonValue]) -> None:
        if self.fail_next > 0:
            self.fail_next -= 1
            raise ConnectionError("webhook endpoint unavailable")
        self.posts.append((url, dict(payload)))


def build_webhook_registry(url: str, transport: WebhookTransport) -> ToolRegistry:
    """The sink's only tool. Its URL is fixed by configuration and checked by the handler too."""
    registry = ToolRegistry({Capability.NETWORK_EGRESS})

    def post_webhook(args: WebhookArgs) -> JsonValue:
        if args.url != url:
            raise ToolNotAllowedError("the webhook URL is fixed by configuration")
        transport.post(args.url, args.payload)
        return "posted"

    registry.register(
        ToolDefinition(
            name="post_webhook",
            description="Post a validated extraction to the configured webhook.",
            args_model=WebhookArgs,
            capabilities=frozenset({Capability.NETWORK_EGRESS}),
            handler=post_webhook,
        )
    )
    return registry


class SinkState:
    """Outcomes by message id, bounded; owned by the pipeline so it outlives restarts."""

    def __init__(self, capacity: int = 100_000) -> None:
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        self.capacity = capacity
        self._outcomes: OrderedDict[str, SinkOutcome] = OrderedDict()
        self.posted = 0
        self.denied = 0
        self.duplicates = 0

    def __contains__(self, message_id: str) -> bool:
        return message_id in self._outcomes

    def __len__(self) -> int:
        return len(self._outcomes)

    def outcome(self, message_id: str) -> SinkOutcome | None:
        return self._outcomes.get(message_id)

    def record(self, message_id: str, outcome: SinkOutcome) -> None:
        self._outcomes[message_id] = outcome
        if outcome == "posted":
            self.posted += 1
        else:
            self.denied += 1
        while len(self._outcomes) > self.capacity:
            self._outcomes.popitem(last=False)


class WebhookSink(Actor):
    accepts = frozenset({SINK_KIND})

    def __init__(
        self, *, state: SinkState, tools: ToolRegistry, approval_path: str, webhook_url: str
    ) -> None:
        self.state = state
        self.tools = tools
        self.approval_path = approval_path
        self.webhook_url = webhook_url

    async def receive(self, envelope: Envelope, ctx: ActorContext) -> None:
        metrics = ctx.system.metrics
        if envelope.id in self.state:
            self.state.duplicates += 1
            metrics.goodput_total.labels("duplicate").inc()
            return
        call = self._validated_call(envelope)
        decision = await self._decision(ctx, call, envelope)
        if decision.call_hash != call.call_hash or decision.message_id != envelope.id:
            raise MessageFailed("approval_mismatch", retryable=False)
        if not decision.approved:
            self._finish(ctx, envelope, "denied")
            return
        try:
            call.run(authorized_hash=decision.call_hash)
        except ToolArgumentError as exc:
            raise MessageFailed("approval_mismatch", retryable=False) from exc
        except ToolNotAllowedError as exc:
            raise MessageFailed("tool_rejected", retryable=False) from exc
        except ConnectionError as exc:
            raise MessageFailed("webhook_failed") from exc
        self._finish(ctx, envelope, "posted")

    def _validated_call(self, envelope: Envelope) -> ValidatedCall:
        try:
            extraction = validate_extraction(envelope.payload)
        except ValidationError as exc:
            raise MessageFailed("schema_violation", retryable=False) from exc
        proposed = ToolUseBlock(
            id=envelope.id,
            name="post_webhook",
            arguments={"url": self.webhook_url, "payload": extraction.model_dump(mode="json")},
        )
        try:
            return self.tools.validate_call(proposed)
        except (ToolNotAllowedError, ToolArgumentError) as exc:
            raise MessageFailed("tool_rejected", retryable=False) from exc

    async def _decision(
        self, ctx: ActorContext, call: ValidatedCall, envelope: Envelope
    ) -> ApprovalDecision:
        try:
            return await ask_approval(
                ctx,
                self.approval_path,
                call,
                message_id=envelope.id,
                deadline=envelope.deadline,
            )
        except DeadlineExceededError as exc:
            raise MessageFailed("deadline_exceeded", retryable=False) from exc

    def _finish(self, ctx: ActorContext, envelope: Envelope, outcome: SinkOutcome) -> None:
        self.state.record(envelope.id, outcome)
        ctx.system.metrics.goodput_total.labels(outcome).inc()
        ctx.system.metrics.message_latency_seconds.observe(ctx.clock() - envelope.created_at)


# --- fault injection -----------------------------------------------------------------------


class FaultInjector(Actor):
    """Admin-only chaos: crash an actor, hang provider calls, inject poison, set the fault rate."""

    accepts = frozenset({FAULT_KIND})

    def __init__(
        self,
        *,
        kill: Callable[[str, ExitReason], None],
        ingest: Callable[[DocumentMessage], bool],
        provider: FaultingFakeProvider | None,
    ) -> None:
        self.kill = kill
        self.ingest = ingest
        self.provider = provider
        self._poison_batches = 0

    async def receive(self, envelope: Envelope, ctx: ActorContext) -> None:
        try:
            command = FaultCommand.model_validate(envelope.payload)
        except ValidationError as exc:
            raise MessageFailed("invalid_command", retryable=False) from exc
        detail = self._apply(command)
        ctx.audit(
            "fault.injected",
            {"op": command.op, "path": command.path, "count": command.count, "detail": detail},
        )

    def _apply(self, command: FaultCommand) -> str:
        if command.op == "crash":
            if command.path is None:
                raise MessageFailed("invalid_command", "crash needs a path", retryable=False)
            try:
                self.kill(command.path, ExitReason.killed("fault_injection"))
            except ActorNotRunningError as exc:
                raise MessageFailed("unknown_actor", retryable=False) from exc
            return "killed"
        if command.op == "poison":
            self._poison_batches += 1
            documents = generate_documents(
                command.count, seed=0xF000 + self._poison_batches, poison_rate=1.0
            )
            accepted = sum(
                self.ingest(DocumentMessage(id=doc.id, text=doc.text)) for doc in documents
            )
            return f"accepted {accepted}/{len(documents)}"
        if self.provider is None:
            raise MessageFailed("unsupported", "no fake provider", retryable=False)
        if command.op == "hang":
            self.provider.hang_next(command.count)
            return f"next {command.count} calls hang"
        if command.rate is None:
            raise MessageFailed("invalid_command", "fault_rate needs a rate", retryable=False)
        self.provider.set_fault_rate(command.rate)
        return f"fault rate {command.rate}"
