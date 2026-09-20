"""Wiring: the supervision tree, the quota ledger that mirrors it, and the ingress.

The tree is static, so every actor path, mailbox capacity, ledger node and tool allowlist is
known before anything starts::

    root                       (one_for_one, restart intensity max_restarts / max_restart_seconds)
      provider                 ProviderActor: bulkhead, breaker, retries, fallback, defer
      approval                 ApprovalActor: decides the sink's outbound effect
      sink                     WebhookSink: approval-gated, deduplicates by message id
      validator                Validator: consistency and limits
      extractors               (one_for_one) w0 ... wN: schema output only, no tools
      ingestor                 Ingestor: routes to the worker with the most free credits
      faults                   FaultInjector (only when faults are enabled)

Children are listed in dependency order, so the graceful shutdown -- which drains and stops in
*reverse* start order -- stops the ingestor first and the provider last, and a message already
in the tree can still finish its way to the sink.

The quota ledger uses the same paths (``root``, ``root/extractors``, ``root/extractors/w0``), so
a worker's reservation is checked against its own limit, the pool's and the system's.
"""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

from prometheus_client import CollectorRegistry

from agent_supervision_tree.agentic.tools import ToolRegistry
from agent_supervision_tree.runtime.breaker import CircuitBreaker
from agent_supervision_tree.runtime.errors import MailboxClosedError, MessageTooLargeError
from agent_supervision_tree.runtime.ledger import QuotaLedger
from agent_supervision_tree.runtime.messages import Envelope
from agent_supervision_tree.runtime.provider_actor import (
    AsyncLLMProvider,
    FallbackPolicy,
    ProviderActor,
    RetryPolicy,
)
from agent_supervision_tree.runtime.quarantine import QuarantineRecord
from agent_supervision_tree.runtime.supervisor import (
    ActorSpec,
    ChildSpec,
    SupervisorSpec,
    TreeNode,
)
from agent_supervision_tree.runtime.system import ActorSystem
from agent_supervision_tree.workload.actors import (
    INGEST_KIND,
    ApprovalActor,
    ApprovalPolicy,
    ExtractorWorker,
    FaultInjector,
    Ingestor,
    InMemoryWebhook,
    PoolRouter,
    SinkState,
    Validator,
    WebhookSink,
    WebhookTransport,
    build_webhook_registry,
    default_approval_policy,
)
from agent_supervision_tree.workload.documents import Document
from agent_supervision_tree.workload.fake_llm import FaultingFakeProvider, FaultProfile
from agent_supervision_tree.workload.schemas import DocumentMessage

if TYPE_CHECKING:
    from agent_supervision_tree.core.config import Settings

__all__ = [
    "APPROVAL_PATH",
    "EXTRACTORS_PATH",
    "INGESTOR_PATH",
    "PROVIDER_PATH",
    "ROOT_PATH",
    "SINK_PATH",
    "VALIDATOR_PATH",
    "Pipeline",
    "PipelineConfig",
    "worker_path",
]

ROOT_PATH = "root"
PROVIDER_PATH = f"{ROOT_PATH}/provider"
APPROVAL_PATH = f"{ROOT_PATH}/approval"
SINK_PATH = f"{ROOT_PATH}/sink"
VALIDATOR_PATH = f"{ROOT_PATH}/validator"
EXTRACTORS_PATH = f"{ROOT_PATH}/extractors"
INGESTOR_PATH = f"{ROOT_PATH}/ingestor"
FAULTS_PATH = f"{ROOT_PATH}/faults"
EXTRACTOR_ROLE = "extractor"
# How many recent message ids the pipeline remembers, for introspection and tests only.
ACCEPTED_HISTORY = 100_000
FALLBACK_ROLE = "fallback"


def worker_path(index: int) -> str:
    return f"{EXTRACTORS_PATH}/w{index}"


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    """Everything the tree needs. Defaults are the ones the demo and the benches run with."""

    workers: int = 4
    ingest_capacity: int = 64
    worker_capacity: int = 8
    stage_capacity: int = 64
    max_attempts: int = 3
    max_restarts: int = 5
    max_restart_seconds: float = 5.0
    message_ttl: float = 10.0
    max_message_bytes: int = 65_536
    provider_call_timeout: float = 2.0
    provider_retries: int = 3
    bulkhead: int = 8
    breaker_failures: int = 5
    breaker_reset_seconds: float = 1.0
    budget_tokens: int = 2_000_000
    budget_usd_micros: int = 5_000_000
    quarantine_max_records: int = 10_000
    webhook_url: str = "https://hooks.example.internal/invoices"
    extractor_model: str = "claude-sonnet-5"
    fallback_model: str = "claude-haiku-4-5"
    allow_defer: bool = True
    faults_enabled: bool = False
    fault_rate: float = 0.0
    seed: int = 0
    max_total_cents: int = 100_000_000
    approval_max_total_cents: int = 50_000_000

    def __post_init__(self) -> None:
        if self.workers < 1:
            raise ValueError("a pipeline needs at least one extractor worker")
        if min(self.ingest_capacity, self.worker_capacity, self.stage_capacity) < 1:
            raise ValueError("mailbox capacities must be at least 1")
        if not 0.0 <= self.fault_rate <= 1.0:
            raise ValueError("fault_rate must be within [0, 1]")

    @classmethod
    def from_settings(cls, settings: Settings) -> PipelineConfig:
        return cls(
            workers=settings.workers,
            ingest_capacity=settings.mailbox_capacity,
            worker_capacity=settings.worker_mailbox_capacity,
            stage_capacity=settings.mailbox_capacity,
            max_attempts=settings.max_attempts,
            max_restarts=settings.max_restarts,
            max_restart_seconds=settings.max_restart_seconds,
            message_ttl=settings.message_ttl_seconds,
            max_message_bytes=settings.max_message_bytes,
            provider_call_timeout=settings.provider_call_timeout_seconds,
            bulkhead=settings.provider_bulkhead,
            budget_tokens=settings.budget_max_tokens,
            budget_usd_micros=settings.budget_max_usd_micros,
            quarantine_max_records=settings.quarantine_max_records,
            webhook_url=settings.webhook_url,
            extractor_model=settings.llm_extractor_model,
            fallback_model=settings.llm_fallback_model,
            allow_defer=settings.llm_allow_defer,
            faults_enabled=settings.faults_enabled,
            fault_rate=settings.fault_rate,
        )

    @property
    def worker_paths(self) -> tuple[str, ...]:
        return tuple(worker_path(index) for index in range(self.workers))

    @property
    def role_models(self) -> dict[str, str]:
        return {EXTRACTOR_ROLE: self.extractor_model, FALLBACK_ROLE: self.fallback_model}


@dataclass
class PipelineStats:
    """Counters the demo, the API and the benches report (all derived, never estimated)."""

    accepted: int = 0
    shed: int = 0
    rejected: int = 0
    posted: int = 0
    denied: int = 0
    duplicates: int = 0
    quarantined: int = 0
    restarts: int = 0
    escalations: int = 0
    provider_calls: int = 0
    settled: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "accepted": self.accepted,
            "shed": self.shed,
            "rejected": self.rejected,
            "posted": self.posted,
            "denied": self.denied,
            "duplicates": self.duplicates,
            "quarantined": self.quarantined,
            "settled": self.settled,
            "restarts": self.restarts,
            "escalations": self.escalations,
            "provider_calls": self.provider_calls,
        }


class Pipeline:
    """Owns the actor system, the ledger and the state that must outlive restarts."""

    def __init__(
        self,
        config: PipelineConfig | None = None,
        *,
        provider: AsyncLLMProvider | None = None,
        transport: WebhookTransport | None = None,
        approval_policy: ApprovalPolicy | None = None,
        registry: CollectorRegistry | None = None,
        audit_path: Path | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.config = config or PipelineConfig()
        self.webhook = transport if transport is not None else InMemoryWebhook()
        self.sink_state = SinkState()
        self.fake_provider: FaultingFakeProvider | None = None
        if provider is None:
            self.fake_provider = FaultingFakeProvider(
                FaultProfile.at(self.config.fault_rate),
                seed=self.config.seed,
                model_for_role=self.config.role_models,
            )
            provider = self.fake_provider
        self.provider = provider
        self.ledger = self._build_ledger()
        self.breaker = CircuitBreaker(
            failure_threshold=self.config.breaker_failures,
            reset_timeout=self.config.breaker_reset_seconds,
            clock=self._clock_now,
        )
        self.approval_policy = approval_policy or default_approval_policy(
            max_total_cents=self.config.approval_max_total_cents
        )
        # Recent ids only: a long-running service must not accumulate one string per message.
        # The counter below is the one that never forgets.
        self.accepted_ids: deque[str] = deque(maxlen=ACCEPTED_HISTORY)
        self.accepted_total = 0
        self.shed_total = 0
        self._provider_actor: ProviderActor | None = None
        self.system = ActorSystem(
            self._build_spec(),
            root_id=ROOT_PATH,
            registry=registry,
            audit_path=audit_path,
            quarantine_max_records=self.config.quarantine_max_records,
            max_message_bytes=self.config.max_message_bytes,
            clock=clock,
        )

    # --- construction --------------------------------------------------------------------

    def _clock_now(self) -> float:
        """The event loop clock of the system (the breaker is built before the system is)."""
        return self.system.clock()

    def _build_ledger(self) -> QuotaLedger:
        config = self.config
        ledger = QuotaLedger(
            ROOT_PATH, tokens=config.budget_tokens, usd_micros=config.budget_usd_micros
        )
        ledger.add_node(
            EXTRACTORS_PATH, tokens=config.budget_tokens, usd_micros=config.budget_usd_micros
        )
        # Twice the fair share: the pool and the root remain the binding limits, but a single
        # runaway worker can never spend the whole system budget on its own.
        share_tokens = max(1, 2 * config.budget_tokens // config.workers)
        share_usd = max(1, 2 * config.budget_usd_micros // config.workers)
        for path in config.worker_paths:
            ledger.add_node(path, tokens=share_tokens, usd_micros=share_usd)
        return ledger

    def _build_provider_actor(self) -> ProviderActor:
        config = self.config
        actor = ProviderActor(
            self.provider,
            ledger=self.ledger,
            breaker=self.breaker,
            bulkhead_size=config.bulkhead,
            retry=RetryPolicy(max_attempts=config.provider_retries),
            fallback=FallbackPolicy(chain=(FALLBACK_ROLE,), allow_defer=config.allow_defer),
            call_timeout=config.provider_call_timeout,
            model_for_role=config.role_models,
            seed=config.seed,
        )
        self._provider_actor = actor
        return actor

    def _build_worker(self, path: str) -> ExtractorWorker:
        # Least privilege: the extractor's registry is empty, so the request carries no tools at
        # all and any tool_use in the answer is a message failure, not an execution.
        return ExtractorWorker(
            provider_path=PROVIDER_PATH,
            validator_path=VALIDATOR_PATH,
            ledger_path=path,
            tools=ToolRegistry(()),
        )

    def _build_sink(self) -> WebhookSink:
        registry = build_webhook_registry(self.config.webhook_url, self.webhook)
        return WebhookSink(
            state=self.sink_state,
            tools=registry,
            approval_path=APPROVAL_PATH,
            webhook_url=self.config.webhook_url,
        )

    def _build_spec(self) -> SupervisorSpec:
        config = self.config
        stage = config.stage_capacity
        workers = tuple(
            ChildSpec(
                id=f"w{index}",
                spec=ActorSpec(
                    factory=partial(self._build_worker, worker_path(index)),
                    mailbox_capacity=config.worker_capacity,
                    max_attempts=config.max_attempts,
                ),
            )
            for index in range(config.workers)
        )
        children: list[ChildSpec] = [
            ChildSpec(
                id="provider",
                spec=ActorSpec(
                    factory=self._build_provider_actor,
                    mailbox_capacity=stage,
                    max_attempts=1,
                ),
            ),
            ChildSpec(
                id="approval",
                spec=ActorSpec(
                    factory=lambda: ApprovalActor(self.approval_policy),
                    mailbox_capacity=stage,
                    max_attempts=1,
                ),
            ),
            ChildSpec(
                id="sink",
                spec=ActorSpec(
                    factory=self._build_sink,
                    mailbox_capacity=stage,
                    max_attempts=config.max_attempts,
                ),
            ),
            ChildSpec(
                id="validator",
                spec=ActorSpec(
                    factory=lambda: Validator(SINK_PATH, max_total_cents=config.max_total_cents),
                    mailbox_capacity=stage,
                    max_attempts=config.max_attempts,
                ),
            ),
            ChildSpec(
                id="extractors",
                spec=SupervisorSpec(
                    strategy="one_for_one",
                    children=workers,
                    max_restarts=config.max_restarts,
                    max_seconds=config.max_restart_seconds,
                ),
            ),
            ChildSpec(
                id="ingestor",
                spec=ActorSpec(
                    factory=lambda: Ingestor(PoolRouter(config.worker_paths)),
                    mailbox_capacity=config.ingest_capacity,
                    max_attempts=config.max_attempts,
                ),
            ),
        ]
        if config.faults_enabled:
            children.append(
                ChildSpec(
                    id="faults",
                    spec=ActorSpec(
                        factory=lambda: FaultInjector(
                            kill=self.system.kill,
                            ingest=lambda document: self.submit(document) is not None,
                            provider=self.fake_provider,
                        ),
                        mailbox_capacity=16,
                        max_attempts=1,
                    ),
                    restart="transient",
                )
            )
        return SupervisorSpec(
            strategy="one_for_one",
            children=tuple(children),
            max_restarts=config.max_restarts,
            max_seconds=config.max_restart_seconds,
        )

    # --- lifecycle -----------------------------------------------------------------------

    @property
    def running(self) -> bool:
        return self.system.running

    async def start(self) -> None:
        await self.system.start()

    async def stop(self, timeout: float = 10.0) -> int:
        return await self.system.stop(timeout)

    async def __aenter__(self) -> Pipeline:
        await self.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self.system.running:
            await self.stop()

    # --- ingress -------------------------------------------------------------------------

    def submit(self, document: DocumentMessage | Document) -> str | None:
        """Accept a document, or return None when the ingestor has no credit left (shed)."""
        metrics = self.system.metrics
        payload = {"id": document.id, "text": document.text}
        try:
            envelope = Envelope.create(
                INGEST_KIND,
                payload,
                ttl=self.config.message_ttl,
                max_bytes=self.config.max_message_bytes,
            )
        except MessageTooLargeError:
            self._shed()
            raise
        try:
            accepted = self.system.mailbox(INGESTOR_PATH).try_send(envelope)
        except MailboxClosedError:
            self._shed()
            raise
        if not accepted:
            self._shed()
            return None
        metrics.ingress_accepted_total.inc()
        self.accepted_ids.append(envelope.id)
        self.accepted_total += 1
        return envelope.id

    def _shed(self) -> None:
        self.shed_total += 1
        self.system.metrics.ingress_shed_total.inc()

    def submit_all(self, documents: Iterable[Document]) -> tuple[list[str], int]:
        """Submit a batch without waiting; returns the accepted ids and how many were shed."""
        accepted: list[str] = []
        shed = 0
        for document in documents:
            message_id = self.submit(document)
            if message_id is None:
                shed += 1
            else:
                accepted.append(message_id)
        return accepted, shed

    def settled(self, message_id: str) -> bool:
        """True once the message reached a sink outcome or the quarantine (terminal states)."""
        return message_id in self.sink_state or message_id in self.system.quarantine

    def unsettled(self, message_ids: Sequence[str]) -> list[str]:
        return [item for item in message_ids if not self.settled(item)]

    async def wait_settled(self, message_ids: Sequence[str], timeout: float = 30.0) -> bool:
        """Poll until every id is terminal. Returns False when the timeout hits first."""
        deadline = self.system.clock() + timeout
        while self.system.clock() < deadline:
            if not self.unsettled(message_ids):
                return True
            await asyncio.sleep(0.005)
        return not self.unsettled(message_ids)

    async def drain(self, timeout: float = 10.0) -> bool:
        """Wait until every mailbox is idle (nothing queued or in flight)."""
        deadline = self.system.clock() + timeout
        while self.system.clock() < deadline:
            if all(mailbox.idle for mailbox in self.system.mailboxes().values()):
                return True
            await asyncio.sleep(0.005)
        return all(mailbox.idle for mailbox in self.system.mailboxes().values())

    # --- introspection -------------------------------------------------------------------

    @property
    def provider_actor(self) -> ProviderActor | None:
        return self._provider_actor

    def replay(self, message_ids: Sequence[str]) -> tuple[list[str], list[str]]:
        """Re-ingest quarantined messages. Returns (replayed ids, ids that were shed)."""
        replayed: list[str] = []
        shed: list[str] = []
        for record in self.system.quarantine.take(message_ids):
            document = _record_document(record)
            if document is None:
                shed.append(record.message_id)
                continue
            envelope = Envelope.create(
                INGEST_KIND,
                {"id": document.id, "text": document.text},
                ttl=self.config.message_ttl,
                max_bytes=self.config.max_message_bytes,
                message_id=record.message_id,
            )
            if self.system.mailbox(INGESTOR_PATH).try_send(envelope):
                replayed.append(record.message_id)
                self.system.audit.append(
                    "system", "quarantine.replayed", {"message_id": record.message_id}
                )
            else:
                shed.append(record.message_id)
        return replayed, shed

    def stats(self) -> PipelineStats:
        state = self.sink_state
        stats = PipelineStats(
            accepted=self.accepted_total,
            shed=self.shed_total,
            posted=state.posted,
            denied=state.denied,
            duplicates=state.duplicates,
            quarantined=len(self.system.quarantine),
            restarts=self.system.audit.count("actor.restarted"),
            escalations=self.system.audit.count("supervisor.escalated"),
            provider_calls=self._provider_actor.calls_total if self._provider_actor else 0,
        )
        stats.settled = stats.posted + stats.denied + stats.quarantined
        return stats

    def budget(self) -> dict[str, Any]:
        return self.ledger.snapshot()

    def tree(self) -> TreeNode:
        node = self.system.tree()
        node["budget"] = {
            "spent_usd_micros": self.ledger.node(ROOT_PATH).spent_usd_micros,
            "remaining_tokens": self.ledger.node(ROOT_PATH).remaining_tokens,
            "remaining_usd_micros": self.ledger.node(ROOT_PATH).remaining_usd_micros,
        }
        node["breaker"] = self.breaker.state
        node["sink"] = {
            "posted": self.sink_state.posted,
            "denied": self.sink_state.denied,
            "duplicates": self.sink_state.duplicates,
        }
        return node

    def refresh_gauges(self) -> None:
        """Update the gauges that are sampled rather than incremented, before a scrape."""
        self.system.refresh_gauges()
        metrics = self.system.metrics
        metrics.breaker_state.labels(self.provider.name).set(self.breaker.state_value)
        for path in self.ledger.paths():
            node = self.ledger.node(path)
            metrics.budget_remaining.labels(path, "tokens").set(node.remaining_tokens)
            metrics.budget_remaining.labels(path, "usd_micros").set(node.remaining_usd_micros)

    def healthy(self, lag_threshold: float) -> bool:
        return self.system.loop_lag_seconds <= lag_threshold

    def ready(self) -> bool:
        root = self.system.root
        return bool(root is not None and root.alive and self.breaker.state != "open")


def _record_document(record: QuarantineRecord) -> DocumentMessage | None:
    """The original document of a quarantined record, when its payload still is one."""
    payload = record.payload
    if not isinstance(payload, dict):
        return None
    try:
        return DocumentMessage.model_validate(payload, strict=True)
    except ValueError:
        return None
