"""The HTTP service: ingress with credit-based 429 shedding, plus a scoped admin API.

Scopes come from the API key that was presented (``core.security``), never from the route:

===========================  =========  ====================================================
Route                        Scope      Purpose
===========================  =========  ====================================================
``POST /v1/documents``       ingress    Submit one document. 429 when the root has no credit.
``GET  /v1/tree``            admin      The live supervision tree, budgets and breaker state.
``GET  /v1/stats``           admin      Delivery counters (accepted, sunk, quarantined, ...).
``GET  /v1/quarantine``      admin      Quarantined messages with their reasons.
``POST /v1/quarantine/replay`` admin    Re-ingest quarantined messages after a fix.
``POST /v1/faults``          fault      Chaos: crash, hang, poison, fault rate.
===========================  =========  ====================================================

The fault routes exist only when ``APP_FAULTS_ENABLED`` is true *and* a fault-scope key is
configured, so a production deployment cannot be told to crash itself by an attacker who
somehow obtained an admin key.

``/healthz`` is liveness (the event loop keeps up), ``/readyz`` is readiness (the root
supervisor is alive and the provider breaker is not open), and ``/metrics`` refreshes the
sampled gauges before rendering.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import Body, Depends, FastAPI, Query
from prometheus_client import CollectorRegistry
from pydantic import BaseModel, ConfigDict, Field

from agent_supervision_tree.core.app import create_app
from agent_supervision_tree.core.config import Settings, get_settings
from agent_supervision_tree.core.errors import AppError
from agent_supervision_tree.core.logging import get_logger
from agent_supervision_tree.core.security import Principal, require_scope
from agent_supervision_tree.runtime.errors import MailboxClosedError, MessageTooLargeError
from agent_supervision_tree.runtime.messages import Envelope
from agent_supervision_tree.runtime.provider_actor import AsyncLLMProvider, ThreadedProvider
from agent_supervision_tree.workload.actors import FAULT_KIND
from agent_supervision_tree.workload.pipeline import FAULTS_PATH, Pipeline, PipelineConfig
from agent_supervision_tree.workload.schemas import DocumentMessage, FaultCommand

__all__ = ["ReplayRequest", "build_provider", "create_service_app"]

_log = get_logger("agent_supervision_tree.service")
_SHED_RETRY_AFTER = "1"
AdminPrincipal = Annotated[Principal, Depends(require_scope("admin"))]
IngressPrincipal = Annotated[Principal, Depends(require_scope("ingress"))]
FaultPrincipal = Annotated[Principal, Depends(require_scope("fault"))]


class ReplayRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message_ids: list[str] = Field(min_length=1, max_length=100)


def build_provider(settings: Settings) -> AsyncLLMProvider | None:
    """The live provider when a key is configured, else None (the pipeline uses the fake one)."""
    if settings.anthropic_api_key is None or not settings.anthropic_api_key.get_secret_value():
        return None
    from agent_supervision_tree.agentic.providers.anthropic import (  # noqa: PLC0415 - lazy SDK
        AnthropicProvider,
    )

    provider = AnthropicProvider.from_settings(
        settings,
        role_models={
            "extractor": settings.llm_extractor_model,
            "fallback": settings.llm_fallback_model,
        },
    )
    return ThreadedProvider(provider)


def create_service_app(
    settings: Settings | None = None,
    *,
    pipeline: Pipeline | None = None,
    registry: CollectorRegistry | None = None,
) -> FastAPI:
    """Build the API around a pipeline. The actor system starts and stops with the app."""
    cfg = settings if settings is not None else get_settings()
    metrics_registry = registry if registry is not None else CollectorRegistry()
    tree = pipeline if pipeline is not None else _build_pipeline(cfg, metrics_registry)
    # Both halves must be present: the flag, a fault-scope key, and a tree that actually has
    # the fault actor (a caller may pass a pipeline built without it).
    faults_open = (
        cfg.faults_enabled and bool(cfg.fault_api_key_sha256) and tree.config.faults_enabled
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        await tree.start()
        try:
            yield
        finally:
            exit_code = await tree.stop(timeout=10.0)
            _log.info("system_stopped", exit_code=exit_code)

    app = create_app(
        cfg,
        readiness_checks={"supervision_tree": tree.ready},
        liveness_check=lambda: tree.healthy(cfg.loop_lag_threshold_seconds),
        registry=metrics_registry,
        before_metrics=tree.refresh_gauges,
        lifespan=lifespan,
    )
    app.state.pipeline = tree
    app.state.faults_enabled = faults_open

    @app.post("/v1/documents", status_code=202)
    async def submit_document(
        principal: IngressPrincipal,
        document: Annotated[DocumentMessage, Body()],
    ) -> dict[str, str]:
        """Accept a document, or shed it with 429 when the ingestor has no credit left."""
        try:
            message_id = tree.submit(document)
        except MessageTooLargeError as exc:
            raise AppError(413, "payload_too_large", "The document is too large.") from exc
        except MailboxClosedError as exc:
            raise AppError(503, "shutting_down", "The service is draining.") from exc
        if message_id is None:
            raise AppError(
                429,
                "overloaded",
                "The ingestor has no free credits; retry later.",
                headers={"Retry-After": _SHED_RETRY_AFTER},
            )
        return {"message_id": message_id, "accepted_by": principal.key_id}

    @app.get("/v1/tree")
    async def read_tree(principal: AdminPrincipal) -> dict[str, Any]:
        return dict(tree.tree())

    @app.get("/v1/stats")
    async def read_stats(principal: AdminPrincipal) -> dict[str, Any]:
        stats = tree.stats().as_dict()
        return {
            "stats": stats,
            "budget": tree.budget(),
            "breaker": tree.breaker.state,
            "loop_lag_seconds": round(tree.system.loop_lag_seconds, 6),
            "audit_head": tree.system.audit.head,
        }

    @app.get("/v1/quarantine")
    async def read_quarantine(
        principal: AdminPrincipal,
        reason: Annotated[str | None, Query(max_length=64, pattern=r"^[a-z0-9_:]+$")] = None,
        limit: Annotated[int, Query(ge=1, le=500)] = 50,
    ) -> dict[str, Any]:
        records = tree.system.quarantine.records(reason=reason, limit=limit)
        return {
            "size": len(tree.system.quarantine),
            "reasons": tree.system.quarantine.reasons(),
            "records": [record.model_dump(mode="json") for record in records],
        }

    @app.post("/v1/quarantine/replay")
    async def replay_quarantine(
        principal: AdminPrincipal, body: Annotated[ReplayRequest, Body()]
    ) -> dict[str, Any]:
        replayed, shed = tree.replay(body.message_ids)
        return {"replayed": replayed, "shed": shed}

    if faults_open:

        @app.post("/v1/faults", status_code=202)
        async def inject_fault(
            principal: FaultPrincipal, command: Annotated[FaultCommand, Body()]
        ) -> dict[str, str]:
            """Queue a fault command for the admin-only fault actor."""
            envelope = Envelope.create(
                FAULT_KIND, command.model_dump(mode="json"), ttl=tree.config.message_ttl
            )
            if not tree.system.mailbox(FAULTS_PATH).try_send(envelope):
                raise AppError(429, "overloaded", "The fault actor is busy; retry later.")
            tree.system.audit.append(
                "api", "fault.requested", {"op": command.op, "key_id": principal.key_id}
            )
            return {"message_id": envelope.id}

    @app.get("/v1/audit")
    async def read_audit(
        principal: AdminPrincipal,
        limit: Annotated[int, Query(ge=1, le=500)] = 50,
        action: Annotated[str | None, Query(max_length=64, pattern=r"^[a-z0-9_.]+$")] = None,
    ) -> dict[str, Any]:
        entries = tree.system.audit.recent(limit, action=action)
        return {
            "head": tree.system.audit.head,
            "entries": [entry.model_dump(mode="json") for entry in entries],
        }

    return app


def _build_pipeline(settings: Settings, registry: CollectorRegistry) -> Pipeline:
    return Pipeline(
        PipelineConfig.from_settings(settings),
        provider=build_provider(settings),
        registry=registry,
        audit_path=settings.audit_log_path,
    )
