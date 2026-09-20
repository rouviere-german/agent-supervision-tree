"""FastAPI application factory with the baseline hardening applied."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Mapping
from contextlib import AbstractAsyncContextManager
from typing import Any

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from opentelemetry.sdk.trace import TracerProvider
from prometheus_client import CollectorRegistry

from agent_supervision_tree import __version__
from agent_supervision_tree.core.config import Settings, get_settings
from agent_supervision_tree.core.errors import install_error_handlers
from agent_supervision_tree.core.logging import get_logger
from agent_supervision_tree.core.metrics import create_http_metrics, render_metrics
from agent_supervision_tree.core.middleware import (
    BodySizeLimitMiddleware,
    RateLimitMiddleware,
    RequestContextMiddleware,
)
from agent_supervision_tree.core.ratelimit import RateLimiter
from agent_supervision_tree.core.security import ScopedApiKeyVerifier, require_api_key
from agent_supervision_tree.core.tracing import build_tracer_provider

__all__ = ["Lifespan", "LivenessCheck", "ReadinessCheck", "create_app"]

ReadinessCheck = Callable[[], bool | Awaitable[bool]]
LivenessCheck = Callable[[], bool]
Lifespan = Callable[[FastAPI], AbstractAsyncContextManager[None]]
_log = get_logger("agent_supervision_tree.app")


async def _run_check(name: str, check: ReadinessCheck) -> bool:
    try:
        result = check()
        if inspect.isawaitable(result):
            result = await result
        return bool(result)
    except Exception as exc:  # noqa: BLE001 - a failing probe must report not-ready, not crash
        _log.warning("readiness_check_failed", check=name, error_type=type(exc).__name__)
        return False


def create_app(
    settings: Settings | None = None,
    *,
    readiness_checks: Mapping[str, ReadinessCheck] | None = None,
    liveness_check: LivenessCheck | None = None,
    registry: CollectorRegistry | None = None,
    tracer_provider: TracerProvider | None = None,
    before_metrics: Callable[[], None] | None = None,
    lifespan: Lifespan | None = None,
) -> FastAPI:
    """Build the app. All state hangs off ``app.state``; nothing global is mutated.

    ``lifespan`` is the startup/shutdown context a service uses to own long-lived state (this
    repo starts and gracefully stops the actor system there).
    """
    cfg = settings if settings is not None else get_settings()
    checks: dict[str, ReadinessCheck] = dict(readiness_checks or {})
    provider = tracer_provider or build_tracer_provider(cfg.service_name, cfg.tracing_exporter)
    metrics = create_http_metrics(registry)
    verifier = ScopedApiKeyVerifier(
        {
            "ingress": cfg.api_key_sha256,
            "admin": cfg.admin_api_key_sha256,
            "fault": cfg.fault_api_key_sha256,
        }
    )
    limiter = RateLimiter(
        cfg.rate_limit_per_second, cfg.rate_limit_burst, max_keys=cfg.rate_limit_max_clients
    )

    app = FastAPI(
        title=cfg.service_name,
        version=__version__,
        docs_url="/docs" if cfg.enable_docs else None,
        redoc_url=None,
        openapi_url="/openapi.json" if cfg.enable_docs else None,
        lifespan=lifespan,
    )
    app.state.settings = cfg
    app.state.metrics = metrics
    app.state.api_key_verifier = verifier
    app.state.rate_limiter = limiter
    app.state.tracer_provider = provider
    install_error_handlers(app)

    @app.get("/healthz", include_in_schema=False)
    async def healthz() -> JSONResponse:
        """Liveness: the process serves requests and its event loop keeps up."""
        alive = liveness_check() if liveness_check is not None else True
        return JSONResponse(
            {"status": "ok" if alive else "unhealthy"}, status_code=200 if alive else 503
        )

    @app.get("/readyz", include_in_schema=False)
    async def readyz() -> JSONResponse:
        """Readiness: every registered dependency check passes. Reports names, not errors."""
        results = {name: await _run_check(name, check) for name, check in checks.items()}
        ready = all(results.values())
        body: dict[str, Any] = {
            "status": "ready" if ready else "not_ready",
            "checks": {name: "ok" if ok else "fail" for name, ok in results.items()},
        }
        return JSONResponse(body, status_code=200 if ready else 503)

    metrics_deps = [Depends(require_api_key)] if cfg.metrics_require_auth else []

    @app.get("/metrics", include_in_schema=False, dependencies=metrics_deps)
    async def metrics_endpoint(request: Request) -> Response:
        if before_metrics is not None:
            before_metrics()
        payload, content_type = render_metrics(request.app.state.metrics.registry)
        return Response(content=payload, media_type=content_type)

    # add_middleware prepends, so the last one added is the outermost.
    app.add_middleware(BodySizeLimitMiddleware, max_bytes=cfg.max_body_bytes, metrics=metrics)
    app.add_middleware(RateLimitMiddleware, limiter=limiter, verifier=verifier, metrics=metrics)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(cfg.cors_allow_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "X-API-Key", "Authorization", "X-Request-ID"],
        max_age=600,
    )
    app.add_middleware(
        RequestContextMiddleware,
        tracer=provider.get_tracer("agent_supervision_tree.http"),
        metrics=metrics,
        docs_enabled=cfg.enable_docs,
    )
    return app
