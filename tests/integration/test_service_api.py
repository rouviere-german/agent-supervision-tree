"""The HTTP service: scoped routes, credit-based shedding, introspection and fault injection."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, cast

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from prometheus_client import CollectorRegistry

from agent_supervision_tree.core.config import Settings
from agent_supervision_tree.core.security import hash_api_key
from agent_supervision_tree.service import build_provider, create_service_app
from agent_supervision_tree.workload.documents import generate_documents
from agent_supervision_tree.workload.fake_llm import FaultingFakeProvider, FaultProfile
from agent_supervision_tree.workload.pipeline import Pipeline, PipelineConfig

pytestmark = pytest.mark.integration

INGRESS_KEY = "ingress-key-" + "i" * 24
ADMIN_KEY = "admin-key-" + "a" * 24
FAULT_KEY = "fault-key-" + "f" * 24
INGRESS = {"X-API-Key": INGRESS_KEY}
ADMIN = {"X-API-Key": ADMIN_KEY}
FAULT = {"X-API-Key": FAULT_KEY}


def make_settings(**overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "environment": "test",
        "api_key_sha256": [hash_api_key(INGRESS_KEY)],
        "admin_api_key_sha256": [hash_api_key(ADMIN_KEY)],
        "fault_api_key_sha256": [hash_api_key(FAULT_KEY)],
        "rate_limit_per_second": 10_000,
        "rate_limit_burst": 10_000,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def make_app(
    settings: Settings | None = None,
    *,
    provider: Any | None = None,
    **pipeline_options: Any,
) -> FastAPI:
    cfg = settings if settings is not None else make_settings()
    options: dict[str, Any] = {
        "workers": 2,
        "worker_capacity": 4,
        "ingest_capacity": 16,
        "stage_capacity": 16,
        "fault_rate": 0.0,
        "message_ttl": 5.0,
        "provider_call_timeout": 0.3,
        "faults_enabled": cfg.faults_enabled,
    }
    options.update(pipeline_options)
    registry = CollectorRegistry()
    pipeline = Pipeline(PipelineConfig(**options), provider=provider, registry=registry)
    return create_service_app(cfg, pipeline=pipeline, registry=registry)


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(make_app(), raise_server_exceptions=False) as test_client:
        yield test_client


def document(index: int = 0) -> dict[str, str]:
    doc = generate_documents(index + 1, seed=21)[index]
    return {"id": doc.id, "text": doc.text}


def pipeline_of(client: TestClient) -> Pipeline:
    app = cast(FastAPI, client.app)
    pipeline = app.state.pipeline
    assert isinstance(pipeline, Pipeline)
    return pipeline


def test_ingress_accepts_a_document_and_the_tree_processes_it(client: TestClient) -> None:
    response = client.post("/v1/documents", json=document(), headers=INGRESS)
    assert response.status_code == 202
    message_id = response.json()["message_id"]
    assert len(message_id) == 32
    pipeline = pipeline_of(client)
    for _ in range(200):
        if pipeline.settled(message_id):
            break
        client.get("/healthz")
    assert pipeline.settled(message_id)
    assert pipeline.stats().posted == 1


def test_every_route_needs_the_right_scope(client: TestClient) -> None:
    assert client.post("/v1/documents", json=document()).status_code == 401
    assert client.get("/v1/tree").status_code == 401
    # A valid key with the wrong scope is a 403, not a 401: the caller is known, not allowed.
    forbidden = client.get("/v1/tree", headers=INGRESS)
    assert forbidden.status_code == 403
    assert forbidden.json()["error"]["code"] == "forbidden"
    assert client.post("/v1/documents", json=document(), headers=ADMIN).status_code == 403
    assert client.get("/v1/tree", headers=ADMIN).status_code == 200


def test_a_malformed_document_is_rejected_before_it_enters_the_tree(client: TestClient) -> None:
    bad = client.post("/v1/documents", json={"id": "nope", "text": "x"}, headers=INGRESS)
    assert bad.status_code == 422
    assert bad.json()["error"]["code"] == "validation_error"
    extra = client.post(
        "/v1/documents", json={**document(), "webhook_url": "https://evil.example"}, headers=INGRESS
    )
    assert extra.status_code == 422
    assert pipeline_of(client).stats().accepted == 0


def test_an_oversized_document_is_refused_with_413() -> None:
    app = make_app(max_message_bytes=512)
    with TestClient(app, raise_server_exceptions=False) as client:
        payload = {"id": document()["id"], "text": "x" * 5_000}
        response = client.post("/v1/documents", json=payload, headers=INGRESS)
        assert response.status_code == 413
        assert response.json()["error"]["code"] == "payload_too_large"


def test_the_ingress_sheds_with_429_when_the_mailbox_is_full() -> None:
    """A slow model plus a two-slot ingress mailbox: the burst is shed, not queued."""
    slow = FaultingFakeProvider(
        FaultProfile(latency_ms=50.0, latency_jitter_ms=0.0),
        seed=0,
        model_for_role={"extractor": "claude-sonnet-5"},
    )
    app = make_app(provider=slow, ingest_capacity=2, worker_capacity=1, workers=1)
    with TestClient(app, raise_server_exceptions=False) as client:
        responses = [
            client.post("/v1/documents", json=document(index), headers=INGRESS)
            for index in range(40)
        ]
        codes = [response.status_code for response in responses]
        assert 202 in codes
        shed = [response for response in responses if response.status_code == 429]
        assert shed, "a burst into a two-slot mailbox must be shed"
        assert shed[0].json()["error"]["code"] == "overloaded"
        assert shed[0].headers["retry-after"] == "1"
        assert pipeline_of(client).stats().shed == len(shed)


def test_the_tree_endpoint_describes_the_live_system(client: TestClient) -> None:
    body = client.get("/v1/tree", headers=ADMIN).json()
    assert body["path"] == "root"
    assert body["strategy"] == "one_for_one"
    children = {child["path"] for child in body["children"]}
    assert {"root/provider", "root/ingestor", "root/extractors"} <= children
    assert body["breaker"] == "closed"
    assert body["budget"]["remaining_usd_micros"] > 0
    assert body["sink"] == {"posted": 0, "denied": 0, "duplicates": 0}


def test_stats_and_audit_are_available_to_operators(client: TestClient) -> None:
    stats = client.get("/v1/stats", headers=ADMIN).json()
    assert stats["stats"]["accepted"] == 0
    assert stats["budget"]["path"] == "root"
    assert stats["breaker"] == "closed"
    audit = client.get("/v1/audit", headers=ADMIN, params={"limit": 10}).json()
    assert audit["entries"][0]["action"] == "system.started"
    assert len(audit["head"]) == 64
    filtered = client.get("/v1/audit", headers=ADMIN, params={"action": "system.started"}).json()
    assert [entry["action"] for entry in filtered["entries"]] == ["system.started"]


def test_quarantined_messages_can_be_listed_and_replayed() -> None:
    app = make_app()
    with TestClient(app, raise_server_exceptions=False) as client:
        pipeline = pipeline_of(client)
        doc = generate_documents(1, seed=31, injection_rate=1.0)[0]
        message_id = client.post(
            "/v1/documents", json={"id": doc.id, "text": doc.text}, headers=INGRESS
        ).json()["message_id"]
        for _ in range(300):
            if pipeline.settled(message_id):
                break
            client.get("/healthz")
        listing = client.get("/v1/quarantine", headers=ADMIN).json()
        assert listing["size"] == 1
        assert listing["reasons"] == {"schema_violation": 1}
        assert listing["records"][0]["message_id"] == message_id

        assert pipeline.fake_provider is not None
        pipeline.fake_provider.gullible = False  # the operator fixes the cause
        replay = client.post(
            "/v1/quarantine/replay", json={"message_ids": [message_id]}, headers=ADMIN
        )
        assert replay.status_code == 200
        assert replay.json() == {"replayed": [message_id], "shed": []}
        for _ in range(300):
            if pipeline.sink_state.outcome(message_id) is not None:
                break
            client.get("/healthz")
        assert pipeline.sink_state.outcome(message_id) == "posted"
        assert client.get("/v1/quarantine", headers=ADMIN).json()["size"] == 0


def test_replay_requests_are_bounded(client: TestClient) -> None:
    empty = client.post("/v1/quarantine/replay", json={"message_ids": []}, headers=ADMIN)
    assert empty.status_code == 422
    too_many = client.post(
        "/v1/quarantine/replay", json={"message_ids": ["x"] * 101}, headers=ADMIN
    )
    assert too_many.status_code == 422
    unknown = client.post(
        "/v1/quarantine/replay", json={"message_ids": ["does-not-exist"]}, headers=ADMIN
    )
    assert unknown.json() == {"replayed": [], "shed": []}


def test_fault_routes_do_not_exist_unless_they_are_enabled(client: TestClient) -> None:
    response = client.post("/v1/faults", json={"op": "crash", "path": "root/sink"}, headers=FAULT)
    assert response.status_code == 404


def test_fault_injection_needs_the_fault_scope_and_is_audited() -> None:
    settings = make_settings(faults_enabled=True)
    app = make_app(settings)
    with TestClient(app, raise_server_exceptions=False) as client:
        pipeline = pipeline_of(client)
        assert (
            client.post("/v1/faults", json={"op": "hang", "count": 1}, headers=ADMIN).status_code
            == 403
        )
        accepted = client.post(
            "/v1/faults", json={"op": "crash", "path": "root/sink"}, headers=FAULT
        )
        assert accepted.status_code == 202
        for _ in range(200):
            if pipeline.system.audit.count("fault.injected"):
                break
            client.get("/healthz")
        assert pipeline.system.audit.count("fault.requested") == 1
        assert pipeline.system.audit.count("fault.injected") == 1
        # The sink was restarted by its supervisor, so the tree is still serving.
        assert client.get("/readyz").status_code == 200


def test_health_readiness_and_metrics(client: TestClient) -> None:
    assert client.get("/healthz").json() == {"status": "ok"}
    ready = client.get("/readyz")
    assert ready.status_code == 200
    assert ready.json()["checks"] == {"supervision_tree": "ok"}
    assert client.get("/metrics").status_code == 401
    body = client.get("/metrics", headers=ADMIN).text
    assert "ast_mailbox_credits" in body
    assert 'ast_budget_remaining{dimension="tokens",node="root"}' in body
    assert 'ast_breaker_state{provider="fake-faulting"}' in body


def test_without_an_api_key_the_live_provider_is_not_built() -> None:
    assert build_provider(make_settings()) is None
