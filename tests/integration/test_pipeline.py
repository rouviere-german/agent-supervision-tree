"""End-to-end properties of the reference pipeline under crashes, faults and overload.

These are the repo's headline claims: nothing the ingress accepted is lost, nothing is sunk
twice, budgets hold, injected text never reaches the outbound effect, shutdown drains, and the
process returns to its baseline task count.
"""

from __future__ import annotations

import asyncio
import random
from contextlib import suppress
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from agent_supervision_tree.runtime.actor import ExitReason
from agent_supervision_tree.runtime.errors import ActorNotRunningError, MailboxClosedError
from agent_supervision_tree.runtime.system import EXIT_SOFTWARE
from agent_supervision_tree.workload.actors import InMemoryWebhook
from agent_supervision_tree.workload.documents import generate_documents
from agent_supervision_tree.workload.pipeline import (
    INGESTOR_PATH,
    PROVIDER_PATH,
    Pipeline,
    PipelineConfig,
    worker_path,
)
from agent_supervision_tree.workload.schemas import Extraction
from helpers import wait_for

pytestmark = pytest.mark.integration

ALLOWED_FIELDS = set(Extraction.model_fields)


def config(**overrides: object) -> PipelineConfig:
    base: dict[str, object] = {
        "workers": 3,
        "worker_capacity": 4,
        "ingest_capacity": 24,
        "stage_capacity": 24,
        "message_ttl": 5.0,
        "provider_call_timeout": 0.3,
        "seed": 5,
        "budget_tokens": 400_000,
        "budget_usd_micros": 800_000,
    }
    base.update(overrides)
    return PipelineConfig(**base)  # type: ignore[arg-type]


async def feed(
    pipeline: Pipeline,
    count: int,
    *,
    seed: int = 1,
    injection_rate: float = 0.0,
    poison_rate: float = 0.0,
) -> list[str]:
    """Submit ``count`` documents, waiting for credits instead of shedding."""
    accepted: list[str] = []
    corpus = generate_documents(
        count, seed=seed, injection_rate=injection_rate, poison_rate=poison_rate
    )
    for document in corpus:
        while True:
            message_id = pipeline.submit(document)
            if message_id is not None:
                accepted.append(message_id)
                break
            await asyncio.sleep(0.002)
    return accepted


def posts(pipeline: Pipeline) -> list[tuple[str, dict[str, Any]]]:
    """What the in-memory webhook received (the pipeline's default transport)."""
    transport = pipeline.webhook
    assert isinstance(transport, InMemoryWebhook)
    return transport.posts


def assert_delivery_invariant(pipeline: Pipeline, accepted: list[str]) -> None:
    sunk = [item for item in accepted if item in pipeline.sink_state]
    quarantined = [item for item in accepted if item in pipeline.system.quarantine]
    lost = pipeline.unsettled(accepted)
    assert lost == [], f"{len(lost)} accepted messages were neither sunk nor quarantined"
    assert not set(sunk) & set(quarantined), "a message both reached a sink and was quarantined"
    document_ids = [str(payload.get("document_id")) for _, payload in posts(pipeline)]
    assert len(document_ids) == len(set(document_ids)), "a document was posted twice"


async def test_a_clean_run_sinks_every_document_exactly_once() -> None:
    pipeline = Pipeline(config(fault_rate=0.0))
    async with pipeline:
        accepted = await feed(pipeline, 24)
        assert await pipeline.wait_settled(accepted, timeout=20)
        assert_delivery_invariant(pipeline, accepted)
        assert pipeline.stats().posted == 24
        assert len(pipeline.system.quarantine) == 0
        assert pipeline.ledger.check_invariants() == []


async def test_injected_documents_are_quarantined_and_never_posted() -> None:
    pipeline = Pipeline(config(fault_rate=0.0))
    async with pipeline:
        accepted = await feed(pipeline, 16, seed=3, injection_rate=1.0)
        assert await pipeline.wait_settled(accepted, timeout=20)
        assert pipeline.stats().posted == 0
        assert len(pipeline.system.quarantine) == 16
        assert set(pipeline.system.quarantine.reasons()) == {"schema_violation"}
        assert posts(pipeline) == []
        # The fake model did obey the injection: the schema is what stopped it.
        assert pipeline.fake_provider is not None
        assert pipeline.fake_provider.outcomes["obeyed_injection"] > 0


async def test_no_posted_payload_carries_a_field_outside_the_schema() -> None:
    pipeline = Pipeline(config(fault_rate=0.1))
    async with pipeline:
        accepted = await feed(pipeline, 24, seed=8, injection_rate=0.5)
        assert await pipeline.wait_settled(accepted, timeout=20)
        for _, payload in posts(pipeline):
            assert set(payload) <= ALLOWED_FIELDS


async def test_a_worker_crash_loses_nothing() -> None:
    pipeline = Pipeline(config(fault_rate=0.2))
    async with pipeline:
        accepted = await feed(pipeline, 24)
        await asyncio.sleep(0.02)
        pipeline.system.kill(worker_path(0), ExitReason.killed("test"))
        pipeline.system.kill(worker_path(2), ExitReason.killed("test"))
        assert await pipeline.wait_settled(accepted, timeout=20)
        assert_delivery_invariant(pipeline, accepted)
        assert pipeline.stats().restarts >= 2


async def test_a_provider_crash_loses_nothing() -> None:
    pipeline = Pipeline(config(fault_rate=0.2))
    async with pipeline:
        accepted = await feed(pipeline, 20)
        await asyncio.sleep(0.02)
        pipeline.system.kill(PROVIDER_PATH, ExitReason.killed("test"))
        assert await pipeline.wait_settled(accepted, timeout=20)
        assert_delivery_invariant(pipeline, accepted)


@settings(
    max_examples=8,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
@given(
    crashes=st.lists(
        st.tuples(st.integers(min_value=0, max_value=2), st.integers(min_value=0, max_value=20)),
        min_size=1,
        max_size=6,
    ),
    fault_rate=st.sampled_from([0.0, 0.2, 0.5]),
)
def test_no_crash_schedule_loses_or_duplicates_a_message(
    crashes: list[tuple[int, int]], fault_rate: float
) -> None:
    """Whatever the crash schedule, every accepted id ends up sunk once or quarantined."""

    async def scenario() -> None:
        pipeline = Pipeline(config(fault_rate=fault_rate, max_restarts=20))
        async with pipeline:
            accepted = await feed(pipeline, 12)
            for worker, delay_ms in crashes:
                await asyncio.sleep(delay_ms / 1000)
                with suppress(ActorNotRunningError):  # may be between incarnations
                    pipeline.system.kill(worker_path(worker), ExitReason.killed("chaos"))
            assert await pipeline.wait_settled(accepted, timeout=25)
            assert_delivery_invariant(pipeline, accepted)
            assert pipeline.ledger.check_invariants() == []

    asyncio.run(scenario())


async def test_the_budget_is_never_exceeded_and_exhaustion_defers() -> None:
    pipeline = Pipeline(config(fault_rate=0.3, budget_tokens=4_000, budget_usd_micros=8_000))
    async with pipeline:
        accepted = await feed(pipeline, 24)
        assert await pipeline.wait_settled(accepted, timeout=25)
        root = pipeline.ledger.node("root")
        assert root.spent_tokens <= root.limit_tokens
        assert root.spent_usd_micros <= root.limit_usd_micros
        assert pipeline.ledger.check_invariants() == []
        # Some documents could not be paid for: they are deferred, then quarantined, not dropped.
        refusals = sum(
            pipeline.ledger.node(path).refusals_total for path in pipeline.ledger.paths()
        )
        assert refusals > 0
        assert_delivery_invariant(pipeline, accepted)
        assert "deferred" in pipeline.system.quarantine.reasons()


async def test_the_ingress_sheds_instead_of_growing_beyond_its_bound() -> None:
    pipeline = Pipeline(config(ingest_capacity=8, fault_rate=0.0))
    async with pipeline:
        accepted, shed = pipeline.submit_all(generate_documents(200, seed=2))
        assert shed > 0, "a 200-document burst into an 8-slot mailbox must shed"
        mailbox = pipeline.system.mailbox(INGESTOR_PATH)
        assert mailbox.depth + mailbox.inflight <= mailbox.capacity
        assert pipeline.stats().shed == shed
        assert await pipeline.wait_settled(accepted, timeout=25)
        assert_delivery_invariant(pipeline, accepted)


async def test_graceful_shutdown_drains_the_tree_and_returns_exit_code_zero() -> None:
    pipeline = Pipeline(config(fault_rate=0.1))
    accepted: list[str] = []
    async with pipeline:
        accepted = await feed(pipeline, 20)
        exit_code = await pipeline.stop(timeout=10)
    assert exit_code == 0
    assert_delivery_invariant(pipeline, accepted)
    assert pipeline.stats().settled == len(accepted)
    with pytest.raises(MailboxClosedError):
        pipeline.submit(generate_documents(1, seed=99)[0])


async def test_the_task_count_returns_to_its_baseline() -> None:
    before = len(asyncio.all_tasks())
    pipeline = Pipeline(config(fault_rate=0.3))
    async with pipeline:
        accepted = await feed(pipeline, 16)
        pipeline.system.kill(worker_path(1), ExitReason.killed("test"))
        assert await pipeline.wait_settled(accepted, timeout=20)
    assert await wait_for(lambda: len(asyncio.all_tasks()) <= before + 1, timeout=3)


async def test_an_escalating_root_stops_the_system_with_a_software_exit_code() -> None:
    pipeline = Pipeline(config(max_restarts=1, max_restart_seconds=60.0, fault_rate=0.0))
    async with pipeline:
        for _ in range(3):
            with suppress(ActorNotRunningError):  # the ingestor may already be gone
                pipeline.system.kill(INGESTOR_PATH, ExitReason.killed("chaos"))
            await asyncio.sleep(0.05)
        assert await wait_for(lambda: pipeline.system.exit_code == EXIT_SOFTWARE, timeout=5)
        assert not pipeline.ready()
        assert pipeline.system.audit.count("supervisor.escalated") == 1


async def test_quarantined_messages_can_be_replayed_after_the_cause_is_fixed() -> None:
    pipeline = Pipeline(config(fault_rate=0.0))
    async with pipeline:
        accepted = await feed(pipeline, 8, seed=4, injection_rate=1.0)
        assert await pipeline.wait_settled(accepted, timeout=20)
        assert len(pipeline.system.quarantine) == 8
        # The operator fixes the cause: the model stops obeying injected instructions.
        assert pipeline.fake_provider is not None
        pipeline.fake_provider.gullible = False
        replayed, shed = pipeline.replay(accepted[:4])
        assert len(replayed) == 4
        assert shed == []
        assert await pipeline.wait_settled(replayed, timeout=20)
        assert pipeline.stats().posted == 4
        assert len(pipeline.system.quarantine) == 4
        assert pipeline.system.audit.count("quarantine.replayed") == 4


async def test_the_metrics_gauges_describe_the_live_system() -> None:
    pipeline = Pipeline(config(fault_rate=0.0))
    async with pipeline:
        accepted = await feed(pipeline, 8)
        assert await pipeline.wait_settled(accepted, timeout=20)
        pipeline.refresh_gauges()
        registry = pipeline.system.metrics.registry
        assert registry.get_sample_value("ast_ingress_accepted_total") == 8
        assert registry.get_sample_value("ast_goodput_total", {"outcome": "posted"}) == 8
        assert registry.get_sample_value("ast_breaker_state", {"provider": "fake-faulting"}) == 0
        remaining = registry.get_sample_value(
            "ast_budget_remaining", {"node": "root", "dimension": "tokens"}
        )
        assert remaining is not None
        assert remaining < 400_000
        assert pipeline.healthy(lag_threshold=1.0)
        assert pipeline.ready()


async def test_a_random_mix_of_faults_keeps_every_invariant() -> None:
    rng = random.Random(17)
    pipeline = Pipeline(config(fault_rate=0.4, max_restarts=20, workers=4))
    async with pipeline:
        accepted = await feed(pipeline, 40, seed=6, injection_rate=0.1, poison_rate=0.1)
        for _ in range(6):
            await asyncio.sleep(rng.uniform(0.0, 0.02))
            with suppress(ActorNotRunningError):  # between incarnations
                pipeline.system.kill(worker_path(rng.randrange(4)), ExitReason.killed("chaos"))
        assert await pipeline.wait_settled(accepted, timeout=30)
        assert_delivery_invariant(pipeline, accepted)
        assert pipeline.ledger.check_invariants() == []
        stats = pipeline.stats()
        assert stats.settled == len(accepted)
        assert stats.posted > 0
