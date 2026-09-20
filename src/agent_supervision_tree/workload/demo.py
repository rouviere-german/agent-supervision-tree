"""The offline demo: a fault-injected run of the reference workload, end to end.

``uv run python tasks.py demo`` starts the tree with a seeded fake model at a 30% fault rate,
feeds synthetic invoices (some carrying a prompt injection), crashes a worker mid-flight and
injects a batch of poison documents through the admin fault actor, then reports the live tree,
the quarantine, the budget burn and the goodput. It needs no API key and no network.

The summary is the demo's own evidence: ``lost`` counts accepted messages that reached neither a
sink outcome nor the quarantine, and ``duplicate_sink_writes`` counts webhook posts beyond one
per document. Both must be zero, and the demo raises if they are not.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

from pydantic import JsonValue

from agent_supervision_tree.agentic.audit import verify_file
from agent_supervision_tree.runtime.actor import ExitReason
from agent_supervision_tree.runtime.messages import Envelope
from agent_supervision_tree.runtime.supervisor import TreeNode
from agent_supervision_tree.workload.actors import FAULT_KIND, InMemoryWebhook
from agent_supervision_tree.workload.documents import generate_documents
from agent_supervision_tree.workload.pipeline import (
    FAULTS_PATH,
    INGESTOR_PATH,
    Pipeline,
    PipelineConfig,
    worker_path,
)
from agent_supervision_tree.workload.schemas import Extraction

__all__ = ["render_tree", "run_supervision_demo"]

_CRASHED_WORKER = 1
_POISON_BATCH = 3


def deferrals_by_reason(pipeline: Pipeline) -> dict[str, int]:
    """Why calls were deferred (an open breaker, a refused reservation, a cost breaker)."""
    counts: dict[str, int] = {}
    for family in pipeline.system.metrics.registry.collect():
        if family.name not in {"ast_provider_deferred", "ast_provider_deferred_total"}:
            continue
        for item in family.samples:
            if item.name.endswith("_total"):
                counts[item.labels["reason"]] = int(item.value)
    return dict(sorted(counts.items()))


def render_tree(node: TreeNode, prefix: str = "") -> list[str]:
    """A compact ASCII rendering of the live tree (state, mailboxes, restarts, exits)."""
    label = str(node.get("path", "?")).rsplit("/", 1)[-1]
    details = [f"#{node.get('incarnation', 0)}", "up" if node.get("running") else "down"]
    if node.get("kind") == "supervisor":
        details.append(str(node.get("strategy", "")))
        details.append(f"restarts={node.get('restarts_in_window', 0)}")
    mailbox = node.get("mailbox")
    if isinstance(mailbox, dict):
        details.append(f"mailbox={mailbox['depth']}+{mailbox['inflight']}/{mailbox['capacity']}")
    if "processed" in node:
        details.append(f"processed={node['processed']}")
    exits = node.get("exits")
    if exits:
        details.append("exits=" + ",".join(str(item) for item in exits))
    lines = [f"{prefix}{label} [{' '.join(item for item in details if item)}]"]
    children = node.get("children")
    if isinstance(children, list):
        for child in children:
            lines.extend(render_tree(child, prefix + "  "))
    return lines


async def _run(workdir: Path, documents: int, fault_rate: float, seed: int) -> dict[str, Any]:
    workdir.mkdir(parents=True, exist_ok=True)
    audit_path = workdir / "audit.jsonl"
    quarantine_path = workdir / "quarantine.jsonl"
    config = PipelineConfig(
        workers=4,
        worker_capacity=4,
        ingest_capacity=32,
        fault_rate=fault_rate,
        faults_enabled=True,
        seed=seed,
        message_ttl=5.0,
        provider_call_timeout=0.5,
        # The demo lasts a couple of seconds. A breaker tuned for a long-running service
        # would spend most of that time open, so it trips a little later and retries a lot
        # sooner here; when it does trip, the summary reports the deferrals it caused.
        breaker_failures=8,
        breaker_reset_seconds=0.25,
        budget_tokens=200_000,
        budget_usd_micros=400_000,
        # Low enough that a few invoices are refused by the approval actor, so the demo shows
        # the gate denying an outbound effect as well as allowing one.
        approval_max_total_cents=400_000,
    )
    webhook = InMemoryWebhook()
    pipeline = Pipeline(config, transport=webhook, audit_path=audit_path)
    corpus = generate_documents(documents, seed=seed, injection_rate=0.1)
    started = time.perf_counter()

    async with pipeline:
        accepted, shed = pipeline.submit_all(corpus[: documents // 2])
        await asyncio.sleep(0.05)
        # A worker crash while it holds in-flight work: the supervisor restarts it and the
        # runner hands what the dead incarnation never acknowledged back to the mailbox.
        pipeline.system.kill(worker_path(_CRASHED_WORKER), ExitReason.killed("demo_crash"))
        more, shed_more = pipeline.submit_all(corpus[documents // 2 :])
        accepted += more
        shed += shed_more
        # Let the ingress backlog clear first: the injector ingests through the same bounded
        # mailbox as any other client and would otherwise be shed by backpressure.
        await pipeline.system.mailbox(INGESTOR_PATH).wait_idle()
        # Poison documents through the admin fault actor: the fake model always answers them
        # with garbage, so they use up their attempts and land in the quarantine.
        await pipeline.system.mailbox(FAULTS_PATH).send(
            Envelope(kind=FAULT_KIND, payload={"op": "poison", "count": _POISON_BATCH})
        )
        await pipeline.system.mailbox(FAULTS_PATH).wait_idle()
        # Everything the ingress accepted, the poison batch included.
        tracked = list(pipeline.accepted_ids)
        settled = await pipeline.wait_settled(tracked, timeout=30.0)
        await pipeline.drain(timeout=10.0)
        tree = pipeline.tree()
        stats = pipeline.stats()
        reasons = pipeline.system.quarantine.reasons()
        quarantined = pipeline.system.quarantine.export_jsonl(quarantine_path)
        audit_entries = len(pipeline.system.audit)
        outcomes = dict(pipeline.fake_provider.outcomes) if pipeline.fake_provider else {}
        deferrals = deferrals_by_reason(pipeline)
        breaker_state = pipeline.breaker.state
        breaker_opened = pipeline.breaker.opened_total
        exit_code = await pipeline.stop(timeout=5.0)

    elapsed = max(time.perf_counter() - started, 1e-9)
    budget = pipeline.ledger.node("root")
    posts = len(webhook.posts)
    unique = len({str(payload.get("document_id")) for _, payload in webhook.posts})
    lost = pipeline.unsettled(tracked)
    # Fields the closed extraction schema does not have: an obeyed injection would add them.
    allowed_fields = set(Extraction.model_fields)
    leaked = [payload for _, payload in webhook.posts if set(payload) - allowed_fields]
    chain = verify_file(audit_path)
    return {
        "documents": documents,
        "fault_rate": fault_rate,
        "accepted": len(tracked),
        "poison_injected": _POISON_BATCH,
        "shed": shed,
        "settled_before_timeout": settled,
        "posted": stats.posted,
        "denied": stats.denied,
        "quarantined": quarantined,
        "quarantine_reasons": reasons,
        "redelivered_duplicates": stats.duplicates,
        "duplicate_sink_writes": posts - unique,
        "injection_leaks": len(leaked),
        "lost": len(lost),
        "restarts": stats.restarts,
        "escalations": stats.escalations,
        "provider_calls": stats.provider_calls,
        "provider_outcomes": dict(sorted(outcomes.items())),
        "deferrals_by_reason": deferrals,
        "breaker_opened_total": breaker_opened,
        "breaker_state": breaker_state,
        "budget_tokens_spent": budget.spent_tokens,
        "budget_usd_micros_spent": budget.spent_usd_micros,
        "budget_usd_micros_limit": budget.limit_usd_micros,
        "budget_invariants_ok": pipeline.ledger.check_invariants() == [],
        "goodput_per_second": round(stats.posted / elapsed, 2),
        "seconds": round(elapsed, 3),
        "audit_entries": audit_entries,
        "audit_chain_ok": chain.ok,
        "audit_head": chain.head,
        "exit_code": exit_code,
        "tree": tree,
        "tree_render": render_tree(tree),
        "workdir": str(workdir),
    }


def run_supervision_demo(
    workdir: Path, *, documents: int = 48, fault_rate: float = 0.3, seed: int = 7
) -> dict[str, JsonValue]:
    """Run the demo and return its summary. Raises when a delivery invariant is violated."""
    summary = asyncio.run(_run(workdir, documents, fault_rate, seed))
    problems: list[str] = []
    if summary["lost"]:
        problems.append(f"{summary['lost']} accepted messages were neither sunk nor quarantined")
    if summary["duplicate_sink_writes"]:
        problems.append(f"{summary['duplicate_sink_writes']} duplicate webhook posts")
    if summary["injection_leaks"]:
        problems.append(f"{summary['injection_leaks']} posts carried fields outside the schema")
    if not summary["budget_invariants_ok"]:
        problems.append("the quota ledger reports violated invariants")
    if not summary["audit_chain_ok"]:
        problems.append("the audit chain does not verify")
    if problems:
        raise AssertionError("; ".join(problems))
    result: dict[str, JsonValue] = dict(summary)
    return result
