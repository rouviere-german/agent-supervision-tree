#!/usr/bin/env python3
"""Fixed-seed fault-injection soak: run the tree under constant chaos and watch it hold.

Usage:
  uv run python bench/soak.py --seconds 300          # the CI soak job
  uv run python bench/soak.py --seconds 20 --seed 5  # a quick local check

The soak feeds documents continuously while crashing workers, injecting poison batches and
raising the model's fault rate. Every ``--tick`` seconds it classifies the messages that are
older than the grace period: each one must have reached a sink outcome or the quarantine.
Anything else is a lost message and fails the run. It also watches RSS, the asyncio task count
and the ledger invariants, and verifies the audit chain at the end.

Everything is deterministic given ``--seed`` apart from the event loop's own scheduling.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from collections import deque
from contextlib import suppress
from pathlib import Path
from typing import Any

from agent_supervision_tree.agentic.audit import verify_file
from agent_supervision_tree.runtime.actor import ExitReason
from agent_supervision_tree.runtime.errors import ActorNotRunningError
from agent_supervision_tree.runtime.messages import Envelope
from agent_supervision_tree.workload.actors import FAULT_KIND
from agent_supervision_tree.workload.documents import generate_documents
from agent_supervision_tree.workload.pipeline import (
    FAULTS_PATH,
    Pipeline,
    PipelineConfig,
    worker_path,
)
from benchlib import ROOT, print_table, rss_mb, write_results

RESULTS = ROOT / "bench" / "results" / "soak.json"
WORKERS = 4
GRACE_SECONDS = 20.0
CORPUS_CHUNK = 2_000


class CountingWebhook:
    """Counts posts and duplicate document ids without keeping the payloads.

    The in-memory webhook of the demo keeps every post; over a five-minute soak that would be
    the biggest object in the process and would swamp the RSS check.
    """

    def __init__(self) -> None:
        self.posted = 0
        self.duplicates = 0
        self._seen: set[str] = set()

    def post(self, url: str, payload: Any) -> None:
        self.posted += 1
        document_id = str(payload.get("document_id", ""))
        if document_id in self._seen:
            self.duplicates += 1
        else:
            self._seen.add(document_id)


class Outcomes:
    """Counts of how accepted messages ended, plus the ids that ended nowhere."""

    def __init__(self) -> None:
        self.posted = 0
        self.denied = 0
        self.quarantined = 0
        self.lost: list[str] = []

    @property
    def settled(self) -> int:
        return self.posted + self.denied + self.quarantined


def classify(
    pipeline: Pipeline, pending: deque[tuple[str, float]], *, cutoff: float, outcomes: Outcomes
) -> None:
    """Settle every message older than ``cutoff``; drain their quarantine records as we go."""
    drained: list[str] = []
    while pending and pending[0][1] <= cutoff:
        message_id, _ = pending.popleft()
        outcome = pipeline.sink_state.outcome(message_id)
        if outcome == "posted":
            outcomes.posted += 1
        elif outcome == "denied":
            outcomes.denied += 1
        elif message_id in pipeline.system.quarantine:
            outcomes.quarantined += 1
            drained.append(message_id)
        else:
            outcomes.lost.append(message_id)
    if drained:
        # An operator exporting and clearing the quarantine; it keeps memory flat too.
        pipeline.system.quarantine.take(drained)


def inject_chaos(pipeline: Pipeline, rng: random.Random) -> None:
    """One tick of chaos: crash a worker, move the model's fault rate, maybe poison a batch."""
    with suppress(ActorNotRunningError):
        pipeline.system.kill(worker_path(rng.randrange(WORKERS)), ExitReason.killed("soak"))
    if pipeline.fake_provider is not None:
        pipeline.fake_provider.set_fault_rate(rng.choice((0.1, 0.3, 0.5)))
    if rng.random() < 0.5:
        pipeline.system.mailbox(FAULTS_PATH).try_send(
            Envelope.create(FAULT_KIND, {"op": "poison", "count": 5}, ttl=5.0)
        )


def sample(pipeline: Pipeline, outcomes: Outcomes, *, at: float) -> dict[str, float]:
    """One row of the soak table: outcomes so far plus the resources the process is using."""
    return {
        "at_s": round(at, 1),
        "posted": float(outcomes.posted),
        "quarantined": float(outcomes.quarantined),
        "lost": float(len(outcomes.lost)),
        "restarts": float(pipeline.stats().restarts),
        "tasks": float(len(asyncio.all_tasks())),
        "rss_mb": rss_mb(),
        "quarantine_size": float(len(pipeline.system.quarantine)),
        "mailbox_depth": float(sum(box.depth for box in pipeline.system.mailboxes().values())),
    }


async def soak(seconds: float, seed: int, tick: float, workdir: Path) -> dict[str, Any]:
    rng = random.Random(seed)  # reproducible chaos, not cryptography
    audit_path = workdir / "soak-audit.jsonl"
    config = PipelineConfig(
        workers=WORKERS,
        worker_capacity=4,
        ingest_capacity=64,
        stage_capacity=64,
        message_ttl=5.0,
        provider_call_timeout=0.5,
        max_restarts=1_000,
        max_restart_seconds=5.0,
        budget_tokens=10**12,
        budget_usd_micros=10**12,
        quarantine_max_records=100_000,
        faults_enabled=True,
        fault_rate=0.1,
        seed=seed,
    )
    webhook = CountingWebhook()
    pipeline = Pipeline(config, transport=webhook, audit_path=audit_path)
    outcomes = Outcomes()
    pending: deque[tuple[str, float]] = deque()
    samples: list[dict[str, float]] = []
    problems: list[str] = []
    chunk = 0
    documents = iter(generate_documents(CORPUS_CHUNK, seed=seed, injection_rate=0.05))

    async with pipeline:
        loop = asyncio.get_running_loop()
        start = loop.time()
        baseline_tasks = len(asyncio.all_tasks())
        next_tick = start + tick
        rss_start = rss_mb()
        while True:
            now = loop.time()
            if now - start >= seconds:
                break
            document = next(documents, None)
            if document is None:
                chunk += 1
                documents = iter(
                    generate_documents(CORPUS_CHUNK, seed=seed + chunk, injection_rate=0.05)
                )
                document = next(documents)
            message_id = pipeline.submit(document)
            if message_id is not None:
                pending.append((message_id, now))
            else:
                await asyncio.sleep(0.002)
            if now < next_tick:
                continue

            next_tick = now + tick
            inject_chaos(pipeline, rng)
            classify(pipeline, pending, cutoff=now - GRACE_SECONDS, outcomes=outcomes)
            problems.extend(pipeline.ledger.check_invariants())
            samples.append(sample(pipeline, outcomes, at=now - start))

        # Tail: let everything in flight finish, then classify the rest.
        remaining = [message_id for message_id, _ in pending]
        settled = await pipeline.wait_settled(remaining, timeout=60.0)
        classify(pipeline, pending, cutoff=loop.time(), outcomes=outcomes)
        stats = pipeline.stats()
        tasks_end = len(asyncio.all_tasks())
        rss_end = rss_mb()
        duplicates = webhook.duplicates
        exit_code = await pipeline.stop(timeout=15.0)

    chain = verify_file(audit_path)
    summary: dict[str, Any] = {
        "seconds": seconds,
        "seed": seed,
        "accepted": float(pipeline.accepted_total),
        "posted": float(outcomes.posted),
        "denied": float(outcomes.denied),
        "quarantined": float(outcomes.quarantined),
        "lost": float(len(outcomes.lost)),
        "duplicate_sink_writes": float(duplicates),
        "settled_tail": float(settled),
        "restarts": float(stats.restarts),
        "escalations": float(stats.escalations),
        "shed": float(stats.shed),
        "ledger_problems": float(len(problems)),
        "baseline_tasks": float(baseline_tasks),
        "tasks_end": float(tasks_end),
        "rss_start_mb": rss_start,
        "rss_end_mb": rss_end,
        "audit_chain_ok": chain.ok,
        "audit_entries": float(chain.entries),
        "exit_code": float(exit_code),
    }
    summary["samples"] = samples
    return summary


def failures(summary: dict[str, Any], max_rss_growth_mb: float) -> list[str]:
    problems: list[str] = []
    if summary["lost"]:
        problems.append(f"{int(summary['lost'])} accepted messages were lost")
    if summary["duplicate_sink_writes"]:
        problems.append(f"{int(summary['duplicate_sink_writes'])} duplicate webhook posts")
    if summary["ledger_problems"]:
        problems.append("the quota ledger reported violated invariants")
    if not summary["audit_chain_ok"]:
        problems.append("the audit chain does not verify")
    if summary["escalations"]:
        problems.append(f"{int(summary['escalations'])} supervisor escalations")
    if summary["tasks_end"] > summary["baseline_tasks"] + 2:
        problems.append(
            f"asyncio tasks did not return to baseline "
            f"({int(summary['tasks_end'])} vs {int(summary['baseline_tasks'])})"
        )
    growth = float(summary["rss_end_mb"]) - float(summary["rss_start_mb"])
    if growth > max_rss_growth_mb:
        problems.append(f"RSS grew by {growth:.1f} MB (limit {max_rss_growth_mb} MB)")
    if not summary["settled_tail"]:
        problems.append("messages in flight at the end did not settle within 60s")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="fault-injection soak")
    parser.add_argument("--seconds", type=float, default=300.0)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--tick", type=float, default=5.0)
    parser.add_argument("--max-rss-growth-mb", type=float, default=100.0)
    parser.add_argument("--workdir", type=Path, default=ROOT / "var" / "soak")
    args = parser.parse_args(argv)

    args.workdir.mkdir(parents=True, exist_ok=True)
    summary = asyncio.run(soak(args.seconds, args.seed, args.tick, args.workdir))
    write_results(RESULTS, summary)
    samples = summary.pop("samples")
    print_table(
        ("t (s)", "posted", "quarantined", "lost", "restarts", "tasks", "RSS MB", "queued"),
        [
            (
                row["at_s"],
                int(row["posted"]),
                int(row["quarantined"]),
                int(row["lost"]),
                int(row["restarts"]),
                int(row["tasks"]),
                row["rss_mb"],
                int(row["mailbox_depth"]),
            )
            for row in samples[:: max(1, len(samples) // 12)]
        ],
    )
    print()
    print(json.dumps(summary, indent=2, sort_keys=True))
    problems = failures(summary, args.max_rss_growth_mb)
    for problem in problems:
        print(f"SOAK FAILURE: {problem}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
