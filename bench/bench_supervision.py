#!/usr/bin/env python3
"""End-to-end evaluation of the supervision runtime, on CPU, in minutes.

Usage:
  uv run python bench/bench_supervision.py                 # default sizes, writes results
  uv run python bench/bench_supervision.py --check         # enforce the thresholds file
  uv run python bench/bench_supervision.py --schedules 1000 --seconds 5

What is measured:

``restart_latency``   time from killing a worker to a new incarnation running (p50/p95/p99).
``delivery``          randomized crash schedules at several fault rates, crashing every actor
                      class in turn (workers, sink, validator, approval, provider, ingestor and
                      a mix); every accepted message must end exactly once in a sink or in the
                      quarantine (lost = 0, duplicate_sink_writes = 0), with the redelivery
                      count reported. ``delivery_<target>`` holds the per-class breakdown --
                      the sink column is the one the "no duplicate post" claim rests on, since
                      a crash between the outbound post and its acknowledgement is what would
                      produce one.
``overload_{1,3,10}x`` open-loop arrivals at multiples of the measured capacity: goodput, p50/p99
                      end-to-end latency, how much was shed, RSS and the peak asyncio task count.
``budget_supervised`` a tight budget under faults: spend must never exceed it.
``baseline_naive_10x`` the same offered load through ``asyncio.gather`` with retries and no
                      supervision: unbounded tasks, spend charged after the fact, failures lost.

Everything runs against the seeded fault-injecting fake model: no network, no API key.
"""

from __future__ import annotations

import argparse
import asyncio
import random
import sys
import time
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from agent_supervision_tree.agentic.errors import ProviderError
from agent_supervision_tree.agentic.pricing import cost_usd_micros
from agent_supervision_tree.agentic.tools import ToolRegistry
from agent_supervision_tree.runtime.actor import ExitReason
from agent_supervision_tree.runtime.errors import ActorNotRunningError
from agent_supervision_tree.workload.actors import build_extraction_request
from agent_supervision_tree.workload.documents import Document, generate_documents
from agent_supervision_tree.workload.fake_llm import FaultingFakeProvider, FaultProfile
from agent_supervision_tree.workload.pipeline import (
    APPROVAL_PATH,
    INGESTOR_PATH,
    PROVIDER_PATH,
    SINK_PATH,
    VALIDATOR_PATH,
    Pipeline,
    PipelineConfig,
    worker_path,
)
from agent_supervision_tree.workload.schemas import DocumentMessage, validate_extraction
from benchlib import ROOT, check_thresholds, percentile, print_table, rss_mb, write_results

RESULTS = ROOT / "bench" / "results" / "supervision.json"
THRESHOLDS = ROOT / "bench" / "thresholds_supervision.json"
BUDGET_TOKENS = 10_000_000
BUDGET_USD_MICROS = 20_000_000
LATENCY_MS = 2.0
WORKERS_IN_SCHEDULE = 3
_WORKER_PATHS = tuple(worker_path(index) for index in range(WORKERS_IN_SCHEDULE))
_SINGLE_ACTORS = (SINK_PATH, VALIDATOR_PATH, APPROVAL_PATH, PROVIDER_PATH, INGESTOR_PATH)
CRASH_TARGETS: dict[str, tuple[str, ...]] = {
    "worker": _WORKER_PATHS,
    "sink": (SINK_PATH,),
    "validator": (VALIDATOR_PATH,),
    "approval": (APPROVAL_PATH,),
    "provider": (PROVIDER_PATH,),
    "ingestor": (INGESTOR_PATH,),
    "mixed": _WORKER_PATHS + _SINGLE_ACTORS,
}


@dataclass
class TimedWebhook:
    """Records what was posted and when, so the bench can measure end-to-end latency itself."""

    posts: list[tuple[str, float]] = field(default_factory=list)

    def post(self, url: str, payload: Any) -> None:
        document_id = str(payload.get("document_id", ""))
        self.posts.append((document_id, time.perf_counter()))

    def document_ids(self) -> list[str]:
        return [document_id for document_id, _ in self.posts]


def corpus(count: int, seed: int, **rates: float) -> list[Document]:
    return generate_documents(count, seed=seed, **rates)


def provider_for(fault_rate: float, seed: int) -> FaultingFakeProvider:
    return FaultingFakeProvider(
        FaultProfile.at(fault_rate, latency_ms=LATENCY_MS, hang_seconds=5.0),
        seed=seed,
        model_for_role={"extractor": "claude-sonnet-5", "fallback": "claude-haiku-4-5"},
    )


def pipeline_for(
    fault_rate: float, seed: int, *, webhook: TimedWebhook | None = None, **overrides: Any
) -> Pipeline:
    options: dict[str, Any] = {
        "workers": 4,
        "worker_capacity": 4,
        "ingest_capacity": 64,
        "stage_capacity": 64,
        "message_ttl": 5.0,
        "provider_call_timeout": 0.5,
        "max_restarts": 50,
        "max_restart_seconds": 5.0,
        "budget_tokens": BUDGET_TOKENS,
        "budget_usd_micros": BUDGET_USD_MICROS,
        "fault_rate": fault_rate,
        "seed": seed,
    }
    options.update(overrides)
    return Pipeline(
        PipelineConfig(**options),
        provider=provider_for(fault_rate, seed),
        transport=webhook or TimedWebhook(),
    )


# --- restart latency --------------------------------------------------------------------------


async def bench_restart_latency(crashes: int) -> dict[str, float]:
    """Kill a worker repeatedly and measure how long a replacement takes to be running."""
    pipeline = pipeline_for(0.0, seed=1, workers=2, max_restarts=crashes * 2 + 10)
    samples: list[float] = []
    async with pipeline:
        for index in range(crashes):
            path = worker_path(index % 2)
            before = pipeline.system.resolve(path).incarnation
            started = time.perf_counter()
            pipeline.system.kill(path, ExitReason.killed("bench"))
            while True:
                await asyncio.sleep(0)
                try:
                    runnable = pipeline.system.resolve(path)
                except ActorNotRunningError:
                    continue  # between incarnations: the replacement is not up yet
                if runnable.incarnation > before and runnable.running:
                    break
            samples.append((time.perf_counter() - started) * 1000)
            if index % 8 == 7:
                # Let the restart intensity window slide instead of escalating.
                await asyncio.sleep(0.01)
    return {
        "crashes": float(crashes),
        "p50_ms": round(percentile(samples, 0.50), 3),
        "p95_ms": round(percentile(samples, 0.95), 3),
        "p99_ms": round(percentile(samples, 0.99), 3),
        "max_ms": round(max(samples), 3),
    }


# --- delivery under randomized crash schedules --------------------------------------------------


async def _one_schedule(
    rng: random.Random, documents: int, fault_rate: float, seed: int, targets: tuple[str, ...]
) -> tuple[int, int, int, int, int]:
    webhook = TimedWebhook()
    pipeline = pipeline_for(
        fault_rate, seed, webhook=webhook, workers=WORKERS_IN_SCHEDULE, worker_capacity=3
    )
    async with pipeline:
        accepted: list[str] = []
        for document in corpus(documents, seed=seed, injection_rate=0.1, poison_rate=0.1):
            message_id = pipeline.submit(document)
            while message_id is None:
                await asyncio.sleep(0.001)
                message_id = pipeline.submit(document)
            accepted.append(message_id)
        for _ in range(rng.randint(1, 4)):
            await asyncio.sleep(rng.uniform(0.0, 0.01))
            with suppress(ActorNotRunningError):  # the actor may be between incarnations
                pipeline.system.kill(rng.choice(targets), ExitReason.killed("bench"))
        settled = await pipeline.wait_settled(accepted, timeout=30.0)
        if not settled:
            raise RuntimeError("a schedule did not settle within 30s")
        lost = len(pipeline.unsettled(accepted))
        posted_ids = webhook.document_ids()
        duplicates = len(posted_ids) - len(set(posted_ids))
        redeliveries = int(
            pipeline.system.metrics.registry.get_sample_value(
                "ast_redeliveries_total", {"actor": worker_path(0)}
            )
            or 0
        )
        quarantined = len(pipeline.system.quarantine)
        return lost, duplicates, redeliveries, quarantined, len(accepted)


def _empty_totals(schedules: int, documents: int) -> dict[str, float]:
    return {
        "schedules": float(schedules),
        "documents_per_schedule": float(documents),
        "accepted": 0.0,
        "lost": 0.0,
        "duplicate_sink_writes": 0.0,
        "redeliveries": 0.0,
        "quarantined": 0.0,
        "seconds": 0.0,
    }


async def bench_delivery(schedules: int, documents: int) -> dict[str, dict[str, float]]:
    """Randomized crash schedules, spread evenly over every actor class the tree contains.

    Crashing only the extractor workers would leave out the actor whose crash could actually
    produce a duplicate outbound post (the sink, between posting and acknowledging), so the
    campaign rotates through the classes and reports each one separately.
    """
    rng = random.Random(20260920)
    names = list(CRASH_TARGETS)
    per_target = {name: _empty_totals(0, documents) for name in names}
    totals = _empty_totals(schedules, documents)
    started = time.perf_counter()
    for index in range(schedules):
        fault_rate = (0.0, 0.25, 0.5)[index % 3]
        name = names[index % len(names)]
        target_started = time.perf_counter()
        lost, duplicates, redeliveries, quarantined, accepted = await _one_schedule(
            rng, documents, fault_rate, seed=1_000 + index, targets=CRASH_TARGETS[name]
        )
        counted = {
            "accepted": float(accepted),
            "lost": float(lost),
            "duplicate_sink_writes": float(duplicates),
            "redeliveries": float(redeliveries),
            "quarantined": float(quarantined),
        }
        row = per_target[name]
        row["schedules"] += 1
        row["seconds"] = round(row["seconds"] + time.perf_counter() - target_started, 2)
        for key, value in counted.items():
            row[key] += value
            totals[key] += value
    totals["seconds"] = round(time.perf_counter() - started, 2)
    return {"delivery": totals} | {f"delivery_{name}": per_target[name] for name in names}


# --- overload ---------------------------------------------------------------------------------


async def _drive(
    pipeline: Pipeline,
    documents: Sequence[Document],
    rate: float,
    seconds: float,
    submitted: dict[str, float],
) -> tuple[int, int, int]:
    """Offer documents at ``rate`` per second for ``seconds``; returns (offered, accepted, tasks).

    Open loop: arrivals do not wait for the tree. Each wakeup submits everything that became
    due since the last one, so a 10x target is actually offered instead of being throttled by
    the driver's own scheduling.
    """
    loop = asyncio.get_running_loop()
    tick = min(1.0 / rate, 0.002)
    start = loop.time()
    offered = accepted = 0
    peak_tasks = 0
    index = 0
    while True:
        elapsed = loop.time() - start
        if elapsed >= seconds:
            break
        due = int(elapsed * rate) - offered
        for _ in range(max(due, 0)):
            if index >= len(documents):
                break
            document = documents[index]
            index += 1
            offered += 1
            if pipeline.submit(document) is not None:
                accepted += 1
                submitted[document.id] = time.perf_counter()
        peak_tasks = max(peak_tasks, len(asyncio.all_tasks()))
        await asyncio.sleep(tick)
    return offered, accepted, peak_tasks


async def measure_capacity(seconds: float, runs: int = 3) -> dict[str, float]:
    """Median of ``runs`` capacity measurements, because one is noisy on a shared machine.

    Throughput moves with whatever else the machine is doing, and every overload target is a
    multiple of it, so the median of a few runs is what the load levels are scaled from. The
    spread is reported too, so a reader can see how much the number moves.
    """
    samples = [await _one_capacity(seconds) for _ in range(runs)]
    return {
        "documents_per_second": round(percentile(samples, 0.50), 1),
        "min_documents_per_second": round(min(samples), 1),
        "max_documents_per_second": round(max(samples), 1),
        "runs": float(runs),
    }


async def _one_capacity(seconds: float, budget: int = 20_000) -> float:
    """Closed-loop throughput: how many documents a second the tree sinks under backpressure.

    The corpus is generated *before* the clock starts; generating an invoice costs about as
    much as processing one, so leaving it inside would measure the generator, not the tree.
    """
    webhook = TimedWebhook()
    pipeline = pipeline_for(0.0, seed=7, webhook=webhook)
    prepared = corpus(budget, seed=77, injection_rate=0.0)
    async with pipeline:
        loop = asyncio.get_running_loop()
        documents = iter(prepared)
        start = loop.time()
        for document in documents:
            if loop.time() - start >= seconds:
                break
            if pipeline.submit(document) is None:
                await asyncio.sleep(0.001)
        await pipeline.drain(timeout=15.0)
        elapsed = loop.time() - start
        posted = len(webhook.posts)
    return max(posted / max(elapsed, 1e-9), 1.0)


async def bench_overload(
    multiplier: int, capacity: float, seconds: float, seed: int
) -> dict[str, float]:
    webhook = TimedWebhook()
    pipeline = pipeline_for(0.0, seed=seed, webhook=webhook)
    submitted: dict[str, float] = {}
    rate = capacity * multiplier
    prepared = corpus(int(rate * seconds * 1.1) + 200, seed=seed, injection_rate=0.0)
    async with pipeline:
        offered, accepted, peak_tasks = await _drive(pipeline, prepared, rate, seconds, submitted)
        await pipeline.drain(timeout=15.0)
        resident = rss_mb()
        stats = pipeline.stats()
    latencies = [
        (posted_at - submitted[document_id]) * 1000
        for document_id, posted_at in webhook.posts
        if document_id in submitted
    ]
    return {
        "multiplier": float(multiplier),
        "target_per_second": round(rate, 1),
        "offered_per_second": round(offered / seconds, 1),
        "offered": float(offered),
        "accepted": float(accepted),
        "shed": float(stats.shed),
        "posted": float(stats.posted),
        "goodput_per_second": round(stats.posted / seconds, 1),
        "p50_ms": round(percentile(latencies, 0.50), 2),
        "p99_ms": round(percentile(latencies, 0.99), 2),
        "rss_mb": resident,
        "peak_tasks": float(peak_tasks),
    }


# --- budgets: supervised versus naive -----------------------------------------------------------


async def bench_budget_supervised(documents: int, budget_usd_micros: int) -> dict[str, float]:
    webhook = TimedWebhook()
    pipeline = pipeline_for(
        0.3,
        seed=11,
        webhook=webhook,
        budget_usd_micros=budget_usd_micros,
        budget_tokens=BUDGET_TOKENS,
    )
    async with pipeline:
        accepted: list[str] = []
        for document in corpus(documents, seed=31, injection_rate=0.05):
            message_id = pipeline.submit(document)
            while message_id is None:
                await asyncio.sleep(0.001)
                message_id = pipeline.submit(document)
            accepted.append(message_id)
        settled = await pipeline.wait_settled(accepted, timeout=60.0)
        root = pipeline.ledger.node("root")
        refusals = sum(
            pipeline.ledger.node(path).refusals_total for path in pipeline.ledger.paths()
        )
        lost = len(pipeline.unsettled(accepted))
        problems = pipeline.ledger.check_invariants()
    return {
        "documents": float(documents),
        "settled": float(settled),
        "limit_usd_micros": float(root.limit_usd_micros),
        "spent_usd_micros": float(root.spent_usd_micros),
        "over_budget_usd_micros": float(max(0, root.spent_usd_micros - root.limit_usd_micros)),
        "refusals": float(refusals),
        "ledger_problems": float(len(problems)),
        "lost": float(lost),
    }


async def bench_naive_baseline(
    documents: int, budget_usd_micros: int, call_timeout: float
) -> dict[str, float]:
    """``asyncio.gather`` with retries: no mailbox, no reservation, no quarantine.

    This is the shape most agent code has. It is measured the same way as the supervised run so
    the README can compare like with like.
    """
    provider = provider_for(0.3, seed=11)
    registry = ToolRegistry(())
    spent = 0
    posted = 0
    lost = 0
    peak_tasks = 0

    async def handle(document: Document) -> None:
        nonlocal spent, posted, lost
        message = DocumentMessage(id=document.id, text=document.text)
        request = build_extraction_request(message, registry)
        for _ in range(3):
            try:
                async with asyncio.timeout(call_timeout):
                    response = await provider.acomplete(request)
            except (ProviderError, TimeoutError):
                await asyncio.sleep(0.01)
                continue
            # The spend is only known after the call: nothing could have refused it up front.
            spent += cost_usd_micros(
                response.model, response.usage.input_tokens, response.usage.output_tokens
            )
            if response.parsed is None:
                continue
            try:
                validate_extraction(response.parsed)
            except ValidationError:
                continue
            posted += 1
            return
        lost += 1  # nowhere to put it: no quarantine, no redelivery

    started = time.perf_counter()
    tasks = [
        asyncio.create_task(handle(document))
        for document in corpus(documents, seed=31, injection_rate=0.05)
    ]
    peak_tasks = max(peak_tasks, len(asyncio.all_tasks()))
    await asyncio.gather(*tasks, return_exceptions=True)
    elapsed = time.perf_counter() - started
    return {
        "documents": float(documents),
        "limit_usd_micros": float(budget_usd_micros),
        "spent_usd_micros": float(spent),
        "over_budget_usd_micros": float(max(0, spent - budget_usd_micros)),
        "posted": float(posted),
        "lost": float(lost),
        "peak_tasks": float(peak_tasks),
        "rss_mb": rss_mb(),
        "seconds": round(elapsed, 2),
    }


# --- driver -------------------------------------------------------------------------------------


async def run(args: argparse.Namespace) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = {}
    results["restart_latency"] = await bench_restart_latency(args.crashes)
    results.update(await bench_delivery(args.schedules, args.documents))
    capacity = await measure_capacity(args.seconds, args.capacity_runs)
    for multiplier in (1, 3, 10):
        results[f"overload_{multiplier}x"] = await bench_overload(
            multiplier, capacity["documents_per_second"], args.seconds, seed=200 + multiplier
        )
    budget = args.budget_usd_micros
    results["budget_supervised"] = await bench_budget_supervised(args.budget_documents, budget)
    results["baseline_naive_10x"] = await bench_naive_baseline(
        args.budget_documents, budget, call_timeout=0.5
    )
    results["capacity"] = capacity
    return results


def report(results: dict[str, dict[str, float]]) -> None:
    restart = results["restart_latency"]
    delivery = results["delivery"]
    print_table(
        ("metric", "value"),
        [
            (
                "restart latency p50 / p95 / p99 (ms)",
                f"{restart['p50_ms']} / {restart['p95_ms']} / {restart['p99_ms']}",
            ),
            ("crash schedules", int(delivery["schedules"])),
            ("messages accepted", int(delivery["accepted"])),
            ("messages lost", int(delivery["lost"])),
            ("duplicate sink writes", int(delivery["duplicate_sink_writes"])),
            ("redeliveries (worker 0)", int(delivery["redeliveries"])),
            ("quarantined", int(delivery["quarantined"])),
            (
                f"measured capacity (docs/s, median of {int(results['capacity']['runs'])})",
                f"{results['capacity']['documents_per_second']} "
                f"({results['capacity']['min_documents_per_second']}"
                f"-{results['capacity']['max_documents_per_second']})",
            ),
        ],
    )
    print()
    print_table(
        ("crashed actor", "schedules", "accepted", "lost", "duplicate posts", "quarantined"),
        [
            (
                name,
                int(results[f"delivery_{name}"]["schedules"]),
                int(results[f"delivery_{name}"]["accepted"]),
                int(results[f"delivery_{name}"]["lost"]),
                int(results[f"delivery_{name}"]["duplicate_sink_writes"]),
                int(results[f"delivery_{name}"]["quarantined"]),
            )
            for name in CRASH_TARGETS
            if f"delivery_{name}" in results
        ],
    )
    print()
    print_table(
        (
            "load",
            "target/s",
            "offered/s",
            "accepted",
            "shed",
            "goodput/s",
            "p50 ms",
            "p99 ms",
            "RSS MB",
            "peak tasks",
        ),
        [
            (
                f"{int(row['multiplier'])}x",
                row["target_per_second"],
                row["offered_per_second"],
                int(row["accepted"]),
                int(row["shed"]),
                row["goodput_per_second"],
                row["p50_ms"],
                row["p99_ms"],
                row["rss_mb"],
                int(row["peak_tasks"]),
            )
            for key, row in results.items()
            if key.startswith("overload_")
        ],
    )
    print()
    supervised = results["budget_supervised"]
    naive = results["baseline_naive_10x"]
    print_table(
        ("budget run", "limit (uUSD)", "spent (uUSD)", "over budget", "lost", "peak tasks"),
        [
            (
                "supervised",
                int(supervised["limit_usd_micros"]),
                int(supervised["spent_usd_micros"]),
                int(supervised["over_budget_usd_micros"]),
                int(supervised["lost"]),
                "bounded",
            ),
            (
                "naive gather+retry",
                int(naive["limit_usd_micros"]),
                int(naive["spent_usd_micros"]),
                int(naive["over_budget_usd_micros"]),
                int(naive["lost"]),
                int(naive["peak_tasks"]),
            ),
        ],
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="supervision runtime evaluation")
    parser.add_argument("--check", action="store_true", help="enforce the thresholds file")
    parser.add_argument("--crashes", type=int, default=40, help="crashes for the latency bench")
    parser.add_argument("--schedules", type=int, default=120, help="randomized crash schedules")
    parser.add_argument("--documents", type=int, default=8, help="documents per crash schedule")
    parser.add_argument("--seconds", type=float, default=2.0, help="seconds per load level")
    parser.add_argument(
        "--capacity-runs", type=int, default=3, help="capacity runs to take the median of"
    )
    parser.add_argument("--budget-documents", type=int, default=400)
    parser.add_argument("--budget-usd-micros", type=int, default=40_000)
    args = parser.parse_args(argv)

    results = asyncio.run(run(args))
    write_results(RESULTS, results)
    report(results)
    if args.check:
        problems = check_thresholds(results, THRESHOLDS)
        for problem in problems:
            print(f"REGRESSION: {problem}")
        if problems:
            return 1
        print("all supervision benchmarks within thresholds")
    return 0


if __name__ == "__main__":
    sys.exit(main())
