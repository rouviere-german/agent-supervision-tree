#!/usr/bin/env python3
"""Micro-benchmarks for the hot paths, with CI regression thresholds.

Usage:
  uv run python bench/bench_baseline.py            # run, write bench/results/latest.json
  uv run python bench/bench_baseline.py --check    # exit 1 when bench/thresholds.json is exceeded

These cover the per-message costs the runtime pays on every hop (mailbox handoff, ledger
reservation, supervision decision, audit append) plus the shared baseline (redaction, rate
limiting, request hashing). Thresholds are deliberately loose -- several times the numbers
measured on a laptop -- because shared CI runners are noisy; they catch order-of-magnitude
regressions, not small drifts. The end-to-end evaluation lives in bench/bench_supervision.py.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import tempfile
from pathlib import Path

from agent_supervision_tree.agentic.audit import AuditLog, verify_file
from agent_supervision_tree.agentic.providers.base import ChatMessage, LLMRequest
from agent_supervision_tree.core.ratelimit import RateLimiter
from agent_supervision_tree.core.redaction import redact_text
from agent_supervision_tree.runtime.ledger import QuotaLedger
from agent_supervision_tree.runtime.mailbox import Mailbox
from agent_supervision_tree.runtime.messages import Envelope
from agent_supervision_tree.runtime.reference import ReferenceSupervisor
from benchlib import ROOT, check_thresholds, print_table, timed, write_results

RESULTS = ROOT / "bench" / "results" / "latest.json"
THRESHOLDS = ROOT / "bench" / "thresholds.json"


def bench_redaction() -> dict[str, float]:
    line = (
        "user jane.doe@example.com paid with 4111 1111 1111 1111 from +49 30 1234 5678; "
        + "lorem ipsum dolor sit amet " * 35
    )
    return timed(lambda: redact_text(line), runs=2_000)


def bench_rate_limiter() -> dict[str, float]:
    limiter = RateLimiter(rate=1e9, burst=1_000_000_000)
    return timed(lambda: limiter.check("client"), runs=20_000)


def bench_request_hash() -> dict[str, float]:
    messages = tuple(ChatMessage.user(f"message {i} " + "x" * 400) for i in range(10))
    request = LLMRequest(system="bench", messages=messages)
    return timed(request.sha256, runs=2_000)


def bench_audit_append(workdir: Path) -> dict[str, float]:
    log = AuditLog(workdir / "append.jsonl")
    counter = iter(range(10**9))
    return timed(
        lambda: log.append("bench", "step", {"index": next(counter), "note": "x" * 100}), runs=200
    )


def bench_audit_verify(workdir: Path) -> dict[str, float]:
    path = workdir / "verify.jsonl"
    log = AuditLog(path)
    for index in range(1_000):
        log.append("bench", "step", {"index": index, "note": "x" * 100})
    return timed(lambda: verify_file(path), runs=10, warmup=1)


def bench_envelope_create() -> dict[str, float]:
    payload = {"id": "doc-abcdef", "text": "invoice line " * 20}
    return timed(lambda: Envelope.create("doc.ingest", payload, ttl=5.0), runs=5_000)


def bench_mailbox_roundtrip() -> dict[str, float]:
    """One full mailbox hop: send, receive, ack (the per-message cost of every stage)."""
    loop = asyncio.new_event_loop()
    try:
        mailbox = Mailbox("bench", 8)
        envelope = Envelope.create("doc.ingest", {"id": "doc-abcdef", "text": "hello"})

        async def hop() -> None:
            mailbox.try_send(envelope)
            await mailbox.receive()
            mailbox.ack()

        return timed(lambda: loop.run_until_complete(hop()), runs=5_000)
    finally:
        loop.close()


def bench_ledger_reserve_settle() -> dict[str, float]:
    ledger = QuotaLedger("root", tokens=10**12, usd_micros=10**12)
    ledger.add_node("root/pool", tokens=10**12, usd_micros=10**12)
    ledger.add_node("root/pool/w0", tokens=10**12, usd_micros=10**12)

    def reserve_and_settle() -> None:
        reservation = ledger.reserve("root/pool/w0", tokens=100, usd_micros=100)
        ledger.settle(reservation, tokens=80, usd_micros=80)

    return timed(reserve_and_settle, runs=20_000)


def bench_supervision_decision() -> dict[str, float]:
    """The pure decision the operational supervisor is verified against."""
    specs = [(f"c{index}", "permanent") for index in range(8)]
    model = ReferenceSupervisor.build("one_for_all", 10**9, 10.0, specs)  # type: ignore[arg-type]
    counter = iter(range(10**9))

    def decide() -> None:
        model.child_exit("c3", "abnormal", float(next(counter)))

    return timed(decide, runs=5_000)


def run() -> dict[str, dict[str, float]]:
    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        return {
            "redact_text_1kb": bench_redaction(),
            "rate_limiter_check": bench_rate_limiter(),
            "llm_request_sha256_10_messages": bench_request_hash(),
            "audit_append_fsync": bench_audit_append(workdir),
            "audit_verify_1000_entries": bench_audit_verify(workdir),
            "envelope_create": bench_envelope_create(),
            "mailbox_send_receive_ack": bench_mailbox_roundtrip(),
            "ledger_reserve_settle": bench_ledger_reserve_settle(),
            "supervision_decision_one_for_all_8": bench_supervision_decision(),
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="baseline micro-benchmarks")
    parser.add_argument("--check", action="store_true", help="enforce bench/thresholds.json")
    args = parser.parse_args(argv)

    results = run()
    write_results(RESULTS, results)
    print_table(
        ("benchmark", "runs", "p50 (us)", "p95 (us)"),
        [
            (name, int(stats["runs"]), stats["p50_us"], stats["p95_us"])
            for name, stats in results.items()
        ],
    )
    if args.check:
        problems = check_thresholds(results, THRESHOLDS)
        for problem in problems:
            print(f"REGRESSION: {problem}")
        if problems:
            return 1
        print("all benchmarks within thresholds")
    return 0


if __name__ == "__main__":
    sys.exit(main())
