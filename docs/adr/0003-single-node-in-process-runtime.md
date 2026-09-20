# 3. A single-node, in-process runtime; durability and distribution are non-goals

- Status: accepted
- Date: 2026-09-20

## Context

"Do not lose work when something fails" can mean two very different things:

1. *Fault containment*: a worker crash, a provider brownout, a poison message or a runaway retry
   must not take down the process, lose in-flight messages or overspend the budget.
2. *Durability*: work survives the death of the process itself, so a `kill -9` or a node failure
   loses nothing.

(2) needs a write-ahead log or an external queue on every hop, an idempotency story for every
side effect, and a recovery protocol. That is a different system, with different benchmarks
(fsync cost, recovery time) and a different failure matrix.

## Decision

This repo does (1) only, in one process, with everything in memory: mailboxes, the in-flight
table of unacknowledged messages, the quarantine, the quota ledger and the audit chain. The tree
is static and single-node. There is no persistence of message state, no leader election and no
cross-process queue.

Concretely:

- the delivery guarantee is stated *within one process life*: every message the ingress accepted
  ends exactly once in a sink or in the quarantine, unless the process itself dies;
- when the root supervisor escalates, the system stops with exit code 70 (`EX_SOFTWARE`) and
  expects a process supervisor (systemd, Kubernetes) to restart it; in-flight work is lost then,
  and the README says so under Non-goals;
- the audit chain is written through to a file when a path is configured, but the hot path keeps
  it in memory: an `fsync` per restart would dominate the restart latency this repo reports;
- durable, replayable agent execution is the subject of a different project
  (`replayable-agent-runtime`), and horizontal scale-out of the tree is not attempted here.

## Consequences

- The numbers in the README mean what they say: restart latency is measured in fractions of a
  millisecond because nothing touches the disk on that path.
- Memory is the bound that matters, so every queue is credit-bounded and the overload bench
  reports RSS. There is no unbounded on-disk backlog to hide behind.
- An operator who needs (2) must put a durable queue in front of the ingress and make the
  webhook sink idempotent; the sink's deduplication by message id is the hook for that.
- A single event loop means a CPU-bound behaviour would stall the tree. The loop-lag gauge and
  the `/healthz` threshold exist to make that visible, and the reference workload keeps its
  behaviours I/O-bound.
