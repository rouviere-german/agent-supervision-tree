# 4. Budgets reserve before the call and settle after it

- Status: accepted
- Date: 2026-09-20

## Context

Agent systems usually charge a budget *after* a model call returns, because that is when the
usage is known. With one worker that is fine. With a pool of workers sharing a budget it is not:
*n* workers can each read the same remaining balance, all find it sufficient, and all spend. The
overspend is bounded by the number of concurrent calls times the cost of the largest call, which
is exactly what a cost incident is made of.

A second problem is hierarchy. A limit per worker does not bound the pool, and a limit on the
pool does not bound a single runaway worker. Both are needed, and a check against one of them
says nothing about the other.

## Decision

The ledger mirrors the supervision tree (`root`, `root/extractors`, `root/extractors/w0`) and
works in two phases:

1. **Reserve** an upper bound before the call: the estimated input tokens plus `max_tokens`, and
   what that would cost at the model's price. The reservation is checked against the remainder
   (`limit - spent - reserved`) of the node *and every ancestor*, and applied to all of them
   under one lock, or refused without touching any.
2. **Settle** with the real usage afterwards, or **release** when the call produced no spend.
   Settling above the reservation is recorded honestly and counted as an overshoot rather than
   hidden.

Because reservations are held until settlement, the invariant `spent + reserved <= limit` holds
at every node at every moment, whatever the concurrency. Calls that are billed even though they
failed (truncations, refusals after partial output) are settled, not released, so failures are
paid for in the ledger too.

When the root node is exhausted, the provider actor stops calling any model at all and answers
with a deterministic defer: a system-wide cost breaker.

## Consequences

- Refusals happen *before* the money is spent, so a budget can be enforced rather than merely
  observed. `tests/unit/test_ledger.py::test_no_node_ever_exceeds_its_limit_under_random_reservations` and the budget bench check the invariant
  under randomized interleavings, and the bench compares it with a naive gather-and-retry
  baseline that charges after the fact and does exceed its budget.
- Reserving an upper bound means the budget is used conservatively: a call that turns out cheap
  still held `max_tokens` worth of quota while it ran. With a small budget and high concurrency
  that refuses calls which would have fitted. `max_tokens` per request is the tuning knob.
- The estimate must be an upper bound, so the token estimator errs high and unknown models are
  charged at the highest listed price.
- The ledger is in memory and synchronous (one lock, no `await`), which is what keeps a
  reservation atomic. Making it durable or shared across processes is out of scope (ADR 0003).
