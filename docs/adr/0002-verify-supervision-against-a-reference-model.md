# 2. Verify supervision against a pure reference model

- Status: accepted
- Date: 2026-09-20

## Context

The point of this repo is the *semantics* of supervision, not a wrapper around `asyncio`. Those
semantics are easy to state and easy to get subtly wrong: which siblings a `one_for_all` restart
terminates and in which order, whether a `transient` child comes back after a normal exit,
whether a `temporary` child's spec is deleted when it is caught in a sweep, when the restart
intensity window slides, and what a supervisor does after it has escalated.

Example-based tests catch the cases their author thought of. The cases that break real systems
are the interleavings nobody wrote down: a second child exiting while the first restart sweep is
still running, an exit arriving for an incarnation that is already gone, a window boundary hit
exactly between two crashes.

Three options were considered:

1. Example-based tests only, one per rule and per strategy.
2. Randomized sequences against the operational supervisor alone, asserting only generic
   invariants (nothing crashes, children eventually run).
3. A second, independent implementation of the OTP rules as a pure state machine, with the two
   required to agree on randomly generated exit sequences.

## Decision

We do (3), and keep (1) as well.

`runtime/reference.py` is a pure model: no asyncio, no tasks, no mailboxes, about 120 lines of
dataclasses that answer one question -- given this exit at this time, what does the supervisor
do? The operational supervisor does not import it; they are independent implementations of the
same specification, with the OTP rules quoted in both docstrings.

`tests/unit/test_supervisor_reference.py` drives both with a hypothesis `RuleBasedStateMachine`.
Each step kills a randomly chosen running child with a randomly chosen exit class, or advances a
controlled clock. After every step, the live tree and the model must agree on the running
children, the number of starts per child, whether the supervisor is still alive, and the exact
decision: which children were terminated, restarted, removed, and whether it escalated.

The operational supervisor therefore exposes its decisions as values (`reference.Decision`)
rather than only as side effects.

## Consequences

- A behavioural difference from OTP shows up as a shrunk, replayable counterexample naming the
  strategy, the restart types and the exit sequence, instead of as a flaky integration test.
- The model doubles as documentation: `docs/SUPERVISION.md` describes what it computes, and the
  one deliberate difference from OTP (escalation exits `error`, not `shutdown`) is written down
  in both implementations and in the failure matrix.
- Two implementations must be changed together. That is the intended cost: it makes changing the
  semantics a deliberate act rather than an accident.
- The model covers decisions, not timing. Restart latency and delivery under concurrency are
  measured operationally instead (`bench/bench_supervision.py`).
