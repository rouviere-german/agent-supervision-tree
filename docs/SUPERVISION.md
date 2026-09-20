# Supervision semantics and the failure matrix

This is the contract the runtime implements. The rules follow the Erlang/OTP `supervisor`
behaviour (OTP design principles, *Supervisor Behaviour*); the differences are listed at the
end. Everything here is verified twice: once against a pure reference model
(`src/agent_supervision_tree/runtime/reference.py`, driven by
`tests/unit/test_supervisor_reference.py::TestSupervisionMachine`) and once operationally
(`tests/unit/test_supervisor.py`).

## Vocabulary

| Term | Meaning |
|---|---|
| Slot | A named position in a supervisor's child list. It owns the mailbox and survives restarts. |
| Incarnation | One running instance of an actor in a slot. A restart creates incarnation *n+1*. |
| Exit class | `normal` (the behaviour returned or raised `ActorExit`), `shutdown` (asked to stop), `abnormal` (an exception, a kill, or a spawned task that failed). |
| Restart type | `permanent` (always restarted), `transient` (restarted only after an abnormal exit), `temporary` (never restarted; the spec is deleted on exit). |
| Restart intensity | More than `max_restarts` restarts within `max_seconds` makes the supervisor give up and escalate. |

## Strategies

| Strategy | On an abnormal child exit |
|---|---|
| `one_for_one` | Restart that child only. |
| `one_for_all` | Terminate every other running child in reverse start order, then start every remaining spec in start order. |
| `rest_for_one` | The same, but only for the specs that come *after* the exited one in start order. |

A `temporary` sibling caught in a `one_for_all` or `rest_for_one` sweep is terminated and its
spec is deleted, so it is not started again.

## Failure matrix

| What fails | Detected by | Immediate effect | Message in flight | Where to look |
|---|---|---|---|---|
| Behaviour raises `MessageFailed` (malformed model output, schema violation) | The runner around `receive` | The actor keeps running; the message is requeued at the tail with `attempt + 1` | Redelivered until `max_attempts`, then quarantined | `ast_message_failures_total`, `ast_redeliveries_total` |
| Behaviour raises `MessageFailed(retryable=False)` (refusal, invalid document) | The runner | The actor keeps running | Quarantined at once | `ast_quarantine_total{reason}` |
| Behaviour raises any other exception, including `SystemExit` from a third-party handler | The runner | The incarnation exits `error`; the supervisor decides | Pushed back to the head of the mailbox with `attempt + 1`; quarantined if the attempts are used up | `ast_child_exits_total`, `ast_restarts_total` |
| A task the actor spawned fails | `ActorContext.spawn` done-callback | The actor exits `error:spawned_task_failed` (no failure is lost in the background) | As above | `ast_child_exits_total` |
| `on_start` raises or takes longer than `start_timeout` | The runner | The incarnation exits `error`; a restart loop that exceeds the intensity escalates | Nothing was dequeued yet | `ast_restarts_total` |
| Deadline passes before the message is handled | The runner, before `receive` | The actor keeps running | Quarantined as `deadline_exceeded` | `ast_quarantine_total{reason="deadline_exceeded"}` |
| Message kind the actor does not accept | The runner, before `receive` | The actor keeps running | Quarantined as `unexpected_kind` | `ast_quarantine_total` |
| Provider hangs | `asyncio.timeout` in the provider actor, bounded by the call timeout *and* the message deadline | The call is cancelled; the next attempt or role is tried | Deferred, then quarantined if nothing serves it | `ast_provider_calls_total{outcome="timeout"}` |
| Provider returns 429 / 5xx | The provider actor | Jittered retry (`retry-after` is a floor); after `failure_threshold` consecutive failures the breaker opens | The call is deferred if the chain is exhausted | `ast_breaker_state`, `ast_provider_calls_total` |
| Provider actor itself crashes | Its supervisor | Every in-flight reply future fails with `ProviderCrashedError`; the actor is restarted | The caller turns it into `MessageFailed`, so the message is redelivered | `ast_child_exits_total{actor="root/provider"}` |
| Budget would be exceeded | The quota ledger, before the call | The reservation is refused and audited; the next role or a deterministic defer is used | Quarantined as `deferred` | `ast_budget_refusals_total`, `ast_budget_remaining` |
| Root budget exhausted | The ledger root node | System-wide cost breaker: no model is called at all, everything defers | Quarantined as `deferred` | `ast_provider_deferred_total{reason="cost_breaker_open"}` |
| Downstream mailbox full | `forward` in the workload actors | The producer waits for a credit until the message deadline | Failed as `downstream_full`, then redelivered | `ast_mailbox_credits`, `ast_mailbox_depth` |
| Ingress mailbox full | `Pipeline.submit` | The request is shed with HTTP 429 and `Retry-After` | Never accepted, so never owed | `ast_ingress_shed_total` |
| Supervisor exceeds its restart intensity | The supervisor itself | Terminates its children in reverse start order and exits `error:max_restart_intensity` | The mailboxes survive; the parent restarts the whole subtree | `ast_escalations_total`, audit `supervisor.escalated` |
| The **root** supervisor escalates | `ActorSystem` | The system stops and sets exit code 70 (`EX_SOFTWARE`) so a process supervisor restarts it | In-flight work is lost with the process: durability is a non-goal (ADR 0003) | `/readyz` fails, audit `system.stopped` |

## Delivery guarantee

For every message the ingress accepted, exactly one of these is true once the system is idle:

* it reached the sink exactly once (`posted` or `denied`), or
* it is in the quarantine with a reason and its attempt count.

The mechanisms behind that:

1. A mailbox credit is returned only on *acknowledgement*, and a message is acknowledged only
   after `receive` returns. A crash therefore never acknowledges anything.
2. The runner holds unacknowledged messages in its in-flight table and hands them back to the
   mailbox (head of the queue, in order) when the incarnation ends. That exit path is a
   `finally` block, and the runner catches `BaseException`, so a behaviour that raises
   `SystemExit` -- `sys.exit()` inside a third-party tool handler -- is contained like any other
   crash instead of stranding the message and its credit. The single exception is
   `KeyboardInterrupt`: the exit path runs, and then it is re-raised so an operator's interrupt
   still reaches the process.
3. Attempts are counted per message. When they run out, the message moves to the quarantine
   instead of circling forever.
4. The sink deduplicates by message id, so a redelivery after a crash cannot post twice.

The one hole is deliberate and measured: if the quarantine is full, the record is dropped and
`ast_quarantine_overflow_total` is incremented. The delivery bench asserts that counter is zero.

## Differences from OTP

| OTP | Here | Why |
|---|---|---|
| A supervisor that exceeds its intensity exits with reason `shutdown` | Exits with `error:max_restart_intensity` | The parent should treat an escalation as a failure, so that a `transient` sub-supervisor is restarted as well as a `permanent` one. |
| `significant` children and `auto_shutdown` | Not implemented | Not needed by the reference workload; the root's own exit stops the system. |
| Dynamic children (`start_child`, `terminate_child`) | Static trees only | Every actor path, mailbox capacity, ledger node and tool allowlist is known at build time, which is what makes the introspection API and the least-privilege checks total. |
| Process links across the whole VM | `ActorSystem.link` between two paths | Enough for the workload; there is no distribution (ADR 0003). |

## Tuning

| Setting | Effect |
|---|---|
| `APP_WORKERS`, `APP_WORKER_MAILBOX_CAPACITY` | The memory bound of the pool: at most `workers * capacity` documents are in the extractor stage. |
| `APP_MAILBOX_CAPACITY` | Ingress depth, and therefore how bursty a load is absorbed before 429s start. |
| `APP_MAX_ATTEMPTS` | How often a message is redelivered before it is quarantined. Higher means more work repeated after a crash. |
| `APP_MAX_RESTARTS`, `APP_MAX_RESTART_SECONDS` | Restart intensity. Too high hides a crash loop; too low turns a transient provider failure into an escalation. |
| `APP_MESSAGE_TTL_SECONDS` | The end-to-end deadline that bounds every wait (mailbox, provider call, approval). |
| `APP_PROVIDER_BULKHEAD` | Concurrent provider calls. It caps how much a slow provider can occupy. |
