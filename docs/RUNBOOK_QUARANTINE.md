# Runbook: the quarantine

A message lands in the quarantine when it could not be processed and the runtime has stopped
trying: its attempts ran out, its deadline passed, it was addressed to an actor that does not
accept its kind, or no model could serve it within the budget. The quarantine is in memory,
bounded, and exportable, so the messages are visible instead of silently dropped.

## 1. Notice

Alert on either of these:

* `ast_quarantine_size` rising steadily (work is arriving that the pipeline cannot process);
* `ast_quarantine_overflow_total > 0` (the bound was hit and records were dropped: those
  messages *are* lost, and this must never be non-zero).

`/readyz` stays ready while the quarantine fills: quarantining is the system working as
designed, not an outage.

## 2. Look

```bash
# Group by cause first; the reason tells you which of the sections below applies.
curl -sS -H "X-API-Key: $ADMIN_KEY" http://localhost:8000/v1/quarantine | jq '.reasons'

# Then read the records for one cause.
curl -sS -H "X-API-Key: $ADMIN_KEY" \
  "http://localhost:8000/v1/quarantine?reason=schema_violation&limit=20" | jq '.records[0]'
```

Payloads are redacted before they are stored, so a record is safe to paste into a ticket. Each
record carries `message_id`, `kind`, `attempts`, `reason`, the `actor` that gave up and the
`traceparent`, so the trace of the original request is one click away.

| Reason | What it means | Usual cause |
|---|---|---|
| `schema_violation` | The model's answer did not match the closed extraction schema | A prompt injection in the document, or a genuinely unreadable document |
| `malformed_output` | The answer was not JSON at all | A degraded model, or a poison document |
| `deferred` | No model in the chain could serve the call | Budget exhausted, breaker open, or the provider is down |
| `deadline_exceeded` | The message ran past its TTL | Overload, or a provider that hangs |
| `downstream_full` | A later stage never had a free credit before the deadline | The sink or validator is stuck |
| `actor_error` / `actor_killed` | The worker crashed repeatedly while holding this message | A bug in the behaviour, or fault injection |
| `webhook_failed` | The outbound call failed on every attempt | The webhook endpoint is down |
| `approval_mismatch` | A decision did not match the call it was asked about | A bug or an attack; investigate before replaying |

## 3. Fix the cause

* **Budget** (`deferred`): check `/v1/stats` for `budget`. Raise `APP_BUDGET_MAX_USD` or wait
  for the window to reset. The system-wide cost breaker opens when the root node is exhausted.
* **Provider** (`deferred`, `deadline_exceeded`): check `breaker` in `/v1/stats`. An `open`
  breaker means consecutive failures; it half-opens by itself after the reset timeout.
* **Injections** (`schema_violation`): the closed schema did its job. Nothing was posted and
  nothing was rerouted. Keep the records as evidence, and only replay documents you trust.
* **Poison** (`malformed_output`): the document cannot be extracted at all. Do not replay it.
* **`approval_mismatch`**: never replay before finding out why a decision did not match its
  call. This is the one reason that indicates a security problem rather than a capacity one.

## 4. Export before you replay

```bash
curl -sS -H "X-API-Key: $ADMIN_KEY" \
  "http://localhost:8000/v1/quarantine?limit=500" > quarantine-$(date +%F).json
```

The runtime writes the same records with `Quarantine.export_jsonl`; the demo does it at the end
of every run (`var/demo/quarantine.jsonl`). Replaying removes the record, so export first.

## 5. Replay

```bash
curl -sS -X POST -H "X-API-Key: $ADMIN_KEY" -H "Content-Type: application/json" \
  -d '{"message_ids": ["<id1>", "<id2>"]}' \
  http://localhost:8000/v1/quarantine/replay
```

What happens:

* the record is removed from the quarantine and re-ingested with its **original message id**,
  so the sink's deduplication still applies and a message that was in fact posted earlier
  cannot be posted twice;
* replaying goes through the normal ingress mailbox, so a replay during overload is shed and
  reported in the `shed` list of the response: retry those later;
* records whose payload was not JSON (internal control messages) are not replayable; they are
  reported in `shed` and stay in the quarantine;
* every replay is written to the audit chain as `quarantine.replayed`.

Replay in batches of at most 100 ids, and check `/v1/stats` between batches.

## 6. Verify

```bash
curl -sS -H "X-API-Key: $ADMIN_KEY" http://localhost:8000/v1/stats | jq '.stats'
```

`posted + denied + quarantined` should equal `accepted` once the queues are idle; that is the
invariant the delivery bench and the soak job check continuously
(`bench/bench_supervision.py`, `bench/soak.py`).

Then verify the audit log that recorded the whole incident:

```bash
uv run agent-supervision-tree audit verify var/audit.jsonl
```

The command checks the hash chain and, by default, every head recorded in `var/audit.jsonl.anchor`
against the log's prefix of that length, so a truncated or rolled-back log fails here instead of
looking clean. `--no-anchor` checks the chain alone, and `--expect-head <hash>` compares against a
head you kept elsewhere -- which is the only check that survives an attacker who can rewrite both
files, so ship the head (`/v1/stats`, field `audit_head`) off the box.
