"""Agent Supervision Tree: an asyncio actor runtime with Erlang/OTP-style supervision.

Sub-packages:
- ``runtime``: mailboxes with credit backpressure, actors, supervisors and their pure reference
  model, the hierarchical quota ledger, the quarantine, the hash-chained audit and the provider
  actor (bulkhead, circuit breaker, retries, deadlines, model fallback).
- ``workload``: the reference pipeline (ingestor, extractor pool, validator, approval-gated
  webhook sink), its synthetic documents and the seeded fault-injecting fake model.
- ``core``: settings, logging with PII redaction, tracing, metrics, hardened FastAPI app factory.
- ``agentic``: LLM provider protocol, untrusted-data envelopes, tool registry, approvals,
  budgets and a hash-chained audit log.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
