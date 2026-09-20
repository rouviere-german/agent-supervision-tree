"""Prometheus metrics of the runtime, on the same per-app registry as the HTTP metrics.

Counters are incremented inline by the runtime. Gauges (mailbox depth, breaker state, budget
remaining, quarantine size) are refreshed by :meth:`RuntimeMetrics.refresh` when ``/metrics``
is scraped or the tree is introspected, which keeps the hot path free of gauge updates.
"""

from __future__ import annotations

from dataclasses import dataclass

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

__all__ = ["RuntimeMetrics", "create_runtime_metrics"]


@dataclass(frozen=True, slots=True)
class RuntimeMetrics:
    registry: CollectorRegistry
    restarts_total: Counter
    escalations_total: Counter
    child_exits_total: Counter
    messages_processed_total: Counter
    message_failures_total: Counter
    redeliveries_total: Counter
    quarantine_total: Counter
    quarantine_overflow_total: Counter
    ingress_accepted_total: Counter
    ingress_shed_total: Counter
    goodput_total: Counter
    provider_calls_total: Counter
    provider_fallbacks_total: Counter
    provider_deferred_total: Counter
    budget_refusals_total: Counter
    approvals_total: Counter
    mailbox_depth: Gauge
    mailbox_credits: Gauge
    breaker_state: Gauge
    budget_remaining: Gauge
    quarantine_size: Gauge
    tasks_active: Gauge
    restart_latency_seconds: Histogram
    message_latency_seconds: Histogram
    provider_call_seconds: Histogram
    event_loop_lag_seconds: Gauge


def create_runtime_metrics(
    registry: CollectorRegistry | None = None, namespace: str = "ast"
) -> RuntimeMetrics:
    reg = registry if registry is not None else CollectorRegistry()

    def counter(name: str, doc: str, labels: tuple[str, ...] = ()) -> Counter:
        return Counter(name, doc, list(labels), namespace=namespace, registry=reg)

    def gauge(name: str, doc: str, labels: tuple[str, ...] = ()) -> Gauge:
        return Gauge(name, doc, list(labels), namespace=namespace, registry=reg)

    return RuntimeMetrics(
        registry=reg,
        restarts_total=counter("restarts_total", "Actor restarts.", ("actor", "reason")),
        escalations_total=counter(
            "escalations_total", "Supervisors that exceeded restart intensity.", ("supervisor",)
        ),
        child_exits_total=counter("child_exits_total", "Child exits seen.", ("actor", "reason")),
        messages_processed_total=counter(
            "messages_processed_total", "Messages acknowledged by actors.", ("actor",)
        ),
        message_failures_total=counter(
            "message_failures_total", "Messages rejected by a behavior.", ("actor", "reason")
        ),
        redeliveries_total=counter("redeliveries_total", "Messages redelivered.", ("actor",)),
        quarantine_total=counter("quarantine_total", "Messages quarantined.", ("reason",)),
        quarantine_overflow_total=counter(
            "quarantine_overflow_total", "Messages dropped because the quarantine was full."
        ),
        ingress_accepted_total=counter("ingress_accepted_total", "Messages accepted."),
        ingress_shed_total=counter("ingress_shed_total", "Messages shed with 429."),
        goodput_total=counter("goodput_total", "Messages that reached a sink.", ("outcome",)),
        provider_calls_total=counter(
            "provider_calls_total", "Provider call attempts.", ("model", "outcome")
        ),
        provider_fallbacks_total=counter(
            "provider_fallbacks_total", "Model fallbacks.", ("from_role", "to_role")
        ),
        provider_deferred_total=counter(
            "provider_deferred_total", "Calls answered by the deterministic defer.", ("reason",)
        ),
        budget_refusals_total=counter(
            "budget_refusals_total", "Reservations refused by the ledger.", ("node",)
        ),
        approvals_total=counter("approvals_total", "Approval decisions.", ("decision",)),
        mailbox_depth=gauge("mailbox_depth", "Queued messages.", ("actor",)),
        mailbox_credits=gauge("mailbox_credits", "Free credits.", ("actor",)),
        breaker_state=gauge("breaker_state", "0 closed, 1 half-open, 2 open.", ("provider",)),
        budget_remaining=gauge("budget_remaining", "Remaining quota.", ("node", "dimension")),
        quarantine_size=gauge("quarantine_size", "Records in the quarantine."),
        tasks_active=gauge("tasks_active", "asyncio tasks alive in the process."),
        restart_latency_seconds=Histogram(
            "restart_latency_seconds",
            "Time from handling a child exit to the replacement being started.",
            namespace=namespace,
            registry=reg,
            buckets=(0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
        ),
        message_latency_seconds=Histogram(
            "message_latency_seconds",
            "Ingress to sink latency.",
            namespace=namespace,
            registry=reg,
            buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
        ),
        provider_call_seconds=Histogram(
            "provider_call_seconds",
            "Provider call latency per attempt.",
            ["model"],
            namespace=namespace,
            registry=reg,
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
        ),
        event_loop_lag_seconds=gauge("event_loop_lag_seconds", "Measured event-loop lag."),
    )
