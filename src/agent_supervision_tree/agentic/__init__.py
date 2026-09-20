"""Agentic kit: providers, untrusted-data envelopes, tool registry, approvals, budgets, audit log.

The security model does not depend on the model behaving well. Tool outputs are wrapped as data,
but the enforcing controls are architectural: a least-privilege registry decides which tools
exist, a human approval bound to the exact call hash gates side effects, budgets cap spend, and
every decision lands in a tamper-evident audit log.
"""
