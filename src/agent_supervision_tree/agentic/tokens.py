"""Token estimators.

Two different jobs that must not share one expression, or the budget guarantee becomes a
tautology (reserving exactly what the provider will report can never overshoot):

:func:`estimate_tokens`
    What the provider actor *reserves* before a call. It errs high on purpose: the reservation
    is an upper bound held against every ancestor budget, so an estimate that is too small is
    the one failure mode that lets real spend pass a limit. It counts UTF-8 bytes, not
    characters, because a byte-level BPE charges three tokens for a CJK character that
    ``len()`` counts once, and it assumes ``BYTES_PER_TOKEN`` bytes per token rather than the
    familiar four: the text being estimated is canonical JSON, which is punctuation-dense and
    tokenizes far denser than prose.

:func:`simulate_usage_tokens`
    What the *fake* providers report as usage, modelling an ordinary tokenizer at roughly four
    characters per token. Tests and benches use it so the reported usage stays below the
    reservation the way a real tokenizer's output normally does -- but not by construction.

Neither is exact. Only the provider's own counter (``usage`` in the response, or Anthropic's
``count_tokens`` endpoint) is, and the ledger therefore still settles honestly when a provider
reports more than was held: see :meth:`QuotaLedger.settle` and ADR 0004.
"""

from __future__ import annotations

import math

__all__ = ["BYTES_PER_TOKEN", "USAGE_CHARS_PER_TOKEN", "estimate_tokens", "simulate_usage_tokens"]

BYTES_PER_TOKEN = 2.5
"""Conservative bytes-per-token ratio for reservations (about 1.6x the 4-chars heuristic)."""

USAGE_CHARS_PER_TOKEN = 4.0
"""Average-case characters-per-token ratio the fake providers report usage with."""


def estimate_tokens(text: str, *, bytes_per_token: float = BYTES_PER_TOKEN) -> int:
    """Estimate, erring high, how many tokens ``text`` costs. At least 1."""
    if bytes_per_token <= 0:
        raise ValueError("bytes_per_token must be positive")
    return max(1, math.ceil(len(text.encode("utf-8")) / bytes_per_token))


def simulate_usage_tokens(text: str, *, chars_per_token: float = USAGE_CHARS_PER_TOKEN) -> int:
    """Model the usage an ordinary tokenizer would report for ``text``. At least 1."""
    if chars_per_token <= 0:
        raise ValueError("chars_per_token must be positive")
    return max(1, math.ceil(len(text) / chars_per_token))
