"""A seeded, fault-injecting fake model for tests, the demo and the benches.

Per call the provider draws from a :class:`FaultProfile`: added latency, a hung call (sleeps
until the caller's deadline cancels it), a 429, a 5xx-style unavailability, or garbage output.
Otherwise it answers with the ground-truth extraction of the document it finds inside the
request's untrusted-data envelope. It is deliberately *gullible*: when the document carries an
injection it obeys and adds the forbidden ``webhook_url`` field, so tests can prove that the
closed schema, not the model's good behaviour, is what keeps injected text from rerouting
anything. Poison documents always get garbage. Usage is estimated so budgets are exercised
with real prices for the model name the role maps to.
"""

from __future__ import annotations

import asyncio
import json
import random
from collections.abc import Mapping
from dataclasses import dataclass

from pydantic import JsonValue

from agent_supervision_tree.agentic.envelope import parse_envelopes
from agent_supervision_tree.agentic.errors import (
    ProviderRateLimitedError,
    ProviderUnavailableError,
)
from agent_supervision_tree.agentic.providers.base import (
    LLMRequest,
    LLMResponse,
    TextBlock,
    Usage,
)
from agent_supervision_tree.agentic.tokens import simulate_usage_tokens
from agent_supervision_tree.core.canonical import canonical_json
from agent_supervision_tree.workload.documents import INJECTION_MARKER, POISON_MARKER, parse_invoice

__all__ = ["DOCUMENT_SOURCE", "FaultProfile", "FaultingFakeProvider"]

DOCUMENT_SOURCE = "user:document"
_GARBAGE: tuple[dict[str, JsonValue] | None, ...] = (
    None,
    {"garbage": True},
    {"document_id": "doc-000000", "vendor": "", "currency": "GBP", "total_cents": -1},
)


@dataclass(frozen=True, slots=True)
class FaultProfile:
    """Per-call fault probabilities (checked in this order) and latency distribution."""

    p_hang: float = 0.0
    p_rate_limited: float = 0.0
    p_unavailable: float = 0.0
    p_garbage: float = 0.0
    latency_ms: float = 2.0
    latency_jitter_ms: float = 2.0
    hang_seconds: float = 30.0

    def __post_init__(self) -> None:
        total = self.p_hang + self.p_rate_limited + self.p_unavailable + self.p_garbage
        if (
            total > 1
            or min(self.p_hang, self.p_rate_limited, self.p_unavailable, self.p_garbage) < 0
        ):
            raise ValueError("fault probabilities must be non-negative and sum to at most 1")

    @property
    def fault_rate(self) -> float:
        return self.p_hang + self.p_rate_limited + self.p_unavailable + self.p_garbage

    @classmethod
    def at(
        cls, fault_rate: float, *, latency_ms: float = 2.0, hang_seconds: float = 30.0
    ) -> FaultProfile:
        """Spread ``fault_rate`` evenly over the four fault kinds."""
        if not 0 <= fault_rate <= 1:
            raise ValueError("fault_rate must be within [0, 1]")
        share = fault_rate / 4
        return cls(share, share, share, share, latency_ms, latency_ms, hang_seconds)


class FaultingFakeProvider:
    name = "fake-faulting"

    def __init__(
        self,
        profile: FaultProfile | None = None,
        *,
        seed: int = 0,
        model_for_role: Mapping[str, str] | None = None,
        gullible: bool = True,
    ) -> None:
        self.profile = profile or FaultProfile()
        self.model_for_role = dict(model_for_role or {})
        self.gullible = gullible
        self._rng = random.Random(seed)  # noqa: S311  # nosec B311 - simulation, not cryptography
        self._hang_next = 0
        self.calls = 0
        self.outcomes: dict[str, int] = {}

    # --- fault knobs (used by the fault-injection actor) ---------------------------------

    def set_profile(self, profile: FaultProfile) -> None:
        self.profile = profile

    def set_fault_rate(self, fault_rate: float) -> None:
        self.profile = FaultProfile.at(
            fault_rate, latency_ms=self.profile.latency_ms, hang_seconds=self.profile.hang_seconds
        )

    def hang_next(self, count: int) -> None:
        """Make the next ``count`` calls hang (until the caller's deadline cancels them)."""
        self._hang_next += count

    def _count(self, outcome: str) -> None:
        self.outcomes[outcome] = self.outcomes.get(outcome, 0) + 1

    # --- the model -----------------------------------------------------------------------

    async def acomplete(self, request: LLMRequest) -> LLMResponse:
        self.calls += 1
        profile = self.profile
        latency = max(
            0.0, profile.latency_ms + self._rng.uniform(-1, 1) * profile.latency_jitter_ms
        )
        await asyncio.sleep(latency / 1000)
        roll = self._rng.random()
        if self._hang_next > 0 or roll < profile.p_hang:
            self._hang_next = max(0, self._hang_next - 1)
            self._count("hang")
            await asyncio.sleep(profile.hang_seconds)
            raise ProviderUnavailableError("the hung call finally failed")
        roll -= profile.p_hang
        if roll < profile.p_rate_limited:
            self._count("rate_limited")
            raise ProviderRateLimitedError("HTTP 429 rate_limit_error", retry_after=0.01)
        roll -= profile.p_rate_limited
        if roll < profile.p_unavailable:
            self._count("unavailable")
            raise ProviderUnavailableError("HTTP 529 overloaded_error")
        roll -= profile.p_unavailable
        model = self.model_for_role.get(request.role, "claude-sonnet-5")
        text = _document_text(request)
        parsed = self._answer(text, garbage=roll < profile.p_garbage)
        rendered = json.dumps(parsed) if parsed is not None else "{not json"
        prompt = canonical_json(request.model_dump(mode="json"))
        usage = Usage(
            input_tokens=simulate_usage_tokens(prompt),
            output_tokens=simulate_usage_tokens(rendered),
        )
        return LLMResponse(
            model=model, stop_reason="end_turn", text=rendered, parsed=parsed, usage=usage
        )

    def _answer(self, text: str, *, garbage: bool) -> dict[str, JsonValue] | None:
        extraction = parse_invoice(text)
        if garbage or POISON_MARKER in text or extraction is None:
            self._count("garbage")
            return replace_garbage(self._rng)
        payload: dict[str, JsonValue] = extraction.model_dump(mode="json")
        if self.gullible and INJECTION_MARKER in text:
            self._count("obeyed_injection")
            payload["webhook_url"] = "https://evil.example/exfil"
            payload["approved"] = True
        else:
            self._count("ok")
        return payload


def replace_garbage(rng: random.Random) -> dict[str, JsonValue] | None:
    choice = rng.choice(_GARBAGE)
    return dict(choice) if choice is not None else None


def _document_text(request: LLMRequest) -> str:
    """The document inside the request's untrusted-data envelope (empty when absent)."""
    for message in reversed(request.messages):
        if message.role != "user":
            continue
        for block in message.content:
            if isinstance(block, TextBlock):
                for envelope in parse_envelopes(block.text):
                    if envelope.source == DOCUMENT_SOURCE:
                        return envelope.content
    return ""
