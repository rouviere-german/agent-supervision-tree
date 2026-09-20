"""FakeProvider: deterministic, seeded, scripted policies for offline tests and demos."""

from __future__ import annotations

import json
import math
import random
import re
from collections.abc import Callable, Sequence

from pydantic import JsonValue

from agent_supervision_tree.agentic.errors import ProviderError
from agent_supervision_tree.agentic.providers.base import (
    LLMRequest,
    LLMResponse,
    ToolResultBlock,
    ToolUseBlock,
    Usage,
)
from agent_supervision_tree.core.canonical import canonical_json

__all__ = [
    "FakeProvider",
    "GullibleInjectionPolicy",
    "Policy",
    "PolicyResult",
    "ScriptedPolicy",
    "SeededChoicePolicy",
    "estimate_tokens",
]

PolicyResult = LLMResponse | ProviderError
Policy = Callable[[LLMRequest, random.Random, int], PolicyResult]

_INJECTED_CALL = re.compile(r"CALL_TOOL\s+([A-Za-z0-9_-]{1,64})\s+(\{.*\})")


def estimate_tokens(text: str) -> int:
    """Rough, deterministic token estimate (4 characters per token, at least 1)."""
    return max(1, math.ceil(len(text) / 4))


class ScriptedPolicy:
    """Return the scripted steps in order; a ProviderError step is raised by the provider."""

    def __init__(self, steps: Sequence[PolicyResult]) -> None:
        self._steps = tuple(steps)

    def __call__(self, request: LLMRequest, rng: random.Random, turn: int) -> PolicyResult:
        del request, rng
        if turn >= len(self._steps):
            return ProviderError(f"scripted policy exhausted after {len(self._steps)} turns")
        return self._steps[turn]


class SeededChoicePolicy:
    """Pick one of ``choices`` per turn using the provider's seeded RNG."""

    def __init__(self, choices: Sequence[LLMResponse]) -> None:
        if not choices:
            raise ValueError("choices must not be empty")
        self._choices = tuple(choices)

    def __call__(self, request: LLMRequest, rng: random.Random, turn: int) -> PolicyResult:
        del request, turn
        return rng.choice(self._choices)


class GullibleInjectionPolicy:
    """A deliberately gullible stand-in for a model that obeys prompt injections.

    It optionally makes ``first_call`` on turn 0, then obeys any ``CALL_TOOL <name> <json>`` line it
    finds in the latest tool results. Tests use it to show that containment comes from the tool
    registry and approval gate, not from the model resisting the injection.
    """

    def __init__(self, first_call: ToolUseBlock | None = None, final_text: str = "Done.") -> None:
        self._first_call = first_call
        self._final_text = final_text

    def __call__(self, request: LLMRequest, rng: random.Random, turn: int) -> PolicyResult:
        del rng
        if turn == 0 and self._first_call is not None:
            return LLMResponse(
                model="fake-gullible", stop_reason="tool_use", tool_calls=(self._first_call,)
            )
        for block in reversed(request.messages[-1].content):
            if not isinstance(block, ToolResultBlock):
                continue
            match = _INJECTED_CALL.search(block.content)
            if match is None:
                continue
            try:
                arguments: JsonValue = json.loads(match.group(2))
            except json.JSONDecodeError:
                continue
            if isinstance(arguments, dict):
                call = ToolUseBlock(id=f"injected_{turn}", name=match.group(1), arguments=arguments)
                return LLMResponse(
                    model="fake-gullible", stop_reason="tool_use", tool_calls=(call,)
                )
        return LLMResponse(model="fake-gullible", stop_reason="end_turn", text=self._final_text)


class FakeProvider:
    """Offline provider driven by a policy and a seeded RNG. Never touches the network."""

    name = "fake"

    def __init__(self, policy: Policy, *, seed: int = 0) -> None:
        self._policy = policy
        self._rng = random.Random(seed)  # noqa: S311  # nosec B311 - simulation, not cryptography
        self._turn = 0
        self.request_hashes: list[str] = []

    @property
    def calls(self) -> int:
        return len(self.request_hashes)

    def complete(self, request: LLMRequest) -> LLMResponse:
        result = self._policy(request, self._rng, self._turn)
        self._turn += 1
        self.request_hashes.append(request.sha256())
        if isinstance(result, ProviderError):
            raise result
        if result.usage.total_tokens == 0:
            prompt = canonical_json(request.model_dump(mode="json"))
            output = result.text + "".join(
                canonical_json(call.arguments) for call in result.tool_calls
            )
            usage = Usage(
                input_tokens=estimate_tokens(prompt), output_tokens=estimate_tokens(output)
            )
            result = result.model_copy(update={"usage": usage})
        return result
