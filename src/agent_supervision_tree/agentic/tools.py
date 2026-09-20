"""Least-privilege tool registry with declared capabilities and strict argument schemas."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, JsonValue, ValidationError

from agent_supervision_tree.agentic.errors import ToolArgumentError, ToolNotAllowedError
from agent_supervision_tree.agentic.providers.base import TOOL_NAME_PATTERN, ToolSpec, ToolUseBlock
from agent_supervision_tree.core.canonical import canonical_json, sha256_hex

__all__ = [
    "SIDE_EFFECT_CAPABILITIES",
    "Capability",
    "ToolDefinition",
    "ToolRegistry",
    "ValidatedCall",
    "tool_call_hash",
]


class Capability(StrEnum):
    READ_DATA = "read_data"
    WRITE_DATA = "write_data"
    NETWORK_EGRESS = "network_egress"
    SEND_MESSAGE = "send_message"
    EXECUTE_CODE = "execute_code"


SIDE_EFFECT_CAPABILITIES = frozenset(
    {
        Capability.WRITE_DATA,
        Capability.NETWORK_EGRESS,
        Capability.SEND_MESSAGE,
        Capability.EXECUTE_CODE,
    }
)

_NAME = re.compile(TOOL_NAME_PATTERN)


def tool_call_hash(tool_name: str, canonical_args: Mapping[str, JsonValue]) -> str:
    """SHA-256 over the canonical JSON of ``{"v": 1, "tool": name, "args": args}``."""
    return sha256_hex(canonical_json({"v": 1, "tool": tool_name, "args": dict(canonical_args)}))


@dataclass(frozen=True, slots=True)
class ToolDefinition[ArgsT: BaseModel]:
    """A tool: name, argument model, declared capabilities and handler.

    ``requires_approval`` defaults to True whenever any side-effect capability is declared. The
    argument model must forbid extra fields, so the schema sent to the model is closed. Its JSON
    dump must also round-trip (no ``exclude=True`` fields, ``SecretStr`` or lossy serializers):
    the dump is what approvals hash, so it has to be exactly what the handler receives.
    """

    name: str
    description: str
    args_model: type[ArgsT]
    capabilities: frozenset[Capability]
    handler: Callable[[ArgsT], JsonValue]
    requires_approval: bool | None = None

    def __post_init__(self) -> None:
        if not _NAME.fullmatch(self.name):
            raise ValueError(f"invalid tool name {self.name!r}")
        if not self.capabilities:
            raise ValueError(f"tool {self.name} must declare at least one capability")
        if self.args_model.model_config.get("extra") != "forbid":
            raise ValueError(f"tool {self.name} args model must set extra='forbid'")
        if self.requires_approval is False and self.capabilities & SIDE_EFFECT_CAPABILITIES:
            raise ValueError(f"tool {self.name} has side effects and cannot opt out of approval")

    @property
    def needs_approval(self) -> bool:
        if self.requires_approval is not None:
            return self.requires_approval
        return bool(self.capabilities & SIDE_EFFECT_CAPABILITIES)

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            description=self.description,
            input_schema=self.args_model.model_json_schema(),
        )


@dataclass(frozen=True, slots=True)
class ValidatedCall:
    """A resolved, strictly validated call whose ``canonical_args`` are exactly what runs.

    ``ToolRegistry.validate_call`` proves that ``args`` is reconstructed from ``canonical_args``,
    so :attr:`call_hash` stands for the call in approvals and audit records. Mutating either
    after validation is detected before authorization and before the handler runs.
    """

    tool: ToolDefinition[Any]
    args: BaseModel
    canonical_args: dict[str, JsonValue]

    @property
    def call_hash(self) -> str:
        return tool_call_hash(self.tool.name, self.canonical_args)

    def ensure_unchanged(self) -> None:
        """Raise ToolArgumentError if ``args`` no longer matches ``canonical_args``."""
        current = canonical_json(self.args.model_dump(mode="json"))
        if current != canonical_json(self.canonical_args):
            raise ToolArgumentError(f"arguments for {self.tool.name} changed after validation")

    def run(self, *, authorized_hash: str | None = None) -> JsonValue:
        """Run the handler; with ``authorized_hash``, only if the call still hashes to it."""
        self.ensure_unchanged()
        if authorized_hash is not None and self.call_hash != authorized_hash:
            raise ToolArgumentError(f"arguments for {self.tool.name} changed after authorization")
        return self.tool.handler(self.args)


def _round_trip_error(tool_name: str) -> ToolArgumentError:
    return ToolArgumentError(
        f"arguments for {tool_name} do not round-trip through their canonical form; the args "
        "model must not exclude, mask or transform fields when dumped"
    )


class ToolRegistry:
    """Tools available to one agent. Registration fails if a tool needs ungranted capabilities."""

    def __init__(self, granted: Iterable[Capability]) -> None:
        self._granted = frozenset(granted)
        self._tools: dict[str, ToolDefinition[Any]] = {}

    @property
    def granted(self) -> frozenset[Capability]:
        return self._granted

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._tools))

    def register(self, tool: ToolDefinition[Any]) -> None:
        if tool.name in self._tools:
            raise ValueError(f"tool {tool.name} is already registered")
        missing = tool.capabilities - self._granted
        if missing:
            needed = ", ".join(sorted(missing))
            raise ToolNotAllowedError(f"tool {tool.name} needs capabilities not granted: {needed}")
        self._tools[tool.name] = tool

    def scoped(self, allowed: Iterable[str]) -> ToolRegistry:
        """A narrower registry holding only the ``allowed`` tools (per-workflow allowlist)."""
        names = set(allowed)
        unknown = names - self._tools.keys()
        if unknown:
            raise ValueError(f"unknown tools in allowlist: {', '.join(sorted(unknown))}")
        tools = [self._tools[name] for name in sorted(names)]
        granted: set[Capability] = set()
        for tool in tools:
            granted |= tool.capabilities
        child = ToolRegistry(granted)
        for tool in tools:
            child.register(tool)
        return child

    def specs(self) -> tuple[ToolSpec, ...]:
        """Tool specs in a stable order, so prompts (and their hashes) are deterministic."""
        return tuple(self._tools[name].spec() for name in self.names)

    def get(self, name: str) -> ToolDefinition[Any]:
        tool = self._tools.get(name)
        if tool is None:
            raise ToolNotAllowedError(f"tool {name} is not available to this agent")
        return tool

    def validate_call(self, call: ToolUseBlock) -> ValidatedCall:
        """Resolve and strictly validate a model-proposed call. Nothing is executed here.

        The validated arguments are dumped to canonical JSON and validated again from that dump;
        the handler receives the second object, and both must agree. An args model whose dump
        drops or masks data (``exclude=True``, ``SecretStr``, lossy serializers) is refused,
        because otherwise arguments hidden from the hash could be changed after approval.
        """
        tool = self.get(call.name)
        try:
            args = tool.args_model.model_validate_json(canonical_json(call.arguments), strict=True)
        except ValidationError as exc:
            locations = sorted(
                {".".join(str(p) for p in err["loc"]) or "$" for err in exc.errors()}
            )
            raise ToolArgumentError(
                f"invalid arguments for {tool.name} at: {', '.join(locations)}"
            ) from exc
        canonical: dict[str, JsonValue] = args.model_dump(mode="json")
        try:
            bound = tool.args_model.model_validate_json(canonical_json(canonical), strict=True)
        except ValidationError as exc:
            raise _round_trip_error(tool.name) from exc
        if bound != args or canonical_json(bound.model_dump(mode="json")) != canonical_json(
            canonical
        ):
            raise _round_trip_error(tool.name)
        return ValidatedCall(tool=tool, args=bound, canonical_args=canonical)
