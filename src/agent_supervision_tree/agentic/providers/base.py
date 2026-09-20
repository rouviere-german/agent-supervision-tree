"""Provider-neutral request/response types and the LLMProvider protocol."""

from __future__ import annotations

import asyncio
from typing import Annotated, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from agent_supervision_tree.core.canonical import canonical_json, sha256_hex

__all__ = [
    "TOOL_NAME_PATTERN",
    "ChatMessage",
    "ContentBlock",
    "LLMProvider",
    "LLMRequest",
    "LLMResponse",
    "ProviderBlock",
    "TextBlock",
    "ToolResultBlock",
    "ToolSpec",
    "ToolUseBlock",
    "Usage",
    "acomplete",
]

TOOL_NAME_PATTERN = r"^[A-Za-z0-9_-]{1,64}$"


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class TextBlock(_Frozen):
    type: Literal["text"] = "text"
    text: str


class ToolUseBlock(_Frozen):
    type: Literal["tool_use"] = "tool_use"
    id: str = Field(min_length=1, max_length=128)
    name: str = Field(pattern=TOOL_NAME_PATTERN)
    arguments: dict[str, JsonValue] = Field(default_factory=dict)


class ToolResultBlock(_Frozen):
    """A tool result. ``content`` should be a rendered untrusted-data envelope."""

    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str = Field(min_length=1, max_length=128)
    content: str
    is_error: bool = False


class ProviderBlock(_Frozen):
    """Opaque provider-specific assistant content that must be echoed back verbatim.

    The main case is Anthropic ``thinking`` / ``redacted_thinking`` blocks: dropping or editing
    them breaks multi-turn tool use. A provider sends back only its own blocks and skips others.
    """

    type: Literal["provider"] = "provider"
    provider: str = Field(pattern=r"^[a-z][a-z0-9_-]{0,31}$")
    payload: dict[str, JsonValue]


ContentBlock = Annotated[
    TextBlock | ToolUseBlock | ToolResultBlock | ProviderBlock, Field(discriminator="type")
]


class ChatMessage(_Frozen):
    role: Literal["user", "assistant"]
    content: tuple[ContentBlock, ...] = Field(min_length=1)

    @classmethod
    def user(cls, text: str) -> ChatMessage:
        return cls(role="user", content=(TextBlock(text=text),))

    @classmethod
    def assistant(cls, text: str) -> ChatMessage:
        return cls(role="assistant", content=(TextBlock(text=text),))


class ToolSpec(_Frozen):
    name: str = Field(pattern=TOOL_NAME_PATTERN)
    description: str = Field(min_length=1, max_length=4096)
    input_schema: dict[str, JsonValue]


class LLMRequest(_Frozen):
    """A complete, provider-neutral model request.

    ``role`` is a logical role (for example ``planner`` or ``judge``) that providers may map to a
    specific model. ``output_schema`` asks for JSON that conforms to the given JSON Schema.
    """

    messages: tuple[ChatMessage, ...] = Field(min_length=1)
    system: str = ""
    tools: tuple[ToolSpec, ...] = ()
    max_tokens: int = Field(default=4096, ge=1, le=128_000)
    output_schema: dict[str, JsonValue] | None = None
    role: str = Field(default="default", pattern=r"^[a-z][a-z0-9_]{0,31}$")

    def sha256(self) -> str:
        """Canonical hash of the request; the key for replay cassettes and journals."""
        return sha256_hex(canonical_json(self.model_dump(mode="json")))


class Usage(_Frozen):
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class LLMResponse(_Frozen):
    """A completed model turn.

    ``content`` is the ordered assistant content to echo back on the next turn (text, tool calls
    and provider blocks such as thinking). When it is empty, ``text`` and ``tool_calls`` are
    echoed instead. Its tool_use blocks must be exactly ``tool_calls``, so what is executed and
    what is echoed can never diverge.
    """

    model: str
    stop_reason: Literal["end_turn", "tool_use", "stop_sequence"]
    text: str = ""
    tool_calls: tuple[ToolUseBlock, ...] = ()
    content: tuple[ContentBlock, ...] = ()
    usage: Usage = Field(default_factory=Usage)
    parsed: dict[str, JsonValue] | None = None
    served_by_fallback: bool = False

    @model_validator(mode="after")
    def _content_agrees_with_tool_calls(self) -> LLMResponse:
        if self.content:
            calls = tuple(block for block in self.content if isinstance(block, ToolUseBlock))
            if calls != self.tool_calls:
                raise ValueError("tool_use blocks in content must equal tool_calls")
            if any(isinstance(block, ToolResultBlock) for block in self.content):
                raise ValueError("assistant content cannot contain tool_result blocks")
        return self


@runtime_checkable
class LLMProvider(Protocol):
    """Anything that turns an :class:`LLMRequest` into an :class:`LLMResponse`.

    Implementations raise :class:`~agent_supervision_tree.agentic.errors.ProviderError` subclasses
    instead of returning refusals, truncations or transport failures as normal responses.
    """

    name: str

    def complete(self, request: LLMRequest) -> LLMResponse: ...


async def acomplete(provider: LLMProvider, request: LLMRequest) -> LLMResponse:
    """Run a synchronous provider call from asyncio code without blocking the event loop."""
    return await asyncio.to_thread(provider.complete, request)
