"""AnthropicProvider: the live provider, active only when ANTHROPIC_API_KEY is configured.

Request shape (see the Anthropic API documentation at docs.anthropic.com):
- ``client.beta.messages.create``, or ``client.beta.messages.stream`` when ``max_tokens`` exceeds
  16K (non-streaming requests that large risk SDK HTTP timeouts). The client timeout defaults to
  the SDK's 10 minutes;
- adaptive thinking and ``output_config.effort`` for the models the reference documents them for
  (Claude Opus 5, Sonnet 5, Fable 5.x and the 4.6+ Opus/Sonnet models). Other models, such as
  claude-haiku-4-5, get neither and run on API defaults rather than failing with a 400;
- JSON output through ``output_config.format`` (JSON Schema) and ``strict: true`` tools. Both
  schemas go through the SDK's ``transform_schema`` (the conversion ``messages.parse`` applies):
  keywords structured outputs reject, such as ``minLength`` or ``minimum``, move into the
  description and every object gets ``additionalProperties: false``. The tool registry still
  validates arguments against the full pydantic model;
- the server-side refusal fallback: ``fallbacks="default"`` with the
  ``server-side-fallback-2026-07-01`` beta header, which re-runs a policy-declined request on the
  recommended fallback model inside the same call. It is sent only for requested models the
  reference documents it for (Claude Opus 5, Fable 5.x); extend ``_REFUSAL_FALLBACK_MODELS`` after
  checking a model's ``allowed_fallback_models`` on ``/v1/models``.

Responses are classified by ``stop_reason`` *before* content is read: ``refusal`` and
``max_tokens`` raise instead of returning partial output, and those errors carry the billed usage
so budgets are still charged. So does a response whose content fails validation. Cache writes and
reads count as input tokens (charged at the full input price, which errs on the safe side).
Thinking blocks are kept (as :class:`ProviderBlock`) and echoed back unchanged on the next turn, as
multi-turn tool use requires. After a fallback, thinking and
tool_use blocks that precede the last ``fallback`` marker belong to the declined attempt and are
neither executed nor echoed. SDK exceptions are mapped most-specific-first onto the
provider-neutral error hierarchy. The SDK is imported lazily so the offline code paths
(FakeProvider, ReplayProvider) need neither the package nor a key.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal

from pydantic import JsonValue, SecretStr, ValidationError

from agent_supervision_tree.agentic.errors import (
    ProviderAuthError,
    ProviderError,
    ProviderRateLimitedError,
    ProviderRefusalError,
    ProviderRequestError,
    ProviderTruncatedError,
    ProviderUnavailableError,
)
from agent_supervision_tree.agentic.providers.base import (
    ContentBlock,
    LLMRequest,
    LLMResponse,
    ProviderBlock,
    TextBlock,
    ToolUseBlock,
    Usage,
)

if TYPE_CHECKING:
    from agent_supervision_tree.core.config import Settings

__all__ = [
    "DEFAULT_MODEL",
    "MAX_NON_STREAMING_TOKENS",
    "REFUSAL_FALLBACK_BETA",
    "AnthropicProvider",
    "Effort",
]

DEFAULT_MODEL = "claude-opus-5"
REFUSAL_FALLBACK_BETA = "server-side-fallback-2026-07-01"
MAX_NON_STREAMING_TOKENS = 16_000
Effort = Literal["low", "medium", "high", "xhigh", "max"]
_PROVIDER = "anthropic"
_OK_STOP_REASONS = frozenset({"end_turn", "tool_use", "stop_sequence"})
_INPUT_TOKEN_FIELDS = ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")
# Model families, matched by ID or ID prefix followed by "-" (so dated IDs match too).
_ADAPTIVE_THINKING_MODELS = (
    "claude-opus-5",
    "claude-opus-4-8",
    "claude-opus-4-7",
    "claude-opus-4-6",
    "claude-sonnet-5",
    "claude-sonnet-4-6",
    "claude-fable-5",
    "claude-mythos-5",
)
_REFUSAL_FALLBACK_MODELS = ("claude-opus-5", "claude-fable-5")


def _in_family(model: str, families: tuple[str, ...]) -> bool:
    return any(model == family or model.startswith(f"{family}-") for family in families)


def _block_to_wire(block: ContentBlock) -> dict[str, Any] | None:
    if isinstance(block, TextBlock):
        return {"type": "text", "text": block.text}
    if isinstance(block, ToolUseBlock):
        return {"type": "tool_use", "id": block.id, "name": block.name, "input": block.arguments}
    if isinstance(block, ProviderBlock):
        # Our own blocks (thinking) go back exactly as received; other providers' blocks are
        # meaningless to this API and are skipped.
        return dict(block.payload) if block.provider == _PROVIDER else None
    return {
        "type": "tool_result",
        "tool_use_id": block.tool_use_id,
        "content": block.content,
        "is_error": block.is_error,
    }


def _strict_schema(schema: Mapping[str, JsonValue], *, context: str) -> dict[str, Any]:
    """Rewrite a JSON Schema into the subset strict tools and structured outputs accept."""
    from anthropic import transform_schema  # noqa: PLC0415 - lazy, like the client itself

    try:
        return transform_schema(copy.deepcopy(dict(schema)))
    except (ValueError, TypeError, AttributeError) as exc:
        message = f"{context} cannot be expressed as a strict JSON schema"
        raise ProviderRequestError(message) from exc


def _status_message(exc: Any) -> str:
    status = getattr(exc, "status_code", "unknown")
    error_type = getattr(exc, "type", None) or "unknown"
    request_id = getattr(exc, "request_id", None) or "n/a"
    return f"Anthropic API returned HTTP {status} ({error_type}); request_id={request_id}"


def _retry_after(exc: Any) -> float | None:
    response = getattr(exc, "response", None)
    raw = response.headers.get("retry-after") if response is not None else None
    try:
        return float(raw) if raw is not None else None
    except ValueError:
        return None


def _usage_of(usage_obj: Any) -> tuple[Usage | None, bool]:
    """Billed usage and whether a fallback model served the message.

    ``usage.iterations`` is the per-attempt source of truth when present; every attempt is summed,
    declined ones included, which errs on the safe side for budgets. ``input_tokens`` excludes
    prompt-cache writes and reads, so those are added to the input count.
    """
    if usage_obj is None:
        return None, False

    def input_tokens(item: Any) -> int:
        return sum(int(getattr(item, name, 0) or 0) for name in _INPUT_TOKEN_FIELDS)

    iterations = list(getattr(usage_obj, "iterations", None) or [])
    attempts = iterations or [usage_obj]
    usage = Usage(
        input_tokens=sum(input_tokens(item) for item in attempts),
        output_tokens=sum(int(getattr(item, "output_tokens", 0) or 0) for item in attempts),
    )
    served_by_fallback = any(getattr(it, "type", None) == "fallback_message" for it in iterations)
    return usage, served_by_fallback


def _read_content(blocks: list[Any]) -> tuple[str, list[ToolUseBlock], list[ContentBlock]]:
    """Split response blocks into answer text, tool calls and the ordered blocks to echo back.

    Blocks other than text that precede the last ``fallback`` marker belong to a declined attempt
    and are dropped; ``fallback`` markers and unknown block types are not echoed.
    """
    boundary = max(
        (index for index, block in enumerate(blocks) if getattr(block, "type", None) == "fallback"),
        default=-1,
    )
    texts: list[str] = []
    calls: list[ToolUseBlock] = []
    echo: list[ContentBlock] = []
    for index, block in enumerate(blocks):
        block_type = getattr(block, "type", None)
        if index < boundary and block_type != "text":
            continue
        if block_type == "text":
            texts.append(block.text)
            if block.text:
                echo.append(TextBlock(text=block.text))
        elif block_type == "tool_use":
            if not isinstance(block.input, Mapping):
                raise TypeError("tool_use input is not a JSON object")
            call = ToolUseBlock(id=block.id, name=block.name, arguments=dict(block.input))
            calls.append(call)
            echo.append(call)
        elif block_type == "thinking":
            payload: dict[str, JsonValue] = {
                "type": "thinking",
                "thinking": block.thinking,
                "signature": block.signature,
            }
            echo.append(ProviderBlock(provider=_PROVIDER, payload=payload))
        elif block_type == "redacted_thinking":
            echo.append(
                ProviderBlock(
                    provider=_PROVIDER, payload={"type": "redacted_thinking", "data": block.data}
                )
            )
    return "".join(texts), calls, echo


class AnthropicProvider:
    name = _PROVIDER

    def __init__(
        self,
        *,
        api_key: SecretStr | None,
        model: str = DEFAULT_MODEL,
        role_models: Mapping[str, str] | None = None,
        effort: Effort = "high",
        refusal_fallback: bool = True,
        max_retries: int = 2,
        timeout_seconds: float = 600.0,
        client: Any | None = None,
    ) -> None:
        self._api_key = api_key
        self.model = model
        self._role_models = dict(role_models or {})
        self.effort: Effort = effort
        self.refusal_fallback = refusal_fallback
        self._max_retries = max_retries
        self._timeout_seconds = timeout_seconds
        self._client = client

    @classmethod
    def from_settings(
        cls, settings: Settings, *, role_models: Mapping[str, str] | None = None
    ) -> AnthropicProvider:
        return cls(
            api_key=settings.anthropic_api_key,
            model=settings.llm_model,
            role_models=role_models,
            effort=settings.llm_effort,
            refusal_fallback=settings.llm_refusal_fallback,
            max_retries=settings.llm_max_retries,
            timeout_seconds=settings.llm_timeout_seconds,
        )

    def model_for(self, role: str) -> str:
        return self._role_models.get(role, self.model)

    def build_params(self, request: LLMRequest) -> dict[str, Any]:
        """Translate a provider-neutral request into ``beta.messages`` keyword arguments."""
        model = self.model_for(request.role)
        adaptive = _in_family(model, _ADAPTIVE_THINKING_MODELS)
        output_config: dict[str, Any] = {"effort": self.effort} if adaptive else {}
        if request.output_schema is not None:
            output_config["format"] = {
                "type": "json_schema",
                "schema": _strict_schema(request.output_schema, context="output_schema"),
            }
        messages: list[dict[str, Any]] = []
        for message in request.messages:
            wire = [item for item in map(_block_to_wire, message.content) if item is not None]
            messages.append({"role": message.role, "content": wire})
        params: dict[str, Any] = {
            "model": model,
            "max_tokens": request.max_tokens,
            "messages": messages,
        }
        if adaptive:
            params["thinking"] = {"type": "adaptive"}
        if output_config:
            params["output_config"] = output_config
        if request.system:
            params["system"] = request.system
        if request.tools:
            params["tools"] = [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": _strict_schema(
                        tool.input_schema, context=f"tool {tool.name} input_schema"
                    ),
                    "strict": True,
                }
                for tool in request.tools
            ]
        if self.refusal_fallback and _in_family(model, _REFUSAL_FALLBACK_MODELS):
            params["betas"] = [REFUSAL_FALLBACK_BETA]
            params["fallbacks"] = "default"
        return params

    def _get_client(self) -> Any:
        if self._client is None:
            if self._api_key is None or not self._api_key.get_secret_value():
                raise ProviderAuthError(
                    "ANTHROPIC_API_KEY is not set; use FakeProvider or ReplayProvider offline"
                )
            import anthropic  # noqa: PLC0415 - lazy import keeps offline paths SDK-free

            self._client = anthropic.Anthropic(
                api_key=self._api_key.get_secret_value(),
                max_retries=self._max_retries,
                timeout=self._timeout_seconds,
            )
        return self._client

    def complete(self, request: LLMRequest) -> LLMResponse:
        client = self._get_client()
        import anthropic  # noqa: PLC0415 - exception classes; the SDK is present once a client exists

        params = self.build_params(request)
        try:
            if request.max_tokens > MAX_NON_STREAMING_TOKENS:
                with client.beta.messages.stream(**params) as stream:
                    message = stream.get_final_message()
            else:
                message = client.beta.messages.create(**params)
        except anthropic.RateLimitError as exc:
            raise ProviderRateLimitedError(
                _status_message(exc), retry_after=_retry_after(exc)
            ) from exc
        except (anthropic.AuthenticationError, anthropic.PermissionDeniedError) as exc:
            raise ProviderAuthError(_status_message(exc)) from exc
        except (
            anthropic.BadRequestError,
            anthropic.NotFoundError,
            anthropic.RequestTooLargeError,
            anthropic.UnprocessableEntityError,
        ) as exc:
            raise ProviderRequestError(_status_message(exc)) from exc
        except (
            anthropic.OverloadedError,
            anthropic.ServiceUnavailableError,
            anthropic.InternalServerError,
        ) as exc:
            raise ProviderUnavailableError(_status_message(exc)) from exc
        except anthropic.APIStatusError as exc:
            retryable = exc.status_code in (408, 409) or exc.status_code >= 500
            raise ProviderError(_status_message(exc), retryable=retryable) from exc
        except anthropic.APITimeoutError as exc:
            raise ProviderUnavailableError("Anthropic API request timed out") from exc
        except anthropic.APIConnectionError as exc:
            raise ProviderUnavailableError("could not connect to the Anthropic API") from exc
        return self.parse_message(message, request)

    def parse_message(self, message: Any, request: LLMRequest) -> LLMResponse:
        """Classify ``stop_reason`` first, then read content blocks.

        Usage (metadata, not content) is read up front so billed failures carry it.
        """
        stop_reason = getattr(message, "stop_reason", None)
        model = str(getattr(message, "model", None) or self.model_for(request.role))
        usage, served_by_fallback = _usage_of(getattr(message, "usage", None))
        if stop_reason == "refusal":
            details = getattr(message, "stop_details", None)
            raise ProviderRefusalError(
                category=getattr(details, "category", None),
                explanation=getattr(details, "explanation", None),
                model=model,
                usage=usage,
            )
        if stop_reason == "max_tokens":
            raise ProviderTruncatedError(
                f"output reached max_tokens={request.max_tokens}; partial output discarded",
                model=model,
                usage=usage,
            )
        if stop_reason == "model_context_window_exceeded":
            raise ProviderRequestError(
                "request exceeded the model context window", model=model, usage=usage
            )
        if stop_reason not in _OK_STOP_REASONS:
            # pause_turn only occurs with server tools, which this provider never sends.
            raise ProviderError(f"unexpected stop_reason: {stop_reason}", model=model, usage=usage)

        try:
            text, calls, echo = _read_content(list(message.content))
        except (ValidationError, TypeError, ValueError, AttributeError) as exc:
            # The call was billed even though its content is unusable.
            raise ProviderError("malformed response content", model=model, usage=usage) from exc
        parsed: dict[str, JsonValue] | None = None
        if request.output_schema is not None and stop_reason != "tool_use":
            try:
                decoded = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ProviderError(
                    "structured output was not valid JSON", model=model, usage=usage
                ) from exc
            if not isinstance(decoded, dict):
                raise ProviderError(
                    "structured output was not a JSON object", model=model, usage=usage
                )
            parsed = decoded
        return LLMResponse(
            model=model,
            stop_reason=stop_reason,
            text=text,
            tool_calls=tuple(calls),
            content=tuple(echo),
            usage=usage or Usage(),
            parsed=parsed,
            served_by_fallback=served_by_fallback,
        )
