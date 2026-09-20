"""AnthropicProvider request construction and response handling against a mocked client."""

from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import anthropic
import httpx2
import pytest
from pydantic import SecretStr

from agent_supervision_tree.agentic.errors import (
    ProviderAuthError,
    ProviderError,
    ProviderRateLimitedError,
    ProviderRefusalError,
    ProviderRequestError,
    ProviderTruncatedError,
    ProviderUnavailableError,
)
from agent_supervision_tree.agentic.providers.anthropic import (
    DEFAULT_MODEL,
    REFUSAL_FALLBACK_BETA,
    AnthropicProvider,
)
from agent_supervision_tree.agentic.providers.base import (
    ChatMessage,
    LLMRequest,
    ProviderBlock,
    TextBlock,
    ToolResultBlock,
    ToolSpec,
    ToolUseBlock,
    Usage,
)
from agent_supervision_tree.core.config import Settings

pytestmark = pytest.mark.unit

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}


def full_request(**overrides: Any) -> LLMRequest:
    values: dict[str, Any] = {
        "system": "You are a test.",
        "messages": (
            ChatMessage.user("Find the file."),
            ChatMessage(
                role="assistant",
                content=(
                    TextBlock(text="Looking."),
                    ToolUseBlock(id="toolu_1", name="read_file", arguments={"path": "a.txt"}),
                ),
            ),
            ChatMessage(
                role="user", content=(ToolResultBlock(tool_use_id="toolu_1", content="data"),)
            ),
        ),
        "tools": (
            ToolSpec(
                name="read_file",
                description="Read a file.",
                input_schema={"type": "object", "properties": {"path": {"type": "string"}}},
            ),
        ),
        "max_tokens": 2048,
    }
    values.update(overrides)
    return LLMRequest(**values)


def message(
    stop_reason: str = "end_turn",
    content: list[Any] | None = None,
    usage: Any = None,
    model: str = DEFAULT_MODEL,
) -> SimpleNamespace:
    return SimpleNamespace(
        stop_reason=stop_reason,
        stop_details=None,
        content=content if content is not None else [SimpleNamespace(type="text", text="hi")],
        usage=usage or SimpleNamespace(input_tokens=10, output_tokens=5, iterations=None),
        model=model,
    )


def provider_with(client: Any, **kwargs: Any) -> AnthropicProvider:
    return AnthropicProvider(api_key=SecretStr("test-key-not-real"), client=client, **kwargs)


def test_request_uses_adaptive_thinking_strict_tools_and_refusal_fallback() -> None:
    params = provider_with(MagicMock()).build_params(full_request())
    assert params["model"] == "claude-opus-5"
    assert params["max_tokens"] == 2048
    assert params["thinking"] == {"type": "adaptive"}
    assert params["output_config"] == {"effort": "high"}
    assert params["betas"] == [REFUSAL_FALLBACK_BETA] == ["server-side-fallback-2026-07-01"]
    assert params["fallbacks"] == "default"
    assert params["system"] == "You are a test."
    assert params["tools"][0]["strict"] is True
    assert params["messages"][1]["content"] == [
        {"type": "text", "text": "Looking."},
        {"type": "tool_use", "id": "toolu_1", "name": "read_file", "input": {"path": "a.txt"}},
    ]
    assert params["messages"][2]["content"] == [
        {"type": "tool_result", "tool_use_id": "toolu_1", "content": "data", "is_error": False}
    ]
    for removed in ("temperature", "top_p", "top_k", "budget_tokens"):
        assert removed not in params


def test_structured_output_roles_and_fallback_opt_out() -> None:
    provider = provider_with(
        MagicMock(), role_models={"judge": "claude-sonnet-5"}, refusal_fallback=False, effort="low"
    )
    params = provider.build_params(
        LLMRequest(messages=(ChatMessage.user("rate it"),), output_schema=SCHEMA, role="judge")
    )
    assert params["model"] == "claude-sonnet-5"
    assert params["thinking"] == {"type": "adaptive"}
    assert params["output_config"] == {
        "effort": "low",
        "format": {"type": "json_schema", "schema": SCHEMA},
    }
    assert "betas" not in params
    assert "fallbacks" not in params
    assert "system" not in params
    assert "tools" not in params


def test_from_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    monkeypatch.setenv("APP_LLM_MODEL", "claude-sonnet-5")
    monkeypatch.setenv("APP_LLM_REFUSAL_FALLBACK", "false")
    provider = AnthropicProvider.from_settings(Settings(_env_file=None))
    assert provider.model == "claude-sonnet-5"
    assert provider.refusal_fallback is False


def test_missing_key_fails_before_any_client_is_built() -> None:
    for key in (None, SecretStr("")):
        with pytest.raises(ProviderAuthError, match="ANTHROPIC_API_KEY"):
            AnthropicProvider(api_key=key).complete(full_request())


def test_sdk_client_is_built_lazily_with_retry_settings() -> None:
    provider = AnthropicProvider(api_key=SecretStr("test-key-not-real"), max_retries=4)
    client = provider._get_client()
    assert isinstance(client, anthropic.Anthropic)
    assert client.max_retries == 4
    assert provider._get_client() is client


def test_text_response_passes_through_without_thinking_text() -> None:
    client = MagicMock()
    client.beta.messages.create.return_value = message(
        content=[
            SimpleNamespace(type="thinking", thinking="", signature="sig-1"),
            SimpleNamespace(type="text", text="hel"),
            SimpleNamespace(type="text", text="lo"),
        ]
    )
    provider = provider_with(client)
    response = provider.complete(full_request())
    assert response.text == "hello"
    assert response.stop_reason == "end_turn"
    assert (response.usage.input_tokens, response.usage.output_tokens) == (10, 5)
    assert client.beta.messages.create.call_args.kwargs == provider.build_params(full_request())


def test_tool_use_response() -> None:
    client = MagicMock()
    client.beta.messages.create.return_value = message(
        stop_reason="tool_use",
        content=[
            SimpleNamespace(type="tool_use", id="toolu_9", name="read_file", input={"path": "b"})
        ],
    )
    response = provider_with(client).complete(full_request(output_schema=SCHEMA))
    assert response.tool_calls == (
        ToolUseBlock(id="toolu_9", name="read_file", arguments={"path": "b"}),
    )
    assert response.parsed is None


class _RefusedMessage:
    stop_reason = "refusal"
    stop_details = SimpleNamespace(category="cyber", explanation="declined")
    model = DEFAULT_MODEL

    @property
    def content(self) -> list[Any]:
        raise AssertionError("content must not be read before stop_reason is checked")


def test_refusal_is_raised_before_content_is_read() -> None:
    client = MagicMock()
    client.beta.messages.create.return_value = _RefusedMessage()
    with pytest.raises(ProviderRefusalError) as info:
        provider_with(client).complete(full_request())
    assert info.value.category == "cyber"
    assert info.value.explanation == "declined"
    assert info.value.retryable is False


@pytest.mark.parametrize(
    ("stop_reason", "error"),
    [
        ("max_tokens", ProviderTruncatedError),
        ("model_context_window_exceeded", ProviderRequestError),
        ("pause_turn", ProviderError),
        ("compaction", ProviderError),
    ],
)
def test_non_final_stop_reasons_raise(stop_reason: str, error: type[Exception]) -> None:
    client = MagicMock()
    client.beta.messages.create.return_value = message(stop_reason=stop_reason)
    with pytest.raises(error):
        provider_with(client).complete(full_request())


def test_fallback_iterations_are_summed_and_reported() -> None:
    usage = SimpleNamespace(
        input_tokens=7,
        output_tokens=3,
        iterations=[
            SimpleNamespace(type="message", input_tokens=100, output_tokens=0),
            SimpleNamespace(type="fallback_message", input_tokens=100, output_tokens=40),
        ],
    )
    client = MagicMock()
    client.beta.messages.create.return_value = message(usage=usage, model="claude-opus-4-8")
    response = provider_with(client).complete(full_request())
    assert response.served_by_fallback is True
    assert response.model == "claude-opus-4-8"
    assert (response.usage.input_tokens, response.usage.output_tokens) == (200, 40)


def test_structured_output_is_parsed_and_validated_as_json_object() -> None:
    client = MagicMock()
    client.beta.messages.create.return_value = message(
        content=[SimpleNamespace(type="text", text='{"answer": "42"}')]
    )
    assert provider_with(client).complete(full_request(output_schema=SCHEMA)).parsed == {
        "answer": "42"
    }
    for bad in ("not json", "[1, 2]"):
        client.beta.messages.create.return_value = message(
            content=[SimpleNamespace(type="text", text=bad)]
        )
        with pytest.raises(ProviderError, match="structured output"):
            provider_with(client).complete(full_request(output_schema=SCHEMA))


def test_thinking_blocks_are_kept_and_echoed_back_unchanged() -> None:
    client = MagicMock()
    client.beta.messages.create.return_value = message(
        stop_reason="tool_use",
        content=[
            SimpleNamespace(type="thinking", thinking="", signature="sig-abc"),
            SimpleNamespace(type="redacted_thinking", data="opaque-data"),
            SimpleNamespace(type="text", text="Reading."),
            SimpleNamespace(type="tool_use", id="toolu_2", name="read_file", input={"path": "b"}),
        ],
    )
    provider = provider_with(client)
    response = provider.complete(full_request())
    followup = full_request(
        messages=(
            *full_request().messages,
            ChatMessage(role="assistant", content=response.content),
            ChatMessage(
                role="user", content=(ToolResultBlock(tool_use_id="toolu_2", content="x"),)
            ),
        )
    )
    assert provider.build_params(followup)["messages"][3]["content"] == [
        {"type": "thinking", "thinking": "", "signature": "sig-abc"},
        {"type": "redacted_thinking", "data": "opaque-data"},
        {"type": "text", "text": "Reading."},
        {"type": "tool_use", "id": "toolu_2", "name": "read_file", "input": {"path": "b"}},
    ]


def test_blocks_of_a_declined_attempt_before_the_fallback_marker_are_dropped() -> None:
    client = MagicMock()
    client.beta.messages.create.return_value = message(
        stop_reason="tool_use",
        model="claude-opus-4-8",
        content=[
            SimpleNamespace(type="thinking", thinking="", signature="declined-sig"),
            SimpleNamespace(type="text", text="Partial. "),
            SimpleNamespace(type="tool_use", id="toolu_no", name="read_file", input={"path": "x"}),
            SimpleNamespace(
                type="fallback",
                from_=SimpleNamespace(model="claude-opus-5"),
                to=SimpleNamespace(model="claude-opus-4-8"),
            ),
            SimpleNamespace(type="thinking", thinking="", signature="fallback-sig"),
            SimpleNamespace(type="tool_use", id="toolu_ok", name="read_file", input={"path": "y"}),
        ],
    )
    response = provider_with(client).complete(full_request())
    kept_call = ToolUseBlock(id="toolu_ok", name="read_file", arguments={"path": "y"})
    assert response.tool_calls == (kept_call,)
    assert response.content == (
        TextBlock(text="Partial. "),
        ProviderBlock(
            provider="anthropic",
            payload={"type": "thinking", "thinking": "", "signature": "fallback-sig"},
        ),
        kept_call,
    )


def _schema_keys(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            found.add(str(key))
            found |= _schema_keys(item)
    elif isinstance(value, list):
        for item in value:
            found |= _schema_keys(item)
    return found


CONSTRAINED: dict[str, Any] = {
    "type": "object",
    "properties": {
        "to": {"type": "string", "minLength": 3, "maxLength": 254},
        "count": {"type": "integer", "minimum": 1, "maximum": 5},
        "meta": {"type": "object", "properties": {"note": {"type": "string"}}},
    },
    "required": ["to"],
}


def test_strict_schemas_drop_constraints_the_api_rejects_and_close_every_object() -> None:
    request = full_request(
        tools=(ToolSpec(name="send", description="Send.", input_schema=CONSTRAINED),),
        output_schema=CONSTRAINED,
    )
    params = provider_with(MagicMock()).build_params(request)
    for schema in (params["tools"][0]["input_schema"], params["output_config"]["format"]["schema"]):
        assert not _schema_keys(schema) & {"minLength", "maxLength", "minimum", "maximum"}
        assert schema["additionalProperties"] is False
        assert schema["properties"]["meta"]["additionalProperties"] is False
    assert request.tools[0].input_schema == CONSTRAINED  # the request itself is not mutated


def test_schemas_that_cannot_be_made_strict_fail_before_any_call() -> None:
    client = MagicMock()
    loose: dict[str, Any] = {"type": "object", "properties": {"anything": {}}}
    with pytest.raises(ProviderRequestError, match="strict"):
        provider_with(client).complete(
            full_request(tools=(ToolSpec(name="send", description="Send.", input_schema=loose),))
        )
    client.beta.messages.create.assert_not_called()


def test_paid_truncations_and_refusals_carry_model_and_usage() -> None:
    client = MagicMock()
    billed = SimpleNamespace(input_tokens=50, output_tokens=2048, iterations=None)
    client.beta.messages.create.return_value = message(stop_reason="max_tokens", usage=billed)
    with pytest.raises(ProviderTruncatedError) as truncated:
        provider_with(client).complete(full_request())
    assert truncated.value.usage == Usage(input_tokens=50, output_tokens=2048)
    assert truncated.value.model == DEFAULT_MODEL

    client.beta.messages.create.return_value = message(stop_reason="refusal", usage=billed)
    with pytest.raises(ProviderRefusalError) as refused:
        provider_with(client).complete(full_request())
    assert refused.value.usage == Usage(input_tokens=50, output_tokens=2048)


_REQUEST = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")


def _status_error(
    cls: type[anthropic.APIStatusError], status: int, headers: dict[str, str] | None = None
) -> anthropic.APIStatusError:
    response = httpx2.Response(status, request=_REQUEST, headers=headers or {"request-id": "req_1"})
    return cls("error", response=response, body=None)


@pytest.mark.parametrize(
    ("exception", "expected", "retryable"),
    [
        (_status_error(anthropic.BadRequestError, 400), ProviderRequestError, False),
        (_status_error(anthropic.AuthenticationError, 401), ProviderAuthError, False),
        (_status_error(anthropic.PermissionDeniedError, 403), ProviderAuthError, False),
        (_status_error(anthropic.NotFoundError, 404), ProviderRequestError, False),
        (_status_error(anthropic.RequestTooLargeError, 413), ProviderRequestError, False),
        (_status_error(anthropic.UnprocessableEntityError, 422), ProviderRequestError, False),
        (_status_error(anthropic.ConflictError, 409), ProviderError, True),
        (_status_error(anthropic.InternalServerError, 500), ProviderUnavailableError, True),
        (_status_error(anthropic.ServiceUnavailableError, 503), ProviderUnavailableError, True),
        (_status_error(anthropic.OverloadedError, 529), ProviderUnavailableError, True),
        (anthropic.APITimeoutError(request=_REQUEST), ProviderUnavailableError, True),
        (anthropic.APIConnectionError(request=_REQUEST), ProviderUnavailableError, True),
    ],
)
def test_sdk_errors_map_to_typed_provider_errors(
    exception: Exception, expected: type[ProviderError], retryable: bool
) -> None:
    client = MagicMock()
    client.beta.messages.create.side_effect = exception
    with pytest.raises(expected) as info:
        provider_with(client).complete(full_request())
    assert type(info.value) is expected
    assert info.value.retryable is retryable
    assert info.value.__cause__ is exception


def test_rate_limits_carry_retry_after() -> None:
    client = MagicMock()
    client.beta.messages.create.side_effect = _status_error(
        anthropic.RateLimitError, 429, {"retry-after": "7", "request-id": "req_2"}
    )
    with pytest.raises(ProviderRateLimitedError) as info:
        provider_with(client).complete(full_request())
    assert info.value.retry_after == 7.0
    assert info.value.retryable is True
    assert "req_2" in str(info.value)


def test_models_without_adaptive_thinking_get_no_thinking_effort_or_fallbacks() -> None:
    # The bundled reference documents adaptive thinking and effort only for newer models; Haiku
    # 4.5 takes neither, and the fallbacks beta is documented only for Opus 5 and Fable 5.x.
    provider = provider_with(MagicMock(), role_models={"judge": "claude-haiku-4-5"})
    plain = provider.build_params(LLMRequest(messages=(ChatMessage.user("rate"),), role="judge"))
    assert plain["model"] == "claude-haiku-4-5"
    assert {"thinking", "output_config", "betas", "fallbacks"}.isdisjoint(plain)
    structured = provider.build_params(
        LLMRequest(messages=(ChatMessage.user("rate"),), output_schema=SCHEMA, role="judge")
    )
    assert structured["output_config"] == {"format": {"type": "json_schema", "schema": SCHEMA}}


@pytest.mark.parametrize(
    ("model", "fallback"),
    [
        ("claude-opus-5", True),
        ("claude-fable-5-1", True),
        ("claude-sonnet-5", False),
        ("claude-opus-4-8", False),
    ],
)
def test_refusal_fallback_is_sent_only_to_models_documented_to_accept_it(
    model: str, fallback: bool
) -> None:
    params = provider_with(MagicMock(), model=model).build_params(full_request())
    assert params["thinking"] == {"type": "adaptive"}
    assert params["output_config"] == {"effort": "high"}
    assert ("fallbacks" in params) is fallback
    assert ("betas" in params) is fallback


@pytest.mark.parametrize("cached_in_iterations", [False, True])
def test_cache_tokens_are_charged_as_input_tokens(cached_in_iterations: bool) -> None:
    cache = {"cache_creation_input_tokens": 300, "cache_read_input_tokens": 700}
    if cached_in_iterations:
        usage = SimpleNamespace(
            input_tokens=10,
            output_tokens=5,
            iterations=[SimpleNamespace(type="message", input_tokens=10, output_tokens=5, **cache)],
        )
    else:
        usage = SimpleNamespace(input_tokens=10, output_tokens=5, iterations=None, **cache)
    client = MagicMock()
    client.beta.messages.create.return_value = message(usage=usage)
    response = provider_with(client).complete(full_request())
    assert (response.usage.input_tokens, response.usage.output_tokens) == (1_010, 5)


@pytest.mark.parametrize(
    "block",
    [
        SimpleNamespace(type="tool_use", id="x" * 500, name="read_file", input={"path": "a"}),
        SimpleNamespace(type="tool_use", id="toolu_1", name="read file!", input={"path": "a"}),
        SimpleNamespace(type="tool_use", id="toolu_1", name="read_file", input=["pa", "th"]),
    ],
)
def test_billed_responses_that_fail_validation_raise_provider_errors_with_usage(
    block: Any,
) -> None:
    client = MagicMock()
    client.beta.messages.create.return_value = message(stop_reason="tool_use", content=[block])
    with pytest.raises(ProviderError, match="malformed") as info:
        provider_with(client).complete(full_request())
    assert type(info.value) is ProviderError
    assert info.value.usage == Usage(input_tokens=10, output_tokens=5)
    assert info.value.model == DEFAULT_MODEL


def test_requests_above_the_non_streaming_limit_are_streamed() -> None:
    client = MagicMock()
    stream = client.beta.messages.stream.return_value.__enter__.return_value
    stream.get_final_message.return_value = message()
    provider = provider_with(client)
    request = full_request(max_tokens=32_000)
    assert provider.complete(request).text == "hi"
    assert client.beta.messages.stream.call_args.kwargs == provider.build_params(request)
    client.beta.messages.create.assert_not_called()


def test_streamed_request_errors_are_mapped_like_non_streamed_ones() -> None:
    client = MagicMock()
    client.beta.messages.stream.return_value.__enter__.side_effect = _status_error(
        anthropic.OverloadedError, 529
    )
    with pytest.raises(ProviderUnavailableError):
        provider_with(client).complete(full_request(max_tokens=32_000))


def test_client_timeout_defaults_to_the_sdk_default_of_ten_minutes() -> None:
    # A shorter total timeout cuts off non-streaming requests the SDK expects to take longer, and
    # every retry of a cut-off request can be billed again.
    assert Settings(_env_file=None).llm_timeout_seconds == 600.0
    client = AnthropicProvider(api_key=SecretStr("test-key-not-real"))._get_client()
    assert client.timeout == 600.0


def test_offline_modules_do_not_import_the_sdk() -> None:
    code = (
        "import sys\n"
        "import agent_supervision_tree.agentic.providers.anthropic\n"
        "import agent_supervision_tree.workload.pipeline\n"
        "import agent_supervision_tree.workload.demo\n"
        "print('anthropic' in sys.modules)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, timeout=120
    )
    assert result.stdout.strip() == "False"
