from __future__ import annotations

import pytest
from pydantic import BaseModel, ConfigDict, Field, JsonValue, SecretStr

from agent_supervision_tree.agentic.errors import ToolArgumentError, ToolNotAllowedError
from agent_supervision_tree.agentic.providers.base import ToolUseBlock
from agent_supervision_tree.agentic.tools import Capability, ToolDefinition, ToolRegistry

pytestmark = pytest.mark.unit


class EchoArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(max_length=20)
    count: int = 1


class WriteArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str


class OpenArgs(BaseModel):
    path: str


def echo(args: EchoArgs) -> JsonValue:
    return args.text * args.count


def write(args: WriteArgs) -> JsonValue:
    return f"wrote {args.path}"


ECHO = ToolDefinition(
    name="echo",
    description="Echo text.",
    args_model=EchoArgs,
    capabilities=frozenset({Capability.READ_DATA}),
    handler=echo,
)
WRITE = ToolDefinition(
    name="write_file",
    description="Write a file.",
    args_model=WriteArgs,
    capabilities=frozenset({Capability.WRITE_DATA}),
    handler=write,
)


def make_registry() -> ToolRegistry:
    registry = ToolRegistry({Capability.READ_DATA, Capability.WRITE_DATA})
    registry.register(ECHO)
    registry.register(WRITE)
    return registry


def call(name: str, **arguments: JsonValue) -> ToolUseBlock:
    return ToolUseBlock(id="call-1", name=name, arguments=arguments)


def test_registration_requires_granted_capabilities() -> None:
    registry = ToolRegistry({Capability.READ_DATA})
    with pytest.raises(ToolNotAllowedError, match="write_data"):
        registry.register(WRITE)


def test_duplicate_registration_is_rejected() -> None:
    registry = make_registry()
    with pytest.raises(ValueError, match="already registered"):
        registry.register(ECHO)


def test_tool_definitions_are_validated() -> None:
    with pytest.raises(ValueError, match="extra='forbid'"):
        ToolDefinition(
            name="open",
            description="Open.",
            args_model=OpenArgs,
            capabilities=frozenset({Capability.READ_DATA}),
            handler=lambda args: args.path,
        )
    with pytest.raises(ValueError, match="cannot opt out"):
        ToolDefinition(
            name="write2",
            description="Write.",
            args_model=WriteArgs,
            capabilities=frozenset({Capability.WRITE_DATA}),
            handler=write,
            requires_approval=False,
        )
    with pytest.raises(ValueError, match="invalid tool name"):
        ToolDefinition(
            name="bad name",
            description="x",
            args_model=EchoArgs,
            capabilities=frozenset({Capability.READ_DATA}),
            handler=echo,
        )
    with pytest.raises(ValueError, match="at least one capability"):
        ToolDefinition(
            name="none",
            description="x",
            args_model=EchoArgs,
            capabilities=frozenset(),
            handler=echo,
        )


def test_approval_defaults_follow_capabilities() -> None:
    assert ECHO.needs_approval is False
    assert WRITE.needs_approval is True
    cautious = ToolDefinition(
        name="echo_reviewed",
        description="Echo with review.",
        args_model=EchoArgs,
        capabilities=frozenset({Capability.READ_DATA}),
        handler=echo,
        requires_approval=True,
    )
    assert cautious.needs_approval is True


def test_unknown_tools_are_not_allowed() -> None:
    with pytest.raises(ToolNotAllowedError, match="delete_all"):
        make_registry().validate_call(call("delete_all"))


@pytest.mark.parametrize(
    "arguments",
    [
        {"text": "hi", "extra": "SECRETVALUE"},
        {"text": "hi", "count": "3"},
        {"text": "SECRETVALUE" * 3},
        {},
    ],
)
def test_invalid_arguments_are_rejected_without_echoing_values(
    arguments: dict[str, JsonValue],
) -> None:
    with pytest.raises(ToolArgumentError) as info:
        make_registry().validate_call(call("echo", **arguments))
    assert "SECRETVALUE" not in str(info.value)


def test_valid_calls_are_canonicalized_and_runnable() -> None:
    validated = make_registry().validate_call(call("echo", text="hi"))
    assert validated.canonical_args == {"text": "hi", "count": 1}
    assert validated.run() == "hi"


def test_scoped_registry_is_a_narrower_allowlist() -> None:
    registry = make_registry()
    scoped = registry.scoped(["echo"])
    assert scoped.names == ("echo",)
    assert scoped.granted == frozenset({Capability.READ_DATA})
    with pytest.raises(ToolNotAllowedError):
        scoped.validate_call(call("write_file", path="x"))
    with pytest.raises(ValueError, match="unknown tools"):
        registry.scoped(["echo", "missing"])


def test_specs_are_sorted_and_schemas_are_closed() -> None:
    specs = make_registry().specs()
    assert [spec.name for spec in specs] == ["echo", "write_file"]
    assert all(spec.input_schema.get("additionalProperties") is False for spec in specs)


class TokenArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    token: SecretStr


def test_argument_models_that_do_not_round_trip_are_refused_without_echoing_values() -> None:
    registry = ToolRegistry({Capability.READ_DATA})
    registry.register(
        ToolDefinition(
            name="use_token",
            description="Use a token.",
            args_model=TokenArgs,
            capabilities=frozenset({Capability.READ_DATA}),
            handler=lambda args: "ok",
        )
    )
    # SecretStr dumps as "**********", so every token would share one call hash.
    with pytest.raises(ToolArgumentError, match="canonical") as info:
        registry.validate_call(call("use_token", token="s3cr3t-value"))
    assert "s3cr3t-value" not in str(info.value)
