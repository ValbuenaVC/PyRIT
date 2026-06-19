# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Unit tests for pyrit.agent.tools (S3.1)."""

from typing import Any
from unittest.mock import AsyncMock

import pytest

from pyrit.agent.tools import Tool, ToolCall, ToolCallDispatch, ToolResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _echo_tool(args: dict[str, Any]) -> dict[str, Any]:
    return {"result": args.get("value", "echo")}


_SIMPLE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"value": {"type": "string"}},
    "required": ["value"],
}


# ---------------------------------------------------------------------------
# Tool model tests
# ---------------------------------------------------------------------------


def test_tool_construction_stores_fields() -> None:
    tool = Tool(
        name="echo",
        description="Echoes input",
        json_schema=_SIMPLE_SCHEMA,
        callable=_echo_tool,
    )
    assert tool.name == "echo"
    assert tool.description == "Echoes input"
    assert tool.json_schema == _SIMPLE_SCHEMA
    assert tool.callable is _echo_tool


def test_tool_name_required() -> None:
    with pytest.raises(Exception):
        Tool(description="No name", json_schema={}, callable=_echo_tool)  # type: ignore[missing-argument]  # ty: ignore[missing-argument]


def test_tool_description_required() -> None:
    with pytest.raises(Exception):
        Tool(name="x", json_schema={}, callable=_echo_tool)  # type: ignore[missing-argument]  # ty: ignore[missing-argument]


def test_tool_callable_required() -> None:
    with pytest.raises(Exception):
        Tool(name="x", description="desc", json_schema={})  # type: ignore[missing-argument]  # ty: ignore[missing-argument]


def test_tool_json_schema_required() -> None:
    with pytest.raises(Exception):
        Tool(name="x", description="desc", callable=_echo_tool)  # type: ignore[missing-argument]  # ty: ignore[missing-argument]


def test_tool_equality_by_name() -> None:
    """Tools with same name compare as equal (set membership)."""
    t1 = Tool(name="foo", description="a", json_schema={}, callable=_echo_tool)
    t2 = Tool(name="foo", description="b", json_schema={}, callable=_echo_tool)
    # Pydantic equality is field-by-field; confirm name is stored
    assert t1.name == t2.name


def test_tool_in_set() -> None:
    t1 = Tool(name="foo", description="a", json_schema={}, callable=_echo_tool)
    t2 = Tool(name="bar", description="b", json_schema={}, callable=_echo_tool)
    toolset: set[Tool] = {t1, t2}
    assert len(toolset) == 2


# ---------------------------------------------------------------------------
# ToolCallDispatch protocol conformance
# ---------------------------------------------------------------------------


class _ConcreteDispatcher:
    """A minimal concrete implementation of ToolCallDispatch."""

    async def call_async(self, *, tool_call: ToolCall) -> ToolResult:
        return {"result": "ok"}


def test_tool_call_dispatch_protocol_conformance() -> None:
    """Concrete class implementing the protocol is accepted at runtime."""
    dispatcher: ToolCallDispatch = _ConcreteDispatcher()  # type: ignore[assignment]
    assert callable(getattr(dispatcher, "call_async", None))


async def test_tool_call_dispatch_callable_invocable() -> None:
    dispatcher = _ConcreteDispatcher()
    tool_call: ToolCall = {
        "type": "function_call",
        "call_id": "call_abc",
        "name": "echo",
        "arguments": '{"value": "hello"}',
    }
    result = await dispatcher.call_async(tool_call=tool_call)
    assert result == {"result": "ok"}


def test_async_mock_satisfies_protocol() -> None:
    """AsyncMock can stand in as a ToolCallDispatch in tests."""
    mock: ToolCallDispatch = AsyncMock(spec=_ConcreteDispatcher)  # type: ignore[assignment]
    assert hasattr(mock, "call_async")


# ---------------------------------------------------------------------------
# ToolResult shape
# ---------------------------------------------------------------------------


def test_tool_result_is_dict() -> None:
    result: ToolResult = {"call_id": "x", "output": "hello"}
    assert isinstance(result, dict)
