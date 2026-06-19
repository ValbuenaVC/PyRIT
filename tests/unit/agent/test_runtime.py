# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Unit tests for pyrit.agent.runtime (S3.2 — InProcessRuntime)."""

from typing import Any

import pytest

from pyrit.agent.runtime import InProcessRuntime
from pyrit.agent.tools import Tool, ToolCall

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _add_tool(args: dict[str, Any]) -> dict[str, Any]:
    return {"sum": args["a"] + args["b"]}


async def _boom_tool(args: dict[str, Any]) -> dict[str, Any]:
    raise RuntimeError("tool exploded")


_ADD_TOOL = Tool(
    name="add",
    description="Adds two numbers",
    json_schema={"type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}}},
    callable=_add_tool,
)

_BOOM_TOOL = Tool(
    name="boom",
    description="Always raises",
    json_schema={},
    callable=_boom_tool,
)


def _make_tool_call(name: str, args: dict[str, Any] | None = None, call_id: str = "call_1") -> ToolCall:
    import json

    return {
        "type": "function_call",
        "call_id": call_id,
        "name": name,
        "arguments": json.dumps(args or {}),
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_in_process_runtime_dispatches_registered_tool() -> None:
    runtime = InProcessRuntime(tools=[_ADD_TOOL])
    tool_call = _make_tool_call("add", {"a": 3, "b": 4})
    result = await runtime.call_async(tool_call=tool_call)
    assert result["sum"] == 7


async def test_in_process_runtime_unknown_tool_returns_structured_error() -> None:
    runtime = InProcessRuntime(tools=[_ADD_TOOL])
    tool_call = _make_tool_call("nonexistent")
    result = await runtime.call_async(tool_call=tool_call)
    assert result.get("error") == "function_not_found"
    assert result.get("missing_function") == "nonexistent"
    assert "available_functions" in result
    assert "add" in result["available_functions"]


async def test_in_process_runtime_exception_captured_as_error() -> None:
    runtime = InProcessRuntime(tools=[_BOOM_TOOL])
    tool_call = _make_tool_call("boom")
    result = await runtime.call_async(tool_call=tool_call)
    assert result.get("error") == "tool_execution_error"
    assert "tool exploded" in result.get("message", "")


async def test_in_process_runtime_malformed_arguments_returns_structured_error() -> None:
    runtime = InProcessRuntime(tools=[_ADD_TOOL])
    bad_call: ToolCall = {
        "type": "function_call",
        "call_id": "call_2",
        "name": "add",
        "arguments": "NOT VALID JSON {{{",
    }
    result = await runtime.call_async(tool_call=bad_call)
    assert result.get("error") == "malformed_arguments"


async def test_in_process_runtime_missing_name_returns_structured_error() -> None:
    runtime = InProcessRuntime(tools=[_ADD_TOOL])
    bad_call: ToolCall = {"type": "function_call", "call_id": "call_3", "arguments": "{}"}
    result = await runtime.call_async(tool_call=bad_call)
    assert result.get("error") == "missing_function_name"


async def test_in_process_runtime_empty_toolset() -> None:
    runtime = InProcessRuntime(tools=[])
    result = await runtime.call_async(tool_call=_make_tool_call("anything"))
    assert result.get("error") == "function_not_found"
    assert result.get("available_functions") == []


async def test_in_process_runtime_multiple_tools() -> None:
    runtime = InProcessRuntime(tools=[_ADD_TOOL, _BOOM_TOOL])
    result = await runtime.call_async(tool_call=_make_tool_call("add", {"a": 1, "b": 2}))
    assert result["sum"] == 3


def test_in_process_runtime_keyword_only_init() -> None:
    with pytest.raises(TypeError):
        InProcessRuntime([_ADD_TOOL])  # type: ignore[missing-argument,too-many-positional-arguments]  # ty: ignore[missing-argument,too-many-positional-arguments]
