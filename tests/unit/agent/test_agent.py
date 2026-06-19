# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Unit tests for pyrit.agent.agent (S3.4 — Agent)."""

import json
import uuid
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pyrit.agent.agent import Agent
from pyrit.agent.runtime import InProcessRuntime
from pyrit.agent.tools import Tool
from pyrit.models import Message, MessagePiece
from pyrit.models.target_capabilities import TargetCapabilities, ToolUsageSchema
from pyrit.prompt_target.common.target_capabilities import TargetCapabilities
from unit.mocks import MockPromptTarget, get_mock_target_identifier

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CONV_ID = str(uuid.uuid4())


def _make_text_message(text: str = "hello", role: str = "user") -> Message:
    return MessagePiece(
        role=role,  # type: ignore[arg-type]
        original_value=text,
        converted_value=text,
        conversation_id=_CONV_ID,
    ).to_message()


def _make_function_call_message(name: str, call_id: str = "call_1", args: dict[str, Any] | None = None) -> Message:
    value = json.dumps(
        {
            "type": "function_call",
            "call_id": call_id,
            "name": name,
            "arguments": json.dumps(args or {}),
        }
    )
    return MessagePiece(
        role="assistant",
        original_value=value,
        original_value_data_type="function_call",  # type: ignore[arg-type]
        conversation_id=_CONV_ID,
    ).to_message()


def _make_tool_result_message(call_id: str, output: str) -> Message:
    value = json.dumps({"type": "function_call_output", "call_id": call_id, "output": output})
    return MessagePiece(
        role="tool",
        original_value=value,
        original_value_data_type="function_call_output",  # type: ignore[arg-type]
        conversation_id=_CONV_ID,
    ).to_message()


async def _add_fn(args: dict[str, Any]) -> dict[str, Any]:
    return {"sum": args["a"] + args["b"]}


_ADD_TOOL = Tool(
    name="add",
    description="Adds two numbers",
    json_schema={},
    callable=_add_fn,
)

_TOOL_USAGE_SCHEMA = ToolUsageSchema()

_TOOL_USING_CAPABILITIES = TargetCapabilities(
    supports_tool_usage=True,
    tool_usage_schema=_TOOL_USAGE_SCHEMA,
    supports_multi_turn=True,
    supports_multi_message_pieces=True,
)


def _make_tool_using_mock_target() -> MagicMock:
    """Return a MagicMock PromptTarget that reports tool-calling capability."""
    from pyrit.prompt_target import PromptTarget

    target = MagicMock(spec=PromptTarget)
    target.get_identifier.return_value = get_mock_target_identifier("MockToolTarget")
    target.capabilities = _TOOL_USING_CAPABILITIES
    return target


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("patch_central_database")
def test_agent_keyword_only_init() -> None:
    inner = MockPromptTarget()
    runtime = InProcessRuntime(tools=[])
    # Should succeed with keyword args
    agent = Agent(target=inner, toolset=set(), dispatcher=runtime)
    assert agent is not None


@pytest.mark.usefixtures("patch_central_database")
def test_agent_uses_inner_target_capabilities() -> None:
    inner = MockPromptTarget()
    agent = Agent(target=inner, toolset=set())
    # Agent is a PromptTarget with its own capabilities
    assert agent.capabilities is not None


@pytest.mark.usefixtures("patch_central_database")
def test_agent_is_prompt_target() -> None:
    from pyrit.prompt_target import PromptTarget

    inner = MockPromptTarget()
    agent = Agent(target=inner, toolset=set())
    assert isinstance(agent, PromptTarget)


# ---------------------------------------------------------------------------
# No-tool-call passthrough
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("patch_central_database")
async def test_agent_passthrough_no_tool_call() -> None:
    """When inner target returns plain text, Agent returns it unchanged."""
    inner = MockPromptTarget()
    agent = Agent(target=inner, toolset=set())

    user_msg = _make_text_message("What is 2+2?")
    inner_response = _make_text_message("4", role="assistant")

    with patch.object(
        inner,
        "_send_prompt_to_target_async",
        new_callable=AsyncMock,
        return_value=[inner_response],
    ):
        result = await agent._send_prompt_to_target_async(normalized_conversation=[user_msg])

    assert len(result) >= 1
    assert result[0].get_value() == "4"


# ---------------------------------------------------------------------------
# Single tool call executes and loops
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("patch_central_database")
async def test_agent_executes_tool_call_and_returns_final_response() -> None:
    """Agent executes a tool call and fetches the follow-up response."""
    inner_mock = _make_tool_using_mock_target()
    runtime = InProcessRuntime(tools=[_ADD_TOOL])
    agent = Agent(target=inner_mock, toolset={_ADD_TOOL}, dispatcher=runtime)

    user_msg = _make_text_message("Add 3 and 4")
    tool_call_msg = _make_function_call_message("add", "call_1", {"a": 3, "b": 4})
    final_text_msg = _make_text_message("The sum is 7", role="assistant")

    # First call: returns tool call; second call: returns final text
    inner_mock._send_prompt_to_target_async = AsyncMock(
        side_effect=[
            [tool_call_msg],
            [final_text_msg],
        ]
    )

    result = await agent._send_prompt_to_target_async(normalized_conversation=[user_msg])

    # Should include the tool call response, the tool result, and the final response
    assert any("7" in r.get_value() for r in result)
    assert inner_mock._send_prompt_to_target_async.call_count == 2


# ---------------------------------------------------------------------------
# Max iteration guard
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("patch_central_database")
async def test_agent_max_iteration_guard() -> None:
    """Agent stops looping after max_tool_iterations even if tool calls keep coming."""
    inner_mock = _make_tool_using_mock_target()
    runtime = InProcessRuntime(tools=[_ADD_TOOL])
    agent = Agent(target=inner_mock, toolset={_ADD_TOOL}, dispatcher=runtime, max_tool_iterations=3)

    user_msg = _make_text_message("Add forever")
    # Each call returns another tool call
    tool_call_msg = _make_function_call_message("add", "call_x", {"a": 1, "b": 1})
    inner_mock._send_prompt_to_target_async = AsyncMock(return_value=[tool_call_msg])

    result = await agent._send_prompt_to_target_async(normalized_conversation=[user_msg])

    # Should stop after max_tool_iterations + 1 total calls (initial + max)
    assert inner_mock._send_prompt_to_target_async.call_count <= 4  # max_tool_iterations + 1
    assert isinstance(result, list)


# ---------------------------------------------------------------------------
# Results are well-formed list[Message]
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("patch_central_database")
async def test_agent_results_are_messages() -> None:
    inner = MockPromptTarget()
    agent = Agent(target=inner, toolset=set())
    user_msg = _make_text_message("test")
    inner_response = _make_text_message("response", role="assistant")

    with patch.object(
        inner,
        "_send_prompt_to_target_async",
        new_callable=AsyncMock,
        return_value=[inner_response],
    ):
        result = await agent._send_prompt_to_target_async(normalized_conversation=[user_msg])

    assert isinstance(result, list)
    for r in result:
        assert isinstance(r, Message)
        for piece in r.message_pieces:
            assert isinstance(piece, MessagePiece)
