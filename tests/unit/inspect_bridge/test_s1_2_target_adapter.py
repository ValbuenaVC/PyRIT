# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Tests for S1.2: TargetToModelAdapter (full implementation).

No network; mocks PyRIT PromptTarget and TargetRegistry.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pyrit.inspect_bridge.errors import InspectBridgeError
from pyrit.prompt_target import PromptTarget


def _make_mock_target(*, unique_name: str = "mock_target_abc") -> MagicMock:
    """Create a MagicMock with PromptTarget spec."""
    target = MagicMock(spec=PromptTarget)
    identifier = MagicMock()
    identifier.unique_name = unique_name
    target.get_identifier.return_value = identifier
    return target


def _make_chat_msg_user(text: str = "Hello") -> object:
    from inspect_ai.model import ChatMessageUser

    return ChatMessageUser(content=text)


def _make_chat_msg_assistant(text: str = "Response") -> object:
    from inspect_ai.model import ChatMessageAssistant

    return ChatMessageAssistant(content=text)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_target_adapter_direct_target() -> None:
    """When target is provided directly, it is stored and accessible via .target."""
    from pyrit.inspect_bridge._target_adapter import TargetToModelAdapter

    mock_target = _make_mock_target()
    adapter = TargetToModelAdapter("pyrit/mock_target_abc", target=mock_target)
    assert adapter.target is mock_target


def test_target_adapter_is_model_api() -> None:
    """TargetToModelAdapter is a subclass of inspect_ai.model.ModelAPI."""
    from inspect_ai.model import ModelAPI

    from pyrit.inspect_bridge._target_adapter import TargetToModelAdapter

    mock_target = _make_mock_target()
    adapter = TargetToModelAdapter("pyrit/mock_target_abc", target=mock_target)
    assert isinstance(adapter, ModelAPI)


def test_target_adapter_model_name_preserved() -> None:
    """The model_name is preserved and accessible via .model_name."""
    from pyrit.inspect_bridge._target_adapter import TargetToModelAdapter

    mock_target = _make_mock_target()
    adapter = TargetToModelAdapter("pyrit/my_target", target=mock_target)
    assert adapter.model_name == "pyrit/my_target"


def test_target_adapter_resolves_from_registry() -> None:
    """When target=None, the target is resolved from TargetRegistry by name."""
    from pyrit.inspect_bridge._target_adapter import TargetToModelAdapter

    mock_target = _make_mock_target(unique_name="registered_target")
    mock_registry = MagicMock()
    mock_registry.get_instance_by_name.return_value = mock_target

    patch_target = "pyrit.registry.object_registries.target_registry.TargetRegistry.get_registry_singleton"
    with patch(patch_target, return_value=mock_registry):
        adapter = TargetToModelAdapter("pyrit/registered_target")
        assert adapter.target is mock_target
        mock_registry.get_instance_by_name.assert_called_once_with("registered_target")


def test_target_adapter_registry_miss_raises() -> None:
    """When target=None and registry lookup fails, InspectBridgeError is raised."""
    from pyrit.inspect_bridge._target_adapter import TargetToModelAdapter

    mock_registry = MagicMock()
    mock_registry.get_instance_by_name.return_value = None

    patch_target = "pyrit.registry.object_registries.target_registry.TargetRegistry.get_registry_singleton"
    with patch(patch_target, return_value=mock_registry):
        with pytest.raises(InspectBridgeError):
            TargetToModelAdapter("pyrit/nonexistent_target")


# ---------------------------------------------------------------------------
# model_name_for static method
# ---------------------------------------------------------------------------


def test_model_name_for_returns_pyrit_prefix() -> None:
    """model_name_for returns 'pyrit/<unique_name>'."""
    from pyrit.inspect_bridge._target_adapter import TargetToModelAdapter

    mock_target = _make_mock_target(unique_name="my_chat_target")
    name = TargetToModelAdapter.model_name_for(target=mock_target)
    assert name == "pyrit/my_chat_target"


# ---------------------------------------------------------------------------
# generate — tools rejection
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_raises_on_nonempty_tools() -> None:
    """generate() raises InspectBridgeError when tools is non-empty."""
    from inspect_ai.tool import ToolInfo

    from pyrit.inspect_bridge._target_adapter import TargetToModelAdapter

    mock_target = _make_mock_target()
    adapter = TargetToModelAdapter("pyrit/mock_target_abc", target=mock_target)

    fake_tool = MagicMock(spec=ToolInfo)
    with pytest.raises(InspectBridgeError):
        await adapter.generate(
            input=[_make_chat_msg_user()],
            tools=[fake_tool],
            tool_choice="none",
            config=None,
        )


# ---------------------------------------------------------------------------
# generate — happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_calls_target_send_prompt() -> None:
    """generate() calls _send_prompt_to_target_async on the wrapped target."""
    from inspect_ai.model import ModelOutput

    from pyrit.inspect_bridge._target_adapter import TargetToModelAdapter

    mock_target = _make_mock_target()

    # Target returns a list of Message objects
    from pyrit.models import Message, MessagePiece

    response_piece = MessagePiece(
        role="assistant",
        original_value="AI response",
        original_value_data_type="text",
        conversation_id="conv-1",
        sequence=1,
    )
    response_msg = Message(message_pieces=[response_piece])
    mock_target._send_prompt_to_target_async = AsyncMock(return_value=[response_msg])

    adapter = TargetToModelAdapter("pyrit/mock_target_abc", target=mock_target)

    input_msgs = [_make_chat_msg_user("What is 2+2?")]
    result = await adapter.generate(
        input=input_msgs,
        tools=[],
        tool_choice="none",
        config=None,
    )

    assert isinstance(result, ModelOutput)
    mock_target._send_prompt_to_target_async.assert_called_once()
    # Verify normalized_conversation kwarg was passed
    call_kwargs = mock_target._send_prompt_to_target_async.call_args.kwargs
    assert "normalized_conversation" in call_kwargs


@pytest.mark.asyncio
async def test_generate_output_contains_response_text() -> None:
    """generate() output contains the text returned by the target."""
    from inspect_ai.model import ModelOutput

    from pyrit.inspect_bridge._target_adapter import TargetToModelAdapter
    from pyrit.models import Message, MessagePiece

    mock_target = _make_mock_target()
    response_piece = MessagePiece(
        role="assistant",
        original_value="The answer is 42",
        original_value_data_type="text",
        conversation_id="conv-1",
        sequence=1,
    )
    response_msg = Message(message_pieces=[response_piece])
    mock_target._send_prompt_to_target_async = AsyncMock(return_value=[response_msg])

    adapter = TargetToModelAdapter("pyrit/mock_target_abc", target=mock_target)
    result = await adapter.generate(input=[_make_chat_msg_user("?")], tools=[], tool_choice="none", config=None)

    assert isinstance(result, ModelOutput)
    content = result.choices[0].message.content
    assert "42" in (content if isinstance(content, str) else str(content))
