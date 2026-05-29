# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""F2 integration test: AzureMLChatTarget end-to-end through @tool_loop.

Validates that PyRIT's full client-side tool-calling stack works against
an AzureML chat target whose scoring script emits the canonical
function_call envelope per plan §12.9.2. Only the outbound HTTP layer is
mocked; the @tool_loop decorator, CanonicalEnvelopeParser, LocalToolBackend
dispatch, ChatMessageNormalizer tool-piece serialization, and Memory
persistence all run unmocked.

The test asserts the canonical four-piece transcript shape:
``[user, assistant function_call, tool function_call_output, assistant text]``.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pyrit.memory import CentralMemory
from pyrit.models import Message, MessagePiece
from pyrit.prompt_normalizer import PromptNormalizer
from pyrit.prompt_target import AzureMLChatTarget
from pyrit.prompt_target.common.target_capabilities import TargetCapabilities
from pyrit.prompt_target.common.target_configuration import TargetConfiguration
from pyrit.tools import (
    CanonicalEnvelopeParser,
    LocalToolBackend,
    ToolEventBehavior,
    ToolEventPolicy,
)


@pytest.fixture
def echo_backend():
    async def _echo(args):
        return {"echoed": args.get("text", "")}

    return LocalToolBackend(
        callables={"echo": _echo},
        schemas=[
            {
                "name": "echo",
                "description": "Echo back the given text.",
                "parameters": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
            }
        ],
    )


@pytest.mark.run_only_if_all_tests
async def test_azure_ml_chat_target_tool_loop_round_trip(patch_central_database, echo_backend):
    """User asks for a tool call; the loop dispatches; the model produces final text."""
    target = AzureMLChatTarget(
        endpoint="https://mock-endpoint.example.com/score",
        api_key="dummy",
        tool_parser=CanonicalEnvelopeParser(),
        tool_backend=echo_backend,
        custom_configuration=TargetConfiguration(
            capabilities=TargetCapabilities(
                supports_multi_message_pieces=True,
                supports_editable_history=True,
                supports_multi_turn=True,
                supports_system_prompt=True,
                supports_tool_use=True,
            ),
            tool_event_policy=ToolEventPolicy(
                behavior=ToolEventBehavior.EXECUTE,
                max_tool_iterations=3,
            ),
            tool_backend=echo_backend,
        ),
    )

    first_response = MagicMock()
    first_response.json.return_value = {
        "output": "",
        "tool_calls": [
            {
                "type": "function_call",
                "call_id": "call_0",
                "name": "echo",
                "arguments": '{"text":"hi"}',
            }
        ],
    }
    second_response = MagicMock()
    second_response.json.return_value = {"output": "The echoed text is: hi"}

    user_msg = Message(
        message_pieces=[
            MessagePiece(
                role="user",
                original_value="Use the echo tool to repeat 'hi'.",
                original_value_data_type="text",
            )
        ]
    )

    with patch(
        "pyrit.common.net_utility.make_request_and_raise_if_error_async",
        new_callable=AsyncMock,
    ) as mock_http:
        mock_http.side_effect = [first_response, second_response]
        normalizer = PromptNormalizer()
        result = await normalizer.send_prompt_async(message=user_msg, target=target)

    assert mock_http.call_count == 2
    final_text = result.get_value()
    assert "The echoed text is" in final_text

    conv = CentralMemory.get_memory_instance().get_conversation(
        conversation_id=result.message_pieces[0].conversation_id
    )
    # Canonical four-piece transcript: user -> assistant fc -> tool fco -> assistant text.
    flat_pieces = [p for msg in conv for p in msg.message_pieces]
    types = [p.original_value_data_type for p in flat_pieces]
    roles = [p.api_role for p in flat_pieces]
    assert types == ["text", "function_call", "function_call_output", "text"]
    assert roles == ["user", "assistant", "tool", "assistant"]

    # call_id round-trips between the assistant and tool messages.
    fc_envelope = json.loads(flat_pieces[1].original_value)
    fco_envelope = json.loads(flat_pieces[2].original_value)
    assert fc_envelope["call_id"] == fco_envelope["call_id"] == "call_0"
    # Tool dispatched the local echo callable; the output reflects the args.
    assert json.loads(fco_envelope["output"])["echoed"] == "hi"


@pytest.mark.run_only_if_all_tests
async def test_azure_ml_chat_target_no_tools_backward_compat(patch_central_database):
    """Without tool_parser / tool_backend, the request body has no tools key."""
    target = AzureMLChatTarget(
        endpoint="https://mock-endpoint.example.com/score",
        api_key="dummy",
    )
    user_msg = Message(
        message_pieces=[MessagePiece(role="user", original_value="Hello", original_value_data_type="text")]
    )

    response = MagicMock()
    response.json.return_value = {"output": "Hi back"}

    with patch(
        "pyrit.common.net_utility.make_request_and_raise_if_error_async",
        new_callable=AsyncMock,
    ) as mock_http:
        mock_http.return_value = response
        normalizer = PromptNormalizer()
        result = await normalizer.send_prompt_async(message=user_msg, target=target)

    assert result.get_value() == "Hi back"
    body = mock_http.call_args.kwargs["request_body"]
    assert "tools" not in body
