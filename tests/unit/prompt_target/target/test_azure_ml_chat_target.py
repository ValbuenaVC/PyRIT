# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from collections.abc import MutableSequence
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import HTTPStatusError
from openai import RateLimitError
from unit.mocks import get_sample_conversations

from pyrit.exceptions import EmptyResponseException, RateLimitException
from pyrit.models import Message, MessagePiece
from pyrit.prompt_target import AzureMLChatTarget


@pytest.fixture
def sample_conversations() -> MutableSequence[MessagePiece]:
    conversations = get_sample_conversations()
    return Message.flatten_to_message_pieces(conversations)


@pytest.fixture
def aml_online_chat(patch_central_database) -> AzureMLChatTarget:
    return AzureMLChatTarget(
        endpoint="http://aml-test-endpoint.com",
        api_key="valid_api_key",
        extra_param1="sample",
        extra_param2=1.0,
    )


def test_initialization_with_required_parameters(
    aml_online_chat: AzureMLChatTarget,
):
    assert aml_online_chat._endpoint == "http://aml-test-endpoint.com"
    assert aml_online_chat._api_key == "valid_api_key"


def test_initialization_with_extra_model_parameters(aml_online_chat: AzureMLChatTarget):
    assert aml_online_chat._extra_parameters == {"extra_param1": "sample", "extra_param2": 1.0}


def test_initialization_with_no_key_raises():
    os.environ[AzureMLChatTarget.api_key_environment_variable] = ""
    with pytest.raises(ValueError):
        AzureMLChatTarget(endpoint="http://aml-test-endpoint.com")


def test_initialization_with_no_api_raises():
    os.environ[AzureMLChatTarget.endpoint_uri_environment_variable] = ""
    with pytest.raises(ValueError):
        AzureMLChatTarget(api_key="xxxxx")


def test_get_headers_with_valid_api_key(aml_online_chat: AzureMLChatTarget):
    expected_headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer valid_api_key",
    }
    assert aml_online_chat._get_headers() == expected_headers


async def test_complete_chat_async(aml_online_chat: AzureMLChatTarget):
    messages = [
        Message(message_pieces=[MessagePiece(role="user", conversation_id="123", original_value="user content")]),
    ]

    with patch("pyrit.common.net_utility.make_request_and_raise_if_error_async") as mock:
        mock_response = MagicMock()
        mock_response.json.return_value = {"output": "extracted response"}
        mock.return_value = mock_response
        response = await aml_online_chat._complete_chat_async(messages)
        assert response == {"output": "extracted response"}
        mock.assert_called_once()


# Test that ChatMessageNormalizer (the default) passes messages through correctly
async def test_complete_chat_async_with_default_normalizer(
    aml_online_chat: AzureMLChatTarget,
):
    messages = [
        Message(message_pieces=[MessagePiece(role="system", conversation_id="123", original_value="system content")]),
        Message(message_pieces=[MessagePiece(role="user", conversation_id="123", original_value="user content")]),
    ]

    with patch("pyrit.common.net_utility.make_request_and_raise_if_error_async", new_callable=AsyncMock) as mock:
        mock_response = MagicMock()
        mock_response.json.return_value = {"output": "extracted response"}
        mock.return_value = mock_response
        response = await aml_online_chat._complete_chat_async(messages)
        assert response == {"output": "extracted response"}

        args, kwargs = mock.call_args
        body = kwargs["request_body"]

        assert body
        assert len(body["input_data"]["input_string"]) == 2
        assert body["input_data"]["input_string"][0]["role"] == "system"


async def test_complete_chat_async_bad_json_response(aml_online_chat: AzureMLChatTarget):
    messages = [
        Message(message_pieces=[MessagePiece(role="user", conversation_id="123", original_value="user content")]),
    ]

    with patch("pyrit.common.net_utility.make_request_and_raise_if_error_async", new_callable=AsyncMock) as mock:
        mock_response = MagicMock()
        # Set is a non-dict body that previously raised TypeError when the code
        # subscripted response.json()["output"]; the new code raises ValueError
        # because the body is not a dict.
        mock_response.json.return_value = {"bad response"}
        mock.return_value = mock_response
        with pytest.raises((TypeError, ValueError, EmptyResponseException)):
            await aml_online_chat._complete_chat_async(messages)


async def test_send_prompt_async_bad_request_error_adds_to_memory(aml_online_chat: AzureMLChatTarget):
    mock_memory = MagicMock()
    mock_memory.get_conversation.return_value = []
    mock_memory.add_message_to_memory = AsyncMock()

    aml_online_chat._memory = mock_memory

    response = MagicMock()
    response.status_code = 400
    response.text = "Bad Request"
    mock_complete_chat_async = AsyncMock(
        side_effect=HTTPStatusError(message="Bad Request", request=MagicMock(), response=response)
    )
    aml_online_chat._complete_chat_async = mock_complete_chat_async
    message = Message(message_pieces=[MessagePiece(role="user", conversation_id="123", original_value="Hello")])

    with pytest.raises(HTTPStatusError) as bre:
        await aml_online_chat.send_prompt_async(message=message)
        aml_online_chat._memory.get_conversation.assert_called_once_with(conversation_id="123")
        aml_online_chat._memory.add_message_to_memory.assert_called_once_with(request=message)

    assert str(bre.value) == "Bad Request"


async def test_send_prompt_async_rate_limit_exception_adds_to_memory(aml_online_chat: AzureMLChatTarget):
    mock_memory = MagicMock()
    mock_memory.get_conversation.return_value = []
    mock_memory.add_message_to_memory = AsyncMock()

    aml_online_chat._memory = mock_memory

    response = MagicMock()
    response.status_code = 429
    mock_complete_chat_async = AsyncMock(
        side_effect=HTTPStatusError(message="Rate Limit Reached", request=MagicMock(), response=response)
    )
    aml_online_chat._complete_chat_async = mock_complete_chat_async
    message = Message(message_pieces=[MessagePiece(role="user", conversation_id="123", original_value="Hello")])

    with pytest.raises(RateLimitException) as rle:
        await aml_online_chat.send_prompt_async(message=message)
        aml_online_chat._memory.get_conversation.assert_called_once_with(conversation_id="123")
        aml_online_chat._memory.add_message_to_memory.assert_called_once_with(request=message)

    assert str(rle.value) == "Status Code: 429, Message: Rate Limit Exception"


async def test_send_prompt_async_rate_limit_exception_retries(aml_online_chat: AzureMLChatTarget):
    response = MagicMock()
    response.status_code = 429
    mock_complete_chat_async = AsyncMock(
        side_effect=RateLimitError("Rate Limit Reached", response=response, body="Rate limit reached")
    )
    aml_online_chat._complete_chat_async = mock_complete_chat_async
    message = Message(message_pieces=[MessagePiece(role="user", conversation_id="12345", original_value="Hello")])

    with pytest.raises(RateLimitError):
        await aml_online_chat.send_prompt_async(message=message)
        # RETRY_MAX_NUM_ATTEMPTS is set to 2 in conftest.py
        assert mock_complete_chat_async.call_count == 2


async def test_send_prompt_async_empty_response_retries(aml_online_chat: AzureMLChatTarget):
    response = MagicMock()
    response.status_code = 429
    # Return an empty dict; _materialize_response raises EmptyResponseException
    # when both output and tool_calls are missing.
    mock_complete_chat_async = AsyncMock()
    mock_complete_chat_async.return_value = {}

    aml_online_chat._complete_chat_async = mock_complete_chat_async
    message = Message(message_pieces=[MessagePiece(role="user", conversation_id="12345", original_value="Hello")])

    with pytest.raises(EmptyResponseException):
        await aml_online_chat.send_prompt_async(message=message)
        # RETRY_MAX_NUM_ATTEMPTS is set to 2 in conftest.py
        assert mock_complete_chat_async.call_count == 2


def test_invalid_temperature_too_low_raises(patch_central_database):
    with pytest.raises(Exception, match="temperature must be between 0 and 2"):
        AzureMLChatTarget(
            endpoint="http://aml-test-endpoint.com",
            api_key="valid_api_key",
            temperature=-0.1,
        )


def test_invalid_temperature_too_high_raises(patch_central_database):
    with pytest.raises(Exception, match="temperature must be between 0 and 2"):
        AzureMLChatTarget(
            endpoint="http://aml-test-endpoint.com",
            api_key="valid_api_key",
            temperature=2.1,
        )


def test_invalid_top_p_too_low_raises(patch_central_database):
    with pytest.raises(Exception, match="top_p must be between 0 and 1"):
        AzureMLChatTarget(
            endpoint="http://aml-test-endpoint.com",
            api_key="valid_api_key",
            top_p=-0.1,
        )


def test_invalid_top_p_too_high_raises(patch_central_database):
    with pytest.raises(Exception, match="top_p must be between 0 and 1"):
        AzureMLChatTarget(
            endpoint="http://aml-test-endpoint.com",
            api_key="valid_api_key",
            top_p=1.1,
        )


def test_valid_temperature_and_top_p(patch_central_database):
    # Should not raise any exceptions
    target = AzureMLChatTarget(
        endpoint="http://aml-test-endpoint.com",
        api_key="valid_api_key",
        temperature=1.5,
        top_p=0.9,
    )
    assert target._temperature == 1.5
    assert target._top_p == 0.9


# ---------------------------------------------------------------------------
# Tool calling: tool_parser + tool_backend kwargs
# ---------------------------------------------------------------------------


@pytest.fixture
def echo_backend():
    from pyrit.tools import LocalToolBackend

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


def test_tool_parser_kwarg_flips_supports_tool_use_capability(patch_central_database):
    from pyrit.prompt_target.common.target_capabilities import CapabilityName
    from pyrit.tools import CanonicalEnvelopeParser

    target = AzureMLChatTarget(
        endpoint="http://aml-test-endpoint.com",
        api_key="k",
        tool_parser=CanonicalEnvelopeParser(),
    )
    assert target.configuration.includes(capability=CapabilityName.TOOL_USE)
    assert target._tool_parser is not None


def test_no_tool_parser_leaves_supports_tool_use_off(aml_online_chat: AzureMLChatTarget):
    from pyrit.prompt_target.common.target_capabilities import CapabilityName

    assert not aml_online_chat.configuration.includes(capability=CapabilityName.TOOL_USE)
    assert aml_online_chat._tool_parser is None


def test_tool_backend_kwarg_installed_into_configuration(patch_central_database, echo_backend):
    from pyrit.tools import CanonicalEnvelopeParser

    target = AzureMLChatTarget(
        endpoint="http://aml-test-endpoint.com",
        api_key="k",
        tool_parser=CanonicalEnvelopeParser(),
        tool_backend=echo_backend,
    )
    assert target.configuration.tool_backend is echo_backend


def test_tool_backend_kwarg_without_parser_raises(patch_central_database, echo_backend):
    # Without tool_parser, the default configuration has supports_tool_use=False,
    # so attaching a backend must raise.
    with pytest.raises(ValueError, match="supports_tool_use"):
        AzureMLChatTarget(
            endpoint="http://aml-test-endpoint.com",
            api_key="k",
            tool_backend=echo_backend,
        )


def test_tool_schemas_wraps_backend_schemas_in_chat_completions_shape(patch_central_database, echo_backend):
    from pyrit.tools import CanonicalEnvelopeParser

    target = AzureMLChatTarget(
        endpoint="http://aml-test-endpoint.com",
        api_key="k",
        tool_parser=CanonicalEnvelopeParser(),
        tool_backend=echo_backend,
    )
    schemas = target._tool_schemas()
    assert len(schemas) == 1
    assert schemas[0]["type"] == "function"
    assert schemas[0]["function"]["name"] == "echo"


def test_tool_schemas_empty_when_no_backend(aml_online_chat: AzureMLChatTarget):
    assert aml_online_chat._tool_schemas() == []


async def test_request_body_omits_tools_key_when_no_backend(aml_online_chat: AzureMLChatTarget):
    messages = [Message(message_pieces=[MessagePiece(role="user", original_value="hi")])]
    body = await aml_online_chat._construct_http_body_async(messages)
    assert "tools" not in body


async def test_request_body_includes_tools_when_backend_set(patch_central_database, echo_backend):
    from pyrit.tools import CanonicalEnvelopeParser

    target = AzureMLChatTarget(
        endpoint="http://aml-test-endpoint.com",
        api_key="k",
        tool_parser=CanonicalEnvelopeParser(),
        tool_backend=echo_backend,
    )
    messages = [Message(message_pieces=[MessagePiece(role="user", original_value="hi")])]
    body = await target._construct_http_body_async(messages)
    assert "tools" in body
    assert body["tools"][0]["function"]["name"] == "echo"


async def test_materialize_response_handles_text_and_tool_calls(patch_central_database, echo_backend):
    from pyrit.tools import CanonicalEnvelopeParser

    target = AzureMLChatTarget(
        endpoint="http://aml-test-endpoint.com",
        api_key="k",
        tool_parser=CanonicalEnvelopeParser(),
        tool_backend=echo_backend,
    )
    request = MessagePiece(role="user", original_value="hi", conversation_id="abc")
    response = {
        "output": "ok",
        "tool_calls": [
            {
                "type": "function_call",
                "call_id": "call_0",
                "name": "echo",
                "arguments": '{"text":"hi"}',
            }
        ],
    }
    msg = target._materialize_response(response=response, request=request)
    types = [p.original_value_data_type for p in msg.message_pieces]
    assert types == ["text", "function_call"]
