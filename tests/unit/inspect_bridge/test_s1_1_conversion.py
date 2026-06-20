# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Tests for S1.1: Message conversion helpers (pyrit/inspect_bridge/conversion.py).

Pure / deterministic — no memory, no network.
"""

from __future__ import annotations

import pytest

from pyrit.models import Message, MessagePiece

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_text_piece(*, role: str, text: str, conversation_id: str = "conv-1", sequence: int = 0) -> MessagePiece:
    return MessagePiece(
        role=role,
        original_value=text,
        original_value_data_type="text",
        conversation_id=conversation_id,
        sequence=sequence,
    )


def _make_message(*pieces: MessagePiece) -> Message:
    return Message(message_pieces=list(pieces))


# ---------------------------------------------------------------------------
# to_inspect_messages
# ---------------------------------------------------------------------------


def test_to_inspect_messages_user_text() -> None:
    """A single user text piece converts to a ChatMessageUser."""
    from inspect_ai.model import ChatMessageUser

    from pyrit.inspect_bridge.conversion import to_inspect_messages

    piece = _make_text_piece(role="user", text="Hello world")
    msg = _make_message(piece)
    result = to_inspect_messages(messages=[msg])
    assert len(result) == 1
    cm = result[0]
    assert isinstance(cm, ChatMessageUser)
    assert cm.content == "Hello world"


def test_to_inspect_messages_assistant_text() -> None:
    """A single assistant text piece converts to a ChatMessageAssistant."""
    from inspect_ai.model import ChatMessageAssistant

    from pyrit.inspect_bridge.conversion import to_inspect_messages

    piece = _make_text_piece(role="assistant", text="Hi there")
    msg = _make_message(piece)
    result = to_inspect_messages(messages=[msg])
    assert len(result) == 1
    assert isinstance(result[0], ChatMessageAssistant)
    assert result[0].content == "Hi there"


def test_to_inspect_messages_system_text() -> None:
    """A single system text piece converts to a ChatMessageSystem."""
    from inspect_ai.model import ChatMessageSystem

    from pyrit.inspect_bridge.conversion import to_inspect_messages

    piece = _make_text_piece(role="system", text="You are a helpful assistant.")
    msg = _make_message(piece)
    result = to_inspect_messages(messages=[msg])
    assert len(result) == 1
    assert isinstance(result[0], ChatMessageSystem)
    assert result[0].content == "You are a helpful assistant."


def test_to_inspect_messages_multi_turn() -> None:
    """Multiple messages in order produce multiple ChatMessage objects in order."""
    from inspect_ai.model import ChatMessageAssistant, ChatMessageUser

    from pyrit.inspect_bridge.conversion import to_inspect_messages

    user_piece = _make_text_piece(role="user", text="Question?")
    asst_piece = _make_text_piece(role="assistant", text="Answer!")
    messages = [_make_message(user_piece), _make_message(asst_piece)]
    result = to_inspect_messages(messages=messages)
    assert len(result) == 2
    assert isinstance(result[0], ChatMessageUser)
    assert isinstance(result[1], ChatMessageAssistant)


def test_to_inspect_messages_multi_piece_message() -> None:
    """A Message with multiple pieces of the same role produces multiple ChatMessages."""
    from pyrit.inspect_bridge.conversion import to_inspect_messages

    p1 = _make_text_piece(role="user", text="First part")
    p2 = _make_text_piece(role="user", text="Second part")
    msg = _make_message(p1, p2)
    result = to_inspect_messages(messages=[msg])
    assert len(result) == 2


def test_to_inspect_messages_unknown_role_raises() -> None:
    """An unrecognised PyRIT role (simulated_assistant) maps without error (uses assistant)."""
    from pyrit.inspect_bridge.conversion import to_inspect_messages

    piece = _make_text_piece(role="simulated_assistant", text="Hi")
    msg = _make_message(piece)
    # simulated_assistant should map to assistant without raising
    result = to_inspect_messages(messages=[msg])
    from inspect_ai.model import ChatMessageAssistant

    assert isinstance(result[0], ChatMessageAssistant)


# ---------------------------------------------------------------------------
# to_model_output
# ---------------------------------------------------------------------------


def test_to_model_output_single_text() -> None:
    """A single assistant text message produces a ModelOutput with one choice."""
    from inspect_ai.model import ModelOutput

    from pyrit.inspect_bridge.conversion import to_model_output

    piece = _make_text_piece(role="assistant", text="Response text")
    msg = _make_message(piece)
    output = to_model_output(messages=[msg])
    assert isinstance(output, ModelOutput)
    assert len(output.choices) >= 1
    # The first choice's message should contain the response text
    choice_content = output.choices[0].message.content
    if isinstance(choice_content, str):
        assert "Response text" in choice_content
    else:
        # List of content parts
        assert any("Response text" in str(part) for part in choice_content)


def test_to_model_output_multi_message_uses_last_assistant() -> None:
    """Multi-turn input: the last message content is used for the ModelOutput."""
    from inspect_ai.model import ModelOutput

    from pyrit.inspect_bridge.conversion import to_model_output

    user_piece = _make_text_piece(role="user", text="Question?")
    asst_piece = _make_text_piece(role="assistant", text="Final answer")
    messages = [_make_message(user_piece), _make_message(asst_piece)]
    output = to_model_output(messages=messages)
    assert isinstance(output, ModelOutput)
    choice_content = output.choices[0].message.content
    if isinstance(choice_content, str):
        assert "Final answer" in choice_content
    else:
        assert any("Final answer" in str(p) for p in choice_content)


def test_to_model_output_empty_messages_raises() -> None:
    """An empty message list should raise ValueError."""
    from pyrit.inspect_bridge.conversion import to_model_output

    with pytest.raises(ValueError):
        to_model_output(messages=[])


# ---------------------------------------------------------------------------
# to_pyrit_message_pieces
# ---------------------------------------------------------------------------


def test_to_pyrit_message_pieces_user_message() -> None:
    """A ChatMessageUser event converts to a user MessagePiece."""
    from inspect_ai.model import ChatMessageUser

    from pyrit.inspect_bridge.conversion import to_pyrit_message_pieces

    chat_msg = ChatMessageUser(content="Hello from Inspect")
    pieces = to_pyrit_message_pieces(
        messages=[chat_msg],
        conversation_id="conv-test",
        sequence_start=0,
    )
    assert len(pieces) >= 1
    assert pieces[0].role == "user"
    assert "Hello from Inspect" in pieces[0].original_value


def test_to_pyrit_message_pieces_assistant_message() -> None:
    """A ChatMessageAssistant event converts to an assistant MessagePiece."""
    from inspect_ai.model import ChatMessageAssistant

    from pyrit.inspect_bridge.conversion import to_pyrit_message_pieces

    chat_msg = ChatMessageAssistant(content="Hello response")
    pieces = to_pyrit_message_pieces(
        messages=[chat_msg],
        conversation_id="conv-test",
        sequence_start=0,
    )
    assert len(pieces) >= 1
    assert pieces[0].role == "assistant"


def test_to_pyrit_message_pieces_conversation_id_propagated() -> None:
    """The conversation_id is set on every returned piece."""
    from inspect_ai.model import ChatMessageUser

    from pyrit.inspect_bridge.conversion import to_pyrit_message_pieces

    chat_msg = ChatMessageUser(content="Hi")
    pieces = to_pyrit_message_pieces(
        messages=[chat_msg],
        conversation_id="my-conv-id",
        sequence_start=0,
    )
    assert all(p.conversation_id == "my-conv-id" for p in pieces)


def test_to_pyrit_message_pieces_sequence_increments() -> None:
    """sequence increases across multiple messages."""
    from inspect_ai.model import ChatMessageAssistant, ChatMessageUser

    from pyrit.inspect_bridge.conversion import to_pyrit_message_pieces

    msgs = [ChatMessageUser(content="Q"), ChatMessageAssistant(content="A")]
    pieces = to_pyrit_message_pieces(messages=msgs, conversation_id="c", sequence_start=0)
    assert len(pieces) == 2
    assert pieces[0].sequence < pieces[1].sequence
