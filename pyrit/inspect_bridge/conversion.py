# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Message conversion helpers between PyRIT and Inspect AI formats.

All inspect_ai imports are deferred inside functions so that importing this
module never triggers an inspect_ai import.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pyrit.models import Message, MessagePiece

if TYPE_CHECKING:
    from inspect_ai.model import ChatMessage


# Map PyRIT roles to Inspect roles (Inspect uses "user"/"assistant"/"system"/"tool")
_PYRIT_ROLE_TO_INSPECT: dict[str, str] = {
    "user": "user",
    "assistant": "assistant",
    "simulated_assistant": "assistant",
    "system": "system",
    "tool": "tool",
    "developer": "system",
}

# Map Inspect roles to PyRIT roles
_INSPECT_ROLE_TO_PYRIT: dict[str, str] = {
    "user": "user",
    "assistant": "assistant",
    "system": "system",
    "tool": "tool",
}


def to_inspect_messages(*, messages: list[Message]) -> list[ChatMessage]:
    """
    Convert a PyRIT message list to a list of Inspect ChatMessage objects.

    Each ``MessagePiece`` inside a ``Message`` is converted to a separate
    ``ChatMessage``.  Multi-piece messages therefore produce multiple Inspect
    messages (each piece maps to one LLM turn boundary).

    Args:
        messages: Ordered list of PyRIT ``Message`` objects.

    Returns:
        Ordered list of Inspect ``ChatMessage`` objects.
    """
    from inspect_ai.model import ChatMessageAssistant, ChatMessageSystem, ChatMessageTool, ChatMessageUser

    inspect_messages: list[ChatMessage] = []
    for message in messages:
        for piece in message.message_pieces:
            inspect_role = _PYRIT_ROLE_TO_INSPECT.get(piece.role, "user")
            text = piece.original_value or ""
            if inspect_role == "user":
                inspect_messages.append(ChatMessageUser(content=text))
            elif inspect_role == "assistant":
                inspect_messages.append(ChatMessageAssistant(content=text))
            elif inspect_role == "system":
                inspect_messages.append(ChatMessageSystem(content=text))
            elif inspect_role == "tool":
                inspect_messages.append(ChatMessageTool(content=text, tool_call_id=""))
            else:
                inspect_messages.append(ChatMessageUser(content=text))
    return inspect_messages


def to_model_output(*, messages: list[Message]) -> object:
    """
    Convert a list of PyRIT messages to an Inspect ``ModelOutput``.

    The last message in the list is used to populate the single
    ``ChatCompletionChoice`` inside the output.

    Args:
        messages: Non-empty ordered list of PyRIT ``Message`` objects.

    Returns:
        An Inspect ``ModelOutput`` wrapping the final message content.

    Raises:
        ValueError: If ``messages`` is empty.
    """
    if not messages:
        raise ValueError("messages must not be empty")

    from inspect_ai.model import ChatCompletionChoice, ChatMessageAssistant, ModelOutput

    last_message = messages[-1]
    last_piece = last_message.message_pieces[-1] if last_message.message_pieces else None
    text = last_piece.original_value if last_piece else ""

    choice = ChatCompletionChoice(
        message=ChatMessageAssistant(content=text or ""),
        stop_reason="stop",
    )
    return ModelOutput(
        model="pyrit/unknown",
        choices=[choice],
    )


def to_pyrit_message_pieces(
    *,
    messages: list[ChatMessage],
    conversation_id: str,
    sequence_start: int = 0,
) -> list[MessagePiece]:
    """
    Convert a list of Inspect ``ChatMessage`` objects to PyRIT ``MessagePiece`` objects.

    Args:
        messages: Ordered list of Inspect ``ChatMessage`` objects.
        conversation_id: The PyRIT conversation identifier to assign to all pieces.
        sequence_start: The starting sequence number (incremented for each piece).

    Returns:
        Ordered list of ``MessagePiece`` objects.
    """
    pieces: list[MessagePiece] = []
    seq = sequence_start
    for chat_msg in messages:
        role = _INSPECT_ROLE_TO_PYRIT.get(chat_msg.role, "user")
        content = chat_msg.content
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            # List of ContentText / ContentImage parts — join text parts
            text = " ".join(part.text if hasattr(part, "text") else str(part) for part in content)
        else:
            text = str(content)

        pieces.append(
            MessagePiece(
                role=role,
                original_value=text,
                original_value_data_type="text",
                conversation_id=conversation_id,
                sequence=seq,
            )
        )
        seq += 1
    return pieces
