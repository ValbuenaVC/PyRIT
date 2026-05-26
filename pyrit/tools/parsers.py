# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from pyrit.models import Message, MessagePiece
    from pyrit.tools.models import ToolCall


@runtime_checkable
class ToolCallParser(Protocol):
    """
    Protocol for extracting tool calls from a target response message.

    Concrete parsers live next to the target whose response shape they
    understand (see :class:`OpenAIChatTarget` and :class:`OpenAIResponseTarget`
    after C7/C8). Parsers MUST return an empty list when the model has
    issued a stop response — the tool loop uses the empty list as the
    signal to exit.
    """

    def parse(self, message: Message) -> list[ToolCall]:
        """
        Extract tool calls from a target response message.

        Args:
            message (Message): The most recent assistant response.

        Returns:
            list[ToolCall]: Tool calls, in declaration order. An empty list
                signals that the model produced a stop response.
        """
        ...


def _extract_function_call_pieces(message: Message) -> list[MessagePiece]:
    """
    Return every :class:`MessagePiece` in *message* whose
    ``original_value_data_type`` is ``"function_call"``.

    This is the canonical envelope produced by OpenAI-style targets after
    the C6 normalization commit. It is exposed here so concrete parsers
    can reuse the filter rather than re-implementing it.

    Args:
        message (Message): The message to scan.

    Returns:
        list[MessagePiece]: Pieces whose ``original_value_data_type`` is
            ``"function_call"``, in their declaration order.
    """
    return [piece for piece in message.message_pieces if piece.original_value_data_type == "function_call"]
