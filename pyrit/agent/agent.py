# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
``Agent`` — a ``PromptTarget`` that augments an inner target with tool calling.

``Agent`` wraps any ``PromptTarget`` and adds a tool-execution loop: when the
inner target's response contains a tool-call piece (``function_call`` or
``tool_call`` data type), ``Agent`` dispatches it via a ``ToolCallDispatch``
backend (default: ``InProcessRuntime``), appends the result as a
``function_call_output`` piece, and forwards the extended conversation back to
the inner target — repeating until no pending tool call remains or the
``max_tool_iterations`` guard fires.

Because ``Agent`` is itself a ``PromptTarget``, attacks and scenarios can use
it anywhere they expect a ``PromptTarget`` — no attack-side changes required.

Tool-loop design mirrors ``OpenAIResponseTarget._send_prompt_to_target_async``
(the inner target's method is called directly, not via the public
``send_prompt_async``, so interim tool turns stay in-context without being
written to memory twice).
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pyrit.agent.tools import Tool, ToolCall, ToolCallDispatch, ToolResult

from pyrit.agent.runtime import InProcessRuntime
from pyrit.models import ComponentIdentifier, Message, MessagePiece
from pyrit.models.target_capabilities import TargetCapabilities, ToolUsageSchema
from pyrit.prompt_target.common.prompt_target import PromptTarget
from pyrit.prompt_target.common.target_configuration import TargetConfiguration

logger = logging.getLogger(__name__)

# Default tool-call data types when no ToolUsageSchema is configured on the inner target.
_DEFAULT_TOOL_CALL_TYPES: frozenset[str] = frozenset({"function_call", "tool_call"})
_DEFAULT_MAX_TOOL_ITERATIONS = 10


class Agent(PromptTarget):
    """
    A ``PromptTarget`` that wraps another target and executes tool calls.

    The agent loop:

    1. Forwards the full conversation to ``_target._send_prompt_to_target_async``
       directly (not via the public ``send_prompt_async``) to keep interim tool
       turns in-context without memory duplication.
    2. Inspects the response for a pending tool-call piece (data type in
       ``tool_usage_schema.tool_call_data_types`` or the default set).
    3. Dispatches the tool call via ``_dispatcher.call_async``.
    4. Appends a ``function_call_output`` piece to the working conversation.
    5. Repeats until no tool call is found or ``max_tool_iterations`` is reached.

    Args:
        target: The inner target that generates responses and may emit tool calls.
        toolset: The set of tools available to this agent.
        dispatcher: The backend that executes tool calls.  Defaults to an
            ``InProcessRuntime`` built from ``toolset``.
        max_tool_iterations: Maximum number of tool-call/result round-trips
            before the loop is forcibly terminated (prevents infinite loops on
            misbehaving models).  Defaults to 10.
        custom_configuration: Optional per-instance capability override.
    """

    _DEFAULT_CONFIGURATION: TargetConfiguration = TargetConfiguration(
        capabilities=TargetCapabilities(
            supports_multi_turn=True,
            supports_multi_message_pieces=True,
            supports_system_prompt=True,
            supports_editable_history=True,
        )
    )

    def __init__(
        self,
        *,
        target: PromptTarget,
        toolset: set[Tool],
        dispatcher: ToolCallDispatch | None = None,
        max_tool_iterations: int = _DEFAULT_MAX_TOOL_ITERATIONS,
        custom_configuration: TargetConfiguration | None = None,
    ) -> None:
        """Initialize Agent with an inner target, toolset, and optional dispatcher."""
        super().__init__(custom_configuration=custom_configuration)
        self._target = target
        self._toolset = toolset
        self._dispatcher: ToolCallDispatch = dispatcher or InProcessRuntime(tools=list(toolset))
        self._max_tool_iterations = max_tool_iterations

    def _build_identifier(self) -> ComponentIdentifier:
        return self._create_identifier(
            params={"max_tool_iterations": self._max_tool_iterations},
            targets=[self._target.get_identifier()],
        )

    def _validate_request(self, *, normalized_conversation: list[Message]) -> None:
        """Pass-through; the inner target validates against its own capabilities."""

    async def _send_prompt_to_target_async(self, *, normalized_conversation: list[Message]) -> list[Message]:
        """
        Execute the tool-calling agentic loop.

        Calls the inner target's ``_send_prompt_to_target_async`` directly to
        keep interim tool turns in-context without touching memory.

        Args:
            normalized_conversation: The full conversation (history + current
                message) after running the normalization pipeline.

        Returns:
            All ``Message`` objects generated during this interaction
            (assistant responses and tool-result messages).
        """
        schema: ToolUsageSchema | None = getattr(self._target.capabilities, "tool_usage_schema", None)
        tool_call_types = schema.tool_call_data_types if schema else _DEFAULT_TOOL_CALL_TYPES
        tool_result_data_type = schema.tool_result_data_type if schema else "function_call_output"
        tool_result_role = schema.tool_result_role if schema else "tool"

        working_conversation: list[Message] = list(normalized_conversation)
        all_responses: list[Message] = []

        # Reference piece for conversation context propagation
        reference_piece = normalized_conversation[-1].message_pieces[0]

        for iteration in range(self._max_tool_iterations + 1):
            logger.debug("Agent loop iteration %d, conversation length=%d", iteration, len(working_conversation))

            responses = await self._target._send_prompt_to_target_async(normalized_conversation=working_conversation)
            working_conversation.extend(responses)
            all_responses.extend(responses)

            # Find the last tool call in the just-received responses
            tool_call = self._find_tool_call(responses=responses, tool_call_types=tool_call_types)

            if tool_call is None:
                # No pending tool call → we're done
                break

            if iteration >= self._max_tool_iterations:
                logger.warning(
                    "Agent reached max_tool_iterations=%d; stopping tool loop.",
                    self._max_tool_iterations,
                )
                break

            # Dispatch the tool call
            tool_result = await self._dispatcher.call_async(tool_call=tool_call)

            # Create a tool-result message and add it to the working conversation
            call_id: str = tool_call.get("call_id", "")
            result_piece = self._make_tool_result_piece(
                result=tool_result,
                call_id=call_id,
                role=tool_result_role,
                data_type=tool_result_data_type,
                reference_piece=reference_piece,
            )
            result_message = Message(message_pieces=[result_piece])
            working_conversation.append(result_message)
            all_responses.append(result_message)

        return all_responses

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _find_tool_call(
        self,
        *,
        responses: list[Message],
        tool_call_types: frozenset[str],
    ) -> ToolCall | None:
        """
        Return the last pending tool-call dict from assistant messages, or ``None``.

        Scans all provided messages in reverse piece order for a piece whose
        ``original_value_data_type`` is in ``tool_call_types``.  Returns the
        first (last in the response) matching parsed JSON dict, or ``None``.

        Args:
            responses: The most recently received messages to scan.
            tool_call_types: Data type strings that identify tool-call pieces.

        Returns:
            The tool-call dict (mirrors ``function_call`` shape), or ``None``.
        """
        for message in reversed(responses):
            for piece in reversed(message.message_pieces):
                if piece.original_value_data_type in tool_call_types:
                    try:
                        section: Any = json.loads(piece.original_value)
                        if isinstance(section, dict):
                            return section
                    except Exception:
                        continue
        return None

    def _make_tool_result_piece(
        self,
        *,
        result: ToolResult,
        call_id: str,
        role: str,
        data_type: str,
        reference_piece: MessagePiece,
    ) -> MessagePiece:
        """
        Build a ``function_call_output`` ``MessagePiece`` from a tool result.

        Args:
            result: The dict returned by the dispatcher.
            call_id: The ``call_id`` from the originating tool call.
            role: The role for the result piece (e.g., ``"tool"``).
            data_type: The ``original_value_data_type`` (e.g.,
                ``"function_call_output"``).
            reference_piece: A piece to copy ``conversation_id`` from.

        Returns:
            A ``MessagePiece`` containing the serialised tool result.
        """
        output_str = result if isinstance(result, str) else json.dumps(result, separators=(",", ":"))
        return MessagePiece(
            role=role,  # type: ignore[arg-type]
            original_value=json.dumps(
                {"type": data_type, "call_id": call_id, "output": output_str},
                separators=(",", ":"),
            ),
            original_value_data_type=data_type,  # type: ignore[arg-type]
            conversation_id=reference_piece.conversation_id,
        )
