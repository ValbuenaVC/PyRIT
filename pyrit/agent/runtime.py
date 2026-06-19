# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
``InProcessRuntime`` — default in-process ``ToolCallDispatch`` backend.

Executes registered Python callables synchronously within the current process.
Unknown tools and exceptions are captured and returned as structured error dicts
rather than raised, mirroring the tolerant mode of ``OpenAIResponseTarget``'s
``_execute_call_section_async``.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pyrit.agent.tools import Tool, ToolCall, ToolResult

logger = logging.getLogger(__name__)


class InProcessRuntime:
    """
    Default ``ToolCallDispatch`` backend that executes tools in-process.

    Tools are registered at construction time via a list of ``Tool`` objects.
    All errors (unknown tool, malformed arguments, runtime exceptions) are
    captured and returned as structured error dicts so the outer agent loop
    can forward them to the model and potentially recover.

    Args:
        tools: The set of tools available to this runtime.
    """

    def __init__(self, *, tools: list[Tool]) -> None:
        """Initialize InProcessRuntime and register the provided tools by name."""
        self._registry: dict[str, Tool] = {t.name: t for t in tools}

    async def call_async(self, *, tool_call: ToolCall) -> ToolResult:
        """
        Dispatch a tool call and return the result.

        Error shapes (mirrors ``OpenAIResponseTarget._execute_call_section_async``):

        - Missing name → ``{"error": "missing_function_name", ...}``
        - Unknown tool → ``{"error": "function_not_found", "missing_function": ..., "available_functions": [...]}``
        - Malformed JSON arguments → ``{"error": "malformed_arguments", ...}``
        - Runtime exception → ``{"error": "tool_execution_error", "message": ..., "function": ...}``

        Args:
            tool_call: A dict with ``name`` and ``arguments`` (JSON string).

        Returns:
            The tool's return dict, or a structured error dict.
        """
        name: str | None = tool_call.get("name")
        if not name:
            return {
                "error": "missing_function_name",
                "tool_call_section": tool_call,
            }

        args_json: str = tool_call.get("arguments", "{}")
        try:
            args: dict[str, Any] = json.loads(args_json)
        except Exception:
            logger.warning("Malformed arguments for tool '%s': %s", name, args_json)
            return {
                "error": "malformed_arguments",
                "function": name,
                "raw_arguments": args_json,
            }

        tool = self._registry.get(name)
        if tool is None:
            available = sorted(self._registry.keys())
            logger.warning("Tool '%s' not registered. Available: %s", name, available)
            return {
                "error": "function_not_found",
                "missing_function": name,
                "available_functions": available,
            }

        try:
            return await tool.callable(args)
        except Exception as exc:
            logger.warning("Tool '%s' raised an exception: %s", name, exc)
            return {
                "error": "tool_execution_error",
                "function": name,
                "message": str(exc),
            }
