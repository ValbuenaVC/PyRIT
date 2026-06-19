# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Tool primitives for the ``pyrit.agent`` package.

Defines:

- ``Tool`` — a Pydantic model describing a callable tool (name, description,
  JSON schema, callable).
- ``ToolCall`` — type alias for the dict representation of a tool invocation
  (mirrors the ``function_call`` shape used by ``OpenAIResponseTarget``).
- ``ToolResult`` — type alias for a tool's return value dict.
- ``ToolCallDispatch`` — ``Protocol`` satisfied by any object that can
  dispatch a tool call and return a ``ToolResult``.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict

# ---------------------------------------------------------------------------
# Type aliases (mirror the OpenAIResponseTarget function_call shape)
# ---------------------------------------------------------------------------

# A tool call is a dict with at minimum:
#   {"type": "function_call", "call_id": <str>, "name": <str>, "arguments": <str JSON>}
ToolCall = dict[str, Any]

# A tool result is the dict payload returned by the callable and later
# serialised as the ``function_call_output`` piece:
#   {"output": <str | dict>, ...}  (call_id injected by the dispatcher)
ToolResult = dict[str, Any]

# Runtime-level alias; kept at module scope so Pydantic can resolve the field
# annotation for ``Tool.callable`` when ``get_type_hints`` is called at class
# creation time (``from __future__ import annotations`` makes all annotations
# lazy strings, but Pydantic resolves them against the module globals).
ToolCallable = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


# ---------------------------------------------------------------------------
# Tool model
# ---------------------------------------------------------------------------


class Tool(BaseModel):
    """
    A single tool that an ``Agent`` can invoke during its agentic loop.

    Attributes:
        name: Unique identifier for the tool (must match the name the target
            emits in ``function_call.name``).
        description: Human-readable description of what the tool does.
        json_schema: JSON Schema dict describing the tool's input arguments.
        callable: Async callable ``(args: dict) -> dict`` that executes the
            tool and returns a result dict.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    description: str
    json_schema: dict[str, Any]
    callable: ToolCallable

    def __hash__(self) -> int:
        """
        Hash by tool name so ``Tool`` instances can live in sets.

        Returns:
            Integer hash of ``self.name``.
        """
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        """
        Compare tools by name (consistent with ``__hash__``).

        Args:
            other: The object to compare against.

        Returns:
            ``True`` if ``other`` is a ``Tool`` with the same name; ``NotImplemented``
            if ``other`` is not a ``Tool`` instance.
        """
        if not isinstance(other, Tool):
            return NotImplemented
        return self.name == other.name


# ---------------------------------------------------------------------------
# ToolCallDispatch protocol
# ---------------------------------------------------------------------------


class ToolCallDispatch(Protocol):
    """
    Protocol satisfied by any object that can dispatch a ``ToolCall`` to the
    appropriate tool implementation and return a ``ToolResult``.

    Implementors include ``InProcessRuntime`` (default) and future runtimes
    such as Docker or MCP-based dispatchers.
    """

    async def call_async(self, *, tool_call: ToolCall) -> ToolResult:
        """
        Dispatch a tool call and return the result.

        Args:
            tool_call: A dict with at least ``name`` and ``arguments``
                (mirrors the ``function_call`` shape from
                ``OpenAIResponseTarget``).

        Returns:
            A ``ToolResult`` dict; the caller is responsible for wrapping it
            in a ``function_call_output`` ``MessagePiece``.
        """
        ...
