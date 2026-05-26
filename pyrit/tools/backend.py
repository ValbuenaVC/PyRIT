# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from pyrit.tools.models import ToolCall


@runtime_checkable
class ToolBackend(Protocol):
    """
    Protocol for backends that dispatch tool calls produced by a target.

    A :class:`ToolBackend` is a per-target dispatch table — it owns the
    ``name -> async callable`` mapping a target uses to execute the tool
    calls extracted from a model response. This is intentionally distinct
    from :mod:`pyrit.registry`, whose ``Registry`` classes register named
    framework singletons (targets, scorers, attacks) for CLI lookup.

    Two concrete implementations ship with PyRIT:

    * :class:`~pyrit.tools.CallableToolBackend` — pure-Python backend
      backed by ``async def`` callables. Useful for unit tests and for
      embedding tools inside the PyRIT process.
    * :class:`pyrit.tools.MCPToolBackend` (lands in C3) — proxies
      dispatch through one or more MCP servers.

    The :attr:`schemas` property exposes the JSON-schema descriptors the
    target injects into its request body (e.g. ``tools=[...]`` for the
    OpenAI APIs).

    :meth:`dispatch_all_sequential_async` is the contract the tool loop
    uses: backends that wish to parallelize dispatch should override it.
    The default sequencing — one ``await dispatch_async`` per call, in
    declaration order — is what every PyRIT backend ships with today.
    """

    @property
    def schemas(self) -> list[dict[str, Any]]:
        """
        The JSON-schema descriptors for every tool the backend exposes.

        Returns:
            list[dict[str, Any]]: One schema per tool, in a target-agnostic
                format that concrete targets serialize into their request
                body.
        """
        ...

    async def dispatch_async(self, call: ToolCall) -> dict[str, Any]:
        """
        Execute a single tool call and return the structured result.

        Implementations MUST NOT raise on tool-side failures; they MUST
        return an error envelope (e.g. ``{"error": "...", "tool": "..."}``)
        so the tool loop can carry the failure back to the model.

        Args:
            call (ToolCall): The tool call to dispatch.

        Returns:
            dict[str, Any]: The structured tool result.
        """
        ...

    async def dispatch_all_sequential_async(
        self,
        calls: list[ToolCall],
    ) -> list[tuple[ToolCall, dict[str, Any]]]:
        """
        Dispatch every call in *calls* sequentially, preserving declaration order.

        Args:
            calls (list[ToolCall]): The calls to dispatch, in declaration order.

        Returns:
            list[tuple[ToolCall, dict[str, Any]]]: ``(call, result)`` pairs,
                in the same order as *calls*.
        """
        ...
