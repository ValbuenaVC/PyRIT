# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Generic tool-use scaffolding for ``PromptTarget``.

This package provides a transport-agnostic tool-calling loop. The
``tool_loop`` decorator, when applied to ``send_prompt_async``, runs
the standard PyRIT validate+normalize work once and then repeatedly
re-enters the target's protected ``_send_prompt_to_target_async`` until
the model issues a stop response (or a configured limit is hit).

A target opts in by declaring two collaborators:

* ``self._tool_parser`` — a ``ToolCallParser`` that walks a
  response message and extracts pending ``ToolCall`` instances.
* ``self.configuration.tool_event_policy`` — a ``ToolEventPolicy``
  whose ``ToolEventBehavior`` decides whether to ``EXECUTE``,
  ``RAISE``, or ``RETURN_RAW`` on each detected call.

When the policy is ``EXECUTE``, calls are dispatched through
``self.configuration.tool_backend``, an implementation of
``ToolBackend``. ``LocalToolBackend`` is the in-process backend;
``MCPToolBackend`` proxies through one or more MCP servers.

The ``ToolBackend`` abstract base is intentionally distinct from
``pyrit.registry`` — that namespace is reserved for framework-level
identity registries (``TargetRegistry``, ``ScorerRegistry``) that
register named singletons for CLI lookup, which a per-target tool
dispatch table is not.

``@tool_loop`` is wired onto ``PromptTarget.send_prompt_async`` from
the base class, and the ``tool_event_policy`` / ``tool_backend``
fields hang off ``TargetConfiguration``.

The two exception types the loop raises
(``ToolCallNotSupported`` and
``ToolCallLoopLimitExceeded``) live in
``pyrit.exceptions`` alongside the rest of PyRIT's exception
catalog, so non-tools callers (attacks, normalizers) can import them
without taking a subsystem-level dependency on ``pyrit.tools``.
"""

from pyrit.tools.backend import ToolBackend
from pyrit.tools.inline_parser import InlineToolCallParser, InlineToolCallParserMode
from pyrit.tools.local_backend import LocalToolBackend
from pyrit.tools.mcp_backend import MCPToolBackend
from pyrit.tools.mcp_client import (
    DockerMCPServerSpec,
    LocalMCPServerSpec,
    MCPClient,
    MCPServerSpec,
    RemoteMCPServerSpec,
)
from pyrit.tools.models import ToolCall, ToolEventBehavior, ToolEventPolicy, tool_loop
from pyrit.tools.parsers import CanonicalEnvelopeParser, ToolCallParser

__all__ = [
    "CanonicalEnvelopeParser",
    "DockerMCPServerSpec",
    "InlineToolCallParser",
    "InlineToolCallParserMode",
    "LocalMCPServerSpec",
    "LocalToolBackend",
    "MCPClient",
    "MCPServerSpec",
    "MCPToolBackend",
    "RemoteMCPServerSpec",
    "ToolBackend",
    "ToolCall",
    "ToolCallParser",
    "ToolEventBehavior",
    "ToolEventPolicy",
    "tool_loop",
]
