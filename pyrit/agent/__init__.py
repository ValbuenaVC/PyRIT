# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
``pyrit.agent`` — universal tool calling and agent primitive for PyRIT.

This package provides the ``Agent`` class (a ``PromptTarget`` that wraps
another target and executes tool calls), the ``Tool`` model, and the
``InProcessRuntime`` dispatcher.
"""

from pyrit.agent.agent import Agent
from pyrit.agent.runtime import InProcessRuntime
from pyrit.agent.tools import Tool, ToolCall, ToolCallDispatch, ToolResult

__all__ = [
    "Agent",
    "InProcessRuntime",
    "Tool",
    "ToolCall",
    "ToolCallDispatch",
    "ToolResult",
]
