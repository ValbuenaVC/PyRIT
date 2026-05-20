# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Scenario state vocabulary for the state-machine refactor.

``ScenarioCoreState`` provides the small set of states every scenario shares:
lifecycle transitions before any work runs, while work is running, and after
work has finished or failed.

Scenarios that need richer states (e.g., ``OPENING_PHASE``, ``ESCALATING``)
declare their own enum that satisfies the ``ScenarioStateLike`` protocol.

This module is part of the scenario core refactor scaffold (Phase 0). It is
not yet wired into ``Scenario.run_async``; ``StrategyGraph`` in Phase 3
consumes it.
"""

from __future__ import annotations

from enum import Enum
from typing import Protocol, runtime_checkable


@runtime_checkable
class ScenarioStateLike(Protocol):
    """
    Structural marker for any enum used as a scenario state.

    Both ``ScenarioCoreState`` and per-scenario state enums satisfy this
    protocol. ``StrategyGraph`` accepts any value that is hashable and has a
    string ``name`` (the standard ``Enum`` contract), so the only constraint
    is that scenarios declare their states as enums.
    """

    name: str
    value: object


class ScenarioCoreState(Enum):
    """
    Lifecycle states shared by every scenario.

    Per-scenario state enums extend this vocabulary by declaring their own
    enum class with additional members.
    """

    #: Scenario object constructed but ``initialize_async`` has not run.
    UNINITIALIZED = "uninitialized"

    #: ``initialize_async`` is running (pre-flight, graph build, state init).
    INITIALIZING = "initializing"

    #: At least one step has started; graph traversal is active.
    EXECUTING = "executing"

    #: All steps finished and the graph reached a terminal accepting state.
    COMPLETE = "complete"

    #: A step raised or the graph reached a terminal rejecting state.
    FAILED = "failed"
