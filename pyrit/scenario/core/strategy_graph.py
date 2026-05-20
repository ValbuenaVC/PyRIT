# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
``StrategyGraph`` — the orchestrator side of the scenario state machine.

A ``StrategyGraph`` owns a ``policy: dict[state, action]`` mapping and walks
through states by invoking the action bound to the current state. Each action
returns a ``(next_state, ScenarioStepResult | None)`` pair; the graph yields
the result and advances. Terminal states stop the loop.

This module is part of the scenario core refactor scaffold (Phase 0). The
skeleton supports straight-line execution; richer policy composition (graph
validation, cycle detection, parallel branches) lands as Phase 3 needs it.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import TYPE_CHECKING, Generic, TypeVar

from pyrit.scenario.core.scenario_state import ScenarioCoreState, ScenarioStateLike

if TYPE_CHECKING:
    from pyrit.scenario.core.scenario_step import ScenarioStep, ScenarioStepResult

logger = logging.getLogger(__name__)

StepT = TypeVar("StepT", bound="ScenarioStep")
StateT = TypeVar("StateT", bound=ScenarioStateLike)

#: A policy action receives the graph and returns the next state plus the
#: result produced (or ``None`` if the action did no observable work).
PolicyAction = Callable[
    ["StrategyGraph[StepT, StateT]"],
    Awaitable[tuple[StateT, "ScenarioStepResult | None"]],
]


class StrategyGraph(Generic[StepT, StateT]):
    """
    Policy-driven state machine over ``ScenarioStep``s.

    Construct with a ``policy`` dict mapping each non-terminal state to an
    async callable that returns the next state and (optionally) the step
    result produced. ``event_loop_async`` iterates the graph: at each state
    it invokes the bound action, yields any returned result, and advances to
    the next state until a terminal state is reached.

    The graph maintains ``current_state``, ``current_step``, and ``history``
    so that retries can resume from the last persisted state without
    replaying completed work.
    """

    def __init__(
        self,
        *,
        policy: dict[StateT, PolicyAction[StepT, StateT]],
        initial_state: StateT,
        terminal_states: set[StateT],
    ) -> None:
        """
        Initialize a ``StrategyGraph``.

        Args:
            policy (dict[StateT, PolicyAction[StepT, StateT]]): Mapping from
                each non-terminal state to the async action that fires while
                in that state.
            initial_state (StateT): Starting state. Must not be in
                ``terminal_states`` (a graph that starts in a terminal state
                does no work and is almost certainly a bug).
            terminal_states (set[StateT]): States that stop ``event_loop_async``
                when reached. Must be non-empty.

        Raises:
            ValueError: If ``terminal_states`` is empty, ``initial_state``
                is terminal, or a non-terminal state lacks a policy entry.
        """
        if not terminal_states:
            raise ValueError("StrategyGraph requires at least one terminal state.")
        if initial_state in terminal_states:
            raise ValueError(
                f"initial_state {initial_state!r} is in terminal_states; the graph would do no work."
            )

        missing_policy = [state for state in policy if state in terminal_states]
        if missing_policy:
            raise ValueError(
                f"Terminal states must not appear in policy: {missing_policy!r}."
            )

        self._policy = dict(policy)
        self._initial_state = initial_state
        self._terminal_states = set(terminal_states)
        self._current_state: StateT = initial_state
        self._current_step: StepT | None = None
        self._history: list[tuple[StateT, ScenarioStepResult]] = []

    @property
    def current_state(self) -> StateT:
        """Return the graph's current state."""
        return self._current_state

    @property
    def current_step(self) -> StepT | None:
        """Return the step bound to the current state, if the action set one."""
        return self._current_step

    @property
    def history(self) -> list[tuple[StateT, ScenarioStepResult]]:
        """Return the ordered history of ``(state, result)`` pairs produced so far."""
        return list(self._history)

    @property
    def is_terminal(self) -> bool:
        """Return ``True`` if the graph is in a terminal state."""
        return self._current_state in self._terminal_states

    def bind_current_step(self, step: StepT | None) -> None:
        """
        Set the step bound to the current state.

        Policy actions call this so external observers (e.g., the surrounding
        ``Scenario``) can read ``graph.current_step`` while the action runs.

        Args:
            step (StepT | None): The step the action is about to execute, or
                ``None`` to clear.
        """
        self._current_step = step

    def reset(self) -> None:
        """
        Reset the graph back to ``initial_state`` and clear history.

        Used by retry paths that want a clean slate rather than resuming
        from the last persisted state.
        """
        self._current_state = self._initial_state
        self._current_step = None
        self._history = []

    async def event_loop_async(self) -> AsyncIterator[ScenarioStepResult]:
        """
        Walk the policy graph, yielding each non-null ``ScenarioStepResult``.

        Restartable: callers may resume from the current state after an
        exception or external interruption. Each iteration:

        1. Looks up the action for ``current_state``.
        2. Awaits the action to receive ``(next_state, result)``.
        3. Appends ``(state_before, result)`` to history when ``result`` is
           non-null.
        4. Advances ``current_state`` to ``next_state``.
        5. Stops when ``current_state`` becomes terminal.

        Yields:
            ScenarioStepResult: Each non-null result produced by a policy
                action, in execution order.

        Raises:
            KeyError: If the graph reaches a non-terminal state with no
                policy entry (indicates malformed policy).
        """
        while not self.is_terminal:
            state_before = self._current_state
            action = self._policy.get(state_before)
            if action is None:
                raise KeyError(
                    f"StrategyGraph reached non-terminal state {state_before!r} "
                    f"with no policy entry."
                )

            next_state, result = await action(self)
            if result is not None:
                self._history.append((state_before, result))
                yield result

            self._current_state = next_state


__all__ = [
    "StrategyGraph",
    "PolicyAction",
    "ScenarioCoreState",
]
