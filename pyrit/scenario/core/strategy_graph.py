# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
``StrategyGraph`` — the orchestrator side of the scenario state machine.

A ``StrategyGraph`` owns a ``policy: dict[state, action]`` mapping and walks
through states by invoking the action bound to the current state. Each action
returns a ``(next_state, ScenarioStepResult | None)`` pair; the graph yields
the result and advances. Terminal states stop the loop.

This module also exposes ``linear_strategy_policy``, a convenience builder
that produces a trivial "run steps 0..N-1 in order" policy. Phase 5 will use
it to silently upgrade scenarios that still declare their steps as a flat
list (via the legacy ``_get_atomic_attacks_async`` override) without forcing
those scenarios to author a custom policy.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Generic, TypeVar

if TYPE_CHECKING:
    from pyrit.scenario.core.scenario_step import ScenarioStep, ScenarioStepResult

logger = logging.getLogger(__name__)

StepT = TypeVar("StepT", bound="ScenarioStep")
StateT = TypeVar("StateT", bound=Hashable)

#: A policy action receives the graph and returns the next state plus the
#: result produced (or ``None`` if the action did no observable work).
PolicyAction = Callable[
    ["StrategyGraph[StepT, StateT]"],
    Awaitable[tuple[StateT, "ScenarioStepResult | None"]],
]


@dataclass(frozen=True)
class StrategyPolicy(Generic[StepT, StateT]):
    """
    Frozen declaration of how a ``StrategyGraph`` traverses its states.

    Mirrors the codebase's other policy wrappers (``CapabilityHandlingPolicy``,
    ``TargetRequirements``): a typed wrapper around a mapping plus the
    structural invariants that mapping must satisfy. Bundling the action
    map, ``initial_state``, and ``terminal_states`` into one frozen value
    means the policy can be constructed once at scenario class definition
    time and shared across runs without risk of mutation.

    The action mapping is defensively copied into a ``MappingProxyType``
    in ``__post_init__`` so the policy is genuinely read-only.

    Attributes:
        actions (Mapping[StateT, PolicyAction[StepT, StateT]]): Per-state
            async callable that produces the next state plus an optional
            ``ScenarioStepResult``.
        initial_state (StateT): The state the graph starts in. Must not be
            in ``terminal_states`` (a graph that starts terminal does no
            work and is almost certainly a bug).
        terminal_states (frozenset[StateT]): States that stop the event
            loop when reached. Must be non-empty. Terminal states must not
            appear in ``actions``.
    """

    actions: Mapping[StateT, PolicyAction[StepT, StateT]]
    initial_state: StateT
    terminal_states: frozenset[StateT] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        """
        Validate the policy structure and freeze the action map.

        Raises:
            ValueError: If ``terminal_states`` is empty, ``initial_state``
                is itself a terminal state, or a terminal state appears as
                a key in ``actions`` (which would never fire).
        """
        if not self.terminal_states:
            raise ValueError("StrategyPolicy requires at least one terminal state.")
        if self.initial_state in self.terminal_states:
            raise ValueError(
                f"initial_state {self.initial_state!r} is in terminal_states; "
                f"the graph would do no work."
            )

        overlap = [state for state in self.actions if state in self.terminal_states]
        if overlap:
            raise ValueError(
                f"Terminal states must not appear in actions: {overlap!r}."
            )

        object.__setattr__(self, "actions", MappingProxyType(dict(self.actions)))
        object.__setattr__(self, "terminal_states", frozenset(self.terminal_states))

    def get_action(self, *, state: StateT) -> PolicyAction[StepT, StateT]:
        """
        Return the action bound to ``state``.

        Args:
            state (StateT): The state to look up.

        Returns:
            PolicyAction[StepT, StateT]: The configured async action.

        Raises:
            KeyError: If ``state`` is non-terminal and has no entry in
                ``actions`` (indicates a malformed policy).
        """
        try:
            return self.actions[state]
        except KeyError:
            known = ", ".join(sorted(str(s) for s in self.actions))
            raise KeyError(
                f"No action defined for state {state!r}. Known states: {known or '(none)'}."
            ) from None

    def is_terminal(self, *, state: StateT) -> bool:
        """Return ``True`` if ``state`` is a terminal state."""
        return state in self.terminal_states


class StrategyGraph(Generic[StepT, StateT]):
    """
    Policy-driven state machine over ``ScenarioStep``s.

    Construct with a frozen ``StrategyPolicy`` describing the action map,
    starting state, and terminal states. ``event_loop_async`` iterates the
    graph: at each state it invokes the bound action, yields any returned
    result, and advances to the next state until a terminal state is
    reached.

    The graph maintains ``current_state``, ``current_step``, and ``history``
    so that retries can resume from the last persisted state without
    replaying completed work.
    """

    def __init__(
        self,
        *,
        policy: StrategyPolicy[StepT, StateT],
    ) -> None:
        """
        Initialize a ``StrategyGraph`` from a frozen policy.

        Args:
            policy (StrategyPolicy[StepT, StateT]): Frozen policy describing
                the action map, initial state, and terminal states. All
                structural validation lives on ``StrategyPolicy``; the graph
                trusts the policy it receives.
        """
        self._policy = policy
        self._current_state: StateT = policy.initial_state
        self._current_step: StepT | None = None
        self._history: list[tuple[StateT, ScenarioStepResult]] = []

    @property
    def policy(self) -> StrategyPolicy[StepT, StateT]:
        """Return the frozen policy that drives this graph."""
        return self._policy

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
        return self._policy.is_terminal(state=self._current_state)

    def bind_current_step(self, *, step: StepT | None) -> None:
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
        Reset the graph back to ``policy.initial_state`` and clear history.

        Used by retry paths that want a clean slate rather than resuming
        from the last persisted state.
        """
        self._current_state = self._policy.initial_state
        self._current_step = None
        self._history = []

    async def event_loop_async(self) -> AsyncIterator[ScenarioStepResult]:
        """
        Walk the policy graph, yielding each non-null ``ScenarioStepResult``.

        Restartable: callers may resume from the current state after an
        exception or external interruption. Each iteration:

        1. Looks up the action for ``current_state`` via the policy.
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
            action = self._policy.get_action(state=state_before)

            next_state, result = await action(self)
            if result is not None:
                self._history.append((state_before, result))
                yield result

            self._current_state = next_state


def linear_strategy_policy(
    steps: Sequence[ScenarioStep],
) -> StrategyPolicy[ScenarioStep, int]:
    """
    Build a trivial linear-traversal policy over an ordered list of steps.

    State ``i`` binds ``steps[i]`` as the graph's current step, awaits its
    ``process_async``, and transitions to state ``i + 1``. State ``len(steps)``
    is the sole terminal state.

    Args:
        steps (Sequence[ScenarioStep]): Steps to execute in order. Must be
            non-empty.

    Returns:
        StrategyPolicy[ScenarioStep, int]: A policy that walks the steps
        sequentially. Pass it to ``StrategyGraph(policy=...)`` to get the
        runnable graph.

    Raises:
        ValueError: If ``steps`` is empty.
    """
    if not steps:
        raise ValueError("linear_strategy_policy requires at least one step.")

    terminal = len(steps)
    actions: dict[int, PolicyAction[ScenarioStep, int]] = {}

    for index, step in enumerate(steps):

        async def _action(
            graph: StrategyGraph[ScenarioStep, int],
            _step: ScenarioStep = step,
            _next: int = index + 1,
        ) -> tuple[int, ScenarioStepResult | None]:
            graph.bind_current_step(step=_step)
            try:
                result = await _step.process_async()
            finally:
                graph.bind_current_step(step=None)
            return _next, result

        actions[index] = _action

    return StrategyPolicy(
        actions=actions,
        initial_state=0,
        terminal_states=frozenset({terminal}),
    )


__all__ = [
    "StrategyGraph",
    "StrategyPolicy",
    "PolicyAction",
    "linear_strategy_policy",
]
