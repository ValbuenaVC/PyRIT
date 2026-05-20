# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Tests for the ``linear_strategy_policy`` convenience builder.

These tests double as the silent-upgrade contract for Phase 5: the legacy
flat ``_get_atomic_attacks_async`` -> ``list[AtomicAttack]`` contract becomes
a ``StrategyGraph`` driven by ``linear_strategy_policy``. If anything here
breaks, the silent-upgrade path is broken.
"""

from typing import Any

import pytest

from pyrit.scenario.core.scenario_step import ScenarioStep, ScenarioStepResult
from pyrit.scenario.core.strategy_graph import StrategyGraph, linear_strategy_policy


class _RecordingStep(ScenarioStep):
    """Minimal ``ScenarioStep`` that records each invocation."""

    def __init__(self, *, name: str, outcome: str = "done"):
        self._name = name
        self._outcome = outcome
        self.call_count = 0

    @property
    def name(self) -> str:  # type: ignore[override]
        return self._name

    @property
    def outputs(self) -> list[str]:  # type: ignore[override]
        return [self._outcome]

    async def process_async(self) -> ScenarioStepResult:
        self.call_count += 1
        return ScenarioStepResult(
            outcome=self._outcome,
            metadata={"name": self._name, "call": self.call_count},
        )


def test_empty_steps_raises():
    with pytest.raises(ValueError, match="at least one step"):
        linear_strategy_policy([])


def test_single_step_policy_has_one_terminal_state():
    step = _RecordingStep(name="solo")
    policy = linear_strategy_policy([step])

    assert policy.initial_state == 0
    assert policy.terminal_states == frozenset({1})
    assert policy.is_terminal(state=1)
    assert not policy.is_terminal(state=0)


async def test_linear_policy_executes_steps_in_order():
    steps = [_RecordingStep(name=f"step_{i}") for i in range(3)]
    graph: StrategyGraph[ScenarioStep, int] = StrategyGraph(policy=linear_strategy_policy(steps))

    results = [r async for r in graph.event_loop_async()]

    assert len(results) == 3
    assert [r.metadata["name"] for r in results] == ["step_0", "step_1", "step_2"]
    assert [s.call_count for s in steps] == [1, 1, 1]
    assert graph.current_state == 3
    assert graph.is_terminal


async def test_linear_policy_binds_current_step_during_execution():
    """The action must bind the step before invoking it and clear after."""
    binding_history: list[Any] = []

    class _BindingObserverStep(_RecordingStep):
        def __init__(self, *, name: str, graph_ref: list[Any]):
            super().__init__(name=name)
            self._graph_ref = graph_ref

        async def process_async(self) -> ScenarioStepResult:
            graph = self._graph_ref[0]
            binding_history.append(graph.current_step)
            return await super().process_async()

    graph_ref: list[Any] = [None]
    steps = [_BindingObserverStep(name="a", graph_ref=graph_ref), _BindingObserverStep(name="b", graph_ref=graph_ref)]
    graph: StrategyGraph[ScenarioStep, int] = StrategyGraph(policy=linear_strategy_policy(steps))
    graph_ref[0] = graph

    _ = [r async for r in graph.event_loop_async()]

    # During each step's process_async, current_step should be that step.
    assert binding_history == steps
    # After traversal completes, current_step is cleared.
    assert graph.current_step is None


async def test_linear_policy_clears_current_step_on_failure():
    """If process_async raises, the bound step is still cleared via finally."""

    class _FailingStep(_RecordingStep):
        async def process_async(self) -> ScenarioStepResult:
            raise RuntimeError("boom")

    steps: list[ScenarioStep] = [_FailingStep(name="fails")]
    graph: StrategyGraph[ScenarioStep, int] = StrategyGraph(policy=linear_strategy_policy(steps))

    with pytest.raises(RuntimeError, match="boom"):
        _ = [r async for r in graph.event_loop_async()]

    assert graph.current_step is None
    # Graph stayed at initial state — retry can re-enter.
    assert graph.current_state == 0


async def test_linear_policy_each_action_runs_its_own_step():
    """Late binding bug guard: every action must run its own indexed step."""
    steps = [_RecordingStep(name=f"step_{i}", outcome=f"out_{i}") for i in range(4)]
    graph: StrategyGraph[ScenarioStep, int] = StrategyGraph(policy=linear_strategy_policy(steps))

    results = [r async for r in graph.event_loop_async()]
    assert [r.outcome for r in results] == ["out_0", "out_1", "out_2", "out_3"]
