# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Tests for the R3 ``StrategyGraph.active_steps`` state-split.

The R3 refactor widens ``StrategyGraph``'s singular ``current_step`` cursor
into a tuple-valued ``active_steps`` field so a future concurrent-dispatch
policy (R4) has a stable, observable surface for fan-out execution. The
legacy ``current_step`` property is preserved as a backward-compat shim and
deprecated for direct use; it emits a ``DeprecationWarning`` only when the
graph has fanned out to more than one step (where collapsing to a single
``StepT | None`` would be ambiguous).
"""

from __future__ import annotations

import warnings
from unittest.mock import MagicMock

import pytest

from pyrit.scenario.core.scenario_step import ScenarioStep
from pyrit.scenario.core.strategy_graph import StrategyGraph, StrategyPolicy


def _build_idle_graph() -> StrategyGraph[ScenarioStep, int]:
    """Build a minimal graph parked at its terminal state so binding tests don't drive the loop."""

    async def _noop(graph: StrategyGraph[ScenarioStep, int]) -> tuple[int, None]:
        return 1, None

    policy: StrategyPolicy[ScenarioStep, int] = StrategyPolicy(
        actions={0: _noop},
        initial_state=0,
        terminal_states=frozenset({1}),
    )
    return StrategyGraph(policy=policy)


class TestActiveStepsDefault:
    """The graph starts idle with an empty active_steps tuple."""

    def test_initial_active_steps_is_empty_tuple(self) -> None:
        graph = _build_idle_graph()
        assert graph.active_steps == ()
        assert isinstance(graph.active_steps, tuple)

    def test_initial_current_step_is_none(self) -> None:
        graph = _build_idle_graph()
        assert graph.current_step is None


class TestBindActiveStepsSequential:
    """Binding a single step exposes the same value through both surfaces."""

    def test_single_step_visible_via_active_steps(self) -> None:
        graph = _build_idle_graph()
        step = MagicMock(spec=ScenarioStep)
        graph.bind_active_steps(steps=(step,))
        assert graph.active_steps == (step,)

    def test_single_step_visible_via_current_step_without_warning(self) -> None:
        graph = _build_idle_graph()
        step = MagicMock(spec=ScenarioStep)
        graph.bind_active_steps(steps=(step,))
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            assert graph.current_step is step

    def test_clear_via_empty_tuple(self) -> None:
        graph = _build_idle_graph()
        graph.bind_active_steps(steps=(MagicMock(spec=ScenarioStep),))
        graph.bind_active_steps(steps=())
        assert graph.active_steps == ()
        assert graph.current_step is None


class TestBindCurrentStepShim:
    """The legacy singular binder still works and routes through bind_active_steps."""

    def test_bind_step_sets_active_steps_to_singleton(self) -> None:
        graph = _build_idle_graph()
        step = MagicMock(spec=ScenarioStep)
        graph.bind_current_step(step=step)
        assert graph.active_steps == (step,)
        assert graph.current_step is step

    def test_bind_none_clears_active_steps(self) -> None:
        graph = _build_idle_graph()
        graph.bind_current_step(step=MagicMock(spec=ScenarioStep))
        graph.bind_current_step(step=None)
        assert graph.active_steps == ()
        assert graph.current_step is None


class TestConcurrentBinding:
    """Multi-step binding is the R3 contract for R4's concurrent dispatch."""

    def test_multiple_steps_visible_via_active_steps(self) -> None:
        graph = _build_idle_graph()
        steps = (MagicMock(spec=ScenarioStep), MagicMock(spec=ScenarioStep), MagicMock(spec=ScenarioStep))
        graph.bind_active_steps(steps=steps)
        assert graph.active_steps == steps

    def test_current_step_warns_on_concurrent_dispatch(self) -> None:
        graph = _build_idle_graph()
        steps = (MagicMock(spec=ScenarioStep), MagicMock(spec=ScenarioStep))
        graph.bind_active_steps(steps=steps)
        with pytest.warns(DeprecationWarning, match=r"current_step"):
            first = graph.current_step
        assert first is steps[0]


class TestResetClearsActiveSteps:
    """reset() returns the graph to its idle state, including active_steps."""

    def test_reset_empties_active_steps(self) -> None:
        graph = _build_idle_graph()
        graph.bind_active_steps(steps=(MagicMock(spec=ScenarioStep), MagicMock(spec=ScenarioStep)))
        graph.reset()
        assert graph.active_steps == ()
        assert graph.current_step is None
