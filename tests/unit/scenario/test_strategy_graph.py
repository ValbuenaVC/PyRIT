# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Tests for ``pyrit.scenario.core.strategy_graph``."""

from enum import Enum

import pytest

from pyrit.scenario.core.scenario_state import ScenarioCoreState
from pyrit.scenario.core.scenario_step import ScenarioStepResult
from pyrit.scenario.core.strategy_graph import StrategyGraph


class _State(Enum):
    START = "start"
    MIDDLE = "middle"
    END = "end"


def test_init_requires_non_empty_terminal_states():
    with pytest.raises(ValueError, match="at least one terminal state"):
        StrategyGraph(
            policy={},
            initial_state=_State.START,
            terminal_states=set(),
        )


def test_init_rejects_initial_state_in_terminals():
    with pytest.raises(ValueError, match="would do no work"):
        StrategyGraph(
            policy={},
            initial_state=_State.END,
            terminal_states={_State.END},
        )


def test_init_rejects_terminal_state_with_policy_entry():
    async def _action(graph):
        return _State.END, None

    with pytest.raises(ValueError, match="Terminal states must not appear in policy"):
        StrategyGraph(
            policy={_State.END: _action},
            initial_state=_State.START,
            terminal_states={_State.END},
        )


async def test_event_loop_yields_results_in_order():
    async def start_action(graph):
        return _State.MIDDLE, ScenarioStepResult(outcome="from_start")

    async def middle_action(graph):
        return _State.END, ScenarioStepResult(outcome="from_middle")

    graph: StrategyGraph = StrategyGraph(
        policy={_State.START: start_action, _State.MIDDLE: middle_action},
        initial_state=_State.START,
        terminal_states={_State.END},
    )

    results = [r async for r in graph.event_loop_async()]
    assert [r.outcome for r in results] == ["from_start", "from_middle"]


async def test_event_loop_advances_current_state():
    async def start_action(graph):
        return _State.END, ScenarioStepResult(outcome="done")

    graph: StrategyGraph = StrategyGraph(
        policy={_State.START: start_action},
        initial_state=_State.START,
        terminal_states={_State.END},
    )

    assert graph.current_state == _State.START
    _ = [r async for r in graph.event_loop_async()]
    assert graph.current_state == _State.END
    assert graph.is_terminal


async def test_event_loop_skips_yield_when_action_returns_no_result():
    async def silent_action(graph):
        return _State.END, None

    graph: StrategyGraph = StrategyGraph(
        policy={_State.START: silent_action},
        initial_state=_State.START,
        terminal_states={_State.END},
    )

    results = [r async for r in graph.event_loop_async()]
    assert results == []
    assert graph.current_state == _State.END


async def test_event_loop_raises_on_missing_policy_entry():
    async def start_action(graph):
        # Transition to MIDDLE but the graph has no policy entry for MIDDLE.
        return _State.MIDDLE, None

    graph: StrategyGraph = StrategyGraph(
        policy={_State.START: start_action},
        initial_state=_State.START,
        terminal_states={_State.END},
    )

    with pytest.raises(KeyError, match="no policy entry"):
        _ = [r async for r in graph.event_loop_async()]


async def test_history_records_yielded_results_only():
    async def start_action(graph):
        return _State.MIDDLE, ScenarioStepResult(outcome="recorded")

    async def middle_action(graph):
        return _State.END, None

    graph: StrategyGraph = StrategyGraph(
        policy={_State.START: start_action, _State.MIDDLE: middle_action},
        initial_state=_State.START,
        terminal_states={_State.END},
    )

    _ = [r async for r in graph.event_loop_async()]
    assert len(graph.history) == 1
    state_before, result = graph.history[0]
    assert state_before == _State.START
    assert result.outcome == "recorded"


async def test_reset_returns_graph_to_initial_state():
    async def start_action(graph):
        return _State.END, ScenarioStepResult(outcome="done")

    graph: StrategyGraph = StrategyGraph(
        policy={_State.START: start_action},
        initial_state=_State.START,
        terminal_states={_State.END},
    )

    _ = [r async for r in graph.event_loop_async()]
    assert graph.current_state == _State.END
    assert len(graph.history) == 1

    graph.reset()
    assert graph.current_state == _State.START
    assert graph.history == []
    assert graph.current_step is None


def test_bind_current_step_sets_and_clears():
    async def noop_action(graph):
        return _State.END, None

    graph: StrategyGraph = StrategyGraph(
        policy={_State.START: noop_action},
        initial_state=_State.START,
        terminal_states={_State.END},
    )

    graph.bind_current_step(None)
    assert graph.current_step is None

    sentinel = object()
    graph.bind_current_step(sentinel)  # type: ignore[arg-type]
    assert graph.current_step is sentinel


async def test_event_loop_is_restartable_from_current_state():
    """After an exception, the loop can be re-entered from the failed state."""
    call_count = {"n": 0}

    async def flaky_action(graph):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("first call fails")
        return _State.END, ScenarioStepResult(outcome="recovered")

    graph: StrategyGraph = StrategyGraph(
        policy={_State.START: flaky_action},
        initial_state=_State.START,
        terminal_states={_State.END},
    )

    # First attempt blows up; graph stays at START.
    with pytest.raises(RuntimeError, match="first call fails"):
        _ = [r async for r in graph.event_loop_async()]
    assert graph.current_state == _State.START

    # Retry: the loop restarts from current_state and now succeeds.
    results = [r async for r in graph.event_loop_async()]
    assert [r.outcome for r in results] == ["recovered"]
    assert graph.current_state == _State.END


async def test_strategy_graph_works_with_scenario_core_state():
    async def initializing_action(graph):
        return ScenarioCoreState.EXECUTING, None

    async def executing_action(graph):
        return ScenarioCoreState.COMPLETE, ScenarioStepResult(outcome="finished")

    graph: StrategyGraph = StrategyGraph(
        policy={
            ScenarioCoreState.INITIALIZING: initializing_action,
            ScenarioCoreState.EXECUTING: executing_action,
        },
        initial_state=ScenarioCoreState.INITIALIZING,
        terminal_states={ScenarioCoreState.COMPLETE, ScenarioCoreState.FAILED},
    )

    results = [r async for r in graph.event_loop_async()]
    assert [r.outcome for r in results] == ["finished"]
    assert graph.current_state == ScenarioCoreState.COMPLETE
