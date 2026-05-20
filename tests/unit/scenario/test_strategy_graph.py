# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Tests for ``pyrit.scenario.core.strategy_graph``."""

from enum import Enum
from unittest.mock import AsyncMock

import pytest

from pyrit.scenario.core.scenario_state import ScenarioCoreState
from pyrit.scenario.core.scenario_step import ScenarioStepResult
from pyrit.scenario.core.strategy_graph import StrategyGraph, StrategyPolicy


class _State(Enum):
    START = "start"
    MIDDLE = "middle"
    END = "end"


def _make_graph(*, policy: StrategyPolicy) -> StrategyGraph:
    return StrategyGraph(policy=policy)


# ---------------------------------------------------------------------------
# StrategyPolicy construction & validation
# ---------------------------------------------------------------------------


class TestStrategyPolicyInit:
    def test_requires_non_empty_terminal_states(self):
        with pytest.raises(ValueError, match="at least one terminal state"):
            StrategyPolicy(
                actions={},
                initial_state=_State.START,
                terminal_states=frozenset(),
            )

    def test_rejects_initial_state_in_terminals(self):
        with pytest.raises(ValueError, match="would do no work"):
            StrategyPolicy(
                actions={},
                initial_state=_State.END,
                terminal_states=frozenset({_State.END}),
            )

    def test_rejects_terminal_state_with_action_entry(self):
        async def _action(graph):
            return _State.END, None

        with pytest.raises(ValueError, match="Terminal states must not appear"):
            StrategyPolicy(
                actions={_State.END: _action},
                initial_state=_State.START,
                terminal_states=frozenset({_State.END}),
            )

    def test_actions_mapping_is_read_only(self):
        async def _action(graph):
            return _State.END, None

        policy = StrategyPolicy(
            actions={_State.START: _action},
            initial_state=_State.START,
            terminal_states=frozenset({_State.END}),
        )

        with pytest.raises(TypeError):
            policy.actions[_State.MIDDLE] = _action  # type: ignore[ty:invalid-assignment]

    def test_get_action_returns_configured_action(self):
        async def _action(graph):
            return _State.END, None

        policy = StrategyPolicy(
            actions={_State.START: _action},
            initial_state=_State.START,
            terminal_states=frozenset({_State.END}),
        )
        assert policy.get_action(state=_State.START) is _action

    def test_get_action_raises_informative_error_for_unknown_state(self):
        async def _action(graph):
            return _State.END, None

        policy = StrategyPolicy(
            actions={_State.START: _action},
            initial_state=_State.START,
            terminal_states=frozenset({_State.END}),
        )

        with pytest.raises(KeyError, match="No action defined for state"):
            policy.get_action(state=_State.MIDDLE)

    def test_is_terminal_predicate(self):
        async def _action(graph):
            return _State.END, None

        policy = StrategyPolicy(
            actions={_State.START: _action},
            initial_state=_State.START,
            terminal_states=frozenset({_State.END}),
        )
        assert policy.is_terminal(state=_State.END)
        assert not policy.is_terminal(state=_State.START)

    def test_terminal_states_frozen_against_input_mutation(self):
        """Mutating the original set passed in must not leak into the policy."""

        async def _action(graph):
            return _State.END, None

        mutable_terminals: set = {_State.END}
        policy: StrategyPolicy = StrategyPolicy(
            actions={_State.START: _action},
            initial_state=_State.START,
            terminal_states=mutable_terminals,  # type: ignore[arg-type]
        )

        mutable_terminals.add(_State.MIDDLE)

        assert isinstance(policy.terminal_states, frozenset)
        assert policy.terminal_states == frozenset({_State.END})
        assert not policy.is_terminal(state=_State.MIDDLE)


# ---------------------------------------------------------------------------
# StrategyGraph traversal
# ---------------------------------------------------------------------------


async def test_event_loop_yields_results_in_order():
    async def start_action(graph):
        return _State.MIDDLE, ScenarioStepResult(outcome="from_start")

    async def middle_action(graph):
        return _State.END, ScenarioStepResult(outcome="from_middle")

    graph = _make_graph(
        policy=StrategyPolicy(
            actions={_State.START: start_action, _State.MIDDLE: middle_action},
            initial_state=_State.START,
            terminal_states=frozenset({_State.END}),
        ),
    )

    results = [r async for r in graph.event_loop_async()]
    assert [r.outcome for r in results] == ["from_start", "from_middle"]


async def test_event_loop_advances_current_state():
    async def start_action(graph):
        return _State.END, ScenarioStepResult(outcome="done")

    graph = _make_graph(
        policy=StrategyPolicy(
            actions={_State.START: start_action},
            initial_state=_State.START,
            terminal_states=frozenset({_State.END}),
        ),
    )

    assert graph.current_state == _State.START
    _ = [r async for r in graph.event_loop_async()]
    assert graph.current_state == _State.END
    assert graph.is_terminal


async def test_event_loop_skips_yield_when_action_returns_no_result():
    async def silent_action(graph):
        return _State.END, None

    graph = _make_graph(
        policy=StrategyPolicy(
            actions={_State.START: silent_action},
            initial_state=_State.START,
            terminal_states=frozenset({_State.END}),
        ),
    )

    results = [r async for r in graph.event_loop_async()]
    assert results == []
    assert graph.current_state == _State.END


async def test_event_loop_raises_on_missing_action_entry():
    async def start_action(graph):
        # Transition to MIDDLE but the graph has no policy entry for MIDDLE.
        return _State.MIDDLE, None

    graph = _make_graph(
        policy=StrategyPolicy(
            actions={_State.START: start_action},
            initial_state=_State.START,
            terminal_states=frozenset({_State.END}),
        ),
    )

    with pytest.raises(KeyError, match="No action defined for state"):
        _ = [r async for r in graph.event_loop_async()]


async def test_history_records_yielded_results_only():
    async def start_action(graph):
        return _State.MIDDLE, ScenarioStepResult(outcome="recorded")

    async def middle_action(graph):
        return _State.END, None

    graph = _make_graph(
        policy=StrategyPolicy(
            actions={_State.START: start_action, _State.MIDDLE: middle_action},
            initial_state=_State.START,
            terminal_states=frozenset({_State.END}),
        ),
    )

    _ = [r async for r in graph.event_loop_async()]
    assert len(graph.history) == 1
    state_before, result = graph.history[0]
    assert state_before == _State.START
    assert result.outcome == "recorded"


async def test_reset_returns_graph_to_initial_state():
    async def start_action(graph):
        return _State.END, ScenarioStepResult(outcome="done")

    graph = _make_graph(
        policy=StrategyPolicy(
            actions={_State.START: start_action},
            initial_state=_State.START,
            terminal_states=frozenset({_State.END}),
        ),
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

    graph = _make_graph(
        policy=StrategyPolicy(
            actions={_State.START: noop_action},
            initial_state=_State.START,
            terminal_states=frozenset({_State.END}),
        ),
    )

    graph.bind_current_step(step=None)
    assert graph.current_step is None

    sentinel = object()
    graph.bind_current_step(step=sentinel)  # type: ignore[arg-type]
    assert graph.current_step is sentinel


async def test_event_loop_is_restartable_from_current_state():
    """After an exception, the loop can be re-entered from the failed state."""
    call_count = {"n": 0}

    async def flaky_action(graph):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("first call fails")
        return _State.END, ScenarioStepResult(outcome="recovered")

    graph = _make_graph(
        policy=StrategyPolicy(
            actions={_State.START: flaky_action},
            initial_state=_State.START,
            terminal_states=frozenset({_State.END}),
        ),
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

    graph = _make_graph(
        policy=StrategyPolicy(
            actions={
                ScenarioCoreState.INITIALIZING: initializing_action,
                ScenarioCoreState.EXECUTING: executing_action,
            },
            initial_state=ScenarioCoreState.INITIALIZING,
            terminal_states=frozenset({ScenarioCoreState.COMPLETE, ScenarioCoreState.FAILED}),
        ),
    )

    results = [r async for r in graph.event_loop_async()]
    assert [r.outcome for r in results] == ["finished"]
    assert graph.current_state == ScenarioCoreState.COMPLETE


async def test_event_loop_invokes_each_action_exactly_once_in_linear_graph():
    """Performance guard: an N-state linear graph triggers exactly N actions, not N**2."""
    n = 10
    actions: dict[int, AsyncMock] = {
        i: AsyncMock(return_value=(i + 1, ScenarioStepResult(outcome=f"step_{i}"))) for i in range(n)
    }

    graph: StrategyGraph[object, int] = StrategyGraph(
        policy=StrategyPolicy(
            actions=actions,
            initial_state=0,
            terminal_states=frozenset({n}),
        ),
    )

    results = [r async for r in graph.event_loop_async()]

    assert len(results) == n
    for i, action in actions.items():
        assert action.call_count == 1, f"action[{i}] invoked {action.call_count} times"
    assert sum(action.call_count for action in actions.values()) == n


async def test_event_loop_reaches_alternate_terminal_state():
    """A policy with multiple terminals must allow stopping on any of them."""

    async def failing_action(graph):
        return ScenarioCoreState.FAILED, ScenarioStepResult(outcome="aborted")

    graph = _make_graph(
        policy=StrategyPolicy(
            actions={ScenarioCoreState.EXECUTING: failing_action},
            initial_state=ScenarioCoreState.EXECUTING,
            terminal_states=frozenset({ScenarioCoreState.COMPLETE, ScenarioCoreState.FAILED}),
        ),
    )

    results = [r async for r in graph.event_loop_async()]

    assert [r.outcome for r in results] == ["aborted"]
    assert graph.current_state == ScenarioCoreState.FAILED
    assert graph.is_terminal


async def test_history_order_is_deterministic_across_runs():
    """Two consecutive runs of the same graph yield identical history orderings."""
    n = 5

    def _make_action(*, index: int):
        async def _action(graph):
            return index + 1, ScenarioStepResult(outcome=f"step_{index}")

        return _action

    actions = {i: _make_action(index=i) for i in range(n)}

    graph: StrategyGraph[object, int] = StrategyGraph(
        policy=StrategyPolicy(
            actions=actions,
            initial_state=0,
            terminal_states=frozenset({n}),
        ),
    )

    _ = [r async for r in graph.event_loop_async()]
    first_run = [(state, result.outcome) for state, result in graph.history]

    graph.reset()
    _ = [r async for r in graph.event_loop_async()]
    second_run = [(state, result.outcome) for state, result in graph.history]

    expected = [(i, f"step_{i}") for i in range(n)]
    assert first_run == expected
    assert second_run == expected
    assert first_run == second_run
