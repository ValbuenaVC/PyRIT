# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Branching ``StrategyGraph`` integration test.

The rest of Phase 0/3's coverage exercises the policy machinery on toy
two-step flat graphs. This module forces the API through a non-trivial
branching scenario *before* Phase 5 commits ``Scenario.run_async`` to it.

Scenario: ``BroadSweepThenDeepDive``
    OPENING_PHASE
        runs a broad-sweep step that emits one of:
            ``"safe"``      — every category passed, transition to COMPLETE
            ``"violation"`` — at least one category leaked, transition to
                              ESCALATION_PHASE
    ESCALATION_PHASE
        runs a deep-dive step that always emits ``"done"`` and transitions
        to COMPLETE.

This is intentionally not a real scenario (lives in tests/), but it has the
exact shape the first real graph-based scenarios will need: outcome-driven
branching with at least one self-loop-free path that skips a step.
"""

from enum import Enum

import pytest

from pyrit.scenario.core.scenario_step import ScenarioStep, ScenarioStepResult
from pyrit.scenario.core.strategy_graph import StrategyGraph, StrategyPolicy


class _Phase(str, Enum):
    OPENING = "opening"
    ESCALATION = "escalation"
    COMPLETE = "complete"


class _SweepStep(ScenarioStep):
    """Broad-sweep step. Emits ``"safe"`` or ``"violation"`` per fixture state."""

    def __init__(self, *, sweep_outcome: str):
        self._sweep_outcome = sweep_outcome
        self.call_count = 0

    @property
    def name(self) -> str:  # type: ignore[override]
        return "broad_sweep"

    @property
    def outputs(self) -> list[str]:  # type: ignore[override]
        return ["safe", "violation"]

    async def process_async(self) -> ScenarioStepResult:
        self.call_count += 1
        return ScenarioStepResult(outcome=self._sweep_outcome)


class _EscalationStep(ScenarioStep):
    """Deep-dive step. Only runs if the sweep emitted a violation."""

    def __init__(self) -> None:
        self.call_count = 0

    @property
    def name(self) -> str:  # type: ignore[override]
        return "deep_dive"

    @property
    def outputs(self) -> list[str]:  # type: ignore[override]
        return ["done"]

    async def process_async(self) -> ScenarioStepResult:
        self.call_count += 1
        return ScenarioStepResult(outcome="done", metadata={"escalated": True})


def _build_graph(*, sweep_outcome: str) -> tuple[StrategyGraph, _SweepStep, _EscalationStep]:
    sweep = _SweepStep(sweep_outcome=sweep_outcome)
    escalation = _EscalationStep()

    async def opening_action(graph):
        graph.bind_current_step(step=sweep)
        try:
            result = await sweep.process_async()
        finally:
            graph.bind_current_step(step=None)
        next_state = _Phase.COMPLETE if result.outcome == "safe" else _Phase.ESCALATION
        return next_state, result

    async def escalation_action(graph):
        graph.bind_current_step(step=escalation)
        try:
            result = await escalation.process_async()
        finally:
            graph.bind_current_step(step=None)
        return _Phase.COMPLETE, result

    policy: StrategyPolicy[ScenarioStep, _Phase] = StrategyPolicy(
        actions={_Phase.OPENING: opening_action, _Phase.ESCALATION: escalation_action},
        initial_state=_Phase.OPENING,
        terminal_states=frozenset({_Phase.COMPLETE}),
    )
    return StrategyGraph(policy=policy), sweep, escalation


async def test_safe_sweep_short_circuits_to_complete():
    graph, sweep, escalation = _build_graph(sweep_outcome="safe")

    results = [r async for r in graph.event_loop_async()]
    outcomes = [r.outcome for r in results]

    assert outcomes == ["safe"]
    assert sweep.call_count == 1
    assert escalation.call_count == 0
    assert graph.current_state == _Phase.COMPLETE
    assert graph.is_terminal


async def test_violation_sweep_triggers_escalation():
    graph, sweep, escalation = _build_graph(sweep_outcome="violation")

    results = [r async for r in graph.event_loop_async()]
    outcomes = [r.outcome for r in results]

    assert outcomes == ["violation", "done"]
    assert sweep.call_count == 1
    assert escalation.call_count == 1
    assert graph.current_state == _Phase.COMPLETE
    assert graph.is_terminal
    # The escalation step's metadata survived the round trip.
    assert results[1].metadata["escalated"] is True


async def test_history_records_both_branches():
    graph, _, _ = _build_graph(sweep_outcome="violation")
    _ = [r async for r in graph.event_loop_async()]

    states_before = [state for state, _ in graph.history]
    assert states_before == [_Phase.OPENING, _Phase.ESCALATION]


async def test_reset_replays_branching_graph():
    """A graph that branched once can be reset and re-run on the other branch."""
    graph, sweep, escalation = _build_graph(sweep_outcome="violation")
    _ = [r async for r in graph.event_loop_async()]
    assert graph.current_state == _Phase.COMPLETE

    graph.reset()
    assert graph.current_state == _Phase.OPENING
    assert graph.history == []

    # Re-run produces the same branch (action closures captured the same state).
    results = [r async for r in graph.event_loop_async()]
    assert [r.outcome for r in results] == ["violation", "done"]
    assert sweep.call_count == 2
    assert escalation.call_count == 2


# ---------------------------------------------------------------------------
# 3-way dispatch — confirms transitions are dict-lookup based, not isinstance chains.
# ---------------------------------------------------------------------------


class _DispatchState(str, Enum):
    DISPATCH = "dispatch"
    BRANCH_A = "branch_a"
    BRANCH_B = "branch_b"
    BRANCH_C = "branch_c"
    COMPLETE = "complete"


def _build_three_way_graph(*, target: _DispatchState) -> StrategyGraph:
    async def dispatch_action(graph):
        return target, ScenarioStepResult(outcome=f"to_{target.value}")

    async def branch_action(graph):
        return _DispatchState.COMPLETE, ScenarioStepResult(outcome=f"reached_{graph.current_state.value}")

    policy: StrategyPolicy[ScenarioStep, _DispatchState] = StrategyPolicy(
        actions={
            _DispatchState.DISPATCH: dispatch_action,
            _DispatchState.BRANCH_A: branch_action,
            _DispatchState.BRANCH_B: branch_action,
            _DispatchState.BRANCH_C: branch_action,
        },
        initial_state=_DispatchState.DISPATCH,
        terminal_states=frozenset({_DispatchState.COMPLETE}),
    )
    return StrategyGraph(policy=policy)


@pytest.mark.parametrize(
    "target",
    [_DispatchState.BRANCH_A, _DispatchState.BRANCH_B, _DispatchState.BRANCH_C],
)
async def test_three_way_branch_dispatches_to_target_state(target: _DispatchState) -> None:
    graph = _build_three_way_graph(target=target)

    results = [r async for r in graph.event_loop_async()]

    assert [r.outcome for r in results] == [f"to_{target.value}", f"reached_{target.value}"]
    states_before = [state for state, _ in graph.history]
    assert states_before == [_DispatchState.DISPATCH, target]
    assert graph.current_state == _DispatchState.COMPLETE
    assert graph.is_terminal
