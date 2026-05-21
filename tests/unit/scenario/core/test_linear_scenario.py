# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Tests for :class:`LinearScenario` — the L0 authoring tier for scenarios.

Pins the contract that LinearScenario gives users a runnable scenario from
a list of pre-built ScenarioStep instances without subclassing, without a
graph vocabulary, and without registry-driven technique selection.
"""

from __future__ import annotations

from unittest.mock import MagicMock, PropertyMock

import pytest

from pyrit.executor.attack.core import AttackExecutorResult
from pyrit.identifiers import ComponentIdentifier
from pyrit.memory import CentralMemory
from pyrit.models import AttackOutcome, AttackResult
from pyrit.scenario import AtomicAttack, DatasetConfiguration, ScenarioResult
from pyrit.scenario.core import (
    BaselineAttackPolicy,
    LinearScenario,
    LinearScenarioStrategy,
    Scenario,
    ScenarioStrategy,
)
from pyrit.scenario.core.scenario_step import ScenarioStepResult
from pyrit.score import Scorer

_TEST_SCORER_ID = ComponentIdentifier(
    class_name="MockScorer",
    class_module="tests.unit.scenarios",
)


def _make_scorer() -> MagicMock:
    scorer = MagicMock(spec=Scorer)
    scorer.get_identifier.return_value = _TEST_SCORER_ID
    scorer.get_scorer_metrics.return_value = None
    return scorer


def _make_atomic_attack_mock(name: str, attack_result: AttackResult) -> MagicMock:
    """Build a fake AtomicAttack whose run_async returns the supplied result.

    Mirrors the canonical fixture pattern in test_scenario_graph_execution.py.
    """
    mock_attack = MagicMock()
    mock_attack.get_objective_target.return_value = MagicMock()
    mock_attack.get_attack_scoring_config.return_value = MagicMock()

    attack = MagicMock(spec=AtomicAttack)
    attack.atomic_attack_name = name
    attack.name = name
    attack.display_group = name
    attack._attack = mock_attack
    attack._scenario_result_id = None
    attack._scenario_max_concurrency = 1
    type(attack).objectives = PropertyMock(return_value=[attack_result.objective])

    def _set_scenario_result_id(sid):
        attack._scenario_result_id = sid

    attack.set_scenario_result_id = MagicMock(side_effect=_set_scenario_result_id)

    def _set_scenario_max_concurrency(mc):
        attack._scenario_max_concurrency = mc

    attack.set_scenario_max_concurrency = MagicMock(side_effect=_set_scenario_max_concurrency)

    async def _fake_run(*args, **kwargs):
        memory = CentralMemory.get_memory_instance()
        sid = attack._scenario_result_id
        if sid:
            attack_result.attribution_parent_id = sid
            attack_result.attribution_data = {"parent_collection": name}
        memory.add_attack_results_to_memory(attack_results=[attack_result])
        return AttackExecutorResult(completed_results=[attack_result], incomplete_objectives=[])

    attack.run_async = MagicMock(side_effect=_fake_run)

    async def _fake_process(*args, **kwargs):
        executor_result = await attack.run_async(
            max_concurrency=attack._scenario_max_concurrency,
            return_partial_on_failure=True,
        )
        return ScenarioStepResult(
            outcome="done",
            attack_results=list(executor_result.completed_results),
            metadata={
                "incomplete_objectives": list(executor_result.incomplete_objectives),
                "input_indices": list(executor_result.input_indices),
            },
        )

    attack.process_async = MagicMock(side_effect=_fake_process)
    return attack


def _sample_result(index: int) -> AttackResult:
    result = AttackResult(
        conversation_id=f"conv-{index}",
        objective=f"objective-{index}",
        outcome=AttackOutcome.SUCCESS,
        executed_turns=1,
    )
    result.atomic_attack_identifier = ComponentIdentifier(
        class_name="MockAttack",
        class_module="tests.unit.scenarios",
        params={"name": f"attack-{index}"},
    )
    return result


@pytest.fixture
def mock_objective_target():
    target = MagicMock()
    target.get_identifier.return_value = ComponentIdentifier(
        class_name="MockTarget",
        class_module="test",
    )
    return target


class TestLinearScenarioConstruction:
    """LinearScenario rejects empty step lists and exposes its strategy class."""

    def test_rejects_empty_steps(self):
        with pytest.raises(ValueError, match="at least one step"):
            LinearScenario(steps=[], objective_scorer=_make_scorer())

    def test_is_scenario_subclass(self):
        assert issubclass(LinearScenario, Scenario)

    def test_baseline_policy_is_forbidden(self):
        assert BaselineAttackPolicy.Forbidden == LinearScenario.BASELINE_ATTACK_POLICY

    def test_strategy_class_is_single_member(self):
        cls = LinearScenario.get_strategy_class()
        assert cls is LinearScenarioStrategy
        members = list(cls)
        assert len(members) == 1
        assert members[0] is LinearScenarioStrategy.DEFAULT

    def test_default_strategy_is_default_member(self):
        assert LinearScenario.get_default_strategy() is LinearScenarioStrategy.DEFAULT

    def test_default_dataset_config_is_empty(self):
        config = LinearScenario.default_dataset_config()
        assert isinstance(config, DatasetConfiguration)

    def test_strategy_aggregate_tag_is_all(self):
        assert LinearScenarioStrategy.get_aggregate_tags() == {"all"}

    def test_strategy_is_scenario_strategy_subclass(self):
        assert issubclass(LinearScenarioStrategy, ScenarioStrategy)


@pytest.mark.usefixtures("patch_central_database")
class TestLinearScenarioExecution:
    """End-to-end pin that LinearScenario walks its step list in order."""

    async def test_runs_steps_in_order(self, mock_objective_target):
        attacks = [_make_atomic_attack_mock(f"a{i}", _sample_result(i)) for i in range(3)]
        scenario = LinearScenario(steps=attacks, objective_scorer=_make_scorer())
        await scenario.initialize_async(objective_target=mock_objective_target)

        result = await scenario.run_async()

        assert isinstance(result, ScenarioResult)
        names = [r.metadata.get("step_name") for r in scenario.execution_history]
        assert names == ["a0", "a1", "a2"]

    async def test_each_step_runs_exactly_once(self, mock_objective_target):
        attacks = [_make_atomic_attack_mock(f"a{i}", _sample_result(i)) for i in range(3)]
        scenario = LinearScenario(steps=attacks, objective_scorer=_make_scorer())
        await scenario.initialize_async(objective_target=mock_objective_target)

        await scenario.run_async()

        for attack in attacks:
            attack.run_async.assert_called_once()

    async def test_max_concurrency_propagates_through_setter(self, mock_objective_target):
        attacks = [_make_atomic_attack_mock(f"a{i}", _sample_result(i)) for i in range(2)]
        scenario = LinearScenario(steps=attacks, objective_scorer=_make_scorer())
        await scenario.initialize_async(objective_target=mock_objective_target, max_concurrency=5)

        await scenario.run_async()

        for attack in attacks:
            attack.set_scenario_max_concurrency.assert_called_with(5)
            attack.run_async.assert_called_once_with(max_concurrency=5, return_partial_on_failure=True)

    async def test_explicit_steps_preserved_across_initialize(self, mock_objective_target):
        # ``initialize_async`` calls ``_get_atomic_attacks_async`` under the hood
        # to populate ``self._atomic_attacks``. LinearScenario must return the
        # exact steps supplied at construction time.
        attacks = [_make_atomic_attack_mock(f"a{i}", _sample_result(i)) for i in range(2)]
        scenario = LinearScenario(steps=attacks, objective_scorer=_make_scorer())
        await scenario.initialize_async(objective_target=mock_objective_target)

        assert list(scenario._atomic_attacks) == attacks

    async def test_history_length_matches_step_count(self, mock_objective_target):
        attacks = [_make_atomic_attack_mock(f"a{i}", _sample_result(i)) for i in range(4)]
        scenario = LinearScenario(steps=attacks, objective_scorer=_make_scorer())
        await scenario.initialize_async(objective_target=mock_objective_target)

        await scenario.run_async()

        assert len(scenario.execution_history) == 4
