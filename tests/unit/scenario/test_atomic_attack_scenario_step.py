# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Tests for ``AtomicAttack`` as a ``ScenarioStep`` (Phase 2 adapter).

Verifies that ``AtomicAttack`` satisfies the ``ScenarioStep`` contract so
the upcoming ``StrategyGraph`` orchestrator can drive it uniformly with
richer step types. Existing ``AtomicAttack`` behavior is covered by
``test_atomic_attack.py``.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pyrit.executor.attack import AttackExecutor, AttackStrategy
from pyrit.executor.attack.core import AttackExecutorResult
from pyrit.identifiers import ComponentIdentifier, Identifiable
from pyrit.models import (
    AttackOutcome,
    AttackResult,
    SeedAttackGroup,
    SeedObjective,
    SeedPrompt,
)
from pyrit.scenario import AtomicAttack
from pyrit.scenario.core.attack_technique import AttackTechnique
from pyrit.scenario.core.scenario_step import ScenarioStep, ScenarioStepResult


@pytest.fixture
def mock_attack():
    return MagicMock(spec=AttackStrategy)


@pytest.fixture
def seed_groups():
    return [
        SeedAttackGroup(seeds=[SeedObjective(value="obj1"), SeedPrompt(value="p1")]),
        SeedAttackGroup(seeds=[SeedObjective(value="obj2"), SeedPrompt(value="p2")]),
    ]


@pytest.fixture
def attack_results():
    return [
        AttackResult(
            conversation_id="conv-1",
            objective="obj1",
            outcome=AttackOutcome.SUCCESS,
            executed_turns=1,
        ),
        AttackResult(
            conversation_id="conv-2",
            objective="obj2",
            outcome=AttackOutcome.FAILURE,
            executed_turns=1,
        ),
    ]


def test_atomic_attack_is_scenario_step():
    assert issubclass(AtomicAttack, ScenarioStep)


def test_atomic_attack_is_identifiable():
    assert issubclass(AtomicAttack, Identifiable)


@pytest.mark.usefixtures("patch_central_database")
class TestAtomicAttackScenarioStepShape:
    """Static shape: name/outputs/identifier behavior independent of execution."""

    def test_name_aliases_atomic_attack_name(self, mock_attack, seed_groups):
        atomic = AtomicAttack(
            attack_technique=AttackTechnique(attack=mock_attack),
            seed_groups=seed_groups,
            atomic_attack_name="my_step",
        )
        assert atomic.name == "my_step"
        assert atomic.name == atomic.atomic_attack_name

    def test_outputs_is_done_only(self, mock_attack, seed_groups):
        atomic = AtomicAttack(
            attack_technique=AttackTechnique(attack=mock_attack),
            seed_groups=seed_groups,
            atomic_attack_name="my_step",
        )
        assert atomic.outputs == ["done"]

    def test_outputs_property_returns_fresh_list(self, mock_attack, seed_groups):
        atomic = AtomicAttack(
            attack_technique=AttackTechnique(attack=mock_attack),
            seed_groups=seed_groups,
            atomic_attack_name="my_step",
        )
        first = atomic.outputs
        first.append("extra")
        # Mutation of the returned list must not affect the next read.
        assert atomic.outputs == ["done"]

    def test_get_identifier_returns_component_identifier(self, mock_attack, seed_groups):
        atomic = AtomicAttack(
            attack_technique=AttackTechnique(attack=mock_attack),
            seed_groups=seed_groups,
            atomic_attack_name="my_step",
        )
        identifier = atomic.get_identifier()
        assert isinstance(identifier, ComponentIdentifier)
        assert identifier.params["atomic_attack_name"] == "my_step"
        assert identifier.params["outputs"] == ["done"]
        assert "attack_technique" in identifier.children

    def test_get_identifier_is_cached(self, mock_attack, seed_groups):
        atomic = AtomicAttack(
            attack_technique=AttackTechnique(attack=mock_attack),
            seed_groups=seed_groups,
            atomic_attack_name="my_step",
        )
        first = atomic.get_identifier()
        second = atomic.get_identifier()
        assert first is second

    def test_identifier_hash_differs_when_name_differs(self, mock_attack, seed_groups):
        atomic_a = AtomicAttack(
            attack_technique=AttackTechnique(attack=mock_attack),
            seed_groups=seed_groups,
            atomic_attack_name="step_a",
        )
        atomic_b = AtomicAttack(
            attack_technique=AttackTechnique(attack=mock_attack),
            seed_groups=seed_groups,
            atomic_attack_name="step_b",
        )
        assert atomic_a.get_identifier().hash != atomic_b.get_identifier().hash


@pytest.mark.usefixtures("patch_central_database")
class TestAtomicAttackProcessAsync:
    """``process_async`` wraps ``run_async`` into a ``ScenarioStepResult``."""

    async def test_returns_scenario_step_result_with_done_outcome(self, mock_attack, seed_groups, attack_results):
        atomic = AtomicAttack(
            attack_technique=AttackTechnique(attack=mock_attack),
            seed_groups=seed_groups,
            atomic_attack_name="my_step",
        )

        with patch.object(AttackExecutor, "execute_attack_from_seed_groups_async", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = AttackExecutorResult(
                completed_results=attack_results,
                incomplete_objectives=[],
                input_indices=[0, 1],
            )
            result = await atomic.process_async()

        assert isinstance(result, ScenarioStepResult)
        assert result.outcome == "done"
        assert result.attack_results == attack_results

    async def test_metadata_carries_incomplete_objectives(self, mock_attack, seed_groups, attack_results):
        atomic = AtomicAttack(
            attack_technique=AttackTechnique(attack=mock_attack),
            seed_groups=seed_groups,
            atomic_attack_name="my_step",
        )

        incomplete: list[tuple[str, BaseException]] = [("obj_failed", RuntimeError("boom"))]
        with patch.object(AttackExecutor, "execute_attack_from_seed_groups_async", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = AttackExecutorResult(
                completed_results=attack_results[:1],
                incomplete_objectives=incomplete,
                input_indices=[0],
            )
            result = await atomic.process_async()

        assert result.metadata["incomplete_objectives"] == incomplete
        assert result.metadata["input_indices"] == [0]

    async def test_propagates_run_async_failures(self, mock_attack, seed_groups):
        atomic = AtomicAttack(
            attack_technique=AttackTechnique(attack=mock_attack),
            seed_groups=seed_groups,
            atomic_attack_name="my_step",
        )

        with patch.object(AttackExecutor, "execute_attack_from_seed_groups_async", new_callable=AsyncMock) as mock_exec:
            mock_exec.side_effect = RuntimeError("execution exploded")
            # run_async wraps the underlying exception in a ValueError; process_async surfaces it.
            with pytest.raises(ValueError, match="Failed to execute atomic attack"):
                await atomic.process_async()

    async def test_returns_empty_results_when_no_completions(self, mock_attack, seed_groups):
        atomic = AtomicAttack(
            attack_technique=AttackTechnique(attack=mock_attack),
            seed_groups=seed_groups,
            atomic_attack_name="my_step",
        )

        with patch.object(AttackExecutor, "execute_attack_from_seed_groups_async", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = AttackExecutorResult(
                completed_results=[],
                incomplete_objectives=[("obj1", RuntimeError("fail"))],
                input_indices=[],
            )
            result = await atomic.process_async()

        assert result.outcome == "done"
        assert result.attack_results == []
        assert len(result.metadata["incomplete_objectives"]) == 1


@pytest.mark.usefixtures("patch_central_database")
class TestAtomicAttackFilterSeedGroupsByObjectives:
    """``filter_seed_groups_by_objectives`` is part of the duck-typed ScenarioStep surface."""

    def test_remaining_objectives_is_keyword_only(self, mock_attack, seed_groups):
        atomic = AtomicAttack(
            attack_technique=AttackTechnique(attack=mock_attack),
            seed_groups=seed_groups,
            atomic_attack_name="my_step",
        )
        with pytest.raises(TypeError):
            atomic.filter_seed_groups_by_objectives(["obj1"])  # type: ignore[misc]

    def test_filter_drops_groups_not_in_remaining(self, mock_attack, seed_groups):
        atomic = AtomicAttack(
            attack_technique=AttackTechnique(attack=mock_attack),
            seed_groups=seed_groups,
            atomic_attack_name="my_step",
        )
        atomic.filter_seed_groups_by_objectives(remaining_objectives=["obj2"])
        assert atomic.objectives == ["obj2"]
        assert len(atomic.seed_groups) == 1

    def test_filter_keeps_all_when_all_remain(self, mock_attack, seed_groups):
        atomic = AtomicAttack(
            attack_technique=AttackTechnique(attack=mock_attack),
            seed_groups=seed_groups,
            atomic_attack_name="my_step",
        )
        atomic.filter_seed_groups_by_objectives(remaining_objectives=["obj1", "obj2"])
        assert atomic.objectives == ["obj1", "obj2"]
        assert len(atomic.seed_groups) == 2

    def test_filter_drops_all_when_none_remain(self, mock_attack, seed_groups):
        atomic = AtomicAttack(
            attack_technique=AttackTechnique(attack=mock_attack),
            seed_groups=seed_groups,
            atomic_attack_name="my_step",
        )
        atomic.filter_seed_groups_by_objectives(remaining_objectives=[])
        assert atomic.objectives == []
        assert atomic.seed_groups == []

    def test_filter_ignores_unknown_objectives(self, mock_attack, seed_groups):
        atomic = AtomicAttack(
            attack_technique=AttackTechnique(attack=mock_attack),
            seed_groups=seed_groups,
            atomic_attack_name="my_step",
        )
        atomic.filter_seed_groups_by_objectives(remaining_objectives=["obj1", "does_not_exist"])
        assert atomic.objectives == ["obj1"]
