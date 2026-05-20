# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Phase 5 — coverage for ``Scenario`` driving execution through ``StrategyGraph``.

These tests pin the new public surface (``execution_graph``, ``execution_history``,
``_build_execution_graph``, ``_build_default_linear_policy``) plus the contract
that the default linear policy produces the same end-to-end behavior as the
legacy flat loop (max_concurrency propagation, step_identifier stamping,
custom-policy overrides, non-AtomicAttack ``ScenarioStep`` dispatch).
"""

from typing import ClassVar, cast
from unittest.mock import MagicMock, PropertyMock

import pytest

from pyrit.executor.attack.core import AttackExecutorResult
from pyrit.identifiers import ComponentIdentifier
from pyrit.memory import CentralMemory
from pyrit.models import AttackOutcome, AttackResult
from pyrit.scenario import DatasetConfiguration, ScenarioResult
from pyrit.scenario.core import AtomicAttack, BaselinePolicy, Scenario, ScenarioStrategy
from pyrit.scenario.core.scenario_step import ScenarioStep, ScenarioStepResult
from pyrit.scenario.core.strategy_graph import (
    PolicyAction,
    StrategyGraph,
    StrategyPolicy,
)
from pyrit.score import Scorer

_TEST_SCORER_ID = ComponentIdentifier(
    class_name="MockScorer",
    class_module="tests.unit.scenarios",
)


def _save_results_to_memory(attack_results):
    memory = CentralMemory.get_memory_instance()
    memory.add_attack_results_to_memory(attack_results=attack_results)


def _make_atomic_attack_mock(name: str, attack_result: AttackResult) -> MagicMock:
    """Build a fake AtomicAttack whose run_async returns the supplied result."""
    mock_attack = MagicMock()
    mock_attack.get_objective_target.return_value = MagicMock()
    mock_attack.get_attack_scoring_config.return_value = MagicMock()

    attack = MagicMock(spec=AtomicAttack)
    attack.atomic_attack_name = name
    attack.display_group = name
    attack._attack = mock_attack
    type(attack).objectives = PropertyMock(return_value=[attack_result.objective])

    async def _fake_run(*args, **kwargs):
        _save_results_to_memory([attack_result])
        return AttackExecutorResult(completed_results=[attack_result], incomplete_objectives=[])

    attack.run_async = MagicMock(side_effect=_fake_run)
    return attack


def _sample_result(index: int) -> AttackResult:
    result = AttackResult(
        conversation_id=f"conv-{index}",
        objective=f"objective-{index}",
        outcome=AttackOutcome.SUCCESS,
        executed_turns=1,
    )
    # Stamp a real atomic_attack_identifier so step_identifier stamping can wrap it.
    result.atomic_attack_identifier = ComponentIdentifier(
        class_name="MockAttack",
        class_module="tests.unit.scenarios",
        params={"name": f"attack-{index}"},
    )
    return result


class _GraphConcreteScenario(Scenario):
    """Concrete Scenario for graph-execution tests.

    Mirrors ``ConcreteScenario`` from test_scenario.py but stays local so we can
    swap the execution-graph builder per test without coupling to the broader
    test_scenario fixture.
    """

    BASELINE_POLICY: ClassVar[BaselinePolicy] = BaselinePolicy.Forbidden

    def __init__(self, atomic_attacks_to_return=None, **kwargs):
        class _TestStrategy(ScenarioStrategy):
            TEST = ("test", {"concrete"})
            ALL = ("all", {"all"})

            @classmethod
            def get_aggregate_tags(cls) -> set[str]:
                return {"all"}

        kwargs.setdefault("strategy_class", _TestStrategy)

        if "objective_scorer" not in kwargs:
            mock_scorer = MagicMock(spec=Scorer)
            mock_scorer.get_identifier.return_value = _TEST_SCORER_ID
            mock_scorer.get_scorer_metrics.return_value = None
            kwargs["objective_scorer"] = mock_scorer

        super().__init__(**kwargs)
        self._atomic_attacks_to_return = atomic_attacks_to_return or []

    @classmethod
    def get_strategy_class(cls):
        class _TestStrategy(ScenarioStrategy):
            TEST = ("test", {"concrete"})
            ALL = ("all", {"all"})

            @classmethod
            def get_aggregate_tags(cls) -> set[str]:
                return {"all"}

        return _TestStrategy

    @classmethod
    def get_default_strategy(cls):
        return cls.get_strategy_class().ALL

    @classmethod
    def default_dataset_config(cls) -> DatasetConfiguration:
        return DatasetConfiguration()

    async def _get_atomic_attacks_async(self):
        return self._atomic_attacks_to_return


@pytest.fixture
def mock_objective_target():
    target = MagicMock()
    target.get_identifier.return_value = ComponentIdentifier(
        class_name="MockTarget",
        class_module="test",
    )
    return target


@pytest.mark.usefixtures("patch_central_database")
class TestBuildExecutionGraph:
    """Pin the default ``_build_execution_graph`` factory contract."""

    def test_raises_when_no_steps_and_no_atomic_attacks(self, mock_objective_target):
        scenario = _GraphConcreteScenario(name="Empty", version=1)
        with pytest.raises(ValueError, match="no steps"):
            scenario._build_execution_graph()

    def test_raises_when_explicit_steps_empty(self, mock_objective_target):
        scenario = _GraphConcreteScenario(name="Empty", version=1)
        with pytest.raises(ValueError, match="no steps"):
            scenario._build_execution_graph(steps=[])

    async def test_default_graph_terminates_at_len_steps(self, mock_objective_target):
        attacks = [_make_atomic_attack_mock(f"a{i}", _sample_result(i)) for i in range(3)]
        scenario = _GraphConcreteScenario(
            name="Default",
            version=1,
            atomic_attacks_to_return=attacks,
        )
        await scenario.initialize_async(objective_target=mock_objective_target)

        graph = scenario._build_execution_graph()

        assert graph.policy.initial_state == 0
        assert graph.policy.terminal_states == frozenset({3})
        # Three non-terminal states must have actions.
        for state in range(3):
            assert callable(graph.policy.get_action(state=state))

    async def test_explicit_steps_override_atomic_attacks(self, mock_objective_target):
        attacks = [_make_atomic_attack_mock(f"a{i}", _sample_result(i)) for i in range(3)]
        scenario = _GraphConcreteScenario(
            name="Default",
            version=1,
            atomic_attacks_to_return=attacks,
        )
        await scenario.initialize_async(objective_target=mock_objective_target)

        # Build with only the first two — terminal moves down to 2.
        graph = scenario._build_execution_graph(steps=attacks[:2])
        assert graph.policy.terminal_states == frozenset({2})


@pytest.mark.usefixtures("patch_central_database")
class TestExecutionGraphPropertyAndHistory:
    """``execution_graph`` and ``execution_history`` reflect the active attempt."""

    def test_graph_and_history_are_empty_before_run(self):
        scenario = _GraphConcreteScenario(name="Pre-run", version=1)
        assert scenario.execution_graph is None
        assert scenario.execution_history == []

    async def test_run_async_populates_graph_and_history(self, mock_objective_target):
        attacks = [_make_atomic_attack_mock(f"a{i}", _sample_result(i)) for i in range(2)]
        scenario = _GraphConcreteScenario(
            name="Populated",
            version=1,
            atomic_attacks_to_return=attacks,
        )
        await scenario.initialize_async(objective_target=mock_objective_target)

        await scenario.run_async()

        assert scenario.execution_graph is not None
        # Two steps executed; history records both.
        assert len(scenario.execution_history) == 2
        names = [r.metadata.get("step_name") for r in scenario.execution_history]
        assert names == ["a0", "a1"]


@pytest.mark.usefixtures("patch_central_database")
class TestStepIdentifierStamping:
    """Default linear path stamps a step_identifier on every persisted AttackResult."""

    async def test_each_result_has_step_identifier(self, mock_objective_target):
        attacks = [_make_atomic_attack_mock(f"a{i}", _sample_result(i)) for i in range(2)]
        scenario = _GraphConcreteScenario(
            name="Stamping",
            version=1,
            atomic_attacks_to_return=attacks,
        )
        await scenario.initialize_async(objective_target=mock_objective_target)

        result = await scenario.run_async()

        assert isinstance(result, ScenarioResult)
        flat_results = [ar for arr in result.attack_results.values() for ar in arr]
        assert flat_results, "expected at least one persisted result"
        for ar in flat_results:
            assert ar.step_identifier is not None
            assert ar.step_identifier.class_name == "ScenarioStep"
            # The step name shows up in the params (inlined by ComponentIdentifier.to_dict).
            id_dict = ar.step_identifier.to_dict()
            assert id_dict["step_name"] in {"a0", "a1"}
            assert id_dict["outcome"] == "done"

    async def test_pre_stamped_step_identifier_is_preserved(self, mock_objective_target):
        """If a step pre-stamps its own step_identifier, the orchestrator must not overwrite."""
        result_obj = _sample_result(0)
        pre_stamped = ComponentIdentifier(
            class_name="ScenarioStep",
            class_module="custom",
            params={"step_name": "custom_step", "outcome": "custom_outcome", "eval_version": 99},
        )
        result_obj.step_identifier = pre_stamped

        attack = _make_atomic_attack_mock("a0", result_obj)

        async def _run_returning_stamped(*args, **kwargs):
            _save_results_to_memory([result_obj])
            return AttackExecutorResult(completed_results=[result_obj], incomplete_objectives=[])

        attack.run_async = MagicMock(side_effect=_run_returning_stamped)

        scenario = _GraphConcreteScenario(
            name="Pre-stamped",
            version=1,
            atomic_attacks_to_return=[attack],
        )
        await scenario.initialize_async(objective_target=mock_objective_target)

        await scenario.run_async()

        stamped = scenario.execution_history[0].attack_results[0].step_identifier
        # The orchestrator must not have replaced the pre-stamped identifier with a
        # default ScenarioStep-shape one. We compare by params since equality is structural.
        assert stamped is not None
        assert stamped.params.get("step_name") == "custom_step"
        assert stamped.params.get("outcome") == "custom_outcome"
        assert stamped.params.get("eval_version") == 99


@pytest.mark.usefixtures("patch_central_database")
class TestStepIdentifierStampingNoDuplication:
    """The orchestrator's per-step ``update_attack_result_by_id`` enrichment must
    not introduce duplicate ``AttackResultEntry`` rows.

    Phase 5 routes results through ``StrategyGraph`` and enriches each result with
    a ``step_identifier`` via ``update_attack_result_by_id``. The inner attack is
    the sole insert site (mirrored here by ``_save_results_to_memory``); the
    orchestrator only updates. This regression test guards against accidentally
    flipping the update into an insert.
    """

    async def test_no_duplicate_attack_results_after_run(self, mock_objective_target):
        # Two atomic attacks, one result each → memory should hold exactly 2 rows.
        attacks = [_make_atomic_attack_mock(f"a{i}", _sample_result(i)) for i in range(2)]
        scenario = _GraphConcreteScenario(
            name="Dedup",
            version=1,
            atomic_attacks_to_return=attacks,
        )
        await scenario.initialize_async(objective_target=mock_objective_target)

        await scenario.run_async()

        memory = CentralMemory.get_memory_instance()
        persisted = memory.get_attack_results()
        assert len(persisted) == 2
        # Each persisted row must carry the step_identifier stamped by the orchestrator.
        for ar in persisted:
            assert ar.step_identifier is not None
            assert ar.step_identifier.class_name == "ScenarioStep"

    async def test_no_duplicate_results_for_multi_result_step(self, mock_objective_target):
        # One atomic attack returning two results — still exactly two rows after stamping.
        result_a = _sample_result(0)
        result_b = _sample_result(1)
        attack = _make_atomic_attack_mock("multi", result_a)

        async def _run_multi(*args, **kwargs):
            _save_results_to_memory([result_a, result_b])
            return AttackExecutorResult(completed_results=[result_a, result_b], incomplete_objectives=[])

        attack.run_async = MagicMock(side_effect=_run_multi)

        scenario = _GraphConcreteScenario(
            name="DedupMulti",
            version=1,
            atomic_attacks_to_return=[attack],
        )
        await scenario.initialize_async(objective_target=mock_objective_target)

        await scenario.run_async()

        persisted = CentralMemory.get_memory_instance().get_attack_results()
        assert len(persisted) == 2


@pytest.mark.usefixtures("patch_central_database")
class TestExecutionGraphRebuildOnRetry:
    """The execution graph is rebuilt from resume-filtered steps on each attempt.

    After a failed attempt, the next attempt's ``execution_graph`` must reflect
    only the still-pending steps. This pins the Phase 5 contract that
    ``_build_execution_graph(steps=remaining_attacks)`` is invoked per attempt.
    """

    async def test_graph_terminal_states_shrink_after_partial_success(self, mock_objective_target):
        # Attack 1 always succeeds; attack 2 fails once then succeeds.
        attack_success = _make_atomic_attack_mock("a_success", _sample_result(0))

        call_count = [0]
        result_second = _sample_result(1)

        async def _run_flaky(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("first attempt failure")
            _save_results_to_memory([result_second])
            return AttackExecutorResult(completed_results=[result_second], incomplete_objectives=[])

        attack_flaky = _make_atomic_attack_mock("a_flaky", result_second)
        attack_flaky.run_async = MagicMock(side_effect=_run_flaky)
        # Mock the resume filter; orchestrator drops attacks whose objectives are all done.
        attack_success.filter_seed_groups_by_objectives = MagicMock()
        attack_flaky.filter_seed_groups_by_objectives = MagicMock()

        scenario = _GraphConcreteScenario(
            name="RebuildOnRetry",
            version=1,
            atomic_attacks_to_return=[attack_success, attack_flaky],
        )
        await scenario.initialize_async(objective_target=mock_objective_target, max_retries=1)

        await scenario.run_async()

        # After retry, the graph should reflect only the one remaining step (attack_flaky).
        # ``execution_graph`` holds the most-recent attempt's graph.
        graph = scenario.execution_graph
        assert graph is not None
        assert graph.policy.terminal_states == frozenset({1})
        # Sanity: the flaky attack ran twice (initial + retry); the success attack only once.
        assert call_count[0] == 2
        attack_success.run_async.assert_called_once()


@pytest.mark.usefixtures("patch_central_database")
class TestMaxConcurrencyPropagation:
    """``max_concurrency`` flows from the scenario through the default linear policy."""

    async def test_atomic_attack_receives_scenario_max_concurrency(self, mock_objective_target):
        attacks = [_make_atomic_attack_mock(f"a{i}", _sample_result(i)) for i in range(2)]
        scenario = _GraphConcreteScenario(
            name="Concurrency",
            version=1,
            atomic_attacks_to_return=attacks,
        )
        await scenario.initialize_async(objective_target=mock_objective_target, max_concurrency=7)

        await scenario.run_async()

        for attack in attacks:
            attack.run_async.assert_called_once_with(max_concurrency=7, return_partial_on_failure=True)


@pytest.mark.usefixtures("patch_central_database")
class TestPartialFailureSurfacing:
    """Partial-failure metadata propagates through the step result and raises ValueError."""

    async def test_incomplete_objectives_in_metadata_raise(self, mock_objective_target):
        result = _sample_result(0)
        attack = _make_atomic_attack_mock("a0", result)

        async def _run_partial(*args, **kwargs):
            _save_results_to_memory([result])
            return AttackExecutorResult(
                completed_results=[result],
                incomplete_objectives=[("partial-obj", RuntimeError("boom"))],
            )

        attack.run_async = MagicMock(side_effect=_run_partial)

        scenario = _GraphConcreteScenario(
            name="Partial",
            version=1,
            atomic_attacks_to_return=[attack],
        )
        await scenario.initialize_async(objective_target=mock_objective_target)

        with pytest.raises(ValueError, match="partially failed"):
            await scenario.run_async()


class _CountingStep(ScenarioStep):
    """Non-AtomicAttack step that records every process_async call."""

    def __init__(self, *, name: str) -> None:
        self.name = name
        self.outputs = ["done"]
        self.call_count = 0

    async def process_async(self) -> ScenarioStepResult:
        self.call_count += 1
        return ScenarioStepResult(outcome="done", attack_results=[])


class _CustomStepScenario(_GraphConcreteScenario):
    """Scenario that drives non-AtomicAttack steps via an explicit StrategyGraph.

    Overrides ``_get_remaining_atomic_attacks_async`` (which is shaped for
    ``AtomicAttack`` instances) and ``_build_execution_graph`` to swap in a
    policy over arbitrary ``ScenarioStep`` subclasses without touching the
    legacy resume path.
    """

    def __init__(self, *, steps: list[_CountingStep], **kwargs):
        super().__init__(**kwargs)
        self._custom_steps = steps

    async def _get_remaining_atomic_attacks_async(self):  # type: ignore[override]
        # The custom steps are not AtomicAttacks; bypass the legacy resume filter
        # by returning them as-is. The orchestrator forwards them to our overridden
        # ``_build_execution_graph``, which wraps them in a hand-rolled policy.
        return self._custom_steps

    def _build_execution_graph(self, *, steps=None):
        effective_steps = list(steps) if steps is not None else list(self._custom_steps)

        actions: dict[int, PolicyAction[ScenarioStep, int]] = {}
        for index, step in enumerate(effective_steps):

            async def _action(
                graph: StrategyGraph[ScenarioStep, int],
                _step: ScenarioStep = step,
                _next: int = index + 1,
            ) -> tuple[int, ScenarioStepResult | None]:
                graph.bind_current_step(step=_step)
                try:
                    base = await _step.process_async()
                    merged = {"step_name": _step.name, **base.metadata}
                    return _next, ScenarioStepResult(
                        outcome=base.outcome,
                        attack_results=base.attack_results,
                        step_identifier=base.step_identifier,
                        metadata=merged,
                    )
                finally:
                    graph.bind_current_step(step=None)

            actions[index] = _action

        policy = StrategyPolicy(
            actions=actions,
            initial_state=0,
            terminal_states=frozenset({len(effective_steps)}),
        )
        return StrategyGraph(policy=policy)


@pytest.mark.usefixtures("patch_central_database")
class TestNonAtomicAttackStepDispatch:
    """Non-AtomicAttack ``ScenarioStep`` subclasses route through ``process_async``."""

    async def test_custom_step_routed_through_process_async(self, mock_objective_target):
        step_a = _CountingStep(name="custom_a")
        step_b = _CountingStep(name="custom_b")

        scenario = _CustomStepScenario(
            steps=[step_a, step_b],
            name="Custom-steps",
            version=1,
        )
        await scenario.initialize_async(objective_target=mock_objective_target)

        # Mirror what initialize_async would do for atomic attacks so the
        # orchestrator's progress-bar math sees the right total. The cast satisfies
        # the field's nominal ``list[AtomicAttack]`` annotation even though our
        # steps are non-atomic ``ScenarioStep`` subclasses.
        scenario._atomic_attacks = cast("list", [step_a, step_b])

        await scenario.run_async()

        assert step_a.call_count == 1
        assert step_b.call_count == 1
        history_names = [r.metadata.get("step_name") for r in scenario.execution_history]
        assert history_names == ["custom_a", "custom_b"]
