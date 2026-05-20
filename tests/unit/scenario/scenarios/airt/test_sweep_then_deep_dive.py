# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Phase 7 — coverage for ``BroadSweepThenDeepDive``.

These tests pin the contract of the first real branching scenario:

- ``CategoryAggregatingSweepStep`` classifies each ``AttackResult`` via the
  ``OutcomeScorer`` and emits the right outcome label plus weak-category set.
- ``FilteredDeepDiveStep`` gates each wrapped atomic on its category being in
  the live weak set; non-matching atomics are skipped and stamped into
  ``metadata['skipped_categories']``.
- ``BroadSweepThenDeepDive._build_execution_graph`` produces a two-action
  branching policy that walks SWEEPING -> {DEEP_DIVING -> COMPLETE | ALL_SAFE}
  and resets ``self._weak_categories`` on every invocation.
- The closure-shared ``weak_categories_ref`` correctly threads sweep output
  into the deep-dive step at policy-dispatch time.
"""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import MagicMock, PropertyMock

import pytest

from pyrit.executor.attack.core import AttackExecutorResult
from pyrit.identifiers import ComponentIdentifier
from pyrit.models import AttackOutcome, AttackResult, MessagePiece
from pyrit.scenario import DatasetConfiguration
from pyrit.scenario.core import AtomicAttack
from pyrit.scenario.core.scenario import BaselineAttackPolicy
from pyrit.scenario.core.scenario_step import ScenarioStep, ScenarioStepResult
from pyrit.scenario.core.strategy_graph import StrategyGraph, StrategyPolicy
from pyrit.scenario.scenarios.airt.sweep_then_deep_dive import (
    BroadSweepThenDeepDive,
    BroadSweepThenDeepDiveStrategy,
    CategoryAggregatingSweepStep,
    FilteredDeepDiveStep,
    SweepThenDeepDiveState,
)
from pyrit.score import Scorer
from pyrit.score.decorators.outcome_scorer import OutcomeScorer

_WEAKNESS_LABEL = "safety_violation"
_SAFE_LABEL = "safe"


def _save_results_to_memory(attack_results: list[AttackResult]) -> None:
    # Memory persistence is intentionally skipped in mocks; these tests
    # exercise the branching policy + classification path, not the
    # persistence layer (covered by Phase 5 graph-execution tests).
    return


def _make_scorer_id(name: str) -> ComponentIdentifier:
    return ComponentIdentifier(class_name=name, class_module="tests.unit.scenario.scenarios.airt")


def _make_outcome_scorer(*, label_for: dict[str, str] | None = None) -> OutcomeScorer:
    """Build an ``OutcomeScorer`` whose wrapped scorer maps response.text to a label.

    The wrapped scorer pulls the response text out of the supplied ``Message``
    and looks it up in ``label_for`` to decide whether the score predicate
    should return weakness or safe. This lets each test feed deterministic
    per-category labels without writing a full ``Scorer`` subclass.
    """
    label_for = label_for or {}
    scorer = MagicMock(spec=Scorer)
    scorer.get_identifier.return_value = _make_scorer_id("MockScorer")

    async def _score_async(message: Any, *, objective: Any = None) -> list[Any]:
        text = message.message_pieces[0].original_value if message.message_pieces else ""
        label = label_for.get(text, _SAFE_LABEL)
        score = MagicMock()
        score.score_value = label
        return [score]

    scorer.score_async = MagicMock(side_effect=_score_async)

    return OutcomeScorer(
        wrapped_scorer=scorer,
        outcome_map={
            _WEAKNESS_LABEL: lambda s: s.score_value == _WEAKNESS_LABEL,
            _SAFE_LABEL: lambda s: s.score_value == _SAFE_LABEL,
        },
    )


def _attack_result(*, conversation_id: str, objective: str, response_text: str | None) -> AttackResult:
    """Build an AttackResult with a real last_response piece (or None)."""
    result = AttackResult(
        conversation_id=conversation_id,
        objective=objective,
        outcome=AttackOutcome.SUCCESS,
        executed_turns=1,
    )
    result.atomic_attack_identifier = ComponentIdentifier(
        class_name="MockAttack",
        class_module="tests.unit.scenario.scenarios.airt",
        params={"name": conversation_id},
    )
    if response_text is not None:
        result.last_response = MessagePiece(role="assistant", original_value=response_text)
    return result


def _make_atomic_mock(
    *,
    name: str,
    display_group: str,
    attack_results: list[AttackResult],
    objectives: list[str] | None = None,
) -> MagicMock:
    """Mock an ``AtomicAttack`` whose ``run_async`` returns the supplied results."""
    attack = MagicMock(spec=AtomicAttack)
    attack.atomic_attack_name = name
    attack.display_group = display_group
    type(attack).objectives = PropertyMock(
        return_value=list(objectives) if objectives is not None else [r.objective for r in attack_results]
    )
    type(attack).seed_groups = PropertyMock(return_value=[])
    attack.get_identifier.return_value = ComponentIdentifier(
        class_name="AtomicAttack",
        class_module="tests.unit.scenario.scenarios.airt",
        params={"name": name},
    )

    async def _fake_run(*args: Any, **kwargs: Any) -> AttackExecutorResult:
        _save_results_to_memory(attack_results)
        return AttackExecutorResult(completed_results=attack_results, incomplete_objectives=[])

    attack.run_async = MagicMock(side_effect=_fake_run)
    return attack


@pytest.mark.usefixtures("patch_central_database")
class TestCategoryAggregatingSweepStep:
    """Pin the sweep step's classification, outcome, and metadata contract."""

    def test_init_rejects_weakness_label_not_in_outcomes(self) -> None:
        scorer = _make_outcome_scorer()
        atomic = _make_atomic_mock(name="sweep", display_group="cat-a", attack_results=[])
        with pytest.raises(ValueError, match="weakness_label"):
            CategoryAggregatingSweepStep(
                atomic_attack_name="sweep",
                atomic_attack=atomic,
                outcome_scorer=scorer,
                weakness_label="not_a_real_label",
            )

    def test_init_surfaces_atomic_attack_attributes(self) -> None:
        scorer = _make_outcome_scorer()
        atomic = _make_atomic_mock(
            name="sweep-name",
            display_group="harms",
            attack_results=[],
            objectives=["obj-1", "obj-2"],
        )
        step = CategoryAggregatingSweepStep(
            atomic_attack_name="sweep-name",
            atomic_attack=atomic,
            outcome_scorer=scorer,
            weakness_label=_WEAKNESS_LABEL,
        )
        assert isinstance(step, ScenarioStep)
        assert step.name == "sweep-name"
        assert step.atomic_attack_name == "sweep-name"
        assert step.display_group == "harms"
        assert step.objectives == ["obj-1", "obj-2"]
        assert _WEAKNESS_LABEL in step.outputs or step.outputs == list(
            CategoryAggregatingSweepStep._OUTPUTS,
        )

    def test_filter_seed_groups_forwards_to_wrapped_atomic(self) -> None:
        scorer = _make_outcome_scorer()
        atomic = _make_atomic_mock(name="sweep", display_group="cat-a", attack_results=[])
        step = CategoryAggregatingSweepStep(
            atomic_attack_name="sweep",
            atomic_attack=atomic,
            outcome_scorer=scorer,
            weakness_label=_WEAKNESS_LABEL,
        )
        step.filter_seed_groups_by_objectives(remaining_objectives=["obj-x"])
        atomic.filter_seed_groups_by_objectives.assert_called_once_with(remaining_objectives=["obj-x"])

    async def test_process_emits_found_weaknesses_when_any_match(self) -> None:
        scorer = _make_outcome_scorer(
            label_for={"breach-A": _WEAKNESS_LABEL, "breach-B": _WEAKNESS_LABEL},
        )
        result_one = _attack_result(
            conversation_id="c1",
            objective="o1",
            response_text="breach-A",
        )
        result_two = _attack_result(
            conversation_id="c2",
            objective="o2",
            response_text="breach-B",
        )
        atomic = _make_atomic_mock(
            name="sweep",
            display_group="cat-a",
            attack_results=[result_one, result_two],
        )
        step = CategoryAggregatingSweepStep(
            atomic_attack_name="sweep",
            atomic_attack=atomic,
            outcome_scorer=scorer,
            weakness_label=_WEAKNESS_LABEL,
        )

        step_result = await step.process_async()

        # All results in this sweep share display_group "cat-a", so weak_categories
        # is OR-aggregated across results.
        assert step_result.outcome == "found_weaknesses"
        assert step_result.metadata["weak_categories"] == {"cat-a"}
        assert step_result.metadata["category_outcomes"]["cat-a"] == _WEAKNESS_LABEL
        assert len(step_result.attack_results) == 2

    async def test_process_marks_category_weak_even_when_only_one_result_is_weak(self) -> None:
        """OR-semantics: a single weak result in a category flags the whole category."""
        scorer = _make_outcome_scorer(
            label_for={"breach-A": _WEAKNESS_LABEL, "safe-text": _SAFE_LABEL},
        )
        result_weak = _attack_result(
            conversation_id="c1",
            objective="o1",
            response_text="breach-A",
        )
        result_safe = _attack_result(
            conversation_id="c2",
            objective="o2",
            response_text="safe-text",
        )
        atomic = _make_atomic_mock(
            name="sweep",
            display_group="cat-a",
            attack_results=[result_weak, result_safe],
        )
        step = CategoryAggregatingSweepStep(
            atomic_attack_name="sweep",
            atomic_attack=atomic,
            outcome_scorer=scorer,
            weakness_label=_WEAKNESS_LABEL,
        )

        step_result = await step.process_async()

        assert step_result.outcome == "found_weaknesses"
        assert step_result.metadata["weak_categories"] == {"cat-a"}

    async def test_process_emits_all_safe_when_no_matches(self) -> None:
        scorer = _make_outcome_scorer(label_for={"safe-text": _SAFE_LABEL})
        result_safe = _attack_result(
            conversation_id="c1",
            objective="o1",
            response_text="safe-text",
        )
        atomic = _make_atomic_mock(
            name="sweep",
            display_group="cat-a",
            attack_results=[result_safe],
        )
        step = CategoryAggregatingSweepStep(
            atomic_attack_name="sweep",
            atomic_attack=atomic,
            outcome_scorer=scorer,
            weakness_label=_WEAKNESS_LABEL,
        )

        step_result = await step.process_async()

        assert step_result.outcome == "all_safe"
        assert step_result.metadata["weak_categories"] == set()
        assert step_result.metadata["category_outcomes"] == {"cat-a": _SAFE_LABEL}

    async def test_process_returns_unscored_when_last_response_is_none(self) -> None:
        scorer = _make_outcome_scorer()
        result_no_response = _attack_result(conversation_id="c1", objective="o1", response_text=None)
        atomic = _make_atomic_mock(
            name="sweep",
            display_group="cat-a",
            attack_results=[result_no_response],
        )
        step = CategoryAggregatingSweepStep(
            atomic_attack_name="sweep",
            atomic_attack=atomic,
            outcome_scorer=scorer,
            weakness_label=_WEAKNESS_LABEL,
        )

        step_result = await step.process_async()

        assert step_result.metadata["category_outcomes"]["cat-a"] == OutcomeScorer.UNSCORED
        assert step_result.outcome == "all_safe"

    async def test_process_empty_results_emits_all_safe(self) -> None:
        scorer = _make_outcome_scorer()
        atomic = _make_atomic_mock(name="sweep", display_group="cat-a", attack_results=[])
        step = CategoryAggregatingSweepStep(
            atomic_attack_name="sweep",
            atomic_attack=atomic,
            outcome_scorer=scorer,
            weakness_label=_WEAKNESS_LABEL,
        )

        step_result = await step.process_async()

        assert step_result.outcome == "all_safe"
        assert step_result.metadata["weak_categories"] == set()
        assert step_result.attack_results == []

    def test_build_identifier_nests_attack_and_scorer(self) -> None:
        scorer = _make_outcome_scorer()
        atomic = _make_atomic_mock(name="sweep", display_group="cat-a", attack_results=[])
        step = CategoryAggregatingSweepStep(
            atomic_attack_name="sweep",
            atomic_attack=atomic,
            outcome_scorer=scorer,
            weakness_label=_WEAKNESS_LABEL,
        )
        identifier = step._build_identifier()
        assert "sweep_attack" in identifier.children
        assert "outcome_scorer" in identifier.children
        assert identifier.params["weakness_label"] == _WEAKNESS_LABEL


@pytest.mark.usefixtures("patch_central_database")
class TestFilteredDeepDiveStep:
    """Pin the deep-dive step's gating, metadata, and aggregation contract."""

    def test_init_rejects_empty_atomic_list(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            FilteredDeepDiveStep(
                atomic_attack_name="deep",
                atomic_attacks=[],
                weak_categories_ref=lambda: set(),
            )

    def test_init_dedupes_objectives_across_atomics(self) -> None:
        a = _make_atomic_mock(name="a", display_group="cat-a", attack_results=[], objectives=["o1", "o2"])
        b = _make_atomic_mock(name="b", display_group="cat-b", attack_results=[], objectives=["o2", "o3"])
        step = FilteredDeepDiveStep(
            atomic_attack_name="deep",
            atomic_attacks=[a, b],
            weak_categories_ref=lambda: set(),
        )
        assert isinstance(step, ScenarioStep)
        assert step.objectives == ["o1", "o2", "o3"]

    async def test_process_dispatches_only_flagged_categories(self) -> None:
        result_a = _attack_result(conversation_id="da", objective="oa", response_text="ra")
        result_b = _attack_result(conversation_id="db", objective="ob", response_text="rb")
        atomic_a = _make_atomic_mock(name="a", display_group="cat-a", attack_results=[result_a])
        atomic_b = _make_atomic_mock(name="b", display_group="cat-b", attack_results=[result_b])

        step = FilteredDeepDiveStep(
            atomic_attack_name="deep",
            atomic_attacks=[atomic_a, atomic_b],
            weak_categories_ref=lambda: {"cat-a"},
        )

        step_result = await step.process_async()

        atomic_a.run_async.assert_called_once()
        atomic_b.run_async.assert_not_called()
        assert step_result.outcome == "done"
        assert step_result.metadata["dispatched_categories"] == ["cat-a"]
        assert step_result.metadata["skipped_categories"] == ["cat-b"]
        assert step_result.attack_results == [result_a]

    async def test_process_skips_all_when_weak_set_empty(self) -> None:
        result_a = _attack_result(conversation_id="da", objective="oa", response_text="ra")
        atomic_a = _make_atomic_mock(name="a", display_group="cat-a", attack_results=[result_a])

        step = FilteredDeepDiveStep(
            atomic_attack_name="deep",
            atomic_attacks=[atomic_a],
            weak_categories_ref=lambda: set(),
        )

        step_result = await step.process_async()

        atomic_a.run_async.assert_not_called()
        assert step_result.outcome == "done"
        assert step_result.metadata["dispatched_categories"] == []
        assert step_result.metadata["skipped_categories"] == ["cat-a"]
        assert step_result.attack_results == []

    async def test_process_reads_weak_categories_live_at_dispatch_time(self) -> None:
        """The closure must be re-evaluated at dispatch time, not at init."""
        weak: set[str] = set()
        result_a = _attack_result(conversation_id="da", objective="oa", response_text="ra")
        atomic_a = _make_atomic_mock(name="a", display_group="cat-a", attack_results=[result_a])
        step = FilteredDeepDiveStep(
            atomic_attack_name="deep",
            atomic_attacks=[atomic_a],
            weak_categories_ref=lambda: weak,
        )
        # Mutate the set AFTER init but BEFORE process_async runs.
        weak.add("cat-a")

        step_result = await step.process_async()

        atomic_a.run_async.assert_called_once()
        assert step_result.metadata["dispatched_categories"] == ["cat-a"]

    def test_filter_seed_groups_forwards_to_each_atomic(self) -> None:
        a = _make_atomic_mock(name="a", display_group="cat-a", attack_results=[])
        b = _make_atomic_mock(name="b", display_group="cat-b", attack_results=[])
        step = FilteredDeepDiveStep(
            atomic_attack_name="deep",
            atomic_attacks=[a, b],
            weak_categories_ref=lambda: set(),
        )
        step.filter_seed_groups_by_objectives(remaining_objectives=["obj-z"])
        a.filter_seed_groups_by_objectives.assert_called_once_with(remaining_objectives=["obj-z"])
        b.filter_seed_groups_by_objectives.assert_called_once_with(remaining_objectives=["obj-z"])
        assert step.objectives == ["obj-z"]

    def test_build_identifier_sorts_nested_atomics(self) -> None:
        a = _make_atomic_mock(name="b-second", display_group="cat-b", attack_results=[])
        b = _make_atomic_mock(name="a-first", display_group="cat-a", attack_results=[])

        step_ab = FilteredDeepDiveStep(
            atomic_attack_name="deep",
            atomic_attacks=[a, b],
            weak_categories_ref=lambda: set(),
        )
        step_ba = FilteredDeepDiveStep(
            atomic_attack_name="deep",
            atomic_attacks=[b, a],
            weak_categories_ref=lambda: set(),
        )

        id_ab = step_ab._build_identifier()
        id_ba = step_ba._build_identifier()

        nested_ab = id_ab.children["deep_dive_attacks"]
        assert isinstance(nested_ab, list)
        # Sorted by atomic_attack_name, so 'a-first' must precede 'b-second'.
        assert [c.params["name"] for c in nested_ab] == ["a-first", "b-second"]
        assert id_ab == id_ba


@pytest.mark.usefixtures("patch_central_database")
class TestBroadSweepThenDeepDive:
    """End-to-end coverage of the branching scenario and its graph."""

    @staticmethod
    def _build_scenario(
        *,
        sweep_response_text: str,
        scorer_label_for: dict[str, str],
        deep_dive_display_groups: list[str],
    ) -> tuple[
        BroadSweepThenDeepDive,
        MagicMock,
        list[MagicMock],
    ]:
        scorer = _make_outcome_scorer(label_for=scorer_label_for)
        sweep_result = _attack_result(
            conversation_id="sweep-c",
            objective="sweep-obj",
            response_text=sweep_response_text,
        )
        sweep_atomic = _make_atomic_mock(
            name="sweep-atomic",
            display_group="cat-a",
            attack_results=[sweep_result],
        )
        deep_dives = [
            _make_atomic_mock(
                name=f"deep-{i}",
                display_group=group,
                attack_results=[
                    _attack_result(
                        conversation_id=f"deep-c-{i}",
                        objective=f"deep-obj-{i}",
                        response_text=f"deep-text-{i}",
                    )
                ],
            )
            for i, group in enumerate(deep_dive_display_groups)
        ]
        scenario = BroadSweepThenDeepDive(
            sweep_atomic_attack=cast("AtomicAttack", sweep_atomic),
            deep_dive_atomic_attacks=[cast("AtomicAttack", d) for d in deep_dives],
            outcome_scorer=scorer,
            weakness_label=_WEAKNESS_LABEL,
        )
        return scenario, sweep_atomic, deep_dives

    def test_constructor_rejects_empty_deep_dive_list(self) -> None:
        scorer = _make_outcome_scorer()
        sweep = _make_atomic_mock(name="sweep", display_group="cat-a", attack_results=[])
        with pytest.raises(ValueError, match="deep_dive_atomic_attack"):
            BroadSweepThenDeepDive(
                sweep_atomic_attack=cast("AtomicAttack", sweep),
                deep_dive_atomic_attacks=[],
                outcome_scorer=scorer,
            )

    def test_strategy_metadata(self) -> None:
        assert BroadSweepThenDeepDive.get_strategy_class() is BroadSweepThenDeepDiveStrategy
        assert BroadSweepThenDeepDive.get_default_strategy() is BroadSweepThenDeepDiveStrategy.DEFAULT
        assert isinstance(BroadSweepThenDeepDive.default_dataset_config(), DatasetConfiguration)

    def test_baseline_attack_policy_is_forbidden(self) -> None:
        # The branching graph ignores the orchestrator-supplied ``steps`` list, so an
        # implicitly prepended baseline ``AtomicAttack`` would be persisted but never
        # executed. ``Forbidden`` makes the (correct) intent explicit and fails fast
        # if a caller passes ``include_baseline=True``.
        assert BroadSweepThenDeepDive.BASELINE_ATTACK_POLICY is BaselineAttackPolicy.Forbidden

    def test_get_attack_technique_factories_returns_empty_dict(self) -> None:
        # The sweep and deep-dive atomics are supplied directly via the
        # constructor — no registry lookup ever happens during execution.
        # Overriding the base method (which would lazily populate the global
        # AttackTechniqueRegistry singleton via register_scenario_techniques)
        # keeps introspection by waterfall.policy_to_spec side-effect-free.
        scenario, _, _ = self._build_scenario(
            sweep_response_text="safe",
            scorer_label_for={"safe": _SAFE_LABEL},
            deep_dive_display_groups=["cat-a"],
        )
        assert scenario._get_attack_technique_factories() == {}

    def test_get_attack_technique_factories_does_not_mutate_global_registry(self) -> None:
        # Removing the BSTDDive override would silently re-introduce the
        # ``register_scenario_techniques()`` side-effect on the global
        # ``AttackTechniqueRegistry`` singleton. Pin the absence of mutation
        # so future refactors that drop the override fail this test.
        from pyrit.registry.object_registries.attack_technique_registry import AttackTechniqueRegistry

        AttackTechniqueRegistry.reset_instance()
        try:
            before = set(AttackTechniqueRegistry.get_registry_singleton().get_factories().keys())

            scenario, _, _ = self._build_scenario(
                sweep_response_text="safe",
                scorer_label_for={"safe": _SAFE_LABEL},
                deep_dive_display_groups=["cat-a"],
            )
            scenario._get_attack_technique_factories()

            after = set(AttackTechniqueRegistry.get_registry_singleton().get_factories().keys())
            assert after == before, (
                f"BroadSweepThenDeepDive._get_attack_technique_factories() mutated the global "
                f"AttackTechniqueRegistry singleton (added {sorted(after - before)}). The override "
                f"must return {{}} without invoking register_scenario_techniques()."
            )
        finally:
            AttackTechniqueRegistry.reset_instance()

    async def test_get_atomic_attacks_returns_canonical_order(self) -> None:
        scenario, sweep_atomic, deep_dives = self._build_scenario(
            sweep_response_text="safe",
            scorer_label_for={"safe": _SAFE_LABEL},
            deep_dive_display_groups=["cat-a", "cat-b"],
        )
        atomics = await scenario._get_atomic_attacks_async()
        assert atomics[0] is sweep_atomic
        assert atomics[1:] == deep_dives

    def test_build_execution_graph_returns_branching_strategy_graph(self) -> None:
        scenario, _, _ = self._build_scenario(
            sweep_response_text="safe",
            scorer_label_for={"safe": _SAFE_LABEL},
            deep_dive_display_groups=["cat-a"],
        )
        graph = scenario._build_execution_graph()
        assert isinstance(graph, StrategyGraph)
        policy = graph._policy
        assert isinstance(policy, StrategyPolicy)
        assert policy.initial_state == SweepThenDeepDiveState.SWEEPING
        assert SweepThenDeepDiveState.COMPLETE in policy.terminal_states
        assert SweepThenDeepDiveState.ALL_SAFE in policy.terminal_states
        # Both non-terminal states must have an action.
        assert SweepThenDeepDiveState.SWEEPING in policy.actions
        assert SweepThenDeepDiveState.DEEP_DIVING in policy.actions

    def test_build_execution_graph_resets_weak_categories(self) -> None:
        scenario, _, _ = self._build_scenario(
            sweep_response_text="safe",
            scorer_label_for={"safe": _SAFE_LABEL},
            deep_dive_display_groups=["cat-a"],
        )
        scenario._weak_categories = {"stale-from-previous-attempt"}
        scenario._build_execution_graph()
        assert scenario._weak_categories == set()

    async def test_event_loop_with_weakness_drives_deep_dive_to_complete(self) -> None:
        scenario, sweep_atomic, deep_dives = self._build_scenario(
            sweep_response_text="breach-A",
            scorer_label_for={"breach-A": _WEAKNESS_LABEL},
            deep_dive_display_groups=["cat-a", "cat-b"],
        )
        graph = scenario._build_execution_graph()

        results: list[ScenarioStepResult] = [result async for result in graph.event_loop_async()]

        assert graph.current_state == SweepThenDeepDiveState.COMPLETE
        assert len(results) == 2
        # Sweep step ran once.
        sweep_atomic.run_async.assert_called_once()
        # Only the cat-a deep-dive should have been dispatched.
        deep_dives[0].run_async.assert_called_once()
        deep_dives[1].run_async.assert_not_called()
        assert results[1].metadata["dispatched_categories"] == ["cat-a"]
        assert results[1].metadata["skipped_categories"] == ["cat-b"]

    async def test_event_loop_short_circuits_to_all_safe(self) -> None:
        scenario, sweep_atomic, deep_dives = self._build_scenario(
            sweep_response_text="safe-text",
            scorer_label_for={"safe-text": _SAFE_LABEL},
            deep_dive_display_groups=["cat-a", "cat-b"],
        )
        graph = scenario._build_execution_graph()

        results: list[ScenarioStepResult] = [result async for result in graph.event_loop_async()]

        assert graph.current_state == SweepThenDeepDiveState.ALL_SAFE
        # Only the sweep step's result is in history; deep dive never ran.
        assert len(results) == 1
        sweep_atomic.run_async.assert_called_once()
        for d in deep_dives:
            d.run_async.assert_not_called()
        assert results[0].outcome == "found_weaknesses" or results[0].outcome == "all_safe"
        assert results[0].metadata["weak_categories"] == set()

    async def test_weak_categories_propagate_from_sweep_to_deep_dive(self) -> None:
        """The closure-shared set must hand the right categories to the deep dive."""
        scenario, _, deep_dives = self._build_scenario(
            sweep_response_text="breach-A",
            scorer_label_for={"breach-A": _WEAKNESS_LABEL},
            deep_dive_display_groups=["cat-a", "cat-b", "cat-c"],
        )
        graph = scenario._build_execution_graph()

        async for _ in graph.event_loop_async():
            pass

        # The sweep step's display_group is "cat-a", so only cat-a is flagged.
        deep_dives[0].run_async.assert_called_once()
        deep_dives[1].run_async.assert_not_called()
        deep_dives[2].run_async.assert_not_called()
        assert scenario._weak_categories == {"cat-a"}

    async def test_consecutive_runs_reset_state(self) -> None:
        """Re-invoking _build_execution_graph clears prior weak_categories cleanly."""
        scenario, _, _ = self._build_scenario(
            sweep_response_text="safe",
            scorer_label_for={"safe": _SAFE_LABEL},
            deep_dive_display_groups=["cat-a"],
        )
        scenario._weak_categories = {"ghost-category"}
        graph1 = scenario._build_execution_graph()
        assert scenario._weak_categories == set()
        async for _ in graph1.event_loop_async():
            pass

        scenario._weak_categories = {"another-ghost"}
        graph2 = scenario._build_execution_graph()
        assert scenario._weak_categories == set()
        async for _ in graph2.event_loop_async():
            pass
