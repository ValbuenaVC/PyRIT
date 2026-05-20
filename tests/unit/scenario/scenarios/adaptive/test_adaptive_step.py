# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Tests for ``AdaptiveStep`` — the StrategyGraph-native replacement for AdaptiveDispatchAttack."""

from __future__ import annotations

import random
from unittest.mock import AsyncMock, MagicMock

import pytest

from pyrit.models import AttackOutcome, AttackResult, SeedAttackGroup, SeedObjective
from pyrit.scenario.core.atomic_attack import AtomicAttack
from pyrit.scenario.core.scenario_step import ScenarioStep, ScenarioStepResult
from pyrit.scenario.scenarios.adaptive.adaptive_step import AdaptiveStep
from pyrit.scenario.scenarios.adaptive.dispatcher import (
    ADAPTIVE_ATTEMPT_LABEL,
    ADAPTIVE_CONTEXT_LABEL,
    ADAPTIVE_TECHNIQUE_LABEL,
    TechniqueBundle,
)
from pyrit.scenario.scenarios.adaptive.selector import (
    GLOBAL_CONTEXT,
    AdaptiveTechniqueSelector,
)


def _make_bundle(*, name: str, outcomes: list[AttackOutcome], seed_technique=None) -> TechniqueBundle:
    """Build a TechniqueBundle whose attack stub yields the given outcomes in order.

    The step routes execution through ``_run_inner_attack_async``; tests
    patch that method directly so we only need a placeholder attack here.
    """
    attack = MagicMock(name=f"attack-{name}")
    attack._outcomes = outcomes
    attack._name = name
    return TechniqueBundle(attack=attack, seed_technique=seed_technique)


def _patch_inner(
    *,
    step: AdaptiveStep,
    bundles: dict[str, TechniqueBundle],
) -> AsyncMock:
    """Replace ``_run_inner_attack_async`` with a stub backed by per-bundle outcomes.

    Returns the AsyncMock so tests can introspect call history (kwargs include
    ``bundle`` and ``attempt_labels``).
    """
    name_for_attack = {id(b.attack): name for name, b in bundles.items()}
    counters: dict[str, int] = dict.fromkeys(bundles, 0)

    async def _stub(*, bundle: TechniqueBundle, attempt_labels: dict[str, str]) -> AttackResult:
        name = name_for_attack[id(bundle.attack)]
        idx = counters[name]
        counters[name] = idx + 1
        outcome = bundle.attack._outcomes[idx]
        return AttackResult(
            conversation_id=f"conv-{name}-{idx}",
            objective="obj",
            outcome=outcome,
        )

    inner_mock = AsyncMock(side_effect=_stub)
    step._run_inner_attack_async = inner_mock  # type: ignore[method-assign]
    return inner_mock


@pytest.fixture
def selector() -> AdaptiveTechniqueSelector:
    # epsilon=0 makes selection deterministic given the table.
    return AdaptiveTechniqueSelector(epsilon=0.0, pool_threshold=1, rng=random.Random(0))


@pytest.fixture
def target() -> MagicMock:
    return MagicMock(name="objective_target")


@pytest.fixture
def seed_group() -> SeedAttackGroup:
    return SeedAttackGroup(seeds=[SeedObjective(value="obj")])


@pytest.mark.usefixtures("patch_central_database")
class TestInit:
    def test_init_rejects_empty_techniques(self, target, selector, seed_group):
        with pytest.raises(ValueError, match="techniques"):
            AdaptiveStep(
                atomic_attack_name="step-1",
                objective_target=target,
                techniques={},
                selector=selector,
                seed_group=seed_group,
            )

    @pytest.mark.parametrize("bad_max", [0, -1])
    def test_init_rejects_invalid_max_attempts(self, target, selector, seed_group, bad_max):
        with pytest.raises(ValueError, match="max_attempts_per_objective"):
            AdaptiveStep(
                atomic_attack_name="step-1",
                objective_target=target,
                techniques={"a": _make_bundle(name="a", outcomes=[AttackOutcome.SUCCESS])},
                selector=selector,
                seed_group=seed_group,
                max_attempts_per_objective=bad_max,
            )

    def test_display_group_defaults_to_name(self, target, selector, seed_group):
        step = AdaptiveStep(
            atomic_attack_name="step-x",
            objective_target=target,
            techniques={"a": _make_bundle(name="a", outcomes=[AttackOutcome.SUCCESS])},
            selector=selector,
            seed_group=seed_group,
        )
        assert step.display_group == "step-x"

    def test_explicit_display_group_overrides_default(self, target, selector, seed_group):
        step = AdaptiveStep(
            atomic_attack_name="step-x",
            display_group="my-group",
            objective_target=target,
            techniques={"a": _make_bundle(name="a", outcomes=[AttackOutcome.SUCCESS])},
            selector=selector,
            seed_group=seed_group,
        )
        assert step.display_group == "my-group"

    def test_outputs_are_success_and_exhausted(self, target, selector, seed_group):
        step = AdaptiveStep(
            atomic_attack_name="step-x",
            objective_target=target,
            techniques={"a": _make_bundle(name="a", outcomes=[AttackOutcome.SUCCESS])},
            selector=selector,
            seed_group=seed_group,
        )
        assert step.outputs == ["success", "exhausted"]
        assert step.name == "step-x"


@pytest.mark.usefixtures("patch_central_database")
class TestAtomicAttackParity:
    """AdaptiveStep must expose AtomicAttack-like attributes for the resume filter."""

    def test_objectives_drawn_from_seed_group(self, target, selector, seed_group):
        step = AdaptiveStep(
            atomic_attack_name="step",
            objective_target=target,
            techniques={"a": _make_bundle(name="a", outcomes=[AttackOutcome.SUCCESS])},
            selector=selector,
            seed_group=seed_group,
        )
        assert step.objectives == ["obj"]

    def test_seed_groups_is_single_element_list(self, target, selector, seed_group):
        step = AdaptiveStep(
            atomic_attack_name="step",
            objective_target=target,
            techniques={"a": _make_bundle(name="a", outcomes=[AttackOutcome.SUCCESS])},
            selector=selector,
            seed_group=seed_group,
        )
        assert step.seed_groups == [seed_group]

    def test_filter_seed_groups_by_objectives_is_noop_when_matched(self, target, selector, seed_group):
        step = AdaptiveStep(
            atomic_attack_name="step",
            objective_target=target,
            techniques={"a": _make_bundle(name="a", outcomes=[AttackOutcome.SUCCESS])},
            selector=selector,
            seed_group=seed_group,
        )
        # Should not raise; bound seed group's objective ("obj") is in the list.
        step.filter_seed_groups_by_objectives(remaining_objectives=["obj"])
        assert step.seed_groups == [seed_group]


@pytest.mark.usefixtures("patch_central_database")
class TestProcess:
    async def test_stops_on_first_success(self, target, selector, seed_group):
        bundles = {
            "a": _make_bundle(name="a", outcomes=[AttackOutcome.SUCCESS]),
            "b": _make_bundle(name="b", outcomes=[AttackOutcome.SUCCESS]),
        }
        step = AdaptiveStep(
            atomic_attack_name="step",
            objective_target=target,
            techniques=bundles,
            selector=selector,
            seed_group=seed_group,
            max_attempts_per_objective=5,
        )
        inner = _patch_inner(step=step, bundles=bundles)

        result = await step.process_async()

        assert isinstance(result, ScenarioStepResult)
        assert result.outcome == "success"
        assert inner.call_count == 1
        assert len(result.attack_results) == 1
        assert result.attack_results[0].outcome == AttackOutcome.SUCCESS

    async def test_retries_until_max_attempts_on_failure_emits_exhausted(self, target, selector, seed_group):
        bundles = {
            "a": _make_bundle(name="a", outcomes=[AttackOutcome.FAILURE] * 3),
            "b": _make_bundle(name="b", outcomes=[AttackOutcome.FAILURE] * 3),
        }
        step = AdaptiveStep(
            atomic_attack_name="step",
            objective_target=target,
            techniques=bundles,
            selector=selector,
            seed_group=seed_group,
            max_attempts_per_objective=3,
        )
        inner = _patch_inner(step=step, bundles=bundles)

        result = await step.process_async()

        assert result.outcome == "exhausted"
        assert inner.call_count == 3

    async def test_updates_selector_on_each_attempt(self, target, selector, seed_group):
        bundles = {
            "a": _make_bundle(name="a", outcomes=[AttackOutcome.FAILURE, AttackOutcome.SUCCESS]),
            "b": _make_bundle(name="b", outcomes=[AttackOutcome.SUCCESS]),
        }
        step = AdaptiveStep(
            atomic_attack_name="step",
            objective_target=target,
            techniques=bundles,
            selector=selector,
            seed_group=seed_group,
            max_attempts_per_objective=3,
        )
        inner = _patch_inner(step=step, bundles=bundles)

        await step.process_async()

        total_attempts = sum(selector.counts(context=GLOBAL_CONTEXT, technique=t)[1] for t in ("a", "b"))
        assert total_attempts == inner.call_count

    async def test_passes_attempt_labels_to_inner(self, target, selector, seed_group):
        bundles = {"a": _make_bundle(name="a", outcomes=[AttackOutcome.SUCCESS])}
        step = AdaptiveStep(
            atomic_attack_name="step",
            objective_target=target,
            techniques=bundles,
            selector=selector,
            seed_group=seed_group,
            memory_labels={"foo": "bar"},
        )
        inner = _patch_inner(step=step, bundles=bundles)

        await step.process_async()

        labels = inner.call_args.kwargs["attempt_labels"]
        assert labels["foo"] == "bar"  # caller labels preserved
        assert labels[ADAPTIVE_TECHNIQUE_LABEL] == "a"
        assert labels[ADAPTIVE_ATTEMPT_LABEL] == "1"

    async def test_uses_per_step_adaptive_context(self, target, selector, seed_group):
        # Two techniques; "b" has been heavily rewarded under context "violence".
        bundles = {
            "a": _make_bundle(name="a", outcomes=[AttackOutcome.SUCCESS]),
            "b": _make_bundle(name="b", outcomes=[AttackOutcome.SUCCESS]),
        }
        for _ in range(5):
            selector.record_outcome(context="violence", technique="b", success=True)
        for _ in range(5):
            selector.record_outcome(context="violence", technique="a", success=False)

        step = AdaptiveStep(
            atomic_attack_name="step",
            objective_target=target,
            techniques=bundles,
            selector=selector,
            seed_group=seed_group,
            adaptive_context="violence",
        )
        inner = _patch_inner(step=step, bundles=bundles)
        await step.process_async()

        # Exploit should have picked "b" first.
        chosen_bundle = inner.call_args.kwargs["bundle"]
        assert chosen_bundle is bundles["b"]

    async def test_default_context_is_global(self, target, selector, seed_group):
        bundles = {"a": _make_bundle(name="a", outcomes=[AttackOutcome.SUCCESS])}
        step = AdaptiveStep(
            atomic_attack_name="step",
            objective_target=target,
            techniques=bundles,
            selector=selector,
            seed_group=seed_group,
        )
        _patch_inner(step=step, bundles=bundles)
        await step.process_async()

        # The global context bucket received the update.
        assert selector.counts(context=GLOBAL_CONTEXT, technique="a") == (1, 1)

    async def test_metadata_records_adaptive_trail_on_step_result_and_attack_result(self, target, selector, seed_group):
        bundles = {"a": _make_bundle(name="a", outcomes=[AttackOutcome.FAILURE, AttackOutcome.SUCCESS])}
        step = AdaptiveStep(
            atomic_attack_name="step-trail",
            objective_target=target,
            techniques=bundles,
            selector=selector,
            seed_group=seed_group,
            max_attempts_per_objective=3,
        )
        _patch_inner(step=step, bundles=bundles)
        result = await step.process_async()

        expected_trail = [
            {"technique": "a", "outcome": "failure"},
            {"technique": "a", "outcome": "success"},
        ]
        # Trail surfaces on both ScenarioStepResult.metadata (for the orchestrator
        # / step_identifier) and on the wrapped AttackResult.metadata (for the
        # persisted row that downstream consumers like the rehydration path read).
        assert result.metadata["adaptive_attempts"] == expected_trail
        assert result.metadata["adaptive_context"] == GLOBAL_CONTEXT
        assert result.metadata["step_name"] == "step-trail"
        assert result.attack_results[0].metadata["adaptive_attempts"] == expected_trail
        assert result.attack_results[0].metadata["adaptive_context"] == GLOBAL_CONTEXT

    async def test_returns_fresh_attack_result_distinct_from_inner(self, target, selector, seed_group):
        # The step must NOT return the inner attack's ``AttackResult`` instance —
        # doing so would cause a duplicate-PK insert when both the inner attack's
        # and the scenario's post-execute persistence paths try to write the same
        # row. Verify the wrapped result has a fresh ``attack_result_id`` while
        # preserving the inner's identifying fields and stamping the trail.
        bundles = {"a": _make_bundle(name="a", outcomes=[AttackOutcome.SUCCESS])}
        step = AdaptiveStep(
            atomic_attack_name="step",
            objective_target=target,
            techniques=bundles,
            selector=selector,
            seed_group=seed_group,
        )
        inner_ids: list[str] = []

        async def _spy(*, bundle, attempt_labels):
            inner_result = AttackResult(
                conversation_id="conv-a-0",
                objective="obj",
                outcome=AttackOutcome.SUCCESS,
            )
            inner_ids.append(inner_result.attack_result_id)
            return inner_result

        step._run_inner_attack_async = AsyncMock(side_effect=_spy)  # type: ignore[method-assign]

        result = await step.process_async()
        outer = result.attack_results[0]

        assert len(inner_ids) == 1
        assert outer.attack_result_id != inner_ids[0]
        assert outer.conversation_id == "conv-a-0"
        assert outer.outcome == AttackOutcome.SUCCESS
        assert outer.metadata["adaptive_attempts"] == [{"technique": "a", "outcome": "success"}]
        assert outer.metadata["adaptive_context"] == GLOBAL_CONTEXT


@pytest.mark.usefixtures("patch_central_database")
class TestIdentifier:
    def test_identifier_nests_technique_identifiers(self, target, selector, seed_group):
        bundle_a = _make_bundle(name="a", outcomes=[AttackOutcome.SUCCESS])
        bundle_b = _make_bundle(name="b", outcomes=[AttackOutcome.SUCCESS])
        # Stub each technique's identifier so the call surface is observable.
        from pyrit.identifiers import ComponentIdentifier

        bundle_a.attack.get_identifier.return_value = ComponentIdentifier(class_name="A", class_module="test")
        bundle_b.attack.get_identifier.return_value = ComponentIdentifier(class_name="B", class_module="test")
        step = AdaptiveStep(
            atomic_attack_name="step-id",
            objective_target=target,
            techniques={"a": bundle_a, "b": bundle_b},
            selector=selector,
            seed_group=seed_group,
            max_attempts_per_objective=5,
            adaptive_context="violence",
        )

        identifier = step.get_identifier()
        # Params capture the step's behavioral identity.
        assert identifier.params["atomic_attack_name"] == "step-id"
        assert identifier.params["outputs"] == ["success", "exhausted"]
        assert identifier.params["max_attempts_per_objective"] == 5
        assert identifier.params["adaptive_context"] == "violence"
        # Children nest each technique's identifier under "techniques",
        # sorted by technique name for deterministic hash stability.
        nested = identifier.children["techniques"]
        assert isinstance(nested, list)
        assert [child.class_name for child in nested] == ["A", "B"]

    def test_identifier_is_stable_under_reversed_technique_order(self, target, selector, seed_group):
        # The _build_identifier sort key (`sorted(self._techniques.items())`)
        # must collapse reversed input dicts to the same identifier, otherwise
        # step_identifier hashes would drift between runs that pick the same
        # techniques in different insertion orders.
        from pyrit.identifiers import ComponentIdentifier

        bundle_a = _make_bundle(name="a", outcomes=[AttackOutcome.SUCCESS])
        bundle_b = _make_bundle(name="b", outcomes=[AttackOutcome.SUCCESS])
        bundle_a.attack.get_identifier.return_value = ComponentIdentifier(class_name="A", class_module="test")
        bundle_b.attack.get_identifier.return_value = ComponentIdentifier(class_name="B", class_module="test")

        def _make(techniques):
            return AdaptiveStep(
                atomic_attack_name="step-id",
                objective_target=target,
                techniques=techniques,
                selector=selector,
                seed_group=seed_group,
                max_attempts_per_objective=5,
                adaptive_context="violence",
            )

        forward = _make({"a": bundle_a, "b": bundle_b}).get_identifier()
        reversed_ = _make({"b": bundle_b, "a": bundle_a}).get_identifier()
        assert forward == reversed_


@pytest.mark.usefixtures("patch_central_database")
class TestScenarioStepIdentity:
    """Phase 6b invariant: AdaptiveStep is a ScenarioStep, NOT an AtomicAttack."""

    def test_is_scenario_step_not_atomic_attack(self, target, selector, seed_group):
        step = AdaptiveStep(
            atomic_attack_name="step",
            objective_target=target,
            techniques={"a": _make_bundle(name="a", outcomes=[AttackOutcome.SUCCESS])},
            selector=selector,
            seed_group=seed_group,
        )
        assert isinstance(step, ScenarioStep)
        assert not isinstance(step, AtomicAttack)

    def test_name_aliases_atomic_attack_name_for_resume_bookkeeping(self, target, selector, seed_group):
        # The orchestrator's resume filter reads ``step.name`` uniformly across
        # step types; AdaptiveStep must alias it to atomic_attack_name without
        # subclassing AtomicAttack.
        step = AdaptiveStep(
            atomic_attack_name="step-resume-key",
            objective_target=target,
            techniques={"a": _make_bundle(name="a", outcomes=[AttackOutcome.SUCCESS])},
            selector=selector,
            seed_group=seed_group,
        )
        assert step.name == "step-resume-key"
        assert step.name == step.atomic_attack_name


@pytest.mark.usefixtures("patch_central_database")
class TestPersistAdaptiveContextLabel:
    """The scenario stamps ADAPTIVE_CONTEXT_LABEL into memory_labels at construction."""

    async def test_context_label_round_trips_through_attempt_labels(self, target, selector, seed_group):
        bundles = {"a": _make_bundle(name="a", outcomes=[AttackOutcome.SUCCESS])}
        step = AdaptiveStep(
            atomic_attack_name="step",
            objective_target=target,
            techniques=bundles,
            selector=selector,
            seed_group=seed_group,
            memory_labels={ADAPTIVE_CONTEXT_LABEL: "violence"},
            adaptive_context="violence",
        )
        inner = _patch_inner(step=step, bundles=bundles)
        await step.process_async()

        labels = inner.call_args.kwargs["attempt_labels"]
        assert labels[ADAPTIVE_CONTEXT_LABEL] == "violence"
