# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import random
from unittest.mock import AsyncMock, MagicMock

import pytest

from pyrit.executor.attack.core.attack_parameters import AttackParameters
from pyrit.models import AttackOutcome, AttackResult, SeedAttackGroup, SeedObjective
from pyrit.scenario.scenarios.adaptive.dispatcher import (
    ADAPTIVE_ATTEMPT_LABEL,
    ADAPTIVE_CONTEXT_LABEL,
    ADAPTIVE_TECHNIQUE_LABEL,
    AdaptiveDispatchAttack,
    AdaptiveDispatchContext,
    TechniqueBundle,
)
from pyrit.scenario.scenarios.adaptive.selector import (
    GLOBAL_CONTEXT,
    AdaptiveTechniqueSelector,
)

# ``AdaptiveDispatchAttack`` is deprecated as of 0.15.0 in favor of
# ``AdaptiveStep`` (see ``test_adaptive_step.py``). Suppress the per-instantiation
# DeprecationWarning here so the regression suite for the dispatcher's existing
# behavior stays clean during the deprecation window. The warning is asserted
# explicitly in a dedicated test below.
pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def _make_bundle(*, name: str, outcomes: list[AttackOutcome], seed_technique=None) -> TechniqueBundle:
    """Build a TechniqueBundle whose attack stub yields the given outcomes in order.

    The dispatcher routes execution through ``_run_inner_attack_async``; tests
    patch that method directly so we only need a placeholder attack here.
    """
    attack = MagicMock(name=f"attack-{name}")
    attack._outcomes = outcomes
    attack._name = name
    return TechniqueBundle(attack=attack, seed_technique=seed_technique)


def _make_context(*, objective: str = "obj", labels: dict[str, str] | None = None) -> AdaptiveDispatchContext:
    return AdaptiveDispatchContext(params=AttackParameters(objective=objective, memory_labels=labels or {}))


def _patch_inner(
    *,
    dispatcher: AdaptiveDispatchAttack,
    bundles: dict[str, TechniqueBundle],
) -> AsyncMock:
    """Replace ``_run_inner_attack_async`` with a stub backed by per-bundle outcomes.

    Returns the AsyncMock so tests can introspect call history (kwargs include
    ``bundle`` and ``attempt_labels``).
    """
    # Each call consumes one outcome from the chosen bundle's deque.
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
    dispatcher._run_inner_attack_async = inner_mock  # type: ignore[method-assign]
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


class TestInit:
    @pytest.mark.usefixtures("patch_central_database")
    def test_init_rejects_empty_techniques(self, target, selector, seed_group):
        with pytest.raises(ValueError, match="techniques"):
            AdaptiveDispatchAttack(
                objective_target=target,
                techniques={},
                selector=selector,
                seed_group=seed_group,
            )

    @pytest.mark.parametrize("bad_max", [0, -1])
    @pytest.mark.usefixtures("patch_central_database")
    def test_init_rejects_invalid_max_attempts(self, target, selector, seed_group, bad_max):
        with pytest.raises(ValueError, match="max_attempts_per_objective"):
            AdaptiveDispatchAttack(
                objective_target=target,
                techniques={"a": _make_bundle(name="a", outcomes=[AttackOutcome.SUCCESS])},
                selector=selector,
                seed_group=seed_group,
                max_attempts_per_objective=bad_max,
            )


@pytest.mark.usefixtures("patch_central_database")
class TestPerform:
    async def test_stops_on_first_success(self, target, selector, seed_group):
        bundles = {
            "a": _make_bundle(name="a", outcomes=[AttackOutcome.SUCCESS]),
            "b": _make_bundle(name="b", outcomes=[AttackOutcome.SUCCESS]),
        }
        dispatcher = AdaptiveDispatchAttack(
            objective_target=target,
            techniques=bundles,
            selector=selector,
            seed_group=seed_group,
            max_attempts_per_objective=5,
        )
        inner = _patch_inner(dispatcher=dispatcher, bundles=bundles)

        result = await dispatcher._perform_async(context=_make_context())

        assert result.outcome == AttackOutcome.SUCCESS
        assert inner.call_count == 1

    async def test_retries_until_max_attempts_on_failure(self, target, selector, seed_group):
        bundles = {
            "a": _make_bundle(name="a", outcomes=[AttackOutcome.FAILURE] * 3),
            "b": _make_bundle(name="b", outcomes=[AttackOutcome.FAILURE] * 3),
        }
        dispatcher = AdaptiveDispatchAttack(
            objective_target=target,
            techniques=bundles,
            selector=selector,
            seed_group=seed_group,
            max_attempts_per_objective=3,
        )
        inner = _patch_inner(dispatcher=dispatcher, bundles=bundles)

        result = await dispatcher._perform_async(context=_make_context())

        assert result.outcome == AttackOutcome.FAILURE
        assert inner.call_count == 3

    async def test_updates_selector_on_each_attempt(self, target, selector, seed_group):
        bundles = {
            "a": _make_bundle(name="a", outcomes=[AttackOutcome.FAILURE, AttackOutcome.SUCCESS]),
            "b": _make_bundle(name="b", outcomes=[AttackOutcome.SUCCESS]),
        }
        dispatcher = AdaptiveDispatchAttack(
            objective_target=target,
            techniques=bundles,
            selector=selector,
            seed_group=seed_group,
            max_attempts_per_objective=3,
        )
        inner = _patch_inner(dispatcher=dispatcher, bundles=bundles)

        await dispatcher._perform_async(context=_make_context())

        total_attempts = sum(selector.counts(context=GLOBAL_CONTEXT, technique=t)[1] for t in ("a", "b"))
        assert total_attempts == inner.call_count

    async def test_passes_attempt_labels_to_inner(self, target, selector, seed_group):
        bundles = {"a": _make_bundle(name="a", outcomes=[AttackOutcome.SUCCESS])}
        dispatcher = AdaptiveDispatchAttack(
            objective_target=target,
            techniques=bundles,
            selector=selector,
            seed_group=seed_group,
        )
        inner = _patch_inner(dispatcher=dispatcher, bundles=bundles)

        await dispatcher._perform_async(context=_make_context(labels={"foo": "bar"}))

        labels = inner.call_args.kwargs["attempt_labels"]
        assert labels["foo"] == "bar"  # caller labels preserved
        assert labels[ADAPTIVE_TECHNIQUE_LABEL] == "a"
        assert labels[ADAPTIVE_ATTEMPT_LABEL] == "1"

    async def test_uses_adaptive_context_from_label(self, target, selector, seed_group):
        # Two techniques; one has been heavily rewarded under context "violence" only.
        bundles = {
            "a": _make_bundle(name="a", outcomes=[AttackOutcome.SUCCESS]),
            "b": _make_bundle(name="b", outcomes=[AttackOutcome.SUCCESS]),
        }
        for _ in range(5):
            selector.record_outcome(context="violence", technique="b", success=True)
        for _ in range(5):
            selector.record_outcome(context="violence", technique="a", success=False)

        dispatcher = AdaptiveDispatchAttack(
            objective_target=target,
            techniques=bundles,
            selector=selector,
            seed_group=seed_group,
        )
        inner = _patch_inner(dispatcher=dispatcher, bundles=bundles)
        ctx = _make_context(labels={ADAPTIVE_CONTEXT_LABEL: "violence"})
        await dispatcher._perform_async(context=ctx)

        # Exploit should have picked "b" first.
        chosen_bundle = inner.call_args.kwargs["bundle"]
        assert chosen_bundle is bundles["b"]

    async def test_falls_back_to_global_context_when_label_missing(self, target, selector, seed_group):
        bundles = {"a": _make_bundle(name="a", outcomes=[AttackOutcome.SUCCESS])}
        dispatcher = AdaptiveDispatchAttack(
            objective_target=target,
            techniques=bundles,
            selector=selector,
            seed_group=seed_group,
        )
        _patch_inner(dispatcher=dispatcher, bundles=bundles)
        await dispatcher._perform_async(context=_make_context(labels={}))

        # The global context bucket received the update.
        assert selector.counts(context=GLOBAL_CONTEXT, technique="a") == (1, 1)

    async def test_metadata_records_adaptive_trail(self, target, selector, seed_group):
        bundles = {"a": _make_bundle(name="a", outcomes=[AttackOutcome.FAILURE, AttackOutcome.SUCCESS])}
        dispatcher = AdaptiveDispatchAttack(
            objective_target=target,
            techniques=bundles,
            selector=selector,
            seed_group=seed_group,
            max_attempts_per_objective=3,
        )
        _patch_inner(dispatcher=dispatcher, bundles=bundles)
        result = await dispatcher._perform_async(context=_make_context())

        trail = result.metadata["adaptive_attempts"]
        assert trail == [
            {"technique": "a", "outcome": "failure"},
            {"technique": "a", "outcome": "success"},
        ]
        assert result.metadata["adaptive_context"] == GLOBAL_CONTEXT

    async def test_returns_fresh_result_distinct_from_inner(self, target, selector, seed_group):
        # The dispatcher must NOT return the inner attack's ``AttackResult``
        # instance — doing so would cause a duplicate-PK insert when both the
        # inner and the dispatcher's ``execute_async`` post-execute hooks try
        # to persist the same row. Verify the returned result has a fresh
        # ``attack_result_id`` while preserving the inner's identifying fields
        # and stamping the dispatch trail.
        bundles = {"a": _make_bundle(name="a", outcomes=[AttackOutcome.SUCCESS])}
        dispatcher = AdaptiveDispatchAttack(
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

        dispatcher._run_inner_attack_async = AsyncMock(side_effect=_spy)  # type: ignore[method-assign]

        result = await dispatcher._perform_async(context=_make_context())

        assert len(inner_ids) == 1
        assert result.attack_result_id != inner_ids[0]
        assert result.conversation_id  # carried over from inner
        assert result.outcome == AttackOutcome.SUCCESS
        assert result.metadata["adaptive_attempts"] == [{"technique": "a", "outcome": "success"}]
        assert result.metadata["adaptive_context"] == GLOBAL_CONTEXT


@pytest.mark.usefixtures("patch_central_database")
class TestValidate:
    @pytest.mark.parametrize("bad_objective", ["", "   ", "\n\t"])
    def test_validate_rejects_empty_objective(self, target, selector, seed_group, bad_objective):
        dispatcher = AdaptiveDispatchAttack(
            objective_target=target,
            techniques={"a": _make_bundle(name="a", outcomes=[AttackOutcome.SUCCESS])},
            selector=selector,
            seed_group=seed_group,
        )
        with pytest.raises(ValueError, match="objective"):
            dispatcher._validate_context(context=_make_context(objective=bad_objective))

    def test_validate_accepts_normal_objective(self, target, selector, seed_group):
        dispatcher = AdaptiveDispatchAttack(
            objective_target=target,
            techniques={"a": _make_bundle(name="a", outcomes=[AttackOutcome.SUCCESS])},
            selector=selector,
            seed_group=seed_group,
        )
        # Does not raise.
        dispatcher._validate_context(context=_make_context(objective="ok"))


@pytest.mark.usefixtures("patch_central_database")
@pytest.mark.filterwarnings("default::DeprecationWarning")
class TestDeprecation:
    def test_instantiation_emits_deprecation_warning(self, target, selector, seed_group):
        with pytest.warns(DeprecationWarning, match="AdaptiveDispatchAttack.*AdaptiveStep"):
            AdaptiveDispatchAttack(
                objective_target=target,
                techniques={"a": _make_bundle(name="a", outcomes=[AttackOutcome.SUCCESS])},
                selector=selector,
                seed_group=seed_group,
            )
