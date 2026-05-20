# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Tests for the ``TextAdaptive`` scenario."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from pyrit.identifiers import ComponentIdentifier
from pyrit.models import SeedAttackGroup, SeedObjective
from pyrit.prompt_target import PromptTarget
from pyrit.registry.object_registries.attack_technique_registry import AttackTechniqueRegistry
from pyrit.scenario.core.dataset_configuration import DatasetConfiguration
from pyrit.scenario.core.scenario import BaselinePolicy
from pyrit.scenario.scenarios.adaptive.adaptive_step import AdaptiveStep
from pyrit.scenario.scenarios.adaptive.dispatcher import (
    ADAPTIVE_CONTEXT_LABEL,
)
from pyrit.scenario.scenarios.adaptive.selector import (
    GLOBAL_CONTEXT,
    harm_category_context,
)
from pyrit.scenario.scenarios.adaptive.text_adaptive import TextAdaptive
from pyrit.score import TrueFalseScorer

_MOCK_MANY_SHOT_EXAMPLES = [{"question": f"q{i}", "answer": f"a{i}"} for i in range(100)]


def _mock_id(name: str) -> ComponentIdentifier:
    return ComponentIdentifier(class_name=name, class_module="test")


@pytest.fixture
def mock_objective_target() -> MagicMock:
    mock = MagicMock(spec=PromptTarget)
    mock.get_identifier.return_value = _mock_id("MockObjectiveTarget")
    return mock


@pytest.fixture
def mock_objective_scorer() -> MagicMock:
    mock = MagicMock(spec=TrueFalseScorer)
    mock.get_identifier.return_value = _mock_id("MockObjectiveScorer")
    return mock


@pytest.fixture(autouse=True)
def reset_technique_registry():
    """Reset registries and the cached strategy class between tests."""
    from pyrit.registry import TargetRegistry

    AttackTechniqueRegistry.reset_instance()
    TargetRegistry.reset_instance()
    TextAdaptive._cached_strategy_class = None
    yield
    AttackTechniqueRegistry.reset_instance()
    TargetRegistry.reset_instance()
    TextAdaptive._cached_strategy_class = None


@pytest.fixture(autouse=True)
def patch_many_shot_load():
    with patch(
        "pyrit.executor.attack.single_turn.many_shot_jailbreak.load_many_shot_jailbreaking_dataset",
        return_value=_MOCK_MANY_SHOT_EXAMPLES,
    ):
        yield


@pytest.fixture
def mock_runtime_env():
    with patch.dict(
        "os.environ",
        {
            "OPENAI_CHAT_ENDPOINT": "https://test.openai.azure.com/",
            "OPENAI_CHAT_KEY": "test-key",
            "OPENAI_CHAT_MODEL": "gpt-4",
        },
    ):
        yield


def _make_seed_group(*, value: str, harm_categories: list[str] | None = None) -> SeedAttackGroup:
    return SeedAttackGroup(seeds=[SeedObjective(value=value, harm_categories=harm_categories)])


def _make_fake_factory(*, seed_technique=None, adversarial_chat=None) -> MagicMock:
    """Return a stub attack-technique factory that produces a fake ``AttackTechnique``.

    Mocks the surface ``AdaptiveScenario._build_techniques_dict`` consumes
    (``factory.create(...)`` and ``factory.adversarial_chat``).
    """
    fake_technique = MagicMock()
    fake_technique.attack = MagicMock(name="fake-attack-strategy")
    fake_technique.seed_technique = seed_technique
    factory = MagicMock()
    factory.create.return_value = fake_technique
    factory.adversarial_chat = adversarial_chat
    return factory


FIXTURES = ["patch_central_database", "mock_runtime_env"]


@pytest.mark.usefixtures(*FIXTURES)
class TestTextAdaptiveBasics:
    def test_version(self):
        assert TextAdaptive.VERSION == 1

    def test_baseline_forbidden(self):
        assert TextAdaptive.BASELINE_POLICY is BaselinePolicy.Forbidden

    def test_default_dataset_config(self):
        config = TextAdaptive.default_dataset_config()
        assert isinstance(config, DatasetConfiguration)
        assert config.max_dataset_size == 4

    def test_required_datasets_non_empty(self):
        assert len(TextAdaptive.required_datasets()) > 0

    def test_get_strategy_class_is_cached(self):
        cls_a = TextAdaptive.get_strategy_class()
        cls_b = TextAdaptive.get_strategy_class()
        assert cls_a is cls_b

    def test_get_default_strategy(self):
        strat = TextAdaptive.get_default_strategy()
        # The default aggregate must resolve to something runnable.
        assert strat is not None

    @patch("pyrit.scenario.core.scenario.Scenario._get_default_objective_scorer")
    def test_init_stores_adaptive_params(self, mock_get_scorer, mock_objective_scorer):
        mock_get_scorer.return_value = mock_objective_scorer
        scenario = TextAdaptive(
            epsilon=0.4,
            pool_threshold=5,
            max_attempts_per_objective=7,
            seed=42,
        )
        assert scenario._epsilon == 0.4
        assert scenario._pool_threshold == 5
        assert scenario._max_attempts_per_objective == 7
        assert scenario._seed == 42


@pytest.mark.usefixtures(*FIXTURES)
class TestTextAdaptiveAtomicAttacks:
    """Tests for ``_get_atomic_attacks_async`` overriding."""

    async def _build_scenario_and_attacks(
        self,
        *,
        mock_objective_target,
        mock_objective_scorer,
        seed_groups: dict[str, list[SeedAttackGroup]],
        **scenario_kwargs,
    ):
        with patch.object(DatasetConfiguration, "get_seed_attack_groups", return_value=seed_groups):
            scenario = TextAdaptive(
                objective_scorer=mock_objective_scorer,
                **scenario_kwargs,
            )
            await scenario.initialize_async(
                objective_target=mock_objective_target,
                include_baseline=False,
            )
            return scenario, await scenario._get_atomic_attacks_async()

    async def test_one_atomic_per_objective(self, mock_objective_target, mock_objective_scorer):
        groups = {
            "violence": [
                _make_seed_group(value="obj-v1", harm_categories=["violence"]),
                _make_seed_group(value="obj-v2", harm_categories=["violence"]),
            ],
            "hate": [
                _make_seed_group(value="obj-h1", harm_categories=["hate"]),
            ],
        }
        _scenario, attacks = await self._build_scenario_and_attacks(
            mock_objective_target=mock_objective_target,
            mock_objective_scorer=mock_objective_scorer,
            seed_groups=groups,
        )
        assert len(attacks) == 3
        for atomic in attacks:
            # Each atomic carries exactly one seed group.
            assert len(atomic.objectives) == 1

    async def test_atomics_share_one_selector_across_dispatchers(self, mock_objective_target, mock_objective_scorer):
        groups = {
            "violence": [
                _make_seed_group(value="obj-v1", harm_categories=["violence"]),
                _make_seed_group(value="obj-v2", harm_categories=["violence"]),
            ],
        }
        _scenario, attacks = await self._build_scenario_and_attacks(
            mock_objective_target=mock_objective_target,
            mock_objective_scorer=mock_objective_scorer,
            seed_groups=groups,
        )
        # Each objective is now driven by its own AdaptiveStep instance...
        assert all(isinstance(step, AdaptiveStep) for step in attacks)
        assert len({id(step) for step in attacks}) == len(attacks)
        # ...but they all share the same selector so learning is global.
        selectors = {id(step._selector) for step in attacks}
        assert len(selectors) == 1

    async def test_global_context_label_when_using_global_extractor(self, mock_objective_target, mock_objective_scorer):
        groups = {
            "violence": [_make_seed_group(value="obj-1", harm_categories=["violence"])],
            "hate": [_make_seed_group(value="obj-2", harm_categories=["hate"])],
        }
        _scenario, attacks = await self._build_scenario_and_attacks(
            mock_objective_target=mock_objective_target,
            mock_objective_scorer=mock_objective_scorer,
            seed_groups=groups,
        )
        for atomic in attacks:
            assert atomic._memory_labels[ADAPTIVE_CONTEXT_LABEL] == GLOBAL_CONTEXT

    async def test_harm_category_extractor_partitions_labels(self, mock_objective_target, mock_objective_scorer):
        groups = {
            "violence": [_make_seed_group(value="obj-v", harm_categories=["violence"])],
            "hate": [_make_seed_group(value="obj-h", harm_categories=["hate"])],
            "uncat": [_make_seed_group(value="obj-u", harm_categories=None)],
        }
        _scenario, attacks = await self._build_scenario_and_attacks(
            mock_objective_target=mock_objective_target,
            mock_objective_scorer=mock_objective_scorer,
            seed_groups=groups,
            context_extractor=harm_category_context,
        )
        contexts = {atomic._memory_labels[ADAPTIVE_CONTEXT_LABEL] for atomic in attacks}
        # Each objective gets its own context bucket from harm_category_context.
        assert contexts == {"violence", "hate", "_uncategorized"}

    async def test_atomic_names_are_unique(self, mock_objective_target, mock_objective_scorer):
        groups = {
            "violence": [_make_seed_group(value=f"obj-{i}", harm_categories=["violence"]) for i in range(5)],
        }
        _scenario, attacks = await self._build_scenario_and_attacks(
            mock_objective_target=mock_objective_target,
            mock_objective_scorer=mock_objective_scorer,
            seed_groups=groups,
        )
        names = [atomic.atomic_attack_name for atomic in attacks]
        assert len(set(names)) == len(names)

    async def test_display_group_is_dataset_name(self, mock_objective_target, mock_objective_scorer):
        groups = {
            "violence": [_make_seed_group(value="obj-v", harm_categories=["violence"])],
            "hate": [_make_seed_group(value="obj-h", harm_categories=["hate"])],
        }
        _scenario, attacks = await self._build_scenario_and_attacks(
            mock_objective_target=mock_objective_target,
            mock_objective_scorer=mock_objective_scorer,
            seed_groups=groups,
        )
        display_groups = {atomic.display_group for atomic in attacks}
        assert display_groups == {"violence", "hate"}

    async def test_no_usable_techniques_raises(self, mock_objective_target, mock_objective_scorer):
        groups = {"violence": [_make_seed_group(value="obj")]}
        with patch.object(DatasetConfiguration, "get_seed_attack_groups", return_value=groups):
            scenario = TextAdaptive(objective_scorer=mock_objective_scorer)
            await scenario.initialize_async(
                objective_target=mock_objective_target,
                include_baseline=False,
            )
            # Force the factory map to be empty.
            with patch.object(scenario, "_get_attack_technique_factories", return_value={}):
                with pytest.raises(ValueError, match="no usable techniques"):
                    await scenario._get_atomic_attacks_async()

    async def test_techniques_with_seed_technique_are_kept(self, mock_objective_target, mock_objective_scorer):
        """Factories that declare a ``seed_technique`` participate in the pool
        (the old behavior silently dropped them with a warning).
        """
        groups = {"violence": [_make_seed_group(value="obj")]}
        plain_factory = _make_fake_factory(seed_technique=None)
        seeded_factory = _make_fake_factory(seed_technique=MagicMock(name="seed_technique"))

        with (
            patch.object(DatasetConfiguration, "get_seed_attack_groups", return_value=groups),
            patch.object(SeedAttackGroup, "is_compatible_with_technique", return_value=True),
        ):
            scenario = TextAdaptive(objective_scorer=mock_objective_scorer)
            with patch.object(
                scenario,
                "_get_attack_technique_factories",
                return_value={"prompt_sending": plain_factory, "many_shot": seeded_factory},
            ):
                await scenario.initialize_async(
                    objective_target=mock_objective_target,
                    include_baseline=False,
                )
                attacks = scenario._atomic_attacks

        assert len(attacks) == 1
        step = attacks[0]
        assert isinstance(step, AdaptiveStep)
        # Both factories survive; in particular the seeded one is no longer
        # silently dropped.
        assert "prompt_sending" in step._techniques
        assert "many_shot" in step._techniques

    async def test_incompatible_seed_technique_is_filtered_per_objective(
        self, mock_objective_target, mock_objective_scorer
    ):
        """Per-objective candidate pool drops techniques whose seed_technique
        is incompatible with the seed group; compatible techniques remain.
        """
        groups = {"violence": [_make_seed_group(value="obj")]}
        plain_factory = _make_fake_factory(seed_technique=None)
        incompatible_factory = _make_fake_factory(seed_technique=MagicMock(name="incompatible_seed_technique"))

        with (
            patch.object(DatasetConfiguration, "get_seed_attack_groups", return_value=groups),
            patch.object(SeedAttackGroup, "is_compatible_with_technique", return_value=False),
        ):
            scenario = TextAdaptive(objective_scorer=mock_objective_scorer)
            with patch.object(
                scenario,
                "_get_attack_technique_factories",
                return_value={"prompt_sending": plain_factory, "many_shot": incompatible_factory},
            ):
                await scenario.initialize_async(
                    objective_target=mock_objective_target,
                    include_baseline=False,
                )
                attacks = scenario._atomic_attacks

        assert len(attacks) == 1
        step = attacks[0]
        assert isinstance(step, AdaptiveStep)
        # Only the plain technique survives; the seed_technique-bearing one is filtered out
        # because is_compatible_with_technique returned False.
        assert "prompt_sending" in step._techniques
        assert "many_shot" not in step._techniques

    async def test_objective_skipped_when_no_compatible_techniques(
        self, mock_objective_target, mock_objective_scorer, caplog
    ):
        """When every technique requires an incompatible seed_technique, the
        objective is dropped with a warning rather than producing an atomic
        attack with an empty technique pool.
        """
        groups = {
            "violence": [_make_seed_group(value="obj-keep")],
            "hate": [_make_seed_group(value="obj-skip")],
        }
        seeded_factory = _make_fake_factory(seed_technique=MagicMock(name="seed_technique"))

        # is_compatible_with_technique returns True for "obj-keep", False for "obj-skip".
        def _selective_compat(self_group, *, technique):
            return self_group.objective.value == "obj-keep"

        with (
            patch.object(DatasetConfiguration, "get_seed_attack_groups", return_value=groups),
            patch.object(SeedAttackGroup, "is_compatible_with_technique", _selective_compat),
        ):
            scenario = TextAdaptive(objective_scorer=mock_objective_scorer)
            with patch.object(
                scenario,
                "_get_attack_technique_factories",
                return_value={"prompt_sending": seeded_factory},
            ):
                import logging

                with caplog.at_level(logging.WARNING):
                    await scenario.initialize_async(
                        objective_target=mock_objective_target,
                        include_baseline=False,
                    )
                    attacks = scenario._atomic_attacks

        # Only the compatible objective produced an atomic attack.
        assert len(attacks) == 1
        # Skip was logged with the affected objective value.
        assert any("obj-skip" in record.getMessage() for record in caplog.records)


@pytest.mark.usefixtures(*FIXTURES)
class TestTextAdaptiveSelectorRehydration:
    """When resuming, prior dispatch trails should replay into the new selector."""

    def _build_scenario_no_resume_id(self, *, scorer):
        return TextAdaptive(objective_scorer=scorer)

    def test_no_scenario_result_id_is_noop(self, mock_objective_scorer):
        from pyrit.scenario.scenarios.adaptive.selector import AdaptiveTechniqueSelector

        scenario = TextAdaptive(objective_scorer=mock_objective_scorer)
        selector = AdaptiveTechniqueSelector()
        # No scenario_result_id set -> early return, no errors, no replays.
        scenario._rehydrate_selector_from_memory(selector=selector, known_techniques={"a", "b"})
        assert selector.snapshot() == {}

    def test_replays_attempts_from_metadata(self, mock_objective_scorer):
        from pyrit.models import AttackResult
        from pyrit.scenario.scenarios.adaptive.selector import AdaptiveTechniqueSelector

        scenario = TextAdaptive(objective_scorer=mock_objective_scorer, scenario_result_id="rid")

        prior_result = MagicMock()
        prior_result.attack_results = {
            "adaptive_violence_o1": [
                AttackResult(
                    conversation_id="c1",
                    objective="o1",
                    metadata={
                        "adaptive_attempts": [
                            {"technique": "a", "outcome": "failure"},
                            {"technique": "b", "outcome": "success"},
                        ],
                        "adaptive_context": "violence",
                    },
                ),
            ],
            "adaptive_hate_o2": [
                AttackResult(
                    conversation_id="c2",
                    objective="o2",
                    metadata={
                        "adaptive_attempts": [{"technique": "a", "outcome": "success"}],
                        "adaptive_context": "hate",
                    },
                ),
            ],
        }

        selector = AdaptiveTechniqueSelector()
        with patch.object(scenario._memory, "get_scenario_results", return_value=[prior_result]):
            scenario._rehydrate_selector_from_memory(selector=selector, known_techniques={"a", "b"})

        # Trails replayed verbatim into the per-context table.
        assert selector.counts(context="violence", technique="a") == (0, 1)
        assert selector.counts(context="violence", technique="b") == (1, 1)
        assert selector.counts(context="hate", technique="a") == (1, 1)

    def test_skips_unknown_techniques(self, mock_objective_scorer):
        from pyrit.models import AttackResult
        from pyrit.scenario.scenarios.adaptive.selector import AdaptiveTechniqueSelector

        scenario = TextAdaptive(objective_scorer=mock_objective_scorer, scenario_result_id="rid")
        prior_result = MagicMock()
        prior_result.attack_results = {
            "x": [
                AttackResult(
                    conversation_id="c1",
                    objective="o1",
                    metadata={
                        "adaptive_attempts": [
                            {"technique": "removed_technique", "outcome": "success"},
                            {"technique": "a", "outcome": "failure"},
                        ],
                        "adaptive_context": "ctx",
                    },
                ),
            ],
        }

        selector = AdaptiveTechniqueSelector()
        with patch.object(scenario._memory, "get_scenario_results", return_value=[prior_result]):
            scenario._rehydrate_selector_from_memory(selector=selector, known_techniques={"a"})

        # Only the known technique was recorded.
        assert selector.counts(context="ctx", technique="a") == (0, 1)
        assert selector.counts(context="ctx", technique="removed_technique") == (0, 0)

    def test_ignores_results_without_adaptive_metadata(self, mock_objective_scorer):
        from pyrit.models import AttackResult
        from pyrit.scenario.scenarios.adaptive.selector import AdaptiveTechniqueSelector

        scenario = TextAdaptive(objective_scorer=mock_objective_scorer, scenario_result_id="rid")
        prior_result = MagicMock()
        prior_result.attack_results = {
            "baseline": [AttackResult(conversation_id="c", objective="o", metadata={})],
        }

        selector = AdaptiveTechniqueSelector()
        with patch.object(scenario._memory, "get_scenario_results", return_value=[prior_result]):
            scenario._rehydrate_selector_from_memory(selector=selector, known_techniques={"a"})
        assert selector.snapshot() == {}

    def test_memory_load_failure_is_swallowed(self, mock_objective_scorer):
        from pyrit.scenario.scenarios.adaptive.selector import AdaptiveTechniqueSelector

        scenario = TextAdaptive(objective_scorer=mock_objective_scorer, scenario_result_id="rid")

        selector = AdaptiveTechniqueSelector()
        with patch.object(scenario._memory, "get_scenario_results", side_effect=RuntimeError("db down")):
            # Must not raise; selector remains empty.
            scenario._rehydrate_selector_from_memory(selector=selector, known_techniques={"a"})
        assert selector.snapshot() == {}


@pytest.mark.usefixtures(*FIXTURES)
class TestTextAdaptiveBaselinePolicy:
    async def test_initialize_async_rejects_explicit_baseline(self, mock_objective_target, mock_objective_scorer):
        groups = {"violence": [_make_seed_group(value="obj", harm_categories=["violence"])]}
        with patch.object(DatasetConfiguration, "get_seed_attack_groups", return_value=groups):
            scenario = TextAdaptive(objective_scorer=mock_objective_scorer)
            with pytest.raises(ValueError):
                await scenario.initialize_async(
                    objective_target=mock_objective_target,
                    include_baseline=True,
                )
