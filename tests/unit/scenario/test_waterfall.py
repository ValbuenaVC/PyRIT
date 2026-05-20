# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Phase 8a — coverage for the forward waterfall in ``pyrit.scenario.core.waterfall``.

Pins:
- ``policy_to_spec`` extracts ``AttackTechniqueSpec`` instances from a
  configured scenario via the factory's ``source_spec`` attribute.
- ``spec_to_enum`` resolves specs to ``ScenarioStrategy`` members of a
  scenario class, returning ``None`` when any spec doesn't match.
- Both functions degrade gracefully (return ``[]`` or ``None``) for
  scenarios that don't use the technique registry pattern.
"""

from __future__ import annotations

from typing import ClassVar, cast
from unittest.mock import MagicMock

import pytest

from pyrit.executor.attack.single_turn.prompt_sending import PromptSendingAttack
from pyrit.identifiers import ComponentIdentifier
from pyrit.registry.object_registries.attack_technique_registry import (
    AttackTechniqueRegistry,
    AttackTechniqueSpec,
)
from pyrit.scenario.core.attack_technique_factory import AttackTechniqueFactory
from pyrit.scenario.core.dataset_configuration import DatasetConfiguration
from pyrit.scenario.core.scenario import BaselineAttackPolicy, Scenario
from pyrit.scenario.core.scenario_strategy import ScenarioStrategy
from pyrit.scenario.core.waterfall import policy_to_spec, spec_to_enum
from pyrit.score import Scorer

# ---------- helpers ----------------------------------------------------------


class _DummyStrategy(ScenarioStrategy):
    ALPHA = ("alpha", set())
    BETA = ("beta", set())
    ALL = ("all", {"all"})

    @classmethod
    def get_aggregate_tags(cls) -> set[str]:
        return {"all"}


def _make_scorer_mock() -> Scorer:
    mock_scorer = MagicMock(spec=Scorer)
    mock_scorer.get_identifier.return_value = ComponentIdentifier(
        class_name="MockScorer", class_module="tests.unit.scenario"
    )
    mock_scorer.get_scorer_metrics.return_value = None
    return mock_scorer


class _DummyScenario(Scenario):
    """Bare Scenario subclass; tests poke ``_scenario_strategies`` directly."""

    BASELINE_ATTACK_POLICY: ClassVar[BaselineAttackPolicy] = BaselineAttackPolicy.Forbidden

    def __init__(self, factories: dict[str, AttackTechniqueFactory] | None = None, **kwargs):
        kwargs.setdefault("strategy_class", _DummyStrategy)
        kwargs.setdefault("objective_scorer", _make_scorer_mock())
        super().__init__(**kwargs)
        self._factories_override = factories or {}
        self._scenario_strategies = []

    @classmethod
    def get_strategy_class(cls) -> type[ScenarioStrategy]:
        return _DummyStrategy

    @classmethod
    def get_default_strategy(cls) -> ScenarioStrategy:
        return _DummyStrategy.ALL

    @classmethod
    def default_dataset_config(cls) -> DatasetConfiguration:
        return DatasetConfiguration()

    def _get_attack_technique_factories(self) -> dict[str, AttackTechniqueFactory]:
        return self._factories_override

    async def _get_atomic_attacks_async(self):
        return []


def _spec(name: str) -> AttackTechniqueSpec:
    return AttackTechniqueSpec(name=name, attack_class=PromptSendingAttack)


def _factory_from_spec(spec: AttackTechniqueSpec) -> AttackTechniqueFactory:
    return AttackTechniqueRegistry.build_factory_from_spec(spec)


def _factory_without_spec() -> AttackTechniqueFactory:
    return AttackTechniqueFactory(attack_class=PromptSendingAttack)


# ---------- policy_to_spec ---------------------------------------------------


@pytest.mark.usefixtures("patch_central_database")
class TestPolicyToSpec:
    def test_returns_empty_for_unset_strategies(self):
        scenario = _DummyScenario(name="t", version=1)
        scenario._scenario_strategies = []
        assert policy_to_spec(scenario) == []

    def test_returns_empty_when_get_factories_raises_not_implemented(self):
        scenario = _DummyScenario(name="t", version=1)
        scenario._scenario_strategies = [_DummyStrategy.ALPHA]

        def _no_factories() -> dict[str, AttackTechniqueFactory]:
            raise NotImplementedError("policy-parameterized scenario")

        scenario._get_attack_technique_factories = _no_factories  # type: ignore[assignment]
        assert policy_to_spec(scenario) == []

    def test_extracts_spec_for_selected_strategy(self):
        spec = _spec("alpha")
        scenario = _DummyScenario(name="t", version=1, factories={"alpha": _factory_from_spec(spec)})
        scenario._scenario_strategies = [_DummyStrategy.ALPHA]
        result = policy_to_spec(scenario)
        assert result == [spec]

    def test_preserves_strategy_order(self):
        alpha, beta = _spec("alpha"), _spec("beta")
        scenario = _DummyScenario(
            name="t",
            version=1,
            factories={"alpha": _factory_from_spec(alpha), "beta": _factory_from_spec(beta)},
        )
        scenario._scenario_strategies = [_DummyStrategy.BETA, _DummyStrategy.ALPHA]
        result = policy_to_spec(scenario)
        assert result == [beta, alpha]

    def test_skips_factory_without_source_spec(self):
        alpha = _spec("alpha")
        scenario = _DummyScenario(
            name="t",
            version=1,
            factories={
                "alpha": _factory_from_spec(alpha),
                "beta": _factory_without_spec(),
            },
        )
        scenario._scenario_strategies = [_DummyStrategy.ALPHA, _DummyStrategy.BETA]
        # Beta has no source spec → silently skipped
        assert policy_to_spec(scenario) == [alpha]

    def test_skips_strategy_with_missing_factory(self):
        alpha = _spec("alpha")
        scenario = _DummyScenario(name="t", version=1, factories={"alpha": _factory_from_spec(alpha)})
        scenario._scenario_strategies = [_DummyStrategy.ALPHA, _DummyStrategy.BETA]
        assert policy_to_spec(scenario) == [alpha]


# ---------- spec_to_enum -----------------------------------------------------


class TestSpecToEnum:
    def test_empty_specs_returns_empty(self):
        assert spec_to_enum(_DummyScenario, []) == []

    def test_resolves_single_spec(self):
        result = spec_to_enum(_DummyScenario, [_spec("alpha")])
        assert result == [_DummyStrategy.ALPHA]

    def test_preserves_input_order(self):
        result = spec_to_enum(_DummyScenario, [_spec("beta"), _spec("alpha")])
        assert result == [_DummyStrategy.BETA, _DummyStrategy.ALPHA]

    def test_unknown_spec_returns_none(self):
        # 'gamma' is not a member of _DummyStrategy
        result = spec_to_enum(_DummyScenario, [_spec("alpha"), _spec("gamma")])
        assert result is None

    def test_returns_none_when_strategy_class_not_implemented(self):
        class _NoStrategyScenario(_DummyScenario):
            @classmethod
            def get_strategy_class(cls) -> type[ScenarioStrategy]:
                raise NotImplementedError("policy-parameterized")

        cls = cast("type[Scenario]", _NoStrategyScenario)
        assert spec_to_enum(cls, [_spec("alpha")]) is None


# ---------- factory.source_spec round-trip -----------------------------------


class TestFactorySourceSpec:
    def test_factory_built_from_spec_carries_source_spec(self):
        spec = _spec("alpha")
        factory = AttackTechniqueRegistry.build_factory_from_spec(spec)
        assert factory.source_spec is spec

    def test_factory_constructed_directly_has_no_source_spec(self):
        factory = AttackTechniqueFactory(attack_class=PromptSendingAttack)
        assert factory.source_spec is None

    def test_explicit_source_spec_kwarg_is_honored(self):
        spec = _spec("alpha")
        factory = AttackTechniqueFactory(attack_class=PromptSendingAttack, source_spec=spec)
        assert factory.source_spec is spec


@pytest.mark.usefixtures("patch_central_database")
@pytest.mark.parametrize("name", ["alpha", "beta"])
def test_round_trip_policy_to_spec_to_enum(name: str):
    """A scenario with one strategy round-trips back to that strategy."""
    spec = _spec(name)
    scenario = _DummyScenario(name="t", version=1, factories={name: _factory_from_spec(spec)})
    scenario._scenario_strategies = [_DummyStrategy[name.upper()]]
    specs = policy_to_spec(scenario)
    enums = spec_to_enum(_DummyScenario, specs)
    assert enums == [_DummyStrategy[name.upper()]]
