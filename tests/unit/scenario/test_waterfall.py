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
from pyrit.scenario.core.input_schema import RoleDescriptor, RoleTag
from pyrit.scenario.core.scenario import BaselineAttackPolicy, Scenario
from pyrit.scenario.core.scenario_strategy import ScenarioStrategy
from pyrit.scenario.core.waterfall import (
    enum_to_spec,
    policy_to_spec,
    spec_to_enum,
    spec_to_policy_inputs,
)
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


# ---------- first-party scenario round-trips ---------------------------------
#
# The DummyScenario coverage above pins behavior against a hand-crafted strategy
# class. These tests exercise the same surface against the real first-party
# AdaptiveScenario subclass (TextAdaptive), which inherits the base
# ``_get_attack_technique_factories`` implementation and therefore relies on
# the global ``AttackTechniqueRegistry`` populated by
# ``register_scenario_techniques()``. Regressions in registry-population
# (factories built without ``source_spec``) would break the wizard's ability to
# reconstruct an Adaptive policy from CLI inputs.


@pytest.mark.usefixtures("patch_central_database")
class TestTextAdaptiveRoundTrip:
    """Pin that registry-backed first-party scenarios round-trip cleanly."""

    @pytest.fixture(autouse=True)
    def _reset_registry(self):
        from pyrit.registry import TargetRegistry
        from pyrit.scenario.scenarios.adaptive.text_adaptive import TextAdaptive

        AttackTechniqueRegistry.reset_instance()
        TargetRegistry.reset_instance()
        TextAdaptive._cached_strategy_class = None
        yield
        AttackTechniqueRegistry.reset_instance()
        TargetRegistry.reset_instance()
        TextAdaptive._cached_strategy_class = None

    def _build_scenario(self, monkeypatch: pytest.MonkeyPatch):
        from pyrit.prompt_target import PromptTarget
        from pyrit.scenario.scenarios.adaptive.text_adaptive import TextAdaptive

        adversarial = MagicMock(spec=PromptTarget)
        adversarial.get_identifier.return_value = ComponentIdentifier(
            class_name="MockAdversarial", class_module="tests.unit.scenario"
        )
        monkeypatch.setattr(
            "pyrit.scenario.core.scenario_techniques.get_default_adversarial_target",
            lambda: adversarial,
        )
        monkeypatch.setattr(
            "pyrit.scenario.core.scenario.Scenario._get_default_objective_scorer",
            lambda self: _make_scorer_mock(),
        )
        return TextAdaptive(), TextAdaptive

    def test_default_strategies_round_trip(self, monkeypatch: pytest.MonkeyPatch):
        scenario, scenario_cls = self._build_scenario(monkeypatch)
        strategy_cls = scenario_cls.get_strategy_class()
        default = scenario_cls.get_default_strategy()
        resolved = strategy_cls.resolve(None, default=default)
        scenario._scenario_strategies = resolved

        specs = policy_to_spec(scenario)
        assert [sp.name for sp in specs] == [m.value for m in resolved]

        enums = spec_to_enum(scenario_cls, specs)
        assert enums == resolved

    def test_explicit_leaf_subset_round_trips(self, monkeypatch: pytest.MonkeyPatch):
        scenario, scenario_cls = self._build_scenario(monkeypatch)
        strategy_cls = scenario_cls.get_strategy_class()
        # Use the first three concrete (non-aggregate) members. The exact set
        # is whatever ``SCENARIO_TECHNIQUES`` registers; we don't hardcode it.
        leaves = strategy_cls.get_all_strategies()[:3]
        assert len(leaves) >= 1, "TextAdaptive should register at least one technique"
        scenario._scenario_strategies = leaves

        specs = policy_to_spec(scenario)
        assert [sp.name for sp in specs] == [m.value for m in leaves]

        enums = spec_to_enum(scenario_cls, specs)
        assert enums == leaves

    def test_every_registered_factory_carries_source_spec(self, monkeypatch: pytest.MonkeyPatch):
        scenario, _ = self._build_scenario(monkeypatch)
        factories = scenario._get_attack_technique_factories()
        assert factories, "registry should populate factories for TextAdaptive"
        missing = [name for name, fac in factories.items() if fac.source_spec is None]
        assert missing == [], (
            "Every registry-built factory must expose source_spec so "
            "policy_to_spec can reconstruct the technique catalog "
            f"(found {len(missing)} without source_spec: {missing})"
        )


# ---------- enum_to_spec (inverse) -------------------------------------------


@pytest.mark.usefixtures("patch_central_database")
class TestEnumToSpec:
    """Inverse half of the waterfall — enum → spec via the global registry."""

    @pytest.fixture(autouse=True)
    def _reset_registry(self, monkeypatch: pytest.MonkeyPatch):
        from pyrit.prompt_target import PromptTarget
        from pyrit.registry import TargetRegistry

        AttackTechniqueRegistry.reset_instance()
        TargetRegistry.reset_instance()

        adversarial = MagicMock(spec=PromptTarget)
        adversarial.get_identifier.return_value = ComponentIdentifier(
            class_name="MockAdversarial", class_module="tests.unit.scenario"
        )
        monkeypatch.setattr(
            "pyrit.scenario.core.scenario_techniques.get_default_adversarial_target",
            lambda: adversarial,
        )

        yield
        AttackTechniqueRegistry.reset_instance()
        TargetRegistry.reset_instance()

    def test_empty_input_returns_empty(self):
        assert enum_to_spec([]) == []

    def test_skips_strategies_not_in_global_registry(self):
        # _DummyStrategy.ALPHA is not registered in SCENARIO_TECHNIQUES, so the
        # global registry has no factory for "alpha" → silently skipped.
        result = enum_to_spec([_DummyStrategy.ALPHA])
        assert result == []

    def test_resolves_registered_strategies(self):
        from pyrit.scenario.scenarios.adaptive.text_adaptive import TextAdaptive

        TextAdaptive._cached_strategy_class = None
        strategy_cls = TextAdaptive.get_strategy_class()
        leaves = strategy_cls.get_all_strategies()[:2]
        assert leaves, "TextAdaptive should register at least one leaf strategy"

        specs = enum_to_spec(leaves)
        assert [sp.name for sp in specs] == [m.value for m in leaves]


# ---------- spec_to_policy_inputs (inverse) ----------------------------------


class _EmptySchemaScenario(_DummyScenario):
    @classmethod
    def input_schema(cls) -> list[RoleDescriptor]:
        return []


class _OptionalSchemaScenario(_DummyScenario):
    @classmethod
    def input_schema(cls) -> list[RoleDescriptor]:
        return [
            RoleDescriptor(
                name="alpha",
                description="optional scalar",
                tag=RoleTag.SCALAR,
                param_type=int,
                default=3,
                required=False,
            )
        ]


class _RequiredScalarWithDefaultScenario(_DummyScenario):
    @classmethod
    def input_schema(cls) -> list[RoleDescriptor]:
        return [
            RoleDescriptor(
                name="alpha",
                description="required scalar with default",
                tag=RoleTag.SCALAR,
                param_type=int,
                default=7,
                required=True,
            )
        ]


class _RequiredScalarNoDefaultScenario(_DummyScenario):
    @classmethod
    def input_schema(cls) -> list[RoleDescriptor]:
        return [
            RoleDescriptor(
                name="alpha",
                description="required scalar with no default",
                tag=RoleTag.SCALAR,
                param_type=int,
                required=True,
            )
        ]


class _RequiredOpaqueScenario(_DummyScenario):
    @classmethod
    def input_schema(cls) -> list[RoleDescriptor]:
        return [
            RoleDescriptor(
                name="step",
                description="required opaque step",
                tag=RoleTag.OPAQUE,
                required=True,
            )
        ]


class _SchemaRaisesScenario(_DummyScenario):
    @classmethod
    def input_schema(cls):
        raise NotImplementedError("not yet declared")


class TestSpecToPolicyInputs:
    """Inverse half — derive ``__init__`` kwargs from a spec list when possible."""

    def test_returns_empty_when_schema_empty(self):
        assert spec_to_policy_inputs(_EmptySchemaScenario, []) == {}

    def test_returns_empty_when_default_schema_inherited(self):
        # _DummyScenario inherits the base ``input_schema`` returning [].
        assert spec_to_policy_inputs(_DummyScenario, []) == {}

    def test_returns_empty_when_only_optional_roles(self):
        assert spec_to_policy_inputs(_OptionalSchemaScenario, []) == {}

    def test_returns_empty_when_required_scalar_has_default(self):
        assert spec_to_policy_inputs(_RequiredScalarWithDefaultScenario, []) == {}

    def test_returns_none_when_required_scalar_has_no_default(self):
        assert spec_to_policy_inputs(_RequiredScalarNoDefaultScenario, []) is None

    def test_returns_none_when_required_opaque_role_present(self):
        assert spec_to_policy_inputs(_RequiredOpaqueScenario, []) is None

    def test_treats_not_implemented_schema_as_empty(self):
        assert spec_to_policy_inputs(_SchemaRaisesScenario, []) == {}

    def test_specs_argument_is_currently_ignored(self):
        # The placeholder signature accepts a spec list; verify the function
        # tolerates arbitrary spec inputs without affecting its return value.
        assert spec_to_policy_inputs(_EmptySchemaScenario, [_spec("alpha"), _spec("beta")]) == {}


# ---------- Phase 8f: 4-function round-trip across first-party scenarios -----


@pytest.mark.usefixtures("patch_central_database")
class TestFourFunctionRoundTrip:
    """
    Integration coverage that walks every waterfall function in sequence for
    the three first-party scenario shapes:

    * **Legacy linear** (``Encoding``) — inherits the default ``input_schema``
      of ``[]``; the scenario is reconstructible from CLI flags alone.
    * **Adaptive** (``TextAdaptive``) — declares 4 optional scalar inputs,
      all with defaults; ``spec_to_policy_inputs`` returns ``{}``.
    * **Policy-parameterized** (``BroadSweepThenDeepDive``) — declares 3
      required OPAQUE roles; ``spec_to_policy_inputs`` is forced to return
      ``None`` because closures and bound ``Identifiable`` instances cannot
      be reconstructed from spec metadata alone.

    For each scenario we run ``policy_to_spec → spec_to_enum → enum_to_spec``
    and assert the spec catalog is preserved by name through the round trip,
    then assert ``spec_to_policy_inputs`` returns the documented value.
    """

    @pytest.fixture(autouse=True)
    def _reset(self, monkeypatch: pytest.MonkeyPatch):
        from pyrit.prompt_target import PromptTarget
        from pyrit.registry import TargetRegistry

        AttackTechniqueRegistry.reset_instance()
        TargetRegistry.reset_instance()

        adversarial = MagicMock(spec=PromptTarget)
        adversarial.get_identifier.return_value = ComponentIdentifier(
            class_name="MockAdversarial", class_module="tests.unit.scenario"
        )
        monkeypatch.setattr(
            "pyrit.scenario.core.scenario_techniques.get_default_adversarial_target",
            lambda: adversarial,
        )
        monkeypatch.setattr(
            "pyrit.scenario.core.scenario.Scenario._get_default_objective_scorer",
            lambda self: _make_scorer_mock(),
        )

        yield
        AttackTechniqueRegistry.reset_instance()
        TargetRegistry.reset_instance()

    def _round_trip(self, scenario: Scenario, scenario_cls: type[Scenario]):
        """policy_to_spec → spec_to_enum → enum_to_spec; assert catalog preserved."""
        specs = policy_to_spec(scenario)
        original_names = [sp.name for sp in specs]

        enums = spec_to_enum(scenario_cls, specs)
        assert enums is not None, f"{scenario_cls.__name__}: spec_to_enum returned None"
        assert [m.value for m in enums] == original_names

        rebuilt = enum_to_spec(enums)
        # `enum_to_spec` is best-effort over the global registry; for scenarios
        # whose catalog is fully registered (legacy linear, adaptive) this is a
        # full round-trip.
        assert [sp.name for sp in rebuilt] == original_names
        return specs

    def test_encoding_round_trip(self):
        """Legacy linear scenario (non-registry catalog): spec catalog is empty by design."""
        from pyrit.scenario.scenarios.garak.encoding import Encoding

        Encoding._cached_strategy_class = None
        strategy_cls = Encoding.get_strategy_class()
        default = Encoding.get_default_strategy()
        resolved = strategy_cls.resolve(None, default=default)

        scenario = Encoding()
        scenario._scenario_strategies = resolved

        # Encoding doesn't participate in the AttackTechniqueRegistry catalog
        # — its strategies map to per-encoding atomic attacks rather than
        # registered techniques. The forward waterfall degrades to empty.
        specs = policy_to_spec(scenario)
        assert specs == [], "Encoding should expose no registry-catalog specs"

        # spec_to_enum on an empty list is the empty list (not None).
        enums = spec_to_enum(Encoding, specs)
        assert enums == []

        rebuilt = enum_to_spec(enums)
        assert rebuilt == []

        # Default schema → no required constructor inputs.
        assert spec_to_policy_inputs(Encoding, specs) == {}

    def test_text_adaptive_round_trip(self):
        """Adaptive scenario: 4 optional scalars, full registry catalog."""
        from pyrit.scenario.scenarios.adaptive.text_adaptive import TextAdaptive

        TextAdaptive._cached_strategy_class = None
        strategy_cls = TextAdaptive.get_strategy_class()
        default = TextAdaptive.get_default_strategy()
        resolved = strategy_cls.resolve(None, default=default)

        scenario = TextAdaptive()
        scenario._scenario_strategies = resolved

        specs = self._round_trip(scenario, TextAdaptive)
        assert specs, "TextAdaptive should expose at least one technique spec"

        # All 4 input_schema roles are optional with defaults → returns {}.
        assert spec_to_policy_inputs(TextAdaptive, specs) == {}

    def test_broad_sweep_then_deep_dive_returns_none_from_policy_inputs(self):
        """Policy-parameterized scenario: opaque-required schema forces ``None``."""
        from pyrit.scenario.scenarios.airt.sweep_then_deep_dive import BroadSweepThenDeepDive

        # The scenario's input_schema declares required OPAQUE roles. We don't
        # need to instantiate it to assert this — spec_to_policy_inputs is a
        # classmethod-style query over the type.
        result = spec_to_policy_inputs(BroadSweepThenDeepDive, [])
        assert result is None, (
            "BroadSweepThenDeepDive declares required OPAQUE roles; "
            "spec_to_policy_inputs must return None to signal the wizard / CLI "
            "should fall back to --from-artifact."
        )
