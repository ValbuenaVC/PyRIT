# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Phase 8 waterfall — translation between scenario configuration layers.

Three layers, four translations:

* **policy**         — a configured :class:`Scenario` with a ready
                       :class:`StrategyGraph` (the executable artifact).
* **spec**           — the technique catalog the policy uses
                       (``list[AttackTechniqueSpec]``).
* **strategy enum**  — the public ``ScenarioStrategy`` enum members that
                       represent those techniques (the CLI / wizard surface).

Forward direction (lossy but well-defined):

* ``policy_to_spec`` extracts the spec layer. Returns ``[]`` when the scenario
  does not use the technique registry pattern (e.g. policy-parameterized
  scenarios like ``BroadSweepThenDeepDive``).
* ``spec_to_enum``  resolves specs to ``ScenarioStrategy`` members of the
  scenario's strategy class. Returns ``None`` when no member matches.

Inverse direction (best-effort, partial; Phase 8e):

* ``enum_to_spec``  looks up specs for selected enum members via the global
  ``AttackTechniqueRegistry`` singleton (``register_scenario_techniques()``
  is called once to populate it). Skips members that are not registered
  there — that's the documented "best-effort" semantics.
* ``spec_to_policy_inputs`` returns the rich-object ``__init__`` payload
  needed to reconstruct a scenario from a spec list. Returns ``{}`` when
  the scenario's ``input_schema()`` is empty or all required roles have
  defaults; returns ``None`` for scenarios with required OPAQUE roles
  (the inverse waterfall can't materialize closures / instances from
  spec metadata — the caller is expected to fall back to
  ``pyrit_scan --from-artifact``).

Bugs in any direction stay localized to this file.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pyrit.registry.object_registries.attack_technique_registry import (
        AttackTechniqueSpec,
    )
    from pyrit.scenario.core.scenario import Scenario
    from pyrit.scenario.core.scenario_strategy import ScenarioStrategy


def policy_to_spec(scenario: Scenario) -> list[AttackTechniqueSpec]:
    """
    Extract the technique catalog (spec layer) backing a configured scenario.

    Walks the scenario's selected strategies and the factories it returns from
    ``_get_attack_technique_factories``, returning each factory's
    :attr:`AttackTechniqueFactory.source_spec`. Factories without a source
    spec (constructed directly rather than via
    ``AttackTechniqueRegistry.build_factory_from_spec``) are skipped silently.

    Args:
        scenario (Scenario): An initialized scenario (``initialize_async`` has
            been called, so ``_scenario_strategies`` is populated).

    Returns:
        list[AttackTechniqueSpec]: One spec per selected technique that has a
        recoverable source spec. Empty list when the scenario uses no
        technique registry (policy-parameterized scenarios like
        ``BroadSweepThenDeepDive``) or when no factories carry source specs.
    """
    selected = getattr(scenario, "_scenario_strategies", None)
    if not selected:
        return []

    try:
        factories = scenario._get_attack_technique_factories()
    except (AttributeError, NotImplementedError):
        return []

    specs: list[AttackTechniqueSpec] = []
    for strategy in selected:
        factory = factories.get(strategy.value)
        if factory is None or factory.source_spec is None:
            continue
        specs.append(factory.source_spec)
    return specs


def spec_to_enum(
    scenario_cls: type[Scenario],
    specs: list[AttackTechniqueSpec],
) -> list[ScenarioStrategy] | None:
    """
    Resolve a spec list to ``ScenarioStrategy`` members of a scenario class.

    Matches by :attr:`AttackTechniqueSpec.name` against
    ``scenario_cls.get_strategy_class()`` members. Returns ``None`` when any
    spec cannot be resolved (e.g. the spec name is not a member of the
    scenario's strategy class) — the caller is expected to fall back to
    ``--from-artifact`` (Phase 8e).

    Args:
        scenario_cls (type[Scenario]): The scenario class whose strategy enum
            should be inspected.
        specs (list[AttackTechniqueSpec]): Specs to resolve. An empty list
            returns ``[]`` (vacuously valid).

    Returns:
        list[ScenarioStrategy] | None: Matching enum members in input order,
        or ``None`` when at least one spec cannot be resolved.
    """
    if not specs:
        return []

    try:
        strategy_cls = scenario_cls.get_strategy_class()
    except (AttributeError, NotImplementedError):
        return None

    by_value = {member.value: member for member in strategy_cls}
    resolved: list[ScenarioStrategy] = []
    for spec in specs:
        member = by_value.get(spec.name)
        if member is None:
            return None
        resolved.append(member)
    return resolved


def enum_to_spec(strategies: list[ScenarioStrategy]) -> list[AttackTechniqueSpec]:
    """
    Look up :class:`AttackTechniqueSpec` instances for selected strategy enum members.

    Inverse of :func:`spec_to_enum`. Populates the global
    :class:`AttackTechniqueRegistry` singleton via
    :func:`register_scenario_techniques` (idempotent), then matches each
    strategy's ``value`` against the registry's factory catalog. Each factory's
    :attr:`AttackTechniqueFactory.source_spec` is the spec we return.

    Best-effort by design: strategies that are not registered in the global
    catalog (e.g. members of a per-scenario benchmark catalog like
    :class:`AdversarialBenchmark`) are silently skipped. Use
    :func:`policy_to_spec` on an instantiated scenario for catalog-specific
    spec extraction.

    Args:
        strategies (list[ScenarioStrategy]): Selected enum members.

    Returns:
        list[AttackTechniqueSpec]: One spec per strategy that resolves through
        the global registry, in input order. May be shorter than the input list
        when some strategies belong to a per-scenario catalog not in the global
        registry.
    """
    if not strategies:
        return []

    from pyrit.registry.object_registries.attack_technique_registry import (
        AttackTechniqueRegistry,
    )
    from pyrit.scenario.core.scenario_techniques import register_scenario_techniques

    register_scenario_techniques()
    factories = AttackTechniqueRegistry.get_registry_singleton().get_factories()

    specs: list[AttackTechniqueSpec] = []
    for strategy in strategies:
        factory = factories.get(strategy.value)
        if factory is None or factory.source_spec is None:
            continue
        specs.append(factory.source_spec)
    return specs


def spec_to_policy_inputs(
    scenario_cls: type[Scenario],
    specs: list[AttackTechniqueSpec],
) -> dict[str, Any] | None:
    """
    Derive the rich-object ``__init__`` payload from a spec list, when possible.

    Inverse-direction half of the waterfall. Designed to be partial:

    * Returns ``{}`` for scenarios whose :meth:`Scenario.input_schema` is empty
      or only declares roles with defaults — most first-party scenarios fall
      here. The spec list itself becomes ``scenario_strategies`` at
      ``initialize_async`` time (via :func:`spec_to_enum`), not a constructor
      argument.
    * Returns ``None`` for scenarios with required OPAQUE roles
      (e.g. :class:`BroadSweepThenDeepDive`'s pre-built ``AtomicAttack`` /
      ``OutcomeScorer``). Spec metadata can't materialize a closure or a
      bound :class:`Identifiable` instance — the caller is expected to fall
      back to ``pyrit_scan --from-artifact path.yaml``.

    Args:
        scenario_cls (type[Scenario]): The scenario class to derive inputs for.
        specs (list[AttackTechniqueSpec]): The spec catalog the policy uses.
            Currently not used to fill in scalar inputs — kept in the signature
            so future scenarios that derive constructor kwargs from spec
            metadata (e.g. shared seed-technique counts) can do so without a
            signature break.

    Returns:
        dict[str, Any] | None: Constructor kwargs (often ``{}``), or ``None``
        when the scenario declares required inputs that cannot be reconstructed
        from spec metadata alone.
    """
    del specs  # placeholder: future scenarios may derive scalars from specs

    try:
        schema = list(scenario_cls.input_schema())
    except (AttributeError, NotImplementedError):
        return {}

    if not schema:
        return {}

    from pyrit.scenario.core.input_schema import RoleTag

    for role in schema:
        if role.required and role.tag is RoleTag.OPAQUE:
            return None

    # Schemas with only optional or non-OPAQUE roles can fall through to
    # constructor defaults; the wizard / artifact path supplies anything else.
    for role in schema:
        if role.required and role.default is None:
            return None

    return {}
