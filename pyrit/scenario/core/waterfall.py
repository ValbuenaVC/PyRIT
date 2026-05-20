# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Phase 8 waterfall — translation between scenario configuration layers.

Three layers, two forward translations (this module):

* **policy**         — a configured :class:`Scenario` with a ready
                       :class:`StrategyGraph` (the executable artifact).
* **spec**           — the technique catalog the policy uses
                       (``list[AttackTechniqueSpec]``).
* **strategy enum**  — the public ``ScenarioStrategy`` enum members that
                       represent those techniques (the CLI / wizard surface).

The forward direction is lossy but well-defined:

* ``policy_to_spec`` extracts the spec layer. Returns ``[]`` when the scenario
  does not use the technique registry pattern (e.g. policy-parameterized
  scenarios like ``BroadSweepThenDeepDive``).
* ``spec_to_enum``  resolves specs to ``ScenarioStrategy`` members of the
  scenario's strategy class. Returns ``None`` when no member matches.

The inverse direction (``enum_to_spec`` + ``spec_to_policy_inputs``) lands in
Phase 8e; both are partial and best-effort. Bugs in any direction stay
localized to this file.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

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
