# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Import prepared plug-ins and discover owned PyRIT contributions."""

from __future__ import annotations

import asyncio
import importlib
import inspect
import pkgutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

from pyrit.exceptions import PluginDiscoveryError, PluginImportError
from pyrit.models import class_name_to_snake_case
from pyrit.scenario import AttackTechniqueFactory, Scenario

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from types import ModuleType

    from pyrit.setup.plugin_formats import PreparedPlugin


@dataclass(frozen=True)
class ImportedPlugin:
    """The imported modules owned by one prepared plug-in."""

    prepared: PreparedPlugin
    modules: tuple[ModuleType, ...]


@dataclass(frozen=True)
class ScenarioContribution:
    """A discovered scenario class and its proposed registry name."""

    scenario_class: type[Scenario]
    registry_name: str


@dataclass(frozen=True)
class TechniqueContribution:
    """A configured attack technique and the scenarios that expose it."""

    factory: AttackTechniqueFactory
    scenario_names: frozenset[str]


async def import_plugin_async(*, prepared: PreparedPlugin) -> ImportedPlugin:
    """
    Import a prepared plug-in and all submodules under package entry modules.

    Args:
        prepared (PreparedPlugin): The prepared source or wheel import root.

    Returns:
        ImportedPlugin: The imported, ownership-validated modules.

    Raises:
        PluginImportError: If import or ownership validation fails.
    """
    return await asyncio.to_thread(_import_plugin, prepared=prepared)


def _import_plugin(*, prepared: PreparedPlugin) -> ImportedPlugin:
    import_root = str(prepared.import_root)
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

    try:
        for entry_module in prepared.entry_modules:
            module = importlib.import_module(entry_module)
            _verify_module_location(module=module, import_root=prepared.import_root)
            _import_submodules(module=module)
    except Exception as exc:
        raise PluginImportError(
            f"Could not import plug-in '{prepared.spec.name}' from {prepared.import_root}: {exc}"
        ) from exc

    modules = tuple(
        module
        for name, module in sorted(sys.modules.items())
        if module is not None and _name_owned_by_any(name=name, prefixes=prepared.owned_module_prefixes)
    )
    for module in modules:
        _verify_module_location(module=module, import_root=prepared.import_root)
    return ImportedPlugin(prepared=prepared, modules=modules)


def _import_submodules(*, module: ModuleType) -> None:
    module_path = getattr(module, "__path__", None)
    if not module_path:
        return
    for item in pkgutil.walk_packages(module_path, prefix=f"{module.__name__}."):
        importlib.import_module(item.name)


def _verify_module_location(*, module: ModuleType, import_root: Path) -> None:
    root = import_root.resolve()
    raw_locations = list(getattr(module, "__path__", []) or [])
    module_file = getattr(module, "__file__", None)
    if module_file:
        raw_locations.append(module_file)
    locations = [Path(location).resolve() for location in raw_locations if location]
    if locations and not any(location.is_relative_to(root) for location in locations):
        raise ValueError(f"Imported module '{module.__name__}' resolved outside plug-in root {root}.")


def discover_scenarios(*, imported: ImportedPlugin) -> list[ScenarioContribution]:
    """
    Discover concrete, module-owned Scenario subclasses.

    Args:
        imported (ImportedPlugin): The imported plug-in modules.

    Returns:
        list[ScenarioContribution]: Deterministically ordered scenario contributions.

    Raises:
        PluginDiscoveryError: If one module defines multiple implicit scenarios.
    """
    contributions: list[ScenarioContribution] = []
    for module in imported.modules:
        classes = _module_defined_subclasses(module=module, base_class=Scenario)
        if len(classes) > 1:
            raise PluginDiscoveryError(
                f"Plug-in module '{module.__name__}' defines multiple Scenario classes. "
                "Provide explicit registry names through the plug-in manifest."
            )
        if classes:
            contributions.append(
                ScenarioContribution(
                    scenario_class=classes[0],
                    registry_name=_scenario_registry_name(imported=imported, module=module, cls=classes[0]),
                )
            )
    return sorted(contributions, key=lambda item: item.registry_name)


def discover_techniques(*, imported: ImportedPlugin) -> list[TechniqueContribution]:
    """
    Discover configured attack-technique factories from owned modules.

    Args:
        imported (ImportedPlugin): The imported plug-in modules.

    Returns:
        list[TechniqueContribution]: Deterministically ordered technique contributions.

    Raises:
        PluginDiscoveryError: If a factory export is malformed, lacks applicability,
            or duplicates another contribution name.
    """
    contributions: list[TechniqueContribution] = []
    seen_objects: set[int] = set()
    for module in imported.modules:
        explicit = [
            value
            for value in vars(module).values()
            if isinstance(value, TechniqueContribution) and id(value.factory) not in seen_objects
        ]
        contributions.extend(explicit)
        seen_objects.update(id(item.factory) for item in explicit)

        factory_builder = getattr(module, "get_technique_factories", None)
        if callable(factory_builder) and getattr(factory_builder, "__module__", None) == module.__name__:
            contributions.extend(
                _contributions_from_factory_builder(
                    module=module,
                    factory_builder=factory_builder,
                    seen_objects=seen_objects,
                )
            )

        for value in vars(module).values():
            if isinstance(value, AttackTechniqueFactory) and id(value) not in seen_objects:
                contributions.append(_contribution_from_factory(factory=value, module_name=module.__name__))
                seen_objects.add(id(value))

    by_name: dict[str, TechniqueContribution] = {}
    for contribution in contributions:
        if not contribution.scenario_names:
            raise PluginDiscoveryError(
                f"Attack technique '{contribution.factory.name}' must declare at least one applicable scenario."
            )
        existing = by_name.get(contribution.factory.name)
        if existing is not None and existing.factory is not contribution.factory:
            raise PluginDiscoveryError(
                f"Plug-in discovered duplicate attack technique name '{contribution.factory.name}'."
            )
        contribution.factory.get_identifier()
        by_name[contribution.factory.name] = contribution
    return [by_name[name] for name in sorted(by_name)]


def _contributions_from_factory_builder(
    *,
    module: ModuleType,
    factory_builder: Callable[[], object],
    seen_objects: set[int],
) -> list[TechniqueContribution]:
    signature = inspect.signature(factory_builder)
    required = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.default is inspect.Parameter.empty
        and parameter.kind
        not in {
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        }
    ]
    if required:
        raise PluginDiscoveryError(
            f"Plug-in function '{module.__name__}.get_technique_factories' must take no required arguments."
        )
    result = factory_builder()
    if not isinstance(result, (list, tuple)) or not all(isinstance(item, AttackTechniqueFactory) for item in result):
        raise PluginDiscoveryError(
            f"Plug-in function '{module.__name__}.get_technique_factories' must return "
            "AttackTechniqueFactory instances."
        )
    factories = cast("Sequence[AttackTechniqueFactory]", result)
    contributions = [
        _contribution_from_factory(factory=factory, module_name=module.__name__)
        for factory in factories
        if id(factory) not in seen_objects
    ]
    seen_objects.update(id(item.factory) for item in contributions)
    return contributions


def _contribution_from_factory(*, factory: AttackTechniqueFactory, module_name: str) -> TechniqueContribution:
    prefix = "scenario:"
    scenario_names = frozenset(
        tag.removeprefix(prefix) for tag in factory.strategy_tags if tag.startswith(prefix) and tag != prefix
    )
    if not scenario_names:
        raise PluginDiscoveryError(
            f"Attack technique '{factory.name}' from '{module_name}' must declare an applicable scenario "
            f"using a '{prefix}<registry-name>' tag or an explicit TechniqueContribution."
        )
    return TechniqueContribution(factory=factory, scenario_names=scenario_names)


def _module_defined_subclasses(*, module: ModuleType, base_class: type[Scenario]) -> list[type[Scenario]]:
    return [
        candidate
        for _, candidate in inspect.getmembers(module, inspect.isclass)
        if candidate is not base_class
        and issubclass(candidate, base_class)
        and not inspect.isabstract(candidate)
        and candidate.__module__ == module.__name__
    ]


def _scenario_registry_name(
    *,
    imported: ImportedPlugin,
    module: ModuleType,
    cls: type[Scenario],
) -> str:
    prefix = next(
        (
            owned
            for owned in sorted(imported.prepared.owned_module_prefixes, key=len, reverse=True)
            if module.__name__ == owned or module.__name__.startswith(f"{owned}.")
        ),
        "",
    )
    relative = module.__name__[len(prefix) :].lstrip(".") if prefix else module.__name__
    for marker in ("scenario.scenarios.", "scenarios."):
        if marker in relative:
            relative = relative.split(marker, 1)[1]
            break
    if relative:
        return relative
    source = imported.prepared.spec.source
    if source is not None and source.is_file():
        return source.stem
    return class_name_to_snake_case(cls.__name__, suffix="Scenario")


def _name_owned_by_any(*, name: str, prefixes: tuple[str, ...]) -> bool:
    return any(name == prefix or name.startswith(f"{prefix}.") for prefix in prefixes)
